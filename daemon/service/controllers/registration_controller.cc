// Copyright (c) 2025, TensorCast Team.

// Implementation of RegistrationController

#include "daemon/service/controllers/registration_controller.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "daemon/status_utils.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

grpc::Status RegistrationController::begin(
    RpcContext& rctx,
    const v1::BeginRegisterArtifactRequest& req,
    v1::BeginRegisterArtifactResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req.device_id()));
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req.total_size()));

  store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
  reg.device_id = req.device_id();
  reg.total_size_bytes = req.total_size();
  reg.enable_p2p = true;
  if (req.has_ttl_ms())
    reg.ttl_ms = req.ttl_ms();
  if (req.owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid is required (>0)"};
  }
  RegistrationManager::RegPlan plan = RegistrationManager::RegPlan::COALESCED;
  if (req.has_lease())
    plan = RegistrationManager::RegPlan::LEASE;
  RegistrationManager::RegMeta meta;
  meta.plan = plan;
  meta.total_size = req.total_size();
  meta.device_id = req.device_id();
  meta.owner_pid = req.owner_pid();
  if (req.has_lease())
    meta.lease_in_place = req.lease().in_place();
  if (req.has_ttl_ms() && req.ttl_ms() > 0) {
    meta.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(req.ttl_ms());
    meta.ttl_ms = static_cast<uint32_t>(req.ttl_ms());
  }
  if (req.has_tensor_index_key())
    meta.index_key_hex = req.tensor_index_key();
  else if (req.has_tensor_index_data())
    meta.index_data = std::string(req.tensor_index_data().data().begin(), req.tensor_index_data().data().end());

  if (plan == RegistrationManager::RegPlan::COALESCED) {
    if (req.has_tensor_index_key())
      reg.tensor_index_key = req.tensor_index_key();
    if (req.has_tensor_index_data()) {
      reg.tensor_index_data = meta.index_data;
      reg.schema_version = req.tensor_index_data().schema_version();
      reg.encoding = req.tensor_index_data().encoding();
    }
    auto begin_or = d_.engine.begin_register_artifact(reg);
    if (!begin_or.ok())
      return to_grpc_status(begin_or.status());
    const auto& out = begin_or.value();
    resp.set_registration_id(out.registration_id);
    auto* hs = resp.mutable_coalesced();
    hs->set_daemon_ipc_handle(
        reinterpret_cast<const char*>(out.cuda_ipc_handle_bytes.data()), out.cuda_ipc_handle_bytes.size());
    resp.set_device_id(out.device_id);
    resp.set_total_size(out.size_bytes);
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_begin_coalesced_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_begin_coalesced_total unavailable";
    }
    d_.reg.set_meta(out.registration_id, meta);
    rctx.mark_success();
    return Status::OK;
  }
  // CPU plan removed
  if (plan == RegistrationManager::RegPlan::LEASE) {
    // Only LIP (in_place=true) is supported in current release
    if (!meta.lease_in_place) {
      return {StatusCode::UNIMPLEMENTED, "vram_leased (in_place=false) is not implemented; set lease_in_place=true"};
    }
    std::string reg_id = absl::StrCat("reg_", absl::ToUnixNanos(absl::Now()), "_", getpid());
    resp.set_registration_id(reg_id);
    resp.mutable_lease();
    resp.set_device_id(req.device_id());
    resp.set_total_size(req.total_size());
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_begin_lease_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_begin_lease_total unavailable";
    }
    d_.reg.set_meta(reg_id, meta);
    rctx.mark_success();
    return Status::OK;
  }
  return Status::OK;
}

grpc::Status RegistrationController::feed_stream(
    RpcContext& rctx,
    ::grpc::ServerReader<v1::FeedRegisterArtifactStreamRequest>& reader,
    v1::FeedRegisterArtifactStreamResponse& /*resp*/) {
  v1::FeedRegisterArtifactStreamRequest req;
  std::string reg_id;
  while (reader.Read(&req)) {
    if (reg_id.empty()) {
      reg_id = req.registration_id();
      if (!d_.reg.has_meta(reg_id)) {
        return {StatusCode::NOT_FOUND, "registration_id not found"};
      }
      if (d_.reg.expire_if_ttl_elapsed(reg_id)) {
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_feed_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_ttl_expired_feed_total unavailable";
        }
        return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
      }
    } else if (req.registration_id() != reg_id) {
      return {StatusCode::INVALID_ARGUMENT, "registration_id changed in stream"};
    }

    uint32_t extend_ms = d_.reg.extend_if_has_ttl(reg_id);
    if (extend_ms > 0) {
      auto st = d_.engine.keep_alive_registered_artifact(reg_id, extend_ms);
      if (!st.ok()) {
        LOG(WARNING) << "keep_alive_registered_artifact failed (stream): reg_id=" << reg_id << ": " << st;
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto ctr = meter->CreateDoubleCounter("tc_register_keepalive_failed_total");
          ctr->Add(1.0);
        } catch (...) {
        }
      }
    }
    if (req.has_lease_segments()) {
      std::vector<LeaseSegMeta> to_add;
      to_add.reserve(req.lease_segments().segments_size());
      for (const auto& s : req.lease_segments().segments()) {
        LeaseSegMeta m;
        m.device_id = s.device_id();
        m.handle_bytes = s.cuda_ipc_handle();
        m.base_offset = s.base_addr();
        m.length = s.length();
        m.dst_offset = s.dst_offset();
        to_add.push_back(std::move(m));
      }
      d_.reg.append_lease_segments(reg_id, std::move(to_add));
    } else if (!req.storage_entries().empty() || !req.tensor_aliases().empty()) {
      // allow requests that only carry storage/alias metadata without segments
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
    }
    if (!req.storage_entries().empty()) {
      std::vector<RegisterStorageMeta> storages;
      storages.reserve(req.storage_entries().size());
      for (const auto& entry : req.storage_entries()) {
        RegisterStorageMeta meta;
        meta.storage_id = entry.storage_id();
        meta.device_id = entry.device_id();
        meta.handle_bytes = entry.cuda_ipc_handle();
        meta.storage_length = entry.storage_length();
        storages.push_back(std::move(meta));
      }
      if (!storages.empty())
        d_.reg.append_storage_entries(reg_id, std::move(storages));
    }
    if (!req.tensor_aliases().empty()) {
      std::vector<RegisterTensorAliasMeta> aliases;
      aliases.reserve(req.tensor_aliases().size());
      for (const auto& alias : req.tensor_aliases()) {
        RegisterTensorAliasMeta meta;
        meta.name = alias.name();
        meta.storage_id = alias.storage_id();
        meta.storage_offset = alias.storage_offset();
        meta.logical_length = alias.logical_length();
        meta.shape.reserve(alias.shape().size());
        for (int64_t v : alias.shape())
          meta.shape.push_back(v);
        meta.stride.reserve(alias.stride().size());
        for (int64_t v : alias.stride())
          meta.stride.push_back(v);
        meta.dtype = alias.dtype();
        aliases.push_back(std::move(meta));
      }
      if (!aliases.empty())
        d_.reg.append_tensor_aliases(reg_id, std::move(aliases));
    }
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status RegistrationController::feed_vector(const std::vector<v1::FeedRegisterArtifactStreamRequest>& reqs) {
  std::string reg_id;
  for (const auto& req : reqs) {
    if (reg_id.empty()) {
      reg_id = req.registration_id();
      if (!d_.reg.has_meta(reg_id)) {
        return {StatusCode::NOT_FOUND, "registration_id not found"};
      }
      if (d_.reg.expire_if_ttl_elapsed(reg_id)) {
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_feed_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_ttl_expired_feed_total unavailable";
        }
        return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
      }
    } else if (req.registration_id() != reg_id) {
      return {StatusCode::INVALID_ARGUMENT, "registration_id changed in stream"};
    }
    {
      uint32_t extend_ms = d_.reg.extend_if_has_ttl(reg_id);
      if (extend_ms > 0) {
        auto st = d_.engine.keep_alive_registered_artifact(reg_id, extend_ms);
        if (!st.ok()) {
          LOG(WARNING) << "keep_alive_registered_artifact failed (vector): reg_id=" << reg_id << ": " << st;
          try {
            static auto meter =
                opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
            static auto ctr = meter->CreateDoubleCounter("tc_register_keepalive_failed_total");
            ctr->Add(1.0);
          } catch (...) {
          }
        }
      }
    }
    if (req.has_lease_segments()) {
      std::vector<LeaseSegMeta> to_add;
      to_add.reserve(req.lease_segments().segments_size());
      for (const auto& s : req.lease_segments().segments()) {
        LeaseSegMeta m;
        m.device_id = s.device_id();
        m.handle_bytes = s.cuda_ipc_handle();
        m.base_offset = s.base_addr();
        m.length = s.length();
        m.dst_offset = s.dst_offset();
        to_add.push_back(std::move(m));
      }
      d_.reg.append_lease_segments(reg_id, std::move(to_add));
    } else if (!req.storage_entries().empty() || !req.tensor_aliases().empty()) {
      // allow metadata-only payloads
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
    }
    if (!req.storage_entries().empty()) {
      std::vector<RegisterStorageMeta> storages;
      storages.reserve(req.storage_entries().size());
      for (const auto& entry : req.storage_entries()) {
        RegisterStorageMeta meta;
        meta.storage_id = entry.storage_id();
        meta.device_id = entry.device_id();
        meta.handle_bytes = entry.cuda_ipc_handle();
        meta.storage_length = entry.storage_length();
        storages.push_back(std::move(meta));
      }
      if (!storages.empty())
        d_.reg.append_storage_entries(reg_id, std::move(storages));
    }
    if (!req.tensor_aliases().empty()) {
      std::vector<RegisterTensorAliasMeta> aliases;
      aliases.reserve(req.tensor_aliases().size());
      for (const auto& alias : req.tensor_aliases()) {
        RegisterTensorAliasMeta meta;
        meta.name = alias.name();
        meta.storage_id = alias.storage_id();
        meta.storage_offset = alias.storage_offset();
        meta.logical_length = alias.logical_length();
        meta.shape.reserve(alias.shape().size());
        for (int64_t v : alias.shape())
          meta.shape.push_back(v);
        meta.stride.reserve(alias.stride().size());
        for (int64_t v : alias.stride())
          meta.stride.push_back(v);
        meta.dtype = alias.dtype();
        aliases.push_back(std::move(meta));
      }
      if (!aliases.empty())
        d_.reg.append_tensor_aliases(reg_id, std::move(aliases));
    }
  }
  return Status::OK;
}

grpc::Status RegistrationController::keep_alive(
    RpcContext& rctx,
    const v1::KeepAliveRegisterArtifactRequest& req,
    v1::KeepAliveRegisterArtifactResponse& /*resp*/) {
  {
    auto st = d_.reg.keepalive_precommit(
        req.registration_id(), req.owner_pid(), req.epoch(), req.ttl_ms() > 0 ? req.ttl_ms() : 0, d_.engine);
    if (st.ok())
      goto KEEPALIVE_OK;
    else if (!absl::IsNotFound(st))
      return to_grpc_status(st);
  }
  {
    auto st = d_.lip.keepalive_lease(
        req.registration_id(), req.owner_pid(), req.epoch(), req.ttl_ms() > 0 ? req.ttl_ms() : 0);
    if (!st.ok())
      return to_grpc_status(st);
  }
KEEPALIVE_OK:
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_keepalive_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_keepalive_total unavailable";
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status RegistrationController::commit(
    RpcContext& rctx,
    const v1::CommitRegisteredArtifactRequest& req,
    v1::CommitRegisteredArtifactResponse& resp) {
  auto meta_opt = d_.reg.get_meta(req.registration_id());
  RegistrationManager::RegMeta meta;
  if (meta_opt.has_value()) {
    meta = *meta_opt;
    LOG(INFO) << "RegistrationController::commit: " << req.registration_id() << ",  plan=" << meta.plan
              << ", lease_in_place=" << meta.lease_in_place << ", expiry=" << meta.expiry.time_since_epoch().count();
  } else {
    LOG(INFO) << "RegistrationController::commit: " << req.registration_id() << " no meta";
  }

  if (meta.plan == RegistrationManager::RegPlan::LEASE) {
    // Only LIP (in_place=true) is supported in current release
    if (!meta.lease_in_place) {
      return {StatusCode::UNIMPLEMENTED, "vram_leased (in_place=false) is not implemented; set lease_in_place=true"};
    }
    if (meta.expiry.time_since_epoch().count() > 0 && std::chrono::steady_clock::now() > meta.expiry) {
      d_.reg.erase_all_for(req.registration_id());
      try {
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
        static auto counter = meter->CreateDoubleCounter("tc_register_ttl_expired_commit_total");
        counter->Add(1.0);
      } catch (...) {
        VLOG(1) << "metrics counter tc_register_ttl_expired_commit_total unavailable";
      }
      return {StatusCode::DEADLINE_EXCEEDED, "registration expired (TTL)"};
    }

    // Always commit lease in-place: do not allocate destination GPU memory
    auto lease_vec = d_.reg.get_lease_segments(req.registration_id());
    if (lease_vec.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "no lease segments fed"};
    }

    auto storage_entries = d_.reg.get_storage_entries(req.registration_id());
    auto alias_vec = d_.reg.get_tensor_aliases(req.registration_id());
    if (storage_entries.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "registration missing storage_entries payload"};
    }
    if (alias_vec.empty()) {
      return {StatusCode::FAILED_PRECONDITION, "registration missing tensor_aliases payload"};
    }
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto storage_counter = meter->CreateDoubleCounter("tc_register_storage_count");
      static auto alias_counter = meter->CreateDoubleCounter("tc_register_tensor_count");
      storage_counter->Add(static_cast<double>(storage_entries.size()));
      alias_counter->Add(static_cast<double>(alias_vec.size()));
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_storage_count/tc_register_tensor_count unavailable";
    }

    auto out_or = d_.lip.commit_lease_in_place(
        req.registration_id(),
        meta.device_id,
        meta.owner_pid,
        meta.ttl_ms,
        meta.epoch,
        meta.total_size,
        meta.index_data,
        meta.index_key_hex,
        std::move(lease_vec),
        std::move(storage_entries),
        std::move(alias_vec));
    if (!out_or.ok()) {
      if (absl::IsAlreadyExists(out_or.status())) {
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto counter = meter->CreateDoubleCounter("tc_register_commit_denied_total");
          counter->Add(1.0);
        } catch (...) {
          VLOG(1) << "metrics counter tc_register_commit_denied_total unavailable";
        }
      }
      return to_grpc_status(out_or.status());
    }

    const auto& out = *out_or;
    auto* desc = resp.mutable_artifact_descriptor();
    desc->set_artifact_id(out.artifact_id);
    desc->set_index_multihash(out.index_multihash);
    desc->set_data_multihash(out.data_multihash);
    desc->set_schema_version(out.schema_version);
    desc->set_encoding(out.encoding);
    desc->set_total_size(out.total_size);
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_lip_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_lip_total unavailable";
    }
    // Create CommitLease for VRAM_LEASED in-place ownership (device-unique)
    if (d_.lifecycle) {
      SessionLifecycleManager::CommitSubject subj{.artifact_id = out.artifact_id, .device_id = meta.device_id};
      auto lid_or = d_.lifecycle->create_commit_lease(subj, meta.owner_pid);
      if (!lid_or.ok()) {
        LOG(WARNING) << "create_commit_lease failed: artifact_id=" << out.artifact_id << " dev=" << meta.device_id
                     << ": " << lid_or.status();
        try {
          static auto meter =
              opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
          static auto ctr = meter->CreateDoubleCounter("tc_lease_create_failed_total");
          ctr->Add(1.0);
        } catch (...) {
        }
      }
    }
    // Log lease-in-place registration summary including plan.
    LOG(INFO) << "Registered memory replica: " << out.artifact_id
              << " plan=vram_leased(in_place) device=gpu:" << meta.device_id << " size=" << out.total_size << "B";
    d_.reg.erase_all_for(req.registration_id());
    rctx.mark_success();
    return Status::OK;
  }
  {
    auto commit_or = d_.engine.commit_registered_artifact(req.registration_id());
    if (!commit_or.ok())
      return to_grpc_status(commit_or.status());
    const auto& out = commit_or.value();
    auto* desc = resp.mutable_artifact_descriptor();
    desc->set_artifact_id(out.artifact_id);
    desc->set_index_multihash(out.index_multihash);
    desc->set_data_multihash(out.data_multihash);
    desc->set_schema_version(out.schema_version);
    desc->set_encoding(out.encoding);
    desc->set_total_size(out.size_bytes);
    resp.set_existed(out.existed);
    if (out.existed) {
      // Join reference to existing GPU replica for coalesced plan duplicates
      store::loading::ReplicaKey key{
          .artifact_id = out.artifact_id,
          .device = store::DeviceKey{.type = DeviceType::GPU, .ordinal = meta.device_id, .uuid = ""},
          .replica = 0};
      d_.refs.add_ref(key, meta.owner_pid);
      if (d_.lifecycle && meta.ttl_ms > 0) {
        SessionLifecycleManager::ReplicaSubject subj{.artifact_id = out.artifact_id, .device_id = meta.device_id};
        auto lease_or = d_.lifecycle->create_ttl_use_lease(subj, meta.owner_pid, absl::Milliseconds(meta.ttl_ms));
        if (lease_or.ok()) {
          meta.use_lease_id = *lease_or;
        } else {
          LOG(ERROR) << "failed to create ttl use lease: " << lease_or.status();
        }
      }
      meta.joined_existing = true;
      meta.artifact_id_mi2 = out.artifact_id;
      d_.reg.set_meta(req.registration_id(), meta);
    } else {
      d_.reg.erase_meta(req.registration_id());
    }
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_coalesced_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_coalesced_total unavailable";
    }
    rctx.mark_success();
    return Status::OK;
  }
}

grpc::Status RegistrationController::abort(
    RpcContext& rctx,
    const v1::AbortRegisteredArtifactRequest& req,
    v1::AbortRegisteredArtifactResponse& /*resp*/) {
  auto st = d_.engine.abort_registered_artifact(req.registration_id());
  if (!st.ok())
    return to_grpc_status(st);
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_abort_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_abort_total unavailable";
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status RegistrationController::revoke(
    RpcContext& rctx,
    const v1::RevokeRegisteredArtifactRequest& req,
    v1::RevokeRegisteredArtifactResponse& /*resp*/) {
  // Capture meta for potential joined-reference cleanup
  auto meta_opt = d_.reg.get_meta(req.registration_id());
  {
    auto st = d_.lip.revoke_by_registration_id(req.registration_id());
    if (st.ok())
      goto REVOKE_DONE;
  }
  {
    absl::Status _st = d_.engine.abort_registered_artifact(req.registration_id());
    if (!_st.ok()) {
      LOG(WARNING) << "revoke: abort_registered_artifact failed for id=" << req.registration_id() << ": " << _st;
    }
    d_.reg.erase_all_for(req.registration_id());
  }
REVOKE_DONE:
  if (meta_opt.has_value() && meta_opt->joined_existing) {
    const auto& m = *meta_opt;
    // Release lifecycle UseLease precisely, if recorded
    if (m.use_lease_id != 0) {
      d_.lifecycle->release_lease(static_cast<SessionLifecycleManager::LeaseId>(m.use_lease_id));
    } else {
      // Fallback by subject+pid
      SessionLifecycleManager::ReplicaSubject subj{.artifact_id = m.artifact_id_mi2, .device_id = m.device_id};
      auto st = d_.lifecycle->release_use_lease(subj, m.owner_pid);
      if (!st.ok()) {
        LOG(WARNING) << "release_use_lease failed (revoke fallback): artifact_id=" << m.artifact_id_mi2
                     << " dev=" << m.device_id << ": " << st;
      }
    }
    store::DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = m.device_id, .uuid = ""};
    store::loading::ReplicaKey key{.artifact_id = m.artifact_id_mi2, .device = dev_key, .replica = 0};
    d_.refs.drop_ref(key, m.owner_pid);
  }
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_register_revoke_total");
    counter->Add(1.0);
  } catch (...) {
    VLOG(1) << "metrics counter tc_register_revoke_total unavailable";
  }
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
