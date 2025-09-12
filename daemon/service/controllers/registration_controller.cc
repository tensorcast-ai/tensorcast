// Copyright (c) 2025, TensorCast Team.

// Implementation of RegistrationController

#include "daemon/service/controllers/registration_controller.h"

#include <cstddef>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/store/loader/segment_plan_source.h"
#include "daemon/cuda_ipc_raii.h"
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
  if (req.has_dvmp())
    plan = RegistrationManager::RegPlan::DVMP;
  else if (req.has_lease())
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
  if (plan == RegistrationManager::RegPlan::DVMP) {
    store::StoreEngine::ArtifactRegistration a;
    a.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
    if (req.has_tensor_index_key())
      a.tensor_index_key = req.tensor_index_key();
    if (req.has_tensor_index_data()) {
      a.tensor_index_data = std::string(req.tensor_index_data().data().begin(), req.tensor_index_data().data().end());
      a.schema_version = req.tensor_index_data().schema_version();
      a.encoding = req.tensor_index_data().encoding();
    }
    a.device_id = req.device_id();
    a.total_size_bytes = req.total_size();
    a.enable_p2p = true;
    if (req.has_ttl_ms())
      a.ttl_ms = req.ttl_ms();
    auto begin_or = d_.engine.begin_register_artifact_dvmp(a);
    if (!begin_or.ok())
      return to_grpc_status(begin_or.status());
    const auto& out = begin_or.value();
    resp.set_registration_id(out.registration_id);
    auto* hs = resp.mutable_dvmp()->mutable_stream();
    hs->set_token(out.registration_id);
    resp.set_device_id(out.device_id);
    resp.set_total_size(out.size_bytes);
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_begin_dvmp_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_begin_dvmp_total unavailable";
    }
    d_.reg.set_meta(out.registration_id, meta);
    rctx.mark_success();
    return Status::OK;
  }
  if (plan == RegistrationManager::RegPlan::LEASE) {
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
    if (req.has_dvmp_chunk()) {
      const auto& ck = req.dvmp_chunk();
      auto st = d_.engine.feed_register_dvmp_chunk(reg_id, ck.offset(), ck.data().data(), ck.data().size());
      if (!st.ok())
        return to_grpc_status(st);
    } else if (req.has_lease_segments()) {
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
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
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
    if (req.has_dvmp_chunk()) {
      const auto& ck = req.dvmp_chunk();
      auto st = d_.engine.feed_register_dvmp_chunk(reg_id, ck.offset(), ck.data().data(), ck.data().size());
      if (!st.ok())
        return to_grpc_status(st);
    } else if (req.has_lease_segments()) {
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
    } else {
      return {StatusCode::INVALID_ARGUMENT, "missing feed payload"};
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
  if (meta_opt.has_value())
    meta = *meta_opt;
  if (meta.plan == RegistrationManager::RegPlan::DVMP) {
    if (meta.expiry.time_since_epoch().count() > 0 && std::chrono::steady_clock::now() > meta.expiry) {
      d_.reg.erase_meta(req.registration_id());
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
      // Join a lightweight reference to the existing replica (CPU for DVMP)
      store::loading::ReplicaKey key{
          .artifact_id = out.artifact_id,
          .device = store::DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""},
          .replica = 0};
      d_.refs.add_ref(key, meta.owner_pid);
      // Preserve meta for optional TTL keepalive and mark joined
      meta.joined_existing = true;
      meta.artifact_id_mi2 = out.artifact_id;
      d_.reg.set_meta(req.registration_id(), meta);
    } else {
      d_.reg.erase_meta(req.registration_id());
    }
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_dvmp_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_dvmp_total unavailable";
    }
    rctx.mark_success();
    return Status::OK;
  }
  if (meta.plan == RegistrationManager::RegPlan::LEASE) {
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

    if (meta.lease_in_place) {
      auto lease_vec = d_.reg.get_lease_segments(req.registration_id());
      if (lease_vec.empty())
        return {StatusCode::FAILED_PRECONDITION, "no lease segments fed"};
      auto out_or = d_.lip.commit_lease_in_place(
          req.registration_id(),
          meta.device_id,
          meta.owner_pid,
          meta.ttl_ms,
          meta.epoch,
          meta.total_size,
          meta.index_data,
          meta.index_key_hex,
          std::move(lease_vec));
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
        static auto meter =
            opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
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
    if (meta.index_data.empty() && meta.index_key_hex.empty()) {
      return {StatusCode::INVALID_ARGUMENT, "lease commit requires index (data or key)"};
    }
    if (meta.index_data.empty()) {
      return {StatusCode::INVALID_ARGUMENT, "lease commit requires canonical index bytes (v2 json)"};
    }

    auto plan_or = store::loader::build_segment_plan_from_canonical_index_json(meta.index_data, meta.total_size, 8);
    if (!plan_or.ok())
      return to_grpc_status(plan_or.status());
    auto plan = *plan_or;

    std::vector<LeaseSegMeta> lease_vec;
    lease_vec = d_.reg.get_lease_segments(req.registration_id());
    if (lease_vec.empty())
      return {StatusCode::FAILED_PRECONDITION, "no lease segments fed"};

    struct Opened {
      int device_id;
      CudaIpcMapping map;
      uint64_t base;
      uint64_t len;
    };

    std::vector<Opened> opened;
    opened.reserve(lease_vec.size());

    for (const auto& seg : lease_vec) {
      auto map_or = CudaIpcMapping::open(seg.handle_bytes, cudaIpcMemLazyEnablePeerAccess);
      if (!map_or.ok())
        return to_grpc_status(map_or.status());
      opened.push_back(
          Opened{.device_id = seg.device_id, .map = std::move(*map_or), .base = seg.base_offset, .len = seg.length});
    }

    store::StoreEngine::ArtifactRegistration areg;
    areg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()), ":", getpid());
    areg.tensor_index_key = meta.index_key_hex;
    areg.tensor_index_data = meta.index_data;
    areg.schema_version = "v2";
    areg.encoding = "json";
    areg.device_id = meta.device_id;
    areg.total_size_bytes = meta.total_size;
    areg.enable_p2p = true;
    auto begin2_or = d_.engine.begin_register_artifact(areg);

    if (!begin2_or.ok())
      return to_grpc_status(begin2_or.status());
    const auto& out2 = begin2_or.value();

    struct RegAbortGuard {
      store::StoreEngine* engine;
      std::string id;
      bool active{true};

      ~RegAbortGuard() {
        if (active && engine) {
          absl::Status _st = engine->abort_registered_artifact(id);
          if (!_st.ok()) {
            LOG(WARNING) << "RegAbortGuard: abort_registered_artifact failed for id=" << id << ": " << _st;
          }
        }
      }

      void release() {
        active = false;
      }
    } abort_guard{.engine = &d_.engine, .id = out2.registration_id};

    auto dst_map_or = CudaIpcMapping::open(out2.cuda_ipc_handle_bytes, cudaIpcMemLazyEnablePeerAccess);
    if (!dst_map_or.ok())
      return to_grpc_status(dst_map_or.status());
    auto dst_map = std::move(*dst_map_or);
    void* dst_dev = dst_map.get();
    {
      auto _set = cuda::set_device(meta.device_id);
      if (!_set.ok())
        return to_grpc_status(_set);
    }

    for (const auto& p : plan) {
      if (p.kind != store::loader::SegmentPiece::PAD)
        continue;
      if (p.length == 0)
        continue;
      auto st = cuda::memset(static_cast<uint8_t*>(dst_dev) + p.dst_offset, 0, static_cast<size_t>(p.length));
      if (!st.ok())
        return to_grpc_status(st);
    }

    for (size_t j = 0; j < opened.size(); ++j) {
      const auto& o = opened[j];
      const uint64_t dst_off = lease_vec[j].dst_offset;
      if (dst_off > meta.total_size || o.len > meta.total_size || dst_off + o.len > meta.total_size) {
        return {StatusCode::OUT_OF_RANGE, "lease segment dst range out of bounds"};
      }
      auto st = cuda::memcpy(
          static_cast<uint8_t*>(dst_dev) + dst_off,
          static_cast<uint8_t*>(o.map.get()) + o.base,
          static_cast<size_t>(o.len),
          cudaMemcpyDeviceToDevice);
      if (!st.ok())
        return to_grpc_status(st);
    }

    {
      auto _sync = cuda::device_synchronize();
      if (!_sync.ok())
        return to_grpc_status(_sync);
    }
    auto commit2_or = d_.engine.commit_registered_artifact(out2.registration_id);
    if (!commit2_or.ok())
      return to_grpc_status(commit2_or.status());
    abort_guard.release();
    const auto& d = commit2_or.value();
    auto* desc = resp.mutable_artifact_descriptor();
    desc->set_artifact_id(d.artifact_id);
    desc->set_index_multihash(d.index_multihash);
    desc->set_data_multihash(d.data_multihash);
    desc->set_schema_version(d.schema_version);
    desc->set_encoding(d.encoding);
    desc->set_total_size(d.size_bytes);
    resp.set_existed(d.existed);
    if (d.existed) {
      // Join reference to existing GPU replica for LIP materialized-to-VRAM path
      store::loading::ReplicaKey key{
          .artifact_id = d.artifact_id,
          .device = store::DeviceKey{.type = DeviceType::GPU, .ordinal = meta.device_id, .uuid = ""},
          .replica = 0};
      d_.refs.add_ref(key, meta.owner_pid);
      if (d_.lifecycle && meta.ttl_ms > 0) {
        SessionLifecycleManager::ReplicaSubject subj{.artifact_id = d.artifact_id, .device_id = meta.device_id};
        auto lease_or = d_.lifecycle->create_ttl_use_lease(subj, meta.owner_pid, absl::Milliseconds(meta.ttl_ms));
        if (lease_or.ok()) {
          meta.use_lease_id = *lease_or;
        } else {
          LOG(ERROR) << "failed to create ttl use lease: " << lease_or.status();
        }
      }
      meta.joined_existing = true;
      meta.artifact_id_mi2 = d.artifact_id;
      d_.reg.set_meta(req.registration_id(), meta);
    } else {
      d_.reg.erase_all_for(req.registration_id());
    }
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto counter = meter->CreateDoubleCounter("tc_register_commit_lease_total");
      counter->Add(1.0);
    } catch (...) {
      VLOG(1) << "metrics counter tc_register_commit_lease_total unavailable";
    }
    // Log lease materialized-to-VRAM registration summary including plan.
    LOG(INFO) << "Registered memory replica: " << d.artifact_id
              << " plan=vram_leased(materialized) device=gpu:" << meta.device_id << " size=" << d.size_bytes << "B";
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
    if (m.use_lease_id != 0 && m.plan != RegistrationManager::RegPlan::DVMP) {
      d_.lifecycle->release_lease(static_cast<SessionLifecycleManager::LeaseId>(m.use_lease_id));
    } else if (m.plan != RegistrationManager::RegPlan::DVMP) {
      // Fallback by subject+pid
      SessionLifecycleManager::ReplicaSubject subj{.artifact_id = m.artifact_id_mi2, .device_id = m.device_id};
      auto st = d_.lifecycle->release_use_lease(subj, m.owner_pid);
      if (!st.ok()) {
        LOG(WARNING) << "release_use_lease failed (revoke fallback): artifact_id=" << m.artifact_id_mi2
                     << " dev=" << m.device_id << ": " << st;
      }
    }
    store::DeviceKey dev_key{
        .type = (m.plan == RegistrationManager::RegPlan::DVMP ? DeviceType::CPU : DeviceType::GPU),
        .ordinal = (m.plan == RegistrationManager::RegPlan::DVMP ? -1 : m.device_id),
        .uuid = ""};
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
