// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/grpc_service_impl.h"

#include <nlohmann/json.hpp>
#include <unistd.h>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "core/store/components/global_store_client.h"
#include "daemon/service/artifact_retire_utils.h"
#include "daemon/util/status_utils.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

absl::StatusOr<std::optional<std::string>> parse_view_id(const tensorcast::common::v1::ByteSpaceRef& space) {
  switch (space.kind()) {
    case tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED:
    case tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL:
      return std::nullopt;
    case tensorcast::common::v1::BYTE_SPACE_KIND_VIEW:
      if (space.id().empty()) {
        return absl::InvalidArgumentError("byte_space VIEW requires id");
      }
      return std::optional<std::string>(space.id());
    default:
      return absl::InvalidArgumentError("unsupported byte_space kind");
  }
}

struct ActiveLeaseResolution {
  ArtifactDeviceKey key;
  LipLeaseEntry lease;
};

absl::StatusOr<ActiveLeaseResolution> resolve_active_lease_by_artifact(
    LipManager* lip_manager,
    std::string_view artifact_id,
    const std::optional<std::string>& view_id,
    const std::optional<int32_t>& device_id) {
  if (device_id.has_value()) {
    ArtifactDeviceKey key{
        .artifact_id = std::string(artifact_id),
        .view_id = view_id.value_or(""),
        .device_id = *device_id,
    };
    auto lease = lip_manager->find_active_by_key(key);
    if (!lease.has_value()) {
      return absl::NotFoundError("no active lease for artifact");
    }
    return ActiveLeaseResolution{.key = std::move(key), .lease = std::move(*lease)};
  }

  auto leases = lip_manager->list_active_by_artifact_id(std::string(artifact_id), view_id);
  if (leases.empty()) {
    return absl::NotFoundError("no active lease for artifact");
  }
  if (leases.size() > 1) {
    return absl::InvalidArgumentError("device_id required to disambiguate replicas");
  }

  ArtifactDeviceKey key{
      .artifact_id = std::string(artifact_id),
      .view_id = view_id.value_or(""),
      .device_id = leases.front().device_id,
  };
  return ActiveLeaseResolution{.key = std::move(key), .lease = std::move(leases.front())};
}

absl::StatusOr<ActiveLeaseResolution> resolve_active_lease_for_retire(
    LipManager* lip_manager,
    const v2::RetirePublishedReplicaRequest& req,
    const std::optional<std::string>& view_id) {
  if (req.has_lease_id() && !req.lease_id().empty()) {
    auto key_opt = lip_manager->find_key_by_registration_id(req.lease_id());
    if (!key_opt.has_value()) {
      return absl::NotFoundError("lease_id not found");
    }
    ArtifactDeviceKey key = *key_opt;
    auto lease = lip_manager->find_active_by_key(key);
    if (!lease.has_value()) {
      return absl::NotFoundError("no active lease for lease_id");
    }
    if (key.artifact_id != req.artifact_id()) {
      return absl::InvalidArgumentError("lease_id does not match artifact_id");
    }
    if (view_id.has_value() && key.view_id != *view_id) {
      return absl::InvalidArgumentError("lease_id does not match byte_space");
    }
    if (req.has_device_id() && key.device_id != req.device_id()) {
      return absl::InvalidArgumentError("lease_id does not match device_id");
    }
    return ActiveLeaseResolution{.key = std::move(key), .lease = std::move(*lease)};
  }

  std::optional<int32_t> device_id;
  if (req.has_device_id()) {
    device_id = req.device_id();
  }
  return resolve_active_lease_by_artifact(lip_manager, req.artifact_id(), view_id, device_id);
}

absl::Status ensure_owner_pid_matches(const std::optional<int32_t>& owner_pid, const LipLeaseEntry& lease) {
  if (owner_pid.has_value() && lease.owner_pid != *owner_pid) {
    return absl::PermissionDeniedError("owner_pid mismatch for active lease");
  }
  return absl::OkStatus();
}

absl::StatusOr<artifact_retire::DrainOutcome> drain_and_revoke_active_lease(
    LipManager* lip_manager,
    store::components::IGlobalStoreClient* global_store_client,
    std::string_view artifact_id,
    const ArtifactDeviceKey& key,
    const LipLeaseEntry& lease,
    bool wait_for_drain,
    uint32_t timeout_ms,
    std::optional<std::string_view> operation_id) {
  auto drain_or = artifact_retire::retire_replica_with_drain(
      lip_manager, global_store_client, artifact_id, key, wait_for_drain, timeout_ms, operation_id);
  if (!drain_or.ok()) {
    return drain_or.status();
  }

  absl::Status revoke_status = lip_manager->revoke_by_registration_id(lease.registration_id);
  if (!revoke_status.ok()) {
    return revoke_status;
  }
  return *drain_or;
}

struct DrainedLeaseResolution {
  ArtifactDeviceKey key;
  LipLeaseEntry lease;
  artifact_retire::DrainOutcome drain;
  std::optional<std::string> view_id;
};

bool matches_view_id(
    const std::optional<std::string>& requested_view_id,
    const std::optional<std::string>& key_view_id) {
  if (requested_view_id.has_value()) {
    if (requested_view_id->empty()) {
      return !key_view_id.has_value() || key_view_id->empty();
    }
    return key_view_id.has_value() && *key_view_id == *requested_view_id;
  }
  return !key_view_id.has_value() || key_view_id->empty();
}

bool matches_requested_device(const std::optional<int32_t>& requested_device_id, const store::DeviceKey& device) {
  if (!requested_device_id.has_value()) {
    return true;
  }
  if (*requested_device_id < 0) {
    return device.type == DeviceType::CPU;
  }
  return device.type == DeviceType::GPU && device.ordinal == *requested_device_id;
}

std::vector<store::loading::ReplicaKey> resolve_retire_replica_keys(
    store::StoreEngine* engine,
    std::string_view artifact_id,
    const std::optional<std::string>& view_id,
    const std::optional<int32_t>& requested_device_id,
    const std::optional<ArtifactDeviceKey>& drained_key) {
  absl::flat_hash_set<store::loading::ReplicaKey, store::loading::ReplicaKeyHash> keys;
  for (const auto& info : engine->get_all_replicas_info()) {
    if (info.artifact_id != artifact_id) {
      continue;
    }
    if (!matches_view_id(view_id, info.key.view_id)) {
      continue;
    }
    if (!matches_requested_device(requested_device_id, info.key.device)) {
      continue;
    }
    keys.insert(info.key);
  }

  std::vector<store::loading::ReplicaKey> out;
  out.reserve(keys.size() + 1);
  for (const auto& key : keys) {
    out.push_back(key);
  }
  if (!out.empty()) {
    return out;
  }

  std::optional<store::DeviceKey> fallback_device;
  if (requested_device_id.has_value()) {
    const DeviceType type = *requested_device_id < 0 ? DeviceType::CPU : DeviceType::GPU;
    fallback_device = store::DeviceKey{.type = type, .ordinal = *requested_device_id, .uuid = ""};
  } else if (drained_key.has_value()) {
    const DeviceType type = drained_key->device_id < 0 ? DeviceType::CPU : DeviceType::GPU;
    fallback_device = store::DeviceKey{.type = type, .ordinal = drained_key->device_id, .uuid = ""};
  }
  if (fallback_device.has_value()) {
    out.push_back(
        store::loading::ReplicaKey{
            .artifact_id = std::string(artifact_id),
            .view_id = view_id,
            .device = *fallback_device,
            .replica = 0,
        });
  }
  return out;
}

void retire_local_replicas(
    store::StoreEngine* engine,
    std::string_view artifact_id,
    const std::optional<std::string>& view_id,
    const std::optional<int32_t>& requested_device_id,
    const std::optional<ArtifactDeviceKey>& drained_key,
    v2::DeregisterArtifactResponse* resp) {
  const std::vector<store::loading::ReplicaKey> replica_keys =
      resolve_retire_replica_keys(engine, artifact_id, view_id, requested_device_id, drained_key);
  for (const auto& replica_key : replica_keys) {
    absl::Status st = engine->retire_replica_status(replica_key);
    if (!st.ok() && !absl::IsNotFound(st)) {
      artifact_retire::append_deregister_message(
          resp, absl::StrCat("local retire failed for device_id=", replica_key.device.ordinal, ": ", st.message()));
    }
  }
}

absl::StatusOr<DrainedLeaseResolution> drain_lease_for_deregister(
    LipManager* lip_manager,
    store::components::IGlobalStoreClient* global_store_client,
    const v2::DeregisterArtifactRequest& req) {
  auto view_id_or = parse_view_id(req.byte_space());
  if (!view_id_or.ok()) {
    return view_id_or.status();
  }
  std::optional<std::string> view_id = *view_id_or;

  std::optional<int32_t> device_id;
  if (req.has_device_id()) {
    device_id = req.device_id();
  }
  auto active_lease_or = resolve_active_lease_by_artifact(lip_manager, req.artifact_id(), view_id, device_id);
  if (!active_lease_or.ok()) {
    return active_lease_or.status();
  }
  ActiveLeaseResolution active_lease = std::move(*active_lease_or);

  std::optional<int32_t> owner_pid;
  if (req.has_owner_pid()) {
    owner_pid = req.owner_pid();
  }
  auto owner_status = ensure_owner_pid_matches(owner_pid, active_lease.lease);
  if (!owner_status.ok()) {
    return owner_status;
  }

  if (req.has_extend_ttl_ms() && req.extend_ttl_ms() > 0) {
    auto st = lip_manager->extend_ttl_for_artifact(req.artifact_id(), req.extend_ttl_ms(), view_id);
    if (!st.ok()) {
      return st;
    }
  }

  lip_manager->quiesce_lease(active_lease.key);

  const uint32_t timeout_ms = req.has_drain_timeout_ms() ? req.drain_timeout_ms() : 30000U;
  auto drain_or = drain_and_revoke_active_lease(
      lip_manager,
      global_store_client,
      req.artifact_id(),
      active_lease.key,
      active_lease.lease,
      req.wait_for_drain(),
      timeout_ms,
      req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt);
  if (!drain_or.ok()) {
    return drain_or.status();
  }

  DrainedLeaseResolution out{
      .key = std::move(active_lease.key),
      .lease = std::move(active_lease.lease),
      .drain = std::move(*drain_or),
      .view_id = std::move(view_id),
  };
  return out;
}

absl::StatusOr<DrainedLeaseResolution> drain_lease_for_retire(
    LipManager* lip_manager,
    store::components::IGlobalStoreClient* global_store_client,
    const v2::RetirePublishedReplicaRequest& req) {
  auto view_id_or = parse_view_id(req.byte_space());
  if (!view_id_or.ok()) {
    return view_id_or.status();
  }
  std::optional<std::string> view_id = *view_id_or;

  auto active_lease_or = resolve_active_lease_for_retire(lip_manager, req, view_id);
  if (!active_lease_or.ok()) {
    return active_lease_or.status();
  }
  ActiveLeaseResolution active_lease = std::move(*active_lease_or);

  std::optional<int32_t> owner_pid;
  if (req.has_owner_pid()) {
    owner_pid = req.owner_pid();
  }
  auto owner_status = ensure_owner_pid_matches(owner_pid, active_lease.lease);
  if (!owner_status.ok()) {
    return owner_status;
  }

  lip_manager->quiesce_lease(active_lease.key);

  const uint32_t timeout_ms = req.has_drain_timeout_ms() ? req.drain_timeout_ms() : 30000U;
  auto drain_or = drain_and_revoke_active_lease(
      lip_manager,
      global_store_client,
      req.artifact_id(),
      active_lease.key,
      active_lease.lease,
      req.wait_for_drain(),
      timeout_ms,
      req.has_operation_id() ? std::optional<std::string_view>(req.operation_id()) : std::nullopt);
  if (!drain_or.ok()) {
    return drain_or.status();
  }

  DrainedLeaseResolution out{
      .key = std::move(active_lease.key),
      .lease = std::move(active_lease.lease),
      .drain = std::move(*drain_or),
      .view_id = std::move(view_id),
  };
  return out;
}

} // namespace

StoreDaemonServiceImpl::StoreDaemonServiceImpl(Deps deps, Options opts)
    : engine_(&deps.engine),
      materialization_controller_(&deps.materialization_controller),
      registration_controller_(&deps.registration_controller),
      transport_controller_(&deps.transport_controller),
      status_controller_(&deps.status_controller),
      region_registry_(&deps.region_registry),
      lip_manager_(&deps.lip_manager),
      global_store_client_(std::move(deps.global_store_client)),
      lifecycle_manager_(&deps.lifecycle_manager),
      key_mapping_controller_(&deps.key_mapping_controller),
      persistence_rpc_controller_(&deps.persistence_rpc_controller),
      replica_session_controller_(&deps.replica_session_controller),
      lease_controller_(&deps.lease_controller),
      shutdown_signal_(&deps.shutdown_signal),
      source_registry_(deps.source_registry),
      opts_(std::move(opts)) {}

Status StoreDaemonServiceImpl::ClearMem(
    grpc::ServerContext* ctx,
    const v2::ClearMemRequest* /*req*/,
    v2::ClearMemResponse* /*resp*/) {
  RpcContext rctx{"ClearMem", *ctx, opts_.allow_high_card_attrs};
  const int rc = engine_->clear_mem();
  if (rc == 0) {
    rctx.mark_success();
    return Status::OK;
  }
  return {StatusCode::INTERNAL, absl::StrFormat("clear_mem() returned %d", rc)};
}

Status StoreDaemonServiceImpl::RegisterVramRegion(
    grpc::ServerContext* ctx,
    const v2::RegisterVramRegionRequest* req,
    v2::RegisterVramRegionResponse* resp) {
  RpcContext rctx{"RegisterVramRegion", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.device.id", static_cast<int64_t>(req->device_id()));
  span->SetAttribute("tc.region.size_bytes", static_cast<int64_t>(req->size_bytes()));
  span->SetAttribute("tc.region.ttl_ms", static_cast<int64_t>(req->ttl_ms()));

  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid must be > 0"};
  }
  if (req->device_id() < 0) {
    return {StatusCode::INVALID_ARGUMENT, "device_id must be >= 0"};
  }
  if (req->size_bytes() == 0) {
    return {StatusCode::INVALID_ARGUMENT, "size_bytes must be > 0"};
  }
  if (req->cuda_ipc_handle().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "cuda_ipc_handle must not be empty"};
  }

  IpcRegionRegistry::RegisterParams params;
  params.device_id = req->device_id();
  params.owner_pid = req->owner_pid();
  params.size_bytes = req->size_bytes();
  params.ttl_ms = req->ttl_ms();
  if (req->has_session_id()) {
    params.session_id = req->session_id();
  }
  if (req->has_region_name()) {
    params.region_name = req->region_name();
  }
  params.handle_bytes = std::string(req->cuda_ipc_handle());

  auto desc_or = region_registry_->register_region(params);
  if (!desc_or.ok()) {
    return to_grpc_status(desc_or.status());
  }
  const auto& desc = *desc_or;
  if (desc.ttl_ms == 0 && lifecycle_manager_ != nullptr) {
    lifecycle_manager_->watch_pid(static_cast<pid_t>(desc.owner_pid));
  }
  resp->set_region_id(desc.region_id);
  resp->set_ttl_ms(desc.ttl_ms);
  if (desc.expires_at != absl::InfiniteFuture()) {
    const int64_t micros = absl::ToUnixMicros(desc.expires_at);
    auto* ts = resp->mutable_expires_at();
    ts->set_seconds(micros / 1'000'000);
    ts->set_nanos(static_cast<int32_t>((micros % 1'000'000) * 1'000));
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::UnregisterVramRegion(
    grpc::ServerContext* ctx,
    const v2::UnregisterVramRegionRequest* req,
    v2::UnregisterVramRegionResponse* resp) {
  RpcContext rctx{"UnregisterVramRegion", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  span->SetAttribute("tc.region.id", req->region_id());

  if (req->region_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "region_id is required"};
  }
  if (req->owner_pid() <= 0) {
    return {StatusCode::INVALID_ARGUMENT, "owner_pid must be > 0"};
  }

  const bool force = req->has_force() ? req->force() : false;
  auto released_or = region_registry_->unregister_region(req->region_id(), req->owner_pid(), force);
  if (!released_or.ok()) {
    return to_grpc_status(released_or.status());
  }
  resp->set_released(*released_or);
  if (*released_or && lifecycle_manager_ != nullptr) {
    lifecycle_manager_->unwatch_pid(static_cast<pid_t>(req->owner_pid()));
  }
  rctx.mark_success();
  return Status::OK;
}

Status StoreDaemonServiceImpl::DeregisterArtifact(
    grpc::ServerContext* ctx,
    const v2::DeregisterArtifactRequest* req,
    v2::DeregisterArtifactResponse* resp) {
  RpcContext rctx{"DeregisterArtifact", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  if (!req->artifact_id().empty()) {
    span->SetAttribute("tc.artifact.id", req->artifact_id());
  }
  if (rctx.allow_high_card_attrs() && req->has_operation_id()) {
    span->SetAttribute("tc.operation.id", req->operation_id());
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  const std::string artifact_id = req->artifact_id();
  const bool keep_shared_disk_copy = req->has_keep_shared_disk_copy() ? req->keep_shared_disk_copy() : false;
  auto view_id_or = parse_view_id(req->byte_space());
  if (!view_id_or.ok()) {
    return to_grpc_status(view_id_or.status());
  }
  std::optional<std::string> view_id = *view_id_or;
  std::optional<int32_t> requested_device_id;
  if (req->has_device_id()) {
    requested_device_id = req->device_id();
  }

  std::optional<ArtifactDeviceKey> drained_key;
  std::optional<LipLeaseEntry> drained_lease;
  artifact_retire::DrainOutcome drain{.drained = true, .replica_id = std::nullopt};
  auto drained_or = drain_lease_for_deregister(lip_manager_, global_store_client_.get(), *req);
  if (drained_or.ok()) {
    DrainedLeaseResolution drained = std::move(*drained_or);
    drained_key = std::move(drained.key);
    drained_lease = std::move(drained.lease);
    drain = std::move(drained.drain);
    view_id = std::move(drained.view_id);
  } else if (!absl::IsNotFound(drained_or.status())) {
    return to_grpc_status(drained_or.status());
  } else {
    artifact_retire::append_deregister_message(resp, "no active lease found; proceeding with stateless retire");
  }

  resp->set_drained(drain.drained);
  resp->set_removed(true);

  if (!view_id.has_value()) {
    absl::flat_hash_set<int32_t> unregister_device_ids;
    if (drained_key.has_value()) {
      unregister_device_ids.insert(drained_key->device_id);
    }
    if (requested_device_id.has_value()) {
      unregister_device_ids.insert(*requested_device_id);
    }
    if (unregister_device_ids.empty()) {
      const auto local_keys =
          resolve_retire_replica_keys(engine_, artifact_id, view_id, requested_device_id, drained_key);
      for (const auto& key : local_keys) {
        if (key.device.type != DeviceType::GPU || key.device.ordinal < 0) {
          continue;
        }
        unregister_device_ids.insert(key.device.ordinal);
      }
    }

    for (const int32_t device_id : unregister_device_ids) {
      absl::Status gs_st = engine_->unregister_replica_from_global_store(artifact_id, device_id);
      if (!gs_st.ok() && !absl::IsNotFound(gs_st)) {
        artifact_retire::append_deregister_message(
            resp, absl::StrCat("Global Store deregister failed: ", gs_st.message()));
      }
    }
  }

  absl::flat_hash_set<std::string> unique_regions;
  if (drained_lease.has_value()) {
    for (const auto& s : drained_lease->storages) {
      if (s.has_region()) {
        unique_regions.insert(s.region_id);
      }
    }
  }
  for (const auto& rid : unique_regions) {
    resp->add_released_region_ids(rid);
  }

  retire_local_replicas(engine_, artifact_id, view_id, requested_device_id, drained_key, resp);

  if (!keep_shared_disk_copy && !view_id.has_value()) {
    artifact_retire::purge_managed_shared_disk_artifact(
        global_store_client_.get(), artifact_id, opts_.storage_path, resp);
  }

  // Keep version keys append-only, but retire local disk-import fallback for this artifact
  // so old versions cannot be re-materialized through daemon-local source paths.
  if (source_registry_ != nullptr) {
    const bool erased = source_registry_->erase_binding(artifact_id);
    if (!erased) {
      VLOG(1) << "DeregisterArtifact: no local disk import entry for artifact_id=" << artifact_id;
    }
  }

  rctx.mark_success();
  return grpc::Status::OK;
}

Status StoreDaemonServiceImpl::RetirePublishedReplica(
    grpc::ServerContext* ctx,
    const v2::RetirePublishedReplicaRequest* req,
    v2::RetirePublishedReplicaResponse* resp) {
  RpcContext rctx{"RetirePublishedReplica", *ctx, opts_.allow_high_card_attrs};
  auto& span = rctx.span();
  if (!req->artifact_id().empty()) {
    span->SetAttribute("tc.artifact.id", req->artifact_id());
  }
  if (rctx.allow_high_card_attrs() && req->has_operation_id()) {
    span->SetAttribute("tc.operation.id", req->operation_id());
  }
  if (shutdown_signal_->is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req->artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  auto drained_or = drain_lease_for_retire(lip_manager_, global_store_client_.get(), *req);
  if (!drained_or.ok()) {
    return to_grpc_status(drained_or.status());
  }
  const auto& drain = drained_or->drain;

  resp->set_drained(drain.drained);
  resp->set_removed(true);
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
