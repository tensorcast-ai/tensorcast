// Copyright (c) 2025, TensorCast Team.

// Implementation of MaterializationController

#include "daemon/service/controllers/materialization_controller.h"

#include <future>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "core/store/device_registry.h"
#include "daemon/deadline_utils.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

grpc::Status MaterializationController::MaterializeReplica(
    RpcContext& rctx,
    const v1::MaterializeReplicaRequest& req,
    v1::MaterializeReplicaResponse& resp) {
  auto& span = rctx.span();
  // Always attach artifact_id for correlation
  if (req.has_artifact_id() && !req.artifact_id().empty()) {
    span->SetAttribute("tc.artifact.id", req.artifact_id());
  }
  if (rctx.allow_high_card_attrs()) {
    if (req.has_disk_path() && !req.disk_path().empty()) {
      span->SetAttribute("tc.disk.path", req.disk_path());
    }
    span->SetAttribute("tc.device.uuid", req.device_uuid());
  }
  span->SetAttribute("tc.size.bytes", static_cast<int64_t>(req.size_bytes()));

  using v1::MaterializeReplicaStatus;
  if (d_.is_shutting_down.load()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  const bool has_artifact = req.has_artifact_id() && !req.artifact_id().empty();
  const bool has_disk = req.has_disk_path() && !req.disk_path().empty();
  if (has_artifact == has_disk) {
    return {StatusCode::INVALID_ARGUMENT, "Exactly one of artifact_id or disk_path must be provided"};
  }

  const auto dev = d_.devices.From(req.target_device_type(), req.device_uuid(), std::nullopt);

  // Artifact LIP fast path: try cross-device consumption
  if (has_artifact) {
    auto satisfied = d_.lip.try_satisfy_from_lip(
        req.artifact_id(),
        dev.ordinal,
        [&](const store::loading::ReplicaKey& rkey) {
          if (!req.replica_uuid().empty()) {
            std::promise<absl::Status> p;
            p.set_value(absl::OkStatus());
            d_.sessions.put_with_verification(req.replica_uuid(), rkey, p.get_future().share());
          }
          if (req.pid() > 0) {
            d_.refs.add_ref(rkey, req.pid(), /*keep_for_global=*/req.keep_for_global());
          }
        },
        resp.mutable_mem_handle());
    if (!satisfied.ok()) {
      // Same-device denial or copy failure: set FAILED for parity and propagate as gRPC error
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
      return to_grpc_status(satisfied.status());
    }
    if (*satisfied) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
      rctx.mark_success();
      return Status::OK;
    }
  }

  // Engine-backed materialization
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  if (has_artifact)
    hints.artifact_id = req.artifact_id();
  if (has_disk)
    hints.disk_path = req.disk_path();
  const auto mode =
      has_disk ? store::StoreEngine::MaterializeMode::LOAD_ONLY : store::StoreEngine::MaterializeMode::AUTO;

  auto result = d_.engine.materialize_replica(dev, mode, hints);
  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  if (!req.replica_uuid().empty()) {
    d_.sessions.put_with_verification(req.replica_uuid(), handle.replica_key, handle.ready_future);
  }
  if (req.pid() > 0) {
    d_.refs.add_ref(handle.replica_key, req.pid(), req.keep_for_global());
  }
  if (has_disk)
    resp.set_disk_path(req.disk_path());
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  if (handle.cuda_ipc_handle.is_valid()) {
    resp.mutable_mem_handle()->set_cuda_ipc_handle(handle.cuda_ipc_handle.to_string());
  }
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::MaterializeByKey(
    RpcContext& rctx,
    const v1::MaterializeByKeyRequest& req,
    v1::MaterializeByKeyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req.key());

  using v1::MaterializeReplicaStatus;
  if (d_.is_shutting_down.load()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (req.key().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "key is required"};
  }

  auto mapping_or = d_.engine.resolve_key_mapping(req.key());
  if (!mapping_or.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(mapping_or.status());
  }
  const auto mapping = *mapping_or;
  span->SetAttribute("tc.artifact.id", mapping.artifact_id);

  // Try LIP fast path first
  {
    auto satisfied = d_.lip.try_satisfy_from_lip(
        mapping.artifact_id,
        req.device_id(),
        [&](const store::loading::ReplicaKey& rkey) {
          if (!req.replica_uuid().empty()) {
            std::promise<absl::Status> p;
            p.set_value(absl::OkStatus());
            d_.sessions.put_with_verification(req.replica_uuid(), rkey, p.get_future().share());
          }
          if (req.pid() > 0) {
            d_.refs.add_ref(rkey, req.pid(), /*keep_for_global=*/false);
          }
        },
        resp.mutable_mem_handle());
    if (!satisfied.ok()) {
      return to_grpc_status(satisfied.status());
    }
    if (*satisfied) {
      resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
      resp.set_artifact_id(mapping.artifact_id);
      resp.set_used_disk_path(mapping.disk_path);
      rctx.mark_success();
      return Status::OK;
    }
  }

  // Engine path
  const auto dev = store::DeviceKey{.type = DeviceType::GPU, .ordinal = req.device_id(), .uuid = ""};
  store::loading::MaterializeHints hints;
  if (req.pinned_allocation_timeout_ms() > 0) {
    hints.pinned_timeout = std::chrono::milliseconds(req.pinned_allocation_timeout_ms());
  }
  hints.artifact_id = mapping.artifact_id;
  if (!mapping.disk_path.empty())
    hints.disk_path = mapping.disk_path;

  auto result = d_.engine.materialize_replica(dev, store::StoreEngine::MaterializeMode::AUTO, hints);
  if (!result.ok()) {
    resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_FAILED);
    return to_grpc_status(result.status());
  }
  const auto& handle = *result;
  if (!req.replica_uuid().empty()) {
    d_.sessions.put_with_verification(req.replica_uuid(), handle.replica_key, handle.ready_future);
  }
  if (req.pid() > 0) {
    d_.refs.add_ref(handle.replica_key, req.pid(), /*keep_for_global=*/false);
  }
  if (handle.cuda_ipc_handle.is_valid()) {
    resp.mutable_mem_handle()->set_cuda_ipc_handle(handle.cuda_ipc_handle.to_string());
  }
  resp.set_status(MaterializeReplicaStatus::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  resp.set_artifact_id(mapping.artifact_id);
  resp.set_used_disk_path(mapping.disk_path);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::GetArtifactIndexById(
    RpcContext& rctx,
    const v1::GetArtifactIndexByIdRequest& req,
    v1::GetArtifactIndexByIdResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());

  if (req.artifact_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "artifact_id is required"};
  }
  if (d_.is_shutting_down.load()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  auto bytes_or = d_.engine.get_canonical_index_by_id(req.artifact_id());
  if (!bytes_or.ok())
    return to_grpc_status(bytes_or.status());
  resp.set_tensor_index_data(*bytes_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status MaterializationController::Confirm(
    RpcContext& rctx,
    const v1::ConfirmReplicaRequest& req,
    v1::ConfirmReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.disk.path", req.disk_path());
  }
  span->SetAttribute("tc.device.type", static_cast<int64_t>(req.target_device_type()));
  resp.set_disk_path(req.disk_path());

  if (req.replica_uuid().empty()) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  auto entry = d_.sessions.get(req.replica_uuid());
  if (!entry.has_value()) {
    // Parity: unknown replica_uuid → code=0 OK
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  // Wait bounded by a 30s cap (Confirm has no user timeout)
  auto wait_ms = std::chrono::milliseconds(30000);
  absl::Status st = entry->ready.wait_for(wait_ms) == std::future_status::ready
      ? entry->ready.get()
      : absl::DeadlineExceededError("confirm timeout");
  if (st.ok()) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_code(1);
  return to_grpc_status(st);
}

grpc::Status MaterializationController::Unload(
    RpcContext& rctx,
    const v1::UnloadReplicaRequest& req,
    v1::UnloadReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    if (!req.disk_path().empty())
      span->SetAttribute("tc.disk.path", req.disk_path());
    if (req.has_pid())
      span->SetAttribute("tc.pid", static_cast<int64_t>(req.pid()));
  }
  resp.set_disk_path(req.disk_path());

  if (req.target_device_type() == v1::DeviceType::DEVICE_TYPE_DISK) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  store::loading::ReplicaKey key;
  if (!req.replica_uuid().empty()) {
    auto entry = d_.sessions.get(req.replica_uuid());
    if (entry.has_value())
      key = entry->key;
  }
  if (key.artifact_id.empty()) {
    if (!req.disk_path().empty()) {
      key.artifact_id = req.disk_path();
      key.device = d_.devices.From(req.target_device_type(), /*uuid=*/"", /*ordinal_hint=*/std::nullopt);
      key.replica = 0;
    } else {
      resp.set_code(0);
      rctx.mark_success();
      return Status::OK;
    }
  }
  if (req.has_pid()) {
    d_.refs.drop_ref(key, req.pid());
    if (d_.refs.ref_count(key) > 0) {
      resp.set_code(0);
      rctx.mark_success();
      return Status::OK;
    }
  }
  const int rc = d_.engine.unload_replica(key);
  if (rc == 0) {
    if (!req.replica_uuid().empty()) {
      (void)d_.sessions.erase(req.replica_uuid());
    }
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_code(1);
  return {StatusCode::INTERNAL, absl::StrFormat("unload_replica() returned %d", rc)};
}

grpc::Status MaterializationController::WaitVerification(
    RpcContext& rctx,
    const v1::WaitReplicaVerificationRequest& req,
    v1::WaitReplicaVerificationResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    if (!req.replica_uuid().empty())
      span->SetAttribute("tc.replica.id", req.replica_uuid());
  }

  // If known terminal state, return immediately
  // Access via SessionsService (VerificationTracker behind it)
  // There is no direct getter here; fallback to session lookup then wait
  // Check known terminal state first via tracker
  if (auto known = d_.sessions.get_known(req.replica_uuid()); known.has_value()) {
    resp.set_status(known->first);
    if (!known->second.empty())
      resp.set_err_msg(known->second);
    rctx.mark_success();
    return Status::OK;
  }
  auto entry = d_.sessions.get(req.replica_uuid());
  if (!entry.has_value()) {
    resp.set_status(v1::VerificationStatus::VERIFICATION_STATUS_UNSPECIFIED);
    rctx.mark_success();
    return Status::OK;
  }
  using namespace std::chrono;
  const auto user_ms = milliseconds(req.timeout_ms() > 0 ? req.timeout_ms() : 30000);
  const auto wait_ms = ClampToDeadline(rctx.server_context(), user_ms, milliseconds(30000));
  auto st_wait = entry->ready.wait_for(wait_ms);
  if (st_wait == std::future_status::timeout) {
    return {StatusCode::DEADLINE_EXCEEDED, "verification wait timeout"};
  }
  absl::Status st = entry->ready.get();
  if (st.ok()) {
    resp.set_status(v1::VerificationStatus::VERIFICATION_STATUS_PASSED);
    d_.sessions.update_verification_status(req.replica_uuid(), v1::VerificationStatus::VERIFICATION_STATUS_PASSED);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_status(v1::VerificationStatus::VERIFICATION_STATUS_FAILED);
  resp.set_err_msg(std::string(st.message()));
  d_.sessions.update_verification_status(
      req.replica_uuid(), v1::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
  return to_grpc_status(st);
}

} // namespace tensorcast::daemon
