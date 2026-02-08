// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/replica_lifecycle_service.h"

#include <chrono>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "daemon/util/deadline_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

ReplicaLifecycleService::ReplicaLifecycleService(Dep d) : d_(std::move(d)) {}

grpc::Status ReplicaLifecycleService::confirm(
    RpcContext& rctx,
    const v2::ConfirmReplicaRequest& req,
    v2::ConfirmReplicaResponse& resp) const {
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
    // Parity: unknown replica_uuid -> code=0 OK.
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  // Wait bounded by gRPC deadline with a 30s hard cap (Confirm has no user timeout).
  using namespace std::chrono;
  const auto wait_ms = ClampToDeadline(rctx.server_context(), milliseconds(30000), milliseconds(30000));
  const absl::Status st = entry->wait_ready(wait_ms);
  if (absl::IsDeadlineExceeded(st)) {
    return {StatusCode::DEADLINE_EXCEEDED, "confirm timeout"};
  }
  if (st.ok()) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_code(1);
  return to_grpc_status(st);
}

grpc::Status ReplicaLifecycleService::unload(
    RpcContext& rctx,
    const v2::UnloadReplicaRequest& req,
    v2::UnloadReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    if (!req.disk_path().empty()) {
      span->SetAttribute("tc.disk.path", req.disk_path());
    }
    if (req.has_pid()) {
      span->SetAttribute("tc.pid", static_cast<int64_t>(req.pid()));
    }
  }
  resp.set_disk_path(req.disk_path());

  if (req.target_device_type() == v2::DeviceType::DEVICE_TYPE_DISK) {
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }

  store::loading::ReplicaKey key;
  if (!req.replica_uuid().empty()) {
    auto entry = d_.sessions.get(req.replica_uuid());
    if (entry.has_value()) {
      key = entry->key;
    }
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
    bool lease_released = false;
    if (d_.lifecycle) {
      const auto status = d_.lifecycle->release_use_lease(key, req.pid());
      if (status.ok()) {
        lease_released = true;
      } else if (absl::IsNotFound(status)) {
        VLOG(1) << "release_use_lease not found: artifact_id=" << key.artifact_id << " dev=" << key.device.ordinal
                << " pid=" << req.pid() << " replica_uuid=" << req.replica_uuid();
      } else {
        LOG(WARNING) << "failed to release use lease: artifact_id=" << key.artifact_id << " dev=" << key.device.ordinal
                     << " pid=" << req.pid() << " replica_uuid=" << req.replica_uuid() << " status=" << status;
      }
    }
    if (!lease_released) {
      d_.refs.drop_ref(key, req.pid());
    }
    if (d_.refs.ref_count(key) > 0) {
      resp.set_code(0);
      rctx.mark_success();
      return Status::OK;
    }
  }
  const absl::Status unload_status = d_.engine.unload_replica_status(key);
  if (unload_status.ok()) {
    if (!req.replica_uuid().empty()) {
      const bool erased = d_.sessions.erase(req.replica_uuid());
      if (!erased) {
        VLOG(2) << "unload: session not found for replica_uuid=" << req.replica_uuid();
      }
    }
    resp.set_code(0);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_code(1);
  return to_grpc_status(unload_status);
}

grpc::Status ReplicaLifecycleService::wait_verification(
    RpcContext& rctx,
    const v2::WaitReplicaVerificationRequest& req,
    v2::WaitReplicaVerificationResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs()) {
    if (!req.replica_uuid().empty()) {
      span->SetAttribute("tc.replica.id", req.replica_uuid());
    }
  }

  // If known terminal state, return immediately.
  if (auto known = d_.sessions.get_known(req.replica_uuid()); known.has_value()) {
    resp.set_status(known->first);
    if (!known->second.empty()) {
      resp.set_err_msg(known->second);
    }
    rctx.mark_success();
    return Status::OK;
  }
  auto entry = d_.sessions.get(req.replica_uuid());
  if (!entry.has_value()) {
    resp.set_status(v2::VerificationStatus::VERIFICATION_STATUS_UNSPECIFIED);
    rctx.mark_success();
    return Status::OK;
  }
  using namespace std::chrono;
  const auto user_ms = milliseconds(req.timeout_ms() > 0 ? req.timeout_ms() : 30000);
  const auto wait_ms = ClampToDeadline(rctx.server_context(), user_ms, milliseconds(30000));
  const absl::Status st = entry->wait_ready(wait_ms);
  if (absl::IsDeadlineExceeded(st)) {
    return {StatusCode::DEADLINE_EXCEEDED, "verification wait timeout"};
  }
  if (st.ok()) {
    resp.set_status(v2::VerificationStatus::VERIFICATION_STATUS_PASSED);
    d_.sessions.update_verification_status(req.replica_uuid(), v2::VerificationStatus::VERIFICATION_STATUS_PASSED);
    rctx.mark_success();
    return Status::OK;
  }
  resp.set_status(v2::VerificationStatus::VERIFICATION_STATUS_FAILED);
  resp.set_err_msg(std::string(st.message()));
  d_.sessions.update_verification_status(
      req.replica_uuid(), v2::VerificationStatus::VERIFICATION_STATUS_FAILED, std::string(st.message()));
  return to_grpc_status(st);
}

} // namespace tensorcast::daemon
