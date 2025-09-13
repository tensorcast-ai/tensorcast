// Copyright (c) 2025, TensorCast Team.

// Implementation of TransportController

#include "daemon/service/controllers/transport_controller.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/device_registry.h"
#include "daemon/status_utils.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using status_utils::to_grpc_status;

grpc::Status TransportController::lock(
    RpcContext& rctx,
    const v1::LockTransportChunksRequest& req,
    v1::LockTransportChunksResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.artifact_id());
  if (req.has_device_id())
    span->SetAttribute("tc.device.id", static_cast<int64_t>(req.device_id()));

  store::loading::ReplicaKey key;
  key.artifact_id = req.artifact_id();
  key.replica = 0;
  // Resolve device lazily to avoid defaulting to GPU0 prematurely
  if (req.has_device_id()) {
    key.device = store::DeviceRegistry::instance().gpu_key(req.device_id());
  }

  // LIP staged export path
  if (auto lip_opt = d_.lip.find_active_by_artifact_id(req.artifact_id()); lip_opt.has_value()) {
    const auto& lip = *lip_opt;
    std::vector<uint32_t> indices(req.chunk_indices().begin(), req.chunk_indices().end());
    auto tok_or = d_.lip.create_staged_export(lip, absl::MakeSpan(indices), d_.engine);
    if (!tok_or.ok())
      return to_grpc_status(tok_or.status());
    resp.set_lock_token(*tok_or);
    if (!lip.verification_json.empty())
      resp.set_verification_json(lip.verification_json);
    rctx.mark_success();
    return Status::OK;
  }

  if (!req.has_device_id()) {
    auto dev_or = d_.engine.get_unique_gpu_residency(req.artifact_id());
    if (!dev_or.ok())
      return to_grpc_status(dev_or.status());
    if (*dev_or >= 0) {
      key.device = store::DeviceRegistry::instance().gpu_key(*dev_or);
    } else {
      // No GPU residency found; do not default to GPU0. Require explicit device_id or pre-load.
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "artifact not resident on any GPU; specify device_id or load");
    }
  }

  // UMA V3: No engine-level transfer locks; UMA plan/commit supersedes any prior locking.
  // Maintain a daemon-local token workflow strictly for bookkeeping.
  std::vector<uint32_t> indices(req.chunk_indices().begin(), req.chunk_indices().end());
  std::string token = d_.locks.mint_token();
  d_.locks.put(token, key, std::move(indices));
  resp.set_lock_token(token);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TransportController::unlock(
    RpcContext& rctx,
    const v1::UnlockTransportChunksRequest& req,
    v1::UnlockTransportChunksResponse& /*resp*/) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs())
    span->SetAttribute("tc.lock.token", req.lock_token());
  auto entry = d_.locks.get(req.lock_token());
  if (!entry.has_value()) {
    // Try staged LIP export unlock; if not found there either, treat as idempotent success.
    auto st = d_.lip.release_staged_export(req.lock_token(), d_.engine);
    if (!st.ok()) {
      // If unknown token in LIP exports, do not fail; return OK for idempotency
      rctx.mark_success();
      return Status::OK;
    }
    rctx.mark_success();
    return Status::OK;
  }
  // UMA V3: No engine-level unlock; treat daemon unlock as idempotent bookkeeping: just erase the token.
  d_.locks.erase(req.lock_token());
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
