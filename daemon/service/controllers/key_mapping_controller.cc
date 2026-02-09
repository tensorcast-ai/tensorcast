// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/key_mapping_controller.h"

#include <optional>

#include "absl/status/status.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using status_utils::to_grpc_status;

grpc::Status KeyMappingController::publish_replica_key(
    RpcContext& rctx,
    const v2::PublishReplicaKeyRequest& req,
    v2::PublishReplicaKeyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req.key());

  if (req.key().empty() || !req.has_artifact_descriptor() || req.artifact_descriptor().artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key and artifact_descriptor.artifact_id are required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto up = d_.engine.upsert_key_mapping(req.key(), req.artifact_descriptor().artifact_id());
  if (!up.ok()) {
    if (absl::IsAlreadyExists(up)) {
      resp.set_ok(false);
      resp.set_conflict_reason(std::string(up.message()));
      rctx.mark_success();
      return grpc::Status::OK;
    }
    return to_grpc_status(up);
  }
  resp.set_ok(true);
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status KeyMappingController::resolve_key_mapping(
    RpcContext& rctx,
    const v2::ResolveKeyMappingRequest& req,
    v2::ResolveKeyMappingResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req.key());

  if (req.key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto mapping_or = d_.engine.resolve_key_mapping(req.key());
  if (!mapping_or.ok()) {
    return to_grpc_status(mapping_or.status());
  }
  const auto& m = *mapping_or;
  resp.set_artifact_id(m.artifact_id);
  resp.set_generation(m.generation);
  resp.set_cache_ttl_seconds(m.cache_ttl_seconds);
  rctx.mark_success();
  return grpc::Status::OK;
}

grpc::Status KeyMappingController::swap_key_mapping(
    RpcContext& rctx,
    const v2::SwapKeyMappingRequest& req,
    v2::SwapKeyMappingResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.key", req.key());
  if (req.key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (req.new_artifact_id().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "new_artifact_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  std::optional<std::string_view> expected_artifact_id;
  if (!req.expected_artifact_id().empty()) {
    expected_artifact_id = req.expected_artifact_id();
  }
  std::optional<uint64_t> expected_generation;
  if (req.has_expected_generation()) {
    expected_generation = req.expected_generation();
  }

  auto result_or =
      d_.engine.swap_key_mapping(req.key(), req.new_artifact_id(), expected_artifact_id, expected_generation);
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  const auto& result = *result_or;
  resp.set_ok(result.ok);
  resp.set_artifact_id(result.artifact_id);
  resp.set_generation(result.generation);
  rctx.mark_success();
  return grpc::Status::OK;
}

} // namespace tensorcast::daemon
