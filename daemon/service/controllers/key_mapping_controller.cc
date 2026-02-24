// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/key_mapping_controller.h"

#include <algorithm>
#include <chrono>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon {

using status_utils::to_grpc_status;

namespace {

constexpr absl::Duration kResolveKeyMappingFallbackTimeout = absl::Seconds(5);
constexpr uint32_t kResolveKeyMappingMaxRetries = 0;

absl::StatusOr<store::components::RpcOptions> make_resolve_key_mapping_rpc_options(grpc::ServerContext& ctx) {
  store::components::RpcOptions options;
  options.max_retries = kResolveKeyMappingMaxRetries;
  options.cancel_check = [&ctx]() { return ctx.IsCancelled(); };

  const auto grpc_deadline = ctx.deadline();
  if (grpc_deadline == std::chrono::system_clock::time_point::max()) {
    options.timeout = kResolveKeyMappingFallbackTimeout;
    return options;
  }

  const auto now = std::chrono::system_clock::now();
  if (grpc_deadline <= now) {
    return absl::DeadlineExceededError("ResolveKeyMapping request deadline already exceeded");
  }

  const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(grpc_deadline - now).count();
  options.timeout = absl::Milliseconds(std::max<int64_t>(1, remaining_ms));
  return options;
}

} // namespace

std::optional<store::components::KeyMapping> KeyMappingController::lookup_local_cache(std::string_view key) const {
  if (key.empty()) {
    return std::nullopt;
  }
  const auto now = std::chrono::steady_clock::now();
  absl::MutexLock lock(&local_cache_mu_);
  auto it = local_cache_.find(std::string(key));
  if (it == local_cache_.end()) {
    return std::nullopt;
  }
  if (it->second.expires_at <= now) {
    local_cache_.erase(it);
    return std::nullopt;
  }
  return it->second.mapping;
}

void KeyMappingController::update_local_cache(
    std::string_view key,
    const store::components::KeyMapping& mapping,
    std::optional<uint32_t> ttl_override_seconds) {
  if (key.empty()) {
    return;
  }
  const uint32_t ttl_seconds = ttl_override_seconds.has_value() ? *ttl_override_seconds : mapping.cache_ttl_seconds;
  if (ttl_seconds == 0) {
    erase_local_cache(key);
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  {
    absl::MutexLock lock(&local_cache_mu_);
    auto it = local_cache_.find(std::string(key));
    if (it != local_cache_.end() && it->second.expires_at > now) {
      const uint64_t existing_generation = it->second.mapping.generation;
      const uint64_t incoming_generation = mapping.generation;
      // Guard against stale writes: never overwrite a known generation with an
      // unknown/older one.
      if (existing_generation > 0 && (incoming_generation == 0 || incoming_generation < existing_generation)) {
        return;
      }
    }
  }
  auto cached_mapping = mapping;
  cached_mapping.cache_ttl_seconds = ttl_seconds;
  LocalCacheEntry entry{
      .mapping = std::move(cached_mapping),
      .expires_at = now + std::chrono::seconds(ttl_seconds),
  };
  {
    absl::MutexLock lock(&local_cache_mu_);
    local_cache_[std::string(key)] = std::move(entry);
  }
}

void KeyMappingController::erase_local_cache(std::string_view key) {
  if (key.empty()) {
    return;
  }
  absl::MutexLock lock(&local_cache_mu_);
  local_cache_.erase(std::string(key));
}

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
  store::components::KeyMapping mapping;
  mapping.artifact_id = req.artifact_descriptor().artifact_id();
  mapping.cache_ttl_seconds = kLocalMutationCacheTtlSeconds;
  update_local_cache(req.key(), mapping, kLocalMutationCacheTtlSeconds);
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

  if (rctx.server_context().IsCancelled()) {
    return {grpc::StatusCode::CANCELLED, "request cancelled"};
  }
  if (req.key().empty()) {
    return {grpc::StatusCode::INVALID_ARGUMENT, "key is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {grpc::StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  if (auto cached = lookup_local_cache(req.key()); cached.has_value()) {
    resp.set_artifact_id(cached->artifact_id);
    resp.set_generation(cached->generation);
    resp.set_cache_ttl_seconds(cached->cache_ttl_seconds);
    rctx.mark_success();
    return grpc::Status::OK;
  }

  auto rpc_options_or = make_resolve_key_mapping_rpc_options(rctx.server_context());
  if (!rpc_options_or.ok()) {
    return to_grpc_status(rpc_options_or.status());
  }
  auto mapping_or = d_.engine.resolve_key_mapping(req.key(), *rpc_options_or);
  if (!mapping_or.ok()) {
    return to_grpc_status(mapping_or.status());
  }
  if (rctx.server_context().IsCancelled()) {
    return {grpc::StatusCode::CANCELLED, "request cancelled"};
  }
  const auto& m = *mapping_or;
  update_local_cache(req.key(), m);
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
  if (result.ok) {
    store::components::KeyMapping mapping;
    mapping.artifact_id = result.artifact_id;
    mapping.generation = result.generation;
    mapping.cache_ttl_seconds = kLocalMutationCacheTtlSeconds;
    update_local_cache(req.key(), mapping, kLocalMutationCacheTtlSeconds);
  } else {
    erase_local_cache(req.key());
  }
  resp.set_ok(result.ok);
  resp.set_artifact_id(result.artifact_id);
  resp.set_generation(result.generation);
  rctx.mark_success();
  return grpc::Status::OK;
}

} // namespace tensorcast::daemon
