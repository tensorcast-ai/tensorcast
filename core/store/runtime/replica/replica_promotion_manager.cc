// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/replica/replica_promotion_manager.h"

#include <algorithm>
#include <utility>

#include "absl/status/status.h"

namespace tensorcast::store::runtime {

ReplicaPromotionManager::ReplicaPromotionManager(Config config)
    : runtime_context_(config.runtime_context), replica_runtime_(config.replica_runtime) {
  auto* hub = runtime_context_->ingestion_event_hub();
  if (hub != nullptr) {
    ingestion_subscription_ =
        hub->subscribe_completed([this](const IngestionCompletedEvent& event) { handle_ingestion_completed(event); });
  }
}

void ReplicaPromotionManager::set_sync_hooks(PromotionSyncHooks hooks) {
  sync_hooks_ = std::move(hooks);
}

void ReplicaPromotionManager::handle_ingestion_completed(const IngestionCompletedEvent& event) {
  if (!event.status.ok()) {
    return;
  }
  if (!event.replica_key.has_value()) {
    return;
  }
  if (event.export_policy == loading::ExportPolicy::kNever) {
    return;
  }

  const loading::ReplicaKey key = *event.replica_key;
  const loading::ExportPolicy request_policy = event.export_policy;
  auto runtime = runtime_context_->async_runtime();
  if (runtime && !runtime->is_shutting_down()) {
    auto executor = runtime->blocking_executor();
    executor->add([this, key, request_policy]() {
      auto st = promote_replica(key, request_policy);
      if (!st.ok()) {
        LOG(WARNING) << "ReplicaPromotionManager promote failed for " << key << ": " << st;
      }
    });
    return;
  }
  auto st = promote_replica(key, request_policy);
  if (!st.ok()) {
    LOG(WARNING) << "ReplicaPromotionManager promote failed for " << key << ": " << st;
  }
}

absl::Status ReplicaPromotionManager::mark_replica_draining(const loading::ReplicaKey& key) {
  const auto current = replica_runtime_->get_transport_state(key);
  if (current.export_state == ReplicaExportState::kPresenceOnly) {
    return absl::OkStatus();
  }
  if (current.export_state == ReplicaExportState::kDraining) {
    request_state_sync();
    return absl::OkStatus();
  }
  auto next = current;
  next.export_state = ReplicaExportState::kDraining;
  replica_runtime_->update_transport_state(key, next);
  request_state_sync();
  return absl::OkStatus();
}

absl::Status ReplicaPromotionManager::finalize_replica_demotion(const loading::ReplicaKey& key) {
  auto state = replica_runtime_->get_transport_state(key);
  if (state.export_state == ReplicaExportState::kPresenceOnly && state.remote_memory_keys.empty()) {
    return absl::OkStatus();
  }

  const auto location = location_for_key(key);
  absl::Status unexport_status = replica_runtime_->disable_remote_replica_access(key, location);
  if (!unexport_status.ok()) {
    LOG(WARNING) << "disable_remote_replica_access failed for " << key << ": " << unexport_status;
  }

  state.export_state = ReplicaExportState::kPresenceOnly;
  state.remote_memory_keys.clear();
  state.buffer_sizes.clear();
  state.verification_json.clear();
  replica_runtime_->update_transport_state(key, state);

  request_state_sync();
  return unexport_status.ok() ? absl::OkStatus() : unexport_status;
}

absl::Status ReplicaPromotionManager::record_export_registration(
    const loading::ReplicaKey& key,
    const ExportRegistration& registration,
    const std::optional<std::string>& verification_json) {
  if (registration.remote_memory_keys.empty()) {
    return absl::InvalidArgumentError("registration missing remote_memory_keys");
  }
  if (registration.remote_memory_keys.size() != registration.buffer_sizes.size()) {
    return absl::InvalidArgumentError("registration buffer_sizes mismatch");
  }

  auto state = replica_runtime_->get_transport_state(key);
  const uint64_t next_generation = state.export_generation + 1;
  state.export_state = ReplicaExportState::kExportable;
  state.export_generation = std::max<uint64_t>(1, next_generation);
  state.remote_memory_keys = registration.remote_memory_keys;
  state.buffer_sizes.clear();
  state.buffer_sizes.reserve(registration.buffer_sizes.size());
  for (const auto size : registration.buffer_sizes) {
    state.buffer_sizes.push_back(static_cast<uint64_t>(size));
  }
  if (verification_json.has_value() && !verification_json->empty()) {
    state.verification_json = *verification_json;
  }
  replica_runtime_->update_transport_state(key, state);

  request_state_sync();
  return absl::OkStatus();
}

ReplicaPromotionManager::PromotionDecision ReplicaPromotionManager::evaluate_promotion(
    loading::ExportPolicy request_policy) const {
  const auto& options = runtime_context_->options().promotion;
  if (options.policy == StoreEngineOptions::PromotionPolicy::kNever) {
    return {.allow = false, .reason = "policy=never"};
  }
  if (request_policy == loading::ExportPolicy::kNever) {
    return {.allow = false, .reason = "request=never"};
  }
  if (options.policy == StoreEngineOptions::PromotionPolicy::kOnHotness &&
      request_policy != loading::ExportPolicy::kForce) {
    return {.allow = false, .reason = "policy=on_hotness requires force"};
  }
  return {.allow = true, .reason = "allowed"};
}

absl::Status ReplicaPromotionManager::promote_replica(
    const loading::ReplicaKey& key,
    loading::ExportPolicy request_policy) {
  const auto comm = runtime_context_->communication_manager();
  if (!comm || !comm->is_enabled()) {
    return absl::FailedPreconditionError("communication disabled");
  }

  auto size_or = replica_runtime_->get_replica_size(key);
  if (!size_or.ok()) {
    return size_or.status();
  }
  const uint64_t replica_bytes = *size_or;
  const auto location = location_for_key(key);
  const auto decision = evaluate_promotion(request_policy);
  if (!decision.allow) {
    VLOG(1) << "Promotion skipped for " << key << ": " << decision.reason;
    return absl::OkStatus();
  }

  std::optional<std::string> verification_json;
  const auto& options = runtime_context_->options().promotion;
  if (options.require_verified) {
    const auto existing_state = replica_runtime_->get_transport_state(key);
    if (!existing_state.verification_json.empty()) {
      verification_json = existing_state.verification_json;
    } else {
      auto ver_or = generate_verification_json(key, location);
      if (!ver_or.ok()) {
        return ver_or.status();
      }
      verification_json = *ver_or;
    }
  }

  auto export_or = replica_runtime_->enable_remote_replica_access(key, location);
  if (!export_or.ok()) {
    return export_or.status();
  }

  auto state = replica_runtime_->get_transport_state(key);
  const uint64_t next_generation = state.export_generation + 1;
  state.export_state = ReplicaExportState::kExportable;
  state.export_generation = std::max<uint64_t>(1, next_generation);
  state.remote_memory_keys = export_or->remote_memory_keys;
  state.buffer_sizes.clear();
  state.buffer_sizes.reserve(export_or->buffer_sizes.size());
  for (const auto size : export_or->buffer_sizes) {
    state.buffer_sizes.push_back(static_cast<uint64_t>(size));
  }
  if (verification_json.has_value() && !verification_json->empty()) {
    state.verification_json = *verification_json;
  }
  replica_runtime_->update_transport_state(key, state);

  request_state_sync();
  return absl::OkStatus();
}

void ReplicaPromotionManager::request_state_sync() {
  if (sync_hooks_.request_state_sync) {
    sync_hooks_.request_state_sync();
  }
}

absl::StatusOr<std::string> ReplicaPromotionManager::generate_verification_json(
    const loading::ReplicaKey& key,
    common::memory::MemoryLocation location) const {
  auto replica_or = replica_runtime_->registry().find(key);
  if (!replica_or.ok()) {
    return replica_or.status();
  }
  const auto& replica = replica_or.value();
  auto info_or = replica->generate_verification_info(location);
  if (!info_or.ok()) {
    return info_or.status();
  }
  return info_or->to_json();
}

common::memory::MemoryLocation ReplicaPromotionManager::location_for_key(const loading::ReplicaKey& key) {
  if (key.device.type == DeviceType::GPU) {
    return common::memory::MemoryLocation::GPU;
  }
  return common::memory::MemoryLocation::CPU;
}

} // namespace tensorcast::store::runtime
