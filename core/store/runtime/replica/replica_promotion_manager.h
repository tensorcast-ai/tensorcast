// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/communication_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica/replica_info.h"
#include "core/store/runtime/replica/replica_runtime.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime {

struct PromotionSyncHooks {
  std::function<void()> request_state_sync;
};

class ReplicaPromotionManager {
 public:
  struct Config {
    gsl::not_null<RuntimeContext*> runtime_context;
    gsl::not_null<ReplicaRuntime*> replica_runtime;
  };

  explicit ReplicaPromotionManager(Config config);
  ~ReplicaPromotionManager() = default;

  ReplicaPromotionManager(const ReplicaPromotionManager&) = delete;
  ReplicaPromotionManager& operator=(const ReplicaPromotionManager&) = delete;

  void set_sync_hooks(PromotionSyncHooks hooks);

  void handle_ingestion_completed(const IngestionCompletedEvent& event);

  absl::Status mark_replica_draining(const loading::ReplicaKey& key);
  absl::Status finalize_replica_demotion(const loading::ReplicaKey& key);

  absl::Status record_export_registration(
      const loading::ReplicaKey& key,
      const ExportRegistration& registration,
      const std::optional<std::string>& verification_json);

 private:
  struct PromotionDecision {
    bool allow{false};
    std::string reason;
  };

  PromotionDecision evaluate_promotion(loading::ExportPolicy request_policy) const;

  absl::Status promote_replica(const loading::ReplicaKey& key, loading::ExportPolicy request_policy);
  void request_state_sync();
  absl::StatusOr<std::string> generate_verification_json(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location) const;

  static common::memory::MemoryLocation location_for_key(const loading::ReplicaKey& key);

  gsl::not_null<RuntimeContext*> runtime_context_;
  gsl::not_null<ReplicaRuntime*> replica_runtime_;

  PromotionSyncHooks sync_hooks_;
  std::unique_ptr<RuntimeContextEvents::Subscription> ingestion_subscription_;
};

} // namespace tensorcast::store::runtime
