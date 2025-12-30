// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/runtime/metadata/metadata_types.h"

namespace tensorcast::store::runtime::metadata {

class RegistrationBackend {
 public:
  RegistrationBackend(
      RegistrationResources resources,
      ReplicaFactory replica_factory,
      size_t artifact_chunk_bytes,
      std::chrono::milliseconds pinned_memory_timeout,
      RegistrationPublisher* publisher);

  RegistrationBackend(const RegistrationBackend&) = delete;
  RegistrationBackend& operator=(const RegistrationBackend&) = delete;

  absl::StatusOr<RegistrationBeginResult> begin(const ArtifactRegistration& reg);
  absl::StatusOr<RegistrationCommitResult> commit(std::string_view registration_id);
  absl::Status abort(std::string_view registration_id);
  absl::Status keep_alive(std::string_view registration_id, uint32_t ttl_ms);
  absl::Status ingest_view_chunk(
      std::string_view registration_id,
      uint64_t view_offset,
      absl::Span<const std::byte> data);
  absl::StatusOr<uint64_t> get_view_ingested_bytes(std::string_view registration_id) const;
  absl::StatusOr<uint64_t> get_registration_gpu_ptr(std::string_view registration_id) const;

 private:
  struct PendingRegistrationContext;

  std::shared_ptr<PendingRegistrationContext> erase_pending(
      std::string_view registration_id,
      size_t* pending_size_after = nullptr);
  std::shared_ptr<PendingRegistrationContext> lookup_pending(std::string_view registration_id) const;
  static void release_replica_memory(
      const std::shared_ptr<replica::Replica>& replica,
      common::memory::MemoryLocation location);
  void record_pending_gauge(size_t pending_count) const;
  void record_commit_latency(const PendingRegistrationContext& ctx, std::string_view status) const;

  gsl::not_null<components::DeviceManager*> device_manager_;
  gsl::not_null<components::ReplicaRegistry*> replica_registry_;
  gsl::not_null<components::MetricsCollector*> metrics_collector_;
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool_;
  std::shared_ptr<components::CommunicationManager> communication_manager_;
  std::shared_ptr<components::StableDramCacheManager> stable_cache_manager_;
  std::shared_ptr<common::AsyncRuntime> async_runtime_;
  std::shared_ptr<MemoryTierBudget> memory_tier_budget_;
  std::optional<MemoryTierConfig> memory_tier_config_;
  ReplicaFactory replica_factory_;
  size_t artifact_chunk_bytes_{0};
  std::chrono::milliseconds pinned_memory_timeout_{0};
  RegistrationPublisher* publisher_;

  mutable std::mutex pending_mutex_;
  absl::flat_hash_map<std::string, std::shared_ptr<PendingRegistrationContext>> pending_regs_;
};

} // namespace tensorcast::store::runtime::metadata
