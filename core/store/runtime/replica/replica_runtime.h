// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/communication_types.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/runtime/ingestion_events.h"
#include "core/store/runtime/replica/replica_info.h"
#include "gsl/pointers"

namespace tensorcast::store::runtime {

using components::CommunicationManager;
using components::DeviceManager;
using components::MetricsCollector;
using components::ReplicaRegistry;

class ReplicaRuntime {
 public:
  struct Config {
    gsl::not_null<RuntimeContext*> runtime_context;
  };

  explicit ReplicaRuntime(gsl::not_null<RuntimeContext*> context);
  explicit ReplicaRuntime(Config config);
  ~ReplicaRuntime() = default;

  ReplicaRuntime(const ReplicaRuntime&) = delete;
  ReplicaRuntime& operator=(const ReplicaRuntime&) = delete;

  size_t get_available_memory() const;
  void update_memory_pool_metrics();
  std::vector<ReplicaInfo> get_all_replicas_info() const;
  std::vector<ReplicaInventoryEntry> get_ha_inventory() const;

  std::vector<DeviceKey> get_resident_devices(std::string_view artifact_id) const;
  absl::StatusOr<int> get_unique_gpu_residency(std::string_view artifact_id) const;
  std::vector<loading::ReplicaKey> list_device_replicas(const DeviceKey& device) const;

  int wait_replica_ready(const loading::ReplicaKey& key) const;
  int unload_replica(const loading::ReplicaKey& key) const;
  replica::MemoryState get_replica_state(const loading::ReplicaKey& key, DeviceType memory_type) const;
  absl::StatusOr<uint64_t> get_replica_gpu_ptr(const loading::ReplicaKey& key) const;
  absl::StatusOr<uint64_t> get_replica_size(const loading::ReplicaKey& key) const;

  void set_replica_publish_state(const loading::ReplicaKey& key, ReplicaPublishState state);
  ReplicaPublishState get_replica_publish_state(const loading::ReplicaKey& key) const;

  absl::StatusOr<ExportRegistration> enable_remote_replica_access(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location) const;
  absl::Status disable_remote_replica_access(const loading::ReplicaKey& key, common::memory::MemoryLocation location)
      const;

  absl::Status try_evict_memory_for_replica(size_t required_size);

  std::shared_ptr<replica::Replica> get_or_create_replica(
      const std::string& artifact_identifier,
      replica::ReplicaConfig config);

  int clear_mem();

  std::vector<replica::ChunkState> get_chunk_states_telemetry(std::string_view artifact_id) const;
  std::vector<replica::ChunkState> get_chunk_states_for_device(std::string_view artifact_id, int device_id) const;
  std::vector<replica::ChunkState> get_chunk_states_cpu_uma(std::string_view artifact_id) const;
  absl::StatusOr<size_t> get_device_total_memory(int device_id) const;
  absl::StatusOr<size_t> get_device_free_memory(int device_id) const;
  void record_ingestion_result(const IngestionResultEvent& event);

  ReplicaRegistry& registry();
  const ReplicaRegistry& registry() const;

  DeviceManager& device_manager();
  const DeviceManager& device_manager() const;

  std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool() const;
  std::shared_ptr<CommunicationManager> communication_manager() const;
  MetricsCollector& metrics();
  const MetricsCollector& metrics() const;

 private:
  void publish_replica_event(RuntimeEventType type, const loading::ReplicaKey& key, size_t size_bytes) const;
  void publish_remote_access_event(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location,
      bool enabled) const;
  size_t get_replica_size_or_zero(const loading::ReplicaKey& key) const;

  gsl::not_null<RuntimeContext*> context_;
  RuntimeContextEvents::Publisher event_publisher_;
  std::unique_ptr<RuntimeContextEvents::Subscription> ingestion_event_subscription_;
  mutable absl::Mutex publish_state_mu_;
  absl::flat_hash_map<loading::ReplicaKey, ReplicaPublishState, loading::ReplicaKeyHash> publish_states_
      ABSL_GUARDED_BY(publish_state_mu_);
};

} // namespace tensorcast::store::runtime
