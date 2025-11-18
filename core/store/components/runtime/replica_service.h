// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/communication_types.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/metrics_collector.h"
#include "core/store/components/replica_registry.h"
#include "core/store/components/runtime/component_catalog.h"
#include "core/store/components/runtime/replica_info.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica.h"
#include "gsl/pointers"

namespace tensorcast::store::components::runtime {

class ReplicaService {
 public:
  explicit ReplicaService(gsl::not_null<ComponentCatalog*> catalog);
  ~ReplicaService() = default;

  ReplicaService(const ReplicaService&) = delete;
  ReplicaService& operator=(const ReplicaService&) = delete;

  size_t get_available_memory() const;
  void update_memory_pool_metrics();
  std::vector<ReplicaInfo> get_all_replicas_info() const;

  std::vector<DeviceKey> get_resident_devices(std::string_view artifact_id) const;
  std::vector<loading::ReplicaKey> list_device_replicas(const DeviceKey& device) const;

  int wait_replica_ready(const loading::ReplicaKey& key) const;
  int unload_replica(const loading::ReplicaKey& key) const;
  replica::MemoryState get_replica_state(const loading::ReplicaKey& key, DeviceType memory_type) const;
  absl::StatusOr<uint64_t> get_replica_gpu_ptr(const loading::ReplicaKey& key) const;
  absl::StatusOr<uint64_t> get_replica_size(const loading::ReplicaKey& key) const;

  absl::StatusOr<ExportRegistration> enable_remote_replica_access(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location) const;
  absl::Status disable_remote_replica_access(const loading::ReplicaKey& key, common::memory::MemoryLocation location)
      const;

  absl::Status try_evict_memory_for_replica(size_t required_size);

  std::shared_ptr<replica::Replica> get_or_create_replica(
      const std::string& artifact_identifier,
      const replica::ReplicaConfig& config);

  int clear_mem();

  std::vector<replica::ChunkState> get_chunk_states_telemetry(std::string_view artifact_id) const;
  std::vector<replica::ChunkState> get_chunk_states_for_device(std::string_view artifact_id, int device_id) const;
  std::vector<replica::ChunkState> get_chunk_states_cpu_uma(std::string_view artifact_id) const;

  ReplicaRegistry& registry();
  const ReplicaRegistry& registry() const;

  DeviceManager& device_manager();
  const DeviceManager& device_manager() const;

  std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool() const;
  std::shared_ptr<CommunicationManager> communication_manager() const;
  MetricsCollector& metrics();
  const MetricsCollector& metrics() const;

 private:
  gsl::not_null<ComponentCatalog*> catalog_;
};

} // namespace tensorcast::store::components::runtime
