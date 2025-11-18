// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/components/runtime/component_catalog.h"
#include "core/store/components/runtime/ingestion_events.h"
#include "core/store/components/runtime/replica_info.h"
#include "core/store/components/runtime/replica_service.h"
#include "core/store/device_types.h"
#include "core/store/replica/chunk_state.h"
#include "gsl/pointers"

namespace tensorcast::store::components::runtime {

class TelemetryService {
 public:
  struct Config {
    ComponentCatalog* component_catalog;
    ReplicaService* replica_service;
  };

  explicit TelemetryService(Config config);

  [[nodiscard]] size_t get_available_memory() const;
  void update_memory_pool_metrics();
  [[nodiscard]] std::vector<ReplicaInfo> get_all_replicas_info() const;

  [[nodiscard]] std::vector<DeviceKey> get_resident_devices(std::string_view artifact_id) const;
  [[nodiscard]] absl::StatusOr<int> get_unique_gpu_residency(std::string_view artifact_id) const;
  [[nodiscard]] std::vector<loading::ReplicaKey> list_device_replicas(const DeviceKey& device) const;

  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states_telemetry(std::string_view artifact_id) const;
  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states_for_device(
      std::string_view artifact_id,
      int device_id) const;
  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states_cpu_uma(std::string_view artifact_id) const;

  [[nodiscard]] absl::StatusOr<size_t> get_device_total_memory(int device_id) const;
  [[nodiscard]] absl::StatusOr<size_t> get_device_free_memory(int device_id) const;

  void record_ingestion_result(const IngestionResultEvent& event);

 private:
  gsl::not_null<ComponentCatalog*> component_catalog_;
  gsl::not_null<ReplicaService*> replica_service_;
};

} // namespace tensorcast::store::components::runtime
