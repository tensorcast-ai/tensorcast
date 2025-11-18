// Copyright (c) 2025, TensorCast Team.

#include "core/store/components/runtime/telemetry_service.h"

#include <string>

#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"

namespace tensorcast::store::components::runtime {

TelemetryService::TelemetryService(Config config)
    : component_catalog_(gsl::not_null<ComponentCatalog*>{config.component_catalog}),
      replica_service_(gsl::not_null<ReplicaService*>{config.replica_service}) {}

size_t TelemetryService::get_available_memory() const {
  return replica_service_->get_available_memory();
}

void TelemetryService::update_memory_pool_metrics() {
  replica_service_->update_memory_pool_metrics();
}

std::vector<ReplicaInfo> TelemetryService::get_all_replicas_info() const {
  return replica_service_->get_all_replicas_info();
}

std::vector<DeviceKey> TelemetryService::get_resident_devices(std::string_view artifact_id) const {
  return replica_service_->get_resident_devices(artifact_id);
}

absl::StatusOr<int> TelemetryService::get_unique_gpu_residency(std::string_view artifact_id) const {
  int unique_gpu_device = -2; // -2: unknown, -1: none, >=0: unique device
  for (const auto& info : get_all_replicas_info()) {
    if (info.artifact_id == artifact_id && info.gpu_state == common::memory::MemoryLocation::GPU) {
      if (unique_gpu_device == -2) {
        unique_gpu_device = info.gpu_device_id;
      } else if (unique_gpu_device != info.gpu_device_id) {
        return absl::InvalidArgumentError("ambiguous artifact residency across multiple GPUs; device_id required");
      }
    }
  }
  if (unique_gpu_device == -2) {
    return -1;
  }
  return unique_gpu_device;
}

std::vector<loading::ReplicaKey> TelemetryService::list_device_replicas(const DeviceKey& device) const {
  return replica_service_->list_device_replicas(device);
}

std::vector<replica::ChunkState> TelemetryService::get_chunk_states_telemetry(std::string_view artifact_id) const {
  return replica_service_->get_chunk_states_telemetry(artifact_id);
}

std::vector<replica::ChunkState> TelemetryService::get_chunk_states_for_device(
    std::string_view artifact_id,
    int device_id) const {
  return replica_service_->get_chunk_states_for_device(artifact_id, device_id);
}

std::vector<replica::ChunkState> TelemetryService::get_chunk_states_cpu_uma(std::string_view artifact_id) const {
  return replica_service_->get_chunk_states_cpu_uma(artifact_id);
}

absl::StatusOr<size_t> TelemetryService::get_device_total_memory(int device_id) const {
  auto info_or = replica_service_->device_manager().get_gpu_info(device_id);
  if (!info_or.ok()) {
    return info_or.status();
  }
  return static_cast<size_t>((*info_or)->total_memory);
}

absl::StatusOr<size_t> TelemetryService::get_device_free_memory(int device_id) const {
  return replica_service_->device_manager().get_free_memory(device_id);
}

void TelemetryService::record_ingestion_result(const IngestionResultEvent& event) {
  auto& metrics = component_catalog_->metrics_collector();
  const bool success = event.status.ok();
  const bool from_p2p = event.source == IngestionSource::kP2P;

  if (from_p2p) {
    metrics.record_p2p_transfer(success ? event.bytes_transferred : 0, success);
  }

  metrics.record_operation(from_p2p ? "load_from_p2p" : "load_from_disk", event.duration_seconds);

  const std::string device_scope = (event.target_device.type == DeviceType::GPU) ? "gpu" : "cpu";
  std::optional<std::string_view> view_scope;
  if (event.view_id.has_value()) {
    view_scope = std::string_view(*event.view_id);
  }
  metrics.record_artifact_load(
      from_p2p ? "remote" : "disk", device_scope, success ? "finalize" : "error", event.duration_seconds, view_scope);

  if (success) {
    replica_service_->update_memory_pool_metrics();
  }
}

} // namespace tensorcast::store::components::runtime
