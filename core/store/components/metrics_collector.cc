// Copyright (c) 2025, TensorCast Team.

#include "metrics_collector.h"

#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/replica_registry.h"
#include "core/store/memory_types.h"

namespace tensorcast::store {

MetricsCollector::MetricsCollector()
    : tc_memory_pool_bytes_cpu_available_("tc_memory_pool_bytes", {{"location", "cpu"}, {"memory_type", "available"}}),
      tc_p2p_bytes_total_("tc_p2p_bytes_total"),
      tc_artifact_load_seconds_("tc_artifact_load_seconds") {
  // Initialize tc_* metrics with zero values
  tc_memory_pool_bytes_cpu_available_.set(0.0);
}

void MetricsCollector::update_memory_pool_metrics(const PinnedMemoryPool& memory_pool) {
  // Track available size in unified gauge only
  size_t available_size = memory_pool.get_available_size();
  tc_memory_pool_bytes_cpu_available_.set(static_cast<double>(available_size));
}

void MetricsCollector::update_replica_metrics(const ReplicaRegistry& replica_registry) {
  // Phase 5: converge on tc_*; replica counters moved out of C++ daemon.
  (void)replica_registry; // no-op
}

void MetricsCollector::update_gpu_metrics(DeviceManager& device_manager) {
  // GPU metrics are updated within DeviceManager itself
  device_manager.update_gpu_metrics();
}

void MetricsCollector::record_operation(const std::string& operation_type, double duration_seconds) {
  // Phase 5: legacy operation metrics removed; rely on record_artifact_load
  (void)operation_type;
  (void)duration_seconds;
}

void MetricsCollector::record_p2p_transfer(size_t bytes_transferred, bool success) {
  if (success) {
    // Unified counter for P2P throughput only
    tc_p2p_bytes_total_.add(static_cast<double>(bytes_transferred));
  }
}

void MetricsCollector::record_memory_eviction() {
  // Phase 5: evictions no longer reported from daemon; no-op
}

void MetricsCollector::update_all_metrics(
    const PinnedMemoryPool& memory_pool,
    const ReplicaRegistry& replica_registry,
    DeviceManager& device_manager) {
  update_memory_pool_metrics(memory_pool);
  update_replica_metrics(replica_registry);
  update_gpu_metrics(device_manager);
}

void MetricsCollector::record_artifact_load(
    const std::string& source,
    const std::string& device,
    const std::string& phase,
    double duration_seconds) {
  // Record into unified histogram with low-cardinality labels
  tc_artifact_load_seconds_.with_labels({{"source", source}, {"device", device}, {"phase", phase}})
      .observe(duration_seconds);
}

} // namespace tensorcast::store
