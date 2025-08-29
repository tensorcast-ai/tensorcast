// Copyright (c) 2025, TensorCast Team.

#include "metrics_collector.h"

#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/replica_registry.h"
#include "core/store/memory_types.h"

namespace tensorcast::store {

MetricsCollector::MetricsCollector()
    : // Memory Pool Metrics
      memory_pool_total_gauge_("store_daemon_memory_pool_total_bytes"),
      memory_pool_available_gauge_("store_daemon_memory_pool_available_bytes"),
      memory_pool_allocated_chunks_gauge_("store_daemon_memory_pool_allocated_chunks"),
      // Replica Metrics
      replicas_in_memory_cpu_gauge_("store_daemon_replicas_in_memory", {{"location", "cpu"}}),
      replicas_in_memory_gpu_gauge_("store_daemon_replicas_in_memory", {{"location", "gpu"}}),
      total_replica_size_bytes_gauge_("store_daemon_total_replica_size_bytes"),
      // Operation Metrics
      operations_total_counter_("store_daemon_cpp_operations_total"),
      operation_latency_histogram_("store_daemon_cpp_operation_latency_seconds"),
      // P2P/RDMA Metrics
      p2p_transfers_total_("store_daemon_p2p_transfers_total"),
      p2p_bytes_transferred_total_("store_daemon_p2p_bytes_transferred_total"),
      p2p_transfer_errors_total_("store_daemon_p2p_transfer_errors_total"),
      memory_evictions_total_("store_daemon_memory_evictions_total") {
  // Initialize metrics with zero values
  memory_pool_total_gauge_.set(0.0);
  memory_pool_available_gauge_.set(0.0);
  memory_pool_allocated_chunks_gauge_.set(0.0);
  replicas_in_memory_cpu_gauge_.set(0.0);
  replicas_in_memory_gpu_gauge_.set(0.0);
  total_replica_size_bytes_gauge_.set(0.0);
  operations_total_counter_.inc(0.0);
  operation_latency_histogram_.observe(0.0);
  p2p_transfers_total_.inc(0.0);
  p2p_bytes_transferred_total_.inc(0.0);
  p2p_transfer_errors_total_.inc(0.0);
  memory_evictions_total_.inc(0.0);
}

void MetricsCollector::update_memory_pool_metrics(const PinnedMemoryPool& memory_pool) {
  // For now, we can only track available size
  size_t available_size = memory_pool.get_available_size();

  // We'll need to track total size externally or modify PinnedMemoryPool
  // For now, just update available size
  memory_pool_available_gauge_.set(static_cast<double>(available_size));
}

void MetricsCollector::update_replica_metrics(const ReplicaRegistry& replica_registry) {
  size_t cpu_models = replica_registry.get_replica_count_by_location(MemoryLocation::PAGEABLE_CPU);
  size_t gpu_models = replica_registry.get_replica_count_by_location(MemoryLocation::GPU);
  uint64_t total_size = replica_registry.get_total_replica_size();

  replicas_in_memory_cpu_gauge_.set(static_cast<double>(cpu_models));
  replicas_in_memory_gpu_gauge_.set(static_cast<double>(gpu_models));
  total_replica_size_bytes_gauge_.set(static_cast<double>(total_size));
}

void MetricsCollector::update_gpu_metrics(DeviceManager& device_manager) {
  // GPU metrics are updated within DeviceManager itself
  device_manager.update_gpu_metrics();
}

void MetricsCollector::record_operation(const std::string& operation_type, double duration_seconds) {
  operations_total_counter_.with_labels({{"operation_type", operation_type}}).inc();
  operation_latency_histogram_.with_labels({{"operation_type", operation_type}}).observe(duration_seconds);
}

void MetricsCollector::record_p2p_transfer(size_t bytes_transferred, bool success) {
  if (success) {
    p2p_transfers_total_.inc();
    p2p_bytes_transferred_total_.add(static_cast<double>(bytes_transferred));
  } else {
    p2p_transfer_errors_total_.inc();
  }
}

void MetricsCollector::record_memory_eviction() {
  memory_evictions_total_.inc();
}

void MetricsCollector::update_all_metrics(
    const PinnedMemoryPool& memory_pool,
    const ReplicaRegistry& replica_registry,
    DeviceManager& device_manager) {
  update_memory_pool_metrics(memory_pool);
  update_replica_metrics(replica_registry);
  update_gpu_metrics(device_manager);
}

} // namespace tensorcast::store