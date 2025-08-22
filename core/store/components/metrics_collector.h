// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "core/common/memory/memory_location.h"
#include "core/common/metrics/metric_objects.h"

namespace stepcast::store {

// Forward declarations
class ReplicaRegistry;
class DeviceManager;
class PinnedMemoryPool;

/**
 * @brief Centralized metrics collection for the store engine.
 *
 * This component handles:
 * - Memory pool metrics
 * - Replica count and size metrics
 * - Operation metrics (counters and histograms)
 * - GPU metrics
 * - P2P/RDMA transfer metrics
 */
class MetricsCollector {
 public:
  MetricsCollector();
  ~MetricsCollector() = default;

  // Disable copy and move
  MetricsCollector(const MetricsCollector&) = delete;
  MetricsCollector& operator=(const MetricsCollector&) = delete;
  MetricsCollector(MetricsCollector&&) = delete;
  MetricsCollector& operator=(MetricsCollector&&) = delete;

  /**
   * @brief Update memory pool metrics.
   * @param memory_pool Reference to the memory pool
   */
  void update_memory_pool_metrics(const PinnedMemoryPool& memory_pool);

  /**
   * @brief Update replica-related metrics.
   * @param replica_registry Replica registry to query
   */
  void update_replica_metrics(const ReplicaRegistry& replica_registry);

  /**
   * @brief Update GPU metrics.
   * @param device_manager Device manager to query
   */
  static void update_gpu_metrics(DeviceManager& device_manager);

  /**
   * @brief Record an operation.
   * @param operation_type Type of operation (e.g., "load_from_disk")
   * @param duration_seconds Duration of the operation
   */
  void record_operation(const std::string& operation_type, double duration_seconds);

  /**
   * @brief Record a P2P transfer.
   * @param bytes_transferred Number of bytes transferred
   * @param success Whether the transfer succeeded
   */
  void record_p2p_transfer(size_t bytes_transferred, bool success);

  /**
   * @brief Record a memory eviction event.
   */
  void record_memory_eviction();

  /**
   * @brief Update all metrics.
   * This is a convenience method that updates all metric types.
   */
  void update_all_metrics(
      const PinnedMemoryPool& memory_pool,
      const ReplicaRegistry& replica_registry,
      DeviceManager& device_manager);

 private:
  // Memory Pool Metrics
  stepcast::metrics::Gauge memory_pool_total_gauge_;
  stepcast::metrics::Gauge memory_pool_available_gauge_;
  stepcast::metrics::Gauge memory_pool_allocated_chunks_gauge_;

  // Replica Metrics
  stepcast::metrics::Gauge replicas_in_memory_cpu_gauge_;
  stepcast::metrics::Gauge replicas_in_memory_gpu_gauge_;
  stepcast::metrics::Gauge total_replica_size_bytes_gauge_;

  // Operation Metrics
  stepcast::metrics::Counter operations_total_counter_;
  stepcast::metrics::Histogram operation_latency_histogram_;

  // P2P/RDMA Metrics
  stepcast::metrics::Counter p2p_transfers_total_;
  stepcast::metrics::Counter p2p_bytes_transferred_total_;
  stepcast::metrics::Counter p2p_transfer_errors_total_;
  stepcast::metrics::Counter memory_evictions_total_;
};

} // namespace stepcast::store