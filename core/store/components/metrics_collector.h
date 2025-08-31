// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "core/common/memory/memory_location.h"
#include "core/common/metrics/metric_objects.h"

namespace tensorcast::store {

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
   * @brief Record artifact load latency with unified labels.
   * @param source Source of data (e.g., "remote" or "disk")
   * @param device Target device ("gpu" or "cpu")
   * @param phase  Pipeline phase (e.g., "finalize")
   * @param duration_seconds Duration in seconds
   */
  void record_artifact_load(
      const std::string& source,
      const std::string& device,
      const std::string& phase,
      double duration_seconds);

  /**
   * @brief Update all metrics.
   * This is a convenience method that updates all metric types.
   */
  void update_all_metrics(
      const PinnedMemoryPool& memory_pool,
      const ReplicaRegistry& replica_registry,
      DeviceManager& device_manager);

 private:
  // Unified tc_* metrics (Phase 3)
  tensorcast::metrics::Gauge tc_memory_pool_bytes_cpu_available_;
  tensorcast::metrics::Counter tc_p2p_bytes_total_;
  tensorcast::metrics::Histogram tc_artifact_load_seconds_;
};

} // namespace tensorcast::store
