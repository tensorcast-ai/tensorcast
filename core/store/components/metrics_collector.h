// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// OpenTelemetry Metrics API (types used in member declarations)
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/pinned_memory_authority.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/observer_result.h"
// Prefer explicit includes over forward declarations
#include "core/store/components/device_manager.h"
#include "core/store/components/replica_registry.h"

namespace tensorcast::store::components {

// Note: PinnedBufferPool lives under tensorcast::common::memory

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
  struct P2PTransferSnapshot {
    std::uint64_t total_transfers{0};
    std::uint64_t total_bytes_transferred{0};
    std::uint64_t total_transfer_errors{0};
  };

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
  void update_memory_pool_metrics(const common::memory::PinnedBufferPool& memory_pool);

  /**
   * @brief Update daemon-wide pinned memory authority metrics.
   * @param authority Reference to the pinned memory authority
   */
  void update_pinned_authority_metrics(const common::memory::PinnedMemoryAuthority& authority);

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

  [[nodiscard]] P2PTransferSnapshot get_p2p_transfer_snapshot() const;

  /**
   * @brief Record the number of in-flight registrations managed by ARM.
   */
  void record_registration_pending(size_t pending_count);

  /**
   * @brief Record the latency of registration commits along with the result state.
   */
  void record_registration_commit(double duration_seconds, std::string_view result);

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
      double duration_seconds,
      std::optional<std::string_view> view_id = std::nullopt);

  /**
   * @brief Update all metrics.
   * This is a convenience method that updates all metric types.
   */
  void update_all_metrics(
      const tensorcast::common::memory::PinnedBufferPool& memory_pool,
      const ReplicaRegistry& replica_registry,
      DeviceManager& device_manager);

 private:
  // ObservableGauge callback trampoline
  static void cpu_mem_available_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void registration_pending_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_class_capacity_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_class_in_use_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_class_free_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_class_waiters_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_class_timeouts_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_total_bytes_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_committed_bytes_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;
  static void pinned_budget_exhausted_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept;

  // OTel Meter
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter_;

  // Synchronous instruments
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> p2p_bytes_total_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> artifact_load_seconds_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> registration_commit_seconds_;
  std::atomic<std::uint64_t> p2p_total_transfers_{0};
  std::atomic<std::uint64_t> p2p_total_bytes_transferred_{0};
  std::atomic<std::uint64_t> p2p_total_transfer_errors_{0};

  // Async gauge for CPU memory available (exposes last observed value)
  // We keep the latest snapshot here; the ObservableGauge callback reads it.
  double cpu_available_bytes_last_ = 0.0;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> cpu_memory_available_gauge_;
  double registration_pending_last_ = 0.0;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> registration_pending_gauge_;

  struct PinnedClassSnapshot {
    std::string name;
    double capacity_slices = 0.0;
    double in_use_slices = 0.0;
    double free_slices = 0.0;
    double waiters = 0.0;
    double acquire_timeouts_total = 0.0;
  };

  std::mutex pinned_snapshot_mutex_;
  std::vector<PinnedClassSnapshot> pinned_classes_last_;
  double pinned_total_bytes_last_ = 0.0;
  double pinned_committed_bytes_last_ = 0.0;
  double pinned_budget_exhausted_total_last_ = 0.0;

  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> pinned_class_capacity_slices_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> pinned_class_in_use_slices_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> pinned_class_free_slices_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> pinned_class_waiters_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument>
      pinned_class_acquire_timeouts_total_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> pinned_total_bytes_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> pinned_committed_bytes_gauge_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> pinned_budget_exhausted_total_gauge_;
};

} // namespace tensorcast::store::components
