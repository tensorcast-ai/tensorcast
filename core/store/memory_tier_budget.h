// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/memory_tier_config.h"

namespace tensorcast::store {

/**
 * @brief Node-local accounting for stable/preemptible UMA tiers.
 *
 * MemoryTierBudget tracks configured capacities (stable + preemptible) and
 * provides atomic reservation helpers for stable leases. The class is
 * intentionally lightweight so callers can hold a shared instance across
 * replicas and the daemon lifecycle.
 */
class MemoryTierBudget {
 public:
  struct Snapshot {
    uint64_t stable_total_bytes{0};
    uint64_t stable_used_bytes{0};
    uint64_t preemptible_total_bytes{0};
    uint64_t preemptible_marked_bytes{0};
    double faults_per_sec{0.0};
    uint64_t rehydrate_p99_ns{0};
  };

  MemoryTierBudget(uint64_t stable_total_bytes, uint64_t preemptible_total_bytes);
  MemoryTierBudget(MemoryTierBudget&& other) noexcept;

  MemoryTierBudget(const MemoryTierBudget&) = delete;
  MemoryTierBudget& operator=(const MemoryTierBudget&) = delete;
  MemoryTierBudget& operator=(MemoryTierBudget&&) = delete;

  /**
   * @brief Reserve stable bytes. Returns ResourceExhausted if insufficient.
   */
  absl::Status try_acquire_stable(uint64_t bytes);

  /**
   * @brief Release previously acquired stable bytes (clamped to zero).
   */
  void release_stable(uint64_t bytes);

  /**
   * @brief Update the number of bytes currently marked preemptible.
   *
   * This is a best-effort telemetry helper; callers should provide the most
   * recent known value. The budget keeps the latest absolute count.
   */
  void set_preemptible_marked_bytes(uint64_t bytes);

  /**
   * @brief Record that a preemptible chunk faulted and required rehydration.
   */
  void record_fault();

  /**
   * @brief Record a rehydration latency sample in nanoseconds.
   */
  void record_rehydrate_latency(uint64_t latency_ns);

  [[nodiscard]] Snapshot snapshot() const;

  [[nodiscard]] uint64_t stable_total_bytes() const {
    return stable_total_bytes_;
  }

  [[nodiscard]] uint64_t preemptible_total_bytes() const {
    return preemptible_total_bytes_;
  }

  /**
   * @brief Compute UMA CPU capacity after subtracting pinned pool bytes.
   */
  static absl::StatusOr<uint64_t> compute_uma_capacity(uint64_t host_dram_bytes, uint64_t pinned_pool_bytes);

  /**
   * @brief Factory that validates config against host capacity and constructs a budget.
   */
  static absl::StatusOr<MemoryTierBudget> from_config(
      const MemoryTierConfig& config,
      uint64_t host_dram_bytes,
      uint64_t pinned_pool_bytes);

 private:
  const uint64_t stable_total_bytes_;
  const uint64_t preemptible_total_bytes_;

  mutable absl::Mutex mu_;
  uint64_t stable_used_bytes_ ABSL_GUARDED_BY(mu_){0};
  std::atomic<uint64_t> preemptible_marked_bytes_{0};
  std::atomic<uint64_t> fault_count_{0};
  const absl::Time started_at_;
  static constexpr size_t kMaxRehydrateSamples = 256;
  std::vector<uint64_t> rehydrate_samples_ns_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::store
