// Copyright (c) 2025, TensorCast Team.

#include "core/store/memory_tier_budget.h"

#include <algorithm>
#include <cmath>
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

namespace tensorcast::store {

MemoryTierBudget::MemoryTierBudget(uint64_t stable_total_bytes, uint64_t preemptible_total_bytes)
    : stable_total_bytes_(stable_total_bytes),
      preemptible_total_bytes_(preemptible_total_bytes),
      started_at_(absl::Now()) {}

MemoryTierBudget::MemoryTierBudget(MemoryTierBudget&& other) noexcept
    : stable_total_bytes_(other.stable_total_bytes_),
      preemptible_total_bytes_(other.preemptible_total_bytes_),
      started_at_(other.started_at_) {
  absl::MutexLock lk(&other.mu_);
  stable_used_bytes_ = other.stable_used_bytes_;
  other.stable_used_bytes_ = 0;
  const uint64_t marked_bytes = other.preemptible_marked_bytes_.load(std::memory_order_acquire);
  preemptible_marked_bytes_.store(marked_bytes, std::memory_order_release);
  other.preemptible_marked_bytes_.store(0, std::memory_order_release);
  fault_count_.store(other.fault_count_.load(std::memory_order_acquire), std::memory_order_release);
  other.fault_count_.store(0, std::memory_order_release);
  rehydrate_samples_ns_ = std::move(other.rehydrate_samples_ns_);
}

absl::Status MemoryTierBudget::try_acquire_stable(uint64_t bytes) {
  if (bytes == 0) {
    return absl::OkStatus();
  }
  absl::MutexLock lk(&mu_);
  if (bytes > stable_total_bytes_ || bytes > (stable_total_bytes_ - stable_used_bytes_)) {
    return absl::ResourceExhaustedError(
        absl::StrFormat(
            "Insufficient stable bytes: requested=%llu used=%llu total=%llu",
            bytes,
            stable_used_bytes_,
            static_cast<uint64_t>(stable_total_bytes_)));
  }
  stable_used_bytes_ += bytes;
  return absl::OkStatus();
}

void MemoryTierBudget::release_stable(uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  absl::MutexLock lk(&mu_);
  if (bytes >= stable_used_bytes_) {
    stable_used_bytes_ = 0;
    return;
  }
  stable_used_bytes_ -= bytes;
}

void MemoryTierBudget::set_preemptible_marked_bytes(uint64_t bytes) {
  preemptible_marked_bytes_.store(bytes, std::memory_order_release);
}

void MemoryTierBudget::record_fault() {
  fault_count_.fetch_add(1, std::memory_order_relaxed);
}

void MemoryTierBudget::record_rehydrate_latency(uint64_t latency_ns) {
  if (latency_ns == 0) {
    return;
  }
  absl::MutexLock lk(&mu_);
  if (rehydrate_samples_ns_.size() >= kMaxRehydrateSamples) {
    rehydrate_samples_ns_.erase(rehydrate_samples_ns_.begin());
  }
  rehydrate_samples_ns_.push_back(latency_ns);
}

MemoryTierBudget::Snapshot MemoryTierBudget::snapshot() const {
  absl::MutexLock lk(&mu_);
  const absl::Duration elapsed = absl::Now() - started_at_;
  double faults_per_sec = 0.0;
  if (elapsed > absl::ZeroDuration()) {
    faults_per_sec = static_cast<double>(fault_count_.load(std::memory_order_acquire)) / absl::ToDoubleSeconds(elapsed);
  }

  uint64_t rehydrate_p99 = 0;
  if (!rehydrate_samples_ns_.empty()) {
    std::vector<uint64_t> samples = rehydrate_samples_ns_;
    const size_t idx = static_cast<size_t>(std::ceil(samples.size() * 0.99)) - 1;
    const size_t clamp_idx = std::min(idx, samples.size() - 1);
    std::nth_element(samples.begin(), samples.begin() + static_cast<std::ptrdiff_t>(clamp_idx), samples.end());
    rehydrate_p99 = samples[clamp_idx];
  }

  return Snapshot{
      .stable_total_bytes = stable_total_bytes_,
      .stable_used_bytes = stable_used_bytes_,
      .preemptible_total_bytes = preemptible_total_bytes_,
      .preemptible_marked_bytes = preemptible_marked_bytes_.load(std::memory_order_acquire),
      .faults_per_sec = faults_per_sec,
      .rehydrate_p99_ns = rehydrate_p99,
  };
}

absl::StatusOr<uint64_t> MemoryTierBudget::compute_uma_capacity(uint64_t host_dram_bytes, uint64_t pinned_pool_bytes) {
  if (pinned_pool_bytes > host_dram_bytes) {
    return absl::InvalidArgumentError("Pinned pool bytes exceed detected host DRAM");
  }
  return host_dram_bytes - pinned_pool_bytes;
}

absl::StatusOr<MemoryTierBudget> MemoryTierBudget::from_config(
    const MemoryTierConfig& config,
    uint64_t host_dram_bytes,
    uint64_t pinned_pool_bytes) {
  auto uma_cap_or = compute_uma_capacity(host_dram_bytes, pinned_pool_bytes);
  if (!uma_cap_or.ok()) {
    return uma_cap_or.status();
  }
  const uint64_t uma_cap = *uma_cap_or;
  if (config.stable_bytes == 0) {
    return absl::InvalidArgumentError("engine.memory_tiers.stable_bytes must be greater than 0");
  }
  if (config.stable_bytes > uma_cap) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "engine.memory_tiers.stable_bytes=%llu exceeds UMA capacity %llu bytes (host=%llu pinned_pool=%llu)",
            static_cast<uint64_t>(config.stable_bytes),
            static_cast<uint64_t>(uma_cap),
            host_dram_bytes,
            pinned_pool_bytes));
  }

  const uint64_t preempt_cap = config.enable_preemptible_memory ? config.preemptible_limit_bytes : 0;
  return MemoryTierBudget(config.stable_bytes, preempt_cap);
}

} // namespace tensorcast::store
