// Copyright (c) 2025-2026, TensorCast Team.

#include "core/communicator/engine/host_pinned_cpu_stager.h"

#include <chrono>
#include <cstring>

#include "absl/log/log.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::engine {

namespace {
class NoOpLeaseHandle : public HostPinnedCpuStager::LeaseHandle {};

class NoOpLeaseProvider : public HostPinnedCpuStager::LeaseProvider {
 public:
  std::unique_ptr<HostPinnedCpuStager::LeaseHandle> acquire(
      const std::string& /*tensor_key*/,
      uint64_t /*offset*/,
      uint64_t /*bytes*/) override {
    return std::make_unique<NoOpLeaseHandle>();
  }
};
} // namespace

std::shared_ptr<HostPinnedCpuStager::LeaseProvider> HostPinnedCpuStager::make_noop_lease_provider() {
  static auto provider = std::make_shared<NoOpLeaseProvider>();
  return provider;
}

HostPinnedCpuStager::HostPinnedCpuStager(
    gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> pool,
    size_t num_buffers_hint)
    : pool_(std::move(pool)), chunk_size_(pool_->slice_bytes()), num_buffers_hint_(num_buffers_hint) {}

absl::StatusOr<void*> HostPinnedCpuStager::stage(
    const std::shared_ptr<communicator::transport::PartitionTensor>& tensor,
    uint64_t offset,
    uint64_t bytes,
    StageMode mode) {
  const auto total_started_at = std::chrono::steady_clock::now();
  // pool_ is guaranteed non-null
  if (bytes == 0 || bytes > chunk_size_) {
    return absl::InvalidArgumentError("bytes must be in [1, chunk_size]");
  }

  if (mode == StageMode::kTry) {
    const size_t available = pool_->get_available_size();
    if (available < chunk_size_) {
      return absl::UnavailableError("PinnedBufferPool has no free buffers");
    }
  }

  std::vector<char*> bufs;
  // Immediate allocation attempt; the send loop can back off if needed
  const auto allocate_started_at = std::chrono::steady_clock::now();
  int ret = pool_->allocate(chunk_size_, bufs);
  const auto allocate_finished_at = std::chrono::steady_clock::now();
  if (ret != 0 || bufs.empty()) {
    return absl::ResourceExhaustedError("PinnedBufferPool out of buffers for staging");
  }

  void* exposed_ptr = bufs[0];
  // memcpy from source VA
  std::unique_ptr<LeaseHandle> lease;
  const auto lease_started_at = std::chrono::steady_clock::now();
  if (lease_provider_) {
    lease = lease_provider_->acquire(tensor->get_key(), offset, bytes);
  }
  const auto lease_finished_at = std::chrono::steady_clock::now();
  auto* src = tensor->get_addr<uint8_t>() + offset;
  const auto memcpy_started_at = std::chrono::steady_clock::now();
  std::memcpy(exposed_ptr, src, static_cast<size_t>(bytes));
  const auto memcpy_finished_at = std::chrono::steady_clock::now();

  auto to_us = [](const auto start, const auto end) -> uint64_t {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  };
  StageStats stage_stats{
      .requested_bytes = bytes,
      .pool_allocate_us = to_us(allocate_started_at, allocate_finished_at),
      .lease_acquire_us = to_us(lease_started_at, lease_finished_at),
      .memcpy_us = to_us(memcpy_started_at, memcpy_finished_at),
      .total_us = to_us(total_started_at, memcpy_finished_at),
  };

  {
    absl::MutexLock lock(&mu_);
    AllocationRecord record;
    record.buffers.reserve(bufs.size());
    for (auto* p : bufs)
      record.buffers.emplace_back(gsl::not_null<char*>{p});
    record.stage_stats = stage_stats;
    allocations_[exposed_ptr] = std::move(record);
  }
  return exposed_ptr;
}

std::optional<HostPinnedCpuStager::StageStats> HostPinnedCpuStager::stage_stats_for_ptr(
    gsl::not_null<void*> exposed_ptr) const {
  absl::MutexLock lock(&mu_);
  auto it = allocations_.find(exposed_ptr.get());
  if (it == allocations_.end()) {
    return std::nullopt;
  }
  return it->second.stage_stats;
}

absl::Status HostPinnedCpuStager::release_staged_buffer(gsl::not_null<void*> exposed_ptr) {
  // pool_ is guaranteed non-null
  std::vector<char*> bufs;
  {
    absl::MutexLock lock(&mu_);
    auto it = allocations_.find(exposed_ptr.get());
    if (it == allocations_.end()) {
      return absl::InvalidArgumentError("Unknown exposed_ptr for HostPinnedCpuStager::release_staged_buffer");
    }
    bufs.reserve(it->second.buffers.size());
    for (auto p : it->second.buffers)
      bufs.push_back(p.get());
    allocations_.erase(it);
  }
  int ret = pool_->deallocate(bufs);
  if (ret != 0) {
    return absl::InternalError("Failed to deallocate pinned buffers");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::communicator::engine
