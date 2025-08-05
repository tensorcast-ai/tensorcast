// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/common/memory/distributed_memory_pool.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace stepcast::memory {

DistributedMemoryPool::~DistributedMemoryPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [id, info] : models_) {
    if (info.base != nullptr && info.bytes > 0) {
      if (munmap(info.base, info.bytes) != 0) {
        LOG(WARNING) << "munmap failed for model " << id << ": " << std::strerror(errno);
      }
    }
  }
}

absl::StatusOr<DistributedMemoryPool::VirtualRegion> DistributedMemoryPool::allocate(
    std::string_view model_id,
    size_t bytes,
    int /*numa*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string key(model_id);
  if (models_.find(key) != models_.end()) {
    return absl::AlreadyExistsError("Model already has an allocated region");
  }

  void* addr = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap failed while reserving VA space");
  }

  const size_t num_chunks = (bytes + kChunk - 1) / kChunk;
  ModelInfo info;
  info.base = addr;
  info.bytes = bytes;
  info.metadata = std::make_unique<stepcast::store::ChunkMeta[]>(num_chunks);
  info.chunk_count = num_chunks;
  for (size_t i = 0; i < num_chunks; ++i) {
    info.metadata[i].state.store(stepcast::store::ChunkState::COLD, std::memory_order_relaxed);
    info.metadata[i].last_touch_s.store(0, std::memory_order_relaxed);
  }

  models_.emplace(key, std::move(info));

  VirtualRegion region{addr, nullptr, bytes};
  VLOG(1) << "DVMP: allocated " << bytes << " bytes for model " << key << " at " << addr;
  return region;
}

absl::Span<const stepcast::store::ChunkMeta> DistributedMemoryPool::chunk_snapshot(
    std::string_view model_id) const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(std::string(model_id));
  if (it == models_.end()) {
    return {};
  }
  if (it->second.metadata == nullptr) {
    return {};
  }
  return {it->second.metadata.get(), it->second.chunk_count};
}

namespace {
constexpr int kMadviseFlagsEvict = MADV_PAGEOUT; // Real page-out on modern kernels
constexpr int kMadviseFlagsFree = MADV_FREE; // Hints that pages can be reclaimed lazily
} // namespace

absl::Status DistributedMemoryPool::lock_chunks(std::string_view model_id, absl::Span<const uint32_t> idx) {
  const std::string key(model_id);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(key);
  if (it == models_.end()) {
    return absl::NotFoundError("Model not found in DVMP");
  }
  ModelInfo& info = it->second;

  if (idx.empty()) {
    LOG(WARNING) << "lock_chunks called with empty index span for model " << key;
    return absl::InvalidArgumentError("idx span is empty");
  }
  // Track successfully locked indices and their previous states for rollback if needed.
  std::vector<std::pair<uint32_t, store::ChunkState>> locked;
  locked.reserve(idx.size());

  // Convenience lambda to roll back any chunks that were already transitioned to
  // LOCKED_TX and to emit a warning with contextual information.
  auto rollback_and_log = [&](const absl::Status& status, std::optional<uint32_t> fail_idx) -> absl::Status {
    if (fail_idx.has_value()) {
      LOG(WARNING) << "lock_chunks rollback for model " << key << ", chunk " << *fail_idx << ": " << status.message();
    } else {
      LOG(WARNING) << "lock_chunks rollback for model " << key << ": " << status.message();
    }
    for (const auto& [j_idx, prev] : locked) {
      info.metadata[j_idx].state.store(prev, std::memory_order_release);
      void* j_addr = static_cast<char*>(info.base) + static_cast<size_t>(j_idx) * kChunk;
      if (::munlock(j_addr, kChunk) != 0) {
        PLOG(WARNING) << "munlock failed during rollback for chunk " << j_idx;
      }
    }
    return status;
  };
  for (uint32_t i : idx) {
    if (i >= info.chunk_count) {
      return rollback_and_log(absl::OutOfRangeError("Chunk index out of range"), i);
    }
    auto& meta = info.metadata[i];
    store::ChunkState expected = meta.state.load(std::memory_order_acquire);
    bool ok = false;
    // We allow locking from HOT, COLD, PREEMPTIBLE, or COPIED_GPU (target may recopy).
    for (;;) {
      if (expected == store::ChunkState::HOT || expected == store::ChunkState::COLD ||
          expected == store::ChunkState::PREEMPTIBLE || expected == store::ChunkState::COPIED_GPU) {
        ok = meta.state.compare_exchange_strong(expected, store::ChunkState::LOCKED_TX, std::memory_order_acq_rel);
        if (ok) {
          break;
        }
        // retry with new expected value
      } else {
        break; // Busy or invalid state
      }
    }
    if (!ok) {
      return rollback_and_log(absl::ResourceExhaustedError("Failed to lock chunks – one or more are busy"), i);
    }

    // Attempt to lock the corresponding memory pages so they cannot be reclaimed.
    void* addr = static_cast<char*>(info.base) + static_cast<size_t>(i) * kChunk;
    if (::mlock(addr, kChunk) != 0) {
      const int err = errno;
      // If the failure is due to insufficient mlock limit or permissions, degrade
      // gracefully instead of aborting the entire load. In such cases we keep the
      // chunk state as LOCKED_TX so that the caller can still proceed. The pages
      // are not physically pinned, but the higher-level logic will still mark
      // them HOT after the unlock dance, allowing the system to reclaim them if
      // necessary. For other error types we still perform a full rollback.
      if (err == ENOMEM || err == EPERM) {
        PLOG(ERROR) << "mlock failed for chunk " << i << " — proceeding without page lock";
      } else {
        // Failed to lock pages, rollback this chunk and any previously locked ones.
        meta.state.store(expected, std::memory_order_release); // revert to previous state for this chunk
        return rollback_and_log(absl::ErrnoToStatus(err, "mlock failed while locking chunk memory"), i);
      }
    }

    meta.last_touch_s.store(now_s(), std::memory_order_relaxed);
    locked.emplace_back(i, expected);
  }
  return absl::OkStatus();
}

absl::Status DistributedMemoryPool::unlock_chunks(
    std::string_view model_id,
    absl::Span<const uint32_t> idx,
    bool copied_gpu) {
  const std::string key(model_id);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(key);
  if (it == models_.end()) {
    return absl::NotFoundError("Model not found in DVMP");
  }
  ModelInfo& info = it->second;
  store::ChunkState new_state = copied_gpu ? store::ChunkState::COPIED_GPU : store::ChunkState::HOT;
  for (uint32_t i : idx) {
    if (i >= info.chunk_count) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    auto& meta = info.metadata[i];
    store::ChunkState expected = store::ChunkState::LOCKED_TX;
    if (!meta.state.compare_exchange_strong(expected, new_state, std::memory_order_acq_rel)) {
      return absl::FailedPreconditionError("unlock_chunks: chunk not in LOCKED_TX state");
    }

    // Release the page lock so that it can be evicted/preempted again if necessary.
    void* addr = static_cast<char*>(info.base) + static_cast<size_t>(i) * kChunk;
    int rc = ::munlock(addr, kChunk);
    if (rc != 0) {
      LOG(WARNING) << "munlock failed for chunk " << i << ": " << std::strerror(errno);
    }

    meta.last_touch_s.store(now_s(), std::memory_order_relaxed);
  }
  return absl::OkStatus();
}

size_t DistributedMemoryPool::evict_tail_bytes(std::string_view model_id, size_t bytes) {
  const std::string key(model_id);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(key);
  if (it == models_.end()) {
    return 0;
  }
  ModelInfo& info = it->second;
  size_t freed = 0;
  // Iterate from tail backwards.
  for (ssize_t idx = static_cast<ssize_t>(info.chunk_count) - 1; idx >= 0 && freed < bytes; --idx) {
    auto& meta = info.metadata[idx];
    store::ChunkState st = meta.state.load(std::memory_order_acquire);
    if (st == store::ChunkState::LOCKED_TX || st == store::ChunkState::EVICTED) {
      continue; // Cannot evict locked or already evicted
    }
    if (st == store::ChunkState::HOT || st == store::ChunkState::COLD || st == store::ChunkState::COPIED_GPU ||
        st == store::ChunkState::PREEMPTIBLE) {
      // Attempt CAS to EVICTED.
      store::ChunkState expected = st;
      if (!meta.state.compare_exchange_strong(expected, store::ChunkState::EVICTED, std::memory_order_acq_rel)) {
        continue; // state changed concurrently, skip
      }
      void* addr = static_cast<char*>(info.base) + static_cast<size_t>(idx) * kChunk;
      size_t advise_len = kChunk;
      // Use MADV_PAGEOUT if available else MADV_DONTNEED
      int madv_flag = kMadviseFlagsEvict;
#ifndef MADV_PAGEOUT
      madv_flag = MADV_DONTNEED;
#endif
      int rc = ::madvise(addr, advise_len, madv_flag);
      if (rc != 0) {
        LOG(WARNING) << "madvise PAGEOUT failed: " << std::strerror(errno);
      }
      freed += kChunk;
    }
  }
  return freed;
}

void DistributedMemoryPool::refresh_chunks(std::string_view model_id, absl::Span<const uint32_t> idx) {
  const std::string key(model_id);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(key);
  if (it == models_.end()) {
    return;
  }
  ModelInfo& info = it->second;
  uint64_t ts = now_s();
  for (uint32_t i : idx) {
    if (i < info.chunk_count) {
      info.metadata[i].last_touch_s.store(ts, std::memory_order_relaxed);
    }
  }
}

absl::Status DistributedMemoryPool::ensure_chunk_resident(std::string_view model_id, uint32_t chunk_idx) {
  const std::string key(model_id);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(key);
  if (it == models_.end()) {
    return absl::NotFoundError("Model not found in DVMP");
  }
  ModelInfo& info = it->second;
  if (chunk_idx >= info.chunk_count) {
    return absl::OutOfRangeError("Chunk index out of range");
  }
  auto st = info.metadata[chunk_idx].state.load(std::memory_order_acquire);
  if (st == store::ChunkState::EVICTED) {
    return {kErrChunkRemote, "Chunk data not resident locally"};
  }
  return absl::OkStatus();
}

absl::Status DistributedMemoryPool::mark_preemptible(std::string_view model_id, absl::Span<const uint32_t> idx) {
  const std::string key(model_id);
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(key);
  if (it == models_.end()) {
    return absl::NotFoundError("Model not found in DVMP");
  }
  ModelInfo& info = it->second;
  for (uint32_t i : idx) {
    if (i >= info.chunk_count) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    auto& meta = info.metadata[i];
    store::ChunkState expected = meta.state.load(std::memory_order_acquire);
    // Eligible states: HOT, COLD, COPIED_GPU (may still hold identical data but no longer needed)
    if (expected == store::ChunkState::HOT || expected == store::ChunkState::COLD ||
        expected == store::ChunkState::COPIED_GPU) {
      if (meta.state.compare_exchange_strong(expected, store::ChunkState::PREEMPTIBLE, std::memory_order_acq_rel)) {
        void* addr = static_cast<char*>(info.base) + static_cast<size_t>(i) * kChunk;
        int rc = ::madvise(addr, kChunk, kMadviseFlagsFree);
        if (rc != 0) {
          // Kernel might not support MADV_FREE (EINVAL). Fallback to MADV_DONTNEED.
          if (errno == EINVAL) {
            rc = ::madvise(addr, kChunk, MADV_DONTNEED);
          }
          if (rc != 0) {
            LOG(WARNING) << "madvise FREE/DONTNEED failed: " << std::strerror(errno);
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

} // namespace stepcast::memory