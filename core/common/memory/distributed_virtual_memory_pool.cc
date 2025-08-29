// Copyright (c) 2025, TensorCast Team.

#include "core/common/memory/distributed_virtual_memory_pool.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "core/common/metrics/metric_objects.h"
#include "core/common/system_capabilities.h"

namespace tensorcast::memory {

DistributedVirtualMemoryPool::DistributedVirtualMemoryPool(size_t chunk_size) : chunk_size_(chunk_size) {
  LOG(INFO) << "Initialized DVMP with chunk size: " << chunk_size_ / (1024 * 1024) << " MiB";
  // Ensure capabilities are detected early even if communicator is not constructed.
  (void)common::SystemCapabilities::instance();
  // Warn once per process if mlock/munlock are unavailable so later paths can be quiet.
  static std::once_flag warn_once;
  std::call_once(warn_once, []() {
    if (!common::SystemCapabilities::instance().mlock_enabled()) {
      LOG(WARNING)
          << "DVMP: mlock/munlock unavailable; page pinning is disabled. unlock() will not physically unpin pages.";
    }
  });
}

DistributedVirtualMemoryPool::~DistributedVirtualMemoryPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [id, info_sp] : artifacts_) {
    if (!info_sp) {
      continue;
    }
    if (info_sp->cpu_base != nullptr && info_sp->bytes > 0) {
      if (munmap(info_sp->cpu_base, info_sp->bytes) != 0) {
        PLOG(WARNING) << "munmap failed for " << id;
      }
    }
  }
}

absl::StatusOr<DistributedVirtualMemoryPool::VirtualRegion> DistributedVirtualMemoryPool::allocate(
    std::string_view artifact_id,
    size_t bytes,
    int /*numa*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string key(artifact_id);
  if (artifacts_.find(key) != artifacts_.end()) {
    return absl::AlreadyExistsError("Artifact replica already has an allocated region");
  }

  // Reserve virtual address range with read/write permissions so that loaders
  // can directly stream data into the region without needing per-page mprotect
  // calls. Using PROT_NONE previously caused segmentation faults when streaming
  // into pages that were not writable.
  void* addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap failed while reserving VA space");
  }

  const size_t num_chunks = (bytes + chunk_size_ - 1) / chunk_size_;
  auto info = std::make_shared<DvmpRegionState>();
  info->cpu_base = addr;
  info->bytes = bytes;
  info->metadata = std::make_unique<store::ChunkMeta[]>(num_chunks);
  info->chunk_count = num_chunks;
  info->pin_refcnt = std::make_unique<std::atomic<uint32_t>[]>(num_chunks);
  info->mlock_refcnt = std::make_unique<std::atomic<uint32_t>[]>(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    info->metadata[i].state.store(store::ChunkState::COLD, std::memory_order_relaxed);
    info->metadata[i].last_touch_s.store(0, std::memory_order_relaxed);
    info->pin_refcnt[i].store(0, std::memory_order_relaxed);
    info->mlock_refcnt[i].store(0, std::memory_order_relaxed);
  }

  artifacts_.emplace(key, std::move(info));

  VirtualRegion region{addr, nullptr, bytes};
  VLOG(1) << "DVMP: allocated " << bytes << " bytes for replica " << key << " at " << addr;
  return region;
}

absl::StatusOr<DistributedVirtualMemoryPool::VirtualRegion> DistributedVirtualMemoryPool::allocate(
    std::string_view artifact_id,
    size_t bytes) {
  return allocate(artifact_id, bytes, -1);
}

absl::Span<const store::ChunkMeta> DistributedVirtualMemoryPool::chunk_snapshot(
    std::string_view artifact_id) const noexcept {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return {};
  }
  const auto& info_sp = *info_sp_or;
  if (!info_sp || info_sp->metadata == nullptr) {
    return {};
  }
  return {info_sp->metadata.get(), info_sp->chunk_count};
}

namespace {
constexpr int kMadviseFlagsFree = MADV_FREE; // Hints that pages can be reclaimed lazily
} // namespace

absl::Status DistributedVirtualMemoryPool::lock_chunks(std::string_view artifact_id, absl::Span<const uint32_t> idx) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  const std::string key(artifact_id);
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);

  if (idx.empty()) {
    // No-op for empty input per test expectation
    return absl::OkStatus();
  }
  // Track successfully locked indices, their previous states, and whether mlock succeeded
  struct LockedChunk {
    uint32_t idx;
    store::ChunkState prev_state;
    bool mlocked;
  };
  std::vector<LockedChunk> locked;
  locked.reserve(idx.size());

  // Convenience lambda to roll back any chunks that were already transitioned to
  // LOCKED_TX and to emit a warning with contextual information.
  auto rollback_and_log = [&](const absl::Status& status, std::optional<uint32_t> fail_idx) -> absl::Status {
    const bool expected_contention = status.code() == absl::StatusCode::kResourceExhausted;
    if (fail_idx.has_value()) {
      if (expected_contention) {
        VLOG(1) << "lock_chunks contention rollback for replica " << key << ", chunk " << *fail_idx << ": "
                << status.message();
      } else {
        LOG(WARNING) << "lock_chunks rollback for replica " << key << ", chunk " << *fail_idx << ": "
                     << status.message();
      }
    } else {
      if (expected_contention) {
        VLOG(1) << "lock_chunks contention rollback for replica " << key << ": " << status.message();
      } else {
        LOG(WARNING) << "lock_chunks rollback for replica " << key << ": " << status.message();
      }
    }
    for (const auto& lc : locked) {
      info.metadata[lc.idx].state.store(lc.prev_state, std::memory_order_release);
      void* j_addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(lc.idx) * chunk_size_;
      if (lc.mlocked) {
        auto prev_m = info.mlock_refcnt[lc.idx].fetch_sub(1, std::memory_order_acq_rel);
        if (prev_m == 1) {
          if (::munlock(j_addr, chunk_size_) != 0) {
            VLOG(1) << "munlock failed during rollback for chunk " << lc.idx;
          }
        }
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
    void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
    bool mlocked = false;
    if (common::SystemCapabilities::instance().mlock_enabled() && ::mlock(addr, chunk_size_) != 0) {
      const int err = errno;
      // If the failure is due to insufficient mlock limit or permissions, degrade
      // gracefully instead of aborting the entire load. In such cases we keep the
      // chunk state as LOCKED_TX so that the caller can still proceed. The pages
      // are not physically pinned, but the higher-level logic will still mark
      // them HOT after the unlock dance, allowing the system to reclaim them if
      // necessary. For other error types we still perform a full rollback.
      if (err == ENOMEM || err == EPERM) {
        // Warn once and disable mlock globally to avoid repeated spam in noisy environments.
        static std::atomic<bool> warned_demote{false};
        if (!warned_demote.exchange(true, std::memory_order_acq_rel)) {
          PLOG(WARNING) << "mlock failed (EPERM/ENOMEM) — disabling page pinning; proceeding without mlock/munlock";
        }
        common::SystemCapabilities::instance().set_mlock_enabled(false);
      } else {
        // Failed to lock pages, rollback this chunk and any previously locked ones.
        meta.state.store(expected, std::memory_order_release); // revert to previous state for this chunk
        return rollback_and_log(absl::ErrnoToStatus(err, "mlock failed while locking chunk memory"), i);
      }
    } else if (common::SystemCapabilities::instance().mlock_enabled()) {
      info.mlock_refcnt[i].fetch_add(1, std::memory_order_acq_rel);
      mlocked = true;
    }
    // If mlock globally disabled, opportunistically prefetch to reduce first-touch stalls
    if (!common::SystemCapabilities::instance().mlock_enabled() &&
        common::SystemCapabilities::instance().madv_willneed_available()) {
      (void)::madvise(addr, chunk_size_, MADV_WILLNEED);
    }

    meta.last_touch_s.store(now_s(), std::memory_order_relaxed);
    locked.push_back(LockedChunk{.idx = i, .prev_state = expected, .mlocked = mlocked});
  }
  return absl::OkStatus();
}

absl::Status DistributedVirtualMemoryPool::unlock_chunks(
    std::string_view artifact_id,
    absl::Span<const uint32_t> idx,
    bool copied_gpu) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);

  // No-op for empty input per test expectation
  if (idx.empty()) {
    return absl::OkStatus();
  }

  // First pass: validate all indices are in range and currently LOCKED_TX to
  // ensure all-or-nothing semantics without needing rollback.
  for (uint32_t i : idx) {
    if (i >= info.chunk_count) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    const auto state = info.metadata[i].state.load(std::memory_order_acquire);
    if (state != store::ChunkState::LOCKED_TX) {
      return absl::FailedPreconditionError("Chunk not locked for transfer");
    }
  }

  store::ChunkState new_state = copied_gpu ? store::ChunkState::COPIED_GPU : store::ChunkState::HOT;

  // Second pass: perform the unlock transition and housekeeping.
  for (uint32_t i : idx) {
    auto& meta = info.metadata[i];
    meta.state.store(new_state, std::memory_order_release);

    void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
    auto pin_cnt = info.pin_refcnt[i].load(std::memory_order_acquire);
    auto m_cnt = info.mlock_refcnt[i].load(std::memory_order_acquire);
    if (m_cnt > pin_cnt) {
      auto prev = info.mlock_refcnt[i].fetch_sub(1, std::memory_order_acq_rel);
      if (prev == 1) {
        if (::munlock(addr, chunk_size_) != 0) {
          VLOG(1) << "munlock failed for chunk " << i;
        }
      }
    }

    meta.last_touch_s.store(now_s(), std::memory_order_relaxed);
  }
  return absl::OkStatus();
}

size_t DistributedVirtualMemoryPool::evict_tail_bytes(std::string_view artifact_id, size_t bytes) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return 0;
  }
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  size_t freed = 0;
  // Iterate from tail backwards.
  for (ssize_t idx = static_cast<ssize_t>(info.chunk_count) - 1; idx >= 0 && freed < bytes; --idx) {
    auto& meta = info.metadata[idx];
    store::ChunkState st = meta.state.load(std::memory_order_acquire);
    // Skip pinned chunks
    if (info.pin_refcnt[idx].load(std::memory_order_acquire) > 0) {
      continue;
    }
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
      void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(idx) * chunk_size_;
      size_t advise_len = chunk_size_;
      // Choose once based on capabilities to avoid repeated failing syscalls
      int madv_flag = MADV_DONTNEED;
#ifdef MADV_PAGEOUT
      if (common::SystemCapabilities::instance().madv_pageout_available()) {
        madv_flag = MADV_PAGEOUT;
      }
#endif
      int rc = ::madvise(addr, advise_len, madv_flag);
      if (rc != 0) {
        PLOG(WARNING) << "madvise PAGEOUT failed";
      }
      freed += chunk_size_;
    }
  }
  return freed;
}

void DistributedVirtualMemoryPool::refresh_chunks(std::string_view artifact_id, absl::Span<const uint32_t> idx) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return;
  }
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  uint64_t ts = now_s();
  for (uint32_t i : idx) {
    if (i < info.chunk_count) {
      info.metadata[i].last_touch_s.store(ts, std::memory_order_relaxed);
    }
  }
}

absl::Status DistributedVirtualMemoryPool::ensure_chunk_resident(std::string_view artifact_id, uint32_t chunk_idx) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  if (chunk_idx >= info.chunk_count) {
    return absl::OutOfRangeError("Chunk index out of range");
  }
  auto st = info.metadata[chunk_idx].state.load(std::memory_order_acquire);
  if (st == store::ChunkState::EVICTED) {
    return {kErrChunkRemote, "Chunk data not resident locally"};
  }
  return absl::OkStatus();
}

absl::Status DistributedVirtualMemoryPool::mark_preemptible(
    std::string_view artifact_id,
    absl::Span<const uint32_t> idx) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  for (uint32_t i : idx) {
    if (i >= info.chunk_count) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    auto& meta = info.metadata[i];
    // Skip pinned chunks
    if (info.pin_refcnt[i].load(std::memory_order_acquire) > 0) {
      continue;
    }
    store::ChunkState expected = meta.state.load(std::memory_order_acquire);
    // Eligible states: HOT, COLD, COPIED_GPU (may still hold identical data but no longer needed)
    if (expected == store::ChunkState::HOT || expected == store::ChunkState::COLD ||
        expected == store::ChunkState::COPIED_GPU) {
      if (meta.state.compare_exchange_strong(expected, store::ChunkState::PREEMPTIBLE, std::memory_order_acq_rel)) {
        void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
        // Prefer MADV_FREE if capability detected, else MADV_DONTNEED directly
        int rc = 0;
        if (common::SystemCapabilities::instance().madv_free_available()) {
          rc = ::madvise(addr, chunk_size_, kMadviseFlagsFree);
          if (rc != 0 && errno == EINVAL) {
            rc = ::madvise(addr, chunk_size_, MADV_DONTNEED);
          }
        } else {
          rc = ::madvise(addr, chunk_size_, MADV_DONTNEED);
        }
        if (rc != 0) {
          PLOG(WARNING) << "madvise FREE/DONTNEED failed";
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status DistributedVirtualMemoryPool::write_at(
    std::string_view artifact_id,
    uint64_t va_offset,
    const void* src,
    size_t bytes) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }

  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  if (va_offset >= info.bytes) {
    return absl::OutOfRangeError("write_at offset beyond artifact size");
  }

  // Check if write would exceed bounds - make this an explicit error
  if (va_offset + bytes > info.bytes) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Write would exceed replica bounds: offset=%lu + bytes=%lu > artifact_size=%lu",
            va_offset,
            bytes,
            info.bytes));
  }

  // Ensure destination range is writable; if any part is file-backed read-only,
  // try mprotect to upgrade, otherwise remap anonymously for the subrange.
  const int64_t page = sysconf(_SC_PAGESIZE);
  uint64_t page_aligned_off = (va_offset / page) * page;
  uint64_t end_off = va_offset + bytes;
  uint64_t page_aligned_end = ((end_off + page - 1) / page) * page;
  size_t aligned_len = static_cast<size_t>(page_aligned_end - page_aligned_off);
  void* aligned_addr = static_cast<char*>(info.cpu_base) + page_aligned_off;
  if (::mprotect(aligned_addr, aligned_len, PROT_READ | PROT_WRITE) != 0) {
    void* mapped =
        ::mmap(aligned_addr, aligned_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (mapped == MAP_FAILED || mapped != aligned_addr) {
      return absl::ErrnoToStatus(errno, "DVMP write_at: failed to ensure writable mapping");
    }
  }

  // If MADV_WILLNEED is supported, proactively fault pages to reduce first-write stalls
  if (common::SystemCapabilities::instance().madv_willneed_available()) {
    (void)::madvise(aligned_addr, aligned_len, MADV_WILLNEED);
  }

  void* dst = static_cast<char*>(info.cpu_base) + va_offset;
  std::memcpy(dst, src, bytes);

  // Update metadata for affected chunks
  const uint64_t first = va_offset / chunk_size_;
  const uint64_t last = (va_offset + bytes - 1) / chunk_size_;
  uint64_t ts = now_s();
  for (uint64_t i = first; i <= last && i < info.chunk_count; ++i) {
    info.metadata[i].state.store(store::ChunkState::HOT, std::memory_order_release);
    info.metadata[i].last_touch_s.store(static_cast<uint32_t>(ts), std::memory_order_relaxed);
  }
  // Metrics: bytes written
  try {
    static const metrics::Counter kWriteBytes("dvmp_write_bytes_total");
    kWriteBytes.inc(static_cast<double>(bytes));
  } catch (...) {
    VLOG(1) << "metrics counter kWriteBytes unavailable";
  }
  return absl::OkStatus();
}

absl::Status DistributedVirtualMemoryPool::map_file_segments(
    std::string_view artifact_id,
    absl::Span<const FileSegment> segs) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);

  const int64_t page = sysconf(_SC_PAGESIZE);
  for (const auto& s : segs) {
    if (!std::filesystem::exists(s.path)) {
      return absl::NotFoundError("File segment path not found");
    }
    if (s.va_offset + s.length > info.bytes) {
      return absl::OutOfRangeError("File segment range exceeds VA");
    }
    if ((s.file_offset % page) != 0 || (s.va_offset % page) != 0) {
      return absl::InvalidArgumentError("file_offset and va_offset must be page-aligned for mmap");
    }
    int fd = ::open(s.path.c_str(), O_RDONLY);
    if (fd < 0) {
      return absl::ErrnoToStatus(errno, "open failed for segment");
    }
    int flags = MAP_PRIVATE | MAP_FIXED;
#ifdef MAP_POPULATE
    if (s.populate) {
      flags |= MAP_POPULATE;
    }
#endif
    void* target = static_cast<char*>(info.cpu_base) + s.va_offset;
    void* mapped = ::mmap(target, s.length, PROT_READ, flags, fd, static_cast<off_t>(s.file_offset));
    int saved = errno;
    ::close(fd);
    if (mapped == MAP_FAILED) {
      return absl::ErrnoToStatus(saved, "mmap failed for file segment");
    }
    if (mapped != target) {
      ::munmap(mapped, s.length);
      return absl::InternalError("mmap returned unexpected address");
    }

    // Update metadata for affected chunks
    const uint64_t first = s.va_offset / chunk_size_;
    const uint64_t last = (s.va_offset + s.length - 1) / chunk_size_;
    uint64_t ts = now_s();
    for (uint64_t i = first; i <= last && i < info.chunk_count; ++i) {
      info.metadata[i].state.store(store::ChunkState::HOT, std::memory_order_release);
      info.metadata[i].last_touch_s.store(static_cast<uint32_t>(ts), std::memory_order_relaxed);
    }
  }
  // Metrics: bytes mapped
  try {
    static const metrics::Counter kMapBytes("dvmp_map_bytes_total");
    uint64_t total = 0;
    for (const auto& s : segs) {
      total += s.length;
    }
    kMapBytes.inc(static_cast<double>(total));
  } catch (...) {
    VLOG(1) << "metrics counter kMapBytes unavailable";
  }
  return absl::OkStatus();
}

absl::StatusOr<DistributedVirtualMemoryPool::VirtualRegion> DistributedVirtualMemoryPool::region_info(
    std::string_view artifact_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = artifacts_.find(std::string(artifact_id));
  if (it == artifacts_.end() || !it->second) {
    return absl::NotFoundError("Artifact not found in DVMP");
  }
  const auto& info_sp = it->second;
  return VirtualRegion{.cpu_base = info_sp->cpu_base, .gpu_base = nullptr, .bytes = info_sp->bytes};
}

// ChunkResidencyLease impl --------------------------------------------------------------
DistributedVirtualMemoryPool::ChunkResidencyLease::~ChunkResidencyLease() {
  release();
}

bool DistributedVirtualMemoryPool::ChunkResidencyLease::is_expired() const {
  if (!impl_ || !impl_->expiry_time) {
    return false;
  }
  return std::chrono::steady_clock::now() > *impl_->expiry_time;
}

DistributedVirtualMemoryPool::ChunkResidencyLease::ChunkResidencyLease(ChunkResidencyLease&& other) noexcept {
  impl_ = std::move(other.impl_);
}
DistributedVirtualMemoryPool::ChunkResidencyLease& DistributedVirtualMemoryPool::ChunkResidencyLease::operator=(
    ChunkResidencyLease&& other) noexcept {
  if (this != &other) {
    // Release any currently held lease before taking ownership
    release();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void DistributedVirtualMemoryPool::ChunkResidencyLease::release() noexcept {
  if (!impl_) {
    return;
  }
  // Warn if expired before release (debug signal only)
  if (is_expired()) {
    LOG(WARNING) << "ChunkResidencyLease expired before being destroyed for replica: " << impl_->artifact_id;
  }

  gsl::not_null<DistributedVirtualMemoryPool*> dvmp = impl_->dvmp;
  if (!dvmp) {
    impl_.reset();
    return;
  }
  auto info_sp_or = dvmp->get_artifact_info(impl_->artifact_id);
  if (!info_sp_or.ok()) {
    impl_.reset();
    return;
  }
  const auto& info_sp = *info_sp_or;
  if (!info_sp) {
    impl_.reset();
    return;
  }
  std::lock_guard<std::mutex> artifact(info_sp->artifact_mutex);
  dvmp->release_pins_unlocked(*info_sp, impl_->chunks);
  impl_.reset();
}

void DistributedVirtualMemoryPool::release_pins_unlocked(DvmpRegionState& info, absl::Span<const uint32_t> chunks)
    const {
  for (uint32_t i : chunks) {
    if (i >= info.chunk_count) {
      continue;
    }
    // Decrement pin lease refcount
    auto cnt = info.pin_refcnt[i].load(std::memory_order_acquire);
    if (cnt > 0) {
      info.pin_refcnt[i].store(cnt - 1, std::memory_order_release);
    }
    // Decrement mlock refcount and munlock when it reaches zero
    auto m_cnt = info.mlock_refcnt[i].load(std::memory_order_acquire);
    if (m_cnt > 0) {
      auto prev = info.mlock_refcnt[i].fetch_sub(1, std::memory_order_acq_rel);
      if (prev == 1) {
        void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
        if (::munlock(addr, chunk_size_) != 0) {
          // Ignore errors; best-effort only
        }
      }
    }
  }
}

absl::StatusOr<DistributedVirtualMemoryPool::ChunkResidencyLease> DistributedVirtualMemoryPool::pin_range(
    std::string_view artifact_id,
    uint64_t va_offset,
    uint64_t bytes,
    std::string_view reason) {
  return pin_range(artifact_id, va_offset, bytes, reason, std::nullopt);
}

absl::StatusOr<DistributedVirtualMemoryPool::ChunkResidencyLease> DistributedVirtualMemoryPool::pin_range(
    std::string_view artifact_id,
    uint64_t va_offset,
    uint64_t bytes,
    std::string_view reason,
    std::optional<std::chrono::milliseconds> timeout_ms) {
  const std::string key(artifact_id);
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  DvmpRegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  if (va_offset >= info.bytes) {
    return absl::OutOfRangeError("pin_range offset beyond artifact size");
  }
  uint64_t to_cover = std::min<uint64_t>(bytes, info.bytes - va_offset);
  if (to_cover == 0) {
    return absl::OutOfRangeError("pin_range length zero");
  }

  const uint32_t first = static_cast<uint32_t>(va_offset / chunk_size_);
  const uint32_t last = static_cast<uint32_t>((va_offset + to_cover - 1) / chunk_size_);
  std::vector<uint32_t> chunks;
  chunks.reserve(last - first + 1);
  for (uint32_t i = first; i <= last && i < info.chunk_count; ++i) {
    chunks.push_back(i);
    // Inc refcount and best-effort mlock
    auto prev = info.pin_refcnt[i].load(std::memory_order_acquire);
    info.pin_refcnt[i].store(prev + 1, std::memory_order_release);
    void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
    if (common::SystemCapabilities::instance().mlock_enabled()) {
      if (::mlock(addr, chunk_size_) == 0) {
        info.mlock_refcnt[i].fetch_add(1, std::memory_order_acq_rel);
      } else {
        const int err = errno;
        if (err == ENOMEM || err == EPERM) {
          static std::atomic<bool> warned_demote{false};
          if (!warned_demote.exchange(true, std::memory_order_acq_rel)) {
            PLOG(WARNING) << "mlock failed (EPERM/ENOMEM) — disabling page pinning; proceeding without mlock/munlock";
          }
          common::SystemCapabilities::instance().set_mlock_enabled(false);
        }
      }
    }
  }
  // Calculate expiry time if timeout is specified
  std::optional<std::chrono::steady_clock::time_point> expiry_time;
  if (timeout_ms.has_value() && timeout_ms->count() > 0) {
    expiry_time = std::chrono::steady_clock::now() + *timeout_ms;
  }

  // Metrics: record a pin-lease acquisition event for external safety/export.
  try {
    static const metrics::Counter kPinLeasesTotal("dvmp_pin_leases_total");
    kPinLeasesTotal.with_labels({{"reason", std::string(reason)}}).inc();
  } catch (...) {
    VLOG(1) << "metrics counter kPinLeasesTotal unavailable";
  }

  return ChunkResidencyLease(
      ChunkResidencyLease::Impl{
          .dvmp = this, .artifact_id = key, .chunks = std::move(chunks), .expiry_time = expiry_time});
}

absl::StatusOr<DistributedVirtualMemoryPool::DvmpRegion> DistributedVirtualMemoryPool::open(
    std::string_view artifact_id) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  return DvmpRegion(this, std::string(artifact_id));
}

} // namespace tensorcast::memory
