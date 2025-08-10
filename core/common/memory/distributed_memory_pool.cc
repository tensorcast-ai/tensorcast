// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/common/memory/distributed_memory_pool.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "core/common/metrics/metric_objects.h"

namespace stepcast::memory {

DistributedMemoryPool::~DistributedMemoryPool() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& [id, info_sp] : models_) {
    if (!info_sp)
      continue;
    if (info_sp->base != nullptr && info_sp->bytes > 0) {
      if (munmap(info_sp->base, info_sp->bytes) != 0) {
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

  // Reserve virtual address range with read/write permissions so that loaders
  // can directly stream data into the region without needing per-page mprotect
  // calls. Using PROT_NONE previously caused segmentation faults when streaming
  // into pages that were not writable.
  void* addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap failed while reserving VA space");
  }

  const size_t num_chunks = (bytes + kChunk - 1) / kChunk;
  auto info = std::make_shared<ModelInfo>();
  info->base = addr;
  info->bytes = bytes;
  info->metadata = std::make_unique<stepcast::store::ChunkMeta[]>(num_chunks);
  info->chunk_count = num_chunks;
  info->pin_refcnt = std::make_unique<std::atomic<uint32_t>[]>(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    info->metadata[i].state.store(stepcast::store::ChunkState::COLD, std::memory_order_relaxed);
    info->metadata[i].last_touch_s.store(0, std::memory_order_relaxed);
    info->pin_refcnt[i].store(0, std::memory_order_relaxed);
  }

  models_.emplace(key, std::move(info));

  VirtualRegion region{addr, nullptr, bytes};
  VLOG(1) << "DVMP: allocated " << bytes << " bytes for model " << key << " at " << addr;
  return region;
}

absl::Span<const stepcast::store::ChunkMeta> DistributedMemoryPool::chunk_snapshot(
    std::string_view model_id) const noexcept {
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return {};
  const auto& info_sp = *info_sp_or;
  if (!info_sp || info_sp->metadata == nullptr)
    return {};
  return {info_sp->metadata.get(), info_sp->chunk_count};
}

namespace {
constexpr int kMadviseFlagsEvict = MADV_PAGEOUT; // Real page-out on modern kernels
constexpr int kMadviseFlagsFree = MADV_FREE; // Hints that pages can be reclaimed lazily
} // namespace

absl::Status DistributedMemoryPool::lock_chunks(std::string_view model_id, absl::Span<const uint32_t> idx) {
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  const std::string key(model_id);
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);

  if (idx.empty()) {
    // No-op for empty input per test expectation
    return absl::OkStatus();
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
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);
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
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return 0;
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);
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
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return;
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);
  uint64_t ts = now_s();
  for (uint32_t i : idx) {
    if (i < info.chunk_count) {
      info.metadata[i].last_touch_s.store(ts, std::memory_order_relaxed);
    }
  }
}

absl::Status DistributedMemoryPool::ensure_chunk_resident(std::string_view model_id, uint32_t chunk_idx) {
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);
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
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);
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

absl::Status DistributedMemoryPool::write_at(
    std::string_view model_id,
    uint64_t va_offset,
    const void* src,
    size_t bytes) {
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  const std::string key(model_id);
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);
  if (va_offset >= info.bytes) {
    return absl::OutOfRangeError("write_at offset beyond model size");
  }

  // Check if write would exceed bounds - make this an explicit error
  if (va_offset + bytes > info.bytes) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Write would exceed model bounds: offset=%lu + bytes=%lu > model_size=%lu", va_offset, bytes, info.bytes));
  }

  // Ensure destination range is writable; if any part is file-backed read-only,
  // try mprotect to upgrade, otherwise remap anonymously for the subrange.
  const long page = sysconf(_SC_PAGESIZE);
  uint64_t page_aligned_off = (va_offset / page) * page;
  uint64_t end_off = va_offset + bytes;
  uint64_t page_aligned_end = ((end_off + page - 1) / page) * page;
  size_t aligned_len = static_cast<size_t>(page_aligned_end - page_aligned_off);
  void* aligned_addr = static_cast<char*>(info.base) + page_aligned_off;
  if (::mprotect(aligned_addr, aligned_len, PROT_READ | PROT_WRITE) != 0) {
    void* mapped =
        ::mmap(aligned_addr, aligned_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (mapped == MAP_FAILED || mapped != aligned_addr) {
      return absl::ErrnoToStatus(errno, "DVMP write_at: failed to ensure writable mapping");
    }
  }

  void* dst = static_cast<char*>(info.base) + va_offset;
  std::memcpy(dst, src, bytes);

  // Update metadata for affected chunks
  const uint64_t first = va_offset / kChunk;
  const uint64_t last = (va_offset + bytes - 1) / kChunk;
  uint64_t ts = now_s();
  for (uint64_t i = first; i <= last && i < info.chunk_count; ++i) {
    info.metadata[i].state.store(store::ChunkState::HOT, std::memory_order_release);
    info.metadata[i].last_touch_s.store(static_cast<uint32_t>(ts), std::memory_order_relaxed);
  }
  // Metrics: bytes written
  try {
    static const stepcast::metrics::Counter kWriteBytes("dvmp_write_bytes_total");
    kWriteBytes.inc(static_cast<double>(bytes));
  } catch (...) {
  }
  return absl::OkStatus();
}

absl::Status DistributedMemoryPool::map_file_segments(std::string_view model_id, absl::Span<const FileSegment> segs) {
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);

  const long page = sysconf(_SC_PAGESIZE);
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
    if (s.populate)
      flags |= MAP_POPULATE;
#endif
    void* target = static_cast<char*>(info.base) + s.va_offset;
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
    const uint64_t first = s.va_offset / kChunk;
    const uint64_t last = (s.va_offset + s.length - 1) / kChunk;
    uint64_t ts = now_s();
    for (uint64_t i = first; i <= last && i < info.chunk_count; ++i) {
      info.metadata[i].state.store(store::ChunkState::HOT, std::memory_order_release);
      info.metadata[i].last_touch_s.store(static_cast<uint32_t>(ts), std::memory_order_relaxed);
    }
  }
  // Metrics: bytes mapped
  try {
    static const stepcast::metrics::Counter kMapBytes("dvmp_map_bytes_total");
    uint64_t total = 0;
    for (const auto& s : segs)
      total += s.length;
    kMapBytes.inc(static_cast<double>(total));
  } catch (...) {
  }
  return absl::OkStatus();
}

absl::StatusOr<DistributedMemoryPool::VirtualRegion> DistributedMemoryPool::region_info(
    std::string_view model_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(std::string(model_id));
  if (it == models_.end() || !it->second) {
    return absl::NotFoundError("Model not found in DVMP");
  }
  const auto& info_sp = it->second;
  return VirtualRegion{info_sp->base, nullptr, info_sp->bytes};
}

// PinLease impl --------------------------------------------------------------
DistributedMemoryPool::PinLease::~PinLease() {
  if (!impl_)
    return;

  // Check if the lease has expired before releasing pins
  if (is_expired()) {
    LOG(WARNING) << "PinLease expired before being destroyed for model: " << impl_->model_key;
  }

  DistributedMemoryPool* dvmp = impl_->dvmp;
  if (!dvmp)
    return;
  auto info_sp_or = dvmp->get_model_info(impl_->model_key);
  if (!info_sp_or.ok())
    return;
  auto info_sp = *info_sp_or;
  if (!info_sp)
    return;
  std::lock_guard<std::mutex> model_lock(info_sp->model_mu);
  dvmp->release_pins_unlocked(*info_sp, impl_->chunks);
}

bool DistributedMemoryPool::PinLease::is_expired() const {
  if (!impl_ || !impl_->expiry_time)
    return false;
  return std::chrono::steady_clock::now() > *impl_->expiry_time;
}

DistributedMemoryPool::PinLease::PinLease(PinLease&& other) noexcept {
  impl_ = std::move(other.impl_);
}
DistributedMemoryPool::PinLease& DistributedMemoryPool::PinLease::operator=(PinLease&& other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void DistributedMemoryPool::release_pins_unlocked(ModelInfo& info, absl::Span<const uint32_t> chunks) {
  for (uint32_t i : chunks) {
    if (i >= info.chunk_count)
      continue;
    // Decrement refcount and best-effort munlock
    auto cnt = info.pin_refcnt[i].load(std::memory_order_acquire);
    if (cnt > 0) {
      info.pin_refcnt[i].store(cnt - 1, std::memory_order_release);
    }
    void* addr = static_cast<char*>(info.base) + static_cast<size_t>(i) * kChunk;
    if (::munlock(addr, kChunk) != 0) {
      // Ignore errors; best-effort only
    }
  }
}

absl::StatusOr<DistributedMemoryPool::PinLease> DistributedMemoryPool::pin_range(
    std::string_view model_id,
    uint64_t va_offset,
    uint64_t bytes,
    std::string_view reason) {
  return pin_range(model_id, va_offset, bytes, reason, std::nullopt);
}

absl::StatusOr<DistributedMemoryPool::PinLease> DistributedMemoryPool::pin_range(
    std::string_view model_id,
    uint64_t va_offset,
    uint64_t bytes,
    std::string_view reason,
    std::optional<std::chrono::milliseconds> timeout_ms) {
  const std::string key(model_id);
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  ModelInfo& info = **info_sp_or;
  std::lock_guard<std::mutex> model_lock(info.model_mu);
  if (va_offset >= info.bytes) {
    return absl::OutOfRangeError("pin_range offset beyond model size");
  }
  uint64_t to_cover = std::min<uint64_t>(bytes, info.bytes - va_offset);
  if (to_cover == 0) {
    return absl::OutOfRangeError("pin_range length zero");
  }

  const uint32_t first = static_cast<uint32_t>(va_offset / kChunk);
  const uint32_t last = static_cast<uint32_t>((va_offset + to_cover - 1) / kChunk);
  std::vector<uint32_t> chunks;
  chunks.reserve(last - first + 1);
  for (uint32_t i = first; i <= last && i < info.chunk_count; ++i) {
    chunks.push_back(i);
    // Inc refcount and best-effort mlock
    auto prev = info.pin_refcnt[i].load(std::memory_order_acquire);
    info.pin_refcnt[i].store(prev + 1, std::memory_order_release);
    void* addr = static_cast<char*>(info.base) + static_cast<size_t>(i) * kChunk;
    ::mlock(addr, kChunk);
  }
  // Calculate expiry time if timeout is specified
  std::optional<std::chrono::steady_clock::time_point> expiry_time;
  if (timeout_ms.has_value() && timeout_ms->count() > 0) {
    expiry_time = std::chrono::steady_clock::now() + *timeout_ms;
  }

  // Metrics: record a pin-lease acquisition event for external safety/export.
  try {
    static const stepcast::metrics::Counter kPinLeasesTotal("dvmp_pin_leases_total");
    kPinLeasesTotal.with_labels({{"reason", std::string(reason)}}).inc();
  } catch (...) {
  }

  return PinLease(
      PinLease::Impl{.dvmp = this, .model_key = key, .chunks = std::move(chunks), .expiry_time = expiry_time});
}

absl::StatusOr<DistributedMemoryPool::DvmpRegion> DistributedMemoryPool::open(std::string_view model_id) {
  auto info_sp_or = get_model_info(model_id);
  if (!info_sp_or.ok())
    return info_sp_or.status();
  return DvmpRegion(this, std::string(model_id));
}

} // namespace stepcast::memory
