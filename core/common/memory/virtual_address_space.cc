// Copyright (c) 2025, TensorCast Team.

#include "core/common/memory/virtual_address_space.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "core/common/system_capabilities.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::common::memory {

// UMA is the sole ledger authority in V3 final state; VS never mutates
// authoritative chunk state, only telemetry and OS advisories.

VirtualAddressSpace::VirtualAddressSpace(size_t chunk_size) : chunk_size_(chunk_size) {
  LOG(INFO) << "Initialized VirtualAddressSpace with chunk size: " << chunk_size_ / (1024 * 1024) << " MiB";
  // Ensure capabilities are detected early even if communicator is not constructed.
  SystemCapabilities::instance();
  // Warn once per process if mlock/munlock are unavailable so later paths can be quiet.
  static std::once_flag warn_once;
  std::call_once(warn_once, []() {
    if (!SystemCapabilities::instance().mlock_enabled()) {
      LOG(WARNING)
          << "VA: mlock/munlock unavailable; page pinning is disabled. unlock() will not physically unpin pages.";
    }
  });
}

VirtualAddressSpace::~VirtualAddressSpace() {
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

absl::StatusOr<VirtualAddressSpace::VirtualRegion> VirtualAddressSpace::allocate(
    std::string_view artifact_id,
    size_t bytes,
    int /*numa*/) {
  std::lock_guard<std::mutex> lock(mutex_);
  const std::string key(artifact_id);
  if (artifacts_.find(key) != artifacts_.end()) {
    return absl::AlreadyExistsError("Artifact replica already has an allocated region");
  }

  void* addr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (addr == MAP_FAILED) {
    return absl::ErrnoToStatus(errno, "mmap failed while reserving VA space");
  }

  const size_t num_chunks = (bytes + chunk_size_ - 1) / chunk_size_;
  auto info = std::make_shared<RegionState>();
  info->cpu_base = addr;
  info->bytes = bytes;
  info->metadata = std::make_unique<store::replica::ChunkMeta[]>(num_chunks);
  info->chunk_count = num_chunks;
  info->pin_refcnt = std::make_unique<std::atomic<uint32_t>[]>(num_chunks);
  info->mlock_refcnt = std::make_unique<std::atomic<uint32_t>[]>(num_chunks);
  for (size_t i = 0; i < num_chunks; ++i) {
    info->metadata[i].state.store(store::replica::ChunkState::COLD, std::memory_order_relaxed);
    info->metadata[i].last_touch_s.store(0, std::memory_order_relaxed);
    info->pin_refcnt[i].store(0, std::memory_order_relaxed);
    info->mlock_refcnt[i].store(0, std::memory_order_relaxed);
  }

  artifacts_.emplace(key, std::move(info));

  VirtualRegion region{addr, nullptr, bytes};
  VLOG(1) << "VA: allocated " << bytes << " bytes for replica " << key << " at " << addr;
  return region;
}

absl::StatusOr<VirtualAddressSpace::VirtualRegion> VirtualAddressSpace::allocate(
    std::string_view artifact_id,
    size_t bytes) {
  return allocate(artifact_id, bytes, -1);
}

absl::Span<const store::replica::ChunkMeta> VirtualAddressSpace::chunk_telemetry_snapshot(
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

size_t VirtualAddressSpace::evict_tail_bytes(std::string_view artifact_id, size_t bytes) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return 0;
  }
  RegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  size_t freed = 0;
  for (ssize_t idx = static_cast<ssize_t>(info.chunk_count) - 1; idx >= 0 && freed < bytes; --idx) {
    auto& meta = info.metadata[idx];
    store::replica::ChunkState st = meta.state.load(std::memory_order_acquire);
    if (info.pin_refcnt[idx].load(std::memory_order_acquire) > 0) {
      continue;
    }
    if (st == store::replica::ChunkState::LOCKED_TX || st == store::replica::ChunkState::EVICTED) {
      continue;
    }
    if (st == store::replica::ChunkState::HOT || st == store::replica::ChunkState::COLD ||
        st == store::replica::ChunkState::COPIED_GPU || st == store::replica::ChunkState::PREEMPTIBLE) {
      void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(idx) * chunk_size_;
      size_t advise_len = chunk_size_;
      int madv_flag = MADV_DONTNEED;
#ifdef MADV_PAGEOUT
      if (SystemCapabilities::instance().madv_pageout_available()) {
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

void VirtualAddressSpace::refresh_chunks(std::string_view artifact_id, absl::Span<const uint32_t> idx) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return;
  }
  RegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  uint64_t ts = now_s();
  for (uint32_t i : idx) {
    if (i < info.chunk_count) {
      info.metadata[i].last_touch_s.store(ts, std::memory_order_relaxed);
    }
  }
}

absl::Status VirtualAddressSpace::ensure_chunk_resident(std::string_view artifact_id, uint32_t chunk_idx) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  RegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  if (chunk_idx >= info.chunk_count) {
    return absl::OutOfRangeError("Chunk index out of range");
  }
  auto st = info.metadata[chunk_idx].state.load(std::memory_order_acquire);
  if (st == store::replica::ChunkState::EVICTED) {
    return {kErrChunkRemote, "Chunk data not resident locally"};
  }
  return absl::OkStatus();
}

absl::Status VirtualAddressSpace::mark_preemptible(std::string_view artifact_id, absl::Span<const uint32_t> idx) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  RegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  for (uint32_t i : idx) {
    if (i >= info.chunk_count) {
      return absl::OutOfRangeError("Chunk index out of range");
    }
    auto& meta = info.metadata[i];
    if (info.pin_refcnt[i].load(std::memory_order_acquire) > 0) {
      continue;
    }
    store::replica::ChunkState expected = meta.state.load(std::memory_order_acquire);
    if (expected == store::replica::ChunkState::HOT || expected == store::replica::ChunkState::COLD ||
        expected == store::replica::ChunkState::COPIED_GPU) {
      void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
      int rc = 0;
      if (SystemCapabilities::instance().madv_free_available()) {
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
  return absl::OkStatus();
}

absl::Status VirtualAddressSpace::write_at(
    std::string_view artifact_id,
    uint64_t va_offset,
    const void* src,
    size_t bytes) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }

  RegionState& info = **info_sp_or;
  std::function<void(uint64_t, uint64_t)> hook_copy;
  {
    std::lock_guard<std::mutex> artifact(info.artifact_mutex);
    if (va_offset >= info.bytes) {
      return absl::OutOfRangeError("write_at offset beyond artifact size");
    }
    if (va_offset + bytes > info.bytes) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "Write would exceed replica bounds: offset=%lu + bytes=%lu > artifact_size=%lu",
              va_offset,
              bytes,
              info.bytes));
    }

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
        return absl::ErrnoToStatus(errno, "VA write_at: failed to ensure writable mapping");
      }
    }

    if (SystemCapabilities::instance().madv_willneed_available()) {
      int rc = ::madvise(aligned_addr, aligned_len, MADV_WILLNEED);
      if (rc != 0 && errno != EINVAL) {
        PLOG(WARNING) << "madvise WILLNEED failed in write_at";
      }
    }

    void* dst = static_cast<char*>(info.cpu_base) + va_offset;
    std::memcpy(dst, src, bytes);

    const uint64_t first = va_offset / chunk_size_;
    const uint64_t last = (va_offset + bytes - 1) / chunk_size_;
    uint64_t ts = now_s();
    for (uint64_t i = first; i <= last && i < info.chunk_count; ++i) {
      info.metadata[i].last_touch_s.store(static_cast<uint32_t>(ts), std::memory_order_relaxed);
    }
    hook_copy = info.write_hook;
  }
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_va_write_bytes_total");
    counter->Add(static_cast<double>(bytes));
  } catch (...) {
    VLOG(1) << "metrics counter tc_va_write_bytes_total unavailable";
  }
  if (hook_copy) {
    hook_copy(va_offset, bytes);
  }
  return absl::OkStatus();
}

absl::Status VirtualAddressSpace::map_file_segments(std::string_view artifact_id, absl::Span<const FileSegment> segs) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  RegionState& info = **info_sp_or;
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

    const uint64_t first = s.va_offset / chunk_size_;
    const uint64_t last = (s.va_offset + s.length - 1) / chunk_size_;
    uint64_t ts = now_s();
    for (uint64_t i = first; i <= last && i < info.chunk_count; ++i) {
      // UMA is the sole ledger; VS only updates telemetry in final cutover.
      info.metadata[i].last_touch_s.store(static_cast<uint32_t>(ts), std::memory_order_relaxed);
    }
  }
  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_va_map_bytes_total");
    uint64_t total = 0;
    for (const auto& s : segs) {
      total += s.length;
    }
    counter->Add(static_cast<double>(total));
  } catch (...) {
    VLOG(1) << "metrics counter tc_va_map_bytes_total unavailable";
  }
  return absl::OkStatus();
}

absl::StatusOr<VirtualAddressSpace::VirtualRegion> VirtualAddressSpace::region_info(
    std::string_view artifact_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = artifacts_.find(std::string(artifact_id));
  if (it == artifacts_.end() || !it->second) {
    return absl::NotFoundError("Artifact not found in VA");
  }
  const auto& info_sp = it->second;
  return VirtualRegion{.cpu_base = info_sp->cpu_base, .gpu_base = nullptr, .bytes = info_sp->bytes};
}

// CpuPinLease impl
VirtualAddressSpace::CpuPinLease::~CpuPinLease() {
  release();
}

bool VirtualAddressSpace::CpuPinLease::is_expired() const {
  if (!impl_ || !impl_->expiry_time)
    return false;
  return std::chrono::steady_clock::now() > *impl_->expiry_time;
}

VirtualAddressSpace::CpuPinLease::CpuPinLease(CpuPinLease&& other) noexcept {
  impl_ = std::move(other.impl_);
}

VirtualAddressSpace::CpuPinLease& VirtualAddressSpace::CpuPinLease::operator=(CpuPinLease&& other) noexcept {
  if (this != &other) {
    release();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

void VirtualAddressSpace::CpuPinLease::release() noexcept {
  if (!impl_)
    return;
  if (is_expired()) {
    LOG(WARNING) << "CpuPinLease expired before being destroyed for replica: " << impl_->artifact_id;
  }
  gsl::not_null<VirtualAddressSpace*> virtual_addr_space = impl_->virtual_addr_space;
  if (!virtual_addr_space) {
    impl_.reset();
    return;
  }
  auto info_sp_or = virtual_addr_space->get_artifact_info(impl_->artifact_id);
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
  virtual_addr_space->release_pins_unlocked(*info_sp, impl_->chunks);
  impl_.reset();
}

absl::StatusOr<VirtualAddressSpace::CpuPinLease> VirtualAddressSpace::pin_range(
    std::string_view artifact_id,
    uint64_t va_offset,
    uint64_t bytes,
    std::string_view reason) {
  return pin_range(artifact_id, va_offset, bytes, reason, std::nullopt);
}

absl::StatusOr<VirtualAddressSpace::CpuPinLease> VirtualAddressSpace::pin_range(
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
  RegionState& info = **info_sp_or;
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
    auto prev = info.pin_refcnt[i].load(std::memory_order_acquire);
    info.pin_refcnt[i].store(prev + 1, std::memory_order_release);
    void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
    if (SystemCapabilities::instance().mlock_enabled()) {
      if (::mlock(addr, chunk_size_) == 0) {
        info.mlock_refcnt[i].fetch_add(1, std::memory_order_acq_rel);
      } else {
        const int err = errno;
        if (err == ENOMEM || err == EPERM) {
          static std::atomic<bool> warned_demote{false};
          if (!warned_demote.exchange(true, std::memory_order_acq_rel)) {
            PLOG(WARNING) << "mlock failed (EPERM/ENOMEM) — disabling page pinning; proceeding without mlock/munlock";
          }
          SystemCapabilities::instance().set_mlock_enabled(false);
        }
      }
    }
  }
  std::optional<std::chrono::steady_clock::time_point> expiry_time;
  if (timeout_ms.has_value() && timeout_ms->count() > 0) {
    expiry_time = std::chrono::steady_clock::now() + *timeout_ms;
  }

  try {
    static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto counter = meter->CreateDoubleCounter("tc_va_pin_leases_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string(reason)));
    counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
    VLOG(1) << "metrics counter tc_va_pin_leases_total unavailable";
  }

  return CpuPinLease(
      CpuPinLease::Impl{
          .virtual_addr_space = this, .artifact_id = key, .chunks = std::move(chunks), .expiry_time = expiry_time});
}

absl::StatusOr<VirtualAddressSpace::VaRegion> VirtualAddressSpace::open(std::string_view artifact_id) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return info_sp_or.status();
  }
  return VaRegion(this, std::string(artifact_id));
}

void VirtualAddressSpace::register_write_hook(
    std::string_view artifact_id,
    std::function<void(uint64_t off, uint64_t len)> cb) {
  auto info_sp_or = get_artifact_info(artifact_id);
  if (!info_sp_or.ok()) {
    return; // silently ignore for non-existent artifact
  }
  RegionState& info = **info_sp_or;
  std::lock_guard<std::mutex> artifact(info.artifact_mutex);
  info.write_hook = std::move(cb);
}

void VirtualAddressSpace::release_pins_unlocked(RegionState& info, absl::Span<const uint32_t> chunks) const {
  for (uint32_t i : chunks) {
    if (i >= info.chunk_count)
      continue;
    // Drop pin refcount
    auto prev_pin = info.pin_refcnt[i].load(std::memory_order_acquire);
    if (prev_pin > 0) {
      info.pin_refcnt[i].store(prev_pin - 1, std::memory_order_release);
    }
    // If there remain extra mlock refs beyond pins, drop one mlock
    auto m_prev = info.mlock_refcnt[i].load(std::memory_order_acquire);
    auto p_now = info.pin_refcnt[i].load(std::memory_order_acquire);
    if (m_prev > p_now && SystemCapabilities::instance().mlock_enabled()) {
      auto prev = info.mlock_refcnt[i].fetch_sub(1, std::memory_order_acq_rel);
      if (prev == 1) {
        void* addr = static_cast<char*>(info.cpu_base) + static_cast<size_t>(i) * chunk_size_;
        if (::munlock(addr, chunk_size_) != 0) {
          LOG(WARNING) << "munlock failed during release_pins_unlocked for chunk " << i;
        }
      }
    }
  }
}

} // namespace tensorcast::common::memory
