// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gsl/pointers"

#include "core/store/replica/chunk_meta.h"

namespace tensorcast::common::memory {

class VirtualAddressSpace {
 public:
  static constexpr size_t kDefaultChunkSize = 256ULL * 1024ULL * 1024ULL; // 256 MiB
  static constexpr absl::StatusCode kErrChunkRemote = absl::StatusCode::kUnavailable;

  struct VirtualRegion {
    void* cpu_base{nullptr}; // CPU virtual address (may be nullptr if GPU-only)
    void* gpu_base{nullptr}; // GPU virtual address (reserved but not required)
    size_t bytes{0};
  };

  explicit VirtualAddressSpace(size_t chunk_size);

  VirtualAddressSpace() : VirtualAddressSpace(kDefaultChunkSize) {}

  virtual ~VirtualAddressSpace();

  VirtualAddressSpace(const VirtualAddressSpace&) = delete;
  VirtualAddressSpace& operator=(const VirtualAddressSpace&) = delete;

  // Expose the configured chunk size for this VA instance. Callers should use
  // this rather than relying on kDefaultChunkSize to ensure global alignment.
  [[nodiscard]] size_t chunk_size() const noexcept {
    return chunk_size_;
  }

  // ===== Allocation =====
  // Reserve contiguous VA range for the whole replica. Physical pages are mapped
  // on-demand. NUMA affinity is optional. Returns VirtualRegion on success.
  virtual absl::StatusOr<VirtualRegion> allocate(std::string_view artifact_id, size_t bytes, int numa);
  // Convenience overload without NUMA argument
  virtual absl::StatusOr<VirtualRegion> allocate(std::string_view artifact_id, size_t bytes);

  // Open a per-replica VS region handle. Returns NotFound if the artifact_id was
  // not previously allocated. The handle exposes replica-scoped operations and
  // uses a per-replica lock to avoid global contention.
  class VaRegion;
  virtual absl::StatusOr<VaRegion> open(std::string_view artifact_id);

  // NEW: Query existing region information (base pointer and size) without modifying state.
  // Returns NotFound if the artifact_id does not exist.
  virtual absl::StatusOr<VirtualRegion> region_info(std::string_view artifact_id) const;

  // ===== Snapshot & State =====
  // Telemetry-only view of per-chunk metadata. VS is not authoritative.
  virtual absl::Span<const tensorcast::store::replica::ChunkMeta> chunk_telemetry_snapshot(
      std::string_view artifact_id) const noexcept;

  // Evict bytes from the tail of the replica (chunk-level granularity).
  virtual size_t evict_tail_bytes(std::string_view artifact_id, size_t bytes);
  virtual void refresh_chunks(std::string_view artifact_id, absl::Span<const uint32_t> idx);

  // Ensure the specified chunk is resident in local memory. If not available
  // locally and state==EVICTED, returns kErrChunkRemote.
  virtual absl::Status ensure_chunk_resident(std::string_view artifact_id, uint32_t chunk_idx);

  // Mark the specified chunks as PREEMPTIBLE. The actual madvise implementation
  // will be added in a subsequent iteration.
  virtual absl::Status mark_preemptible(std::string_view artifact_id, absl::Span<const uint32_t> idx);

  // ===== VS-owned IO =====
  struct FileSegment {
    std::filesystem::path path;
    uint64_t file_offset; // file offset in bytes
    uint64_t va_offset; // destination VA offset (within replica region)
    uint64_t length; // bytes to map
    bool populate{false}; // MAP_POPULATE hint
  };

  virtual absl::Status map_file_segments(std::string_view artifact_id, absl::Span<const FileSegment> segs);

  /**
   * @brief Write data to the VA region and update metadata.
   *
   * This method writes data to the CPU virtual address space and automatically
   * updates chunk metadata to reflect the write operation.
   */
  virtual absl::Status write_at(std::string_view artifact_id, uint64_t va_offset, const void* src, size_t bytes);

  // Optional: register a callback invoked after successful writes into a replica's VA range.
  // The callback receives (offset, length). Intended for UMA to update last_access telemetry.
  virtual void register_write_hook(std::string_view artifact_id, std::function<void(uint64_t off, uint64_t len)> cb);

  // ===== Pin Leases =====
  class CpuPinLease {
   public:
    CpuPinLease() = default;
    ~CpuPinLease();
    CpuPinLease(CpuPinLease&&) noexcept;
    CpuPinLease& operator=(CpuPinLease&&) noexcept;
    CpuPinLease(const CpuPinLease&) = delete;
    CpuPinLease& operator=(const CpuPinLease&) = delete;

    // Check if lease has expired (returns false if no timeout was set)
    [[nodiscard]] bool is_expired() const;

   private:
    friend class VirtualAddressSpace;
    // Release the currently held lease, if any. Safe to call multiple times.
    void release() noexcept;

    struct Impl {
      gsl::not_null<VirtualAddressSpace*> virtual_addr_space;
      std::string artifact_id;
      std::vector<uint32_t> chunks;
      std::optional<std::chrono::steady_clock::time_point> expiry_time;
    };

    explicit CpuPinLease(Impl impl) : impl_(std::make_shared<Impl>(std::move(impl))) {}

    std::shared_ptr<Impl> impl_;
  };

  virtual absl::StatusOr<CpuPinLease> pin_range(
      std::string_view artifact_id,
      uint64_t va_offset,
      uint64_t bytes,
      std::string_view reason);

  // Overload with optional timeout
  virtual absl::StatusOr<CpuPinLease> pin_range(
      std::string_view artifact_id,
      uint64_t va_offset,
      uint64_t bytes,
      std::string_view reason,
      std::optional<std::chrono::milliseconds> timeout_ms);

 private:
  struct RegionState {
    // Base CPU virtual address reserved for the replica's region
    void* cpu_base{nullptr};
    size_t bytes{0};
    std::unique_ptr<tensorcast::store::replica::ChunkMeta[]> metadata;
    size_t chunk_count{0};
    // Optional write hook per replica (called outside artifact lock)
    std::function<void(uint64_t, uint64_t)> write_hook;
    // Per-chunk pin refcounts used by CpuPinLease API
    std::unique_ptr<std::atomic<uint32_t>[]> pin_refcnt;
    // Per-chunk mlock refcounts tracking pin leases
    std::unique_ptr<std::atomic<uint32_t>[]> mlock_refcnt;
    // Per-replica mutex guarding replica-local state
    mutable std::mutex artifact_mutex;
  };

  const size_t chunk_size_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<RegionState>> artifacts_;

  // Helpers for CpuPinLease
  void release_pins_unlocked(RegionState& info, absl::Span<const uint32_t> chunks) const;

  // Lookup helper to reduce boilerplate in public methods. Returns shared_ptr
  // to ReplicaInfo or NotFound status.
  absl::StatusOr<std::shared_ptr<RegionState>> get_artifact_info(std::string_view artifact_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = artifacts_.find(std::string(artifact_id));
    if (it == artifacts_.end() || !it->second) {
      return absl::NotFoundError("Artifact not found in VA");
    }
    return it->second;
  }

  // Internal helpers -------------------------------------------------------
  static uint64_t now_s() noexcept {
    return static_cast<uint64_t>(::time(nullptr));
  }
};

// Per-replica VS region handle. Lightweight wrapper that reuses the parent
// implementation with replica scoping. Copyable and cheap (stores shared_ptr internally).
class VirtualAddressSpace::VaRegion {
 public:
  VaRegion() = default;

  absl::Status mark_preemptible(absl::Span<const uint32_t> idx) {
    return va_space_ ? va_space_->mark_preemptible(artifact_id_, idx)
                     : absl::FailedPreconditionError("null virtual_addr_space region");
  }

  absl::Status ensure_chunk_resident(uint32_t chunk_idx) {
    return va_space_ ? va_space_->ensure_chunk_resident(artifact_id_, chunk_idx)
                     : absl::FailedPreconditionError("null virtual_addr_space region");
  }

  size_t evict_tail_bytes(size_t bytes) {
    return va_space_ ? va_space_->evict_tail_bytes(artifact_id_, bytes) : 0;
  }

  void refresh_chunks(absl::Span<const uint32_t> idx) {
    if (va_space_) {
      va_space_->refresh_chunks(artifact_id_, idx);
    }
  }

  absl::Status map_file_segments(absl::Span<const FileSegment> segs) {
    return va_space_ ? va_space_->map_file_segments(artifact_id_, segs)
                     : absl::FailedPreconditionError("null virtual_addr_space region");
  }

  absl::Status write_at(uint64_t va_offset, const void* src, size_t bytes) {
    return va_space_ ? va_space_->write_at(artifact_id_, va_offset, src, bytes)
                     : absl::FailedPreconditionError("null virtual_addr_space region");
  }

  absl::StatusOr<CpuPinLease> pin_range(uint64_t va_offset, uint64_t bytes, std::string_view reason) {
    return va_space_ ? va_space_->pin_range(artifact_id_, va_offset, bytes, reason)
                     : absl::FailedPreconditionError("null virtual_addr_space region");
  }

  absl::StatusOr<CpuPinLease> pin_range(
      uint64_t va_offset,
      uint64_t bytes,
      std::string_view reason,
      std::optional<std::chrono::milliseconds> timeout_ms) {
    return va_space_ ? va_space_->pin_range(artifact_id_, va_offset, bytes, reason, timeout_ms)
                     : absl::FailedPreconditionError("null virtual_addr_space region");
  }

  void register_write_hook(std::function<void(uint64_t off, uint64_t len)> cb) {
    if (va_space_) {
      va_space_->register_write_hook(artifact_id_, std::move(cb));
    }
  }

 private:
  friend class VirtualAddressSpace;

  VaRegion(VirtualAddressSpace* virtual_addr_space, std::string artifact_id)
      : va_space_(virtual_addr_space), artifact_id_(std::move(artifact_id)) {}

  VirtualAddressSpace* va_space_{nullptr};
  std::string artifact_id_;
};

} // namespace tensorcast::common::memory
