// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <filesystem>
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

namespace stepcast::memory {

class DistributedVirtualMemoryPool {
 public:
  static constexpr size_t kDefaultChunkSize = 256ULL * 1024ULL * 1024ULL; // 256 MiB
  static constexpr absl::StatusCode kErrChunkRemote = absl::StatusCode::kUnavailable;

  struct VirtualRegion {
    void* cpu_base{nullptr}; // CPU virtual address (may be nullptr if GPU-only)
    void* gpu_base{nullptr}; // GPU virtual address (reserved but not required)
    size_t bytes{0};
  };

  explicit DistributedVirtualMemoryPool(size_t chunk_size);
  DistributedVirtualMemoryPool() : DistributedVirtualMemoryPool(kDefaultChunkSize) {}
  virtual ~DistributedVirtualMemoryPool();

  DistributedVirtualMemoryPool(const DistributedVirtualMemoryPool&) = delete;
  DistributedVirtualMemoryPool& operator=(const DistributedVirtualMemoryPool&) = delete;

  // ===== Allocation =====
  // Reserve contiguous VA range for the whole replica. Physical pages are mapped
  // on-demand. NUMA affinity is optional. Returns VirtualRegion on success.
  virtual absl::StatusOr<VirtualRegion> allocate(std::string_view artifact_id, size_t bytes, int numa);
  // Convenience overload without NUMA argument
  virtual absl::StatusOr<VirtualRegion> allocate(std::string_view artifact_id, size_t bytes);

  // Open a per-replica DVMP region handle. Returns NotFound if the artifact_id was
  // not previously allocated. The handle exposes replica-scoped operations and
  // uses a per-replica lock to avoid global contention.
  class DvmpRegion;
  virtual absl::StatusOr<DvmpRegion> open(std::string_view artifact_id);

  // NEW: Query existing region information (base pointer and size) without modifying state.
  // Returns NotFound if the artifact_id does not exist.
  virtual absl::StatusOr<VirtualRegion> region_info(std::string_view artifact_id) const;

  // ===== Snapshot & State =====
  virtual absl::Span<const stepcast::store::ChunkMeta> chunk_snapshot(std::string_view artifact_id) const noexcept;

  // Lock/unlock a set of chunks for H2D or P2P transfer.
  virtual absl::Status lock_chunks(std::string_view artifact_id, absl::Span<const uint32_t> idx);
  virtual absl::Status unlock_chunks(std::string_view artifact_id, absl::Span<const uint32_t> idx, bool copied_gpu);

  // Evict bytes from the tail of the replica (chunk-level granularity).
  virtual size_t evict_tail_bytes(std::string_view artifact_id, size_t bytes);
  virtual void refresh_chunks(std::string_view artifact_id, absl::Span<const uint32_t> idx);

  // Ensure the specified chunk is resident in local memory. If not available
  // locally and state==EVICTED, returns kErrChunkRemote.
  virtual absl::Status ensure_chunk_resident(std::string_view artifact_id, uint32_t chunk_idx);

  // Mark the specified chunks as PREEMPTIBLE. The actual madvise implementation
  // will be added in a subsequent iteration.
  virtual absl::Status mark_preemptible(std::string_view artifact_id, absl::Span<const uint32_t> idx);

  // ===== DVMP-owned IO =====
  struct FileSegment {
    std::filesystem::path path;
    uint64_t file_offset; // file offset in bytes
    uint64_t va_offset; // destination VA offset (within replica region)
    uint64_t length; // bytes to map
    bool populate{false}; // MAP_POPULATE hint
  };

  virtual absl::Status map_file_segments(std::string_view artifact_id, absl::Span<const FileSegment> segs);

  /**
   * @brief Write data to the DVMP region and update metadata.
   *
   * This method writes data to the CPU virtual address space managed by DVMP
   * and automatically updates chunk metadata to reflect the write operation.
   * It ensures CPU metadata visibility by updating chunk states to at least HOT
   * and refreshing last_touch_s timestamps.
   *
   * @param artifact_id The replica identifier
   * @param va_offset Offset within the virtual address region
   * @param src Source data pointer
   * @param bytes Number of bytes to write
   * @return Status indicating success or failure
   *
   * @note Updates chunk residency state (at least HOT) and last_touch_s for touched chunks
   * @note Thread-safe and ensures metadata consistency
   */
  virtual absl::Status write_at(std::string_view artifact_id, uint64_t va_offset, const void* src, size_t bytes);

  // ===== Pin Leases =====
  class ChunkResidencyLease {
   public:
    ChunkResidencyLease() = default;
    ~ChunkResidencyLease();
    ChunkResidencyLease(ChunkResidencyLease&&) noexcept;
    ChunkResidencyLease& operator=(ChunkResidencyLease&&) noexcept;
    ChunkResidencyLease(const ChunkResidencyLease&) = delete;
    ChunkResidencyLease& operator=(const ChunkResidencyLease&) = delete;

    // Check if lease has expired (returns false if no timeout was set)
    [[nodiscard]] bool is_expired() const;

   private:
    friend class DistributedVirtualMemoryPool;
    // Release the currently held lease, if any. Safe to call multiple times.
    void release() noexcept;
    struct Impl {
      gsl::not_null<DistributedVirtualMemoryPool*> dvmp;
      std::string artifact_id;
      std::vector<uint32_t> chunks;
      std::optional<std::chrono::steady_clock::time_point> expiry_time;
    };
    explicit ChunkResidencyLease(Impl impl) : impl_(std::make_shared<Impl>(std::move(impl))) {}
    std::shared_ptr<Impl> impl_;
  };

  virtual absl::StatusOr<ChunkResidencyLease> pin_range(
      std::string_view artifact_id,
      uint64_t va_offset,
      uint64_t bytes,
      std::string_view reason);

  // Overload with optional timeout
  virtual absl::StatusOr<ChunkResidencyLease> pin_range(
      std::string_view artifact_id,
      uint64_t va_offset,
      uint64_t bytes,
      std::string_view reason,
      std::optional<std::chrono::milliseconds> timeout_ms);

 private:
  struct DvmpRegionState {
    // Base CPU virtual address reserved for the replica's region
    void* cpu_base{nullptr};
    size_t bytes{0};
    std::unique_ptr<stepcast::store::ChunkMeta[]> metadata;
    size_t chunk_count{0};
    // Per-chunk pin refcounts used by ChunkResidencyLease API
    std::unique_ptr<std::atomic<uint32_t>[]> pin_refcnt;
    // Per-chunk mlock refcounts tracking both transfer locks and pin leases
    std::unique_ptr<std::atomic<uint32_t>[]> mlock_refcnt;
    // Per-replica mutex guarding replica-local state; global mutex_ only protects
    // the models_ map and retrieval of shared_ptrs.
    // Per-replica mutex guarding replica-local state
    mutable std::mutex artifact_mutex;
  };

  const size_t chunk_size_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<DvmpRegionState>> artifacts_;

  // Helpers for ChunkResidencyLease
  void release_pins_unlocked(DvmpRegionState& info, absl::Span<const uint32_t> chunks) const;

  // Lookup helper to reduce boilerplate in public methods. Returns shared_ptr
  // to ReplicaInfo or NotFound status.
  absl::StatusOr<std::shared_ptr<DvmpRegionState>> get_artifact_info(std::string_view artifact_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = artifacts_.find(std::string(artifact_id));
    if (it == artifacts_.end() || !it->second) {
      return absl::NotFoundError("Artifact not found in DVMP");
    }
    return it->second;
  }

  // Internal helpers -------------------------------------------------------
  static uint64_t now_s() noexcept {
    return static_cast<uint64_t>(::time(nullptr));
  }
};

// Per-replica DVMP region handle. Lightweight wrapper that reuses the parent
// DVMP implementation with replica scoping. Copyable and cheap (stores shared
//_ptr to ReplicaInfo internally).
class DistributedVirtualMemoryPool::DvmpRegion {
 public:
  DvmpRegion() = default;
  absl::Status lock_chunks(absl::Span<const uint32_t> idx) {
    return dvmp_ ? dvmp_->lock_chunks(artifact_id_, idx) : absl::FailedPreconditionError("null dvmp region");
  }
  absl::Status unlock_chunks(absl::Span<const uint32_t> idx, bool copied_gpu) {
    return dvmp_ ? dvmp_->unlock_chunks(artifact_id_, idx, copied_gpu)
                 : absl::FailedPreconditionError("null dvmp region");
  }
  absl::Status mark_preemptible(absl::Span<const uint32_t> idx) {
    return dvmp_ ? dvmp_->mark_preemptible(artifact_id_, idx) : absl::FailedPreconditionError("null dvmp region");
  }
  absl::Status ensure_chunk_resident(uint32_t chunk_idx) {
    return dvmp_ ? dvmp_->ensure_chunk_resident(artifact_id_, chunk_idx)
                 : absl::FailedPreconditionError("null dvmp region");
  }
  size_t evict_tail_bytes(size_t bytes) {
    return dvmp_ ? dvmp_->evict_tail_bytes(artifact_id_, bytes) : 0;
  }
  void refresh_chunks(absl::Span<const uint32_t> idx) {
    if (dvmp_) {
      dvmp_->refresh_chunks(artifact_id_, idx);
    }
  }
  absl::Status map_file_segments(absl::Span<const FileSegment> segs) {
    return dvmp_ ? dvmp_->map_file_segments(artifact_id_, segs) : absl::FailedPreconditionError("null dvmp region");
  }
  absl::Status write_at(uint64_t va_offset, const void* src, size_t bytes) {
    return dvmp_ ? dvmp_->write_at(artifact_id_, va_offset, src, bytes)
                 : absl::FailedPreconditionError("null dvmp region");
  }
  absl::StatusOr<ChunkResidencyLease> pin_range(uint64_t va_offset, uint64_t bytes, std::string_view reason) {
    return dvmp_ ? dvmp_->pin_range(artifact_id_, va_offset, bytes, reason)
                 : absl::FailedPreconditionError("null dvmp region");
  }
  absl::StatusOr<ChunkResidencyLease> pin_range(
      uint64_t va_offset,
      uint64_t bytes,
      std::string_view reason,
      std::optional<std::chrono::milliseconds> timeout_ms) {
    return dvmp_ ? dvmp_->pin_range(artifact_id_, va_offset, bytes, reason, timeout_ms)
                 : absl::FailedPreconditionError("null dvmp region");
  }

 private:
  friend class DistributedVirtualMemoryPool;
  DvmpRegion(DistributedVirtualMemoryPool* dvmp, std::string artifact_id)
      : dvmp_(dvmp), artifact_id_(std::move(artifact_id)) {}
  DistributedVirtualMemoryPool* dvmp_{nullptr};
  std::string artifact_id_;
};

} // namespace stepcast::memory
