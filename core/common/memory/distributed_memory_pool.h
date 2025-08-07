// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstddef>
#include <ctime>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "core/store/model/chunk_meta.h"

namespace stepcast::memory {

class DistributedMemoryPool {
 public:
  static constexpr size_t kChunk = 256ULL * 1024ULL * 1024ULL; // 256 MiB
  static constexpr absl::StatusCode kErrChunkRemote = absl::StatusCode::kUnavailable;

  struct VirtualRegion {
    void* cpu_base{nullptr}; // CPU virtual address (may be nullptr if GPU-only)
    void* gpu_base{nullptr}; // GPU virtual address (reserved but not required)
    size_t bytes{0};
  };

  DistributedMemoryPool() = default;
  virtual ~DistributedMemoryPool();

  DistributedMemoryPool(const DistributedMemoryPool&) = delete;
  DistributedMemoryPool& operator=(const DistributedMemoryPool&) = delete;

  // ===== Allocation =====
  // Reserve contiguous VA range for the whole model. Physical pages are mapped
  // on-demand. NUMA affinity is optional. Returns VirtualRegion on success.
  virtual absl::StatusOr<VirtualRegion> allocate(std::string_view model_id, size_t bytes, int numa = -1);

  // ===== Snapshot & State =====
  virtual absl::Span<const stepcast::store::ChunkMeta> chunk_snapshot(std::string_view model_id) const noexcept;

  // Lock/unlock a set of chunks for H2D or P2P transfer.
  virtual absl::Status lock_chunks(std::string_view model_id, absl::Span<const uint32_t> idx);
  virtual absl::Status unlock_chunks(std::string_view model_id, absl::Span<const uint32_t> idx, bool copied_gpu);

  // Evict bytes from the tail of the model (chunk-level granularity).
  virtual size_t evict_tail_bytes(std::string_view model_id, size_t bytes);
  virtual void refresh_chunks(std::string_view model_id, absl::Span<const uint32_t> idx);

  // Ensure the specified chunk is resident in local memory. If not available
  // locally and state==EVICTED, returns kErrChunkRemote.
  virtual absl::Status ensure_chunk_resident(std::string_view model_id, uint32_t chunk_idx);

  // Mark the specified chunks as PREEMPTIBLE. The actual madvise implementation
  // will be added in a subsequent iteration.
  virtual absl::Status mark_preemptible(std::string_view model_id, absl::Span<const uint32_t> idx);

  // ===== DVMP-owned IO =====
  struct FileSegment {
    std::filesystem::path path;
    uint64_t file_offset; // file offset in bytes
    uint64_t va_offset; // destination VA offset (within model region)
    uint64_t length; // bytes to map
    bool populate{false}; // MAP_POPULATE hint
  };

  virtual absl::Status map_file_segments(std::string_view model_id, absl::Span<const FileSegment> segs);

  virtual absl::Status write_at(std::string_view model_id, uint64_t va_offset, const void* src, size_t bytes);

  // ===== Pin Leases =====
  class PinLease {
   public:
    PinLease() = default;
    ~PinLease();
    PinLease(PinLease&&) noexcept;
    PinLease& operator=(PinLease&&) noexcept;
    PinLease(const PinLease&) = delete;
    PinLease& operator=(const PinLease&) = delete;

   private:
    friend class DistributedMemoryPool;
    struct Impl {
      DistributedMemoryPool* dvmp;
      std::string model_key;
      std::vector<uint32_t> chunks;
    };
    explicit PinLease(Impl impl) : impl_(std::make_shared<Impl>(std::move(impl))) {}
    std::shared_ptr<Impl> impl_;
  };

  virtual absl::StatusOr<PinLease> pin_range(
      std::string_view model_id,
      uint64_t va_offset,
      uint64_t bytes,
      std::string_view reason);

 private:
  struct ModelInfo {
    void* base{nullptr};
    size_t bytes{0};
    std::unique_ptr<stepcast::store::ChunkMeta[]> metadata;
    size_t chunk_count{0};
    // Per-chunk pin refcounts used by PinLease API
    std::unique_ptr<std::atomic<uint32_t>[]> pin_refcnt;
  };

  mutable std::mutex mutex_;
  std::unordered_map<std::string, ModelInfo> models_;

  // Helpers for PinLease
  void release_pins_unlocked(ModelInfo& info, absl::Span<const uint32_t> chunks) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Internal helpers -------------------------------------------------------
  static uint64_t now_s() noexcept {
    return static_cast<uint64_t>(::time(nullptr));
  }
};

} // namespace stepcast::memory
