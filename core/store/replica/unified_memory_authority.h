// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gsl/pointers"

#include "core/common/memory/cuda_memory.h"
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/replica/chunk_state.h"
#include "core/store/replica/types/direct_write_grant.h"

namespace tensorcast::store::replica {

/**
 * @brief Unified memory management for artifacts/replicas across DRAM and VRAM.
 *
 * This class manages chunk-based memory state for replicas, tracking:
 * - One contiguous DRAM allocation per replica (via VS)
 * - One contiguous VRAM allocation per GPU device
 * - Per-chunk state tracking for fine-grained memory management
 *
 * The design enables efficient P2P transfers, intelligent memory eviction,
 * and automatic memory reclamation while maintaining zero-copy access.
 */
class UnifiedMemoryAuthority {
 public:
  struct Options {
    // When true, CPU allocations are backed by memfd + MAP_SHARED so they can be
    // exported cross-process for zero-copy CPU materialization.
    bool cpu_shared_memory_enabled{true};
  };

  struct ArtifactLayout {
    uint64_t artifact_bytes{0};
    size_t artifact_chunk_bytes{0};
    size_t transfer_slice_bytes{0};
  };

  struct TransferPlan {
    uint64_t session_id{0};
    // Chunk-aligned byte ranges (offset, length)
    std::vector<std::pair<uint64_t, size_t>> ranges;
    // Explicit chunk indices included in plan
    std::vector<uint32_t> chunk_indices;
    // Optional direct-write grant for CPU target
    std::optional<DirectWriteGrant> cpu_direct_grant;
  };

  struct StableLease {
    loading::ReplicaKey key;
    std::vector<uint32_t> chunk_indices;
    uint64_t bytes{0};
    uint64_t ledger_version{0};
  };

  /**
   * @brief Chunk source information for loading operations.
   */
  struct ChunkSource {
    enum Type : std::uint8_t {
      LOCAL_CPU, // Available in local DRAM
      LOCAL_GPU, // Available on local GPU
      REMOTE_P2P, // Available via P2P transfer
      DISK // Must load from disk
    };

    Type type;
    int device_id{-1}; // For LOCAL_GPU sources
    std::string remote_node; // For REMOTE_P2P sources
  };

  struct ChunkRecordView {
    uint32_t chunk_idx{0};
    ChunkState cpu{ChunkState::COLD};
    bool exported_cpu{false};
    uint32_t pin_refcnt{0};
    uint32_t stable_lease_count{0};
    uint64_t last_access_ns{0};
    uint64_t version{0};
  };

  // Snapshot helpers for telemetry dashboards (immutable views)
  std::vector<ChunkRecordView> snapshot_cpu_chunks(const loading::ReplicaKey& key) const;

  explicit UnifiedMemoryAuthority(size_t artifact_chunk_bytes);
  explicit UnifiedMemoryAuthority(size_t artifact_chunk_bytes, Options options);
  ~UnifiedMemoryAuthority() = default;

  // Disable copy/move
  UnifiedMemoryAuthority(const UnifiedMemoryAuthority&) = delete;
  UnifiedMemoryAuthority& operator=(const UnifiedMemoryAuthority&) = delete;

  /**
   * @brief Allocate unified memory for a replica.
   *
   * Reserves virtual address space in DRAM via VS. GPU allocations
   * are created lazily on first use.
   *
   * @param key Replica instance key
   * @param bytes Total artifact size
   * @return Status of allocation
   */
  absl::Status allocate(const loading::ReplicaKey& key, size_t bytes);

  /**
   * @brief Get or create GPU allocation for a device.
   *
   * Lazily creates VRAM allocation on first access to minimize
   * upfront memory cost on unused GPUs.
   *
   * @param key Replica instance key
   * @param device_id GPU device ID
   * @return GpuDeviceMemory pointer or error
   */
  absl::StatusOr<std::shared_ptr<common::memory::GpuDeviceMemory>> get_or_create_gpu_allocation(
      const loading::ReplicaKey& key,
      int device_id);

  /**
   * @brief Get missing chunks for a target location.
   *
   * @param key Replica instance key
   * @param target Target memory location
   * @param device_id Device ID for GPU targets
   * @return List of missing chunk indices
   */
  std::vector<uint32_t> get_missing_chunks(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation target,
      std::optional<int> device_id = std::nullopt) const;

  /**
   * @brief Get best source for loading a chunk.
   *
   * Prioritizes based on:
   * 1. Local memory (CPU/GPU)
   * 2. Remote P2P transfer
   * 3. Disk loading
   *
   * @param key Replica instance key
   * @param chunk_idx Chunk index
   * @param target Target location
   * @return Best available source
   */
  ChunkSource get_best_source_for_chunk(
      const loading::ReplicaKey& key,
      uint32_t chunk_idx,
      common::memory::MemoryLocation target) const;

  /**
   * @brief Get memory statistics for a replica.
   *
   * @param key Replica instance key
   * @return Map of location to allocated bytes
   */
  std::unordered_map<common::memory::MemoryLocation, size_t> get_memory_stats(const loading::ReplicaKey& key) const;

  /**
   * @brief Check if replica has any allocations.
   *
   * @param key Replica instance key
   * @return True if replica has DRAM or VRAM allocations
   */
  bool has_allocation(const loading::ReplicaKey& key) const;

  /**
   * @brief Convenience: return the UMA artifact chunk size (bytes).
   */
  size_t get_artifact_chunk_bytes() const;

  // Record-based queries (preferred over deprecated mapping view)
  absl::StatusOr<ChunkState> get_cpu_chunk_state(const loading::ReplicaKey& key, uint32_t chunk_idx) const;
  absl::StatusOr<ChunkState> get_gpu_chunk_state(const loading::ReplicaKey& key, int device_id, uint32_t chunk_idx)
      const;

  /**
   * @brief Retrieve the artifact layout (bytes, chunk size, slice size).
   *
   * transfer_slice_bytes is reported as 0 here (caller uses pinned pool slice
   * size). UMA does not consult environment variables in V3; data-plane
   * window size comes from the configured PinnedBufferPool.
   */
  absl::StatusOr<ArtifactLayout> get_layout(const loading::ReplicaKey& key) const;

  // Phase 2 — Transactional transfer (lightweight, idempotent)
  absl::StatusOr<TransferPlan> plan_load(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation target,
      std::optional<int> device_id,
      std::optional<absl::Span<const uint32_t>> chunk_indices);

  absl::Status commit(
      uint64_t session_id,
      common::memory::MemoryLocation target,
      absl::Span<const uint32_t> committed_chunks,
      std::optional<int> device_id);

  absl::Status abort(uint64_t session_id);

  // Stable lease management for CPU residency protection
  absl::StatusOr<StableLease> acquire_stable_lease(
      const loading::ReplicaKey& key,
      absl::Span<const uint32_t> chunk_indices);
  absl::Status release_stable_lease(const StableLease& lease);
  absl::Status release_stable_lease(const loading::ReplicaKey& key, absl::Span<const uint32_t> chunk_indices);
  absl::StatusOr<uint64_t> get_ledger_version(const loading::ReplicaKey& key) const;

  // Optional node-level tier budget for telemetry and admission control.
  void set_memory_tier_budget(std::shared_ptr<MemoryTierBudget> budget) {
    memory_tier_budget_ = std::move(budget);
  }

  // Export ledger and lease management (UMA‑owned export lifecycle)
  struct ExportRegistration {
    // Coalesced chunk ranges [start_idx, end_idx] (inclusive)
    std::vector<std::pair<uint32_t, uint32_t>> chunk_ranges;
    // Opaque keepalive for UMA-managed CPU pin leases (nullptr for GPU)
    std::shared_ptr<void> keepalive;
  };

  // Set exported flag for the given chunks at a location and return coalesced
  // ranges along with an optional keepalive for CPU pin leases when turning on.
  absl::StatusOr<ExportRegistration> set_exported(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      bool on);

  /**
   * @brief Mark a ratio of CPU chunks as PREEMPTIBLE via UMA-owned arena.
   *
   * The number of chunks selected is `ratio * total_chunks`, rounded down.
   * Selection order is from the beginning of the replica (lowest indices first)
   * which matches the existing test expectations.
   */
  absl::Status mark_cpu_chunks_preemptible(const loading::ReplicaKey& key, float ratio);

  /**
   * @brief Get total artifact size.
   *
   * @param key Replica instance key
   * @return Total bytes or error if not allocated
   */
  absl::StatusOr<size_t> get_artifact_size(const loading::ReplicaKey& key) const;

  struct CpuMemfdRegion {
    int fd{-1};
    uint64_t size_bytes{0};
    uint64_t offset_bytes{0};
  };

  absl::StatusOr<CpuMemfdRegion> get_cpu_memfd_region(const loading::ReplicaKey& key) const;

  /**
   * @brief Get DRAM base pointer.
   *
   * @param key Replica instance key
   * @return Base pointer or nullptr if not allocated
   */
  void* get_cpu_base_ptr(const loading::ReplicaKey& key) const;

  /**
   * @brief Get GPU base pointer for device.
   *
   * @param key Replica instance key
   * @param device_id GPU device ID
   * @return Base pointer or nullptr if not allocated
   */
  void* get_gpu_base_ptr(const loading::ReplicaKey& key, int device_id) const;

  /**
   * @brief Check if GPU loading is complete for a device.
   *
   * Returns true if all chunks are in HOT or COPIED_GPU state on the
   * specified GPU device.
   *
   * @param key Replica instance key
   * @param device_id GPU device ID
   * @return True if all chunks are loaded on GPU
   */
  bool is_gpu_loading_complete(const loading::ReplicaKey& key, int device_id) const;

  /**
   * @brief Record GPU access for a set of chunks.
   *
   * Updates the last_access_ns timestamp for the provided chunks so that LRU
   * selection algorithms (e.g., mark_cpu_chunks_preemptible) account for GPU
   * reads/writes as well as CPU touches.
   */
  void record_gpu_touch(const loading::ReplicaKey& key, int device_id, absl::Span<const uint32_t> chunks);

  /**
   * @brief Release all allocations for a replica.
   *
   * @param key Replica instance key
   * @return Status of release
   */
  absl::Status release(const loading::ReplicaKey& key);

  /**
   * @brief Release GPU residency and optional allocation for a specific device.
   *
   * Clears UMA ledger residency for all chunks on the given GPU device and,
   * when drop_allocation is true, removes the UMA-owned VRAM allocation handle
   * so that VRAM is actually returned when no other owners exist.
   *
   * This method acquires only UMA's internal mutex and does not invoke VS or
   * external registries, preserving the UMA→VS→Export lock order.
   */
  absl::Status release_gpu_device(const loading::ReplicaKey& key, int device_id, bool drop_allocation = true);

  // UMA V3: Direct write grant returning windowed authorization directly.
  // Exposes DirectWriteGrant with Window entries and keepalive semantics.
  absl::StatusOr<DirectWriteGrant> grant_direct_write(const loading::ReplicaKey& key, absl::Span<const VaRange> ranges);

  enum class PostGpuLoadPolicy : std::uint8_t { EvictCPU, MarkPreemptible, Keep };

  /**
   * @brief Apply post-GPU-load policy affecting CPU residency.
   */
  absl::Status post_gpu_load_policy(const loading::ReplicaKey& key, size_t bytes, PostGpuLoadPolicy policy);

  /**
   * @brief Direct CPU write helper used by sinks and GPU→CPU copies.
   */
  absl::Status write_cpu_span(const loading::ReplicaKey& key, uint64_t va_offset, const void* src, size_t bytes);

  // UMA-side telemetry for CPU writes (invoked after memcpy/mmap)
  void record_cpu_write(const loading::ReplicaKey& key, uint64_t va_offset, uint64_t bytes);

 private:
  // Internal helpers that assume mutex_ is already held. These avoid re-entrant
  // locking bugs when called from methods that already hold mutex_.
  std::vector<uint32_t> get_missing_chunks_locked_(
      const loading::ReplicaKey& key,
      common::memory::MemoryLocation target,
      std::optional<int> device_id) const;

  struct SessionRecord {
    loading::ReplicaKey key;
    common::memory::MemoryLocation target;
    std::optional<int> device_id;
    std::vector<uint32_t> chunks;
    // RAII: UMA-managed pin keepalives for CPU ranges held during session
    std::vector<std::shared_ptr<void>> cpu_keepalives;
  };

  static std::vector<std::pair<uint32_t, uint32_t>> coalesce_runs_(absl::Span<const uint32_t> sorted);

  struct ReplicaAllocation;

  absl::Status mark_chunks_preemptible_locked_(ReplicaAllocation& alloc, absl::Span<const uint32_t> indices);
  bool is_preemptible_resident_locked_(const ReplicaAllocation& alloc, uint32_t chunk_idx) const;
  void record_preemptible_fault_locked_(const loading::ReplicaKey& key, uint32_t chunk_idx) const;
  void maybe_record_rehydrate_latency_locked_(const loading::ReplicaKey& key, uint32_t chunk_idx);
  void clear_rehydrate_records_for_key_(const loading::ReplicaKey& key);
  static uint64_t chunk_bytes_for_index_(const ReplicaAllocation& alloc, uint64_t chunk_size_bytes, uint32_t idx);
  static std::vector<uint32_t> normalize_chunk_indices_(absl::Span<const uint32_t> indices, size_t num_chunks);
  static bool is_cpu_resident_state_(ChunkState state);
  uint64_t compute_preemptible_bytes_locked_() const;
  void update_preemptible_budget_locked_() const;

  struct ReplicaAllocation {
    struct CpuRegion {
      void* base{nullptr};
      size_t bytes{0};
      int memfd{-1};
      uint64_t offset_bytes{0};
    };

    CpuRegion cpu_region;
    std::vector<uint32_t> mlock_refcnt; // tracks successful mlock calls per chunk

    // GPU allocations per device (lazy creation)
    std::unordered_map<DeviceKey, std::shared_ptr<common::memory::GpuDeviceMemory>, DeviceKeyHash> gpu_allocations;

    // Phase 1: Internal orthogonal chunk record (not exported). Mirrors UMA ledger
    // without relying on VS. Over time, APIs will migrate to this structure.
    struct ChunkRecord {
      uint32_t chunk_idx{0};
      // CPU residency (ChunkState used as enum carrier; semantics under UMA)
      ChunkState cpu{ChunkState::COLD};
      // GPU residency per device
      std::unordered_map<DeviceKey, ChunkState, DeviceKeyHash> gpu;
      // Export flags
      bool exported_cpu{false};
      std::unordered_map<DeviceKey, bool, DeviceKeyHash> exported_gpu;
      // UMA-managed CPU pin refcount for this chunk
      uint32_t pin_refcnt{0};
      // Number of active stable leases guarding this chunk
      uint32_t stable_lease_count{0};
      // Last access timestamp (ns since steady_clock epoch)
      uint64_t last_access_ns{0};
      // Monotonic version; incremented on state updates
      uint64_t version{0};
    };

    std::vector<ChunkRecord> chunk_records;

    // Total artifact size
    size_t total_bytes{0};

    // Number of chunks
    size_t num_chunks{0};
    // Per-device counter of chunks that are fully resident (HOT | COPIED_GPU).
    std::unordered_map<DeviceKey, size_t, DeviceKeyHash> loaded_chunk_counts;
    // Monotonic ledger version (incremented on state transitions)
    uint64_t ledger_version{0};
  };

  void record_cpu_write_locked_(ReplicaAllocation& alloc, uint64_t va_offset, uint64_t bytes);

  class CpuArena {
   public:
    struct Options {
      bool cpu_shared_memory_enabled{true};
    };

    explicit CpuArena(size_t chunk_bytes, Options options);

    absl::Status allocate_region(ReplicaAllocation& alloc, size_t bytes) const;
    void release_region(ReplicaAllocation& alloc) const;
    absl::Status write_span(ReplicaAllocation& alloc, uint64_t va_offset, const void* src, size_t bytes) const;
    absl::Status mark_preemptible(ReplicaAllocation& alloc, absl::Span<const uint32_t> idx) const;
    size_t evict_tail_bytes(ReplicaAllocation& alloc, size_t bytes) const;
    absl::StatusOr<std::shared_ptr<void>> pin_chunks(
        ReplicaAllocation& alloc,
        absl::Span<const uint32_t> chunk_indices,
        std::string_view reason,
        std::optional<std::chrono::milliseconds> timeout) const;

    [[nodiscard]] size_t chunk_bytes() const {
      return chunk_bytes_;
    }

   private:
    struct PinHandle {
      const CpuArena* arena{nullptr};
      ReplicaAllocation* alloc{nullptr};
      std::vector<uint32_t> chunks;
      ~PinHandle();
    };

    absl::Status ensure_bounds(const ReplicaAllocation& alloc, uint64_t va_offset, size_t bytes) const;
    void release_pins(ReplicaAllocation& alloc, absl::Span<const uint32_t> chunk_indices) const;
    Options options_;
    size_t chunk_bytes_{0};
  };

  mutable std::mutex mutex_;
  size_t chunk_size_bytes_;
  CpuArena cpu_arena_;
  std::unordered_map<loading::ReplicaKey, ReplicaAllocation, loading::ReplicaKeyHash> allocations_;
  std::shared_ptr<MemoryTierBudget> memory_tier_budget_;

  // Sessions for plan/commit/abort
  std::unordered_map<uint64_t, SessionRecord> sessions_;
  uint64_t next_session_id_{1};

  struct PendingRehydrateKey {
    loading::ReplicaKey key;
    uint32_t chunk_idx{0};
    bool operator==(const PendingRehydrateKey&) const = default;
  };

  struct PendingRehydrateKeyHash {
    size_t operator()(const PendingRehydrateKey& k) const noexcept {
      const size_t base = loading::ReplicaKeyHash{}(k.key);
      return absl::HashOf(base, k.chunk_idx);
    }
  };

  mutable std::mutex telemetry_mu_;
  mutable std::unordered_map<PendingRehydrateKey, std::chrono::steady_clock::time_point, PendingRehydrateKeyHash>
      pending_rehydrate_;
};

} // namespace tensorcast::store::replica
