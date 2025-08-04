// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "core/common/memory/cuda_memory.h"
#include "core/common/memory/distributed_memory_pool.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/chunk_meta.h"
#include "core/store/model/model_location.h"

namespace stepcast::store {

/**
 * @brief Unified memory management for models across VRAM and DRAM.
 *
 * This class manages chunk-based memory state for models, tracking:
 * - One contiguous DRAM allocation per model (via DVMP)
 * - One contiguous VRAM allocation per GPU device
 * - Per-chunk state tracking for fine-grained memory management
 *
 * The design enables efficient P2P transfers, intelligent memory eviction,
 * and automatic memory reclamation while maintaining zero-copy access.
 */
class UnifiedModelMemory {
 public:
  /**
   * @brief Chunk mapping tracking state across memory locations.
   */
  struct ChunkMapping {
    uint32_t chunk_idx;
    ChunkState cpu_state; // State in DRAM (from DVMP)
    std::unordered_map<DeviceKey, ChunkState, DeviceKeyHash> gpu_state; // Per-device VRAM state
    uint64_t last_access_ns{0};
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

  explicit UnifiedModelMemory(std::shared_ptr<stepcast::memory::DistributedMemoryPool> dvmp);
  ~UnifiedModelMemory() = default;

  // Disable copy/move
  UnifiedModelMemory(const UnifiedModelMemory&) = delete;
  UnifiedModelMemory& operator=(const UnifiedModelMemory&) = delete;

  /**
   * @brief Allocate unified memory for a model.
   *
   * Reserves virtual address space in DRAM via DVMP. GPU allocations
   * are created lazily on first use.
   *
   * @param key Model instance key
   * @param bytes Total model size
   * @return Status of allocation
   */
  absl::Status allocate(const InstanceKey& key, size_t bytes);

  /**
   * @brief Get or create GPU allocation for a device.
   *
   * Lazily creates VRAM allocation on first access to minimize
   * upfront memory cost on unused GPUs.
   *
   * @param key Model instance key
   * @param device_id GPU device ID
   * @return CudaMemory pointer or error
   */
  absl::StatusOr<std::shared_ptr<CudaMemory>> get_or_create_gpu_allocation(const InstanceKey& key, int device_id);

  /**
   * @brief Get chunk mappings for scheduling decisions.
   *
   * @param key Model instance key
   * @return View of chunk mappings
   */
  absl::Span<const ChunkMapping> get_chunk_mappings(const InstanceKey& key) const;

  /**
   * @brief Get missing chunks for a target location.
   *
   * @param key Model instance key
   * @param target Target memory location
   * @param device_id Device ID for GPU targets
   * @return List of missing chunk indices
   */
  std::vector<uint32_t> get_missing_chunks(
      const InstanceKey& key,
      ModelLocation target,
      std::optional<int> device_id = std::nullopt) const;

  /**
   * @brief Lock chunks for transfer operation.
   *
   * Atomically transitions chunks to LOCKED_TX state. Must be
   * followed by update_chunk_states() after transfer.
   *
   * @param key Model instance key
   * @param source Source memory location
   * @param target Target memory location
   * @param chunks Chunk indices to lock
   * @return Status of lock operation
   */
  absl::Status lock_chunks_for_transfer(
      const InstanceKey& key,
      ModelLocation source,
      ModelLocation target,
      const std::vector<uint32_t>& chunks);

  /**
   * @brief Update chunk states after transfer.
   *
   * @param key Model instance key
   * @param location Memory location that was updated
   * @param chunks Chunk indices that were transferred
   * @param new_state New state for chunks
   * @param device_id Device ID for GPU operations
   * @return Status of update
   */
  absl::Status update_chunk_states(
      const InstanceKey& key,
      ModelLocation location,
      const std::vector<uint32_t>& chunks,
      ChunkState new_state,
      std::optional<int> device_id = std::nullopt);

  /**
   * @brief Get best source for loading a chunk.
   *
   * Prioritizes based on:
   * 1. Local memory (CPU/GPU)
   * 2. Remote P2P transfer
   * 3. Disk loading
   *
   * @param key Model instance key
   * @param chunk_idx Chunk index
   * @param target Target location
   * @return Best available source
   */
  ChunkSource get_best_source_for_chunk(const InstanceKey& key, uint32_t chunk_idx, ModelLocation target) const;

  /**
   * @brief Get memory statistics for a model.
   *
   * @param key Model instance key
   * @return Map of location to allocated bytes
   */
  std::unordered_map<ModelLocation, size_t> get_memory_stats(const InstanceKey& key) const;

  /**
   * @brief Check if model has any allocations.
   *
   * @param key Model instance key
   * @return True if model has DRAM or VRAM allocations
   */
  bool has_allocation(const InstanceKey& key) const;

  /**
   * @brief Convenience: return the global chunk size used by DVMP.
   */
  size_t get_chunk_size() const;

  /**
   * @brief Mark a ratio of CPU chunks as PREEMPTIBLE.
   *
   * Relies on DVMP's mark_preemptible implementation. The number of
   * chunks selected is `ratio * total_chunks`, rounded down. Selection
   * order is from the beginning of the model (lowest indices first)
   * which matches the existing test expectations.
   *
   * @param key   Model instance key.
   * @param ratio Fraction \([0,1]\] of chunks to mark.
   */
  absl::Status mark_cpu_chunks_preemptible(const InstanceKey& key, float ratio);

  /**
   * @brief Get total model size.
   *
   * @param key Model instance key
   * @return Total bytes or error if not allocated
   */
  absl::StatusOr<size_t> get_model_size(const InstanceKey& key) const;

  /**
   * @brief Get DRAM base pointer.
   *
   * @param key Model instance key
   * @return Base pointer or nullptr if not allocated
   */
  void* get_cpu_base_ptr(const InstanceKey& key) const;

  /**
   * @brief Get GPU base pointer for device.
   *
   * @param key Model instance key
   * @param device_id GPU device ID
   * @return Base pointer or nullptr if not allocated
   */
  void* get_gpu_base_ptr(const InstanceKey& key, int device_id) const;

  /**
   * @brief Check if GPU loading is complete for a device.
   *
   * Returns true if all chunks are in HOT or COPIED_GPU state on the
   * specified GPU device.
   *
   * @param key Model instance key
   * @param device_id GPU device ID
   * @return True if all chunks are loaded on GPU
   */
  bool is_gpu_loading_complete(const InstanceKey& key, int device_id) const;

  /**
   * @brief Record GPU access for a set of chunks.
   *
   * Updates the last_access_ns timestamp for the provided chunks so that LRU
   * selection algorithms (e.g., mark_cpu_chunks_preemptible) account for GPU
   * reads/writes as well as CPU touches.
   */
  void record_gpu_touch(const InstanceKey& key, int device_id, absl::Span<const uint32_t> chunks);

  /**
   * @brief Release all allocations for a model.
   *
   * @param key Model instance key
   * @return Status of release
   */
  absl::Status release(const InstanceKey& key);

 private:
  struct ModelAllocation {
    // DRAM allocation info from DVMP
    stepcast::memory::DistributedMemoryPool::VirtualRegion dram_region;

    // GPU allocations per device (lazy creation)
    std::unordered_map<DeviceKey, std::shared_ptr<CudaMemory>, DeviceKeyHash> gpu_allocations;

    // Chunk mappings (shared across all devices)
    std::vector<ChunkMapping> chunk_mappings;

    // Total model size
    size_t total_bytes{0};

    // Number of chunks
    size_t num_chunks{0};
    // Per-device counter of chunks that are fully resident (HOT | COPIED_GPU).
    std::unordered_map<DeviceKey, size_t, DeviceKeyHash> loaded_chunk_counts;
  };

  mutable std::mutex mutex_;
  std::shared_ptr<stepcast::memory::DistributedMemoryPool> dvmp_;
  std::unordered_map<InstanceKey, ModelAllocation, InstanceKeyHash> allocations_;

  // Helper to sync chunk states with DVMP
  void sync_cpu_chunk_states(const InstanceKey& key);
};

} // namespace stepcast::store