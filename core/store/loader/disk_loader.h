// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <filesystem>
#include <future>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

#include "core/store/loader/loader.h"
#include "core/store/loading/loading_spec.h" // For DiskSource
// Forward declaration instead of include to avoid circular dependency

namespace stepcast::store {

// Forward declaration
class MemoryManager;

/**
 * @brief Loader implementation for reading model data partitions from disk.
 */
class DiskLoader : public IModelLoader {
 public:
  /**
   * @brief Constructs a DiskLoader.
   * @param source Configuration detailing the disk storage path.
   */
  explicit DiskLoader(DiskSource source);

  ~DiskLoader() override = default;

  // Disable copy and move
  DiskLoader(const DiskLoader&) = delete;
  DiskLoader& operator=(const DiskLoader&) = delete;
  DiskLoader(DiskLoader&&) = delete;
  DiskLoader& operator=(DiskLoader&&) = delete;

  /**
   * @brief Scans the configured directory to find partitions and calculate total model size.
   * Must be called before get_model_size() or load_async() if not implicitly called.
   * Thread-safe.
   * @return absl::Status OkStatus on success, error otherwise (e.g., path not found).
   */
  absl::Status initialize() override ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Returns the total model size discovered during initialization.
   * Calls initialize() if not already done. Thread-safe.
   * @return absl::StatusOr<uint64_t> Total model size in bytes or error status.
   */
  absl::StatusOr<uint64_t> get_model_size() override ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Asynchronously loads data from disk partitions into CPU pinned memory
   *        managed by the provided MemoryManager.
   *
   * Ensures target CPU memory in MemoryManager is allocated (using the size
   * discovered by initialize()). Uses multiple threads for reading partitions.
   * Transitions MemoryManager CPU state: ALLOCATED -> LOADING -> LOADED/FAILED.
   * Thread-safe with respect to internal state and MemoryManager interaction.
   *
   * @param mem_manager Shared pointer to the MemoryManager handling the target CPU memory.
   * @param target_location Must be ModelLocation::PAGEABLE_CPU.
   * @param concurrency Number of threads to use for parallel disk reads.
   * @return std::future<absl::Status> A future indicating the completion status of the load operation.
   */
  std::future<absl::Status> load_async(
      std::shared_ptr<MemoryManager> mem_manager,
      ModelLocation target_location,
      int concurrency) override ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Asynchronously loads specific chunks from disk into the target location.
   *
   * This method enables chunk-aware loading, allowing the loader to skip chunks
   * that are already available in other memory tiers. It integrates with
   * UnifiedModelMemory to provide fine-grained loading control.
   *
   * @param mem_manager Shared pointer to the MemoryManager handling the target memory.
   * @param target_location The destination for the data (PAGEABLE_CPU or GPU).
   * @param chunk_indices Vector of chunk indices to load (0-based).
   * @param concurrency Number of threads to use for parallel disk reads.
   * @return std::future<absl::Status> A future indicating the completion status.
   */
  std::future<absl::Status> load_chunks_async(
      std::shared_ptr<MemoryManager> mem_manager,
      ModelLocation target_location,
      const std::vector<uint32_t>& chunk_indices,
      int concurrency) ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_; // Protects initialization state and partition info access

  DiskSource source_ ABSL_GUARDED_BY(mutex_);
  uint64_t model_size_ ABSL_GUARDED_BY(mutex_) = 0;
  std::vector<std::filesystem::path> partition_paths_ ABSL_GUARDED_BY(mutex_);
  std::vector<size_t> partition_sizes_ ABSL_GUARDED_BY(mutex_);
  bool initialized_ ABSL_GUARDED_BY(mutex_) = false;
};

} // namespace stepcast::store