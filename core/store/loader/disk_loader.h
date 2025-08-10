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

#include "core/common/model_verification.h"
#include "core/store/loader/loader.h"
#include "core/store/loading/loading_spec.h" // For DiskSource
// Forward declaration instead of including memory manager to avoid heavy deps

namespace stepcast::store::loader {
class SeekableSource;
}

namespace stepcast::store {

// MemoryManager is already defined in this namespace via included header.

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
   * Must be called before get_model_size() if not implicitly called.
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
   * @brief Returns the model verification information loaded from disk (if any).
   */
  absl::StatusOr<ModelVerificationInfo> get_verification_info() const;

  // NEW: Provide disk-backed source for pumping
  absl::StatusOr<std::unique_ptr<loader::SeekableSource>> open_source() override ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_; // Protects initialization state and partition info access

  DiskSource source_ ABSL_GUARDED_BY(mutex_);
  uint64_t model_size_ = 0;
  std::vector<std::filesystem::path> partition_paths_;
  std::vector<size_t> partition_sizes_;
  bool initialized_ = false;
};

} // namespace stepcast::store