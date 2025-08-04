// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <future>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/model/model_location.h"

namespace stepcast::store {

// Forward declaration
class MemoryManager;

/**
 * @brief Interface for loading model data from a source (Disk, P2P)
 *        into memory managed by a MemoryManager.
 */
class IModelLoader {
 public:
  virtual ~IModelLoader() = default;

  /**
   * @brief Initialize the loader.
   * @return absl::Status OkStatus if initialization succeeds, error otherwise.
   */
  virtual absl::Status initialize() = 0;

  /**
   * @brief Get the total size of the model data in bytes.
   * May require initialization or connection depending on the loader type.
   * @return absl::StatusOr<uint64_t> Model size or error.
   */
  virtual absl::StatusOr<uint64_t> get_model_size() = 0;

  /**
   * @brief Asynchronously loads data from the source into the target location
   *        managed by the MemoryManager.
   *
   * The Loader assumes the destination memory has already been prepared
   * (e.g., via allocate_memory) for the target_location
   * and will transition the MemoryManager's state for that location
   * within the MemoryManager from ALLOCATED to LOADING and then
   * to LOADED (on success) or FAILED (on error).
   *
   * @param mem_manager Reference to the MemoryManager handling the target memory.
   * @param target_location The destination for the data (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   *                        Loaders might only support specific targets (e.g., DiskLoader to CPU, P2PLoader to GPU).
   * @param concurrency Hint for the number of parallel tasks to use (e.g., threads for disk I/O).
   * @return std::future<absl::Status> A future indicating the completion status of the load operation.
   */
  virtual std::future<absl::Status> load_async(
      std::shared_ptr<MemoryManager> mem_manager,
      ModelLocation target_location,
      int concurrency) = 0;

  // Disable copy and move semantics
  IModelLoader(const IModelLoader&) = delete;
  IModelLoader& operator=(const IModelLoader&) = delete;
  IModelLoader(IModelLoader&&) = delete;
  IModelLoader& operator=(IModelLoader&&) = delete;

 protected:
  IModelLoader() = default; // Constructor accessible only to derived classes
};

} // namespace stepcast::store