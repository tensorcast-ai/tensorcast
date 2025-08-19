// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <future>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "gsl/pointers"

#include "core/common/model_verification.h"
#include "core/communicator/engine/engine.h"
#include "core/store/communication_types.h"
#include "core/store/loader/loader.h"
#include "core/store/model/memory_manager.h"
#include "core/store/model/memory_state.h"
#include "core/store/model/model_config.h"
#include "core/store/model/model_location.h"

namespace stepcast::store {

/**
 * @brief Facade class for managing a machine learning model's lifecycle,
 *        including loading from different sources and memory management.
 *
 * This class orchestrates the interactions between Loaders (Disk, RDMA)
 * and the MemoryManager to provide a simplified interface for accessing model data
 * on CPU or GPU.
 */
class Model {
 public:
  /**
   * @brief Factory function to create and initialize a Model instance.
   * This handles selecting the appropriate loader based on the config.
   * @param config Configuration specifying the model source, target device, pools, etc.
   * @return absl::StatusOr<std::unique_ptr<Model>> The created Model instance or an error status.
   */
  static absl::StatusOr<std::unique_ptr<Model>> create(ModelConfig config);

  ~Model();

  // Disable copy and move
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&&) = delete;
  Model& operator=(Model&&) = delete;

  /**
   * @brief Gets the unique identifier for this model.
   */
  const std::string& model_id() const;

  /**
   * @brief Returns the InstanceKey uniquely identifying this model instance.
   */
  const InstanceKey& instance_key() const {
    return key_;
  }

  /**
   * @brief Returns the DeviceKey where this model instance is bound.
   */
  const DeviceKey& device() const {
    return key_.device;
  }

  /**
   * @brief Copies model data from another Model instance (GPU↔GPU or CPU↔GPU).
   */
  absl::Status copy_from(const Model& src);

  /**
   * @brief Gets the total size of the model data in bytes.
   */
  absl::StatusOr<uint64_t> get_model_size() const;

  /**
   * @brief Asynchronously ensures the model data is loaded to the specified target location.
   *
   * If the data is not already present at the target location and in the LOADED state,
   * this function will:
   * 1. Allocate memory at the target location (if not already allocated).
   * 2. Initiate loading from the source (Disk/RDMA) via the appropriate Loader, OR
   * 3. Initiate a copy from another location (e.g., CPU -> GPU) if data is already loaded elsewhere.
   *
   * @param target_location The desired location (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   * @param concurrency Hint for loader concurrency (e.g., disk read threads). Defaults to 4.
   * @param device_id Optional device ID for GPU operations.
   * @return std::shared_future<absl::Status> A future indicating the completion status of the load/copy operation.
   */
  std::shared_future<absl::Status> ensure_loaded_async(
      ModelLocation target_location,
      int concurrency = 4,
      std::optional<int> device_id = std::nullopt) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Releases the memory associated with the specified location.
   * @param location The location to release (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   * @param safe_release If true, fails if the memory is currently being loaded into.
   * @return absl::Status OkStatus on success, error otherwise.
   */
  absl::Status release_memory(ModelLocation location, bool safe_release = false) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets the current state of the memory for the specified location.
   * @param location ModelLocation::PAGEABLE_CPU or ModelLocation::GPU.
   * @return MemoryState The current state of the memory at the specified location.
   */
  MemoryState get_memory_state(ModelLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets a pointer to the model data at the specified location.
   * Returns nullptr if the data is not in the LOADED state at that location.
   * Note: For CPU, direct access to chunked memory might require `get_memory_manager()`.
   * @param location ModelLocation::PAGEABLE_CPU or ModelLocation::GPU.
   * @return std::vector<void*> Vector of pointers to the data, or empty vector if not loaded.
   */
  std::vector<void*> get_data_pointer(ModelLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Waits until the model data is fully loaded at the specified location or an error occurs.
   * @param location The location to wait for (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   * @param timeout Optional maximum duration to wait.
   * @return absl::Status OkStatus if loaded, DeadlineExceeded if timeout, FailedPrecondition if loading failed.
   */
  absl::Status wait_until_loaded(ModelLocation location, absl::Duration timeout = absl::InfiniteDuration())
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Provides access to the underlying MemoryManager. Use with caution.
   * Allows advanced operations like accessing CPU chunks directly if needed.
   */
  MemoryManager& get_memory_manager() const;

  /**
   * @brief Registers the loaded memory (CPU or GPU) for communication access via the communicator engine.
   * Requires the model to be loaded at the specified location.
   * @param location The memory location to register (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   * @param comm_engine The communicator engine to use for communication registration.
   * @return absl::StatusOr<CommRegistrationInfo> Information needed by remote peers to access the memory, or an error.
   */
  absl::StatusOr<CommRegistrationInfo> enable_remote_memory_access(
      ModelLocation location,
      stepcast::communicator::CommunicateEngine& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Disables the communication access for the specified location.
   * @param location The memory location to disable communication access for (ModelLocation::PAGEABLE_CPU or
   * ModelLocation::GPU).
   * @param comm_engine The communicator engine to use for communication disconnection.
   * @return absl::Status OkStatus on success, error if location not loaded or communication disconnection fails.
   */
  absl::Status disable_remote_memory_access(
      ModelLocation location,
      stepcast::communicator::CommunicateEngine& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Generates verification information for the model data at the specified location.
   * The model must be loaded at the specified location before calling this method.
   * @param location The memory location to generate verification info for (ModelLocation::PAGEABLE_CPU or
   * ModelLocation::GPU).
   * @return absl::StatusOr<ModelVerificationInfo> Generated verification information or an error.
   */
  absl::StatusOr<ModelVerificationInfo> generate_verification_info(ModelLocation location) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Verifies the model data at the specified location against expected verification information.
   * @param location The memory location to verify (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   * @param expected_info The expected verification information to compare against.
   * @param level The verification level to use (default: SEGMENT_HASHES for balanced speed/accuracy).
   * @return absl::Status OkStatus if verification passes, error if verification fails or location not loaded.
   */
  absl::Status verify_model_data(
      ModelLocation location,
      const ModelVerificationInfo& expected_info,
      VerificationLevel level = VerificationLevel::SEGMENT_HASHES) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Fast key-point verification of model data (first, middle, last positions).
   * Provides near-zero overhead integrity checking for critical scenarios.
   * @param location The memory location to verify (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   * @param expected_info The expected verification information containing key values.
   * @return absl::Status OkStatus if key points match, error if verification fails.
   */
  absl::Status verify_key_points(ModelLocation location, const ModelVerificationInfo& expected_info) const
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  // Immutable identifier for multi-device binding.
  const InstanceKey key_{};

  // Private constructor, use Model::create()
  Model(
      InstanceKey key,
      std::unique_ptr<IModelLoader> loader,
      std::shared_ptr<MemoryManager> memory_manager,
      ModelLocation source_type);

  // Helper to determine the optimal source location for loading `target_location`
  absl::StatusOr<ModelLocation> find_best_source_for_target(ModelLocation target_location) const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_; // Protects internal state consistency, loader/manager access

  const gsl::not_null<std::unique_ptr<IModelLoader>> loader_;
  const gsl::not_null<std::shared_ptr<MemoryManager>> memory_manager_;

  // Store the original source type for reference (e.g., to know if RDMA registration makes sense)
  const ModelLocation original_source_type_;

  // Futures tracking ongoing load/copy operations to prevent duplicate requests
  std::shared_future<absl::Status> cpu_load_future_ ABSL_GUARDED_BY(mutex_);
  std::shared_future<absl::Status> gpu_load_future_ ABSL_GUARDED_BY(mutex_);
};

} // namespace stepcast::store