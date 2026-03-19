// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "folly/futures/Future.h"
#include "gsl/pointers"

#include "core/common/artifact_verification.h"
#include "core/common/async_runtime.h"
#include "core/common/memory/memory_location.h"
#include "core/common/ready_signal.h"
#include "core/communicator/engine/engine.h"
#include "core/store/communication_types.h"
#include "core/store/materialization/dataplane/contracts/loader.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica_config.h"
#include "core/store/replica/replica_load_controller.h"

namespace tensorcast::store::replica {

/**
 * @brief Facade class for managing a machine learning replica's lifecycle,
 *        including loading from different sources and memory management.
 *
 * This class orchestrates the interactions between Loaders (Disk, RDMA)
 * and the ReplicaLoadController to provide a simplified interface for accessing replica data
 * on CPU or GPU.
 */
class Replica {
 public:
  /**
   * @brief Factory function to create and initialize a Replica instance.
   * This handles selecting the appropriate loader based on the config.
   * @param config Configuration specifying the replica source, target device, pools, etc.
   * @return absl::StatusOr<std::unique_ptr<Replica>> The created Replica instance or an error status.
   */
  static absl::StatusOr<std::unique_ptr<Replica>> create(ReplicaConfig config);

  ~Replica();

  // Disable copy and move
  Replica(const Replica&) = delete;
  Replica& operator=(const Replica&) = delete;
  Replica(Replica&&) = delete;
  Replica& operator=(Replica&&) = delete;

  /**
   * @brief Gets the unique identifier for this replica.
   */
  const std::string& artifact_id() const;

  /**
   * @brief Returns the ReplicaKey uniquely identifying this replica instance.
   */
  const loading::ReplicaKey& replica_key() const {
    return key_;
  }

  /**
   * @brief Returns the DeviceKey where this replica instance is bound.
   */
  const DeviceKey& device() const {
    return key_.device;
  }

  /**
   * @brief Copies replica data from another Replica instance (GPU↔GPU or CPU↔GPU).
   */
  absl::Status copy_from(const replica::Replica& src);

  /**
   * @brief Gets the total size of the replica data in bytes.
   */
  absl::StatusOr<uint64_t> get_artifact_size() const;

  /**
   * @brief Asynchronously ensures the replica data is loaded to the specified target location.
   *
   * If the data is not already present at the target location and in the LOADED state,
   * this function will:
   * 1. Allocate memory at the target location (if not already allocated).
   * 2. Initiate loading from the source (Disk/RDMA) via the appropriate Loader, OR
   * 3. Initiate a copy from another location (e.g., CPU -> GPU) if data is already loaded elsewhere.
   *
   * @param target_location The desired location (MemoryLocation::CPU or MemoryLocation::GPU).
   * @param concurrency Hint for loader concurrency (e.g., disk read threads). Defaults to 4.
   * @param device_id Optional device ID for GPU operations.
   * @return folly::SemiFuture<absl::Status> A future indicating the completion status of the load/copy operation.
   */
  folly::SemiFuture<absl::Status> ensure_loaded_async(
      common::memory::MemoryLocation target_location,
      int concurrency = 4,
      std::optional<int> device_id = std::nullopt) ABSL_LOCKS_EXCLUDED(mutex_);

  [[nodiscard]] std::shared_ptr<common::ReadySignal<absl::Status>> ready_signal_for(
      common::memory::MemoryLocation target_location) const ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status mark_loaded(common::memory::MemoryLocation location) ABSL_LOCKS_EXCLUDED(mutex_);
  void set_ready_signal(common::memory::MemoryLocation location, const absl::Status& status)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Releases the memory associated with the specified location.
   *        Refuses to release if the memory is currently LOADING.
   * @param location The location to release (MemoryLocation::CPU or MemoryLocation::GPU).
   * @return absl::Status OkStatus on success, FailedPrecondition if LOADING, error otherwise.
   */
  absl::Status release_memory(common::memory::MemoryLocation location) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets the current state of the memory for the specified location.
   * @param location MemoryLocation::CPU or MemoryLocation::GPU.
   * @return MemoryState The current state of the memory at the specified location.
   */
  replica::MemoryState get_memory_state(common::memory::MemoryLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets a pointer to the replica data at the specified location.
   * Returns nullptr if the data is not in the LOADED state at that location.
   * Note: For CPU, direct access to chunked memory might require `get_memory_manager()`.
   * @param location MemoryLocation::CPU or MemoryLocation::GPU.
   * @return std::vector<void*> Vector of pointers to the data, or empty vector if not loaded.
   */
  std::vector<void*> get_data_pointer(common::memory::MemoryLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Waits until the replica data is fully loaded at the specified location or an error occurs.
   * @param location The location to wait for (MemoryLocation::CPU or MemoryLocation::GPU).
   * @param timeout Optional maximum duration to wait.
   * @return absl::Status OkStatus if loaded, DeadlineExceeded if timeout, FailedPrecondition if loading failed.
   */
  absl::Status wait_until_loaded(
      common::memory::MemoryLocation location,
      absl::Duration timeout = absl::InfiniteDuration()) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Provides access to the underlying ReplicaLoadController. Use with caution.
   * Allows advanced operations like accessing CPU chunks directly if needed.
   */
  ReplicaLoadController& get_memory_manager() const;

  /**
   * @brief Registers the loaded memory (CPU or GPU) for communication access via the communicator engine.
   * Requires the replica to be loaded at the specified location.
   * @param location The memory location to register (MemoryLocation::CPU or MemoryLocation::GPU).
   * @param comm_engine The communicator engine to use for communication registration.
   * @return absl::StatusOr<ExportRegistration> Information needed by remote peers to access the memory, or an error.
   */
  absl::StatusOr<ExportRegistration> enable_remote_memory_access(
      common::memory::MemoryLocation location,
      tensorcast::communicator::engine::Communicator& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Disables the communication access for the specified location.
   * @param location The memory location to disable communication access for (MemoryLocation::CPU or
   * MemoryLocation::GPU).
   * @param comm_engine The communicator engine to use for communication disconnection.
   * @return absl::Status OkStatus on success, error if location not loaded or communication disconnection fails.
   */
  absl::Status disable_remote_memory_access(
      common::memory::MemoryLocation location,
      tensorcast::communicator::engine::Communicator& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Generates verification information for the replica data at the specified location.
   * The replica must be loaded at the specified location before calling this method.
   * @param location The memory location to generate verification info for (MemoryLocation::CPU or
   * MemoryLocation::GPU).
   * @return absl::StatusOr<ArtifactVerificationInfo> Generated verification information or an error.
   */
  absl::StatusOr<common::ArtifactVerificationInfo> generate_verification_info(
      common::memory::MemoryLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Verifies the replica data at the specified location against expected verification information.
   * @param location The memory location to verify (MemoryLocation::CPU or MemoryLocation::GPU).
   * @param expected_info The expected verification information to compare against.
   * @param level The verification level to use (default: SEGMENT_HASHES for balanced speed/accuracy).
   * @return absl::Status OkStatus if verification passes, error if verification fails or location not loaded.
   */
  absl::Status verify_artifact_data(
      common::memory::MemoryLocation location,
      const common::ArtifactVerificationInfo& expected_info,
      common::VerificationLevel level = common::VerificationLevel::SEGMENT_HASHES) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Fast key-point verification of replica data (first, middle, last positions).
   * Provides near-zero overhead integrity checking for critical scenarios.
   * @param location The memory location to verify (MemoryLocation::CPU or MemoryLocation::GPU).
   * @param expected_info The expected verification information containing key values.
   * @return absl::Status OkStatus if key points match, error if verification fails.
   */
  absl::Status verify_key_points(
      common::memory::MemoryLocation location,
      const common::ArtifactVerificationInfo& expected_info) const ABSL_LOCKS_EXCLUDED(mutex_);

  [[nodiscard]] const std::optional<loader::ViewPlan>& view_plan() const {
    return view_plan_;
  }

  [[nodiscard]] const std::optional<std::string>& canonical_index_json() const {
    return canonical_index_json_;
  }

  [[nodiscard]] const std::optional<loading::CollectiveLoadGroupHint>& collective_load_group() const {
    return collective_load_group_;
  }

 private:
  // Immutable identifier for multi-device binding.
  const loading::ReplicaKey key_{};

  // Private constructor, use Replica::create()
  Replica(
      loading::ReplicaKey key,
      std::unique_ptr<IArtifactLoader> loader,
      std::shared_ptr<ReplicaLoadController> memory_manager,
      gsl::not_null<std::shared_ptr<common::AsyncRuntime>> async_runtime,
      common::memory::MemoryLocation source_type,
      bool source_is_view,
      std::optional<loader::ViewPlan> view_plan,
      std::optional<std::string> canonical_index_json,
      std::optional<std::string> source_index_json,
      std::optional<loading::CollectiveLoadGroupHint> collective_load_group,
      std::optional<loading::VariantIdentity> variant_identity,
      loading::TransformPlacement transform_placement,
      StoreEngineOptions::ByteMappingConfig byte_mapping_config);

  // Helper to determine the optimal source location for loading `target_location`
  absl::StatusOr<common::memory::MemoryLocation> find_best_source_for_target(
      common::memory::MemoryLocation target_location) const ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  mutable absl::Mutex mutex_; // Protects internal state consistency, loader/manager access

  const gsl::not_null<std::unique_ptr<IArtifactLoader>> loader_;
  const gsl::not_null<std::shared_ptr<ReplicaLoadController>> memory_manager_;
  const gsl::not_null<std::shared_ptr<common::AsyncRuntime>> async_runtime_;

  // Store the original source type for reference (e.g., to know if RDMA registration makes sense)
  const common::memory::MemoryLocation original_source_type_;
  const bool source_is_view_;

  // Optional view execution plan when this replica represents a variant byte space.
  const std::optional<loader::ViewPlan> view_plan_;
  const std::optional<std::string> canonical_index_json_;
  const std::optional<std::string> source_index_json_;
  const std::optional<loading::CollectiveLoadGroupHint> collective_load_group_;
  const std::optional<loading::VariantIdentity> variant_identity_;
  const loading::TransformPlacement transform_placement_;
  const StoreEngineOptions::ByteMappingConfig byte_mapping_config_;

  // Producer-side completion signals for in-flight load/copy operations.
  std::shared_ptr<common::ReadySignal<absl::Status>> cpu_ready_signal_ ABSL_GUARDED_BY(mutex_);
  std::shared_ptr<common::ReadySignal<absl::Status>> gpu_ready_signal_ ABSL_GUARDED_BY(mutex_);
};

} // namespace tensorcast::store::replica
