// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/communication_types.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/memory_manager.h"
#include "core/store/replica/replica_memory_coordinator.h"
// Prefer explicit includes over forward declarations
#include "core/store/components/global_store_client.h"

namespace tensorcast::store::loading {

/**
 * @brief Strategy for determining optimal chunk loading plan across multiple sources.
 *
 * This class implements the intelligent loading logic that:
 * 1. Queries chunk availability across local and remote locations
 * 2. Creates optimized loading plans based on source priority
 * 3. Coordinates parallel chunk transfers from multiple sources
 * 4. Tracks loading progress with real-time updates
 */
class ChunkAwareLoadingStrategy {
 public:
  // Type of chunk source
  enum class ChunkSource : std::uint8_t {
    LOCAL_CPU, // Copy from local DRAM
    LOCAL_GPU, // Copy from local GPU (different device)
    REMOTE_P2P, // Transfer from remote peer
    DISK // Load from local disk
  };

  // A single loading operation for a batch of chunks
  struct LoadOperation {
    ChunkSource source;
    std::vector<uint32_t> chunks;
    common::memory::MemoryLocation target;

    // For REMOTE_P2P operations
    std::optional<P2PSource> p2p_source;

    // For LOCAL_GPU operations
    std::optional<int> source_device_id;
  };

  // Complete loading plan with all operations
  struct LoadPlan {
    std::vector<LoadOperation> operations;
    common::memory::MemoryLocation target;
    size_t total_chunks;
    size_t total_bytes;

    // Maps chunk_idx -> remote source for P2P operations
    std::map<uint32_t, P2PSource> remote_sources;
  };

  // Progress tracking
  struct LoadProgress {
    size_t completed_chunks = 0;
    size_t total_chunks = 0;
    size_t completed_bytes = 0;
    size_t total_bytes = 0;
    absl::Status last_error;
  };

  using ProgressCallback = std::function<void(size_t completed, size_t total)>;

  /**
   * @brief Constructor
   */
  ChunkAwareLoadingStrategy() = default;

  /**
   * @brief Generate optimal loading plan for missing chunks
   *
   * @param key Replica instance key
   * @param target Target location (GPU or PAGEABLE_CPU)
   * @param memory Unified memory to query current chunk states
   * @param global_store Client to query remote chunk locations
   * @return Optimized loading plan
   */
  static LoadPlan create_loading_plan(
      const ReplicaKey& key,
      common::memory::MemoryLocation target,
      const replica::ReplicaMemoryCoordinator& memory,
      components::GlobalStoreClient& global_store);

  /**
   * @brief Execute loading plan asynchronously
   *
   * @param plan Loading plan to execute
   * @param memory Unified memory to update chunk states
   * @param mem_manager Memory manager for allocations and loaders
   * @return Future that resolves when loading completes
   */
  static std::future<absl::Status> execute_plan(
      const LoadPlan& plan,
      replica::ReplicaMemoryCoordinator& memory,
      const std::shared_ptr<store::replica::MemoryManager>& mem_manager);

  /**
   * @brief Execute loading plan with progress tracking
   *
   * @param plan Loading plan to execute
   * @param memory Unified memory to update chunk states
   * @param mem_manager Memory manager for allocations and loaders
   * @param progress_cb Callback for progress updates
   * @return Status of the execution
   */
  static absl::Status execute_plan_with_progress(
      const LoadPlan& plan,
      replica::ReplicaMemoryCoordinator& memory,
      const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
      const ProgressCallback& progress_cb);

 private:
  // Helper to select best remote source based on load and network topology
  static P2PSource select_best_remote(const std::vector<P2PSource>& candidates);

  // Helper to optimize chunk order for sequential disk access
  static std::vector<uint32_t> optimize_chunk_order(const std::vector<uint32_t>& chunks);

  // Execute a single load operation
  static absl::Status execute_operation(
      const LoadOperation& op,
      replica::ReplicaMemoryCoordinator& memory,
      const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
      const ProgressCallback& progress_cb);

  // Execute local CPU->GPU copy
  static absl::Status execute_local_cpu_copy(
      const std::vector<uint32_t>& chunks,
      common::memory::MemoryLocation target,
      replica::ReplicaMemoryCoordinator& memory,
      const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
      const ProgressCallback& progress_cb);

  // Execute P2P transfer
  static absl::Status execute_p2p_transfer(
      const LoadOperation& op,
      replica::ReplicaMemoryCoordinator& memory,
      const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
      const ProgressCallback& progress_cb);

  // Execute disk load
  static absl::Status execute_disk_load(
      const std::vector<uint32_t>& chunks,
      common::memory::MemoryLocation target,
      replica::ReplicaMemoryCoordinator& memory,
      const std::shared_ptr<store::replica::MemoryManager>& mem_manager,
      const ProgressCallback& progress_cb);
};

} // namespace tensorcast::store::loading
