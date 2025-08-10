// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <future>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

#include "core/store/communication_types.h"
#include "core/store/loader/loader.h"
#include "core/store/model/model_location.h"

// Forward declaration instead of include to avoid circular dependency

// Forward declaration
namespace stepcast::communicator {
class CommunicateEngine;
} // namespace stepcast::communicator

namespace stepcast::store {

// Forward declaration
class MemoryManager;

/**
 * @brief Loader implementation for fetching model data from a remote peer via RDMA or TCP.
 */
class P2PLoader : public IModelLoader {
 public:
  /**
   * @brief Constructs a P2PLoader.
   * @param source Configuration detailing the remote peer source.
   * @param comm_engine A shared communicator engine instance for performing P2P operations.
   */
  explicit P2PLoader(P2PSource source);

  ~P2PLoader() override = default;

  // Disable copy and move
  P2PLoader(const P2PLoader&) = delete;
  P2PLoader& operator=(const P2PLoader&) = delete;
  P2PLoader(P2PLoader&&) = delete;
  P2PLoader& operator=(P2PLoader&&) = delete;

  /**
   * @brief Initializes the loader (currently checks communicator).
   * @return absl::Status OkStatus if communicator is valid.
   */
  absl::Status initialize() override ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets the expected model size from the configuration.
   * Requires the loader to be initialized first.
   * @return absl::StatusOr<uint64_t> Model size or error if not initialized or not configured.
   */
  absl::StatusOr<uint64_t> get_model_size() override ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Asynchronously loads data from a remote source into local memory.
   *
   * Supports the following use cases:
   * 1. Remote GPU → Local GPU (original use case)
   * 2. Remote GPU/CPU → Local PAGEABLE_CPU (new DVMP use case)
   *
   * For PAGEABLE_CPU targets, this loader implements chunk-aware transfer
   * that integrates with the DistributedVirtualMemoryPool (DVMP) for zero-copy
   * and efficient memory management.
   *
   * @param mem_manager       Shared pointer to the `MemoryManager` handling the memory.
   * @param target_location   ModelLocation::GPU or ModelLocation::PAGEABLE_CPU.
   * @param concurrency       Concurrency hint for parallel chunk transfers (used for PAGEABLE_CPU).
   * @return                  A future that resolves to the transfer status.
   */
  std::future<absl::Status> load_async(
      std::shared_ptr<MemoryManager> mem_manager,
      ModelLocation target_location,
      int concurrency) override ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Asynchronously loads specific chunks from remote source into target location.
   *
   * This method enables chunk-aware loading from remote peers, allowing selective
   * chunk transfer based on what's already available locally. It integrates with
   * ModelMemoryCoordinator and supports both GPU and PAGEABLE_CPU targets.
   *
   * @param mem_manager Shared pointer to the MemoryManager handling the target memory.
   * @param target_location The destination for the data (PAGEABLE_CPU or GPU).
   * @param chunk_indices Vector of chunk indices to load (0-based).
   * @param concurrency Number of parallel chunk transfers to use.
   * @return std::future<absl::Status> A future indicating the completion status.
   */
  std::future<absl::Status> load_chunks_async(
      const std::shared_ptr<MemoryManager>& mem_manager,
      ModelLocation target_location,
      const std::vector<uint32_t>& chunk_indices,
      int concurrency) ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;

  P2PSource source_;
  bool initialized_ = false;
};

} // namespace stepcast::store
