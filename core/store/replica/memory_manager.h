// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "gsl/pointers"

#include "core/common/memory/cuda_memory.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/communicator/engine/engine.h"
#include "core/store/communication_types.h"
#include "core/store/direct_write.h"
#include "core/store/loader/source.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/chunk_meta.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/replica_memory_coordinator.h"

namespace stepcast::store {

/**
 * @brief Manages pageable CPU memory and GPU memory allocation, state, and transfers for a single replica instance.
 *
 * This class encapsulates the complexity of dealing with potentially chunked pageable CPU memory
 * and contiguous GPU memory, supporting both pool-based allocation and borrowing external pointers.
 * It ensures thread-safe access to memory resources and their states.
 */
class MemoryManager {
 public:
  /**
   * @brief Constructs a MemoryManager.
   * @param artifact_identifier A unique name for the replica, used for logging.
   * @param local_device_id The target local GPU device ID.
   * @param pinned_pool Shared pool for allocating pinned CPU memory.
   * @param dvmp Shared Distributed Virtual Memory Pool.
   * @param max_buffer_bytes The maximum buffer size in bytes for streaming transfers (default 1 GB).
   * @param pinned_memory_timeout Timeout for pinned memory allocation operations.
   * @param artifact_size Total artifact size in bytes. Must be non-zero.
   *
   * Note: UMA (DVMP-backed virtual address space) is allocated eagerly during
   * construction. This does not consume physical memory and simplifies later
   * code paths by avoiding conditional UMA allocation.
   */
  MemoryManager(
      std::string artifact_identifier,
      int local_device_id,
      const gsl::not_null<std::shared_ptr<PinnedMemoryPool>>& pinned_pool,
      const gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>>& dvmp,
      size_t max_buffer_bytes,
      std::chrono::milliseconds pinned_memory_timeout,
      uint64_t artifact_size);

  ~MemoryManager() noexcept;

  // Disable copy and move
  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  /**
   * @brief Gets the configured artifact size.
   * @return uint64_t Artifact size in bytes, or 0 if not set.
   */
  [[nodiscard]] uint64_t get_artifact_size() const noexcept;

  /**
   * @brief Gets the configured local CUDA device ID (or -1 if not yet set).
   */
  [[nodiscard]] int get_local_device_id() const noexcept {
    return replica_key_.device.ordinal;
  }

  /**
   * @brief Sets the CUDA device that this memory manager should use. This will lazily
   *        create an internal non-blocking stream bound to the device. This function
   *        must be called (directly or indirectly) before any GPU allocation or copy
   *        operation when the device ID was not provided during construction.
   *
   * @param device_id The local CUDA device ID that subsequent GPU operations should
   *                  run on.
   * @return absl::Status OkStatus on success, or an error if stream creation fails or
   *         if attempting to change the device id **after** GPU memory has
   *         already been allocated.  The device id can be changed
   *         freely while no GPU memory is owned by this manager.
   */
  // [[deprecated("MemoryManager is device-bound at construction time; avoid changing after creation")]]
  // absl::Status set_local_device_id(int device_id) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Copies data from a peer MemoryManager located on another device.
   *        This enables GPU↔GPU direct P2P transfers.
   */
  [[nodiscard]] absl::Status copy_from_peer(const MemoryManager& source, cudaStream_t stream = nullptr);

  /**
   * @brief Allocates memory at the specified location using the configured memory pools.
   * Transitions state from UNALLOCATED to ALLOCATED on success.
   * @param location MemoryLocation::PAGEABLE_CPU or GPU.
   * @return absl::Status OkStatus on success, or an error status.
   */
  [[nodiscard]] absl::Status allocate_memory(MemoryLocation location) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Releases memory at the specified location.
   * If the memory was allocated by this manager, it's returned to the pool or freed.
   * Transitions state to UNALLOCATED or UNINITIALIZED.
   * @param location PAGEABLE_CPU or GPU.
   * @param safe_release If true, refuses to release if state is LOADING.
   * @return absl::Status OkStatus or error (e.g., if safe_release is true and state is LOADING).
   */
  [[nodiscard]] absl::Status release_memory(MemoryLocation location, bool safe_release = false)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets the current memory state for the specified location.
   */
  [[nodiscard]] MemoryState get_state(MemoryLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets raw pointer(s) for the specified location.
   * - For GPU: returns one pointer when state is ALLOCATED, LOADING, or LOADED.
   * - For PAGEABLE_CPU: returns the DVMP base pointer when state is ALLOCATED or LOADED.
   * Returns an empty vector otherwise.
   * @param location PAGEABLE_CPU or GPU.
   * @return std::vector<void*> Vector with zero or one pointer.
   */
  [[nodiscard]] std::vector<void*> get_pointer(MemoryLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Retrieves the CUDA IPC memory handle associated with the managed GPU
   *        memory. The replica must have its GPU memory buffer allocated and the
   *        underlying CudaMemory object initialised.
   *
   * The IPC handle can be obtained once the GPU buffer is allocated (states
   * ALLOCATED, LOADING, or LOADED).
   *
   * @return absl::StatusOr<cudaIpcMemHandle_t>  The CUDA IPC handle on success
   *         or an error status if the memory is not available.
   */
  [[nodiscard]] absl::StatusOr<cudaIpcMemHandle_t> get_cuda_ipc_handle() const noexcept ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Asynchronously copies data between PAGEABLE_CPU and GPU memory managed by this instance.
   * Assumes source location is in LOADED state and destination is ALLOCATED.
   * Manages state transitions (LOADING -> LOADED/FAILED) for the destination.
   * Uses asynchronous CUDA operations on a dedicated stream.
   *
   * @param source The source memory location (must be LOADED).
   * @param destination The destination memory location (must be ALLOCATED).
   * @return std::future<absl::Status> Future indicating the completion status of the copy.
   */
  [[nodiscard]] std::future<absl::Status> copy_data_async(MemoryLocation source, MemoryLocation destination)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief NEW: Orchestrate DISK/REMOTE → CPU/GPU using a provided source.
   * Performs allocation, state transitions, buffer setup, pumping, and finalization.
   */
  [[nodiscard]] std::future<absl::Status> load_async_from_source(
      std::unique_ptr<loader::SeekableSource> source,
      MemoryLocation target_location,
      int concurrency,
      std::optional<absl::Span<const uint32_t>> chunk_indices = std::nullopt) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Waits for the memory at the specified location to reach the LOADED state.
   * @param location PAGEABLE_CPU or GPU.
   * @param timeout Optional timeout duration.
   * @return absl::Status OkStatus if LOADED, DeadlineExceeded if timeout, FailedPrecondition if FAILED state reached,
   * Cancelled if interrupted.
   */
  [[nodiscard]] absl::Status wait_for_state(
      MemoryLocation location,
      MemoryState target_state,
      absl::Duration timeout = absl::InfiniteDuration()) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Registers the loaded memory (PAGEABLE_CPU or GPU) for communication access via the communicator engine.
   * Requires the memory to be in the LOADED state at the specified location.
   * NOTE: Deprecated. Use chunk-scoped APIs `export_chunks_for_p2p`/`unexport_chunks_for_p2p`.
   * @param location The memory location to register (MemoryLocation::PAGEABLE_CPU or MemoryLocation::GPU).
   * @param comm_engine The communicator engine to use for communication registration.
   * @return absl::StatusOr<CommRegistrationInfo> Information needed by remote peers to access the memory, or an error.
   */

  // Convenience wrappers for buffer sizing
  // These are lock-free since members are immutable after construction.
  [[nodiscard]] size_t get_pool_chunk_size() const noexcept {
    return pinned_pool_->chunk_size();
  }

  /**
   * @brief Attempts to finalize the state of a location after an asynchronous load/copy.
   * Checks if the state is currently LOADING for the given location. If so,
   * transitions it to LOADED on success or FAILED on error. If the state
   * is not LOADING, it logs a warning and does nothing.
   * @param location The memory location (PAGEABLE_CPU or GPU).
   * @param final_status The status of the completed load/copy operation.
   * @return absl::Status OkStatus, or the final_status if the state transition failed.
   */
  [[nodiscard]] absl::Status finalize_load_state(MemoryLocation location, const absl::Status& final_status)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Streaming buffer helpers moved to TransferService

  // (Deprecated) DVMP region reservation moved to UMA

  // --- Internal State Management ---
  // These might be called by Loaders or copy operations.

  /**
   * @brief Sets the memory state for a location, performing checks and logging. Internal use mostly.
   * @param location PAGEABLE_CPU or GPU.
   * @param new_state The target state.
   * @return absl::Status Ok if transition is valid, error otherwise.
   */
  [[nodiscard]] absl::Status set_state(MemoryLocation location, MemoryState new_state) ABSL_LOCKS_EXCLUDED(mutex_);

  // Thin wrappers delegating to services -------------------------------------
  [[nodiscard]] absl::StatusOr<CommRegistrationInfo> export_chunks_for_p2p(
      MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      communicator::CommunicateEngine& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);
  [[nodiscard]] absl::Status unexport_chunks_for_p2p(
      MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      communicator::CommunicateEngine& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  // --- New DVMP accessors ---------------------------------------------------
  /**
   * @brief Exposes the underlying DistributedVirtualMemoryPool instance used by this
   *        MemoryManager. Loaders can use this to allocate or lock chunks.
   */
  [[nodiscard]] gsl::not_null<memory::DistributedVirtualMemoryPool*> get_dvmp() ABSL_LOCKS_EXCLUDED(mutex_);
  [[nodiscard]] gsl::not_null<const memory::DistributedVirtualMemoryPool*> get_dvmp() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Returns an immutable snapshot view of ChunkMeta for the replica.
   *        Empty span if DVMP not available or replica not allocated.
   */
  [[nodiscard]] absl::Span<const store::ChunkMeta> chunk_snapshot() const noexcept ABSL_LOCKS_EXCLUDED(mutex_);

  // Globally-unique key identifying this replica replica (artifact_id + device + replica).
  store::ReplicaKey replica_key_;

  // Accessor for callers needing the replica key (e.g. schedulers, metrics).
  [[nodiscard]] const store::ReplicaKey& replica_key() const noexcept {
    return replica_key_;
  }

  // --- NEW: Unified Memory Management ---

  /**
   * @brief Allocate per-replica memory bookkeeping in the coordinator.
   * Reserves DRAM via DVMP for this replica. GPU allocations remain lazy.
   * @return Status of allocation.
   */
  [[nodiscard]] absl::Status allocate_replica_memory() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Mark CPU chunks as preemptible after GPU loading.
   * @param ratio Fraction of chunks to mark as preemptible (0.0-1.0).
   * @return Status of operation.
   */
  [[nodiscard]] absl::Status mark_cpu_preemptible(float ratio = 1.0F) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get missing chunks for a target location.
   * @param target Target memory location.
   * @param device_id Device ID for GPU targets.
   * @return Vector of missing chunk indices.
   */
  [[nodiscard]] std::vector<uint32_t> get_missing_chunks(
      MemoryLocation target,
      std::optional<int> device_id = std::nullopt) const ABSL_LOCKS_EXCLUDED(mutex_);

  // Plan a direct-write token for destination VA ranges (PAGEABLE_CPU).
  [[nodiscard]] absl::StatusOr<DirectWriteToken> plan_direct_write(absl::Span<const VaRange> ranges)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Explicit finalize hook to update UMA states after a load into a location
  // completes (replaces UnifiedMemorySink close-time updates).
  [[nodiscard]] absl::Status finalize_load(
      MemoryLocation location,
      std::optional<absl::Span<const uint32_t>> chunk_indices = std::nullopt) const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  /**
   * @brief Logs state transitions.
   */
  void log_state_change(MemoryLocation loc, MemoryState old_state, MemoryState new_state) const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Returns base pointer for PAGEABLE_CPU or GPU using UMA/local cache.
   * Expects mutex_ to be held by the caller.
   * Returns nullptr if state is not eligible or pointer unavailable.
   */
  [[nodiscard]] void* get_base_ptr_locked(MemoryLocation location) const ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Releases GPU resources (cuda memory).
   */
  void release_gpu_resources_locked() noexcept ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Internal helper that assumes mutex_ is already held.
   * Only to be used by methods that already acquired the lock to avoid
   * re‑entrant locking.
   */
  [[nodiscard]] absl::Status set_state_locked(MemoryLocation location, MemoryState new_state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Helper to retrieve state and condition variable pointers under lock
  struct StateCond {
    MemoryState* state;
    absl::CondVar* cond;
  };

  // New helper returning StatusOr for cleaner call-sites
  [[nodiscard]] absl::StatusOr<StateCond> get_state_cond_locked(MemoryLocation location)
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  // --- Refactor helpers (split allocate_memory) ---------------------------
  [[nodiscard]] absl::Status allocate_pageable_cpu() ABSL_LOCKS_EXCLUDED(mutex_);
  [[nodiscard]] absl::Status allocate_gpu_memory() ABSL_LOCKS_EXCLUDED(mutex_);

  // --- Internal copy refactor helpers (no behavior change) ---------------
  struct CopyLaunchParams {
    std::shared_ptr<StreamingPinnedBuffer> streaming_buffer;
    std::shared_ptr<::stepcast::memory::DistributedVirtualMemoryPool> dvmp;
    void* dvmp_base = nullptr;
    std::shared_ptr<CudaMemory> cuda_mem;
    size_t total_size = 0;
    cudaStream_t stream = nullptr;
    uint32_t device_id = 0;
    std::string artifact_id;
  };

  // Capture and validate all state needed for an async copy and set LOADING
  // on destination under lock. Returns parameters for the copy task.
  absl::Status capture_copy_context_(
      MemoryLocation source,
      MemoryLocation destination,
      CopyLaunchParams* out,
      bool* need_allocate_um) ABSL_LOCKS_EXCLUDED(mutex_);

  // Finalize destination MemoryState based on copy status under lock.
  absl::Status finalize_copy_state_(MemoryLocation destination, const absl::Status& copy_status)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // ----------------------------------------------------------------------
  // Small internal helpers for error recording (behavior-preserving)------
  // ----------------------------------------------------------------------
  void set_failure_locked_(MemoryLocation location, std::string message) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  [[nodiscard]] std::string get_last_error_locked_(MemoryLocation location) const ABSL_SHARED_LOCKS_REQUIRED(mutex_);
  // Consolidated helper: atomically record failure message and transition to FAILED.
  void record_failure_and_fail_(MemoryLocation location, std::string message) ABSL_LOCKS_EXCLUDED(mutex_);

  // ----------------------------------------------------------------------
  // Refactor: CPU/GPU pods to group related members under a single lock
  // ----------------------------------------------------------------------
  struct CpuPod {
    MemoryState state = MemoryState::UNINITIALIZED;
    absl::CondVar cond;
    bool comm_registered = false;
    CommRegistrationInfo comm_registration_info;
    std::vector<memory::DistributedVirtualMemoryPool::ChunkResidencyLease> pin_leases;
    // DVMP-backed VA info
    [[deprecated("Use UMA's get_cpu_base_ptr() instead")]] void* dvmp_base = nullptr; // Now managed by UMA
    // Last failure reason for observability
    std::string last_error;
  };

  struct GpuPod {
    MemoryState state = MemoryState::UNINITIALIZED;
    absl::CondVar cond;
    std::shared_ptr<CudaMemory> cuda_mem;
    // Dedicated CUDA stream for memory copies
    cudaStream_t stream = nullptr;
    bool stream_initialized = false;
    // Communication registration
    bool comm_registered = false;
    CommRegistrationInfo comm_registration_info;
    // Last failure reason for observability
    std::string last_error;
  };

  mutable absl::Mutex mutex_; // Protects all member variables below

  // CPU / GPU pods (guarded by the same mutex for now; can be split later)
  CpuPod cpu_ ABSL_GUARDED_BY(mutex_);
  GpuPod gpu_ ABSL_GUARDED_BY(mutex_);

  const uint64_t artifact_size_;

  // Memory Pools (immutable handles; lock-free reads)
  const gsl::not_null<std::shared_ptr<PinnedMemoryPool>> pinned_pool_;

  // Streaming buffer configuration
  const size_t max_buffer_bytes_; // 1 GB default

  // Pinned memory allocation timeout
  const std::chrono::milliseconds pinned_memory_timeout_;

  // Whether to fail transfer if DVMP chunk locking fails
  // [[maybe_unused]] const bool require_dvmp_lock_success_;

  const gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>> dvmp_;

  // Unified memory management instance
  const gsl::not_null<std::shared_ptr<ReplicaMemoryCoordinator>> memory_coordinator_;

  // New services introduced by RFC 0004
  const gsl::not_null<std::shared_ptr<class TransferService>> transfer_service_;
  const gsl::not_null<std::shared_ptr<class ChunkExportService>> export_service_;

  // Ensure GPU stream is initialised (create if needed). Expects mutex_ held.
  [[nodiscard]] absl::Status ensure_gpu_stream_initialized_locked_() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

} // namespace stepcast::store
