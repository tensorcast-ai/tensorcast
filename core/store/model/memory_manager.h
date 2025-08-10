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
#include "core/common/memory/distributed_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/communicator/engine/engine.h"
#include "core/store/communication_types.h"
#include "core/store/direct_write.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/memory_types.h"
#include "core/store/model/chunk_meta.h"
#include "core/store/model/memory_state.h"
#include "core/store/model/model_location.h"
#include "core/store/model/unified_model_memory.h"

namespace stepcast::store {

/**
 * @brief Manages pageable CPU memory and GPU memory allocation, state, and transfers for a single model instance.
 *
 * This class encapsulates the complexity of dealing with potentially chunked pageable CPU memory
 * and contiguous GPU memory, supporting both pool-based allocation and borrowing external pointers.
 * It ensures thread-safe access to memory resources and their states.
 */
class MemoryManager {
 public:
  /**
   * @brief Constructs a MemoryManager.
   * @param model_identifier A unique name for the model, used for logging.
   * @param local_device_id The target local GPU device ID.
   * @param pinned_pool Shared pool for allocating pinned CPU memory.
   * @param dvmp Shared Distributed Virtual Memory Pool.
   * @param max_buffer_bytes The maximum buffer size in bytes for streaming transfers (default 1 GB).
   * @param pinned_memory_timeout Timeout for pinned memory allocation operations.
   * @param require_dvmp_lock_success Whether to fail transfer if DVMP chunk locking fails.
   */
  MemoryManager(
      std::string model_identifier,
      int local_device_id,
      const gsl::not_null<std::shared_ptr<PinnedMemoryPool>>& pinned_pool,
      const gsl::not_null<std::shared_ptr<memory::DistributedMemoryPool>>& dvmp,
      size_t max_buffer_bytes,
      std::chrono::milliseconds pinned_memory_timeout = std::chrono::milliseconds::zero(),
      bool require_dvmp_lock_success = true);

  ~MemoryManager();

  // Disable copy and move
  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;
  MemoryManager(MemoryManager&&) = delete;
  MemoryManager& operator=(MemoryManager&&) = delete;

  /**
   * @brief Sets the expected total size of the model data. Must be called before allocation/borrowing.
   * @param size Size in bytes.
   */
  void set_model_size(uint64_t size);

  /**
   * @brief Gets the configured model size.
   * @return uint64_t Model size in bytes, or 0 if not set.
   */
  uint64_t get_model_size() const;

  /**
   * @brief Gets the configured local CUDA device ID (or -1 if not yet set).
   */
  int get_local_device_id() const;

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
  absl::Status copy_from_peer(const MemoryManager& source, cudaStream_t stream = nullptr);

  /**
   * @brief Allocates memory at the specified location using the configured memory pools.
   * Transitions state from UNALLOCATED to ALLOCATED on success.
   * @param location ModelLocation::PAGEABLE_CPU or ModelLocation::GPU.
   * @return absl::Status OkStatus on success, or an error status.
   */
  absl::Status allocate_memory(ModelLocation location) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Releases memory at the specified location.
   * If the memory was allocated by this manager, it's returned to the pool or freed.
   * Transitions state to UNALLOCATED or UNINITIALIZED.
   * @param location PAGEABLE_CPU or GPU.
   * @param safe_release If true, refuses to release if state is LOADING.
   * @return absl::Status OkStatus or error (e.g., if safe_release is true and state is LOADING).
   */
  absl::Status release_memory(ModelLocation location, bool safe_release = false) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets the current memory state for the specified location.
   */
  MemoryState get_state(ModelLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets raw pointer(s) for the specified location.
   * - For GPU: returns one pointer when state is ALLOCATED, LOADING, or LOADED.
   * - For PAGEABLE_CPU: returns the DVMP base pointer when state is ALLOCATED or LOADED.
   * Returns an empty vector otherwise.
   * @param location PAGEABLE_CPU or GPU.
   * @return std::vector<void*> Vector with zero or one pointer.
   */
  std::vector<void*> get_pointer(ModelLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets access to the BatchVector used for CPU chunk status tracking during loading.
   * Primarily used internally by Loaders.
   * @return std::shared_ptr<BatchVector> Shared pointer to the BatchVector, or nullptr.
   */
  std::shared_ptr<BatchVector> get_host_chunk_queue() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Retrieves the CUDA IPC memory handle associated with the managed GPU
   *        memory. The model must have its GPU memory buffer allocated and the
   *        underlying CudaMemory object initialised.
   *
   * The IPC handle can be obtained once the GPU buffer is allocated (states
   * ALLOCATED, LOADING, or LOADED).
   *
   * @return absl::StatusOr<cudaIpcMemHandle_t>  The CUDA IPC handle on success
   *         or an error status if the memory is not available.
   */
  absl::StatusOr<cudaIpcMemHandle_t> get_cuda_ipc_handle() const ABSL_LOCKS_EXCLUDED(mutex_);

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
  std::future<absl::Status> copy_data_async(ModelLocation source, ModelLocation destination)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Waits for the memory at the specified location to reach the LOADED state.
   * @param location PAGEABLE_CPU or GPU.
   * @param timeout Optional timeout duration.
   * @return absl::Status OkStatus if LOADED, DeadlineExceeded if timeout, FailedPrecondition if FAILED state reached,
   * Cancelled if interrupted.
   */
  absl::Status wait_for_state(
      ModelLocation location,
      MemoryState target_state,
      absl::Duration timeout = absl::InfiniteDuration()) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Registers the loaded memory (PAGEABLE_CPU or GPU) for communication access via the communicator engine.
   * Requires the memory to be in the LOADED state at the specified location.
   * NOTE: Deprecated. Use chunk-scoped APIs `export_chunks_for_p2p`/`unexport_chunks_for_p2p`.
   * @param location The memory location to register (ModelLocation::PAGEABLE_CPU or ModelLocation::GPU).
   * @param comm_engine The communicator engine to use for communication registration.
   * @return absl::StatusOr<CommRegistrationInfo> Information needed by remote peers to access the memory, or an error.
   */

  /**
   * @brief Gets the size of individual chunks used for CPU pinned memory.
   * Returns 0 if CPU memory is not allocated or managed in chunks.
   */
  // Updated: If full pinned memory is not allocated, falls back to streaming
  // buffer chunk size. Returns 0 if neither is available.
  size_t get_cpu_chunk_size() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Attempts to finalize the state of a location after an asynchronous load/copy.
   * Checks if the state is currently LOADING for the given location. If so,
   * transitions it to LOADED on success or FAILED on error. If the state
   * is not LOADING, it logs a warning and does nothing.
   * @param location The memory location (PAGEABLE_CPU or GPU).
   * @param final_status The status of the completed load/copy operation.
   * @return absl::Status OkStatus, or the final_status if the state transition failed.
   */
  absl::Status finalize_load_state(ModelLocation location, const absl::Status& final_status)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Streaming buffer helpers -------------------------------------------------
  /**
   * @brief Ensures a streaming buffer pool is allocated with at least the
   *        specified number of chunks. Thread-safe and idempotent.
   */
  absl::Status ensure_streaming_buffer(size_t num_chunks) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Release the streaming buffer pool (if allocated).
   */
  absl::Status release_buffer_pool() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Accessor to the allocated StreamingPinnedBuffer (may return nullptr).
   */
  std::shared_ptr<StreamingPinnedBuffer> get_streaming_buffer() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get the configured maximum buffer size in bytes.
   */
  size_t get_max_buffer_bytes() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get chunk size of the underlying pinned memory pool (0 if pool null).
   */
  size_t get_pool_chunk_size() const ABSL_LOCKS_EXCLUDED(mutex_);

  // NEW: Reserve a pageable CPU VA region via DistributedMemoryPool (DVMP).
  // This wraps dvmp_->allocate() and records the base address internally.
  // If the region already exists, the returned Status will carry
  // absl::StatusCode::kAlreadyExists, mirroring DVMP semantics.
  // On success, the VirtualRegion structure contains the reserved base
  // address and size. The caller should treat kAlreadyExists as a signal
  // that the region has been allocated previously.
  absl::StatusOr<memory::DistributedMemoryPool::VirtualRegion> allocate_pageable_cpu_region()
      ABSL_LOCKS_EXCLUDED(mutex_);

  // --- Internal State Management ---
  // These might be called by Loaders or copy operations.

  /**
   * @brief Sets the memory state for a location, performing checks and logging. Internal use mostly.
   * @param location PAGEABLE_CPU or GPU.
   * @param new_state The target state.
   * @return absl::Status Ok if transition is valid, error otherwise.
   */
  absl::Status set_state(ModelLocation location, MemoryState new_state) ABSL_LOCKS_EXCLUDED(mutex_);

  // Chunk-scoped export APIs using DVMP pin leases (now supports CPU and GPU)
  absl::StatusOr<CommRegistrationInfo> export_chunks_for_p2p(
      ModelLocation location,
      absl::Span<const uint32_t> chunks,
      communicator::CommunicateEngine& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  absl::Status unexport_chunks_for_p2p(
      ModelLocation location,
      absl::Span<const uint32_t> chunks,
      communicator::CommunicateEngine& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  // --- New DVMP accessors ---------------------------------------------------
  /**
   * @brief Exposes the underlying DistributedMemoryPool instance used by this
   *        MemoryManager. Loaders can use this to allocate or lock chunks.
   */
  memory::DistributedMemoryPool* get_dvmp() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get the model identifier string.
   * @return The model identifier used by this MemoryManager.
   */
  const std::string& get_model_id() const {
    return instance_key_.model_id;
  }

  /**
   * @brief Returns an immutable snapshot view of ChunkMeta for the model.
   *        Empty span if DVMP not available or model not allocated.
   */
  absl::Span<const store::ChunkMeta> chunk_snapshot() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Globally-unique key identifying this model replica (model_id + device + replica).
  store::InstanceKey instance_key_;

  // Accessor for callers needing the replica key (e.g. schedulers, metrics).
  const store::InstanceKey& instance_key() const {
    return instance_key_;
  }

  // --- NEW: Unified Memory Management ---

  /**
   * @brief Get the unified memory instance for chunk-aware operations.
   * @return Pointer to UnifiedModelMemory or nullptr if not initialized.
   */
  std::shared_ptr<UnifiedModelMemory> get_unified_memory() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Allocate unified memory for the model.
   * Creates UnifiedModelMemory instance and reserves DRAM via DVMP.
   * @return Status of allocation.
   */
  absl::Status allocate_unified() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Mark CPU chunks as preemptible after GPU loading.
   * @param ratio Fraction of chunks to mark as preemptible (0.0-1.0).
   * @return Status of operation.
   */
  absl::Status mark_cpu_preemptible(float ratio = 1.0F) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get missing chunks for a target location.
   * @param target Target memory location.
   * @param device_id Device ID for GPU targets.
   * @return Vector of missing chunk indices.
   */
  std::vector<uint32_t> get_missing_chunks(ModelLocation target, std::optional<int> device_id = std::nullopt) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // NEW: Expose base address of DVMP region reserved for PAGEABLE_CPU. Returns nullptr if region not allocated.
  void* get_dvmp_cpu_base() const ABSL_LOCKS_EXCLUDED(mutex_);

  // Plan a direct-write token for destination VA ranges (PAGEABLE_CPU).
  absl::StatusOr<DirectWriteToken> plan_direct_write(absl::Span<const VaRange> ranges) ABSL_LOCKS_EXCLUDED(mutex_);

  // Explicit finalize hook to update UMA states after a load into a location
  // completes (replaces UnifiedMemorySink close-time updates).
  absl::Status finalize_load(
      ModelLocation location,
      std::optional<absl::Span<const uint32_t>> chunk_indices = std::nullopt) ABSL_LOCKS_EXCLUDED(mutex_);

  // Provide a per-model DVMP handle (region) for sinks and loaders.
  absl::StatusOr<memory::DistributedMemoryPool::DvmpRegion> get_dvmp_region() const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  /**
   * @brief Helper to perform the asynchronous CPU -> GPU copy.
   */
  absl::Status perform_copy_cpu_to_gpu_async() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Helper to perform the asynchronous GPU -> CPU copy.
   */
  absl::Status perform_copy_gpu_to_cpu_async() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Logs state transitions.
   */
  void log_state_change(ModelLocation loc, MemoryState old_state, MemoryState new_state) const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Releases CPU resources (pinned memory, batch vector).
   */
  // Updated: Also releases streaming buffer pool if allocated.
  void release_cpu_resources_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Releases GPU resources (cuda memory).
   */
  void release_gpu_resources_locked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Internal helper that assumes mutex_ is already held.
   * Only to be used by methods that already acquired the lock to avoid
   * re‑entrant locking.
   */
  absl::Status set_state_locked(ModelLocation location, MemoryState new_state) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Helper to retrieve state and condition variable pointers under lock
  absl::Status get_state_and_cond_locked_(ModelLocation location, MemoryState** state_ptr, absl::CondVar** cond_ptr)
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Internal helper: allocate streaming buffer pool (caller must hold mutex_)
   */
  absl::Status allocate_buffer_pool(size_t num_chunks) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // --- Internal copy refactor helpers (no behavior change) ---------------
  struct CopyLaunchParams {
    std::shared_ptr<StreamingPinnedBuffer> streaming_buffer;
    std::shared_ptr<::stepcast::memory::DistributedMemoryPool> dvmp;
    void* dvmp_base = nullptr;
    std::shared_ptr<CudaMemory> cuda_mem;
    size_t total_size = 0;
    cudaStream_t stream = nullptr;
    uint32_t device_id = 0;
    std::string model_id;
  };

  // Capture and validate all state needed for an async copy and set LOADING
  // on destination under lock. Returns parameters for the copy task.
  absl::Status capture_copy_context_(
      ModelLocation source,
      ModelLocation destination,
      CopyLaunchParams* out,
      bool* need_allocate_um) ABSL_LOCKS_EXCLUDED(mutex_);

  // Finalize destination MemoryState based on copy status under lock.
  absl::Status finalize_copy_state_(ModelLocation destination, const absl::Status& copy_status)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // ----------------------------------------------------------------------
  // Refactor: CPU/GPU pods to group related members under a single lock
  // ----------------------------------------------------------------------
  struct CpuPod {
    MemoryState state = MemoryState::UNINITIALIZED;
    std::shared_ptr<BatchVector> host_chunk_queue = nullptr;
    absl::CondVar cond;
    bool comm_registered = false;
    CommRegistrationInfo comm_registration_info;
    std::vector<memory::DistributedMemoryPool::PinLease> pin_leases;
    // DVMP-backed VA info
    void* dvmp_base = nullptr;
    size_t dvmp_bytes = 0;
    // Streaming buffer used for staged transfers
    std::shared_ptr<StreamingPinnedBuffer> streaming_buffer = nullptr;
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
  };

  mutable absl::Mutex mutex_; // Protects all member variables below

  // CPU / GPU pods (guarded by the same mutex for now; can be split later)
  CpuPod cpu_ ABSL_GUARDED_BY(mutex_);
  GpuPod gpu_ ABSL_GUARDED_BY(mutex_);

  uint64_t model_size_ = 0;

  // Memory Pools
  gsl::not_null<std::shared_ptr<PinnedMemoryPool>> pinned_pool_ ABSL_GUARDED_BY(mutex_);

  // Streaming buffer configuration
  const size_t max_buffer_bytes_; // 1 GB default

  // Pinned memory allocation timeout
  const std::chrono::milliseconds pinned_memory_timeout_;

  // Whether to fail transfer if DVMP chunk locking fails
  [[maybe_unused]] const bool require_dvmp_lock_success_;

  gsl::not_null<std::shared_ptr<memory::DistributedMemoryPool>> dvmp_ ABSL_GUARDED_BY(mutex_);

  // Unified memory management instance
  std::shared_ptr<UnifiedModelMemory> unified_memory_ ABSL_GUARDED_BY(mutex_);

  // --- New: small internal helpers to reduce duplication -----------------
  // Reserve/open DVMP region and cache base/size. Expects mutex_ held.
  absl::StatusOr<memory::DistributedMemoryPool::VirtualRegion> reserve_dvmp_region_locked_()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Ensure GPU stream is initialised (create if needed). Expects mutex_ held.
  absl::Status ensure_gpu_stream_initialized_locked_() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

} // namespace stepcast::store
