// Copyright (c) 2025, TensorCast Team.

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
#include "core/common/memory/memory_location.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/common/memory/virtual_address_space.h"
#include "core/communicator/engine/engine.h"
#include "core/store/communication_types.h"
#include "core/store/loader/source.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/chunk_meta.h"
#include "core/store/replica/memory_state.h"
#include "core/store/replica/types/direct_write_grant.h"
#include "core/store/replica/unified_memory_authority.h"

namespace tensorcast::store::replica {

/**
 * @brief Manages pageable CPU memory and GPU memory allocation, state, and transfers for a single replica instance.
 *
 * This class encapsulates the complexity of dealing with potentially chunked pageable CPU memory
 * and contiguous GPU memory, supporting both pool-based allocation and borrowing external pointers.
 * It ensures thread-safe access to memory resources and their states.
 */
class ReplicaLoadController {
 public:
  /**
   * @brief Constructs a ReplicaLoadController.
   * @param artifact_identifier A unique name for the replica, used for logging.
   * @param local_device_id The target local GPU device ID.
   * @param pinned_pool Shared pool for allocating pinned CPU memory.
   * @param virtual_addr_space Shared Virtual Address Space (VS).
   * @param max_buffer_bytes The maximum buffer size in bytes for streaming transfers (default 1 GB).
   * @param pinned_memory_timeout Timeout for pinned memory allocation operations.
   * @param artifact_size Total artifact size in bytes. Must be non-zero.
   *
   * Note: UMA (VS-managed virtual memory) is allocated eagerly during
   * construction. This does not consume physical memory and simplifies later
   * code paths by avoiding conditional UMA allocation.
   */
  ReplicaLoadController(
      std::string artifact_identifier,
      int local_device_id,
      const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>& pinned_pool,
      const gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>>& virtual_addr_space,
      size_t max_buffer_bytes,
      std::chrono::milliseconds pinned_memory_timeout,
      uint64_t artifact_size);

  ~ReplicaLoadController() noexcept;

  // Disable copy and move
  ReplicaLoadController(const ReplicaLoadController&) = delete;
  ReplicaLoadController& operator=(const ReplicaLoadController&) = delete;
  ReplicaLoadController(ReplicaLoadController&&) = delete;
  ReplicaLoadController& operator=(ReplicaLoadController&&) = delete;

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

  // Device id is bound at construction; no post-ctor mutation API is provided.

  /**
   * @brief Copies data from a peer ReplicaLoadController located on another device.
   *        This enables GPU↔GPU direct P2P transfers.
   */
  [[nodiscard]] absl::Status copy_from_peer(const ReplicaLoadController& source, cudaStream_t stream = nullptr);

  /**
   * @brief Allocates memory at the specified location using the configured memory pools.
   * Transitions state from UNALLOCATED to ALLOCATED on success.
   * @param location MemoryLocation::CPU or GPU.
   * @return absl::Status OkStatus on success, or an error status.
   */
  [[nodiscard]] absl::Status allocate_memory(common::memory::MemoryLocation location) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Releases memory at the specified location.
   * If the memory was allocated by this manager, it's returned to the pool or freed.
   * Transitions state to UNALLOCATED except when the current state is FAILED (kept for observability).
   * Refuses to release if the state is LOADING.
   * @param location CPU or GPU.
   * @return absl::Status OkStatus or FailedPrecondition if LOADING (GPU always fails while transfers are in-flight).
   */
  [[nodiscard]] absl::Status release_memory(common::memory::MemoryLocation location) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets the current memory state for the specified location.
   */
  [[nodiscard]] MemoryState get_state(common::memory::MemoryLocation location) const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Gets raw pointer(s) for the specified location.
   * - For GPU: returns one pointer when state is ALLOCATED, LOADING, or LOADED.
   * - For CPU: returns the VS base pointer when state is ALLOCATED or LOADED.
   * Returns an empty vector otherwise.
   * @param location CPU or GPU.
   * @return std::vector<void*> Vector with zero or one pointer.
   */
  [[nodiscard]] std::vector<void*> get_pointer(common::memory::MemoryLocation location) const
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Exposes the shared GPU allocation handle; primarily for ensuring the
  // lifetime of UMA-managed allocations during async transfers.
  [[nodiscard]] std::shared_ptr<common::memory::GpuDeviceMemory> get_gpu_allocation_shared() const
      ABSL_LOCKS_EXCLUDED(mutex_);

  struct GpuAllocationView {
    void* base_ptr = nullptr;
    std::shared_ptr<common::memory::GpuDeviceMemory> allocation;
  };

  [[nodiscard]] absl::StatusOr<GpuAllocationView> get_gpu_allocation_view() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Retrieves the CUDA IPC memory handle associated with the managed GPU
   *        memory. The replica must have its GPU memory buffer allocated and the
   *        underlying GpuDeviceMemory object initialised.
   *
   * The IPC handle can be obtained once the GPU buffer is allocated (states
   * ALLOCATED, LOADING, or LOADED).
   *
   * @return absl::StatusOr<cudaIpcMemHandle_t>  The CUDA IPC handle on success
   *         or an error status if the memory is not available.
   */
  [[nodiscard]] absl::StatusOr<cudaIpcMemHandle_t> get_ipc_handle() const noexcept ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Asynchronously copies data between CPU and GPU memory managed by this instance.
   * Assumes source location is in LOADED state and destination is ALLOCATED.
   * Manages state transitions (LOADING -> LOADED/FAILED) for the destination.
   * Uses asynchronous CUDA operations on a dedicated stream.
   *
   * @param source The source memory location (must be LOADED).
   * @param destination The destination memory location (must be ALLOCATED).
   * @return std::future<absl::Status> Future indicating the completion status of the copy.
   */
  [[nodiscard]] std::future<absl::Status> copy_data_async(
      common::memory::MemoryLocation source,
      common::memory::MemoryLocation destination) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief NEW: Orchestrate DISK/REMOTE → CPU/GPU using a provided source.
   * Performs allocation, state transitions, buffer setup, pumping, and finalization.
   */
  [[nodiscard]] std::future<absl::Status> load_async_from_source(
      std::unique_ptr<loader::SeekableSource> source,
      common::memory::MemoryLocation target_location,
      int concurrency,
      std::optional<absl::Span<const uint32_t>> chunk_indices = std::nullopt) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Waits for the memory at the specified location to reach the LOADED state.
   * @param location CPU or GPU.
   * @param timeout Optional timeout duration.
   * @return absl::Status OkStatus if LOADED, DeadlineExceeded if timeout, FailedPrecondition if FAILED state reached,
   * Cancelled if interrupted.
   */
  [[nodiscard]] absl::Status wait_for_state(
      common::memory::MemoryLocation location,
      MemoryState target_state,
      absl::Duration timeout = absl::InfiniteDuration()) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Registers the loaded memory (CPU or GPU) for communication access via the communicator engine.
   * Requires the memory to be in the LOADED state at the specified location.
   * NOTE: Deprecated. Use chunk-scoped APIs `export_chunks_for_p2p`/`unexport_chunks_for_p2p`.
   * @param location The memory location to register (MemoryLocation::CPU or MemoryLocation::GPU).
   * @param comm_engine The communicator engine to use for communication registration.
   * @return absl::StatusOr<ExportRegistration> Information needed by remote peers to access the memory, or an error.
   */

  // Convenience wrappers for buffer sizing
  // These are lock-free since members are immutable after construction.
  [[nodiscard]] size_t get_pool_chunk_size() const noexcept {
    return pinned_pool_->slice_bytes();
  }

  /**
   * @brief Attempts to finalize the state of a location after an asynchronous load/copy.
   * Checks if the state is currently LOADING for the given location. If so,
   * transitions it to LOADED on success or FAILED on error. If the state
   * is not LOADING, it logs a warning and does nothing.
   * @param location The memory location (CPU or GPU).
   * @param final_status The status of the completed load/copy operation.
   * @return absl::Status OkStatus, or the final_status if the state transition failed.
   */
  [[nodiscard]] absl::Status finalize_load_state(
      common::memory::MemoryLocation location,
      const absl::Status& final_status) ABSL_LOCKS_EXCLUDED(mutex_);

  // Streaming buffer helpers moved to TransferService

  // VS region reservation is handled by UMA

  // --- Internal State Management ---
  // These might be called by Loaders or copy operations.

  /**
   * @brief Sets the memory state for a location, performing checks and logging. Internal use mostly.
   * @param location CPU or GPU.
   * @param new_state The target state.
   * @return absl::Status Ok if transition is valid, error otherwise.
   */
  [[nodiscard]] absl::Status set_state(common::memory::MemoryLocation location, MemoryState new_state)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // Thin wrappers delegating to services -------------------------------------
  [[nodiscard]] absl::StatusOr<ExportRegistration> export_chunks_for_p2p(
      common::memory::MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      tensorcast::communicator::engine::Communicator& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);
  [[nodiscard]] absl::Status unexport_chunks_for_p2p(
      common::memory::MemoryLocation location,
      absl::Span<const uint32_t> chunks,
      tensorcast::communicator::engine::Communicator& comm_engine) ABSL_LOCKS_EXCLUDED(mutex_);

  // --- VA Space accessors ---------------------------------------------------
  /**
   * @brief Exposes the underlying VirtualAddressSpace instance. Loaders can
   *        use this to allocate/map/pin ranges in the CPU VA region.
   */
  [[nodiscard]] gsl::not_null<common::memory::VirtualAddressSpace*> get_va_space() ABSL_LOCKS_EXCLUDED(mutex_);
  [[nodiscard]] gsl::not_null<const common::memory::VirtualAddressSpace*> get_va_space() const
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Returns an immutable snapshot view of VS ChunkMeta for telemetry.
   *        Empty span if VS not available or replica not allocated.
   *        UMA is authoritative; prefer UMA-based getters for state.
   */
  [[nodiscard]] absl::Span<const ChunkMeta> chunk_telemetry_snapshot() const noexcept ABSL_LOCKS_EXCLUDED(mutex_);

  // Globally-unique key identifying this replica replica (artifact_id + device + replica).
  loading::ReplicaKey replica_key_;

  // Accessor for callers needing the replica key (e.g. schedulers, metrics).
  [[nodiscard]] const loading::ReplicaKey& replica_key() const noexcept {
    return replica_key_;
  }

  // --- NEW: Unified Memory Management ---

  /**
   * @brief Allocate per-replica memory bookkeeping in the coordinator.
   * Reserves DRAM via VS for this replica. GPU allocations remain lazy.
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
      common::memory::MemoryLocation target,
      std::optional<int> device_id = std::nullopt) const ABSL_LOCKS_EXCLUDED(mutex_);

  // Plan a direct-write grant for destination VA ranges (CPU).
  [[nodiscard]] absl::StatusOr<DirectWriteGrant> plan_direct_write(absl::Span<const VaRange> ranges)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // UMA-backed chunk state snapshot for a target location. Returns a vector
  // of per-chunk states derived from UMA's ledger (device-aware for GPU).
  // On error, returns an empty vector.
  [[nodiscard]] std::vector<replica::ChunkState> get_chunk_states_uma(
      common::memory::MemoryLocation location,
      std::optional<int> device_id = std::nullopt) const ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  /**
   * @brief Logs state transitions.
   */
  void log_state_change(common::memory::MemoryLocation loc, MemoryState old_state, MemoryState new_state) const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Returns base pointer for CPU or GPU using UMA/local cache.
   * Expects mutex_ to be held by the caller.
   * Returns nullptr if state is not eligible or pointer unavailable.
   */
  [[nodiscard]] void* get_base_ptr_locked(common::memory::MemoryLocation location) const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Releases GPU resources (cuda memory).
   */
  void release_gpu_resources_locked() noexcept ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Internal helper that assumes mutex_ is already held.
   * Only to be used by methods that already acquired the lock to avoid
   * re‑entrant locking.
   */
  [[nodiscard]] absl::Status set_state_locked(common::memory::MemoryLocation location, MemoryState new_state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Helper to retrieve state and condition variable pointers under lock
  struct StateCond {
    MemoryState* state;
    absl::CondVar* cond;
    uint64_t* load_epoch;
  };

  // New helper returning StatusOr for cleaner call-sites
  [[nodiscard]] absl::StatusOr<StateCond> get_state_cond_locked(common::memory::MemoryLocation location)
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);

  // --- Refactor helpers (split allocate_memory) ---------------------------
  [[nodiscard]] absl::Status allocate_pageable_cpu() ABSL_LOCKS_EXCLUDED(mutex_);
  [[nodiscard]] absl::Status allocate_gpu_memory() ABSL_LOCKS_EXCLUDED(mutex_);

  // --- Internal copy refactor helpers (no behavior change) ---------------
  struct CopyLaunchParams {
    std::shared_ptr<common::memory::StreamingPinnedBuffer> streaming_buffer;
    std::shared_ptr<common::memory::VirtualAddressSpace> virtual_addr_space;
    void* va_space_base = nullptr;
    std::shared_ptr<common::memory::GpuDeviceMemory> cuda_mem;
    size_t total_size = 0;
    cudaStream_t stream = nullptr;
    uint32_t device_id = 0;
    std::string artifact_id;
  };

  // Capture and validate all state needed for an async copy and set LOADING
  // on destination under lock. Returns parameters for the copy task.
  absl::Status capture_copy_context_(
      common::memory::MemoryLocation source,
      common::memory::MemoryLocation destination,
      CopyLaunchParams* out,
      bool* need_allocate_um) ABSL_LOCKS_EXCLUDED(mutex_);

  // Finalize destination MemoryState based on copy status under lock.
  absl::Status finalize_copy_state_(common::memory::MemoryLocation destination, const absl::Status& copy_status)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // ----------------------------------------------------------------------
  // Small internal helpers for error recording (behavior-preserving)------
  // ----------------------------------------------------------------------
  void set_failure_locked_(common::memory::MemoryLocation location, std::string message)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  [[nodiscard]] std::string get_last_error_locked_(common::memory::MemoryLocation location) const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_);
  // Consolidated helper: atomically record failure message and transition to FAILED.
  void record_failure_and_fail_(common::memory::MemoryLocation location, std::string message)
      ABSL_LOCKS_EXCLUDED(mutex_);

  // ----------------------------------------------------------------------
  // Refactor: CPU/GPU pods to group related members under a single lock
  // ----------------------------------------------------------------------
  struct CpuPod {
    MemoryState state = MemoryState::UNINITIALIZED;
    absl::CondVar cond;
    bool comm_registered = false;
    ExportRegistration comm_registration_info;
    // VS base pointer is managed by UMA; no local cache here
    // Last failure reason for observability
    std::string last_error;
    uint64_t load_epoch = 0;
  };

  struct GpuPod {
    MemoryState state = MemoryState::UNINITIALIZED;
    absl::CondVar cond;
    std::shared_ptr<common::memory::GpuDeviceMemory> cuda_mem;
    // Dedicated CUDA stream for memory copies
    cudaStream_t stream = nullptr;
    bool stream_initialized = false;
    // Communication registration
    bool comm_registered = false;
    ExportRegistration comm_registration_info;
    // Last failure reason for observability
    std::string last_error;
    uint64_t load_epoch = 0;
  };

  mutable absl::Mutex mutex_; // Protects all member variables below

  // CPU / GPU pods (guarded by the same mutex for now; can be split later)
  CpuPod cpu_ ABSL_GUARDED_BY(mutex_);
  GpuPod gpu_ ABSL_GUARDED_BY(mutex_);

  const uint64_t artifact_size_;

  // Memory Pools (immutable handles; lock-free reads)
  const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> pinned_pool_;

  // Streaming buffer configuration
  const size_t max_buffer_bytes_; // 1 GB default

  // Pinned memory allocation timeout
  const std::chrono::milliseconds pinned_memory_timeout_;

  // Whether to fail transfer if VS chunk locking fails (unused in final UMA V3)
  // [[maybe_unused]] const bool require_va_space_lock_success_;

  const gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>> va_space_;

  // Unified memory management instance
  const gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>> memory_coordinator_;

  // New services introduced by RFC 0004
  const gsl::not_null<std::shared_ptr<class TransferService>> transfer_service_;
  const gsl::not_null<std::shared_ptr<class MemoryExportRegistry>> export_service_;

  // Ensure GPU stream is initialised (create if needed). Expects mutex_ held.
  [[nodiscard]] absl::Status ensure_gpu_stream_initialized_locked_() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

} // namespace tensorcast::store::replica
