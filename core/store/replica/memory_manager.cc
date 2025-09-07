// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/memory_manager.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"

#include "core/common/cuda_api.h"
#include "core/common/device_types.h"
#include "core/common/memory/memory_location.h"
#include "core/communicator/engine/engine.h"
#include "core/store/direct_write.h"
#include "core/store/replica/chunk_export_service.h"
#include "core/store/replica/transfer_service.h"

namespace tensorcast::store::replica {

using common::memory::DistributedVirtualMemoryPool;
using common::memory::MemoryLocation;
using common::memory::PinnedMemoryPool;
using loading::ReplicaKey;

MemoryManager::MemoryManager(
    std::string artifact_identifier,
    int local_device_id,
    const gsl::not_null<std::shared_ptr<PinnedMemoryPool>>& pinned_pool,
    const gsl::not_null<std::shared_ptr<common::memory::DistributedVirtualMemoryPool>>& dvmp,
    size_t max_buffer_bytes,
    std::chrono::milliseconds pinned_memory_timeout,
    uint64_t artifact_size)
    : artifact_size_(artifact_size),
      pinned_pool_(pinned_pool),
      max_buffer_bytes_(max_buffer_bytes),
      pinned_memory_timeout_(pinned_memory_timeout),
      dvmp_(dvmp),
      memory_coordinator_(std::make_shared<ReplicaMemoryCoordinator>(dvmp)),
      transfer_service_(
          std::make_shared<TransferService>(
              pinned_pool_,
              dvmp_,
              memory_coordinator_,
              ReplicaKey{
                  .artifact_id = artifact_identifier,
                  .device = {.type = DeviceType::GPU, .ordinal = local_device_id, .uuid = ""},
                  .replica = 0},
              TransferService::Config{
                  .max_buffer_bytes = max_buffer_bytes_,
                  .pinned_memory_timeout = pinned_memory_timeout_})),
      export_service_(std::make_shared<ChunkExportService>(memory_coordinator_, dvmp_)) {
  // Populate replica_key_ using constructor inputs
  replica_key_.artifact_id = std::move(artifact_identifier);
  replica_key_.device.type = DeviceType::GPU;
  replica_key_.device.ordinal = local_device_id;
  // Initialize services already done in initializer list

  {
    absl::MutexLock lock(&mutex_);
    // Initialize states properly based on whether pools are provided
    cpu_.state = MemoryState::UNALLOCATED;
    // GPU state depends on pool or potential borrowing later
    gpu_.state = MemoryState::UNALLOCATED; // Assume potential for allocation/borrowing
  }

  // Eagerly allocate UMA (DVMP-backed virtual memory) at construction time
  // This does not consume physical memory and simplifies later code paths.
  {
    auto uma_status = allocate_replica_memory();
    ABSL_CHECK_OK(uma_status) << "UMA allocation during construction failed: " << uma_status;
  }

  // Initialise CUDA context and non-blocking stream if a valid device id was provided at construction.
  if (replica_key_.device.ordinal >= 0) {
    absl::MutexLock lock(&mutex_);
    auto stream_status = ensure_gpu_stream_initialized_locked_();
    if (!stream_status.ok()) {
      LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id
                 << "): Failed to initialize CUDA stream during construction: " << stream_status;
    }
  }
}

MemoryManager::~MemoryManager() noexcept {
  cudaStream_t local_stream = nullptr;
  std::string id_copy; // Copy identifier for logging after potential lock release
  {
    absl::MutexLock lock(&mutex_);
    id_copy = replica_key_.artifact_id; // Copy identifier while lock is held
    local_stream = gpu_.stream;
    gpu_.stream_initialized = false; // Mark as not initialized early
    gpu_.stream = nullptr; // Prevent use after unlock if sync takes time
  }

  if (local_stream != nullptr) {
    VLOG(1) << "MemoryManager(" << id_copy << "): Synchronizing stream " << local_stream << " in destructor.";
    // cudaStreamSynchronize can block, do it outside the lock.
    auto sync_status = cuda::stream_synchronize(local_stream);
    if (!sync_status.ok()) {
      LOG(ERROR) << "MemoryManager(" << id_copy << "): Failed to synchronize CUDA stream " << local_stream
                 << " during destruction: " << sync_status.message();
    }
    auto destroy_status = cuda::stream_destroy(local_stream);
    if (!destroy_status.ok()) {
      LOG(ERROR) << "MemoryManager(" << id_copy << "): Failed to destroy CUDA stream " << local_stream << ": "
                 << destroy_status.message();
    } else {
      VLOG(1) << "MemoryManager(" << id_copy << "): Successfully destroyed stream " << local_stream << ".";
    }
  }

  {
    absl::MutexLock lock(&mutex_);
    release_gpu_resources_locked();
    VLOG(2) << "MemoryManager(" << replica_key_.artifact_id << "): Destructor finished.";
  }
}

uint64_t MemoryManager::get_artifact_size() const noexcept {
  return artifact_size_;
}

// get_local_device_id is defined inline in the header now.

absl::Status MemoryManager::allocate_memory(MemoryLocation location) {
  ABSL_CHECK_NE(artifact_size_, 0) << "Artifact size not set before allocation.";

  // Delegate to per-location helpers (they handle their own locking needs).
  switch (location) {
    case MemoryLocation::PAGEABLE_CPU:
      return allocate_pageable_cpu();
    case MemoryLocation::GPU:
      return allocate_gpu_memory();
    default:
      return absl::InvalidArgumentError(
          absl::Substitute(
              "MemoryManager($0): Invalid location for allocation: $1",
              replica_key_.artifact_id,
              location_to_string(location)));
  }
}

// --- New helpers extracted from original allocate_memory ------------------

absl::Status MemoryManager::allocate_pageable_cpu() {
  // Streaming buffer is injected and shared; UMA is allocated at construction.
  {
    absl::MutexLock lock(&mutex_);
    if (cpu_.state >= MemoryState::ALLOCATED) {
      VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): PAGEABLE_CPU already in state "
              << state_to_string(cpu_.state) << ". Allocation request ignored.";
      return absl::OkStatus();
    }
    if (cpu_.state != MemoryState::UNALLOCATED) {
      return absl::FailedPreconditionError(
          absl::Substitute(
              "MemoryManager($0): Cannot allocate PAGEABLE_CPU memory. Expected UNALLOCATED state, but found $1.",
              replica_key_.artifact_id,
              state_to_string(cpu_.state)));
    }
  }

  absl::MutexLock lock(&mutex_);
  return set_state_locked(MemoryLocation::PAGEABLE_CPU, MemoryState::ALLOCATED);
}

absl::Status MemoryManager::allocate_gpu_memory() {
  {
    absl::MutexLock lock(&mutex_);
    ABSL_CHECK(gpu_.stream_initialized) << "CUDA stream not initialized";

    if (gpu_.state >= MemoryState::ALLOCATED) {
      VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): GPU memory already in state "
              << state_to_string(gpu_.state) << ". Allocation request ignored.";
      return absl::OkStatus();
    }
    if (gpu_.state != MemoryState::UNALLOCATED) {
      return absl::FailedPreconditionError(
          absl::Substitute(
              "MemoryManager($0): Cannot allocate GPU memory. Expected UNALLOCATED state, but found $1.",
              replica_key_.artifact_id,
              state_to_string(gpu_.state)));
    }
  }

  VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): Requesting GPU allocation from UMA for "
          << artifact_size_ << " bytes on device " << replica_key_.device.ordinal << ".";

  auto gpu_alloc_result = memory_coordinator_->get_or_create_gpu_allocation(replica_key_, replica_key_.device.ordinal);
  if (!gpu_alloc_result.ok()) {
    (void)set_state(MemoryLocation::GPU, MemoryState::FAILED);
    return absl::ResourceExhaustedError(
        absl::Substitute(
            "MemoryManager($0): Failed UMA GPU allocation on device $1: $2",
            replica_key_.artifact_id,
            replica_key_.device.ordinal,
            gpu_alloc_result.status().message()));
  }

  absl::MutexLock lock(&mutex_);
  gpu_.cuda_mem = *gpu_alloc_result;
  VLOG(1) << "MemoryManager(" << replica_key_.artifact_id
          << "): UMA GPU allocation successful (ptr=" << gpu_.cuda_mem->get() << ").";
  return set_state_locked(MemoryLocation::GPU, MemoryState::ALLOCATED);
}

absl::Status MemoryManager::release_memory(MemoryLocation location, bool safe_release) {
  absl::MutexLock lock(&mutex_);

  std::string loc_str = location_to_string(location);
  auto sc_or = get_state_cond_locked(location);
  if (!sc_or.ok()) {
    return absl::InvalidArgumentError(
        absl::Substitute("MemoryManager($0): Invalid location for release: $1", replica_key_.artifact_id, loc_str));
  }
  MemoryState* state_ptr = sc_or->state;
  absl::CondVar* cond_ptr = sc_or->cond;

  // PAGEABLE_CPU has special handling (does not free DVMP region)
  if (location == MemoryLocation::PAGEABLE_CPU) {
    VLOG(2) << "MemoryManager(" << replica_key_.artifact_id
            << "): release_memory called for PAGEABLE_CPU (safe_release=" << safe_release << ")";

    MemoryState current_state = *state_ptr;
    if (current_state == MemoryState::LOADING && safe_release) {
      return absl::FailedPreconditionError(
          absl::Substitute(
              "MemoryManager($0): Cannot safely release PAGEABLE_CPU while LOADING.", replica_key_.artifact_id));
    }

    if (current_state == MemoryState::LOADING && !safe_release) {
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
    } else if (current_state != MemoryState::UNALLOCATED) {
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::UNALLOCATED));
    }

    return absl::OkStatus();
  }

  MemoryState current_state = *state_ptr;
  VLOG(2) << "MemoryManager(" << replica_key_.artifact_id << "): Requesting release for " << loc_str
          << " (current state: " << state_to_string(current_state) << ", safe_release: " << safe_release << ")";

  if (current_state <= MemoryState::UNALLOCATED) {
    VLOG(2) << "MemoryManager(" << replica_key_.artifact_id << "): Memory for " << loc_str
            << " already released or uninitialized. No action taken.";
    return absl::OkStatus();
  }

  if (current_state == MemoryState::LOADING) {
    if (safe_release) {
      LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id << "): Safe release requested for " << loc_str
                   << " while LOADING. Release denied.";
      return absl::FailedPreconditionError(
          absl::Substitute(
              "MemoryManager($0): Cannot safely release $1 memory while in LOADING state.",
              replica_key_.artifact_id,
              loc_str));
    }

    VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): Unsafe release requested for " << loc_str
            << " while LOADING. Attempting to wait briefly...";
    if (cond_ptr->WaitWithTimeout(&mutex_, absl::Milliseconds(500))) {
      LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id
                 << "): Timeout expired while waiting for LOADING state on " << loc_str
                 << " to resolve during unsafe release.";
    }

    current_state = *state_ptr;
    if (current_state == MemoryState::LOADING) {
      LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id << "): Force releasing " << loc_str
                 << " memory while still LOADING after wait. Setting state to FAILED. Potential resource issues.";
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
    } else {
      VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): State for " << loc_str << " changed to "
              << state_to_string(current_state) << " during wait. Proceeding with release.";
    }
  }

  // Proceed with GPU resource release
  release_gpu_resources_locked();

  // Clear communication registration if releasing the registered GPU location
  if (location == MemoryLocation::GPU && gpu_.comm_registered) {
    VLOG(2) << "MemoryManager(" << replica_key_.artifact_id
            << "): Clearing GPU communication registration as memory is being released.";
    gpu_.comm_registered = false;
  }

  if (*state_ptr != MemoryState::FAILED) {
    ABSL_CHECK_OK(set_state_locked(location, MemoryState::UNALLOCATED));
  } else {
    LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id << "): Resources for " << loc_str
                 << " released, but state remains FAILED due to earlier unsafe release during LOADING.";
  }

  VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): Finished release process for " << loc_str
          << ". Final state: " << state_to_string(*state_ptr);
  return absl::OkStatus();
}

// Private helper
void MemoryManager::release_gpu_resources_locked() noexcept {
  // Called with mutex held
  if (gpu_.cuda_mem) {
    VLOG(2) << "MemoryManager(" << replica_key_.artifact_id << "): Releasing allocated GPU memory object (pointer "
            << gpu_.cuda_mem->get() << " will be freed/returned to pool by CudaMemory dtor).";
    // UMA now manages GPU memory lifecycle
    gpu_.cuda_mem.reset();
  }
}

MemoryState MemoryManager::get_state(MemoryLocation location) const {
  absl::MutexLock lock(&mutex_);
  switch (location) {
    case MemoryLocation::PAGEABLE_CPU:
      return cpu_.state;
    case MemoryLocation::GPU:
      return gpu_.state;
    default:
      LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id
                   << "): get_state called with invalid location: " << static_cast<int>(location);
      return MemoryState::UNINITIALIZED;
  }
}

std::vector<void*> MemoryManager::get_pointer(MemoryLocation location) const {
  absl::MutexLock lock(&mutex_);
  void* base = get_base_ptr_locked(location);
  if (base == nullptr) {
    if (location == MemoryLocation::PAGEABLE_CPU) {
      VLOG(2) << "MemoryManager(" << replica_key_.artifact_id
              << "): get_pointer(PAGEABLE_CPU) returning null. State: " << state_to_string(cpu_.state)
              << ", StreamingBuffer valid: " << (transfer_service_->get_streaming_buffer() != nullptr);
    } else if (location == MemoryLocation::GPU) {
      VLOG(2) << "MemoryManager(" << replica_key_.artifact_id
              << "): get_pointer(GPU) returning empty. State: " << state_to_string(gpu_.state);
    } else {
      LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id
                   << "): get_pointer called with invalid location: " << static_cast<int>(location);
    }
    return {};
  }
  return std::vector<void*>({base});
}

void* MemoryManager::get_base_ptr_locked(MemoryLocation location) const {
  switch (location) {
    case MemoryLocation::PAGEABLE_CPU: {
      if (cpu_.state == MemoryState::LOADED || cpu_.state == MemoryState::ALLOCATED) {
        return memory_coordinator_->get_cpu_base_ptr(replica_key_);
      }
      return nullptr;
    }
    case MemoryLocation::GPU: {
      if (gpu_.state >= MemoryState::ALLOCATED && gpu_.state != MemoryState::FAILED) {
        if (gpu_.cuda_mem) {
          return gpu_.cuda_mem->get();
        }
        return memory_coordinator_->get_gpu_base_ptr(replica_key_, replica_key_.device.ordinal);
      }
      return nullptr;
    }
    default:
      return nullptr;
  }
}

// Public thread‑safe wrapper
absl::Status MemoryManager::set_state(MemoryLocation location, MemoryState new_state) {
  absl::MutexLock lock(&mutex_);
  return set_state_locked(location, new_state);
}

// Internal helper to fetch state and cond pointers (expects mutex_ held)
absl::StatusOr<MemoryManager::StateCond> MemoryManager::get_state_cond_locked(MemoryLocation location) {
  switch (location) {
    case MemoryLocation::PAGEABLE_CPU:
      return StateCond{.state = &cpu_.state, .cond = &cpu_.cond};
    case MemoryLocation::GPU:
      return StateCond{.state = &gpu_.state, .cond = &gpu_.cond};
    default:
      return absl::InvalidArgumentError("Invalid location");
  }
}

// Internal helper (expects mutex_ held)
absl::Status MemoryManager::set_state_locked(MemoryLocation location, MemoryState new_state) {
  auto sc_or = get_state_cond_locked(location);
  if (!sc_or.ok()) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id
               << "): Invalid location for set_state_locked: " << location_to_string(location);
    return sc_or.status();
  }

  MemoryState* state_ptr = sc_or->state;
  absl::CondVar* cond_ptr = sc_or->cond;

  MemoryState old_state = *state_ptr;
  if (old_state == new_state) {
    VLOG(2) << "MemoryManager(" << replica_key_.artifact_id << "): State for " << location_to_string(location)
            << " already " << state_to_string(new_state) << ". No change.";
    return absl::OkStatus();
  }

  // If we are recovering from a FAILED state to a non-failed state, clear last_error for that pod
  if (old_state == MemoryState::FAILED && new_state != MemoryState::FAILED) {
    if (location == MemoryLocation::PAGEABLE_CPU) {
      cpu_.last_error.clear();
    } else if (location == MemoryLocation::GPU) {
      gpu_.last_error.clear();
    }
  }

  log_state_change(location, old_state, new_state);
  *state_ptr = new_state;
  cond_ptr->SignalAll();
  return absl::OkStatus();
}

void MemoryManager::log_state_change(MemoryLocation loc, MemoryState old_state, MemoryState new_state) const {
  // Assumes mutex is held
  VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): " << location_to_string(loc) << " state changing from "
          << state_to_string(old_state) << " to " << state_to_string(new_state);
}

std::future<absl::Status> MemoryManager::copy_data_async(MemoryLocation source, MemoryLocation destination) {
  // Phase 1: capture params and mark destination LOADING
  CopyLaunchParams params;
  bool need_allocate_um = false;
  auto context_status = capture_copy_context_(source, destination, &params, &need_allocate_um);
  if (!context_status.ok()) {
    return std::async(std::launch::deferred, [context_status] { return context_status; });
  }

  // Allocate UMA if needed (avoid calling while holding mutex_)
  if (need_allocate_um) {
    auto st_um = allocate_replica_memory();
    if (!st_um.ok()) {
      return std::async(std::launch::deferred, [st_um] { return st_um; });
    }
  }

  // Phase 2: launch async copy task
  return std::async(std::launch::async, [this, source, destination, p = std::move(params)]() -> absl::Status {
    absl::Status copy_status;
    const bool src_host_async = (source == MemoryLocation::PAGEABLE_CPU);
    const bool dst_host_async = (destination == MemoryLocation::PAGEABLE_CPU);

    if (src_host_async && !dst_host_async) { // Host -> GPU
      copy_status = transfer_service_->copy_cpu_to_gpu_streaming(
          p.device_id, p.stream, p.cuda_mem ? p.cuda_mem->get() : nullptr, p.total_size);
    } else if (!src_host_async && dst_host_async) { // GPU -> Host
      copy_status = transfer_service_->copy_gpu_to_cpu_streaming(
          p.device_id, p.stream, p.cuda_mem ? p.cuda_mem->get() : nullptr, p.total_size);
    } else {
      LOG(ERROR) << "MemoryManager(" << p.artifact_id << "): Unsupported copy direction: " << static_cast<int>(source)
                 << " -> " << static_cast<int>(destination);
      copy_status = absl::InvalidArgumentError("Unsupported copy direction in async task.");
    }

    // Phase 3: finalize state and cleanup as needed
    return this->finalize_copy_state_(destination, copy_status);
  });
}

absl::Status MemoryManager::wait_for_state(MemoryLocation location, MemoryState target_state, absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);

  std::string loc_str = location_to_string(location);
  auto state_cond_or = get_state_cond_locked(location);
  if (!state_cond_or.ok()) {
    return absl::InvalidArgumentError(
        absl::Substitute(
            "MemoryManager($0): Invalid location for wait_for_state: $1", replica_key_.artifact_id, loc_str));
  }
  MemoryState* state_ptr = state_cond_or->state;
  absl::CondVar* cond_ptr = state_cond_or->cond;

  VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): Waiting for " << loc_str << " to reach state "
          << state_to_string(target_state) << " (current: " << state_to_string(*state_ptr) << ", timeout: " << timeout
          << ")";

  absl::Time deadline = (timeout == absl::InfiniteDuration()) ? absl::InfiniteFuture() : absl::Now() + timeout;

  while (*state_ptr != target_state && *state_ptr != MemoryState::FAILED) {
    if (absl::Now() >= deadline) {
      if (*state_ptr != target_state && *state_ptr != MemoryState::FAILED) {
        LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id << "): Timeout waiting for " << loc_str
                     << " to reach state " << state_to_string(target_state)
                     << ". Current state: " << state_to_string(*state_ptr);
        return absl::DeadlineExceededError(
            absl::Substitute("Timeout waiting for $0 state $1", loc_str, state_to_string(target_state)));
      }
      break;
    }
    cond_ptr->WaitWithDeadline(&mutex_, deadline);
  }

  if (*state_ptr == target_state) {
    VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): Wait successful. " << loc_str
            << " reached target state " << state_to_string(target_state);
    return absl::OkStatus();
  }
  if (*state_ptr == MemoryState::FAILED) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id << "): Wait completed because " << loc_str
               << " reached FAILED state while waiting for " << state_to_string(target_state);
    // Provide richer context if a failure reason was recorded
    std::string reason;
    {
      // Best-effort read of last error without introducing new locks
      reason = get_last_error_locked_(location);
    }
    if (!reason.empty()) {
      return absl::FailedPreconditionError(absl::Substitute("$0 operation failed: $1", loc_str, reason));
    }
    return absl::FailedPreconditionError(absl::Substitute("$0 operation failed", loc_str));
  }
  LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id << "): Wait loop exited with unexpected state "
             << state_to_string(*state_ptr) << " for " << loc_str;
  return absl::InternalError("Unexpected state after wait loop.");
}

absl::Status MemoryManager::finalize_load_state(MemoryLocation location, const absl::Status& final_status) {
  absl::MutexLock lock(&mutex_);

  auto sc_or = get_state_cond_locked(location);
  if (!sc_or.ok()) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id
               << "): Invalid location for finalize_load_state: " << location_to_string(location);
    return absl::InvalidArgumentError("Invalid location for finalize_load_state");
  }

  MemoryState* state_ptr = sc_or->state;
  std::string loc_str = location_to_string(location);

  MemoryState current_state = *state_ptr;

  if (current_state == MemoryState::LOADING) {
    MemoryState target_final_state = final_status.ok() ? MemoryState::LOADED : MemoryState::FAILED;
    VLOG(1) << "MemoryManager(" << replica_key_.artifact_id << "): Finalizing operation for " << loc_str
            << ". Operation status: " << final_status << ". Setting state from LOADING to "
            << state_to_string(target_final_state);
    // Use set_state_locked to update state and notify condition variables
    return set_state_locked(location, target_final_state); // Already under lock
  }
  // This is not necessarily an error; the state might have been changed by release_memory or another operation.
  LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id << "): Finalize requested for " << loc_str
               << ", but state was not LOADING (current: " << state_to_string(current_state)
               << "). State not updated. Operation status was: " << final_status;
  // Return OkStatus because the finalization logic itself didn't fail, even if no state change occurred.
  // The caller should primarily rely on the future's status (which is final_status).
  return absl::OkStatus();
}

// Small helpers to consolidate failure recording and retrieval under lock
void MemoryManager::set_failure_locked_(MemoryLocation location, std::string message) {
  if (location == MemoryLocation::PAGEABLE_CPU) {
    cpu_.last_error = std::move(message);
  } else if (location == MemoryLocation::GPU) {
    gpu_.last_error = std::move(message);
  }
}

std::string MemoryManager::get_last_error_locked_(MemoryLocation location) const {
  if (location == MemoryLocation::PAGEABLE_CPU) {
    return cpu_.last_error;
  }
  if (location == MemoryLocation::GPU) {
    return gpu_.last_error;
  }
  return {};
}

// Consolidated helper: atomically record failure message and transition to FAILED.
void MemoryManager::record_failure_and_fail_(MemoryLocation location, std::string message) {
  absl::MutexLock lock(&mutex_);
  set_failure_locked_(location, std::move(message));
  (void)set_state_locked(location, MemoryState::FAILED);
}

// ---------------------------------------------------------------------------
// Refactor helpers: capture and finalize copy state (no behavior change)
// ---------------------------------------------------------------------------

absl::Status MemoryManager::capture_copy_context_(
    MemoryLocation source,
    MemoryLocation destination,
    CopyLaunchParams* out,
    bool* need_allocate_um) {
  if (!out || !need_allocate_um) {
    return absl::InvalidArgumentError("Null output pointers for capture_copy_context_");
  }

  const std::string src_str = location_to_string(source);
  const std::string dst_str = location_to_string(destination);

  absl::MutexLock lock(&mutex_);

  if (!gpu_.stream_initialized || gpu_.stream == nullptr) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id << "): Cannot initiate copy. CUDA stream is not valid.";
    return absl::InternalError("CUDA stream not initialized.");
  }

  const bool src_is_host = (source == MemoryLocation::PAGEABLE_CPU);
  const bool dst_is_host = (destination == MemoryLocation::PAGEABLE_CPU);

  MemoryState src_state = src_is_host ? cpu_.state : gpu_.state;
  MemoryState dst_state = dst_is_host ? cpu_.state : gpu_.state;

  LOG(INFO) << "MemoryManager(" << replica_key_.artifact_id << "): Requesting async copy from " << src_str
            << " (state: " << state_to_string(src_state) << ") to " << dst_str
            << " (state: " << state_to_string(dst_state) << ")";

  // Validate states
  if (src_state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::Substitute(
            "MemoryManager($0): Source $1 is not in LOADED state for copy.", replica_key_.artifact_id, src_str));
  }
  if (dst_state != MemoryState::ALLOCATED) {
    return absl::FailedPreconditionError(
        absl::Substitute(
            "MemoryManager($0): Destination $1 is not in ALLOCATED state for copy.",
            replica_key_.artifact_id,
            dst_str));
  }

  // Validate buffers
  if (src_is_host || dst_is_host) {
    auto spb = transfer_service_->get_streaming_buffer();
    if (spb == nullptr) {
      return absl::FailedPreconditionError(
          "StreamingPinnedBuffer must be allocated before host↔device copy operations.");
    }
    out->streaming_buffer = spb;
  }
  // UMA now manages GPU memory, so we only need to check if cuda_mem exists
  if ((source == MemoryLocation::GPU || destination == MemoryLocation::GPU) && !gpu_.cuda_mem) {
    return absl::InternalError("GPU memory not allocated through UMA");
  }

  out->cuda_mem = gpu_.cuda_mem;
  if (src_is_host || dst_is_host) {
    out->dvmp = dvmp_;
    // Get DVMP base from UMA (single source of truth)
    out->dvmp_base = memory_coordinator_->get_cpu_base_ptr(replica_key_);
  }
  out->total_size = artifact_size_;
  out->stream = gpu_.stream;
  out->device_id = replica_key_.device.ordinal;
  out->artifact_id = replica_key_.artifact_id;

  *need_allocate_um = (src_is_host || dst_is_host) && !memory_coordinator_->has_allocation(replica_key_);

  // Mark destination as LOADING
  absl::Status st = set_state_locked(destination, MemoryState::LOADING);
  if (!st.ok()) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id << "): Failed to set destination " << dst_str
               << " state to LOADING: " << st;
  }
  return st;
}

absl::Status MemoryManager::finalize_copy_state_(MemoryLocation destination, const absl::Status& copy_status) {
  // Reuse finalize_load_state to set final state and notify waiters.
  std::string dst_str_async = location_to_string(destination);
  LOG(INFO) << "MemoryManager(" << replica_key_.artifact_id << "): Async copy to " << dst_str_async
            << " finished. Operation status: " << copy_status << ". Finalizing state.";
  auto st = finalize_load_state(destination, copy_status);

  // For successful copies, perform additional finalization
  if (copy_status.ok()) {
    if (destination == MemoryLocation::PAGEABLE_CPU) {
      auto uma_sync_status = finalize_load(destination);
      if (!uma_sync_status.ok()) {
        LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id
                     << "): UMA sync after GPU->CPU copy failed: " << uma_sync_status;
      }
    }

    if (destination == MemoryLocation::GPU) {
      absl::MutexLock release_lock(&mutex_);
      // Apply UMA post-GPU-load policy (EvictCPU by default)
      auto uma_policy_st = memory_coordinator_->post_gpu_load_policy(
          replica_key_, artifact_size_, ReplicaMemoryCoordinator::PostGpuLoadPolicy::EvictCPU);
      if (!uma_policy_st.ok()) {
        LOG(WARNING) << "MemoryManager(" << replica_key_.artifact_id
                     << "): UMA post_gpu_load_policy returned: " << uma_policy_st;
      }
      if (cpu_.state != MemoryState::FAILED) {
        ABSL_CHECK_OK(set_state_locked(MemoryLocation::PAGEABLE_CPU, MemoryState::UNALLOCATED));
      }
    }
  }
  return st;
}

absl::StatusOr<cudaIpcMemHandle_t> MemoryManager::get_cuda_ipc_handle() const noexcept {
  absl::MutexLock lock(&mutex_);

  // A CUDA IPC handle can be created as soon as the underlying GPU buffer is
  // allocated, even if an asynchronous H2D copy is still in progress.
  // Therefore we permit ALLOCATED and LOADING states (as well as the final
  // LOADED state) instead of requiring the operation to have fully
  // completed before exposing the handle.
  if (gpu_.state != MemoryState::LOADED && gpu_.state != MemoryState::ALLOCATED && gpu_.state != MemoryState::LOADING) {
    return absl::FailedPreconditionError("GPU memory is not yet allocated");
  }

  if (gpu_.cuda_mem == nullptr) {
    return absl::NotFoundError("CudaMemory object is not initialised");
  }

  return gpu_.cuda_mem->get_handle();
}

// ---------------------------------------------------------------------------
// Peer-to-Peer copy (GPU↔GPU) – experimental implementation
// ---------------------------------------------------------------------------
absl::Status MemoryManager::copy_from_peer(const MemoryManager& source, cudaStream_t ext_stream) {
  // Quick checks – both managers must have GPU memory LOADED.
  if (this == &source) {
    return absl::InvalidArgumentError("Source and destination MemoryManager are identical.");
  }

  if (source.get_state(MemoryLocation::GPU) != MemoryState::LOADED) {
    return absl::FailedPreconditionError("Source GPU memory not in LOADED state");
  }
  if (get_state(MemoryLocation::GPU) != MemoryState::ALLOCATED) {
    return absl::FailedPreconditionError("Destination GPU memory is not allocated");
  }

  // Use provided stream or our internal gpu_.stream.
  cudaStream_t stream_to_use = ext_stream;
  {
    absl::MutexLock lock(&mutex_);
    if (stream_to_use == nullptr) {
      auto st = ensure_gpu_stream_initialized_locked_();
      if (!st.ok()) {
        return st;
      }
      stream_to_use = gpu_.stream;
    }
  }

  void* src_ptr = nullptr;
  {
    // Source pointer is safe to access without lock as get_pointer uses its own mutex.
    auto vec = source.get_pointer(MemoryLocation::GPU);
    if (vec.empty() || vec[0] == nullptr) {
      return absl::InternalError("Source GPU pointer invalid");
    }
    src_ptr = vec[0];
  }
  void* dst_ptr = nullptr;
  {
    auto vec = get_pointer(MemoryLocation::GPU);
    if (vec.empty() || vec[0] == nullptr) {
      return absl::InternalError("Destination GPU pointer invalid");
    }
    dst_ptr = vec[0];
  }

  // Determine size (same on both managers).
  uint64_t bytes = source.get_artifact_size();
  if (bytes == 0 || bytes != get_artifact_size()) {
    return absl::FailedPreconditionError("Artifact size mismatch between source and destination");
  }

  // Launch async peer copy.
  auto st = cuda::memcpy_async(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream_to_use);
  if (!st.ok()) {
    ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
    return st;
  }

  // Update state to LOADING then LOADED when stream sync completes.
  ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::LOADING));
  auto sync_status = cuda::stream_synchronize(stream_to_use);
  if (!sync_status.ok()) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id
               << "): CUDA stream synchronization failed: " << sync_status;
    ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
    return sync_status;
  }
  ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::LOADED));

  // Update UMA states for GPU chunks after peer copy completes.
  ABSL_CHECK_OK(finalize_load(MemoryLocation::GPU));
  return absl::OkStatus();
}

// --- Chunk-scoped export / unexport for P2P access -------------------------

// Note: removed unused coalesce_ranges helper
absl::StatusOr<CommRegistrationInfo> MemoryManager::export_chunks_for_p2p(
    MemoryLocation location,
    absl::Span<const uint32_t> chunks,
    communicator::engine::CommunicateEngine& comm_engine) {
  absl::MutexLock lock(&mutex_);
  auto info_or = export_service_->export_chunks(replica_key_, location, chunks, comm_engine);
  if (!info_or.ok()) {
    return info_or.status();
  }
  CommRegistrationInfo info = *info_or;
  if (location == MemoryLocation::GPU) {
    gpu_.comm_registration_info = info;
    gpu_.comm_registered = true;
  } else if (location == MemoryLocation::PAGEABLE_CPU) {
    cpu_.comm_registration_info = info;
    cpu_.comm_registered = true;
  }
  return info;
}

absl::Status MemoryManager::unexport_chunks_for_p2p(
    MemoryLocation location,
    absl::Span<const uint32_t> /*chunks*/,
    communicator::engine::CommunicateEngine& comm_engine) {
  absl::MutexLock lock(&mutex_);
  if (location == MemoryLocation::GPU && gpu_.comm_registered) {
    auto st = export_service_->unexport_chunks(replica_key_, gpu_.comm_registration_info, comm_engine);
    if (st.ok()) {
      gpu_.comm_registered = false;
      gpu_.comm_registration_info = {};
    }
    return st;
  }
  if (location == MemoryLocation::PAGEABLE_CPU && cpu_.comm_registered) {
    auto st = export_service_->unexport_chunks(replica_key_, cpu_.comm_registration_info, comm_engine);
    if (st.ok()) {
      cpu_.comm_registered = false;
      cpu_.comm_registration_info = {};
    }
    return st;
  }
  return absl::OkStatus();
}

// --- DVMP accessor implementation ---
gsl::not_null<common::memory::DistributedVirtualMemoryPool*> MemoryManager::get_dvmp() {
  return gsl::not_null<common::memory::DistributedVirtualMemoryPool*>{dvmp_.get().get()};
}

gsl::not_null<const common::memory::DistributedVirtualMemoryPool*> MemoryManager::get_dvmp() const {
  return gsl::not_null<const common::memory::DistributedVirtualMemoryPool*>{dvmp_.get().get()};
}

// Opaque keepalive container for DVMP pin leases held by a DirectWriteToken
namespace {
//------------------------------------------------------------------------------
// Helper: coalesce a sorted list of chunk indices into contiguous [start, end]
// ranges.  This utility is used by finalize_load when syncing CPU chunk states
// with the UMA coordinator.  Behaviour is identical to the in-line implementation
// previously found in finalize_load but is now shared and unit-testable.
//------------------------------------------------------------------------------
std::vector<std::pair<uint32_t, uint32_t>> coalesce_indices_to_ranges(absl::Span<const uint32_t> indices) {
  std::vector<std::pair<uint32_t, uint32_t>> ranges;
  if (indices.empty()) {
    return ranges;
  }

  // Make a local copy for sorting & dedup in-place to avoid mutating caller data.
  std::vector<uint32_t> sorted(indices.begin(), indices.end());
  std::ranges::sort(sorted);
  sorted.erase(std::ranges::unique(sorted).begin(), sorted.end());

  uint32_t range_start = sorted.front();
  uint32_t range_end = range_start;

  for (size_t i = 1; i < sorted.size(); ++i) {
    const uint32_t idx = sorted[i];
    if (idx == range_end + 1) {
      range_end = idx;
    } else {
      ranges.emplace_back(range_start, range_end);
      range_start = range_end = idx;
    }
  }
  ranges.emplace_back(range_start, range_end);
  return ranges;
}
} // namespace

absl::StatusOr<DirectWriteToken> MemoryManager::plan_direct_write(absl::Span<const VaRange> ranges) {
  {
    absl::MutexLock lock(&mutex_);
    if (cpu_.state != MemoryState::LOADED && cpu_.state != MemoryState::ALLOCATED) {
      return absl::FailedPreconditionError("CPU memory must be allocated/loaded for direct write");
    }
  }
  return memory_coordinator_->create_direct_write_token(replica_key_, ranges);
}

absl::Status MemoryManager::finalize_load(
    MemoryLocation location,
    std::optional<absl::Span<const uint32_t>> chunk_indices) const {
  // Handle CPU targets by syncing UMA from DVMP
  if (location == MemoryLocation::PAGEABLE_CPU) {
    if (chunk_indices.has_value() && !chunk_indices->empty()) {
      // Convert arbitrary chunk index list to compact ranges and sync with UMA.
      const auto ranges = coalesce_indices_to_ranges(*chunk_indices);
      memory_coordinator_->sync_cpu_chunk_states(replica_key_, absl::MakeConstSpan(ranges));
    } else {
      memory_coordinator_->sync_cpu_chunk_states(replica_key_);
    }
    return absl::OkStatus();
  }

  // Only GPU updates UMA states explicitly
  if (location != MemoryLocation::GPU) {
    return absl::OkStatus();
  }

  // Determine new state for GPU loads
  replica::ChunkState new_state = replica::ChunkState::COPIED_GPU;

  // Gather chunk list
  std::vector<uint32_t> chunks;
  if (chunk_indices.has_value()) {
    chunks.assign(chunk_indices->begin(), chunk_indices->end());
  } else {
    // Build [0..N-1]
    auto span = chunk_snapshot();
    chunks.resize(span.size());
    std::iota(chunks.begin(), chunks.end(), 0U);
  }

  // Provide local device id for GPU state updates
  int device_id = get_local_device_id();
  return memory_coordinator_->update_chunk_states(replica_key_, location, chunks, new_state, device_id);
}

// --- DVMP metadata snapshot -------------------------------------------------
// Provides a lightweight, read-only view of per-chunk metadata stored inside
// the DistributedVirtualMemoryPool (DVMP).  The span remains valid as long as the
// DVMP instance itself lives.  Callers must treat the returned ChunkMeta
// objects as immutable and use the atomic accessors defined inside ChunkMeta
// for state inspection.
absl::Span<const ChunkMeta> MemoryManager::chunk_snapshot() const noexcept {
  // No expensive operations here – simply delegate to DVMP.
  return dvmp_->chunk_snapshot(replica_key_.artifact_id);
}

// --- NEW: Unified Memory Management implementations ---

absl::Status MemoryManager::allocate_replica_memory() {
  absl::MutexLock lock(&mutex_);

  ABSL_CHECK_NE(artifact_size_, 0) << "Artifact size must be set before UMA allocation";
  ABSL_CHECK(!memory_coordinator_->has_allocation(replica_key_)) << "UMA replica allocation already exists";

  // Allocate via unified memory (which will use DVMP internally)
  auto status = memory_coordinator_->allocate(replica_key_, artifact_size_);
  if (!status.ok()) {
    return status;
  }

  LOG(INFO) << "MemoryManager(" << replica_key_.artifact_id << "): Allocated unified memory for " << artifact_size_
            << " bytes";

  return absl::OkStatus();
}

absl::Status MemoryManager::mark_cpu_preemptible(float ratio) {
  absl::MutexLock lock(&mutex_);

  if (!memory_coordinator_->has_allocation(replica_key_)) {
    return absl::FailedPreconditionError("Unified memory not allocated for this replica");
  }

  if (ratio < 0.0F || ratio > 1.0F) {
    return absl::InvalidArgumentError("Ratio must be between 0.0F and 1.0F");
  }

  // Delegate selection and DVMP interaction to UMA
  auto status = memory_coordinator_->mark_cpu_chunks_preemptible(replica_key_, ratio);
  if (!status.ok()) {
    return status;
  }

  LOG(INFO) << "MemoryManager(" << replica_key_.artifact_id
            << "): Marked CPU chunks as preemptible via UMA (ratio=" << ratio << ")";

  return absl::OkStatus();
}

std::vector<uint32_t> MemoryManager::get_missing_chunks(MemoryLocation target, std::optional<int> device_id) const {
  absl::MutexLock lock(&mutex_);
  ABSL_CHECK(memory_coordinator_->has_allocation(replica_key_)) << "UMA allocation not found";

  return memory_coordinator_->get_missing_chunks(replica_key_, target, device_id);
}

// build_sink_ and range utilities moved to TransferService

std::future<absl::Status> MemoryManager::load_async_from_source(
    std::unique_ptr<loader::SeekableSource> source,
    MemoryLocation target_location,
    int concurrency,
    std::optional<absl::Span<const uint32_t>> chunk_indices) {
  // Phase 1: capture state under lock and ensure allocation + LOADING
  bool need_allocate = false;
  {
    absl::MutexLock lock(&mutex_);
    ABSL_CHECK_NE(artifact_size_, 0) << "Artifact size must be set before loading";
    // Allocate destination if needed
    auto sc_or3 = get_state_cond_locked(target_location);
    if (!sc_or3.ok()) {
      const auto& err = sc_or3.status();
      return std::async(std::launch::deferred, [err] { return err; });
    }
    MemoryState* state_ptr = sc_or3->state;
    if (*state_ptr == MemoryState::UNALLOCATED) {
      need_allocate = true;
    }
  }

  if (need_allocate) {
    auto st = allocate_memory(target_location);
    if (!st.ok()) {
      return std::async(std::launch::deferred, [st] { return st; });
    }
  }

  // Streaming buffer is injected and shared; no allocation here.

  // Mark destination LOADING
  {
    absl::MutexLock lock(&mutex_);
    (void)set_state_locked(target_location, MemoryState::LOADING);
  }

  // Phase 2: launch async pump task delegated to TransferService
  return std::async(
      std::launch::async,
      [this, source = std::move(source), target_location, concurrency, chunk_indices]() mutable -> absl::Status {
        void* gpu_ptr = nullptr;
        int device_id = get_local_device_id();
        if (target_location == MemoryLocation::GPU) {
          auto vec = this->get_pointer(MemoryLocation::GPU);
          if (vec.empty()) {
            this->record_failure_and_fail_(MemoryLocation::GPU, "GPU memory not allocated");
            return absl::FailedPreconditionError("GPU memory not allocated");
          }
          gpu_ptr = vec[0];
        }

        absl::Status pump_status = transfer_service_->load_from_source(
            source, target_location, concurrency, chunk_indices, gpu_ptr, device_id);

        if (!pump_status.ok()) {
          this->record_failure_and_fail_(
              target_location, absl::Substitute("pump_ranges/load_from_source failed: $0", pump_status.message()));
          return pump_status;
        }

        absl::Status fin = this->finalize_load(target_location, chunk_indices);
        if (!fin.ok()) {
          this->record_failure_and_fail_(target_location, absl::Substitute("finalize_load failed: $0", fin.message()));
          return fin;
        }

        absl::Status st = this->set_state(target_location, MemoryState::LOADED);
        if (!st.ok()) {
          return st;
        }
        return absl::OkStatus();
      });
}

absl::Status MemoryManager::ensure_gpu_stream_initialized_locked_() {
  // Expects mutex_ held by caller
  if (gpu_.stream_initialized && gpu_.stream != nullptr) {
    return absl::OkStatus();
  }
  if (replica_key_.device.ordinal < 0) {
    return absl::FailedPreconditionError("Local CUDA device id not set");
  }
  auto dev_st = cuda::set_device(replica_key_.device.ordinal);
  if (!dev_st.ok()) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id
               << "): Failed to set CUDA device in ensure_gpu_stream_initialized_locked_: " << dev_st;
    return dev_st;
  }
  cudaStream_t created = nullptr;
  auto stream_st = cuda::stream_create_with_flags(&created, cudaStreamNonBlocking);
  if (!stream_st.ok()) {
    LOG(ERROR) << "MemoryManager(" << replica_key_.artifact_id
               << "): Failed to create CUDA stream in ensure_gpu_stream_initialized_locked_: " << stream_st;
    return stream_st;
  }
  gpu_.stream = created;
  gpu_.stream_initialized = true;
  VLOG(2) << "MemoryManager(" << replica_key_.artifact_id << "): Created CUDA stream " << gpu_.stream << " on device "
          << replica_key_.device.ordinal << ".";
  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
