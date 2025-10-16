// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/replica_load_controller.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/substitute.h"
#include "absl/time/clock.h"

#include "core/common/async_copy_manager.h"
#include "core/common/cuda_api.h"
#include "core/common/device_types.h"
#include "core/common/memory/memory_location.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/communicator/engine/engine.h"
#include "core/store/replica/memory_export_registry.h"
#include "core/store/replica/transfer_service.h"
#include "core/store/replica/types/direct_write_grant.h"
#include "gsl/pointers"

namespace tensorcast::store::replica {

using common::memory::MemoryLocation;
using common::memory::PinnedBufferPool;
using common::memory::VirtualAddressSpace;
using loading::ReplicaKey;

// V3 final cutover: plan/execute/commit path is always enabled (no flags)

ReplicaLoadController::ReplicaLoadController(
    std::string artifact_identifier,
    int local_device_id,
    const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>& pinned_pool,
    const gsl::not_null<std::shared_ptr<common::memory::VirtualAddressSpace>>& virtual_addr_space,
    size_t max_buffer_bytes,
    std::chrono::milliseconds pinned_memory_timeout,
    uint64_t artifact_size,
    std::optional<std::string> view_id)
    : artifact_size_(artifact_size),
      pinned_pool_(pinned_pool),
      max_buffer_bytes_(max_buffer_bytes),
      pinned_memory_timeout_(pinned_memory_timeout),
      va_space_(virtual_addr_space),
      memory_coordinator_(std::make_shared<UnifiedMemoryAuthority>(virtual_addr_space)),
      transfer_service_(
          std::make_shared<TransferService>(
              pinned_pool_,
              va_space_,
              memory_coordinator_,
              ReplicaKey{
                  .artifact_id = artifact_identifier,
                  .view_id = view_id,
                  .device = {.type = DeviceType::GPU, .ordinal = local_device_id, .uuid = ""},
                  .replica = 0},
              TransferService::Config{
                  .max_buffer_bytes = max_buffer_bytes_,
                  .pinned_memory_timeout = pinned_memory_timeout_})),
      export_service_(
          std::make_shared<MemoryExportRegistry>(
              gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>>{memory_coordinator_},
              va_space_)) {
  // Populate replica_key_ using constructor inputs
  replica_key_.artifact_id = std::move(artifact_identifier);
  replica_key_.view_id = std::move(view_id);
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

  // Eagerly allocate UMA (VS-managed virtual memory) at construction time
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
      LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id
                 << "): Failed to initialize CUDA stream during construction: " << stream_status;
    }
  }
}

ReplicaLoadController::~ReplicaLoadController() noexcept {
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
    VLOG(1) << "ReplicaLoadController(" << id_copy << "): Synchronizing stream " << local_stream << " in destructor.";
    // cudaStreamSynchronize can block, do it outside the lock.
    auto sync_status = cuda::stream_synchronize(local_stream);
    if (!sync_status.ok()) {
      LOG(ERROR) << "ReplicaLoadController(" << id_copy << "): Failed to synchronize CUDA stream " << local_stream
                 << " during destruction: " << sync_status.message();
    }
    auto destroy_status = cuda::stream_destroy(local_stream);
    if (!destroy_status.ok()) {
      LOG(ERROR) << "ReplicaLoadController(" << id_copy << "): Failed to destroy CUDA stream " << local_stream << ": "
                 << destroy_status.message();
    } else {
      VLOG(1) << "ReplicaLoadController(" << id_copy << "): Successfully destroyed stream " << local_stream << ".";
    }
  }

  {
    absl::MutexLock lock(&mutex_);
    release_gpu_resources_locked();
    VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Destructor finished.";
  }
}

uint64_t ReplicaLoadController::get_artifact_size() const noexcept {
  return artifact_size_;
}

// get_local_device_id is defined inline in the header now.

absl::Status ReplicaLoadController::allocate_memory(MemoryLocation location) {
  ABSL_CHECK_NE(artifact_size_, 0) << "Artifact size not set before allocation.";

  // Delegate to per-location helpers (they handle their own locking needs).
  switch (location) {
    case MemoryLocation::CPU:
      return allocate_pageable_cpu();
    case MemoryLocation::GPU:
      return allocate_gpu_memory();
    default:
      return absl::InvalidArgumentError(
          absl::Substitute(
              "ReplicaLoadController($0): Invalid location for allocation: $1",
              replica_key_.artifact_id,
              location_to_string(location)));
  }
}

// --- New helpers extracted from original allocate_memory ------------------

absl::Status ReplicaLoadController::allocate_pageable_cpu() {
  // Streaming buffer is injected and shared; UMA is allocated at construction.
  {
    absl::MutexLock lock(&mutex_);
    if (cpu_.state >= MemoryState::ALLOCATED) {
      VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): CPU already in state "
              << state_to_string(cpu_.state) << ". Allocation request ignored.";
      return absl::OkStatus();
    }
    if (cpu_.state != MemoryState::UNALLOCATED) {
      return absl::FailedPreconditionError(
          absl::Substitute(
              "ReplicaLoadController($0): Cannot allocate CPU memory. Expected UNALLOCATED state, but found $1.",
              replica_key_.artifact_id,
              state_to_string(cpu_.state)));
    }
  }

  absl::MutexLock lock(&mutex_);
  return set_state_locked(MemoryLocation::CPU, MemoryState::ALLOCATED);
}

absl::Status ReplicaLoadController::allocate_gpu_memory() {
  {
    absl::MutexLock lock(&mutex_);
    ABSL_CHECK(gpu_.stream_initialized) << "CUDA stream not initialized";

    if (gpu_.state >= MemoryState::ALLOCATED) {
      VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): GPU memory already in state "
              << state_to_string(gpu_.state) << ". Allocation request ignored.";
      return absl::OkStatus();
    }
    if (gpu_.state != MemoryState::UNALLOCATED) {
      return absl::FailedPreconditionError(
          absl::Substitute(
              "ReplicaLoadController($0): Cannot allocate GPU memory. Expected UNALLOCATED state, but found $1.",
              replica_key_.artifact_id,
              state_to_string(gpu_.state)));
    }
  }

  VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Requesting GPU allocation from UMA for "
          << artifact_size_ << " bytes on device " << replica_key_.device.ordinal << ".";

  auto gpu_alloc_result = memory_coordinator_->get_or_create_gpu_allocation(replica_key_, replica_key_.device.ordinal);
  if (!gpu_alloc_result.ok()) {
    {
      absl::Status _st = set_state(MemoryLocation::GPU, MemoryState::FAILED);
      if (!_st.ok()) {
        LOG(WARNING) << "allocate_gpu_memory: failed to set state to FAILED after UMA alloc error: " << _st;
      }
    }
    return absl::ResourceExhaustedError(
        absl::Substitute(
            "ReplicaLoadController($0): Failed UMA GPU allocation on device $1: $2",
            replica_key_.artifact_id,
            replica_key_.device.ordinal,
            gpu_alloc_result.status().message()));
  }

  absl::MutexLock lock(&mutex_);
  gpu_.cuda_mem = *gpu_alloc_result;
  VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id
          << "): UMA GPU allocation successful (ptr=" << gpu_.cuda_mem->get() << ").";
  return set_state_locked(MemoryLocation::GPU, MemoryState::ALLOCATED);
}

absl::Status ReplicaLoadController::release_memory(MemoryLocation location) {
  absl::MutexLock lock(&mutex_);

  std::string loc_str = location_to_string(location);
  auto sc_or = get_state_cond_locked(location);
  if (!sc_or.ok()) {
    return absl::InvalidArgumentError(
        absl::Substitute(
            "ReplicaLoadController($0): Invalid location for release: $1", replica_key_.artifact_id, loc_str));
  }
  MemoryState* state_ptr = sc_or->state;

  // CPU has special handling (does not free VS region)
  if (location == MemoryLocation::CPU) {
    VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id << "): release_memory called for CPU";

    MemoryState current_state = *state_ptr;
    if (current_state == MemoryState::LOADING) {
      return absl::FailedPreconditionError(
          absl::Substitute(
              "ReplicaLoadController($0): Cannot release CPU memory while a load is in progress.",
              replica_key_.artifact_id));
    }

    if (current_state != MemoryState::UNALLOCATED) {
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::UNALLOCATED));
    }

    return absl::OkStatus();
  }

  MemoryState current_state = *state_ptr;
  VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Requesting release for " << loc_str
          << " (current state: " << state_to_string(current_state) << ")";

  if (current_state <= MemoryState::UNALLOCATED) {
    VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Memory for " << loc_str
            << " already released or uninitialized. No action taken.";
    return absl::OkStatus();
  }

  if (current_state == MemoryState::LOADING) {
    // When a concurrent load is in-flight we refuse to tear down resources.
    // Callers should retry once the load completes.
    VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Release requested for " << loc_str
            << " while LOADING. Returning FailedPrecondition without releasing.";
    return absl::FailedPreconditionError(
        absl::Substitute(
            "ReplicaLoadController($0): Cannot release $1 memory while a load is in progress.",
            replica_key_.artifact_id,
            loc_str));
  }

  // Proceed with GPU resource release
  release_gpu_resources_locked();

  // Inform UMA to drop GPU residency and allocation for this device to keep
  // the authoritative ledger in sync and actually reclaim VRAM.
  if (location == MemoryLocation::GPU) {
    auto uma_st =
        memory_coordinator_->release_gpu_device(replica_key_, replica_key_.device.ordinal, /*drop_allocation=*/true);
    if (!uma_st.ok() && uma_st.code() != absl::StatusCode::kNotFound) {
      LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id
                   << "): UMA release_gpu_device returned: " << uma_st;
    }
  }

  // Clear communication registration if releasing the registered GPU location
  if (location == MemoryLocation::GPU && gpu_.comm_registered) {
    VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id
            << "): Clearing GPU communication registration as memory is being released.";
    gpu_.comm_registered = false;
  }

  if (*state_ptr != MemoryState::FAILED) {
    ABSL_CHECK_OK(set_state_locked(location, MemoryState::UNALLOCATED));
  } else {
    LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Resources for " << loc_str
                 << " released, but state remains FAILED due to earlier unsafe release during LOADING.";
  }

  VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Finished release process for " << loc_str
          << ". Final state: " << state_to_string(*state_ptr);
  return absl::OkStatus();
}

// Private helper
void ReplicaLoadController::release_gpu_resources_locked() noexcept {
  // Called with mutex held
  if (gpu_.cuda_mem) {
    VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id
            << "): Releasing allocated GPU memory object (pointer " << gpu_.cuda_mem->get()
            << " will be freed/returned to pool by GpuDeviceMemory dtor).";
    // UMA now manages GPU memory lifecycle
    gpu_.cuda_mem.reset();
  }
}

MemoryState ReplicaLoadController::get_state(MemoryLocation location) const {
  absl::MutexLock lock(&mutex_);
  switch (location) {
    case MemoryLocation::CPU:
      return cpu_.state;
    case MemoryLocation::GPU:
      return gpu_.state;
    default:
      LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id
                   << "): get_state called with invalid location: " << static_cast<int>(location);
      return MemoryState::UNINITIALIZED;
  }
}

std::vector<void*> ReplicaLoadController::get_pointer(MemoryLocation location) const {
  absl::MutexLock lock(&mutex_);
  void* base = get_base_ptr_locked(location);
  if (base == nullptr) {
    if (location == MemoryLocation::CPU) {
      VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id
              << "): get_pointer(CPU) returning null. State: " << state_to_string(cpu_.state)
              << ", StreamingBuffer valid: " << (transfer_service_->get_streaming_buffer() != nullptr);
    } else if (location == MemoryLocation::GPU) {
      VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id
              << "): get_pointer(GPU) returning empty. State: " << state_to_string(gpu_.state);
    } else {
      LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id
                   << "): get_pointer called with invalid location: " << static_cast<int>(location);
    }
    return {};
  }
  return std::vector<void*>({base});
}

std::shared_ptr<common::memory::GpuDeviceMemory> ReplicaLoadController::get_gpu_allocation_shared() const {
  absl::MutexLock lock(&mutex_);
  return gpu_.cuda_mem;
}

absl::StatusOr<ReplicaLoadController::GpuAllocationView> ReplicaLoadController::get_gpu_allocation_view() const {
  absl::MutexLock lock(&mutex_);
  if (!gpu_.cuda_mem) {
    return absl::FailedPreconditionError("GPU allocation not available");
  }
  void* base = get_base_ptr_locked(MemoryLocation::GPU);
  if (base == nullptr) {
    return absl::FailedPreconditionError("GPU base pointer not available");
  }
  GpuAllocationView view;
  view.base_ptr = base;
  view.allocation = gpu_.cuda_mem;
  return view;
}

void* ReplicaLoadController::get_base_ptr_locked(MemoryLocation location) const {
  switch (location) {
    case MemoryLocation::CPU: {
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
absl::Status ReplicaLoadController::set_state(MemoryLocation location, MemoryState new_state) {
  absl::MutexLock lock(&mutex_);
  return set_state_locked(location, new_state);
}

// Internal helper to fetch state and cond pointers (expects mutex_ held)
absl::StatusOr<ReplicaLoadController::StateCond> ReplicaLoadController::get_state_cond_locked(MemoryLocation location) {
  switch (location) {
    case MemoryLocation::CPU:
      return StateCond{.state = &cpu_.state, .cond = &cpu_.cond, .load_epoch = &cpu_.load_epoch};
    case MemoryLocation::GPU:
      return StateCond{.state = &gpu_.state, .cond = &gpu_.cond, .load_epoch = &gpu_.load_epoch};
    default:
      return absl::InvalidArgumentError("Invalid location");
  }
}

// Internal helper (expects mutex_ held)
absl::Status ReplicaLoadController::set_state_locked(MemoryLocation location, MemoryState new_state) {
  auto sc_or = get_state_cond_locked(location);
  if (!sc_or.ok()) {
    LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id
               << "): Invalid location for set_state_locked: " << location_to_string(location);
    return sc_or.status();
  }

  MemoryState* state_ptr = sc_or->state;
  absl::CondVar* cond_ptr = sc_or->cond;
  uint64_t* load_epoch_ptr = sc_or->load_epoch;

  MemoryState old_state = *state_ptr;
  if (old_state == new_state) {
    VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id << "): State for " << location_to_string(location)
            << " already " << state_to_string(new_state) << ". No change.";
    return absl::OkStatus();
  }

  // If we are recovering from a FAILED state to a non-failed state, clear last_error for that pod
  if (old_state == MemoryState::FAILED && new_state != MemoryState::FAILED) {
    if (location == MemoryLocation::CPU) {
      cpu_.last_error.clear();
    } else if (location == MemoryLocation::GPU) {
      gpu_.last_error.clear();
    }
  }

  log_state_change(location, old_state, new_state);
  *state_ptr = new_state;
  if (new_state == MemoryState::LOADED && load_epoch_ptr != nullptr) {
    ++(*load_epoch_ptr);
  }
  cond_ptr->SignalAll();
  return absl::OkStatus();
}

void ReplicaLoadController::log_state_change(MemoryLocation loc, MemoryState old_state, MemoryState new_state) const {
  // Assumes mutex is held
  VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): " << location_to_string(loc)
          << " state changing from " << state_to_string(old_state) << " to " << state_to_string(new_state);
}

std::future<absl::Status> ReplicaLoadController::copy_data_async(MemoryLocation source, MemoryLocation destination) {
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
    const bool src_host_async = (source == MemoryLocation::CPU);
    const bool dst_host_async = (destination == MemoryLocation::CPU);

    // Plan UMA session for transactional CPU↔GPU copies
    std::optional<int> dev_for_plan;
    if (destination == MemoryLocation::GPU)
      dev_for_plan = p.device_id;
    auto plan_or = memory_coordinator_->plan_load(replica_key_, destination, dev_for_plan, std::nullopt);
    if (!plan_or.ok()) {
      LOG(ERROR) << "ReplicaLoadController(" << p.artifact_id
                 << "): UMA plan_load failed for copy: " << plan_or.status();
      return this->finalize_copy_state_(destination, plan_or.status());
    }
    auto plan = *plan_or;

    if (src_host_async && !dst_host_async) { // Host -> GPU
      copy_status = transfer_service_->copy_cpu_to_gpu_streaming(
          p.device_id, gsl::not_null<void*>{p.cuda_mem->get()}, p.total_size);
    } else if (!src_host_async && dst_host_async) { // GPU -> Host
      copy_status = transfer_service_->copy_gpu_to_cpu_streaming(
          p.device_id, gsl::not_null<void*>{p.cuda_mem->get()}, p.total_size);
    } else {
      LOG(ERROR) << "ReplicaLoadController(" << p.artifact_id
                 << "): Unsupported copy direction: " << static_cast<int>(source) << " -> "
                 << static_cast<int>(destination);
      copy_status = absl::InvalidArgumentError("Unsupported copy direction in async task.");
    }

    if (!copy_status.ok()) {
      // Abort UMA session on failure (idempotent)
      (void)memory_coordinator_->abort(plan.session_id);
      return this->finalize_copy_state_(destination, copy_status);
    }

    // Commit UMA ledger for in-memory copy path
    std::optional<int> commit_dev;
    if (destination == MemoryLocation::GPU)
      commit_dev = p.device_id;
    absl::Status cst =
        memory_coordinator_->commit(plan.session_id, destination, absl::MakeSpan(plan.chunk_indices), commit_dev);
    if (!cst.ok()) {
      LOG(ERROR) << "ReplicaLoadController(" << p.artifact_id << "): UMA commit failed after copy: " << cst;
      return this->finalize_copy_state_(destination, cst);
    }

    // Phase 3: finalize state and cleanup as needed
    return this->finalize_copy_state_(destination, copy_status);
  });
}

absl::Status ReplicaLoadController::wait_for_state(
    MemoryLocation location,
    MemoryState target_state,
    absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);

  std::string loc_str = location_to_string(location);
  auto state_cond_or = get_state_cond_locked(location);
  if (!state_cond_or.ok()) {
    return absl::InvalidArgumentError(
        absl::Substitute(
            "ReplicaLoadController($0): Invalid location for wait_for_state: $1", replica_key_.artifact_id, loc_str));
  }
  MemoryState* state_ptr = state_cond_or->state;
  absl::CondVar* cond_ptr = state_cond_or->cond;
  uint64_t* load_epoch_ptr = state_cond_or->load_epoch;
  const uint64_t initial_epoch =
      (target_state == MemoryState::LOADED && load_epoch_ptr != nullptr) ? *load_epoch_ptr : 0;

  auto observed_target = [&]() -> bool {
    if (*state_ptr == target_state) {
      return true;
    }
    if (target_state == MemoryState::LOADED && load_epoch_ptr != nullptr && *load_epoch_ptr > initial_epoch) {
      return true;
    }
    return false;
  };

  VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Waiting for " << loc_str << " to reach state "
          << state_to_string(target_state) << " (current: " << state_to_string(*state_ptr) << ", timeout: " << timeout
          << ")";

  absl::Time deadline = (timeout == absl::InfiniteDuration()) ? absl::InfiniteFuture() : absl::Now() + timeout;

  if (observed_target()) {
    const bool epoch_advanced =
        target_state == MemoryState::LOADED && load_epoch_ptr != nullptr && *load_epoch_ptr > initial_epoch;
    if (epoch_advanced && *state_ptr != target_state) {
      VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id
              << "): Target state satisfied by a completed load epoch even though current state is "
              << state_to_string(*state_ptr);
    } else {
      VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Wait successful. " << loc_str
              << " already in target state " << state_to_string(target_state);
    }
    return absl::OkStatus();
  }

  while (!observed_target() && *state_ptr != MemoryState::FAILED) {
    if (absl::Now() >= deadline) {
      if (!observed_target() && *state_ptr != MemoryState::FAILED) {
        LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Timeout waiting for " << loc_str
                     << " to reach state " << state_to_string(target_state)
                     << ". Current state: " << state_to_string(*state_ptr);
        return absl::DeadlineExceededError(
            absl::Substitute("Timeout waiting for $0 state $1", loc_str, state_to_string(target_state)));
      }
      break;
    }
    cond_ptr->WaitWithDeadline(&mutex_, deadline);
  }

  if (observed_target()) {
    const bool epoch_advanced =
        target_state == MemoryState::LOADED && load_epoch_ptr != nullptr && *load_epoch_ptr > initial_epoch;
    if (epoch_advanced && *state_ptr != target_state) {
      VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Wait successful via load epoch advance. "
              << loc_str << " experienced a LOADED epoch after wait started; current state is "
              << state_to_string(*state_ptr);
    } else {
      VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Wait successful. " << loc_str
              << " reached target state " << state_to_string(target_state);
    }
    return absl::OkStatus();
  }
  if (*state_ptr == MemoryState::FAILED) {
    LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Wait completed because " << loc_str
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
  LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Wait loop exited with unexpected state "
             << state_to_string(*state_ptr) << " for " << loc_str;
  return absl::InternalError("Unexpected state after wait loop.");
}

absl::Status ReplicaLoadController::finalize_load_state(MemoryLocation location, const absl::Status& final_status) {
  absl::MutexLock lock(&mutex_);

  auto sc_or = get_state_cond_locked(location);
  if (!sc_or.ok()) {
    LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id
               << "): Invalid location for finalize_load_state: " << location_to_string(location);
    return absl::InvalidArgumentError("Invalid location for finalize_load_state");
  }

  MemoryState* state_ptr = sc_or->state;
  std::string loc_str = location_to_string(location);

  MemoryState current_state = *state_ptr;

  if (current_state == MemoryState::LOADING) {
    MemoryState target_final_state = final_status.ok() ? MemoryState::LOADED : MemoryState::FAILED;
    VLOG(1) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Finalizing operation for " << loc_str
            << ". Operation status: " << final_status << ". Setting state from LOADING to "
            << state_to_string(target_final_state);
    // Use set_state_locked to update state and notify condition variables
    return set_state_locked(location, target_final_state); // Already under lock
  }
  // This is not necessarily an error; the state might have been changed by release_memory or another operation.
  LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Finalize requested for " << loc_str
               << ", but state was not LOADING (current: " << state_to_string(current_state)
               << "). State not updated. Operation status was: " << final_status;
  // Return OkStatus because the finalization logic itself didn't fail, even if no state change occurred.
  // The caller should primarily rely on the future's status (which is final_status).
  return absl::OkStatus();
}

// Small helpers to consolidate failure recording and retrieval under lock
void ReplicaLoadController::set_failure_locked_(MemoryLocation location, std::string message) {
  if (location == MemoryLocation::CPU) {
    cpu_.last_error = std::move(message);
  } else if (location == MemoryLocation::GPU) {
    gpu_.last_error = std::move(message);
  }
}

std::string ReplicaLoadController::get_last_error_locked_(MemoryLocation location) const {
  if (location == MemoryLocation::CPU) {
    return cpu_.last_error;
  }
  if (location == MemoryLocation::GPU) {
    return gpu_.last_error;
  }
  return {};
}

// Consolidated helper: atomically record failure message and transition to FAILED.
void ReplicaLoadController::record_failure_and_fail_(MemoryLocation location, std::string message) {
  absl::MutexLock lock(&mutex_);
  set_failure_locked_(location, std::move(message));
  {
    absl::Status _st = set_state_locked(location, MemoryState::FAILED);
    if (!_st.ok()) {
      LOG(WARNING) << "record_failure_and_fail_: failed to set state to FAILED: " << _st;
    }
  }
}

// ---------------------------------------------------------------------------
// Refactor helpers: capture and finalize copy state (no behavior change)
// ---------------------------------------------------------------------------

absl::Status ReplicaLoadController::capture_copy_context_(
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
    LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id
               << "): Cannot initiate copy. CUDA stream is not valid.";
    return absl::InternalError("CUDA stream not initialized.");
  }

  const bool src_is_host = (source == MemoryLocation::CPU);
  const bool dst_is_host = (destination == MemoryLocation::CPU);

  MemoryState src_state = src_is_host ? cpu_.state : gpu_.state;
  MemoryState dst_state = dst_is_host ? cpu_.state : gpu_.state;

  LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Requesting async copy from " << src_str
            << " (state: " << state_to_string(src_state) << ") to " << dst_str
            << " (state: " << state_to_string(dst_state) << ")";

  // Validate states
  if (src_state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::Substitute(
            "ReplicaLoadController($0): Source $1 is not in LOADED state for copy.",
            replica_key_.artifact_id,
            src_str));
  }
  if (dst_state != MemoryState::ALLOCATED) {
    return absl::FailedPreconditionError(
        absl::Substitute(
            "ReplicaLoadController($0): Destination $1 is not in ALLOCATED state for copy.",
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
    out->virtual_addr_space = va_space_;
    // Get VS base from UMA (single source of truth)
    out->va_space_base = memory_coordinator_->get_cpu_base_ptr(replica_key_);
  }
  out->total_size = artifact_size_;
  out->stream = gpu_.stream;
  out->device_id = replica_key_.device.ordinal;
  out->artifact_id = replica_key_.artifact_id;

  *need_allocate_um = (src_is_host || dst_is_host) && !memory_coordinator_->has_allocation(replica_key_);

  // Mark destination as LOADING
  absl::Status st = set_state_locked(destination, MemoryState::LOADING);
  if (!st.ok()) {
    LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Failed to set destination " << dst_str
               << " state to LOADING: " << st;
  }
  return st;
}

absl::Status ReplicaLoadController::finalize_copy_state_(MemoryLocation destination, const absl::Status& copy_status) {
  // Reuse finalize_load_state to set final state and notify waiters.
  std::string dst_str_async = location_to_string(destination);
  LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Async copy to " << dst_str_async
            << " finished. Operation status: " << copy_status << ". Finalizing state.";
  auto st = finalize_load_state(destination, copy_status);

  // For successful copies, perform additional finalization
  if (copy_status.ok()) {
    // UMA ledger is updated via transactional commit in copy_data_async.
    // Avoid duplicate UMA updates here to keep commit the single source of truth.

    if (destination == MemoryLocation::GPU) {
      absl::MutexLock release_lock(&mutex_);
      // Apply UMA post-GPU-load policy (EvictCPU by default)
      auto uma_policy_st = memory_coordinator_->post_gpu_load_policy(
          replica_key_, artifact_size_, UnifiedMemoryAuthority::PostGpuLoadPolicy::EvictCPU);
      if (!uma_policy_st.ok()) {
        LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id
                     << "): UMA post_gpu_load_policy returned: " << uma_policy_st;
      }
      if (cpu_.state != MemoryState::FAILED) {
        ABSL_CHECK_OK(set_state_locked(MemoryLocation::CPU, MemoryState::UNALLOCATED));
      }
    }
  }
  return st;
}

absl::StatusOr<cudaIpcMemHandle_t> ReplicaLoadController::get_ipc_handle() const noexcept {
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
    return absl::NotFoundError("GpuDeviceMemory object is not initialised");
  }

  return gpu_.cuda_mem->get_handle();
}

// ---------------------------------------------------------------------------
// Peer-to-Peer copy (GPU↔GPU) – experimental implementation
// ---------------------------------------------------------------------------
absl::Status ReplicaLoadController::copy_from_peer(const ReplicaLoadController& source, cudaStream_t ext_stream) {
  // Quick checks – both managers must have GPU memory LOADED.
  if (this == &source) {
    return absl::InvalidArgumentError("Source and destination ReplicaLoadController are identical.");
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

  // Launch async peer copy: same-device uses D2D; cross-device falls back to staged D2H+H2D via pinned slots
  ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::LOADING));
  const int dst_dev_id = get_local_device_id();
  const int src_dev_id = source.get_local_device_id();
  // Wrap pointers after validation for safer use
  gsl::not_null<void*> src_nn{src_ptr};
  gsl::not_null<void*> dst_nn{dst_ptr};

  // Plan UMA transactional session for GPU target on our local device
  auto plan_or =
      memory_coordinator_->plan_load(replica_key_, MemoryLocation::GPU, dst_dev_id, /*chunk_indices=*/std::nullopt);
  if (!plan_or.ok()) {
    ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
    return plan_or.status();
  }
  auto plan = *plan_or;

  if (dst_dev_id == src_dev_id) {
    common::DeviceRegion s{.device_id = dst_dev_id, .dev_ptr = src_nn.get(), .length = static_cast<size_t>(bytes)};
    common::DeviceRegion d{.device_id = dst_dev_id, .dev_ptr = dst_nn.get(), .length = static_cast<size_t>(bytes)};
    auto hdl_or = common::AsyncCopyManager::instance().submit_d2d(s, d, {.tracing_stage = "D2D/Copy"});
    if (!hdl_or.ok()) {
      ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
      return hdl_or.status();
    }
    auto wait_st = hdl_or->wait();
    if (!wait_st.ok()) {
      LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id << "): D2D copy failed: " << wait_st;
      ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
      return wait_st;
    }
  } else {
    // Try peer access first; if not available, fall back to staged copy via pinned host
    int can = 0;
    auto can_st = cuda::device_can_access_peer(&can, dst_dev_id, src_dev_id);
    if (can_st.ok() && can) {
      {
        absl::Status _a = cuda::enable_peer_access(dst_dev_id, src_dev_id);
        if (!_a.ok()) {
          VLOG(1) << "enable_peer_access(dst,src) failed: " << _a;
        }
      }
      {
        absl::Status _b = cuda::enable_peer_access(src_dev_id, dst_dev_id);
        if (!_b.ok()) {
          VLOG(1) << "enable_peer_access(src,dst) failed: " << _b;
        }
      }
      auto st = cuda::memcpy_peer_async(dst_nn.get(), dst_dev_id, src_nn.get(), src_dev_id, bytes, stream_to_use);
      if (!st.ok()) {
        LOG(WARNING) << "Peer copy failed, falling back to staged: " << st.message();
      } else {
        auto sync_st = cuda::stream_synchronize(stream_to_use);
        if (!sync_st.ok()) {
          ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
          return sync_st;
        }
        // Commit UMA ledger for entire artifact (plan carries chunk indices)
        absl::Status cst = memory_coordinator_->commit(
            plan.session_id, MemoryLocation::GPU, absl::MakeSpan(plan.chunk_indices), dst_dev_id);
        if (!cst.ok()) {
          ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
          return cst;
        }
        ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::LOADED));
        // Apply UMA post-GPU-load policy and optionally drop CPU facade state
        {
          auto uma_policy_st = memory_coordinator_->post_gpu_load_policy(
              replica_key_, artifact_size_, UnifiedMemoryAuthority::PostGpuLoadPolicy::EvictCPU);
          if (!uma_policy_st.ok()) {
            LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id
                         << "): UMA post_gpu_load_policy returned: " << uma_policy_st;
          }
          absl::MutexLock lk(&mutex_);
          if (cpu_.state != MemoryState::FAILED) {
            (void)set_state_locked(MemoryLocation::CPU, MemoryState::UNALLOCATED);
          }
        }
        return absl::OkStatus();
      }
    }

    // Staged fallback using StreamingPinnedBuffer
    auto spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
        /*num_chunks=*/4, pinned_pool_->slice_bytes(), pinned_pool_);
    auto init_status = spb->initialize(pinned_memory_timeout_);
    if (!init_status.ok()) {
      ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
      return init_status;
    }
    uint64_t offset = 0;
    while (offset < bytes) {
      size_t step = std::min<size_t>(pinned_pool_->slice_bytes(), static_cast<size_t>(bytes - offset));
      auto slot_or = spb->get_free_chunk();
      if (!slot_or.ok()) {
        ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
        return slot_or.status();
      }
      const int slot_id = *slot_or;
      char* host_ptr = spb->get_chunk_ptr(slot_id);
      if (host_ptr == nullptr) {
        {
          absl::Status _rc = spb->return_chunk(slot_id);
          if (!_rc.ok()) {
            LOG(WARNING) << "SPB return_chunk failed (null host_ptr) slot=" << slot_id << ": " << _rc;
          }
        }
        ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
        return absl::InternalError("SPB returned null chunk pointer");
      }
      // D2H from source device
      common::DeviceRegion s{
          .device_id = src_dev_id, .dev_ptr = static_cast<char*>(src_nn.get()) + offset, .length = step};
      common::HostRegion h{.base = host_ptr, .length = step, .pinned = true};
      auto d2h_hdl = common::AsyncCopyManager::instance().submit_d2h(s, h, {.tracing_stage = "D2H/Copy"});
      if (!d2h_hdl.ok()) {
        {
          absl::Status _rc = spb->return_chunk(slot_id);
          if (!_rc.ok()) {
            LOG(WARNING) << "SPB return_chunk failed after d2h submit error slot=" << slot_id << ": " << _rc;
          }
        }
        ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
        return d2h_hdl.status();
      }
      auto d2h_wait = d2h_hdl->wait();
      if (!d2h_wait.ok()) {
        {
          absl::Status _rc = spb->return_chunk(slot_id);
          if (!_rc.ok()) {
            LOG(WARNING) << "SPB return_chunk failed after d2h wait error slot=" << slot_id << ": " << _rc;
          }
        }
        ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
        return d2h_wait;
      }
      // H2D to destination device
      common::DeviceRegion d{
          .device_id = dst_dev_id, .dev_ptr = static_cast<char*>(dst_nn.get()) + offset, .length = step};
      auto h2d_hdl = common::AsyncCopyManager::instance().submit_h2d(h, d, {.tracing_stage = "H2D/Copy"});
      if (!h2d_hdl.ok()) {
        {
          absl::Status _rc = spb->return_chunk(slot_id);
          if (!_rc.ok()) {
            LOG(WARNING) << "SPB return_chunk failed after h2d submit error slot=" << slot_id << ": " << _rc;
          }
        }
        ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
        return h2d_hdl.status();
      }
      auto h2d_wait = h2d_hdl->wait();
      {
        absl::Status _rc = spb->return_chunk(slot_id);
        if (!_rc.ok()) {
          LOG(WARNING) << "SPB return_chunk failed after h2d wait slot=" << slot_id << ": " << _rc;
        }
      }
      if (!h2d_wait.ok()) {
        ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
        return h2d_wait;
      }
      offset += step;
    }
  }
  // Commit UMA ledger for entire artifact; then finalize LOADED and post policy
  {
    absl::Status cst = memory_coordinator_->commit(
        plan.session_id, MemoryLocation::GPU, absl::MakeSpan(plan.chunk_indices), dst_dev_id);
    if (!cst.ok()) {
      ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::FAILED));
      return cst;
    }
  }
  ABSL_CHECK_OK(set_state(MemoryLocation::GPU, MemoryState::LOADED));
  {
    auto uma_policy_st = memory_coordinator_->post_gpu_load_policy(
        replica_key_, artifact_size_, UnifiedMemoryAuthority::PostGpuLoadPolicy::EvictCPU);
    if (!uma_policy_st.ok()) {
      LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id
                   << "): UMA post_gpu_load_policy returned: " << uma_policy_st;
    }
    absl::MutexLock lk(&mutex_);
    if (cpu_.state != MemoryState::FAILED) {
      (void)set_state_locked(MemoryLocation::CPU, MemoryState::UNALLOCATED);
    }
  }
  return absl::OkStatus();
}

// --- Chunk-scoped export / unexport for P2P access -------------------------

absl::StatusOr<ExportRegistration> ReplicaLoadController::export_chunks_for_p2p(
    MemoryLocation location,
    absl::Span<const uint32_t> chunks,
    communicator::engine::Communicator& comm_engine) {
  absl::MutexLock lock(&mutex_);
  auto info_or = export_service_->export_chunks(replica_key_, location, chunks, comm_engine);
  if (!info_or.ok()) {
    return info_or.status();
  }
  ExportRegistration info = *info_or;
  if (location == MemoryLocation::GPU) {
    gpu_.comm_registration_info = info;
    gpu_.comm_registered = true;
  } else if (location == MemoryLocation::CPU) {
    cpu_.comm_registration_info = info;
    cpu_.comm_registered = true;
  }
  return info;
}

absl::Status ReplicaLoadController::unexport_chunks_for_p2p(
    MemoryLocation location,
    absl::Span<const uint32_t> /*chunks*/,
    communicator::engine::Communicator& comm_engine) {
  absl::MutexLock lock(&mutex_);
  if (location == MemoryLocation::GPU && gpu_.comm_registered) {
    auto st = export_service_->unexport_chunks(replica_key_, gpu_.comm_registration_info, comm_engine);
    if (st.ok()) {
      gpu_.comm_registered = false;
      gpu_.comm_registration_info = {};
    }
    return st;
  }
  if (location == MemoryLocation::CPU && cpu_.comm_registered) {
    auto st = export_service_->unexport_chunks(replica_key_, cpu_.comm_registration_info, comm_engine);
    if (st.ok()) {
      cpu_.comm_registered = false;
      cpu_.comm_registration_info = {};
    }
    return st;
  }
  return absl::OkStatus();
}

// --- VS accessor implementation ---
gsl::not_null<common::memory::VirtualAddressSpace*> ReplicaLoadController::get_va_space() {
  return gsl::not_null<common::memory::VirtualAddressSpace*>{va_space_.get().get()};
}

gsl::not_null<const common::memory::VirtualAddressSpace*> ReplicaLoadController::get_va_space() const {
  return gsl::not_null<const common::memory::VirtualAddressSpace*>{va_space_.get().get()};
}

absl::StatusOr<DirectWriteGrant> ReplicaLoadController::plan_direct_write(absl::Span<const VaRange> ranges) {
  {
    absl::MutexLock lock(&mutex_);
    if (cpu_.state != MemoryState::LOADED && cpu_.state != MemoryState::ALLOCATED) {
      return absl::FailedPreconditionError("CPU memory must be allocated/loaded for direct write");
    }
  }
  return memory_coordinator_->grant_direct_write(replica_key_, ranges);
}

// finalize_load removed — UMA plan/commit is authoritative in final cutover.

// --- VS metadata snapshot ---------------------------------------------------
// Provides a lightweight, read-only view of per-chunk metadata stored inside
// the VirtualAddressSpace (VS).  The span remains valid as long as the
// VS instance itself lives.  Callers must treat the returned ChunkMeta
// objects as immutable and use the atomic accessors defined inside ChunkMeta
// for state inspection.
absl::Span<const ChunkMeta> ReplicaLoadController::chunk_telemetry_snapshot() const noexcept {
  // No expensive operations here – simply delegate to VS (telemetry view).
  return va_space_->chunk_telemetry_snapshot(replica_key_.artifact_id);
}

// --- NEW: Unified Memory Management implementations ---

absl::Status ReplicaLoadController::allocate_replica_memory() {
  absl::MutexLock lock(&mutex_);

  ABSL_CHECK_NE(artifact_size_, 0) << "Artifact size must be set before UMA allocation";
  ABSL_CHECK(!memory_coordinator_->has_allocation(replica_key_)) << "UMA replica allocation already exists";

  // Allocate via unified memory (which will use VS internally)
  auto status = memory_coordinator_->allocate(replica_key_, artifact_size_);
  if (!status.ok()) {
    return status;
  }

  LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Allocated unified memory for "
            << artifact_size_ << " bytes";

  return absl::OkStatus();
}

absl::Status ReplicaLoadController::mark_cpu_preemptible(float ratio) {
  absl::MutexLock lock(&mutex_);

  if (!memory_coordinator_->has_allocation(replica_key_)) {
    return absl::FailedPreconditionError("Unified memory not allocated for this replica");
  }

  if (ratio < 0.0F || ratio > 1.0F) {
    return absl::InvalidArgumentError("Ratio must be between 0.0F and 1.0F");
  }

  // Delegate selection and VS interaction to UMA
  auto status = memory_coordinator_->mark_cpu_chunks_preemptible(replica_key_, ratio);
  if (!status.ok()) {
    return status;
  }

  LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id
            << "): Marked CPU chunks as preemptible via UMA (ratio=" << ratio << ")";

  return absl::OkStatus();
}

std::vector<uint32_t> ReplicaLoadController::get_missing_chunks(MemoryLocation target, std::optional<int> device_id)
    const {
  absl::MutexLock lock(&mutex_);
  ABSL_CHECK(memory_coordinator_->has_allocation(replica_key_)) << "UMA allocation not found";

  return memory_coordinator_->get_missing_chunks(replica_key_, target, device_id);
}

// build_sink_ and range utilities moved to TransferService

std::future<absl::Status> ReplicaLoadController::load_async_from_source(
    std::unique_ptr<loader::SeekableSource> source,
    MemoryLocation target_location,
    int concurrency,
    std::optional<absl::Span<const uint32_t>> chunk_indices,
    std::function<absl::Status()> post_load_fn) {
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
    absl::Status _st = set_state_locked(target_location, MemoryState::LOADING);
    if (!_st.ok()) {
      LOG(WARNING) << "ensure_loaded_async: failed to set state to LOADING: " << _st;
    }
  }

  // Phase 2: launch async pump task delegated to TransferService
  return std::async(
      std::launch::async,
      [this,
       source = std::move(source),
       target_location,
       concurrency,
       chunk_indices,
       post_load_fn = std::move(post_load_fn)]() mutable -> absl::Status {
        LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id
                  << "): async load start; target=" << static_cast<int>(target_location) << ", conc=" << concurrency;
        void* gpu_ptr = nullptr;
        std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation;
        int device_id = get_local_device_id();
        if (target_location == MemoryLocation::GPU) {
          auto vec = this->get_pointer(MemoryLocation::GPU);
          if (vec.empty()) {
            this->record_failure_and_fail_(MemoryLocation::GPU, "GPU memory not allocated");
            return absl::FailedPreconditionError("GPU memory not allocated");
          }
          gpu_ptr = vec[0];
          gpu_allocation = this->get_gpu_allocation_shared();
          if (!gpu_allocation) {
            this->record_failure_and_fail_(MemoryLocation::GPU, "GPU allocation handle missing");
            return absl::FailedPreconditionError("GPU allocation handle missing");
          }
        }

        // Plan → Execute → Commit path (final)
        auto plan_or = memory_coordinator_->plan_load(replica_key_, target_location, device_id, chunk_indices);
        if (!plan_or.ok()) {
          this->record_failure_and_fail_(
              target_location, absl::Substitute("UMA plan_load failed: $0", plan_or.status().message()));
          return plan_or.status();
        }
        LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id << "): UMA plan_load ok; launching execute";
        auto plan = std::move(*plan_or);
        absl::Status exec_status =
            transfer_service_->execute(plan, target_location, *source, concurrency, gpu_ptr, gpu_allocation, device_id);
        if (!exec_status.ok()) {
          // Abort session on failure (idempotent)
          auto _ = memory_coordinator_->abort(plan.session_id);
          this->record_failure_and_fail_(
              target_location, absl::Substitute("transfer execute failed: $0", exec_status.message()));
          return exec_status;
        }
        // Commit UMA ledger
        LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id << "): execute ok; committing UMA";
        absl::Status cst = memory_coordinator_->commit(
            plan.session_id, target_location, absl::MakeSpan(plan.chunk_indices), device_id);
        if (!cst.ok()) {
          this->record_failure_and_fail_(target_location, absl::Substitute("UMA commit failed: $0", cst.message()));
          return cst;
        }
        // Post policy for GPU; for CPU do nothing here
        if (target_location == MemoryLocation::GPU) {
          auto uma_policy_st = memory_coordinator_->post_gpu_load_policy(
              replica_key_, artifact_size_, UnifiedMemoryAuthority::PostGpuLoadPolicy::EvictCPU);
          if (!uma_policy_st.ok()) {
            LOG(WARNING) << "ReplicaLoadController(" << replica_key_.artifact_id
                         << "): UMA post_gpu_load_policy returned: " << uma_policy_st;
          }
          // Optionally mark CPU state UNALLOCATED in memory manager facade
          {
            absl::MutexLock lk(&mutex_);
            if (cpu_.state != MemoryState::FAILED) {
              (void)set_state_locked(MemoryLocation::CPU, MemoryState::UNALLOCATED);
            }
          }
        }
        // Optional post-load hook (e.g., view transforms)
        if (post_load_fn) {
          absl::Status hook_status = post_load_fn();
          if (!hook_status.ok()) {
            this->record_failure_and_fail_(
                target_location, absl::Substitute("post-load transform failed: $0", hook_status.message()));
            return hook_status;
          }
        }
        // Facade state LOADED
        LOG(INFO) << "ReplicaLoadController(" << replica_key_.artifact_id << "): setting final LOADED state";
        return this->set_state(target_location, MemoryState::LOADED);
      });
}

std::vector<replica::ChunkState> ReplicaLoadController::get_chunk_states_uma(
    common::memory::MemoryLocation location,
    std::optional<int> device_id) const {
  std::vector<replica::ChunkState> out;
  // Determine number of chunks from UMA layout
  auto sz_or = memory_coordinator_->get_artifact_size(replica_key_);
  if (!sz_or.ok()) {
    return out;
  }
  const size_t total_bytes = *sz_or;
  const size_t chunk_size = memory_coordinator_->get_artifact_chunk_bytes();
  if (chunk_size == 0) {
    return out;
  }
  const size_t num_chunks = (total_bytes + chunk_size - 1) / chunk_size;
  out.reserve(num_chunks);

  for (uint32_t i = 0; i < num_chunks; ++i) {
    absl::StatusOr<replica::ChunkState> st_or = absl::UnavailableError("");
    if (location == common::memory::MemoryLocation::GPU) {
      const int dev = device_id.has_value() ? *device_id : replica_key_.device.ordinal;
      st_or = memory_coordinator_->get_gpu_chunk_state(replica_key_, dev, i);
    } else {
      st_or = memory_coordinator_->get_cpu_chunk_state(replica_key_, i);
    }
    if (st_or.ok()) {
      out.push_back(*st_or);
    } else {
      // Default to EVICTED when UMA has no explicit state for this chunk/device.
      out.push_back(replica::ChunkState::EVICTED);
    }
  }
  return out;
}

absl::Status ReplicaLoadController::ensure_gpu_stream_initialized_locked_() {
  // Expects mutex_ held by caller
  if (gpu_.stream_initialized && gpu_.stream != nullptr) {
    return absl::OkStatus();
  }
  if (replica_key_.device.ordinal < 0) {
    return absl::FailedPreconditionError("Local CUDA device id not set");
  }
  auto dev_st = cuda::set_device(replica_key_.device.ordinal);
  if (!dev_st.ok()) {
    LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id
               << "): Failed to set CUDA device in ensure_gpu_stream_initialized_locked_: " << dev_st;
    return dev_st;
  }
  cudaStream_t created = nullptr;
  auto stream_st = cuda::stream_create_with_flags(&created, cudaStreamNonBlocking);
  if (!stream_st.ok()) {
    LOG(ERROR) << "ReplicaLoadController(" << replica_key_.artifact_id
               << "): Failed to create CUDA stream in ensure_gpu_stream_initialized_locked_: " << stream_st;
    return stream_st;
  }
  gpu_.stream = created;
  gpu_.stream_initialized = true;
  VLOG(2) << "ReplicaLoadController(" << replica_key_.artifact_id << "): Created CUDA stream " << gpu_.stream
          << " on device " << replica_key_.device.ordinal << ".";
  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
