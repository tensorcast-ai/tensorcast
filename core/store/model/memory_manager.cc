// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/model/memory_manager.h"
#include "core/common/trace/trace_macros.h"
#include "core/store/model/model_location.h"
#include "core/store/model/transfer_helpers.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "core/common/cuda_api.h"
#include "core/common/device_types.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/common/metrics/metric_objects.h"
#include "core/communicator/engine/engine.h"
#include "core/store/direct_write.h"

// Remove bridge macros; use structured members directly.
// instance_key_.model_id -> instance_key_.model_id
// instance_key_.device.ordinal -> instance_key_.device.ordinal

namespace stepcast::store {

MemoryManager::MemoryManager(
    std::string model_identifier,
    int local_device_id,
    const gsl::not_null<std::shared_ptr<PinnedMemoryPool>>& pinned_pool,
    const gsl::not_null<std::shared_ptr<memory::DistributedVirtualMemoryPool>>& dvmp,
    size_t max_buffer_bytes,
    std::chrono::milliseconds pinned_memory_timeout)
    : pinned_pool_(pinned_pool),
      max_buffer_bytes_(max_buffer_bytes),
      pinned_memory_timeout_(pinned_memory_timeout),
      dvmp_(dvmp),
      memory_coordinator_(std::make_shared<ModelMemoryCoordinator>(dvmp)) {
  // Populate instance_key_ using constructor inputs
  instance_key_.model_id = std::move(model_identifier);
  instance_key_.device.type = (local_device_id >= 0) ? DeviceType::GPU : DeviceType::CPU;
  instance_key_.device.ordinal = local_device_id;

  {
    absl::MutexLock lock(&mutex_);
    // Initialize states properly based on whether pools are provided
    cpu_.state = MemoryState::UNALLOCATED;
    // GPU state depends on pool or potential borrowing later
    gpu_.state = MemoryState::UNALLOCATED; // Assume potential for allocation/borrowing
  }

  // Initialise CUDA context and non-blocking stream if a valid device id was provided at construction.
  if (instance_key_.device.ordinal >= 0) {
    auto dev_st = cuda::set_device(instance_key_.device.ordinal);
    if (!dev_st.ok()) {
      LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                 << "): Failed to set CUDA device during construction: " << dev_st;
    } else {
      auto stream_st = cuda::stream_create_with_flags(&gpu_.stream, cudaStreamNonBlocking);
      if (!stream_st.ok()) {
        LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                   << "): Failed to create CUDA stream during construction: " << stream_st;
      } else {
        gpu_.stream_initialized = true;
        VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Created CUDA stream " << gpu_.stream
                << " on device " << instance_key_.device.ordinal << ".";
      }
    }
  }
}

MemoryManager::~MemoryManager() {
  cudaStream_t local_stream = nullptr;
  std::string id_copy; // Copy identifier for logging after potential lock release
  {
    absl::MutexLock lock(&mutex_);
    id_copy = instance_key_.model_id; // Copy identifier while lock is held
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
    release_cpu_resources_locked();
    release_gpu_resources_locked();
    VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Destructor finished.";
  }
}

void MemoryManager::set_model_size(uint64_t size) {
  absl::MutexLock lock(&mutex_);
  if (model_size_ > 0 && model_size_ != size) {
    LOG(WARNING) << "MemoryManager(" << instance_key_.model_id << "): Model size being reset from " << model_size_
                 << " to " << size;
    if (cpu_.state >= MemoryState::ALLOCATED || gpu_.state >= MemoryState::ALLOCATED) {
      LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                 << "): Cannot change model size after memory allocation/borrowing. Current PAGEABLE_CPU state: "
                 << state_to_string(cpu_.state) << ", GPU state: " << state_to_string(gpu_.state);
      // Optionally return error status here if function signature allowed
      return;
    }
  }
  model_size_ = size;
  VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Model size set to " << model_size_ << " bytes.";
}

uint64_t MemoryManager::get_model_size() const {
  absl::MutexLock lock(&mutex_);
  return model_size_;
}

int MemoryManager::get_local_device_id() const {
  return instance_key_.device.ordinal;
}

absl::Status MemoryManager::allocate_memory(ModelLocation location) {
  absl::MutexLock lock(&mutex_);
  if (model_size_ == 0) {
    return absl::FailedPreconditionError(
        absl::StrFormat("MemoryManager(%s): Model size not set before allocation.", instance_key_.model_id));
  }
  // Only GPU allocations require a valid CUDA stream.  PAGEABLE_CPU allocations rely solely on
  // pinned host memory and therefore do not depend on CUDA stream initialisation.
  if (location == ModelLocation::GPU && !gpu_.stream_initialized) {
    // Try to lazily initialise stream under lock for resilience
    auto stream_st = ensure_gpu_stream_initialized_locked_();
    if (!stream_st.ok()) {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "MemoryManager(%s): CUDA stream not initialised and lazy init failed.", instance_key_.model_id));
    }
  }

  MemoryState* state_ptr = nullptr;
  std::string loc_str = location_to_string(location);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU: {
      state_ptr = &cpu_.state;
      if (*state_ptr >= MemoryState::ALLOCATED) {
        VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): PAGEABLE_CPU already in state "
                << state_to_string(*state_ptr) << ". Allocation request ignored.";
        return absl::OkStatus();
      }
      if (*state_ptr != MemoryState::UNALLOCATED) {
        return absl::FailedPreconditionError(
            absl::StrFormat(
                "MemoryManager(%s): Cannot allocate PAGEABLE_CPU memory. Expected UNALLOCATED state, but found %s.",
                instance_key_.model_id,
                state_to_string(*state_ptr)));
      }

      auto region_or = reserve_dvmp_region_locked_();
      if (!region_or.ok()) {
        ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
        return region_or.status();
      }

      // Allocate a streaming buffer pool instead of full pinned memory.
      size_t chunk_size = pinned_pool_->chunk_size();
      // Compute the maximum chunks allowed by the buffer cap
      size_t max_chunks_cap = std::max<size_t>(1, max_buffer_bytes_ / chunk_size);
      // Compute how many chunks are actually needed for this model
      auto required_chunks = static_cast<size_t>((model_size_ + chunk_size - 1) / chunk_size);
      size_t num_chunks = std::min(max_chunks_cap, std::max<size_t>(1, required_chunks));
      size_t actual_buffer_bytes = num_chunks * chunk_size;

      // Allocate buffer under lock; alignment policy is enforced in ensure_streaming_buffer for external callers
      ABSL_CHECK_OK(allocate_buffer_pool(num_chunks));

      LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Allocated streaming pinned buffer with "
                << num_chunks << " chunks (chunk_size=" << chunk_size << ", total=" << actual_buffer_bytes
                << " bytes) for PAGEABLE_CPU staging.";
      break;
    }
    case ModelLocation::GPU: {
      state_ptr = &gpu_.state;
      if (*state_ptr >= MemoryState::ALLOCATED) {
        VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): GPU memory already in state "
                << state_to_string(*state_ptr) << ". Allocation request ignored.";
        return absl::OkStatus(); // Already allocated
      }
      if (*state_ptr != MemoryState::UNALLOCATED) {
        return absl::FailedPreconditionError(
            absl::StrFormat(
                "MemoryManager(%s): Cannot allocate GPU memory. Expected UNALLOCATED state, but found %s.",
                instance_key_.model_id,
                state_to_string(*state_ptr)));
      }

      // Ensure UMA is initialized first
      if (!memory_coordinator_->has_allocation(instance_key_)) {
        VLOG(1) << "MemoryManager(" << instance_key_.model_id
                << "): UMA has no allocation for this model, calling allocate_model_memory()";
        // Need to release lock before calling allocate_model_memory to avoid deadlock
        mutex_.Unlock();
        auto uma_status = allocate_model_memory();
        mutex_.Lock();
        if (!uma_status.ok() && uma_status.code() != absl::StatusCode::kAlreadyExists) {
          LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                     << "): Failed to allocate UMA model state: " << uma_status.message();
          ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
          return absl::FailedPreconditionError(
              absl::StrFormat(
                  "MemoryManager(%s): Failed UMA allocation for GPU allocation: %s",
                  instance_key_.model_id,
                  uma_status.message()));
        }
      }

      // Trace allocation of GPU memory.
      SC_TRACE_SCOPE("allocate_gpu_memory");

      // Route GPU allocation through UMA (sole authority for VRAM allocation)
      VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Requesting GPU allocation from UMA for "
              << model_size_ << " bytes on device " << instance_key_.device.ordinal << ".";
      auto gpu_alloc_result =
          memory_coordinator_->get_or_create_gpu_allocation(instance_key_, instance_key_.device.ordinal);
      if (!gpu_alloc_result.ok()) {
        LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                   << "): UMA GPU allocation failed: " << gpu_alloc_result.status().message();
        ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
        return absl::ResourceExhaustedError(
            absl::StrFormat(
                "MemoryManager(%s): Failed UMA GPU allocation on device %d: %s",
                instance_key_.model_id,
                instance_key_.device.ordinal,
                gpu_alloc_result.status().message()));
      }

      // Cache the shared_ptr from UMA (UMA owns lifetime)
      gpu_.cuda_mem = *gpu_alloc_result;
      VLOG(1) << "MemoryManager(" << instance_key_.model_id
              << "): UMA GPU allocation successful (ptr=" << gpu_.cuda_mem->get() << ").";
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("MemoryManager(%s): Invalid location for allocation: %s", instance_key_.model_id, loc_str));
  }

  // If allocation successful, set state to ALLOCATED
  return set_state_locked(location, MemoryState::ALLOCATED);
}

absl::Status MemoryManager::release_memory(ModelLocation location, bool safe_release) {
  absl::MutexLock lock(&mutex_);

  MemoryState* state_ptr = nullptr;
  absl::CondVar* cond_ptr = nullptr;
  std::string loc_str = location_to_string(location);
  auto map_status = get_state_and_cond_locked_(location, &state_ptr, &cond_ptr);
  if (!map_status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MemoryManager(%s): Invalid location for release: %s", instance_key_.model_id, loc_str));
  }

  // PAGEABLE_CPU has special handling (does not free DVMP region)
  if (location == ModelLocation::PAGEABLE_CPU) {
    VLOG(2) << "MemoryManager(" << instance_key_.model_id
            << "): release_memory called for PAGEABLE_CPU (safe_release=" << safe_release << ")";

    MemoryState current_state = *state_ptr;
    if (current_state == MemoryState::LOADING && safe_release) {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "MemoryManager(%s): Cannot safely release PAGEABLE_CPU while LOADING.", instance_key_.model_id));
    }

    if (current_state == MemoryState::LOADING && !safe_release) {
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
    } else if (current_state != MemoryState::UNALLOCATED) {
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::UNALLOCATED));
    }

    // Release staging resources (streaming buffer, host queues). DVMP region remains reserved.
    release_cpu_resources_locked();
    return absl::OkStatus();
  }

  MemoryState current_state = *state_ptr;
  VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Requesting release for " << loc_str
          << " (current state: " << state_to_string(current_state) << ", safe_release: " << safe_release << ")";

  if (current_state <= MemoryState::UNALLOCATED) {
    VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Memory for " << loc_str
            << " already released or uninitialized. No action taken.";
    return absl::OkStatus();
  }

  if (current_state == MemoryState::LOADING) {
    if (safe_release) {
      LOG(WARNING) << "MemoryManager(" << instance_key_.model_id << "): Safe release requested for " << loc_str
                   << " while LOADING. Release denied.";
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "MemoryManager(%s): Cannot safely release %s memory while in LOADING state.",
              instance_key_.model_id,
              loc_str));
    }

    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Unsafe release requested for " << loc_str
            << " while LOADING. Attempting to wait briefly...";
    if (cond_ptr->WaitWithTimeout(&mutex_, absl::Milliseconds(500))) {
      LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                 << "): Timeout expired while waiting for LOADING state on " << loc_str
                 << " to resolve during unsafe release.";
    }

    current_state = *state_ptr;
    if (current_state == MemoryState::LOADING) {
      LOG(ERROR) << "MemoryManager(" << instance_key_.model_id << "): Force releasing " << loc_str
                 << " memory while still LOADING after wait. Setting state to FAILED. Potential resource issues.";
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
    } else {
      VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): State for " << loc_str << " changed to "
              << state_to_string(current_state) << " during wait. Proceeding with release.";
    }
  }

  // Proceed with GPU resource release
  release_gpu_resources_locked();

  // Clear communication registration if releasing the registered GPU location
  if (location == ModelLocation::GPU && gpu_.comm_registered) {
    VLOG(2) << "MemoryManager(" << instance_key_.model_id
            << "): Clearing GPU communication registration as memory is being released.";
    gpu_.comm_registered = false;
  }

  if (*state_ptr != MemoryState::FAILED) {
    ABSL_CHECK_OK(set_state_locked(location, MemoryState::UNALLOCATED));
  } else {
    LOG(WARNING) << "MemoryManager(" << instance_key_.model_id << "): Resources for " << loc_str
                 << " released, but state remains FAILED due to earlier unsafe release during LOADING.";
  }

  VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Finished release process for " << loc_str
          << ". Final state: " << state_to_string(*state_ptr);
  return absl::OkStatus();
}

// Private helper
void MemoryManager::release_cpu_resources_locked() {
  // Called with mutex held
  // Release streaming buffer first (if allocated)
  if (cpu_.streaming_buffer) {
    auto status = cpu_.streaming_buffer->release();
    if (!status.ok()) {
      LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                   << "): Streaming buffer release returned error: " << status;
    }
    cpu_.streaming_buffer.reset();
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Streaming buffer released/reset.";
  }

  // No full PinnedMemory allocation anymore – streaming buffer is the only pinned memory user.
}

// Private helper
void MemoryManager::release_gpu_resources_locked() {
  // Called with mutex held
  if (gpu_.cuda_mem) {
    VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Releasing allocated GPU memory object (pointer "
            << gpu_.cuda_mem->get() << " will be freed/returned to pool by CudaMemory dtor).";
    // UMA now manages GPU memory lifecycle
    gpu_.cuda_mem.reset();
  }
}

MemoryState MemoryManager::get_state(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);
  MemoryState* state_ptr = nullptr;
  absl::CondVar* cond_ptr = nullptr;
  auto st = const_cast<MemoryManager*>(this)->get_state_and_cond_locked_(location, &state_ptr, &cond_ptr);
  if (!st.ok()) {
    LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                 << "): get_state called with invalid location: " << static_cast<int>(location);
    return MemoryState::UNINITIALIZED;
  }
  return *state_ptr;
}

std::vector<void*> MemoryManager::get_pointer(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);
  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      // Return pointer to first chunk if loaded, otherwise null.
      // Documentation should clarify this behaviour.
      if (cpu_.state == MemoryState::LOADED || cpu_.state == MemoryState::ALLOCATED) {
        // Always use UMA as single source of truth for CPU base pointer
        void* base_ptr = nullptr;
        base_ptr = memory_coordinator_->get_cpu_base_ptr(instance_key_);
        if (base_ptr == nullptr) {
          VLOG(1) << "MemoryManager(" << instance_key_.model_id
                  << "): CPU base is null despite ALLOCATED/LOADED state.";
          return {};
        }
        return std::vector<void*>({base_ptr});
      }
      VLOG(2) << "MemoryManager(" << instance_key_.model_id
              << "): get_pointer(PAGEABLE_CPU) returning null. State: " << state_to_string(cpu_.state)
              << ", StreamingBuffer valid: " << (cpu_.streaming_buffer != nullptr);
      return {};
    case ModelLocation::GPU:
      // GPU memory is now managed by UMA - return pointer if allocated
      if (gpu_.state >= MemoryState::ALLOCATED && gpu_.state != MemoryState::FAILED) {
        // UMA guarantees cuda_mem is valid if state is ALLOCATED or higher
        if (gpu_.cuda_mem) {
          return std::vector<void*>({gpu_.cuda_mem->get()});
        } else {
          // Fallback to UMA's GPU allocation if local cache is not set
          auto gpu_alloc = memory_coordinator_->get_gpu_base_ptr(instance_key_, instance_key_.device.ordinal);
          if (gpu_alloc) {
            return std::vector<void*>({gpu_alloc});
          }
        }
      }
      VLOG(2) << "MemoryManager(" << instance_key_.model_id
              << "): get_pointer(GPU) returning empty. State: " << state_to_string(gpu_.state);
      return {};
    default:
      LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                   << "): get_pointer called with invalid location: " << static_cast<int>(location);
      return {};
  }
}

// Public thread‑safe wrapper
absl::Status MemoryManager::set_state(ModelLocation location, MemoryState new_state) {
  absl::MutexLock lock(&mutex_);
  return set_state_locked(location, new_state);
}

// Internal helper to fetch state and cond pointers (expects mutex_ held)
absl::Status MemoryManager::get_state_and_cond_locked_(
    ModelLocation location,
    MemoryState** state_ptr,
    absl::CondVar** cond_ptr) {
  if (state_ptr == nullptr || cond_ptr == nullptr) {
    return absl::InvalidArgumentError("Null output in get_state_and_cond_locked_");
  }
  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      *state_ptr = &cpu_.state;
      *cond_ptr = &cpu_.cond;
      return absl::OkStatus();
    case ModelLocation::GPU:
      *state_ptr = &gpu_.state;
      *cond_ptr = &gpu_.cond;
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError("Invalid location");
  }
}

// Internal helper (expects mutex_ held)
absl::Status MemoryManager::set_state_locked(ModelLocation location, MemoryState new_state) {
  MemoryState* state_ptr = nullptr;
  absl::CondVar* cond_ptr = nullptr;
  auto map_status = get_state_and_cond_locked_(location, &state_ptr, &cond_ptr);
  if (!map_status.ok()) {
    LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
               << "): Invalid location for set_state_locked: " << location_to_string(location);
    return map_status;
  }

  MemoryState old_state = *state_ptr;
  if (old_state == new_state) {
    VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): State for " << location_to_string(location)
            << " already " << state_to_string(new_state) << ". No change.";
    return absl::OkStatus();
  }

  log_state_change(location, old_state, new_state);
  *state_ptr = new_state;
  cond_ptr->SignalAll();
  return absl::OkStatus();
}

void MemoryManager::log_state_change(ModelLocation loc, MemoryState old_state, MemoryState new_state) const {
  // Assumes mutex is held
  VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): " << location_to_string(loc) << " state changing from "
          << state_to_string(old_state) << " to " << state_to_string(new_state);
}

std::future<absl::Status> MemoryManager::copy_data_async(ModelLocation source, ModelLocation destination) {
  // Phase 1: capture params and mark destination LOADING
  CopyLaunchParams params;
  bool need_allocate_um = false;
  auto cap_status = capture_copy_context_(source, destination, &params, &need_allocate_um);
  if (!cap_status.ok()) {
    return std::async(std::launch::deferred, [cap_status] { return cap_status; });
  }

  // Allocate UMA if needed (avoid calling while holding mutex_)
  if (need_allocate_um) {
    auto st_um = allocate_model_memory();
    if (!st_um.ok()) {
      return std::async(std::launch::deferred, [st_um] { return st_um; });
    }
  }

  // Phase 2: launch async copy task
  return std::async(std::launch::async, [this, source, destination, p = std::move(params)]() -> absl::Status {
    absl::Status copy_status;
    const bool src_host_async = (source == ModelLocation::PAGEABLE_CPU);
    const bool dst_host_async = (destination == ModelLocation::PAGEABLE_CPU);

    if (src_host_async && !dst_host_async) { // Host -> GPU
      auto uma_capture = this->get_memory_coordinator();
      copy_status = perform_copy_cpu_to_gpu_streaming(
          p.model_id,
          p.device_id,
          p.streaming_buffer,
          p.cuda_mem ? p.cuda_mem->get() : nullptr,
          p.total_size,
          p.stream,
          p.dvmp_base,
          p.dvmp,
          uma_capture,
          this->instance_key_);
    } else if (!src_host_async && dst_host_async) { // GPU -> Host
      copy_status = perform_copy_gpu_to_cpu_streaming(
          p.model_id,
          p.device_id,
          p.streaming_buffer,
          p.cuda_mem ? p.cuda_mem->get() : nullptr,
          p.total_size,
          p.stream,
          p.dvmp_base,
          p.dvmp);
    } else {
      LOG(ERROR) << "MemoryManager(" << p.model_id << "): Unsupported copy direction: " << static_cast<int>(source)
                 << " -> " << static_cast<int>(destination);
      copy_status = absl::InvalidArgumentError("Unsupported copy direction in async task.");
    }

    // Phase 3: finalize state and cleanup as needed
    return this->finalize_copy_state_(destination, copy_status);
  });
}

absl::Status MemoryManager::wait_for_state(ModelLocation location, MemoryState target_state, absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);

  MemoryState* state_ptr = nullptr;
  absl::CondVar* cond_ptr = nullptr;
  std::string loc_str = location_to_string(location);
  auto map_status = get_state_and_cond_locked_(location, &state_ptr, &cond_ptr);
  if (!map_status.ok()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("MemoryManager(%s): Invalid location for wait_for_state: %s", instance_key_.model_id, loc_str));
  }

  VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Waiting for " << loc_str << " to reach state "
          << state_to_string(target_state) << " (current: " << state_to_string(*state_ptr) << ", timeout: " << timeout
          << ")";

  absl::Time deadline = (timeout == absl::InfiniteDuration()) ? absl::InfiniteFuture() : absl::Now() + timeout;

  while (*state_ptr != target_state && *state_ptr != MemoryState::FAILED) {
    if (absl::Now() >= deadline) {
      if (*state_ptr != target_state && *state_ptr != MemoryState::FAILED) {
        LOG(WARNING) << "MemoryManager(" << instance_key_.model_id << "): Timeout waiting for " << loc_str
                     << " to reach state " << state_to_string(target_state)
                     << ". Current state: " << state_to_string(*state_ptr);
        return absl::DeadlineExceededError(
            absl::StrFormat("Timeout waiting for %s state %s", loc_str, state_to_string(target_state)));
      }
      break;
    }
    cond_ptr->WaitWithDeadline(&mutex_, deadline);
  }

  if (*state_ptr == target_state) {
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Wait successful. " << loc_str
            << " reached target state " << state_to_string(target_state);
    return absl::OkStatus();
  }
  if (*state_ptr == MemoryState::FAILED) {
    LOG(ERROR) << "MemoryManager(" << instance_key_.model_id << "): Wait completed because " << loc_str
               << " reached FAILED state while waiting for " << state_to_string(target_state);
    return absl::FailedPreconditionError(absl::StrFormat("%s operation failed", loc_str));
  }
  LOG(ERROR) << "MemoryManager(" << instance_key_.model_id << "): Wait loop exited with unexpected state "
             << state_to_string(*state_ptr) << " for " << loc_str;
  return absl::InternalError("Unexpected state after wait loop.");
}

absl::Status MemoryManager::finalize_load_state(ModelLocation location, const absl::Status& final_status) {
  absl::MutexLock lock(&mutex_);

  MemoryState* state_ptr = nullptr;
  std::string loc_str = location_to_string(location);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      state_ptr = &cpu_.state;
      break;
    case ModelLocation::GPU:
      state_ptr = &gpu_.state;
      break;
    default:
      LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                 << "): Invalid location for finalize_load_state: " << loc_str;
      return absl::InvalidArgumentError("Invalid location for finalize_load_state");
  }

  MemoryState current_state = *state_ptr;

  if (current_state == MemoryState::LOADING) {
    MemoryState target_final_state = final_status.ok() ? MemoryState::LOADED : MemoryState::FAILED;
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Finalizing operation for " << loc_str
            << ". Operation status: " << final_status << ". Setting state from LOADING to "
            << state_to_string(target_final_state);
    // Use set_state_locked to update state and notify condition variables
    return set_state_locked(location, target_final_state); // Already under lock
  } // This is not necessarily an error, the state might have been changed by release_memory or another operation.
  LOG(WARNING) << "MemoryManager(" << instance_key_.model_id << "): Finalize requested for " << loc_str
               << ", but state was not LOADING (current: " << state_to_string(current_state)
               << "). State not updated. Operation status was: " << final_status;
  // Return OkStatus because the finalization logic itself didn't fail, even if no state change occurred.
  // The caller should primarily rely on the future's status (which is final_status).
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Refactor helpers: capture and finalize copy state (no behavior change)
// ---------------------------------------------------------------------------

absl::Status MemoryManager::capture_copy_context_(
    ModelLocation source,
    ModelLocation destination,
    CopyLaunchParams* out,
    bool* need_allocate_um) {
  if (!out || !need_allocate_um) {
    return absl::InvalidArgumentError("Null output pointers for capture_copy_context_");
  }

  const std::string src_str = location_to_string(source);
  const std::string dst_str = location_to_string(destination);

  absl::MutexLock lock(&mutex_);

  if (!gpu_.stream_initialized || gpu_.stream == nullptr) {
    LOG(ERROR) << "MemoryManager(" << instance_key_.model_id << "): Cannot initiate copy. CUDA stream is not valid.";
    return absl::InternalError("CUDA stream not initialized.");
  }

  const bool src_is_host = (source == ModelLocation::PAGEABLE_CPU);
  const bool dst_is_host = (destination == ModelLocation::PAGEABLE_CPU);

  MemoryState src_state = src_is_host ? cpu_.state : gpu_.state;
  MemoryState dst_state = dst_is_host ? cpu_.state : gpu_.state;

  LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Requesting async copy from " << src_str
            << " (state: " << state_to_string(src_state) << ") to " << dst_str
            << " (state: " << state_to_string(dst_state) << ")";

  // Validate states
  if (src_state != MemoryState::LOADED) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "MemoryManager(%s): Source %s is not in LOADED state for copy.", instance_key_.model_id, src_str));
  }
  if (dst_state != MemoryState::ALLOCATED) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "MemoryManager(%s): Destination %s is not in ALLOCATED state for copy.", instance_key_.model_id, dst_str));
  }

  // Validate buffers
  if (src_is_host || dst_is_host) {
    ABSL_CHECK(cpu_.streaming_buffer) << "StreamingPinnedBuffer must be allocated before host↔device copy operations.";
    out->streaming_buffer = cpu_.streaming_buffer;
  }
  // UMA now manages GPU memory, so we only need to check if cuda_mem exists
  if ((source == ModelLocation::GPU || destination == ModelLocation::GPU) && !gpu_.cuda_mem) {
    return absl::InternalError("GPU memory not allocated through UMA");
  }

  out->cuda_mem = gpu_.cuda_mem;
  if (src_is_host || dst_is_host) {
    out->dvmp = dvmp_;
    // Get DVMP base from UMA (single source of truth)
    out->dvmp_base = memory_coordinator_->get_cpu_base_ptr(instance_key_);
  }
  out->total_size = model_size_;
  out->stream = gpu_.stream;
  out->device_id = instance_key_.device.ordinal;
  out->model_id = instance_key_.model_id;

  *need_allocate_um = (src_is_host || dst_is_host) && !memory_coordinator_->has_allocation(instance_key_);

  // Mark destination as LOADING
  absl::Status st = set_state_locked(destination, MemoryState::LOADING);
  if (!st.ok()) {
    LOG(ERROR) << "MemoryManager(" << instance_key_.model_id << "): Failed to set destination " << dst_str
               << " state to LOADING: " << st;
  }
  return st;
}

absl::Status MemoryManager::finalize_copy_state_(ModelLocation destination, const absl::Status& copy_status) {
  // Reuse finalize_load_state to set final state and notify waiters.
  std::string dst_str_async = location_to_string(destination);
  LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Async copy to " << dst_str_async
            << " finished. Operation status: " << copy_status << ". Finalizing state.";
  auto st = finalize_load_state(destination, copy_status);

  // For successful copies, perform additional finalization
  if (copy_status.ok()) {
    // For GPU->CPU copies, sync UMA from DVMP
    if (destination == ModelLocation::PAGEABLE_CPU) {
      // Call finalize_load to sync UMA's CPU view from DVMP
      // No specific chunk indices - sync all chunks that were written
      auto uma_sync_status = finalize_load(destination);
      if (!uma_sync_status.ok()) {
        LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                     << "): UMA sync after GPU->CPU copy failed: " << uma_sync_status;
      }
    }

    // Mandatory CPU memory release after successful CPU→GPU copy (per RFC 0001 §4.3)
    if (destination == ModelLocation::GPU) {
      absl::MutexLock release_lock(&mutex_);
      // Get CPU base from UMA (single source of truth)
      void* cpu_base = memory_coordinator_->get_cpu_base_ptr(instance_key_);
      if (cpu_base != nullptr) {
        size_t evicted = dvmp_->evict_tail_bytes(instance_key_.model_id, model_size_);
        LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Evicted " << evicted
                  << " bytes from DVMP after GPU copy.";
      }
      release_cpu_resources_locked();
      // Do not override a FAILED state set elsewhere.
      if (cpu_.state != MemoryState::FAILED) {
        ABSL_CHECK_OK(set_state_locked(ModelLocation::PAGEABLE_CPU, MemoryState::UNALLOCATED));
      }
    }
  }
  return st;
}

// --- CPU Chunk Size Getter ---
size_t MemoryManager::get_cpu_chunk_size() const {
  absl::MutexLock lock(&mutex_);
  if (!cpu_.streaming_buffer) {
    return 0;
  }
  return cpu_.streaming_buffer->chunk_size();
}

absl::StatusOr<cudaIpcMemHandle_t> MemoryManager::get_cuda_ipc_handle() const {
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
// Streaming buffer helper implementations
// ---------------------------------------------------------------------------
absl::Status MemoryManager::allocate_buffer_pool(size_t num_chunks) {
  if (cpu_.streaming_buffer) {
    return absl::AlreadyExistsError("Streaming buffer already allocated");
  }
  if (num_chunks == 0) {
    return absl::InvalidArgumentError("num_chunks must be > 0 for allocate_buffer_pool");
  }
  size_t chunk_size = pinned_pool_->chunk_size();
  cpu_.streaming_buffer = std::make_shared<StreamingPinnedBuffer>(num_chunks, chunk_size, pinned_pool_);
  absl::Status st = cpu_.streaming_buffer->initialize(pinned_memory_timeout_);
  if (!st.ok()) {
    cpu_.streaming_buffer.reset();
    return st;
  }
  VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Allocated streaming buffer with " << num_chunks
          << " chunks (chunk_size=" << chunk_size << ")";
  return absl::OkStatus();
}

absl::Status MemoryManager::release_buffer_pool() {
  absl::MutexLock lock(&mutex_);
  if (!cpu_.streaming_buffer) {
    return absl::OkStatus();
  }
  absl::Status st = cpu_.streaming_buffer->release();
  cpu_.streaming_buffer.reset();
  return st;
}

std::shared_ptr<StreamingPinnedBuffer> MemoryManager::get_streaming_buffer() const {
  absl::MutexLock lock(&mutex_);
  return cpu_.streaming_buffer;
}

size_t MemoryManager::get_max_buffer_bytes() const {
  absl::MutexLock lock(&mutex_);
  return max_buffer_bytes_;
}

size_t MemoryManager::get_pool_chunk_size() const {
  absl::MutexLock lock(&mutex_);
  return pinned_pool_->chunk_size();
}

// ---------------------------------------------------------------------------
// DVMP pageable CPU region allocation helper
// ---------------------------------------------------------------------------
absl::StatusOr<memory::DistributedVirtualMemoryPool::VirtualRegion> MemoryManager::allocate_pageable_cpu_region() {
  absl::MutexLock lock(&mutex_);

  if (model_size_ == 0) {
    return absl::FailedPreconditionError(
        absl::StrFormat("MemoryManager(%s): Model size not set before DVMP allocation.", instance_key_.model_id));
  }

  return reserve_dvmp_region_locked_();
}

// --- Internal helpers implementation --------------------------------------
absl::StatusOr<memory::DistributedVirtualMemoryPool::VirtualRegion> MemoryManager::reserve_dvmp_region_locked_() {
  // Assumes mutex_ is held
  // Attempt allocation. This call may return kAlreadyExists if another loader already reserved the region.
  auto region_or = dvmp_->allocate(instance_key_.model_id, model_size_);
  if (region_or.ok()) {
    // UMA now manages the CPU base pointer
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Reserved pageable CPU region of " << region_or->bytes
            << " bytes at " << region_or->cpu_base << " via DVMP.";
    return region_or;
  }
  if (region_or.status().code() == absl::StatusCode::kAlreadyExists) {
    // Region exists: query info and cache
    auto info_or = dvmp_->region_info(instance_key_.model_id);
    if (info_or.ok()) {
      // UMA now manages the CPU base pointer
      VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Using existing DVMP region at " << info_or->cpu_base
              << " with size " << info_or->bytes;
      return info_or;
    }
    LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                 << "): region_info failed after AlreadyExists: " << info_or.status();
    return info_or.status();
  }
  LOG(ERROR) << "MemoryManager(" << instance_key_.model_id << "): DVMP allocation failed: " << region_or.status();
  return region_or.status();
}

absl::Status MemoryManager::ensure_gpu_stream_initialized_locked_() {
  // Assumes mutex_ is held
  if (gpu_.stream_initialized && gpu_.stream != nullptr) {
    return absl::OkStatus();
  }
  auto dev_st = stepcast::cuda::set_device(instance_key_.device.ordinal);
  if (!dev_st.ok()) {
    return dev_st;
  }
  auto stream_st = stepcast::cuda::stream_create_with_flags(&gpu_.stream, cudaStreamNonBlocking);
  if (!stream_st.ok()) {
    return stream_st;
  }
  gpu_.stream_initialized = true;
  VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Lazily created CUDA stream " << gpu_.stream
          << " on device " << instance_key_.device.ordinal << ".";
  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// Peer-to-Peer copy (GPU↔GPU) – experimental implementation
// ---------------------------------------------------------------------------
absl::Status MemoryManager::copy_from_peer(const MemoryManager& source, cudaStream_t ext_stream) {
  // Quick checks – both managers must have GPU memory LOADED.
  if (this == &source) {
    return absl::InvalidArgumentError("Source and destination MemoryManager are identical.");
  }

  if (source.get_state(ModelLocation::GPU) != MemoryState::LOADED) {
    return absl::FailedPreconditionError("Source GPU memory not in LOADED state");
  }
  if (get_state(ModelLocation::GPU) != MemoryState::ALLOCATED) {
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
    auto vec = source.get_pointer(ModelLocation::GPU);
    if (vec.empty() || vec[0] == nullptr) {
      return absl::InternalError("Source GPU pointer invalid");
    }
    src_ptr = vec[0];
  }
  void* dst_ptr = nullptr;
  {
    auto vec = get_pointer(ModelLocation::GPU);
    if (vec.empty() || vec[0] == nullptr) {
      return absl::InternalError("Destination GPU pointer invalid");
    }
    dst_ptr = vec[0];
  }

  // Determine size (same on both managers).
  uint64_t bytes = source.get_model_size();
  if (bytes == 0 || bytes != get_model_size()) {
    return absl::FailedPreconditionError("Model size mismatch between source and destination");
  }

  // Launch async peer copy.
  auto st = cuda::memcpy_async(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream_to_use);
  if (!st.ok()) {
    ABSL_CHECK_OK(set_state(ModelLocation::GPU, MemoryState::FAILED));
    return st;
  }

  // Update state to LOADING then LOADED when stream sync completes.
  ABSL_CHECK_OK(set_state(ModelLocation::GPU, MemoryState::LOADING));
  auto sync_status = cuda::stream_synchronize(stream_to_use);
  if (!sync_status.ok()) {
    ABSL_CHECK_OK(set_state(ModelLocation::GPU, MemoryState::FAILED));
    return sync_status;
  }
  ABSL_CHECK_OK(set_state(ModelLocation::GPU, MemoryState::LOADED));

  // Update UMA states for GPU chunks after peer copy completes.
  absl::Status uma_st = finalize_load(ModelLocation::GPU);
  if (!uma_st.ok()) {
    LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                 << "): UMA finalize after peer copy returned: " << uma_st;
  }
  return absl::OkStatus();
}

// --- Chunk-scoped export / unexport for P2P access -------------------------

namespace {
// Coalesce sorted chunk indices into contiguous [start,end] inclusive ranges
std::vector<std::pair<uint32_t, uint32_t>> coalesce_ranges(std::vector<uint32_t> chunks) {
  std::vector<std::pair<uint32_t, uint32_t>> out;
  if (chunks.empty()) {
    return out;
  }
  std::sort(chunks.begin(), chunks.end());
  uint32_t start = chunks.front();
  uint32_t prev = start;
  for (size_t i = 1; i < chunks.size(); ++i) {
    if (chunks[i] == prev + 1) {
      prev = chunks[i];
      continue;
    }
    out.emplace_back(start, prev);
    start = prev = chunks[i];
  }
  out.emplace_back(start, prev);
  return out;
}
} // namespace

absl::StatusOr<CommRegistrationInfo> MemoryManager::export_chunks_for_p2p(
    ModelLocation location,
    absl::Span<const uint32_t> chunks,
    communicator::CommunicateEngine& comm_engine) {
  if (chunks.empty()) {
    return absl::InvalidArgumentError("No chunks specified for export");
  }

  if (location == ModelLocation::PAGEABLE_CPU) {
    std::shared_ptr<memory::DistributedVirtualMemoryPool> dvmp_capture;
    void* base_capture = nullptr;
    std::string model_id;
    uint64_t model_bytes = 0;
    {
      absl::MutexLock lock(&mutex_);
      if (cpu_.state != MemoryState::LOADED) {
        return absl::FailedPreconditionError("CPU memory must be LOADED to export chunks");
      }
      dvmp_capture = dvmp_;
      // Get CPU base from UMA (single source of truth)
      base_capture = memory_coordinator_->get_cpu_base_ptr(instance_key_);
      model_id = instance_key_.model_id;
      model_bytes = model_size_;
    }
    if (!dvmp_capture || base_capture == nullptr) {
      return absl::FailedPreconditionError("DVMP not available or base not set");
    }

    // Build contiguous ranges and register each as a tensor key with a pin lease
    CommRegistrationInfo info;
    info.model_size = model_bytes;
    info.location = location;
    info.device_id = 1;
    info.comm_dev_type = communicator::COMMUNICATE_ENGINE_DEV_CPU;

    std::vector<uint32_t> chunk_vec(chunks.begin(), chunks.end());
    auto ranges = coalesce_ranges(std::move(chunk_vec));

    constexpr uint64_t kChunk = memory::DistributedVirtualMemoryPool::kChunk;
    size_t range_idx = 0;
    for (const auto& [start, end] : ranges) {
      uint64_t va_off = static_cast<uint64_t>(start) * kChunk;
      uint64_t va_end = std::min<uint64_t>(model_bytes, (static_cast<uint64_t>(end) + 1) * kChunk);
      uint64_t length = (va_end > va_off) ? (va_end - va_off) : 0;
      if (length == 0) {
        continue;
      }

      auto lease_or = dvmp_capture->pin_range(model_id, va_off, length, "ExternalShare");
      if (!lease_or.ok()) {
        return lease_or.status();
      }
      {
        absl::MutexLock lock(&mutex_);
        cpu_.pin_leases.emplace_back(std::move(*lease_or));
      }

      auto addr = reinterpret_cast<uint64_t>(static_cast<char*>(base_capture) + va_off);
      auto key = absl::StrFormat("%s_CPU_chunk_%zu", model_id, range_idx++);
      auto ret = comm_engine.register_tensor(key, addr, length, info.comm_dev_type, info.device_id);
      if (!ret.ok()) {
        return absl::InternalError("Failed to register chunk-range tensor");
      }
      info.buffer_addresses.push_back(addr);
      info.buffer_sizes.push_back(static_cast<size_t>(length));
      info.remote_memory_keys.push_back(std::move(key));
    }

    // Cache CPU registration details for later unexport
    {
      absl::MutexLock lock(&mutex_);
      cpu_.comm_registration_info = info;
      cpu_.comm_registered = true;
    }

    // Metrics: count exported chunks by location
    try {
      static const stepcast::metrics::Counter kChunkExports("chunk_exports_total");
      kChunkExports.with_labels({{"location", "CPU"}}).inc(static_cast<double>(chunks.size()));
    } catch (...) {
      // Best-effort metrics - ignore failures
      VLOG(2) << "Failed to record chunk export metrics";
    }

    return info;
  }

  if (location == ModelLocation::GPU) {
    // GPU path: allow chunk-range keys by logical DVMP chunking on top of contiguous GPU buffer.
    void* gpu_ptr = nullptr;
    uint64_t model_bytes = 0;
    int device_id = -1;
    {
      absl::MutexLock lock(&mutex_);
      if (gpu_.state != MemoryState::LOADED) {
        return absl::FailedPreconditionError("GPU memory must be LOADED to export chunks");
      }
      if (!gpu_.cuda_mem) {
        return absl::InternalError("GPU memory not allocated through UMA");
      }
      gpu_ptr = gpu_.cuda_mem->get();
      model_bytes = model_size_;
      device_id = instance_key_.device.ordinal;
    }

    CommRegistrationInfo info;
    info.model_size = model_bytes;
    info.location = location;
    info.device_id = device_id;
    info.comm_dev_type = communicator::COMMUNICATE_ENGINE_DEV_GPU;

    std::vector<uint32_t> chunk_vec(chunks.begin(), chunks.end());
    auto ranges = coalesce_ranges(std::move(chunk_vec));

    constexpr uint64_t kChunk = memory::DistributedVirtualMemoryPool::kChunk;
    size_t range_idx = 0;
    for (const auto& [start, end] : ranges) {
      uint64_t off = static_cast<uint64_t>(start) * kChunk;
      uint64_t va_end = std::min<uint64_t>(model_bytes, (static_cast<uint64_t>(end) + 1) * kChunk);
      uint64_t length = (va_end > off) ? (va_end - off) : 0;
      if (length == 0) {
        continue;
      }
      auto addr = reinterpret_cast<uint64_t>(static_cast<char*>(gpu_ptr) + off);
      auto key = absl::StrFormat("%s_GPU_chunk%zu", instance_key_.model_id, range_idx++);
      auto ret = comm_engine.register_tensor(key, addr, length, info.comm_dev_type, info.device_id);
      if (!ret.ok()) {
        return absl::InternalError("Failed to register GPU chunk-range tensor");
      }
      info.buffer_addresses.push_back(addr);
      info.buffer_sizes.push_back(static_cast<size_t>(length));
      info.remote_memory_keys.push_back(std::move(key));
    }

    // Cache GPU registration details for later unexport
    {
      absl::MutexLock lock(&mutex_);
      gpu_.comm_registration_info = info;
      gpu_.comm_registered = true;
    }

    try {
      static const stepcast::metrics::Counter kChunkExports("chunk_exports_total");
      kChunkExports.with_labels({{"location", "GPU"}}).inc(static_cast<double>(chunks.size()));
    } catch (...) {
      // Best-effort metrics - ignore failures
      VLOG(2) << "Failed to record chunk export metrics";
    }

    return info;
  }

  return absl::InvalidArgumentError("Invalid location for export_chunks_for_p2p");
}

absl::Status MemoryManager::unexport_chunks_for_p2p(
    ModelLocation location,
    absl::Span<const uint32_t> /*chunks*/,
    communicator::CommunicateEngine& comm_engine) {
  if (location == ModelLocation::PAGEABLE_CPU) {
    // Unregister keys and release pin leases
    absl::Status first_error;
    {
      absl::MutexLock lock(&mutex_);
      for (const auto& key : cpu_.comm_registration_info.remote_memory_keys) {
        absl::Status st = comm_engine.unregister_tensor(key);
        if (!st.ok() && first_error.ok()) {
          first_error = st;
        }
      }
      cpu_.comm_registration_info.buffer_addresses.clear();
      cpu_.comm_registration_info.buffer_sizes.clear();
      cpu_.comm_registration_info.remote_memory_keys.clear();
      cpu_.pin_leases.clear();
      cpu_.comm_registered = false;
    }
    if (!first_error.ok()) {
      return first_error;
    }
    return absl::OkStatus();
  }

  if (location == ModelLocation::GPU) {
    absl::Status first_error;
    {
      absl::MutexLock lock(&mutex_);
      for (const auto& key : gpu_.comm_registration_info.remote_memory_keys) {
        absl::Status st = comm_engine.unregister_tensor(key);
        if (!st.ok() && first_error.ok()) {
          first_error = st;
        }
      }
      gpu_.comm_registration_info.buffer_addresses.clear();
      gpu_.comm_registration_info.buffer_sizes.clear();
      gpu_.comm_registration_info.remote_memory_keys.clear();
      gpu_.comm_registered = false;
    }
    if (!first_error.ok()) {
      return first_error;
    }
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Invalid location for unexport_chunks_for_p2p");
}

// --- DVMP accessor implementation ---
memory::DistributedVirtualMemoryPool* stepcast::store::MemoryManager::get_dvmp() {
  absl::MutexLock lock(&mutex_);
  return dvmp_.get().get();
}

void* MemoryManager::get_dvmp_cpu_base() const {
  absl::MutexLock lock(&mutex_);
  // Always use UMA as single source of truth for CPU base
  return memory_coordinator_->get_cpu_base_ptr(instance_key_);
}

// Opaque keepalive container for DVMP pin leases held by a DirectWriteToken
namespace {
struct DwKeepalive {
  std::vector<memory::DistributedVirtualMemoryPool::ChunkResidencyLease> leases;
};
} // namespace

absl::StatusOr<DirectWriteToken> MemoryManager::plan_direct_write(absl::Span<const VaRange> ranges) {
  std::shared_ptr<memory::DistributedVirtualMemoryPool> dvmp;
  void* base = nullptr;
  std::string model_id;
  uint64_t model_bytes = 0;
  {
    absl::MutexLock lock(&mutex_);
    if (cpu_.state != MemoryState::LOADED && cpu_.state != MemoryState::ALLOCATED) {
      return absl::FailedPreconditionError("CPU memory must be allocated/loaded for direct write");
    }
    dvmp = dvmp_;
    // Get CPU base from UMA (single source of truth)
    base = memory_coordinator_->get_cpu_base_ptr(instance_key_);
    model_id = instance_key_.model_id;
    model_bytes = model_size_;
  }
  if (!dvmp || base == nullptr) {
    return absl::FailedPreconditionError("DVMP base not available");
  }

  auto keep = std::make_shared<DwKeepalive>();
  DirectWriteToken token;
  token.segments.reserve(ranges.size());
  for (const auto& r : ranges) {
    if (r.offset + r.length > model_bytes) {
      return absl::OutOfRangeError("Direct write range exceeds model bounds");
    }
    // Acquire a pin lease to protect this VA range
    auto lease_or = dvmp->pin_range(model_id, r.offset, r.length, "DirectWrite");
    if (!lease_or.ok()) {
      return lease_or.status();
    }
    keep->leases.emplace_back(std::move(*lease_or));
    token.segments.push_back(
        DirectWriteToken::Segment{
            .va_offset = r.offset,
            .local_addr = reinterpret_cast<uint64_t>(static_cast<char*>(base) + r.offset),
            .length = r.length});
  }
  token.keepalive = keep;
  return token;
}

absl::Status MemoryManager::finalize_load(
    ModelLocation location,
    std::optional<absl::Span<const uint32_t>> chunk_indices) {
  // Unified memory optional – if absent, treat as no-op for CPU/GPU.
  auto uma = get_memory_coordinator();
  if (!uma) {
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): UMA not initialized; finalize_load is a no-op.";
    return absl::OkStatus();
  }

  // Handle CPU targets by syncing UMA from DVMP
  if (location == ModelLocation::PAGEABLE_CPU) {
    if (chunk_indices.has_value() && !chunk_indices->empty()) {
      // Build ranges from chunk indices for efficient syncing
      std::vector<std::pair<uint32_t, uint32_t>> ranges;
      uint32_t range_start = (*chunk_indices)[0];
      uint32_t range_end = range_start;

      for (size_t i = 1; i < chunk_indices->size(); ++i) {
        uint32_t idx = (*chunk_indices)[i];
        if (idx == range_end + 1) {
          // Extend current range
          range_end = idx;
        } else {
          // Save current range and start a new one
          ranges.emplace_back(range_start, range_end);
          range_start = idx;
          range_end = idx;
        }
      }
      ranges.emplace_back(range_start, range_end);

      // Sync only the specified ranges
      uma->sync_cpu_chunk_states(instance_key_, absl::MakeConstSpan(ranges));
    } else {
      // Sync all chunks
      uma->sync_cpu_chunk_states(instance_key_);
    }
    return absl::OkStatus();
  }

  // Only GPU updates UMA states explicitly
  if (location != ModelLocation::GPU) {
    return absl::OkStatus();
  }

  // Determine new state for GPU loads
  store::ChunkState new_state = store::ChunkState::COPIED_GPU;

  // Gather chunk list
  std::vector<uint32_t> chunks;
  if (chunk_indices.has_value()) {
    chunks.assign(chunk_indices->begin(), chunk_indices->end());
  } else {
    // Build [0..N-1]
    auto span = chunk_snapshot();
    chunks.reserve(span.size());
    for (uint32_t i = 0; i < span.size(); ++i) {
      chunks.push_back(i);
    }
  }

  // Provide local device id for GPU state updates
  int device_id = get_local_device_id();
  return uma->update_chunk_states(instance_key_, location, chunks, new_state, device_id);
}

absl::StatusOr<memory::DistributedVirtualMemoryPool::DvmpRegion> MemoryManager::get_dvmp_region() const {
  std::shared_ptr<memory::DistributedVirtualMemoryPool> dvmp;
  std::string model_id;
  {
    absl::MutexLock lock(&mutex_);
    dvmp = dvmp_;
    model_id = instance_key_.model_id;
  }
  if (!dvmp) {
    return absl::FailedPreconditionError("DVMP not available");
  }
  return dvmp->open(model_id);
}

// --- DVMP metadata snapshot -------------------------------------------------
// Provides a lightweight, read-only view of per-chunk metadata stored inside
// the DistributedVirtualMemoryPool (DVMP).  The span remains valid as long as the
// DVMP instance itself lives.  Callers must treat the returned ChunkMeta
// objects as immutable and use the atomic accessors defined inside ChunkMeta
// for state inspection.
absl::Span<const store::ChunkMeta> stepcast::store::MemoryManager::chunk_snapshot() const {
  // No expensive operations here – simply delegate to DVMP. We only acquire
  // the mutex to safely access the shared_ptr.  DVMP provides its own
  // internal locking for thread-safe snapshot retrieval.
  absl::MutexLock lock(&mutex_);
  return dvmp_->chunk_snapshot(instance_key_.model_id);
}

// --- NEW: Unified Memory Management implementations ---

std::shared_ptr<ModelMemoryCoordinator> MemoryManager::get_memory_coordinator() const {
  absl::MutexLock lock(&mutex_);
  return memory_coordinator_;
}

absl::Status MemoryManager::allocate_model_memory() {
  absl::MutexLock lock(&mutex_);

  if (model_size_ == 0) {
    return absl::FailedPreconditionError("Model size must be set before UMA allocation");
  }

  if (memory_coordinator_->has_allocation(instance_key_)) {
    return absl::AlreadyExistsError("UMA model allocation already exists");
  }

  // Allocate via unified memory (which will use DVMP internally)
  auto status = memory_coordinator_->allocate(instance_key_, model_size_);
  if (!status.ok()) {
    return status;
  }

  // Update our internal tracking
  cpu_.dvmp_base = memory_coordinator_->get_cpu_base_ptr(instance_key_);

  LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Allocated unified memory for " << model_size_
            << " bytes";

  return absl::OkStatus();
}

absl::Status MemoryManager::mark_cpu_preemptible(float ratio) {
  absl::MutexLock lock(&mutex_);

  if (!memory_coordinator_->has_allocation(instance_key_)) {
    return absl::FailedPreconditionError("Unified memory not allocated for this model");
  }

  if (ratio < 0.0F || ratio > 1.0F) {
    return absl::InvalidArgumentError("Ratio must be between 0.0 and 1.0");
  }

  // Get current chunk mappings
  auto mappings = memory_coordinator_->get_chunk_mappings(instance_key_);
  if (mappings.empty()) {
    return absl::OkStatus(); // No chunks to mark
  }

  // Calculate number of chunks to mark as preemptible
  auto num_chunks_to_mark = static_cast<size_t>(mappings.size() * ratio);
  if (num_chunks_to_mark == 0 && ratio > 0.0F) {
    num_chunks_to_mark = 1; // At least one chunk if ratio > 0
  }

  // Sort chunks by last access time (oldest first)
  std::vector<std::pair<uint32_t, uint64_t>> chunk_access_times;
  chunk_access_times.reserve(mappings.size());

  const bool mark_all = std::fabs(ratio - 1.0F) < 1e-6F;

  // Collect candidate chunks if we need to sort; otherwise we'll push directly.
  std::vector<uint32_t> chunks_to_mark;
  chunks_to_mark.reserve(mark_all ? mappings.size() : num_chunks_to_mark);

  for (size_t i = 0; i < mappings.size(); ++i) {
    const auto& mapping = mappings[i];
    const bool eligible =
        (mapping.cpu_state == ChunkState::HOT || mapping.cpu_state == ChunkState::COLD ||
         mapping.cpu_state == ChunkState::COPIED_GPU);

    if (eligible) {
      if (mark_all) {
        // Fast path: select all eligible chunks and skip sorting when ratio==1
        chunks_to_mark.push_back(static_cast<uint32_t>(i));
      } else {
        chunk_access_times.emplace_back(i, mapping.last_access_ns);
      }
    }
  }

  if (!mark_all) {
    // Sort by access time (oldest first)
    std::sort(chunk_access_times.begin(), chunk_access_times.end(), [](const auto& a, const auto& b) {
      return a.second < b.second;
    });

    // Build list of chunks to mark from the sorted vector
    for (size_t i = 0; i < std::min(num_chunks_to_mark, chunk_access_times.size()); ++i) {
      chunks_to_mark.push_back(chunk_access_times[i].first);
    }
  }

  // If no eligible chunks, return.
  if (chunks_to_mark.empty()) {
    return absl::OkStatus();
  }

  // Use DVMP to mark chunks as preemptible
  auto status = dvmp_->mark_preemptible(instance_key_.model_id, chunks_to_mark);
  if (!status.ok()) {
    return status;
  }

  LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Marked " << chunks_to_mark.size()
            << " CPU chunks as preemptible (ratio=" << ratio << ")";

  return absl::OkStatus();
}

std::vector<uint32_t> MemoryManager::get_missing_chunks(ModelLocation target, std::optional<int> device_id) const {
  absl::MutexLock lock(&mutex_);

  if (!memory_coordinator_->has_allocation(instance_key_)) {
    return {}; // No UMA allocation for this model
  }

  return memory_coordinator_->get_missing_chunks(instance_key_, target, device_id);
}

absl::Status MemoryManager::ensure_streaming_buffer(size_t num_chunks) {
  absl::MutexLock lock(&mutex_);
  if (cpu_.streaming_buffer) {
    return absl::OkStatus();
  }
  // Alignment policy enforcement: ensure pool chunk size divides DVMP chunk size
  // and is O_DIRECT-friendly (multiple of 4 KiB).
  const size_t pool_chunk = pinned_pool_->chunk_size();
  const size_t dvmp_chunk = ::stepcast::memory::DistributedVirtualMemoryPool::kChunk;
  if (pool_chunk == 0) {
    return absl::InvalidArgumentError("Pinned memory pool chunk size must be > 0");
  }
  if (dvmp_chunk % pool_chunk != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Pinned pool chunk size (%zu) must divide DVMP chunk size (%zu)", pool_chunk, dvmp_chunk));
  }
  if (pool_chunk % 4096 != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Pinned pool chunk size (%zu) must be a multiple of 4096 for O_DIRECT alignment", pool_chunk));
  }
  return allocate_buffer_pool(num_chunks);
}

} // namespace stepcast::store
