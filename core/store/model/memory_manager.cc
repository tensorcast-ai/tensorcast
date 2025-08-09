// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/model/memory_manager.h"
#include "core/common/trace/trace_macros.h"
#include "core/store/model/model_location.h"

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
// Forward declaration of streaming copy helper (definition later in this file)
class StreamingPinnedBuffer;
absl::Status perform_copy_cpu_to_gpu_streaming(
    const std::string& model_id,
    uint32_t device_id,
    const std::shared_ptr<StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<::stepcast::memory::DistributedMemoryPool>& dvmp,
    const std::shared_ptr<UnifiedModelMemory>& uma,
    const stepcast::store::InstanceKey& ikey);

// Forward declaration (add after perform_copy_cpu_to_gpu_streaming declaration)
absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& model_id,
    uint32_t device_id,
    const std::shared_ptr<StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<::stepcast::memory::DistributedMemoryPool>& dvmp);

MemoryManager::MemoryManager(
    std::string model_identifier,
    int local_device_id,
    const gsl::not_null<std::shared_ptr<PinnedMemoryPool>>& pinned_pool,
    const gsl::not_null<std::shared_ptr<memory::DistributedMemoryPool>>& dvmp,
    size_t max_buffer_bytes,
    std::chrono::milliseconds pinned_memory_timeout,
    bool require_dvmp_lock_success)
    : pinned_pool_(pinned_pool),
      max_buffer_bytes_(max_buffer_bytes),
      pinned_memory_timeout_(pinned_memory_timeout),
      require_dvmp_lock_success_(require_dvmp_lock_success),
      dvmp_(dvmp) {
  // Populate instance_key_ using constructor inputs
  instance_key_.model_id = std::move(model_identifier);
  instance_key_.device.type = (local_device_id >= 0) ? stepcast::DeviceType::GPU : stepcast::DeviceType::CPU;
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
    auto dev_st = stepcast::cuda::set_device(instance_key_.device.ordinal);
    if (!dev_st.ok()) {
      LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                 << "): Failed to set CUDA device during construction: " << dev_st;
    } else {
      auto stream_st = stepcast::cuda::stream_create_with_flags(&gpu_.stream, cudaStreamNonBlocking);
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
    auto sync_status = stepcast::cuda::stream_synchronize(local_stream);
    if (!sync_status.ok()) {
      LOG(ERROR) << "MemoryManager(" << id_copy << "): Failed to synchronize CUDA stream " << local_stream
                 << " during destruction: " << sync_status.message();
    }
    auto destroy_status = stepcast::cuda::stream_destroy(local_stream);
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
      size_t required_chunks = static_cast<size_t>((model_size_ + chunk_size - 1) / chunk_size);
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

      gpu_.cuda_mem = std::make_shared<CudaMemory>();

      // Trace allocation of GPU memory.
      SC_TRACE_SCOPE("allocate_gpu_memory");

      VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Attempting direct cudaMalloc for " << model_size_
              << " bytes on device " << instance_key_.device.ordinal << ".";
      absl::Status alloc_status = gpu_.cuda_mem->allocate(model_size_, instance_key_.device.ordinal);
      if (!alloc_status.ok()) {
        LOG(ERROR) << "MemoryManager(" << instance_key_.model_id
                   << "): Direct cudaMalloc failed: " << alloc_status.message();
        gpu_.cuda_mem.reset();
        ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
        return absl::ResourceExhaustedError(
            absl::StrFormat(
                "MemoryManager(%s): Failed direct cudaMalloc on device %d: %s",
                instance_key_.model_id,
                instance_key_.device.ordinal,
                alloc_status.message()));
      }

      VLOG(1) << "MemoryManager(" << instance_key_.model_id
              << "): Direct cudaMalloc successful (ptr=" << gpu_.cuda_mem->get() << ").";
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

  if (cpu_.host_chunk_queue) {
    cpu_.host_chunk_queue.reset();
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): CPU host_chunk_queue released/reset.";
  }
}

// Private helper
void MemoryManager::release_gpu_resources_locked() {
  // Called with mutex held
  if (gpu_.cuda_mem) {
    VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Releasing allocated GPU memory object (pointer "
            << gpu_.cuda_mem->get() << " will be freed/returned to pool by CudaMemory dtor).";
    // CudaMemory destructor handles cudaFree or pool deallocate appropriately
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
        // For mmap/zero-copy path (DVMP memory). cpu_.dvmp_base may be lazily resolved via UMA.
        void* base_ptr = cpu_.dvmp_base;
        if (base_ptr == nullptr) {
          auto uma = unified_memory_;
          if (uma) {
            base_ptr = uma->get_cpu_base_ptr(instance_key_);
          }
        }
        if (base_ptr == nullptr) {
          VLOG(1) << "MemoryManager(" << instance_key_.model_id
                  << "): CPU base is null despite ALLOCATED/LOADED state.";
          return {};
        }
        return std::vector<void*>({base_ptr});
      }
      VLOG(2) << "MemoryManager(" << instance_key_.model_id
              << "): get_pointer(PAGEABLE_CPU) returning null. State: " << state_to_string(cpu_.state)
              << ", StreamingBuffer valid: " << (cpu_.streaming_buffer != nullptr) << ", DVMP base: " << cpu_.dvmp_base;
      return {};
    case ModelLocation::GPU:
      // Return pointer if memory is allocated, loading, or loaded (pointer valid during streaming and copy)
      if ((gpu_.state >= MemoryState::ALLOCATED && gpu_.state != MemoryState::FAILED) && gpu_.cuda_mem) {
        auto* const ptr = gpu_.cuda_mem->get();
        return std::vector<void*>({ptr});
      }
      VLOG(2) << "MemoryManager(" << instance_key_.model_id
              << "): get_pointer(GPU) returning null. State: " << state_to_string(gpu_.state)
              << ", CudaMem valid: " << (gpu_.cuda_mem != nullptr);
      return {};
    default:
      LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                   << "): get_pointer called with invalid location: " << static_cast<int>(location);
      return {};
  }
}

std::shared_ptr<BatchVector> MemoryManager::get_host_chunk_queue() const {
  absl::MutexLock lock(&mutex_);
  return cpu_.host_chunk_queue;
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
    auto st_um = allocate_unified();
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
      auto uma_capture = this->get_unified_memory();
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

absl::StatusOr<CommRegistrationInfo> MemoryManager::enable_remote_memory_access(
    ModelLocation location,
    stepcast::communicator::CommunicateEngine& comm_engine) {
  // Pointers to cached flags/infos (set under lock); used again when we cache results.
  bool* registered_ptr = nullptr;
  CommRegistrationInfo* cached_info_ptr = nullptr;
  uint64_t model_size_snapshot = 0;
  {
    absl::MutexLock lock(&mutex_);
    // Check if already registered for the specific location
    if (location == ModelLocation::PAGEABLE_CPU) {
      registered_ptr = &cpu_.comm_registered;
      cached_info_ptr = &cpu_.comm_registration_info;
    } else if (location == ModelLocation::GPU) {
      registered_ptr = &gpu_.comm_registered;
      cached_info_ptr = &gpu_.comm_registration_info;
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "MemoryManager(%s): Invalid location specified for Comm registration: %d",
              instance_key_.model_id,
              static_cast<int>(location)));
    }

    if (*registered_ptr) {
      LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Memory for " << location_to_string(location)
                << " already registered for communication. Returning cached registration info.";
      return *cached_info_ptr;
    }

    if (model_size_ == 0) {
      return absl::FailedPreconditionError(
          absl::StrFormat("MemoryManager(%s): Model size is 0, cannot register memory.", instance_key_.model_id));
    }
    model_size_snapshot = model_size_;
  }

  CommRegistrationInfo reg_info;
  reg_info.model_size = model_size_snapshot;
  reg_info.location = location;

  std::string location_str;
  reg_info.comm_dev_type = stepcast::communicator::COMMUNICATE_ENGINE_DEV_CPU;

  if (location == ModelLocation::PAGEABLE_CPU) {
    // New default: chunk-scoped export using DVMP pin leases.
    location_str = "PAGEABLE_CPU";
    reg_info.device_id = 1; // Explicitly 1 for CPU

    // Snapshot necessary state under lock to satisfy thread-safety analysis.
    bool cpu_loaded = false;
    void* dvmp_base_copy = nullptr;
    std::string model_id_copy;
    size_t chunk_count = 0;
    {
      absl::MutexLock lock(&mutex_);
      cpu_loaded = (cpu_.state == MemoryState::LOADED);
      dvmp_base_copy = cpu_.dvmp_base;
      model_id_copy = instance_key_.model_id;
      // Compute count from model size to avoid calling into DVMP while holding our mutex.
      chunk_count =
          (model_size_snapshot + memory::DistributedMemoryPool::kChunk - 1) / memory::DistributedMemoryPool::kChunk;
    }

    if (!cpu_loaded) {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "MemoryManager(%s): PAGEABLE_CPU memory must be in LOADED state for Comm registration.", model_id_copy));
    }
    if (dvmp_base_copy == nullptr) {
      return absl::FailedPreconditionError("DVMP CPU base address is null; cannot register PAGEABLE_CPU memory.");
    }

    // Build full chunk index list [0..N-1] and export via chunk API.
    if (chunk_count == 0) {
      return absl::FailedPreconditionError("No DVMP chunks available for CPU export");
    }
    std::vector<uint32_t> all_chunks;
    all_chunks.reserve(chunk_count);
    for (uint32_t i = 0; i < chunk_count; ++i)
      all_chunks.push_back(i);

    // Use chunk-scoped export; this acquires pin leases and registers keys per coalesced range.
    // Cache the returned info below to enable proper unregistration.
    auto info_or = export_chunks_for_p2p(ModelLocation::PAGEABLE_CPU, all_chunks, comm_engine);
    if (!info_or.ok()) {
      return info_or.status();
    }
    reg_info = *info_or;
  } else if (location == ModelLocation::GPU) {
    location_str = "GPU";
    uint32_t device_id_copy = 0;
    void* ptr_to_register = nullptr;
    size_t size_to_register = model_size_snapshot;
    {
      absl::MutexLock lock(&mutex_);
      reg_info.device_id = instance_key_.device.ordinal; // Use actual device ID for GPU
      reg_info.comm_dev_type = stepcast::communicator::COMMUNICATE_ENGINE_DEV_GPU;

      // GPU must be ready (LOADED) to be registered
      if (gpu_.state != MemoryState::LOADED) {
        return absl::FailedPreconditionError(
            absl::StrFormat(
                "MemoryManager(%s): GPU memory must be in LOADED state for Comm registration (current: %s).",
                instance_key_.model_id,
                state_to_string(gpu_.state)));
      }
      if (!gpu_.cuda_mem || gpu_.cuda_mem->get() == nullptr) {
        return absl::InternalError(
            absl::StrFormat(
                "MemoryManager(%s): Invalid GPU memory object (null=%d) or size (%d) for Comm registration.",
                instance_key_.model_id,
                gpu_.cuda_mem == nullptr || gpu_.cuda_mem->get() == nullptr,
                model_size_));
      }
      ptr_to_register = gpu_.cuda_mem->get();
      device_id_copy = reg_info.device_id;
    }
    auto addr_to_register = reinterpret_cast<uint64_t>(ptr_to_register);

    // Construct a unique key for the single GPU buffer registration
    // Using "chunk0" suffix for consistency, even though it's one block.
    auto tensor_name_for_comm =
        absl::StrFormat("%s_%s_dev%d_chunk0", instance_key_.model_id, location_str, reg_info.device_id);

    VLOG(1) << "MemoryManager(" << instance_key_.model_id
            << "): Registering GPU memory (1 chunk) for Comm via CommunicateEngine. Name: '" << tensor_name_for_comm
            << "', Addr: " << ptr_to_register << " (" << addr_to_register << ")"
            << ", Size: " << size_to_register << ", DevType: " << reg_info.comm_dev_type
            << ", DevId: " << device_id_copy;

    // Call the communicator engine's registration function
    auto ret = comm_engine.register_tensor(
        tensor_name_for_comm, addr_to_register, size_to_register, reg_info.comm_dev_type, reg_info.device_id);

    if (!ret.ok()) {
      LOG(ERROR) << "MemoryManager(" << instance_key_.model_id << "): Failed to register GPU tensor '"
                 << tensor_name_for_comm << "' with CommunicateEngine. Error code: " << ret;
      return absl::InternalError(
          absl::StrFormat("Failed to register GPU tensor %s via CommunicateEngine", tensor_name_for_comm));
    }

    // Store registration info
    reg_info.buffer_addresses.push_back(addr_to_register);
    reg_info.buffer_sizes.push_back(size_to_register);
    reg_info.remote_memory_keys.push_back(tensor_name_for_comm);
  }

  // Cache the registration result for the specific location
  {
    absl::MutexLock lock(&mutex_);
    *registered_ptr = true;
    *cached_info_ptr = reg_info;
  }

  return reg_info; // Return the populated struct
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
  if (source == ModelLocation::GPU && (!gpu_.cuda_mem || gpu_.cuda_mem->get() == nullptr)) {
    return absl::InternalError("Source GPU memory is invalid for copy.");
  }
  if (destination == ModelLocation::GPU && (!gpu_.cuda_mem || gpu_.cuda_mem->get() == nullptr)) {
    return absl::InternalError("Destination GPU memory is invalid for copy.");
  }

  out->cuda_mem = gpu_.cuda_mem;
  if (src_is_host || dst_is_host) {
    out->dvmp = dvmp_;
    out->dvmp_base = cpu_.dvmp_base;
  }
  out->total_size = model_size_;
  out->stream = gpu_.stream;
  out->device_id = instance_key_.device.ordinal;
  out->model_id = instance_key_.model_id;

  *need_allocate_um = (src_is_host || dst_is_host) && !unified_memory_;

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
  // Mandatory CPU memory release after successful CPU→GPU copy (per RFC 0001 §4.3)
  if (copy_status.ok() && destination == ModelLocation::GPU) {
    absl::MutexLock release_lock(&mutex_);
    if (cpu_.dvmp_base != nullptr) {
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

bool MemoryManager::is_comm_registered(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      return cpu_.comm_registered;
    case ModelLocation::GPU:
      return gpu_.comm_registered;
    default:
      LOG(WARNING) << "MemoryManager(" << instance_key_.model_id
                   << "): is_comm_registered called with invalid location: " << static_cast<int>(location);
      return false;
  }
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
absl::StatusOr<memory::DistributedMemoryPool::VirtualRegion> MemoryManager::allocate_pageable_cpu_region() {
  absl::MutexLock lock(&mutex_);

  if (model_size_ == 0) {
    return absl::FailedPreconditionError(
        absl::StrFormat("MemoryManager(%s): Model size not set before DVMP allocation.", instance_key_.model_id));
  }

  return reserve_dvmp_region_locked_();
}

// --- Internal helpers implementation --------------------------------------
absl::StatusOr<memory::DistributedMemoryPool::VirtualRegion> MemoryManager::reserve_dvmp_region_locked_() {
  // Assumes mutex_ is held
  // Attempt allocation. This call may return kAlreadyExists if another loader already reserved the region.
  auto region_or = dvmp_->allocate(instance_key_.model_id, model_size_);
  if (region_or.ok()) {
    cpu_.dvmp_base = region_or->cpu_base;
    cpu_.dvmp_bytes = region_or->bytes;
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Reserved pageable CPU region of " << region_or->bytes
            << " bytes at " << region_or->cpu_base << " via DVMP.";
    return region_or;
  }
  if (region_or.status().code() == absl::StatusCode::kAlreadyExists) {
    // Region exists: query info and cache
    auto info_or = dvmp_->region_info(instance_key_.model_id);
    if (info_or.ok()) {
      cpu_.dvmp_base = info_or->cpu_base;
      cpu_.dvmp_bytes = info_or->bytes;
      VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): Using existing DVMP region at " << cpu_.dvmp_base
              << " with size " << cpu_.dvmp_bytes;
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
  auto st = stepcast::cuda::memcpy_async(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream_to_use);
  if (!st.ok()) {
    ABSL_CHECK_OK(set_state(ModelLocation::GPU, MemoryState::FAILED));
    return st;
  }

  // Update state to LOADING then LOADED when stream sync completes.
  ABSL_CHECK_OK(set_state(ModelLocation::GPU, MemoryState::LOADING));
  auto sync_status = stepcast::cuda::stream_synchronize(stream_to_use);
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

// ---------------------------------------------------------------------------
// Disable remote memory access (unregistration)
// ---------------------------------------------------------------------------

absl::Status MemoryManager::disable_remote_memory_access(
    ModelLocation location,
    stepcast::communicator::CommunicateEngine& comm_engine) {
  absl::MutexLock lock(&mutex_);

  bool* registered_ptr = nullptr;
  CommRegistrationInfo* info_ptr = nullptr;

  if (location == ModelLocation::PAGEABLE_CPU) {
    registered_ptr = &cpu_.comm_registered;
    info_ptr = &cpu_.comm_registration_info;
  } else if (location == ModelLocation::GPU) {
    registered_ptr = &gpu_.comm_registered;
    info_ptr = &gpu_.comm_registration_info;
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "MemoryManager(%s): Invalid location specified for Comm unregistration: %d",
            instance_key_.model_id,
            static_cast<int>(location)));
  }

  if (!*registered_ptr) {
    // Nothing was registered – treat as a no-op.
    LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): No existing Comm registration for "
              << location_to_string(location) << ". Nothing to unregister.";
    return absl::OkStatus();
  }

  absl::Status first_error;

  for (const auto& key : info_ptr->remote_memory_keys) {
    absl::Status st = comm_engine.unregister_tensor(key);
    if (!st.ok()) {
      LOG(WARNING) << "MemoryManager(" << instance_key_.model_id << "): Failed to unregister tensor '" << key
                   << "' from CommunicateEngine. Status: " << st;
      if (first_error.ok()) {
        first_error = st;
      }
    } else {
      VLOG(2) << "MemoryManager(" << instance_key_.model_id << "): Unregistered tensor '" << key << "'";
    }
  }

  // Mark as unregistered irrespective of individual failures to avoid leaking state.
  *registered_ptr = false;

  // Optionally clear cached info to free memory (addresses still valid but ok).
  info_ptr->buffer_addresses.clear();
  info_ptr->buffer_sizes.clear();
  info_ptr->remote_memory_keys.clear();

  if (!first_error.ok()) {
    return absl::InternalError(absl::StrFormat("One or more tensors failed to unregister: %s", first_error.ToString()));
  }

  LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Successfully unregistered memory at "
            << location_to_string(location) << " for communication.";
  return absl::OkStatus();
}

// --- Chunk-scoped export / unexport for P2P access -------------------------

namespace {
// Coalesce sorted chunk indices into contiguous [start,end] inclusive ranges
static std::vector<std::pair<uint32_t, uint32_t>> coalesce_ranges(std::vector<uint32_t> chunks) {
  std::vector<std::pair<uint32_t, uint32_t>> out;
  if (chunks.empty())
    return out;
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
  // For now, implement CPU path; GPU remains whole-region.
  if (location != ModelLocation::PAGEABLE_CPU) {
    return absl::UnimplementedError("export_chunks_for_p2p currently supports PAGEABLE_CPU only");
  }
  if (chunks.empty()) {
    return absl::InvalidArgumentError("No chunks specified for export");
  }

  std::shared_ptr<memory::DistributedMemoryPool> dvmp_capture;
  void* base_capture = nullptr;
  std::string model_id;
  uint64_t model_bytes = 0;
  {
    absl::MutexLock lock(&mutex_);
    if (cpu_.state != MemoryState::LOADED) {
      return absl::FailedPreconditionError("CPU memory must be LOADED to export chunks");
    }
    dvmp_capture = dvmp_;
    base_capture = cpu_.dvmp_base;
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

  constexpr uint64_t kChunk = memory::DistributedMemoryPool::kChunk;
  size_t range_idx = 0;
  for (const auto& [start, end] : ranges) {
    uint64_t va_off = static_cast<uint64_t>(start) * kChunk;
    uint64_t va_end = std::min<uint64_t>(model_bytes, (static_cast<uint64_t>(end) + 1) * kChunk);
    uint64_t length = (va_end > va_off) ? (va_end - va_off) : 0;
    if (length == 0)
      continue;

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

  // Metrics: count exported chunks by location
  try {
    static const stepcast::metrics::Counter kChunkExports("chunk_exports_total");
    const char* loc = (location == ModelLocation::GPU) ? "GPU" : "CPU";
    kChunkExports.with_labels({{"location", loc}}).inc(static_cast<double>(chunks.size()));
  } catch (...) {
    // Best-effort metrics
  }

  return info;
}

absl::Status MemoryManager::unexport_chunks_for_p2p(
    ModelLocation location,
    absl::Span<const uint32_t> /*chunks*/,
    communicator::CommunicateEngine& comm_engine) {
  if (location != ModelLocation::PAGEABLE_CPU) {
    return absl::UnimplementedError("unexport_chunks_for_p2p currently supports PAGEABLE_CPU only");
  }
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
  }
  if (!first_error.ok()) {
    return first_error;
  }
  return absl::OkStatus();
}

// --- DVMP accessor implementation ---
memory::DistributedMemoryPool* stepcast::store::MemoryManager::get_dvmp() {
  absl::MutexLock lock(&mutex_);
  return dvmp_.get().get();
}

void* MemoryManager::get_dvmp_cpu_base() const {
  absl::MutexLock lock(&mutex_);
  return cpu_.dvmp_base;
}

// Opaque keepalive container for DVMP pin leases held by a DirectWriteToken
namespace {
struct DwKeepalive {
  std::vector<memory::DistributedMemoryPool::PinLease> leases;
};
} // namespace

absl::StatusOr<DirectWriteToken> MemoryManager::plan_direct_write(absl::Span<const VaRange> ranges) {
  std::shared_ptr<memory::DistributedMemoryPool> dvmp;
  void* base = nullptr;
  std::string model_id;
  uint64_t model_bytes = 0;
  {
    absl::MutexLock lock(&mutex_);
    if (cpu_.state != MemoryState::LOADED && cpu_.state != MemoryState::ALLOCATED) {
      return absl::FailedPreconditionError("CPU memory must be allocated/loaded for direct write");
    }
    dvmp = dvmp_;
    base = cpu_.dvmp_base;
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
  auto uma = get_unified_memory();
  if (!uma) {
    VLOG(1) << "MemoryManager(" << instance_key_.model_id << "): UMA not initialized; finalize_load is a no-op.";
    return absl::OkStatus();
  }

  // Only GPU updates UMA. CPU metadata is authoritative in DVMP and read via snapshots.
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
    for (uint32_t i = 0; i < span.size(); ++i)
      chunks.push_back(i);
  }

  // Provide local device id for GPU state updates
  int device_id = get_local_device_id();
  return uma->update_chunk_states(instance_key_, location, chunks, new_state, device_id);
}

absl::StatusOr<memory::DistributedMemoryPool::DvmpRegion> MemoryManager::get_dvmp_region() const {
  std::shared_ptr<memory::DistributedMemoryPool> dvmp;
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
// the DistributedMemoryPool (DVMP).  The span remains valid as long as the
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

std::shared_ptr<UnifiedModelMemory> MemoryManager::get_unified_memory() const {
  absl::MutexLock lock(&mutex_);
  return unified_memory_;
}

absl::Status MemoryManager::allocate_unified() {
  absl::MutexLock lock(&mutex_);

  if (unified_memory_) {
    return absl::AlreadyExistsError("Unified memory already allocated");
  }

  if (model_size_ == 0) {
    return absl::FailedPreconditionError("Model size must be set before unified allocation");
  }

  // Pass the shared DVMP instance directly to UnifiedModelMemory
  unified_memory_ = std::make_shared<UnifiedModelMemory>(dvmp_);

  // Allocate via unified memory (which will use DVMP internally)
  auto status = unified_memory_->allocate(instance_key_, model_size_);
  if (!status.ok()) {
    unified_memory_.reset();
    return status;
  }

  // Update our internal tracking
  cpu_.dvmp_base = unified_memory_->get_cpu_base_ptr(instance_key_);
  cpu_.dvmp_bytes = model_size_;

  LOG(INFO) << "MemoryManager(" << instance_key_.model_id << "): Allocated unified memory for " << model_size_
            << " bytes";

  return absl::OkStatus();
}

absl::Status MemoryManager::mark_cpu_preemptible(float ratio) {
  absl::MutexLock lock(&mutex_);

  if (!unified_memory_) {
    return absl::FailedPreconditionError("Unified memory not allocated");
  }

  if (ratio < 0.0F || ratio > 1.0F) {
    return absl::InvalidArgumentError("Ratio must be between 0.0 and 1.0");
  }

  // Get current chunk mappings
  auto mappings = unified_memory_->get_chunk_mappings(instance_key_);
  if (mappings.empty()) {
    return absl::OkStatus(); // No chunks to mark
  }

  // Calculate number of chunks to mark as preemptible
  size_t num_chunks_to_mark = static_cast<size_t>(mappings.size() * ratio);
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

  if (!unified_memory_) {
    return {}; // No unified memory, return empty
  }

  return unified_memory_->get_missing_chunks(instance_key_, target, device_id);
}

// ---------------------------------------------------------------------------
// New helper using StreamingPinnedBuffer for staged CPU->GPU copy
// ---------------------------------------------------------------------------

absl::Status perform_copy_cpu_to_gpu_streaming(
    const std::string& model_id,
    uint32_t device_id,
    const std::shared_ptr<StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<::stepcast::memory::DistributedMemoryPool>& dvmp,
    const std::shared_ptr<UnifiedModelMemory>& uma,
    const stepcast::store::InstanceKey& ikey) {
  // Required components must be present – enforce via CHECKKs
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK(gpu_ptr) << "GPU destination pointer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";

  const size_t dvmp_chunk = ::stepcast::memory::DistributedMemoryPool::kChunk;
  const size_t copy_chunk = streaming_buf->chunk_size();

  auto device_status = stepcast::cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  // Chunk-aware copy: iterate DVMP chunk ranges so that UMA can lock/unlock
  // per-chunk and update states correctly.
  const size_t num_dvmp_chunks = (total_size + dvmp_chunk - 1) / dvmp_chunk;
  for (size_t dvmp_idx = 0; dvmp_idx < num_dvmp_chunks; ++dvmp_idx) {
    const size_t dvmp_off = dvmp_idx * dvmp_chunk;
    const size_t this_len = std::min(dvmp_chunk, total_size - dvmp_off);

    // UMA lock this DVMP chunk for transfer (CPU -> GPU)
    if (uma) {
      std::vector<uint32_t> one{static_cast<uint32_t>(dvmp_idx)};
      auto st = uma->lock_chunks_for_transfer(ikey, ModelLocation::PAGEABLE_CPU, ModelLocation::GPU, one);
      if (!st.ok()) {
        return st;
      }
    }

    size_t copied = 0;
    while (copied < this_len) {
      size_t step = std::min(copy_chunk, this_len - copied);
      // Acquire a free chunk slot
      auto slot_or = streaming_buf->get_free_chunk();
      if (!slot_or.ok()) {
        return slot_or.status();
      }
      int slot_id = *slot_or;
      char* host_ptr = streaming_buf->get_chunk_ptr(slot_id);
      if (host_ptr == nullptr) {
        ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
        return absl::InternalError("Failed to get chunk pointer from streaming buffer");
      }

      // Copy from DVMP region to pinned chunk
      void* src_host = static_cast<char*>(dvmp_base) + dvmp_off + copied;
      std::memcpy(host_ptr, src_host, step);

      // Async copy H2D
      void* dst_device = static_cast<char*>(gpu_ptr) + dvmp_off + copied;
      auto memcpy_status = stepcast::cuda::memcpy_async(dst_device, host_ptr, step, cudaMemcpyHostToDevice, stream);
      if (!memcpy_status.ok()) {
        ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
        return memcpy_status;
      }

      // Synchronize to ensure chunk can be reused safely
      auto sync_status = stepcast::cuda::stream_synchronize(stream);
      if (!sync_status.ok()) {
        ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
        return sync_status;
      }

      // Return chunk to buffer
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      copied += step;
    }

    // UMA update: mark DVMP chunk as COPIED_GPU which triggers DVMP unlock
    if (uma) {
      std::vector<uint32_t> one{static_cast<uint32_t>(dvmp_idx)};
      auto st = uma->update_chunk_states(ikey, ModelLocation::GPU, one, ChunkState::COPIED_GPU, device_id);
      if (!st.ok()) {
        // Best-effort unlock on failure to avoid holding LOCKED_TX
        (void)dvmp->unlock_chunks(model_id, one, /*copied_gpu=*/true);
        return st;
      }
    }
  }

  return absl::OkStatus();
}

// ---------------------------------------------------------------------------
// New helper using StreamingPinnedBuffer for staged GPU->CPU copy
// ---------------------------------------------------------------------------

absl::Status perform_copy_gpu_to_cpu_streaming(
    const std::string& model_id,
    uint32_t device_id,
    const std::shared_ptr<StreamingPinnedBuffer>& streaming_buf,
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream,
    void* dvmp_base,
    const std::shared_ptr<::stepcast::memory::DistributedMemoryPool>& dvmp) {
  ABSL_CHECK(streaming_buf) << "StreamingPinnedBuffer must not be null";
  ABSL_CHECK(gpu_ptr) << "GPU source pointer must not be null";
  ABSL_CHECK_GT(total_size, 0) << "Total size must be positive";
  ABSL_CHECK(dvmp_base) << "DVMP base pointer must not be null";

  size_t chunk_size = streaming_buf->chunk_size();
  size_t offset = 0;

  auto device_status = stepcast::cuda::set_device(device_id);
  if (!device_status.ok()) {
    return device_status;
  }

  while (offset < total_size) {
    size_t current_chunk_size = std::min(chunk_size, total_size - offset);

    // Acquire a free chunk slot
    auto slot_or = streaming_buf->get_free_chunk();
    if (!slot_or.ok()) {
      return slot_or.status();
    }
    int slot_id = *slot_or;
    char* host_ptr = streaming_buf->get_chunk_ptr(slot_id);
    if (host_ptr == nullptr) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return absl::InternalError("Failed to get chunk pointer from streaming buffer");
    }

    // Async copy from GPU to pinned host chunk
    void* src_device = static_cast<char*>(gpu_ptr) + offset;
    auto memcpy_status =
        stepcast::cuda::memcpy_async(host_ptr, src_device, current_chunk_size, cudaMemcpyDeviceToHost, stream);
    if (!memcpy_status.ok()) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return memcpy_status;
    }

    // Synchronize to ensure chunk hosts valid data before copying to DVMP
    auto sync_status = stepcast::cuda::stream_synchronize(stream);
    if (!sync_status.ok()) {
      ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));
      return sync_status;
    }

    // Copy from pinned chunk to DVMP destination
    void* dst_host = static_cast<char*>(dvmp_base) + offset;
    std::memcpy(dst_host, host_ptr, current_chunk_size);

    // Return chunk to buffer
    ABSL_CHECK_OK(streaming_buf->return_chunk(slot_id));

    offset += current_chunk_size;
  }

  return absl::OkStatus();
}

absl::Status MemoryManager::ensure_streaming_buffer(size_t num_chunks) {
  absl::MutexLock lock(&mutex_);
  if (cpu_.streaming_buffer) {
    return absl::OkStatus();
  }
  // Alignment policy enforcement: ensure pool chunk size divides DVMP chunk size
  // and is O_DIRECT-friendly (multiple of 4 KiB).
  const size_t pool_chunk = pinned_pool_->chunk_size();
  const size_t dvmp_chunk = ::stepcast::memory::DistributedMemoryPool::kChunk;
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
