// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/model/memory_manager.h"
#include "core/common/trace/trace_macros.h"
#include "core/store/model/model_location.h"

#include <algorithm>
#include <cmath>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "core/common/cuda_api.h"
#include "core/common/device_types.h"
#include "core/communicator/engine/engine.h"

#define model_identifier_ instance_key_.model_id
#define local_device_id_ instance_key_.device.ordinal

namespace stepcast::store {
MemoryManager::MemoryManager(
    std::string model_identifier,
    int local_device_id,
    std::shared_ptr<PinnedMemoryPool> pinned_pool,
    size_t max_buffer_bytes,
    std::chrono::milliseconds pinned_memory_timeout,
    bool require_dvmp_lock_success)
    : pinned_pool_(std::move(pinned_pool)),
      pageable_cpu_state_(MemoryState::UNALLOCATED),
      gpu_state_(MemoryState::UNALLOCATED),
      copy_stream_(nullptr),
      stream_initialized_(false),
      max_buffer_bytes_(max_buffer_bytes),
      pinned_memory_timeout_(pinned_memory_timeout),
      require_dvmp_lock_success_(require_dvmp_lock_success),
      dvmp_(std::make_unique<memory::DistributedMemoryPool>()) {
  // DVMP is always available in this codebase (per review requirement)
  ABSL_CHECK(dvmp_ != nullptr) << "MemoryManager: DVMP allocation failed, which should not happen";
  
  // Populate instance_key_ using constructor inputs
  instance_key_.model_id = std::move(model_identifier);
  instance_key_.device.type = (local_device_id >= 0) ? stepcast::DeviceType::GPU : stepcast::DeviceType::CPU;
  instance_key_.device.ordinal = local_device_id;

  {
    absl::MutexLock lock(&mutex_);
    // Initialize states properly based on whether pools are provided
    pageable_cpu_state_ = pinned_pool_ ? MemoryState::UNALLOCATED : MemoryState::UNINITIALIZED;
    // GPU state depends on pool or potential borrowing later
    gpu_state_ = MemoryState::UNALLOCATED; // Assume potential for allocation/borrowing
  }

  // Initialise CUDA context and non-blocking stream if a valid device id was provided at construction.
  if (local_device_id_ >= 0) {
    auto dev_st = stepcast::cuda::set_device(local_device_id_);
    if (!dev_st.ok()) {
      LOG(ERROR) << "MemoryManager(" << model_identifier_
                 << "): Failed to set CUDA device during construction: " << dev_st;
    } else {
      auto stream_st = stepcast::cuda::stream_create_with_flags(&copy_stream_, cudaStreamNonBlocking);
      if (!stream_st.ok()) {
        LOG(ERROR) << "MemoryManager(" << model_identifier_
                   << "): Failed to create CUDA stream during construction: " << stream_st;
      } else {
        stream_initialized_ = true;
        VLOG(2) << "MemoryManager(" << model_identifier_ << "): Created CUDA stream " << copy_stream_ << " on device "
                << local_device_id_ << ".";
      }
    }
  }
}

MemoryManager::~MemoryManager() {
  cudaStream_t local_stream = nullptr;
  std::string id_copy; // Copy identifier for logging after potential lock release
  {
    absl::MutexLock lock(&mutex_);
    id_copy = model_identifier_; // Copy identifier while lock is held
    local_stream = copy_stream_;
    stream_initialized_ = false; // Mark as not initialized early
    copy_stream_ = nullptr; // Prevent use after unlock if sync takes time
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
    VLOG(2) << "MemoryManager(" << model_identifier_ << "): Destructor finished.";
  }
}

void MemoryManager::set_model_size(uint64_t size) {
  absl::MutexLock lock(&mutex_);
  if (model_size_ > 0 && model_size_ != size) {
    LOG(WARNING) << "MemoryManager(" << model_identifier_ << "): Model size being reset from " << model_size_ << " to "
                 << size;
    if (pageable_cpu_state_ >= MemoryState::ALLOCATED || gpu_state_ >= MemoryState::ALLOCATED) {
      LOG(ERROR) << "MemoryManager(" << model_identifier_
                 << "): Cannot change model size after memory allocation/borrowing. Current PAGEABLE_CPU state: "
                 << state_to_string(pageable_cpu_state_) << ", GPU state: " << state_to_string(gpu_state_);
      // Optionally return error status here if function signature allowed
      return;
    }
  }
  model_size_ = size;
  VLOG(1) << "MemoryManager(" << model_identifier_ << "): Model size set to " << model_size_ << " bytes.";
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
        absl::StrFormat("MemoryManager(%s): Model size not set before allocation.", model_identifier_));
  }
  // Only GPU allocations require a valid CUDA stream.  PAGEABLE_CPU allocations rely solely on
  // pinned host memory and therefore do not depend on CUDA stream initialisation.
  if (location == ModelLocation::GPU && !stream_initialized_) {
    return absl::FailedPreconditionError(
        absl::StrFormat(
            "MemoryManager(%s): CUDA stream not initialised. Provide a valid device id during MemoryManager construction before GPU allocation.",
            model_identifier_));
  }

  MemoryState* state_ptr = nullptr;
  std::string loc_str = location_to_string(location);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU: {
      state_ptr = &pageable_cpu_state_;
      if (*state_ptr >= MemoryState::ALLOCATED) {
        VLOG(1) << "MemoryManager(" << model_identifier_ << "): PAGEABLE_CPU already in state "
                << state_to_string(*state_ptr) << ". Allocation request ignored.";
        return absl::OkStatus();
      }
      if (*state_ptr != MemoryState::UNALLOCATED) {
        return absl::FailedPreconditionError(
            absl::StrFormat(
                "MemoryManager(%s): Cannot allocate PAGEABLE_CPU memory. Expected UNALLOCATED state, but found %s.",
                model_identifier_,
                state_to_string(*state_ptr)));
      }

      auto region_or = dvmp_->allocate(model_identifier_, model_size_);
      if (region_or.ok()) {
        dvmp_cpu_base_ = region_or->cpu_base;
        dvmp_cpu_bytes_ = region_or->bytes;
      } else if (region_or.status().code() == absl::StatusCode::kAlreadyExists) {
        VLOG(1) << "MemoryManager(" << model_identifier_ << "): DVMP region already exists for PAGEABLE_CPU.";
        // Treat as success and continue.
      } else {
        ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
        return region_or.status();
      }

      // -------------------------------------------------------------------
      // Allocate pinned memory so that H2D/D2H transfers can still leverage
      // fast host buffers exactly the same way the historical CPU path did.
      // -------------------------------------------------------------------
      if (!pinned_pool_) {
        return absl::FailedPreconditionError(
            absl::StrFormat(
                "MemoryManager(%s): No PinnedMemoryPool provided for PAGEABLE_CPU allocation.", model_identifier_));
      }
      pinned_mem_ = std::make_shared<PinnedMemory>();
      int ret = pinned_mem_->allocate(model_size_, pinned_pool_, pinned_memory_timeout_);
      if (ret != 0) {
        pinned_mem_.reset();
        ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
        return absl::ResourceExhaustedError(
            absl::StrFormat(
                "MemoryManager(%s): Failed to allocate %d bytes of pinned host memory for PAGEABLE_CPU (pool error: %d).",
                model_identifier_,
                model_size_,
                ret));
      }
      host_chunk_queue_ = std::make_shared<BatchVector>();
      host_chunk_queue_->init(model_identifier_ + "_host_queue", pinned_mem_->num_chunks());
      LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Allocated " << model_size_
                << " bytes pinned host memory (" << pinned_mem_->num_chunks() << " chunks) for PAGEABLE_CPU.";

      break;
    }
    case ModelLocation::GPU: {
      state_ptr = &gpu_state_;
      if (*state_ptr >= MemoryState::ALLOCATED) {
        VLOG(1) << "MemoryManager(" << model_identifier_ << "): GPU memory already in state "
                << state_to_string(*state_ptr) << ". Allocation request ignored.";
        return absl::OkStatus(); // Already allocated
      }
      if (*state_ptr != MemoryState::UNALLOCATED) {
        return absl::FailedPreconditionError(
            absl::StrFormat(
                "MemoryManager(%s): Cannot allocate GPU memory. Expected UNALLOCATED state, but found %s.",
                model_identifier_,
                state_to_string(*state_ptr)));
      }

      cuda_mem_ = std::make_shared<CudaMemory>();

      // Trace allocation of GPU memory.
      SC_TRACE_SCOPE("allocate_gpu_memory");

      VLOG(1) << "MemoryManager(" << model_identifier_ << "): Attempting direct cudaMalloc for " << model_size_
              << " bytes on device " << local_device_id_ << ".";
      absl::Status alloc_status = cuda_mem_->allocate(model_size_, local_device_id_);
      if (!alloc_status.ok()) {
        LOG(ERROR) << "MemoryManager(" << model_identifier_
                   << "): Direct cudaMalloc failed: " << alloc_status.message();
        cuda_mem_.reset();
        ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
        return absl::ResourceExhaustedError(
            absl::StrFormat(
                "MemoryManager(%s): Failed direct cudaMalloc on device %d: %s",
                model_identifier_,
                local_device_id_,
                alloc_status.message()));
      }

      VLOG(1) << "MemoryManager(" << model_identifier_ << "): Direct cudaMalloc successful (ptr=" << cuda_mem_->get()
              << ").";
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("MemoryManager(%s): Invalid location for allocation: %s", model_identifier_, loc_str));
  }

  // If allocation successful, set state to ALLOCATED
  return set_state_locked(location, MemoryState::ALLOCATED);
}

absl::Status MemoryManager::release_memory(ModelLocation location, bool safe_release) {
  absl::MutexLock lock(&mutex_);

  MemoryState* state_ptr = nullptr;
  absl::CondVar* cond_ptr = nullptr;
  std::string loc_str = location_to_string(location);

  // Set up state and condition variable pointers based on location
  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      state_ptr = &pageable_cpu_state_;
      cond_ptr = &pageable_cpu_cond_;
      break;
    case ModelLocation::GPU:
      state_ptr = &gpu_state_;
      cond_ptr = &gpu_cond_;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("MemoryManager(%s): Invalid location for release: %s", model_identifier_, loc_str));
  }

  // ---------------------------------------------------------------------
  // Unified host-memory path: PAGEABLE_CPU does **not** free the underlying
  // DVMP region or pinned memory buffers.  Releasing simply transitions the
  // state back to UNALLOCATED (or leaves it as-is) so that other components
  // can lock the chunks again.  All CPU-specific resource destruction calls
  // are intentionally skipped.
  // ---------------------------------------------------------------------
  if (location == ModelLocation::PAGEABLE_CPU) {
    VLOG(2) << "MemoryManager(" << model_identifier_
            << "): release_memory called for PAGEABLE_CPU (safe_release=" << safe_release << ")";

    MemoryState current_state = *state_ptr;
    if (current_state == MemoryState::LOADING && safe_release) {
      return absl::FailedPreconditionError(
          absl::StrFormat("MemoryManager(%s): Cannot safely release PAGEABLE_CPU while LOADING.", model_identifier_));
    }

    // For LOADING + unsafe path we mark FAILED, similarly to old behaviour.
    if (current_state == MemoryState::LOADING && !safe_release) {
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::FAILED));
    } else if (current_state != MemoryState::UNALLOCATED) {
      ABSL_CHECK_OK(set_state_locked(location, MemoryState::UNALLOCATED));
    }

    // TODO: Consider dvmp_->unlock_chunks(...) once chunk-level tracking is
    // integrated. For now we rely on external callers to handle unlock.
    return absl::OkStatus();
  }

  // GPU release logic follows
  MemoryState current_state = *state_ptr;
  VLOG(2) << "MemoryManager(" << model_identifier_ << "): Requesting release for " << loc_str
          << " (current state: " << state_to_string(current_state) << ", safe_release: " << safe_release << ")";

  if (current_state <= MemoryState::UNALLOCATED) {
    VLOG(2) << "MemoryManager(" << model_identifier_ << "): Memory for " << loc_str
            << " already released or uninitialized. No action taken.";
    return absl::OkStatus(); // Nothing to release
  }

  if (current_state == MemoryState::LOADING) {
    if (safe_release) {
      LOG(WARNING) << "MemoryManager(" << model_identifier_ << "): Safe release requested for " << loc_str
                   << " while LOADING. Release denied.";
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "MemoryManager(%s): Cannot safely release %s memory while in LOADING state.",
              model_identifier_,
              loc_str));
    }

    VLOG(1) << "MemoryManager(" << model_identifier_ << "): Unsafe release requested for " << loc_str
            << " while LOADING. Attempting to wait briefly...";
    // Wait briefly for ongoing async operations to potentially finish or fail.
    if (cond_ptr->WaitWithTimeout(&mutex_, absl::Milliseconds(500))) {
      // Timeout occurred
      LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Timeout expired while waiting for LOADING state on "
                 << loc_str << " to resolve during unsafe release.";
    }

    // Re-check state after waiting (or if notified early)
    current_state = *state_ptr; // Update current_state
    if (current_state == MemoryState::LOADING) {
      LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Force releasing " << loc_str
                 << " memory while still LOADING after wait. Setting state to FAILED. Potential resource issues.";
      // Force state change to FAILED, hoping async op cleans up eventually or errors out.
      ABSL_CHECK_OK(set_state_locked(
          location, MemoryState::FAILED)); // Mark as failed due to forced release during load
                                           // Proceed to release resources below, but state is now FAILED
    } else {
      VLOG(1) << "MemoryManager(" << model_identifier_ << "): State for " << loc_str << " changed to "
              << state_to_string(current_state) << " during wait. Proceeding with release.";
      // State changed (e.g., to LOADED or FAILED), proceed with release normally.
    }
  }

  // Proceed with GPU resource release
  release_gpu_resources_locked();
  MemoryState target_state = MemoryState::UNALLOCATED;

  // Clear communication registration if releasing the registered location
  if (gpu_comm_registered_) {
    VLOG(2) << "MemoryManager(" << model_identifier_
            << "): Clearing GPU communication registration as memory is being released.";
    gpu_comm_registered_ = false;
  }

  // Set the final state only if we didn't force it to FAILED above during unsafe release.
  // If state is already FAILED, leave it as FAILED.
  if (*state_ptr != MemoryState::FAILED) {
    ABSL_CHECK_OK(set_state_locked(location, target_state));
  } else {
    LOG(WARNING) << "MemoryManager(" << model_identifier_ << "): Resources for " << loc_str
                 << " released, but state remains FAILED due to earlier unsafe release during LOADING.";
  }

  VLOG(1) << "MemoryManager(" << model_identifier_ << "): Finished release process for " << loc_str
          << ". Final state: " << state_to_string(*state_ptr); // Log the actual final state

  return absl::OkStatus();
}

// Private helper
void MemoryManager::release_cpu_resources_locked() {
  // Called with mutex held
  if (pinned_mem_) {
    // PinnedMemory destructor handles returning chunks to the pool (if applicable)
    pinned_mem_.reset();
    LOG(INFO) << "MemoryManager(" << model_identifier_ << "): CPU PinnedMemory object released/reset.";
  }
  if (host_chunk_queue_) {
    host_chunk_queue_.reset();
    VLOG(1) << "MemoryManager(" << model_identifier_ << "): CPU host_chunk_queue released/reset.";
  }
}

// Private helper
void MemoryManager::release_gpu_resources_locked() {
  // Called with mutex held
  if (cuda_mem_) {
    VLOG(2) << "MemoryManager(" << model_identifier_ << "): Releasing allocated GPU memory object (pointer "
            << cuda_mem_->get() << " will be freed/returned to pool by CudaMemory dtor).";
    // CudaMemory destructor handles cudaFree or pool deallocate appropriately
    cuda_mem_.reset();
  }
}

MemoryState MemoryManager::get_state(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);
  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      return pageable_cpu_state_;
    case ModelLocation::GPU:
      return gpu_state_;
    default:
      LOG(WARNING) << "MemoryManager(" << model_identifier_
                   << "): get_state called with invalid location: " << static_cast<int>(location);
      return MemoryState::UNINITIALIZED; // Or FAILED
  }
}

std::vector<void*> MemoryManager::get_pointer(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);
  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      // Return pointer to first chunk if loaded, otherwise null.
      // Documentation should clarify this behaviour.
      if ((pageable_cpu_state_ == MemoryState::LOADED || pageable_cpu_state_ == MemoryState::ALLOCATED) &&
          pinned_mem_ != nullptr && !pinned_mem_->get().empty()) {
        const auto vec = pinned_mem_->get(); // Pointer to first chunk data
        return std::vector<void*>(vec.begin(), vec.end());
      }
      VLOG(2) << "MemoryManager(" << model_identifier_
              << "): get_pointer(PAGEABLE_CPU) returning null. State: " << state_to_string(pageable_cpu_state_)
              << ", PinnedMem valid: " << (pinned_mem_ != nullptr);
      return {};
    case ModelLocation::GPU:
      // Return pointer if memory is allocated, loading, or loaded (pointer valid during streaming and copy)
      if ((gpu_state_ >= MemoryState::ALLOCATED && gpu_state_ != MemoryState::FAILED) && cuda_mem_) {
        auto* const ptr = cuda_mem_->get();
        return std::vector<void*>({ptr});
      }
      VLOG(2) << "MemoryManager(" << model_identifier_
              << "): get_pointer(GPU) return`ing null. State: " << state_to_string(gpu_state_)
              << ", CudaMem valid: " << (cuda_mem_ != nullptr);
      return {};
    default:
      LOG(WARNING) << "MemoryManager(" << model_identifier_
                   << "): get_pointer called with invalid location: " << static_cast<int>(location);
      return {};
  }
}

std::shared_ptr<PinnedMemory> MemoryManager::get_pinned_memory() const {
  absl::MutexLock lock(&mutex_);
  return pinned_mem_;
}

std::shared_ptr<BatchVector> MemoryManager::get_host_chunk_queue() const {
  absl::MutexLock lock(&mutex_);
  return host_chunk_queue_;
}

// Public thread‑safe wrapper
absl::Status MemoryManager::set_state(ModelLocation location, MemoryState new_state) {
  absl::MutexLock lock(&mutex_);
  return set_state_locked(location, new_state);
}

// Internal helper (expects mutex_ held)
absl::Status MemoryManager::set_state_locked(ModelLocation location, MemoryState new_state) {
  MemoryState* state_ptr = nullptr;
  absl::CondVar* cond_ptr = nullptr;
  std::string loc_str = location_to_string(location);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      state_ptr = &pageable_cpu_state_;
      cond_ptr = &pageable_cpu_cond_;
      break;
    case ModelLocation::GPU:
      state_ptr = &gpu_state_;
      cond_ptr = &gpu_cond_;
      break;
    default:
      LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Invalid location for set_state_locked: " << loc_str;
      // Avoid returning Status while holding lock if possible, but here it indicates a programming error.
      return absl::InvalidArgumentError("Invalid location for set_state_locked");
  }

  MemoryState old_state = *state_ptr;
  if (old_state == new_state) {
    VLOG(2) << "MemoryManager(" << model_identifier_ << "): State for " << loc_str << " already "
            << state_to_string(new_state) << ". No change.";
    return absl::OkStatus(); // No change
  }

  // Log the state change
  log_state_change(location, old_state, new_state);
  *state_ptr = new_state;

  // Notify waiters on significant state changes (terminal states, ready states, initial allocation states)
  // This ensures waiters for LOADED, FAILED, ALLOCATED, or UNALLOCATED are woken up.
  cond_ptr->SignalAll();

  return absl::OkStatus();
}

void MemoryManager::log_state_change(ModelLocation loc, MemoryState old_state, MemoryState new_state) const {
  // Assumes mutex is held
  VLOG(1) << "MemoryManager(" << model_identifier_ << "): " << location_to_string(loc) << " state changing from "
          << state_to_string(old_state) << " to " << state_to_string(new_state);
}

// --- Asynchronous Data Copy ---

// Helper function to perform the actual copy from CPU to GPU.
// This function is intended to be run asynchronously.
// It does NOT acquire the manager's mutex directly but operates on captured data.
absl::Status perform_copy_cpu_to_gpu_async_internal(
    const std::string& model_id,
    uint32_t device_id,
    const std::shared_ptr<PinnedMemory>& pinned_mem_copy, // Captured shared_ptr keeps memory alive
    void* gpu_ptr,
    size_t total_size,
    cudaStream_t stream) {
  if (gpu_ptr == nullptr || !pinned_mem_copy || pinned_mem_copy->get().empty() || total_size == 0) {
    LOG(ERROR) << "MemoryManager(" << model_id
               << "): Invalid arguments for internal CPU->GPU copy. GPU Ptr: " << gpu_ptr
               << ", PinnedMem Valid: " << (pinned_mem_copy != nullptr) << ", Total Size: " << total_size;
    return absl::InternalError("Invalid arguments provided for internal CPU->GPU copy.");
  }

  LOG(INFO) << "MemoryManager(" << model_id << "): Starting async CPU->GPU copy (" << total_size << " bytes) on stream "
            << stream << "...";

  auto device_status = stepcast::cuda::set_device(device_id);
  if (!device_status.ok()) {
    LOG(ERROR) << "MemoryManager(" << model_id << "): set_device(" << device_id
               << ") failed before CPU->GPU copy: " << device_status.message();
    return absl::InternalError(absl::StrFormat("set_device failed: %s", device_status.message()));
  }

  size_t offset = 0;
  auto& cpu_chunks = pinned_mem_copy->get();
  size_t chunk_size = pinned_mem_copy->chunk_size();
  size_t num_chunks = pinned_mem_copy->num_chunks();

  for (size_t i = 0; i < num_chunks; ++i) {
    if (offset >= total_size) {
      LOG(WARNING) << "MemoryManager(" << model_id << "): CPU->GPU copy offset (" << offset << ") reached total size ("
                   << total_size << ") prematurely at chunk " << i << "/" << num_chunks;
      break; // Avoid over-copying if total_size is less than chunk sum
    }
    size_t current_chunk_size = std::min(chunk_size, total_size - offset);
    if (current_chunk_size == 0) {
      break; // Should not happen if offset < total_size
    }

    void* src_host = cpu_chunks[i];
    void* dst_device = static_cast<char*>(gpu_ptr) + offset;

    VLOG(2) << "MemoryManager(" << model_id << "): Copying chunk " << i << " (" << current_chunk_size
            << " bytes) from CPU:" << src_host << " to GPU:" << dst_device << " on stream " << stream;
    auto memcpy_status =
        stepcast::cuda::memcpy_async(dst_device, src_host, current_chunk_size, cudaMemcpyHostToDevice, stream);
    if (!memcpy_status.ok()) {
      LOG(ERROR) << "MemoryManager(" << model_id << "): memcpy_async (H2D) failed for chunk " << i << ": "
                 << memcpy_status.message();
      // Attempt to synchronize stream before returning error to potentially catch earlier errors?
      auto sync_status = stepcast::cuda::stream_synchronize(stream); // Best effort sync on error
      if (!sync_status.ok()) {
        LOG(ERROR) << "MemoryManager(" << model_id << "): Best effort stream sync failed: " << sync_status.message();
      }
      return absl::InternalError(
          absl::StrFormat("memcpy_async (H2D) failed for chunk %d: %s", i, memcpy_status.message()));
    }
    offset += current_chunk_size;
  }

  if (offset != total_size) {
    LOG(WARNING) << "MemoryManager(" << model_id << "): CPU->GPU copy finished, but total bytes copied (" << offset
                 << ") does not match expected total size (" << total_size << ").";
    // Continue to synchronize, but log potential issue.
  }

  // Synchronize the specific stream to ensure copy completion before future resolves
  VLOG(1) << "MemoryManager(" << model_id << "): Synchronizing stream " << stream << " after CPU->GPU copy.";
  auto sync_status = stepcast::cuda::stream_synchronize(stream);
  if (!sync_status.ok()) {
    LOG(ERROR) << "MemoryManager(" << model_id
               << " ): stream_synchronize failed after CPU->GPU copy: " << sync_status.message();
    return absl::InternalError(
        absl::StrFormat("stream_synchronize failed after CPU->GPU copy: %s", sync_status.message()));
  }

  LOG(INFO) << "MemoryManager(" << model_id << "): Async CPU->GPU copy completed successfully on stream " << stream
            << ".";
  return absl::OkStatus();
}

// Helper function to perform the actual copy from GPU to CPU.
// Similar structure to the CPU->GPU helper.
absl::Status perform_copy_gpu_to_cpu_async_internal(
    const std::string& model_id,
    uint32_t device_id,
    void* gpu_ptr,
    const std::shared_ptr<PinnedMemory>& pinned_mem_copy, // Captured shared_ptr
    size_t total_size,
    cudaStream_t stream) {
  if (gpu_ptr == nullptr || !pinned_mem_copy || pinned_mem_copy->get().empty() || total_size == 0) {
    LOG(ERROR) << "MemoryManager(" << model_id
               << "): Invalid arguments for internal GPU->CPU copy. GPU Ptr: " << gpu_ptr
               << ", PinnedMem Valid: " << (pinned_mem_copy != nullptr) << ", Total Size: " << total_size;
    return absl::InternalError("Invalid arguments provided for internal GPU->CPU copy.");
  }

  LOG(INFO) << "MemoryManager(" << model_id << "): Starting async GPU->CPU copy (" << total_size << " bytes) on stream "
            << stream << "...";

  auto device_status = stepcast::cuda::set_device(device_id);
  if (!device_status.ok()) {
    LOG(ERROR) << "MemoryManager(" << model_id << "): set_device(" << device_id
               << ") failed before GPU->CPU copy: " << device_status.message();
    return absl::InternalError(absl::StrFormat("set_device failed: %s", device_status.message()));
  }

  size_t offset = 0;
  auto& cpu_chunks = pinned_mem_copy->get();
  size_t chunk_size = pinned_mem_copy->chunk_size();
  size_t num_chunks = pinned_mem_copy->num_chunks();

  for (size_t i = 0; i < num_chunks; ++i) {
    if (offset >= total_size) {
      LOG(WARNING) << "MemoryManager(" << model_id << "): GPU->CPU copy offset (" << offset << ") reached total size ("
                   << total_size << ") prematurely at chunk " << i << "/" << num_chunks;
      break; // Avoid over-copying
    }
    size_t current_chunk_size = std::min(chunk_size, total_size - offset);
    if (current_chunk_size == 0) {
      break;
    }

    void* src_device = static_cast<char*>(gpu_ptr) + offset;
    void* dst_host = cpu_chunks[i];

    VLOG(2) << "MemoryManager(" << model_id << "): Copying chunk " << i << " (" << current_chunk_size
            << " bytes) from GPU:" << src_device << " to CPU:" << dst_host << " on stream " << stream;
    auto memcpy_status =
        stepcast::cuda::memcpy_async(dst_host, src_device, current_chunk_size, cudaMemcpyDeviceToHost, stream);
    if (!memcpy_status.ok()) {
      LOG(ERROR) << "MemoryManager(" << model_id << "): memcpy_async (D2H) failed for chunk " << i << ": "
                 << memcpy_status.message();
      auto sync_status = stepcast::cuda::stream_synchronize(stream); // Best effort sync
      if (!sync_status.ok()) {
        LOG(ERROR) << "MemoryManager(" << model_id << "): Best effort stream sync failed: " << sync_status.message();
      }
      return absl::InternalError(
          absl::StrFormat("memcpy_async (D2H) failed for chunk %d: %s", i, memcpy_status.message()));
    }
    offset += current_chunk_size;
  }

  if (offset != total_size) {
    LOG(WARNING) << "MemoryManager(" << model_id << "): GPU->CPU copy finished, but total bytes copied (" << offset
                 << ") does not match expected total size (" << total_size << ").";
  }

  // Synchronize the stream
  VLOG(1) << "MemoryManager(" << model_id << "): Synchronizing stream " << stream << " after GPU->CPU copy.";
  auto sync_status = stepcast::cuda::stream_synchronize(stream);
  if (!sync_status.ok()) {
    LOG(ERROR) << "MemoryManager(" << model_id
               << "): stream_synchronize failed after GPU->CPU copy: " << sync_status.message();
    return absl::InternalError(
        absl::StrFormat("stream_synchronize failed after GPU->CPU copy: %s", sync_status.message()));
  }

  LOG(INFO) << "MemoryManager(" << model_id << "): Async GPU->CPU copy completed successfully on stream " << stream
            << ".";
  return absl::OkStatus();
}

std::future<absl::Status> MemoryManager::copy_data_async(ModelLocation source, ModelLocation destination) {
  // --- Phase 1: Acquire Lock, Check State, Prepare Data for Capture ---
  std::shared_ptr<PinnedMemory> pinned_mem_capture;
  std::shared_ptr<CudaMemory> cuda_mem_capture;
  size_t size_capture = 0;
  cudaStream_t stream_capture = nullptr;
  uint32_t device_id_capture = 0;
  std::string model_id_capture;
  std::string src_str = location_to_string(source);
  std::string dst_str = location_to_string(destination);

  { // Mutex Lock Scope
    absl::MutexLock lock(&mutex_);

    if (!stream_initialized_ || copy_stream_ == nullptr) {
      LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Cannot initiate copy. CUDA stream is not valid.";
      return std::async(std::launch::deferred, [] { return absl::InternalError("CUDA stream not initialized."); });
    }

    // Normalise host locations (only PAGEABLE_CPU is supported)
    bool src_is_host = (source == ModelLocation::PAGEABLE_CPU);
    bool dst_is_host = (destination == ModelLocation::PAGEABLE_CPU);

    MemoryState src_state = src_is_host ? pageable_cpu_state_ : gpu_state_;
    MemoryState dst_state = dst_is_host ? pageable_cpu_state_ : gpu_state_;

    LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Requesting async copy from " << src_str
              << " (state: " << state_to_string(src_state) << ") to " << dst_str
              << " (state: " << state_to_string(dst_state) << ")";

    // --- Basic State Checks ---
    if (src_state != MemoryState::LOADED) {
      return std::async(std::launch::deferred, [id = model_identifier_, src_str] { // Capture id
        return absl::FailedPreconditionError(
            absl::StrFormat("MemoryManager(%s): Source %s is not in LOADED state for copy.", id, src_str));
      });
    }
    // Allow copy destination to be ALLOCATED
    if (dst_state != MemoryState::ALLOCATED) {
      return std::async(std::launch::deferred, [id = model_identifier_, dst_str] { // Capture id
        return absl::FailedPreconditionError(
            absl::StrFormat("MemoryManager(%s): Destination %s is not in ALLOCATED state for copy.", id, dst_str));
      });
    }

    // --- Capture Necessary Data Under Lock ---
    // Check if underlying memory pointers are valid before capturing
    if (src_is_host && (!pinned_mem_ || pinned_mem_->get().empty())) {
      return std::async(std::launch::deferred, [id = model_identifier_] {
        return absl::InternalError("MemoryManager(" + id + "): Source PAGEABLE_CPU memory is invalid for copy.");
      });
    }
    if (source == ModelLocation::GPU && (!cuda_mem_ || cuda_mem_->get() == nullptr)) {
      return std::async(std::launch::deferred, [id = model_identifier_] {
        return absl::InternalError("MemoryManager(" + id + "): Source GPU memory is invalid for copy.");
      });
    }
    if (dst_is_host && (!pinned_mem_ || pinned_mem_->get().empty())) {
      return std::async(std::launch::deferred, [id = model_identifier_] {
        return absl::InternalError("MemoryManager(" + id + "): Destination PAGEABLE_CPU memory is invalid for copy.");
      });
    }
    if (destination == ModelLocation::GPU && (!cuda_mem_ || cuda_mem_->get() == nullptr)) {
      return std::async(std::launch::deferred, [id = model_identifier_] {
        return absl::InternalError("MemoryManager(" + id + "): Destination GPU memory is invalid for copy.");
      });
    }

    pinned_mem_capture = pinned_mem_; // Copy shared_ptr
    cuda_mem_capture = cuda_mem_; // Copy shared_ptr
    size_capture = model_size_;
    stream_capture = copy_stream_;
    device_id_capture = local_device_id_;
    model_id_capture = model_identifier_; // Copy string

    // Note: lock_chunks/unlock_chunks calls removed per review feedback
    // DVMP chunk locking is handled at a higher layer if needed

    // --- Mark Destination as Loading ---
    absl::Status state_status = set_state_locked(destination, MemoryState::LOADING);
    if (!state_status.ok()) {
      // If setting state fails, return the error immediately
      LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Failed to set destination " << dst_str
                 << " state to LOADING: " << state_status;
      // Create a future that holds the status
      return std::async(std::launch::deferred, [state_status] { return state_status; });
    }
  } // Mutex Released

  // --- Phase 2: Launch Asynchronous Task ---
  return std::async(
      std::launch::async,
      [this, // Capture this *only* for final state update and logging within lambda
       source,
       destination, // Captured location enums
       // Captured data needed for the copy operation itself:
       pinned_mem_copy = std::move(pinned_mem_capture),
       cuda_mem_copy = std::move(cuda_mem_capture),
       total_size = size_capture,
       stream = stream_capture,
       device_id = device_id_capture,
       model_id = std::move(model_id_capture)]() -> absl::Status {
        absl::Status copy_status;
        std::string dst_str_async = location_to_string(destination); // For logging inside lambda

        // Select and execute the correct internal copy function
        bool src_host_async = (source == ModelLocation::PAGEABLE_CPU);
        bool dst_host_async = (destination == ModelLocation::PAGEABLE_CPU);

        if (src_host_async && !dst_host_async) { // Host -> GPU
          copy_status = perform_copy_cpu_to_gpu_async_internal(
              model_id, device_id, pinned_mem_copy, cuda_mem_copy ? cuda_mem_copy->get() : nullptr, total_size, stream);
        } else if (!src_host_async && dst_host_async) { // GPU -> Host
          copy_status = perform_copy_gpu_to_cpu_async_internal(
              model_id, device_id, cuda_mem_copy ? cuda_mem_copy->get() : nullptr, pinned_mem_copy, total_size, stream);
        } else {
          LOG(ERROR) << "MemoryManager(" << model_id
                     << "): Invalid source/destination combination for async copy task: " << static_cast<int>(source)
                     << " -> " << static_cast<int>(destination);
          copy_status = absl::InvalidArgumentError("Unsupported copy direction in async task.");
        }

        // --- Phase 3: Acquire Lock Again, Update Final State ---
        {
          // Acquire lock to safely update state
          absl::MutexLock final_lock(&this->mutex_);
          MemoryState final_state = copy_status.ok() ? MemoryState::LOADED : MemoryState::FAILED;
          LOG(INFO) << "MemoryManager(" << this->model_identifier_ << "): Async copy to " << dst_str_async
                    << " finished. Operation status: " << copy_status << ". Attempting to set final state.";
          // Check the current state under lock before finalizing
          bool dst_is_host_final = (destination == ModelLocation::PAGEABLE_CPU);
          MemoryState current_dst_state = dst_is_host_final ? this->pageable_cpu_state_ : this->gpu_state_;
          if (current_dst_state == MemoryState::LOADING) {
            ABSL_CHECK_OK(this->set_state_locked(destination, final_state));
            LOG(INFO) << "MemoryManager(" << this->model_identifier_ << "): Final state for " << dst_str_async
                      << " set to " << state_to_string(final_state) << ".";
          } else {
            LOG(WARNING) << "MemoryManager(" << this->model_identifier_ << "): State of " << dst_str_async
                         << " was no longer LOADING during async copy finalization (current state: "
                         << state_to_string(current_dst_state)
                         << "). Final state not updated. Copy operation status was: " << copy_status;
          }
        }
        // Mandatory CPU memory release after successful CPU→GPU copy (per RFC 0001 §4.3)
        bool src_host_release = (source == ModelLocation::PAGEABLE_CPU);
        if (copy_status.ok() && src_host_release && destination == ModelLocation::GPU) {
          LOG(INFO) << "MemoryManager(" << this->model_identifier_
                    << "): Releasing host pinned memory after successful H→D copy (mandatory per RFC 0001).";
          
          // DVMP eviction for physical page reclamation
          if (this->dvmp_ && this->dvmp_cpu_base_ != nullptr) {
            // Trigger eviction to reclaim physical pages while retaining virtual mapping
            size_t evicted = this->dvmp_->evict_tail_bytes(this->model_identifier_, this->model_size_);
            LOG(INFO) << "MemoryManager(" << this->model_identifier_
                      << "): Evicted " << evicted << " bytes from DVMP after GPU copy.";
          }
          
          // Release CPU resources and mark state as UNALLOCATED
          {
            absl::MutexLock release_lock(&this->mutex_);
            this->release_cpu_resources_locked();
            ABSL_CHECK_OK(this->set_state_locked(ModelLocation::PAGEABLE_CPU, MemoryState::UNALLOCATED));
          }
          
          LOG(INFO) << "MemoryManager(" << this->model_identifier_
                    << "): CPU memory release completed.";
        }

        return copy_status; // Return the status of the copy operation itself
      });
}

absl::Status MemoryManager::wait_for_state(ModelLocation location, MemoryState target_state, absl::Duration timeout) {
  absl::MutexLock lock(&mutex_);

  MemoryState* state_ptr = nullptr;
  absl::CondVar* cond_ptr = nullptr;
  std::string loc_str = location_to_string(location);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      state_ptr = &pageable_cpu_state_;
      cond_ptr = &pageable_cpu_cond_;
      break;
    case ModelLocation::GPU:
      state_ptr = &gpu_state_;
      cond_ptr = &gpu_cond_;
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("MemoryManager(%s): Invalid location for wait_for_state: %s", model_identifier_, loc_str));
  }

  VLOG(1) << "MemoryManager(" << model_identifier_ << "): Waiting for " << loc_str << " to reach state "
          << state_to_string(target_state) << " (current: " << state_to_string(*state_ptr) << ", timeout: " << timeout
          << ")";

  absl::Time deadline = (timeout == absl::InfiniteDuration()) ? absl::InfiniteFuture() : absl::Now() + timeout;

  // Wait loop: Continue waiting as long as the current state is NOT the target state AND NOT the FAILED state.
  while (*state_ptr != target_state && *state_ptr != MemoryState::FAILED) {
    if (absl::Now() >= deadline) {
      // Double check state right after deadline check before declaring timeout
      if (*state_ptr != target_state && *state_ptr != MemoryState::FAILED) {
        LOG(WARNING) << "MemoryManager(" << model_identifier_ << "): Timeout waiting for " << loc_str
                     << " to reach state " << state_to_string(target_state)
                     << ". Current state: " << state_to_string(*state_ptr);
        return absl::DeadlineExceededError(
            absl::StrFormat("Timeout waiting for %s state %s", loc_str, state_to_string(target_state)));
      } // State changed just before timeout check, break loop to return correct status below.
      break;
    }
    // WaitWithDeadline returns true if the deadline was exceeded.
    // We handle the deadline check explicitly above for clarity and immediate check after wake.
    cond_ptr->WaitWithDeadline(&mutex_, deadline);
    // Loop condition will re-evaluate state after waking up.
  }

  // Check final state after wait loop exits
  if (*state_ptr == target_state) {
    VLOG(1) << "MemoryManager(" << model_identifier_ << "): Wait successful. " << loc_str << " reached target state "
            << state_to_string(target_state);
    return absl::OkStatus();
  }
  if (*state_ptr == MemoryState::FAILED) {
    LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Wait completed because " << loc_str
               << " reached FAILED state while waiting for " << state_to_string(target_state);
    return absl::FailedPreconditionError(absl::StrFormat("%s operation failed", loc_str));
  }
  // Should be unreachable if loop logic is correct
  LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Wait loop exited with unexpected state "
             << state_to_string(*state_ptr) << " for " << loc_str;
  return absl::InternalError("Unexpected state after wait loop.");
}

absl::StatusOr<CommRegistrationInfo> MemoryManager::enable_remote_memory_access(
    ModelLocation location,
    stepcast::communicator::CommunicateEngine& comm_engine) {
  absl::MutexLock lock(&mutex_);

  // Check if already registered for the specific location
  bool* registered_ptr = nullptr;
  CommRegistrationInfo* cached_info_ptr = nullptr;

  if (location == ModelLocation::PAGEABLE_CPU) {
    registered_ptr = &pageable_cpu_comm_registered_;
    cached_info_ptr = &pageable_cpu_comm_registration_info_;
  } else if (location == ModelLocation::GPU) {
    registered_ptr = &gpu_comm_registered_;
    cached_info_ptr = &gpu_comm_registration_info_;
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "MemoryManager(%s): Invalid location specified for Comm registration: %d",
            model_identifier_,
            static_cast<int>(location)));
  }

  if (*registered_ptr) {
    LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Memory for " << location_to_string(location)
              << " already registered for communication. Returning cached registration info.";
    return *cached_info_ptr;
  }

  if (model_size_ == 0) {
    return absl::FailedPreconditionError(
        absl::StrFormat("MemoryManager(%s): Model size is 0, cannot register memory.", model_identifier_));
  }

  CommRegistrationInfo reg_info;
  reg_info.model_size = model_size_;
  reg_info.location = location;

  std::string location_str;
  reg_info.comm_dev_type = stepcast::communicator::COMMUNICATE_ENGINE_DEV_CPU;

  if (location == ModelLocation::PAGEABLE_CPU) {
    location_str = "PAGEABLE_CPU";
    reg_info.device_id = 1; // Explicitly 1 for CPU

    if (pageable_cpu_state_ != MemoryState::LOADED) {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "MemoryManager(%s): PAGEABLE_CPU memory must be in LOADED state for Comm registration (current: %s).",
              model_identifier_,
              state_to_string(pageable_cpu_state_)));
    }
    if (!pinned_mem_ || pinned_mem_->get().empty()) {
      return absl::InternalError(
          absl::StrFormat(
              "MemoryManager(%s): Invalid or empty PAGEABLE_CPU PinnedMemory object for Comm registration.",
              model_identifier_));
    }

    LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Registering PAGEABLE_CPU memory ("
              << pinned_mem_->num_chunks() << " chunks) for Comm via CommunicateEngine.";

    auto& cpu_chunks = pinned_mem_->get();
    size_t chunk_size = pinned_mem_->chunk_size();
    size_t num_chunks = pinned_mem_->num_chunks();
    size_t registered_size = 0;

    // Reserve space for efficiency
    reg_info.buffer_addresses.reserve(num_chunks);
    reg_info.buffer_sizes.reserve(num_chunks);
    reg_info.remote_memory_keys.reserve(num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
      void* chunk_ptr = cpu_chunks[i];
      uint64_t chunk_addr = reinterpret_cast<uint64_t>(chunk_ptr);
      // Calculate the actual size of the current chunk (last chunk might be smaller)
      size_t current_chunk_bytes = std::min(chunk_size, model_size_ - registered_size);
      if (current_chunk_bytes == 0) {
        LOG(WARNING) << "MemoryManager(" << model_identifier_ << "): Reached 0 chunk size at index " << i
                     << " while registering CPU chunks. Stopping registration.";
        break; // Avoid registering zero-sized chunks
      }

      // Construct a unique key for each chunk
      auto tensor_name_for_comm = absl::StrFormat("%s_%s_chunk_%d", model_identifier_, location_str, i);

      LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Registering Chunk " << i << "/" << num_chunks
                << " Name: '" << tensor_name_for_comm << "', Addr: " << chunk_ptr << " (" << chunk_addr << ")"
                << ", Size: " << current_chunk_bytes << ", DevType: " << reg_info.comm_dev_type
                << ", DevId: " << reg_info.device_id;

      // Call the communicator engine's registration function for the chunk
      auto ret = comm_engine.register_tensor(
          tensor_name_for_comm, chunk_addr, current_chunk_bytes, reg_info.comm_dev_type, reg_info.device_id);

      if (!ret.ok()) {
        LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Failed to register CPU chunk " << i << " tensor '"
                   << tensor_name_for_comm << "' with CommunicateEngine. Error code: " << ret;
        // TODO: Consider unregistering successfully registered chunks before returning error.
        return absl::InternalError(
            absl::StrFormat(
                "Failed to register CPU chunk %d tensor %s via CommunicateEngine", i, tensor_name_for_comm));
      }

      // Store registration info
      reg_info.buffer_addresses.push_back(chunk_addr);
      reg_info.buffer_sizes.push_back(current_chunk_bytes);
      reg_info.remote_memory_keys.push_back(tensor_name_for_comm);

      registered_size += current_chunk_bytes;
      if (registered_size >= model_size_) {
        // Stop if we have registered enough bytes to cover the model size
        LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Registered total of " << registered_size
                  << " bytes across " << (i + 1) << " chunks, matching model size " << model_size_
                  << ". Stopping chunk registration.";
        break;
      }
    }
    if (registered_size < model_size_) {
      LOG(WARNING) << "MemoryManager(" << model_identifier_
                   << "): Finished registering CPU chunks, but total registered size (" << registered_size
                   << ") is less than model size (" << model_size_ << ").";
      // This might indicate an issue with chunk calculation or model size.
      // Return an error because registration is incomplete/inconsistent
      return absl::InternalError(
          absl::StrFormat(
              "Incomplete CPU memory registration: registered %d bytes, expected %d", registered_size, model_size_));
    }

    LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Successfully registered all "
              << reg_info.buffer_addresses.size() << " CPU memory chunks for Comm.";
  } else if (location == ModelLocation::GPU) {
    location_str = "GPU";
    reg_info.device_id = local_device_id_; // Use actual device ID for GPU
    reg_info.comm_dev_type = stepcast::communicator::COMMUNICATE_ENGINE_DEV_GPU;

    // GPU must be ready (LOADED) to be registered
    if (gpu_state_ != MemoryState::LOADED) {
      return absl::FailedPreconditionError(
          absl::StrFormat(
              "MemoryManager(%s): GPU memory must be in LOADED state for Comm registration (current: %s).",
              model_identifier_,
              state_to_string(gpu_state_)));
    }
    if (!cuda_mem_ || cuda_mem_->get() == nullptr) {
      return absl::InternalError(
          absl::StrFormat(
              "MemoryManager(%s): Invalid GPU memory object (null=%d) or size (%d) for Comm registration.",
              model_identifier_,
              cuda_mem_ == nullptr || cuda_mem_->get() == nullptr,
              model_size_));
    }

    void* ptr_to_register = cuda_mem_->get();
    auto addr_to_register = reinterpret_cast<uint64_t>(ptr_to_register);
    size_t size_to_register = model_size_;

    // Construct a unique key for the single GPU buffer registration
    // Using "chunk0" suffix for consistency, even though it's one block.
    auto tensor_name_for_comm =
        absl::StrFormat("%s_%s_dev%d_chunk0", model_identifier_, location_str, reg_info.device_id);

    VLOG(1) << "MemoryManager(" << model_identifier_
            << "): Registering GPU memory (1 chunk) for Comm via CommunicateEngine. Name: '" << tensor_name_for_comm
            << "', Addr: " << ptr_to_register << " (" << addr_to_register << ")"
            << ", Size: " << size_to_register << ", DevType: " << reg_info.comm_dev_type
            << ", DevId: " << reg_info.device_id;

    // Call the communicator engine's registration function
    auto ret = comm_engine.register_tensor(
        tensor_name_for_comm, addr_to_register, size_to_register, reg_info.comm_dev_type, reg_info.device_id);

    if (!ret.ok()) {
      LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Failed to register GPU tensor '"
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
  *registered_ptr = true;
  *cached_info_ptr = reg_info;

  return reg_info; // Return the populated struct
}

absl::Status MemoryManager::finalize_load_state(ModelLocation location, const absl::Status& final_status) {
  absl::MutexLock lock(&mutex_);

  MemoryState* state_ptr = nullptr;
  std::string loc_str = location_to_string(location);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      state_ptr = &pageable_cpu_state_;
      break;
    case ModelLocation::GPU:
      state_ptr = &gpu_state_;
      break;
    default:
      LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): Invalid location for finalize_load_state: " << loc_str;
      return absl::InvalidArgumentError("Invalid location for finalize_load_state");
  }

  MemoryState current_state = *state_ptr;

  if (current_state == MemoryState::LOADING) {
    MemoryState target_final_state = final_status.ok() ? MemoryState::LOADED : MemoryState::FAILED;
    VLOG(1) << "MemoryManager(" << model_identifier_ << "): Finalizing operation for " << loc_str
            << ". Operation status: " << final_status << ". Setting state from LOADING to "
            << state_to_string(target_final_state);
    // Use set_state_locked to update state and notify condition variables
    return set_state_locked(location, target_final_state); // Already under lock
  } // This is not necessarily an error, the state might have been changed by release_memory or another operation.
  LOG(WARNING) << "MemoryManager(" << model_identifier_ << "): Finalize requested for " << loc_str
               << ", but state was not LOADING (current: " << state_to_string(current_state)
               << "). State not updated. Operation status was: " << final_status;
  // Return OkStatus because the finalization logic itself didn't fail, even if no state change occurred.
  // The caller should primarily rely on the future's status (which is final_status).
  return absl::OkStatus();
}

// --- CPU Chunk Size Getter ---
size_t MemoryManager::get_cpu_chunk_size() const {
  absl::MutexLock lock(&mutex_); // Lock needed to safely access pinned_mem_
  if (pinned_mem_ && !pinned_mem_->get().empty()) {
    return pinned_mem_->chunk_size();
  }
  return 0; // Return 0 if not allocated or no chunks
}

absl::StatusOr<cudaIpcMemHandle_t> MemoryManager::get_cuda_ipc_handle() const {
  absl::MutexLock lock(&mutex_);

  // A CUDA IPC handle can be created as soon as the underlying GPU buffer is
  // allocated, even if an asynchronous H2D copy is still in progress.
  // Therefore we permit ALLOCATED and LOADING states (as well as the final
  // LOADED state) instead of requiring the operation to have fully
  // completed before exposing the handle.
  if (gpu_state_ != MemoryState::LOADED && gpu_state_ != MemoryState::ALLOCATED && gpu_state_ != MemoryState::LOADING) {
    return absl::FailedPreconditionError("GPU memory is not yet allocated");
  }

  if (cuda_mem_ == nullptr) {
    return absl::NotFoundError("CudaMemory object is not initialised");
  }

  return cuda_mem_->get_handle();
}

bool MemoryManager::is_comm_registered(ModelLocation location) const {
  absl::MutexLock lock(&mutex_);

  switch (location) {
    case ModelLocation::PAGEABLE_CPU:
      return pageable_cpu_comm_registered_;
    case ModelLocation::GPU:
      return gpu_comm_registered_;
    default:
      LOG(WARNING) << "MemoryManager(" << model_identifier_
                   << "): is_comm_registered called with invalid location: " << static_cast<int>(location);
      return false;
  }
}

// ---------------------------------------------------------------------------
// Streaming buffer helper implementations
// ---------------------------------------------------------------------------
absl::Status MemoryManager::allocate_buffer_pool(size_t num_chunks) {
  absl::MutexLock lock(&mutex_);
  if (!pinned_pool_) {
    return absl::FailedPreconditionError("No pinned memory pool available in MemoryManager");
  }
  if (streaming_buffer_) {
    return absl::AlreadyExistsError("Streaming buffer already allocated");
  }
  if (num_chunks == 0) {
    return absl::InvalidArgumentError("num_chunks must be > 0 for allocate_buffer_pool");
  }
  size_t chunk_size = pinned_pool_->chunk_size();
  streaming_buffer_ = std::make_shared<StreamingPinnedBuffer>(num_chunks, chunk_size, pinned_pool_);
  absl::Status st = streaming_buffer_->initialize(pinned_memory_timeout_);
  if (!st.ok()) {
    streaming_buffer_.reset();
    return st;
  }
  VLOG(1) << "MemoryManager(" << model_identifier_ << "): Allocated streaming buffer with " << num_chunks
          << " chunks (chunk_size=" << chunk_size << ")";
  return absl::OkStatus();
}

absl::Status MemoryManager::release_buffer_pool() {
  absl::MutexLock lock(&mutex_);
  if (!streaming_buffer_) {
    return absl::OkStatus();
  }
  absl::Status st = streaming_buffer_->release();
  streaming_buffer_.reset();
  return st;
}

std::shared_ptr<StreamingPinnedBuffer> MemoryManager::get_streaming_buffer() const {
  absl::MutexLock lock(&mutex_);
  return streaming_buffer_;
}

size_t MemoryManager::get_max_buffer_bytes() const {
  absl::MutexLock lock(&mutex_);
  return max_buffer_bytes_;
}

size_t MemoryManager::get_pool_chunk_size() const {
  absl::MutexLock lock(&mutex_);
  if (pinned_pool_) {
    return pinned_pool_->chunk_size();
  }
  return 0;
}

// ---------------------------------------------------------------------------
// DVMP pageable CPU region allocation helper
// ---------------------------------------------------------------------------
absl::StatusOr<memory::DistributedMemoryPool::VirtualRegion> MemoryManager::allocate_pageable_cpu_region() {
  absl::MutexLock lock(&mutex_);

  if (model_size_ == 0) {
    return absl::FailedPreconditionError(
        absl::StrFormat("MemoryManager(%s): Model size not set before DVMP allocation.", model_identifier_));
  }

  // Attempt allocation. This call may return kAlreadyExists if another loader
  // has already reserved the region for the same model_id.
  auto region_or = dvmp_->allocate(model_identifier_, model_size_);

  if (region_or.ok()) {
    // Cache base address information for potential future use.
    dvmp_cpu_base_ = region_or->cpu_base;
    dvmp_cpu_bytes_ = region_or->bytes;
    VLOG(1) << "MemoryManager(" << model_identifier_ << "): Reserved pageable CPU region of " << region_or->bytes
            << " bytes at " << region_or->cpu_base << " via DVMP.";
  } else if (region_or.status().code() == absl::StatusCode::kAlreadyExists) {
    VLOG(1) << "MemoryManager(" << model_identifier_
            << "): DVMP region already exists for model. Skipping reservation.";
  } else {
    LOG(ERROR) << "MemoryManager(" << model_identifier_ << "): DVMP allocation failed: " << region_or.status();
  }

  return region_or;
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

  // Use provided stream or our internal copy_stream_.
  cudaStream_t stream_to_use = ext_stream;
  {
    absl::MutexLock lock(&mutex_);
    if (stream_to_use == nullptr) {
      stream_to_use = copy_stream_;
      if (!stream_initialized_) {
        // Lazily create stream.
        auto st = stepcast::cuda::stream_create_with_flags(&copy_stream_, /*non-blocking*/ 0);
        if (!st.ok()) {
          return st;
        }
        stream_initialized_ = true;
        stream_to_use = copy_stream_;
      }
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
    registered_ptr = &pageable_cpu_comm_registered_;
    info_ptr = &pageable_cpu_comm_registration_info_;
  } else if (location == ModelLocation::GPU) {
    registered_ptr = &gpu_comm_registered_;
    info_ptr = &gpu_comm_registration_info_;
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "MemoryManager(%s): Invalid location specified for Comm unregistration: %d",
            model_identifier_,
            static_cast<int>(location)));
  }

  if (!*registered_ptr) {
    // Nothing was registered – treat as a no-op.
    LOG(INFO) << "MemoryManager(" << model_identifier_ << "): No existing Comm registration for "
              << location_to_string(location) << ". Nothing to unregister.";
    return absl::OkStatus();
  }

  absl::Status first_error;

  for (const auto& key : info_ptr->remote_memory_keys) {
    absl::Status st = comm_engine.unregister_tensor(key);
    if (!st.ok()) {
      LOG(WARNING) << "MemoryManager(" << model_identifier_ << "): Failed to unregister tensor '" << key
                   << "' from CommunicateEngine. Status: " << st;
      if (first_error.ok()) {
        first_error = st;
      }
    } else {
      VLOG(2) << "MemoryManager(" << model_identifier_ << "): Unregistered tensor '" << key << "'";
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

  LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Successfully unregistered memory at "
            << location_to_string(location) << " for communication.";
  return absl::OkStatus();
}

// --- DVMP accessor implementation ---
memory::DistributedMemoryPool* stepcast::store::MemoryManager::get_dvmp() const {
  absl::MutexLock lock(&mutex_);
  return dvmp_.get();
}

void* MemoryManager::get_dvmp_cpu_base() const {
  absl::MutexLock lock(&mutex_);
  return dvmp_cpu_base_;
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

  // Create shared pointer to DVMP with a non-owning deleter
  // This allows sharing between MemoryManager and UnifiedModelMemory
  auto shared_dvmp = std::shared_ptr<memory::DistributedMemoryPool>(dvmp_.get(), [](memory::DistributedMemoryPool*) {
    // Non-owning deleter - MemoryManager retains ownership
  });

  unified_memory_ = std::make_shared<UnifiedModelMemory>(shared_dvmp);

  // Allocate via unified memory (which will use DVMP internally)
  auto status = unified_memory_->allocate(instance_key_, model_size_);
  if (!status.ok()) {
    unified_memory_.reset();
    return status;
  }

  // Update our internal tracking
  dvmp_cpu_base_ = unified_memory_->get_cpu_base_ptr(instance_key_);
  dvmp_cpu_bytes_ = model_size_;

  LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Allocated unified memory for " << model_size_ << " bytes";

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

  LOG(INFO) << "MemoryManager(" << model_identifier_ << "): Marked " << chunks_to_mark.size()
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

} // namespace stepcast::store