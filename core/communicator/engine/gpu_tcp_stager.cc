// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/communicator/engine/gpu_tcp_stager.h"

#include "core/common/cuda_api.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "core/common/device_guard.h"
#include "core/communicator/engine/engine.h" // For COMMUNICATE_ENGINE_DEV_GPU
#include "core/communicator/transport/partition_tensor.h" // For PartitionTensor

namespace stepcast::communicator {

// Implementation of ScopedStagedBuffer
ScopedStagedBuffer::~ScopedStagedBuffer() {
  reset();
}

void ScopedStagedBuffer::reset() {
  if (data_ && stager_) {
    auto status = stager_->release_staged_buffer(data_);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to release staged buffer in ScopedStagedBuffer destructor: " << status;
    }
  }
  stager_ = nullptr;
  data_ = nullptr;
  size_ = 0;
}

GpuTcpStager::GpuTcpStager(size_t chunk_size, size_t num_chunks, std::shared_ptr<store::PinnedMemoryPool> pool)
    : chunk_size_(chunk_size), num_chunks_(num_chunks), memory_pool_(pool) {
  // Create memory pool if not provided
  if (!memory_pool_) {
    // Create a pool with enough memory for all chunks
    memory_pool_ = std::make_shared<store::PinnedMemoryPool>(chunk_size_ * num_chunks_, chunk_size_);
  }

  // Create streaming buffer
  streaming_buffer_ = std::make_unique<store::StreamingPinnedBuffer>(num_chunks_, chunk_size_, memory_pool_);

  // Initialize streaming buffer
  auto status = streaming_buffer_->initialize();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize streaming buffer: " << status;
    streaming_buffer_.reset();
    return;
  }

  // Create CUDA stream for async copies
  auto stream_status = cuda::stream_create(&copy_stream_);
  if (!stream_status.ok()) {
    LOG(ERROR) << "Failed to create CUDA stream: " << stream_status.message();
    CHECK_OK(streaming_buffer_->release());
    streaming_buffer_.reset();
    copy_stream_ = nullptr;
    return;
  }

  // Pre-create CUDA events
  cuda_events_.resize(num_chunks_);
  for (size_t i = 0; i < num_chunks_; ++i) {
    auto event_status = cuda::event_create_with_flags(&cuda_events_[i], cudaEventDisableTiming);
    if (!event_status.ok()) {
      LOG(ERROR) << "Failed to create CUDA event: " << event_status.message();
      // Clean up
      for (size_t j = 0; j < i; ++j) {
        auto destroy_status = cuda::event_destroy(cuda_events_[j]);
        if (!destroy_status.ok()) {
          LOG(ERROR) << "Failed to destroy CUDA event during cleanup: " << destroy_status.message();
        }
      }
      cuda_events_.clear();
      auto stream_destroy_status = cuda::stream_destroy(copy_stream_);
      if (!stream_destroy_status.ok()) {
        LOG(ERROR) << "Failed to destroy CUDA stream during cleanup: " << stream_destroy_status.message();
      }
      CHECK_OK(streaming_buffer_->release());
      streaming_buffer_.reset();
      return;
    }
    free_events_.push(cuda_events_[i]);
  }

  VLOG(1) << "GpuTcpStager initialized with " << num_chunks_ << " chunks of " << chunk_size_ / (1024 * 1024)
          << " MiB each";
}

GpuTcpStager::~GpuTcpStager() {
  // Wait for all pending copies to complete
  if (copy_stream_) {
    auto sync_status = cuda::stream_synchronize(copy_stream_);
    if (!sync_status.ok()) {
      LOG(ERROR) << "Failed to synchronize CUDA stream during destruction: " << sync_status.message();
    }
    auto destroy_status = cuda::stream_destroy(copy_stream_);
    if (!destroy_status.ok()) {
      LOG(ERROR) << "Failed to destroy CUDA stream during destruction: " << destroy_status.message();
    }
  }

  // Destroy CUDA events
  for (auto* event : cuda_events_) {
    if (event) {
      auto destroy_status = cuda::event_destroy(event);
      if (!destroy_status.ok()) {
        LOG(ERROR) << "Failed to destroy CUDA event during destruction: " << destroy_status.message();
      }
    }
  }

  // Release streaming buffer
  if (streaming_buffer_) {
    CHECK_OK(streaming_buffer_->release());
  }
}

absl::StatusOr<void*> GpuTcpStager::stage(
    const std::shared_ptr<PartitionTensor>& tensor,
    uint64_t offset,
    uint64_t bytes) {
  VLOG(2) << "[GpuTcpStager::stage] tensor=" << tensor->get_key() << " offset=" << offset << " bytes=" << bytes;
  // Validate inputs
  if (!tensor) {
    return absl::InvalidArgumentError("Null tensor provided to stage()");
  }

  if (tensor->get_mem_type() != COMMUNICATE_ENGINE_DEV_GPU) {
    return absl::InvalidArgumentError("Tensor is not on GPU");
  }

  if (offset + bytes > tensor->get_bytes()) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Stage request out of bounds: offset=%lu, bytes=%lu, tensor_size=%lu", offset, bytes, tensor->get_bytes()));
  }

  if (bytes > chunk_size_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Requested bytes (%lu) exceeds chunk size (%lu)", bytes, chunk_size_));
  }

  if (!streaming_buffer_) {
    return absl::FailedPreconditionError("GpuTcpStager not properly initialized");
  }

  // Get a free chunk from streaming buffer
  auto slot_result = streaming_buffer_->get_free_chunk();
  if (!slot_result.ok()) {
    absl::MutexLock lock(&mutex_);
    stats_.buffer_wait_count++;
    return slot_result.status();
  }

  int slot_id = *slot_result;
  void* host_ptr = streaming_buffer_->get_chunk_ptr(slot_id);
  if (!host_ptr) {
    CHECK_OK(streaming_buffer_->return_chunk(slot_id));
    return absl::InternalError("Failed to get chunk pointer");
  }

  // Get a CUDA event
  cudaEvent_t event = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    event = get_free_event();

    // Track the operation
    StagingOperation op;
    op.slot_id = slot_id;
    op.copy_complete_event = event;
    op.host_ptr = host_ptr;
    active_operations_[slot_id] = op;
    ptr_to_slot_[host_ptr] = slot_id;

    stats_.total_stage_calls++;
    stats_.total_staged_bytes += bytes;
  }

  // Perform the staging copy
  auto copy_status = perform_staging_copy(tensor, offset, bytes, host_ptr, event);
  if (!copy_status.ok()) {
    // Clean up on error
    absl::MutexLock lock(&mutex_);
    active_operations_.erase(slot_id);
    ptr_to_slot_.erase(host_ptr);
    return_event(event);
    CHECK_OK(streaming_buffer_->return_chunk(slot_id));
    return copy_status;
  }

  // For synchronous interface, wait for completion
  auto sync_status = cuda::event_synchronize(event);
  if (!sync_status.ok()) {
    // Clean up on error
    absl::MutexLock lock(&mutex_);
    active_operations_.erase(slot_id);
    ptr_to_slot_.erase(host_ptr);
    return_event(event);
    CHECK_OK(streaming_buffer_->return_chunk(slot_id));
    return sync_status;
  }

  return host_ptr;
}

absl::StatusOr<int> GpuTcpStager::stage_async(
    const std::shared_ptr<PartitionTensor>& tensor,
    uint64_t offset,
    uint64_t bytes) {
  // Validate inputs
  if (!tensor) {
    return absl::InvalidArgumentError("Null tensor provided to stage_async()");
  }

  if (tensor->get_mem_type() != COMMUNICATE_ENGINE_DEV_GPU) {
    return absl::InvalidArgumentError("Tensor is not on GPU");
  }

  if (offset + bytes > tensor->get_bytes()) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "Stage request out of bounds: offset=%lu, bytes=%lu, tensor_size=%lu", offset, bytes, tensor->get_bytes()));
  }

  if (bytes > chunk_size_) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Requested bytes (%lu) exceeds chunk size (%lu)", bytes, chunk_size_));
  }

  if (!streaming_buffer_) {
    return absl::FailedPreconditionError("GpuTcpStager not properly initialized");
  }

  // Get a free chunk from streaming buffer
  auto slot_result = streaming_buffer_->get_free_chunk();
  if (!slot_result.ok()) {
    absl::MutexLock lock(&mutex_);
    stats_.buffer_wait_count++;
    return slot_result.status();
  }

  int slot_id = *slot_result;
  void* host_ptr = streaming_buffer_->get_chunk_ptr(slot_id);
  if (!host_ptr) {
    CHECK_OK(streaming_buffer_->return_chunk(slot_id));
    return absl::InternalError("Failed to get chunk pointer");
  }

  // Get a CUDA event
  cudaEvent_t event = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    event = get_free_event();

    // Track the operation
    StagingOperation op;
    op.slot_id = slot_id;
    op.copy_complete_event = event;
    op.host_ptr = host_ptr;
    active_operations_[slot_id] = op;
    ptr_to_slot_[host_ptr] = slot_id;

    stats_.total_stage_calls++;
    stats_.total_staged_bytes += bytes;
  }

  // Perform the staging copy
  auto copy_status = perform_staging_copy(tensor, offset, bytes, host_ptr, event);
  if (!copy_status.ok()) {
    // Clean up on error
    absl::MutexLock lock(&mutex_);
    active_operations_.erase(slot_id);
    ptr_to_slot_.erase(host_ptr);
    return_event(event);
    CHECK_OK(streaming_buffer_->return_chunk(slot_id));
    return copy_status;
  }

  // Return slot ID immediately for async operation
  return slot_id;
}

absl::StatusOr<void*> GpuTcpStager::wait_staging_complete(int slot_id) {
  StagingOperation op;
  {
    absl::MutexLock lock(&mutex_);
    auto it = active_operations_.find(slot_id);
    if (it == active_operations_.end()) {
      return absl::NotFoundError("Invalid or completed staging operation");
    }
    op = it->second;
  }

  // Wait for the copy to complete
  auto event_sync_status = cuda::event_synchronize(op.copy_complete_event);
  if (!event_sync_status.ok()) {
    // Clean up on error
    absl::MutexLock lock(&mutex_);
    active_operations_.erase(slot_id);
    ptr_to_slot_.erase(op.host_ptr);
    return_event(op.copy_complete_event);
    CHECK_OK(streaming_buffer_->return_chunk(slot_id));
    return event_sync_status;
  }

  return op.host_ptr;
}

absl::Status GpuTcpStager::release_staged_buffer(void* host_ptr) {
  if (!host_ptr) {
    return absl::InvalidArgumentError("Null host pointer provided");
  }

  int slot_id = -1;
  cudaEvent_t event = nullptr;

  {
    absl::MutexLock lock(&mutex_);

    auto ptr_it = ptr_to_slot_.find(host_ptr);
    if (ptr_it == ptr_to_slot_.end()) {
      return absl::NotFoundError("Host pointer not found in active operations");
    }
    slot_id = ptr_it->second;

    auto op_it = active_operations_.find(slot_id);
    if (op_it == active_operations_.end()) {
      return absl::InternalError("Inconsistent internal state");
    }

    event = op_it->second.copy_complete_event;

    // Remove from tracking
    ptr_to_slot_.erase(ptr_it);
    active_operations_.erase(op_it);

    // Return event to pool
    return_event(event);
  }

  // Return chunk to streaming buffer
  return streaming_buffer_->return_chunk(slot_id);
}

GpuTcpStager::Stats GpuTcpStager::get_stats() const {
  absl::MutexLock lock(&mutex_);
  return stats_;
}

absl::Status GpuTcpStager::perform_staging_copy(
    const std::shared_ptr<PartitionTensor>& tensor,
    uint64_t offset,
    uint64_t bytes,
    void* dest_ptr,
    cudaEvent_t event) {
  // Ensure we're on the correct device
  int tensor_device = tensor->get_device_id();
  if (tensor_device < 0) {
    tensor_device = 0;
  }

  common::DeviceGuard guard(tensor_device);
  if (!guard.status().ok()) {
    return guard.status();
  }

  // Perform async GPU->CPU copy
  void* gpu_ptr = reinterpret_cast<void*>(tensor->get_uint64_addr() + offset);
  auto copy_status = cuda::memcpy_async(dest_ptr, gpu_ptr, bytes, cudaMemcpyDeviceToHost, copy_stream_);
  if (!copy_status.ok()) {
    return copy_status;
  }

  // Record event for tracking copy completion
  auto record_status = cuda::event_record(event, copy_stream_);
  if (!record_status.ok()) {
    return record_status;
  }

  return absl::OkStatus();
}

cudaEvent_t GpuTcpStager::get_free_event() {
  if (free_events_.empty()) {
    // This should not happen if we sized the event pool correctly
    LOG(ERROR) << "No free CUDA events available";
    return nullptr;
  }
  cudaEvent_t event = free_events_.front();
  free_events_.pop();
  return event;
}

void GpuTcpStager::return_event(cudaEvent_t event) {
  if (event) {
    free_events_.push(event);
  }
}

absl::StatusOr<ScopedStagedBuffer> GpuTcpStager::stage_scoped(
    const std::shared_ptr<PartitionTensor>& tensor,
    uint64_t offset,
    uint64_t bytes) {
  // Use the existing stage() method
  auto result = stage(tensor, offset, bytes);
  if (!result.ok()) {
    return result.status();
  }

  return ScopedStagedBuffer(this, *result, bytes);
}

} // namespace stepcast::communicator