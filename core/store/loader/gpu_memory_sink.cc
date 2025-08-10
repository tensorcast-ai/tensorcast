// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/gpu_memory_sink.h"

#include "absl/log/log.h"
#include "core/common/device_guard.h"

namespace stepcast::store::loader {

GPUMemorySink::GPUMemorySink(Options options) : options_(std::move(options)) {
  if (!options_.gpu_base_ptr) {
    overall_status_ = absl::InvalidArgumentError("GPU base pointer is null");
    return;
  }

  // Create non-blocking CUDA stream for H2D transfers
  common::DeviceGuard guard(options_.device_id);
  if (!guard.status().ok()) {
    overall_status_ = guard.status();
    return;
  }

  auto stream_status = cuda::stream_create_with_flags(&h2d_stream_, cudaStreamNonBlocking);
  if (!stream_status.ok()) {
    overall_status_ = stream_status;
    LOG(ERROR) << "Failed to create CUDA stream: " << stream_status;
    return;
  }
  stream_created_ = true;
}

GPUMemorySink::~GPUMemorySink() {
  if (stream_created_) {
    auto destroy_status = cuda::stream_destroy(h2d_stream_);
    if (!destroy_status.ok()) {
      LOG(ERROR) << "Failed to destroy CUDA stream: " << destroy_status;
    }
  }
}

absl::Status GPUMemorySink::write(const void* src, size_t bytes) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  auto st = write_at(current_offset_, src, bytes);
  if (st.ok()) {
    current_offset_ += bytes;
  }
  return st;
}

absl::Status GPUMemorySink::write_at(uint64_t offset, const void* src, size_t bytes) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }

  if (offset + bytes > options_.total_size) {
    return absl::InvalidArgumentError("Write would exceed total GPU memory size");
  }

  // Set device context
  common::DeviceGuard guard(options_.device_id);
  if (!guard.status().ok()) {
    overall_status_ = guard.status();
    return overall_status_;
  }

  // Calculate destination GPU pointer
  char* gpu_dest = static_cast<char*>(options_.gpu_base_ptr) + offset;

  // Perform async H2D transfer
  auto copy_status = cuda::memcpy_async(gpu_dest, src, bytes, cudaMemcpyHostToDevice, h2d_stream_);

  if (!copy_status.ok()) {
    overall_status_ = copy_status;
    LOG(ERROR) << "Failed to copy data to GPU: " << copy_status;
    return copy_status;
  }

  VLOG(3) << "Copied " << bytes << " bytes to GPU at offset " << offset;

  return absl::OkStatus();
}

absl::Status GPUMemorySink::close() {
  if (!overall_status_.ok()) {
    return overall_status_;
  }

  if (!stream_created_) {
    return absl::OkStatus();
  }

  // Validate that all expected data was written
  if (current_offset_ != options_.total_size) {
    LOG(WARNING) << "GPU memory sink closed with incomplete transfer. "
                 << "Expected " << options_.total_size << " bytes, "
                 << "but only " << current_offset_ << " bytes were written.";
  }

  // Set device context
  common::DeviceGuard guard(options_.device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }

  // Synchronize stream to ensure all transfers complete
  auto sync_status = cuda::stream_synchronize(h2d_stream_);
  if (!sync_status.ok()) {
    LOG(ERROR) << "Failed to synchronize CUDA stream: " << sync_status;
    return sync_status;
  }

  VLOG(2) << "GPU memory sink closed successfully. Total bytes written: " << current_offset_;

  return absl::OkStatus();
}

} // namespace stepcast::store::loader
