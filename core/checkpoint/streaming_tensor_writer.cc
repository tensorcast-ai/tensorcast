// Copyright (c) 2025-2026, TensorCast Team.

#include "core/checkpoint/streaming_tensor_writer.h"

#include <algorithm>
#include <cstring>
#include "core/common/async_copy_manager.h"
#include "core/cuda/cuda_api.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::checkpoint {

StreamingTensorWriter::StreamingTensorWriter(
    std::string filename,
    Config config,
    std::shared_ptr<common::memory::PinnedBufferPool> pool)
    : filename_(std::move(filename)),
      config_(std::move(config)),
      buffer_size_(config_.buffer_size_mb << 20) { // Convert MB to bytes

  streaming_buffer_ =
      std::make_unique<common::memory::StreamingPinnedBuffer>(config_.num_buffers, buffer_size_, std::move(pool));
  tensor_writer_ = std::make_unique<TensorWriter>(filename_);
}

StreamingTensorWriter::~StreamingTensorWriter() {
  if (initialized_ && !shutdown_.load()) {
    auto status = finalize();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to finalize StreamingTensorWriter: " << status;
    }
  }
}

absl::Status StreamingTensorWriter::initialize() {
  if (initialized_) {
    return absl::AlreadyExistsError("StreamingTensorWriter already initialized");
  }

  // Initialize the streaming buffer
  auto status = streaming_buffer_->initialize();
  if (!status.ok()) {
    return status;
  }

  // Start the disk writer thread if async write is enabled
  if (config_.enable_async_write) {
    disk_writer_thread_ = std::thread(&StreamingTensorWriter::disk_writer_thread, this);
  }

  initialized_ = true;
  LOG(INFO) << "StreamingTensorWriter initialized with " << config_.num_buffers << " buffers of "
            << config_.buffer_size_mb << " MB each";

  return absl::OkStatus();
}

absl::StatusOr<uint64_t> StreamingTensorWriter::write_tensor(
    const void* data,
    size_t size,
    bool is_gpu,
    cudaStream_t stream) {
  static_cast<void>(stream);
  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingTensorWriter not initialized");
  }

  const uint64_t tensor_offset = current_offset_.load();
  size_t remaining = size;
  size_t processed = 0;

  // Process data in chunks
  while (remaining > 0) {
    // Get a free buffer slot
    auto slot_result = streaming_buffer_->get_free_chunk();
    if (!slot_result.ok()) {
      return slot_result.status();
    }
    int slot_id = slot_result.value();

    // Calculate chunk size (raw data bytes)
    size_t chunk_size = std::min(remaining, buffer_size_);

    // TensorWriter::write_record() **always** rounds the written length up to an
    // 8-byte boundary via TensorWriter::aligned_size().  The previous logic only
    // added this padding for the *last* chunk of a tensor, which meant that for
    // every *intermediate* chunk where chunk_size % 8 != 0 the Streaming writer
    // advanced its logical offset by *less* than the actual on-disk bytes
    // written.  As a consequence, all subsequent tensor offsets recorded in
    // tensor_index.json became progressively smaller than their real
    // positions, leading to silent corruption when the checkpoint was re-read.
    //
    // The correct behaviour is therefore to account for the padding on **every
    // chunk**, irrespective of whether it closes the tensor.
    const size_t padded_chunk_size = TensorWriter::aligned_size(chunk_size);

    char* buffer_ptr = streaming_buffer_->get_chunk_ptr(slot_id);

    if (!buffer_ptr) {
      return absl::InternalError("Invalid buffer pointer");
    }

    // Allocate a single consistent global chunk id for both CPU and GPU paths.
    const size_t global_chunk_id = next_chunk_id_.fetch_add(1, std::memory_order_relaxed);

    // Track chunk offset for this global chunk using the same id
    chunk_offsets_[global_chunk_id] = current_offset_.load();

    // Copy data to buffer
    if (is_gpu) {
      // Detect device id from the GPU pointer and schedule D2H via ACM.
      const void* gpu_ptr = static_cast<const char*>(data) + processed;
      cudaPointerAttributes attrs{};
      auto attr_status = tensorcast::cuda::pointer_get_attributes_full(gpu_ptr, &attrs);
      if (!attr_status.ok()) {
        return absl::InternalError(absl::StrCat("Failed to query CUDA pointer attributes: ", attr_status.message()));
      }
      const int device_id = attrs.device;

      common::DeviceRegion src{.device_id = device_id, .dev_ptr = const_cast<void*>(gpu_ptr), .length = chunk_size};
      common::HostRegion dst{.base = buffer_ptr, .length = chunk_size, .pinned = true};

      // Use a callback that marks the chunk ready with the same global id.
      auto on_done = [spb = streaming_buffer_.get(), slot_id, global_chunk_id, chunk_size](absl::Status st) {
        if (!st.ok()) {
          LOG(ERROR) << "AsyncCopyManager::submit_d2h failed for chunk_id=" << global_chunk_id << ": " << st;
          absl::Status rc = spb->return_chunk(slot_id);
          if (!rc.ok()) {
            LOG(WARNING) << "StreamingPinnedBuffer::return_chunk failed after async error slot=" << slot_id << ": "
                         << rc;
          }
          return;
        }
        absl::Status rc = spb->mark_chunk_ready(slot_id, global_chunk_id, chunk_size);
        if (!rc.ok()) {
          LOG(WARNING) << "StreamingPinnedBuffer::mark_chunk_ready failed slot=" << slot_id << ": " << rc;
        }
      };
      common::CopyOptions opts{.tracing_stage = "D2H/Copy", .callbacks = {.on_copy_done = on_done}};
      auto hdl_or = common::AsyncCopyManager::instance().submit_d2h(src, dst, opts);
      if (!hdl_or.ok()) {
        return hdl_or.status();
      }
      // Record the in-flight copy handle to wait on during finalize().
      {
        absl::MutexLock lock(&pending_mu_);
        pending_copies_.emplace_back(std::move(hdl_or.value()));
      }
    } else {
      // CPU data - direct memory copy
      std::memcpy(buffer_ptr, static_cast<const char*>(data) + processed, chunk_size);
      // Mark chunk as ready for CPU path immediately.
      auto ready_status = streaming_buffer_->mark_chunk_ready(slot_id, global_chunk_id, chunk_size);
      if (!ready_status.ok()) {
        return ready_status;
      }
    }

    // If synchronous write mode, write immediately
    if (!config_.enable_async_write) {
      auto ready_chunk_result = streaming_buffer_->get_ready_chunk();
      if (!ready_chunk_result.ok()) {
        return ready_chunk_result.status();
      }

      auto ready_chunk = ready_chunk_result.value();
      tensor_writer_->write_record(ready_chunk.data_ptr, ready_chunk.bytes_in_chunk);

      auto return_status = streaming_buffer_->return_chunk(ready_chunk.slot_id);
      if (!return_status.ok()) {
        return return_status;
      }
    }

    // Update counters (including alignment padding)
    processed += chunk_size;
    remaining -= chunk_size;
    current_offset_ += padded_chunk_size;
    total_bytes_written_ += padded_chunk_size;
  }

  return tensor_offset;
}

// Removed copy_gpu_to_buffer helper in favor of ACM-driven D2H

void StreamingTensorWriter::disk_writer_thread() {
  LOG(INFO) << "Disk writer thread started";

  while (!shutdown_.load() || !streaming_buffer_->is_consumption_complete()) {
    // Get ready chunk
    auto ready_chunk_result = streaming_buffer_->get_ready_chunk();

    if (!ready_chunk_result.ok()) {
      if (shutdown_.load() && ready_chunk_result.status().code() == absl::StatusCode::kOutOfRange) {
        // Normal shutdown condition
        break;
      }
      LOG(ERROR) << "Failed to get ready chunk: " << ready_chunk_result.status();
      continue;
    }

    auto ready_chunk = ready_chunk_result.value();

    // Write to disk
    uint64_t written_offset = tensor_writer_->write_record(ready_chunk.data_ptr, ready_chunk.bytes_in_chunk);

    // Update chunk offset mapping
    if (chunk_offsets_.contains(ready_chunk.global_chunk_id)) {
      // Verify offset matches expected
      uint64_t expected_offset = chunk_offsets_[ready_chunk.global_chunk_id];
      if (written_offset != expected_offset) {
        LOG(ERROR) << "Chunk offset mismatch: expected " << expected_offset << " but got " << written_offset;
      }
    }

    // Return chunk to free pool
    auto return_status = streaming_buffer_->return_chunk(ready_chunk.slot_id);
    if (!return_status.ok()) {
      LOG(ERROR) << "Failed to return chunk: " << return_status;
    }
  }

  LOG(INFO) << "Disk writer thread finished. Total bytes written: " << total_bytes_written_.load();
}

absl::Status StreamingTensorWriter::finalize() {
  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingTensorWriter not initialized");
  }

  // Ensure all GPU D2H copies have completed and invoked their callbacks
  // before signaling production complete to the consumer. This prevents
  // premature termination and avoids use-after-free of the buffer in callbacks.
  std::vector<common::CopyHandle> handles_to_wait;
  {
    absl::MutexLock lock(&pending_mu_);
    handles_to_wait.swap(pending_copies_);
  }
  for (const auto& h : handles_to_wait) {
    absl::Status st = h.wait();
    if (!st.ok()) {
      return st;
    }
  }

  // Signal no more data will be produced only after pending copies are done.
  streaming_buffer_->signal_production_complete();

  // Wait for disk writer thread to finish
  if (config_.enable_async_write && disk_writer_thread_.joinable()) {
    shutdown_.store(true);
    disk_writer_thread_.join();
  }

  initialized_ = false;
  LOG(INFO) << "StreamingTensorWriter finalized. Total bytes written: " << total_bytes_written_.load();

  return absl::OkStatus();
}

} // namespace tensorcast::checkpoint
