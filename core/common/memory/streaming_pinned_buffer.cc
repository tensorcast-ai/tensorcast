// Copyright (c) 2025, TensorCast Team.

#include "core/common/memory/streaming_pinned_buffer.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace tensorcast::common::memory {

StreamingPinnedBuffer::StreamingPinnedBuffer(
    size_t num_chunks,
    size_t chunk_size,
    std::shared_ptr<PinnedBufferPool> pool)
    : num_chunks_(num_chunks), chunk_size_(chunk_size), pool_(std::move(pool)) {
  chunk_buffers_.reserve(num_chunks_);

  // Verify that chunk_size from pool is aligned for DIRECT_IO
  if (pool_ && pool_->chunk_size() % PinnedBufferPool::kDirectIOAlignment != 0) {
    LOG(WARNING) << "StreamingPinnedBuffer: Pool chunk size " << pool_->chunk_size() << " is not aligned for DIRECT_IO";
  }
}

StreamingPinnedBuffer::~StreamingPinnedBuffer() {
  if (initialized_) {
    auto status = release();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to release streaming buffer: " << status;
    }
  }
}

absl::Status StreamingPinnedBuffer::initialize(const std::chrono::milliseconds& timeout) {
  absl::MutexLock lock(&mutex_);

  if (initialized_) {
    return absl::AlreadyExistsError("StreamingPinnedBuffer already initialized");
  }

  // Allocate all chunks from the pool (with optional timeout)
  size_t total_size = num_chunks_ * chunk_size_;
  int result = pool_->allocate(total_size, chunk_buffers_, timeout);
  if (result != 0) {
    if (result > 0 && timeout.count() > 0) {
      return absl::DeadlineExceededError(
          absl::StrCat(
              "Timeout waiting for ", result, " chunks from pinned memory pool after ", timeout.count(), "ms"));
    }
    return absl::ResourceExhaustedError("Failed to allocate chunks from pinned memory pool");
  }

  if (chunk_buffers_.size() != num_chunks_) {
    // This shouldn't happen if pool chunk size matches our chunk size
    pool_->deallocate(chunk_buffers_);
    chunk_buffers_.clear();
    return absl::InternalError("Allocated chunk count mismatch");
  }

  // Initialize all chunks as free
  for (size_t i = 0; i < num_chunks_; ++i) {
    free_queue_.push(static_cast<int>(i));
  }

  initialized_ = true;
  VLOG(1) << "Initialized streaming buffer with " << num_chunks_ << " chunks of " << chunk_size_ << " bytes each"
          << (timeout.count() > 0 ? absl::StrCat(" (timeout: ", timeout.count(), "ms)") : "");

  return absl::OkStatus();
}

absl::Status StreamingPinnedBuffer::release() {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  // Wait for all chunks to be returned with a timeout
  const auto timeout = absl::Seconds(30); // 30 second timeout

  while (free_queue_.size() < num_chunks_) {
    LOG(WARNING) << "Waiting for " << (num_chunks_ - free_queue_.size()) << " chunks to be returned before release";

    if (!mutex_.AwaitWithTimeout(absl::Condition(this, &StreamingPinnedBuffer::all_chunks_returned), timeout)) {
      // Timeout occurred - this is a fatal error
      LOG(FATAL) << "FATAL: StreamingPinnedBuffer::release() timed out after " << absl::FormatDuration(timeout) << ". "
                 << (num_chunks_ - free_queue_.size()) << " chunks were not returned. "
                 << "This indicates a resource leak - buffers obtained via get_free_chunk() "
                 << "were not returned via return_chunk(). "
                 << "Total chunks: " << num_chunks_ << ", Free chunks: " << free_queue_.size()
                 << ", Chunks produced: " << chunks_produced_ << ", Chunks consumed: " << chunks_consumed_;
    }
  }

  // Deallocate all chunks back to the pool
  int result = pool_->deallocate(chunk_buffers_);
  if (result != 0) {
    return absl::InternalError("Failed to deallocate chunks to pinned memory pool");
  }

  chunk_buffers_.clear();
  while (!free_queue_.empty()) {
    free_queue_.pop();
  }
  while (!ready_queue_.empty()) {
    ready_queue_.pop();
  }

  initialized_ = false;
  production_complete_ = false;
  chunks_produced_ = 0;
  chunks_consumed_ = 0;

  VLOG(1) << "Released streaming buffer";
  return absl::OkStatus();
}

bool StreamingPinnedBuffer::all_chunks_returned() const {
  return free_queue_.size() == num_chunks_;
}

absl::Status StreamingPinnedBuffer::reset_for_new_production() {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  // Reset producer/consumer coordination state.
  production_complete_ = false;
  chunks_produced_ = 0;
  chunks_consumed_ = 0;

  // Move any in-flight or ready chunks back to free. This is a best-effort
  // reset and assumes no threads are actively using the buffer.
  while (!ready_queue_.empty()) {
    auto rc = ready_queue_.front();
    ready_queue_.pop();
    free_queue_.push(rc.slot_id);
  }

  // Ensure all slots are present in free_queue_. If some were lost due to
  // previous error, we conservatively rebuild the free list.
  if (free_queue_.size() != num_chunks_) {
    std::queue<int> empty;
    std::swap(free_queue_, empty);
    for (size_t i = 0; i < num_chunks_; ++i) {
      free_queue_.push(static_cast<int>(i));
    }
  }

  // Wake up any potential waiters
  ready_cv_.SignalAll();
  free_cv_.SignalAll();

  return absl::OkStatus();
}

absl::StatusOr<int> StreamingPinnedBuffer::get_free_chunk() {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  // Wait for a free chunk to become available
  while (free_queue_.empty() && !production_complete_) {
    free_cv_.Wait(&mutex_);
  }

  if (free_queue_.empty() && production_complete_) {
    return absl::OutOfRangeError("No more free chunks and production is complete");
  }

  int slot_id = free_queue_.front();
  free_queue_.pop();

  return slot_id;
}

absl::Status StreamingPinnedBuffer::mark_chunk_ready(int slot_id, size_t global_chunk_id, size_t bytes_in_chunk) {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  if (slot_id < 0 || slot_id >= static_cast<int>(num_chunks_)) {
    return absl::InvalidArgumentError("Invalid slot_id");
  }

  ReadyChunk chunk{
      .slot_id = slot_id,
      .global_chunk_id = global_chunk_id,
      .bytes_in_chunk = bytes_in_chunk,
      .data_ptr = chunk_buffers_[slot_id]};

  ready_queue_.push(chunk);
  chunks_produced_++;
  ready_cv_.Signal();

  return absl::OkStatus();
}

absl::StatusOr<StreamingPinnedBuffer::ReadyChunk> StreamingPinnedBuffer::get_ready_chunk() {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  // Wait for a ready chunk or production complete
  while (ready_queue_.empty() && !production_complete_) {
    ready_cv_.Wait(&mutex_);
  }

  if (ready_queue_.empty() && production_complete_) {
    return absl::OutOfRangeError("No more ready chunks and production is complete");
  }

  ReadyChunk chunk = ready_queue_.front();
  ready_queue_.pop();

  return chunk;
}

absl::Status StreamingPinnedBuffer::return_chunk(int slot_id) {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  if (slot_id < 0 || slot_id >= static_cast<int>(num_chunks_)) {
    return absl::InvalidArgumentError("Invalid slot_id");
  }

  free_queue_.push(slot_id);
  chunks_consumed_++;
  free_cv_.Signal();

  return absl::OkStatus();
}

char* StreamingPinnedBuffer::get_chunk_ptr(int slot_id) const {
  if (slot_id < 0 || slot_id >= static_cast<int>(num_chunks_)) {
    return nullptr;
  }
  return chunk_buffers_[slot_id];
}

void StreamingPinnedBuffer::signal_production_complete() {
  absl::MutexLock lock(&mutex_);
  production_complete_ = true;
  ready_cv_.SignalAll();
  free_cv_.SignalAll();
  VLOG(1) << "Production complete. Total chunks produced: " << chunks_produced_;
}

bool StreamingPinnedBuffer::is_consumption_complete() const {
  absl::MutexLock lock(&mutex_);
  return production_complete_ && ready_queue_.empty() && chunks_consumed_ == chunks_produced_;
}

absl::StatusOr<int> StreamingPinnedBuffer::try_get_free_chunk() {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  // Non-blocking check - return immediately if no free chunks
  if (free_queue_.empty()) {
    return absl::UnavailableError("No free chunks available");
  }

  int slot_id = free_queue_.front();
  free_queue_.pop();

  return slot_id;
}

size_t StreamingPinnedBuffer::inflight() const {
  absl::MutexLock lock(&mutex_);

  // In-flight chunks are those that are neither free nor ready
  // Total chunks - free chunks - ready chunks = in-flight chunks
  return num_chunks_ - free_queue_.size() - ready_queue_.size();
}

bool StreamingPinnedBuffer::production_done() const {
  absl::MutexLock lock(&mutex_);
  return production_complete_;
}

} // namespace tensorcast::common::memory
