// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/memory/streaming_pinned_buffer.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

namespace tensorcast::common::memory {

StreamingPinnedBuffer::StreamingPinnedBuffer(
    size_t num_chunks,
    size_t chunk_size,
    std::shared_ptr<PinnedBufferPool> pool)
    : num_chunks_(num_chunks), chunk_size_(chunk_size), pool_(std::move(pool)) {
  chunk_buffers_.reserve(num_chunks_);
  slot_states_.assign(num_chunks_, SlotState::kFree);
  slot_chunk_ids_.assign(num_chunks_, 0);
  slot_epochs_.assign(num_chunks_, 0);

  // Verify that chunk_size from pool is aligned for DIRECT_IO
  if (pool_ && pool_->slice_bytes() % PinnedBufferPool::kDirectIOAlignment != 0) {
    LOG(WARNING) << "StreamingPinnedBuffer: Pool chunk size " << pool_->slice_bytes()
                 << " is not aligned for DIRECT_IO";
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
      const std::string_view pool_name = pool_ ? pool_->name() : std::string_view();
      return absl::DeadlineExceededError(
          absl::StrCat(
              "Timeout waiting for ",
              result,
              " pinned slices from pool",
              (pool_name.empty() ? "" : absl::StrCat("[name=", pool_name, "]")),
              " after ",
              timeout.count(),
              "ms"));
    }
    const std::string_view pool_name = pool_ ? pool_->name() : std::string_view();
    return absl::ResourceExhaustedError(
        absl::StrCat(
            "Failed to allocate pinned slices from pool",
            (pool_name.empty() ? "" : absl::StrCat("[name=", pool_name, "]"))));
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
    slot_states_[i] = SlotState::kFree;
    slot_chunk_ids_[i] = 0;
    slot_epochs_[i] = 0;
  }
  next_epoch_ = 0;

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
      const size_t missing = num_chunks_ - free_queue_.size();
      LOG(ERROR) << "StreamingPinnedBuffer::release() timed out after " << absl::FormatDuration(timeout)
                 << " missing_chunks=" << missing << " total_chunks=" << num_chunks_
                 << " free_chunks=" << free_queue_.size() << " chunks_produced=" << chunks_produced_
                 << " chunks_consumed=" << chunks_consumed_
                 << " (refusing to deallocate buffers while chunks are still in use)";
      return absl::DeadlineExceededError(
          absl::StrCat(
              "StreamingPinnedBuffer::release timed out: missing_chunks=",
              missing,
              " total_chunks=",
              num_chunks_,
              " free_chunks=",
              free_queue_.size(),
              " chunks_produced=",
              chunks_produced_,
              " chunks_consumed=",
              chunks_consumed_));
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
  for (size_t i = 0; i < slot_states_.size(); ++i) {
    slot_states_[i] = SlotState::kFree;
    slot_chunk_ids_[i] = 0;
    slot_epochs_[i] = 0;
  }
  next_epoch_ = 0;
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
    slot_states_[rc.slot_id] = SlotState::kFree;
    slot_chunk_ids_[rc.slot_id] = 0;
  }

  // Ensure all slots are present in free_queue_. If some were lost due to
  // previous error, we conservatively rebuild the free list.
  if (free_queue_.size() != num_chunks_) {
    std::queue<int> empty;
    std::swap(free_queue_, empty);
    for (size_t i = 0; i < num_chunks_; ++i) {
      free_queue_.push(static_cast<int>(i));
      slot_states_[i] = SlotState::kFree;
      slot_chunk_ids_[i] = 0;
      slot_epochs_[i] = 0;
    }
  }
  next_epoch_ = 0;

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
  absl::Time wait_start;
  absl::Time next_log;
  bool wait_started = false;
  bool waiting_logged = false;
  while (free_queue_.empty() && !production_complete_) {
    const absl::Time now = absl::Now();
    // Avoid log spam in steady-state backpressure scenarios where waits are on the
    // order of microseconds. Only start logging if the wait becomes noticeable.
    constexpr absl::Duration kLogAfter = absl::Milliseconds(100);
    if (!wait_started) {
      wait_started = true;
      wait_start = now;
      next_log = now + kLogAfter;
    }

    if (!waiting_logged && now >= next_log) {
      LOG(WARNING) << "StreamingPinnedBuffer capacity exhausted (num_chunks=" << num_chunks_
                   << ", chunk_size=" << chunk_size_ << ") — waiting for consumer to return staging buffers";
      waiting_logged = true;
      next_log = now + absl::Seconds(5);
    } else if (waiting_logged && now >= next_log) {
      LOG(WARNING) << "StreamingPinnedBuffer still waiting for free chunk after "
                   << absl::FormatDuration(now - wait_start) << " (produced=" << chunks_produced_
                   << ", consumed=" << chunks_consumed_ << ")";
      next_log = now + absl::Seconds(5);
    }

    // Wake periodically even if no notifier so we can re-log the wait status
    const absl::Time deadline = now + absl::Seconds(1);
    free_cv_.WaitWithDeadline(&mutex_, deadline);
  }

  if (free_queue_.empty() && production_complete_) {
    return absl::OutOfRangeError("No more free chunks and production is complete");
  }

  int slot_id = free_queue_.front();
  free_queue_.pop();
  slot_chunk_ids_[slot_id] = 0;
  slot_epochs_[slot_id] = ++next_epoch_;
  set_slot_state_unsafe(slot_id, SlotState::kProducerOwned);

  if (waiting_logged) {
    LOG(INFO) << "StreamingPinnedBuffer wait resolved after " << absl::FormatDuration(absl::Now() - wait_start)
              << ": slot=" << slot_id << " now available";
  }

  return slot_id;
}

absl::Status StreamingPinnedBuffer::promote_producer_slot_to_consumer(
    int slot_id,
    size_t global_chunk_id,
    size_t bytes_in_chunk) {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  if (const auto status = validate_slot_id(slot_id); !status.ok()) {
    return status;
  }

  if (bytes_in_chunk == 0 || bytes_in_chunk > chunk_size_) {
    return absl::InvalidArgumentError("bytes_in_chunk must be within (0, chunk_size]");
  }

  if (get_slot_state_unsafe(slot_id) != SlotState::kProducerOwned) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "promote_producer_slot_to_consumer requires producer-owned slot; slot=",
            slot_id,
            " state=",
            static_cast<int>(get_slot_state_unsafe(slot_id))));
  }

  slot_chunk_ids_[slot_id] = global_chunk_id;
  slot_epochs_[slot_id] = ++next_epoch_;
  set_slot_state_unsafe(slot_id, SlotState::kConsumerOwned);
  chunks_produced_++;

  return absl::OkStatus();
}

absl::Status StreamingPinnedBuffer::abort_producer_slot(int slot_id) {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  if (const auto status = validate_slot_id(slot_id); !status.ok()) {
    return status;
  }

  if (get_slot_state_unsafe(slot_id) != SlotState::kProducerOwned) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "abort_producer_slot expects producer-owned slot; slot=",
            slot_id,
            " state=",
            static_cast<int>(get_slot_state_unsafe(slot_id))));
  }

  set_slot_state_unsafe(slot_id, SlotState::kFree);
  slot_chunk_ids_[slot_id] = 0;
  free_queue_.push(slot_id);
  free_cv_.Signal();
  return absl::OkStatus();
}

absl::Status StreamingPinnedBuffer::mark_chunk_ready(int slot_id, size_t global_chunk_id, size_t bytes_in_chunk) {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  if (const auto status = validate_slot_id(slot_id); !status.ok()) {
    return status;
  }

  if (get_slot_state_unsafe(slot_id) != SlotState::kProducerOwned) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "mark_chunk_ready expects producer-owned slot; slot=",
            slot_id,
            " state=",
            static_cast<int>(get_slot_state_unsafe(slot_id))));
  }

  ReadyChunk chunk{
      .slot_id = slot_id,
      .global_chunk_id = global_chunk_id,
      .bytes_in_chunk = bytes_in_chunk,
      .data_ptr = chunk_buffers_[slot_id]};

  ready_queue_.push(chunk);
  slot_chunk_ids_[slot_id] = global_chunk_id;
  set_slot_state_unsafe(slot_id, SlotState::kReady);
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
  if (const auto status = validate_slot_id(chunk.slot_id); !status.ok()) {
    return status;
  }
  if (get_slot_state_unsafe(chunk.slot_id) != SlotState::kReady) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "get_ready_chunk found slot in unexpected state=",
            static_cast<int>(get_slot_state_unsafe(chunk.slot_id)),
            " slot=",
            chunk.slot_id));
  }
  set_slot_state_unsafe(chunk.slot_id, SlotState::kConsumerOwned);
  slot_chunk_ids_[chunk.slot_id] = chunk.global_chunk_id;
  return chunk;
}

absl::Status StreamingPinnedBuffer::return_chunk(int slot_id) {
  absl::MutexLock lock(&mutex_);

  if (!initialized_) {
    return absl::FailedPreconditionError("StreamingPinnedBuffer not initialized");
  }

  if (const auto status = validate_slot_id(slot_id); !status.ok()) {
    return status;
  }

  if (get_slot_state_unsafe(slot_id) != SlotState::kConsumerOwned) {
    return absl::FailedPreconditionError(
        absl::StrCat(
            "return_chunk expects consumer-owned slot; slot=",
            slot_id,
            " state=",
            static_cast<int>(get_slot_state_unsafe(slot_id))));
  }

  set_slot_state_unsafe(slot_id, SlotState::kFree);
  slot_chunk_ids_[slot_id] = 0;
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
  set_slot_state_unsafe(slot_id, SlotState::kProducerOwned);
  slot_chunk_ids_[slot_id] = 0;
  slot_epochs_[slot_id] = ++next_epoch_;

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

void StreamingPinnedBuffer::set_slot_state_unsafe(int slot_id, SlotState state) {
  slot_states_[slot_id] = state;
}

StreamingPinnedBuffer::SlotState StreamingPinnedBuffer::get_slot_state_unsafe(int slot_id) const {
  return slot_states_[slot_id];
}

absl::Status StreamingPinnedBuffer::validate_slot_id(int slot_id) const {
  if (slot_id < 0 || slot_id >= static_cast<int>(num_chunks_)) {
    return absl::InvalidArgumentError("Invalid slot_id");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::common::memory
