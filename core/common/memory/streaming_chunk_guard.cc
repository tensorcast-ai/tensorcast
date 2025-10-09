// Copyright (c) 2025, TensorCast Team.

#include "core/common/memory/streaming_chunk_guard.h"

#include "absl/log/log.h"

namespace tensorcast::common::memory {

StreamingChunkGuard::StreamingChunkGuard(std::shared_ptr<StreamingPinnedBuffer> buffer)
    : buffer_shared_(std::move(buffer)), buffer_raw_(buffer_shared_.get()) {}

StreamingChunkGuard::StreamingChunkGuard(gsl::not_null<StreamingPinnedBuffer*> buffer) : buffer_raw_(buffer.get()) {}

StreamingChunkGuard::StreamingChunkGuard(StreamingPinnedBuffer* buffer) : buffer_raw_(buffer) {}

StreamingChunkGuard::StreamingChunkGuard(StreamingChunkGuard&& other) noexcept {
  *this = std::move(other);
}

StreamingChunkGuard& StreamingChunkGuard::operator=(StreamingChunkGuard&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  reset();
  buffer_shared_ = std::move(other.buffer_shared_);
  buffer_raw_ = buffer_shared_ ? buffer_shared_.get() : other.buffer_raw_;
  host_ptr_ = other.host_ptr_;
  slot_id_ = other.slot_id_;
  promoted_ = other.promoted_;

  other.host_ptr_ = nullptr;
  other.slot_id_ = -1;
  other.promoted_ = false;
  other.buffer_raw_ = nullptr;
  return *this;
}

StreamingChunkGuard::~StreamingChunkGuard() {
  reset();
}

void StreamingChunkGuard::reset() {
  if (!buffer_raw_ || slot_id_ < 0) {
    return;
  }

  absl::Status status;
  if (promoted_) {
    status = buffer_raw_->return_chunk(slot_id_);
  } else {
    status = buffer_raw_->abort_producer_slot(slot_id_);
  }
  if (!status.ok()) {
    LOG(WARNING) << "StreamingChunkGuard cleanup failed slot=" << slot_id_ << ": " << status;
  }

  slot_id_ = -1;
  host_ptr_ = nullptr;
  promoted_ = false;
}

absl::StatusOr<char*> StreamingChunkGuard::acquire() {
  return acquire_internal(/*blocking=*/true);
}

absl::StatusOr<char*> StreamingChunkGuard::try_acquire() {
  return acquire_internal(/*blocking=*/false);
}

absl::StatusOr<char*> StreamingChunkGuard::acquire_internal(bool blocking) {
  if (!buffer_raw_) {
    return absl::FailedPreconditionError("StreamingChunkGuard buffer is null");
  }
  if (slot_id_ >= 0) {
    return absl::FailedPreconditionError("StreamingChunkGuard already acquired");
  }

  absl::StatusOr<int> slot_or = blocking ? buffer_raw_->get_free_chunk() : buffer_raw_->try_get_free_chunk();
  if (!slot_or.ok()) {
    return slot_or.status();
  }

  slot_id_ = *slot_or;
  host_ptr_ = buffer_raw_->get_chunk_ptr(slot_id_);
  if (host_ptr_ == nullptr) {
    absl::Status rc = buffer_raw_->abort_producer_slot(slot_id_);
    if (!rc.ok()) {
      LOG(WARNING) << "StreamingChunkGuard abort_producer_slot failed slot=" << slot_id_ << ": " << rc;
    }
    slot_id_ = -1;
    return absl::InternalError("StreamingChunkGuard failed to map chunk pointer");
  }
  return host_ptr_;
}

absl::Status StreamingChunkGuard::promote_to_consumer(size_t global_chunk_id, size_t bytes_in_chunk) {
  if (slot_id_ < 0) {
    return absl::FailedPreconditionError("StreamingChunkGuard has no active slot");
  }
  if (promoted_) {
    return absl::FailedPreconditionError("StreamingChunkGuard already promoted");
  }
  auto status = buffer_raw_->promote_producer_slot_to_consumer(slot_id_, global_chunk_id, bytes_in_chunk);
  if (status.ok()) {
    promoted_ = true;
  }
  return status;
}

int StreamingChunkGuard::release_for_async() {
  if (slot_id_ < 0) {
    return -1;
  }
  if (!promoted_) {
    LOG(FATAL) << "StreamingChunkGuard::release_for_async requires promoted slot";
  }
  int slot = slot_id_;
  slot_id_ = -1;
  host_ptr_ = nullptr;
  return slot;
}

} // namespace tensorcast::common::memory
