// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/streaming_buffer_adapter.h"

#include "absl/log/log.h"

namespace stepcast::store::loader {

StreamingBufferAdapter::StreamingBufferAdapter(std::shared_ptr<::stepcast::store::StreamingPinnedBuffer> buffer)
    : buffer_(std::move(buffer)) {
  if (!buffer_) {
    LOG(ERROR) << "StreamingPinnedBuffer is null";
  }
}

size_t StreamingBufferAdapter::chunk_size() const {
  return buffer_ ? buffer_->chunk_size() : 0;
}

int StreamingBufferAdapter::capacity() const {
  return buffer_ ? buffer_->num_chunks() : 0;
}

absl::StatusOr<int> StreamingBufferAdapter::get_free_chunk() {
  if (!buffer_) {
    return absl::InvalidArgumentError("StreamingPinnedBuffer is null");
  }
  return buffer_->get_free_chunk();
}

void StreamingBufferAdapter::return_chunk(int slot_id) {
  if (!buffer_) {
    LOG(ERROR) << "StreamingPinnedBuffer is null";
    return;
  }
  auto status = buffer_->return_chunk(slot_id);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to return chunk: " << status;
  }
}

absl::Status StreamingBufferAdapter::mark_chunk_ready(int slot_id, uint64_t global_chunk_idx, size_t valid_bytes) {
  if (!buffer_) {
    return absl::InvalidArgumentError("StreamingPinnedBuffer is null");
  }
  // Note: StreamingPinnedBuffer uses size_t for global_chunk_id
  return buffer_->mark_chunk_ready(slot_id, static_cast<size_t>(global_chunk_idx), valid_bytes);
}

absl::StatusOr<ReadyChunk> StreamingBufferAdapter::get_ready_chunk() {
  if (!buffer_) {
    return absl::InvalidArgumentError("StreamingPinnedBuffer is null");
  }

  auto result = buffer_->get_ready_chunk();
  if (!result.ok()) {
    return result.status();
  }

  // Convert from StreamingPinnedBuffer::ReadyChunk to our ReadyChunk
  const auto& spb_chunk = *result;
  ReadyChunk chunk;
  chunk.slot_id = spb_chunk.slot_id;
  chunk.global_chunk_id = spb_chunk.global_chunk_id;
  chunk.bytes_in_chunk = spb_chunk.bytes_in_chunk;
  chunk.data_ptr = spb_chunk.data_ptr;

  return chunk;
}

void StreamingBufferAdapter::signal_production_complete() {
  if (!buffer_) {
    LOG(ERROR) << "StreamingPinnedBuffer is null";
    return;
  }
  buffer_->signal_production_complete();
}

void* StreamingBufferAdapter::get_chunk_data_ptr(int slot_id) {
  if (!buffer_) {
    return nullptr;
  }
  return buffer_->get_chunk_ptr(slot_id);
}

} // namespace stepcast::store::loader