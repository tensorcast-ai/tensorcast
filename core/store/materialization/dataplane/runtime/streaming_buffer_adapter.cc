// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/runtime/streaming_buffer_adapter.h"

#include "absl/log/log.h"
#include "absl/status/status.h"

namespace tensorcast::store::loader {

StreamingBufferAdapter::StreamingBufferAdapter(
    gsl::not_null<std::shared_ptr<tensorcast::common::memory::StreamingPinnedBuffer>> buffer)
    : buffer_(std::move(buffer)) {}

size_t StreamingBufferAdapter::chunk_size() const {
  return buffer_->chunk_size();
}

int StreamingBufferAdapter::capacity() const {
  return buffer_->num_chunks();
}

absl::StatusOr<int> StreamingBufferAdapter::get_free_chunk() {
  return buffer_->get_free_chunk();
}

void StreamingBufferAdapter::return_chunk(int slot_id) {
  absl::Status status = buffer_->return_chunk(slot_id);
  if (status.ok()) {
    return;
  }
  if (!absl::IsFailedPrecondition(status)) {
    LOG(ERROR) << "Failed to return chunk: " << status;
    return;
  }

  // Producer-side error paths use the same BufferPool::return_chunk contract
  // via RAII slot guards. StreamingPinnedBuffer requires producer-owned slots
  // to be released with abort_producer_slot().
  absl::Status abort_status = buffer_->abort_producer_slot(slot_id);
  if (abort_status.ok()) {
    VLOG(1) << "Recovered producer-owned slot via abort_producer_slot slot=" << slot_id;
    return;
  }
  LOG(ERROR) << "Failed to release slot after return_chunk fallback slot=" << slot_id
             << " return_status=" << status << " abort_status=" << abort_status;
}

absl::Status StreamingBufferAdapter::mark_chunk_ready(int slot_id, uint64_t global_chunk_idx, size_t valid_bytes) {
  // Note: StreamingPinnedBuffer uses size_t for global_chunk_id
  return buffer_->mark_chunk_ready(slot_id, static_cast<size_t>(global_chunk_idx), valid_bytes);
}

absl::StatusOr<ReadyChunk> StreamingBufferAdapter::get_ready_chunk() {
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
  buffer_->signal_production_complete();
}

void StreamingBufferAdapter::shutdown() {
  // For StreamingPinnedBuffer, best-effort shutdown means signaling production
  // complete and waking up all waiters so they can exit. The adapter forwards
  // this behavior.
  buffer_->signal_production_complete();
}

void* StreamingBufferAdapter::get_chunk_data_ptr(int slot_id) {
  return buffer_->get_chunk_ptr(slot_id);
}

} // namespace tensorcast::store::loader
