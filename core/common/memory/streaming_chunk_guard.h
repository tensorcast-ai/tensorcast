// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gsl/pointers"

#include "core/common/memory/streaming_pinned_buffer.h"

namespace tensorcast::common::memory {

// StreamingChunkGuard wraps the lifecycle of a StreamingPinnedBuffer slot.
// Typical usage:
//   StreamingChunkGuard guard(buffer);
//   TCC_ASSIGN_OR_RETURN(auto host_ptr, guard.acquire());
//   ... populate host_ptr ...
//   TCC_RETURN_IF_ERROR(guard.promote_to_consumer(global_chunk_id, bytes));
//   const int slot = guard.release_for_async();  // transfers ownership to async path
//   // async callback must call StreamingPinnedBuffer::return_chunk(slot).
//
// If promote_to_consumer fails or the guard goes out of scope before release_for_async
// is called, the guard automatically aborts or returns the slot, preventing leaks.
class StreamingChunkGuard {
 public:
  explicit StreamingChunkGuard(std::shared_ptr<StreamingPinnedBuffer> buffer);
  explicit StreamingChunkGuard(gsl::not_null<StreamingPinnedBuffer*> buffer);
  explicit StreamingChunkGuard(StreamingPinnedBuffer* buffer);
  StreamingChunkGuard(StreamingChunkGuard&& other) noexcept;
  StreamingChunkGuard& operator=(StreamingChunkGuard&& other) noexcept;

  StreamingChunkGuard(const StreamingChunkGuard&) = delete;
  StreamingChunkGuard& operator=(const StreamingChunkGuard&) = delete;

  ~StreamingChunkGuard();

  // Acquire a producer-owned slot (blocking) and expose its host pointer.
  absl::StatusOr<char*> acquire();

  // Try to acquire a producer slot without blocking; returns Unavailable on contention.
  absl::StatusOr<char*> try_acquire();

  // Promote the currently-held slot into consumer ownership so async consumers
  // (e.g. AsyncCopyManager callbacks) can safely recycle it via return_chunk().
  absl::Status promote_to_consumer(size_t global_chunk_id, size_t bytes_in_chunk);

  // Transfer responsibility for the slot to an async consumer. After this call,
  // the guard will no longer clean up the slot in its destructor.
  // Requires promote_to_consumer() to have succeeded.
  int release_for_async();

  bool has_slot() const {
    return slot_id_ >= 0;
  }

  int slot_id() const {
    return slot_id_;
  }

 private:
  absl::StatusOr<char*> acquire_internal(bool blocking);
  void reset();

  std::shared_ptr<StreamingPinnedBuffer> buffer_shared_;
  StreamingPinnedBuffer* buffer_raw_ = nullptr;
  char* host_ptr_ = nullptr;
  int slot_id_ = -1;
  bool promoted_ = false;
};

} // namespace tensorcast::common::memory
