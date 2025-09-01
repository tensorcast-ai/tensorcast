// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorcast::store::loader {

struct ReadyChunk {
  int slot_id;
  uint64_t global_chunk_id;
  size_t bytes_in_chunk;
  void* data_ptr;
};

class BufferPool {
 public:
  virtual ~BufferPool() = default;

  virtual size_t chunk_size() const = 0;

  virtual int capacity() const = 0;

  virtual absl::StatusOr<int> get_free_chunk() = 0;

  virtual void return_chunk(int slot_id) = 0;

  virtual absl::Status mark_chunk_ready(int slot_id, uint64_t global_chunk_idx, size_t valid_bytes) = 0;

  virtual absl::StatusOr<ReadyChunk> get_ready_chunk() = 0;

  virtual void signal_production_complete() = 0;

  // Gracefully cancel ongoing waits and wake up all waiters.
  // Implementations should ensure that any threads blocked in
  // get_free_chunk()/get_ready_chunk() are unblocked promptly.
  // After shutdown(), further calls to blocking getters should
  // return a non-OK status (e.g., Cancelled/OutOfRange/Unavailable).
  virtual void shutdown() = 0;

  // Get pointer to chunk data for writing. Must be called only after
  // get_free_chunk() and before return_chunk() or mark_chunk_ready().
  virtual void* get_chunk_data_ptr(int slot_id) = 0;
};

} // namespace tensorcast::store::loader