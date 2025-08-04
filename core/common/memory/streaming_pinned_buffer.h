// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <chrono>
#include <memory>
#include <queue>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

#include "core/common/memory/pinned_memory_pool.h"

namespace stepcast::store {

/**
 * @brief Manages a circular buffer pool for streaming data transfers.
 *
 * This class provides a producer-consumer pattern where:
 * - Producers can get free chunks, fill them with data, and mark them as ready
 * - Consumers can get ready chunks, use them, and return them to the free pool
 */
class StreamingPinnedBuffer {
 public:
  /**
   * @brief Construct a streaming buffer with specified number of chunks.
   * @param num_chunks Number of chunks in the circular buffer
   * @param chunk_size Size of each chunk (must match pool's chunk size)
   * @param pool Shared pinned memory pool to allocate from
   */
  StreamingPinnedBuffer(size_t num_chunks, size_t chunk_size, std::shared_ptr<PinnedMemoryPool> pool);
  ~StreamingPinnedBuffer();

  // Disable copy and move
  StreamingPinnedBuffer(const StreamingPinnedBuffer&) = delete;
  StreamingPinnedBuffer& operator=(const StreamingPinnedBuffer&) = delete;
  StreamingPinnedBuffer(StreamingPinnedBuffer&&) = delete;
  StreamingPinnedBuffer& operator=(StreamingPinnedBuffer&&) = delete;

  /**
   * @brief Initialize the buffer pool by allocating chunks from the memory pool.
   * @param timeout Optional timeout for allocation (default: no timeout)
   * @return Status indicating success or failure
   */
  absl::Status initialize(const std::chrono::milliseconds& timeout = std::chrono::milliseconds::zero())
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Release all chunks back to the memory pool.
   * @return Status indicating success or failure
   */
  absl::Status release() ABSL_LOCKS_EXCLUDED(mutex_) ABSL_NO_THREAD_SAFETY_ANALYSIS;

  // Helper for release(): checks if all free chunks are returned
  bool all_chunks_returned() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  /**
   * @brief Get a free chunk slot for producer to write data.
   * Blocks until a free slot is available.
   * @return Slot ID or error status
   */
  absl::StatusOr<int> get_free_chunk() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Mark a chunk as ready with data for consumer.
   * @param slot_id The slot ID returned by get_free_chunk
   * @param global_chunk_id The global chunk ID in the model
   * @param bytes_in_chunk Actual bytes written to this chunk
   * @return Status indicating success or failure
   */
  absl::Status mark_chunk_ready(int slot_id, size_t global_chunk_id, size_t bytes_in_chunk) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Structure representing a ready chunk for consumption.
   */
  struct ReadyChunk {
    int slot_id;
    size_t global_chunk_id;
    size_t bytes_in_chunk;
    char* data_ptr;
  };

  /**
   * @brief Get the next ready chunk for consumer to process.
   * Blocks until a chunk is ready.
   * @return ReadyChunk or error status
   */
  absl::StatusOr<ReadyChunk> get_ready_chunk() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Return a chunk back to the free pool after consumption.
   * @param slot_id The slot ID to return
   * @return Status indicating success or failure
   */
  absl::Status return_chunk(int slot_id) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get the pointer to a specific chunk slot.
   * @param slot_id The slot ID
   * @return Pointer to the chunk data or nullptr if invalid
   */
  char* get_chunk_ptr(int slot_id) const;

  /**
   * @brief Signal that no more chunks will be produced.
   */
  void signal_production_complete() ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Check if all chunks have been consumed after production is complete.
   * @return true if all chunks consumed, false otherwise
   */
  bool is_consumption_complete() const ABSL_LOCKS_EXCLUDED(mutex_);

  size_t chunk_size() const {
    return chunk_size_;
  }
  size_t num_chunks() const {
    return num_chunks_;
  }

 private:
  const size_t num_chunks_;
  const size_t chunk_size_;
  std::shared_ptr<PinnedMemoryPool> pool_;

  // Chunk buffers allocated from the pool
  std::vector<char*> chunk_buffers_;

  // Free chunks available for producers
  std::queue<int> free_queue_ ABSL_GUARDED_BY(mutex_);

  // Ready chunks waiting for consumers
  std::queue<ReadyChunk> ready_queue_ ABSL_GUARDED_BY(mutex_);

  // Synchronization
  mutable absl::Mutex mutex_;
  absl::CondVar free_cv_ ABSL_GUARDED_BY(mutex_);
  absl::CondVar ready_cv_ ABSL_GUARDED_BY(mutex_);

  // State tracking
  bool initialized_ ABSL_GUARDED_BY(mutex_) = false;
  bool production_complete_ ABSL_GUARDED_BY(mutex_) = false;
  size_t chunks_produced_ ABSL_GUARDED_BY(mutex_) = 0;
  size_t chunks_consumed_ ABSL_GUARDED_BY(mutex_) = 0;
};

} // namespace stepcast::store