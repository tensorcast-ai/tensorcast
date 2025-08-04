// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"

namespace stepcast::communicator {

// Forward declaration
class PartitionTensor;
class GpuTcpStager;

/**
 * @brief RAII wrapper for staged GPU buffers.
 *
 * This class ensures that staged buffers are automatically released when they go
 * out of scope, preventing resource leaks. Users should prefer using this wrapper
 * instead of manually calling stage() and release_staged_buffer().
 *
 * Example usage:
 * @code
 * {
 *   auto staged_buffer_or = stager->stage_scoped(tensor, offset, bytes);
 *   if (!staged_buffer_or.ok()) {
 *     return staged_buffer_or.status();
 *   }
 *   auto staged = std::move(*staged_buffer_or);
 *   // Use staged->data() to access the buffer
 *   send_over_network(staged->data(), staged->size());
 *   // Buffer is automatically released when 'staged' goes out of scope
 * }
 * @endcode
 */
class ScopedStagedBuffer {
 public:
  ScopedStagedBuffer() = default;

  ScopedStagedBuffer(GpuTcpStager* stager, void* data, size_t size) : stager_(stager), data_(data), size_(size) {}

  ~ScopedStagedBuffer();

  // Move-only semantics
  ScopedStagedBuffer(const ScopedStagedBuffer&) = delete;
  ScopedStagedBuffer& operator=(const ScopedStagedBuffer&) = delete;

  ScopedStagedBuffer(ScopedStagedBuffer&& other) noexcept
      : stager_(other.stager_), data_(other.data_), size_(other.size_) {
    other.stager_ = nullptr;
    other.data_ = nullptr;
    other.size_ = 0;
  }

  ScopedStagedBuffer& operator=(ScopedStagedBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      stager_ = other.stager_;
      data_ = other.data_;
      size_ = other.size_;
      other.stager_ = nullptr;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /**
   * @brief Release the buffer early (before destructor).
   */
  void reset();

  /**
   * @brief Get the staged data pointer.
   */
  void* data() const {
    return data_;
  }

  /**
   * @brief Get the size of staged data.
   */
  size_t size() const {
    return size_;
  }

  /**
   * @brief Check if this object holds a valid buffer.
   */
  bool valid() const {
    return data_ != nullptr;
  }

  /**
   * @brief Release ownership of the buffer without deallocating.
   * The caller becomes responsible for calling release_staged_buffer().
   */
  void* release() {
    void* ptr = data_;
    stager_ = nullptr;
    data_ = nullptr;
    size_ = 0;
    return ptr;
  }

 private:
  GpuTcpStager* stager_ = nullptr;
  void* data_ = nullptr;
  size_t size_ = 0;
};

/**
 * @brief Manages GPU to CPU staging for TCP transfers.
 *
 * This class handles asynchronous GPU->CPU memory copies using pinned buffers,
 * enabling TCP transport to work with GPU tensors without requiring the entire
 * model to be resident in CPU memory.
 *
 * Internally uses StreamingPinnedBuffer for buffer management.
 */
class GpuTcpStager {
 public:
  /**
   * @brief Constructs a GPU TCP stager with specified buffer configuration.
   * @param chunk_size Size of each staging chunk (e.g., 64 MiB)
   * @param num_chunks Number of chunks for double buffering (default: 2)
   * @param pool Shared pinned memory pool (optional, will create one if not provided)
   */
  explicit GpuTcpStager(
      size_t chunk_size = 64 * 1024 * 1024,
      size_t num_chunks = 2,
      std::shared_ptr<stepcast::store::PinnedMemoryPool> pool = nullptr);
  ~GpuTcpStager();

  // Disable copy and move
  GpuTcpStager(const GpuTcpStager&) = delete;
  GpuTcpStager& operator=(const GpuTcpStager&) = delete;
  GpuTcpStager(GpuTcpStager&&) = delete;
  GpuTcpStager& operator=(GpuTcpStager&&) = delete;

  /**
   * @brief Stage data from GPU tensor to pinned CPU buffer.
   *
   * Performs asynchronous GPU->CPU copy and returns a host pointer to the staged data.
   * This method may block if all staging buffers are in use.
   *
   * WARNING: When using this method, you MUST call release_staged_buffer() with the
   * returned pointer when done. Consider using stage_scoped() instead for automatic
   * resource management.
   *
   * @param tensor The GPU tensor to read from
   * @param offset Offset within the tensor
   * @param bytes Number of bytes to stage
   * @return Host pointer to staged data or error status
   */
  absl::StatusOr<void*> stage(const std::shared_ptr<PartitionTensor>& tensor, uint64_t offset, uint64_t bytes)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Stage data with automatic resource management (RECOMMENDED).
   *
   * Returns a ScopedStagedBuffer that automatically releases the buffer when it
   * goes out of scope. This is the preferred method to avoid resource leaks.
   *
   * @param tensor The GPU tensor to read from
   * @param offset Offset within the tensor
   * @param bytes Number of bytes to stage
   * @return ScopedStagedBuffer or error status
   */
  absl::StatusOr<ScopedStagedBuffer> stage_scoped(
      const std::shared_ptr<PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Asynchronously stage data from GPU tensor to pinned CPU buffer.
   *
   * Starts an asynchronous GPU->CPU copy and returns immediately.
   * The caller must call wait_staging_complete() before using the data.
   *
   * @param tensor The GPU tensor to read from
   * @param offset Offset within the tensor
   * @param bytes Number of bytes to stage
   * @return Slot ID for the staging operation or error status
   */
  absl::StatusOr<int> stage_async(const std::shared_ptr<PartitionTensor>& tensor, uint64_t offset, uint64_t bytes)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Wait for an async staging operation to complete.
   *
   * @param slot_id The slot ID returned by stage_async
   * @return Host pointer to staged data or error status
   */
  absl::StatusOr<void*> wait_staging_complete(int slot_id) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Mark a staged buffer as available for reuse.
   *
   * Should be called after the staged data has been sent over the network.
   *
   * @param host_ptr The host pointer returned by stage()
   * @return Status indicating success or failure
   */
  absl::Status release_staged_buffer(void* host_ptr) ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get statistics about staging operations.
   */
  struct Stats {
    uint64_t total_staged_bytes = 0;
    uint64_t total_stage_calls = 0;
    uint64_t buffer_wait_count = 0;
  };

  Stats get_stats() const ABSL_LOCKS_EXCLUDED(mutex_);

  /**
   * @brief Get the chunk size configured for this stager.
   */
  size_t get_chunk_size() const {
    return chunk_size_;
  }

  /**
   * @brief Get the number of buffers configured for this stager.
   */
  size_t get_num_buffers() const {
    return num_chunks_;
  }

 private:
  struct StagingOperation {
    int slot_id;
    cudaEvent_t copy_complete_event = nullptr;
    void* host_ptr = nullptr;
  };

  const size_t chunk_size_;
  const size_t num_chunks_;

  mutable absl::Mutex mutex_;

  // Use existing infrastructure
  std::shared_ptr<stepcast::store::PinnedMemoryPool> memory_pool_;
  std::unique_ptr<stepcast::store::StreamingPinnedBuffer> streaming_buffer_;

  // Track ongoing operations
  std::unordered_map<int, StagingOperation> active_operations_ ABSL_GUARDED_BY(mutex_);
  std::unordered_map<void*, int> ptr_to_slot_ ABSL_GUARDED_BY(mutex_);

  // Statistics
  Stats stats_ ABSL_GUARDED_BY(mutex_);

  // CUDA resources
  cudaStream_t copy_stream_ = nullptr;
  std::vector<cudaEvent_t> cuda_events_; // Pool of reusable events
  std::queue<cudaEvent_t> free_events_ ABSL_GUARDED_BY(mutex_);

  // Helper methods
  absl::Status perform_staging_copy(
      const std::shared_ptr<PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes,
      void* dest_ptr,
      cudaEvent_t event) ABSL_LOCKS_EXCLUDED(mutex_);

  cudaEvent_t get_free_event() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  void return_event(cudaEvent_t event) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

} // namespace stepcast::communicator