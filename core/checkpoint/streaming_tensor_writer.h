// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/cuda_api.h"

#include "core/checkpoint/tensor_writer.h"
#include "core/common/memory/streaming_pinned_buffer.h"

namespace tensorcast::checkpoint {

/**
 * @brief A tensor writer that uses streaming pinned buffer for efficient GPU tensor saving.
 *
 * This class provides asynchronous GPU->Host copy and disk writing using a producer-consumer
 * pattern with StreamingPinnedBuffer.
 */
class StreamingTensorWriter {
 public:
  /**
   * @brief Configuration for StreamingTensorWriter
   */
  struct Config {
    size_t num_buffers = 4; // Number of buffers in the circular buffer
    size_t buffer_size_mb = 256; // Size of each buffer in MB
    bool enable_async_write = true; // Enable asynchronous disk writing
    size_t pool_size_gb = 10; // Total pinned pool size in GB
  } __attribute__((aligned(32))) __attribute__((packed));

  /**
   * @brief Construct a streaming tensor writer
   * @param filename Base filename for tensor data
   * @param config Configuration parameters
   * @param pool Shared pinned memory pool
   */
  StreamingTensorWriter(
      std::string filename,
      Config config,
      std::shared_ptr<tensorcast::common::memory::PinnedMemoryPool> pool);

  ~StreamingTensorWriter();

  /**
   * @brief Initialize the writer and start background threads
   * @return Status indicating success or failure
   */
  absl::Status initialize();

  /**
   * @brief Write tensor data (CPU or GPU)
   * @param data Pointer to tensor data
   * @param size Size of tensor data in bytes
   * @param is_gpu Whether the data is on GPU
   * @param stream CUDA stream for async copy (optional)
   * @return Offset where data was written or error status
   */
  absl::StatusOr<uint64_t> write_tensor(const void* data, size_t size, bool is_gpu, cudaStream_t stream = nullptr);

  /**
   * @brief Finalize writing and wait for all pending operations
   * @return Status indicating success or failure
   */
  absl::Status finalize();

  /**
   * @brief Get total bytes written
   */
  uint64_t total_bytes_written() const {
    return total_bytes_written_.load();
  }

 private:
  // Worker thread for disk writing
  void disk_writer_thread();

  // Helper to copy data from GPU to pinned buffer
  static absl::Status copy_gpu_to_buffer(const void* gpu_data, size_t size, char* buffer, cudaStream_t stream);

  // Configuration
  const std::string filename_;
  const Config config_;
  const size_t buffer_size_;

  // Streaming buffer for GPU->Host->Disk pipeline
  std::unique_ptr<tensorcast::common::memory::StreamingPinnedBuffer> streaming_buffer_;

  // Underlying tensor writer for disk I/O
  std::unique_ptr<TensorWriter> tensor_writer_;

  // Background thread for disk writing
  std::thread disk_writer_thread_;
  std::atomic<bool> shutdown_{false};

  // Statistics
  std::atomic<uint64_t> total_bytes_written_{0};
  std::atomic<uint64_t> current_offset_{0};

  // Offset tracking for chunks
  std::unordered_map<size_t, uint64_t> chunk_offsets_;

  // State
  bool initialized_{false};
};

} // namespace tensorcast::checkpoint
