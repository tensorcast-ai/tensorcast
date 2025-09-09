// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>

#include "absl/status/status.h"
#include "core/store/loader/sink.h"

namespace tensorcast::store::loader {

class GPUMemorySink : public Sink, public PositionedSink, public AsyncPositionedSink {
 public:
  struct Options {
    void* gpu_base_ptr = nullptr;
    uint64_t total_size = 0;
    size_t chunk_size = 128 * 1024 * 1024; // 128MB default
    int device_id = 0;
  };

  explicit GPUMemorySink(Options options);
  ~GPUMemorySink() override;

  absl::Status write(const void* src, size_t bytes) override;

  // Positioned write into GPU base + offset
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;

  absl::StatusOr<common::CopyHandle> write_at_async(uint64_t offset, const void* src, size_t bytes) override;

  absl::Status close() override;

 private:
  Options options_;
  cudaStream_t h2d_stream_ = nullptr;
  uint64_t current_offset_ = 0;
  bool stream_created_ = false;
  absl::Status overall_status_;
  // Tracks total bytes successfully enqueued via write_at to validate completeness
  uint64_t total_bytes_written_ = 0;
};

} // namespace tensorcast::store::loader
