// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include "gsl/pointers"

#include "absl/status/status.h"
#include "core/store/loader/sink.h"

namespace tensorcast::store::loader {

class GpuMemorySink : public Sink, public PositionedSink, public AsyncPositionedSink {
 public:
  struct Options {
    gsl::not_null<void*> gpu_base_ptr;
    uint64_t total_size = 0;
    size_t chunk_size = 128 * 1024 * 1024; // 128MB default
    int device_id = 0;
  };

  explicit GpuMemorySink(Options options);
  ~GpuMemorySink() override;

  absl::Status write(const void* src, size_t bytes) override;

  // Positioned write into GPU base + offset
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;

  absl::StatusOr<common::CopyHandle> write_at_async(uint64_t offset, const void* src, size_t bytes) override;

  absl::Status close() override;

 private:
  Options options_;
  uint64_t current_offset_ = 0;
  absl::Status overall_status_;
  // Tracks total bytes successfully transferred via write/write_at for
  // compatibility with tests that validate completeness on close().
  uint64_t total_bytes_written_ = 0;
};

} // namespace tensorcast::store::loader
