// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include "gsl/pointers"

#include "absl/status/status.h"
#include "core/common/memory/cuda_memory.h"
#include "core/store/materialization/dataplane/contracts/sink.h"

namespace tensorcast::store::loader {

inline constexpr uint64_t DEFAULT_GPU_SCHED_LIMIT_BYTES = 512ull * 1024ull * 1024ull;
inline constexpr uint64_t DEFAULT_GPU_SCHED_LIMIT_COPIES = 2;

struct GpuSchedulerStats {
  uint64_t waits = 0;
  double wait_sec = 0.0;
  uint64_t inflight_bytes = 0;
  uint64_t inflight_copies = 0;
  uint64_t limit_bytes = DEFAULT_GPU_SCHED_LIMIT_BYTES;
  uint64_t limit_copies = DEFAULT_GPU_SCHED_LIMIT_COPIES;
  bool enabled = true;
};

GpuSchedulerStats get_gpu_scheduler_stats(int device_id);
void reset_gpu_scheduler_stats_for_testing();

class GpuMemorySink : public Sink, public PositionedSink, public AsyncPositionedSink {
 public:
  struct Options {
    gsl::not_null<void*> gpu_base_ptr;
    uint64_t total_size = 0;
    size_t chunk_size = 128 * 1024 * 1024; // 128MB default
    int device_id = 0;
    std::shared_ptr<common::memory::GpuDeviceMemory> allocation;
    bool gpu_sched_enabled = true;
    uint64_t gpu_sched_limit_bytes = DEFAULT_GPU_SCHED_LIMIT_BYTES;
    uint64_t gpu_sched_limit_copies = DEFAULT_GPU_SCHED_LIMIT_COPIES;
  };

  explicit GpuMemorySink(Options options);
  ~GpuMemorySink() override;

  absl::Status write(const void* src, size_t bytes) override;

  // Positioned write into GPU base + offset
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;

  absl::StatusOr<common::CopyHandle> write_at_async(
      uint64_t offset,
      const void* src,
      size_t bytes,
      const AsyncWriteOptions& options = AsyncWriteOptions{}) override;

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
