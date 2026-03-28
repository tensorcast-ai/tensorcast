// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "core/store/materialization/dataplane/contracts/sink.h"
#include "core/store/materialization/dataplane/sinks/gpu_memory_sink.h"
#include "gsl/pointers"

namespace tensorcast::store::loader {

struct TargetStorage {
  gsl::not_null<void*> gpu_base_ptr;
  uint64_t length{0};
};

class TargetLayoutGpuSink : public Sink, public PositionedSink, public AsyncPositionedSink {
 public:
  struct Options {
    std::vector<TargetStorage> storages;
    size_t chunk_size = 128 * 1024 * 1024;
    int device_id = 0;
    bool gpu_sched_enabled = true;
    uint64_t gpu_sched_limit_bytes = DEFAULT_GPU_SCHED_LIMIT_BYTES;
    uint64_t gpu_sched_limit_copies = DEFAULT_GPU_SCHED_LIMIT_COPIES;
  };

  explicit TargetLayoutGpuSink(Options options);
  ~TargetLayoutGpuSink() override;

  absl::Status write(const void* src, size_t bytes) override;
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override;
  absl::StatusOr<common::CopyHandle> write_at_async(
      uint64_t offset,
      const void* src,
      size_t bytes,
      const AsyncWriteOptions& options = AsyncWriteOptions{}) override;

  absl::Status close() override;

  [[nodiscard]] uint64_t total_size() const {
    return total_size_;
  }

 private:
  struct StorageState {
    uint64_t base_offset{0};
    uint64_t length{0};
    uint64_t covered_bytes{0};
    std::map<uint64_t, uint64_t> covered_intervals;
    std::unique_ptr<GpuMemorySink> sink;
  };

  size_t locate_storage(uint64_t offset, size_t bytes) const;
  void mark_storage_covered(StorageState& state, uint64_t local_offset, size_t bytes);

  std::vector<StorageState> storage_states_;
  uint64_t total_size_{0};
  uint64_t current_offset_{0};
  absl::Status overall_status_;
};

} // namespace tensorcast::store::loader
