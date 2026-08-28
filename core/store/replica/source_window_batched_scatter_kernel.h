// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace tensorcast::store::replica {

struct SourceWindowBatchedScatterDescriptor {
  uint64_t src_ptr{0};
  uint64_t dst_ptr{0};
  uint64_t row_bytes{0};
  uint64_t row_count{0};
  uint64_t source_stride_bytes{0};
  uint64_t target_stride_bytes{0};
};

constexpr size_t source_window_batched_scatter_descriptor_bytes() {
  return sizeof(SourceWindowBatchedScatterDescriptor);
}

absl::Status launch_source_window_batched_scatter(
    absl::Span<const SourceWindowBatchedScatterDescriptor> descriptors,
    void* device_descriptor_buffer,
    size_t device_descriptor_capacity_bytes,
    int device_id,
    cudaStream_t stream);

absl::Status prewarm_source_window_batched_scatter_kernel_for_device(int device_id);

absl::Status prewarm_source_window_batched_scatter_kernel_for_visible_devices();

} // namespace tensorcast::store::replica
