// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/loader/source.h"

namespace tensorcast::store::loader {

// Segment piece of the linearized Artifact Virtual Byte Stream (AVBS).
// When kind == DATA, bytes are read from GPU memory at src_offset.
// When kind == PAD, the source is implicit zeros for the given length.
struct SegmentPiece {
  enum Kind : uint8_t { DATA = 0, PAD = 1 };
  Kind kind{PAD};
  uint64_t dst_offset{0};
  uint64_t length{0};
  // For DATA pieces only: offset within the underlying coalesced buffer
  uint64_t src_offset{0};
};

// Build a linear SegmentPlan from a canonical index (v2) JSON string.
// The index JSON maps tensor name -> [offset, size, shape, stride, dtype, storage_offset].
// The resulting plan covers [0, total_size) with DATA and PAD segments (8B alignment assumed).
absl::StatusOr<std::vector<SegmentPiece>> build_segment_plan_from_canonical_index_json(
    std::string_view index_json,
    uint64_t total_size,
    uint64_t align_bytes = 8);

// A SeekableSource that linearizes GPU memory according to a SegmentPlan
// and yields zero bytes over PAD regions.
class LinearizedGpuPlanSource final : public SeekableSource {
 public:
  LinearizedGpuPlanSource(void* device_ptr, int device_id, absl::Span<const SegmentPiece> plan, uint64_t total_size);

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

 private:
  void* device_ptr_;
  int device_id_;
  std::vector<SegmentPiece> plan_;
  uint64_t total_size_;
  uint64_t current_offset_{0};
};

// Compute data multihash by streaming over a SegmentPlan (PAD=0 semantics).
absl::StatusOr<std::string> compute_data_multihash_from_gpu_plan(
    void* device_ptr,
    int device_id,
    absl::Span<const SegmentPiece> plan,
    uint64_t total_size,
    size_t leaf_chunk_bytes = 4ULL * 1024 * 1024);

} // namespace tensorcast::store::loader
