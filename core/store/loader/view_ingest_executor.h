// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/common/memory/memory_location.h"
#include "core/store/loader/view_planner.h"

namespace tensorcast::store::loader {

/**
 * ViewIngestExecutor incrementally copies view-formatted bytes into canonical
 * replica memory and applies inverse transforms to recover canonical layout.
 *
 * Registration streaming feeds call ingest_chunk() with server-received view
 * payloads. finalize() must be called once all expected bytes have been
 * ingested to run any pending inverse transforms (e.g., transpose).
 */
class ViewIngestExecutor {
 public:
  ViewIngestExecutor(ViewWritePlan write_plan, TransformPlan inverse_transform);

  [[nodiscard]] uint64_t expected_view_bytes() const {
    return total_view_bytes_;
  }

  [[nodiscard]] uint64_t ingested_bytes() const {
    return ingested_bytes_;
  }

  [[nodiscard]] bool is_complete() const {
    return ingested_bytes_ == total_view_bytes_;
  }

  absl::Status ingest_chunk(
      uint64_t view_offset,
      absl::Span<const std::byte> data,
      common::memory::MemoryLocation location,
      void* canonical_base_ptr,
      int device_id);

  absl::Status finalize(common::memory::MemoryLocation location, void* canonical_base_ptr, int device_id);

 private:
  struct ChunkState {
    ViewWritePlan::Chunk chunk;
    uint64_t written{0};
  };

  absl::Status copy_into_canonical(
      ChunkState& chunk,
      uint64_t chunk_offset_bytes,
      absl::Span<const std::byte> data,
      common::memory::MemoryLocation location,
      void* canonical_base_ptr,
      int device_id);

  std::vector<ChunkState> chunks_;
  TransformPlan inverse_transform_;
  uint64_t total_view_bytes_{0};
  uint64_t ingested_bytes_{0};
  size_t current_chunk_idx_{0};
  bool finalized_{false};
};

} // namespace tensorcast::store::loader
