// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/store/materialization/dataplane/contracts/buffer_pool.h"
#include "core/store/materialization/dataplane/contracts/sink.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "folly/Executor.h"

namespace tensorcast::store::loader {

// Convenience alias for byte ranges used by pump_ranges
using Range = std::pair<uint64_t, size_t>;

struct PumpDebugStats {
  std::uint64_t produced_chunks{0};
  std::uint64_t produced_bytes{0};
  std::uint64_t source_read_at_us_total{0};
  std::uint64_t gpu_write_wait_us_total{0};
  std::uint64_t gpu_write_bytes_total{0};
};

struct PumpDirectWriteOptions {
  size_t direct_write_batch_bytes = 0;
  size_t direct_write_batch_ops = 0;
};

// Execute a direct-write-only transfer for the provided ranges without
// allocating staging buffers from a BufferPool. The caller owns destination
// lifecycle and must call dst.close() separately when appropriate.
absl::Status pump_ranges_direct_write(
    SeekableSource& src,
    PositionedSink& dst,
    absl::Span<const Range> ranges,
    size_t window_bytes,
    int concurrency,
    PumpDebugStats* debug_stats = nullptr,
    PumpDirectWriteOptions direct_write_options = {});

absl::Status pump_ranges(
    SeekableSource& src,
    PositionedSink& dst,
    BufferPool& pool,
    absl::Span<const Range> ranges,
    int concurrency,
    folly::Executor::KeepAlive<> executor,
    PumpDebugStats* debug_stats = nullptr,
    PumpDirectWriteOptions direct_write_options = {});

} // namespace tensorcast::store::loader
