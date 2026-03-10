// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/dataplane/contracts/sink.h"
#include "core/store/materialization/dataplane/contracts/source.h"

namespace tensorcast::store::loader {

class SourceWindowScheduler {
 public:
  struct Options {
    uint64_t merge_max_gap_bytes{0};
    uint32_t merge_max_amplification{0};
    uint32_t prefetch_depth{1};
    uint64_t window_cap_bytes{0};
    std::string path;
  };

  explicit SourceWindowScheduler(Options options);

  absl::Status Execute(
      const ByteRangeMap& map,
      absl::Span<const std::shared_ptr<SeekableSource>> sources,
      PositionedSink& sink,
      std::shared_ptr<common::memory::PinnedBufferPool> pinned_pool,
      std::chrono::milliseconds pinned_timeout,
      bool use_pinned_buffers);

 private:
  Options options_;
};

} // namespace tensorcast::store::loader
