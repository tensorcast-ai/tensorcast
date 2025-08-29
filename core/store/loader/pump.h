// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/store/loader/buffer_pool.h"
#include "core/store/loader/sink.h"
#include "core/store/loader/source.h"

namespace tensorcast::store::loader {

// Convenience alias for byte ranges used by pump_ranges
using Range = std::pair<uint64_t, size_t>;

absl::Status pump_ranges(
    SeekableSource& src,
    PositionedSink& dst,
    BufferPool& pool,
    absl::Span<const Range> ranges,
    int concurrency = 2);

} // namespace tensorcast::store::loader
