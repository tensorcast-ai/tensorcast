// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/store/loader/buffer_pool.h"
#include "core/store/loader/sink.h"
#include "core/store/loader/source.h"

namespace stepcast::store::loader {

absl::Status pump(Source& src, Sink& dst, BufferPool& pool, int concurrency = 2);

absl::Status pump_ranges(
    SeekableSource& src,
    Sink& dst,
    BufferPool& pool,
    absl::Span<const std::pair<uint64_t, size_t>> ranges,
    int concurrency = 2);

} // namespace stepcast::store::loader