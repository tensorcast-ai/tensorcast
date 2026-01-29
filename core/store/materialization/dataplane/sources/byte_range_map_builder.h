// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string_view>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"

namespace tensorcast::store::loader {

// Build a ByteRangeMap from a canonical index JSON string.
absl::StatusOr<ByteRangeMap> build_byte_range_map_from_canonical_index_json(
    std::string_view index_json,
    uint64_t total_size,
    uint64_t align_bytes = 8);

} // namespace tensorcast::store::loader
