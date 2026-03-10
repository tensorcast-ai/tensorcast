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

// Build a ByteRangeMap from canonical + source index JSON strings.
// Canonical offsets define dst offsets; source offsets define src offsets.
absl::StatusOr<ByteRangeMap> build_byte_range_map_from_canonical_and_source_index_json(
    std::string_view canonical_index_json,
    std::string_view source_index_json,
    uint64_t total_size,
    uint64_t align_bytes = 8);

// Compose an outer map (dst->canonical) with an inner map (canonical->source)
// into a single map (dst->source).
absl::StatusOr<ByteRangeMap> compose_byte_range_maps(const ByteRangeMap& outer, const ByteRangeMap& inner);

} // namespace tensorcast::store::loader
