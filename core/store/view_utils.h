// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "core/store/loader/view_planner.h"

namespace tensorcast::store::view {

struct CanonicalRange {
  uint64_t offset{0};
  uint64_t length{0};
};

std::string build_view_spec_json(const loader::ViewSpec& spec);

uint64_t align_up(uint64_t value, uint64_t align);
uint64_t align_down(uint64_t value, uint64_t align);

std::vector<uint64_t> compute_fully_covered_canonical_leaf_indices(
    absl::Span<const CanonicalRange> ranges,
    uint64_t chunk_bytes);

} // namespace tensorcast::store::view
