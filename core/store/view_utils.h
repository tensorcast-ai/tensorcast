// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/materialization/dataplane/view/view_planner.h"

namespace tensorcast::store::view {

struct CanonicalRange {
  uint64_t offset{0};
  uint64_t length{0};
};

struct ParsedViewSelection {
  loader::ViewSpec spec;
  std::vector<std::string> tensor_names;
};

std::string build_view_spec_json(const loader::ViewSpec& spec, absl::Span<const std::string> tensor_names = {});
absl::StatusOr<ParsedViewSelection> parse_view_selection_json(std::string_view view_spec_json);
absl::StatusOr<loader::ViewSpec> parse_view_spec_json(std::string_view view_spec_json);

uint64_t align_up(uint64_t value, uint64_t align);
uint64_t align_down(uint64_t value, uint64_t align);

std::vector<uint64_t> compute_fully_covered_canonical_leaf_indices(
    absl::Span<const CanonicalRange> ranges,
    uint64_t chunk_bytes);

} // namespace tensorcast::store::view
