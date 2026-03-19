// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "core/store/runtime/metadata/metadata_types.h"

namespace tensorcast::store::materialization::common {

struct NormalizedViewRegistration {
  ::tensorcast::store::runtime::metadata::ViewRegistration registration;
  loader::BidirectionalViewPlan plan;
  std::string view_spec_json;
  uint64_t canonical_size_bytes{0};
  uint64_t canonical_bytes_covered{0};
  uint64_t expected_view_bytes{0};
  uint64_t view_size_bytes{0};
  std::vector<::tensorcast::store::runtime::metadata::CanonicalRange> canonical_ranges;
};

absl::StatusOr<NormalizedViewRegistration> normalize_view_registration(
    const ::tensorcast::store::runtime::metadata::ViewRegistration& registration,
    std::string_view canonical_index_json,
    uint64_t default_canonical_size_bytes);

} // namespace tensorcast::store::materialization::common
