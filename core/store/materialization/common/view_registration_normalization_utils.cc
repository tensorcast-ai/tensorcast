// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/common/view_registration_normalization_utils.h"

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "core/store/materialization/common/piece_view_state_utils.h"
#include "core/store/materialization/dataplane/view/view_identity.h"
#include "core/store/view_utils.h"

namespace tensorcast::store::materialization::common {

namespace {

bool ranges_cover_interval(
    const std::vector<::tensorcast::store::runtime::metadata::CanonicalRange>& ranges,
    uint64_t start,
    uint64_t length) {
  if (length == 0) {
    return true;
  }
  uint64_t cursor = start;
  const uint64_t end = start + length;
  for (const auto& range : ranges) {
    if (range.length == 0) {
      continue;
    }
    const uint64_t range_start = range.offset;
    const uint64_t range_end = range.offset + range.length;
    if (range_end <= cursor) {
      continue;
    }
    if (range_start > cursor) {
      return false;
    }
    cursor = std::min(end, range_end);
    if (cursor >= end) {
      return true;
    }
  }
  return cursor >= end;
}

} // namespace

absl::StatusOr<NormalizedViewRegistration> normalize_view_registration(
    const ::tensorcast::store::runtime::metadata::ViewRegistration& registration,
    std::string_view canonical_index_json,
    uint64_t default_canonical_size_bytes) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("view registration requires inline canonical index data");
  }
  if (registration.spec.tensors.empty() && registration.tensor_names.empty()) {
    return absl::InvalidArgumentError("view registration requires a view spec or tensor_names");
  }
  if (registration.placement == ::tensorcast::store::runtime::metadata::ViewPlacement::kUnspecified) {
    return absl::InvalidArgumentError("view registration requires explicit placement");
  }
  if (registration.registration_kind == ::tensorcast::store::runtime::metadata::ViewRegistrationKind::kUnspecified) {
    return absl::InvalidArgumentError("view.registration_kind must be specified");
  }

  NormalizedViewRegistration normalized;
  normalized.registration = registration;

  auto computed_view_id_or = loader::compute_view_id_from_spec(registration.spec, canonical_index_json);
  if (!computed_view_id_or.ok()) {
    return computed_view_id_or.status();
  }
  if (normalized.registration.view_id.empty()) {
    normalized.registration.view_id = *computed_view_id_or;
  } else if (normalized.registration.view_id != *computed_view_id_or) {
    return absl::InvalidArgumentError("view_id does not match view spec");
  }

  normalized.view_spec_json =
      store::view::build_view_spec_json(normalized.registration.spec, normalized.registration.tensor_names);

  auto plan_or = normalized.registration.tensor_names.empty()
      ? loader::ViewPlanner::compute_bidirectional_view_plan(canonical_index_json, normalized.registration.spec)
      : loader::ViewPlanner::compute_bidirectional_view_plan(
            canonical_index_json, normalized.registration.spec, normalized.registration.tensor_names);
  if (!plan_or.ok()) {
    return plan_or.status();
  }
  normalized.plan = std::move(*plan_or);
  normalized.expected_view_bytes = 0;
  for (const auto& chunk : normalized.plan.write.chunks) {
    normalized.expected_view_bytes += chunk.length;
  }
  normalized.view_size_bytes = normalized.plan.forward.view_size_bytes;

  const auto computed_ranges = canonical_ranges_from_write_plan(normalized.plan.write);
  normalized.canonical_ranges.reserve(computed_ranges.size());
  for (const auto& range : computed_ranges) {
    ::tensorcast::store::runtime::metadata::CanonicalRange converted;
    converted.offset = range.offset;
    converted.length = range.length;
    normalized.canonical_ranges.push_back(converted);
    normalized.canonical_bytes_covered += converted.length;
  }

  if (normalized.registration.registration_kind ==
      ::tensorcast::store::runtime::metadata::ViewRegistrationKind::kPiece) {
    if (normalized.registration.canonical_size_bytes == 0) {
      return absl::InvalidArgumentError("piece registration requires canonical_size_bytes");
    }
    normalized.canonical_size_bytes = normalized.registration.canonical_size_bytes;
  } else {
    if (normalized.registration.canonical_size_bytes != 0 &&
        normalized.registration.canonical_size_bytes != default_canonical_size_bytes) {
      return absl::InvalidArgumentError("view.canonical_size_bytes must match total_size_bytes");
    }
    normalized.canonical_size_bytes = normalized.registration.canonical_size_bytes != 0
        ? normalized.registration.canonical_size_bytes
        : default_canonical_size_bytes;
  }

  if (normalized.canonical_bytes_covered > normalized.canonical_size_bytes) {
    return absl::InvalidArgumentError("view registration exceeds canonical byte space");
  }

  const bool fully_covers_canonical =
      ranges_cover_interval(normalized.canonical_ranges, 0, normalized.canonical_size_bytes);
  if (normalized.registration.registration_kind ==
      ::tensorcast::store::runtime::metadata::ViewRegistrationKind::kPiece) {
    if (fully_covers_canonical) {
      return absl::InvalidArgumentError(
          "piece registration must not fully cover canonical bytes; use "
          "registration_kind=CANONICAL");
    }
  } else if (normalized.canonical_bytes_covered != normalized.canonical_size_bytes) {
    return absl::InvalidArgumentError(
        "canonical view registration must fully cover canonical bytes; use "
        "registration_kind=PIECE for partial");
  }

  normalized.registration.canonical_size_bytes = normalized.canonical_size_bytes;
  normalized.registration.canonical_ranges = normalized.canonical_ranges;
  return normalized;
}

} // namespace tensorcast::store::materialization::common
