// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/representation_contract.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/representation_layout_types.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::representation_transform_builder {

struct TransformWorkCompatibilityStats {
  uint64_t total_dst_tensors{0};
  uint64_t compatible_candidates{0};
  uint64_t compatible_bytes{0};
  uint64_t concat_candidates{0};
  uint64_t concat_bytes{0};
  uint64_t rejected_mixed_src_or_dim{0};
  uint64_t rejected_mixed_src_or_dim_bytes{0};
  uint64_t rejected_non_contiguous{0};
  uint64_t rejected_non_contiguous_bytes{0};
  uint64_t rejected_unsupported_distribution{0};
  uint64_t rejected_unsupported_distribution_bytes{0};
};

struct BuildRepresentationTransformResult {
  tensorcast::store::loader::ByteRangeMap generic_fallback_map;
  tensorcast::store::materialization::contracts::RepresentationTransformContract transform_contract;
  uint64_t total_bytes_copied{0};
  TransformWorkCompatibilityStats compatibility_stats;
};

absl::StatusOr<BuildRepresentationTransformResult> build_representation_transform_contract(
    const v2::CopyPlan& copy_plan,
    const materialization_layout::CanonicalIndexTable& source_table,
    const materialization_layout::CanonicalIndexTable& canonical_source_table,
    const absl::flat_hash_map<std::string, representation_layout::TensorLayoutSpec>& dst_specs,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec>& view_narrows,
    const tensorcast::common::v1::ByteSpaceRef& source_byte_space,
    std::string_view representation_family);

absl::StatusOr<BuildRepresentationTransformResult> build_representation_transform_contract(
    const v2::BindingRealizationPlan& realization_plan,
    const materialization_layout::CanonicalIndexTable& source_table,
    const materialization_layout::CanonicalIndexTable& canonical_source_table,
    const absl::flat_hash_map<std::string, representation_layout::TensorLayoutSpec>& dst_specs,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec>& view_narrows,
    const tensorcast::common::v1::ByteSpaceRef& source_byte_space,
    std::string_view representation_family);

} // namespace tensorcast::daemon::representation_transform_builder
