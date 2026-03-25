// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/materialization_target_plan_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "catch2/catch_test_macros.hpp"

namespace tensorcast::daemon::materialization_target_plan {
namespace {

using tensorcast::store::loader::ByteRangeMap;
using tensorcast::store::loader::ByteRangeSegment;
using tensorcast::store::materialization::contracts::BindingOpKind;
using tensorcast::store::materialization::contracts::CoverageKind;
using tensorcast::store::materialization::contracts::normalize_representation_transform_contract;
using tensorcast::store::materialization::contracts::RealizationKind;
using tensorcast::store::materialization::contracts::RepresentationTensorBinding;
using tensorcast::store::materialization::contracts::RepresentationTensorSpec;
using tensorcast::store::materialization::contracts::RepresentationTransformContract;
using tensorcast::store::materialization::contracts::SourceFragment;

RepresentationTensorSpec make_tensor_spec(
    std::string name,
    uint64_t logical_offset,
    uint64_t logical_length,
    std::vector<int64_t> shape,
    std::vector<int64_t> stride) {
  return RepresentationTensorSpec{
      .name = std::move(name),
      .shape = std::move(shape),
      .stride = std::move(stride),
      .dtype = "torch.float16",
      .logical_offset = logical_offset,
      .logical_length = logical_length,
      .storage_offset = 0,
      .element_size = 2,
  };
}

tensorcast::common::v1::ByteSpaceRef canonical_byte_space() {
  tensorcast::common::v1::ByteSpaceRef out;
  out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  return out;
}

} // namespace

TEST_CASE(
    "build_resolved_mapped_materialization_plan keeps contract hash immutable while rebinding work plan sources",
    "[materialization_target_plan]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.target_representation.realization_kind = RealizationKind::kEphemeralIntoTarget;
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "tensor",
          .dst_spec = make_tensor_spec("tensor", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kExactCopy,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("tensor", 0, 16, {8}, {1}),
          }},
          .coverage_kind = CoverageKind::kExact,
      },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  const std::string original_hash = normalized_or->target_representation.representation_contract_hash;

  ByteRangeMap generic_fallback_map;
  generic_fallback_map.total_bytes = 16;
  generic_fallback_map.num_sources = 1;
  generic_fallback_map.segments.push_back(
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 16,
          .src_offset = 0,
          .source_index = 0,
      });

  MappedTargetMaterializationPlan mapped_plan;
  mapped_plan.representation = representation_transform_builder::BuildRepresentationTransformResult{
      .generic_fallback_map = generic_fallback_map,
      .transform_contract = *normalized_or,
      .total_bytes_copied = 16,
  };
  mapped_plan.canonical_index_json = R"({"tensor":[0,16,[8],[1],"torch.float16",0]})";
  mapped_plan.logical_total_size = 16;

  store::loading::IntoTargetLayout target_layout;
  target_layout.total_size = 16;
  const std::string physical_source_index_json = R"({"tensor":[32,16,[8],[1],"torch.float16",0]})";
  auto resolved_plan_or = build_resolved_mapped_materialization_plan(
      "artifact",
      /*generation=*/7,
      target_layout,
      mapped_plan,
      std::nullopt,
      std::optional<std::string_view>(physical_source_index_json));
  REQUIRE(resolved_plan_or.ok());
  REQUIRE(resolved_plan_or->representation_transform_contract.has_value());
  REQUIRE(resolved_plan_or->representation_work_plan.has_value());

  CHECK(
      resolved_plan_or->representation_transform_contract->target_representation.representation_contract_hash ==
      original_hash);
  CHECK(
      resolved_plan_or->representation_transform_contract->tensor_bindings.front()
          .sources.front()
          .source_spec.logical_offset == 0);
  REQUIRE(resolved_plan_or->representation_work_plan->items.size() == 1);
  REQUIRE(resolved_plan_or->representation_work_plan->items.front().sources.size() == 1);
  CHECK(
      resolved_plan_or->representation_work_plan->items.front().sources.front().fragment.source_spec.logical_offset ==
      32);
}

} // namespace tensorcast::daemon::materialization_target_plan
