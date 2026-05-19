// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/materialization_target_plan_utils.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/store/runtime/ingestion/source_bound_strategy_planner.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::daemon::materialization_target_plan {
namespace {

using materialization_layout::parse_canonical_index;
using representation_layout::TensorLayoutSpec;
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

TensorLayoutSpec make_layout_spec(uint64_t logical_length, std::vector<int64_t> shape, std::vector<int64_t> stride) {
  return TensorLayoutSpec{
      .shape = std::move(shape),
      .stride = std::move(stride),
      .dtype = "torch.float16",
      .storage_offset = 0,
      .logical_length = logical_length,
      .element_size = 2,
  };
}

tensorcast::common::v1::ByteSpaceRef canonical_byte_space() {
  tensorcast::common::v1::ByteSpaceRef out;
  out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  return out;
}

uint64_t covered_bytes(const ByteRangeMap& map) {
  uint64_t total = 0;
  for (const auto& segment : map.segments) {
    total += segment.length;
  }
  return total;
}

store::runtime::ingestion::strategy::SourceBoundSourceFacts safetensors_disk_source() {
  return store::runtime::ingestion::strategy::SourceBoundSourceFacts{
      .disk_source_available = true,
      .disk_source_is_safetensors = true,
  };
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
  REQUIRE(resolved_plan_or->resolved_plan.representation_transform_contract.has_value());
  REQUIRE(resolved_plan_or->resolved_plan.representation_work_plan.has_value());

  CHECK(
      resolved_plan_or->resolved_plan.representation_transform_contract->target_representation
          .representation_contract_hash == original_hash);
  CHECK(
      resolved_plan_or->resolved_plan.representation_transform_contract->tensor_bindings.front()
          .sources.front()
          .source_spec.logical_offset == 0);
  REQUIRE(resolved_plan_or->resolved_plan.representation_work_plan->items.size() == 1);
  REQUIRE(resolved_plan_or->resolved_plan.representation_work_plan->items.front().sources.size() == 1);
  CHECK(
      resolved_plan_or->resolved_plan.representation_work_plan->items.front()
          .sources.front()
          .fragment.source_spec.logical_offset == 32);
}

TEST_CASE("binding realization copy ranges convert without descriptor mismatch", "[materialization_target_plan]") {
  const std::string index_json = R"({"alpha":[0,8,[4],[1],"torch.float16",0]})";
  auto source_table_or = parse_canonical_index(index_json);
  REQUIRE(source_table_or.ok());
  auto canonical_source_table_or = parse_canonical_index(index_json);
  REQUIRE(canonical_source_table_or.ok());

  absl::flat_hash_map<std::string, TensorLayoutSpec> dst_specs;
  dst_specs.emplace("alpha", make_layout_spec(8, {4}, {1}));
  absl::flat_hash_map<std::string, uint64_t> dst_base_offsets;
  dst_base_offsets.emplace("alpha", 0);
  absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec> view_narrows;

  v2::BindingRealizationPlan realization_plan;
  realization_plan.set_version(1);
  for (const auto [start, end] : std::array<std::pair<int64_t, int64_t>, 2>{{{0, 2}, {2, 4}}}) {
    auto* entry = realization_plan.add_entries();
    entry->set_op_kind(v2::BINDING_REALIZATION_OP_KIND_COPY);
    entry->set_source_name("alpha");
    entry->set_dst_name("alpha");
    auto* src_range = entry->add_source_ranges();
    src_range->set_dim(0);
    src_range->set_start(start);
    src_range->set_end(end);
    auto* dst_range = entry->add_dst_ranges();
    dst_range->set_dim(0);
    dst_range->set_start(start);
    dst_range->set_end(end);
  }

  auto result_or = representation_transform_builder::build_representation_transform_contract(
      realization_plan,
      *source_table_or,
      *canonical_source_table_or,
      dst_specs,
      dst_base_offsets,
      view_narrows,
      canonical_byte_space(),
      "local_seal_then_promote");
  REQUIRE(result_or.ok());
  REQUIRE(result_or->transform_contract.tensor_bindings.size() == 1);
  const auto& binding = result_or->transform_contract.tensor_bindings.front();
  REQUIRE(binding.sources.size() == 1);
  CHECK(binding.dst_name == "alpha");
  CHECK(binding.sources.front().source_spec.name == "alpha");
}

TEST_CASE(
    "binding realization multi-axis copy ranges preserve semantic bindings and byte segments",
    "[materialization_target_plan]") {
  const std::string index_json =
      R"({"gate":[0,16,[2,2,2],[4,2,1],"torch.float16",0],"up":[16,16,[2,2,2],[4,2,1],"torch.float16",0]})";
  auto source_table_or = parse_canonical_index(index_json);
  REQUIRE(source_table_or.ok());
  auto canonical_source_table_or = parse_canonical_index(index_json);
  REQUIRE(canonical_source_table_or.ok());

  absl::flat_hash_map<std::string, TensorLayoutSpec> dst_specs;
  dst_specs.emplace("packed", make_layout_spec(32, {2, 4, 2}, {8, 2, 1}));
  absl::flat_hash_map<std::string, uint64_t> dst_base_offsets;
  dst_base_offsets.emplace("packed", 0);
  absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec> view_narrows;

  v2::BindingRealizationPlan realization_plan;
  realization_plan.set_version(1);
  for (int row = 0; row < 2; ++row) {
    auto* gate = realization_plan.add_entries();
    gate->set_op_kind(v2::BINDING_REALIZATION_OP_KIND_COPY);
    gate->set_source_name("gate");
    gate->set_dst_name("packed");
    auto* gate_src_range = gate->add_source_ranges();
    gate_src_range->set_dim(0);
    gate_src_range->set_start(row);
    gate_src_range->set_end(row + 1);
    auto* gate_dst_row = gate->add_dst_ranges();
    gate_dst_row->set_dim(0);
    gate_dst_row->set_start(row);
    gate_dst_row->set_end(row + 1);
    auto* gate_dst_col = gate->add_dst_ranges();
    gate_dst_col->set_dim(1);
    gate_dst_col->set_start(0);
    gate_dst_col->set_end(2);

    auto* up = realization_plan.add_entries();
    up->set_op_kind(v2::BINDING_REALIZATION_OP_KIND_COPY);
    up->set_source_name("up");
    up->set_dst_name("packed");
    auto* up_src_range = up->add_source_ranges();
    up_src_range->set_dim(0);
    up_src_range->set_start(row);
    up_src_range->set_end(row + 1);
    auto* up_dst_row = up->add_dst_ranges();
    up_dst_row->set_dim(0);
    up_dst_row->set_start(row);
    up_dst_row->set_end(row + 1);
    auto* up_dst_col = up->add_dst_ranges();
    up_dst_col->set_dim(1);
    up_dst_col->set_start(2);
    up_dst_col->set_end(4);
  }

  auto result_or = representation_transform_builder::build_representation_transform_contract(
      realization_plan,
      *source_table_or,
      *canonical_source_table_or,
      dst_specs,
      dst_base_offsets,
      view_narrows,
      canonical_byte_space(),
      "local_seal_then_promote");
  REQUIRE(result_or.ok());
  REQUIRE(result_or->transform_contract.tensor_bindings.size() == 4);
  CHECK(result_or->transform_contract.tensor_bindings.front().dst_name == "packed");
  REQUIRE(result_or->transform_contract.tensor_bindings.front().sources.size() == 1);
  CHECK(result_or->transform_contract.tensor_bindings.front().sources.front().destination_range.axes.size() == 2);
  CHECK(result_or->generic_fallback_map.segments.size() == 4);
  CHECK(result_or->generic_fallback_map.segments.front().length == 8);
  CHECK(result_or->total_bytes_copied == 32);
}

TEST_CASE(
    "binding realization mixed copy and const fill keeps fill-covered bytes out of residual fallback",
    "[materialization_target_plan]") {
  const std::string source_index_json = R"({"weight":[0,8,[4],[1],"torch.float16",0]})";
  auto source_table_or = parse_canonical_index(source_index_json);
  REQUIRE(source_table_or.ok());
  auto canonical_source_table_or = parse_canonical_index(source_index_json);
  REQUIRE(canonical_source_table_or.ok());

  absl::flat_hash_map<std::string, TensorLayoutSpec> dst_specs;
  dst_specs.emplace("weight", make_layout_spec(8, {4}, {1}));
  dst_specs.emplace(
      "manifest",
      TensorLayoutSpec{
          .shape = {4},
          .stride = {1},
          .dtype = "torch.uint8",
          .storage_offset = 0,
          .logical_length = 4,
          .element_size = 1,
      });
  absl::flat_hash_map<std::string, uint64_t> dst_base_offsets;
  dst_base_offsets.emplace("weight", 0);
  dst_base_offsets.emplace("manifest", 8);
  absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec> view_narrows;

  v2::BindingRealizationPlan realization_plan;
  realization_plan.set_version(1);
  {
    auto* copy = realization_plan.add_entries();
    copy->set_op_kind(v2::BINDING_REALIZATION_OP_KIND_COPY);
    copy->set_source_name("weight");
    copy->set_dst_name("weight");
  }
  {
    auto* fill = realization_plan.add_entries();
    fill->set_op_kind(v2::BINDING_REALIZATION_OP_KIND_CONST_FILL);
    fill->set_dst_name("manifest");
    fill->set_fill_value(std::string("\x01\x02\x03\x04", 4));
  }

  auto result_or = representation_transform_builder::build_representation_transform_contract(
      realization_plan,
      *source_table_or,
      *canonical_source_table_or,
      dst_specs,
      dst_base_offsets,
      view_narrows,
      canonical_byte_space(),
      "local_seal_then_promote");
  REQUIRE(result_or.ok());
  REQUIRE(result_or->transform_contract.tensor_bindings.size() == 2);
  REQUIRE(result_or->generic_fallback_map.segments.size() == 1);
  CHECK(result_or->generic_fallback_map.segments[0].kind == ByteRangeSegment::Kind::kData);
  CHECK(result_or->generic_fallback_map.segments[0].dst_offset == 0);
  CHECK(result_or->generic_fallback_map.segments[0].length == 8);

  MappedTargetMaterializationPlan mapped_plan;
  mapped_plan.representation = *result_or;
  mapped_plan.canonical_index_json =
      R"({"weight":[0,8,[4],[1],"torch.float16",0],"manifest":[8,4,[4],[1],"torch.uint8",0]})";
  mapped_plan.logical_total_size = 12;

  store::loading::IntoTargetLayout target_layout;
  target_layout.total_size = 12;
  auto resolved_plan_or = build_resolved_mapped_materialization_plan(
      "artifact",
      /*generation=*/11,
      target_layout,
      mapped_plan,
      std::nullopt,
      std::optional<std::string_view>(source_index_json));
  REQUIRE(resolved_plan_or.ok());
  REQUIRE(resolved_plan_or->resolved_plan.representation_work_plan.has_value());
  CHECK(resolved_plan_or->resolved_plan.representation_work_plan->residual_fallback_map.segments.empty());
  REQUIRE(resolved_plan_or->resolved_plan.representation_work_plan->items.size() == 2);
  CHECK(
      resolved_plan_or->resolved_plan.representation_work_plan->items[0].kind ==
      tensorcast::store::materialization::contracts::RepresentationWorkItemKind::kConstFill);
}

TEST_CASE("copy-plan collective map only keeps source-overlap tensors", "[materialization_target_plan]") {
  const std::string source_index_json = R"({
    "copy":[0,8,[4],[1],"torch.float16",0],
    "left":[8,4,[2],[1],"torch.float16",0],
    "right":[12,4,[2],[1],"torch.float16",0]
  })";
  auto source_table_or = parse_canonical_index(source_index_json);
  REQUIRE(source_table_or.ok());
  auto canonical_source_table_or = parse_canonical_index(source_index_json);
  REQUIRE(canonical_source_table_or.ok());

  absl::flat_hash_map<std::string, TensorLayoutSpec> dst_specs;
  dst_specs.emplace("copy_dst", make_layout_spec(8, {4}, {1}));
  dst_specs.emplace("concat_dst", make_layout_spec(8, {4}, {1}));
  absl::flat_hash_map<std::string, uint64_t> dst_base_offsets;
  dst_base_offsets.emplace("copy_dst", 0);
  dst_base_offsets.emplace("concat_dst", 8);
  absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec> view_narrows;

  v2::CopyPlan copy_plan;
  copy_plan.set_version(1);
  {
    auto* entry = copy_plan.add_entries();
    entry->set_ckpt_name("copy");
    entry->set_dst_name("copy_dst");
  }
  {
    auto* entry = copy_plan.add_entries();
    entry->set_ckpt_name("left");
    entry->set_dst_name("concat_dst");
    auto* src_range = entry->mutable_ckpt_range();
    src_range->set_dim(0);
    src_range->set_start(0);
    src_range->set_end(2);
    auto* dst_range = entry->mutable_dst_range();
    dst_range->set_dim(0);
    dst_range->set_start(0);
    dst_range->set_end(2);
  }
  {
    auto* entry = copy_plan.add_entries();
    entry->set_ckpt_name("right");
    entry->set_dst_name("concat_dst");
    auto* src_range = entry->mutable_ckpt_range();
    src_range->set_dim(0);
    src_range->set_start(0);
    src_range->set_end(2);
    auto* dst_range = entry->mutable_dst_range();
    dst_range->set_dim(0);
    dst_range->set_start(2);
    dst_range->set_end(4);
  }

  auto transform_or = representation_transform_builder::build_representation_transform_contract(
      copy_plan,
      *source_table_or,
      *canonical_source_table_or,
      dst_specs,
      dst_base_offsets,
      view_narrows,
      canonical_byte_space(),
      "local_seal_then_promote");
  REQUIRE(transform_or.ok());
  CHECK(transform_or->compatibility_stats.compatible_candidates == 1);
  CHECK(transform_or->compatibility_stats.concat_candidates == 1);
  CHECK(covered_bytes(transform_or->generic_fallback_map) == 16);
  CHECK(covered_bytes(transform_or->compatibility_lowered_map) == 16);
  REQUIRE(transform_or->compatibility_lowered_map.segments.size() == 3);

  MappedTargetMaterializationPlan mapped_plan;
  mapped_plan.representation = *transform_or;
  mapped_plan.canonical_index_json =
      R"({"copy_dst":[0,8,[4],[1],"torch.float16",0],"concat_dst":[8,8,[4],[1],"torch.float16",0]})";
  mapped_plan.logical_total_size = 16;

  store::loading::IntoTargetLayout target_layout;
  target_layout.total_size = 16;
  auto prepared_or = build_resolved_mapped_materialization_plan(
      "artifact",
      /*generation=*/13,
      target_layout,
      mapped_plan,
      std::nullopt,
      std::optional<std::string_view>(source_index_json));
  REQUIRE(prepared_or.ok());
  REQUIRE(prepared_or->lowering_artifacts.has_value());
  REQUIRE(prepared_or->lowering_artifacts->collective_data_map.has_value());
  CHECK(covered_bytes(*prepared_or->lowering_artifacts->collective_data_map) == 8);
  REQUIRE(prepared_or->lowering_artifacts->executor_generic_data_map.has_value());
  CHECK(covered_bytes(*prepared_or->lowering_artifacts->executor_generic_data_map) == 16);

  store::StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  strategy_config.enable_owner_file_collective = true;
  strategy_config.allow_mixed_execution = true;
  strategy_config.owner_file_collective_min_dedup_saving_bytes = 0;

  store::loading::ExecutionTopologyContext topology;
  topology.collective_load_group = store::loading::CollectiveLoadGroupHint{
      .group_id = "group",
      .world_size = 2,
      .rank = 0,
  };
  topology.source_locality = store::loading::SourceLocalityHint::kSharedSource;
  topology.source_sharing_domain = "shared-fs";

  auto strategy_plan_or = store::runtime::ingestion::strategy::build_source_bound_execution_strategy_plan(
      prepared_or->resolved_plan,
      prepared_or->lowering_artifacts,
      store::runtime::ingestion::strategy::SourceBoundPolicy::kCollectiveFirst,
      strategy_config,
      topology,
      safetensors_disk_source());
  REQUIRE(strategy_plan_or.ok());
  CHECK(
      strategy_plan_or->lane_plan.mode ==
      store::runtime::ingestion::strategy::SourceBoundExecutionMode::kCollectiveFirstMixed);
  CHECK(strategy_plan_or->summary.planned_collective_candidate_bytes == 8);
  CHECK(strategy_plan_or->summary.planned_non_admitted_typed_bytes == 8);
  CHECK(strategy_plan_or->summary.planned_collective_admitted_bytes == 8);
  CHECK(covered_bytes(strategy_plan_or->lane_plan.collective_lane_map) == 8);
  REQUIRE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(strategy_plan_or->summary.planner_reject_reason_buckets.at("typed_work_without_source_overlap") == 8);
  CHECK_FALSE(strategy_plan_or->summary.planner_reject_reason_buckets.contains("generic_backend_coverage_unproven"));
}

TEST_CASE("copy-plan compatibility map admits source-only dim1 shard copies", "[materialization_target_plan]") {
  const std::string source_index_json = R"({
    "down":[0,32,[2,8],[8,1],"torch.float16",0]
  })";
  auto source_table_or = parse_canonical_index(source_index_json);
  REQUIRE(source_table_or.ok());
  auto canonical_source_table_or = parse_canonical_index(source_index_json);
  REQUIRE(canonical_source_table_or.ok());

  absl::flat_hash_map<std::string, TensorLayoutSpec> dst_specs;
  dst_specs.emplace("down", make_layout_spec(16, {2, 4}, {4, 1}));
  absl::flat_hash_map<std::string, uint64_t> dst_base_offsets;
  dst_base_offsets.emplace("down", 0);
  absl::flat_hash_map<std::string, representation_layout::ViewNarrowSpec> view_narrows;

  v2::CopyPlan copy_plan;
  copy_plan.set_version(1);
  auto* entry = copy_plan.add_entries();
  entry->set_ckpt_name("down");
  entry->set_dst_name("down");
  auto* src_range = entry->mutable_ckpt_range();
  src_range->set_dim(1);
  src_range->set_start(0);
  src_range->set_end(4);

  auto transform_or = representation_transform_builder::build_representation_transform_contract(
      copy_plan,
      *source_table_or,
      *canonical_source_table_or,
      dst_specs,
      dst_base_offsets,
      view_narrows,
      canonical_byte_space(),
      "local_seal_then_promote");
  REQUIRE(transform_or.ok());
  CHECK(transform_or->compatibility_stats.compatible_candidates == 1);
  CHECK(transform_or->compatibility_stats.compatible_bytes == 16);
  CHECK(covered_bytes(transform_or->compatibility_lowered_map) == 16);
}

} // namespace tensorcast::daemon::materialization_target_plan
