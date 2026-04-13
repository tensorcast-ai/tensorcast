// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/contracts/representation_contract.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "catch2/catch_test_macros.hpp"

namespace tensorcast::store::materialization::contracts {
namespace {

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

TensorCoordinateSpec full_range() {
  return TensorCoordinateSpec{};
}

TensorCoordinateSpec dim0_range(int64_t start, int64_t end) {
  TensorCoordinateSpec out;
  out.axes.push_back(TensorAxisRange{.dim = 0, .start = start, .end = end});
  return out;
}

TensorCoordinateSpec multi_range(std::initializer_list<TensorAxisRange> axes) {
  TensorCoordinateSpec out;
  out.axes.assign(axes.begin(), axes.end());
  return out;
}

TensorCoordinateSpec scalar_index(std::initializer_list<int64_t> indices) {
  TensorCoordinateSpec out;
  int32_t dim = 0;
  for (int64_t index : indices) {
    out.axes.push_back(TensorAxisRange{.dim = dim, .start = index, .end = index + 1});
    ++dim;
  }
  return out;
}

} // namespace

TEST_CASE("representation contract normalization is order stable", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "b",
          .dst_spec = make_tensor_spec("b", 16, 16, {8}, {1}),
          .op_kind = BindingOpKind::kSliceCopy,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("source_b", 32, 16, {8}, {1}),
              .source_range = dim0_range(0, 8),
              .destination_range = dim0_range(0, 8),
              .role = SourceFragmentRole::kDefault,
          }},
      },
      RepresentationTensorBinding{
          .dst_name = "a",
          .dst_spec = make_tensor_spec("a", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kExactCopy,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("source_a", 0, 16, {8}, {1}),
              .source_range = full_range(),
              .destination_range = full_range(),
              .role = SourceFragmentRole::kDefault,
          }},
      },
  };

  RepresentationTransformContract reordered = contract;
  std::swap(reordered.tensor_bindings[0], reordered.tensor_bindings[1]);

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto reordered_or = normalize_representation_transform_contract(std::move(reordered));
  REQUIRE(reordered_or.ok());

  CHECK(normalized_or->tensor_bindings == reordered_or->tensor_bindings);
  CHECK(
      normalized_or->target_representation.tensor_schema_hash ==
      reordered_or->target_representation.tensor_schema_hash);
  CHECK(
      normalized_or->target_representation.representation_contract_hash ==
      reordered_or->target_representation.representation_contract_hash);
}

TEST_CASE(
    "representation contract hash normalizes logical topology ordering and keeps topology semantic",
    "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "runtime_serving";
  contract.target_representation.realization_kind = RealizationKind::kArtifactPublishable;
  contract.target_representation.logical_topology = TopologyContract{
      .family = "tp",
      .version = "v1",
      .dimensions =
          {
              TopologyDimension{.name = "rank", .size = 1},
              TopologyDimension{.name = "world", .size = 8},
          },
  };
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "weights",
          .dst_spec = make_tensor_spec("weights", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kExactCopy,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("weights", 0, 16, {8}, {1}),
              .source_range = full_range(),
              .destination_range = full_range(),
              .role = SourceFragmentRole::kDefault,
          }},
      },
  };

  RepresentationTransformContract reordered = contract;
  std::swap(
      reordered.target_representation.logical_topology->dimensions[0],
      reordered.target_representation.logical_topology->dimensions[1]);

  RepresentationTransformContract changed = contract;
  changed.target_representation.logical_topology->dimensions[1].size = 4;

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto reordered_or = normalize_representation_transform_contract(std::move(reordered));
  REQUIRE(reordered_or.ok());
  auto changed_or = normalize_representation_transform_contract(std::move(changed));
  REQUIRE(changed_or.ok());

  CHECK(normalized_or->target_representation.logical_topology == reordered_or->target_representation.logical_topology);
  CHECK(
      normalized_or->target_representation.representation_contract_hash ==
      reordered_or->target_representation.representation_contract_hash);
  CHECK(
      normalized_or->target_representation.representation_contract_hash !=
      changed_or->target_representation.representation_contract_hash);
}

TEST_CASE("representation work plan derives copy and concat items", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "replicated",
          .dst_spec = make_tensor_spec("replicated", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kExactCopy,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("source_replicated", 0, 16, {8}, {1}),
              .source_range = full_range(),
              .destination_range = full_range(),
              .role = SourceFragmentRole::kDefault,
          }},
      },
      RepresentationTensorBinding{
          .dst_name = "concat",
          .dst_spec = make_tensor_spec("concat", 16, 16, {8}, {1}),
          .op_kind = BindingOpKind::kConcat,
          .sources =
              {
                  SourceFragment{
                      .source_spec = make_tensor_spec("left", 32, 8, {4}, {1}),
                      .source_range = dim0_range(0, 4),
                      .destination_range = dim0_range(0, 4),
                      .role = SourceFragmentRole::kConcat,
                  },
                  SourceFragment{
                      .source_spec = make_tensor_spec("right", 40, 8, {4}, {1}),
                      .source_range = dim0_range(0, 4),
                      .destination_range = dim0_range(4, 8),
                      .role = SourceFragmentRole::kConcat,
                  },
              },
      },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto plan_or = build_representation_work_plan(*normalized_or);
  REQUIRE(plan_or.ok());

  REQUIRE(plan_or->items.size() == 2);
  CHECK(plan_or->items[0].kind == RepresentationWorkItemKind::kConcatAssemble);
  REQUIRE(plan_or->items[0].sources.size() == 2);
  CHECK(plan_or->items[0].sources[0].prefix_count == 1);
  CHECK(plan_or->items[0].sources[0].dst_block_offset_bytes == 0);
  CHECK(plan_or->items[0].sources[0].dst_block_stride_bytes == 16);
  CHECK(plan_or->items[0].sources[0].dst_block_bytes == 8);
  CHECK(plan_or->items[0].sources[1].prefix_count == 1);
  CHECK(plan_or->items[0].sources[1].dst_block_offset_bytes == 8);
  CHECK(plan_or->items[0].sources[1].dst_block_stride_bytes == 16);
  CHECK(plan_or->items[0].sources[1].dst_block_bytes == 8);
  CHECK(plan_or->items[1].kind == RepresentationWorkItemKind::kTensorCopy);
  CHECK(plan_or->items[1].partition_kind == WorkPartitionKind::kReplicated);
}

TEST_CASE("representation contract rejects overlapping destination spans", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "tensor",
          .dst_spec = make_tensor_spec("tensor", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kSliceCopy,
          .sources =
              {
                  SourceFragment{
                      .source_spec = make_tensor_spec("left", 0, 8, {4}, {1}),
                      .source_range = dim0_range(0, 4),
                      .destination_range = dim0_range(0, 4),
                  },
                  SourceFragment{
                      .source_spec = make_tensor_spec("right", 8, 8, {4}, {1}),
                      .source_range = dim0_range(0, 4),
                      .destination_range = dim0_range(2, 6),
                  },
              },
      },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE_FALSE(normalized_or.ok());
}

TEST_CASE("index-backed representation work rejects source target mismatch", "[representation_contract]") {
  const std::string source_index_json = R"({"a":[0,16,[8],[1],"torch.float16",0]})";
  const std::string target_index_json =
      R"({"a":[0,16,[8],[1],"torch.float16",0],"b":[16,16,[8],[1],"torch.float16",0]})";

  auto result_or = build_index_backed_representation_work(
      source_index_json, target_index_json, std::nullopt, canonical_byte_space(), "ephemeral_into_target");
  REQUIRE_FALSE(result_or.ok());
  CHECK(result_or.status().code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE("representation work plan lowers const fill items", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "filled",
          .dst_spec = make_tensor_spec("filled", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kConstFill,
          .fill_rule = FillRule{.constant_value = {0, 0}},
      },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto plan_or = build_representation_work_plan(*normalized_or);
  REQUIRE(plan_or.ok());
  REQUIRE(plan_or->items.size() == 1);
  CHECK(plan_or->items.front().kind == RepresentationWorkItemKind::kConstFill);
  REQUIRE(plan_or->items.front().fill_rule.has_value());
  CHECK(plan_or->items.front().fill_rule->constant_value == std::vector<std::uint8_t>{0, 0});
  CHECK(plan_or->items.front().committed_bytes == 16);
}

TEST_CASE("representation work plan preserves partial const fill destination ranges", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "filled",
          .dst_spec = make_tensor_spec("filled", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kConstFill,
          .fill_rule =
              FillRule{
                  .constant_value = {0x34, 0x12},
                  .destination_range = dim0_range(2, 6),
              },
      },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto plan_or = build_representation_work_plan(*normalized_or);
  REQUIRE(plan_or.ok());
  REQUIRE(plan_or->items.size() == 1);
  CHECK(plan_or->items.front().kind == RepresentationWorkItemKind::kConstFill);
  REQUIRE(plan_or->items.front().fill_rule.has_value());
  CHECK(plan_or->items.front().fill_rule->destination_range == dim0_range(2, 6));
  CHECK(plan_or->items.front().committed_bytes == 8);
}

TEST_CASE("representation work plan lowers scalar broadcast fill items", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "filled",
          .dst_spec = make_tensor_spec("filled", 0, 16, {8}, {1}),
          .op_kind = BindingOpKind::kScalarFromSource,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("scalar_src", 32, 8, {4}, {1}),
              .source_range = scalar_index({2}),
              .destination_range = full_range(),
          }},
      },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto plan_or = build_representation_work_plan(*normalized_or);
  REQUIRE(plan_or.ok());
  REQUIRE(plan_or->items.size() == 1);
  CHECK(plan_or->items.front().kind == RepresentationWorkItemKind::kScalarBroadcastFill);
  REQUIRE(plan_or->items.front().sources.size() == 1);
  CHECK(plan_or->items.front().sources.front().fragment.source_range.axes.size() == 1);
}

TEST_CASE("representation work plan accepts multi-axis slice copy coordinates", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "dst",
          .dst_spec = make_tensor_spec("dst", 0, 46080, {36, 20, 32}, {640, 32, 1}),
          .op_kind = BindingOpKind::kSliceCopy,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("src", 0, 184320, {288, 10, 32}, {320, 32, 1}),
              .source_range = dim0_range(180, 181),
              .destination_range = multi_range({
                  TensorAxisRange{.dim = 0, .start = 0, .end = 1},
                  TensorAxisRange{.dim = 1, .start = 0, .end = 10},
              }),
              .role = SourceFragmentRole::kDefault,
          }},
      },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto plan_or = build_representation_work_plan(*normalized_or);
  REQUIRE(plan_or.ok());
  REQUIRE(plan_or->items.size() == 1);
  CHECK(plan_or->items.front().kind == RepresentationWorkItemKind::kTensorCopy);
  CHECK(plan_or->items.front().committed_bytes == 640);
  CHECK(plan_or->items.front().partition_kind == WorkPartitionKind::kUnknown);
}

TEST_CASE("representation work plan preserves normalized residual fallback accounting", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.residual_fallback_map = loader::ByteRangeMap{
      .total_bytes = 8,
      .num_sources = 1,
      .segments =
          {
              loader::ByteRangeSegment{
                  .kind = loader::ByteRangeSegment::Kind::kData,
                  .dst_offset = 4,
                  .length = 4,
                  .src_offset = 4,
                  .source_index = 0,
              },
              loader::ByteRangeSegment{
                  .kind = loader::ByteRangeSegment::Kind::kData,
                  .dst_offset = 0,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 0,
              },
          },
  };

  RepresentationTransformContract reordered = contract;
  std::swap(reordered.residual_fallback_map.segments[0], reordered.residual_fallback_map.segments[1]);

  RepresentationTransformContract changed = contract;
  changed.residual_fallback_map.segments[1].src_offset = 8;

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto reordered_or = normalize_representation_transform_contract(std::move(reordered));
  REQUIRE(reordered_or.ok());
  auto changed_or = normalize_representation_transform_contract(std::move(changed));
  REQUIRE(changed_or.ok());

  CHECK(
      normalized_or->target_representation.representation_contract_hash ==
      reordered_or->target_representation.representation_contract_hash);
  CHECK(
      normalized_or->target_representation.representation_contract_hash !=
      changed_or->target_representation.representation_contract_hash);

  auto plan_or = build_representation_work_plan(*normalized_or);
  REQUIRE(plan_or.ok());
  REQUIRE(plan_or->items.size() == 1);
  CHECK(plan_or->items.back().kind == RepresentationWorkItemKind::kResidualByteRange);
  CHECK(plan_or->items.back().committed_bytes == 8);
  CHECK(plan_or->committed_bytes == 0);
}

TEST_CASE("representation work plan rejects residual overlap with typed work", "[representation_contract]") {
  RepresentationTransformContract contract;
  contract.source_byte_space = canonical_byte_space();
  contract.target_representation.family = "ephemeral_into_target";
  contract.tensor_bindings = {
      RepresentationTensorBinding{
          .dst_name = "copied",
          .dst_spec = make_tensor_spec("copied", 0, 8, {4}, {1}),
          .op_kind = BindingOpKind::kExactCopy,
          .sources = {SourceFragment{
              .source_spec = make_tensor_spec("copied", 0, 8, {4}, {1}),
              .source_range = full_range(),
              .destination_range = full_range(),
              .role = SourceFragmentRole::kDefault,
          }},
      },
  };
  contract.residual_fallback_map = loader::ByteRangeMap{
      .total_bytes = 8,
      .num_sources = 1,
      .segments =
          {
              loader::ByteRangeSegment{
                  .kind = loader::ByteRangeSegment::Kind::kData,
                  .dst_offset = 0,
                  .length = 8,
                  .src_offset = 0,
                  .source_index = 0,
              },
          },
  };

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  REQUIRE(normalized_or.ok());
  auto plan_or = build_representation_work_plan(*normalized_or);
  REQUIRE_FALSE(plan_or.ok());
}

} // namespace tensorcast::store::materialization::contracts
