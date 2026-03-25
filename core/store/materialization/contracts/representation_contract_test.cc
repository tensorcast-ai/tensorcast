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

} // namespace tensorcast::store::materialization::contracts
