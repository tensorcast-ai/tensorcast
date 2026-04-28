// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/contracts/view/view_id.h"
#include "tensorcast/common/v1/common.pb.h"

namespace tensorcast::store::materialization::contracts {

struct TopologyDimension {
  std::string name;
  int64_t size{0};

  bool operator==(const TopologyDimension&) const = default;
};

struct TopologyContract {
  std::string family;
  std::string version;
  std::vector<TopologyDimension> dimensions;

  bool operator==(const TopologyContract&) const = default;
};

enum class RealizationKind : std::uint8_t {
  kEphemeralIntoTarget = 0,
  kArtifactPublishable = 1,
  kLocalSealThenPromote = 2,
};

struct RepresentationDescriptor {
  std::string family;
  std::string tensor_schema_hash;
  std::string representation_contract_hash;
  std::optional<TopologyContract> logical_topology;
  RealizationKind realization_kind{RealizationKind::kEphemeralIntoTarget};

  bool operator==(const RepresentationDescriptor&) const = default;
};

struct RepresentationTensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  uint64_t logical_offset{0};
  uint64_t logical_length{0};
  uint64_t storage_offset{0};
  uint64_t element_size{0};

  bool operator==(const RepresentationTensorSpec&) const = default;
};

struct TensorAxisRange {
  int32_t dim{0};
  int64_t start{0};
  int64_t end{0};

  bool operator==(const TensorAxisRange&) const = default;
};

struct TensorCoordinateSpec {
  std::vector<TensorAxisRange> axes;
  bool selects_scalar{false};

  bool operator==(const TensorCoordinateSpec&) const = default;
};

struct TensorByteSpan {
  uint64_t offset{0};
  uint64_t length{0};
};

enum class SourceFragmentRole : std::uint8_t {
  kDefault = 0,
  kConcat = 1,
  kBroadcastEqual = 2,
};

struct FillRule {
  std::vector<std::uint8_t> constant_value;
  TensorCoordinateSpec destination_range;

  bool operator==(const FillRule&) const = default;
};

struct SourceFragment {
  RepresentationTensorSpec source_spec;
  TensorCoordinateSpec source_range;
  TensorCoordinateSpec destination_range;
  SourceFragmentRole role{SourceFragmentRole::kDefault};

  bool operator==(const SourceFragment&) const = default;
};

enum class BindingOpKind : std::uint8_t {
  kExactCopy = 0,
  kSliceCopy = 1,
  kConcat = 2,
  kScalarFromSource = 3,
  kConstFill = 4,
};

enum class CoverageKind : std::uint8_t {
  kExact = 0,
  kPartialWithResidualFallback = 1,
};

struct RepresentationTensorBinding {
  std::string dst_name;
  RepresentationTensorSpec dst_spec;
  BindingOpKind op_kind{BindingOpKind::kExactCopy};
  std::vector<SourceFragment> sources;
  std::optional<FillRule> fill_rule;
  CoverageKind coverage_kind{CoverageKind::kExact};

  bool operator==(const RepresentationTensorBinding&) const = default;
};

struct RepresentationTransformContract {
  tensorcast::common::v1::ByteSpaceRef source_byte_space;
  RepresentationDescriptor target_representation;
  std::vector<RepresentationTensorBinding> tensor_bindings;
  loader::ByteRangeMap residual_fallback_map;

  bool operator==(const RepresentationTransformContract& other) const;
};

enum class RepresentationWorkItemKind : std::uint8_t {
  kTensorCopy = 0,
  kConcatAssemble = 1,
  kResidualByteRange = 2,
  kScalarBroadcastFill = 3,
  kConstFill = 4,
  kPadFill = 5,
  kExpertDim0Concat = 6,
};

enum class WorkPartitionKind : std::uint8_t {
  kUnknown = 0,
  kReplicated = 1,
  kDim0Partitioned = 2,
  kDim1Partitioned = 3,
};

struct RepresentationWorkSourceFragment {
  SourceFragment fragment;
  uint64_t prefix_count{1};
  uint64_t dst_block_offset_bytes{0};
  uint64_t dst_block_stride_bytes{0};
  uint64_t dst_block_bytes{0};

  bool operator==(const RepresentationWorkSourceFragment&) const = default;
};

struct RepresentationWorkItem {
  RepresentationWorkItemKind kind{RepresentationWorkItemKind::kTensorCopy};
  WorkPartitionKind partition_kind{WorkPartitionKind::kUnknown};
  std::string dst_name;
  RepresentationTensorSpec dst_spec;
  std::vector<RepresentationWorkSourceFragment> sources;
  std::optional<FillRule> fill_rule;
  loader::ByteRangeMap byte_range_map;
  uint64_t committed_bytes{0};

  bool operator==(const RepresentationWorkItem& other) const;
};

struct RepresentationWorkPlan {
  std::vector<RepresentationWorkItem> items;
  loader::ByteRangeMap residual_fallback_map;
  uint64_t committed_bytes{0};

  bool operator==(const RepresentationWorkPlan& other) const;
};

struct IndexBackedRepresentationBuildResult {
  RepresentationTransformContract transform_contract;
  RepresentationWorkPlan work_plan;
};

absl::Status validate_representation_transform_contract(const RepresentationTransformContract& contract);

absl::StatusOr<RepresentationTransformContract> normalize_representation_transform_contract(
    RepresentationTransformContract contract);

absl::StatusOr<std::string> compute_tensor_schema_hash(const RepresentationTransformContract& contract);

absl::StatusOr<std::string> compute_representation_contract_hash(const RepresentationTransformContract& contract);

absl::StatusOr<std::vector<TensorByteSpan>> build_coordinate_byte_spans(
    const RepresentationTensorSpec& spec,
    const TensorCoordinateSpec& range);

absl::StatusOr<RepresentationWorkPlan> build_representation_work_plan(const RepresentationTransformContract& contract);

absl::StatusOr<IndexBackedRepresentationBuildResult> build_index_backed_representation_work(
    std::string_view source_index_json,
    std::string_view target_index_json,
    const std::optional<materialization::view::VariantIdentity>& variant_identity,
    const tensorcast::common::v1::ByteSpaceRef& source_byte_space,
    std::string_view target_representation_family);

} // namespace tensorcast::store::materialization::contracts
