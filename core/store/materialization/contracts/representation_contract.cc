// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/contracts/representation_contract.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::materialization::contracts {

namespace {

using nlohmann::json;

constexpr std::string_view kTensorSchemaHashVersion = "tensorcast.representation.tensor_schema.v1";
constexpr std::string_view kRepresentationContractHashVersion = "tensorcast.representation.contract.v1";

std::string_view realization_kind_name(RealizationKind kind) {
  switch (kind) {
    case RealizationKind::kEphemeralIntoTarget:
      return "ephemeral_into_target";
    case RealizationKind::kArtifactPublishable:
      return "artifact_publishable";
    case RealizationKind::kLocalSealThenPromote:
      return "local_seal_then_promote";
  }
  return "ephemeral_into_target";
}

std::string_view binding_op_kind_name(BindingOpKind kind) {
  switch (kind) {
    case BindingOpKind::kExactCopy:
      return "exact_copy";
    case BindingOpKind::kSliceCopy:
      return "slice_copy";
    case BindingOpKind::kConcat:
      return "concat";
    case BindingOpKind::kScalarFromSource:
      return "scalar_from_source";
    case BindingOpKind::kConstFill:
      return "const_fill";
  }
  return "exact_copy";
}

std::string_view coverage_kind_name(CoverageKind kind) {
  switch (kind) {
    case CoverageKind::kExact:
      return "exact";
    case CoverageKind::kPartialWithResidualFallback:
      return "partial_with_residual_fallback";
  }
  return "exact";
}

std::string_view source_fragment_role_name(SourceFragmentRole role) {
  switch (role) {
    case SourceFragmentRole::kDefault:
      return "default";
    case SourceFragmentRole::kConcat:
      return "concat";
    case SourceFragmentRole::kBroadcastEqual:
      return "broadcast_equal";
  }
  return "default";
}

std::string_view work_item_kind_name(RepresentationWorkItemKind kind) {
  switch (kind) {
    case RepresentationWorkItemKind::kTensorCopy:
      return "tensor_copy";
    case RepresentationWorkItemKind::kConcatAssemble:
      return "concat_assemble";
    case RepresentationWorkItemKind::kResidualByteRange:
      return "residual_byte_range";
    case RepresentationWorkItemKind::kScalarBroadcastFill:
      return "scalar_broadcast_fill";
    case RepresentationWorkItemKind::kConstFill:
      return "const_fill";
    case RepresentationWorkItemKind::kPadFill:
      return "pad_fill";
  }
  return "tensor_copy";
}

std::string_view work_partition_kind_name(WorkPartitionKind kind) {
  switch (kind) {
    case WorkPartitionKind::kUnknown:
      return "unknown";
    case WorkPartitionKind::kReplicated:
      return "replicated";
    case WorkPartitionKind::kDim0Partitioned:
      return "dim0_partitioned";
    case WorkPartitionKind::kDim1Partitioned:
      return "dim1_partitioned";
  }
  return "unknown";
}

absl::StatusOr<uint64_t> infer_dtype_element_size(std::string_view dtype) {
  static const absl::flat_hash_map<std::string_view, uint64_t> kMap = {
      {"torch.float16", 2},
      {"torch.bfloat16", 2},
      {"torch.float8_e4m3fn", 1},
      {"torch.float8_e5m2", 1},
      {"torch.float32", 4},
      {"torch.float64", 8},
      {"torch.int8", 1},
      {"torch.uint8", 1},
      {"torch.int16", 2},
      {"torch.int32", 4},
      {"torch.int64", 8},
      {"torch.bool", 1},
      {"torch.float", 4},
      {"torch.double", 8},
  };
  if (auto it = kMap.find(dtype); it != kMap.end()) {
    return it->second;
  }
  return absl::InvalidArgumentError(absl::StrCat("unsupported dtype in index json: ", dtype));
}

bool is_row_major_contiguous(const RepresentationTensorSpec& spec) {
  if (spec.shape.empty()) {
    return spec.stride.empty();
  }
  if (spec.shape.size() != spec.stride.size()) {
    return false;
  }
  int64_t expected = 1;
  for (int64_t index = static_cast<int64_t>(spec.shape.size()) - 1; index >= 0; --index) {
    if (spec.stride[static_cast<size_t>(index)] != expected) {
      return false;
    }
    expected *= spec.shape[static_cast<size_t>(index)];
  }
  return true;
}

json to_json(const TensorAxisRange& axis) {
  return json{
      {"dim", axis.dim},
      {"start", axis.start},
      {"end", axis.end},
  };
}

json to_json(const TensorCoordinateSpec& spec) {
  json out = json::object();
  out["selects_scalar"] = spec.selects_scalar;
  json axes = json::array();
  for (const auto& axis : spec.axes) {
    axes.push_back(to_json(axis));
  }
  out["axes"] = std::move(axes);
  return out;
}

json to_json(const RepresentationTensorSpec& spec, bool include_offsets) {
  json out = json::object();
  out["name"] = spec.name;
  out["dtype"] = spec.dtype;
  out["shape"] = spec.shape;
  out["stride"] = spec.stride;
  out["element_size"] = spec.element_size;
  if (include_offsets) {
    out["logical_offset"] = spec.logical_offset;
    out["logical_length"] = spec.logical_length;
    out["storage_offset"] = spec.storage_offset;
  }
  return out;
}

json to_json(const TopologyContract& topology) {
  json out = json::object();
  out["family"] = topology.family;
  out["version"] = topology.version;
  json dimensions = json::array();
  for (const auto& dimension : topology.dimensions) {
    dimensions.push_back(json{{"name", dimension.name}, {"size", dimension.size}});
  }
  out["dimensions"] = std::move(dimensions);
  return out;
}

json to_json(const SourceFragment& fragment) {
  return json{
      {"source_spec", to_json(fragment.source_spec, true)},
      {"source_range", to_json(fragment.source_range)},
      {"destination_range", to_json(fragment.destination_range)},
      {"role", source_fragment_role_name(fragment.role)},
  };
}

json to_json(const RepresentationTensorBinding& binding) {
  json out = json::object();
  out["dst_name"] = binding.dst_name;
  out["dst_spec"] = to_json(binding.dst_spec, true);
  out["op_kind"] = binding_op_kind_name(binding.op_kind);
  out["coverage_kind"] = coverage_kind_name(binding.coverage_kind);
  json sources = json::array();
  for (const auto& fragment : binding.sources) {
    sources.push_back(to_json(fragment));
  }
  out["sources"] = std::move(sources);
  if (binding.fill_rule.has_value()) {
    out["fill_rule"] = json{
        {"constant_value", binding.fill_rule->constant_value},
        {"destination_range", to_json(binding.fill_rule->destination_range)},
    };
  } else {
    out["fill_rule"] = nullptr;
  }
  return out;
}

json to_json(const loader::ByteRangeMap& map) {
  json out = json::object();
  out["total_bytes"] = map.total_bytes;
  out["num_sources"] = map.num_sources;
  json segments = json::array();
  for (const auto& segment : map.segments) {
    segments.push_back(
        json{
            {"kind", segment.kind == loader::ByteRangeSegment::Kind::kData ? "data" : "pad"},
            {"dst_offset", segment.dst_offset},
            {"length", segment.length},
            {"src_offset", segment.src_offset},
            {"source_index", segment.source_index},
        });
  }
  out["segments"] = std::move(segments);
  return out;
}

bool byte_range_segment_equal(const loader::ByteRangeSegment& lhs, const loader::ByteRangeSegment& rhs) {
  return lhs.kind == rhs.kind && lhs.dst_offset == rhs.dst_offset && lhs.length == rhs.length &&
      lhs.src_offset == rhs.src_offset && lhs.source_index == rhs.source_index;
}

bool byte_range_map_equal(const loader::ByteRangeMap& lhs, const loader::ByteRangeMap& rhs) {
  if (lhs.total_bytes != rhs.total_bytes || lhs.num_sources != rhs.num_sources ||
      lhs.segments.size() != rhs.segments.size()) {
    return false;
  }
  for (size_t index = 0; index < lhs.segments.size(); ++index) {
    if (!byte_range_segment_equal(lhs.segments[index], rhs.segments[index])) {
      return false;
    }
  }
  return true;
}

uint64_t byte_range_map_covered_bytes(const loader::ByteRangeMap& map) {
  uint64_t total = 0;
  for (const auto& segment : map.segments) {
    total += segment.length;
  }
  return total;
}

std::string hash_serialized_payload(std::string_view version, const std::string& serialized) {
  std::string payload;
  payload.reserve(version.size() + 1 + serialized.size());
  payload.append(version.data(), version.size());
  payload.push_back('\n');
  payload.append(serialized);
  const std::vector<std::uint8_t> digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(payload.data()), payload.size()));
  return tensorcast::common::multibase_multihash_sha256(digest);
}

std::optional<TensorAxisRange> single_axis_range(const TensorCoordinateSpec& spec) {
  if (spec.selects_scalar || spec.axes.size() != 1) {
    return std::nullopt;
  }
  return spec.axes.front();
}

struct ByteSpan {
  uint64_t offset{0};
  uint64_t length{0};
};

absl::Status validate_axis_range(
    const TensorAxisRange& axis,
    const RepresentationTensorSpec& spec,
    std::string_view role) {
  if (spec.shape.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(role, " range requires tensor rank > 0"));
  }
  if (axis.dim < 0 || axis.dim >= static_cast<int32_t>(spec.shape.size())) {
    return absl::InvalidArgumentError(absl::StrCat(role, " range dim out of bounds"));
  }
  if (axis.start < 0 || axis.end <= axis.start || axis.end > spec.shape[static_cast<size_t>(axis.dim)]) {
    return absl::InvalidArgumentError(absl::StrCat(role, " range bounds invalid"));
  }
  return absl::OkStatus();
}

absl::Status validate_coordinate_spec(
    const TensorCoordinateSpec& spec,
    const RepresentationTensorSpec& tensor,
    std::string_view role) {
  if (spec.selects_scalar && !spec.axes.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(role, " scalar selection must not carry axes"));
  }
  std::array<bool, 16> seen_small{};
  std::vector<int32_t> seen_large;
  for (const auto& axis : spec.axes) {
    if (axis.dim >= 0 && axis.dim < static_cast<int32_t>(seen_small.size())) {
      if (seen_small[static_cast<size_t>(axis.dim)]) {
        return absl::InvalidArgumentError(absl::StrCat(role, " contains duplicate axis"));
      }
      seen_small[static_cast<size_t>(axis.dim)] = true;
    } else {
      if (std::find(seen_large.begin(), seen_large.end(), axis.dim) != seen_large.end()) {
        return absl::InvalidArgumentError(absl::StrCat(role, " contains duplicate axis"));
      }
      seen_large.push_back(axis.dim);
    }
    auto status = validate_axis_range(axis, tensor, role);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

bool coordinate_selects_single_element(const TensorCoordinateSpec& spec, const RepresentationTensorSpec& tensor) {
  if (spec.selects_scalar) {
    return tensor.shape.empty();
  }
  if (tensor.shape.empty()) {
    return spec.axes.empty();
  }
  if (spec.axes.size() != tensor.shape.size()) {
    return false;
  }
  std::vector<bool> seen(tensor.shape.size(), false);
  for (const auto& axis : spec.axes) {
    if (axis.dim < 0 || axis.dim >= static_cast<int32_t>(tensor.shape.size())) {
      return false;
    }
    if (seen[static_cast<size_t>(axis.dim)]) {
      return false;
    }
    seen[static_cast<size_t>(axis.dim)] = true;
    if (axis.end - axis.start != 1) {
      return false;
    }
  }
  return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

absl::StatusOr<uint64_t> product_dims(const std::vector<int64_t>& dims, size_t begin) {
  uint64_t total = 1;
  for (size_t index = begin; index < dims.size(); ++index) {
    if (dims[index] < 0) {
      return absl::InvalidArgumentError("negative tensor dimension is invalid");
    }
    const uint64_t extent = static_cast<uint64_t>(dims[index]);
    if (extent != 0 && total > std::numeric_limits<uint64_t>::max() / extent) {
      return absl::OutOfRangeError("tensor element count overflow");
    }
    total *= extent;
  }
  return total;
}

absl::StatusOr<std::vector<ByteSpan>> build_coordinate_byte_spans(
    const RepresentationTensorSpec& spec,
    const TensorCoordinateSpec& range) {
  if (spec.element_size == 0 && spec.logical_length == 0) {
    return absl::InvalidArgumentError("tensor requires non-zero element_size or logical_length");
  }
  const uint64_t element_bytes = spec.element_size == 0 ? spec.logical_length : spec.element_size;
  const uint64_t base_offset = spec.logical_offset;
  if (range.selects_scalar || coordinate_selects_single_element(range, spec)) {
    uint64_t element_offset = 0;
    for (const auto& axis : range.axes) {
      element_offset +=
          static_cast<uint64_t>(axis.start) * static_cast<uint64_t>(spec.stride[static_cast<size_t>(axis.dim)]);
    }
    return std::vector<ByteSpan>{{.offset = base_offset + element_offset * element_bytes, .length = element_bytes}};
  }
  if (spec.shape.empty() || range.axes.empty()) {
    return std::vector<ByteSpan>{{.offset = base_offset, .length = spec.logical_length}};
  }
  if (!is_row_major_contiguous(spec)) {
    return absl::InvalidArgumentError("multi-axis coordinates require row-major contiguous tensors");
  }
  std::vector<int64_t> starts(spec.shape.size(), 0);
  std::vector<int64_t> ends = spec.shape;
  for (const auto& axis : range.axes) {
    starts[static_cast<size_t>(axis.dim)] = axis.start;
    ends[static_cast<size_t>(axis.dim)] = axis.end;
  }
  std::vector<ByteSpan> spans;
  auto emit = [&](auto&& self, size_t dim, uint64_t element_offset) -> absl::Status {
    if (dim == spec.shape.size()) {
      spans.push_back(ByteSpan{.offset = base_offset + element_offset * element_bytes, .length = element_bytes});
      return absl::OkStatus();
    }
    bool later_full = true;
    for (size_t next = dim + 1; next < spec.shape.size(); ++next) {
      if (starts[next] != 0 || ends[next] != spec.shape[next]) {
        later_full = false;
        break;
      }
    }
    if (later_full) {
      auto tail_or = product_dims(spec.shape, dim + 1);
      if (!tail_or.ok()) {
        return tail_or.status();
      }
      const uint64_t elem_start =
          element_offset + static_cast<uint64_t>(starts[dim]) * static_cast<uint64_t>(spec.stride[dim]);
      const uint64_t span_elems = static_cast<uint64_t>(ends[dim] - starts[dim]) * (*tail_or);
      spans.push_back(
          ByteSpan{.offset = base_offset + elem_start * element_bytes, .length = span_elems * element_bytes});
      return absl::OkStatus();
    }
    for (int64_t index = starts[dim]; index < ends[dim]; ++index) {
      auto status =
          self(self, dim + 1, element_offset + static_cast<uint64_t>(index) * static_cast<uint64_t>(spec.stride[dim]));
      if (!status.ok()) {
        return status;
      }
    }
    return absl::OkStatus();
  };
  auto status = emit(emit, 0, 0);
  if (!status.ok()) {
    return status;
  }
  if (spans.empty()) {
    return absl::InvalidArgumentError("coordinate selection produced no byte spans");
  }
  std::vector<ByteSpan> merged;
  merged.reserve(spans.size());
  for (const auto& span : spans) {
    if (merged.empty()) {
      merged.push_back(span);
      continue;
    }
    auto& previous = merged.back();
    if (previous.offset + previous.length == span.offset) {
      previous.length += span.length;
      continue;
    }
    merged.push_back(span);
  }
  return merged;
}

absl::StatusOr<uint64_t> slice_size_bytes(const RepresentationTensorSpec& spec, const TensorCoordinateSpec& range) {
  auto spans_or = build_coordinate_byte_spans(spec, range);
  if (!spans_or.ok()) {
    return spans_or.status();
  }
  uint64_t total = 0;
  for (const auto& span : *spans_or) {
    total += span.length;
  }
  return total;
}

absl::Status add_spans_to_covered_ranges(
    const std::vector<ByteSpan>& spans,
    std::vector<std::pair<uint64_t, uint64_t>>* covered_ranges) {
  if (covered_ranges == nullptr) {
    return absl::InvalidArgumentError("covered range output must not be null");
  }
  for (const auto& span : spans) {
    if (span.length == 0) {
      continue;
    }
    covered_ranges->push_back({span.offset, span.offset + span.length});
  }
  return absl::OkStatus();
}

absl::Status validate_no_overlap_with_ranges(
    const std::vector<std::pair<uint64_t, uint64_t>>& covered_ranges,
    const loader::ByteRangeMap& map,
    std::string_view map_name) {
  if (map.segments.empty()) {
    return absl::OkStatus();
  }
  for (const auto& segment : map.segments) {
    if (segment.length == 0) {
      continue;
    }
    const uint64_t begin = segment.dst_offset;
    const uint64_t end = segment.dst_offset + segment.length;
    for (const auto& range : covered_ranges) {
      if (range.second <= begin || end <= range.first) {
        continue;
      }
      return absl::FailedPreconditionError(absl::StrCat("typed work overlaps ", map_name, " coverage"));
    }
  }
  return absl::OkStatus();
}

absl::Status validate_work_plan_residual_coverage(const RepresentationWorkPlan& plan) {
  std::vector<std::pair<uint64_t, uint64_t>> covered_ranges;
  for (const auto& item : plan.items) {
    switch (item.kind) {
      case RepresentationWorkItemKind::kTensorCopy:
      case RepresentationWorkItemKind::kConcatAssemble:
      case RepresentationWorkItemKind::kScalarBroadcastFill:
        for (const auto& source : item.sources) {
          auto spans_or = build_coordinate_byte_spans(item.dst_spec, source.fragment.destination_range);
          if (!spans_or.ok()) {
            return spans_or.status();
          }
          auto status = add_spans_to_covered_ranges(*spans_or, &covered_ranges);
          if (!status.ok()) {
            return status;
          }
        }
        break;
      case RepresentationWorkItemKind::kConstFill:
        if (!item.fill_rule.has_value()) {
          return absl::InvalidArgumentError("const fill work item requires fill_rule");
        }
        {
          auto spans_or = build_coordinate_byte_spans(item.dst_spec, item.fill_rule->destination_range);
          if (!spans_or.ok()) {
            return spans_or.status();
          }
          auto status = add_spans_to_covered_ranges(*spans_or, &covered_ranges);
          if (!status.ok()) {
            return status;
          }
        }
        break;
      case RepresentationWorkItemKind::kPadFill: {
        auto overlap_status = validate_no_overlap_with_ranges(covered_ranges, item.byte_range_map, "typed work");
        if (!overlap_status.ok()) {
          return overlap_status;
        }
        for (const auto& segment : item.byte_range_map.segments) {
          if (segment.length == 0) {
            continue;
          }
          covered_ranges.push_back({segment.dst_offset, segment.dst_offset + segment.length});
        }
      } break;
      case RepresentationWorkItemKind::kResidualByteRange:
        break;
    }
  }
  return validate_no_overlap_with_ranges(covered_ranges, plan.residual_fallback_map, "residual_fallback_map");
}

bool coordinate_less(const TensorCoordinateSpec& lhs, const TensorCoordinateSpec& rhs) {
  if (lhs.selects_scalar != rhs.selects_scalar) {
    return lhs.selects_scalar < rhs.selects_scalar;
  }
  if (lhs.axes.size() != rhs.axes.size()) {
    return lhs.axes.size() < rhs.axes.size();
  }
  for (size_t index = 0; index < lhs.axes.size(); ++index) {
    if (lhs.axes[index].dim != rhs.axes[index].dim) {
      return lhs.axes[index].dim < rhs.axes[index].dim;
    }
    if (lhs.axes[index].start != rhs.axes[index].start) {
      return lhs.axes[index].start < rhs.axes[index].start;
    }
    if (lhs.axes[index].end != rhs.axes[index].end) {
      return lhs.axes[index].end < rhs.axes[index].end;
    }
  }
  return false;
}

bool source_fragment_less(const SourceFragment& lhs, const SourceFragment& rhs) {
  if (lhs.role != rhs.role) {
    return static_cast<int>(lhs.role) < static_cast<int>(rhs.role);
  }
  if (lhs.source_spec.name != rhs.source_spec.name) {
    return lhs.source_spec.name < rhs.source_spec.name;
  }
  if (coordinate_less(lhs.destination_range, rhs.destination_range)) {
    return true;
  }
  if (coordinate_less(rhs.destination_range, lhs.destination_range)) {
    return false;
  }
  if (coordinate_less(lhs.source_range, rhs.source_range)) {
    return true;
  }
  if (coordinate_less(rhs.source_range, lhs.source_range)) {
    return false;
  }
  return lhs.source_spec.logical_offset < rhs.source_spec.logical_offset;
}

bool binding_less(const RepresentationTensorBinding& lhs, const RepresentationTensorBinding& rhs) {
  if (lhs.dst_name != rhs.dst_name) {
    return lhs.dst_name < rhs.dst_name;
  }
  if (lhs.op_kind != rhs.op_kind) {
    return static_cast<int>(lhs.op_kind) < static_cast<int>(rhs.op_kind);
  }
  if (lhs.coverage_kind != rhs.coverage_kind) {
    return static_cast<int>(lhs.coverage_kind) < static_cast<int>(rhs.coverage_kind);
  }
  if (lhs.sources.size() != rhs.sources.size()) {
    return lhs.sources.size() < rhs.sources.size();
  }
  for (size_t index = 0; index < lhs.sources.size(); ++index) {
    if (source_fragment_less(lhs.sources[index], rhs.sources[index])) {
      return true;
    }
    if (source_fragment_less(rhs.sources[index], lhs.sources[index])) {
      return false;
    }
  }
  return false;
}

absl::Status validate_binding(const RepresentationTensorBinding& binding) {
  if (binding.dst_name.empty()) {
    return absl::InvalidArgumentError("representation binding requires dst_name");
  }
  if (binding.dst_spec.name.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("binding ", binding.dst_name, " requires dst_spec.name"));
  }
  if (binding.dst_spec.name != binding.dst_name) {
    return absl::InvalidArgumentError(absl::StrCat("binding ", binding.dst_name, " dst_spec.name mismatch"));
  }
  for (const auto& fragment : binding.sources) {
    auto src_status = validate_coordinate_spec(fragment.source_range, fragment.source_spec, "source");
    if (!src_status.ok()) {
      return src_status;
    }
    auto dst_status = validate_coordinate_spec(fragment.destination_range, binding.dst_spec, "destination");
    if (!dst_status.ok()) {
      return dst_status;
    }
  }

  switch (binding.op_kind) {
    case BindingOpKind::kConstFill:
      if (!binding.fill_rule.has_value()) {
        return absl::InvalidArgumentError(absl::StrCat("binding ", binding.dst_name, " const fill requires fill_rule"));
      }
      if (!binding.sources.empty()) {
        return absl::InvalidArgumentError(absl::StrCat("binding ", binding.dst_name, " const fill forbids sources"));
      }
      if (auto dst_status =
              validate_coordinate_spec(binding.fill_rule->destination_range, binding.dst_spec, "destination");
          !dst_status.ok()) {
        return dst_status;
      }
      break;
    case BindingOpKind::kExactCopy:
    case BindingOpKind::kSliceCopy:
    case BindingOpKind::kConcat:
      if (binding.sources.empty()) {
        return absl::InvalidArgumentError(absl::StrCat("binding ", binding.dst_name, " requires source fragments"));
      }
      break;
    case BindingOpKind::kScalarFromSource:
      if (binding.sources.size() != 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("binding ", binding.dst_name, " scalar-from-source requires exactly one source fragment"));
      }
      if (!coordinate_selects_single_element(
              binding.sources.front().source_range, binding.sources.front().source_spec)) {
        return absl::InvalidArgumentError(
            absl::StrCat("binding ", binding.dst_name, " scalar-from-source requires a single source element"));
      }
      break;
  }

  std::vector<std::pair<uint64_t, uint64_t>> dst_spans;
  for (const auto& fragment : binding.sources) {
    auto spans_or = build_coordinate_byte_spans(binding.dst_spec, fragment.destination_range);
    if (!spans_or.ok()) {
      return spans_or.status();
    }
    for (const auto& span : *spans_or) {
      dst_spans.push_back({span.offset, span.offset + span.length});
    }
  }
  if (binding.fill_rule.has_value()) {
    auto spans_or = build_coordinate_byte_spans(binding.dst_spec, binding.fill_rule->destination_range);
    if (!spans_or.ok()) {
      return spans_or.status();
    }
    for (const auto& span : *spans_or) {
      dst_spans.push_back({span.offset, span.offset + span.length});
    }
  }
  std::sort(dst_spans.begin(), dst_spans.end());
  for (size_t index = 1; index < dst_spans.size(); ++index) {
    if (dst_spans[index - 1].second > dst_spans[index].first) {
      return absl::InvalidArgumentError(
          absl::StrCat("binding ", binding.dst_name, " has overlapping destination spans"));
    }
  }
  return absl::OkStatus();
}

json build_tensor_schema_json(const RepresentationTransformContract& contract) {
  absl::flat_hash_map<std::string, RepresentationTensorSpec> unique_specs;
  unique_specs.reserve(contract.tensor_bindings.size());
  for (const auto& binding : contract.tensor_bindings) {
    unique_specs.emplace(binding.dst_name, binding.dst_spec);
  }
  std::vector<std::string> names;
  names.reserve(unique_specs.size());
  for (const auto& [name, _] : unique_specs) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  json tensors = json::array();
  for (const auto& name : names) {
    tensors.push_back(to_json(unique_specs.at(name), false));
  }
  return json{{"tensors", std::move(tensors)}};
}

struct ConcatSourceGroup {
  const SourceFragment* first_fragment{nullptr};
  std::vector<const SourceFragment*> fragments;
  uint64_t prefix_count{0};
  int64_t first_src_start{0};
  int64_t first_src_end{0};
  int64_t block_offset_rows{0};
  int64_t block_rows{0};
  int64_t dst_stride_rows{0};
};

absl::StatusOr<std::optional<std::vector<RepresentationWorkSourceFragment>>> derive_concat_work_source_fragments(
    const RepresentationTensorBinding& binding) {
  if (!is_row_major_contiguous(binding.dst_spec)) {
    return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
  }
  auto dst_inner_or = product_dims(binding.dst_spec.shape, 1);
  if (!dst_inner_or.ok()) {
    return dst_inner_or.status();
  }
  const uint64_t dst_inner = binding.dst_spec.shape.size() <= 1 ? 1 : *dst_inner_or;

  absl::flat_hash_map<std::string, ConcatSourceGroup> groups;
  groups.reserve(binding.sources.size());
  for (const auto& fragment : binding.sources) {
    const auto src_axis = single_axis_range(fragment.source_range);
    const auto dst_axis = single_axis_range(fragment.destination_range);
    if (!src_axis.has_value() || !dst_axis.has_value() || src_axis->dim != 0 || dst_axis->dim != 0 ||
        !is_row_major_contiguous(fragment.source_spec)) {
      return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
    }
    auto [it, inserted] = groups.try_emplace(fragment.source_spec.name);
    if (inserted) {
      it->second.first_fragment = &fragment;
    }
    it->second.fragments.push_back(&fragment);
  }
  if (groups.size() < 2) {
    return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
  }

  std::optional<uint64_t> expected_prefix_count;
  std::optional<int64_t> expected_dst_stride_rows;
  std::vector<ConcatSourceGroup*> ordered_groups;
  ordered_groups.reserve(groups.size());
  for (auto& [_, group] : groups) {
    std::sort(group.fragments.begin(), group.fragments.end(), [](const SourceFragment* lhs, const SourceFragment* rhs) {
      return lhs->destination_range.axes.front().start < rhs->destination_range.axes.front().start;
    });
    const auto* first = group.fragments.front();
    const auto& first_src_axis = first->source_range.axes.front();
    const auto& first_dst_axis = first->destination_range.axes.front();
    group.prefix_count = group.fragments.size();
    group.first_src_start = first_src_axis.start;
    group.first_src_end = first_src_axis.end;
    group.block_rows = first_dst_axis.end - first_dst_axis.start;
    if (group.block_rows <= 0) {
      return absl::InvalidArgumentError("concat binding block rows must be positive");
    }
    for (size_t index = 1; index < group.fragments.size(); ++index) {
      const auto* current = group.fragments[index];
      const auto& current_src_axis = current->source_range.axes.front();
      const auto& current_dst_axis = current->destination_range.axes.front();
      if ((current_src_axis.end - current_src_axis.start) != (first_src_axis.end - first_src_axis.start) ||
          (current_dst_axis.end - current_dst_axis.start) != group.block_rows) {
        return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
      }
    }
    if (group.prefix_count > 1) {
      group.dst_stride_rows = group.fragments[1]->destination_range.axes.front().start -
          group.fragments[0]->destination_range.axes.front().start;
      if (group.dst_stride_rows < group.block_rows) {
        return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
      }
      for (size_t index = 0; index < group.fragments.size(); ++index) {
        const auto* current = group.fragments[index];
        const auto& current_src_axis = current->source_range.axes.front();
        const auto& current_dst_axis = current->destination_range.axes.front();
        const int64_t src_block_rows = group.first_src_end - group.first_src_start;
        if (current_src_axis.start != group.first_src_start + static_cast<int64_t>(index) * src_block_rows) {
          return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
        }
        if (current_dst_axis.start !=
            group.fragments[0]->destination_range.axes.front().start +
                static_cast<int64_t>(index) * group.dst_stride_rows) {
          return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
        }
      }
      group.block_offset_rows = group.fragments[0]->destination_range.axes.front().start;
      if (group.block_offset_rows >= group.dst_stride_rows) {
        return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
      }
      if (!expected_prefix_count.has_value()) {
        expected_prefix_count = group.prefix_count;
      } else if (*expected_prefix_count != group.prefix_count) {
        return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
      }
      if (!expected_dst_stride_rows.has_value()) {
        expected_dst_stride_rows = group.dst_stride_rows;
      } else if (*expected_dst_stride_rows != group.dst_stride_rows) {
        return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
      }
    } else {
      group.block_offset_rows = group.fragments[0]->destination_range.axes.front().start;
    }
    ordered_groups.push_back(&group);
  }

  std::sort(
      ordered_groups.begin(), ordered_groups.end(), [](const ConcatSourceGroup* lhs, const ConcatSourceGroup* rhs) {
        return lhs->block_offset_rows < rhs->block_offset_rows;
      });
  const uint64_t prefix_count = expected_prefix_count.value_or(1);
  int64_t dst_stride_rows = expected_dst_stride_rows.value_or(0);
  if (prefix_count == 1) {
    int64_t cursor = 0;
    int64_t max_end = 0;
    for (const ConcatSourceGroup* group : ordered_groups) {
      if (group->block_offset_rows != cursor) {
        return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
      }
      cursor += group->block_rows;
      max_end = std::max(max_end, group->block_offset_rows + group->block_rows);
    }
    dst_stride_rows = max_end;
  } else {
    int64_t cursor = 0;
    for (const ConcatSourceGroup* group : ordered_groups) {
      if (group->block_offset_rows != cursor) {
        return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
      }
      cursor += group->block_rows;
    }
    if (cursor != dst_stride_rows) {
      return std::optional<std::vector<RepresentationWorkSourceFragment>>{};
    }
  }

  std::vector<RepresentationWorkSourceFragment> out;
  out.reserve(ordered_groups.size());
  for (const ConcatSourceGroup* group : ordered_groups) {
    RepresentationWorkSourceFragment work_fragment;
    work_fragment.fragment = *group->first_fragment;
    work_fragment.prefix_count = group->prefix_count;
    work_fragment.dst_block_offset_bytes =
        static_cast<uint64_t>(group->block_offset_rows) * dst_inner * binding.dst_spec.element_size;
    work_fragment.dst_block_stride_bytes =
        static_cast<uint64_t>(dst_stride_rows) * dst_inner * binding.dst_spec.element_size;
    work_fragment.dst_block_bytes =
        static_cast<uint64_t>(group->block_rows) * dst_inner * binding.dst_spec.element_size;
    out.push_back(std::move(work_fragment));
  }
  return std::optional<std::vector<RepresentationWorkSourceFragment>>(std::move(out));
}

absl::StatusOr<RepresentationWorkSourceFragment> build_work_source_fragment(
    BindingOpKind op_kind,
    const RepresentationTensorSpec& dst_spec,
    const SourceFragment& fragment,
    uint64_t prefix_count) {
  RepresentationWorkSourceFragment out;
  out.fragment = fragment;
  out.prefix_count = prefix_count;
  if (op_kind != BindingOpKind::kConcat) {
    auto slice_bytes_or = slice_size_bytes(dst_spec, fragment.destination_range);
    if (!slice_bytes_or.ok()) {
      return slice_bytes_or.status();
    }
    out.dst_block_bytes = *slice_bytes_or;
    return out;
  }

  const auto src_axis = single_axis_range(fragment.source_range);
  const auto dst_axis = single_axis_range(fragment.destination_range);
  if (!src_axis.has_value() || !dst_axis.has_value() || src_axis->dim != 0 || dst_axis->dim != 0 ||
      !is_row_major_contiguous(fragment.source_spec) || !is_row_major_contiguous(dst_spec)) {
    return out;
  }

  auto dst_inner_or = product_dims(dst_spec.shape, 1);
  if (!dst_inner_or.ok()) {
    return dst_inner_or.status();
  }
  const uint64_t dst_inner = dst_spec.shape.size() <= 1 ? 1 : *dst_inner_or;
  auto src_block_bytes_or = slice_size_bytes(fragment.source_spec, fragment.source_range);
  if (!src_block_bytes_or.ok()) {
    return src_block_bytes_or.status();
  }
  const int64_t block_rows = dst_axis->end - dst_axis->start;
  if (block_rows <= 0) {
    return absl::InvalidArgumentError("concat binding block rows must be positive");
  }
  out.dst_block_offset_bytes = static_cast<uint64_t>(dst_axis->start) * dst_inner * dst_spec.element_size;
  out.dst_block_bytes = static_cast<uint64_t>(block_rows) * dst_inner * dst_spec.element_size;
  out.dst_block_stride_bytes = prefix_count > 1
      ? static_cast<uint64_t>(dst_spec.shape.empty() ? block_rows : dst_spec.shape.front()) * dst_inner *
          dst_spec.element_size
      : out.dst_block_bytes;
  if (*src_block_bytes_or != out.dst_block_bytes) {
    out.dst_block_bytes = *src_block_bytes_or;
  }
  return out;
}

struct ParsedTensorMeta {
  uint64_t offset{0};
  uint64_t size_bytes{0};
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  uint64_t storage_offset{0};
  uint64_t elem_size{0};
};

absl::StatusOr<absl::flat_hash_map<std::string, ParsedTensorMeta>> parse_index_json(std::string_view index_json) {
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse index json: ", ex.what()));
  }
  if (!parsed.is_object()) {
    return absl::InvalidArgumentError("index json must be an object");
  }
  absl::flat_hash_map<std::string, ParsedTensorMeta> out;
  out.reserve(parsed.size());
  for (auto it = parsed.begin(); it != parsed.end(); ++it) {
    const auto& value = it.value();
    if (!value.is_array() || value.size() < 6) {
      return absl::InvalidArgumentError("index entry must be [offset,size,shape,stride,dtype,storage_offset]");
    }
    ParsedTensorMeta meta;
    meta.offset = value[0].get<uint64_t>();
    meta.size_bytes = value[1].get<uint64_t>();
    for (const auto& dim : value[2]) {
      meta.shape.push_back(dim.get<int64_t>());
    }
    for (const auto& dim : value[3]) {
      meta.stride.push_back(dim.get<int64_t>());
    }
    meta.dtype = value[4].get<std::string>();
    meta.storage_offset = value[5].get<uint64_t>();
    if (auto elem_or = infer_dtype_element_size(meta.dtype); elem_or.ok()) {
      meta.elem_size = *elem_or;
    } else {
      return elem_or.status();
    }
    out.emplace(it.key(), std::move(meta));
  }
  return out;
}

RepresentationTensorSpec to_representation_tensor_spec(std::string_view name, const ParsedTensorMeta& meta) {
  return RepresentationTensorSpec{
      .name = std::string(name),
      .shape = meta.shape,
      .stride = meta.stride,
      .dtype = meta.dtype,
      .logical_offset = meta.offset,
      .logical_length = meta.size_bytes,
      .storage_offset = meta.storage_offset,
      .element_size = meta.elem_size,
  };
}

TensorCoordinateSpec full_coordinate_spec() {
  return TensorCoordinateSpec{};
}

TensorCoordinateSpec axis_coordinate_spec(int32_t dim, int64_t start, int64_t end) {
  TensorCoordinateSpec out;
  out.axes.push_back(TensorAxisRange{.dim = dim, .start = start, .end = end});
  return out;
}

struct DerivedTensorSlice {
  enum class Kind : uint8_t { kFull = 0, kDim0 = 1, kDim1 = 2 };

  Kind kind{Kind::kFull};
  int64_t start{0};
  uint64_t length{0};
};

bool is_row_major_contiguous(const ParsedTensorMeta& meta) {
  if (meta.shape.size() != meta.stride.size()) {
    return false;
  }
  int64_t acc = 1;
  for (int64_t index = static_cast<int64_t>(meta.shape.size()) - 1; index >= 0; --index) {
    if (meta.stride[static_cast<size_t>(index)] != acc) {
      return false;
    }
    acc *= meta.shape[static_cast<size_t>(index)];
  }
  return true;
}

absl::StatusOr<DerivedTensorSlice> derive_tensor_slice(
    const ParsedTensorMeta& source,
    const ParsedTensorMeta& target,
    const std::optional<materialization::view::TensorViewOps>& view_ops) {
  DerivedTensorSlice slice;
  if (!view_ops.has_value() || view_ops->ops.empty()) {
    slice.kind = DerivedTensorSlice::Kind::kFull;
    return slice;
  }
  if (view_ops->ops.size() != 1 || view_ops->ops[0].kind != materialization::view::ViewOp::Kind::kNarrow) {
    return absl::UnimplementedError("only a single narrow op is supported in index-backed representation work");
  }
  const auto& narrow = view_ops->ops[0].narrow;
  if (narrow.dim < 0 || narrow.dim >= static_cast<int32_t>(source.shape.size())) {
    return absl::InvalidArgumentError("narrow dim out of range in index-backed representation work");
  }
  int64_t dim_extent = source.shape[static_cast<size_t>(narrow.dim)];
  int64_t normalized_start = narrow.start;
  if (normalized_start < 0) {
    normalized_start += dim_extent;
  }
  if (normalized_start < 0 || normalized_start >= dim_extent) {
    return absl::InvalidArgumentError("narrow start out of range in index-backed representation work");
  }
  slice.start = normalized_start;
  slice.length = narrow.length;
  if (narrow.dim == 0) {
    slice.kind = DerivedTensorSlice::Kind::kDim0;
    return slice;
  }
  if (narrow.dim == 1 && source.shape.size() == 2 && is_row_major_contiguous(source) &&
      is_row_major_contiguous(target)) {
    slice.kind = DerivedTensorSlice::Kind::kDim1;
    return slice;
  }
  return absl::UnimplementedError("index-backed representation work only supports dim0 or 2D dim1 narrow");
}

} // namespace

absl::Status validate_representation_transform_contract(const RepresentationTransformContract& contract) {
  if (contract.source_byte_space.kind() == tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED) {
    return absl::InvalidArgumentError("representation contract requires source_byte_space.kind");
  }
  if (contract.target_representation.family.empty()) {
    return absl::InvalidArgumentError("representation contract requires target_representation.family");
  }
  for (const auto& binding : contract.tensor_bindings) {
    auto status = validate_binding(binding);
    if (!status.ok()) {
      return status;
    }
  }
  return absl::OkStatus();
}

bool RepresentationTransformContract::operator==(const RepresentationTransformContract& other) const {
  return source_byte_space.kind() == other.source_byte_space.kind() &&
      source_byte_space.id() == other.source_byte_space.id() && target_representation == other.target_representation &&
      tensor_bindings == other.tensor_bindings &&
      byte_range_map_equal(residual_fallback_map, other.residual_fallback_map);
}

absl::StatusOr<std::string> compute_tensor_schema_hash(const RepresentationTransformContract& contract) {
  auto status = validate_representation_transform_contract(contract);
  if (!status.ok()) {
    return status;
  }
  const std::string serialized = build_tensor_schema_json(contract).dump(-1, ' ', true);
  return hash_serialized_payload(kTensorSchemaHashVersion, serialized);
}

absl::StatusOr<std::string> compute_representation_contract_hash(const RepresentationTransformContract& contract) {
  auto status = validate_representation_transform_contract(contract);
  if (!status.ok()) {
    return status;
  }
  json payload = json::object();
  payload["source_byte_space"] = json{
      {"kind", contract.source_byte_space.kind()},
      {"id", contract.source_byte_space.id()},
  };
  payload["target_representation"] = json{
      {"family", contract.target_representation.family},
      {"realization_kind", realization_kind_name(contract.target_representation.realization_kind)},
      {"logical_topology",
       contract.target_representation.logical_topology.has_value()
           ? to_json(*contract.target_representation.logical_topology)
           : json(nullptr)},
  };
  payload["tensor_schema_hash"] = contract.target_representation.tensor_schema_hash;
  json bindings = json::array();
  for (const auto& binding : contract.tensor_bindings) {
    bindings.push_back(to_json(binding));
  }
  payload["tensor_bindings"] = std::move(bindings);
  payload["residual_fallback_map"] = to_json(contract.residual_fallback_map);
  const std::string serialized = payload.dump(-1, ' ', true);
  return hash_serialized_payload(kRepresentationContractHashVersion, serialized);
}

absl::StatusOr<RepresentationTransformContract> normalize_representation_transform_contract(
    RepresentationTransformContract contract) {
  if (contract.target_representation.logical_topology.has_value()) {
    auto& dimensions = contract.target_representation.logical_topology->dimensions;
    std::sort(dimensions.begin(), dimensions.end(), [](const TopologyDimension& lhs, const TopologyDimension& rhs) {
      if (lhs.name != rhs.name) {
        return lhs.name < rhs.name;
      }
      return lhs.size < rhs.size;
    });
  }
  for (auto& binding : contract.tensor_bindings) {
    std::sort(binding.sources.begin(), binding.sources.end(), source_fragment_less);
  }
  std::sort(contract.tensor_bindings.begin(), contract.tensor_bindings.end(), binding_less);
  if (contract.residual_fallback_map.total_bytes > 0 || !contract.residual_fallback_map.segments.empty()) {
    auto normalized_map_or = loader::normalize_byte_range_map(contract.residual_fallback_map);
    if (!normalized_map_or.ok()) {
      return normalized_map_or.status();
    }
    contract.residual_fallback_map = std::move(*normalized_map_or);
  } else {
    contract.residual_fallback_map = loader::ByteRangeMap{};
  }
  auto tensor_schema_hash_or = compute_tensor_schema_hash(contract);
  if (!tensor_schema_hash_or.ok()) {
    return tensor_schema_hash_or.status();
  }
  contract.target_representation.tensor_schema_hash = *tensor_schema_hash_or;
  auto contract_hash_or = compute_representation_contract_hash(contract);
  if (!contract_hash_or.ok()) {
    return contract_hash_or.status();
  }
  contract.target_representation.representation_contract_hash = *contract_hash_or;
  return contract;
}

absl::StatusOr<RepresentationWorkPlan> build_representation_work_plan(const RepresentationTransformContract& contract) {
  auto status = validate_representation_transform_contract(contract);
  if (!status.ok()) {
    return status;
  }
  RepresentationWorkPlan plan;
  plan.residual_fallback_map = contract.residual_fallback_map;
  plan.items.reserve(contract.tensor_bindings.size() + (contract.residual_fallback_map.segments.empty() ? 0 : 1));

  for (const auto& binding : contract.tensor_bindings) {
    RepresentationWorkItem item;
    item.dst_name = binding.dst_name;
    item.dst_spec = binding.dst_spec;
    switch (binding.op_kind) {
      case BindingOpKind::kConcat:
        item.kind = RepresentationWorkItemKind::kConcatAssemble;
        break;
      case BindingOpKind::kConstFill:
        item.kind = RepresentationWorkItemKind::kConstFill;
        item.fill_rule = binding.fill_rule;
        {
          auto committed_bytes_or = slice_size_bytes(binding.dst_spec, binding.fill_rule->destination_range);
          if (!committed_bytes_or.ok()) {
            return committed_bytes_or.status();
          }
          item.committed_bytes = *committed_bytes_or;
        }
        plan.committed_bytes += item.committed_bytes;
        plan.items.push_back(std::move(item));
        continue;
      case BindingOpKind::kExactCopy:
      case BindingOpKind::kSliceCopy:
        item.kind = RepresentationWorkItemKind::kTensorCopy;
        break;
      case BindingOpKind::kScalarFromSource:
        item.kind = RepresentationWorkItemKind::kScalarBroadcastFill;
        break;
    }

    std::optional<std::vector<RepresentationWorkSourceFragment>> concat_work_sources;
    if (binding.op_kind == BindingOpKind::kConcat) {
      auto concat_sources_or = derive_concat_work_source_fragments(binding);
      if (!concat_sources_or.ok()) {
        return concat_sources_or.status();
      }
      concat_work_sources = std::move(*concat_sources_or);
    }

    if (concat_work_sources.has_value()) {
      item.sources = std::move(*concat_work_sources);
    } else {
      item.sources.reserve(binding.sources.size());
      for (const auto& fragment : binding.sources) {
        auto work_fragment_or =
            build_work_source_fragment(binding.op_kind, binding.dst_spec, fragment, /*prefix_count=*/1);
        if (!work_fragment_or.ok()) {
          return work_fragment_or.status();
        }
        item.sources.push_back(std::move(*work_fragment_or));
      }
    }

    for (const auto& source : item.sources) {
      auto committed_bytes_or = slice_size_bytes(binding.dst_spec, source.fragment.destination_range);
      if (!committed_bytes_or.ok()) {
        return committed_bytes_or.status();
      }
      item.committed_bytes += *committed_bytes_or;
    }

    if (item.kind == RepresentationWorkItemKind::kTensorCopy && item.sources.size() == 1) {
      const auto src_axis = single_axis_range(item.sources.front().fragment.source_range);
      const auto dst_axis = single_axis_range(item.sources.front().fragment.destination_range);
      if (!src_axis.has_value() && !dst_axis.has_value()) {
        item.partition_kind = WorkPartitionKind::kReplicated;
      } else if (src_axis.has_value() && dst_axis.has_value() && src_axis->dim == dst_axis->dim) {
        if (src_axis->dim == 0) {
          item.partition_kind = WorkPartitionKind::kDim0Partitioned;
        } else if (
            src_axis->dim == 1 && item.sources.front().fragment.source_spec.shape.size() == 2 &&
            item.dst_spec.shape.size() == 2) {
          item.partition_kind = WorkPartitionKind::kDim1Partitioned;
        }
      }
    } else if (item.kind == RepresentationWorkItemKind::kScalarBroadcastFill) {
      item.partition_kind = WorkPartitionKind::kUnknown;
    }

    plan.committed_bytes += item.committed_bytes;
    plan.items.push_back(std::move(item));
  }

  if (!plan.residual_fallback_map.segments.empty()) {
    RepresentationWorkItem residual_item;
    residual_item.kind = RepresentationWorkItemKind::kResidualByteRange;
    residual_item.partition_kind = WorkPartitionKind::kUnknown;
    residual_item.byte_range_map = plan.residual_fallback_map;
    residual_item.committed_bytes = byte_range_map_covered_bytes(plan.residual_fallback_map);
    plan.items.push_back(std::move(residual_item));
  }

  auto coverage_status = validate_work_plan_residual_coverage(plan);
  if (!coverage_status.ok()) {
    return coverage_status;
  }
  return plan;
}

bool RepresentationWorkItem::operator==(const RepresentationWorkItem& other) const {
  return kind == other.kind && partition_kind == other.partition_kind && dst_name == other.dst_name &&
      dst_spec == other.dst_spec && sources == other.sources && fill_rule == other.fill_rule &&
      byte_range_map_equal(byte_range_map, other.byte_range_map) && committed_bytes == other.committed_bytes;
}

bool RepresentationWorkPlan::operator==(const RepresentationWorkPlan& other) const {
  return items == other.items && committed_bytes == other.committed_bytes &&
      byte_range_map_equal(residual_fallback_map, other.residual_fallback_map);
}

absl::StatusOr<IndexBackedRepresentationBuildResult> build_index_backed_representation_work(
    std::string_view source_index_json,
    std::string_view target_index_json,
    const std::optional<materialization::view::VariantIdentity>& variant_identity,
    const tensorcast::common::v1::ByteSpaceRef& source_byte_space,
    std::string_view target_representation_family) {
  auto source_or = parse_index_json(source_index_json);
  if (!source_or.ok()) {
    return source_or.status();
  }
  auto target_or = parse_index_json(target_index_json);
  if (!target_or.ok()) {
    return target_or.status();
  }

  RepresentationTransformContract contract;
  contract.source_byte_space = source_byte_space;
  contract.target_representation.family = std::string(target_representation_family);
  contract.target_representation.realization_kind = RealizationKind::kEphemeralIntoTarget;
  contract.tensor_bindings.reserve(target_or->size());

  std::vector<std::string> names;
  names.reserve(target_or->size());
  for (const auto& [name, _] : *target_or) {
    if (!source_or->contains(name)) {
      return absl::FailedPreconditionError(absl::StrCat("source/target tensor set mismatch for ", name));
    }
    names.push_back(name);
  }
  if (source_or->size() != target_or->size()) {
    for (const auto& [name, _] : *source_or) {
      if (!target_or->contains(name)) {
        return absl::FailedPreconditionError(absl::StrCat("source/target tensor set mismatch for ", name));
      }
    }
  }
  std::sort(names.begin(), names.end(), [&](const std::string& lhs, const std::string& rhs) {
    return source_or->find(lhs)->second.offset < source_or->find(rhs)->second.offset;
  });

  for (const auto& name : names) {
    auto source_it = source_or->find(name);
    auto target_it = target_or->find(name);
    if (source_it == source_or->end() || target_it == target_or->end()) {
      return absl::FailedPreconditionError(absl::StrCat("source/target tensor set mismatch for ", name));
    }

    std::optional<materialization::view::TensorViewOps> tensor_ops;
    if (variant_identity.has_value() && variant_identity->view_spec.has_value()) {
      const auto& tensors = variant_identity->view_spec->tensors;
      if (auto spec_it = tensors.find(name); spec_it != tensors.end()) {
        tensor_ops = spec_it->second;
      }
    }
    auto slice_or = derive_tensor_slice(source_it->second, target_it->second, tensor_ops);
    if (!slice_or.ok()) {
      return slice_or.status();
    }

    RepresentationTensorBinding binding;
    binding.dst_name = name;
    binding.dst_spec = to_representation_tensor_spec(name, target_it->second);
    binding.coverage_kind = CoverageKind::kExact;

    SourceFragment fragment;
    fragment.source_spec = to_representation_tensor_spec(name, source_it->second);
    switch (slice_or->kind) {
      case DerivedTensorSlice::Kind::kFull:
        binding.op_kind = BindingOpKind::kExactCopy;
        fragment.source_range = full_coordinate_spec();
        fragment.destination_range = full_coordinate_spec();
        break;
      case DerivedTensorSlice::Kind::kDim0:
        binding.op_kind = BindingOpKind::kSliceCopy;
        fragment.source_range =
            axis_coordinate_spec(0, slice_or->start, slice_or->start + static_cast<int64_t>(slice_or->length));
        fragment.destination_range =
            axis_coordinate_spec(0, 0, binding.dst_spec.shape.empty() ? 1 : binding.dst_spec.shape.front());
        break;
      case DerivedTensorSlice::Kind::kDim1:
        binding.op_kind = BindingOpKind::kSliceCopy;
        fragment.source_range =
            axis_coordinate_spec(1, slice_or->start, slice_or->start + static_cast<int64_t>(slice_or->length));
        fragment.destination_range = axis_coordinate_spec(
            1,
            0,
            binding.dst_spec.shape.size() < 2 ? static_cast<int64_t>(slice_or->length) : binding.dst_spec.shape[1]);
        break;
    }
    binding.sources.push_back(std::move(fragment));
    contract.tensor_bindings.push_back(std::move(binding));
  }

  auto normalized_or = normalize_representation_transform_contract(std::move(contract));
  if (!normalized_or.ok()) {
    return normalized_or.status();
  }
  auto work_plan_or = build_representation_work_plan(*normalized_or);
  if (!work_plan_or.ok()) {
    return work_plan_or.status();
  }
  return IndexBackedRepresentationBuildResult{
      .transform_contract = std::move(*normalized_or),
      .work_plan = std::move(*work_plan_or),
  };
}

} // namespace tensorcast::store::materialization::contracts
