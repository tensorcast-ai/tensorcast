// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/representation_transform_builder.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace tensorcast::daemon::representation_transform_builder {

namespace {

using materialization_layout::CanonicalIndexEntry;
using materialization_layout::dtype_element_size;
using materialization_layout::product_dims;
using representation_layout::TensorLayoutSpec;
using representation_layout::ViewNarrowSpec;
using tensorcast::store::materialization::contracts::BindingOpKind;
using tensorcast::store::materialization::contracts::CoverageKind;
using tensorcast::store::materialization::contracts::RealizationKind;
using tensorcast::store::materialization::contracts::RepresentationTensorBinding;
using tensorcast::store::materialization::contracts::RepresentationTensorSpec;
using tensorcast::store::materialization::contracts::SourceFragment;
using tensorcast::store::materialization::contracts::SourceFragmentRole;
using tensorcast::store::materialization::contracts::TensorAxisRange;
using tensorcast::store::materialization::contracts::TensorCoordinateSpec;

struct RangeSpec {
  bool has_range{false};
  int32_t dim{0};
  int64_t start{0};
  int64_t end{0};
};

struct CandidateEntry {
  std::string src_name;
  int32_t dim{-1};
  RangeSpec src_range;
  RangeSpec dst_range;
};

RangeSpec build_range(const v2::CopyPlanRange& range) {
  return RangeSpec{
      .has_range = true,
      .dim = static_cast<int32_t>(range.dim()),
      .start = range.start(),
      .end = range.end(),
  };
}

bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
  return representation_layout::is_contiguous(shape, stride);
}

std::vector<int64_t> effective_slice_shape(const std::vector<int64_t>& shape, const RangeSpec& range, int32_t dim) {
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> result = shape;
  if (dim >= 0 && dim < static_cast<int32_t>(result.size()) && range.has_range) {
    result[static_cast<size_t>(dim)] = range.end - range.start;
  }
  result.erase(std::remove_if(result.begin(), result.end(), [](int64_t size) { return size == 1; }), result.end());
  return result;
}

absl::Status validate_and_build_segments(
    int idx,
    const v2::CopyPlanEntry& entry,
    const CanonicalIndexEntry& src_entry,
    const TensorLayoutSpec& dst_spec,
    uint64_t dst_base_offset,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows,
    RangeSpec src_range,
    RangeSpec dst_range,
    absl::flat_hash_map<std::string, int32_t>& dst_dim_by_name,
    absl::flat_hash_map<std::string, std::vector<std::tuple<int64_t, int64_t, int>>>& dst_intervals,
    std::vector<tensorcast::store::loader::ByteRangeSegment>& out_segments,
    uint64_t& total_bytes_copied) {
  auto elem_or = dtype_element_size(src_entry.dtype);
  if (!elem_or.ok()) {
    return elem_or.status();
  }
  const uint64_t src_elem_size = *elem_or;
  if (src_entry.dtype != dst_spec.dtype) {
    return absl::InvalidArgumentError(
        std::format(
            "copy_plan entry {} dtype mismatch: {}({}) -> {}({})",
            idx,
            entry.ckpt_name(),
            src_entry.dtype,
            entry.dst_name(),
            dst_spec.dtype));
  }
  if (!is_contiguous(src_entry.shape, src_entry.stride)) {
    return absl::InvalidArgumentError(
        std::format("copy_plan entry {} source '{}' must be contiguous", idx, entry.ckpt_name()));
  }
  if (!is_contiguous(dst_spec.shape, dst_spec.stride)) {
    return absl::InvalidArgumentError(
        std::format("copy_plan entry {} target '{}' must be contiguous", idx, entry.dst_name()));
  }

  if (src_entry.shape.empty() && src_range.has_range) {
    return absl::InvalidArgumentError(std::format("copy_plan entry {} source scalar must use full range", idx));
  }
  if (dst_spec.shape.empty() && dst_range.has_range) {
    return absl::InvalidArgumentError(std::format("copy_plan entry {} target scalar must use full range", idx));
  }

  int32_t dim = 0;
  bool dim_set = false;
  if (src_range.has_range) {
    dim = src_range.dim;
    dim_set = true;
  }
  if (dst_range.has_range) {
    if (dim_set && dim != dst_range.dim) {
      return absl::InvalidArgumentError(
          std::format("copy_plan entry {} range dim mismatch ({} vs {})", idx, dim, dst_range.dim));
    }
    dim = dst_range.dim;
    dim_set = true;
  }
  if (!dim_set) {
    dim = 0;
  }
  if (dim < 0 || dim > 1) {
    return absl::InvalidArgumentError(std::format("copy_plan entry {} dim must be 0 or 1 (got {})", idx, dim));
  }

  const auto normalize_range =
      [&](const std::vector<int64_t>& shape, RangeSpec& range, std::string_view role) -> absl::Status {
    if (shape.empty()) {
      return absl::OkStatus();
    }
    if (dim >= static_cast<int32_t>(shape.size())) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} {} dim {} out of range", idx, role, dim));
    }
    if (!range.has_range) {
      range.has_range = true;
      range.dim = dim;
      range.start = 0;
      range.end = shape[static_cast<size_t>(dim)];
      return absl::OkStatus();
    }
    if (range.dim != dim) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} {} dim mismatch", idx, role));
    }
    if (range.start < 0 || range.end <= range.start) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} {} range invalid", idx, role));
    }
    if (range.end > shape[static_cast<size_t>(dim)]) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} {} range out of bounds", idx, role));
    }
    return absl::OkStatus();
  };

  auto src_shape = src_entry.shape;
  auto dst_shape = dst_spec.shape;
  auto src_stride = src_entry.stride;
  auto dst_stride = dst_spec.stride;
  if (src_shape.empty()) {
    src_shape = {1};
    src_stride = {1};
  }
  if (dst_shape.empty()) {
    dst_shape = {1};
    dst_stride = {1};
  }

  if (view_narrows.contains(entry.ckpt_name())) {
    const auto& narrow = view_narrows.at(entry.ckpt_name());
    if (!src_range.has_range || src_range.dim != narrow.dim) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} source range required for view narrow", idx));
    }
    if (src_range.start < narrow.start || src_range.end > narrow.end) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} source range outside view bounds", idx));
    }
    src_range.start -= narrow.start;
    src_range.end -= narrow.start;
  }

  if (!src_shape.empty()) {
    auto status = normalize_range(src_shape, src_range, "source");
    if (!status.ok()) {
      return status;
    }
  }
  if (!dst_shape.empty()) {
    auto status = normalize_range(dst_shape, dst_range, "target");
    if (!status.ok()) {
      return status;
    }
  }
  if (effective_slice_shape(src_shape, src_range, dim) != effective_slice_shape(dst_shape, dst_range, dim)) {
    return absl::InvalidArgumentError(std::format("copy_plan entry {} shape mismatch for {}", idx, entry.dst_name()));
  }

  auto src_count_or = product_dims(src_shape);
  if (!src_count_or.ok()) {
    return src_count_or.status();
  }
  auto dst_count_or = product_dims(dst_shape);
  if (!dst_count_or.ok()) {
    return dst_count_or.status();
  }
  uint64_t src_elements = src_shape.empty() ? 1 : *src_count_or;
  uint64_t dst_elements = dst_shape.empty() ? 1 : *dst_count_or;

  uint64_t src_slice_elements = src_elements;
  if (!src_shape.empty()) {
    const uint64_t slice_dim = static_cast<uint64_t>(src_range.end - src_range.start);
    auto tail_or = product_dims(absl::Span<const int64_t>(src_shape).subspan(dim + 1));
    if (!tail_or.ok()) {
      return tail_or.status();
    }
    const uint64_t tail = *tail_or;
    if (dim == 0) {
      src_slice_elements = slice_dim * tail;
    } else {
      src_slice_elements = slice_dim * tail * static_cast<uint64_t>(src_shape.front());
    }
  }

  uint64_t dst_slice_elements = dst_elements;
  if (!dst_shape.empty()) {
    const uint64_t slice_dim = static_cast<uint64_t>(dst_range.end - dst_range.start);
    auto tail_or = product_dims(absl::Span<const int64_t>(dst_shape).subspan(dim + 1));
    if (!tail_or.ok()) {
      return tail_or.status();
    }
    const uint64_t tail = *tail_or;
    if (dim == 0) {
      dst_slice_elements = slice_dim * tail;
    } else {
      dst_slice_elements = slice_dim * tail * static_cast<uint64_t>(dst_shape.front());
    }
  }

  if (src_slice_elements != dst_slice_elements) {
    return absl::InvalidArgumentError(std::format("copy_plan entry {} element count mismatch", idx));
  }

  auto record_interval = [&](const std::string& name, const RangeSpec& range) -> absl::Status {
    auto it = dst_dim_by_name.find(name);
    if (it == dst_dim_by_name.end()) {
      dst_dim_by_name.emplace(name, range.dim);
    } else if (it->second != range.dim) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} mixes slice dims for {}", idx, name));
    }
    auto& intervals = dst_intervals[name];
    intervals.emplace_back(range.start, range.end, idx);
    return absl::OkStatus();
  };
  RangeSpec record_range = dst_range;
  if (!record_range.has_range) {
    record_range.has_range = true;
    record_range.dim = dim;
    record_range.start = 0;
    record_range.end = dst_shape[static_cast<size_t>(dim)];
  }
  auto status = record_interval(entry.dst_name(), record_range);
  if (!status.ok()) {
    return status;
  }

  const uint64_t src_base_offset = src_entry.logical_offset + src_entry.storage_offset * src_elem_size;
  const uint64_t dst_base_offset_bytes = dst_base_offset;

  auto tail_or = product_dims(absl::Span<const int64_t>(src_shape).subspan(dim + 1));
  if (!tail_or.ok()) {
    return tail_or.status();
  }
  uint64_t row_elems = *tail_or;
  if (src_shape.empty()) {
    row_elems = 1;
  }

  if (dim == 0 || src_shape.empty()) {
    const uint64_t length_elems = static_cast<uint64_t>(src_range.end - src_range.start) * row_elems;
    const uint64_t src_offset_elems = static_cast<uint64_t>(src_range.start) * src_stride[0];
    const uint64_t dst_offset_elems = static_cast<uint64_t>(dst_range.start) * dst_stride[0];
    tensorcast::store::loader::ByteRangeSegment seg;
    seg.kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData;
    seg.src_offset = src_base_offset + src_offset_elems * src_elem_size;
    seg.dst_offset = dst_base_offset_bytes + dst_offset_elems * dst_spec.element_size;
    seg.length = length_elems * src_elem_size;
    seg.source_index = 0;
    out_segments.push_back(seg);
    total_bytes_copied += seg.length;
    return absl::OkStatus();
  }

  if (dim == 1) {
    if (src_shape.size() < 2 || dst_shape.size() < 2) {
      return absl::InvalidArgumentError(std::format("copy_plan entry {} dim1 requires at least 2D tensors", idx));
    }
    const uint64_t length_elems = static_cast<uint64_t>(src_range.end - src_range.start) * row_elems;
    const int64_t outer = src_shape.front();
    for (int64_t row = 0; row < outer; ++row) {
      const uint64_t src_offset_elems =
          static_cast<uint64_t>(row) * src_stride[0] + static_cast<uint64_t>(src_range.start) * src_stride[1];
      const uint64_t dst_offset_elems =
          static_cast<uint64_t>(row) * dst_stride[0] + static_cast<uint64_t>(dst_range.start) * dst_stride[1];
      tensorcast::store::loader::ByteRangeSegment seg;
      seg.kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData;
      seg.src_offset = src_base_offset + src_offset_elems * src_elem_size;
      seg.dst_offset = dst_base_offset_bytes + dst_offset_elems * dst_spec.element_size;
      seg.length = length_elems * src_elem_size;
      seg.source_index = 0;
      out_segments.push_back(seg);
      total_bytes_copied += seg.length;
    }
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("copy_plan dim must be 0 or 1");
}

RepresentationTensorSpec to_source_tensor_spec(std::string_view name, const CanonicalIndexEntry& entry) {
  auto elem_or = dtype_element_size(entry.dtype);
  const uint64_t element_size = elem_or.ok() ? *elem_or : 0;
  return RepresentationTensorSpec{
      .name = std::string(name),
      .shape = entry.shape,
      .stride = entry.stride,
      .dtype = entry.dtype,
      .logical_offset = entry.logical_offset,
      .logical_length = entry.logical_length,
      .storage_offset = entry.storage_offset,
      .element_size = element_size,
  };
}

RepresentationTensorSpec to_dst_tensor_spec(std::string_view name, const TensorLayoutSpec& spec, uint64_t base_offset) {
  return RepresentationTensorSpec{
      .name = std::string(name),
      .shape = spec.shape,
      .stride = spec.stride,
      .dtype = spec.dtype,
      .logical_offset = base_offset,
      .logical_length = spec.logical_length,
      .storage_offset = spec.storage_offset,
      .element_size = spec.element_size,
  };
}

TensorCoordinateSpec to_coordinate_spec(const RangeSpec& range) {
  TensorCoordinateSpec out;
  if (!range.has_range) {
    return out;
  }
  out.axes.push_back(TensorAxisRange{.dim = range.dim, .start = range.start, .end = range.end});
  return out;
}

void merge_adjacent_fragments(std::vector<SourceFragment>* fragments) {
  if (fragments == nullptr || fragments->size() < 2) {
    return;
  }
  std::vector<SourceFragment> merged;
  merged.reserve(fragments->size());
  for (const auto& fragment : *fragments) {
    if (merged.empty()) {
      merged.push_back(fragment);
      continue;
    }
    auto& previous = merged.back();
    if (previous.role != fragment.role || previous.source_spec != fragment.source_spec) {
      merged.push_back(fragment);
      continue;
    }
    if (previous.source_range.axes.size() != 1 || fragment.source_range.axes.size() != 1 ||
        previous.destination_range.axes.size() != 1 || fragment.destination_range.axes.size() != 1) {
      merged.push_back(fragment);
      continue;
    }
    auto& previous_src = previous.source_range.axes.front();
    auto& previous_dst = previous.destination_range.axes.front();
    const auto& current_src = fragment.source_range.axes.front();
    const auto& current_dst = fragment.destination_range.axes.front();
    if (previous_src.dim == current_src.dim && previous_dst.dim == current_dst.dim &&
        previous_src.end == current_src.start && previous_dst.end == current_dst.start) {
      previous_src.end = current_src.end;
      previous_dst.end = current_dst.end;
      continue;
    }
    merged.push_back(fragment);
  }
  *fragments = std::move(merged);
}

std::optional<int32_t> infer_partition_dim(const RepresentationTensorBinding& binding) {
  if (binding.sources.size() != 1) {
    return std::nullopt;
  }
  const auto& fragment = binding.sources.front();
  if (fragment.source_range.axes.empty() && fragment.destination_range.axes.empty()) {
    return -1;
  }
  if (fragment.source_range.axes.size() != 1 || fragment.destination_range.axes.size() != 1) {
    return std::nullopt;
  }
  const auto& src_axis = fragment.source_range.axes.front();
  const auto& dst_axis = fragment.destination_range.axes.front();
  if (src_axis.dim != dst_axis.dim) {
    return std::nullopt;
  }
  return src_axis.dim;
}

} // namespace

absl::StatusOr<BuildRepresentationTransformResult> build_representation_transform_contract(
    const v2::CopyPlan& copy_plan,
    const materialization_layout::CanonicalIndexTable& source_table,
    const materialization_layout::CanonicalIndexTable& canonical_source_table,
    const absl::flat_hash_map<std::string, TensorLayoutSpec>& dst_specs,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows,
    const tensorcast::common::v1::ByteSpaceRef& source_byte_space,
    std::string_view representation_family) {
  BuildRepresentationTransformResult result;
  result.generic_fallback_map.total_bytes = 0;
  result.generic_fallback_map.num_sources = 1;

  std::vector<tensorcast::store::loader::ByteRangeSegment> mapped_segments;
  mapped_segments.reserve(copy_plan.entries_size());
  absl::flat_hash_map<std::string, std::vector<CandidateEntry>> candidate_entries_by_dst;
  candidate_entries_by_dst.reserve(copy_plan.entries_size());

  absl::flat_hash_map<std::string, int32_t> dst_dim_by_name;
  absl::flat_hash_map<std::string, std::vector<std::tuple<int64_t, int64_t, int>>> dst_intervals;

  int entry_index = 0;
  for (const auto& entry : copy_plan.entries()) {
    if (entry.ckpt_name().empty()) {
      return absl::InvalidArgumentError("copy_plan entry missing ckpt_name");
    }
    if (entry.dst_name().empty()) {
      return absl::InvalidArgumentError("copy_plan entry missing dst_name");
    }
    auto src_it = source_table.entries.find(entry.ckpt_name());
    if (src_it == source_table.entries.end()) {
      return absl::InvalidArgumentError("copy_plan references unknown source tensor");
    }
    auto dst_it = dst_specs.find(entry.dst_name());
    if (dst_it == dst_specs.end()) {
      return absl::InvalidArgumentError("copy_plan references unknown destination tensor");
    }
    auto base_it = dst_base_offsets.find(entry.dst_name());
    if (base_it == dst_base_offsets.end()) {
      return absl::InvalidArgumentError("copy_plan references destination tensor without base offset");
    }

    RangeSpec src_range;
    if (entry.has_ckpt_range()) {
      src_range = build_range(entry.ckpt_range());
    }
    RangeSpec dst_range;
    if (entry.has_dst_range()) {
      dst_range = build_range(entry.dst_range());
    }

    auto status = validate_and_build_segments(
        entry_index,
        entry,
        src_it->second,
        dst_it->second,
        base_it->second,
        view_narrows,
        src_range,
        dst_range,
        dst_dim_by_name,
        dst_intervals,
        mapped_segments,
        result.total_bytes_copied);
    if (!status.ok()) {
      return status;
    }

    candidate_entries_by_dst[entry.dst_name()].push_back(
        CandidateEntry{
            .src_name = entry.ckpt_name(),
            .dim = (dst_range.has_range ? dst_range.dim : (src_range.has_range ? src_range.dim : -1)),
            .src_range = src_range,
            .dst_range = dst_range,
        });
    ++entry_index;
  }

  for (const auto& [name, intervals] : dst_intervals) {
    const auto& spec = dst_specs.at(name);
    int32_t dim = 0;
    if (auto it = dst_dim_by_name.find(name); it != dst_dim_by_name.end()) {
      dim = it->second;
    }
    if (spec.shape.empty()) {
      if (intervals.size() != 1) {
        return absl::InvalidArgumentError("copy_plan must include one entry for scalar dst");
      }
      const auto& interval = intervals.front();
      if (std::get<0>(interval) != 0 || std::get<1>(interval) != 1) {
        return absl::InvalidArgumentError("copy_plan scalar dst must cover full range");
      }
      continue;
    }
    if (dim < 0 || dim >= static_cast<int32_t>(spec.shape.size())) {
      return absl::InvalidArgumentError("copy_plan dim out of range for dst tensor");
    }
    std::vector<std::tuple<int64_t, int64_t, int>> sorted = intervals;
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
      return std::get<0>(lhs) < std::get<0>(rhs);
    });
    int64_t cursor = 0;
    for (const auto& interval : sorted) {
      const int64_t start = std::get<0>(interval);
      const int64_t end = std::get<1>(interval);
      if (start < cursor) {
        return absl::InvalidArgumentError("copy_plan has overlapping dst ranges");
      }
      if (start != cursor) {
        return absl::InvalidArgumentError("copy_plan has gaps in dst ranges");
      }
      cursor = end;
    }
    if (cursor != spec.shape[static_cast<size_t>(dim)]) {
      return absl::InvalidArgumentError("copy_plan does not cover full dst range");
    }
  }

  result.generic_fallback_map.segments = std::move(mapped_segments);
  result.transform_contract.source_byte_space = source_byte_space;
  result.transform_contract.target_representation.family = std::string(representation_family);
  result.transform_contract.target_representation.realization_kind = RealizationKind::kEphemeralIntoTarget;
  result.compatibility_stats.total_dst_tensors = candidate_entries_by_dst.size();
  result.transform_contract.tensor_bindings.reserve(candidate_entries_by_dst.size());

  for (const auto& [dst_name, entries] : candidate_entries_by_dst) {
    if (entries.empty()) {
      continue;
    }
    const auto dst_spec_it = dst_specs.find(dst_name);
    const auto dst_base_it = dst_base_offsets.find(dst_name);
    if (dst_spec_it == dst_specs.end() || dst_base_it == dst_base_offsets.end()) {
      return absl::InvalidArgumentError("destination tensor metadata missing");
    }
    RepresentationTensorBinding binding;
    binding.dst_name = dst_name;
    binding.dst_spec = to_dst_tensor_spec(dst_name, dst_spec_it->second, dst_base_it->second);

    std::vector<CandidateEntry> sorted_entries = entries;
    std::sort(sorted_entries.begin(), sorted_entries.end(), [](const CandidateEntry& lhs, const CandidateEntry& rhs) {
      const int64_t lhs_start = lhs.dst_range.has_range ? lhs.dst_range.start : 0;
      const int64_t rhs_start = rhs.dst_range.has_range ? rhs.dst_range.start : 0;
      if (lhs_start != rhs_start) {
        return lhs_start < rhs_start;
      }
      if (lhs.src_name != rhs.src_name) {
        return lhs.src_name < rhs.src_name;
      }
      return lhs.dim < rhs.dim;
    });

    bool single_source = true;
    for (const auto& entry : sorted_entries) {
      if (entry.src_name != sorted_entries.front().src_name) {
        single_source = false;
        break;
      }
    }

    binding.op_kind = single_source ? BindingOpKind::kSliceCopy : BindingOpKind::kConcat;
    if (single_source && sorted_entries.size() == 1 && !sorted_entries.front().src_range.has_range &&
        !sorted_entries.front().dst_range.has_range) {
      binding.op_kind = BindingOpKind::kExactCopy;
    }

    binding.sources.reserve(sorted_entries.size());
    for (const auto& entry : sorted_entries) {
      auto src_it = canonical_source_table.entries.find(entry.src_name);
      if (src_it == canonical_source_table.entries.end()) {
        return absl::InvalidArgumentError("copy_plan references unknown canonical source tensor");
      }
      binding.sources.push_back(
          SourceFragment{
              .source_spec = to_source_tensor_spec(entry.src_name, src_it->second),
              .source_range = to_coordinate_spec(entry.src_range),
              .destination_range = to_coordinate_spec(entry.dst_range),
              .role = single_source ? SourceFragmentRole::kDefault : SourceFragmentRole::kConcat,
          });
    }
    if (single_source) {
      merge_adjacent_fragments(&binding.sources);
    }

    auto partition_dim = infer_partition_dim(binding);
    if (binding.op_kind == BindingOpKind::kConcat) {
      bool concat_compatible = true;
      for (const auto& fragment : binding.sources) {
        if (fragment.source_range.axes.size() != 1 || fragment.destination_range.axes.size() != 1 ||
            fragment.source_range.axes.front().dim != 0 || fragment.destination_range.axes.front().dim != 0) {
          concat_compatible = false;
          break;
        }
      }
      if (concat_compatible) {
        result.compatibility_stats.concat_candidates += 1;
        result.compatibility_stats.concat_bytes += binding.dst_spec.logical_length;
      } else {
        result.compatibility_stats.rejected_mixed_src_or_dim += 1;
        result.compatibility_stats.rejected_mixed_src_or_dim_bytes += binding.dst_spec.logical_length;
      }
    } else if (!partition_dim.has_value()) {
      result.compatibility_stats.rejected_non_contiguous += 1;
      result.compatibility_stats.rejected_non_contiguous_bytes += binding.dst_spec.logical_length;
    } else if (*partition_dim < -1 || *partition_dim > 1) {
      result.compatibility_stats.rejected_unsupported_distribution += 1;
      result.compatibility_stats.rejected_unsupported_distribution_bytes += binding.dst_spec.logical_length;
    } else {
      result.compatibility_stats.compatible_candidates += 1;
      result.compatibility_stats.compatible_bytes += binding.dst_spec.logical_length;
    }

    result.transform_contract.tensor_bindings.push_back(std::move(binding));
  }

  auto normalized_contract_or =
      tensorcast::store::materialization::contracts::normalize_representation_transform_contract(
          std::move(result.transform_contract));
  if (!normalized_contract_or.ok()) {
    return normalized_contract_or.status();
  }
  result.transform_contract = std::move(*normalized_contract_or);
  return result;
}

} // namespace tensorcast::daemon::representation_transform_builder
