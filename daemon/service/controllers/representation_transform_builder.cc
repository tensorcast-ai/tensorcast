// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/representation_transform_builder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
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

using ProfileClock = std::chrono::steady_clock;

double seconds_between(ProfileClock::time_point start, ProfileClock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

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

using ByteRangeSegment = tensorcast::store::loader::ByteRangeSegment;

std::vector<std::string> sorted_dst_names_by_base_offset(
    const absl::flat_hash_map<std::string, std::vector<ByteRangeSegment>>& segments_by_dst,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets) {
  std::vector<std::string> names;
  names.reserve(segments_by_dst.size());
  for (const auto& [name, segments] : segments_by_dst) {
    if (!segments.empty()) {
      names.push_back(name);
    }
  }
  std::sort(names.begin(), names.end(), [&](const std::string& lhs, const std::string& rhs) {
    const auto lhs_it = dst_base_offsets.find(lhs);
    const auto rhs_it = dst_base_offsets.find(rhs);
    const uint64_t lhs_offset =
        lhs_it == dst_base_offsets.end() ? std::numeric_limits<uint64_t>::max() : lhs_it->second;
    const uint64_t rhs_offset =
        rhs_it == dst_base_offsets.end() ? std::numeric_limits<uint64_t>::max() : rhs_it->second;
    if (lhs_offset != rhs_offset) {
      return lhs_offset < rhs_offset;
    }
    return lhs < rhs;
  });
  return names;
}

void append_segments_preserving_dst_order(
    const std::vector<ByteRangeSegment>& segments,
    std::vector<ByteRangeSegment>* out) {
  if (segments.empty() || out == nullptr) {
    return;
  }
  const bool sorted = std::is_sorted(segments.begin(), segments.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.dst_offset != rhs.dst_offset) {
      return lhs.dst_offset < rhs.dst_offset;
    }
    return lhs.src_offset < rhs.src_offset;
  });
  if (sorted) {
    out->insert(out->end(), segments.begin(), segments.end());
    return;
  }
  std::vector<ByteRangeSegment> ordered = segments;
  std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.dst_offset != rhs.dst_offset) {
      return lhs.dst_offset < rhs.dst_offset;
    }
    return lhs.src_offset < rhs.src_offset;
  });
  out->insert(out->end(), ordered.begin(), ordered.end());
}

std::vector<ByteRangeSegment> flatten_segments_by_dst_order(
    const absl::flat_hash_map<std::string, std::vector<ByteRangeSegment>>& segments_by_dst,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_set<std::string>* included_dst_names = nullptr) {
  std::vector<ByteRangeSegment> out;
  size_t segment_count = 0;
  for (const auto& [name, segments] : segments_by_dst) {
    if (included_dst_names != nullptr && !included_dst_names->contains(name)) {
      continue;
    }
    segment_count += segments.size();
  }
  out.reserve(segment_count);
  const auto names = sorted_dst_names_by_base_offset(segments_by_dst, dst_base_offsets);
  for (const auto& name : names) {
    if (included_dst_names != nullptr && !included_dst_names->contains(name)) {
      continue;
    }
    auto it = segments_by_dst.find(name);
    if (it == segments_by_dst.end()) {
      continue;
    }
    append_segments_preserving_dst_order(it->second, &out);
  }
  return out;
}

tensorcast::store::loader::ByteRangeMap merge_ordered_byte_range_maps(
    tensorcast::store::loader::ByteRangeMap lhs,
    tensorcast::store::loader::ByteRangeMap rhs) {
  if (lhs.segments.empty()) {
    return rhs;
  }
  if (rhs.segments.empty()) {
    return lhs;
  }
  tensorcast::store::loader::ByteRangeMap merged;
  merged.total_bytes = std::max(lhs.total_bytes, rhs.total_bytes);
  merged.num_sources = std::max(lhs.num_sources, rhs.num_sources);
  merged.segments.reserve(lhs.segments.size() + rhs.segments.size());
  size_t lhs_index = 0;
  size_t rhs_index = 0;
  while (lhs_index < lhs.segments.size() || rhs_index < rhs.segments.size()) {
    const bool take_lhs = rhs_index == rhs.segments.size() ||
        (lhs_index < lhs.segments.size() && lhs.segments[lhs_index].dst_offset <= rhs.segments[rhs_index].dst_offset);
    merged.segments.push_back(take_lhs ? lhs.segments[lhs_index++] : rhs.segments[rhs_index++]);
  }
  return merged;
}

RangeSpec build_range(const v2::CopyPlanRange& range) {
  return RangeSpec{
      .has_range = true,
      .dim = static_cast<int32_t>(range.dim()),
      .start = range.start(),
      .end = range.end(),
  };
}

RangeSpec build_range(const v2::BindingRealizationRange& range) {
  return RangeSpec{
      .has_range = true,
      .dim = static_cast<int32_t>(range.dim()),
      .start = range.start(),
      .end = range.end(),
  };
}

void copy_range(const v2::BindingRealizationRange& from, v2::CopyPlanRange* to) {
  if (to == nullptr) {
    return;
  }
  to->set_dim(from.dim());
  to->set_start(from.start());
  to->set_end(from.end());
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
  const bool source_is_full = fragment.source_range.axes.empty();
  const bool destination_is_full = fragment.destination_range.axes.empty();
  if (source_is_full && destination_is_full) {
    return -1;
  }
  if ((!source_is_full && fragment.source_range.axes.size() != 1) ||
      (!destination_is_full && fragment.destination_range.axes.size() != 1)) {
    return std::nullopt;
  }
  const TensorAxisRange* src_axis = source_is_full ? nullptr : &fragment.source_range.axes.front();
  const TensorAxisRange* dst_axis = destination_is_full ? nullptr : &fragment.destination_range.axes.front();
  const int32_t dim = src_axis != nullptr ? src_axis->dim : dst_axis->dim;
  if (src_axis != nullptr && dst_axis != nullptr && src_axis->dim != dst_axis->dim) {
    return std::nullopt;
  }
  if (dim < 0 || dim >= static_cast<int32_t>(fragment.source_spec.shape.size()) ||
      dim >= static_cast<int32_t>(binding.dst_spec.shape.size())) {
    return std::nullopt;
  }
  if (fragment.source_spec.shape.size() != binding.dst_spec.shape.size()) {
    return std::nullopt;
  }
  for (size_t axis = 0; axis < fragment.source_spec.shape.size(); ++axis) {
    const int64_t src_extent = src_axis != nullptr && static_cast<int32_t>(axis) == src_axis->dim
        ? src_axis->end - src_axis->start
        : fragment.source_spec.shape[axis];
    const int64_t dst_extent = dst_axis != nullptr && static_cast<int32_t>(axis) == dst_axis->dim
        ? dst_axis->end - dst_axis->start
        : binding.dst_spec.shape[axis];
    if (src_extent != dst_extent) {
      return std::nullopt;
    }
  }
  return dim;
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
    std::string_view representation_family,
    bool compute_identity_hashes) {
  const auto total_begin = ProfileClock::now();
  BuildRepresentationTransformResult result;
  result.generic_fallback_map.total_bytes = 0;
  result.generic_fallback_map.num_sources = 1;
  result.collective_lowered_map.total_bytes = 0;
  result.collective_lowered_map.num_sources = 1;

  std::vector<tensorcast::store::loader::ByteRangeSegment> mapped_segments;
  mapped_segments.reserve(copy_plan.entries_size());
  absl::flat_hash_map<std::string, std::vector<tensorcast::store::loader::ByteRangeSegment>> mapped_segments_by_dst;
  mapped_segments_by_dst.reserve(copy_plan.entries_size());
  absl::flat_hash_map<std::string, std::vector<CandidateEntry>> candidate_entries_by_dst;
  candidate_entries_by_dst.reserve(copy_plan.entries_size());

  absl::flat_hash_map<std::string, int32_t> dst_dim_by_name;
  absl::flat_hash_map<std::string, std::vector<std::tuple<int64_t, int64_t, int>>> dst_intervals;

  const auto build_segments_begin = ProfileClock::now();
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

    const size_t mapped_segments_begin = mapped_segments.size();
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
    auto& dst_segments = mapped_segments_by_dst[entry.dst_name()];
    for (size_t segment_index = mapped_segments_begin; segment_index < mapped_segments.size(); ++segment_index) {
      dst_segments.push_back(mapped_segments[segment_index]);
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
  const auto build_segments_end = ProfileClock::now();
  const size_t mapped_segment_count = mapped_segments.size();

  const auto validate_intervals_begin = ProfileClock::now();
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
  const auto validate_intervals_end = ProfileClock::now();

  result.transform_contract.source_byte_space = source_byte_space;
  result.transform_contract.target_representation.family = std::string(representation_family);
  result.transform_contract.target_representation.realization_kind = RealizationKind::kEphemeralIntoTarget;
  result.lowering_stats.total_dst_tensors = candidate_entries_by_dst.size();
  result.transform_contract.tensor_bindings.reserve(candidate_entries_by_dst.size());

  const auto build_bindings_begin = ProfileClock::now();
  absl::flat_hash_set<std::string> collective_dst_names;
  collective_dst_names.reserve(candidate_entries_by_dst.size());
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
      bool concat_collective_eligible = true;
      for (const auto& fragment : binding.sources) {
        if (fragment.source_range.axes.size() != 1 || fragment.destination_range.axes.size() != 1 ||
            fragment.source_range.axes.front().dim != 0 || fragment.destination_range.axes.front().dim != 0) {
          concat_collective_eligible = false;
          break;
        }
      }
      if (concat_collective_eligible) {
        result.lowering_stats.concat_candidates += 1;
        result.lowering_stats.concat_bytes += binding.dst_spec.logical_length;
        collective_dst_names.insert(dst_name);
      } else {
        result.lowering_stats.rejected_mixed_src_or_dim += 1;
        result.lowering_stats.rejected_mixed_src_or_dim_bytes += binding.dst_spec.logical_length;
      }
    } else if (!partition_dim.has_value()) {
      result.lowering_stats.rejected_non_contiguous += 1;
      result.lowering_stats.rejected_non_contiguous_bytes += binding.dst_spec.logical_length;
    } else if (*partition_dim < -1 || *partition_dim > 1) {
      result.lowering_stats.rejected_unsupported_distribution += 1;
      result.lowering_stats.rejected_unsupported_distribution_bytes += binding.dst_spec.logical_length;
    } else {
      result.lowering_stats.collective_candidates += 1;
      result.lowering_stats.collective_bytes += binding.dst_spec.logical_length;
      collective_dst_names.insert(dst_name);
    }

    result.transform_contract.tensor_bindings.push_back(std::move(binding));
  }
  result.generic_fallback_map.segments = flatten_segments_by_dst_order(mapped_segments_by_dst, dst_base_offsets);
  result.collective_lowered_map.segments =
      flatten_segments_by_dst_order(mapped_segments_by_dst, dst_base_offsets, &collective_dst_names);
  const auto build_bindings_end = ProfileClock::now();

  const auto normalize_begin = ProfileClock::now();
  auto normalized_contract_or =
      tensorcast::store::materialization::contracts::normalize_representation_transform_contract(
          std::move(result.transform_contract),
          tensorcast::store::materialization::contracts::NormalizeRepresentationTransformContractOptions{
              .compute_identity_hashes = compute_identity_hashes,
          });
  if (!normalized_contract_or.ok()) {
    return normalized_contract_or.status();
  }
  result.transform_contract = std::move(*normalized_contract_or);
  const auto normalize_end = ProfileClock::now();
  LOG(INFO) << "tc_profile representation_copy_plan timings"
            << " total_sec=" << seconds_between(total_begin, normalize_end)
            << " build_segments_sec=" << seconds_between(build_segments_begin, build_segments_end)
            << " validate_intervals_sec=" << seconds_between(validate_intervals_begin, validate_intervals_end)
            << " build_bindings_sec=" << seconds_between(build_bindings_begin, build_bindings_end)
            << " normalize_sec=" << seconds_between(normalize_begin, normalize_end)
            << " compute_identity_hashes=" << (compute_identity_hashes ? 1 : 0)
            << " entries=" << copy_plan.entries_size() << " dst_interval_groups=" << dst_intervals.size()
            << " candidate_dst_count=" << candidate_entries_by_dst.size() << " mapped_segments=" << mapped_segment_count
            << " collective_segments=" << result.collective_lowered_map.segments.size()
            << " fallback_segments=" << result.generic_fallback_map.segments.size()
            << " tensor_bindings=" << result.transform_contract.tensor_bindings.size()
            << " total_bytes_copied=" << result.total_bytes_copied;
  return result;
}

namespace {

struct RealizationManualEntry {
  const v2::BindingRealizationEntry* proto{nullptr};
  BindingOpKind op_kind{BindingOpKind::kExactCopy};
  TensorCoordinateSpec source_coordinate;
  TensorCoordinateSpec destination_coordinate;
  bool has_full_destination{false};
};

struct ByteSpan {
  uint64_t offset{0};
  uint64_t length{0};
};

absl::StatusOr<TensorCoordinateSpec> build_coordinate_from_ranges(
    std::string_view entry_kind,
    int idx,
    std::string_view tensor_name,
    const RepresentationTensorSpec& spec,
    const google::protobuf::RepeatedPtrField<v2::BindingRealizationRange>& ranges,
    const std::optional<ViewNarrowSpec>& view_narrow,
    bool require_view_axis) {
  TensorCoordinateSpec coordinate;
  if (spec.shape.empty()) {
    if (!ranges.empty()) {
      return absl::InvalidArgumentError(
          std::format("{} entry {} scalar tensor '{}' does not accept ranges", entry_kind, idx, tensor_name));
    }
    return coordinate;
  }
  std::vector<bool> seen(spec.shape.size(), false);
  coordinate.axes.reserve(ranges.size());
  bool adjusted_view = !view_narrow.has_value();
  for (const auto& range_proto : ranges) {
    RangeSpec range = build_range(range_proto);
    if (range.dim < 0 || range.dim >= static_cast<int32_t>(spec.shape.size())) {
      return absl::InvalidArgumentError(
          std::format("{} entry {} tensor '{}' dim {} out of range", entry_kind, idx, tensor_name, range.dim));
    }
    if (seen[static_cast<size_t>(range.dim)]) {
      return absl::InvalidArgumentError(
          std::format("{} entry {} tensor '{}' contains duplicate dims", entry_kind, idx, tensor_name));
    }
    seen[static_cast<size_t>(range.dim)] = true;
    if (view_narrow.has_value() && range.dim == view_narrow->dim) {
      if (range.start < view_narrow->start || range.end > view_narrow->end) {
        return absl::InvalidArgumentError(std::format("{} entry {} source range outside view bounds", entry_kind, idx));
      }
      range.start -= view_narrow->start;
      range.end -= view_narrow->start;
      adjusted_view = true;
    }
    if (range.start < 0 || range.end <= range.start || range.end > spec.shape[static_cast<size_t>(range.dim)]) {
      return absl::InvalidArgumentError(
          std::format("{} entry {} tensor '{}' range is out of bounds", entry_kind, idx, tensor_name));
    }
    coordinate.axes.push_back(TensorAxisRange{.dim = range.dim, .start = range.start, .end = range.end});
  }
  if (view_narrow.has_value() && require_view_axis && !adjusted_view) {
    return absl::InvalidArgumentError(
        std::format("{} entry {} source range required for view narrow", entry_kind, idx));
  }
  std::sort(coordinate.axes.begin(), coordinate.axes.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.dim < rhs.dim;
  });
  return coordinate;
}

std::vector<int64_t> effective_coordinate_shape(
    const RepresentationTensorSpec& spec,
    const TensorCoordinateSpec& coordinate) {
  if (spec.shape.empty()) {
    return {};
  }
  std::vector<int64_t> extents = spec.shape;
  for (const auto& axis : coordinate.axes) {
    extents[static_cast<size_t>(axis.dim)] = axis.end - axis.start;
  }
  extents.erase(
      std::remove_if(extents.begin(), extents.end(), [](int64_t extent) { return extent == 1; }), extents.end());
  return extents;
}

bool coordinate_selects_single_element(const RepresentationTensorSpec& spec, const TensorCoordinateSpec& coordinate) {
  if (spec.shape.empty()) {
    return coordinate.axes.empty();
  }
  if (coordinate.axes.size() != spec.shape.size()) {
    return false;
  }
  std::vector<bool> seen(spec.shape.size(), false);
  for (const auto& axis : coordinate.axes) {
    if (axis.dim < 0 || axis.dim >= static_cast<int32_t>(spec.shape.size())) {
      return false;
    }
    if (seen[static_cast<size_t>(axis.dim)]) {
      return false;
    }
    seen[static_cast<size_t>(axis.dim)] = true;
    if (axis.end != axis.start + 1) {
      return false;
    }
  }
  return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

absl::StatusOr<std::vector<ByteSpan>> build_tensor_byte_spans(
    const RepresentationTensorSpec& spec,
    const TensorCoordinateSpec& coordinate,
    bool include_storage_offset) {
  if (spec.element_size == 0 && spec.logical_length == 0) {
    return absl::InvalidArgumentError("tensor requires non-zero element_size or logical_length");
  }
  const uint64_t element_bytes = spec.element_size == 0 ? spec.logical_length : spec.element_size;
  const uint64_t base_offset = spec.logical_offset + (include_storage_offset ? spec.storage_offset * element_bytes : 0);
  if (coordinate.selects_scalar || (spec.shape.empty() && coordinate.axes.empty())) {
    return std::vector<ByteSpan>{{.offset = base_offset, .length = element_bytes}};
  }
  if (!is_contiguous(spec.shape, spec.stride)) {
    return absl::InvalidArgumentError("multi-axis realization requires row-major contiguous tensors");
  }
  const size_t rank = spec.shape.size();
  std::vector<int64_t> starts(rank, 0);
  std::vector<int64_t> ends = spec.shape;
  for (const auto& axis : coordinate.axes) {
    starts[static_cast<size_t>(axis.dim)] = axis.start;
    ends[static_cast<size_t>(axis.dim)] = axis.end;
  }
  std::vector<ByteSpan> spans;
  auto emit_spans = [&](auto&& self, size_t dim, uint64_t element_offset) -> absl::Status {
    if (dim == rank) {
      spans.push_back(ByteSpan{.offset = base_offset + element_offset * element_bytes, .length = element_bytes});
      return absl::OkStatus();
    }
    bool later_full = true;
    for (size_t next = dim + 1; next < rank; ++next) {
      if (starts[next] != 0 || ends[next] != spec.shape[next]) {
        later_full = false;
        break;
      }
    }
    if (later_full) {
      auto tail_or = product_dims(absl::Span<const int64_t>(spec.shape).subspan(dim + 1));
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
  auto status = emit_spans(emit_spans, 0, 0);
  if (!status.ok()) {
    return status;
  }
  if (spans.empty()) {
    return absl::InvalidArgumentError("coordinate produced no byte spans");
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

absl::StatusOr<std::vector<tensorcast::store::loader::ByteRangeSegment>> build_copy_segments_from_coordinates(
    const RepresentationTensorSpec& source_spec,
    const TensorCoordinateSpec& source_coordinate,
    const RepresentationTensorSpec& dst_spec,
    const TensorCoordinateSpec& destination_coordinate) {
  auto source_spans_or = build_tensor_byte_spans(source_spec, source_coordinate, /*include_storage_offset=*/true);
  if (!source_spans_or.ok()) {
    return source_spans_or.status();
  }
  auto destination_spans_or =
      build_tensor_byte_spans(dst_spec, destination_coordinate, /*include_storage_offset=*/false);
  if (!destination_spans_or.ok()) {
    return destination_spans_or.status();
  }
  uint64_t total_source_bytes = 0;
  for (const auto& span : *source_spans_or) {
    total_source_bytes += span.length;
  }
  uint64_t total_destination_bytes = 0;
  for (const auto& span : *destination_spans_or) {
    total_destination_bytes += span.length;
  }
  if (total_source_bytes != total_destination_bytes) {
    return absl::InvalidArgumentError("copy coordinate byte sizes do not match");
  }
  std::vector<tensorcast::store::loader::ByteRangeSegment> segments;
  segments.reserve(source_spans_or->size() + destination_spans_or->size());
  size_t source_idx = 0;
  size_t destination_idx = 0;
  uint64_t source_cursor = source_spans_or->front().offset;
  uint64_t destination_cursor = destination_spans_or->front().offset;
  uint64_t source_remaining = source_spans_or->front().length;
  uint64_t destination_remaining = destination_spans_or->front().length;
  while (source_idx < source_spans_or->size() && destination_idx < destination_spans_or->size()) {
    const uint64_t take = std::min<uint64_t>(source_remaining, destination_remaining);
    segments.push_back(
        tensorcast::store::loader::ByteRangeSegment{
            .kind = tensorcast::store::loader::ByteRangeSegment::Kind::kData,
            .dst_offset = destination_cursor,
            .length = take,
            .src_offset = source_cursor,
            .source_index = 0,
        });
    source_cursor += take;
    destination_cursor += take;
    source_remaining -= take;
    destination_remaining -= take;
    if (source_remaining == 0) {
      ++source_idx;
      if (source_idx < source_spans_or->size()) {
        source_cursor = (*source_spans_or)[source_idx].offset;
        source_remaining = (*source_spans_or)[source_idx].length;
      }
    }
    if (destination_remaining == 0) {
      ++destination_idx;
      if (destination_idx < destination_spans_or->size()) {
        destination_cursor = (*destination_spans_or)[destination_idx].offset;
        destination_remaining = (*destination_spans_or)[destination_idx].length;
      }
    }
  }
  if (source_idx != source_spans_or->size() || destination_idx != destination_spans_or->size()) {
    return absl::InternalError("copy coordinate segment construction ended early");
  }
  return segments;
}

absl::Status normalize_copy_realization_entry(
    int idx,
    const v2::BindingRealizationEntry& entry,
    const CanonicalIndexEntry& src_entry,
    const RepresentationTensorSpec& dst_spec,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows,
    RealizationManualEntry& out) {
  if (entry.source_name().empty()) {
    return absl::InvalidArgumentError(std::format("realization entry {} copy requires source_name", idx));
  }
  const auto source_spec = to_source_tensor_spec(entry.source_name(), src_entry);
  if (src_entry.dtype != dst_spec.dtype) {
    return absl::InvalidArgumentError(
        std::format(
            "realization entry {} dtype mismatch: {}({}) -> {}({})",
            idx,
            entry.source_name(),
            src_entry.dtype,
            entry.dst_name(),
            dst_spec.dtype));
  }
  if (!is_contiguous(src_entry.shape, src_entry.stride)) {
    return absl::InvalidArgumentError(
        std::format("realization entry {} source '{}' must be contiguous", idx, entry.source_name()));
  }
  if (!is_contiguous(dst_spec.shape, dst_spec.stride)) {
    return absl::InvalidArgumentError(
        std::format("realization entry {} target '{}' must be contiguous", idx, entry.dst_name()));
  }
  auto source_coordinate_or = build_coordinate_from_ranges(
      "realization",
      idx,
      entry.source_name(),
      source_spec,
      entry.source_ranges(),
      [&]() -> std::optional<ViewNarrowSpec> {
        auto it = view_narrows.find(entry.source_name());
        if (it == view_narrows.end()) {
          return std::nullopt;
        }
        return it->second;
      }(),
      /*require_view_axis=*/true);
  if (!source_coordinate_or.ok()) {
    return source_coordinate_or.status();
  }
  auto destination_coordinate_or = build_coordinate_from_ranges(
      "realization",
      idx,
      entry.dst_name(),
      dst_spec,
      entry.dst_ranges(),
      std::nullopt,
      /*require_view_axis=*/false);
  if (!destination_coordinate_or.ok()) {
    return destination_coordinate_or.status();
  }
  if (effective_coordinate_shape(source_spec, *source_coordinate_or) !=
      effective_coordinate_shape(dst_spec, *destination_coordinate_or)) {
    return absl::InvalidArgumentError(std::format("realization entry {} shape mismatch for {}", idx, entry.dst_name()));
  }

  out = RealizationManualEntry{
      .proto = &entry,
      .op_kind = (entry.source_ranges_size() == 0 && entry.dst_ranges_size() == 0) ? BindingOpKind::kExactCopy
                                                                                   : BindingOpKind::kSliceCopy,
      .source_coordinate = std::move(*source_coordinate_or),
      .destination_coordinate = std::move(*destination_coordinate_or),
      .has_full_destination = entry.dst_ranges_size() == 0,
  };
  return absl::OkStatus();
}

absl::Status validate_group_destination_coverage(
    std::string_view dst_name,
    const TensorLayoutSpec& dst_spec,
    const std::vector<RealizationManualEntry>& entries) {
  if (entries.empty()) {
    return absl::OkStatus();
  }
  const auto full_dst_spec = to_dst_tensor_spec(std::string(dst_name), dst_spec, /*base_offset=*/0);
  auto full_spans_or = build_tensor_byte_spans(full_dst_spec, TensorCoordinateSpec{}, /*include_storage_offset=*/false);
  if (!full_spans_or.ok()) {
    return full_spans_or.status();
  }
  if (full_spans_or->size() != 1) {
    return absl::InvalidArgumentError(
        std::format("realization destination '{}' must lower to one contiguous coverage span", dst_name));
  }
  std::vector<std::pair<uint64_t, uint64_t>> intervals;
  for (const auto& entry : entries) {
    auto spans_or =
        build_tensor_byte_spans(full_dst_spec, entry.destination_coordinate, /*include_storage_offset=*/false);
    if (!spans_or.ok()) {
      return spans_or.status();
    }
    for (const auto& span : *spans_or) {
      intervals.emplace_back(span.offset, span.offset + span.length);
    }
  }
  std::sort(intervals.begin(), intervals.end());
  uint64_t cursor = full_spans_or->front().offset;
  const uint64_t full_end = full_spans_or->front().offset + full_spans_or->front().length;
  for (const auto& interval : intervals) {
    if (interval.first < cursor) {
      return absl::InvalidArgumentError(
          std::format("realization entries for '{}' have overlapping destination ranges", dst_name));
    }
    if (interval.first != cursor) {
      return absl::InvalidArgumentError(
          std::format("realization entries for '{}' have gaps in destination coverage", dst_name));
    }
    cursor = interval.second;
  }
  if (cursor != full_end) {
    return absl::InvalidArgumentError(
        std::format("realization entries for '{}' do not cover the full destination range", dst_name));
  }
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<BuildRepresentationTransformResult> build_representation_transform_contract(
    const v2::BindingRealizationPlan& realization_plan,
    const materialization_layout::CanonicalIndexTable& source_table,
    const materialization_layout::CanonicalIndexTable& canonical_source_table,
    const absl::flat_hash_map<std::string, TensorLayoutSpec>& dst_specs,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows,
    const tensorcast::common::v1::ByteSpaceRef& source_byte_space,
    std::string_view representation_family,
    bool compute_identity_hashes) {
  const auto total_begin = ProfileClock::now();
  BuildRepresentationTransformResult result;
  result.generic_fallback_map.total_bytes = 0;
  result.generic_fallback_map.num_sources = 1;
  result.collective_lowered_map.total_bytes = 0;
  result.collective_lowered_map.num_sources = 1;
  result.transform_contract.source_byte_space = source_byte_space;
  result.transform_contract.target_representation.family = std::string(representation_family);
  result.transform_contract.target_representation.realization_kind = RealizationKind::kEphemeralIntoTarget;

  v2::CopyPlan copy_plan;
  copy_plan.set_version(1);
  absl::flat_hash_map<std::string, std::vector<RealizationManualEntry>> manual_entries_by_dst;
  manual_entries_by_dst.reserve(realization_plan.entries_size());
  absl::flat_hash_set<std::string> manual_dst_names;
  manual_dst_names.reserve(realization_plan.entries_size());

  const auto manual_scan_begin = ProfileClock::now();
  for (const auto& entry : realization_plan.entries()) {
    if (entry.op_kind() != v2::BINDING_REALIZATION_OP_KIND_COPY || entry.source_ranges_size() > 1 ||
        entry.dst_ranges_size() > 1) {
      manual_dst_names.insert(entry.dst_name());
    }
  }
  const auto manual_scan_end = ProfileClock::now();

  const auto lower_copy_plan_begin = ProfileClock::now();
  for (int idx = 0; idx < realization_plan.entries_size(); ++idx) {
    const auto& entry = realization_plan.entries(idx);
    if (entry.dst_name().empty()) {
      return absl::InvalidArgumentError(std::format("realization entry {} missing dst_name", idx));
    }
    const auto dst_it = dst_specs.find(entry.dst_name());
    if (dst_it == dst_specs.end()) {
      return absl::InvalidArgumentError(
          std::format("realization entry {} references unknown destination tensor '{}'", idx, entry.dst_name()));
    }
    const auto dst_base_it = dst_base_offsets.find(entry.dst_name());
    if (dst_base_it == dst_base_offsets.end()) {
      return absl::InvalidArgumentError(
          std::format(
              "realization entry {} destination tensor '{}' is missing target layout metadata", idx, entry.dst_name()));
    }
    const auto dst_spec = to_dst_tensor_spec(entry.dst_name(), dst_it->second, dst_base_it->second);

    switch (entry.op_kind()) {
      case v2::BINDING_REALIZATION_OP_KIND_COPY: {
        if (entry.source_name().empty()) {
          return absl::InvalidArgumentError(std::format("realization entry {} copy requires source_name", idx));
        }
        if (manual_dst_names.contains(entry.dst_name())) {
          const auto src_it = source_table.entries.find(entry.source_name());
          if (src_it == source_table.entries.end()) {
            return absl::InvalidArgumentError(
                std::format("realization entry {} references unknown source tensor '{}'", idx, entry.source_name()));
          }
          RealizationManualEntry manual_entry;
          auto status =
              normalize_copy_realization_entry(idx, entry, src_it->second, dst_spec, view_narrows, manual_entry);
          if (!status.ok()) {
            return status;
          }
          manual_entries_by_dst[entry.dst_name()].push_back(std::move(manual_entry));
          continue;
        }
        auto* copy_entry = copy_plan.add_entries();
        copy_entry->set_ckpt_name(entry.source_name());
        copy_entry->set_dst_name(entry.dst_name());
        if (entry.source_ranges_size() == 1) {
          copy_range(entry.source_ranges(0), copy_entry->mutable_ckpt_range());
        } else if (entry.source_ranges_size() > 1) {
          const auto src_it = source_table.entries.find(entry.source_name());
          if (src_it == source_table.entries.end()) {
            return absl::InvalidArgumentError(
                std::format("realization entry {} references unknown source tensor '{}'", idx, entry.source_name()));
          }
          RealizationManualEntry manual_entry;
          auto status =
              normalize_copy_realization_entry(idx, entry, src_it->second, dst_spec, view_narrows, manual_entry);
          if (!status.ok()) {
            return status;
          }
          copy_plan.mutable_entries()->RemoveLast();
          manual_entries_by_dst[entry.dst_name()].push_back(std::move(manual_entry));
          continue;
        }
        if (entry.dst_ranges_size() == 1) {
          copy_range(entry.dst_ranges(0), copy_entry->mutable_dst_range());
        } else if (entry.dst_ranges_size() > 1) {
          const auto src_it = source_table.entries.find(entry.source_name());
          if (src_it == source_table.entries.end()) {
            return absl::InvalidArgumentError(
                std::format("realization entry {} references unknown source tensor '{}'", idx, entry.source_name()));
          }
          RealizationManualEntry manual_entry;
          auto status =
              normalize_copy_realization_entry(idx, entry, src_it->second, dst_spec, view_narrows, manual_entry);
          if (!status.ok()) {
            return status;
          }
          copy_plan.mutable_entries()->RemoveLast();
          manual_entries_by_dst[entry.dst_name()].push_back(std::move(manual_entry));
        }
        break;
      }
      case v2::BINDING_REALIZATION_OP_KIND_CONST_FILL: {
        if (entry.fill_value().empty()) {
          return absl::InvalidArgumentError(std::format("realization entry {} const fill requires fill_value", idx));
        }
        RealizationManualEntry manual_entry;
        manual_entry.proto = &entry;
        manual_entry.op_kind = BindingOpKind::kConstFill;
        auto destination_coordinate_or = build_coordinate_from_ranges(
            "realization",
            idx,
            entry.dst_name(),
            dst_spec,
            entry.dst_ranges(),
            std::nullopt,
            /*require_view_axis=*/false);
        if (!destination_coordinate_or.ok()) {
          return destination_coordinate_or.status();
        }
        manual_entry.destination_coordinate = std::move(*destination_coordinate_or);
        manual_entry.has_full_destination = entry.dst_ranges_size() == 0;
        manual_entries_by_dst[entry.dst_name()].push_back(std::move(manual_entry));
        break;
      }
      case v2::BINDING_REALIZATION_OP_KIND_SCALAR_FILL: {
        if (entry.source_name().empty()) {
          return absl::InvalidArgumentError(std::format("realization entry {} scalar fill requires source_name", idx));
        }
        const auto src_it = source_table.entries.find(entry.source_name());
        if (src_it == source_table.entries.end()) {
          return absl::InvalidArgumentError(
              std::format("realization entry {} references unknown source tensor '{}'", idx, entry.source_name()));
        }
        RealizationManualEntry manual_entry;
        manual_entry.proto = &entry;
        manual_entry.op_kind = BindingOpKind::kScalarFromSource;
        auto destination_coordinate_or = build_coordinate_from_ranges(
            "realization",
            idx,
            entry.dst_name(),
            dst_spec,
            entry.dst_ranges(),
            std::nullopt,
            /*require_view_axis=*/false);
        if (!destination_coordinate_or.ok()) {
          return destination_coordinate_or.status();
        }
        manual_entry.destination_coordinate = std::move(*destination_coordinate_or);
        manual_entry.has_full_destination = entry.dst_ranges_size() == 0;
        auto source_coordinate_or = build_coordinate_from_ranges(
            "realization",
            idx,
            entry.source_name(),
            to_source_tensor_spec(entry.source_name(), src_it->second),
            entry.source_ranges(),
            [&]() -> std::optional<ViewNarrowSpec> {
              auto it = view_narrows.find(entry.source_name());
              if (it == view_narrows.end()) {
                return std::nullopt;
              }
              return it->second;
            }(),
            /*require_view_axis=*/true);
        if (!source_coordinate_or.ok()) {
          return source_coordinate_or.status();
        }
        if (!coordinate_selects_single_element(
                to_source_tensor_spec(entry.source_name(), src_it->second), *source_coordinate_or)) {
          return absl::InvalidArgumentError(
              std::format(
                  "realization entry {} source '{}' scalar fill requires one source range per dimension",
                  idx,
                  entry.source_name()));
        }
        manual_entry.source_coordinate = std::move(*source_coordinate_or);
        manual_entries_by_dst[entry.dst_name()].push_back(std::move(manual_entry));
        break;
      }
      case v2::BINDING_REALIZATION_OP_KIND_UNSPECIFIED:
      default:
        return absl::InvalidArgumentError(std::format("realization entry {} has unsupported op_kind", idx));
    }
  }
  const auto lower_copy_plan_end = ProfileClock::now();
  size_t manual_entry_count = 0;
  for (const auto& [_, entries] : manual_entries_by_dst) {
    manual_entry_count += entries.size();
  }

  const auto copy_contract_begin = ProfileClock::now();
  if (copy_plan.entries_size() > 0) {
    auto copy_result_or = build_representation_transform_contract(
        copy_plan,
        source_table,
        canonical_source_table,
        dst_specs,
        dst_base_offsets,
        view_narrows,
        source_byte_space,
        representation_family,
        /*compute_identity_hashes=*/false);
    if (!copy_result_or.ok()) {
      return copy_result_or.status();
    }
    result = std::move(*copy_result_or);
  }
  const auto copy_contract_end = ProfileClock::now();

  const auto manual_bindings_begin = ProfileClock::now();
  absl::flat_hash_map<std::string, std::vector<ByteRangeSegment>> manual_segments_by_dst;
  manual_segments_by_dst.reserve(manual_entries_by_dst.size());
  for (const auto& [dst_name, entries] : manual_entries_by_dst) {
    const auto dst_spec_it = dst_specs.find(dst_name);
    const auto dst_base_it = dst_base_offsets.find(dst_name);
    if (dst_spec_it == dst_specs.end() || dst_base_it == dst_base_offsets.end()) {
      return absl::InvalidArgumentError("destination tensor metadata missing for realization plan");
    }
    if (auto status = validate_group_destination_coverage(dst_name, dst_spec_it->second, entries); !status.ok()) {
      return status;
    }
    const auto full_dst_spec = to_dst_tensor_spec(dst_name, dst_spec_it->second, dst_base_it->second);
    for (const auto& entry : entries) {
      RepresentationTensorBinding binding;
      binding.dst_name = dst_name;
      binding.dst_spec = full_dst_spec;
      binding.op_kind = entry.op_kind;
      binding.coverage_kind = CoverageKind::kExact;
      if (entry.op_kind == BindingOpKind::kConstFill) {
        binding.fill_rule = tensorcast::store::materialization::contracts::FillRule{
            .constant_value =
                std::vector<std::uint8_t>(entry.proto->fill_value().begin(), entry.proto->fill_value().end()),
            .destination_range = entry.destination_coordinate,
        };
      } else {
        const auto canonical_src_it = canonical_source_table.entries.find(entry.proto->source_name());
        if (canonical_src_it == canonical_source_table.entries.end()) {
          return absl::InvalidArgumentError(
              std::format(
                  "realization entry references unknown canonical source tensor '{}'", entry.proto->source_name()));
        }
        binding.sources.push_back(
            SourceFragment{
                .source_spec = to_source_tensor_spec(entry.proto->source_name(), canonical_src_it->second),
                .source_range = entry.source_coordinate,
                .destination_range = entry.destination_coordinate,
                .role = SourceFragmentRole::kDefault,
            });
      }
      result.transform_contract.tensor_bindings.push_back(std::move(binding));
      if (entry.op_kind == BindingOpKind::kExactCopy || entry.op_kind == BindingOpKind::kSliceCopy) {
        const auto source_it = source_table.entries.find(entry.proto->source_name());
        if (source_it == source_table.entries.end()) {
          return absl::InvalidArgumentError(
              std::format("realization entry references unknown source tensor '{}'", entry.proto->source_name()));
        }
        auto segments_or = build_copy_segments_from_coordinates(
            to_source_tensor_spec(entry.proto->source_name(), source_it->second),
            entry.source_coordinate,
            full_dst_spec,
            entry.destination_coordinate);
        if (!segments_or.ok()) {
          return segments_or.status();
        }
        uint64_t copied_bytes = 0;
        auto& dst_segments = manual_segments_by_dst[dst_name];
        for (const auto& segment : *segments_or) {
          copied_bytes += segment.length;
          dst_segments.push_back(segment);
        }
        result.total_bytes_copied += copied_bytes;
        result.lowering_stats.collective_candidates += 1;
        result.lowering_stats.collective_bytes += copied_bytes;
      }
    }
  }
  tensorcast::store::loader::ByteRangeMap manual_data_map;
  manual_data_map.total_bytes = result.generic_fallback_map.total_bytes;
  manual_data_map.num_sources = 1;
  manual_data_map.segments = flatten_segments_by_dst_order(manual_segments_by_dst, dst_base_offsets);
  result.collective_lowered_map =
      merge_ordered_byte_range_maps(std::move(result.collective_lowered_map), manual_data_map);
  result.generic_fallback_map =
      merge_ordered_byte_range_maps(std::move(result.generic_fallback_map), std::move(manual_data_map));
  const auto manual_bindings_end = ProfileClock::now();

  const auto normalize_begin = ProfileClock::now();
  auto normalized_contract_or =
      tensorcast::store::materialization::contracts::normalize_representation_transform_contract(
          std::move(result.transform_contract),
          tensorcast::store::materialization::contracts::NormalizeRepresentationTransformContractOptions{
              .compute_identity_hashes = compute_identity_hashes,
          });
  if (!normalized_contract_or.ok()) {
    return normalized_contract_or.status();
  }
  result.transform_contract = std::move(*normalized_contract_or);
  const auto normalize_end = ProfileClock::now();
  LOG(INFO) << "tc_profile representation_realization_plan timings"
            << " total_sec=" << seconds_between(total_begin, normalize_end)
            << " manual_scan_sec=" << seconds_between(manual_scan_begin, manual_scan_end)
            << " lower_copy_plan_sec=" << seconds_between(lower_copy_plan_begin, lower_copy_plan_end)
            << " copy_contract_sec=" << seconds_between(copy_contract_begin, copy_contract_end)
            << " manual_bindings_sec=" << seconds_between(manual_bindings_begin, manual_bindings_end)
            << " final_normalize_sec=" << seconds_between(normalize_begin, normalize_end)
            << " compute_identity_hashes=" << (compute_identity_hashes ? 1 : 0)
            << " entries=" << realization_plan.entries_size() << " copy_plan_entries=" << copy_plan.entries_size()
            << " manual_dst_count=" << manual_entries_by_dst.size() << " manual_entry_count=" << manual_entry_count
            << " tensor_bindings=" << result.transform_contract.tensor_bindings.size()
            << " collective_segments=" << result.collective_lowered_map.segments.size()
            << " fallback_segments=" << result.generic_fallback_map.segments.size()
            << " total_bytes_copied=" << result.total_bytes_copied;
  return result;
}

} // namespace tensorcast::daemon::representation_transform_builder
