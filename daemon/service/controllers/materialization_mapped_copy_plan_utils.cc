// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_mapped_copy_plan_utils.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"

namespace tensorcast::daemon::materialization_mapped_copy_plan {

namespace {

using materialization_layout::CanonicalIndexEntry;
using materialization_layout::dtype_element_size;
using materialization_layout::product_dims;

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

std::vector<int64_t> compute_compact_stride(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> stride(shape.size());
  int64_t acc = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    stride[static_cast<size_t>(i)] = acc;
    acc *= shape[static_cast<size_t>(i)];
  }
  return stride;
}

RangeSpec build_range(const v2::CopyPlanRange& range) {
  return RangeSpec{
      .has_range = true,
      .dim = static_cast<int32_t>(range.dim()),
      .start = range.start(),
      .end = range.end(),
  };
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
    const MappedTensorSpec& dst_spec,
    uint64_t dst_base_offset,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows,
    RangeSpec src_range,
    RangeSpec dst_range,
    absl::flat_hash_map<std::string, int32_t>& dst_dim_by_name,
    absl::flat_hash_map<std::string, std::vector<std::tuple<int64_t, int64_t, int>>>& dst_intervals,
    std::vector<store::loader::ByteRangeSegment>& out_segments,
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
    auto st = normalize_range(src_shape, src_range, "source");
    if (!st.ok()) {
      return st;
    }
  }
  if (!dst_shape.empty()) {
    auto st = normalize_range(dst_shape, dst_range, "target");
    if (!st.ok()) {
      return st;
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
  auto st = record_interval(entry.dst_name(), record_range);
  if (!st.ok()) {
    return st;
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
    store::loader::ByteRangeSegment seg;
    seg.kind = store::loader::ByteRangeSegment::Kind::kData;
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
      store::loader::ByteRangeSegment seg;
      seg.kind = store::loader::ByteRangeSegment::Kind::kData;
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

} // namespace

bool is_contiguous(const std::vector<int64_t>& shape, const std::vector<int64_t>& stride) {
  if (shape.empty()) {
    return stride.empty();
  }
  return stride == compute_compact_stride(shape);
}

absl::StatusOr<uint64_t> slice_bytes_along_dim0(
    const std::vector<int64_t>& shape,
    uint64_t elem_size,
    int64_t start,
    int64_t end) {
  if (end <= start) {
    return absl::InvalidArgumentError("slice_bytes_along_dim0 requires end > start");
  }
  if (shape.empty()) {
    return elem_size;
  }
  auto tail_or = product_dims(absl::Span<const int64_t>(shape).subspan(1));
  if (!tail_or.ok()) {
    return tail_or.status();
  }
  return static_cast<uint64_t>(end - start) * (*tail_or) * elem_size;
}

absl::StatusOr<std::optional<ConcatJobCandidate>> maybe_build_concat_dim0_candidate(
    std::string_view dst_name,
    const std::vector<CandidateEntry>& entries,
    const materialization_layout::CanonicalIndexTable& source_table,
    const MappedTensorSpec& dst_spec,
    uint64_t dst_base_offset) {
  if (entries.size() < 2 || !is_contiguous(dst_spec.shape, dst_spec.stride)) {
    return std::optional<ConcatJobCandidate>{};
  }
  auto dst_inner_or = product_dims(absl::Span<const int64_t>(dst_spec.shape).subspan(1));
  if (!dst_inner_or.ok()) {
    return dst_inner_or.status();
  }
  const uint64_t dst_inner = dst_spec.shape.size() <= 1 ? 1 : *dst_inner_or;

  struct SourceGroup {
    std::string src_name;
    const materialization_layout::CanonicalIndexEntry* src_entry{nullptr};
    std::vector<CandidateEntry> entries;
    uint64_t prefix_count{0};
    int64_t first_src_start{0};
    int64_t first_src_end{0};
    int64_t block_offset_rows{0};
    int64_t block_rows{0};
    int64_t dst_stride_rows{0};
  };

  absl::flat_hash_map<std::string, SourceGroup> groups;
  groups.reserve(entries.size());
  for (const auto& entry : entries) {
    if (entry.dim != 0 || !entry.src_range.has_range || !entry.dst_range.has_range) {
      return std::optional<ConcatJobCandidate>{};
    }
    auto [it, inserted] = groups.try_emplace(
        entry.src_name,
        SourceGroup{
            .src_name = entry.src_name,
            .src_entry = &source_table.entries.at(entry.src_name),
        });
    it->second.entries.push_back(entry);
    if (!is_contiguous(it->second.src_entry->shape, it->second.src_entry->stride)) {
      return std::optional<ConcatJobCandidate>{};
    }
  }
  if (groups.size() < 2) {
    return std::optional<ConcatJobCandidate>{};
  }

  std::optional<uint64_t> expected_prefix_count;
  std::optional<int64_t> expected_dst_stride_rows;
  std::vector<SourceGroup*> ordered_groups;
  ordered_groups.reserve(groups.size());
  for (auto& [_, group] : groups) {
    std::sort(group.entries.begin(), group.entries.end(), [](const CandidateEntry& a, const CandidateEntry& b) {
      return a.dst_range.start < b.dst_range.start;
    });
    const auto& first = group.entries.front();
    group.prefix_count = group.entries.size();
    group.first_src_start = first.src_range.start;
    group.first_src_end = first.src_range.end;
    group.block_rows = first.dst_range.end - first.dst_range.start;
    if (group.block_rows <= 0) {
      return std::optional<ConcatJobCandidate>{};
    }
    for (size_t idx = 1; idx < group.entries.size(); ++idx) {
      const auto& cur = group.entries[idx];
      if ((cur.src_range.end - cur.src_range.start) != (first.src_range.end - first.src_range.start) ||
          (cur.dst_range.end - cur.dst_range.start) != group.block_rows) {
        return std::optional<ConcatJobCandidate>{};
      }
    }
    if (group.prefix_count > 1) {
      group.dst_stride_rows = group.entries[1].dst_range.start - group.entries[0].dst_range.start;
      if (group.dst_stride_rows < group.block_rows) {
        return std::optional<ConcatJobCandidate>{};
      }
      for (size_t idx = 0; idx < group.entries.size(); ++idx) {
        const auto& cur = group.entries[idx];
        if (cur.src_range.start !=
            group.first_src_start + static_cast<int64_t>(idx) * (group.first_src_end - group.first_src_start)) {
          return std::optional<ConcatJobCandidate>{};
        }
        if (cur.dst_range.start !=
            group.entries[0].dst_range.start + static_cast<int64_t>(idx) * group.dst_stride_rows) {
          return std::optional<ConcatJobCandidate>{};
        }
      }
      group.block_offset_rows = group.entries[0].dst_range.start;
      if (group.block_offset_rows >= group.dst_stride_rows) {
        return std::optional<ConcatJobCandidate>{};
      }
      if (!expected_prefix_count.has_value()) {
        expected_prefix_count = group.prefix_count;
      } else if (*expected_prefix_count != group.prefix_count) {
        return std::optional<ConcatJobCandidate>{};
      }
      if (!expected_dst_stride_rows.has_value()) {
        expected_dst_stride_rows = group.dst_stride_rows;
      } else if (*expected_dst_stride_rows != group.dst_stride_rows) {
        return std::optional<ConcatJobCandidate>{};
      }
    } else {
      group.block_offset_rows = group.entries[0].dst_range.start;
    }
    ordered_groups.push_back(&group);
  }

  std::sort(ordered_groups.begin(), ordered_groups.end(), [](const SourceGroup* a, const SourceGroup* b) {
    return a->block_offset_rows < b->block_offset_rows;
  });

  uint64_t prefix_count = expected_prefix_count.value_or(1);
  int64_t dst_stride_rows = expected_dst_stride_rows.value_or(0);
  if (prefix_count == 1) {
    int64_t max_end = 0;
    int64_t cursor = 0;
    for (const SourceGroup* group : ordered_groups) {
      if (group->block_offset_rows != cursor) {
        return std::optional<ConcatJobCandidate>{};
      }
      cursor += group->block_rows;
      max_end = std::max(max_end, group->block_offset_rows + group->block_rows);
    }
    dst_stride_rows = max_end;
  } else {
    int64_t cursor = 0;
    for (const SourceGroup* group : ordered_groups) {
      if (group->block_offset_rows != cursor) {
        return std::optional<ConcatJobCandidate>{};
      }
      cursor += group->block_rows;
    }
    if (cursor != dst_stride_rows) {
      return std::optional<ConcatJobCandidate>{};
    }
  }

  ConcatJobCandidate candidate{
      .dst_name = std::string(dst_name),
      .dst_shape = dst_spec.shape,
      .dst_stride = dst_spec.stride,
      .dtype = dst_spec.dtype,
      .dst_base_offset = dst_base_offset,
      .dst_size_bytes = dst_spec.logical_length,
      .element_size = dst_spec.element_size,
      .prefix_count = prefix_count,
      .dst_block_stride_bytes = static_cast<uint64_t>(dst_stride_rows) * dst_inner * dst_spec.element_size,
  };
  candidate.sources.reserve(ordered_groups.size());
  for (const SourceGroup* group : ordered_groups) {
    auto src_block_bytes_or = slice_bytes_along_dim0(
        group->src_entry->shape, dst_spec.element_size, group->first_src_start, group->first_src_end);
    if (!src_block_bytes_or.ok()) {
      return src_block_bytes_or.status();
    }
    const uint64_t dst_block_bytes = static_cast<uint64_t>(group->block_rows) * dst_inner * dst_spec.element_size;
    candidate.sources.push_back(
        ConcatSourceFragmentCandidate{
            .src_name = group->src_name,
            .src_shape = group->src_entry->shape,
            .src_stride = group->src_entry->stride,
            .dtype = group->src_entry->dtype,
            .src_logical_offset = group->src_entry->logical_offset,
            .src_storage_offset = group->src_entry->storage_offset,
            .src_size_bytes = group->src_entry->logical_length,
            .element_size = dst_spec.element_size,
            .dim = 0,
            .src_start = group->first_src_start,
            .src_end = group->first_src_end,
            .prefix_count = group->prefix_count,
            .dst_block_offset_bytes =
                static_cast<uint64_t>(group->block_offset_rows) * dst_inner * dst_spec.element_size,
            .dst_block_stride_bytes = static_cast<uint64_t>(dst_stride_rows) * dst_inner * dst_spec.element_size,
            .dst_block_bytes = dst_block_bytes,
        });
  }
  return std::optional<ConcatJobCandidate>(std::move(candidate));
}

absl::StatusOr<BuildCopyPlanResult> build_copy_plan(
    const v2::CopyPlan& copy_plan,
    const materialization_layout::CanonicalIndexTable& source_table,
    const materialization_layout::CanonicalIndexTable& canonical_source_table,
    const absl::flat_hash_map<std::string, MappedTensorSpec>& dst_specs,
    const absl::flat_hash_map<std::string, uint64_t>& dst_base_offsets,
    const absl::flat_hash_map<std::string, ViewNarrowSpec>& view_narrows) {
  BuildCopyPlanResult result;
  result.map.total_bytes = 0;
  result.map.num_sources = 1;

  std::vector<store::loader::ByteRangeSegment> mapped_segments;
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

    auto st = validate_and_build_segments(
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
    if (!st.ok()) {
      return st;
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
    auto dim_it = dst_dim_by_name.find(name);
    if (dim_it != dst_dim_by_name.end()) {
      dim = dim_it->second;
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
    std::sort(
        sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
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

  result.map.segments.reserve(mapped_segments.size());
  for (const auto& segment : mapped_segments) {
    result.map.segments.push_back(segment);
  }

  result.compatibility_stats.total_dst_tensors = candidate_entries_by_dst.size();
  result.tensor_job_candidates.reserve(dst_specs.size());
  for (const auto& [dst_name, entries] : candidate_entries_by_dst) {
    if (entries.empty()) {
      continue;
    }
    const auto& dst_spec = dst_specs.at(dst_name);
    const auto dst_base_offset = dst_base_offsets.at(dst_name);
    const uint64_t dst_bytes = dst_spec.logical_length;
    const auto& first = entries.front();
    bool compatible = true;
    for (const auto& entry : entries) {
      if (entry.src_name != first.src_name || entry.dim != first.dim) {
        compatible = false;
        break;
      }
    }
    if (!compatible) {
      auto concat_candidate_or =
          maybe_build_concat_dim0_candidate(dst_name, entries, canonical_source_table, dst_spec, dst_base_offset);
      if (!concat_candidate_or.ok()) {
        return concat_candidate_or.status();
      }
      if (concat_candidate_or->has_value()) {
        result.concat_job_candidates.push_back(std::move(**concat_candidate_or));
        result.compatibility_stats.concat_candidates += 1;
        result.compatibility_stats.concat_bytes += dst_bytes;
      } else {
        result.compatibility_stats.rejected_mixed_src_or_dim += 1;
        result.compatibility_stats.rejected_mixed_src_or_dim_bytes += dst_bytes;
      }
      continue;
    }
    auto canonical_src_it = canonical_source_table.entries.find(first.src_name);
    if (canonical_src_it == canonical_source_table.entries.end()) {
      return absl::InvalidArgumentError("copy_plan references unknown canonical source tensor");
    }
    const auto& src_entry = canonical_src_it->second;

    auto sorted = entries;
    std::sort(sorted.begin(), sorted.end(), [](const CandidateEntry& a, const CandidateEntry& b) {
      const int64_t a_start = a.dst_range.has_range ? a.dst_range.start : 0;
      const int64_t b_start = b.dst_range.has_range ? b.dst_range.start : 0;
      return a_start < b_start;
    });

    int64_t merged_src_start = sorted.front().src_range.has_range ? sorted.front().src_range.start : 0;
    int64_t merged_src_end = sorted.front().src_range.has_range
        ? sorted.front().src_range.end
        : (src_entry.shape.empty() ? 1 : src_entry.shape.front());
    int64_t merged_dst_start = sorted.front().dst_range.has_range ? sorted.front().dst_range.start : 0;
    int64_t merged_dst_end = sorted.front().dst_range.has_range ? sorted.front().dst_range.end
                                                                : (dst_spec.shape.empty() ? 1 : dst_spec.shape.front());

    for (size_t i = 1; i < sorted.size(); ++i) {
      const auto& prev = sorted[i - 1];
      const auto& cur = sorted[i];
      const int64_t prev_dst_end = prev.dst_range.has_range ? prev.dst_range.end : merged_dst_end;
      const int64_t cur_dst_start = cur.dst_range.has_range ? cur.dst_range.start : 0;
      const int64_t prev_src_end = prev.src_range.has_range ? prev.src_range.end : merged_src_end;
      const int64_t cur_src_start = cur.src_range.has_range ? cur.src_range.start : 0;
      if (prev_dst_end != cur_dst_start || prev_src_end != cur_src_start) {
        compatible = false;
        break;
      }
      merged_src_end = cur.src_range.has_range ? cur.src_range.end : merged_src_end;
      merged_dst_end = cur.dst_range.has_range ? cur.dst_range.end : merged_dst_end;
    }
    if (!compatible) {
      result.compatibility_stats.rejected_non_contiguous += 1;
      result.compatibility_stats.rejected_non_contiguous_bytes += dst_bytes;
      continue;
    }

    TensorJobDistribution distribution = TensorJobDistribution::kUnknown;
    if (first.dim < 0) {
      distribution = TensorJobDistribution::kReplicated;
    } else if (first.dim == 0) {
      distribution = TensorJobDistribution::kDim0Partitioned;
    } else if (first.dim == 1) {
      distribution = TensorJobDistribution::kDim1Partitioned;
    }
    if (distribution == TensorJobDistribution::kUnknown) {
      result.compatibility_stats.rejected_unsupported_distribution += 1;
      result.compatibility_stats.rejected_unsupported_distribution_bytes += dst_bytes;
      continue;
    }

    result.tensor_job_candidates.push_back(
        TensorJobCandidate{
            .src_name = first.src_name,
            .dst_name = dst_name,
            .distribution = distribution,
            .src_shape = src_entry.shape,
            .src_stride = src_entry.stride,
            .dst_shape = dst_spec.shape,
            .dst_stride = dst_spec.stride,
            .dtype = src_entry.dtype,
            .src_logical_offset = src_entry.logical_offset,
            .src_storage_offset = src_entry.storage_offset,
            .src_size_bytes = src_entry.logical_length,
            .dst_base_offset = dst_base_offset,
            .dst_size_bytes = dst_spec.logical_length,
            .element_size = dst_spec.element_size,
            .dim = first.dim,
            .src_start = merged_src_start,
            .src_end = merged_src_end,
            .dst_start = merged_dst_start,
            .dst_end = merged_dst_end,
        });
    result.compatibility_stats.compatible_candidates += 1;
    result.compatibility_stats.compatible_bytes += dst_bytes;
  }
  return result;
}

} // namespace tensorcast::daemon::materialization_mapped_copy_plan
