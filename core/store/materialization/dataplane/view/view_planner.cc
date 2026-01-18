// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_planner.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::loader {

namespace {

constexpr uint64_t kAlignmentBytes = 8;

struct CanonicalEntry {
  uint64_t offset{0};
  uint64_t size_bytes{0};
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  uint64_t storage_offset{0};
};

absl::Status validate_ops(const TensorViewOps& ops) {
  enum class Mode : uint8_t { kNone, kNarrow, kTranspose };
  Mode mode = Mode::kNone;
  bool seen_narrow = false;
  for (const auto& op : ops.ops) {
    switch (op.kind) {
      case ViewOp::Kind::kNarrow:
        if (mode == Mode::kTranspose) {
          return absl::InvalidArgumentError("mixing transpose and narrow in the same tensor is not supported");
        }
        if (seen_narrow) {
          return absl::InvalidArgumentError("multiple narrow ops per tensor are not supported");
        }
        seen_narrow = true;
        mode = Mode::kNarrow;
        break;
      case ViewOp::Kind::kTranspose:
        if (mode == Mode::kNarrow) {
          return absl::InvalidArgumentError("mixing transpose and narrow in the same tensor is not supported");
        }
        if (op.transpose.dim0 == op.transpose.dim1) {
          return absl::InvalidArgumentError("transpose dims must be different");
        }
        mode = Mode::kTranspose;
        break;
      default:
        return absl::InvalidArgumentError("unsupported view operation kind");
    }
  }
  return absl::OkStatus();
}

absl::Status parse_canonical_index(
    std::string_view canonical_index_json,
    std::map<std::string, CanonicalEntry>* out_entries) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical_index_json must not be empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", e.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }
  std::map<std::string, CanonicalEntry> entries;
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& key = it.key();
    if (!it.value().is_array() || it.value().size() < 6) {
      return absl::InvalidArgumentError(
          "canonical index entry must be array [offset,size,shape,stride,dtype,storage_offset]");
    }
    CanonicalEntry entry;
    entry.offset = it.value()[0].get<uint64_t>();
    entry.size_bytes = it.value()[1].get<uint64_t>();
    entry.shape.clear();
    for (const auto& dim : it.value()[2]) {
      entry.shape.push_back(dim.get<int64_t>());
    }
    entry.stride.clear();
    for (const auto& s : it.value()[3]) {
      entry.stride.push_back(s.get<int64_t>());
    }
    entry.dtype = it.value()[4].get<std::string>();
    entry.storage_offset = it.value()[5].get<uint64_t>();
    entries.emplace(key, std::move(entry));
  }
  *out_entries = std::move(entries);
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> dtype_element_size(std::string_view dtype) {
  static const absl::flat_hash_map<std::string_view, uint64_t> kSizeMap = {
      {"torch.float16", 2},
      {"torch.bfloat16", 2},
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
  auto it = kSizeMap.find(dtype);
  if (it == kSizeMap.end()) {
    return absl::InvalidArgumentError(absl::StrCat("unsupported dtype in canonical index: ", dtype));
  }
  return it->second;
}

uint64_t product(const std::vector<int64_t>& values, size_t begin, size_t end) {
  if (begin >= end) {
    return 1;
  }
  uint64_t acc = 1;
  for (size_t i = begin; i < end; ++i) {
    const int64_t v = values[i];
    if (v <= 0) {
      return 0;
    }
    acc *= static_cast<uint64_t>(v);
  }
  return acc;
}

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

bool is_multiple_of(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value == 0;
  }
  return (value % align) == 0;
}

SelectionPlan::Range make_data_range(uint64_t src, uint64_t dst, uint64_t len) {
  SelectionPlan::Range r;
  r.kind = SelectionPlan::Range::Kind::kData;
  r.src_offset = src;
  r.dst_offset = dst;
  r.length = len;
  return r;
}

SelectionPlan::Range make_pad_range(uint64_t dst, uint64_t len) {
  SelectionPlan::Range r;
  r.kind = SelectionPlan::Range::Kind::kPad;
  r.src_offset = 0;
  r.dst_offset = dst;
  r.length = len;
  return r;
}

void finalize_selection_plan(SelectionPlan* plan) {
  plan->num_ranges = static_cast<uint32_t>(plan->ranges.size());
  plan->total_bytes = 0;
  bool contiguous = true;
  bool segment_aligned = true;
  uint64_t expected_dst = 0;
  size_t data_ranges = 0;
  for (const auto& r : plan->ranges) {
    plan->total_bytes += r.length;
    if (r.dst_offset != expected_dst) {
      contiguous = false;
    }
    expected_dst = r.dst_offset + r.length;
    if (r.kind == SelectionPlan::Range::Kind::kData) {
      ++data_ranges;
      if (!is_multiple_of(r.src_offset, kAlignmentBytes) || !is_multiple_of(r.length, kAlignmentBytes)) {
        segment_aligned = false;
      }
    }
  }
  plan->is_contiguous = contiguous && data_ranges <= 1;
  plan->is_segment_aligned = plan->is_contiguous && segment_aligned;
}

ViewWritePlan build_view_write_plan(const SelectionPlan& selection) {
  ViewWritePlan write_plan;
  write_plan.chunks.reserve(selection.ranges.size());
  for (const auto& range : selection.ranges) {
    if (range.kind != SelectionPlan::Range::Kind::kData) {
      continue;
    }
    ViewWritePlan::Chunk chunk;
    chunk.canonical_offset = range.src_offset;
    chunk.view_offset = range.dst_offset;
    chunk.length = range.length;
    chunk.segment_aligned =
        is_multiple_of(range.src_offset, kAlignmentBytes) && is_multiple_of(range.length, kAlignmentBytes);
    write_plan.chunks.push_back(std::move(chunk));
  }
  return write_plan;
}

absl::StatusOr<std::vector<int64_t>> compute_inverse_permutation(
    const std::vector<int64_t>& permutation,
    std::string_view tensor_name) {
  const size_t n = permutation.size();
  std::vector<int64_t> inverse(n, 0);
  std::vector<bool> seen(n, false);
  for (size_t idx = 0; idx < n; ++idx) {
    const int64_t value = permutation[idx];
    if (value < 0 || static_cast<size_t>(value) >= n) {
      return absl::InvalidArgumentError(
          absl::StrCat("transpose permutation entry out of range for tensor ", tensor_name));
    }
    if (seen[static_cast<size_t>(value)]) {
      return absl::InvalidArgumentError(
          absl::StrCat("transpose permutation repeats dimension for tensor ", tensor_name));
    }
    seen[static_cast<size_t>(value)] = true;
    inverse[static_cast<size_t>(value)] = static_cast<int64_t>(idx);
  }
  return inverse;
}

absl::StatusOr<TransformPlan> build_inverse_transform(
    const TransformPlan& forward,
    const std::unordered_map<std::string, uint64_t>& canonical_offsets) {
  TransformPlan inverse;
  inverse.requires_materialization = forward.requires_materialization;
  if (forward.tensors.empty()) {
    return inverse;
  }
  inverse.tensors.reserve(forward.tensors.size());
  for (const TensorTransformPlan& tensor_plan : forward.tensors) {
    auto it = canonical_offsets.find(tensor_plan.tensor_name);
    if (it == canonical_offsets.end()) {
      return absl::InvalidArgumentError(absl::StrCat("missing canonical offset for tensor ", tensor_plan.tensor_name));
    }
    auto inverse_perm_or = compute_inverse_permutation(tensor_plan.permutation, tensor_plan.tensor_name);
    if (!inverse_perm_or.ok()) {
      return inverse_perm_or.status();
    }
    TensorTransformPlan inverse_plan;
    inverse_plan.tensor_name = tensor_plan.tensor_name;
    inverse_plan.dst_offset = it->second;
    inverse_plan.canonical_offset = it->second;
    inverse_plan.storage_offset_elements = tensor_plan.storage_offset_elements;
    inverse_plan.canonical_shape = tensor_plan.canonical_shape;
    inverse_plan.canonical_stride = tensor_plan.canonical_stride;
    inverse_plan.view_shape = tensor_plan.view_shape;
    inverse_plan.view_stride = tensor_plan.view_stride;
    inverse_plan.permutation = std::move(*inverse_perm_or);
    inverse_plan.dtype = tensor_plan.dtype;
    inverse_plan.element_size_bytes = tensor_plan.element_size_bytes;
    inverse.tensors.push_back(std::move(inverse_plan));
  }
  return inverse;
}

absl::StatusOr<BidirectionalViewPlan> compute_bidirectional_internal(
    std::string_view canonical_index_json,
    const ViewSpec& spec,
    absl::Span<const std::string> subset_names) {
  std::map<std::string, CanonicalEntry> canonical_entries;
  if (auto st = parse_canonical_index(canonical_index_json, &canonical_entries); !st.ok()) {
    return st;
  }

  for (const auto& [tensor_name, ops] : spec.tensors) {
    if (!canonical_entries.contains(tensor_name)) {
      return absl::InvalidArgumentError(absl::StrCat("ViewSpec references unknown tensor: ", tensor_name));
    }
    auto st = validate_ops(ops);
    if (!st.ok()) {
      return st;
    }
  }

  absl::flat_hash_set<std::string> subset_filter;
  subset_filter.reserve(subset_names.size());
  for (const auto& name : subset_names) {
    subset_filter.insert(name);
  }
  if (!subset_filter.empty()) {
    for (const auto& name : subset_filter) {
      if (!canonical_entries.contains(name)) {
        return absl::InvalidArgumentError(absl::StrCat("View subset references unknown tensor: ", name));
      }
    }
  }

  const bool subset_full = !subset_filter.empty() && subset_filter.size() == canonical_entries.size();
  const bool subset_enabled = !subset_filter.empty() && !subset_full;

  bool has_transform = false;
  uint64_t view_cursor = 0;
  SelectionPlan selection_plan;
  selection_plan.requires_materialization = false;
  bool any_requires_materialization = false;
  TransformPlan transform_plan;
  std::vector<std::string> ordered_names;
  ordered_names.reserve(canonical_entries.size());
  std::unordered_map<std::string, uint64_t> offsets;
  std::unordered_map<std::string, uint64_t> sizes;
  std::unordered_map<std::string, CanonicalTensorMeta> metas;
  std::unordered_map<std::string, uint64_t> canonical_offsets;
  offsets.reserve(canonical_entries.size());
  sizes.reserve(canonical_entries.size());
  metas.reserve(canonical_entries.size());
  canonical_offsets.reserve(canonical_entries.size());

  for (const auto& [name, entry] : canonical_entries) {
    if (subset_enabled && !subset_filter.contains(name)) {
      continue;
    }
    const uint64_t aligned_start = (view_cursor + (kAlignmentBytes - 1)) / kAlignmentBytes * kAlignmentBytes;
    if (aligned_start > view_cursor) {
      selection_plan.ranges.push_back(make_pad_range(view_cursor, aligned_start - view_cursor));
      view_cursor = aligned_start;
    }

    ordered_names.push_back(name);
    canonical_offsets[name] = entry.offset;
    const auto spec_it = spec.tensors.find(name);
    const TensorViewOps* ops = (spec_it == spec.tensors.end() ? nullptr : &spec_it->second);

    CanonicalTensorMeta meta{
        .shape = entry.shape, .stride = entry.stride, .dtype = entry.dtype, .storage_offset = entry.storage_offset};
    uint64_t tensor_data_bytes = entry.size_bytes;
    bool tensor_identity = true;

    uint64_t element_size = 0;
    if (auto size_or = dtype_element_size(entry.dtype); !size_or.ok()) {
      return size_or.status();
    } else {
      element_size = *size_or;
    }

    const uint64_t tensor_dst_offset = view_cursor;

    std::vector<int64_t> permutation(entry.shape.size());
    std::iota(permutation.begin(), permutation.end(), 0);

    if (ops && !ops->ops.empty()) {
      const ViewOp& first = ops->ops.front();
      if (first.kind == ViewOp::Kind::kNarrow) {
        const NarrowOp& narrow = first.narrow;
        if (narrow.length == 0) {
          return absl::InvalidArgumentError(absl::StrCat("narrow length must be > 0 for tensor ", name));
        }
        if (narrow.dim < 0 || narrow.dim >= static_cast<int32_t>(entry.shape.size())) {
          return absl::InvalidArgumentError(absl::StrCat("narrow dim out of range for tensor ", name));
        }
        if (ops->ops.size() > 1) {
          return absl::InvalidArgumentError(absl::StrCat("multiple narrow ops per tensor are not supported: ", name));
        }
        const int32_t dim = narrow.dim;
        const int64_t dim_extent = entry.shape[static_cast<size_t>(dim)];
        if (dim_extent <= 0) {
          return absl::InvalidArgumentError(
              absl::StrCat("invalid dimension extent in canonical index for tensor ", name));
        }
        if (narrow.length > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
          return absl::InvalidArgumentError("narrow length exceeds supported range");
        }
        int64_t normalized_start = narrow.start;
        if (normalized_start < 0) {
          normalized_start += dim_extent;
        }
        if (normalized_start < 0 || normalized_start >= dim_extent) {
          return absl::InvalidArgumentError(absl::StrCat("narrow start out of range for tensor ", name));
        }
        const int64_t length_i64 = static_cast<int64_t>(narrow.length);
        if (normalized_start + length_i64 > dim_extent) {
          return absl::InvalidArgumentError(absl::StrCat("narrow length exceeds dimension for tensor ", name));
        }

        if (normalized_start == 0 && length_i64 == dim_extent) {
          tensor_identity = true;
        } else {
          tensor_identity = false;
          has_transform = true;

          std::vector<int64_t> view_shape = entry.shape;
          view_shape[static_cast<size_t>(dim)] = length_i64;
          meta.shape = view_shape;
          meta.stride = compute_compact_stride(view_shape);
          meta.dtype = entry.dtype;
          meta.storage_offset = 0;

          const uint64_t inner = product(entry.shape, static_cast<size_t>(dim) + 1, entry.shape.size());
          const uint64_t outer = product(entry.shape, 0, static_cast<size_t>(dim));
          const uint64_t block_elements = static_cast<uint64_t>(length_i64) * inner;
          const uint64_t block_bytes = block_elements * element_size;

          const uint64_t stride_dim =
              (entry.stride.size() > static_cast<size_t>(dim)
                   ? static_cast<uint64_t>(entry.stride[static_cast<size_t>(dim)])
                   : inner);

          const uint64_t outer_count = std::max<uint64_t>(1, outer);

          tensor_data_bytes = block_bytes * outer_count;

          for (uint64_t outer_index = 0; outer_index < outer_count; ++outer_index) {
            uint64_t prefix_elements = entry.storage_offset;
            if (outer_count > 1) {
              uint64_t tmp = outer_index;
              for (int32_t i = dim - 1; i >= 0; --i) {
                const uint64_t extent = static_cast<uint64_t>(entry.shape[static_cast<size_t>(i)]);
                if (extent == 0) {
                  continue;
                }
                const uint64_t idx = tmp % extent;
                tmp /= extent;
                if (entry.stride.size() <= static_cast<size_t>(i)) {
                  return absl::InvalidArgumentError(absl::StrCat("stride missing for tensor ", name));
                }
                prefix_elements += static_cast<uint64_t>(entry.stride[static_cast<size_t>(i)]) * idx;
              }
            }
            const uint64_t start_elements = prefix_elements + static_cast<uint64_t>(normalized_start) * stride_dim;
            const uint64_t byte_offset = entry.offset + start_elements * element_size;
            selection_plan.ranges.push_back(make_data_range(byte_offset, view_cursor, block_bytes));
            view_cursor += block_bytes;
          }
        }
      } else if (first.kind == ViewOp::Kind::kTranspose) {
        tensor_identity = false;
        has_transform = true;
        for (const auto& view_op : ops->ops) {
          if (view_op.kind != ViewOp::Kind::kTranspose) {
            return absl::InvalidArgumentError("mixed operations per tensor are not supported");
          }
          const int32_t dim0 = view_op.transpose.dim0;
          const int32_t dim1 = view_op.transpose.dim1;
          if (dim0 < 0 || dim1 < 0 || dim0 >= static_cast<int32_t>(permutation.size()) ||
              dim1 >= static_cast<int32_t>(permutation.size())) {
            return absl::InvalidArgumentError(absl::StrCat("transpose dims out of range for tensor ", name));
          }
          std::swap(permutation[static_cast<size_t>(dim0)], permutation[static_cast<size_t>(dim1)]);
        }
        bool permutation_identity = true;
        for (size_t idx = 0; idx < permutation.size(); ++idx) {
          if (permutation[idx] != static_cast<int64_t>(idx)) {
            permutation_identity = false;
            break;
          }
        }
        if (permutation_identity) {
          tensor_identity = true;
        } else {
          std::vector<int64_t> view_shape(permutation.size());
          for (size_t idx = 0; idx < permutation.size(); ++idx) {
            view_shape[idx] = entry.shape[static_cast<size_t>(permutation[idx])];
          }
          const auto view_stride = compute_compact_stride(view_shape);
          meta.shape = view_shape;
          meta.stride = view_stride;
          meta.storage_offset = 0;
          tensor_data_bytes = entry.size_bytes;

          selection_plan.ranges.push_back(make_data_range(entry.offset, view_cursor, entry.size_bytes));
          view_cursor += entry.size_bytes;

          TensorTransformPlan tensor_transform;
          tensor_transform.tensor_name = name;
          tensor_transform.dst_offset = tensor_dst_offset;
          tensor_transform.canonical_offset = entry.offset;
          tensor_transform.storage_offset_elements = entry.storage_offset;
          tensor_transform.canonical_shape = entry.shape;
          tensor_transform.canonical_stride = entry.stride.empty() ? compute_compact_stride(entry.shape) : entry.stride;
          tensor_transform.view_shape = view_shape;
          tensor_transform.view_stride = view_stride;
          tensor_transform.permutation.assign(permutation.begin(), permutation.end());
          tensor_transform.dtype = entry.dtype;
          tensor_transform.element_size_bytes = element_size;
          transform_plan.tensors.push_back(std::move(tensor_transform));
          any_requires_materialization = true;
        }
      } else {
        return absl::InvalidArgumentError("unsupported view operation kind");
      }
    }

    if (tensor_identity) {
      selection_plan.ranges.push_back(make_data_range(entry.offset, view_cursor, entry.size_bytes));
      view_cursor += entry.size_bytes;
    }

    offsets[name] = tensor_dst_offset;
    sizes[name] = tensor_data_bytes;
    metas[name] = meta;
  }

  finalize_selection_plan(&selection_plan);
  selection_plan.requires_materialization = any_requires_materialization;
  transform_plan.requires_materialization = any_requires_materialization;

  ViewPlan forward_plan;
  forward_plan.is_identity = !has_transform && !subset_enabled;
  forward_plan.view_size_bytes = view_cursor;
  forward_plan.selection = std::move(selection_plan);
  forward_plan.transform = std::move(transform_plan);

  if (forward_plan.is_identity) {
    auto rebuilt_or = rebuild_stable_canonical_index(std::string(canonical_index_json), 0);
    if (!rebuilt_or.ok()) {
      return rebuilt_or.status();
    }
    forward_plan.view_index_json = *rebuilt_or;
  } else {
    absl::StatusOr<std::string> view_index_json_or = build_canonical_index_json(ordered_names, offsets, sizes, metas);
    if (!view_index_json_or.ok()) {
      return view_index_json_or.status();
    }
    forward_plan.view_index_json = *view_index_json_or;
  }

  ViewWritePlan write_plan = build_view_write_plan(forward_plan.selection);
  auto inverse_transform_or = build_inverse_transform(forward_plan.transform, canonical_offsets);
  if (!inverse_transform_or.ok()) {
    return inverse_transform_or.status();
  }

  BidirectionalViewPlan bidirectional;
  bidirectional.forward = std::move(forward_plan);
  bidirectional.write = std::move(write_plan);
  bidirectional.inverse_transform = std::move(*inverse_transform_or);
  return bidirectional;
}

} // namespace

absl::StatusOr<ViewPlan> ViewPlanner::compute_view_plan(std::string_view canonical_index_json, const ViewSpec& spec) {
  auto bidirectional_or = compute_bidirectional_internal(canonical_index_json, spec, {});
  if (!bidirectional_or.ok()) {
    return bidirectional_or.status();
  }
  BidirectionalViewPlan bidirectional = std::move(*bidirectional_or);
  return std::move(bidirectional.forward);
}

absl::StatusOr<BidirectionalViewPlan> ViewPlanner::compute_bidirectional_view_plan(
    std::string_view canonical_index_json,
    const ViewSpec& spec) {
  return compute_bidirectional_internal(canonical_index_json, spec, {});
}

absl::StatusOr<ViewPlan> ViewPlanner::compute_view_plan(
    std::string_view canonical_index_json,
    const ViewSpec& spec,
    absl::Span<const std::string> subset_names) {
  auto bidirectional_or = compute_bidirectional_internal(canonical_index_json, spec, subset_names);
  if (!bidirectional_or.ok()) {
    return bidirectional_or.status();
  }
  BidirectionalViewPlan bidirectional = std::move(*bidirectional_or);
  return std::move(bidirectional.forward);
}

absl::StatusOr<BidirectionalViewPlan> ViewPlanner::compute_bidirectional_view_plan(
    std::string_view canonical_index_json,
    const ViewSpec& spec,
    absl::Span<const std::string> subset_names) {
  return compute_bidirectional_internal(canonical_index_json, spec, subset_names);
}

} // namespace tensorcast::store::loader
