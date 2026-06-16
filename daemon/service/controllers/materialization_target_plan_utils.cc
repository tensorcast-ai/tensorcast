// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_target_plan_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/view_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_mapped_target_layout_utils.h"
#include "daemon/service/controllers/materialization_mapped_view_narrow_utils.h"
#include "daemon/service/controllers/materialization_mapped_view_spec_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/representation_transform_builder.h"
#include "daemon/service/controllers/selection_validation_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon::materialization_target_plan {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using materialization_layout::CanonicalIndexEntry;
using materialization_layout::CanonicalIndexTable;
using materialization_layout::dtype_element_size;
using materialization_layout::parse_canonical_index_shared;
using materialization_layout::parse_canonical_index_shared_with_identity;
using materialization_layout::TargetOffsetEntry;
using materialization_mapped_target_layout::validate_mapped_target_layout;
using materialization_mapped_target_layout::ValidatedMappedTargetLayout;
using materialization_mapped_target_layout::validation_error_reason;
using materialization_mapped_target_layout::ValidationErrorReason;
using materialization_mapped_view_narrow::build_view_narrows;
using materialization_mapped_view_narrow::view_narrow_error_reason;
using materialization_mapped_view_narrow::ViewNarrowErrorReason;
using materialization_mapped_view_spec::resolve_mapped_view_spec;
using materialization_mapped_view_spec::resolve_view_spec_error_reason;
using materialization_mapped_view_spec::ResolveViewSpecErrorReason;
using materialization_policy::build_view_spec_proto;
using materialization_policy::compute_view_id_from_spec;
using materialization_policy::convert_view_spec;
using representation_layout::TensorLayoutSpec;
using representation_layout::ViewNarrowSpec;
using representation_transform_builder::build_representation_transform_contract;
using store::loader::ViewSpec;

store::runtime::ingestion::strategy::SourceBoundLoweringStats to_source_bound_lowering_stats(
    const representation_transform_builder::TransformWorkLoweringStats& stats) {
  return store::runtime::ingestion::strategy::SourceBoundLoweringStats{
      .total_dst_tensors = stats.total_dst_tensors,
      .collective_candidates = stats.collective_candidates,
      .collective_bytes = stats.collective_bytes,
      .concat_candidates = stats.concat_candidates,
      .concat_bytes = stats.concat_bytes,
      .rejected_mixed_src_or_dim = stats.rejected_mixed_src_or_dim,
      .rejected_mixed_src_or_dim_bytes = stats.rejected_mixed_src_or_dim_bytes,
      .rejected_non_contiguous = stats.rejected_non_contiguous,
      .rejected_non_contiguous_bytes = stats.rejected_non_contiguous_bytes,
      .rejected_unsupported_distribution = stats.rejected_unsupported_distribution,
      .rejected_unsupported_distribution_bytes = stats.rejected_unsupported_distribution_bytes,
  };
}

absl::StatusOr<std::string> build_mapped_target_selected_index_json(const ValidatedMappedTargetLayout& mapped_layout) {
  std::vector<std::string> ordered_names;
  ordered_names.reserve(mapped_layout.dst_specs.size());
  std::unordered_map<std::string, uint64_t> offsets;
  std::unordered_map<std::string, uint64_t> sizes;
  std::unordered_map<std::string, store::loader::CanonicalTensorMeta> metas;
  offsets.reserve(mapped_layout.dst_specs.size());
  sizes.reserve(mapped_layout.dst_specs.size());
  metas.reserve(mapped_layout.dst_specs.size());

  for (const auto& [name, spec] : mapped_layout.dst_specs) {
    auto offset_it = mapped_layout.dst_base_offsets.find(name);
    if (offset_it == mapped_layout.dst_base_offsets.end()) {
      return absl::InvalidArgumentError("mapped target layout missing dst base offset");
    }
    ordered_names.push_back(name);
    offsets.emplace(name, offset_it->second);
    sizes.emplace(name, spec.logical_length);
    metas.emplace(
        name,
        store::loader::CanonicalTensorMeta{
            .shape = spec.shape,
            .stride = spec.stride,
            .dtype = spec.dtype,
            .storage_offset = spec.storage_offset,
        });
  }
  std::sort(ordered_names.begin(), ordered_names.end());
  return store::loader::build_canonical_index_json(ordered_names, offsets, sizes, metas);
}

tensorcast::common::v1::ByteSpaceRef build_source_byte_space(std::optional<std::string_view> view_id) {
  tensorcast::common::v1::ByteSpaceRef byte_space;
  if (view_id.has_value() && !view_id->empty()) {
    byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
    byte_space.set_id(std::string(*view_id));
    return byte_space;
  }
  byte_space.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  return byte_space;
}

void patch_transform_contract_source_specs(
    const CanonicalIndexTable& source_table,
    store::materialization::contracts::RepresentationTransformContract* transform_contract) {
  if (transform_contract == nullptr) {
    return;
  }
  for (auto& binding : transform_contract->tensor_bindings) {
    for (auto& source : binding.sources) {
      auto it = source_table.entries.find(source.source_spec.name);
      if (it == source_table.entries.end()) {
        continue;
      }
      auto elem_or = materialization_layout::dtype_element_size(it->second.dtype);
      if (!elem_or.ok()) {
        continue;
      }
      source.source_spec.shape = it->second.shape;
      source.source_spec.stride = it->second.stride;
      source.source_spec.dtype = it->second.dtype;
      source.source_spec.logical_offset = it->second.logical_offset;
      source.source_spec.logical_length = it->second.logical_length;
      source.source_spec.storage_offset = it->second.storage_offset;
      source.source_spec.element_size = *elem_or;
    }
  }
}

bool source_spec_patch_compatible(
    const store::materialization::contracts::RepresentationTensorSpec& spec,
    const CanonicalIndexEntry& entry,
    uint64_t element_size) {
  return spec.shape == entry.shape && spec.stride == entry.stride && spec.dtype == entry.dtype &&
      spec.logical_length == entry.logical_length && spec.element_size == element_size;
}

absl::StatusOr<bool> can_patch_work_plan_source_specs(
    const CanonicalIndexTable& source_table,
    const store::materialization::contracts::RepresentationTransformContract& transform_contract) {
  for (const auto& binding : transform_contract.tensor_bindings) {
    for (const auto& source : binding.sources) {
      auto it = source_table.entries.find(source.source_spec.name);
      if (it == source_table.entries.end()) {
        continue;
      }
      auto elem_or = materialization_layout::dtype_element_size(it->second.dtype);
      if (!elem_or.ok()) {
        return elem_or.status();
      }
      if (!source_spec_patch_compatible(source.source_spec, it->second, *elem_or)) {
        return false;
      }
    }
  }
  return true;
}

void patch_source_spec_from_entry(
    const CanonicalIndexEntry& entry,
    uint64_t element_size,
    store::materialization::contracts::RepresentationTensorSpec* spec) {
  if (spec == nullptr) {
    return;
  }
  spec->shape = entry.shape;
  spec->stride = entry.stride;
  spec->dtype = entry.dtype;
  spec->logical_offset = entry.logical_offset;
  spec->logical_length = entry.logical_length;
  spec->storage_offset = entry.storage_offset;
  spec->element_size = element_size;
}

absl::Status patch_work_plan_source_specs(
    const CanonicalIndexTable& source_table,
    store::materialization::contracts::RepresentationWorkPlan* work_plan) {
  if (work_plan == nullptr) {
    return absl::OkStatus();
  }
  for (auto& item : work_plan->items) {
    for (auto& source : item.sources) {
      auto it = source_table.entries.find(source.fragment.source_spec.name);
      if (it == source_table.entries.end()) {
        continue;
      }
      auto elem_or = materialization_layout::dtype_element_size(it->second.dtype);
      if (!elem_or.ok()) {
        return elem_or.status();
      }
      patch_source_spec_from_entry(it->second, *elem_or, &source.fragment.source_spec);
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<store::materialization::contracts::RepresentationWorkPlan> build_execution_representation_work_plan(
    const store::materialization::contracts::RepresentationTransformContract& transform_contract,
    const CanonicalIndexTable* physical_source_table,
    std::optional<store::loader::ByteRangeMap> collective_lowered_map = std::nullopt,
    bool* source_rebind_fast_path = nullptr) {
  if (source_rebind_fast_path != nullptr) {
    *source_rebind_fast_path = false;
  }
  auto append_pad_items = [&](store::materialization::contracts::RepresentationWorkPlan* work_plan) {
    if (work_plan == nullptr || !collective_lowered_map.has_value()) {
      return;
    }
    store::loader::ByteRangeMap pad_map;
    pad_map.total_bytes = collective_lowered_map->total_bytes;
    pad_map.num_sources = collective_lowered_map->num_sources;
    for (const auto& segment : collective_lowered_map->segments) {
      if (segment.kind == store::loader::ByteRangeSegment::Kind::kPad) {
        pad_map.segments.push_back(segment);
      }
    }
    if (!pad_map.segments.empty()) {
      store::materialization::contracts::RepresentationWorkItem pad_item;
      pad_item.kind = store::materialization::contracts::RepresentationWorkItemKind::kPadFill;
      pad_item.partition_kind = store::materialization::contracts::WorkPartitionKind::kUnknown;
      pad_item.byte_range_map = std::move(pad_map);
      for (const auto& segment : pad_item.byte_range_map.segments) {
        pad_item.committed_bytes += segment.length;
      }
      work_plan->committed_bytes += pad_item.committed_bytes;
      work_plan->items.push_back(std::move(pad_item));
    }
  };

  if (physical_source_table != nullptr) {
    auto compatible_or = can_patch_work_plan_source_specs(*physical_source_table, transform_contract);
    if (!compatible_or.ok()) {
      return compatible_or.status();
    }
    if (*compatible_or) {
      auto work_plan_or = store::materialization::contracts::build_representation_work_plan(transform_contract);
      if (!work_plan_or.ok()) {
        return work_plan_or.status();
      }
      auto patch_status = patch_work_plan_source_specs(*physical_source_table, &*work_plan_or);
      if (!patch_status.ok()) {
        return patch_status;
      }
      append_pad_items(&*work_plan_or);
      if (source_rebind_fast_path != nullptr) {
        *source_rebind_fast_path = true;
      }
      return std::move(*work_plan_or);
    }
  }

  auto execution_contract = transform_contract;
  if (physical_source_table != nullptr) {
    patch_transform_contract_source_specs(*physical_source_table, &execution_contract);
  }
  auto work_plan_or = store::materialization::contracts::build_representation_work_plan(execution_contract);
  if (!work_plan_or.ok()) {
    return work_plan_or.status();
  }
  append_pad_items(&*work_plan_or);
  return std::move(*work_plan_or);
}

absl::StatusOr<store::loader::ByteRangeMap> filter_byte_range_map(
    const store::loader::ByteRangeMap& map,
    store::loader::ByteRangeSegment::Kind kind) {
  store::loader::ByteRangeMap filtered;
  filtered.total_bytes = map.total_bytes;
  filtered.num_sources = map.num_sources;
  bool sorted_by_dst = true;
  std::optional<uint64_t> previous_end;
  for (const auto& segment : map.segments) {
    if (segment.kind == kind) {
      if (previous_end.has_value() && segment.dst_offset < *previous_end) {
        sorted_by_dst = false;
      }
      previous_end = segment.dst_offset + segment.length;
      filtered.segments.push_back(segment);
    }
  }
  if (!sorted_by_dst) {
    std::sort(filtered.segments.begin(), filtered.segments.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.dst_offset < rhs.dst_offset;
    });
  }
  for (size_t index = 1; index < filtered.segments.size(); ++index) {
    const auto& previous = filtered.segments[index - 1];
    const auto& current = filtered.segments[index];
    if (current.dst_offset < previous.dst_offset + previous.length) {
      return absl::InvalidArgumentError("filtered byte range map contains overlapping segments");
    }
  }
  return filtered;
}

absl::StatusOr<store::loader::ByteRangeMap> merge_byte_range_maps(
    const store::loader::ByteRangeMap& lhs,
    const store::loader::ByteRangeMap& rhs) {
  store::loader::ByteRangeMap merged;
  merged.total_bytes = std::max(lhs.total_bytes, rhs.total_bytes);
  merged.num_sources = std::max(lhs.num_sources, rhs.num_sources);
  merged.segments.reserve(lhs.segments.size() + rhs.segments.size());
  merged.segments.insert(merged.segments.end(), lhs.segments.begin(), lhs.segments.end());
  merged.segments.insert(merged.segments.end(), rhs.segments.begin(), rhs.segments.end());
  std::sort(merged.segments.begin(), merged.segments.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.dst_offset < rhs.dst_offset;
  });
  for (size_t index = 1; index < merged.segments.size(); ++index) {
    const auto& previous = merged.segments[index - 1];
    const auto& current = merged.segments[index];
    if (current.dst_offset < previous.dst_offset + previous.length) {
      return absl::InvalidArgumentError("merged byte range map contains overlapping segments");
    }
  }
  return merged;
}

uint64_t byte_range_map_covered_bytes(const store::loader::ByteRangeMap& map) {
  uint64_t total = 0;
  for (const auto& segment : map.segments) {
    total += segment.length;
  }
  return total;
}

bool is_data_representation_work_item(store::materialization::contracts::RepresentationWorkItemKind kind) {
  using RepresentationWorkItemKind = store::materialization::contracts::RepresentationWorkItemKind;
  switch (kind) {
    case RepresentationWorkItemKind::kTensorCopy:
    case RepresentationWorkItemKind::kConcatAssemble:
    case RepresentationWorkItemKind::kExpertDim0Concat:
      return true;
    case RepresentationWorkItemKind::kResidualByteRange:
    case RepresentationWorkItemKind::kScalarBroadcastFill:
    case RepresentationWorkItemKind::kConstFill:
    case RepresentationWorkItemKind::kPadFill:
      return false;
  }
  return false;
}

void append_coverage_data_segment(store::loader::ByteRangeMap* map, uint64_t dst_offset, uint64_t length) {
  if (map == nullptr || length == 0) {
    return;
  }
  map->segments.push_back(
      store::loader::ByteRangeSegment{
          .kind = store::loader::ByteRangeSegment::Kind::kData,
          .dst_offset = dst_offset,
          .length = length,
          .src_offset = dst_offset,
          .source_index = 0,
      });
}

absl::Status sort_and_merge_coverage_data_segments(store::loader::ByteRangeMap* map) {
  if (map == nullptr || map->segments.empty()) {
    return absl::OkStatus();
  }
  std::sort(map->segments.begin(), map->segments.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.dst_offset != rhs.dst_offset) {
      return lhs.dst_offset < rhs.dst_offset;
    }
    return lhs.length < rhs.length;
  });
  std::vector<store::loader::ByteRangeSegment> merged;
  merged.reserve(map->segments.size());
  for (const auto& segment : map->segments) {
    if (segment.length == 0) {
      continue;
    }
    if (!merged.empty()) {
      auto& previous = merged.back();
      const uint64_t previous_end = previous.dst_offset + previous.length;
      if (segment.dst_offset < previous_end) {
        return absl::InvalidArgumentError("compact coverage map contains overlapping data ranges");
      }
      if (segment.dst_offset == previous_end) {
        previous.length += segment.length;
        continue;
      }
    }
    merged.push_back(segment);
  }
  map->segments = std::move(merged);
  return absl::OkStatus();
}

template <typename IncludeItem>
absl::StatusOr<store::loader::ByteRangeMap> build_compact_data_coverage_map_from_work_plan(
    const store::materialization::contracts::RepresentationWorkPlan& work_plan,
    uint64_t total_bytes,
    IncludeItem include_item) {
  store::loader::ByteRangeMap map;
  map.total_bytes = total_bytes;
  map.num_sources = 1;
  for (const auto& item : work_plan.items) {
    if (!include_item(item) || item.committed_bytes == 0) {
      continue;
    }
    if (item.committed_bytes == item.dst_spec.logical_length) {
      append_coverage_data_segment(&map, item.dst_spec.logical_offset, item.dst_spec.logical_length);
      continue;
    }
    for (const auto& source : item.sources) {
      auto spans_or = store::materialization::contracts::build_coordinate_byte_spans(
          item.dst_spec, source.fragment.destination_range);
      if (!spans_or.ok()) {
        return spans_or.status();
      }
      for (const auto& span : *spans_or) {
        append_coverage_data_segment(&map, span.offset, span.length);
      }
    }
  }
  auto status = sort_and_merge_coverage_data_segments(&map);
  if (!status.ok()) {
    return status;
  }
  return map;
}

absl::StatusOr<store::loader::ByteRangeMap> build_compact_data_coverage_map_from_work_plan(
    const store::materialization::contracts::RepresentationWorkPlan& work_plan,
    uint64_t total_bytes) {
  return build_compact_data_coverage_map_from_work_plan(
      work_plan, total_bytes, [](const store::materialization::contracts::RepresentationWorkItem& item) {
        return is_data_representation_work_item(item.kind);
      });
}

bool work_item_has_collective_source_overlap(const store::materialization::contracts::RepresentationWorkItem& item) {
  using RepresentationWorkItemKind = store::materialization::contracts::RepresentationWorkItemKind;
  return item.kind == RepresentationWorkItemKind::kTensorCopy &&
      item.partition_kind == store::materialization::contracts::WorkPartitionKind::kReplicated;
}

absl::StatusOr<std::vector<std::pair<uint64_t, uint64_t>>> build_collective_dst_ranges(
    const store::materialization::contracts::RepresentationWorkPlan& work_plan) {
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (const auto& item : work_plan.items) {
    if (!work_item_has_collective_source_overlap(item)) {
      continue;
    }
    for (const auto& source : item.sources) {
      auto spans_or = store::materialization::contracts::build_coordinate_byte_spans(
          item.dst_spec, source.fragment.destination_range);
      if (!spans_or.ok()) {
        return spans_or.status();
      }
      for (const auto& span : *spans_or) {
        if (span.length == 0) {
          continue;
        }
        ranges.emplace_back(span.offset, span.offset + span.length);
      }
    }
  }
  std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.first != rhs.first) {
      return lhs.first < rhs.first;
    }
    return lhs.second < rhs.second;
  });
  std::vector<std::pair<uint64_t, uint64_t>> merged;
  merged.reserve(ranges.size());
  for (const auto& range : ranges) {
    if (range.second <= range.first) {
      continue;
    }
    if (merged.empty() || range.first > merged.back().second) {
      merged.push_back(range);
      continue;
    }
    merged.back().second = std::max<uint64_t>(merged.back().second, range.second);
  }
  return merged;
}

absl::StatusOr<store::loader::ByteRangeMap> filter_data_map_to_dst_ranges(
    const store::loader::ByteRangeMap& data_map,
    const std::vector<std::pair<uint64_t, uint64_t>>& dst_ranges) {
  store::loader::ByteRangeMap filtered;
  filtered.total_bytes = data_map.total_bytes;
  filtered.num_sources = data_map.num_sources;
  if (dst_ranges.empty()) {
    return filtered;
  }
  size_t range_index = 0;
  for (const auto& segment : data_map.segments) {
    if (segment.kind != store::loader::ByteRangeSegment::Kind::kData || segment.length == 0) {
      continue;
    }
    const uint64_t segment_begin = segment.dst_offset;
    const uint64_t segment_end = segment.dst_offset + segment.length;
    while (range_index < dst_ranges.size() && dst_ranges[range_index].second <= segment_begin) {
      ++range_index;
    }
    for (size_t index = range_index; index < dst_ranges.size(); ++index) {
      const auto& [range_begin, range_end] = dst_ranges[index];
      if (range_begin >= segment_end) {
        break;
      }
      const uint64_t overlap_begin = std::max<uint64_t>(segment_begin, range_begin);
      const uint64_t overlap_end = std::min<uint64_t>(segment_end, range_end);
      if (overlap_end <= overlap_begin) {
        continue;
      }
      filtered.segments.push_back(
          store::loader::ByteRangeSegment{
              .kind = store::loader::ByteRangeSegment::Kind::kData,
              .dst_offset = overlap_begin,
              .length = overlap_end - overlap_begin,
              .src_offset = segment.src_offset + (overlap_begin - segment_begin),
              .source_index = segment.source_index,
          });
    }
  }
  std::sort(filtered.segments.begin(), filtered.segments.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.dst_offset < rhs.dst_offset;
  });
  for (size_t index = 1; index < filtered.segments.size(); ++index) {
    const auto& previous = filtered.segments[index - 1];
    const auto& current = filtered.segments[index];
    if (current.dst_offset < previous.dst_offset + previous.length) {
      return absl::InvalidArgumentError("collective overlap byte range map contains overlapping segments");
    }
  }
  return filtered;
}

void record_error(RecordMaterializeResultFn record_result, std::string_view reason) {
  if (record_result == nullptr) {
    return;
  }
  record_result("error", reason, v2::MaterializationSource::MATERIALIZATION_SOURCE_UNSPECIFIED);
}

struct TargetLayoutTensorSelection {
  absl::flat_hash_set<std::string> layout_name_set;
  std::vector<std::string> layout_names;
  std::vector<std::string> request_names;
};

Status collect_target_layout_tensor_selection(
    const tensorcast::common::v1::ArtifactSelection& request_selection,
    const std::vector<TargetOffsetEntry>& offsets,
    RecordMaterializeResultFn record_result,
    TargetLayoutTensorSelection& selection) {
  selection.layout_name_set.clear();
  selection.layout_names.clear();
  selection.request_names.clear();

  selection.layout_name_set.reserve(offsets.size());
  selection.layout_names.reserve(offsets.size());
  for (const auto& entry : offsets) {
    if (entry.name.empty()) {
      record_error(record_result, "tensor_name_missing");
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes empty tensor name"};
    }
    if (!selection.layout_name_set.insert(entry.name).second) {
      record_error(record_result, "tensor_name_duplicate");
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes duplicate tensor name"};
    }
    selection.layout_names.push_back(entry.name);
  }

  if (request_selection.tensor_names_size() == 0) {
    return Status::OK;
  }

  absl::flat_hash_set<std::string> request_name_set;
  selection.request_names.reserve(request_selection.tensor_names_size());
  request_name_set.reserve(request_selection.tensor_names_size());
  for (const auto& name : request_selection.tensor_names()) {
    selection.request_names.push_back(name);
    request_name_set.insert(name);
  }
  if (request_name_set.size() != static_cast<size_t>(request_selection.tensor_names_size())) {
    record_error(record_result, "tensor_name_duplicate");
    return {StatusCode::INVALID_ARGUMENT, "selection.tensor_names must not contain duplicates"};
  }
  if (request_name_set.size() != selection.layout_name_set.size()) {
    record_error(record_result, "tensor_name_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "selection.tensor_names do not match target_layout entries"};
  }
  for (const auto& name : selection.layout_name_set) {
    if (!request_name_set.contains(name)) {
      record_error(record_result, "tensor_name_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "selection.tensor_names do not match target_layout entries"};
    }
  }

  return Status::OK;
}

Status validate_layout_names_exist_in_index(
    const absl::flat_hash_set<std::string>& layout_name_set,
    const CanonicalIndexTable& index_table,
    std::string_view error_message,
    RecordMaterializeResultFn record_result) {
  for (const auto& name : layout_name_set) {
    if (!index_table.entries.contains(name)) {
      record_error(record_result, "tensor_name_mismatch");
      return {StatusCode::INVALID_ARGUMENT, std::string(error_message)};
    }
  }
  return Status::OK;
}

Status validate_selected_index_matches_layout(
    const absl::flat_hash_set<std::string>& layout_name_set,
    const CanonicalIndexTable& index_table,
    RecordMaterializeResultFn record_result) {
  if (index_table.entries.size() != layout_name_set.size()) {
    record_error(record_result, "tensor_name_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "target_layout must include every selected tensor"};
  }
  return validate_layout_names_exist_in_index(
      layout_name_set, index_table, "target_layout includes unknown tensor name", record_result);
}

bool is_layout_subset_of_canonical_index(
    const CanonicalIndexTable& canonical_table,
    const absl::flat_hash_set<std::string>& layout_name_set) {
  if (canonical_table.entries.size() != layout_name_set.size()) {
    return true;
  }
  for (const auto& [name, _] : canonical_table.entries) {
    if (!layout_name_set.contains(name)) {
      return true;
    }
  }
  return false;
}

struct StorageRange {
  std::string storage_id;
  uint64_t base_offset{0};
  uint64_t length{0};
};

struct PreparedTargetStorageLayout {
  std::vector<RegisterStorageMeta> publish_storages;
  std::vector<LeaseSegMeta> publish_segments;
  absl::flat_hash_map<std::string, StorageRange> storage_ranges;
  uint64_t total_storage_bytes{0};
};

Status build_target_storage_layout(
    const v2::TargetLayout& layout,
    RecordMaterializeResultFn record_result,
    PreparedTargetStorageLayout& prepared) {
  prepared.publish_storages.clear();
  prepared.publish_segments.clear();
  prepared.storage_ranges.clear();
  prepared.total_storage_bytes = 0;

  prepared.publish_storages.reserve(layout.storages_size());
  prepared.publish_segments.reserve(layout.storages_size());
  prepared.storage_ranges.reserve(layout.storages_size());

  uint64_t range_cursor = 0;
  for (const auto& storage : layout.storages()) {
    if (storage.storage_id().empty()) {
      record_error(record_result, "storage_id_missing");
      return {StatusCode::INVALID_ARGUMENT, "storage_id is required for each storage entry"};
    }
    if (storage.storage_length() == 0) {
      record_error(record_result, "storage_length_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "storage_length must be non-zero"};
    }
    if (prepared.storage_ranges.contains(storage.storage_id())) {
      record_error(record_result, "storage_id_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "storage_id must be unique in target_layout"};
    }

    StorageRange range;
    range.storage_id = storage.storage_id();
    range.base_offset = range_cursor;
    range.length = storage.storage_length();
    prepared.storage_ranges.emplace(range.storage_id, range);

    RegisterStorageMeta meta;
    meta.storage_id = storage.storage_id();
    meta.device_id = storage.device_id();
    meta.storage_length = storage.storage_length();
    if (!storage.vram_region_id().empty()) {
      meta.region_id = storage.vram_region_id();
    }
    if (!storage.cuda_ipc_handle().empty()) {
      meta.handle_bytes = storage.cuda_ipc_handle();
    }
    meta.mapping_base_offset = storage.mapping_base_offset();
    prepared.publish_storages.push_back(std::move(meta));

    LeaseSegMeta seg;
    seg.storage_id = storage.storage_id();
    seg.storage_offset = 0;
    seg.artifact_offset = range_cursor;
    seg.length = storage.storage_length();
    prepared.publish_segments.push_back(std::move(seg));

    if (storage.storage_length() > std::numeric_limits<uint64_t>::max() - range_cursor) {
      record_error(record_result, "storage_length_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "storage_length sum overflow"};
    }
    range_cursor += storage.storage_length();
  }
  prepared.total_storage_bytes = range_cursor;
  return Status::OK;
}

Status validate_storage_span_matches_index(
    uint64_t total_storage_bytes,
    uint64_t logical_total_size,
    RecordMaterializeResultFn record_result) {
  if (total_storage_bytes != logical_total_size) {
    record_error(record_result, "storage_length_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "storage_length must span logical byte space"};
  }
  return Status::OK;
}

Status validate_target_offsets_against_layout(
    const std::vector<TargetOffsetEntry>& offsets,
    const CanonicalIndexTable& index_table,
    const absl::flat_hash_map<std::string, StorageRange>& storage_ranges,
    RecordMaterializeResultFn record_result) {
  absl::flat_hash_set<std::string> seen_offsets;
  seen_offsets.reserve(offsets.size());
  for (const auto& entry : offsets) {
    if (!seen_offsets.insert(entry.name).second) {
      record_error(record_result, "tensor_name_duplicate");
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes duplicate tensor name"};
    }
    auto it = index_table.entries.find(entry.name);
    if (it == index_table.entries.end()) {
      record_error(record_result, "tensor_name_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout includes unknown tensor name"};
    }
    const auto range_it = storage_ranges.find(entry.storage_id);
    if (range_it == storage_ranges.end()) {
      record_error(record_result, "storage_id_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout references unknown storage_id"};
    }
    const auto& range = range_it->second;
    const auto& index_entry = it->second;
    if (entry.logical_length != index_entry.logical_length) {
      record_error(record_result, "layout_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout logical_length mismatch"};
    }
    if (index_entry.logical_offset < range.base_offset) {
      record_error(record_result, "offset_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout offset out of storage bounds"};
    }
    const uint64_t expected_storage_offset = index_entry.logical_offset - range.base_offset;
    if (entry.storage_offset != expected_storage_offset) {
      record_error(record_result, "offset_mismatch");
      return {
          StatusCode::INVALID_ARGUMENT,
          absl::StrCat(
              "target_layout storage_offset mismatch for tensor=",
              entry.name,
              " expected=",
              expected_storage_offset,
              " actual=",
              entry.storage_offset)};
    }
    if (expected_storage_offset + entry.logical_length > range.length) {
      record_error(record_result, "offset_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout exceeds storage bounds"};
    }
  }
  return Status::OK;
}

absl::StatusOr<ValidatedMappedTargetLayout> build_mapped_layout_from_target_index(
    std::string_view target_index_json,
    const std::vector<TargetOffsetEntry>& offsets,
    const absl::flat_hash_map<std::string, StorageRange>& storage_ranges) {
  auto index_table_or = parse_canonical_index_shared(target_index_json);
  if (!index_table_or.ok()) {
    return index_table_or.status();
  }
  const auto& index_table = **index_table_or;

  ValidatedMappedTargetLayout result;
  result.logical_total_size = index_table.logical_total_size;
  result.dst_specs.reserve(index_table.entries.size());
  result.dst_base_offsets.reserve(index_table.entries.size());

  absl::flat_hash_map<std::string, TargetOffsetEntry> offsets_by_name;
  offsets_by_name.reserve(offsets.size());
  for (const auto& offset : offsets) {
    offsets_by_name.emplace(offset.name, offset);
  }

  if (index_table.entries.size() != offsets_by_name.size()) {
    return absl::InvalidArgumentError("target_index must match target_layout offsets");
  }

  for (const auto& [name, index_entry] : index_table.entries) {
    const auto offset_it = offsets_by_name.find(name);
    if (offset_it == offsets_by_name.end()) {
      return absl::InvalidArgumentError("target_index includes unknown tensor name");
    }
    const auto storage_it = storage_ranges.find(offset_it->second.storage_id);
    if (storage_it == storage_ranges.end()) {
      return absl::InvalidArgumentError("target_layout references unknown storage_id");
    }
    auto elem_or = dtype_element_size(index_entry.dtype);
    if (!elem_or.ok()) {
      return elem_or.status();
    }
    result.dst_specs.emplace(
        name,
        TensorLayoutSpec{
            .shape = index_entry.shape,
            .stride = index_entry.stride,
            .dtype = index_entry.dtype,
            .storage_offset = index_entry.storage_offset,
            .logical_length = index_entry.logical_length,
            .element_size = *elem_or,
        });
    result.dst_base_offsets.emplace(name, storage_it->second.base_offset + offset_it->second.storage_offset);
  }
  return result;
}

struct TargetViewResolution {
  std::optional<ViewSpec> view_spec;
  std::optional<tensorcast::common::v1::ViewSpec> view_spec_proto;
  std::optional<std::string> request_view_id;
  std::optional<std::string> view_data_hash;
  std::optional<store::loader::ViewPlan> view_plan;
  std::optional<std::string> resolved_view_id;
  std::vector<std::string> metadata_subset_names;
  bool view_id_requested{false};
  bool has_view_transform{false};
};

Status parse_target_view_identity(
    const tensorcast::common::v1::ArtifactSelection& selection,
    TargetViewResolution& view_resolution) {
  if (selection.has_view_spec()) {
    auto spec_or = convert_view_spec(selection.view_spec());
    if (!spec_or.ok()) {
      return to_grpc_status(spec_or.status());
    }
    view_resolution.view_spec = std::move(*spec_or);
    view_resolution.view_spec_proto = selection.view_spec();
  }
  if (!selection.view_id().empty()) {
    view_resolution.view_id_requested = true;
    view_resolution.request_view_id = selection.view_id();
  }
  return Status::OK;
}

Status hydrate_target_view_from_metadata(
    store::StoreEngine& engine,
    std::string_view resolved_artifact_id,
    RecordMaterializeResultFn record_result,
    TargetViewResolution& view_resolution) {
  if (!view_resolution.view_id_requested || !view_resolution.request_view_id.has_value() ||
      view_resolution.view_spec.has_value()) {
    return Status::OK;
  }

  auto view_meta_or = engine.get_view_metadata(std::string(resolved_artifact_id), *view_resolution.request_view_id);
  if (!view_meta_or.ok()) {
    record_error(record_result, "view_meta_missing");
    return to_grpc_status(view_meta_or.status());
  }
  auto parsed_or = store::view::parse_view_selection_json(view_meta_or->view_spec_json);
  if (!parsed_or.ok()) {
    record_error(record_result, "view_parse_failed");
    return to_grpc_status(parsed_or.status());
  }
  view_resolution.view_spec = std::move(parsed_or->spec);
  view_resolution.view_spec_proto = build_view_spec_proto(*view_resolution.view_spec);
  view_resolution.metadata_subset_names = std::move(parsed_or->tensor_names);
  view_resolution.view_data_hash = view_meta_or->view_data_hash;
  return Status::OK;
}

Status compute_target_view_plan(
    std::string_view canonical_json,
    bool has_subset,
    bool has_ordered_selection,
    const std::vector<std::string>& layout_names,
    const std::vector<std::string>& request_names,
    v2::TargetLayout::IndexKind index_kind,
    RecordMaterializeResultFn record_result,
    TargetViewResolution& view_resolution) {
  const bool has_metadata_subset = !view_resolution.metadata_subset_names.empty();
  if (!view_resolution.view_spec.has_value() && !has_subset && index_kind != v2::TargetLayout::INDEX_KIND_VIEW &&
      !has_ordered_selection && !has_metadata_subset) {
    return Status::OK;
  }

  ViewSpec plan_spec = view_resolution.view_spec.value_or(ViewSpec{});
  std::vector<std::string> subset_names;
  if (has_ordered_selection) {
    subset_names = request_names;
  } else if (has_subset) {
    subset_names = layout_names;
  } else if (has_metadata_subset) {
    subset_names = view_resolution.metadata_subset_names;
  }
  auto plan_or = store::StoreEngine::compute_view_plan(std::string(canonical_json), plan_spec, subset_names);
  if (!plan_or.ok()) {
    record_error(record_result, "view_plan_failed");
    return to_grpc_status(plan_or.status());
  }
  view_resolution.view_plan = std::move(*plan_or);
  return Status::OK;
}

Status validate_target_index_kind_mode(
    const v2::TargetLayout& layout,
    bool has_subset,
    bool has_ordered_selection,
    RecordMaterializeResultFn record_result,
    TargetViewResolution& view_resolution) {
  const bool has_metadata_subset = !view_resolution.metadata_subset_names.empty();
  view_resolution.has_view_transform = view_resolution.view_id_requested ||
      (view_resolution.view_spec.has_value() && view_resolution.view_plan.has_value() &&
       !view_resolution.view_plan->is_identity);
  if (view_resolution.view_id_requested && view_resolution.view_plan.has_value() &&
      view_resolution.view_plan->is_identity && !has_subset && !has_ordered_selection && !has_metadata_subset) {
    record_error(record_result, "view_identity_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "view_id requires a non-identity view spec"};
  }

  if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED) {
    if (view_resolution.has_view_transform || has_subset || has_ordered_selection || has_metadata_subset) {
      record_error(record_result, "index_kind_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "index_kind CANONICAL cannot be used with view/subset selection"};
    }
    return Status::OK;
  }
  if (!view_resolution.has_view_transform && !has_subset && !has_ordered_selection && !has_metadata_subset) {
    record_error(record_result, "index_kind_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "index_kind VIEW requires view or selection order"};
  }
  return Status::OK;
}

Status resolve_target_view_id(
    std::string_view canonical_json,
    RecordMaterializeResultFn record_result,
    TargetViewResolution& view_resolution) {
  if (!view_resolution.has_view_transform) {
    return Status::OK;
  }
  if (!view_resolution.view_spec_proto.has_value()) {
    record_error(record_result, "view_missing");
    return {StatusCode::INVALID_ARGUMENT, "view spec required for view transforms"};
  }
  auto view_id_or = compute_view_id_from_spec(*view_resolution.view_spec_proto, std::string(canonical_json));
  if (!view_id_or.ok()) {
    return to_grpc_status(view_id_or.status());
  }
  if (view_resolution.request_view_id.has_value() && *view_resolution.request_view_id != *view_id_or) {
    record_error(record_result, "view_id_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "view_id does not match view spec"};
  }
  view_resolution.resolved_view_id = *view_id_or;
  return Status::OK;
}

Status validate_target_layout_view_binding(
    const v2::TargetLayout& layout,
    const TargetViewResolution& view_resolution,
    RecordMaterializeResultFn record_result) {
  if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    if (view_resolution.has_view_transform) {
      if (layout.view_id().empty() ||
          (view_resolution.resolved_view_id.has_value() && layout.view_id() != *view_resolution.resolved_view_id)) {
        record_error(record_result, "view_id_mismatch");
        return {StatusCode::INVALID_ARGUMENT, "target_layout.view_id must match resolved view_id"};
      }
      return Status::OK;
    }
    if (!layout.view_id().empty()) {
      record_error(record_result, "view_id_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout.view_id must be empty for subset-only layouts"};
    }
    return Status::OK;
  }
  if (!layout.view_id().empty()) {
    record_error(record_result, "view_id_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "target_layout.view_id not allowed for canonical layout"};
  }
  return Status::OK;
}

Status choose_target_selected_index_json(
    const v2::TargetLayout& layout,
    std::string_view canonical_index_json,
    const TargetViewResolution& view_resolution,
    RecordMaterializeResultFn record_result,
    std::string& selected_index_json) {
  if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW) {
    if (!view_resolution.view_plan.has_value()) {
      record_error(record_result, "view_plan_missing");
      return {StatusCode::FAILED_PRECONDITION, "view plan missing for VIEW layout"};
    }
    selected_index_json = view_resolution.view_plan->view_index_json;
    return Status::OK;
  }
  selected_index_json = std::string(canonical_index_json);
  return Status::OK;
}

Status validate_target_alias_against_entry(
    const v2::TensorAlias& alias,
    const CanonicalIndexEntry& entry,
    RecordMaterializeResultFn record_result) {
  if (alias.logical_length() != entry.logical_length) {
    record_error(record_result, "layout_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "target_layout alias logical_length mismatch"};
  }
  if (alias.dtype() != entry.dtype) {
    record_error(record_result, "layout_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "target_layout alias dtype mismatch"};
  }
  if (alias.shape_size() != static_cast<int>(entry.shape.size()) ||
      alias.stride_size() != static_cast<int>(entry.stride.size())) {
    record_error(record_result, "layout_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "target_layout alias shape/stride mismatch"};
  }
  for (int i = 0; i < alias.shape_size(); ++i) {
    if (alias.shape(i) != entry.shape[static_cast<size_t>(i)]) {
      record_error(record_result, "layout_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout alias shape mismatch"};
    }
  }
  for (int i = 0; i < alias.stride_size(); ++i) {
    if (alias.stride(i) != entry.stride[static_cast<size_t>(i)]) {
      record_error(record_result, "layout_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout alias stride mismatch"};
    }
  }
  return Status::OK;
}

Status validate_target_layout_aliases(
    const v2::TargetLayout& layout,
    const CanonicalIndexTable& index_table,
    RecordMaterializeResultFn record_result) {
  if (layout.tensor_spec_kind() != v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED) {
    return Status::OK;
  }
  for (const auto& alias : layout.aliases()) {
    auto it = index_table.entries.find(alias.name());
    if (it == index_table.entries.end()) {
      record_error(record_result, "tensor_name_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout alias includes unknown tensor name"};
    }
    auto alias_status = validate_target_alias_against_entry(alias, it->second, record_result);
    if (!alias_status.ok()) {
      return alias_status;
    }
  }
  return Status::OK;
}

} // namespace

Status build_target_materialization_plan(
    store::StoreEngine& engine,
    std::string_view resolved_artifact_id,
    const v2::MaterializeIntoTargetRequest& req,
    const v2::TargetLayout& layout,
    const std::vector<TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    TargetMaterializationPlan& plan) {
  if (!req.has_selection()) {
    record_error(record_result, "selection_missing");
    return {StatusCode::INVALID_ARGUMENT, "selection is required"};
  }
  const auto& selection = req.selection();

  plan = TargetMaterializationPlan{};
  plan.canonical_index_json = std::move(canonical_index_json);

  TargetLayoutTensorSelection tensor_selection;
  auto tensor_selection_status =
      collect_target_layout_tensor_selection(selection, offsets, record_result, tensor_selection);
  if (!tensor_selection_status.ok()) {
    return tensor_selection_status;
  }

  auto canonical_table_or = parse_canonical_index_shared(plan.canonical_index_json);
  if (!canonical_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(canonical_table_or.status());
  }
  const CanonicalIndexTable& canonical_table = **canonical_table_or;

  auto canonical_layout_validation_status = validate_layout_names_exist_in_index(
      tensor_selection.layout_name_set, canonical_table, "target_layout includes unknown tensor name", record_result);
  if (!canonical_layout_validation_status.ok()) {
    return canonical_layout_validation_status;
  }
  plan.has_subset = is_layout_subset_of_canonical_index(canonical_table, tensor_selection.layout_name_set);
  const bool has_ordered_selection = !tensor_selection.request_names.empty();

  TargetViewResolution view_resolution;
  auto parse_view_status = parse_target_view_identity(selection, view_resolution);
  if (!parse_view_status.ok()) {
    return parse_view_status;
  }
  auto hydrate_view_status =
      hydrate_target_view_from_metadata(engine, resolved_artifact_id, record_result, view_resolution);
  if (!hydrate_view_status.ok()) {
    return hydrate_view_status;
  }
  auto view_plan_status = compute_target_view_plan(
      plan.canonical_index_json,
      plan.has_subset,
      has_ordered_selection,
      tensor_selection.layout_names,
      tensor_selection.request_names,
      layout.index_kind(),
      record_result,
      view_resolution);
  if (!view_plan_status.ok()) {
    return view_plan_status;
  }
  auto index_kind_status =
      validate_target_index_kind_mode(layout, plan.has_subset, has_ordered_selection, record_result, view_resolution);
  if (!index_kind_status.ok()) {
    return index_kind_status;
  }
  auto resolve_view_id_status = resolve_target_view_id(plan.canonical_index_json, record_result, view_resolution);
  if (!resolve_view_id_status.ok()) {
    return resolve_view_id_status;
  }
  auto layout_view_binding_status = validate_target_layout_view_binding(layout, view_resolution, record_result);
  if (!layout_view_binding_status.ok()) {
    return layout_view_binding_status;
  }

  auto selected_index_status = choose_target_selected_index_json(
      layout, plan.canonical_index_json, view_resolution, record_result, plan.selected_index_json);
  if (!selected_index_status.ok()) {
    return selected_index_status;
  }

  auto index_table_or = parse_canonical_index_shared(plan.selected_index_json);
  if (!index_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(index_table_or.status());
  }
  const CanonicalIndexTable& index_table = **index_table_or;
  plan.logical_total_size = index_table.logical_total_size;

  auto selected_index_validation_status =
      validate_selected_index_matches_layout(tensor_selection.layout_name_set, index_table, record_result);
  if (!selected_index_validation_status.ok()) {
    return selected_index_validation_status;
  }
  if (view_resolution.view_plan.has_value() && view_resolution.view_plan->view_size_bytes > 0 &&
      view_resolution.view_plan->view_size_bytes != plan.logical_total_size) {
    record_error(record_result, "layout_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "view plan size does not match selected index"};
  }

  auto alias_validation_status = validate_target_layout_aliases(layout, index_table, record_result);
  if (!alias_validation_status.ok()) {
    return alias_validation_status;
  }

  PreparedTargetStorageLayout prepared_storage_layout;
  auto storage_layout_status = build_target_storage_layout(layout, record_result, prepared_storage_layout);
  if (!storage_layout_status.ok()) {
    return storage_layout_status;
  }
  auto storage_span_status = validate_storage_span_matches_index(
      prepared_storage_layout.total_storage_bytes, plan.logical_total_size, record_result);
  if (!storage_span_status.ok()) {
    return storage_span_status;
  }
  auto offset_validation_status = validate_target_offsets_against_layout(
      offsets, index_table, prepared_storage_layout.storage_ranges, record_result);
  if (!offset_validation_status.ok()) {
    return offset_validation_status;
  }

  std::string_view selection_error_reason;
  auto subset_hash_status = selection_validation::compute_and_validate_view_subset_hash(
      selection,
      (plan.has_subset || has_ordered_selection)
          ? absl::MakeSpan(has_ordered_selection ? tensor_selection.request_names : tensor_selection.layout_names)
          : absl::Span<const std::string>(),
      plan.view_subset_hash,
      &selection_error_reason);
  if (!subset_hash_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return subset_hash_status;
  }

  const std::vector<std::string> empty_selection_names;
  const std::vector<std::string>& resolved_selection_names = (plan.has_subset || has_ordered_selection)
      ? (has_ordered_selection ? tensor_selection.request_names : tensor_selection.layout_names)
      : empty_selection_names;

  const std::string resolved_view_id = view_resolution.resolved_view_id.value_or("");
  const bool needs_view_index = layout.index_kind() == v2::TargetLayout::INDEX_KIND_VIEW;
  const tensorcast::common::v1::ViewSpec* resolved_view_spec =
      (view_resolution.has_view_transform && view_resolution.view_spec_proto.has_value())
      ? &*view_resolution.view_spec_proto
      : nullptr;
  auto selection_identity_status = selection_validation::validate_hashes_and_build_resolved_selection(
      selection,
      resolved_artifact_id,
      resolved_view_id,
      plan.selected_index_json,
      needs_view_index,
      resolved_selection_names,
      plan.view_subset_hash,
      resolved_view_spec,
      plan.resolved_selection,
      &selection_error_reason);
  if (!selection_identity_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return selection_identity_status;
  }

  plan.layout_names = std::move(tensor_selection.layout_names);
  plan.view_spec = std::move(view_resolution.view_spec);
  plan.view_data_hash = std::move(view_resolution.view_data_hash);
  plan.view_plan = std::move(view_resolution.view_plan);
  plan.resolved_view_id = std::move(view_resolution.resolved_view_id);
  plan.has_view_transform = view_resolution.has_view_transform;
  plan.publish_storages = std::move(prepared_storage_layout.publish_storages);
  plan.publish_segments = std::move(prepared_storage_layout.publish_segments);
  return Status::OK;
}

Status build_mapped_target_materialization_plan(
    store::StoreEngine& engine,
    const v2::MaterializeIntoMappedTargetRequest& req,
    std::string_view resolved_artifact_id,
    const std::vector<TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    MappedTargetMaterializationPlan& plan) {
  if (!req.has_selection()) {
    record_error(record_result, "selection_missing");
    return {StatusCode::INVALID_ARGUMENT, "selection is required"};
  }
  const auto& selection = req.selection();
  if (selection.artifact_id().empty()) {
    record_error(record_result, "selection_missing");
    return {StatusCode::INVALID_ARGUMENT, "selection.artifact_id is required"};
  }

  plan = MappedTargetMaterializationPlan{};
  plan.canonical_index_json = std::move(canonical_index_json);

  std::vector<std::string> ordered_selection_names;
  ordered_selection_names.reserve(selection.tensor_names_size());
  absl::flat_hash_set<std::string> selection_name_set;
  selection_name_set.reserve(selection.tensor_names_size());
  for (const auto& name : selection.tensor_names()) {
    ordered_selection_names.push_back(name);
    selection_name_set.insert(name);
  }
  if (selection_name_set.size() != static_cast<size_t>(selection.tensor_names_size())) {
    record_error(record_result, "tensor_name_duplicate");
    return {StatusCode::INVALID_ARGUMENT, "selection.tensor_names must not contain duplicates"};
  }
  std::string view_subset_hash;
  std::string_view selection_error_reason;
  auto subset_hash_status = selection_validation::compute_and_validate_view_subset_hash(
      selection, absl::MakeSpan(ordered_selection_names), view_subset_hash, &selection_error_reason);
  if (!subset_hash_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return subset_hash_status;
  }

  ValidationErrorReason mapped_layout_reason = ValidationErrorReason::kUnknown;
  auto mapped_layout_or = validate_mapped_target_layout(req, offsets, &mapped_layout_reason);
  if (!mapped_layout_or.ok()) {
    record_error(record_result, validation_error_reason(mapped_layout_reason));
    return to_grpc_status(mapped_layout_or.status());
  }
  ValidatedMappedTargetLayout mapped_layout = std::move(*mapped_layout_or);
  plan.logical_total_size = mapped_layout.logical_total_size;

  ResolveViewSpecErrorReason resolve_view_reason = ResolveViewSpecErrorReason::kUnknown;
  auto resolved_view_or = resolve_mapped_view_spec(req, resolved_artifact_id, engine, &resolve_view_reason);
  if (!resolved_view_or.ok()) {
    record_error(record_result, resolve_view_spec_error_reason(resolve_view_reason));
    return to_grpc_status(resolved_view_or.status());
  }
  auto resolved_view = std::move(*resolved_view_or);
  plan.view_spec = std::move(resolved_view.view_spec);
  std::optional<std::string> request_view_id = std::move(resolved_view.request_view_id);
  std::optional<tensorcast::common::v1::ViewSpec> view_spec_proto;
  std::optional<std::string> resolved_view_id;

  ViewNarrowErrorReason view_narrow_reason = ViewNarrowErrorReason::kUnknown;
  auto view_narrows_or = build_view_narrows(plan.view_spec, &view_narrow_reason);
  if (!view_narrows_or.ok()) {
    record_error(record_result, view_narrow_error_reason(view_narrow_reason));
    return to_grpc_status(view_narrows_or.status());
  }
  absl::flat_hash_map<std::string, ViewNarrowSpec> view_narrows = std::move(*view_narrows_or);

  if (plan.view_spec.has_value()) {
    view_spec_proto = build_view_spec_proto(*plan.view_spec);
    auto view_plan_or = store::StoreEngine::compute_view_plan(plan.canonical_index_json, *plan.view_spec);
    if (!view_plan_or.ok()) {
      record_error(record_result, "view_plan_failed");
      return to_grpc_status(view_plan_or.status());
    }
    plan.view_plan = std::move(*view_plan_or);
    if (plan.view_plan->is_identity) {
      if (request_view_id.has_value()) {
        record_error(record_result, "view_identity_mismatch");
        return {StatusCode::INVALID_ARGUMENT, "selection.view_id requires a non-identity view spec"};
      }
    } else {
      auto view_id_or = compute_view_id_from_spec(*view_spec_proto, plan.canonical_index_json);
      if (!view_id_or.ok()) {
        return to_grpc_status(view_id_or.status());
      }
      if (request_view_id.has_value() && *request_view_id != *view_id_or) {
        record_error(record_result, "view_id_mismatch");
        return {StatusCode::INVALID_ARGUMENT, "selection.view_id does not match selection.view_spec"};
      }
      resolved_view_id = *view_id_or;
    }
  } else if (request_view_id.has_value()) {
    // For mapped targets, selection.view_id may identify the target byte-space
    // even when the source artifact has no metadata-backed view spec. Runtime
    // source selection still decides whether those bytes arrive directly from a
    // view-capable source or are reconstructed from canonical/disk fallback.
    resolved_view_id = *request_view_id;
  }

  std::string source_index_json = plan.canonical_index_json;
  if (plan.view_plan.has_value() && !plan.view_plan->is_identity) {
    source_index_json = plan.view_plan->view_index_json;
  }
  auto target_index_json_or = build_mapped_target_selected_index_json(mapped_layout);
  if (!target_index_json_or.ok()) {
    record_error(record_result, "layout_mismatch");
    return to_grpc_status(target_index_json_or.status());
  }
  // Mapped copy-plan execution reads from source_index_json above, but
  // resolved_selection must describe the packed target byte-space that this RPC
  // materializes and later publishes.
  plan.selected_index_json = std::move(*target_index_json_or);
  auto canonical_source_table_or = parse_canonical_index_shared(plan.canonical_index_json);
  if (!canonical_source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(canonical_source_table_or.status());
  }
  auto source_table_or = source_index_json == plan.canonical_index_json
      ? canonical_source_table_or
      : parse_canonical_index_shared(source_index_json);
  if (!source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(source_table_or.status());
  }
  const CanonicalIndexTable& canonical_source_table = **canonical_source_table_or;
  const CanonicalIndexTable& source_table = **source_table_or;

  auto representation_or = build_representation_transform_contract(
      req.copy_plan(),
      source_table,
      canonical_source_table,
      mapped_layout.dst_specs,
      mapped_layout.dst_base_offsets,
      view_narrows,
      build_source_byte_space(
          resolved_view_id.has_value() ? std::optional<std::string_view>(*resolved_view_id) : std::nullopt),
      "ephemeral_into_target");
  if (!representation_or.ok()) {
    record_error(record_result, "mapping_invalid");
    return to_grpc_status(representation_or.status());
  }
  plan.representation = std::move(*representation_or);
  plan.representation.generic_fallback_map.total_bytes = plan.logical_total_size;
  LOG(INFO) << "MaterializeIntoMappedTarget representation-work lowering"
            << " copy_entries=" << req.copy_plan().entries_size() << " dst_tensors=" << mapped_layout.dst_specs.size()
            << " collective_candidates=" << plan.representation.lowering_stats.collective_candidates
            << " collective_bytes=" << plan.representation.lowering_stats.collective_bytes
            << " concat_candidates=" << plan.representation.lowering_stats.concat_candidates
            << " concat_bytes=" << plan.representation.lowering_stats.concat_bytes
            << " rejected_mixed_src_or_dim=" << plan.representation.lowering_stats.rejected_mixed_src_or_dim
            << " rejected_mixed_src_or_dim_bytes=" << plan.representation.lowering_stats.rejected_mixed_src_or_dim_bytes
            << " rejected_non_contiguous=" << plan.representation.lowering_stats.rejected_non_contiguous
            << " rejected_non_contiguous_bytes=" << plan.representation.lowering_stats.rejected_non_contiguous_bytes
            << " rejected_unsupported_distribution="
            << plan.representation.lowering_stats.rejected_unsupported_distribution
            << " rejected_unsupported_distribution_bytes="
            << plan.representation.lowering_stats.rejected_unsupported_distribution_bytes
            << " representation_contract_hash="
            << plan.representation.transform_contract.target_representation.representation_contract_hash;

  PreparedTargetStorageLayout prepared_storage_layout;
  auto storage_layout_status = build_target_storage_layout(req.target_layout(), record_result, prepared_storage_layout);
  if (!storage_layout_status.ok()) {
    return storage_layout_status;
  }
  auto storage_span_status = validate_storage_span_matches_index(
      prepared_storage_layout.total_storage_bytes, plan.logical_total_size, record_result);
  if (!storage_span_status.ok()) {
    return storage_span_status;
  }

  const bool needs_view_index = resolved_view_id.has_value();
  const tensorcast::common::v1::ViewSpec* resolved_view_spec =
      (plan.view_plan.has_value() && !plan.view_plan->is_identity && view_spec_proto.has_value()) ? &*view_spec_proto
                                                                                                  : nullptr;
  auto selection_identity_status = selection_validation::validate_hashes_and_build_resolved_selection(
      selection,
      resolved_artifact_id,
      resolved_view_id.value_or(""),
      plan.selected_index_json,
      needs_view_index,
      ordered_selection_names,
      view_subset_hash,
      resolved_view_spec,
      plan.resolved_selection,
      &selection_error_reason);
  if (!selection_identity_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return selection_identity_status;
  }
  plan.publish_storages = std::move(prepared_storage_layout.publish_storages);
  plan.publish_segments = std::move(prepared_storage_layout.publish_segments);
  return Status::OK;
}

Status build_binding_realization_materialization_plan(
    store::StoreEngine& engine,
    const tensorcast::common::v1::ArtifactSelection& selection,
    const v2::BindingRealizationPlan& realization_plan,
    std::string_view resolved_artifact_id,
    const v2::TargetLayout& target_layout,
    std::string_view target_index_json,
    const std::vector<TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    BindingRealizationMaterializationPlanOptions options,
    MappedTargetMaterializationPlan& plan) {
  const auto profile_start = std::chrono::steady_clock::now();
  auto elapsed_sec = [](auto start, auto end) { return std::chrono::duration<double>(end - start).count(); };
  double selection_names_sec = 0.0;
  double subset_hash_sec = 0.0;
  double storage_layout_sec = 0.0;
  double mapped_layout_sec = 0.0;
  double parse_target_index_sec = 0.0;
  double validate_target_layout_sec = 0.0;
  double resolve_view_sec = 0.0;
  double view_narrows_sec = 0.0;
  double view_plan_sec = 0.0;
  double parse_source_tables_sec = 0.0;
  bool source_table_reused = false;
  double representation_contract_sec = 0.0;
  double selection_identity_sec = 0.0;
  const bool build_byte_range_maps = options.build_byte_range_maps;

  if (selection.artifact_id().empty()) {
    record_error(record_result, "selection_missing");
    return {StatusCode::INVALID_ARGUMENT, "selection.artifact_id is required"};
  }
  if (realization_plan.entries_size() == 0) {
    record_error(record_result, "mapping_invalid");
    return {StatusCode::INVALID_ARGUMENT, "realization_plan.entries must be non-empty"};
  }

  plan = MappedTargetMaterializationPlan{};
  plan.canonical_index_json = std::move(canonical_index_json);
  plan.selected_index_json = std::string(target_index_json);

  std::vector<std::string> ordered_selection_names;
  ordered_selection_names.reserve(selection.tensor_names_size());
  std::string_view selection_error_reason;
  const auto selection_names_start = std::chrono::steady_clock::now();
  auto tensor_name_status =
      selection_validation::validate_request_tensor_names(selection, ordered_selection_names, &selection_error_reason);
  selection_names_sec = elapsed_sec(selection_names_start, std::chrono::steady_clock::now());
  if (!tensor_name_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return tensor_name_status;
  }
  std::string view_subset_hash;
  const auto subset_hash_start = std::chrono::steady_clock::now();
  auto subset_hash_status = selection_validation::compute_and_validate_view_subset_hash(
      selection, absl::MakeSpan(ordered_selection_names), view_subset_hash, &selection_error_reason);
  subset_hash_sec = elapsed_sec(subset_hash_start, std::chrono::steady_clock::now());
  if (!subset_hash_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return subset_hash_status;
  }

  PreparedTargetStorageLayout prepared_storage_layout;
  const auto storage_layout_start = std::chrono::steady_clock::now();
  auto storage_layout_status = build_target_storage_layout(target_layout, record_result, prepared_storage_layout);
  storage_layout_sec = elapsed_sec(storage_layout_start, std::chrono::steady_clock::now());
  if (!storage_layout_status.ok()) {
    return storage_layout_status;
  }

  const auto mapped_layout_start = std::chrono::steady_clock::now();
  auto mapped_layout_or =
      build_mapped_layout_from_target_index(plan.selected_index_json, offsets, prepared_storage_layout.storage_ranges);
  mapped_layout_sec = elapsed_sec(mapped_layout_start, std::chrono::steady_clock::now());
  if (!mapped_layout_or.ok()) {
    record_error(record_result, "layout_mismatch");
    return to_grpc_status(mapped_layout_or.status());
  }
  const auto parse_target_index_start = std::chrono::steady_clock::now();
  auto target_index_table_or = parse_canonical_index_shared(plan.selected_index_json);
  parse_target_index_sec = elapsed_sec(parse_target_index_start, std::chrono::steady_clock::now());
  if (!target_index_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(target_index_table_or.status());
  }
  const CanonicalIndexTable& target_index_table = **target_index_table_or;
  plan.logical_total_size = target_index_table.logical_total_size;
  const auto validate_target_layout_start = std::chrono::steady_clock::now();
  auto storage_span_status = validate_storage_span_matches_index(
      prepared_storage_layout.total_storage_bytes, plan.logical_total_size, record_result);
  if (!storage_span_status.ok()) {
    return storage_span_status;
  }
  auto offset_validation_status = validate_target_offsets_against_layout(
      offsets, target_index_table, prepared_storage_layout.storage_ranges, record_result);
  if (!offset_validation_status.ok()) {
    return offset_validation_status;
  }
  validate_target_layout_sec = elapsed_sec(validate_target_layout_start, std::chrono::steady_clock::now());

  v2::MaterializeIntoMappedTargetRequest view_request;
  view_request.mutable_selection()->CopyFrom(selection);
  ResolveViewSpecErrorReason resolve_view_reason = ResolveViewSpecErrorReason::kUnknown;
  const auto resolve_view_start = std::chrono::steady_clock::now();
  auto resolved_view_or = resolve_mapped_view_spec(view_request, resolved_artifact_id, engine, &resolve_view_reason);
  resolve_view_sec = elapsed_sec(resolve_view_start, std::chrono::steady_clock::now());
  if (!resolved_view_or.ok()) {
    record_error(record_result, resolve_view_spec_error_reason(resolve_view_reason));
    return to_grpc_status(resolved_view_or.status());
  }
  auto resolved_view = std::move(*resolved_view_or);
  plan.view_spec = std::move(resolved_view.view_spec);
  std::optional<std::string> request_view_id = std::move(resolved_view.request_view_id);
  std::optional<tensorcast::common::v1::ViewSpec> view_spec_proto;
  std::optional<std::string> resolved_view_id;

  ViewNarrowErrorReason view_narrow_reason = ViewNarrowErrorReason::kUnknown;
  const auto view_narrows_start = std::chrono::steady_clock::now();
  auto view_narrows_or = build_view_narrows(plan.view_spec, &view_narrow_reason);
  view_narrows_sec = elapsed_sec(view_narrows_start, std::chrono::steady_clock::now());
  if (!view_narrows_or.ok()) {
    record_error(record_result, view_narrow_error_reason(view_narrow_reason));
    return to_grpc_status(view_narrows_or.status());
  }
  const auto view_narrows = std::move(*view_narrows_or);

  if (plan.view_spec.has_value()) {
    const auto view_plan_start = std::chrono::steady_clock::now();
    view_spec_proto = build_view_spec_proto(*plan.view_spec);
    auto view_plan_or = store::StoreEngine::compute_view_plan(plan.canonical_index_json, *plan.view_spec);
    if (!view_plan_or.ok()) {
      record_error(record_result, "view_plan_failed");
      return to_grpc_status(view_plan_or.status());
    }
    plan.view_plan = std::move(*view_plan_or);
    if (plan.view_plan->is_identity) {
      if (request_view_id.has_value()) {
        record_error(record_result, "view_identity_mismatch");
        return {StatusCode::INVALID_ARGUMENT, "selection.view_id requires a non-identity view spec"};
      }
    } else {
      auto view_id_or = compute_view_id_from_spec(*view_spec_proto, plan.canonical_index_json);
      if (!view_id_or.ok()) {
        return to_grpc_status(view_id_or.status());
      }
      if (request_view_id.has_value() && *request_view_id != *view_id_or) {
        record_error(record_result, "view_id_mismatch");
        return {StatusCode::INVALID_ARGUMENT, "selection.view_id does not match selection.view_spec"};
      }
      resolved_view_id = *view_id_or;
    }
    view_plan_sec = elapsed_sec(view_plan_start, std::chrono::steady_clock::now());
  } else if (request_view_id.has_value()) {
    resolved_view_id = *request_view_id;
  }

  std::string source_index_json = plan.canonical_index_json;
  if (plan.view_plan.has_value() && !plan.view_plan->is_identity) {
    source_index_json = plan.view_plan->view_index_json;
  }
  const auto parse_source_tables_start = std::chrono::steady_clock::now();
  auto canonical_source_table_or =
      options.canonical_index_parse_identity_key.has_value() && !options.canonical_index_parse_identity_key->empty()
      ? parse_canonical_index_shared_with_identity(
            plan.canonical_index_json, *options.canonical_index_parse_identity_key)
      : parse_canonical_index_shared(plan.canonical_index_json);
  if (!canonical_source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(canonical_source_table_or.status());
  }
  source_table_reused = source_index_json == plan.canonical_index_json;
  auto source_table_or =
      source_table_reused ? canonical_source_table_or : parse_canonical_index_shared(source_index_json);
  if (!source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(source_table_or.status());
  }
  const CanonicalIndexTable& canonical_source_table = **canonical_source_table_or;
  const CanonicalIndexTable& source_table = **source_table_or;
  parse_source_tables_sec = elapsed_sec(parse_source_tables_start, std::chrono::steady_clock::now());

  const auto representation_contract_start = std::chrono::steady_clock::now();
  auto representation_or = build_representation_transform_contract(
      realization_plan,
      source_table,
      canonical_source_table,
      mapped_layout_or->dst_specs,
      mapped_layout_or->dst_base_offsets,
      view_narrows,
      build_source_byte_space(
          resolved_view_id.has_value() ? std::optional<std::string_view>(*resolved_view_id) : std::nullopt),
      "local_seal_then_promote",
      representation_transform_builder::BuildRepresentationTransformOptions{
          .compute_identity_hashes = false,
          .build_byte_range_maps = build_byte_range_maps,
      });
  if (!representation_or.ok()) {
    record_error(record_result, "mapping_invalid");
    return to_grpc_status(representation_or.status());
  }
  plan.representation = std::move(*representation_or);
  representation_contract_sec = elapsed_sec(representation_contract_start, std::chrono::steady_clock::now());

  tensorcast::common::v1::ArtifactSelection validated_source_selection;
  const auto selection_identity_start = std::chrono::steady_clock::now();
  auto selection_identity_status = selection_validation::validate_hashes_and_build_resolved_selection(
      selection,
      resolved_artifact_id,
      resolved_view_id.value_or(""),
      source_index_json,
      resolved_view_id.has_value(),
      ordered_selection_names,
      view_subset_hash,
      (plan.view_plan.has_value() && !plan.view_plan->is_identity && view_spec_proto.has_value()) ? &*view_spec_proto
                                                                                                  : nullptr,
      validated_source_selection,
      &selection_error_reason);
  if (!selection_identity_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return selection_identity_status;
  }
  selection_identity_sec = elapsed_sec(selection_identity_start, std::chrono::steady_clock::now());

  plan.publish_storages = std::move(prepared_storage_layout.publish_storages);
  plan.publish_segments = std::move(prepared_storage_layout.publish_segments);
  const auto done = std::chrono::steady_clock::now();
  LOG(INFO) << "tc_profile binding_realization_plan timings"
            << " artifact_id=" << resolved_artifact_id << " entries=" << realization_plan.entries_size()
            << " target_tensors=" << mapped_layout_or->dst_specs.size()
            << " selected_index_bytes=" << plan.selected_index_json.size()
            << " canonical_index_bytes=" << plan.canonical_index_json.size()
            << " build_byte_range_maps=" << (build_byte_range_maps ? 1 : 0) << " canonical_index_identity_parse_key="
            << (options.canonical_index_parse_identity_key.has_value() &&
                        !options.canonical_index_parse_identity_key->empty()
                    ? 1
                    : 0)
            << " selection_names_sec=" << selection_names_sec << " subset_hash_sec=" << subset_hash_sec
            << " storage_layout_sec=" << storage_layout_sec << " mapped_layout_sec=" << mapped_layout_sec
            << " parse_target_index_sec=" << parse_target_index_sec
            << " validate_target_layout_sec=" << validate_target_layout_sec << " resolve_view_sec=" << resolve_view_sec
            << " view_narrows_sec=" << view_narrows_sec << " view_plan_sec=" << view_plan_sec
            << " parse_source_tables_sec=" << parse_source_tables_sec
            << " source_table_reused=" << (source_table_reused ? 1 : 0)
            << " representation_contract_sec=" << representation_contract_sec
            << " selection_identity_sec=" << selection_identity_sec
            << " collective_segments=" << plan.representation.collective_lowered_map.segments.size()
            << " fallback_segments=" << plan.representation.generic_fallback_map.segments.size()
            << " tensor_bindings=" << plan.representation.transform_contract.tensor_bindings.size()
            << " total_sec=" << elapsed_sec(profile_start, done);
  return Status::OK;
}

Status build_binding_realization_materialization_plan(
    store::StoreEngine& engine,
    const tensorcast::common::v1::ArtifactSelection& selection,
    const v2::BindingRealizationPlan& realization_plan,
    std::string_view resolved_artifact_id,
    const v2::TargetLayout& target_layout,
    std::string_view target_index_json,
    const std::vector<TargetOffsetEntry>& offsets,
    std::string canonical_index_json,
    RecordMaterializeResultFn record_result,
    MappedTargetMaterializationPlan& plan) {
  return build_binding_realization_materialization_plan(
      engine,
      selection,
      realization_plan,
      resolved_artifact_id,
      target_layout,
      target_index_json,
      offsets,
      std::move(canonical_index_json),
      record_result,
      BindingRealizationMaterializationPlanOptions{},
      plan);
}

absl::StatusOr<store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan>
build_resolved_mapped_materialization_plan(
    std::string_view resolved_artifact_id,
    uint64_t generation,
    const store::loading::IntoTargetLayout& target_layout,
    const MappedTargetMaterializationPlan& mapped_plan,
    const std::optional<store::loading::VariantIdentity>& variant,
    std::optional<std::string_view> source_index_json,
    std::shared_ptr<const CanonicalIndexTable> physical_source_table,
    ResolvedMappedMaterializationPlanOptions options) {
  const auto profile_start = std::chrono::steady_clock::now();
  auto elapsed_sec = [](auto start, auto end) { return std::chrono::duration<double>(end - start).count(); };
  double parse_source_index_sec = 0.0;
  double init_plan_sec = 0.0;
  double work_plan_sec = 0.0;
  bool work_plan_source_rebind_fast_path = false;
  double filter_collective_candidate_sec = 0.0;
  double collective_ranges_sec = 0.0;
  double filter_collective_sec = 0.0;
  double filter_generic_sec = 0.0;
  double merge_executor_sec = 0.0;
  bool generic_reused_collective_map = false;
  bool executor_merge_skipped = false;
  bool compact_coverage_map_used = false;
  uint64_t compact_coverage_data_bytes = 0;
  size_t compact_coverage_segments = 0;
  bool compact_collective_coverage_map_used = false;
  uint64_t compact_collective_coverage_data_bytes = 0;
  size_t compact_collective_coverage_segments = 0;
  bool source_window_coverage_proof_map_used = false;
  uint64_t collective_candidate_data_bytes = 0;

  using PreparedExecutionPlan = store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan;
  using StrategyPlan = store::runtime::ingestion::strategy::ResolvedMaterializationPlan;

  const bool physical_source_table_preparsed = physical_source_table != nullptr;
  if (physical_source_table == nullptr && source_index_json.has_value()) {
    const auto parse_source_index_start = std::chrono::steady_clock::now();
    auto physical_source_table_or = parse_canonical_index_shared(*source_index_json);
    parse_source_index_sec = elapsed_sec(parse_source_index_start, std::chrono::steady_clock::now());
    if (!physical_source_table_or.ok()) {
      return physical_source_table_or.status();
    }
    physical_source_table = std::move(*physical_source_table_or);
  }

  PreparedExecutionPlan prepared_execution;
  const auto init_plan_start = std::chrono::steady_clock::now();
  StrategyPlan& resolved_plan = prepared_execution.resolved_plan;
  resolved_plan.artifact_id = std::string(resolved_artifact_id);
  resolved_plan.generation = generation;
  resolved_plan.variant = variant;
  resolved_plan.canonical_index_json = mapped_plan.canonical_index_json;
  resolved_plan.target_layout = target_layout;
  resolved_plan.representation_transform_contract = mapped_plan.representation.transform_contract;
  init_plan_sec = elapsed_sec(init_plan_start, std::chrono::steady_clock::now());
  if (resolved_plan.representation_transform_contract.has_value()) {
    prepared_execution.lowering_artifacts = store::runtime::ingestion::strategy::SourceBoundLoweringArtifacts{
        .lowering_stats = to_source_bound_lowering_stats(mapped_plan.representation.lowering_stats),
    };
    const store::loader::ByteRangeMap collective_candidate_map = [&]() {
      if (!mapped_plan.representation.collective_lowered_map.segments.empty()) {
        auto map = mapped_plan.representation.collective_lowered_map;
        if (map.total_bytes == 0) {
          map.total_bytes = target_layout.total_size > 0 ? target_layout.total_size : mapped_plan.logical_total_size;
        }
        return map;
      }
      auto map = mapped_plan.representation.generic_fallback_map;
      if (!map.segments.empty() && map.total_bytes == 0) {
        map.total_bytes = target_layout.total_size > 0 ? target_layout.total_size : mapped_plan.logical_total_size;
      }
      return map;
    }();
    const store::loader::ByteRangeMap generic_fallback_map = [&]() {
      auto map = mapped_plan.representation.generic_fallback_map;
      if (!map.segments.empty() && map.total_bytes == 0) {
        map.total_bytes = target_layout.total_size > 0 ? target_layout.total_size : mapped_plan.logical_total_size;
      }
      return map;
    }();
    const auto work_plan_start = std::chrono::steady_clock::now();
    auto work_plan_or = build_execution_representation_work_plan(
        *resolved_plan.representation_transform_contract,
        physical_source_table.get(),
        collective_candidate_map.segments.empty()
            ? std::nullopt
            : std::optional<store::loader::ByteRangeMap>(collective_candidate_map),
        &work_plan_source_rebind_fast_path);
    work_plan_sec = elapsed_sec(work_plan_start, std::chrono::steady_clock::now());
    if (!work_plan_or.ok()) {
      return work_plan_or.status();
    }
    resolved_plan.representation_work_plan = std::move(*work_plan_or);
    const bool byte_range_maps_omitted = collective_candidate_map.segments.empty() &&
        generic_fallback_map.segments.empty() && mapped_plan.representation.total_bytes_copied > 0;
    if (byte_range_maps_omitted && resolved_plan.representation_work_plan->residual_fallback_map.segments.empty()) {
      const uint64_t coverage_total_bytes =
          target_layout.total_size > 0 ? target_layout.total_size : mapped_plan.logical_total_size;
      if (options.source_window_strict_coverage_proof_only) {
        auto proof_map_or = build_compact_data_coverage_map_from_work_plan(
            *resolved_plan.representation_work_plan, coverage_total_bytes);
        if (!proof_map_or.ok()) {
          return proof_map_or.status();
        }
        if (!proof_map_or->segments.empty()) {
          source_window_coverage_proof_map_used = true;
          compact_coverage_map_used = true;
          compact_coverage_segments = proof_map_or->segments.size();
          compact_coverage_data_bytes = byte_range_map_covered_bytes(*proof_map_or);
          prepared_execution.lowering_artifacts->executor_generic_data_map = *proof_map_or;
          prepared_execution.lowering_artifacts->executor_generic_data_map_coverage_only = true;
          executor_merge_skipped = true;
        }
      } else {
        auto compact_map_or = build_compact_data_coverage_map_from_work_plan(
            *resolved_plan.representation_work_plan, coverage_total_bytes);
        if (!compact_map_or.ok()) {
          return compact_map_or.status();
        }
        auto compact_collective_map_or = build_compact_data_coverage_map_from_work_plan(
            *resolved_plan.representation_work_plan, coverage_total_bytes, work_item_has_collective_source_overlap);
        if (!compact_collective_map_or.ok()) {
          return compact_collective_map_or.status();
        }
        if (!compact_collective_map_or->segments.empty()) {
          compact_collective_coverage_map_used = true;
          compact_collective_coverage_segments = compact_collective_map_or->segments.size();
          compact_collective_coverage_data_bytes = byte_range_map_covered_bytes(*compact_collective_map_or);
          collective_candidate_data_bytes = compact_collective_coverage_data_bytes;
          prepared_execution.lowering_artifacts->collective_data_map = *compact_collective_map_or;
        }
        if (!compact_map_or->segments.empty()) {
          compact_coverage_map_used = true;
          compact_coverage_segments = compact_map_or->segments.size();
          compact_coverage_data_bytes = byte_range_map_covered_bytes(*compact_map_or);
          prepared_execution.lowering_artifacts->executor_generic_data_map = *compact_map_or;
          prepared_execution.lowering_artifacts->executor_generic_data_map_coverage_only = true;
          executor_merge_skipped = true;
        }
      }
      if (source_window_coverage_proof_map_used) {
        prepared_execution.lowering_artifacts->executor_generic_data_map_coverage_only = true;
        executor_merge_skipped = true;
      }
    } else {
      const auto filter_collective_candidate_start = std::chrono::steady_clock::now();
      auto collective_candidate_data_map_or =
          filter_byte_range_map(collective_candidate_map, store::loader::ByteRangeSegment::Kind::kData);
      filter_collective_candidate_sec =
          elapsed_sec(filter_collective_candidate_start, std::chrono::steady_clock::now());
      if (!collective_candidate_data_map_or.ok()) {
        return collective_candidate_data_map_or.status();
      }
      if (!collective_candidate_data_map_or->segments.empty()) {
        collective_candidate_data_bytes = byte_range_map_covered_bytes(*collective_candidate_data_map_or);
        const auto collective_ranges_start = std::chrono::steady_clock::now();
        auto collective_dst_ranges_or = build_collective_dst_ranges(*resolved_plan.representation_work_plan);
        collective_ranges_sec = elapsed_sec(collective_ranges_start, std::chrono::steady_clock::now());
        if (!collective_dst_ranges_or.ok()) {
          return collective_dst_ranges_or.status();
        }
        const auto filter_collective_start = std::chrono::steady_clock::now();
        auto collective_data_map_or =
            filter_data_map_to_dst_ranges(*collective_candidate_data_map_or, *collective_dst_ranges_or);
        filter_collective_sec = elapsed_sec(filter_collective_start, std::chrono::steady_clock::now());
        if (!collective_data_map_or.ok()) {
          return collective_data_map_or.status();
        }
        if (!collective_data_map_or->segments.empty()) {
          prepared_execution.lowering_artifacts->collective_data_map = *collective_data_map_or;
        }
      }
      store::loader::ByteRangeMap collective_candidate_data_map;
      if (!collective_candidate_data_map_or->segments.empty()) {
        collective_candidate_data_map = *collective_candidate_data_map_or;
      }
      store::loader::ByteRangeMap generic_data_map;
      if (!collective_candidate_data_map.segments.empty() &&
          collective_candidate_data_bytes >= mapped_plan.representation.total_bytes_copied) {
        generic_data_map = collective_candidate_data_map;
        generic_reused_collective_map = true;
      } else {
        const auto filter_generic_start = std::chrono::steady_clock::now();
        auto generic_data_map_or =
            filter_byte_range_map(generic_fallback_map, store::loader::ByteRangeSegment::Kind::kData);
        filter_generic_sec = elapsed_sec(filter_generic_start, std::chrono::steady_clock::now());
        if (!generic_data_map_or.ok()) {
          return generic_data_map_or.status();
        }
        if (!generic_data_map_or->segments.empty()) {
          generic_data_map = *generic_data_map_or;
        } else if (!collective_candidate_data_map.segments.empty()) {
          generic_data_map = collective_candidate_data_map;
        }
      }
      if (resolved_plan.representation_work_plan->residual_fallback_map.segments.empty()) {
        executor_merge_skipped = true;
        if (!generic_data_map.segments.empty()) {
          prepared_execution.lowering_artifacts->executor_generic_data_map = generic_data_map;
        }
      } else {
        const auto merge_executor_start = std::chrono::steady_clock::now();
        auto executor_map_or =
            merge_byte_range_maps(generic_data_map, resolved_plan.representation_work_plan->residual_fallback_map);
        merge_executor_sec = elapsed_sec(merge_executor_start, std::chrono::steady_clock::now());
        if (!executor_map_or.ok()) {
          return executor_map_or.status();
        }
        if (!executor_map_or->segments.empty()) {
          prepared_execution.lowering_artifacts->executor_generic_data_map = *executor_map_or;
        }
      }
    }
  }
  const auto done = std::chrono::steady_clock::now();
  LOG(INFO) << "tc_profile resolved_mapped_execution_plan timings"
            << " artifact_id=" << resolved_artifact_id << " target_total_size=" << target_layout.total_size
            << " source_index_bytes=" << (source_index_json.has_value() ? source_index_json->size() : 0)
            << " tensor_bindings="
            << (resolved_plan.representation_transform_contract.has_value()
                    ? resolved_plan.representation_transform_contract->tensor_bindings.size()
                    : 0)
            << " work_items="
            << (resolved_plan.representation_work_plan.has_value()
                    ? resolved_plan.representation_work_plan->items.size()
                    : 0)
            << " physical_source_table_preparsed=" << (physical_source_table_preparsed ? 1 : 0)
            << " parse_source_index_sec=" << parse_source_index_sec << " init_plan_sec=" << init_plan_sec
            << " work_plan_sec=" << work_plan_sec
            << " work_plan_source_rebind_fast_path=" << (work_plan_source_rebind_fast_path ? 1 : 0)
            << " filter_collective_candidate_sec=" << filter_collective_candidate_sec
            << " collective_ranges_sec=" << collective_ranges_sec << " filter_collective_sec=" << filter_collective_sec
            << " filter_generic_sec=" << filter_generic_sec << " merge_executor_sec=" << merge_executor_sec
            << " collective_candidate_data_bytes=" << collective_candidate_data_bytes
            << " total_bytes_copied=" << mapped_plan.representation.total_bytes_copied
            << " generic_reused_collective_map=" << (generic_reused_collective_map ? 1 : 0)
            << " executor_merge_skipped=" << (executor_merge_skipped ? 1 : 0)
            << " compact_coverage_map_used=" << (compact_coverage_map_used ? 1 : 0)
            << " compact_coverage_segments=" << compact_coverage_segments
            << " compact_coverage_data_bytes=" << compact_coverage_data_bytes
            << " compact_collective_coverage_map_used=" << (compact_collective_coverage_map_used ? 1 : 0)
            << " compact_collective_coverage_segments=" << compact_collective_coverage_segments
            << " compact_collective_coverage_data_bytes=" << compact_collective_coverage_data_bytes
            << " source_window_coverage_proof_map_used=" << (source_window_coverage_proof_map_used ? 1 : 0)
            << " total_sec=" << elapsed_sec(profile_start, done);
  return prepared_execution;
}

} // namespace tensorcast::daemon::materialization_target_plan
