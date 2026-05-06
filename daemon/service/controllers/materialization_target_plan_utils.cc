// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_target_plan_utils.h"

#include <algorithm>
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
using materialization_layout::parse_canonical_index;
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
    const representation_transform_builder::TransformWorkCompatibilityStats& stats) {
  return store::runtime::ingestion::strategy::SourceBoundLoweringStats{
      .total_dst_tensors = stats.total_dst_tensors,
      .compatible_candidates = stats.compatible_candidates,
      .compatible_bytes = stats.compatible_bytes,
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

absl::StatusOr<store::materialization::contracts::RepresentationWorkPlan> build_execution_representation_work_plan(
    const store::materialization::contracts::RepresentationTransformContract& transform_contract,
    const std::optional<CanonicalIndexTable>& physical_source_table,
    std::optional<store::loader::ByteRangeMap> compatibility_lowered_map = std::nullopt) {
  auto execution_contract = transform_contract;
  if (physical_source_table.has_value()) {
    patch_transform_contract_source_specs(*physical_source_table, &execution_contract);
  }
  auto work_plan_or = store::materialization::contracts::build_representation_work_plan(execution_contract);
  if (!work_plan_or.ok()) {
    return work_plan_or.status();
  }
  if (compatibility_lowered_map.has_value()) {
    store::loader::ByteRangeMap pad_map;
    pad_map.total_bytes = compatibility_lowered_map->total_bytes;
    pad_map.num_sources = compatibility_lowered_map->num_sources;
    for (const auto& segment : compatibility_lowered_map->segments) {
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
      work_plan_or->committed_bytes += pad_item.committed_bytes;
      work_plan_or->items.push_back(std::move(pad_item));
    }
  }
  return std::move(*work_plan_or);
}

absl::StatusOr<store::loader::ByteRangeMap> filter_byte_range_map(
    const store::loader::ByteRangeMap& map,
    store::loader::ByteRangeSegment::Kind kind) {
  store::loader::ByteRangeMap filtered;
  filtered.total_bytes = map.total_bytes;
  filtered.num_sources = map.num_sources;
  for (const auto& segment : map.segments) {
    if (segment.kind == kind) {
      filtered.segments.push_back(segment);
    }
  }
  std::sort(filtered.segments.begin(), filtered.segments.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.dst_offset < rhs.dst_offset;
  });
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

bool work_item_has_collective_source_overlap(const store::materialization::contracts::RepresentationWorkItem& item) {
  return item.kind == store::materialization::contracts::RepresentationWorkItemKind::kTensorCopy &&
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
  auto index_table_or = parse_canonical_index(target_index_json);
  if (!index_table_or.ok()) {
    return index_table_or.status();
  }
  const auto& index_table = *index_table_or;

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

  auto canonical_table_or = parse_canonical_index(plan.canonical_index_json);
  if (!canonical_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(canonical_table_or.status());
  }
  const CanonicalIndexTable& canonical_table = *canonical_table_or;

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

  auto index_table_or = parse_canonical_index(plan.selected_index_json);
  if (!index_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(index_table_or.status());
  }
  const CanonicalIndexTable& index_table = *index_table_or;
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
  auto canonical_source_table_or = parse_canonical_index(plan.canonical_index_json);
  if (!canonical_source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(canonical_source_table_or.status());
  }
  const CanonicalIndexTable& canonical_source_table = *canonical_source_table_or;
  auto source_table_or = parse_canonical_index(source_index_json);
  if (!source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(source_table_or.status());
  }
  const CanonicalIndexTable& source_table = *source_table_or;

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
  LOG(INFO) << "MaterializeIntoMappedTarget representation-work compatibility"
            << " copy_entries=" << req.copy_plan().entries_size() << " dst_tensors=" << mapped_layout.dst_specs.size()
            << " compatible_candidates=" << plan.representation.compatibility_stats.compatible_candidates
            << " compatible_bytes=" << plan.representation.compatibility_stats.compatible_bytes
            << " concat_candidates=" << plan.representation.compatibility_stats.concat_candidates
            << " concat_bytes=" << plan.representation.compatibility_stats.concat_bytes
            << " rejected_mixed_src_or_dim=" << plan.representation.compatibility_stats.rejected_mixed_src_or_dim
            << " rejected_mixed_src_or_dim_bytes="
            << plan.representation.compatibility_stats.rejected_mixed_src_or_dim_bytes
            << " rejected_non_contiguous=" << plan.representation.compatibility_stats.rejected_non_contiguous
            << " rejected_non_contiguous_bytes="
            << plan.representation.compatibility_stats.rejected_non_contiguous_bytes
            << " rejected_unsupported_distribution="
            << plan.representation.compatibility_stats.rejected_unsupported_distribution
            << " rejected_unsupported_distribution_bytes="
            << plan.representation.compatibility_stats.rejected_unsupported_distribution_bytes
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
    MappedTargetMaterializationPlan& plan) {
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
  auto tensor_name_status =
      selection_validation::validate_request_tensor_names(selection, ordered_selection_names, &selection_error_reason);
  if (!tensor_name_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return tensor_name_status;
  }
  std::string view_subset_hash;
  auto subset_hash_status = selection_validation::compute_and_validate_view_subset_hash(
      selection, absl::MakeSpan(ordered_selection_names), view_subset_hash, &selection_error_reason);
  if (!subset_hash_status.ok()) {
    record_error(record_result, selection_error_reason.empty() ? "transfer_error" : selection_error_reason);
    return subset_hash_status;
  }

  PreparedTargetStorageLayout prepared_storage_layout;
  auto storage_layout_status = build_target_storage_layout(target_layout, record_result, prepared_storage_layout);
  if (!storage_layout_status.ok()) {
    return storage_layout_status;
  }

  auto mapped_layout_or =
      build_mapped_layout_from_target_index(plan.selected_index_json, offsets, prepared_storage_layout.storage_ranges);
  if (!mapped_layout_or.ok()) {
    record_error(record_result, "layout_mismatch");
    return to_grpc_status(mapped_layout_or.status());
  }
  auto target_index_table_or = parse_canonical_index(plan.selected_index_json);
  if (!target_index_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(target_index_table_or.status());
  }
  plan.logical_total_size = target_index_table_or->logical_total_size;
  auto storage_span_status = validate_storage_span_matches_index(
      prepared_storage_layout.total_storage_bytes, plan.logical_total_size, record_result);
  if (!storage_span_status.ok()) {
    return storage_span_status;
  }
  auto offset_validation_status = validate_target_offsets_against_layout(
      offsets, *target_index_table_or, prepared_storage_layout.storage_ranges, record_result);
  if (!offset_validation_status.ok()) {
    return offset_validation_status;
  }

  v2::MaterializeIntoMappedTargetRequest view_request;
  view_request.mutable_selection()->CopyFrom(selection);
  ResolveViewSpecErrorReason resolve_view_reason = ResolveViewSpecErrorReason::kUnknown;
  auto resolved_view_or = resolve_mapped_view_spec(view_request, resolved_artifact_id, engine, &resolve_view_reason);
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
  const auto view_narrows = std::move(*view_narrows_or);

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
    resolved_view_id = *request_view_id;
  }

  std::string source_index_json = plan.canonical_index_json;
  if (plan.view_plan.has_value() && !plan.view_plan->is_identity) {
    source_index_json = plan.view_plan->view_index_json;
  }
  auto canonical_source_table_or = parse_canonical_index(plan.canonical_index_json);
  if (!canonical_source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(canonical_source_table_or.status());
  }
  auto source_table_or = parse_canonical_index(source_index_json);
  if (!source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(source_table_or.status());
  }

  auto representation_or = build_representation_transform_contract(
      realization_plan,
      *source_table_or,
      *canonical_source_table_or,
      mapped_layout_or->dst_specs,
      mapped_layout_or->dst_base_offsets,
      view_narrows,
      build_source_byte_space(
          resolved_view_id.has_value() ? std::optional<std::string_view>(*resolved_view_id) : std::nullopt),
      "local_seal_then_promote");
  if (!representation_or.ok()) {
    record_error(record_result, "mapping_invalid");
    return to_grpc_status(representation_or.status());
  }
  plan.representation = std::move(*representation_or);

  tensorcast::common::v1::ArtifactSelection validated_source_selection;
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

  plan.publish_storages = std::move(prepared_storage_layout.publish_storages);
  plan.publish_segments = std::move(prepared_storage_layout.publish_segments);
  return Status::OK;
}

absl::StatusOr<store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan>
build_resolved_mapped_materialization_plan(
    std::string_view resolved_artifact_id,
    uint64_t generation,
    const store::loading::IntoTargetLayout& target_layout,
    const MappedTargetMaterializationPlan& mapped_plan,
    const std::optional<store::loading::VariantIdentity>& variant,
    std::optional<std::string_view> source_index_json) {
  using PreparedExecutionPlan = store::runtime::ingestion::strategy::PreparedSourceBoundExecutionPlan;
  using StrategyPlan = store::runtime::ingestion::strategy::ResolvedMaterializationPlan;

  std::optional<CanonicalIndexTable> physical_source_table;
  if (source_index_json.has_value()) {
    auto physical_source_table_or = parse_canonical_index(*source_index_json);
    if (!physical_source_table_or.ok()) {
      return physical_source_table_or.status();
    }
    physical_source_table = std::move(*physical_source_table_or);
  }

  PreparedExecutionPlan prepared_execution;
  StrategyPlan& resolved_plan = prepared_execution.resolved_plan;
  resolved_plan.artifact_id = std::string(resolved_artifact_id);
  resolved_plan.generation = generation;
  resolved_plan.variant = variant;
  resolved_plan.canonical_index_json = mapped_plan.canonical_index_json;
  resolved_plan.target_layout = target_layout;
  resolved_plan.representation_transform_contract = mapped_plan.representation.transform_contract;
  if (resolved_plan.representation_transform_contract.has_value()) {
    prepared_execution.lowering_artifacts = store::runtime::ingestion::strategy::SourceBoundLoweringArtifacts{
        .lowering_stats = to_source_bound_lowering_stats(mapped_plan.representation.compatibility_stats),
    };
    const store::loader::ByteRangeMap compatibility_map = [&]() {
      if (!mapped_plan.representation.compatibility_lowered_map.segments.empty()) {
        auto map = mapped_plan.representation.compatibility_lowered_map;
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
    auto work_plan_or = build_execution_representation_work_plan(
        *resolved_plan.representation_transform_contract,
        physical_source_table,
        compatibility_map.segments.empty() ? std::nullopt
                                           : std::optional<store::loader::ByteRangeMap>(compatibility_map));
    if (!work_plan_or.ok()) {
      return work_plan_or.status();
    }
    resolved_plan.representation_work_plan = std::move(*work_plan_or);
    auto compatibility_data_map_or =
        filter_byte_range_map(compatibility_map, store::loader::ByteRangeSegment::Kind::kData);
    if (!compatibility_data_map_or.ok()) {
      return compatibility_data_map_or.status();
    }
    if (!compatibility_data_map_or->segments.empty()) {
      auto collective_dst_ranges_or = build_collective_dst_ranges(*resolved_plan.representation_work_plan);
      if (!collective_dst_ranges_or.ok()) {
        return collective_dst_ranges_or.status();
      }
      auto collective_data_map_or =
          filter_data_map_to_dst_ranges(*compatibility_data_map_or, *collective_dst_ranges_or);
      if (!collective_data_map_or.ok()) {
        return collective_data_map_or.status();
      }
      if (!collective_data_map_or->segments.empty()) {
        prepared_execution.lowering_artifacts->collective_data_map = *collective_data_map_or;
      }
    }
    store::loader::ByteRangeMap compatibility_data_map;
    if (!compatibility_data_map_or->segments.empty()) {
      compatibility_data_map = *compatibility_data_map_or;
    }
    auto generic_data_map_or =
        filter_byte_range_map(generic_fallback_map, store::loader::ByteRangeSegment::Kind::kData);
    if (!generic_data_map_or.ok()) {
      return generic_data_map_or.status();
    }
    store::loader::ByteRangeMap generic_data_map;
    if (!generic_data_map_or->segments.empty()) {
      generic_data_map = *generic_data_map_or;
    } else if (!compatibility_data_map.segments.empty()) {
      generic_data_map = compatibility_data_map;
    }
    auto executor_map_or =
        merge_byte_range_maps(generic_data_map, resolved_plan.representation_work_plan->residual_fallback_map);
    if (!executor_map_or.ok()) {
      return executor_map_or.status();
    }
    if (!executor_map_or->segments.empty()) {
      prepared_execution.lowering_artifacts->executor_generic_data_map = *executor_map_or;
    }
  }
  return prepared_execution;
}

} // namespace tensorcast::daemon::materialization_target_plan
