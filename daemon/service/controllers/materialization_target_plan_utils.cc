// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_target_plan_utils.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/store/view_utils.h"
#include "daemon/service/controllers/materialization_layout_utils.h"
#include "daemon/service/controllers/materialization_mapped_copy_plan_utils.h"
#include "daemon/service/controllers/materialization_mapped_target_layout_utils.h"
#include "daemon/service/controllers/materialization_mapped_view_narrow_utils.h"
#include "daemon/service/controllers/materialization_mapped_view_spec_utils.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/service/controllers/selection_validation_utils.h"
#include "daemon/util/status_utils.h"

namespace tensorcast::daemon::materialization_target_plan {

using ::grpc::Status;
using ::grpc::StatusCode;
using status_utils::to_grpc_status;

namespace {

using materialization_layout::CanonicalIndexEntry;
using materialization_layout::CanonicalIndexTable;
using materialization_layout::parse_canonical_index;
using materialization_layout::TargetOffsetEntry;
using materialization_mapped_copy_plan::build_copy_plan;
using materialization_mapped_copy_plan::ViewNarrowSpec;
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
using store::loader::ViewSpec;

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
      return {StatusCode::INVALID_ARGUMENT, "target_layout storage_offset mismatch"};
    }
    if (expected_storage_offset + entry.logical_length > range.length) {
      record_error(record_result, "offset_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "target_layout exceeds storage bounds"};
    }
  }
  return Status::OK;
}

struct TargetViewResolution {
  std::optional<ViewSpec> view_spec;
  std::optional<tensorcast::common::v1::ViewSpec> view_spec_proto;
  std::optional<std::string> request_view_id;
  std::optional<std::string> view_data_hash;
  std::optional<store::loader::ViewPlan> view_plan;
  std::optional<std::string> resolved_view_id;
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
  auto spec_or = store::view::parse_view_spec_json(view_meta_or->view_spec_json);
  if (!spec_or.ok()) {
    record_error(record_result, "view_parse_failed");
    return to_grpc_status(spec_or.status());
  }
  view_resolution.view_spec = std::move(*spec_or);
  view_resolution.view_spec_proto = build_view_spec_proto(*view_resolution.view_spec);
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
  if (!view_resolution.view_spec.has_value() && !has_subset && index_kind != v2::TargetLayout::INDEX_KIND_VIEW &&
      !has_ordered_selection) {
    return Status::OK;
  }

  ViewSpec plan_spec = view_resolution.view_spec.value_or(ViewSpec{});
  std::vector<std::string> subset_names;
  if (has_ordered_selection) {
    subset_names = request_names;
  } else if (has_subset) {
    subset_names = layout_names;
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
  view_resolution.has_view_transform = view_resolution.view_id_requested ||
      (view_resolution.view_spec.has_value() && view_resolution.view_plan.has_value() &&
       !view_resolution.view_plan->is_identity);
  if (view_resolution.view_id_requested && view_resolution.view_plan.has_value() &&
      view_resolution.view_plan->is_identity && !has_subset) {
    record_error(record_result, "view_identity_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "view_id requires a non-identity view spec"};
  }

  if (layout.index_kind() == v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED) {
    if (view_resolution.has_view_transform || has_subset || has_ordered_selection) {
      record_error(record_result, "index_kind_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "index_kind CANONICAL cannot be used with view/subset selection"};
    }
    return Status::OK;
  }
  if (!view_resolution.has_view_transform && !has_subset && !has_ordered_selection) {
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
    // selection.view_id without metadata-backed view_spec is treated as opaque mapped byte space identity.
    resolved_view_id = *request_view_id;
  }

  std::string source_index_json = plan.canonical_index_json;
  if (plan.view_plan.has_value() && !plan.view_plan->is_identity) {
    source_index_json = plan.view_plan->view_index_json;
  }
  plan.selected_index_json = source_index_json;
  auto source_table_or = parse_canonical_index(source_index_json);
  if (!source_table_or.ok()) {
    record_error(record_result, "index_parse_failed");
    return to_grpc_status(source_table_or.status());
  }
  const CanonicalIndexTable& source_table = *source_table_or;

  auto copy_plan_or = build_copy_plan(
      req.copy_plan(), source_table, mapped_layout.dst_specs, mapped_layout.dst_base_offsets, view_narrows);
  if (!copy_plan_or.ok()) {
    record_error(record_result, "mapping_invalid");
    return to_grpc_status(copy_plan_or.status());
  }
  plan.copy_plan = std::move(*copy_plan_or);
  plan.copy_plan.map.total_bytes = plan.logical_total_size;

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

} // namespace tensorcast::daemon::materialization_target_plan
