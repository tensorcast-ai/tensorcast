// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/selection_validation_utils.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_identity.h"
#include "core/common/selection_identity.h"
#include "daemon/service/artifact_profile_registry.h"

namespace tensorcast::daemon::selection_validation {

namespace {

using ::grpc::Status;
using ::grpc::StatusCode;

const ArtifactProfileRuntime& byte_artifact_runtime() {
  return ArtifactProfileRegistry::runtime_for_profile(ArtifactProfileRegistry::Profile::kByteArtifact);
}

void set_error_reason(std::string_view* error_reason, std::string_view value) {
  if (error_reason != nullptr) {
    *error_reason = value;
  }
}

absl::StatusOr<bool> is_validated_byte_artifact_profile(
    std::string_view artifact_id,
    std::string_view field_name,
    std::string_view* error_reason) {
  if (ArtifactProfileRegistry::classify_artifact_id(artifact_id) != ArtifactProfileRegistry::Profile::kByteArtifact) {
    return false;
  }
  const auto validation_st = byte_artifact_runtime().validate_artifact_id_for_field(artifact_id, field_name);
  if (!validation_st.ok()) {
    set_error_reason(error_reason, "byte_artifact_id_invalid");
    return validation_st;
  }
  return true;
}

} // namespace

Status validate_request_tensor_names(
    const tensorcast::common::v1::ArtifactSelection& selection,
    std::vector<std::string>& ordered_tensor_names,
    std::string_view* error_reason) {
  ordered_tensor_names.clear();
  ordered_tensor_names.reserve(selection.tensor_names_size());

  absl::flat_hash_set<std::string> names;
  names.reserve(selection.tensor_names_size());
  for (const auto& name : selection.tensor_names()) {
    ordered_tensor_names.push_back(name);
    if (!names.insert(name).second) {
      set_error_reason(error_reason, "tensor_name_duplicate");
      return {StatusCode::INVALID_ARGUMENT, "selection.tensor_names must not contain duplicates"};
    }
  }

  return Status::OK;
}

Status compute_and_validate_view_subset_hash(
    const tensorcast::common::v1::ArtifactSelection& selection,
    absl::Span<const std::string> resolved_selection_names,
    std::string& view_subset_hash,
    std::string_view* error_reason) {
  view_subset_hash.clear();
  auto byte_artifact_profile_or =
      is_validated_byte_artifact_profile(selection.artifact_id(), "selection.artifact_id", error_reason);
  if (!byte_artifact_profile_or.ok()) {
    return {StatusCode::INVALID_ARGUMENT, std::string(byte_artifact_profile_or.status().message())};
  }
  const bool byte_artifact_profile = *byte_artifact_profile_or;
  if (byte_artifact_profile) {
    if (!resolved_selection_names.empty()) {
      set_error_reason(error_reason, "byte_artifact_subset_not_supported");
      return {StatusCode::INVALID_ARGUMENT, "byte artifact selection supports full selection only"};
    }
    if (!selection.view_subset_hash().empty()) {
      set_error_reason(error_reason, "byte_artifact_subset_hash_not_allowed");
      return {StatusCode::INVALID_ARGUMENT, "byte artifact selection does not support view_subset_hash"};
    }
    return Status::OK;
  }

  if (!resolved_selection_names.empty()) {
    view_subset_hash = common::compute_view_subset_hash_bytes(resolved_selection_names);
    if (!selection.view_subset_hash().empty() && selection.view_subset_hash() != view_subset_hash) {
      set_error_reason(error_reason, "subset_hash_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "view_subset_hash does not match tensor_names"};
    }
    return Status::OK;
  }

  if (!selection.view_subset_hash().empty()) {
    set_error_reason(error_reason, "subset_hash_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "view_subset_hash must be empty for full selection"};
  }

  return Status::OK;
}

Status validate_hashes_and_build_resolved_selection(
    const tensorcast::common::v1::ArtifactSelection& request_selection,
    std::string_view resolved_artifact_id,
    std::string_view resolved_view_id,
    std::string_view selected_index_json,
    bool needs_view_index,
    absl::Span<const std::string> resolved_selection_names,
    std::string_view view_subset_hash,
    const tensorcast::common::v1::ViewSpec* resolved_view_spec,
    tensorcast::common::v1::ArtifactSelection& resolved_selection,
    std::string_view* error_reason) {
  auto byte_artifact_profile_or = is_validated_byte_artifact_profile(resolved_artifact_id, "artifact_id", error_reason);
  if (!byte_artifact_profile_or.ok()) {
    return {StatusCode::INVALID_ARGUMENT, std::string(byte_artifact_profile_or.status().message())};
  }
  const bool byte_artifact_profile = *byte_artifact_profile_or;
  if (byte_artifact_profile) {
    if (!resolved_view_id.empty() || resolved_view_spec != nullptr || request_selection.has_view_spec() ||
        !request_selection.view_id().empty()) {
      set_error_reason(error_reason, "byte_artifact_view_not_supported");
      return {StatusCode::INVALID_ARGUMENT, "byte artifact selection does not support view transforms"};
    }
    if (!resolved_selection_names.empty()) {
      set_error_reason(error_reason, "byte_artifact_subset_not_supported");
      return {StatusCode::INVALID_ARGUMENT, "byte artifact selection supports full selection only"};
    }
    if (!view_subset_hash.empty()) {
      set_error_reason(error_reason, "byte_artifact_subset_hash_not_allowed");
      return {StatusCode::INVALID_ARGUMENT, "byte artifact selection does not support view_subset_hash"};
    }
    auto normalized_or = byte_artifact_runtime().build_normalized_selection(resolved_artifact_id);
    if (!normalized_or.ok()) {
      set_error_reason(error_reason, "byte_artifact_selection_invalid");
      return {StatusCode::INVALID_ARGUMENT, std::string(normalized_or.status().message())};
    }
    if (!request_selection.logical_layout_hash().empty() &&
        request_selection.logical_layout_hash() != normalized_or->logical_layout_hash()) {
      set_error_reason(error_reason, "logical_layout_hash_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "selection.logical_layout_hash does not match resolved selection"};
    }
    if (!request_selection.selection_hash().empty() &&
        request_selection.selection_hash() != normalized_or->selection_hash()) {
      set_error_reason(error_reason, "selection_hash_mismatch");
      return {StatusCode::INVALID_ARGUMENT, "selection.selection_hash does not match resolved selection"};
    }

    resolved_selection = std::move(*normalized_or);
    return Status::OK;
  }

  const std::string logical_layout_hash = common::compute_logical_layout_hash_bytes(
      absl::Span<const uint8_t>(
          reinterpret_cast<const uint8_t*>(selected_index_json.data()), selected_index_json.size()),
      needs_view_index);
  std::optional<std::string_view> subset_hash_opt;
  if (!view_subset_hash.empty()) {
    subset_hash_opt = view_subset_hash;
  }
  const std::string selection_hash = common::compute_selection_hash_bytes(resolved_view_id, subset_hash_opt);

  if (!request_selection.logical_layout_hash().empty() &&
      request_selection.logical_layout_hash() != logical_layout_hash) {
    set_error_reason(error_reason, "logical_layout_hash_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "selection.logical_layout_hash does not match resolved selection"};
  }
  if (!request_selection.selection_hash().empty() && request_selection.selection_hash() != selection_hash) {
    set_error_reason(error_reason, "selection_hash_mismatch");
    return {StatusCode::INVALID_ARGUMENT, "selection.selection_hash does not match resolved selection"};
  }

  resolved_selection.Clear();
  resolved_selection.set_artifact_id(std::string(resolved_artifact_id));
  resolved_selection.set_view_id(std::string(resolved_view_id));
  resolved_selection.set_logical_layout_hash(logical_layout_hash);
  resolved_selection.set_selection_hash(selection_hash);
  if (!view_subset_hash.empty()) {
    resolved_selection.set_view_subset_hash(std::string(view_subset_hash));
  }
  for (const auto& name : resolved_selection_names) {
    resolved_selection.add_tensor_names(name);
  }
  if (!resolved_view_id.empty() && resolved_view_spec != nullptr) {
    resolved_selection.mutable_view_spec()->CopyFrom(*resolved_view_spec);
  }

  return Status::OK;
}

} // namespace tensorcast::daemon::selection_validation
