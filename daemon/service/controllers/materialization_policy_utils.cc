// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/view/view_identity.h"
#include "daemon/service/rpc_context.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"

namespace tensorcast::daemon::materialization_policy {

namespace global_store = tensorcast::global_store::v1;

namespace {

using store::loader::ViewOp;

constexpr std::string_view kGroupRealizationTransportKind = "group_realization_transport";
constexpr std::string_view kGroupRealizationChildTransportRequestProfile =
    "tensorcast.group_realization.child_transport_request.v1";
constexpr std::string_view kControllerSourceSelectionDigestProfile = "tensorcast.controller.source_selection_digest.v1";

store::loading::SourceLocalityHint to_source_locality(v2::SourceLocality locality) {
  switch (locality) {
    case v2::SourceLocality::SOURCE_LOCALITY_HOST_LOCAL:
      return store::loading::SourceLocalityHint::kHostLocal;
    case v2::SourceLocality::SOURCE_LOCALITY_SHARED_SOURCE:
      return store::loading::SourceLocalityHint::kSharedSource;
    case v2::SourceLocality::SOURCE_LOCALITY_AUTO:
    case v2::SourceLocality::SOURCE_LOCALITY_UNSPECIFIED:
    default:
      return store::loading::SourceLocalityHint::kAuto;
  }
}

absl::Status response_status_to_absl(global_store::Status status, std::string_view rpc_name) {
  if (status == global_store::STATUS_OK) {
    return absl::OkStatus();
  }
  const std::string status_name = global_store::Status_Name(status);
  const std::string message =
      std::format("{} failed: {}", rpc_name, status_name.empty() ? "STATUS_UNKNOWN" : status_name);
  switch (status) {
    case global_store::STATUS_NOT_FOUND:
      return absl::NotFoundError(message);
    case global_store::STATUS_TIMED_OUT:
      return absl::DeadlineExceededError(message);
    case global_store::STATUS_TOO_MANY_REQUESTS:
      return absl::ResourceExhaustedError(message);
    case global_store::STATUS_STATE_SYNC_REQUIRED:
      return absl::FailedPreconditionError(message);
    case global_store::STATUS_ERROR:
    case global_store::STATUS_UNSPECIFIED:
    default:
      return absl::InternalError(message);
  }
}

} // namespace

store::loading::SourcePreference to_hint_preference(v2::SourcePreference preference) {
  switch (preference) {
    case v2::SourcePreference::SOURCE_PREFERENCE_PREFER_P2P:
      return store::loading::SourcePreference::kPreferP2P;
    case v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK:
      return store::loading::SourcePreference::kPreferDisk;
    case v2::SourcePreference::SOURCE_PREFERENCE_AUTO:
    case v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED:
    default:
      return store::loading::SourcePreference::kAuto;
  }
}

store::loading::ExportPolicy to_hint_export_policy(v2::ExportPolicy policy) {
  switch (policy) {
    case v2::ExportPolicy::EXPORT_POLICY_FORCE:
      return store::loading::ExportPolicy::kForce;
    case v2::ExportPolicy::EXPORT_POLICY_AUTO:
      return store::loading::ExportPolicy::kAuto;
    case v2::ExportPolicy::EXPORT_POLICY_NEVER:
    case v2::ExportPolicy::EXPORT_POLICY_UNSPECIFIED:
    default:
      return store::loading::ExportPolicy::kNever;
  }
}

namespace {

absl::Status validate_required_group_parts(const v2::SemanticGroupContext& group) {
  if (group.required_part_ids_size() == 0) {
    return absl::InvalidArgumentError("group_realization.group.required_part_ids is required");
  }
  if (static_cast<uint32_t>(group.required_part_ids_size()) != group.total_parts()) {
    return absl::InvalidArgumentError("group_realization.group.required_part_ids must match total_parts");
  }

  std::vector<std::string> required_part_ids(group.required_part_ids().begin(), group.required_part_ids().end());
  bool contains_part_id = false;
  for (const std::string& part_id : required_part_ids) {
    if (part_id.empty()) {
      return absl::InvalidArgumentError("group_realization.group.required_part_ids entries must be non-empty");
    }
    contains_part_id = contains_part_id || part_id == group.part_id();
  }
  std::sort(required_part_ids.begin(), required_part_ids.end());
  if (std::adjacent_find(required_part_ids.begin(), required_part_ids.end()) != required_part_ids.end()) {
    return absl::InvalidArgumentError("group_realization.group.required_part_ids must be unique");
  }
  if (!contains_part_id) {
    return absl::InvalidArgumentError("group_realization.group.part_id must be present in required_part_ids");
  }
  return absl::OkStatus();
}

absl::Status validate_group_realization_version_reference(const v2::VersionReference& version) {
  switch (version.value_case()) {
    case v2::VersionReference::kExplicitSelection:
      if (version.explicit_selection().artifact_id().empty()) {
        return absl::InvalidArgumentError("group_realization.version.explicit_selection.artifact_id is required");
      }
      return absl::OkStatus();
    case v2::VersionReference::kExplicitVersionSet:
      if (version.explicit_version_set().version_set_id().empty()) {
        return absl::InvalidArgumentError("group_realization.version.explicit_version_set.version_set_id is required");
      }
      return absl::OkStatus();
    case v2::VersionReference::kKeyReference:
      if (version.key_reference().key().empty()) {
        return absl::InvalidArgumentError("group_realization.version.key_reference.key is required");
      }
      return absl::OkStatus();
    case v2::VersionReference::VALUE_NOT_SET:
    default:
      return absl::InvalidArgumentError("group_realization.version value is required");
  }
}

absl::Status validate_group_realization_options(const v2::GroupRealizationOptions& options) {
  if (!options.has_version()) {
    return absl::InvalidArgumentError("group_realization.version is required when group realization is enabled");
  }
  if (auto status = validate_group_realization_version_reference(options.version()); !status.ok()) {
    return status;
  }
  if (!options.has_group()) {
    return absl::InvalidArgumentError("group_realization.group is required when group realization is enabled");
  }
  const auto& group = options.group();
  if (group.group_kind().empty() || group.group_id().empty()) {
    return absl::InvalidArgumentError("group_realization.group group_kind and group_id are required");
  }
  if (group.part_id().empty()) {
    return absl::InvalidArgumentError("group_realization.group.part_id is required");
  }
  if (group.total_parts() == 0) {
    return absl::InvalidArgumentError("group_realization.group.total_parts must be > 0");
  }
  return validate_required_group_parts(group);
}

std::string build_group_realization_transport_group_id(const v2::SemanticGroupContext& group) {
  std::string group_id = group.group_kind();
  group_id.push_back(':');
  group_id.append(group.group_id());
  return group_id;
}

store::loading::TransportSchedulingGroupHint build_group_realization_transport_group(
    const v2::SemanticGroupContext& group) {
  return store::loading::TransportSchedulingGroupHint{
      .group_id = build_group_realization_transport_group_id(group),
      .group_kind = std::string(kGroupRealizationTransportKind),
      .total_parts = group.total_parts(),
      .part_id = group.part_id(),
      .priority = 0,
      .epoch = group.epoch(),
  };
}

void copy_group_version_set_ref(const v2::GroupVersionSetRef& source, global_store::GroupVersionSetRef* destination) {
  destination->set_version_set_id(source.version_set_id());
  destination->set_manifest_hash(source.manifest_hash());
  destination->set_manifest_generation(source.manifest_generation());
}

void copy_key_version_reference(const v2::KeyVersionReference& source, global_store::KeyVersionReference* destination) {
  destination->set_key(source.key());
  destination->set_namespace_(source.namespace_());
  destination->set_alias(source.alias());
  if (source.has_expected_generation()) {
    destination->set_expected_generation(source.expected_generation());
  }
}

absl::Status copy_version_reference(const v2::VersionReference& source, global_store::VersionReference* destination) {
  switch (source.value_case()) {
    case v2::VersionReference::kExplicitSelection:
      destination->mutable_explicit_selection()->CopyFrom(source.explicit_selection());
      return absl::OkStatus();
    case v2::VersionReference::kExplicitVersionSet:
      copy_group_version_set_ref(source.explicit_version_set(), destination->mutable_explicit_version_set());
      return absl::OkStatus();
    case v2::VersionReference::kKeyReference:
      copy_key_version_reference(source.key_reference(), destination->mutable_key_reference());
      return absl::OkStatus();
    case v2::VersionReference::VALUE_NOT_SET:
    default:
      return absl::InvalidArgumentError("group_realization.version value is required");
  }
}

void copy_group_realization_context(
    const v2::SemanticGroupContext& source,
    global_store::GroupRealizationContext* destination) {
  destination->set_group_kind(source.group_kind());
  destination->set_group_id(source.group_id());
  destination->set_epoch(source.epoch());
  destination->set_total_parts(source.total_parts());
  destination->set_part_id(source.part_id());
  for (const std::string& part_id : source.required_part_ids()) {
    destination->add_required_part_ids(part_id);
  }
}

absl::Status validate_begin_or_join_response(
    const global_store::BeginOrJoinGroupRealizationResponse& response,
    const v2::SemanticGroupContext& group) {
  if (auto status = response_status_to_absl(response.status(), "BeginOrJoinGroupRealization"); !status.ok()) {
    return status;
  }
  if (response.transaction_id().empty()) {
    return absl::FailedPreconditionError("BeginOrJoinGroupRealization returned an empty transaction_id");
  }
  if (!response.has_part()) {
    return absl::FailedPreconditionError("BeginOrJoinGroupRealization returned no part");
  }
  if (response.part().part_id() != group.part_id()) {
    return absl::FailedPreconditionError("BeginOrJoinGroupRealization returned a different part_id");
  }
  if (!response.part().has_selection() || response.part().selection().artifact_id().empty()) {
    return absl::FailedPreconditionError("BeginOrJoinGroupRealization returned no frozen part selection");
  }
  return absl::OkStatus();
}

std::string byte_space_identity(const tensorcast::common::v1::ByteSpaceRef& byte_space) {
  if (byte_space.kind() == tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED && byte_space.id().empty()) {
    return "none";
  }
  return std::format("{}:{}", static_cast<int>(byte_space.kind()), byte_space.id());
}

std::string selection_view_identity(const tensorcast::common::v1::ArtifactSelection& selection) {
  if (!selection.view_id().empty()) {
    return selection.view_id();
  }
  if (!selection.view_subset_hash().empty()) {
    return std::format("subset:{}", selection.view_subset_hash());
  }
  if (selection.has_view_spec()) {
    std::string serialized;
    if (selection.view_spec().SerializeToString(&serialized)) {
      return absl::StrCat("spec:", absl::BytesToHexString(serialized));
    }
    return "spec";
  }
  return "canonical";
}

void append_uint64_be(std::string* payload, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    payload->push_back(static_cast<char>((value >> shift) & 0xFF));
  }
}

void append_length_prefixed_field(std::string* payload, std::string_view value) {
  append_uint64_be(payload, value.size());
  payload->append(value.data(), value.size());
}

std::string derive_group_realization_child_transport_request_id(
    const GroupRealizationBeginContext& begin_context,
    std::string_view operation_attempt_id) {
  std::string payload;
  append_length_prefixed_field(&payload, kGroupRealizationChildTransportRequestProfile);
  append_length_prefixed_field(&payload, begin_context.transaction_id);
  append_length_prefixed_field(&payload, begin_context.version_set.version_set_id());
  append_length_prefixed_field(&payload, begin_context.version_set.manifest_hash());
  append_length_prefixed_field(&payload, std::format("{}", begin_context.version_set.manifest_generation()));
  append_length_prefixed_field(&payload, std::format("{}", static_cast<int>(begin_context.realization_kind)));
  append_length_prefixed_field(&payload, begin_context.part_id);
  append_length_prefixed_field(&payload, begin_context.part_selection.artifact_id());
  append_length_prefixed_field(&payload, byte_space_identity(begin_context.requested_byte_space));
  append_length_prefixed_field(&payload, selection_view_identity(begin_context.part_selection));
  append_length_prefixed_field(&payload, begin_context.selection_hash);
  append_length_prefixed_field(&payload, operation_attempt_id);
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return absl::StrCat(
      "grt:", absl::BytesToHexString(std::string_view(reinterpret_cast<const char*>(digest.data()), digest.size())));
}

std::string digest_payload(std::string_view label, std::string_view payload) {
  std::string framed;
  append_length_prefixed_field(&framed, label);
  append_length_prefixed_field(&framed, payload);
  const auto digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(framed.data()), framed.size()));
  return absl::BytesToHexString(std::string_view(reinterpret_cast<const char*>(digest.data()), digest.size()));
}

absl::StatusOr<std::string> target_layout_digest(const v2::TargetLayout& layout) {
  std::string serialized;
  if (!layout.SerializeToString(&serialized)) {
    return absl::InternalError("failed to serialize target_layout for realization plan");
  }
  return digest_payload("daemon-target-layout", serialized);
}

std::string source_selection_mode_for(const GroupRealizationBeginContext* begin_context) {
  if (begin_context == nullptr) {
    return "single_selection";
  }
  switch (begin_context->realization_kind) {
    case global_store::GROUP_REALIZATION_KIND_SAME_SELECTION:
      return "same_selection";
    case global_store::GROUP_REALIZATION_KIND_PER_PART_SELECTION:
      return "per_part_selection";
    case global_store::GROUP_REALIZATION_KIND_UNSPECIFIED:
    default:
      return "group_selection";
  }
}

std::string source_selection_mode_for(const v2::GroupRealizationOptions* group_realization, bool per_part_source) {
  (void)group_realization;
  if (per_part_source) {
    return "per_part_selection";
  }
  return "same_selection";
}

std::vector<std::string> group_barriers_for(const v2::GroupRealizationOptions* group_realization) {
  std::vector<std::string> barriers;
  if (group_realization == nullptr || !group_realization->enabled()) {
    return barriers;
  }
  if (group_realization->group().total_parts() > 1) {
    barriers.push_back("member_readiness");
  }
  barriers.push_back("group_acquire");
  if (group_realization->require_staged_publish()) {
    barriers.push_back("staged_values");
    barriers.push_back("publish_barrier");
  }
  return barriers;
}

std::vector<std::string> acquire_group_barriers_for(const v2::GroupRealizationAcquireRef& acquire) {
  std::vector<std::string> barriers{"group_acquire"};
  if (acquire.wait_for_publish()) {
    barriers.push_back("publish_barrier");
  }
  return barriers;
}

std::string serialize_deterministic(const google::protobuf::MessageLite& message) {
  std::string output;
  {
    google::protobuf::io::StringOutputStream string_stream(&output);
    google::protobuf::io::CodedOutputStream coded_stream(&string_stream);
    coded_stream.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded_stream) || coded_stream.HadError()) {
      return message.SerializeAsString();
    }
  }
  return output;
}

void append_big_endian_u64(std::vector<uint8_t>* out, uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out->push_back(static_cast<uint8_t>((value >> shift) & 0xffU));
  }
}

void append_digest_part(std::vector<uint8_t>* out, std::string_view part) {
  append_big_endian_u64(out, static_cast<uint64_t>(part.size()));
  out->insert(out->end(), part.begin(), part.end());
}

std::string sha256_hex_for_parts(const std::vector<std::string_view>& parts) {
  uint64_t total_size = 0;
  for (std::string_view part : parts) {
    total_size += 8U + static_cast<uint64_t>(part.size());
  }
  std::vector<uint8_t> payload;
  if (total_size <= static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    payload.reserve(static_cast<size_t>(total_size));
  }
  for (std::string_view part : parts) {
    append_digest_part(&payload, part);
  }
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(absl::MakeConstSpan(payload));
  return absl::BytesToHexString(std::string(reinterpret_cast<const char*>(digest.data()), digest.size()));
}

std::string artifact_profile_for(std::string_view artifact_id) {
  if (artifact_id.starts_with("msa1:")) {
    return "mounted_source";
  }
  if (artifact_id.starts_with("cgid:")) {
    return "byte_artifact";
  }
  return "durable_artifact";
}

std::string authority_scope_for(std::string_view artifact_id) {
  if (artifact_id.starts_with("msa1:")) {
    return "daemon_local_mounted_source";
  }
  return "daemon_mediated_durable";
}

std::optional<std::string> requested_generation_hint_for(const v2::GroupRealizationOptions* group_realization) {
  if (group_realization == nullptr || !group_realization->enabled()) {
    return std::nullopt;
  }
  if (group_realization->version().value_case() != v2::VersionReference::kKeyReference) {
    return std::nullopt;
  }
  const v2::KeyVersionReference& key_ref = group_realization->version().key_reference();
  if (!key_ref.has_expected_generation()) {
    return std::nullopt;
  }
  return std::to_string(key_ref.expected_generation());
}

std::optional<std::string> requested_version_set_id_for(const v2::GroupRealizationOptions* group_realization) {
  if (group_realization == nullptr || !group_realization->enabled()) {
    return std::nullopt;
  }
  if (group_realization->version().value_case() != v2::VersionReference::kExplicitVersionSet) {
    return std::nullopt;
  }
  const std::string& version_set_id = group_realization->version().explicit_version_set().version_set_id();
  if (version_set_id.empty()) {
    return std::nullopt;
  }
  return version_set_id;
}

std::optional<std::string> selection_digest_for(
    const v2::GroupRealizationOptions* group_realization,
    const GroupRealizationBeginContext* begin_context,
    const tensorcast::common::v1::ArtifactSelection& selection) {
  const tensorcast::common::v1::ArtifactSelection& effective_selection =
      begin_context != nullptr && !begin_context->part_selection.artifact_id().empty() ? begin_context->part_selection
                                                                                       : selection;
  if (effective_selection.artifact_id().empty()) {
    return std::nullopt;
  }

  const std::string serialized_selection = serialize_deterministic(effective_selection);
  const std::string selection_identity = begin_context != nullptr && !begin_context->selection_hash.empty()
      ? begin_context->selection_hash
      : effective_selection.selection_hash();
  std::string generation_hint;
  if (begin_context != nullptr && begin_context->key_generation != 0) {
    generation_hint = std::to_string(begin_context->key_generation);
  } else if (std::optional<std::string> requested_generation = requested_generation_hint_for(group_realization);
             requested_generation.has_value()) {
    generation_hint = *requested_generation;
  }
  const std::string profile = artifact_profile_for(effective_selection.artifact_id());
  const std::string scope = authority_scope_for(effective_selection.artifact_id());
  return sha256_hex_for_parts({
      kControllerSourceSelectionDigestProfile,
      serialized_selection,
      effective_selection.logical_layout_hash(),
      selection_identity,
      profile,
      scope,
      generation_hint,
  });
}

std::optional<std::string> operation_id_for(const v2::MaterializeIntoTargetRequest& request) {
  return request.has_operation_id() && !request.operation_id().empty()
      ? std::optional<std::string>(request.operation_id())
      : std::nullopt;
}

std::optional<std::string> operation_id_for(const v2::MaterializeIntoMappedTargetRequest& request) {
  return request.has_operation_id() && !request.operation_id().empty()
      ? std::optional<std::string>(request.operation_id())
      : std::nullopt;
}

std::string digest_fields(std::string_view label, const std::vector<std::string>& fields);

std::string target_layout_digest_for_replica(
    const tensorcast::common::v1::ArtifactSelection& resolved_selection,
    const v2::MaterializeReplicaRequest& request,
    std::string_view resolved_artifact_id) {
  if (!resolved_selection.logical_layout_hash().empty()) {
    return absl::BytesToHexString(resolved_selection.logical_layout_hash());
  }
  return digest_fields(
      "daemon-replica-target-layout",
      {
          std::string(resolved_artifact_id),
          resolved_selection.view_id(),
          absl::BytesToHexString(resolved_selection.selection_hash()),
          request.device_uuid(),
          std::format("{}", static_cast<int>(request.target_device_type())),
          std::format("{}", request.size_bytes()),
      });
}

v2::CollectivePolicy replica_collective_policy_for(const NormalizedMaterializationRequestContext& request_context) {
  return request_context.execution_topology.collective_load_group.has_value()
      ? v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST
      : v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE;
}

std::string replica_target_kind_for(const v2::MaterializeReplicaRequest& request) {
  return request.wait_for_completion() ? "tensor_dict" : "retained_replica";
}

std::vector<std::string> replica_release_policy_for(const v2::MaterializeReplicaRequest& request, bool no_lease) {
  if (no_lease) {
    return {"retain_daemon_replica", "release_operation_ticket"};
  }
  if (request.wait_for_completion()) {
    return {"release_handle_lease", "release_pid_ref", "release_replica_session"};
  }
  return {"release_retained_replica_handle", "release_replica_session"};
}

std::string digest_fields(std::string_view label, const std::vector<std::string>& fields) {
  std::string payload;
  for (const auto& field : fields) {
    append_length_prefixed_field(&payload, field);
  }
  return digest_payload(label, payload);
}

std::string target_layout_digest_for_serving_target(const tensorcast::operation::v1::ServingBindingTarget& target) {
  const auto& layout = target.resolved_layout();
  if (!layout.target_layout_hash().empty()) {
    return layout.target_layout_hash();
  }
  if (!layout.target_layout().empty()) {
    return digest_payload("daemon-serving-target-layout", layout.target_layout());
  }
  return {};
}

std::string target_set_layout_digest_for(const tensorcast::operation::v1::ServingBindingSetTarget& target) {
  std::vector<std::string> fields{
      target.runtime(),
      target.group_id(),
  };
  fields.reserve(fields.size() + static_cast<size_t>(target.members_size()) * 4);
  for (const auto& member : target.members()) {
    fields.push_back(member.member().member_id());
    fields.push_back(member.device_uuid());
    fields.push_back(target_layout_digest_for_serving_target(member));
    fields.push_back(member.resolved_layout().spec_digest());
  }
  return digest_fields("daemon-serving-target-set-layout", fields);
}

std::optional<std::string> prefetch_operation_id_for(const v2::PrefetchServingBindingRequest& request) {
  return request.has_operation_id() && !request.operation_id().empty()
      ? std::optional<std::string>(request.operation_id())
      : std::nullopt;
}

std::optional<std::string> publication_operation_id_for(
    const v2::PublishTargetReplicaRequest& request,
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope) {
  if (request.has_operation_id() && !request.operation_id().empty()) {
    return request.operation_id();
  }
  if (!scope.operation_id().empty()) {
    return scope.operation_id();
  }
  return std::nullopt;
}

std::optional<std::string> operation_id_for(const v2::CreateOwnedBindingRequest& request) {
  return request.has_operation_id() && !request.operation_id().empty()
      ? std::optional<std::string>(request.operation_id())
      : std::nullopt;
}

std::optional<std::string> operation_id_for(const v2::RefillOwnedBindingRequest& request) {
  return request.has_operation_id() && !request.operation_id().empty()
      ? std::optional<std::string>(request.operation_id())
      : std::nullopt;
}

std::string publication_target_layout_digest_for(
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope,
    const tensorcast::common::v1::ByteSpaceRef& normalized_byte_space) {
  if (!scope.target_layout_hash().empty()) {
    return absl::BytesToHexString(scope.target_layout_hash());
  }
  return digest_fields(
      "daemon-publication-target-layout",
      {
          scope.publication_id(),
          scope.selection().artifact_id(),
          byte_space_identity(normalized_byte_space),
          scope.device_uuid(),
          scope.binding_id(),
          scope.binding_layout_id(),
          scope.binding_value_id(),
          std::format("{}", scope.seal_generation()),
      });
}

v2::CollectivePolicy prefetch_collective_policy_for(
    const v2::PrefetchServingBindingRequest& request,
    uint32_t member_count) {
  if (member_count > 1 &&
      request.source().source_kind() == tensorcast::operation::v1::SERVING_BINDING_SOURCE_KIND_CHECKPOINT_ARTIFACT) {
    return v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST;
  }
  return v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE;
}

std::string binding_ownership_name(v2::BindingOwnership ownership) {
  switch (ownership) {
    case v2::BindingOwnership::BINDING_OWNERSHIP_DAEMON:
      return "binding_owned";
    case v2::BindingOwnership::BINDING_OWNERSHIP_CLIENT:
      return "binding_adopted";
    case v2::BindingOwnership::BINDING_OWNERSHIP_UNSPECIFIED:
    default:
      return "binding";
  }
}

std::vector<std::string> create_binding_release_policy_for(const v2::CreateBindingRequest& request) {
  if (request.ownership() == v2::BindingOwnership::BINDING_OWNERSHIP_DAEMON) {
    return {"release_handle_lease", "close_binding"};
  }
  if (request.has_initial_selection()) {
    return {"close_binding", "retire_publication_token"};
  }
  return {"close_binding"};
}

std::vector<std::string> binding_materialization_release_policy(bool staged_publish) {
  std::vector<std::string> release_policy{"release_handle_lease", "close_binding"};
  if (staged_publish) {
    release_policy.push_back("release_group_staged_acquire");
  } else {
    release_policy.push_back("retire_publication_token");
  }
  return release_policy;
}

std::vector<std::string> refill_release_policy(bool execution_only_mutable) {
  if (execution_only_mutable) {
    return {"preserve_binding_allocation", "mark_mutable_epoch"};
  }
  return {"replace_binding_current_value", "retire_publication_token"};
}

uint32_t assembly_requirement_count(const v2::AssemblyRequirementSetRef& requirements) {
  if (requirements.requirement_count() > 0) {
    return requirements.requirement_count();
  }
  return static_cast<uint32_t>(requirements.inline_requirements_size());
}

std::string assembly_attempt_layout_digest(const v2::AssemblyAttemptIntent& intent) {
  return digest_fields(
      "daemon-assembly-attempt-layout",
      {
          intent.layout_id(),
          intent.attempt_intent_digest(),
          std::format("{}", assembly_requirement_count(intent.requirements())),
          intent.requirements().carrier_form(),
      });
}

std::string assembly_seal_layout_digest(std::string_view assembly_id, std::string_view layout_id) {
  return digest_fields(
      "daemon-assembly-seal-layout",
      {
          std::string(assembly_id),
          std::string(layout_id),
      });
}

std::string assembly_closeout_coordination(const v2::AssemblyCloseoutContract& closeout_contract) {
  switch (closeout_contract.kind()) {
    case v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH:
      return "representation_publish_closeout";
    case v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY:
      return "source_publish_closeout";
    case v2::ASSEMBLY_CLOSEOUT_KIND_ROLLOUT_GATED_PUBLISH:
      return "rollout_gated_publish_closeout";
    case v2::ASSEMBLY_CLOSEOUT_KIND_UNSPECIFIED:
    default:
      return "assembly_closeout";
  }
}

std::vector<std::string> assembly_attempt_barriers_for(const v2::AssemblyAttemptIntent& intent) {
  std::vector<std::string> barriers{"requirement_registration"};
  if (assembly_requirement_count(intent.requirements()) > 1) {
    barriers.push_back("multi_requirement_readiness");
  }
  barriers.push_back("operation_coordinator");
  return barriers;
}

std::vector<std::string> assembly_seal_barriers_for(const v2::AssemblyCloseoutContract& closeout_contract) {
  std::vector<std::string> barriers{"readiness_cut", "seal_cut"};
  if (closeout_contract.kind() == v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH) {
    barriers.push_back("representation_publish_closeout");
  } else {
    barriers.push_back("source_publish_closeout");
  }
  return barriers;
}

bool has_prefix(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool contains_token(std::string_view value, std::string_view token) {
  return value.find(token) != std::string_view::npos;
}

void append_unique(std::vector<std::string>* values, std::string_view value) {
  const bool found = std::any_of(values->begin(), values->end(), [value](const std::string& current) {
    return std::string_view(current) == value;
  });
  if (found) {
    return;
  }
  values->emplace_back(value);
}

bool envelope_has_resource_authority(
    const ControllerRealizationResourceEnvelope& envelope,
    std::string_view authority) {
  return std::any_of(
      envelope.resource_authorities.begin(),
      envelope.resource_authorities.end(),
      [authority](const std::string& current) { return std::string_view(current) == authority; });
}

bool retained_backing_kind(std::string_view backing_kind) {
  return has_prefix(backing_kind, "daemon_retained") || backing_kind == "daemon_binding" ||
      backing_kind == "daemon_published_replica" || backing_kind == "assembly_workspace";
}

bool process_visible_handle_export(std::string_view export_kind) {
  return export_kind == "cuda_ipc_lease" || export_kind == "cpu_memfd_lease";
}

bool token_backed_export_lifetime(std::string_view lifetime_kind) {
  return lifetime_kind == "handle_lease" || lifetime_kind == "runtime_attachment";
}

ControllerBodyBackingIntentPlan body_backing_intent_for(std::string_view backing_kind, std::string_view export_kind) {
  ControllerBodyBackingIntentPlan intent;
  intent.preferred_residency = export_kind == "cuda_ipc_lease" ? "gpu" : "cpu";
  intent.retention_intent = retained_backing_kind(backing_kind) ? "retained" : "ephemeral";
  intent.stable_retention_requirement =
      backing_kind == "daemon_published_replica" || has_prefix(backing_kind, "daemon_retained") ? "prefer_stable"
                                                                                                : "none";
  if (export_kind == "publication_lease") {
    intent.sharing_intent = "remote_shareable";
  } else if (process_visible_handle_export(export_kind) || contains_token(export_kind, "binding_reservation")) {
    intent.sharing_intent = "local_read_mostly";
  } else {
    intent.sharing_intent = "private_local";
  }
  return intent;
}

std::vector<std::string> resource_authorities_for(
    std::string_view backing_kind,
    std::string_view export_kind,
    std::string_view projection_kind,
    std::string_view owner_kind) {
  std::vector<std::string> authorities;
  if (backing_kind == "caller_region") {
    append_unique(&authorities, "caller_allocation");
  } else if (backing_kind == "assembly_workspace") {
    append_unique(&authorities, "assembly_registry");
  } else {
    append_unique(&authorities, "BodyBackingManager");
  }
  if (contains_token(backing_kind, "binding") || contains_token(export_kind, "binding") ||
      contains_token(projection_kind, "binding") || contains_token(owner_kind, "binding")) {
    append_unique(&authorities, "BindingRegistry");
  }
  if (process_visible_handle_export(export_kind)) {
    append_unique(&authorities, "HandleLeaseRegistry");
    append_unique(&authorities, "SessionLifecycleManager");
    append_unique(&authorities, "LifecycleKernel");
  }
  if (export_kind == "publication_lease" || owner_kind == "runtime_publication") {
    append_unique(&authorities, "LifecycleKernel");
  }
  if (export_kind == "operation_lease") {
    append_unique(&authorities, "OperationLeaseRegistry");
  }
  if (owner_kind == "daemon_session" || retained_backing_kind(backing_kind)) {
    append_unique(&authorities, "SessionLifecycleManager");
  }
  return authorities;
}

std::string body_backing_manager_linkage_for(const ControllerRealizationPlan& plan) {
  if (!envelope_has_resource_authority(plan.resource_envelope, "BodyBackingManager")) {
    return "none";
  }
  return std::format(
      "BodyBackingManager:{}:{}:{}",
      plan.resource_envelope.backing_kind,
      plan.resource_envelope.body_backing_intent.retention_intent,
      plan.resource_envelope.body_backing_intent.sharing_intent);
}

std::string handle_lease_registry_linkage_for(const ControllerRealizationPlan& plan) {
  if (!envelope_has_resource_authority(plan.resource_envelope, "HandleLeaseRegistry")) {
    return "none";
  }
  return std::format("HandleLeaseRegistry:{}", plan.resource_envelope.export_kind);
}

std::string session_lifecycle_manager_linkage_for(const ControllerRealizationPlan& plan) {
  if (!envelope_has_resource_authority(plan.resource_envelope, "SessionLifecycleManager")) {
    return "none";
  }
  if (process_visible_handle_export(plan.resource_envelope.export_kind)) {
    return std::format("SessionLifecycleManager:{}:pid_use_lease", plan.resource_envelope.export_kind);
  }
  if (plan.resource_envelope.export_kind == "operation_lease") {
    return "SessionLifecycleManager:operation_lease";
  }
  if (contains_token(plan.resource_envelope.export_kind, "binding_reservation")) {
    return "SessionLifecycleManager:retained_binding_reservation";
  }
  if (plan.resource_envelope.owner_kind == "daemon_session") {
    return "SessionLifecycleManager:daemon_session_retention";
  }
  if (retained_backing_kind(plan.resource_envelope.backing_kind)) {
    return "SessionLifecycleManager:retained_backing";
  }
  return "SessionLifecycleManager:resource_lifecycle";
}

std::string lifecycle_kernel_linkage_for(const ControllerRealizationPlan& plan) {
  if (!envelope_has_resource_authority(plan.resource_envelope, "LifecycleKernel")) {
    return "none";
  }
  if (process_visible_handle_export(plan.resource_envelope.export_kind)) {
    return "LifecycleKernel:handle_lease_capability";
  }
  if (plan.resource_envelope.export_kind == "publication_lease" ||
      plan.resource_envelope.owner_kind == "runtime_publication") {
    return "LifecycleKernel:publication_capability";
  }
  return "LifecycleKernel:capability";
}

bool collective_policy_linkage_requested(v2::CollectivePolicy policy) {
  switch (policy) {
    case v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE:
    case v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST:
      return true;
    case v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE:
    case v2::CollectivePolicy::COLLECTIVE_POLICY_UNSPECIFIED:
    default:
      return false;
  }
}

std::string execution_commit_report_linkage_for(const ControllerRealizationPlan& plan) {
  const std::string_view target_kind = plan.target.target_kind;
  std::string_view commit_subject;
  if (target_kind == "caller_tensors") {
    commit_subject = "direct_write";
  } else if (target_kind == "binding_adopted") {
    commit_subject = "mapped_target";
  } else if (target_kind == "binding_owned" || target_kind == "binding_owned_refill") {
    commit_subject = "binding_materialization";
  } else if (target_kind == "tensor_dict" || target_kind == "retained_replica") {
    commit_subject = "replica_materialization";
  } else if (target_kind == "retained_binding" || target_kind == "target_set") {
    commit_subject = "retained_binding_materialization";
  } else {
    return "none";
  }
  return std::format(
      "ExecutionCommitReport:{}:{}",
      commit_subject,
      collective_policy_linkage_requested(plan.strategy.collective_policy) ? "collective_capable" : "single_executor");
}

ControllerRealizationResourceEnvelope make_resource_envelope(
    std::string_view backing_kind,
    std::string_view export_kind,
    std::string_view projection_kind,
    std::string_view owner_kind,
    const std::vector<std::string>& release_policy) {
  return ControllerRealizationResourceEnvelope{
      .backing_kind = std::string(backing_kind),
      .export_kind = std::string(export_kind),
      .projection_kind = std::string(projection_kind),
      .owner_kind = std::string(owner_kind),
      .release_policy = release_policy,
      .body_backing_intent = body_backing_intent_for(backing_kind, export_kind),
      .resource_authorities = resource_authorities_for(backing_kind, export_kind, projection_kind, owner_kind),
  };
}

absl::StatusOr<ControllerRealizationPlan> finalize_controller_realization_plan(ControllerRealizationPlan plan) {
  plan.resource_envelope.manager_linkage = controller_resource_manager_linkage_for(plan);
  auto status = validate_controller_realization_plan(plan);
  if (!status.ok()) {
    return status;
  }
  return plan;
}

template <typename RequestT>
absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan_impl(
    const RequestT& request,
    const NormalizedMaterializationRequestContext& request_context,
    const OperationTransportContext& transport_context,
    const GroupRealizationBeginContext* group_begin_context,
    std::string_view resolved_artifact_id,
    std::string_view target_kind,
    std::string_view capability,
    std::string_view projection_kind) {
  if (!request.has_target_layout()) {
    return absl::InvalidArgumentError("target_layout is required for controller realization plan");
  }
  if (request.device_uuid().empty()) {
    return absl::InvalidArgumentError("device_uuid is required for controller realization plan");
  }
  if (request.pid() <= 0) {
    return absl::InvalidArgumentError("pid is required for controller realization plan");
  }
  auto layout_digest_or = target_layout_digest(request.target_layout());
  if (!layout_digest_or.ok()) {
    return layout_digest_or.status();
  }

  const v2::GroupRealizationOptions* group_realization =
      request.has_group_realization() ? &request.group_realization() : nullptr;
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = std::string(target_kind),
      .resolved_artifact_id = std::string(resolved_artifact_id),
      .target_layout_digest = std::move(*layout_digest_or),
      .device_uuid = request.device_uuid(),
      .owner_pid = request.pid(),
      .layout_storage_count = static_cast<uint32_t>(request.target_layout().storages_size()),
      .layout_offset_count = static_cast<uint32_t>(request.target_layout().offsets_size()),
      .member_count =
          group_realization != nullptr && group_realization->enabled() ? group_realization->group().total_parts() : 1,
      .group_id = group_realization != nullptr && group_realization->enabled()
          ? std::optional<std::string>(group_realization->group().group_id())
          : std::nullopt,
      .part_id = group_begin_context != nullptr ? std::optional<std::string>(group_begin_context->part_id)
          : group_realization != nullptr && group_realization->enabled()
          ? std::optional<std::string>(group_realization->group().part_id())
          : std::nullopt,
      .operation_id = operation_id_for(request),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = source_selection_mode_for(group_begin_context),
      .source_coordination =
          transport_context.transport_scheduling_group.has_value() ? "group_realization_transport" : "single_request",
      .collective_policy = default_collective_policy_for_mapped_target(request_context.execution_topology),
      .group_barriers = group_barriers_for(group_realization),
      .version_set_id = group_begin_context != nullptr && !group_begin_context->version_set.version_set_id().empty()
          ? std::optional<std::string>(group_begin_context->version_set.version_set_id())
          : requested_version_set_id_for(group_realization),
      .transaction_id = group_begin_context != nullptr && !group_begin_context->transaction_id.empty()
          ? std::optional<std::string>(group_begin_context->transaction_id)
          : std::nullopt,
      .source_selection_digest = selection_digest_for(group_realization, group_begin_context, request.selection()),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = std::string(capability),
      .export_lifetime_kind = "request_scoped",
      .release_strictness = "strict",
      .mutability_contract = "caller_mutable",
      .release_policy = {"release_external_target_storage_lease"},
      .staged_value_count =
          group_realization != nullptr && group_realization->enabled() && group_realization->require_staged_publish()
          ? 1U
          : 0U,
      .acquire_claim_count = group_begin_context != nullptr ? 1U : 0U,
      .publish_barrier =
          group_realization != nullptr && group_realization->enabled() && group_realization->require_staged_publish(),
  };
  plan.resource_envelope = make_resource_envelope(
      "caller_region", "registered_region_direct_write", projection_kind, "caller_pid", plan.lifecycle.release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_prefetch_target_set_realization_plan(
    const v2::PrefetchServingBindingRequest& request) {
  const auto& target = request.serving_binding_set_target();
  if (target.members_size() == 0) {
    return absl::InvalidArgumentError("serving_binding_set_target.members is required for controller realization plan");
  }
  const uint32_t member_count = static_cast<uint32_t>(target.members_size());
  const bool per_part_source =
      request.source().source_kind() == tensorcast::operation::v1::SERVING_BINDING_SOURCE_KIND_SERVING_ARTIFACT_SET;
  std::vector<std::string> release_policy{"release_binding_reservations"};
  if (request.has_group_realization() && request.group_realization().require_staged_publish()) {
    release_policy.push_back("release_group_staged_acquire");
  }
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "target_set",
      .resolved_artifact_id = request.source_selection().artifact_id(),
      .target_layout_digest = target_set_layout_digest_for(target),
      .device_uuid = "target_set",
      .owner_pid = 0,
      .layout_storage_count = 0,
      .layout_offset_count = 0,
      .member_count = member_count,
      .group_id = target.group_id().empty() ? std::nullopt : std::optional<std::string>(target.group_id()),
      .part_id = std::nullopt,
      .operation_id = prefetch_operation_id_for(request),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = source_selection_mode_for(
          request.has_group_realization() ? &request.group_realization() : nullptr, per_part_source),
      .source_coordination = request.has_group_realization() && request.group_realization().enabled()
          ? "group_realization_transport"
          : "same_daemon_session",
      .collective_policy = prefetch_collective_policy_for(request, member_count),
      .group_barriers = group_barriers_for(request.has_group_realization() ? &request.group_realization() : nullptr),
      .version_set_id =
          requested_version_set_id_for(request.has_group_realization() ? &request.group_realization() : nullptr),
      .transaction_id = std::nullopt,
      .source_selection_digest = !request.source().artifact_selection_digest().empty()
          ? std::optional<std::string>(request.source().artifact_selection_digest())
          : selection_digest_for(
                request.has_group_realization() ? &request.group_realization() : nullptr,
                nullptr,
                request.source_selection()),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "target_set",
      .export_lifetime_kind = "daemon_retained",
      .release_strictness = "strict",
      .mutability_contract = "binding_controlled_read_only",
      .release_policy = release_policy,
      .staged_value_count =
          request.has_group_realization() && request.group_realization().require_staged_publish() ? member_count : 0,
      .acquire_claim_count = member_count,
      .publish_barrier = request.has_group_realization() && request.group_realization().require_staged_publish(),
  };
  plan.resource_envelope = make_resource_envelope(
      "daemon_retained_binding_set",
      "binding_reservation_set",
      "target_set",
      "binding_reservation_capability_set",
      release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_prefetch_member_realization_plan(
    const v2::PrefetchServingBindingRequest& request) {
  const auto& target = request.serving_binding_target();
  if (target.device_uuid().empty()) {
    return absl::InvalidArgumentError("serving_binding_target.device_uuid is required for controller realization plan");
  }
  if (!target.has_member() || target.member().member_id().empty()) {
    return absl::InvalidArgumentError("serving_binding_target.member is required for controller realization plan");
  }
  std::vector<std::string> release_policy{"release_binding_reservation"};
  if (request.has_group_realization() && request.group_realization().require_staged_publish()) {
    release_policy.push_back("release_group_staged_acquire");
  }
  const uint32_t member_count = target.member().member_count() == 0 ? 1 : target.member().member_count();
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "retained_binding",
      .resolved_artifact_id = request.source_selection().artifact_id(),
      .target_layout_digest = target_layout_digest_for_serving_target(target),
      .device_uuid = target.device_uuid(),
      .owner_pid = 0,
      .layout_storage_count = 0,
      .layout_offset_count = 0,
      .member_count = member_count,
      .group_id =
          target.member().group_id().empty() ? std::nullopt : std::optional<std::string>(target.member().group_id()),
      .part_id = target.member().member_id(),
      .operation_id = prefetch_operation_id_for(request),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = source_selection_mode_for(
          request.has_group_realization() ? &request.group_realization() : nullptr,
          /*per_part_source=*/false),
      .source_coordination = request.has_group_realization() && request.group_realization().enabled()
          ? "group_realization_transport"
          : "same_daemon_session",
      .collective_policy = prefetch_collective_policy_for(request, member_count),
      .group_barriers = group_barriers_for(request.has_group_realization() ? &request.group_realization() : nullptr),
      .version_set_id =
          requested_version_set_id_for(request.has_group_realization() ? &request.group_realization() : nullptr),
      .transaction_id = std::nullopt,
      .source_selection_digest = !request.source().artifact_selection_digest().empty()
          ? std::optional<std::string>(request.source().artifact_selection_digest())
          : selection_digest_for(
                request.has_group_realization() ? &request.group_realization() : nullptr,
                nullptr,
                request.source_selection()),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "retained_binding",
      .export_lifetime_kind = "daemon_retained",
      .release_strictness = "strict",
      .mutability_contract = "binding_controlled_read_only",
      .release_policy = release_policy,
      .staged_value_count =
          request.has_group_realization() && request.group_realization().require_staged_publish() ? 1U : 0U,
      .acquire_claim_count = 1,
      .publish_barrier = request.has_group_realization() && request.group_realization().require_staged_publish(),
  };
  plan.resource_envelope = make_resource_envelope(
      "daemon_retained_binding",
      "binding_reservation",
      "prefetch_handoff",
      "binding_reservation_capability",
      release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

std::string acquire_target_subject(const v2::AcquireBindingValueRequest& request) {
  if (!request.binding_value_ref().binding_id().empty()) {
    return absl::StrCat("binding:", request.binding_value_ref().binding_id());
  }
  if (!request.local_serving_ref().empty()) {
    return absl::StrCat("local_serving_ref:", request.local_serving_ref());
  }
  return "binding:unknown";
}

uint32_t acquire_member_count(const v2::AcquireBindingValueRequest& request) {
  if (request.reservation_capability().has_member() && request.reservation_capability().member().member_count() > 0) {
    return request.reservation_capability().member().member_count();
  }
  if (request.has_expected_member() && request.expected_member().member_count() > 0) {
    return request.expected_member().member_count();
  }
  return 1;
}

std::optional<std::string> acquire_group_id(const v2::AcquireBindingValueRequest& request) {
  if (request.reservation_capability().has_member() && !request.reservation_capability().member().group_id().empty()) {
    return request.reservation_capability().member().group_id();
  }
  if (request.has_expected_member() && !request.expected_member().group_id().empty()) {
    return request.expected_member().group_id();
  }
  return std::nullopt;
}

std::optional<std::string> acquire_part_id(const v2::AcquireBindingValueRequest& request) {
  if (request.has_group_realization_acquire() && !request.group_realization_acquire().part_id().empty()) {
    return request.group_realization_acquire().part_id();
  }
  if (request.reservation_capability().has_member() && !request.reservation_capability().member().member_id().empty()) {
    return request.reservation_capability().member().member_id();
  }
  if (request.has_expected_member() && !request.expected_member().member_id().empty()) {
    return request.expected_member().member_id();
  }
  return std::nullopt;
}

} // namespace

ControllerResourceManagerLinkagePlan controller_resource_manager_linkage_for(const ControllerRealizationPlan& plan) {
  return ControllerResourceManagerLinkagePlan{
      .body_backing_manager = body_backing_manager_linkage_for(plan),
      .handle_lease_registry = handle_lease_registry_linkage_for(plan),
      .session_lifecycle_manager = session_lifecycle_manager_linkage_for(plan),
      .lifecycle_kernel = lifecycle_kernel_linkage_for(plan),
      .execution_commit_report = execution_commit_report_linkage_for(plan),
  };
}

void attach_controller_realization_plan_span_attrs(RpcContext& rctx, const ControllerRealizationPlan& plan) {
  auto& span = rctx.span();
  span->SetAttribute("tc.realization.target_kind", plan.target.target_kind);
  span->SetAttribute("tc.realization.source_selection_mode", plan.strategy.source_selection_mode);
  span->SetAttribute("tc.realization.lifecycle_capability", plan.lifecycle.capability);
  span->SetAttribute("tc.realization.resource_backing", plan.resource_envelope.backing_kind);
  span->SetAttribute("tc.realization.resource_export", plan.resource_envelope.export_kind);
  span->SetAttribute("tc.realization.resource_projection", plan.resource_envelope.projection_kind);
  span->SetAttribute("tc.realization.resource_owner", plan.resource_envelope.owner_kind);
  span->SetAttribute(
      "tc.realization.manager.body_backing", plan.resource_envelope.manager_linkage.body_backing_manager);
  span->SetAttribute(
      "tc.realization.manager.handle_lease", plan.resource_envelope.manager_linkage.handle_lease_registry);
  span->SetAttribute(
      "tc.realization.manager.session_lifecycle", plan.resource_envelope.manager_linkage.session_lifecycle_manager);
  span->SetAttribute(
      "tc.realization.manager.lifecycle_kernel", plan.resource_envelope.manager_linkage.lifecycle_kernel);
  span->SetAttribute(
      "tc.realization.manager.execution_commit", plan.resource_envelope.manager_linkage.execution_commit_report);
}

absl::Status validate_controller_realization_plan(const ControllerRealizationPlan& plan) {
  if (plan.target.target_kind.empty()) {
    return absl::InvalidArgumentError("controller realization plan requires target_kind");
  }
  if (plan.lifecycle.capability.empty()) {
    return absl::InvalidArgumentError("controller realization plan requires lifecycle capability");
  }
  if (plan.lifecycle.export_lifetime_kind.empty()) {
    return absl::InvalidArgumentError("controller realization plan requires export_lifetime_kind");
  }
  if (plan.resource_envelope.backing_kind.empty() || plan.resource_envelope.export_kind.empty() ||
      plan.resource_envelope.projection_kind.empty() || plan.resource_envelope.owner_kind.empty()) {
    return absl::InvalidArgumentError("controller realization plan requires resource envelope identity fields");
  }
  if (plan.resource_envelope.resource_authorities.empty()) {
    return absl::InvalidArgumentError("controller realization plan requires resource authorities");
  }
  if (plan.resource_envelope.release_policy != plan.lifecycle.release_policy) {
    return absl::InvalidArgumentError("resource envelope release_policy must match lifecycle release_policy");
  }
  if (process_visible_handle_export(plan.resource_envelope.export_kind) &&
      !token_backed_export_lifetime(plan.lifecycle.export_lifetime_kind)) {
    return absl::InvalidArgumentError("process-visible controller exports require token-backed lifetime");
  }
  const ControllerResourceManagerLinkagePlan expected_linkage = controller_resource_manager_linkage_for(plan);
  if (plan.resource_envelope.manager_linkage != expected_linkage) {
    return absl::InvalidArgumentError(
        "controller realization plan resource manager linkage must match resource envelope authorities");
  }
  return absl::OkStatus();
}

bool controller_plan_has_resource_authority(const ControllerRealizationPlan& plan, std::string_view authority) {
  return envelope_has_resource_authority(plan.resource_envelope, authority);
}

absl::Status require_controller_resource_authority(
    const ControllerRealizationPlan& plan,
    std::string_view authority,
    std::string_view action) {
  if (controller_plan_has_resource_authority(plan, authority)) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(std::format("{} requires controller resource authority {}", action, authority));
}

absl::Status require_controller_export_kind(
    const ControllerRealizationPlan& plan,
    std::string_view export_kind,
    std::string_view action) {
  if (plan.resource_envelope.export_kind == export_kind) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(
      std::format(
          "{} expected controller export_kind={} but admitted export_kind={}",
          action,
          export_kind,
          plan.resource_envelope.export_kind));
}

absl::Status validate_controller_process_visible_export_authorities(
    const ControllerRealizationPlan& plan,
    std::string_view action) {
  if (!process_visible_handle_export(plan.resource_envelope.export_kind)) {
    return absl::OkStatus();
  }
  if (auto status = require_controller_resource_authority(plan, "HandleLeaseRegistry", action); !status.ok()) {
    return status;
  }
  if (auto status = require_controller_resource_authority(plan, "SessionLifecycleManager", action); !status.ok()) {
    return status;
  }
  return require_controller_resource_authority(plan, "LifecycleKernel", action);
}

absl::StatusOr<RetrievalPolicy> resolve_retrieval_policy(const v2::SourcePolicy* policy) {
  RetrievalPolicy resolved;
  if (policy != nullptr) {
    if (policy->preference() != v2::SourcePreference::SOURCE_PREFERENCE_UNSPECIFIED) {
      resolved.preference = to_hint_preference(policy->preference());
    }
    if (policy->has_allow_p2p()) {
      resolved.allow_p2p = policy->allow_p2p();
    }
    if (policy->has_allow_disk()) {
      resolved.allow_disk = policy->allow_disk();
    }
  }
  if (auto policy_status = store::loading::validate_retrieval_policy(resolved); !policy_status.ok()) {
    return policy_status;
  }
  return resolved;
}

std::optional<store::loading::CollectiveLoadGroupHint> resolve_collective_group_hint(
    const v2::CollectiveLoadGroup* group) {
  if (group == nullptr || group->group_id().empty()) {
    return std::nullopt;
  }
  const uint32_t world_size = group->world_size();
  const uint32_t rank = group->rank();
  if (world_size <= 1 || rank >= world_size) {
    return std::nullopt;
  }
  return store::loading::CollectiveLoadGroupHint{
      .group_id = group->group_id(),
      .world_size = world_size,
      .rank = rank,
  };
}

absl::StatusOr<ExecutionTopologyContext> resolve_source_execution_topology(
    const v2::SourceExecutionTopology* topology) {
  ExecutionTopologyContext execution_topology;
  if (topology == nullptr) {
    return execution_topology;
  }
  if (topology->has_collective_load_group()) {
    auto group_hint = resolve_collective_group_hint(&topology->collective_load_group());
    if (!group_hint.has_value()) {
      return absl::InvalidArgumentError("execution_topology.collective_load_group is invalid");
    }
    execution_topology.collective_load_group = std::move(group_hint);
  }
  execution_topology.source_locality = to_source_locality(topology->source_locality());
  if (topology->has_source_sharing_domain() && !topology->source_sharing_domain().empty()) {
    execution_topology.source_sharing_domain = topology->source_sharing_domain();
  }
  return execution_topology;
}

absl::StatusOr<v2::CollectivePolicy> resolve_collective_policy(
    v2::CollectivePolicy requested,
    const ExecutionTopologyContext& execution_topology) {
  const bool has_collective_group = execution_topology.collective_load_group.has_value();
  if (requested == v2::CollectivePolicy::COLLECTIVE_POLICY_UNSPECIFIED) {
    return has_collective_group ? v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE
                                : v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE;
  }
  if (requested == v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE && has_collective_group) {
    return absl::InvalidArgumentError("collective_policy=disable_collective conflicts with a collective_load_group");
  }
  if (requested != v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE && !has_collective_group) {
    return absl::InvalidArgumentError("collective_policy requires execution_topology.collective_load_group");
  }
  return requested;
}

v2::CollectivePolicy default_collective_policy_for_mapped_target(const ExecutionTopologyContext& execution_topology) {
  return execution_topology.collective_load_group.has_value()
      ? v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST
      : v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE;
}

bool collective_policy_requests_collective(v2::CollectivePolicy policy) {
  return policy == v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE ||
      policy == v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST;
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::MaterializeIntoTargetRequest& request,
    const NormalizedMaterializationRequestContext& request_context,
    const OperationTransportContext& transport_context,
    const GroupRealizationBeginContext* group_begin_context,
    std::string_view resolved_artifact_id) {
  return build_controller_realization_plan_impl(
      request,
      request_context,
      transport_context,
      group_begin_context,
      resolved_artifact_id,
      "caller_tensors",
      "caller_tensors",
      "completion");
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::MaterializeIntoMappedTargetRequest& request,
    const NormalizedMaterializationRequestContext& request_context,
    const OperationTransportContext& transport_context,
    const GroupRealizationBeginContext* group_begin_context,
    std::string_view resolved_artifact_id) {
  return build_controller_realization_plan_impl(
      request,
      request_context,
      transport_context,
      group_begin_context,
      resolved_artifact_id,
      "binding_adopted",
      "binding_adopted",
      "binding");
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::MaterializeReplicaRequest& request,
    const NormalizedMaterializationRequestContext& request_context,
    const OperationTransportContext& transport_context,
    const GroupRealizationBeginContext* group_begin_context,
    std::string_view resolved_artifact_id,
    const tensorcast::common::v1::ArtifactSelection& resolved_selection,
    bool cpu_target,
    bool no_lease) {
  if (resolved_artifact_id.empty()) {
    return absl::InvalidArgumentError("resolved_artifact_id is required for controller realization plan");
  }
  if (request.wait_for_completion() && no_lease) {
    return absl::InvalidArgumentError("lease_mode=NO_LEASE cannot export a tensor_dict realization");
  }
  const v2::GroupRealizationOptions* group_realization =
      request.has_group_realization() && request.group_realization().enabled() ? &request.group_realization() : nullptr;
  std::vector<std::string> release_policy = replica_release_policy_for(request, no_lease);
  const std::string target_kind = replica_target_kind_for(request);
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = target_kind,
      .resolved_artifact_id = std::string(resolved_artifact_id),
      .target_layout_digest = target_layout_digest_for_replica(resolved_selection, request, resolved_artifact_id),
      .device_uuid = cpu_target           ? "CPU"
          : request.device_uuid().empty() ? "GPU"
                                          : request.device_uuid(),
      .owner_pid = no_lease ? 0 : request.pid(),
      .layout_storage_count = 0,
      .layout_offset_count = 0,
      .member_count = group_realization != nullptr && group_realization->group().total_parts() > 0
          ? group_realization->group().total_parts()
          : 1,
      .group_id = group_realization != nullptr && !group_realization->group().group_id().empty()
          ? std::optional<std::string>(group_realization->group().group_id())
          : std::nullopt,
      .part_id = group_begin_context != nullptr && !group_begin_context->part_id.empty()
          ? std::optional<std::string>(group_begin_context->part_id)
          : group_realization != nullptr && !group_realization->group().part_id().empty()
          ? std::optional<std::string>(group_realization->group().part_id())
          : std::nullopt,
      .operation_id = std::nullopt,
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = source_selection_mode_for(group_begin_context),
      .source_coordination =
          transport_context.transport_scheduling_group.has_value() ? "group_realization_transport" : "single_request",
      .collective_policy = replica_collective_policy_for(request_context),
      .group_barriers = group_barriers_for(group_realization),
      .version_set_id = group_begin_context != nullptr && !group_begin_context->version_set.version_set_id().empty()
          ? std::optional<std::string>(group_begin_context->version_set.version_set_id())
          : requested_version_set_id_for(group_realization),
      .transaction_id = group_begin_context != nullptr && !group_begin_context->transaction_id.empty()
          ? std::optional<std::string>(group_begin_context->transaction_id)
          : std::nullopt,
      .source_selection_digest = selection_digest_for(group_realization, group_begin_context, resolved_selection),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = target_kind,
      .export_lifetime_kind = no_lease ? "daemon_retained" : "handle_lease",
      .release_strictness = "strict",
      .mutability_contract =
          cpu_target && request.wait_for_completion() ? "read_mostly_private_copy" : "borrowed_read_only",
      .release_policy = release_policy,
      .staged_value_count = 0,
      .acquire_claim_count = no_lease ? 0U : 1U,
      .publish_barrier = false,
  };
  plan.resource_envelope = make_resource_envelope(
      request.wait_for_completion() ? "daemon_replica" : "daemon_retained_replica",
      no_lease ? "none" : (cpu_target ? "cpu_memfd_lease" : "cuda_ipc_lease"),
      request.wait_for_completion() ? "tensor_dict" : "operation_ticket",
      no_lease ? "daemon_session" : "caller_pid",
      release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(const v2::CreateBindingRequest& request) {
  if (request.ownership() == v2::BindingOwnership::BINDING_OWNERSHIP_UNSPECIFIED) {
    return absl::InvalidArgumentError("binding ownership is required for controller realization plan");
  }
  if (!request.has_target_layout()) {
    return absl::InvalidArgumentError("target_layout is required for controller realization plan");
  }
  if (request.device_uuid().empty()) {
    return absl::InvalidArgumentError("device_uuid is required for controller realization plan");
  }
  if (request.pid() <= 0) {
    return absl::InvalidArgumentError("pid is required for controller realization plan");
  }
  auto layout_digest_or = target_layout_digest(request.target_layout());
  if (!layout_digest_or.ok()) {
    return layout_digest_or.status();
  }

  const std::vector<std::string> release_policy = create_binding_release_policy_for(request);
  const std::string target_kind = binding_ownership_name(request.ownership());
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = target_kind,
      .resolved_artifact_id = request.has_initial_selection()
          ? (request.has_source_artifact_id() ? request.source_artifact_id()
                                              : request.initial_selection().artifact_id())
          : std::string(),
      .target_layout_digest = std::move(*layout_digest_or),
      .device_uuid = request.device_uuid(),
      .owner_pid = request.pid(),
      .layout_storage_count = static_cast<uint32_t>(request.target_layout().storages_size()),
      .layout_offset_count = static_cast<uint32_t>(request.target_layout().offsets_size()),
      .member_count = 1,
      .group_id = std::nullopt,
      .part_id = std::nullopt,
      .operation_id = std::nullopt,
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = request.has_initial_selection() ? "single_selection" : "none",
      .source_coordination = request.has_initial_selection() ? "binding_initial_value" : "binding_allocation",
      .collective_policy = v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers = {},
      .version_set_id = std::nullopt,
      .transaction_id = std::nullopt,
      .source_selection_digest = request.has_initial_selection()
          ? selection_digest_for(nullptr, nullptr, request.initial_selection())
          : std::nullopt,
  };
  const bool daemon_owned = request.ownership() == v2::BindingOwnership::BINDING_OWNERSHIP_DAEMON;
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = target_kind,
      .export_lifetime_kind = daemon_owned ? "handle_lease" : "binding_registry",
      .release_strictness = "strict",
      .mutability_contract = daemon_owned ? "binding_controlled_mutable" : "caller_region_borrowed",
      .release_policy = release_policy,
      .staged_value_count = 0,
      .acquire_claim_count = daemon_owned ? 1U : 0U,
      .publish_barrier = false,
  };
  plan.resource_envelope = make_resource_envelope(
      daemon_owned ? "daemon_binding" : "caller_region",
      daemon_owned ? "cuda_ipc_lease" : "publication_token_or_none",
      "binding",
      "caller_pid",
      release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::CreateOwnedBindingRequest& request,
    const NormalizedMaterializationRequestContext& request_context,
    const OperationTransportContext& transport_context,
    const GroupRealizationBeginContext* group_begin_context,
    std::string_view resolved_artifact_id,
    const tensorcast::common::v1::ArtifactSelection& resolved_selection) {
  if (resolved_artifact_id.empty()) {
    return absl::InvalidArgumentError("resolved_artifact_id is required for controller realization plan");
  }
  if (!request.has_target_layout()) {
    return absl::InvalidArgumentError("target_layout is required for controller realization plan");
  }
  if (request.device_uuid().empty()) {
    return absl::InvalidArgumentError("device_uuid is required for controller realization plan");
  }
  if (request.pid() <= 0) {
    return absl::InvalidArgumentError("pid is required for controller realization plan");
  }
  auto layout_digest_or = target_layout_digest(request.target_layout());
  if (!layout_digest_or.ok()) {
    return layout_digest_or.status();
  }

  const v2::GroupRealizationOptions* group_realization =
      request.has_group_realization() && request.group_realization().enabled() ? &request.group_realization() : nullptr;
  const bool staged_publish = group_realization != nullptr && group_realization->require_staged_publish();
  const std::vector<std::string> release_policy = binding_materialization_release_policy(staged_publish);
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "binding_owned",
      .resolved_artifact_id = std::string(resolved_artifact_id),
      .target_layout_digest = std::move(*layout_digest_or),
      .device_uuid = request.device_uuid(),
      .owner_pid = request.pid(),
      .layout_storage_count = static_cast<uint32_t>(request.target_layout().storages_size()),
      .layout_offset_count = static_cast<uint32_t>(request.target_layout().offsets_size()),
      .member_count = group_realization != nullptr ? group_realization->group().total_parts() : 1,
      .group_id = group_realization != nullptr && !group_realization->group().group_id().empty()
          ? std::optional<std::string>(group_realization->group().group_id())
          : std::nullopt,
      .part_id = group_begin_context != nullptr && !group_begin_context->part_id.empty()
          ? std::optional<std::string>(group_begin_context->part_id)
          : group_realization != nullptr && !group_realization->group().part_id().empty()
          ? std::optional<std::string>(group_realization->group().part_id())
          : std::nullopt,
      .operation_id = operation_id_for(request),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = source_selection_mode_for(group_begin_context),
      .source_coordination =
          transport_context.transport_scheduling_group.has_value() ? "group_realization_transport" : "single_request",
      .collective_policy = default_collective_policy_for_mapped_target(request_context.execution_topology),
      .group_barriers = group_barriers_for(group_realization),
      .version_set_id = group_begin_context != nullptr && !group_begin_context->version_set.version_set_id().empty()
          ? std::optional<std::string>(group_begin_context->version_set.version_set_id())
          : std::nullopt,
      .transaction_id = group_begin_context != nullptr && !group_begin_context->transaction_id.empty()
          ? std::optional<std::string>(group_begin_context->transaction_id)
          : std::nullopt,
      .source_selection_digest = selection_digest_for(group_realization, group_begin_context, resolved_selection),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "binding_owned",
      .export_lifetime_kind = "handle_lease",
      .release_strictness = "strict",
      .mutability_contract = "binding_controlled_read_only",
      .release_policy = release_policy,
      .staged_value_count = staged_publish ? 1U : 0U,
      .acquire_claim_count = group_begin_context != nullptr ? 1U : 0U,
      .publish_barrier = staged_publish,
  };
  plan.resource_envelope = make_resource_envelope(
      "daemon_binding", "cuda_ipc_lease", staged_publish ? "staged_binding" : "binding", "caller_pid", release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::RefillOwnedBindingRequest& request,
    const NormalizedMaterializationRequestContext& request_context,
    const OperationTransportContext& transport_context,
    const GroupRealizationBeginContext* group_begin_context,
    std::string_view resolved_artifact_id,
    const tensorcast::common::v1::ArtifactSelection& resolved_selection,
    const v2::TargetLayout& target_layout,
    std::string_view device_uuid,
    int32_t owner_pid,
    bool mapped,
    v2::CollectivePolicy collective_policy,
    bool execution_only_mutable) {
  if (request.binding_id().empty()) {
    return absl::InvalidArgumentError("binding_id is required for controller realization plan");
  }
  if (resolved_artifact_id.empty()) {
    return absl::InvalidArgumentError("resolved_artifact_id is required for controller realization plan");
  }
  if (device_uuid.empty()) {
    return absl::InvalidArgumentError("device_uuid is required for controller realization plan");
  }
  if (owner_pid <= 0) {
    return absl::InvalidArgumentError("owner_pid is required for controller realization plan");
  }
  auto layout_digest_or = target_layout_digest(target_layout);
  if (!layout_digest_or.ok()) {
    return layout_digest_or.status();
  }

  const v2::GroupRealizationOptions* group_realization =
      request.has_group_realization() && request.group_realization().enabled() ? &request.group_realization() : nullptr;
  const std::vector<std::string> release_policy = refill_release_policy(execution_only_mutable);
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "binding_owned_refill",
      .resolved_artifact_id = std::string(resolved_artifact_id),
      .target_layout_digest = std::move(*layout_digest_or),
      .device_uuid = std::string(device_uuid),
      .owner_pid = owner_pid,
      .layout_storage_count = static_cast<uint32_t>(target_layout.storages_size()),
      .layout_offset_count = static_cast<uint32_t>(target_layout.offsets_size()),
      .member_count = group_realization != nullptr ? group_realization->group().total_parts() : 1,
      .group_id = group_realization != nullptr && !group_realization->group().group_id().empty()
          ? std::optional<std::string>(group_realization->group().group_id())
          : std::nullopt,
      .part_id = group_begin_context != nullptr && !group_begin_context->part_id.empty()
          ? std::optional<std::string>(group_begin_context->part_id)
          : group_realization != nullptr && !group_realization->group().part_id().empty()
          ? std::optional<std::string>(group_realization->group().part_id())
          : std::nullopt,
      .operation_id = operation_id_for(request),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = source_selection_mode_for(group_begin_context),
      .source_coordination =
          transport_context.transport_scheduling_group.has_value() ? "group_realization_transport" : "single_request",
      .collective_policy = request_context.execution_topology.collective_load_group.has_value()
          ? collective_policy
          : v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers = group_barriers_for(group_realization),
      .version_set_id = group_begin_context != nullptr && !group_begin_context->version_set.version_set_id().empty()
          ? std::optional<std::string>(group_begin_context->version_set.version_set_id())
          : std::nullopt,
      .transaction_id = group_begin_context != nullptr && !group_begin_context->transaction_id.empty()
          ? std::optional<std::string>(group_begin_context->transaction_id)
          : std::nullopt,
      .source_selection_digest = selection_digest_for(group_realization, group_begin_context, resolved_selection),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "binding_owned",
      .export_lifetime_kind = "binding_current_value",
      .release_strictness = "strict",
      .mutability_contract = execution_only_mutable ? "binding_controlled_mutable" : "binding_controlled_read_only",
      .release_policy = release_policy,
      .staged_value_count = 0,
      .acquire_claim_count = group_begin_context != nullptr ? 1U : 0U,
      .publish_barrier = group_realization != nullptr && group_realization->require_staged_publish(),
  };
  plan.resource_envelope = make_resource_envelope(
      "daemon_binding",
      execution_only_mutable ? "none" : "publication_token_or_none",
      mapped ? "binding_mapped_refill" : "binding_refill",
      "binding_registry",
      release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::StartAssemblyAttemptRequest& request,
    const v2::AssemblyAttemptIntent& intent,
    std::string_view attempt_id,
    std::string_view workspace_assembly_id,
    std::string_view operation_id) {
  if (intent.layout_id().empty()) {
    return absl::InvalidArgumentError("assembly attempt layout_id is required for controller realization plan");
  }
  if (attempt_id.empty()) {
    return absl::InvalidArgumentError("assembly attempt_id is required for controller realization plan");
  }
  if (workspace_assembly_id.empty()) {
    return absl::InvalidArgumentError("assembly workspace_assembly_id is required for controller realization plan");
  }
  if (operation_id.empty()) {
    return absl::InvalidArgumentError("assembly operation_id is required for controller realization plan");
  }
  const bool representation_publish = request.has_representation_publish_spec();
  const std::vector<std::string> release_policy{
      "release_operation_lease",
      "stop_coordinator_keepalive",
      "close_assembly_attempt",
  };
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = representation_publish ? "representation_publish_assembly_attempt" : "assembly_attempt",
      .resolved_artifact_id = std::string(workspace_assembly_id),
      .target_layout_digest = assembly_attempt_layout_digest(intent),
      .device_uuid = "assembly",
      .owner_pid = 0,
      .layout_storage_count = 0,
      .layout_offset_count = assembly_requirement_count(intent.requirements()),
      .member_count = assembly_requirement_count(intent.requirements()),
      .group_id = std::string(attempt_id),
      .part_id = std::nullopt,
      .operation_id = std::string(operation_id),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = "assembly_requirements",
      .source_coordination = "assembly_attempt_coordinator",
      .collective_policy = v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers = assembly_attempt_barriers_for(intent),
      .version_set_id = std::nullopt,
      .transaction_id = std::string(attempt_id),
      .source_selection_digest = intent.attempt_intent_digest().empty()
          ? std::nullopt
          : std::optional<std::string>(intent.attempt_intent_digest()),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "assembly_attempt",
      .export_lifetime_kind = "operation_lease",
      .release_strictness = "strict",
      .mutability_contract = "workspace_mutable_until_seal",
      .release_policy = release_policy,
      .staged_value_count = assembly_requirement_count(intent.requirements()),
      .acquire_claim_count = 1,
      .publish_barrier = representation_publish,
  };
  plan.resource_envelope = make_resource_envelope(
      "assembly_workspace", "operation_lease", "operation_ref", "daemon_coordinator", release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::SealAssemblyAttemptRequest& request,
    const v2::AssemblyAttemptRecord& record,
    std::string_view operation_id) {
  if (request.attempt_id().empty()) {
    return absl::InvalidArgumentError("assembly attempt_id is required for controller realization plan");
  }
  if (record.workspace_assembly_id().empty()) {
    return absl::InvalidArgumentError("assembly workspace_assembly_id is required for controller realization plan");
  }
  if (operation_id.empty()) {
    return absl::InvalidArgumentError("assembly operation_id is required for controller realization plan");
  }
  const std::vector<std::string> release_policy{
      "release_operation_lease",
      "finalize_slot_occupancies",
      "publish_workspace_seal_binding",
  };
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "assembly_attempt_seal",
      .resolved_artifact_id = record.workspace_assembly_id(),
      .target_layout_digest = assembly_attempt_layout_digest(record.intent()),
      .device_uuid = "assembly",
      .owner_pid = 0,
      .layout_storage_count = 0,
      .layout_offset_count = assembly_requirement_count(record.intent().requirements()),
      .member_count = assembly_requirement_count(record.intent().requirements()),
      .group_id = request.attempt_id(),
      .part_id = std::nullopt,
      .operation_id = std::string(operation_id),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = "assembly_requirements",
      .source_coordination = assembly_closeout_coordination(record.intent().closeout_contract()),
      .collective_policy = v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers = assembly_seal_barriers_for(record.intent().closeout_contract()),
      .version_set_id = std::nullopt,
      .transaction_id = request.attempt_id(),
      .source_selection_digest = record.intent().attempt_intent_digest().empty()
          ? std::nullopt
          : std::optional<std::string>(record.intent().attempt_intent_digest()),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "assembly_seal",
      .export_lifetime_kind = "operation_lease",
      .release_strictness = "strict",
      .mutability_contract = "sealed_artifact_immutable",
      .release_policy = release_policy,
      .staged_value_count = 0,
      .acquire_claim_count = 1,
      .publish_barrier = true,
  };
  plan.resource_envelope = make_resource_envelope(
      "assembly_workspace", "operation_lease", "operation_ref", "daemon_coordinator", release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(const v2::SealAssemblyRequest& request) {
  if (request.assembly_id().empty()) {
    return absl::InvalidArgumentError("assembly_id is required for controller realization plan");
  }
  const std::vector<std::string> release_policy{"commit_sealed_artifact"};
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "assembly_seal",
      .resolved_artifact_id = request.assembly_id(),
      .target_layout_digest = assembly_seal_layout_digest(request.assembly_id(), std::string_view()),
      .device_uuid = "assembly",
      .owner_pid = 0,
      .layout_storage_count = 0,
      .layout_offset_count = 0,
      .member_count = 1,
      .group_id = std::nullopt,
      .part_id = std::nullopt,
      .operation_id = std::nullopt,
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = "assembly_workspace",
      .source_coordination = request.publish_canonical() ? "seal_and_publish_canonical" : "seal_only",
      .collective_policy = v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers = {"seal_cut"},
      .version_set_id = std::nullopt,
      .transaction_id = std::nullopt,
      .source_selection_digest = std::nullopt,
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "assembly_seal",
      .export_lifetime_kind = "request_scoped",
      .release_strictness = "strict",
      .mutability_contract = "sealed_artifact_immutable",
      .release_policy = release_policy,
      .staged_value_count = 0,
      .acquire_claim_count = 0,
      .publish_barrier = request.publish_canonical(),
  };
  plan.resource_envelope =
      make_resource_envelope("assembly_workspace", "none", "artifact_descriptor", "daemon_request", release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::StartSealAssemblyRequest& request,
    std::string_view operation_id) {
  if (request.assembly_id().empty()) {
    return absl::InvalidArgumentError("assembly_id is required for controller realization plan");
  }
  if (operation_id.empty()) {
    return absl::InvalidArgumentError("assembly operation_id is required for controller realization plan");
  }
  const std::vector<std::string> release_policy{
      "release_operation_lease",
      "attach_layout_to_artifact",
      "apply_post_seal_policy",
  };
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "assembly_seal",
      .resolved_artifact_id = request.assembly_id(),
      .target_layout_digest = assembly_seal_layout_digest(request.assembly_id(), request.layout_id()),
      .device_uuid = "assembly",
      .owner_pid = 0,
      .layout_storage_count = 0,
      .layout_offset_count = 0,
      .member_count = 1,
      .group_id = std::nullopt,
      .part_id = std::nullopt,
      .operation_id = std::string(operation_id),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = "assembly_workspace",
      .source_coordination = request.layout_id().empty() ? "layout_binding_lookup" : "explicit_layout",
      .collective_policy = v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers = {"operation_coordinator", "seal_cut", "post_seal_policy"},
      .version_set_id = std::nullopt,
      .transaction_id = std::nullopt,
      .source_selection_digest = std::nullopt,
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "assembly_seal",
      .export_lifetime_kind = "operation_lease",
      .release_strictness = "strict",
      .mutability_contract = "sealed_artifact_immutable",
      .release_policy = release_policy,
      .staged_value_count = 0,
      .acquire_claim_count = 1,
      .publish_barrier = true,
  };
  plan.resource_envelope = make_resource_envelope(
      "assembly_workspace", "operation_lease", "operation_ref", "daemon_coordinator", release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::PrefetchServingBindingRequest& request) {
  switch (request.target_case()) {
    case v2::PrefetchServingBindingRequest::kServingBindingSetTarget:
      return build_prefetch_target_set_realization_plan(request);
    case v2::PrefetchServingBindingRequest::kServingBindingTarget:
      return build_prefetch_member_realization_plan(request);
    case v2::PrefetchServingBindingRequest::TARGET_NOT_SET:
    default:
      return absl::InvalidArgumentError("prefetch serving binding target is required for controller realization plan");
  }
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::AcquireBindingValueRequest& request) {
  if (!request.has_caller_pid() || request.caller_pid() <= 0) {
    return absl::InvalidArgumentError("caller_pid is required for controller realization plan");
  }
  const bool group_acquire = request.has_group_realization_acquire();
  const std::vector<std::string> release_policy{"release_handle_lease", "release_attachment_ref"};
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "runtime_attachment",
      .resolved_artifact_id = acquire_target_subject(request),
      .target_layout_digest = request.expected_target_layout_hash(),
      .device_uuid = !request.expected_device_uuid().empty() ? request.expected_device_uuid()
                                                             : request.reservation_capability().device_uuid(),
      .owner_pid = request.caller_pid(),
      .layout_storage_count = 0,
      .layout_offset_count = 0,
      .member_count = acquire_member_count(request),
      .group_id = acquire_group_id(request),
      .part_id = acquire_part_id(request),
      .operation_id = std::nullopt,
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = group_acquire ? "per_part_selection" : "retained_selection",
      .source_coordination = group_acquire ? "group_realization_acquire" : "retained_binding_acquire",
      .collective_policy = v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers =
          group_acquire ? acquire_group_barriers_for(request.group_realization_acquire()) : std::vector<std::string>{},
      .version_set_id = group_acquire && !request.group_realization_acquire().version_set_id().empty()
          ? std::optional<std::string>(request.group_realization_acquire().version_set_id())
          : std::nullopt,
      .transaction_id = group_acquire && !request.group_realization_acquire().transaction_id().empty()
          ? std::optional<std::string>(request.group_realization_acquire().transaction_id())
          : std::nullopt,
      .source_selection_digest = !request.reservation_capability().scope_digest().empty()
          ? std::optional<std::string>(request.reservation_capability().scope_digest())
          : std::nullopt,
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "retained_acquire",
      .export_lifetime_kind = "runtime_attachment",
      .release_strictness = "strict",
      .mutability_contract = "runtime_adapter_owned",
      .release_policy = release_policy,
      .staged_value_count = group_acquire ? 1U : 0U,
      .acquire_claim_count = 1,
      .publish_barrier = group_acquire && request.group_realization_acquire().wait_for_publish(),
  };
  plan.resource_envelope = make_resource_envelope(
      "daemon_retained_binding", "cuda_ipc_lease", "runtime_attachment", "caller_pid", release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

absl::StatusOr<ControllerRealizationPlan> build_controller_realization_plan(
    const v2::PublishTargetReplicaRequest& request,
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope,
    const tensorcast::common::v1::ByteSpaceRef& normalized_byte_space) {
  if (request.binding_current_value_publication_token().empty()) {
    return absl::InvalidArgumentError(
        "binding_current_value_publication_token is required for controller realization plan");
  }
  if (scope.publication_id().empty()) {
    return absl::InvalidArgumentError("publication_id is required for controller realization plan");
  }
  if (scope.selection().artifact_id().empty()) {
    return absl::InvalidArgumentError("publication selection artifact_id is required for controller realization plan");
  }
  if (scope.device_uuid().empty()) {
    return absl::InvalidArgumentError("publication device_uuid is required for controller realization plan");
  }
  const int32_t owner_pid =
      request.has_owner_pid() && request.owner_pid() > 0 ? request.owner_pid() : scope.owner_pid();
  if (owner_pid <= 0) {
    return absl::InvalidArgumentError("publication owner_pid is required for controller realization plan");
  }

  const v2::GroupRealizationOptions* group_realization =
      request.has_group_realization() && request.group_realization().enabled() ? &request.group_realization() : nullptr;
  const std::vector<std::string> release_policy{
      "retire_published_replica",
      "release_publication_lease",
      "release_lifecycle_use_guard",
  };
  ControllerRealizationPlan plan;
  plan.target = ControllerRealizationTargetPlan{
      .target_kind = "publication",
      .resolved_artifact_id = scope.selection().artifact_id(),
      .target_layout_digest = publication_target_layout_digest_for(scope, normalized_byte_space),
      .device_uuid = scope.device_uuid(),
      .owner_pid = owner_pid,
      .layout_storage_count = 0,
      .layout_offset_count = 0,
      .member_count = group_realization != nullptr ? group_realization->group().total_parts() : 1,
      .group_id = group_realization != nullptr && !group_realization->group().group_id().empty()
          ? std::optional<std::string>(group_realization->group().group_id())
          : std::nullopt,
      .part_id = group_realization != nullptr && !group_realization->group().part_id().empty()
          ? std::optional<std::string>(group_realization->group().part_id())
          : std::nullopt,
      .operation_id = publication_operation_id_for(request, scope),
  };
  plan.strategy = ControllerRealizationStrategyPlan{
      .source_selection_mode = group_realization != nullptr ? "same_selection" : "single_selection",
      .source_coordination = group_realization != nullptr ? "publication_group_realization" : "publication_lifecycle",
      .collective_policy = v2::CollectivePolicy::COLLECTIVE_POLICY_DISABLE_COLLECTIVE,
      .group_barriers = group_barriers_for(group_realization),
      .version_set_id = std::nullopt,
      .transaction_id = std::nullopt,
      .source_selection_digest = selection_digest_for(group_realization, nullptr, scope.selection()),
  };
  plan.lifecycle = ControllerRealizationLifecyclePlan{
      .capability = "publication",
      .export_lifetime_kind = "publication_lease",
      .release_strictness = "strict",
      .mutability_contract = "published_read_only",
      .release_policy = release_policy,
      .staged_value_count = group_realization != nullptr && group_realization->require_staged_publish() ? 1U : 0U,
      .acquire_claim_count = 1,
      .publish_barrier = true,
  };
  plan.resource_envelope = make_resource_envelope(
      "daemon_published_replica", "publication_lease", "published_replica", "runtime_publication", release_policy);
  return finalize_controller_realization_plan(std::move(plan));
}

OperationTransportContext resolve_operation_transport_context(std::string_view operation_id) {
  OperationTransportContext context;
  if (!operation_id.empty()) {
    context.transport_request_id = std::string(operation_id);
  }
  return context;
}

absl::StatusOr<OperationTransportContext> resolve_group_realization_transport_context(
    std::string_view operation_id,
    const v2::GroupRealizationOptions* group_realization) {
  OperationTransportContext context = resolve_operation_transport_context(operation_id);
  if (group_realization == nullptr || !group_realization->enabled()) {
    return context;
  }

  auto validation_status = validate_group_realization_options(*group_realization);
  if (!validation_status.ok()) {
    return validation_status;
  }
  context.transport_scheduling_group = build_group_realization_transport_group(group_realization->group());
  return context;
}

absl::Status validate_group_realization_staged_publish_supported(
    const v2::GroupRealizationOptions* group_realization,
    bool staged_publish_supported) {
  if (group_realization == nullptr || !group_realization->enabled()) {
    return absl::OkStatus();
  }
  if (!group_realization->require_staged_publish() || staged_publish_supported) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError("group_realization.require_staged_publish requires a staged resource path");
}

absl::StatusOr<std::optional<GroupRealizationBeginContext>> begin_or_join_group_realization_if_enabled(
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    const v2::GroupRealizationOptions* group_realization,
    std::string_view daemon_id,
    std::string_view daemon_session_id,
    std::string_view worker_id,
    const store::components::RpcOptions& rpc_options) {
  if (group_realization == nullptr || !group_realization->enabled()) {
    return std::nullopt;
  }
  if (auto validation_status = validate_group_realization_options(*group_realization); !validation_status.ok()) {
    return validation_status;
  }
  if (global_store_client == nullptr || !global_store_client->is_connected()) {
    return absl::UnavailableError("GlobalStoreClient is required for group_realization");
  }
  if (daemon_id.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for group_realization");
  }
  if (daemon_session_id.empty()) {
    return absl::FailedPreconditionError("daemon_session_id is required for group_realization");
  }

  global_store::BeginOrJoinGroupRealizationRequest request;
  if (auto status = copy_version_reference(group_realization->version(), request.mutable_version()); !status.ok()) {
    return status;
  }
  copy_group_realization_context(group_realization->group(), request.mutable_context());
  request.set_deadline_unix_nanos(group_realization->deadline_unix_nanos());
  request.set_daemon_id(std::string(daemon_id));
  request.set_daemon_session_id(std::string(daemon_session_id));
  request.set_worker_id(std::string(worker_id));

  auto response_or = global_store_client->begin_or_join_group_realization(request, rpc_options);
  if (!response_or.ok()) {
    return response_or.status();
  }
  if (auto status = validate_begin_or_join_response(*response_or, group_realization->group()); !status.ok()) {
    return status;
  }

  GroupRealizationBeginContext begin_context;
  begin_context.transaction_id = response_or->transaction_id();
  begin_context.version_set = response_or->version_set();
  begin_context.realization_kind = response_or->realization_kind();
  begin_context.part_id = response_or->part().part_id();
  begin_context.part_selection = response_or->part().selection();
  begin_context.requested_byte_space = response_or->part().requested_byte_space();
  begin_context.selection_hash = response_or->part().selection_hash();
  begin_context.state = response_or->state();
  begin_context.key_generation = response_or->key_generation();
  return std::optional<GroupRealizationBeginContext>(std::move(begin_context));
}

void apply_group_realization_begin_context_to_transport_context(
    const GroupRealizationBeginContext& begin_context,
    OperationTransportContext* transport_context) {
  if (transport_context == nullptr || !transport_context->transport_scheduling_group.has_value()) {
    return;
  }
  auto& group = *transport_context->transport_scheduling_group;
  group.group_kind = std::string(kGroupRealizationTransportKind);
  group.group_id = absl::StrCat(
      "txn:",
      begin_context.transaction_id,
      "|vs:",
      begin_context.version_set.version_set_id(),
      "|artifact:",
      begin_context.part_selection.artifact_id(),
      "|part:",
      begin_context.part_id,
      "|byte_space:",
      byte_space_identity(begin_context.requested_byte_space),
      "|view:",
      selection_view_identity(begin_context.part_selection),
      "|selection_hash:",
      absl::BytesToHexString(begin_context.selection_hash));
  group.part_id = begin_context.part_id;
  transport_context->transport_request_id =
      derive_group_realization_child_transport_request_id(begin_context, transport_context->transport_request_id);
}

absl::StatusOr<std::optional<global_store::ReportGroupRealizationPreparedResponse>>
report_group_realization_prepared_if_enabled(
    const std::shared_ptr<store::components::IGlobalStoreClient>& global_store_client,
    const v2::GroupRealizationOptions* group_realization,
    const GroupRealizationBeginContext* begin_context,
    const GroupRealizationPreparedMemberContext& prepared_member,
    std::string_view daemon_id,
    std::string_view daemon_session_id,
    std::string_view worker_id,
    const store::components::RpcOptions& rpc_options) {
  if (group_realization == nullptr || !group_realization->enabled()) {
    return std::nullopt;
  }
  if (begin_context == nullptr || begin_context->transaction_id.empty()) {
    return absl::FailedPreconditionError("group_realization begin context is required before prepared report");
  }
  if (prepared_member.binding_id.empty() || prepared_member.binding_value_id.empty() ||
      prepared_member.staging_token.empty()) {
    return absl::InvalidArgumentError("group_realization prepared report requires staged binding value identity");
  }
  if (prepared_member.staging_epoch == 0) {
    return absl::InvalidArgumentError("group_realization prepared report requires staging_epoch");
  }
  if (begin_context->part_id.empty()) {
    return absl::FailedPreconditionError("group_realization begin context part_id is required");
  }
  if (global_store_client == nullptr || !global_store_client->is_connected()) {
    return absl::UnavailableError("GlobalStoreClient is required for group_realization prepared report");
  }
  if (daemon_id.empty()) {
    return absl::FailedPreconditionError("daemon_id is required for group_realization prepared report");
  }
  if (daemon_session_id.empty()) {
    return absl::FailedPreconditionError("daemon_session_id is required for group_realization prepared report");
  }

  global_store::ReportGroupRealizationPreparedRequest request;
  request.set_transaction_id(begin_context->transaction_id);
  request.set_part_id(begin_context->part_id);
  auto* staged_value = request.mutable_staged_value();
  staged_value->set_daemon_id(std::string(daemon_id));
  staged_value->set_daemon_session_id(std::string(daemon_session_id));
  staged_value->set_binding_id(prepared_member.binding_id);
  staged_value->set_binding_value_id(prepared_member.binding_value_id);
  staged_value->set_staging_token(prepared_member.staging_token);
  staged_value->set_staging_epoch(prepared_member.staging_epoch);
  request.set_expected_previous_seal_generation(prepared_member.expected_previous_seal_generation);
  if (!prepared_member.prepared_value_hash.empty()) {
    request.set_prepared_value_hash(prepared_member.prepared_value_hash);
  }
  request.set_daemon_id(std::string(daemon_id));
  request.set_daemon_session_id(std::string(daemon_session_id));
  request.set_worker_id(std::string(worker_id));
  request.set_materialization_attempt_id(prepared_member.materialization_attempt_id);
  request.set_source_replica_id(prepared_member.source_replica_id);
  request.set_source_export_generation(prepared_member.source_export_generation);
  request.set_child_transport_request_id(prepared_member.child_transport_request_id);

  auto response_or = global_store_client->report_group_realization_prepared(request, rpc_options);
  if (!response_or.ok()) {
    return response_or.status();
  }
  if (response_or->status() != global_store::STATUS_OK) {
    return response_status_to_absl(response_or->status(), "ReportGroupRealizationPrepared");
  }
  return std::optional<global_store::ReportGroupRealizationPreparedResponse>(std::move(*response_or));
}

absl::StatusOr<NormalizedMaterializationRequestContext> resolve_materialization_request_context(
    const v2::SourcePolicy* source_policy,
    ExecutionTopologyContext execution_topology,
    std::optional<std::string> replica_uuid,
    bool verify_checksums,
    int32_t wait_for_shared_disk_ms) {
  auto retrieval_policy_or = resolve_retrieval_policy(source_policy);
  if (!retrieval_policy_or.ok()) {
    return retrieval_policy_or.status();
  }
  return NormalizedMaterializationRequestContext{
      .retrieval_policy = *retrieval_policy_or,
      .execution_topology = std::move(execution_topology),
      .replica_uuid = std::move(replica_uuid),
      .verify_checksums = verify_checksums,
      .wait_for_shared_disk_ms = wait_for_shared_disk_ms,
  };
}

void apply_operation_transport_context(
    const OperationTransportContext& context,
    store::loading::MaterializeHints* hints) {
  if (hints == nullptr) {
    return;
  }
  if (!context.transport_request_id.empty()) {
    hints->transport_request_id = context.transport_request_id;
  }
  if (context.transport_scheduling_group.has_value()) {
    hints->transport_scheduling_group = context.transport_scheduling_group;
  }
}

void apply_request_context_to_hints(
    const NormalizedMaterializationRequestContext& context,
    store::loading::MaterializeHints* hints) {
  if (hints == nullptr) {
    return;
  }
  hints->set_retrieval_policy(context.retrieval_policy);
  hints->set_execution_topology(context.execution_topology);
}

v2::ExecutionDiagnostics build_execution_diagnostics(
    const store::loading::MaterializeIntoTargetResult* result,
    v2::CollectivePolicy collective_policy,
    const ExecutionTopologyContext& execution_topology,
    const HashExecutionDetails& hash_details) {
  v2::ExecutionDiagnostics diagnostics;
  const bool collective_requested =
      collective_policy_requests_collective(collective_policy) && execution_topology.collective_load_group.has_value();
  diagnostics.set_collective_requested(collective_requested);
  diagnostics.set_collective_acknowledged(collective_requested);
  diagnostics.set_collective_policy(collective_policy);
  diagnostics.set_hash_rounds(hash_details.hash_rounds);
  diagnostics.set_hash_location(hash_details.hash_location);
  diagnostics.set_hash_backend(hash_details.hash_backend);
  diagnostics.set_hash_bytes(hash_details.hash_bytes);
  diagnostics.set_hash_wall_time_ms(hash_details.hash_wall_time_ms);
  diagnostics.set_hash_identity_forming(hash_details.hash_identity_forming);
  diagnostics.set_identity_mint_strategy(hash_details.identity_mint_strategy);
  if (result == nullptr) {
    return diagnostics;
  }
  diagnostics.set_collective_used(result->collective_handled);
  diagnostics.set_dominant_executor(result->dominant_executor);
  diagnostics.set_direct_write_supported(result->direct_write_supported);
  diagnostics.set_fallback_bytes(result->fallback_bytes);
  diagnostics.set_residual_bytes(result->residual_bytes);
  diagnostics.set_actual_collective_committed_bytes(result->actual_collective_committed_bytes);
  diagnostics.set_actual_local_typed_bytes(result->actual_local_typed_bytes);
  diagnostics.set_actual_generic_backend_bytes(result->actual_generic_backend_bytes);
  diagnostics.set_collective_unique_source_bytes(result->collective_unique_source_bytes);
  diagnostics.set_collective_peer_transfer_bytes(result->collective_peer_transfer_bytes);
  diagnostics.set_collective_peak_temporary_bytes(result->collective_peak_temporary_bytes);
  diagnostics.set_collective_batch_count(result->collective_batch_count);
  diagnostics.set_collective_dedup_saving_bytes(result->collective_dedup_saving_bytes);
  diagnostics.set_collective_skip_reason(result->collective_skip_reason);
  if (collective_requested && !result->collective_handled) {
    diagnostics.set_collective_failure_class(v2::CollectiveFailureClass::COLLECTIVE_FAILURE_CLASS_NOT_ELIGIBLE);
  }
  return diagnostics;
}

absl::StatusOr<ViewSpec> convert_view_spec(const tensorcast::common::v1::ViewSpec& proto) {
  ViewSpec spec;
  for (const auto& [tensor_name, ops_proto] : proto.tensors()) {
    store::loader::TensorViewOps ops;
    for (const auto& op_proto : ops_proto.ops()) {
      switch (op_proto.kind_case()) {
        case tensorcast::common::v1::Op::kNarrow: {
          const auto& narrow = op_proto.narrow();
          store::loader::NarrowOp op{
              .dim = static_cast<int32_t>(narrow.dim()),
              .start = narrow.start(),
              .length = narrow.length(),
          };
          ops.ops.push_back(ViewOp::Narrow(op));
          break;
        }
        case tensorcast::common::v1::Op::kTranspose: {
          const auto& transpose = op_proto.transpose();
          store::loader::TransposeOp op{
              .dim0 = static_cast<int32_t>(transpose.dim0()),
              .dim1 = static_cast<int32_t>(transpose.dim1()),
          };
          ops.ops.push_back(ViewOp::Transpose(op));
          break;
        }
        case tensorcast::common::v1::Op::KIND_NOT_SET:
          return absl::InvalidArgumentError("view op kind not set");
      }
    }
    spec.tensors.emplace(tensor_name, std::move(ops));
  }
  return spec;
}

absl::StatusOr<std::string> compute_view_id_from_spec(
    const tensorcast::common::v1::ViewSpec& view_spec,
    std::string_view canonical_index_json) {
  auto spec_or = convert_view_spec(view_spec);
  if (!spec_or.ok()) {
    return spec_or.status();
  }
  return store::loader::compute_view_id_from_spec(*spec_or, canonical_index_json);
}

tensorcast::common::v1::ViewSpec build_view_spec_proto(const ViewSpec& spec) {
  tensorcast::common::v1::ViewSpec proto;
  auto* tensors = proto.mutable_tensors();
  for (const auto& [tensor_name, ops] : spec.tensors) {
    auto& container = (*tensors)[tensor_name];
    for (const auto& op : ops.ops) {
      auto* op_proto = container.add_ops();
      switch (op.kind) {
        case ViewOp::Kind::kNarrow:
          op_proto->mutable_narrow()->set_dim(static_cast<uint32_t>(op.narrow.dim));
          op_proto->mutable_narrow()->set_start(op.narrow.start);
          op_proto->mutable_narrow()->set_length(op.narrow.length);
          break;
        case ViewOp::Kind::kTranspose:
          op_proto->mutable_transpose()->set_dim0(static_cast<uint32_t>(op.transpose.dim0));
          op_proto->mutable_transpose()->set_dim1(static_cast<uint32_t>(op.transpose.dim1));
          break;
      }
    }
  }
  return proto;
}

bool spec_includes_transpose(const ViewSpec& spec) {
  for (const auto& [_, ops] : spec.tensors) {
    for (const auto& op : ops.ops) {
      if (op.kind == ViewOp::Kind::kTranspose) {
        return true;
      }
    }
  }
  return false;
}

store::loading::TransformPlacement resolve_transform_placement(
    v2::TransformPlacement requested,
    const std::optional<ViewSpec>& spec) {
  switch (requested) {
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_SERVER:
      return store::loading::TransformPlacement::kServer;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_CLIENT:
      return store::loading::TransformPlacement::kClient;
    case v2::TransformPlacement::TRANSFORM_PLACEMENT_UNSPECIFIED:
    default:
      break;
  }
  if (spec.has_value() && spec_includes_transpose(*spec)) {
    return store::loading::TransformPlacement::kClient;
  }
  return store::loading::TransformPlacement::kServer;
}

} // namespace tensorcast::daemon::materialization_policy
