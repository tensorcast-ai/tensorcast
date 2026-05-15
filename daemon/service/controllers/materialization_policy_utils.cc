// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_policy_utils.h"

#include <algorithm>
#include <cstdint>
#include <format>
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

namespace tensorcast::daemon::materialization_policy {

namespace global_store = tensorcast::global_store::v1;

namespace {

using store::loader::ViewOp;

constexpr std::string_view kGroupRealizationTransportKind = "group_realization_transport";
constexpr std::string_view kGroupRealizationChildTransportRequestProfile =
    "tensorcast.group_realization.child_transport_request.v1";

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

} // namespace

absl::StatusOr<RetrievalPolicy> resolve_retrieval_policy_compat(const v2::SourcePolicy* policy) {
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
  auto retrieval_policy_or = resolve_retrieval_policy_compat(source_policy);
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
