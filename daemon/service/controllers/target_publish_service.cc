// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/target_publish_service.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "daemon/state/routed_authority_protocol.h"
#include "daemon/state/routed_authority_wire.h"
#include "daemon/util/status_utils.h"
#include "google/protobuf/any.pb.h"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
namespace global_store = tensorcast::global_store::v1;
namespace operation = tensorcast::operation::v1;
using status_utils::to_grpc_status;

namespace {

constexpr std::string_view kTargetPublicationOwnerPidObservationKind = "target_publication_owner_pid_assertion";
constexpr std::string_view kPublishWorkflowPathFamily = "gate_continue_then_adopt";
constexpr std::string_view kPublishWorkflowGateStageRef = "workflow_gate";
constexpr std::string_view kPublishIssuerValidateStageRef = "issuer_validate";
constexpr std::string_view kPublishWorkflowToIssuerEdgeRef = "workflow_to_issuer";
constexpr std::string_view kPublishWorkflowGateClaimKind = "publish_workflow_gate";
constexpr std::string_view kPublishWorkflowGateClaimPayload = "admit";
constexpr std::string_view kPublishOperationKind = TargetPublishService::public_operation_kind();
constexpr std::string_view kPublishAuthorityScopeKind = "workflow_owner";
constexpr std::string_view kPublishAttachmentKind = "target_publication";
constexpr std::string_view kPublishRecoveryClass = "ephemeral_process_local";
constexpr std::string_view kProgressiveBytePrefixCoverageOrder{"tensorcast.progressive.byte_prefix.v1"};
constexpr uint64_t kTargetPublicationExportGeneration{1};

absl::Status await_state_sync_barrier(const TargetPublishService::Dep& dep) {
  if (!dep.await_state_sync_barrier) {
    return absl::OkStatus();
  }
  return dep.await_state_sync_barrier();
}

std::optional<std::string> mi2_index_multihash(std::string_view artifact_id) {
  constexpr std::string_view kPrefix{"mi2:"};
  if (!artifact_id.starts_with(kPrefix)) {
    return std::nullopt;
  }
  const std::string_view suffix = artifact_id.substr(kPrefix.size());
  const size_t separator = suffix.find(':');
  if (separator == std::string_view::npos || separator == 0) {
    return std::nullopt;
  }
  return std::string(suffix.substr(0, separator));
}

std::string digest_to_string(std::string_view payload) {
  const std::vector<uint8_t> digest = common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  return std::string(reinterpret_cast<const char*>(digest.data()), digest.size());
}

std::string byte_prefix_coverage_order_hash() {
  return digest_to_string(kProgressiveBytePrefixCoverageOrder);
}

std::optional<global_store::ProgressiveCoverageIdentity> build_progressive_publication_identity(
    const TargetPublicationRegistry::Record& record) {
  if (record.byte_space.kind() != tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL ||
      !record.byte_space.id().empty()) {
    return std::nullopt;
  }
  if (!record.selection.view_id().empty() || record.selection.tensor_names_size() > 0 ||
      !record.selection.view_subset_hash().empty()) {
    return std::nullopt;
  }
  auto index_multihash = mi2_index_multihash(record.selection.artifact_id());
  if (!index_multihash.has_value() || record.selection.logical_layout_hash().empty() ||
      record.selection.selection_hash().empty()) {
    return std::nullopt;
  }

  global_store::ProgressiveCoverageIdentity identity;
  identity.set_artifact_id(record.selection.artifact_id());
  identity.mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  identity.set_selection_hash(record.selection.selection_hash());
  identity.set_logical_layout_hash(record.selection.logical_layout_hash());
  identity.mutable_hash_space()->mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  identity.mutable_hash_space()->set_canonical_index_multihash(*index_multihash);
  identity.set_coverage_order_hash(byte_prefix_coverage_order_hash());
  const bool has_group_version_set = !record.group_version_set_id.empty();
  const bool has_group_part = !record.group_part_id.empty();
  if (has_group_version_set != has_group_part) {
    return std::nullopt;
  }
  if (has_group_version_set) {
    identity.set_group_version_set_id(record.group_version_set_id);
    identity.set_group_part_id(record.group_part_id);
  }
  return identity;
}

void maybe_report_progressive_published_target_coverage(
    const TargetPublishService::Dep& dep,
    const TargetPublicationRegistry::Record& record,
    std::string_view replica_id,
    uint64_t total_size) {
  if (!dep.progressive_replication.enabled || dep.global_store_client == nullptr ||
      !dep.global_store_client->is_connected()) {
    return;
  }
  if (dep.shutdown_signal.is_shutting_down() || replica_id.empty() || total_size == 0) {
    return;
  }
  auto identity = build_progressive_publication_identity(record);
  if (!identity.has_value()) {
    return;
  }

  std::string daemon_id = dep.identity.daemon_id();
  if (daemon_id.empty()) {
    daemon_id = dep.daemon_id;
  }
  const std::string worker_id = dep.identity.worker_id();
  if (daemon_id.empty() || worker_id.empty()) {
    VLOG(1) << "Skipping progressive target publication coverage because daemon or worker identity is unavailable";
    return;
  }
  const std::string node_id = dep.identity.node_id();
  const std::string source_domain = !node_id.empty() ? node_id : daemon_id;
  const std::string materialization_attempt_id = absl::StrCat("target_publish:", record.publication_id.value);

  global_store::ReportProgressiveCoverageRequest request;
  request.set_coverage_id(
      absl::StrCat(
          "progressive:", materialization_attempt_id, ":", replica_id, ":", kTargetPublicationExportGeneration));
  request.mutable_identity()->CopyFrom(*identity);
  request.set_replica_id(std::string(replica_id));
  request.set_daemon_id(daemon_id);
  request.set_daemon_session_id(dep.daemon_session_id);
  request.set_worker_id(worker_id);
  request.set_source_export_generation(kTargetPublicationExportGeneration);
  request.set_coverage_epoch(1);
  request.set_coverage_kind(global_store::PROGRESSIVE_COVERAGE_KIND_BYTE_PREFIX);
  request.set_state(global_store::PROGRESSIVE_COVERAGE_STATE_VERIFIED);
  request.set_export_state(global_store::PROGRESSIVE_EXPORT_STATE_COMPLETE_EXPORTABLE);
  request.set_verified_units(total_size);
  request.set_verified_bytes(total_size);
  request.set_completed_units(total_size);
  request.set_completed_bytes(total_size);
  request.set_total_units(total_size);
  request.set_total_bytes(total_size);
  request.set_materialization_attempt_id(materialization_attempt_id);
  request.set_source_transport_id(record.publication_id.value);
  request.set_source_domain(source_domain);
  request.set_seed_transport_kind("local_replica");

  auto response_or = dep.global_store_client->report_progressive_coverage(request);
  if (!response_or.ok()) {
    LOG(WARNING) << "ReportProgressiveCoverage failed for target publication artifact_id="
                 << record.selection.artifact_id() << " replica_id=" << replica_id << ": " << response_or.status();
    return;
  }
  if (response_or->status() != global_store::STATUS_OK) {
    LOG(WARNING) << "ReportProgressiveCoverage returned status=" << static_cast<int>(response_or->status())
                 << " reason=" << response_or->reason()
                 << " for target publication artifact_id=" << record.selection.artifact_id()
                 << " replica_id=" << replica_id;
  }
}

class OperationLeaseGuard {
 public:
  OperationLeaseGuard(
      std::shared_ptr<store::components::IGlobalStoreClient> client,
      std::string lease_token,
      std::string operation_id)
      : client_(std::move(client)), lease_token_(std::move(lease_token)), operation_id_(std::move(operation_id)) {}

  ~OperationLeaseGuard() {
    release();
  }

  void release() {
    if (released_ || client_ == nullptr || lease_token_.empty()) {
      released_ = true;
      return;
    }
    operation::ReleaseOperationLeaseRequest request;
    request.set_lease_token(lease_token_);
    auto release_or = client_->release_operation_lease(request);
    if (!release_or.ok()) {
      LOG(WARNING) << "release_operation_lease failed for op=" << operation_id_ << ": " << release_or.status();
    }
    released_ = true;
  }

 private:
  std::shared_ptr<store::components::IGlobalStoreClient> client_;
  std::string lease_token_;
  std::string operation_id_;
  bool released_{false};
};

struct TargetPublicationCredentialInspection {
  tensorcast::common::v1::CapabilityTokenEnvelope envelope;
  tensorcast::common::v1::BindingCurrentValuePublicationScope scope;
  tensorcast::common::v1::ByteSpaceRef normalized_byte_space;
  FrontDoorCredentialContext front_door_context;
};

enum class PublicationSubjectMatchResult : std::uint8_t {
  kCurrentInstance = 0,
  kStaleCurrent = 1,
  kOwnerLost = 2,
  kUnavailable = 3,
};

struct PublicationSubjectEvaluation {
  PublicationSubjectMatchResult match_result{PublicationSubjectMatchResult::kUnavailable};
  std::optional<TargetPublicationRegistry::Record> current_record;
};

WorkflowBindingProjection publication_workflow_binding_projection(const TargetPublicationRegistry::Record& record);

absl::StatusOr<tensorcast::common::v1::ByteSpaceRef> normalize_byte_space(
    const tensorcast::common::v1::ByteSpaceRef& space) {
  tensorcast::common::v1::ByteSpaceRef out;
  switch (space.kind()) {
    case tensorcast::common::v1::BYTE_SPACE_KIND_UNSPECIFIED:
    case tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL:
      out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
      out.set_id("");
      return out;
    case tensorcast::common::v1::BYTE_SPACE_KIND_VIEW:
      if (space.id().empty()) {
        return absl::InvalidArgumentError("byte_space VIEW requires id");
      }
      out.set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
      out.set_id(space.id());
      return out;
    default:
      return absl::InvalidArgumentError("unsupported byte_space kind");
  }
}

std::string publication_capability_id(std::string_view publication_id) {
  return absl::StrCat("publication:", publication_id);
}

std::string publication_subject_id(const TargetPublicationRegistry::Record& record) {
  return absl::StrCat("publication-target:", record.publication_subject_key.value);
}

LocalObservationSet build_target_publication_local_observations(const v2::PublishTargetReplicaRequest& req) {
  LocalObservationSet local_observations;
  if (req.has_owner_pid()) {
    local_observations.observations.push_back(
        LocalObservation{
            .observation_kind = std::string(kTargetPublicationOwnerPidObservationKind),
            .observation_payload = std::to_string(req.owner_pid()),
        });
  }
  return local_observations;
}

absl::Status validate_target_publication_local_observations(
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope,
    const LocalObservationSet& local_observations) {
  for (const auto& observation : local_observations.observations) {
    if (observation.observation_kind == kTargetPublicationOwnerPidObservationKind) {
      if (std::to_string(scope.owner_pid()) != observation.observation_payload) {
        return absl::PermissionDeniedError("owner_pid mismatch for binding_current_value_publication_token");
      }
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrCat("unsupported target_publication local observation: ", observation.observation_kind));
  }
  return absl::OkStatus();
}

FrontDoorCredentialContext build_target_publication_front_door_context(
    const tensorcast::common::v1::CapabilityTokenEnvelope& envelope,
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope,
    std::string_view binding_current_value_publication_token,
    CredentialCarriageKind carriage_kind,
    LocalObservationSet local_observations) {
  std::optional<ForwardableCredentialEvidence> forwardable_evidence;
  if (carriage_kind == CredentialCarriageKind::kSelfDescribing) {
    forwardable_evidence = ForwardableCredentialEvidence{
        .evidence_kind = CredentialEvidenceKind::kRawCredential,
        .raw_credential_bytes = std::string(binding_current_value_publication_token),
    };
  }
  return FrontDoorCredentialContext{
      .parsed_credential =
          ParsedCredential{
              .address =
                  CapabilityBindingAddress{
                      .route_principal = make_issuer_route_principal(envelope.issuer_daemon_id()),
                      .family = LifecycleCapabilityFamily::kPublish,
                      .binding_space = LifecycleBindingSpace::kPublication,
                      .binding_key_kind = BindingKeyKind::kPublicationId,
                      .binding_key = scope.publication_id(),
                      .epochs = LifecycleEpochs{},
                  },
              .front_door_kind = LifecycleFrontDoorKind::kTargetPublicationToken,
              .credential_expires_at = absl::FromUnixMillis(static_cast<std::int64_t>(envelope.expires_at_ms())),
              .carriage_kind = carriage_kind,
              .binding_mode = LifecycleBindingMode::kAddressDerived,
              .constraint_claims =
                  ConstraintClaims{
                      .artifact_id = scope.selection().artifact_id(),
                      .operation_id = scope.operation_id(),
                  },
          },
      .forwardable_evidence = std::move(forwardable_evidence),
      .local_observations = std::move(local_observations),
  };
}

void bind_record_to_target_publication_front_door_context(
    const TargetPublicationRegistry::Record& record,
    FrontDoorCredentialContext* front_door_context) {
  if (front_door_context == nullptr) {
    return;
  }
  front_door_context->parsed_credential.address.binding_key = record.publication_id.value;
  front_door_context->parsed_credential.address.epochs.subject_generation =
      record.subject_generation == 0 ? 1 : record.subject_generation;
  front_door_context->parsed_credential.constraint_claims.operation_id = record.request_operation_id;
}

PortableParsedCredential derive_target_publication_portable_credential(
    const FrontDoorCredentialContext& front_door_context) {
  return PortableParsedCredential{
      .address = front_door_context.parsed_credential.address,
      .front_door_kind = front_door_context.parsed_credential.front_door_kind,
      .credential_expires_at = front_door_context.parsed_credential.credential_expires_at,
      .binding_mode = front_door_context.parsed_credential.binding_mode,
      .portable_constraint_claims = front_door_context.parsed_credential.constraint_claims,
  };
}

AuthorityRef publication_routed_workflow_authority_ref(std::string_view issuer_daemon_id) {
  return AuthorityRef{
      .authority_kind = AuthorityKind::kWorkflowOwner,
      .authority_id = std::string(issuer_daemon_id),
  };
}

AuthorityRef publication_routed_issuer_authority_ref(std::string_view issuer_daemon_id) {
  return AuthorityRef{
      .authority_kind = AuthorityKind::kIssuerDaemon,
      .authority_id = std::string(issuer_daemon_id),
  };
}

std::string target_publication_root_request_id(std::string_view publication_id) {
  return absl::StrCat("target-publication:", publication_id);
}

std::string publication_request_idempotency_key(const TargetPublicationRegistry::Record& record) {
  if (record.request_operation_id.empty()) {
    return {};
  }
  return absl::StrCat(
      record.publication_subject_key.value, "|g:", record.subject_generation, "|req:", record.request_operation_id);
}

std::string owner_id_for_operation(const WorkerIdentityStore& identity) {
  auto daemon_id = identity.daemon_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  daemon_id = identity.worker_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  return "unknown-daemon";
}

std::string publish_operation_id(const TargetPublicationRegistry::Record& record) {
  return absl::StrCat("publish-target:", record.publication_id.value);
}

std::optional<std::string> fencing_digest_for_workflow_ref(const WorkflowCompanionRef& workflow_ref) {
  if (!workflow_ref.fencing_context.has_value()) {
    return std::nullopt;
  }
  const auto& fencing = *workflow_ref.fencing_context;
  return absl::StrCat(static_cast<int>(fencing.principal_kind), ":", fencing.principal_id, ":", fencing.epoch);
}

operation::OperationRef build_publish_operation_ref(const TargetPublicationRegistry::Record& record) {
  operation::OperationRef operation_ref;
  operation_ref.set_operation_id(publish_operation_id(record));
  operation_ref.set_kind(std::string(kPublishOperationKind));
  operation_ref.set_target_artifact_id(record.selection.artifact_id());
  operation_ref.set_authority_scope_kind(std::string(kPublishAuthorityScopeKind));
  operation_ref.set_authority_scope_id(record.publication_id.value);
  operation_ref.set_attachment_kind(std::string(kPublishAttachmentKind));
  operation_ref.set_recovery_class(std::string(kPublishRecoveryClass));
  const auto workflow_ref = record.workflow_binding_projection.value_or(publication_workflow_binding_projection(record))
                                .resolved_workflow_ref;
  if (const auto fencing_digest = fencing_digest_for_workflow_ref(workflow_ref); fencing_digest.has_value()) {
    operation_ref.set_fencing_digest(*fencing_digest);
  }
  return operation_ref;
}

google::protobuf::Any build_publish_operation_snapshot(const TargetPublicationRegistry::Record& record) {
  google::protobuf::Any snapshot;
  operation::OperationContinuationMetadata metadata;
  metadata.mutable_ref()->CopyFrom(build_publish_operation_ref(record));
  snapshot.PackFrom(metadata);
  return snapshot;
}

operation::GetOperationRequest build_get_operation_request(const operation::OperationRef& operation_ref) {
  operation::GetOperationRequest request;
  request.set_operation_id(operation_ref.operation_id());
  request.mutable_ref()->CopyFrom(operation_ref);
  return request;
}

void merge_operation_ref_metadata(const operation::OperationRef& source, operation::OperationRef* target) {
  if (target == nullptr) {
    return;
  }
  if (target->operation_id().empty() && !source.operation_id().empty()) {
    target->set_operation_id(source.operation_id());
  }
  if (target->kind().empty() && !source.kind().empty()) {
    target->set_kind(source.kind());
  }
  if (target->target_artifact_id().empty() && !source.target_artifact_id().empty()) {
    target->set_target_artifact_id(source.target_artifact_id());
  }
  if (target->authority_scope_kind().empty() && !source.authority_scope_kind().empty()) {
    target->set_authority_scope_kind(source.authority_scope_kind());
  }
  if (target->authority_scope_id().empty() && !source.authority_scope_id().empty()) {
    target->set_authority_scope_id(source.authority_scope_id());
  }
  if (target->attachment_kind().empty() && !source.attachment_kind().empty()) {
    target->set_attachment_kind(source.attachment_kind());
  }
  if (target->recovery_class().empty() && !source.recovery_class().empty()) {
    target->set_recovery_class(source.recovery_class());
  }
  if (target->fencing_digest().empty() && !source.fencing_digest().empty()) {
    target->set_fencing_digest(source.fencing_digest());
  }
}

std::optional<absl::StatusCode> absl_status_code_from_string(std::string_view status_code) {
  if (status_code == "cancelled" || status_code == "CANCELLED") {
    return absl::StatusCode::kCancelled;
  }
  if (status_code == "invalid_argument" || status_code == "INVALID_ARGUMENT") {
    return absl::StatusCode::kInvalidArgument;
  }
  if (status_code == "deadline_exceeded" || status_code == "DEADLINE_EXCEEDED") {
    return absl::StatusCode::kDeadlineExceeded;
  }
  if (status_code == "not_found" || status_code == "NOT_FOUND") {
    return absl::StatusCode::kNotFound;
  }
  if (status_code == "already_exists" || status_code == "ALREADY_EXISTS") {
    return absl::StatusCode::kAlreadyExists;
  }
  if (status_code == "permission_denied" || status_code == "PERMISSION_DENIED") {
    return absl::StatusCode::kPermissionDenied;
  }
  if (status_code == "resource_exhausted" || status_code == "RESOURCE_EXHAUSTED") {
    return absl::StatusCode::kResourceExhausted;
  }
  if (status_code == "failed_precondition" || status_code == "FAILED_PRECONDITION") {
    return absl::StatusCode::kFailedPrecondition;
  }
  if (status_code == "aborted" || status_code == "ABORTED") {
    return absl::StatusCode::kAborted;
  }
  if (status_code == "out_of_range" || status_code == "OUT_OF_RANGE") {
    return absl::StatusCode::kOutOfRange;
  }
  if (status_code == "unimplemented" || status_code == "UNIMPLEMENTED") {
    return absl::StatusCode::kUnimplemented;
  }
  if (status_code == "internal" || status_code == "INTERNAL") {
    return absl::StatusCode::kInternal;
  }
  if (status_code == "unavailable" || status_code == "UNAVAILABLE") {
    return absl::StatusCode::kUnavailable;
  }
  if (status_code == "data_loss" || status_code == "DATA_LOSS") {
    return absl::StatusCode::kDataLoss;
  }
  if (status_code == "unauthenticated" || status_code == "UNAUTHENTICATED") {
    return absl::StatusCode::kUnauthenticated;
  }
  return std::nullopt;
}

void populate_publish_operation_error(const absl::Status& status, operation::OperationError* error) {
  if (error == nullptr || status.ok()) {
    return;
  }
  error->set_status_code(std::string(absl::StatusCodeToString(status.code())));
  error->set_message(std::string(status.message()));
  error->set_retryable(
      status.code() == absl::StatusCode::kUnavailable || status.code() == absl::StatusCode::kDeadlineExceeded);
}

operation::OperationStatus build_publish_operation_status(
    operation::OperationState state,
    std::string_view message,
    const absl::Status& status = absl::OkStatus(),
    const google::protobuf::Any* result = nullptr) {
  operation::OperationStatus operation_status;
  operation_status.set_state(state);
  operation_status.set_message(std::string(message));
  if (!status.ok()) {
    populate_publish_operation_error(status, operation_status.mutable_error());
  }
  if (result != nullptr) {
    operation_status.mutable_result()->CopyFrom(*result);
  }
  return operation_status;
}

absl::Status status_from_operation_status(const operation::OperationStatus& status) {
  if (status.state() == operation::OPERATION_STATE_SUCCESS) {
    return absl::OkStatus();
  }
  if (status.has_error()) {
    const auto code_or = absl_status_code_from_string(status.error().status_code());
    return absl::Status(code_or.value_or(absl::StatusCode::kUnknown), status.error().message());
  }
  if (status.state() == operation::OPERATION_STATE_CANCELLED) {
    return absl::CancelledError(status.message());
  }
  if (status.state() == operation::OPERATION_STATE_FAILED) {
    return absl::UnknownError(status.message());
  }
  return absl::UnavailableError(status.message());
}

absl::StatusOr<v2::PublishTargetReplicaResponse> publish_result_from_operation(
    const operation::GetOperationResponse& operation_response) {
  if (!operation_response.status().has_result()) {
    return absl::DataLossError("publish_target_replica operation result is missing");
  }
  v2::PublishTargetReplicaResponse result;
  if (!operation_response.status().result().UnpackTo(&result)) {
    return absl::DataLossError("publish_target_replica operation result type is invalid");
  }
  if (result.lease_id().empty()) {
    return absl::DataLossError("publish_target_replica operation returned empty lease_id");
  }
  return result;
}

bool is_publish_workflow_gate_stage(std::string_view path_family, std::string_view stage_ref) {
  return path_family == kPublishWorkflowPathFamily && stage_ref == kPublishWorkflowGateStageRef;
}

bool is_publish_issuer_validate_stage(std::string_view path_family, std::string_view stage_ref) {
  return path_family == kPublishWorkflowPathFamily && stage_ref == kPublishIssuerValidateStageRef;
}

absl::Status undeclared_route_authority_stage_status(std::string_view path_family, std::string_view stage_ref) {
  return absl::FailedPreconditionError(
      absl::StrCat("undeclared routed authority path/stage: ", path_family, "/", stage_ref));
}

absl::StatusOr<std::string_view> require_target_publication_raw_issuer_evidence(
    const ForwardableCredentialEvidence& evidence) {
  if (evidence.evidence_kind != CredentialEvidenceKind::kRawCredential || !evidence.raw_credential_bytes.has_value()) {
    return absl::FailedPreconditionError(
        "target_publication workflow route currently supports only raw_credential issuer evidence; canonical "
        "projection is not supported");
  }
  return *evidence.raw_credential_bytes;
}

DelegationEnvelope make_delegation_envelope(
    const AuthorityRef& audience_authority_ref,
    std::string_view root_request_id,
    std::string_view path_family,
    DelegationPayloadKind payload_kind,
    DelegationClass delegation_class,
    absl::Time expires_at,
    std::optional<std::string_view> edge_ref = std::nullopt) {
  DelegationEnvelope envelope{
      .audience_authority_ref = audience_authority_ref,
      .bound_root_request_id = std::string(root_request_id),
      .bound_path_family = std::string(path_family),
      .payload_kind = payload_kind,
      .delegation_class = delegation_class,
  };
  if (edge_ref.has_value()) {
    envelope.bound_edge = std::string(*edge_ref);
  }
  if (expires_at != absl::InfinitePast() && expires_at != absl::InfiniteFuture()) {
    envelope.expires_at = expires_at;
  }
  return envelope;
}

absl::StatusOr<TargetPublicationCredentialInspection> inspect_target_publication_credential(
    common::CapabilityTokenManager* capability_tokens,
    std::string_view binding_current_value_publication_token,
    absl::Time now,
    std::string_view expected_issuer,
    LocalObservationSet local_observations) {
  if (binding_current_value_publication_token.empty()) {
    return absl::InvalidArgumentError("binding_current_value_publication_token is required");
  }
  if (capability_tokens == nullptr || !capability_tokens->configured()) {
    return absl::FailedPreconditionError("capability tokens not configured");
  }
  auto env_or = capability_tokens->verify(
      binding_current_value_publication_token,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_BINDING_CURRENT_VALUE_PUBLICATION,
      expected_issuer,
      now,
      /*require_not_expired=*/true);
  if (!env_or.ok()) {
    return env_or.status();
  }

  tensorcast::common::v1::BindingCurrentValuePublicationScope scope;
  if (!scope.ParseFromString(env_or->scope())) {
    return absl::InvalidArgumentError("binding_current_value_publication_token scope parse failed");
  }
  auto local_observation_status = validate_target_publication_local_observations(scope, local_observations);
  if (!local_observation_status.ok()) {
    return local_observation_status;
  }
  auto normalized_scope_or = normalize_byte_space(scope.byte_space());
  if (!normalized_scope_or.ok()) {
    return normalized_scope_or.status();
  }
  const auto envelope = *env_or;
  const auto front_door_context = build_target_publication_front_door_context(
      envelope,
      scope,
      binding_current_value_publication_token,
      CredentialCarriageKind::kSelfDescribing,
      std::move(local_observations));

  return TargetPublicationCredentialInspection{
      .envelope = envelope,
      .scope = std::move(scope),
      .normalized_byte_space = std::move(*normalized_scope_or),
      .front_door_context = front_door_context,
  };
}

absl::Status validate_target_publication_request_against_scope(
    const v2::PublishTargetReplicaRequest& req,
    const TargetPublicationCredentialInspection& inspection) {
  auto normalized_req_or = normalize_byte_space(req.byte_space());
  if (!normalized_req_or.ok()) {
    return normalized_req_or.status();
  }
  const auto& normalized_req = *normalized_req_or;
  if (normalized_req.kind() != inspection.normalized_byte_space.kind() ||
      normalized_req.id() != inspection.normalized_byte_space.id()) {
    return absl::InvalidArgumentError("byte_space does not match binding_current_value_publication_token");
  }
  if (!inspection.scope.operation_id().empty()) {
    if (!req.has_operation_id() || req.operation_id().empty()) {
      return absl::InvalidArgumentError("operation_id is required for binding_current_value_publication_token");
    }
    if (inspection.scope.operation_id() != req.operation_id()) {
      return absl::FailedPreconditionError("operation_id mismatch for binding_current_value_publication_token");
    }
  }
  if (inspection.scope.publication_id().empty()) {
    return absl::InvalidArgumentError("binding_current_value_publication_token missing publication_id");
  }
  return absl::OkStatus();
}

absl::Status validate_target_publication_scope_against_record(
    const TargetPublicationRegistry::Record& record,
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope,
    const tensorcast::common::v1::ByteSpaceRef& normalized_byte_space) {
  const PublicationSubjectKey publication_subject_key = build_publication_subject_key(
      scope.selection(), normalized_byte_space, scope.target_layout_hash(), scope.device_uuid());
  if (record.publication_subject_key != publication_subject_key) {
    return absl::FailedPreconditionError("publication target mismatch for binding_current_value_publication_token");
  }
  if (record.device_uuid != scope.device_uuid()) {
    return absl::FailedPreconditionError("device_uuid mismatch for binding_current_value_publication_token");
  }
  if (record.owner_pid != scope.owner_pid()) {
    return absl::FailedPreconditionError("owner_pid mismatch for binding_current_value_publication_token");
  }
  if (record.daemon_id != scope.daemon_id()) {
    return absl::FailedPreconditionError("daemon_id mismatch for binding_current_value_publication_token");
  }
  if (record.daemon_session_id != scope.daemon_session_id()) {
    return absl::FailedPreconditionError("daemon_session_id mismatch for binding_current_value_publication_token");
  }
  if (record.binding_id != scope.binding_id()) {
    return absl::FailedPreconditionError("binding_id mismatch for binding_current_value_publication_token");
  }
  if (record.binding_layout_id != scope.binding_layout_id()) {
    return absl::FailedPreconditionError("binding_layout_id mismatch for binding_current_value_publication_token");
  }
  if (record.binding_value_id != scope.binding_value_id()) {
    return absl::FailedPreconditionError("binding_value_id mismatch for binding_current_value_publication_token");
  }
  if (record.seal_generation != scope.seal_generation()) {
    return absl::FailedPreconditionError("seal_generation mismatch for binding_current_value_publication_token");
  }
  if (record.target_layout_hash != scope.target_layout_hash()) {
    return absl::FailedPreconditionError("target_layout_hash mismatch for binding_current_value_publication_token");
  }
  if (record.byte_space.kind() != normalized_byte_space.kind() ||
      record.byte_space.id() != normalized_byte_space.id()) {
    return absl::FailedPreconditionError("byte_space mismatch for binding_current_value_publication_token");
  }
  if (!scope.operation_id().empty() && record.request_operation_id != scope.operation_id()) {
    return absl::FailedPreconditionError("stored operation_id mismatch for binding_current_value_publication_token");
  }
  if (record.selection.artifact_id() != scope.selection().artifact_id() ||
      record.selection.view_id() != scope.selection().view_id() ||
      record.selection.logical_layout_hash() != scope.selection().logical_layout_hash() ||
      record.selection.selection_hash() != scope.selection().selection_hash() ||
      record.selection.view_subset_hash() != scope.selection().view_subset_hash() ||
      record.selection.tensor_names_size() != scope.selection().tensor_names_size()) {
    return absl::FailedPreconditionError("selection mismatch for binding_current_value_publication_token");
  }
  for (int i = 0; i < record.selection.tensor_names_size(); ++i) {
    if (record.selection.tensor_names(i) != scope.selection().tensor_names(i)) {
      return absl::FailedPreconditionError(
          "selection tensor_names mismatch for binding_current_value_publication_token");
    }
  }
  const bool has_subset_selection =
      !scope.selection().tensor_names().empty() || !scope.selection().view_subset_hash().empty();
  const bool view_scoped_byte_space = normalized_byte_space.kind() == tensorcast::common::v1::BYTE_SPACE_KIND_VIEW &&
      !normalized_byte_space.id().empty();
  if (has_subset_selection && !view_scoped_byte_space) {
    return absl::FailedPreconditionError("selection is not publishable (packed or subset requires view byte-space)");
  }
  if (scope.selection().artifact_id().empty()) {
    return absl::InvalidArgumentError("artifact_id missing from binding_current_value_publication_token");
  }
  if (scope.binding_id().empty() || scope.binding_layout_id().empty() || scope.binding_value_id().empty() ||
      scope.seal_generation() == 0) {
    return absl::InvalidArgumentError("binding identity missing from binding_current_value_publication_token");
  }
  return absl::OkStatus();
}

absl::Status validate_binding_current_value_against_publication(
    BindingRegistry& bindings,
    const TargetPublicationRegistry::Record& record,
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope) {
  auto binding_or = bindings.get(scope.binding_id());
  if (!binding_or.ok()) {
    return binding_or.status();
  }
  const auto binding = *binding_or;
  absl::MutexLock lock(&binding->mu);
  if (binding->closed) {
    return absl::FailedPreconditionError("binding is closed");
  }
  if (binding->retired) {
    return absl::FailedPreconditionError("binding is retired");
  }
  if (binding->state != v2::BINDING_STATE_READY_ARTIFACT) {
    return absl::FailedPreconditionError("binding current value is not artifact-backed");
  }
  if (binding->binding_layout_id != scope.binding_layout_id()) {
    return absl::FailedPreconditionError("binding_layout_id mismatch for binding current value publication");
  }
  if (binding->current_binding_value_id != scope.binding_value_id()) {
    return absl::FailedPreconditionError("binding_value_id mismatch for binding current value publication");
  }
  if (binding->seal_generation != scope.seal_generation()) {
    return absl::FailedPreconditionError("seal_generation mismatch for binding current value publication");
  }
  if (binding->owner_pid != scope.owner_pid()) {
    return absl::FailedPreconditionError("owner_pid mismatch for binding current value publication");
  }
  if (binding->daemon_id != scope.daemon_id()) {
    return absl::FailedPreconditionError("daemon_id mismatch for binding current value publication");
  }
  if (binding->daemon_session_id != scope.daemon_session_id()) {
    return absl::FailedPreconditionError("daemon_session_id mismatch for binding current value publication");
  }
  if (binding->device_uuid != scope.device_uuid()) {
    return absl::FailedPreconditionError("device_uuid mismatch for binding current value publication");
  }
  if (binding->target_layout_hash != scope.target_layout_hash()) {
    return absl::FailedPreconditionError("target_layout_hash mismatch for binding current value publication");
  }
  if (binding->current_artifact_id.empty() || binding->current_artifact_id != scope.selection().artifact_id()) {
    return absl::FailedPreconditionError("artifact_id mismatch for binding current value publication");
  }
  if (binding->current_selection.artifact_id() != scope.selection().artifact_id() ||
      binding->current_selection.view_id() != scope.selection().view_id() ||
      binding->current_selection.logical_layout_hash() != scope.selection().logical_layout_hash() ||
      binding->current_selection.selection_hash() != scope.selection().selection_hash() ||
      binding->current_selection.view_subset_hash() != scope.selection().view_subset_hash() ||
      binding->current_selection.tensor_names_size() != scope.selection().tensor_names_size()) {
    return absl::FailedPreconditionError("selection mismatch for binding current value publication");
  }
  for (int i = 0; i < binding->current_selection.tensor_names_size(); ++i) {
    if (binding->current_selection.tensor_names(i) != scope.selection().tensor_names(i)) {
      return absl::FailedPreconditionError("selection tensor_names mismatch for binding current value publication");
    }
  }
  if (binding->active_published_current.has_value() &&
      binding->active_published_current->binding_id == binding->binding_id &&
      binding->active_published_current->binding_value_id == binding->current_binding_value_id &&
      binding->active_published_current->seal_generation == binding->seal_generation &&
      binding->active_published_current->publication_id != record.publication_id.value) {
    return absl::FailedPreconditionError("binding current value is already published; call retire() first");
  }
  return absl::OkStatus();
}

OwnerStageReply publication_terminal_reply(
    const AuthorityRef& answered_by,
    std::string_view stage_ref,
    std::string_view status_code,
    std::string_view family_payload,
    std::string_view diagnostics) {
  return OwnerStageReply{
      .answered_by = answered_by,
      .path_family = std::string(kPublishWorkflowPathFamily),
      .stage_ref = std::string(stage_ref),
      .reply_kind = OwnerStageReplyKind::kTerminal,
      .terminal_projection =
          TerminalProjection{
              .projection_kind = status_code == "ok" ? TerminalProjectionKind::kSemanticSuccess
                                                     : TerminalProjectionKind::kSemanticReject,
              .status_code = std::string(status_code),
              .family_payload = std::string(family_payload),
          },
      .diagnostics = std::string(diagnostics),
  };
}

WorkflowBindingProjection publication_workflow_binding_projection(const TargetPublicationRegistry::Record& record) {
  return WorkflowBindingProjection{
      .resolved_workflow_ref =
          WorkflowCompanionRef{
              .owner_kind = WorkflowOwnerKind::kPublication,
              .workflow_id = record.publication_id.value,
              .currentness_key = record.publication_subject_key.value,
              .operation_id = record.request_operation_id.empty()
                  ? std::nullopt
                  : std::optional<std::string>(record.request_operation_id),
              .fencing_context = std::nullopt,
          },
  };
}

WorkflowIssueContext publication_workflow_issue_context(const TargetPublicationRegistry::Record& record) {
  return WorkflowIssueContext{
      .family = "publish",
      .requested_workflow_ref = publication_workflow_binding_projection(record).resolved_workflow_ref,
      .currentness_key = record.publication_subject_key.value,
      .request_operation_id =
          record.request_operation_id.empty() ? std::nullopt : std::optional<std::string>(record.request_operation_id),
  };
}

AuthorityRef publication_workflow_authority_ref(const TargetPublicationRegistry::Record& record) {
  const auto binding_projection =
      record.workflow_binding_projection.value_or(publication_workflow_binding_projection(record));
  return AuthorityRef{
      .authority_kind = AuthorityKind::kWorkflowOwner,
      .authority_id = binding_projection.resolved_workflow_ref.workflow_id,
      .fencing_context = binding_projection.resolved_workflow_ref.fencing_context,
  };
}

std::shared_ptr<AuthorityAttachmentRef> publication_attachment_ref(const TargetPublicationRegistry::Record& record) {
  const AuthorityRef authority_ref = publication_workflow_authority_ref(record);
  return std::make_shared<AuthorityAttachmentRef>(AuthorityAttachmentRef{
      .authority_ref = authority_ref,
      .attachment_kind = "target_publication",
      .attachment_id = record.publication_id.value,
      .fencing_context = authority_ref.fencing_context,
  });
}

WorkflowOutcomeProjection publication_replay_outcome_projection(const TargetPublicationRegistry::Record& record) {
  return WorkflowOutcomeProjection{
      .projection_kind = WorkflowOutcomeProjectionKind::kExistingCapability,
      .owner_workflow_id = record.publication_id.value,
      .attachment_ref = publication_attachment_ref(record),
      .existing_capability_id = record.capability_id.empty()
          ? std::optional<std::string>(publication_capability_id(record.publication_id.value))
          : std::optional<std::string>(record.capability_id),
  };
}

void populate_publication_workflow_state(TargetPublicationRegistry::Record* record) {
  if (record == nullptr) {
    return;
  }
  if (!record->workflow_binding_projection.has_value() ||
      record->workflow_binding_projection->resolved_workflow_ref.workflow_id.empty()) {
    const auto workflow_issue_context = publication_workflow_issue_context(*record);
    if (workflow_issue_context.requested_workflow_ref.has_value()) {
      record->workflow_binding_projection =
          WorkflowBindingProjection{.resolved_workflow_ref = *workflow_issue_context.requested_workflow_ref};
    } else {
      record->workflow_binding_projection = publication_workflow_binding_projection(*record);
    }
  }
  if (!record->replay_outcome_projection.has_value()) {
    record->replay_outcome_projection = publication_replay_outcome_projection(*record);
  }
}

WorkflowGateDecision publication_gate_decision(
    const TargetPublicationRegistry::Record& record,
    WorkflowDecisionClass decision_class,
    std::optional<std::string> diagnostics = std::nullopt) {
  const auto binding_projection =
      record.workflow_binding_projection.value_or(publication_workflow_binding_projection(record));
  WorkflowGateDecision decision{
      .decision_class = decision_class,
      .resolved_workflow_ref = binding_projection.resolved_workflow_ref,
      .binding_projection = binding_projection,
      .diagnostics = std::move(diagnostics),
  };
  if (decision_class == WorkflowDecisionClass::kReplay) {
    decision.outcome_projection =
        record.replay_outcome_projection.value_or(publication_replay_outcome_projection(record));
  }
  return decision;
}

WorkflowRedemptionContext publication_workflow_redemption_context(
    const TargetPublicationRegistry::Record& record,
    const AdmittedCapabilityUse& admitted) {
  const auto binding_projection =
      record.workflow_binding_projection.value_or(publication_workflow_binding_projection(record));
  return WorkflowRedemptionContext{
      .family = "publish",
      .workflow_ref = binding_projection.resolved_workflow_ref,
      .capability_id = admitted.capability.capability_id,
      .subject_id = admitted.subject.subject_id,
      .binding_id = admitted.binding_record.has_value()
          ? std::optional<std::string>(admitted.binding_record->binding_id)
          : std::nullopt,
      .lifecycle_fencing_context = admitted.capability.epochs.fencing_context,
      .request_operation_id =
          record.request_operation_id.empty() ? std::nullopt : std::optional<std::string>(record.request_operation_id),
  };
}

PublicationSubjectEvaluation evaluate_publication_subject(
    const TargetPublicationRegistry& target_publication_registry,
    const TargetPublicationRegistry::Record& record,
    absl::Time now) {
  auto current_record_opt = target_publication_registry.lookup_current_for_subject(
      record.publication_subject_key.value, now, /*require_not_expired=*/true);
  if (!current_record_opt.has_value()) {
    return PublicationSubjectEvaluation{.match_result = PublicationSubjectMatchResult::kUnavailable};
  }
  if (current_record_opt->publication_id == record.publication_id) {
    return PublicationSubjectEvaluation{
        .match_result = PublicationSubjectMatchResult::kCurrentInstance,
        .current_record = std::move(current_record_opt),
    };
  }
  if (current_record_opt->subject_generation > record.subject_generation) {
    return PublicationSubjectEvaluation{
        .match_result = PublicationSubjectMatchResult::kOwnerLost,
        .current_record = std::move(current_record_opt),
    };
  }
  return PublicationSubjectEvaluation{
      .match_result = PublicationSubjectMatchResult::kStaleCurrent,
      .current_record = std::move(current_record_opt),
  };
}

absl::Status validate_publish_workflow_gate_claims(
    const RoutedAuthorityRequest& routed_request,
    const AuthorityRef& expected_workflow_authority_ref,
    const AuthorityRef& expected_issuer_authority_ref) {
  for (const auto& forwarded_claim : routed_request.forwarded_claims) {
    if (forwarded_claim.claim_kind != kPublishWorkflowGateClaimKind) {
      continue;
    }
    if (forwarded_claim.provenance != ForwardedClaimProvenance::kAuthorityAuthenticated) {
      return absl::FailedPreconditionError("publish workflow gate claim provenance is not authority_authenticated");
    }
    if (forwarded_claim.claim_payload != kPublishWorkflowGateClaimPayload) {
      return absl::FailedPreconditionError("publish workflow gate claim payload is invalid");
    }
    if (forwarded_claim.minted_by_authority_ref != expected_workflow_authority_ref) {
      return absl::FailedPreconditionError("publish workflow gate claim minting authority mismatch");
    }
    if (forwarded_claim.audience_authority_ref != expected_issuer_authority_ref) {
      return absl::FailedPreconditionError("publish workflow gate claim audience mismatch");
    }
    if (forwarded_claim.bound_root_request_id != routed_request.request_metadata.root_request_id) {
      return absl::FailedPreconditionError("publish workflow gate claim root_request_id mismatch");
    }
    if (!forwarded_claim.bound_path_family.has_value() ||
        *forwarded_claim.bound_path_family != routed_request.path_family) {
      return absl::FailedPreconditionError("publish workflow gate claim path_family mismatch");
    }
    if (!forwarded_claim.bound_edge.has_value() || *forwarded_claim.bound_edge != kPublishWorkflowToIssuerEdgeRef) {
      return absl::FailedPreconditionError("publish workflow gate claim edge mismatch");
    }
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError("publish workflow gate claim is required");
}

struct UseGuardScope {
  LifecycleKernel* kernel{nullptr};
  std::optional<LifecycleUseGuard> use_guard;

  ~UseGuardScope() {
    if (kernel == nullptr || !use_guard.has_value()) {
      return;
    }
    auto st = kernel->release_use_guard(*use_guard);
    LOG_IF(WARNING, !st.ok()) << "publish_target_replica: failed to release lifecycle use guard: " << st;
  }
};

void clear_binding_published_guard(BindingRegistry* bindings, const TargetPublicationRegistry::Record& record) {
  if (bindings == nullptr || record.binding_id.empty()) {
    return;
  }
  auto binding_or = bindings->get(record.binding_id);
  if (!binding_or.ok()) {
    return;
  }
  const auto binding = *binding_or;
  absl::MutexLock lock(&binding->mu);
  if (!binding->active_published_current.has_value()) {
    return;
  }
  if (binding->active_published_current->publication_id == record.publication_id.value) {
    binding->active_published_current.reset();
  }
}

absl::Status mark_binding_published(
    BindingRegistry& bindings,
    const TargetPublicationRegistry::Record& record,
    std::string_view lease_id,
    std::string_view replica_id,
    std::string_view operation_id) {
  auto binding_or = bindings.get(record.binding_id);
  if (!binding_or.ok()) {
    return binding_or.status();
  }
  const auto binding = *binding_or;
  absl::MutexLock lock(&binding->mu);
  if (binding->current_binding_value_id != record.binding_value_id ||
      binding->seal_generation != record.seal_generation) {
    return absl::FailedPreconditionError("binding current value changed before publish completion");
  }
  if (binding->active_published_current.has_value() &&
      binding->active_published_current->binding_id == binding->binding_id &&
      binding->active_published_current->binding_value_id == binding->current_binding_value_id &&
      binding->active_published_current->seal_generation == binding->seal_generation &&
      binding->active_published_current->publication_id != record.publication_id.value) {
    return absl::FailedPreconditionError("binding current value is already published; call retire() first");
  }
  binding->active_published_current = BindingRegistry::Record::PublishedCurrentValue{
      .binding_id = record.binding_id,
      .binding_layout_id = record.binding_layout_id,
      .binding_value_id = record.binding_value_id,
      .seal_generation = record.seal_generation,
      .publication_id = record.publication_id.value,
      .lease_id = std::string(lease_id),
      .replica_id = std::string(replica_id),
      .operation_id = std::string(operation_id),
  };
  return absl::OkStatus();
}

absl::Status release_publication_capability(
    LifecycleKernel* lifecycle_kernel,
    const TargetPublicationRegistry::Record& record) {
  if (lifecycle_kernel == nullptr) {
    return absl::OkStatus();
  }
  const std::string capability_id =
      record.capability_id.empty() ? publication_capability_id(record.publication_id.value) : record.capability_id;
  return lifecycle_kernel->release_capability(capability_id);
}

absl::StatusOr<TargetPublicationRegistry::Record> terminalize_publication_record(
    const std::shared_ptr<TargetPublicationRegistry>& registry,
    LifecycleKernel* lifecycle_kernel,
    BindingRegistry* bindings,
    std::string_view publication_id,
    std::string_view reason) {
  if (registry == nullptr) {
    return absl::FailedPreconditionError("target publication registry unavailable");
  }
  auto record = registry->terminalize(publication_id, reason);
  if (!record.has_value()) {
    return absl::NotFoundError("publication is not active");
  }
  clear_binding_published_guard(bindings, *record);
  auto release_status = release_publication_capability(lifecycle_kernel, *record);
  if (!release_status.ok() && !absl::IsNotFound(release_status)) {
    return release_status;
  }
  return *record;
}

absl::StatusOr<v2::PublishTargetReplicaResponse> execute_publish_target_replica(
    const TargetPublishService::Dep& d,
    std::shared_ptr<TargetPublicationRegistry> target_publication_registry,
    TargetPublishService::TargetPublicationFrontDoorContext publish_context,
    std::string_view request_operation_id,
    uint32_t ttl_ms) {
  auto record = std::move(publish_context.record);
  populate_publication_workflow_state(&record);
  const std::string effective_operation_id =
      !request_operation_id.empty() ? std::string(request_operation_id) : record.request_operation_id;
  if (record.published) {
    if (record.published_operation_id != effective_operation_id) {
      return absl::FailedPreconditionError(
          "binding_current_value_publication_token already published with a different operation_id");
    }
    if (record.published_lease_id.empty() || record.published_replica_id.empty()) {
      return absl::FailedPreconditionError("published target publication record is missing replay ids");
    }
    v2::PublishTargetReplicaResponse response;
    response.set_lease_id(record.published_lease_id);
    response.set_replica_id(record.published_replica_id);
    return response;
  }
  auto scope_record_status = validate_target_publication_scope_against_record(
      record, publish_context.scope, publish_context.normalized_byte_space);
  if (!scope_record_status.ok()) {
    return scope_record_status;
  }
  auto binding_status = validate_binding_current_value_against_publication(d.bindings, record, publish_context.scope);
  if (!binding_status.ok()) {
    return binding_status;
  }
  const auto subject_evaluation = evaluate_publication_subject(*target_publication_registry, record, absl::Now());
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kOwnerLost) {
    return absl::UnavailableError("publish workflow owner lost");
  }
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kUnavailable) {
    return absl::UnavailableError("target_publication workflow state unavailable");
  }
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kStaleCurrent) {
    return absl::FailedPreconditionError("binding_current_value_publication_token is stale for target");
  }

  const auto device =
      d.devices.From(v2::DeviceType::DEVICE_TYPE_GPU, publish_context.scope.device_uuid(), std::nullopt);
  if (device.type != DeviceType::GPU || device.ordinal < 0) {
    return absl::InvalidArgumentError("invalid device_uuid for binding_current_value_publication_token");
  }

  const std::string view_id =
      publish_context.normalized_byte_space.kind() == tensorcast::common::v1::BYTE_SPACE_KIND_VIEW
      ? publish_context.normalized_byte_space.id()
      : "";
  ArtifactDeviceKey key{
      .artifact_id = publish_context.scope.selection().artifact_id(), .view_id = view_id, .device_id = device.ordinal};

  if (auto active = d.lip_manager.find_active_by_key(key); active.has_value()) {
    if (active->registration_id == publish_context.scope.publication_id()) {
      auto replica_id = d.lip_manager.find_replica_id(key);
      if (!replica_id.has_value()) {
        return absl::FailedPreconditionError("target already published without replica_id");
      }
      v2::PublishTargetReplicaResponse response;
      response.set_lease_id(publish_context.scope.publication_id());
      response.set_replica_id(*replica_id);
      return response;
    }
    return absl::AlreadyExistsError("another lease already exists for target");
  }

  uint64_t total_size = 0;
  for (const auto& seg : record.segments) {
    if (seg.length == 0) {
      continue;
    }
    const uint64_t end = seg.artifact_offset + seg.length;
    if (end > total_size) {
      total_size = end;
    }
  }
  if (total_size == 0) {
    return absl::FailedPreconditionError("binding_current_value_publication_token has empty segments");
  }

  struct LipRollback {
    LipManager* lip{nullptr};
    std::string registration_id;
    bool active{true};

    ~LipRollback() {
      if (!active || lip == nullptr) {
        return;
      }
      absl::Status st = lip->revoke_by_registration_id(registration_id);
      if (!st.ok()) {
        LOG(WARNING) << "PublishTargetReplica rollback: revoke failed for id=" << registration_id << ": " << st;
      }
    }

    void release() {
      active = false;
    }
  } lip_rollback{.lip = &d.lip_manager, .registration_id = publish_context.scope.publication_id()};

  const uint64_t epoch = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now()));
  auto lease_or = d.lip_manager.commit_routable_view_lease_in_place(
      publish_context.scope.publication_id(),
      publish_context.scope.selection().artifact_id(),
      view_id,
      device.ordinal,
      publish_context.scope.owner_pid(),
      ttl_ms,
      epoch,
      total_size,
      std::move(record.segments),
      std::move(record.storages));
  if (!lease_or.ok()) {
    lip_rollback.release();
    return lease_or.status();
  }

  std::string worker_id = d.identity.worker_id();
  if (worker_id.empty()) {
    return absl::UnavailableError(
        absl::StrCat(
            "worker identity unavailable while publishing target replica for artifact_id=",
            publish_context.scope.selection().artifact_id(),
            " view_id=",
            view_id,
            "; routable replicas require completed worker lifecycle registration"));
  }

  const std::optional<std::string_view> register_client_request_id = !effective_operation_id.empty()
      ? std::optional<std::string_view>(effective_operation_id)
      : std::optional<std::string_view>(record.publication_id.value);
  auto replica_id_or = d.global_store_client->register_memory_replica_idempotent(
      publish_context.scope.selection().artifact_id(),
      worker_id,
      device,
      total_size,
      record.index_key_hex,
      lease_or->remote_memory_keys,
      lease_or->buffer_sizes,
      record.canonical_index_json,
      /*encoding=*/"json",
      /*schema_version=*/"v3",
      /*max_concurrency=*/std::max<uint32_t>(1, d.max_concurrency),
      /*verification_json=*/std::nullopt,
      view_id.empty() ? std::nullopt : std::optional<std::string_view>(view_id),
      /*descriptor=*/std::nullopt,
      register_client_request_id,
      kTargetPublicationExportGeneration);
  if (!replica_id_or.ok()) {
    return replica_id_or.status();
  }
  const std::string replica_id = *replica_id_or;
  d.lip_manager.attach_replica_id(publish_context.scope.publication_id(), replica_id);

  struct ReplicaRollback {
    store::components::IGlobalStoreClient* client{nullptr};
    std::string artifact_id;
    std::string replica_id;
    bool active{true};

    ~ReplicaRollback() {
      if (!active || client == nullptr || replica_id.empty()) {
        return;
      }
      if (!client->is_connected()) {
        return;
      }
      auto st = client->unregister_replica(artifact_id, replica_id);
      if (!st.ok()) {
        LOG(WARNING) << "PublishTargetReplica rollback: unregister_replica failed for artifact_id=" << artifact_id
                     << " replica_id=" << replica_id << ": " << st;
      }
    }

    void release() {
      active = false;
    }
  } replica_rollback{
      .client = d.global_store_client.get(),
      .artifact_id = publish_context.scope.selection().artifact_id(),
      .replica_id = replica_id,
  };

  auto barrier_status = await_state_sync_barrier(d);
  if (!barrier_status.ok()) {
    return barrier_status;
  }

  maybe_report_progressive_published_target_coverage(d, record, replica_id, total_size);

  auto lifecycle_lease_or = d.lifecycle.create_use_lease(
      store::loading::ReplicaKey{
          .artifact_id = publish_context.scope.selection().artifact_id(),
          .view_id = view_id.empty() ? std::nullopt : std::optional<std::string>(view_id),
          .device =
              store::DeviceKey{
                  .type = DeviceType::GPU,
                  .ordinal = device.ordinal,
                  .uuid = publish_context.scope.device_uuid(),
              },
          .replica = 0,
      },
      publish_context.scope.owner_pid(),
      {[registry = target_publication_registry,
        lifecycle_kernel = &d.lifecycle_kernel,
        bindings = &d.bindings,
        lip_manager = &d.lip_manager,
        global_store_client = d.global_store_client,
        artifact_id = publish_context.scope.selection().artifact_id(),
        replica_id,
        operation_id = effective_operation_id,
        publication_id = record.publication_id.value]() -> absl::Status {
        if (global_store_client && global_store_client->is_connected() && !replica_id.empty()) {
          const std::optional<std::string_view> operation_id_view =
              operation_id.empty() ? std::nullopt : std::optional<std::string_view>(operation_id);
          auto mark_or = global_store_client->mark_replica_unavailable(
              artifact_id, replica_id, /*reason=*/"owner_pid_exit", operation_id_view);
          if (!mark_or.ok() && !absl::IsNotFound(mark_or.status())) {
            LOG(WARNING) << "target publication owner cleanup: mark_replica_unavailable failed for artifact_id="
                         << artifact_id << " replica_id=" << replica_id << ": " << mark_or.status();
          }
        }
        auto terminalized_or =
            terminalize_publication_record(registry, lifecycle_kernel, bindings, publication_id, "owner_pid_exit");
        if (!terminalized_or.ok() && !absl::IsNotFound(terminalized_or.status())) {
          return terminalized_or.status();
        }
        auto revoke_status = lip_manager->revoke_by_registration_id(publication_id);
        if (!revoke_status.ok() && !absl::IsNotFound(revoke_status)) {
          return revoke_status;
        }
        return absl::OkStatus();
      }});
  if (!lifecycle_lease_or.ok()) {
    return lifecycle_lease_or.status();
  }

  auto mark_binding_status =
      mark_binding_published(d.bindings, record, record.publication_id.value, replica_id, effective_operation_id);
  if (!mark_binding_status.ok()) {
    d.lifecycle.release_lease(*lifecycle_lease_or);
    return mark_binding_status;
  }
  auto marked_record = target_publication_registry->mark_published(
      record.publication_id.value,
      effective_operation_id,
      record.publication_id.value,
      replica_id,
      *lifecycle_lease_or,
      absl::Now());
  if (!marked_record.has_value()) {
    d.lifecycle.release_lease(*lifecycle_lease_or);
    clear_binding_published_guard(&d.bindings, record);
    return absl::NotFoundError("publication record disappeared before publish completion");
  }

  lip_rollback.release();
  replica_rollback.release();
  v2::PublishTargetReplicaResponse response;
  response.set_lease_id(publish_context.scope.publication_id());
  response.set_replica_id(replica_id);
  return response;
}

} // namespace

TargetPublishService::TargetPublishService(Dep d)
    : d_(std::move(d)),
      capability_tokens_(d_.capability_tokens),
      target_publication_registry_(
          std::make_shared<TargetPublicationRegistry>(
              TargetPublicationRegistry::Options{.ttl = binding_current_value_publication_token_ttl()})),
      publish_operation_tracker_(std::make_shared<PublishOperationTracker>()) {}

absl::Duration TargetPublishService::binding_current_value_publication_token_ttl() {
  return absl::Minutes(5);
}

absl::Status TargetPublishService::terminalize_publication(
    std::string_view publication_id,
    std::string_view reason,
    bool release_published_lifecycle_lease) {
  auto terminalized_or = terminalize_publication_record(
      target_publication_registry_, &d_.lifecycle_kernel, &d_.bindings, publication_id, reason);
  if (!terminalized_or.ok()) {
    return terminalized_or.status();
  }
  if (release_published_lifecycle_lease && terminalized_or->published_lifecycle_lease_id != 0) {
    d_.lifecycle.release_lease(terminalized_or->published_lifecycle_lease_id);
  }
  return absl::OkStatus();
}

absl::StatusOr<TargetPublicationRegistry::Record> TargetPublishService::remember_target_publication(
    TargetPublicationRegistry::Record record) {
  const absl::Time now = absl::Now();
  record.workflow_recovery_class = WorkflowRecoveryClass::kEphemeralProcessLocal;
  populate_publication_workflow_state(&record);
  const absl::Duration ttl = record.expires_at == absl::InfinitePast() ? binding_current_value_publication_token_ttl()
                                                                       : record.expires_at - now;
  auto lease_or = d_.lifecycle.create_retention_lease(
      ttl,
      std::vector<std::function<absl::Status()>>{
          [this, publication_id = record.publication_id.value]() -> absl::Status {
            auto record =
                target_publication_registry_->lookup(publication_id, absl::Now(), /*require_not_expired=*/false);
            if (record.has_value() && record->published) {
              return absl::OkStatus();
            }
            target_publication_registry_->erase(publication_id);
            auto release_status = d_.lifecycle_kernel.release_capability(publication_capability_id(publication_id));
            if (!release_status.ok() && !absl::IsNotFound(release_status)) {
              return release_status;
            }
            return absl::OkStatus();
          },
      });
  if (!lease_or.ok()) {
    return lease_or.status();
  }
  record.capability_id = publication_capability_id(record.publication_id.value);
  record.lease_id = *lease_or;
  record = target_publication_registry_->insert(std::move(record));

  LifecycleSubjectRecord subject;
  subject.subject_id = publication_subject_id(record);
  subject.epochs.subject_generation = record.subject_generation == 0 ? 1 : record.subject_generation;
  subject.subject_kind = LifecycleSubjectKind::kPublicationTarget;
  subject.created_at = now;
  subject.last_observed_at = now;
  subject.artifact_id = record.selection.artifact_id();
  subject.semantic_ref_id = record.publication_id.value;
  subject.workflow_companion = record.workflow_binding_projection->resolved_workflow_ref;
  auto capability_or = d_.lifecycle_kernel.mint_capability(
      MintCapabilityRequest{
          .subject = subject,
          .address =
              CapabilityBindingAddress{
                  .route_principal = make_issuer_route_principal(d_.identity.daemon_id()),
                  .family = LifecycleCapabilityFamily::kPublish,
                  .binding_space = LifecycleBindingSpace::kPublication,
                  .binding_key_kind = BindingKeyKind::kPublicationId,
                  .binding_key = record.publication_id.value,
                  .epochs = subject.epochs,
              },
          .front_door_kind = LifecycleFrontDoorKind::kTargetPublicationToken,
          .capability_id = record.capability_id,
          .lease_id = *lease_or,
          .capability_expires_at = record.expires_at,
          .carriage_kind = CredentialCarriageKind::kSelfDescribing,
          .binding_mode = LifecycleBindingMode::kAddressDerived,
          .constraint_claims =
              ConstraintClaims{
                  .artifact_id = record.selection.artifact_id(),
                  .operation_id = record.request_operation_id,
              },
          .workflow_companion = record.workflow_binding_projection->resolved_workflow_ref,
          .workflow_gate = WorkflowGateKind::kRequired,
      });
  if (!capability_or.ok()) {
    target_publication_registry_->erase(record.publication_id.value);
    d_.lifecycle.release_lease(*lease_or);
    return capability_or.status();
  }
  return record;
}

absl::StatusOr<TargetPublishService::TargetPublicationFrontDoorContext> TargetPublishService::
    inspect_target_publication_context(const v2::PublishTargetReplicaRequest& req, absl::Time now) const {
  const LocalObservationSet local_observations = build_target_publication_local_observations(req);
  auto inspection_or = inspect_target_publication_credential(
      capability_tokens_,
      req.binding_current_value_publication_token(),
      now,
      d_.identity.daemon_id(),
      local_observations);
  if (!inspection_or.ok()) {
    return inspection_or.status();
  }
  auto scope_validation_status = validate_target_publication_request_against_scope(req, *inspection_or);
  if (!scope_validation_status.ok()) {
    return scope_validation_status;
  }

  auto record_opt =
      target_publication_registry_->lookup(inspection_or->scope.publication_id(), now, /*require_not_expired=*/true);
  if (!record_opt.has_value()) {
    return absl::NotFoundError("binding_current_value_publication_token is no longer valid");
  }
  bind_record_to_target_publication_front_door_context(*record_opt, &inspection_or->front_door_context);

  return TargetPublicationFrontDoorContext{
      .record = std::move(*record_opt),
      .scope = std::move(inspection_or->scope),
      .normalized_byte_space = std::move(inspection_or->normalized_byte_space),
      .front_door_context = std::move(inspection_or->front_door_context),
  };
}

absl::StatusOr<RoutedAuthorityRequest> TargetPublishService::build_target_publication_workflow_routed_request(
    const v2::PublishTargetReplicaRequest& req,
    absl::Time now) const {
  const LocalObservationSet local_observations = build_target_publication_local_observations(req);
  auto inspection_or = inspect_target_publication_credential(
      capability_tokens_,
      req.binding_current_value_publication_token(),
      now,
      /*expected_issuer=*/"",
      local_observations);
  if (!inspection_or.ok()) {
    return inspection_or.status();
  }
  auto scope_validation_status = validate_target_publication_request_against_scope(req, *inspection_or);
  if (!scope_validation_status.ok()) {
    return scope_validation_status;
  }
  if (!inspection_or->front_door_context.forwardable_evidence.has_value()) {
    return absl::FailedPreconditionError("target_publication workflow route requires forwardable_evidence");
  }
  auto evidence_validation_status =
      require_target_publication_raw_issuer_evidence(*inspection_or->front_door_context.forwardable_evidence);
  if (!evidence_validation_status.ok()) {
    return evidence_validation_status.status();
  }
  auto record_opt =
      target_publication_registry_->lookup(inspection_or->scope.publication_id(), now, /*require_not_expired=*/true);
  if (!record_opt.has_value()) {
    return absl::NotFoundError("binding_current_value_publication_token is no longer valid");
  }
  bind_record_to_target_publication_front_door_context(*record_opt, &inspection_or->front_door_context);
  const auto authority_ref = publication_routed_workflow_authority_ref(inspection_or->envelope.issuer_daemon_id());
  const auto portable_credential = derive_target_publication_portable_credential(inspection_or->front_door_context);
  const std::string root_request_id = target_publication_root_request_id(inspection_or->scope.publication_id());
  RoutedAuthorityRequest routed_request{
      .authority_ref = authority_ref,
      .path_family = std::string(kPublishWorkflowPathFamily),
      .stage_ref = std::string(kPublishWorkflowGateStageRef),
      .portable_credential = portable_credential,
      .forwardable_evidence = inspection_or->front_door_context.forwardable_evidence,
      .portable_credential_envelope = make_delegation_envelope(
          authority_ref,
          root_request_id,
          kPublishWorkflowPathFamily,
          DelegationPayloadKind::kPortableCredential,
          DelegationClass::kBootstrapSafe,
          portable_credential.credential_expires_at),
      .forwardable_evidence_envelope = make_delegation_envelope(
          authority_ref,
          root_request_id,
          kPublishWorkflowPathFamily,
          DelegationPayloadKind::kForwardableEvidence,
          DelegationClass::kOwnerScopedSensitive,
          portable_credential.credential_expires_at),
      .request_metadata =
          RoutedRequestMetadata{
              .root_request_id = std::move(root_request_id),
              .hop_budget_remaining = 1,
          },
  };
  if (const std::string idempotency_key = publication_request_idempotency_key(*record_opt); !idempotency_key.empty()) {
    routed_request.request_metadata.idempotency_key = idempotency_key;
  }
  auto validation_status = routed_authority_wire::validate_routed_authority_request_shape(routed_request);
  if (!validation_status.ok()) {
    return validation_status;
  }
  return routed_request;
}

absl::StatusOr<RoutedAuthorityRequest> TargetPublishService::build_target_publication_workflow_continuation_request(
    const RoutedAuthorityRequest& routed_request,
    const OwnerStageReply& workflow_gate_reply) const {
  if (!is_publish_workflow_gate_stage(routed_request.path_family, routed_request.stage_ref)) {
    return absl::FailedPreconditionError("publish workflow continuation requires workflow_gate request");
  }
  if (workflow_gate_reply.reply_kind != OwnerStageReplyKind::kContinueWithAuthority ||
      !workflow_gate_reply.continuation.has_value()) {
    return absl::FailedPreconditionError("publish workflow continuation requires continue_with_authority reply");
  }
  const auto& continuation = *workflow_gate_reply.continuation;
  RoutedAuthorityRequest continued_request{
      .authority_ref = continuation.next_authority_ref,
      .path_family = routed_request.path_family,
      .stage_ref = std::string(kPublishIssuerValidateStageRef),
      .portable_credential = routed_request.portable_credential,
      .forwardable_evidence = routed_request.forwardable_evidence,
      .portable_credential_envelope = make_delegation_envelope(
          continuation.next_authority_ref,
          routed_request.request_metadata.root_request_id,
          routed_request.path_family,
          DelegationPayloadKind::kPortableCredential,
          DelegationClass::kBootstrapSafe,
          routed_request.portable_credential.credential_expires_at,
          kPublishWorkflowToIssuerEdgeRef),
      .hop_auth_context = routed_request.hop_auth_context,
      .forwarded_claims = continuation.forwarded_claims,
      .forwarded_claims_envelope = continuation.forwarded_claims_envelope,
      .request_metadata = routed_request.request_metadata,
  };
  if (routed_request.forwardable_evidence.has_value()) {
    continued_request.forwardable_evidence_envelope = make_delegation_envelope(
        continuation.next_authority_ref,
        routed_request.request_metadata.root_request_id,
        routed_request.path_family,
        DelegationPayloadKind::kForwardableEvidence,
        DelegationClass::kOwnerScopedSensitive,
        routed_request.portable_credential.credential_expires_at,
        kPublishWorkflowToIssuerEdgeRef);
  }
  if (continued_request.request_metadata.hop_budget_remaining > 0) {
    --continued_request.request_metadata.hop_budget_remaining;
  }
  auto validation_status = routed_authority_wire::validate_routed_authority_request_shape(continued_request);
  if (!validation_status.ok()) {
    return validation_status;
  }
  return continued_request;
}

absl::StatusOr<std::optional<OwnerStageReply>> TargetPublishService::maybe_route_authority_stage(
    const RoutedAuthorityRequest& routed_request,
    absl::Time now) {
  if (is_publish_workflow_gate_stage(routed_request.path_family, routed_request.stage_ref)) {
    const auto expected_workflow_authority = publication_routed_workflow_authority_ref(d_.identity.daemon_id());
    if (routed_request.authority_ref != expected_workflow_authority) {
      return absl::FailedPreconditionError("target_publication workflow authority mismatch");
    }
    if (!routed_request.forwardable_evidence.has_value()) {
      return absl::FailedPreconditionError("target_publication workflow route requires forwardable_evidence");
    }
    auto raw_token_or = require_target_publication_raw_issuer_evidence(*routed_request.forwardable_evidence);
    if (!raw_token_or.ok()) {
      return raw_token_or.status();
    }
    auto inspection_or = inspect_target_publication_credential(
        capability_tokens_, *raw_token_or, now, d_.identity.daemon_id(), LocalObservationSet{});
    if (!inspection_or.ok()) {
      return inspection_or.status();
    }
    auto record_opt =
        target_publication_registry_->lookup(inspection_or->scope.publication_id(), now, /*require_not_expired=*/true);
    if (!record_opt.has_value()) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_workflow_authority,
          kPublishWorkflowGateStageRef,
          "unavailable",
          "publish_workflow_state_missing",
          "target_publication workflow state unavailable"));
    }
    auto record = std::move(*record_opt);
    populate_publication_workflow_state(&record);
    bind_record_to_target_publication_front_door_context(record, &inspection_or->front_door_context);
    const auto portable_credential = derive_target_publication_portable_credential(inspection_or->front_door_context);
    if (portable_credential != routed_request.portable_credential) {
      return absl::FailedPreconditionError("target_publication portable credential mismatches workflow-owner state");
    }
    auto scope_record_status = validate_target_publication_scope_against_record(
        record, inspection_or->scope, inspection_or->normalized_byte_space);
    if (!scope_record_status.ok()) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_workflow_authority,
          kPublishWorkflowGateStageRef,
          "failed_precondition",
          std::string(scope_record_status.message()),
          std::string(scope_record_status.message())));
    }
    auto binding_status = validate_binding_current_value_against_publication(d_.bindings, record, inspection_or->scope);
    if (!binding_status.ok()) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_workflow_authority,
          kPublishWorkflowGateStageRef,
          "failed_precondition",
          std::string(binding_status.message()),
          std::string(binding_status.message())));
    }
    const auto subject_evaluation = evaluate_publication_subject(*target_publication_registry_, record, now);
    if (subject_evaluation.match_result == PublicationSubjectMatchResult::kOwnerLost) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_workflow_authority,
          kPublishWorkflowGateStageRef,
          "unavailable",
          "publish_owner_lost",
          "publish owner lost"));
    }
    if (subject_evaluation.match_result == PublicationSubjectMatchResult::kUnavailable) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_workflow_authority,
          kPublishWorkflowGateStageRef,
          "unavailable",
          "publish_workflow_state_missing",
          "target_publication workflow state unavailable"));
    }
    if (subject_evaluation.match_result == PublicationSubjectMatchResult::kStaleCurrent) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_workflow_authority,
          kPublishWorkflowGateStageRef,
          "failed_precondition",
          "publish_stale_current",
          "publish currentness rejected"));
    }
    const auto operation_ref = build_publish_operation_ref(record);
    if (d_.global_store_client != nullptr && d_.global_store_client->is_connected()) {
      auto operation_or = d_.global_store_client->get_operation(build_get_operation_request(operation_ref));
      if (operation_or.ok()) {
        const auto state = operation_or->status().state();
        if (state == operation::OPERATION_STATE_PENDING || state == operation::OPERATION_STATE_RUNNING ||
            state == operation::OPERATION_STATE_DEGRADED) {
          OwnerStageReply reply{
              .answered_by = expected_workflow_authority,
              .path_family = std::string(kPublishWorkflowPathFamily),
              .stage_ref = std::string(kPublishWorkflowGateStageRef),
              .reply_kind = OwnerStageReplyKind::kAttachExisting,
              .attachment_ref = *publication_attachment_ref(record),
              .diagnostics = std::string("publish workflow operation already exists"),
          };
          auto shape_status = routed_authority_wire::validate_owner_stage_reply_shape(reply);
          if (!shape_status.ok()) {
            return shape_status;
          }
          return std::optional<OwnerStageReply>(std::move(reply));
        }
        if (state == operation::OPERATION_STATE_SUCCESS) {
          return std::optional<OwnerStageReply>(publication_terminal_reply(
              expected_workflow_authority,
              kPublishWorkflowGateStageRef,
              "ok",
              "publish_replay_terminal",
              "publish workflow reused terminal operation"));
        }
        if (state == operation::OPERATION_STATE_FAILED || state == operation::OPERATION_STATE_CANCELLED) {
          const auto failed_status = status_from_operation_status(operation_or->status());
          return std::optional<OwnerStageReply>(publication_terminal_reply(
              expected_workflow_authority,
              kPublishWorkflowGateStageRef,
              std::string(absl::StatusCodeToString(failed_status.code())),
              "publish_replay_terminal",
              std::string(failed_status.message())));
        }
      } else if (!absl::IsNotFound(operation_or.status())) {
        return operation_or.status();
      }
    }
    const auto issuer_authority = publication_routed_issuer_authority_ref(d_.identity.daemon_id());
    ForwardedClaim workflow_gate_claim{
        .claim_kind = std::string(kPublishWorkflowGateClaimKind),
        .provenance = ForwardedClaimProvenance::kAuthorityAuthenticated,
        .claim_payload = std::string(kPublishWorkflowGateClaimPayload),
        .minted_by_authority_ref = expected_workflow_authority,
        .audience_authority_ref = issuer_authority,
        .bound_root_request_id = routed_request.request_metadata.root_request_id,
        .bound_path_family = routed_request.path_family,
        .bound_edge = std::string(kPublishWorkflowToIssuerEdgeRef),
        .claim_expires_at = routed_request.portable_credential.credential_expires_at,
    };
    OwnerStageReply reply{
        .answered_by = expected_workflow_authority,
        .path_family = std::string(kPublishWorkflowPathFamily),
        .stage_ref = std::string(kPublishWorkflowGateStageRef),
        .reply_kind = OwnerStageReplyKind::kContinueWithAuthority,
        .continuation =
            AuthorityContinuation{
                .next_authority_ref = issuer_authority,
                .edge_ref = std::string(kPublishWorkflowToIssuerEdgeRef),
                .forwarded_claims = {std::move(workflow_gate_claim)},
                .forwarded_claims_envelope = make_delegation_envelope(
                    issuer_authority,
                    routed_request.request_metadata.root_request_id,
                    kPublishWorkflowPathFamily,
                    DelegationPayloadKind::kForwardedClaim,
                    DelegationClass::kOwnerScopedSensitive,
                    routed_request.portable_credential.credential_expires_at,
                    kPublishWorkflowToIssuerEdgeRef),
                .continuation_reason = std::string("publish_workflow_gate_admitted"),
            },
        .diagnostics = std::string("publish workflow gate admitted"),
    };
    auto shape_status = routed_authority_wire::validate_owner_stage_reply_shape(reply);
    if (!shape_status.ok()) {
      return shape_status;
    }
    return std::optional<OwnerStageReply>(std::move(reply));
  }

  if (is_publish_issuer_validate_stage(routed_request.path_family, routed_request.stage_ref)) {
    const auto expected_issuer_authority = publication_routed_issuer_authority_ref(d_.identity.daemon_id());
    if (routed_request.authority_ref != expected_issuer_authority) {
      return absl::FailedPreconditionError("target_publication issuer authority mismatch");
    }
    if (!routed_request.forwardable_evidence.has_value()) {
      return absl::FailedPreconditionError("target_publication issuer route requires forwardable_evidence");
    }
    auto raw_token_or = require_target_publication_raw_issuer_evidence(*routed_request.forwardable_evidence);
    if (!raw_token_or.ok()) {
      return raw_token_or.status();
    }
    auto inspection_or = inspect_target_publication_credential(
        capability_tokens_, *raw_token_or, now, d_.identity.daemon_id(), LocalObservationSet{});
    if (!inspection_or.ok()) {
      return inspection_or.status();
    }
    auto workflow_gate_status = validate_publish_workflow_gate_claims(
        routed_request, publication_routed_workflow_authority_ref(d_.identity.daemon_id()), expected_issuer_authority);
    if (!workflow_gate_status.ok()) {
      return workflow_gate_status;
    }
    auto record_opt =
        target_publication_registry_->lookup(inspection_or->scope.publication_id(), now, /*require_not_expired=*/true);
    if (!record_opt.has_value()) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_issuer_authority,
          kPublishIssuerValidateStageRef,
          "unavailable",
          "publish_workflow_state_missing",
          "target_publication workflow state unavailable"));
    }
    auto record = std::move(*record_opt);
    populate_publication_workflow_state(&record);
    bind_record_to_target_publication_front_door_context(record, &inspection_or->front_door_context);
    const auto portable_credential = derive_target_publication_portable_credential(inspection_or->front_door_context);
    if (portable_credential != routed_request.portable_credential) {
      return absl::FailedPreconditionError("target_publication portable credential mismatches issuer-validated state");
    }
    auto scope_record_status = validate_target_publication_scope_against_record(
        record, inspection_or->scope, inspection_or->normalized_byte_space);
    if (!scope_record_status.ok()) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_issuer_authority,
          kPublishIssuerValidateStageRef,
          "failed_precondition",
          std::string(scope_record_status.message()),
          std::string(scope_record_status.message())));
    }
    auto binding_status = validate_binding_current_value_against_publication(d_.bindings, record, inspection_or->scope);
    if (!binding_status.ok()) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_issuer_authority,
          kPublishIssuerValidateStageRef,
          "failed_precondition",
          std::string(binding_status.message()),
          std::string(binding_status.message())));
    }
    const auto subject_evaluation = evaluate_publication_subject(*target_publication_registry_, record, now);
    if (subject_evaluation.match_result == PublicationSubjectMatchResult::kOwnerLost) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_issuer_authority,
          kPublishIssuerValidateStageRef,
          "unavailable",
          "publish_owner_lost",
          "publish owner lost"));
    }
    if (subject_evaluation.match_result == PublicationSubjectMatchResult::kUnavailable) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_issuer_authority,
          kPublishIssuerValidateStageRef,
          "unavailable",
          "publish_workflow_state_missing",
          "target_publication workflow state unavailable"));
    }
    if (subject_evaluation.match_result == PublicationSubjectMatchResult::kStaleCurrent) {
      return std::optional<OwnerStageReply>(publication_terminal_reply(
          expected_issuer_authority,
          kPublishIssuerValidateStageRef,
          "failed_precondition",
          "publish_stale_current",
          "publish currentness rejected"));
    }
    OwnerStageReply reply = publication_terminal_reply(
        expected_issuer_authority,
        kPublishIssuerValidateStageRef,
        "ok",
        "publish_workflow_gate_admitted",
        "publish issuer validation admitted");
    auto shape_status = routed_authority_wire::validate_owner_stage_reply_shape(reply);
    if (!shape_status.ok()) {
      return shape_status;
    }
    return std::optional<OwnerStageReply>(std::move(reply));
  }

  return std::optional<OwnerStageReply>();
}

grpc::Status TargetPublishService::publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::PublishTargetReplicaResponse& resp) {
  auto& span = rctx.span();
  if (rctx.allow_high_card_attrs() && req.has_operation_id()) {
    span->SetAttribute("tc.operation.id", req.operation_id());
  }
  if (req.binding_current_value_publication_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_current_value_publication_token is required"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "Global Store client unavailable"};
  }

  auto publish_context_or = inspect_target_publication_context(req, absl::Now());
  if (!publish_context_or.ok()) {
    return to_grpc_status(publish_context_or.status());
  }
  auto publish_context = std::move(*publish_context_or);
  auto record = std::move(publish_context.record);
  populate_publication_workflow_state(&record);
  if (record.request_operation_id.empty() && req.has_operation_id() && !req.operation_id().empty()) {
    record.request_operation_id = req.operation_id();
    record.workflow_binding_projection.reset();
    record.replay_outcome_projection.reset();
    populate_publication_workflow_state(&record);
    record = target_publication_registry_->insert(std::move(record));
    bind_record_to_target_publication_front_door_context(record, &publish_context.front_door_context);
  }
  auto admitted_or = d_.lifecycle_kernel.admit_redemption(publish_context.front_door_context.parsed_credential);
  if (!admitted_or.ok()) {
    return to_grpc_status(admitted_or.status());
  }
  UseGuardScope use_guard_scope{.kernel = &d_.lifecycle_kernel, .use_guard = admitted_or->use_guard};
  const auto workflow_redemption_context = publication_workflow_redemption_context(record, *admitted_or);
  span->SetAttribute(
      "tc.front_door.forwardable_evidence_present",
      publish_context.front_door_context.forwardable_evidence.has_value());
  span->SetAttribute(
      "tc.front_door.local_observation_count",
      static_cast<int64_t>(publish_context.front_door_context.local_observations.observations.size()));
  if (publish_context.front_door_context.forwardable_evidence.has_value()) {
    span->SetAttribute(
        "tc.front_door.forwardable_evidence_kind",
        static_cast<int>(publish_context.front_door_context.forwardable_evidence->evidence_kind));
  }
  span->SetAttribute("tc.workflow.family", workflow_redemption_context.family);
  span->SetAttribute("tc.workflow.owner_kind", static_cast<int>(workflow_redemption_context.workflow_ref.owner_kind));
  span->SetAttribute("tc.workflow.recovery_class", static_cast<int>(record.workflow_recovery_class));
  auto scope_record_status = validate_target_publication_scope_against_record(
      record, publish_context.scope, publish_context.normalized_byte_space);
  if (!scope_record_status.ok()) {
    return to_grpc_status(scope_record_status);
  }
  auto binding_status = validate_binding_current_value_against_publication(d_.bindings, record, publish_context.scope);
  if (!binding_status.ok()) {
    return to_grpc_status(binding_status);
  }
  const auto subject_evaluation = evaluate_publication_subject(*target_publication_registry_, record, absl::Now());
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kOwnerLost) {
    return {StatusCode::UNAVAILABLE, "publish workflow owner lost"};
  }
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kUnavailable) {
    return {StatusCode::UNAVAILABLE, "target_publication workflow state unavailable"};
  }
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kStaleCurrent) {
    const auto stale_current =
        publication_gate_decision(record, WorkflowDecisionClass::kStaleCurrent, "publish currentness rejected");
    span->SetAttribute("tc.workflow.decision_class", static_cast<int>(stale_current.decision_class));
    return {StatusCode::FAILED_PRECONDITION, "binding_current_value_publication_token is stale for target"};
  }

  publish_context.record = record;
  auto result_or = execute_publish_target_replica(
      d_,
      target_publication_registry_,
      std::move(publish_context),
      req.has_operation_id() ? std::string_view(req.operation_id()) : std::string_view(),
      req.has_ttl_ms() ? req.ttl_ms() : 0U);
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  resp = std::move(*result_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status TargetPublishService::start_publish_target_replica(
    RpcContext& rctx,
    const v2::PublishTargetReplicaRequest& req,
    v2::StartPublishTargetReplicaResponse& resp) {
  auto& span = rctx.span();
  if (req.binding_current_value_publication_token().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "binding_current_value_publication_token is required"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "Global Store client unavailable"};
  }

  auto publish_context_or = inspect_target_publication_context(req, absl::Now());
  if (!publish_context_or.ok()) {
    return to_grpc_status(publish_context_or.status());
  }
  auto publish_context = std::move(*publish_context_or);
  auto record = std::move(publish_context.record);
  populate_publication_workflow_state(&record);
  if (record.request_operation_id.empty() && req.has_operation_id() && !req.operation_id().empty()) {
    record.request_operation_id = req.operation_id();
    record.workflow_binding_projection.reset();
    record.replay_outcome_projection.reset();
    populate_publication_workflow_state(&record);
    record = target_publication_registry_->insert(std::move(record));
  }
  bind_record_to_target_publication_front_door_context(record, &publish_context.front_door_context);
  publish_context.record = record;
  const auto operation_ref = build_publish_operation_ref(record);
  if (rctx.allow_high_card_attrs()) {
    span->SetAttribute("tc.operation.id", operation_ref.operation_id());
  }

  auto admitted_or = d_.lifecycle_kernel.admit_redemption(publish_context.front_door_context.parsed_credential);
  if (!admitted_or.ok()) {
    return to_grpc_status(admitted_or.status());
  }
  const auto workflow_redemption_context = publication_workflow_redemption_context(record, *admitted_or);
  span->SetAttribute(
      "tc.front_door.forwardable_evidence_present",
      publish_context.front_door_context.forwardable_evidence.has_value());
  span->SetAttribute(
      "tc.front_door.local_observation_count",
      static_cast<int64_t>(publish_context.front_door_context.local_observations.observations.size()));
  if (publish_context.front_door_context.forwardable_evidence.has_value()) {
    span->SetAttribute(
        "tc.front_door.forwardable_evidence_kind",
        static_cast<int>(publish_context.front_door_context.forwardable_evidence->evidence_kind));
  }
  span->SetAttribute("tc.workflow.family", workflow_redemption_context.family);
  span->SetAttribute("tc.workflow.owner_kind", static_cast<int>(workflow_redemption_context.workflow_ref.owner_kind));
  span->SetAttribute("tc.workflow.recovery_class", static_cast<int>(record.workflow_recovery_class));
  auto scope_record_status = validate_target_publication_scope_against_record(
      record, publish_context.scope, publish_context.normalized_byte_space);
  if (!scope_record_status.ok()) {
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after scope reject: " << release_status;
    return to_grpc_status(scope_record_status);
  }
  auto binding_status = validate_binding_current_value_against_publication(d_.bindings, record, publish_context.scope);
  if (!binding_status.ok()) {
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after binding-current reject: "
        << release_status;
    return to_grpc_status(binding_status);
  }
  const auto subject_evaluation = evaluate_publication_subject(*target_publication_registry_, record, absl::Now());
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kOwnerLost ||
      subject_evaluation.match_result == PublicationSubjectMatchResult::kUnavailable) {
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after owner-loss reject: " << release_status;
    return {
        StatusCode::UNAVAILABLE,
        subject_evaluation.match_result == PublicationSubjectMatchResult::kOwnerLost
            ? "publish workflow owner lost"
            : "target_publication workflow state unavailable",
    };
  }
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kStaleCurrent) {
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after stale-current reject: "
        << release_status;
    const auto stale_current =
        publication_gate_decision(record, WorkflowDecisionClass::kStaleCurrent, "publish currentness rejected");
    span->SetAttribute("tc.workflow.decision_class", static_cast<int>(stale_current.decision_class));
    return {StatusCode::FAILED_PRECONDITION, "binding_current_value_publication_token is stale for target"};
  }

  resp.mutable_operation()->CopyFrom(operation_ref);
  operation::AcquireOperationLeaseRequest lease_request;
  lease_request.set_operation_id(operation_ref.operation_id());
  lease_request.set_kind(operation_ref.kind());
  lease_request.set_target_artifact_id(operation_ref.target_artifact_id());
  lease_request.set_owner_id(owner_id_for_operation(d_.identity));
  lease_request.set_ttl_ms(
      static_cast<uint64_t>(absl::ToInt64Milliseconds(binding_current_value_publication_token_ttl())));

  auto lease_or = d_.global_store_client->acquire_operation_lease(lease_request);
  if (!lease_or.ok()) {
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after acquire failure: " << release_status;
    return to_grpc_status(lease_or.status());
  }
  if (!lease_or->acquired()) {
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after attach_existing: " << release_status;
    rctx.mark_success();
    return Status::OK;
  }

  operation::UpdateOperationRequest update_request;
  update_request.set_operation_id(operation_ref.operation_id());
  update_request.set_lease_generation(lease_or->lease().lease_generation());
  *update_request.mutable_status() =
      build_publish_operation_status(operation::OPERATION_STATE_RUNNING, "publish_target_replica running");
  update_request.mutable_snapshot()->CopyFrom(build_publish_operation_snapshot(record));
  auto initial_update_status = d_.global_store_client->update_operation(update_request);
  if (!initial_update_status.ok()) {
    OperationLeaseGuard lease_guard(
        d_.global_store_client, lease_or->lease().lease_token(), operation_ref.operation_id());
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after initial update failure: "
        << release_status;
    return to_grpc_status(initial_update_status);
  }

  bool should_start = false;
  {
    absl::MutexLock lock(&publish_operation_tracker_->mu);
    should_start = publish_operation_tracker_->active_operations.insert(operation_ref.operation_id()).second;
  }
  if (!should_start) {
    auto release_status = d_.lifecycle_kernel.release_use_guard(admitted_or->use_guard);
    LOG_IF(WARNING, !release_status.ok())
        << "publish_target_replica: failed to release lifecycle use guard after duplicate local start: "
        << release_status;
    rctx.mark_success();
    return Status::OK;
  }

  auto client_sp = d_.global_store_client;
  const auto tracker = publish_operation_tracker_;
  const uint64_t lease_generation = lease_or->lease().lease_generation();
  const std::string lease_token = lease_or->lease().lease_token();
  const std::string operation_id = operation_ref.operation_id();
  const uint32_t ttl_ms = req.has_ttl_ms() ? req.ttl_ms() : 0U;
  const auto use_guard = admitted_or->use_guard;
  d_.async_runtime.blocking_executor()->add([this,
                                             client_sp = std::move(client_sp),
                                             tracker,
                                             publish_context = std::move(publish_context),
                                             operation_id,
                                             lease_token,
                                             lease_generation,
                                             ttl_ms,
                                             use_guard]() mutable {
    if (client_sp == nullptr) {
      return;
    }
    OperationLeaseGuard lease_guard(client_sp, lease_token, operation_id);
    auto cleanup = absl::MakeCleanup([this, tracker, operation_id, use_guard]() {
      {
        absl::MutexLock lock(&tracker->mu);
        tracker->active_operations.erase(operation_id);
      }
      auto release_status = d_.lifecycle_kernel.release_use_guard(use_guard);
      LOG_IF(WARNING, !release_status.ok())
          << "publish_target_replica: failed to release lifecycle use guard: " << release_status;
    });

    auto result_or = execute_publish_target_replica(
        d_, target_publication_registry_, std::move(publish_context), operation_id, ttl_ms);
    operation::UpdateOperationRequest final_update;
    final_update.set_operation_id(operation_id);
    final_update.set_lease_generation(lease_generation);
    if (result_or.ok()) {
      google::protobuf::Any result_any;
      result_any.PackFrom(*result_or);
      *final_update.mutable_status() = build_publish_operation_status(
          operation::OPERATION_STATE_SUCCESS, "publish_target_replica completed", absl::OkStatus(), &result_any);
    } else {
      *final_update.mutable_status() = build_publish_operation_status(
          operation::OPERATION_STATE_FAILED, result_or.status().message(), result_or.status());
    }
    auto final_status = client_sp->update_operation(final_update);
    LOG_IF(WARNING, !final_status.ok()) << "publish_target_replica: failed to update terminal operation state for op="
                                        << operation_id << ": " << final_status;
  });

  rctx.mark_success();
  return Status::OK;
}

absl::Status TargetPublishService::admit_public_operation(const operation::OperationRef& operation_ref, absl::Time now)
    const {
  if (operation_ref.kind() != kPublishOperationKind) {
    return absl::OkStatus();
  }
  if (operation_ref.authority_scope_kind() != kPublishAuthorityScopeKind) {
    return absl::FailedPreconditionError("publish operation authority_scope_kind mismatch");
  }
  if (operation_ref.authority_scope_id().empty()) {
    return absl::FailedPreconditionError("publish operation authority_scope_id is required");
  }
  if (operation_ref.attachment_kind() != kPublishAttachmentKind) {
    return absl::FailedPreconditionError("publish operation attachment_kind mismatch");
  }
  if (operation_ref.recovery_class() != kPublishRecoveryClass) {
    return absl::FailedPreconditionError("publish operation recovery_class mismatch");
  }
  auto record_opt =
      target_publication_registry_->lookup(operation_ref.authority_scope_id(), now, /*require_not_expired=*/true);
  if (!record_opt.has_value()) {
    return absl::UnavailableError("publish workflow owner lost");
  }
  auto record = std::move(*record_opt);
  populate_publication_workflow_state(&record);
  if (publish_operation_id(record) != operation_ref.operation_id()) {
    return absl::FailedPreconditionError("publish operation_id mismatch");
  }
  if (!operation_ref.target_artifact_id().empty() &&
      record.selection.artifact_id() != operation_ref.target_artifact_id()) {
    return absl::FailedPreconditionError("publish operation target_artifact_id mismatch");
  }
  const auto subject_evaluation = evaluate_publication_subject(*target_publication_registry_, record, now);
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kOwnerLost ||
      subject_evaluation.match_result == PublicationSubjectMatchResult::kUnavailable) {
    return absl::UnavailableError("publish workflow owner lost");
  }
  if (subject_evaluation.match_result == PublicationSubjectMatchResult::kStaleCurrent) {
    return absl::FailedPreconditionError("publish operation attachment is stale");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon
