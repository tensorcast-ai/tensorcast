// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/runtime/ingestion/artifact_truth.h"

namespace tensorcast::daemon {

enum class AuthorityKind : std::uint8_t;
struct AuthorityRef;
struct AuthorityAttachmentRef;

enum class LifecycleCapabilityFamily : std::uint8_t {
  kServe = 0,
  kExport = 1,
  kPlacement = 2,
  kRetention = 3,
  kPublish = 4,
  kInternal = 5,
};

enum class LifecycleFrontDoorKind : std::uint8_t {
  kPayloadRef = 0,
  kLocalCpuMemfdExport = 1,
  kLocalCudaIpcExport = 2,
  kPlacementLeaseEnvelope = 3,
  kPlacementLeaseLocalToken = 4,
  kRetentionHandleToken = 5,
  kTargetPublicationToken = 6,
  kInternalRecord = 7,
};

enum class CredentialCarriageKind : std::uint8_t {
  kSelfDescribing = 0,
  kOpaqueLocalCompat = 1,
};

enum class LifecycleBindingMode : std::uint8_t {
  kAddressDerived = 0,
  kBindingRecord = 1,
};

enum class LifecycleBindingSpace : std::uint8_t {
  kPayload = 0,
  kExportHandle = 1,
  kPlacementLease = 2,
  kRetentionHandle = 3,
  kPublication = 4,
  kInternalRecord = 5,
};

enum class BindingKeyKind : std::uint8_t {
  kPayloadId = 0,
  kLeaseId = 1,
  kHandleId = 2,
  kPublicationId = 3,
  kOpaqueLocalToken = 4,
  kInternalRecordId = 5,
};

enum class LifecycleSubjectKind : std::uint8_t {
  kBacking = 0,
  kPolicySource = 1,
  kInlineSnapshot = 2,
  kPlacementTarget = 3,
  kSelectionTarget = 4,
  kPublicationTarget = 5,
  kCommitTarget = 6,
};

enum class LifecycleRecordState : std::uint8_t {
  kMinting = 0,
  kActive = 1,
  kDraining = 2,
  kExpired = 3,
  kReleased = 4,
};

enum class LifecycleBindingState : std::uint8_t {
  kActive = 0,
  kExpired = 1,
  kRevoked = 2,
};

enum class WorkflowGateKind : std::uint8_t {
  kNone = 0,
  kRequired = 1,
};

enum class LifecycleRoutePrincipalKind : std::uint8_t {
  kIssuerDaemon = 0,
  kWorkflowOwner = 1,
  kInternalAuthority = 2,
};

enum class FencingPrincipalKind : std::uint8_t {
  kIssuerDaemon = 0,
  kWorkflowOwner = 1,
  kOperation = 2,
  kQueueLeader = 3,
  kBindingRecord = 4,
  kInternalAuthority = 5,
};

enum class WorkflowOwnerKind : std::uint8_t {
  kPublication = 0,
  kOperation = 1,
  kQueue = 2,
  kInternal = 3,
};

struct LifecycleRoutePrincipal {
  LifecycleRoutePrincipalKind principal_kind{LifecycleRoutePrincipalKind::kIssuerDaemon};
  std::string principal_id;

  bool operator==(const LifecycleRoutePrincipal&) const = default;
};

struct FencingContext {
  FencingPrincipalKind principal_kind{FencingPrincipalKind::kIssuerDaemon};
  std::string principal_id;
  std::uint64_t epoch{0};

  bool operator==(const FencingContext&) const = default;
};

struct WorkflowCompanionRef {
  WorkflowOwnerKind owner_kind{WorkflowOwnerKind::kInternal};
  std::string workflow_id;
  std::optional<std::string> currentness_key;
  std::optional<std::string> operation_id;
  std::optional<FencingContext> fencing_context;

  bool operator==(const WorkflowCompanionRef&) const = default;
};

enum class WorkflowRecoveryClass : std::uint8_t {
  kEphemeralProcessLocal = 0,
  kLocalRecoverable = 1,
  kReplicatedAuthority = 2,
};

enum class WorkflowDecisionClass : std::uint8_t {
  kAdmit = 0,
  kReplay = 1,
  kStaleCurrent = 2,
  kFenced = 3,
  kCancelled = 4,
  kFailedPrecondition = 5,
};

enum class WorkflowObservationKind : std::uint8_t {
  kReplayLookup = 0,
  kJoinLookup = 1,
  kStatus = 2,
  kWait = 3,
  kCurrentness = 4,
};

enum class WorkflowOutcomeProjectionKind : std::uint8_t {
  kExistingCapability = 0,
  kStatusSnapshot = 1,
  kCurrentWinnerHint = 2,
  kFamilyDefined = 3,
};

enum class WorkflowCompletionClass : std::uint8_t {
  kCompleted = 0,
  kFailed = 1,
  kCancelled = 2,
  kReleasedWithoutRealization = 3,
};

struct WorkflowIssueContext {
  std::string family;
  std::optional<std::string> adapter_kind;
  std::optional<WorkflowCompanionRef> requested_workflow_ref;
  std::optional<std::string> currentness_key;
  std::optional<std::string> request_operation_id;
  std::optional<FencingContext> requested_fencing_context;

  bool operator==(const WorkflowIssueContext&) const = default;
};

struct WorkflowBindingProjection {
  WorkflowCompanionRef resolved_workflow_ref;

  bool operator==(const WorkflowBindingProjection&) const = default;
};

struct WorkflowRedemptionContext {
  std::string family;
  std::optional<std::string> adapter_kind;
  WorkflowCompanionRef workflow_ref;
  std::optional<std::string> capability_id;
  std::optional<std::string> subject_id;
  std::optional<std::string> binding_id;
  std::optional<FencingContext> lifecycle_fencing_context;
  std::optional<std::string> request_operation_id;

  bool operator==(const WorkflowRedemptionContext&) const = default;
};

struct WorkflowOutcomeProjection {
  WorkflowOutcomeProjectionKind projection_kind{WorkflowOutcomeProjectionKind::kFamilyDefined};
  std::string owner_workflow_id;
  std::shared_ptr<AuthorityAttachmentRef> attachment_ref;
  std::optional<std::string> existing_capability_id;
  std::optional<std::string> status_summary;
  std::optional<std::string> current_winner_workflow_id;
  std::optional<std::string> family_payload;

  bool operator==(const WorkflowOutcomeProjection&) const = default;
};

struct WorkflowGateDecision {
  WorkflowDecisionClass decision_class{WorkflowDecisionClass::kFailedPrecondition};
  WorkflowCompanionRef resolved_workflow_ref;
  std::optional<WorkflowBindingProjection> binding_projection;
  std::optional<WorkflowOutcomeProjection> outcome_projection;
  std::optional<std::string> diagnostics;

  bool operator==(const WorkflowGateDecision&) const = default;
};

struct WorkflowObservationQuery {
  WorkflowCompanionRef workflow_ref;
  WorkflowObservationKind observation_kind{WorkflowObservationKind::kStatus};
  std::optional<std::string> adapter_kind;
  std::optional<std::string> capability_id;
  std::optional<std::string> subject_id;
  std::optional<std::string> binding_id;
  std::optional<absl::Time> wait_deadline;
  std::optional<std::string> request_operation_id;

  bool operator==(const WorkflowObservationQuery&) const = default;
};

struct WorkflowObservationResult {
  WorkflowObservationKind observation_kind{WorkflowObservationKind::kStatus};
  WorkflowCompanionRef resolved_workflow_ref;
  std::optional<WorkflowOutcomeProjection> outcome_projection;
  std::optional<bool> ready;
  std::optional<std::string> diagnostics;

  bool operator==(const WorkflowObservationResult&) const = default;
};

struct WorkflowCompletionContext {
  std::string family;
  std::optional<std::string> adapter_kind;
  WorkflowCompanionRef workflow_ref;
  WorkflowCompletionClass completion_class{WorkflowCompletionClass::kCompleted};
  std::optional<std::string> capability_id;
  std::optional<std::string> subject_id;
  std::optional<std::string> completion_details;

  bool operator==(const WorkflowCompletionContext&) const = default;
};

[[nodiscard]] inline LifecycleRoutePrincipal make_issuer_route_principal(std::string_view issuer_daemon_id) {
  return LifecycleRoutePrincipal{
      .principal_kind = LifecycleRoutePrincipalKind::kIssuerDaemon,
      .principal_id = std::string(issuer_daemon_id),
  };
}

struct LifecycleEpochs {
  std::uint64_t subject_generation{1};
  std::optional<FencingContext> fencing_context;

  bool operator==(const LifecycleEpochs&) const = default;
};

struct ConstraintClaims {
  std::string artifact_id;
  std::string digest_alg;
  std::string digest_hex;
  std::string direction;
  std::string operation_id;
  std::string holder_scope;
  bool local_only{false};

  bool operator==(const ConstraintClaims&) const = default;
};

struct LifecycleSubjectRecord {
  std::string subject_id;
  LifecycleEpochs epochs;
  LifecycleSubjectKind subject_kind{LifecycleSubjectKind::kBacking};
  absl::Time created_at{absl::InfinitePast()};
  absl::Time last_observed_at{absl::InfinitePast()};
  std::optional<std::string> artifact_id;
  std::optional<store::runtime::ingestion::VerifiedContentDescriptor> verified_content_descriptor;
  std::optional<std::string> semantic_ref_id;
  std::optional<WorkflowCompanionRef> workflow_companion;
  bool caller_visible{true};
};

struct LifecycleCapabilityRecord {
  std::string capability_id;
  LifecycleCapabilityFamily family{LifecycleCapabilityFamily::kInternal};
  LifecycleFrontDoorKind front_door_kind{LifecycleFrontDoorKind::kInternalRecord};
  std::string subject_id;
  LifecycleEpochs epochs;
  std::uint64_t lease_id{0};
  std::optional<std::uint64_t> lease_generation_or_equivalent;
  std::string issuer_daemon_id;
  absl::Time issued_at{absl::InfinitePast()};
  absl::Time capability_expires_at{absl::InfinitePast()};
  LifecycleRecordState state{LifecycleRecordState::kMinting};
  std::optional<WorkflowCompanionRef> workflow_companion;
  std::optional<std::string> holder_scope;
  std::optional<std::string> resolution_hints;
  std::optional<std::string> direction;
  bool local_only{false};
  WorkflowGateKind workflow_gate{WorkflowGateKind::kNone};
  ConstraintClaims constraint_claims;
};

struct CapabilityBindingAddress {
  LifecycleRoutePrincipal route_principal;
  LifecycleCapabilityFamily family{LifecycleCapabilityFamily::kInternal};
  LifecycleBindingSpace binding_space{LifecycleBindingSpace::kInternalRecord};
  BindingKeyKind binding_key_kind{BindingKeyKind::kInternalRecordId};
  std::string binding_key;
  LifecycleEpochs epochs;
  std::optional<std::string> binding_id;

  bool operator==(const CapabilityBindingAddress&) const = default;
};

struct LifecycleBindingRecord {
  std::string binding_id;
  std::string capability_id;
  CapabilityBindingAddress address;
  absl::Time issued_at{absl::InfinitePast()};
  absl::Time credential_expires_at{absl::InfinitePast()};
  LifecycleBindingState state{LifecycleBindingState::kActive};
  std::optional<FencingContext> binding_fencing_context;
};

struct ParsedCredential {
  CapabilityBindingAddress address;
  LifecycleFrontDoorKind front_door_kind{LifecycleFrontDoorKind::kInternalRecord};
  absl::Time credential_expires_at{absl::InfinitePast()};
  CredentialCarriageKind carriage_kind{CredentialCarriageKind::kSelfDescribing};
  LifecycleBindingMode binding_mode{LifecycleBindingMode::kAddressDerived};
  ConstraintClaims constraint_claims;

  bool operator==(const ParsedCredential&) const = default;
};

enum class CredentialEvidenceKind : std::uint8_t {
  kRawCredential = 0,
  kIssuerVerifiableProjection = 1,
  kRelayAttestedProjection = 2,
};

enum class LocalObservationRoutingAction : std::uint8_t {
  kConsume = 0,
  kReject = 1,
  kTranslateToForwardedClaim = 2,
};

struct CanonicalCredentialProjection {
  std::string projection_kind;
  std::string projection_version;
  std::string projection_bytes;
  std::string projection_digest;
  std::string issuer_binding;
  std::optional<std::string> projection_authenticator;

  bool operator==(const CanonicalCredentialProjection&) const = default;
};

struct ForwardableCredentialEvidence {
  CredentialEvidenceKind evidence_kind{CredentialEvidenceKind::kRawCredential};
  std::optional<std::string> raw_credential_bytes;
  std::optional<CanonicalCredentialProjection> canonical_projection;

  bool operator==(const ForwardableCredentialEvidence&) const = default;
};

struct LocalObservation {
  std::string observation_kind;
  std::string observation_payload;

  bool operator==(const LocalObservation&) const = default;
};

struct LocalObservationSet {
  std::vector<LocalObservation> observations;

  [[nodiscard]] bool empty() const {
    return observations.empty();
  }

  bool operator==(const LocalObservationSet&) const = default;
};

struct LocalObservationRoutingRule {
  std::string observation_kind;
  LocalObservationRoutingAction action{LocalObservationRoutingAction::kReject};
  std::optional<std::string> forwarded_claim_kind;
  std::optional<std::string> forwarded_claim_payload;

  bool operator==(const LocalObservationRoutingRule&) const = default;
};

struct FrontDoorCredentialContext {
  ParsedCredential parsed_credential;
  std::optional<ForwardableCredentialEvidence> forwardable_evidence;
  LocalObservationSet local_observations;

  bool operator==(const FrontDoorCredentialContext&) const = default;
};

struct PortableParsedCredential {
  CapabilityBindingAddress address;
  LifecycleFrontDoorKind front_door_kind{LifecycleFrontDoorKind::kInternalRecord};
  absl::Time credential_expires_at{absl::InfinitePast()};
  LifecycleBindingMode binding_mode{LifecycleBindingMode::kAddressDerived};
  ConstraintClaims portable_constraint_claims;

  bool operator==(const PortableParsedCredential&) const = default;
};

struct BindingResolution {
  LifecycleCapabilityRecord capability;
  LifecycleSubjectRecord subject;
  std::optional<LifecycleBindingRecord> binding_record;
};

struct LifecycleUseGuard {
  std::string guard_id;
  std::string capability_id;
};

struct AdmittedCapabilityUse {
  LifecycleCapabilityRecord capability;
  LifecycleSubjectRecord subject;
  std::optional<LifecycleBindingRecord> binding_record;
  LifecycleUseGuard use_guard;
};

struct MintCapabilityRequest {
  LifecycleSubjectRecord subject;
  CapabilityBindingAddress address;
  LifecycleFrontDoorKind front_door_kind{LifecycleFrontDoorKind::kInternalRecord};
  std::string capability_id;
  std::uint64_t lease_id{0};
  std::optional<std::uint64_t> lease_generation_or_equivalent;
  absl::Time capability_expires_at{absl::InfinitePast()};
  CredentialCarriageKind carriage_kind{CredentialCarriageKind::kSelfDescribing};
  LifecycleBindingMode binding_mode{LifecycleBindingMode::kAddressDerived};
  ConstraintClaims constraint_claims;
  std::optional<absl::Time> credential_expires_at;
  std::optional<std::string> binding_id;
  std::optional<WorkflowCompanionRef> workflow_companion;
  std::optional<std::string> holder_scope;
  std::optional<std::string> resolution_hints;
  std::optional<std::string> direction;
  bool local_only{false};
  WorkflowGateKind workflow_gate{WorkflowGateKind::kNone};
  std::optional<FencingContext> binding_fencing_context;
};

struct RenewCapabilityRequest {
  std::string capability_id;
  absl::Time capability_expires_at{absl::InfinitePast()};
  std::optional<absl::Time> credential_expires_at;
  std::optional<std::string> binding_id;
};

class LifecycleKernel {
 public:
  explicit LifecycleKernel(std::string issuer_daemon_id);

  [[nodiscard]] const std::string& issuer_daemon_id() const {
    return issuer_daemon_id_;
  }

  [[nodiscard]] absl::StatusOr<LifecycleCapabilityRecord> mint_capability(const MintCapabilityRequest& request)
      ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::StatusOr<LifecycleCapabilityRecord> renew_capability(const RenewCapabilityRequest& request)
      ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::StatusOr<AdmittedCapabilityUse> admit_redemption(const ParsedCredential& credential)
      ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::Status release_use_guard(const LifecycleUseGuard& use_guard) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::Status revoke_binding_record(std::string_view binding_id) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::Status release_capability(std::string_view capability_id) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::StatusOr<LifecycleCapabilityRecord> inspect_capability(std::string_view capability_id) const
      ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::StatusOr<LifecycleBindingRecord> inspect_binding(std::string_view binding_id) const
      ABSL_LOCKS_EXCLUDED(mu_);

 private:
  struct CapabilityEntry {
    LifecycleCapabilityRecord capability;
    LifecycleSubjectRecord subject;
    CapabilityBindingAddress address;
    CredentialCarriageKind carriage_kind{CredentialCarriageKind::kSelfDescribing};
    LifecycleBindingMode binding_mode{LifecycleBindingMode::kAddressDerived};
    std::size_t active_use_count{0};
    std::optional<std::string> binding_id;
  };

  [[nodiscard]] static std::string address_index_key(const CapabilityBindingAddress& address);

  [[nodiscard]] absl::Status validate_mint_request_locked_(const MintCapabilityRequest& request) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  [[nodiscard]] absl::Status validate_constraint_claims_(
      const ConstraintClaims& expected,
      const ConstraintClaims& observed) const;

  [[nodiscard]] absl::StatusOr<CapabilityEntry*> resolve_entry_for_credential_locked_(
      const ParsedCredential& credential) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void maybe_expire_entry_locked_(CapabilityEntry& entry, absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void erase_entry_locked_(std::string_view capability_id) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  std::string issuer_daemon_id_;

  mutable absl::Mutex mu_;
  std::uint64_t next_guard_id_ ABSL_GUARDED_BY(mu_){1};
  absl::flat_hash_map<std::string, CapabilityEntry> capabilities_by_id_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::string> address_to_capability_id_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, LifecycleBindingRecord> bindings_by_id_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, std::string> use_guard_to_capability_id_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
