// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "daemon/state/routed_authority_protocol.h"
#include "google/protobuf/timestamp.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::routed_authority_wire {

namespace detail {

inline void populate_proto_timestamp(absl::Time time, google::protobuf::Timestamp* proto_timestamp) {
  proto_timestamp->Clear();
  const std::int64_t seconds = absl::ToUnixSeconds(time);
  proto_timestamp->set_seconds(seconds);
  proto_timestamp->set_nanos(
      static_cast<std::int32_t>(absl::ToInt64Nanoseconds(time - absl::FromUnixSeconds(seconds))));
}

inline absl::StatusOr<absl::Time> timestamp_from_proto(const google::protobuf::Timestamp& proto_timestamp) {
  if (proto_timestamp.nanos() < 0 || proto_timestamp.nanos() >= 1000000000) {
    return absl::InvalidArgumentError("timestamp nanos is out of range");
  }
  return absl::FromUnixSeconds(proto_timestamp.seconds()) + absl::Nanoseconds(proto_timestamp.nanos());
}

inline std::string lifecycle_front_door_kind_label(LifecycleFrontDoorKind front_door_kind) {
  switch (front_door_kind) {
    case LifecycleFrontDoorKind::kPayloadRef:
      return "payload_ref";
    case LifecycleFrontDoorKind::kLocalCpuMemfdExport:
      return "local_cpu_memfd_export";
    case LifecycleFrontDoorKind::kLocalCudaIpcExport:
      return "local_cuda_ipc_export";
    case LifecycleFrontDoorKind::kPlacementLeaseEnvelope:
      return "placement_lease_envelope";
    case LifecycleFrontDoorKind::kPlacementLeaseLocalToken:
      return "placement_lease_local_token";
    case LifecycleFrontDoorKind::kRetentionHandleToken:
      return "retention_handle_token";
    case LifecycleFrontDoorKind::kTargetPublicationToken:
      return "binding_current_value_publication_token";
    case LifecycleFrontDoorKind::kInternalRecord:
      return "internal_record";
  }
  return "internal_record";
}

inline absl::StatusOr<LifecycleFrontDoorKind> lifecycle_front_door_kind_from_label(std::string_view label) {
  if (label == "payload_ref") {
    return LifecycleFrontDoorKind::kPayloadRef;
  }
  if (label == "local_cpu_memfd_export") {
    return LifecycleFrontDoorKind::kLocalCpuMemfdExport;
  }
  if (label == "local_cuda_ipc_export") {
    return LifecycleFrontDoorKind::kLocalCudaIpcExport;
  }
  if (label == "placement_lease_envelope") {
    return LifecycleFrontDoorKind::kPlacementLeaseEnvelope;
  }
  if (label == "placement_lease_local_token") {
    return LifecycleFrontDoorKind::kPlacementLeaseLocalToken;
  }
  if (label == "retention_handle_token") {
    return LifecycleFrontDoorKind::kRetentionHandleToken;
  }
  if (label == "binding_current_value_publication_token" || label == "target_publication_token") {
    return LifecycleFrontDoorKind::kTargetPublicationToken;
  }
  if (label == "internal_record") {
    return LifecycleFrontDoorKind::kInternalRecord;
  }
  return absl::InvalidArgumentError("unsupported front_door_kind label");
}

inline std::string lifecycle_binding_mode_label(LifecycleBindingMode binding_mode) {
  switch (binding_mode) {
    case LifecycleBindingMode::kAddressDerived:
      return "address_derived";
    case LifecycleBindingMode::kBindingRecord:
      return "binding_record";
  }
  return "address_derived";
}

inline absl::StatusOr<LifecycleBindingMode> lifecycle_binding_mode_from_label(std::string_view label) {
  if (label == "address_derived") {
    return LifecycleBindingMode::kAddressDerived;
  }
  if (label == "binding_record") {
    return LifecycleBindingMode::kBindingRecord;
  }
  return absl::InvalidArgumentError("unsupported binding_mode label");
}

inline std::string route_principal_kind_label(LifecycleRoutePrincipalKind principal_kind) {
  switch (principal_kind) {
    case LifecycleRoutePrincipalKind::kIssuerDaemon:
      return "issuer_daemon";
    case LifecycleRoutePrincipalKind::kWorkflowOwner:
      return "workflow_owner";
    case LifecycleRoutePrincipalKind::kInternalAuthority:
      return "internal_authority";
  }
  return "internal_authority";
}

inline absl::StatusOr<LifecycleRoutePrincipalKind> route_principal_kind_from_label(std::string_view label) {
  if (label == "issuer_daemon") {
    return LifecycleRoutePrincipalKind::kIssuerDaemon;
  }
  if (label == "workflow_owner") {
    return LifecycleRoutePrincipalKind::kWorkflowOwner;
  }
  if (label == "internal_authority") {
    return LifecycleRoutePrincipalKind::kInternalAuthority;
  }
  return absl::InvalidArgumentError("unsupported route_principal_kind label");
}

inline std::string capability_family_label(LifecycleCapabilityFamily family) {
  switch (family) {
    case LifecycleCapabilityFamily::kServe:
      return "serve";
    case LifecycleCapabilityFamily::kExport:
      return "export";
    case LifecycleCapabilityFamily::kPlacement:
      return "placement";
    case LifecycleCapabilityFamily::kRetention:
      return "retention";
    case LifecycleCapabilityFamily::kPublish:
      return "publish";
    case LifecycleCapabilityFamily::kInternal:
      return "internal";
  }
  return "internal";
}

inline absl::StatusOr<LifecycleCapabilityFamily> capability_family_from_label(std::string_view label) {
  if (label == "serve") {
    return LifecycleCapabilityFamily::kServe;
  }
  if (label == "export") {
    return LifecycleCapabilityFamily::kExport;
  }
  if (label == "placement") {
    return LifecycleCapabilityFamily::kPlacement;
  }
  if (label == "retention") {
    return LifecycleCapabilityFamily::kRetention;
  }
  if (label == "publish") {
    return LifecycleCapabilityFamily::kPublish;
  }
  if (label == "internal") {
    return LifecycleCapabilityFamily::kInternal;
  }
  return absl::InvalidArgumentError("unsupported family label");
}

inline std::string binding_space_label(LifecycleBindingSpace binding_space) {
  switch (binding_space) {
    case LifecycleBindingSpace::kPayload:
      return "payload";
    case LifecycleBindingSpace::kExportHandle:
      return "export_handle";
    case LifecycleBindingSpace::kPlacementLease:
      return "placement_lease";
    case LifecycleBindingSpace::kRetentionHandle:
      return "retention_handle";
    case LifecycleBindingSpace::kPublication:
      return "publication";
    case LifecycleBindingSpace::kInternalRecord:
      return "internal_record";
  }
  return "internal_record";
}

inline absl::StatusOr<LifecycleBindingSpace> binding_space_from_label(std::string_view label) {
  if (label == "payload") {
    return LifecycleBindingSpace::kPayload;
  }
  if (label == "export_handle") {
    return LifecycleBindingSpace::kExportHandle;
  }
  if (label == "placement_lease") {
    return LifecycleBindingSpace::kPlacementLease;
  }
  if (label == "retention_handle") {
    return LifecycleBindingSpace::kRetentionHandle;
  }
  if (label == "publication") {
    return LifecycleBindingSpace::kPublication;
  }
  if (label == "internal_record") {
    return LifecycleBindingSpace::kInternalRecord;
  }
  return absl::InvalidArgumentError("unsupported binding_space label");
}

inline std::string binding_key_kind_label(BindingKeyKind binding_key_kind) {
  switch (binding_key_kind) {
    case BindingKeyKind::kPayloadId:
      return "payload_id";
    case BindingKeyKind::kLeaseId:
      return "lease_id";
    case BindingKeyKind::kHandleId:
      return "handle_id";
    case BindingKeyKind::kPublicationId:
      return "publication_id";
    case BindingKeyKind::kOpaqueLocalToken:
      return "opaque_local_token";
    case BindingKeyKind::kInternalRecordId:
      return "internal_record_id";
  }
  return "internal_record_id";
}

inline absl::StatusOr<BindingKeyKind> binding_key_kind_from_label(std::string_view label) {
  if (label == "payload_id") {
    return BindingKeyKind::kPayloadId;
  }
  if (label == "lease_id") {
    return BindingKeyKind::kLeaseId;
  }
  if (label == "handle_id") {
    return BindingKeyKind::kHandleId;
  }
  if (label == "publication_id") {
    return BindingKeyKind::kPublicationId;
  }
  if (label == "opaque_local_token") {
    return BindingKeyKind::kOpaqueLocalToken;
  }
  if (label == "internal_record_id") {
    return BindingKeyKind::kInternalRecordId;
  }
  return absl::InvalidArgumentError("unsupported binding_key_kind label");
}

inline v2::RoutedAuthorityKind authority_kind_to_proto(AuthorityKind authority_kind) {
  switch (authority_kind) {
    case AuthorityKind::kCatalogOwner:
      return v2::ROUTED_AUTHORITY_KIND_CATALOG_OWNER;
    case AuthorityKind::kShardHomeDaemon:
      return v2::ROUTED_AUTHORITY_KIND_SHARD_HOME_DAEMON;
    case AuthorityKind::kWorkflowOwner:
      return v2::ROUTED_AUTHORITY_KIND_WORKFLOW_OWNER;
    case AuthorityKind::kOperationOwner:
      return v2::ROUTED_AUTHORITY_KIND_OPERATION_OWNER;
    case AuthorityKind::kIssuerDaemon:
      return v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON;
    case AuthorityKind::kInternalAuthority:
      return v2::ROUTED_AUTHORITY_KIND_INTERNAL_AUTHORITY;
  }
  return v2::ROUTED_AUTHORITY_KIND_UNSPECIFIED;
}

inline AuthorityKind authority_kind_from_proto(v2::RoutedAuthorityKind authority_kind) {
  switch (authority_kind) {
    case v2::ROUTED_AUTHORITY_KIND_CATALOG_OWNER:
      return AuthorityKind::kCatalogOwner;
    case v2::ROUTED_AUTHORITY_KIND_SHARD_HOME_DAEMON:
      return AuthorityKind::kShardHomeDaemon;
    case v2::ROUTED_AUTHORITY_KIND_WORKFLOW_OWNER:
      return AuthorityKind::kWorkflowOwner;
    case v2::ROUTED_AUTHORITY_KIND_OPERATION_OWNER:
      return AuthorityKind::kOperationOwner;
    case v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON:
      return AuthorityKind::kIssuerDaemon;
    case v2::ROUTED_AUTHORITY_KIND_UNSPECIFIED:
    case v2::ROUTED_AUTHORITY_KIND_INTERNAL_AUTHORITY:
    case v2::RoutedAuthorityKind_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedAuthorityKind_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return AuthorityKind::kInternalAuthority;
  }
}

inline v2::RoutedFencingPrincipalKind fencing_principal_kind_to_proto(FencingPrincipalKind principal_kind) {
  switch (principal_kind) {
    case FencingPrincipalKind::kIssuerDaemon:
      return v2::ROUTED_FENCING_PRINCIPAL_KIND_ISSUER_DAEMON;
    case FencingPrincipalKind::kWorkflowOwner:
      return v2::ROUTED_FENCING_PRINCIPAL_KIND_WORKFLOW_OWNER;
    case FencingPrincipalKind::kOperation:
      return v2::ROUTED_FENCING_PRINCIPAL_KIND_OPERATION;
    case FencingPrincipalKind::kQueueLeader:
      return v2::ROUTED_FENCING_PRINCIPAL_KIND_QUEUE_LEADER;
    case FencingPrincipalKind::kBindingRecord:
      return v2::ROUTED_FENCING_PRINCIPAL_KIND_BINDING_RECORD;
    case FencingPrincipalKind::kInternalAuthority:
      return v2::ROUTED_FENCING_PRINCIPAL_KIND_INTERNAL_AUTHORITY;
  }
  return v2::ROUTED_FENCING_PRINCIPAL_KIND_UNSPECIFIED;
}

inline FencingPrincipalKind fencing_principal_kind_from_proto(v2::RoutedFencingPrincipalKind principal_kind) {
  switch (principal_kind) {
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_ISSUER_DAEMON:
      return FencingPrincipalKind::kIssuerDaemon;
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_WORKFLOW_OWNER:
      return FencingPrincipalKind::kWorkflowOwner;
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_OPERATION:
      return FencingPrincipalKind::kOperation;
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_QUEUE_LEADER:
      return FencingPrincipalKind::kQueueLeader;
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_BINDING_RECORD:
      return FencingPrincipalKind::kBindingRecord;
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_UNSPECIFIED:
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_INTERNAL_AUTHORITY:
    case v2::RoutedFencingPrincipalKind_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedFencingPrincipalKind_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return FencingPrincipalKind::kInternalAuthority;
  }
}

inline v2::RoutedDaemonHopAuthClass daemon_hop_auth_class_to_proto(DaemonHopAuthClass auth_class) {
  switch (auth_class) {
    case DaemonHopAuthClass::kLegacyUnauthenticated:
      return v2::ROUTED_DAEMON_HOP_AUTH_CLASS_LEGACY_UNAUTHENTICATED;
    case DaemonHopAuthClass::kDeploymentTrustedChannel:
      return v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DEPLOYMENT_TRUSTED_CHANNEL;
    case DaemonHopAuthClass::kDaemonMutualAuth:
      return v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DAEMON_MUTUAL_AUTH;
  }
  return v2::ROUTED_DAEMON_HOP_AUTH_CLASS_UNSPECIFIED;
}

inline DaemonHopAuthClass daemon_hop_auth_class_from_proto(v2::RoutedDaemonHopAuthClass auth_class) {
  switch (auth_class) {
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DEPLOYMENT_TRUSTED_CHANNEL:
      return DaemonHopAuthClass::kDeploymentTrustedChannel;
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DAEMON_MUTUAL_AUTH:
      return DaemonHopAuthClass::kDaemonMutualAuth;
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_UNSPECIFIED:
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_LEGACY_UNAUTHENTICATED:
    case v2::RoutedDaemonHopAuthClass_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedDaemonHopAuthClass_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return DaemonHopAuthClass::kLegacyUnauthenticated;
  }
}

inline v2::RoutedForwardedClaimProvenance forwarded_claim_provenance_to_proto(ForwardedClaimProvenance provenance) {
  switch (provenance) {
    case ForwardedClaimProvenance::kAuthorityAuthenticated:
      return v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_AUTHORITY_AUTHENTICATED;
    case ForwardedClaimProvenance::kRelayAuthenticated:
      return v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_RELAY_AUTHENTICATED;
    case ForwardedClaimProvenance::kIngressLocal:
      return v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_INGRESS_LOCAL;
  }
  return v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_UNSPECIFIED;
}

inline ForwardedClaimProvenance forwarded_claim_provenance_from_proto(v2::RoutedForwardedClaimProvenance provenance) {
  switch (provenance) {
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_AUTHORITY_AUTHENTICATED:
      return ForwardedClaimProvenance::kAuthorityAuthenticated;
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_RELAY_AUTHENTICATED:
      return ForwardedClaimProvenance::kRelayAuthenticated;
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_UNSPECIFIED:
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_INGRESS_LOCAL:
    case v2::RoutedForwardedClaimProvenance_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedForwardedClaimProvenance_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return ForwardedClaimProvenance::kIngressLocal;
  }
}

inline v2::RoutedDelegationPayloadKind delegation_payload_kind_to_proto(DelegationPayloadKind payload_kind) {
  switch (payload_kind) {
    case DelegationPayloadKind::kPortableCredential:
      return v2::ROUTED_DELEGATION_PAYLOAD_KIND_PORTABLE_CREDENTIAL;
    case DelegationPayloadKind::kForwardableEvidence:
      return v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDABLE_EVIDENCE;
    case DelegationPayloadKind::kForwardedClaim:
      return v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDED_CLAIM;
  }
  return v2::ROUTED_DELEGATION_PAYLOAD_KIND_UNSPECIFIED;
}

inline DelegationPayloadKind delegation_payload_kind_from_proto(v2::RoutedDelegationPayloadKind payload_kind) {
  switch (payload_kind) {
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDABLE_EVIDENCE:
      return DelegationPayloadKind::kForwardableEvidence;
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDED_CLAIM:
      return DelegationPayloadKind::kForwardedClaim;
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_UNSPECIFIED:
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_PORTABLE_CREDENTIAL:
    case v2::RoutedDelegationPayloadKind_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedDelegationPayloadKind_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return DelegationPayloadKind::kPortableCredential;
  }
}

inline v2::RoutedDelegationClass delegation_class_to_proto(DelegationClass delegation_class) {
  switch (delegation_class) {
    case DelegationClass::kBootstrapSafe:
      return v2::ROUTED_DELEGATION_CLASS_BOOTSTRAP_SAFE;
    case DelegationClass::kOwnerScopedSensitive:
      return v2::ROUTED_DELEGATION_CLASS_OWNER_SCOPED_SENSITIVE;
    case DelegationClass::kIssuerSecret:
      return v2::ROUTED_DELEGATION_CLASS_ISSUER_SECRET;
  }
  return v2::ROUTED_DELEGATION_CLASS_UNSPECIFIED;
}

inline DelegationClass delegation_class_from_proto(v2::RoutedDelegationClass delegation_class) {
  switch (delegation_class) {
    case v2::ROUTED_DELEGATION_CLASS_OWNER_SCOPED_SENSITIVE:
      return DelegationClass::kOwnerScopedSensitive;
    case v2::ROUTED_DELEGATION_CLASS_ISSUER_SECRET:
      return DelegationClass::kIssuerSecret;
    case v2::ROUTED_DELEGATION_CLASS_UNSPECIFIED:
    case v2::ROUTED_DELEGATION_CLASS_BOOTSTRAP_SAFE:
    case v2::RoutedDelegationClass_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedDelegationClass_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return DelegationClass::kBootstrapSafe;
  }
}

inline v2::RoutedCredentialEvidenceKind credential_evidence_kind_to_proto(CredentialEvidenceKind evidence_kind) {
  switch (evidence_kind) {
    case CredentialEvidenceKind::kRawCredential:
      return v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_RAW_CREDENTIAL;
    case CredentialEvidenceKind::kIssuerVerifiableProjection:
      return v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_ISSUER_VERIFIABLE_PROJECTION;
    case CredentialEvidenceKind::kRelayAttestedProjection:
      return v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_RELAY_ATTESTED_PROJECTION;
  }
  return v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_UNSPECIFIED;
}

inline CredentialEvidenceKind credential_evidence_kind_from_proto(v2::RoutedCredentialEvidenceKind evidence_kind) {
  switch (evidence_kind) {
    case v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_ISSUER_VERIFIABLE_PROJECTION:
      return CredentialEvidenceKind::kIssuerVerifiableProjection;
    case v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_RELAY_ATTESTED_PROJECTION:
      return CredentialEvidenceKind::kRelayAttestedProjection;
    case v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_UNSPECIFIED:
    case v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_RAW_CREDENTIAL:
    case v2::RoutedCredentialEvidenceKind_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedCredentialEvidenceKind_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return CredentialEvidenceKind::kRawCredential;
  }
}

inline v2::RoutedOwnerStageReplyKind owner_stage_reply_kind_to_proto(OwnerStageReplyKind reply_kind) {
  switch (reply_kind) {
    case OwnerStageReplyKind::kReadyForLowering:
      return v2::ROUTED_OWNER_STAGE_REPLY_KIND_READY_FOR_LOWERING;
    case OwnerStageReplyKind::kContinueWithAuthority:
      return v2::ROUTED_OWNER_STAGE_REPLY_KIND_CONTINUE_WITH_AUTHORITY;
    case OwnerStageReplyKind::kRetryLater:
      return v2::ROUTED_OWNER_STAGE_REPLY_KIND_RETRY_LATER;
    case OwnerStageReplyKind::kAttachExisting:
      return v2::ROUTED_OWNER_STAGE_REPLY_KIND_ATTACH_EXISTING;
    case OwnerStageReplyKind::kTerminal:
      return v2::ROUTED_OWNER_STAGE_REPLY_KIND_TERMINAL;
  }
  return v2::ROUTED_OWNER_STAGE_REPLY_KIND_UNSPECIFIED;
}

inline absl::StatusOr<OwnerStageReplyKind> owner_stage_reply_kind_from_proto(v2::RoutedOwnerStageReplyKind reply_kind) {
  switch (reply_kind) {
    case v2::ROUTED_OWNER_STAGE_REPLY_KIND_READY_FOR_LOWERING:
      return OwnerStageReplyKind::kReadyForLowering;
    case v2::ROUTED_OWNER_STAGE_REPLY_KIND_CONTINUE_WITH_AUTHORITY:
      return OwnerStageReplyKind::kContinueWithAuthority;
    case v2::ROUTED_OWNER_STAGE_REPLY_KIND_RETRY_LATER:
      return OwnerStageReplyKind::kRetryLater;
    case v2::ROUTED_OWNER_STAGE_REPLY_KIND_ATTACH_EXISTING:
      return OwnerStageReplyKind::kAttachExisting;
    case v2::ROUTED_OWNER_STAGE_REPLY_KIND_TERMINAL:
      return OwnerStageReplyKind::kTerminal;
    case v2::ROUTED_OWNER_STAGE_REPLY_KIND_UNSPECIFIED:
    case v2::RoutedOwnerStageReplyKind_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedOwnerStageReplyKind_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return absl::InvalidArgumentError("owner_stage_reply.reply_kind is unspecified");
  }
}

inline v2::RoutedTerminalProjectionKind terminal_projection_kind_to_proto(TerminalProjectionKind projection_kind) {
  switch (projection_kind) {
    case TerminalProjectionKind::kSemanticSuccess:
      return v2::ROUTED_TERMINAL_PROJECTION_KIND_SEMANTIC_SUCCESS;
    case TerminalProjectionKind::kSemanticReject:
      return v2::ROUTED_TERMINAL_PROJECTION_KIND_SEMANTIC_REJECT;
    case TerminalProjectionKind::kStatusSnapshot:
      return v2::ROUTED_TERMINAL_PROJECTION_KIND_STATUS_SNAPSHOT;
    case TerminalProjectionKind::kFamilyDefined:
      return v2::ROUTED_TERMINAL_PROJECTION_KIND_FAMILY_DEFINED;
  }
  return v2::ROUTED_TERMINAL_PROJECTION_KIND_UNSPECIFIED;
}

inline absl::StatusOr<TerminalProjectionKind> terminal_projection_kind_from_proto(
    v2::RoutedTerminalProjectionKind projection_kind) {
  switch (projection_kind) {
    case v2::ROUTED_TERMINAL_PROJECTION_KIND_SEMANTIC_SUCCESS:
      return TerminalProjectionKind::kSemanticSuccess;
    case v2::ROUTED_TERMINAL_PROJECTION_KIND_SEMANTIC_REJECT:
      return TerminalProjectionKind::kSemanticReject;
    case v2::ROUTED_TERMINAL_PROJECTION_KIND_STATUS_SNAPSHOT:
      return TerminalProjectionKind::kStatusSnapshot;
    case v2::ROUTED_TERMINAL_PROJECTION_KIND_FAMILY_DEFINED:
      return TerminalProjectionKind::kFamilyDefined;
    case v2::ROUTED_TERMINAL_PROJECTION_KIND_UNSPECIFIED:
    case v2::RoutedTerminalProjectionKind_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v2::RoutedTerminalProjectionKind_INT_MAX_SENTINEL_DO_NOT_USE_:
    default:
      return absl::InvalidArgumentError("terminal_projection.projection_kind is unspecified");
  }
}

} // namespace detail

inline void populate_proto_fencing_context(const FencingContext& fencing_context, v2::RoutedFencingContext* proto) {
  proto->Clear();
  proto->set_principal_kind(detail::fencing_principal_kind_to_proto(fencing_context.principal_kind));
  proto->set_principal_id(fencing_context.principal_id);
  proto->set_epoch(fencing_context.epoch);
}

inline FencingContext fencing_context_from_proto(const v2::RoutedFencingContext& proto) {
  return FencingContext{
      .principal_kind = detail::fencing_principal_kind_from_proto(proto.principal_kind()),
      .principal_id = proto.principal_id(),
      .epoch = proto.epoch(),
  };
}

inline void populate_proto_authority_ref(const AuthorityRef& authority_ref, v2::AuthorityRef* proto_authority_ref) {
  proto_authority_ref->Clear();
  proto_authority_ref->set_authority_kind(detail::authority_kind_to_proto(authority_ref.authority_kind));
  proto_authority_ref->set_authority_id(authority_ref.authority_id);
  if (authority_ref.fencing_context.has_value()) {
    populate_proto_fencing_context(*authority_ref.fencing_context, proto_authority_ref->mutable_fencing_context());
  }
}

inline AuthorityRef authority_ref_from_proto(const v2::AuthorityRef& proto_authority_ref) {
  AuthorityRef authority_ref{
      .authority_kind = detail::authority_kind_from_proto(proto_authority_ref.authority_kind()),
      .authority_id = proto_authority_ref.authority_id(),
  };
  if (proto_authority_ref.has_fencing_context()) {
    authority_ref.fencing_context = fencing_context_from_proto(proto_authority_ref.fencing_context());
  }
  return authority_ref;
}

inline void populate_proto_portable_credential(
    const PortableParsedCredential& portable_credential,
    v2::PortableParsedCredential* proto_portable_credential) {
  proto_portable_credential->Clear();
  auto* address = proto_portable_credential->mutable_address();
  address->set_route_principal_kind(
      detail::route_principal_kind_label(portable_credential.address.route_principal.principal_kind));
  address->set_route_principal_id(portable_credential.address.route_principal.principal_id);
  address->set_family(detail::capability_family_label(portable_credential.address.family));
  address->set_binding_space(detail::binding_space_label(portable_credential.address.binding_space));
  address->set_binding_key_kind(detail::binding_key_kind_label(portable_credential.address.binding_key_kind));
  address->set_binding_key(portable_credential.address.binding_key);
  address->set_subject_generation(portable_credential.address.epochs.subject_generation);
  if (portable_credential.address.binding_id.has_value()) {
    address->set_binding_id(*portable_credential.address.binding_id);
  }
  proto_portable_credential->set_front_door_kind(
      detail::lifecycle_front_door_kind_label(portable_credential.front_door_kind));
  detail::populate_proto_timestamp(
      portable_credential.credential_expires_at, proto_portable_credential->mutable_credential_expires_at());
  proto_portable_credential->set_binding_mode(detail::lifecycle_binding_mode_label(portable_credential.binding_mode));
  auto* claims = proto_portable_credential->mutable_portable_constraint_claims();
  claims->set_artifact_id(portable_credential.portable_constraint_claims.artifact_id);
  claims->set_digest_alg(portable_credential.portable_constraint_claims.digest_alg);
  claims->set_digest_hex(portable_credential.portable_constraint_claims.digest_hex);
  claims->set_direction(portable_credential.portable_constraint_claims.direction);
  claims->set_operation_id(portable_credential.portable_constraint_claims.operation_id);
  claims->set_holder_scope(portable_credential.portable_constraint_claims.holder_scope);
  claims->set_local_only(portable_credential.portable_constraint_claims.local_only);
}

inline absl::StatusOr<PortableParsedCredential> portable_credential_from_proto(
    const v2::PortableParsedCredential& proto_portable_credential) {
  if (!proto_portable_credential.has_address()) {
    return absl::InvalidArgumentError("portable_credential.address is required");
  }
  auto route_principal_kind_or =
      detail::route_principal_kind_from_label(proto_portable_credential.address().route_principal_kind());
  if (!route_principal_kind_or.ok()) {
    return route_principal_kind_or.status();
  }
  auto family_or = detail::capability_family_from_label(proto_portable_credential.address().family());
  if (!family_or.ok()) {
    return family_or.status();
  }
  auto binding_space_or = detail::binding_space_from_label(proto_portable_credential.address().binding_space());
  if (!binding_space_or.ok()) {
    return binding_space_or.status();
  }
  auto binding_key_kind_or =
      detail::binding_key_kind_from_label(proto_portable_credential.address().binding_key_kind());
  if (!binding_key_kind_or.ok()) {
    return binding_key_kind_or.status();
  }
  auto front_door_kind_or = detail::lifecycle_front_door_kind_from_label(proto_portable_credential.front_door_kind());
  if (!front_door_kind_or.ok()) {
    return front_door_kind_or.status();
  }
  auto binding_mode_or = detail::lifecycle_binding_mode_from_label(proto_portable_credential.binding_mode());
  if (!binding_mode_or.ok()) {
    return binding_mode_or.status();
  }
  if (!proto_portable_credential.has_credential_expires_at()) {
    return absl::InvalidArgumentError("portable_credential.credential_expires_at is required");
  }
  auto credential_expires_at_or = detail::timestamp_from_proto(proto_portable_credential.credential_expires_at());
  if (!credential_expires_at_or.ok()) {
    return credential_expires_at_or.status();
  }
  PortableParsedCredential portable_credential{
      .address =
          CapabilityBindingAddress{
              .route_principal =
                  LifecycleRoutePrincipal{
                      .principal_kind = *route_principal_kind_or,
                      .principal_id = proto_portable_credential.address().route_principal_id(),
                  },
              .family = *family_or,
              .binding_space = *binding_space_or,
              .binding_key_kind = *binding_key_kind_or,
              .binding_key = proto_portable_credential.address().binding_key(),
              .epochs =
                  LifecycleEpochs{
                      .subject_generation = proto_portable_credential.address().subject_generation(),
                  },
          },
      .front_door_kind = *front_door_kind_or,
      .credential_expires_at = *credential_expires_at_or,
      .binding_mode = *binding_mode_or,
      .portable_constraint_claims =
          ConstraintClaims{
              .artifact_id = proto_portable_credential.portable_constraint_claims().artifact_id(),
              .digest_alg = proto_portable_credential.portable_constraint_claims().digest_alg(),
              .digest_hex = proto_portable_credential.portable_constraint_claims().digest_hex(),
              .direction = proto_portable_credential.portable_constraint_claims().direction(),
              .operation_id = proto_portable_credential.portable_constraint_claims().operation_id(),
              .holder_scope = proto_portable_credential.portable_constraint_claims().holder_scope(),
              .local_only = proto_portable_credential.portable_constraint_claims().local_only(),
          },
  };
  if (proto_portable_credential.address().has_binding_id()) {
    portable_credential.address.binding_id = proto_portable_credential.address().binding_id();
  }
  return portable_credential;
}

inline void populate_proto_forwardable_evidence(
    const ForwardableCredentialEvidence& forwardable_evidence,
    v2::ForwardableCredentialEvidence* proto_forwardable_evidence) {
  proto_forwardable_evidence->Clear();
  proto_forwardable_evidence->set_evidence_kind(
      detail::credential_evidence_kind_to_proto(forwardable_evidence.evidence_kind));
  if (forwardable_evidence.raw_credential_bytes.has_value()) {
    proto_forwardable_evidence->set_raw_credential_bytes(*forwardable_evidence.raw_credential_bytes);
  }
  if (forwardable_evidence.canonical_projection.has_value()) {
    auto* projection = proto_forwardable_evidence->mutable_canonical_projection();
    projection->set_projection_kind(forwardable_evidence.canonical_projection->projection_kind);
    projection->set_projection_version(forwardable_evidence.canonical_projection->projection_version);
    projection->set_projection_bytes(forwardable_evidence.canonical_projection->projection_bytes);
    projection->set_projection_digest(forwardable_evidence.canonical_projection->projection_digest);
    projection->set_issuer_binding(forwardable_evidence.canonical_projection->issuer_binding);
    if (forwardable_evidence.canonical_projection->projection_authenticator.has_value()) {
      projection->set_projection_authenticator(*forwardable_evidence.canonical_projection->projection_authenticator);
    }
  }
}

inline ForwardableCredentialEvidence forwardable_evidence_from_proto(
    const v2::ForwardableCredentialEvidence& proto_forwardable_evidence) {
  ForwardableCredentialEvidence forwardable_evidence{
      .evidence_kind = detail::credential_evidence_kind_from_proto(proto_forwardable_evidence.evidence_kind()),
  };
  if (proto_forwardable_evidence.has_raw_credential_bytes()) {
    forwardable_evidence.raw_credential_bytes = proto_forwardable_evidence.raw_credential_bytes();
  }
  if (proto_forwardable_evidence.has_canonical_projection()) {
    forwardable_evidence.canonical_projection = CanonicalCredentialProjection{
        .projection_kind = proto_forwardable_evidence.canonical_projection().projection_kind(),
        .projection_version = proto_forwardable_evidence.canonical_projection().projection_version(),
        .projection_bytes = proto_forwardable_evidence.canonical_projection().projection_bytes(),
        .projection_digest = proto_forwardable_evidence.canonical_projection().projection_digest(),
        .issuer_binding = proto_forwardable_evidence.canonical_projection().issuer_binding(),
    };
    if (proto_forwardable_evidence.canonical_projection().has_projection_authenticator()) {
      forwardable_evidence.canonical_projection->projection_authenticator =
          proto_forwardable_evidence.canonical_projection().projection_authenticator();
    }
  }
  return forwardable_evidence;
}

inline void populate_proto_daemon_hop_auth_context(
    const DaemonHopAuthContext& hop_auth_context,
    v2::DaemonHopAuthContext* proto_hop_auth_context) {
  proto_hop_auth_context->Clear();
  proto_hop_auth_context->set_auth_class(detail::daemon_hop_auth_class_to_proto(hop_auth_context.auth_class));
  if (hop_auth_context.authenticated_peer_daemon_id.has_value()) {
    proto_hop_auth_context->set_authenticated_peer_daemon_id(*hop_auth_context.authenticated_peer_daemon_id);
  }
  if (hop_auth_context.transport_peer.has_value()) {
    proto_hop_auth_context->set_transport_peer(*hop_auth_context.transport_peer);
  }
}

inline DaemonHopAuthContext daemon_hop_auth_context_from_proto(const v2::DaemonHopAuthContext& proto_hop_auth_context) {
  DaemonHopAuthContext hop_auth_context{
      .auth_class = detail::daemon_hop_auth_class_from_proto(proto_hop_auth_context.auth_class()),
  };
  if (proto_hop_auth_context.has_authenticated_peer_daemon_id()) {
    hop_auth_context.authenticated_peer_daemon_id = proto_hop_auth_context.authenticated_peer_daemon_id();
  }
  if (proto_hop_auth_context.has_transport_peer()) {
    hop_auth_context.transport_peer = proto_hop_auth_context.transport_peer();
  }
  return hop_auth_context;
}

inline void populate_proto_delegation_envelope(
    const DelegationEnvelope& delegation_envelope,
    v2::DelegationEnvelope* proto_delegation_envelope) {
  proto_delegation_envelope->Clear();
  populate_proto_authority_ref(
      delegation_envelope.audience_authority_ref, proto_delegation_envelope->mutable_audience_authority_ref());
  proto_delegation_envelope->set_bound_root_request_id(delegation_envelope.bound_root_request_id);
  if (delegation_envelope.bound_path_family.has_value()) {
    proto_delegation_envelope->set_bound_path_family(*delegation_envelope.bound_path_family);
  }
  if (delegation_envelope.bound_edge.has_value()) {
    proto_delegation_envelope->set_bound_edge(*delegation_envelope.bound_edge);
  }
  proto_delegation_envelope->set_payload_kind(
      detail::delegation_payload_kind_to_proto(delegation_envelope.payload_kind));
  proto_delegation_envelope->set_delegation_class(
      detail::delegation_class_to_proto(delegation_envelope.delegation_class));
  if (delegation_envelope.not_before.has_value()) {
    detail::populate_proto_timestamp(*delegation_envelope.not_before, proto_delegation_envelope->mutable_not_before());
  }
  if (delegation_envelope.expires_at.has_value()) {
    detail::populate_proto_timestamp(*delegation_envelope.expires_at, proto_delegation_envelope->mutable_expires_at());
  }
  if (delegation_envelope.authenticator.has_value()) {
    proto_delegation_envelope->set_authenticator(*delegation_envelope.authenticator);
  }
}

inline absl::StatusOr<DelegationEnvelope> delegation_envelope_from_proto(
    const v2::DelegationEnvelope& proto_delegation_envelope) {
  if (!proto_delegation_envelope.has_audience_authority_ref()) {
    return absl::InvalidArgumentError("delegation_envelope.audience_authority_ref is required");
  }
  DelegationEnvelope delegation_envelope{
      .audience_authority_ref = authority_ref_from_proto(proto_delegation_envelope.audience_authority_ref()),
      .bound_root_request_id = proto_delegation_envelope.bound_root_request_id(),
      .payload_kind = detail::delegation_payload_kind_from_proto(proto_delegation_envelope.payload_kind()),
      .delegation_class = detail::delegation_class_from_proto(proto_delegation_envelope.delegation_class()),
  };
  if (proto_delegation_envelope.has_bound_path_family()) {
    delegation_envelope.bound_path_family = proto_delegation_envelope.bound_path_family();
  }
  if (proto_delegation_envelope.has_bound_edge()) {
    delegation_envelope.bound_edge = proto_delegation_envelope.bound_edge();
  }
  if (proto_delegation_envelope.has_not_before()) {
    auto not_before_or = detail::timestamp_from_proto(proto_delegation_envelope.not_before());
    if (!not_before_or.ok()) {
      return not_before_or.status();
    }
    delegation_envelope.not_before = *not_before_or;
  }
  if (proto_delegation_envelope.has_expires_at()) {
    auto expires_at_or = detail::timestamp_from_proto(proto_delegation_envelope.expires_at());
    if (!expires_at_or.ok()) {
      return expires_at_or.status();
    }
    delegation_envelope.expires_at = *expires_at_or;
  }
  if (proto_delegation_envelope.has_authenticator()) {
    delegation_envelope.authenticator = proto_delegation_envelope.authenticator();
  }
  return delegation_envelope;
}

inline void populate_proto_forwarded_claim(
    const ForwardedClaim& forwarded_claim,
    v2::ForwardedClaim* proto_forwarded_claim) {
  proto_forwarded_claim->Clear();
  proto_forwarded_claim->set_claim_kind(forwarded_claim.claim_kind);
  proto_forwarded_claim->set_provenance(detail::forwarded_claim_provenance_to_proto(forwarded_claim.provenance));
  proto_forwarded_claim->set_claim_payload(forwarded_claim.claim_payload);
  populate_proto_authority_ref(
      forwarded_claim.minted_by_authority_ref, proto_forwarded_claim->mutable_minted_by_authority_ref());
  populate_proto_authority_ref(
      forwarded_claim.audience_authority_ref, proto_forwarded_claim->mutable_audience_authority_ref());
  proto_forwarded_claim->set_bound_root_request_id(forwarded_claim.bound_root_request_id);
  if (forwarded_claim.bound_credential_binding_digest.has_value()) {
    proto_forwarded_claim->set_bound_credential_binding_digest(*forwarded_claim.bound_credential_binding_digest);
  }
  if (forwarded_claim.bound_path_family.has_value()) {
    proto_forwarded_claim->set_bound_path_family(*forwarded_claim.bound_path_family);
  }
  if (forwarded_claim.bound_edge.has_value()) {
    proto_forwarded_claim->set_bound_edge(*forwarded_claim.bound_edge);
  }
  if (forwarded_claim.claim_expires_at.has_value()) {
    detail::populate_proto_timestamp(
        *forwarded_claim.claim_expires_at, proto_forwarded_claim->mutable_claim_expires_at());
  }
  if (forwarded_claim.claim_authenticator.has_value()) {
    proto_forwarded_claim->set_claim_authenticator(*forwarded_claim.claim_authenticator);
  }
}

inline absl::StatusOr<ForwardedClaim> forwarded_claim_from_proto(const v2::ForwardedClaim& proto_forwarded_claim) {
  if (!proto_forwarded_claim.has_minted_by_authority_ref()) {
    return absl::InvalidArgumentError("forwarded_claim.minted_by_authority_ref is required");
  }
  if (!proto_forwarded_claim.has_audience_authority_ref()) {
    return absl::InvalidArgumentError("forwarded_claim.audience_authority_ref is required");
  }
  ForwardedClaim forwarded_claim{
      .claim_kind = proto_forwarded_claim.claim_kind(),
      .provenance = detail::forwarded_claim_provenance_from_proto(proto_forwarded_claim.provenance()),
      .claim_payload = proto_forwarded_claim.claim_payload(),
      .minted_by_authority_ref = authority_ref_from_proto(proto_forwarded_claim.minted_by_authority_ref()),
      .audience_authority_ref = authority_ref_from_proto(proto_forwarded_claim.audience_authority_ref()),
      .bound_root_request_id = proto_forwarded_claim.bound_root_request_id(),
  };
  if (proto_forwarded_claim.has_bound_credential_binding_digest()) {
    forwarded_claim.bound_credential_binding_digest = proto_forwarded_claim.bound_credential_binding_digest();
  }
  if (proto_forwarded_claim.has_bound_path_family()) {
    forwarded_claim.bound_path_family = proto_forwarded_claim.bound_path_family();
  }
  if (proto_forwarded_claim.has_bound_edge()) {
    forwarded_claim.bound_edge = proto_forwarded_claim.bound_edge();
  }
  if (proto_forwarded_claim.has_claim_expires_at()) {
    auto claim_expires_at_or = detail::timestamp_from_proto(proto_forwarded_claim.claim_expires_at());
    if (!claim_expires_at_or.ok()) {
      return claim_expires_at_or.status();
    }
    forwarded_claim.claim_expires_at = *claim_expires_at_or;
  }
  if (proto_forwarded_claim.has_claim_authenticator()) {
    forwarded_claim.claim_authenticator = proto_forwarded_claim.claim_authenticator();
  }
  return forwarded_claim;
}

inline void populate_proto_routed_request_metadata(
    const RoutedRequestMetadata& request_metadata,
    v2::RoutedRequestMetadata* proto_request_metadata) {
  proto_request_metadata->Clear();
  proto_request_metadata->set_root_request_id(request_metadata.root_request_id);
  if (request_metadata.deadline.has_value()) {
    detail::populate_proto_timestamp(*request_metadata.deadline, proto_request_metadata->mutable_deadline());
  }
  if (request_metadata.trace_context.has_value()) {
    proto_request_metadata->set_trace_context(*request_metadata.trace_context);
  }
  if (request_metadata.idempotency_key.has_value()) {
    proto_request_metadata->set_idempotency_key(*request_metadata.idempotency_key);
  }
  if (request_metadata.credential_binding_digest.has_value()) {
    proto_request_metadata->set_credential_binding_digest(*request_metadata.credential_binding_digest);
  }
  proto_request_metadata->set_hop_budget_remaining(request_metadata.hop_budget_remaining);
  proto_request_metadata->set_retry_attempt(request_metadata.retry_attempt);
}

inline absl::StatusOr<RoutedRequestMetadata> routed_request_metadata_from_proto(
    const v2::RoutedRequestMetadata& proto_request_metadata) {
  RoutedRequestMetadata request_metadata{
      .root_request_id = proto_request_metadata.root_request_id(),
      .hop_budget_remaining = proto_request_metadata.hop_budget_remaining(),
      .retry_attempt = proto_request_metadata.retry_attempt(),
  };
  if (proto_request_metadata.has_deadline()) {
    auto deadline_or = detail::timestamp_from_proto(proto_request_metadata.deadline());
    if (!deadline_or.ok()) {
      return deadline_or.status();
    }
    request_metadata.deadline = *deadline_or;
  }
  if (proto_request_metadata.has_trace_context()) {
    request_metadata.trace_context = proto_request_metadata.trace_context();
  }
  if (proto_request_metadata.has_idempotency_key()) {
    request_metadata.idempotency_key = proto_request_metadata.idempotency_key();
  }
  if (proto_request_metadata.has_credential_binding_digest()) {
    request_metadata.credential_binding_digest = proto_request_metadata.credential_binding_digest();
  }
  return request_metadata;
}

inline absl::Status validate_routed_authority_request_shape(const RoutedAuthorityRequest& routed_request) {
  if (routed_request.authority_ref.authority_id.empty()) {
    return absl::InvalidArgumentError("routed_request.authority_ref.authority_id is required");
  }
  if (routed_request.path_family.empty()) {
    return absl::InvalidArgumentError("routed_request.path_family is required");
  }
  if (routed_request.stage_ref.empty()) {
    return absl::InvalidArgumentError("routed_request.stage_ref is required");
  }
  if (routed_request.portable_credential.address.binding_key.empty()) {
    return absl::InvalidArgumentError("routed_request.portable_credential.address.binding_key is required");
  }
  return absl::OkStatus();
}

inline void populate_proto_routed_authority_request(
    const RoutedAuthorityRequest& routed_request,
    v2::RoutedAuthorityRequest* proto_routed_request) {
  proto_routed_request->Clear();
  populate_proto_authority_ref(routed_request.authority_ref, proto_routed_request->mutable_authority_ref());
  proto_routed_request->set_path_family(routed_request.path_family);
  proto_routed_request->set_stage_ref(routed_request.stage_ref);
  populate_proto_portable_credential(
      routed_request.portable_credential, proto_routed_request->mutable_portable_credential());
  if (routed_request.forwardable_evidence.has_value()) {
    populate_proto_forwardable_evidence(
        *routed_request.forwardable_evidence, proto_routed_request->mutable_forwardable_evidence());
  }
  if (routed_request.portable_credential_envelope.has_value()) {
    populate_proto_delegation_envelope(
        *routed_request.portable_credential_envelope, proto_routed_request->mutable_portable_credential_envelope());
  }
  if (routed_request.forwardable_evidence_envelope.has_value()) {
    populate_proto_delegation_envelope(
        *routed_request.forwardable_evidence_envelope, proto_routed_request->mutable_forwardable_evidence_envelope());
  }
  populate_proto_daemon_hop_auth_context(
      routed_request.hop_auth_context, proto_routed_request->mutable_hop_auth_context());
  for (const auto& forwarded_claim : routed_request.forwarded_claims) {
    populate_proto_forwarded_claim(forwarded_claim, proto_routed_request->add_forwarded_claims());
  }
  if (routed_request.forwarded_claims_envelope.has_value()) {
    populate_proto_delegation_envelope(
        *routed_request.forwarded_claims_envelope, proto_routed_request->mutable_forwarded_claims_envelope());
  }
  populate_proto_routed_request_metadata(
      routed_request.request_metadata, proto_routed_request->mutable_request_metadata());
}

inline absl::StatusOr<RoutedAuthorityRequest> routed_authority_request_from_proto(
    const v2::RoutedAuthorityRequest& proto_routed_request) {
  if (!proto_routed_request.has_authority_ref()) {
    return absl::InvalidArgumentError("routed_request.authority_ref is required");
  }
  if (!proto_routed_request.has_portable_credential()) {
    return absl::InvalidArgumentError("routed_request.portable_credential is required");
  }
  auto portable_credential_or = portable_credential_from_proto(proto_routed_request.portable_credential());
  if (!portable_credential_or.ok()) {
    return portable_credential_or.status();
  }
  RoutedAuthorityRequest routed_request{
      .authority_ref = authority_ref_from_proto(proto_routed_request.authority_ref()),
      .path_family = proto_routed_request.path_family(),
      .stage_ref = proto_routed_request.stage_ref(),
      .portable_credential = *portable_credential_or,
      .hop_auth_context = proto_routed_request.has_hop_auth_context()
          ? daemon_hop_auth_context_from_proto(proto_routed_request.hop_auth_context())
          : DaemonHopAuthContext{},
  };
  if (proto_routed_request.has_forwardable_evidence()) {
    routed_request.forwardable_evidence = forwardable_evidence_from_proto(proto_routed_request.forwardable_evidence());
  }
  if (proto_routed_request.has_portable_credential_envelope()) {
    auto portable_credential_envelope_or =
        delegation_envelope_from_proto(proto_routed_request.portable_credential_envelope());
    if (!portable_credential_envelope_or.ok()) {
      return portable_credential_envelope_or.status();
    }
    routed_request.portable_credential_envelope = *portable_credential_envelope_or;
  }
  if (proto_routed_request.has_forwardable_evidence_envelope()) {
    auto forwardable_evidence_envelope_or =
        delegation_envelope_from_proto(proto_routed_request.forwardable_evidence_envelope());
    if (!forwardable_evidence_envelope_or.ok()) {
      return forwardable_evidence_envelope_or.status();
    }
    routed_request.forwardable_evidence_envelope = *forwardable_evidence_envelope_or;
  }
  for (const auto& proto_forwarded_claim : proto_routed_request.forwarded_claims()) {
    auto forwarded_claim_or = forwarded_claim_from_proto(proto_forwarded_claim);
    if (!forwarded_claim_or.ok()) {
      return forwarded_claim_or.status();
    }
    routed_request.forwarded_claims.push_back(*forwarded_claim_or);
  }
  if (proto_routed_request.has_forwarded_claims_envelope()) {
    auto forwarded_claims_envelope_or =
        delegation_envelope_from_proto(proto_routed_request.forwarded_claims_envelope());
    if (!forwarded_claims_envelope_or.ok()) {
      return forwarded_claims_envelope_or.status();
    }
    routed_request.forwarded_claims_envelope = *forwarded_claims_envelope_or;
  }
  if (proto_routed_request.has_request_metadata()) {
    auto request_metadata_or = routed_request_metadata_from_proto(proto_routed_request.request_metadata());
    if (!request_metadata_or.ok()) {
      return request_metadata_or.status();
    }
    routed_request.request_metadata = *request_metadata_or;
  }
  auto validation_status = validate_routed_authority_request_shape(routed_request);
  if (!validation_status.ok()) {
    return validation_status;
  }
  return routed_request;
}

inline void populate_proto_authority_attachment_ref(
    const AuthorityAttachmentRef& attachment_ref,
    v2::AuthorityAttachmentRef* proto_attachment_ref) {
  proto_attachment_ref->Clear();
  populate_proto_authority_ref(attachment_ref.authority_ref, proto_attachment_ref->mutable_authority_ref());
  proto_attachment_ref->set_attachment_kind(attachment_ref.attachment_kind);
  proto_attachment_ref->set_attachment_id(attachment_ref.attachment_id);
  if (attachment_ref.fencing_context.has_value()) {
    populate_proto_fencing_context(*attachment_ref.fencing_context, proto_attachment_ref->mutable_fencing_context());
  }
}

inline absl::StatusOr<AuthorityAttachmentRef> authority_attachment_ref_from_proto(
    const v2::AuthorityAttachmentRef& proto_attachment_ref) {
  if (!proto_attachment_ref.has_authority_ref()) {
    return absl::InvalidArgumentError("attachment_ref.authority_ref is required");
  }
  AuthorityAttachmentRef attachment_ref{
      .authority_ref = authority_ref_from_proto(proto_attachment_ref.authority_ref()),
      .attachment_kind = proto_attachment_ref.attachment_kind(),
      .attachment_id = proto_attachment_ref.attachment_id(),
  };
  if (proto_attachment_ref.has_fencing_context()) {
    attachment_ref.fencing_context = fencing_context_from_proto(proto_attachment_ref.fencing_context());
  }
  return attachment_ref;
}

inline void populate_proto_authority_continuation(
    const AuthorityContinuation& continuation,
    v2::AuthorityContinuation* proto_continuation) {
  proto_continuation->Clear();
  populate_proto_authority_ref(continuation.next_authority_ref, proto_continuation->mutable_next_authority_ref());
  proto_continuation->set_edge_ref(continuation.edge_ref);
  for (const auto& forwarded_claim : continuation.forwarded_claims) {
    populate_proto_forwarded_claim(forwarded_claim, proto_continuation->add_forwarded_claims());
  }
  if (continuation.forwarded_claims_envelope.has_value()) {
    populate_proto_delegation_envelope(
        *continuation.forwarded_claims_envelope, proto_continuation->mutable_forwarded_claims_envelope());
  }
  if (continuation.continuation_reason.has_value()) {
    proto_continuation->set_continuation_reason(*continuation.continuation_reason);
  }
}

inline absl::StatusOr<AuthorityContinuation> authority_continuation_from_proto(
    const v2::AuthorityContinuation& proto_continuation) {
  if (!proto_continuation.has_next_authority_ref()) {
    return absl::InvalidArgumentError("continuation.next_authority_ref is required");
  }
  AuthorityContinuation continuation{
      .next_authority_ref = authority_ref_from_proto(proto_continuation.next_authority_ref()),
      .edge_ref = proto_continuation.edge_ref(),
  };
  for (const auto& proto_forwarded_claim : proto_continuation.forwarded_claims()) {
    auto forwarded_claim_or = forwarded_claim_from_proto(proto_forwarded_claim);
    if (!forwarded_claim_or.ok()) {
      return forwarded_claim_or.status();
    }
    continuation.forwarded_claims.push_back(*forwarded_claim_or);
  }
  if (proto_continuation.has_forwarded_claims_envelope()) {
    auto forwarded_claims_envelope_or = delegation_envelope_from_proto(proto_continuation.forwarded_claims_envelope());
    if (!forwarded_claims_envelope_or.ok()) {
      return forwarded_claims_envelope_or.status();
    }
    continuation.forwarded_claims_envelope = *forwarded_claims_envelope_or;
  }
  if (proto_continuation.has_continuation_reason()) {
    continuation.continuation_reason = proto_continuation.continuation_reason();
  }
  return continuation;
}

inline void populate_proto_terminal_projection(
    const TerminalProjection& terminal_projection,
    v2::TerminalProjection* proto_terminal_projection) {
  proto_terminal_projection->Clear();
  proto_terminal_projection->set_projection_kind(
      detail::terminal_projection_kind_to_proto(terminal_projection.projection_kind));
  if (terminal_projection.status_code.has_value()) {
    proto_terminal_projection->set_status_code(*terminal_projection.status_code);
  }
  if (terminal_projection.family_payload.has_value()) {
    proto_terminal_projection->set_family_payload(*terminal_projection.family_payload);
  }
}

inline absl::StatusOr<TerminalProjection> terminal_projection_from_proto(
    const v2::TerminalProjection& proto_terminal_projection) {
  auto projection_kind_or = detail::terminal_projection_kind_from_proto(proto_terminal_projection.projection_kind());
  if (!projection_kind_or.ok()) {
    return projection_kind_or.status();
  }
  TerminalProjection terminal_projection{
      .projection_kind = *projection_kind_or,
  };
  if (proto_terminal_projection.has_status_code()) {
    terminal_projection.status_code = proto_terminal_projection.status_code();
  }
  if (proto_terminal_projection.has_family_payload()) {
    terminal_projection.family_payload = proto_terminal_projection.family_payload();
  }
  return terminal_projection;
}

inline absl::Status validate_owner_stage_reply_shape(const OwnerStageReply& owner_stage_reply) {
  if (owner_stage_reply.answered_by.authority_id.empty()) {
    return absl::InvalidArgumentError("owner_stage_reply.answered_by.authority_id is required");
  }
  if (owner_stage_reply.path_family.empty()) {
    return absl::InvalidArgumentError("owner_stage_reply.path_family is required");
  }
  if (owner_stage_reply.stage_ref.empty()) {
    return absl::InvalidArgumentError("owner_stage_reply.stage_ref is required");
  }
  switch (owner_stage_reply.reply_kind) {
    case OwnerStageReplyKind::kReadyForLowering:
      if (owner_stage_reply.resolved_source_capability == nullptr) {
        return absl::InvalidArgumentError("ready_for_lowering requires resolved_source_capability");
      }
      if (owner_stage_reply.continuation.has_value() || owner_stage_reply.attachment_ref.has_value() ||
          owner_stage_reply.terminal_projection.has_value()) {
        return absl::InvalidArgumentError(
            "ready_for_lowering must not carry continuation, attachment_ref, or terminal_projection");
      }
      break;
    case OwnerStageReplyKind::kContinueWithAuthority:
      if (!owner_stage_reply.continuation.has_value()) {
        return absl::InvalidArgumentError("continue_with_authority requires continuation");
      }
      if (owner_stage_reply.resolved_source_capability != nullptr || owner_stage_reply.attachment_ref.has_value() ||
          owner_stage_reply.terminal_projection.has_value()) {
        return absl::InvalidArgumentError("continue_with_authority must carry only continuation");
      }
      break;
    case OwnerStageReplyKind::kRetryLater:
      if (owner_stage_reply.resolved_source_capability != nullptr || owner_stage_reply.continuation.has_value() ||
          owner_stage_reply.terminal_projection.has_value()) {
        return absl::InvalidArgumentError(
            "retry_later must not carry resolved_source_capability, continuation, or terminal_projection");
      }
      break;
    case OwnerStageReplyKind::kAttachExisting:
      if (!owner_stage_reply.attachment_ref.has_value()) {
        return absl::InvalidArgumentError("attach_existing requires attachment_ref");
      }
      if (owner_stage_reply.resolved_source_capability != nullptr || owner_stage_reply.continuation.has_value() ||
          owner_stage_reply.terminal_projection.has_value()) {
        return absl::InvalidArgumentError("attach_existing must carry only attachment_ref");
      }
      break;
    case OwnerStageReplyKind::kTerminal:
      if (!owner_stage_reply.terminal_projection.has_value()) {
        return absl::InvalidArgumentError("terminal requires terminal_projection");
      }
      if (owner_stage_reply.resolved_source_capability != nullptr || owner_stage_reply.continuation.has_value() ||
          owner_stage_reply.attachment_ref.has_value()) {
        return absl::InvalidArgumentError("terminal must carry only terminal_projection");
      }
      break;
  }
  return absl::OkStatus();
}

inline absl::Status validate_owner_stage_reply_shell_shape(
    OwnerStageReplyKind reply_kind,
    bool has_resolved_source_capability_payload,
    bool has_continuation,
    bool has_attachment_ref,
    bool has_terminal_projection) {
  switch (reply_kind) {
    case OwnerStageReplyKind::kReadyForLowering:
      if (!has_resolved_source_capability_payload) {
        return absl::InvalidArgumentError("ready_for_lowering requires resolved_source_capability");
      }
      if (has_continuation || has_attachment_ref || has_terminal_projection) {
        return absl::InvalidArgumentError(
            "ready_for_lowering must not carry continuation, attachment_ref, or terminal_projection");
      }
      break;
    case OwnerStageReplyKind::kContinueWithAuthority:
      if (!has_continuation) {
        return absl::InvalidArgumentError("continue_with_authority requires continuation");
      }
      if (has_resolved_source_capability_payload || has_attachment_ref || has_terminal_projection) {
        return absl::InvalidArgumentError("continue_with_authority must carry only continuation");
      }
      break;
    case OwnerStageReplyKind::kRetryLater:
      if (has_resolved_source_capability_payload || has_continuation || has_terminal_projection) {
        return absl::InvalidArgumentError(
            "retry_later must not carry resolved_source_capability, continuation, or terminal_projection");
      }
      break;
    case OwnerStageReplyKind::kAttachExisting:
      if (!has_attachment_ref) {
        return absl::InvalidArgumentError("attach_existing requires attachment_ref");
      }
      if (has_resolved_source_capability_payload || has_continuation || has_terminal_projection) {
        return absl::InvalidArgumentError("attach_existing must carry only attachment_ref");
      }
      break;
    case OwnerStageReplyKind::kTerminal:
      if (!has_terminal_projection) {
        return absl::InvalidArgumentError("terminal requires terminal_projection");
      }
      if (has_resolved_source_capability_payload || has_continuation || has_attachment_ref) {
        return absl::InvalidArgumentError("terminal must carry only terminal_projection");
      }
      break;
  }
  return absl::OkStatus();
}

struct OwnerStageReplyShell {
  OwnerStageReply reply;
  std::optional<std::string> resolved_source_capability_payload;
};

inline absl::Status populate_proto_owner_stage_reply_shell(
    const OwnerStageReply& owner_stage_reply,
    const std::optional<std::string>& resolved_source_capability_payload,
    v2::OwnerStageReply* proto_reply) {
  auto validation_status = validate_owner_stage_reply_shape(owner_stage_reply);
  if (!validation_status.ok()) {
    return validation_status;
  }
  proto_reply->Clear();
  populate_proto_authority_ref(owner_stage_reply.answered_by, proto_reply->mutable_answered_by());
  proto_reply->set_path_family(owner_stage_reply.path_family);
  proto_reply->set_stage_ref(owner_stage_reply.stage_ref);
  proto_reply->set_reply_kind(detail::owner_stage_reply_kind_to_proto(owner_stage_reply.reply_kind));
  if (resolved_source_capability_payload.has_value()) {
    proto_reply->set_resolved_source_capability(*resolved_source_capability_payload);
  }
  if (owner_stage_reply.continuation.has_value()) {
    populate_proto_authority_continuation(*owner_stage_reply.continuation, proto_reply->mutable_continuation());
  }
  if (owner_stage_reply.attachment_ref.has_value()) {
    populate_proto_authority_attachment_ref(*owner_stage_reply.attachment_ref, proto_reply->mutable_attachment_ref());
  }
  if (owner_stage_reply.terminal_projection.has_value()) {
    populate_proto_terminal_projection(
        *owner_stage_reply.terminal_projection, proto_reply->mutable_terminal_projection());
  }
  if (owner_stage_reply.diagnostics.has_value()) {
    proto_reply->set_diagnostics(*owner_stage_reply.diagnostics);
  }
  return absl::OkStatus();
}

inline absl::StatusOr<OwnerStageReplyShell> owner_stage_reply_shell_from_proto(const v2::OwnerStageReply& proto_reply) {
  if (!proto_reply.has_answered_by()) {
    return absl::InvalidArgumentError("owner_stage_reply.answered_by is required");
  }
  auto reply_kind_or = detail::owner_stage_reply_kind_from_proto(proto_reply.reply_kind());
  if (!reply_kind_or.ok()) {
    return reply_kind_or.status();
  }
  OwnerStageReply reply{
      .answered_by = authority_ref_from_proto(proto_reply.answered_by()),
      .path_family = proto_reply.path_family(),
      .stage_ref = proto_reply.stage_ref(),
      .reply_kind = *reply_kind_or,
  };
  if (proto_reply.has_continuation()) {
    auto continuation_or = authority_continuation_from_proto(proto_reply.continuation());
    if (!continuation_or.ok()) {
      return continuation_or.status();
    }
    reply.continuation = *continuation_or;
  }
  if (proto_reply.has_attachment_ref()) {
    auto attachment_ref_or = authority_attachment_ref_from_proto(proto_reply.attachment_ref());
    if (!attachment_ref_or.ok()) {
      return attachment_ref_or.status();
    }
    reply.attachment_ref = *attachment_ref_or;
  }
  if (proto_reply.has_terminal_projection()) {
    auto terminal_projection_or = terminal_projection_from_proto(proto_reply.terminal_projection());
    if (!terminal_projection_or.ok()) {
      return terminal_projection_or.status();
    }
    reply.terminal_projection = *terminal_projection_or;
  }
  if (proto_reply.has_diagnostics()) {
    reply.diagnostics = proto_reply.diagnostics();
  }
  if (reply.answered_by.authority_id.empty()) {
    return absl::InvalidArgumentError("owner_stage_reply.answered_by.authority_id is required");
  }
  if (reply.path_family.empty()) {
    return absl::InvalidArgumentError("owner_stage_reply.path_family is required");
  }
  if (reply.stage_ref.empty()) {
    return absl::InvalidArgumentError("owner_stage_reply.stage_ref is required");
  }
  auto validation_status = validate_owner_stage_reply_shell_shape(
      reply.reply_kind,
      !proto_reply.resolved_source_capability().empty(),
      proto_reply.has_continuation(),
      proto_reply.has_attachment_ref(),
      proto_reply.has_terminal_projection());
  if (!validation_status.ok()) {
    return validation_status;
  }
  return OwnerStageReplyShell{
      .reply = std::move(reply),
      .resolved_source_capability_payload = proto_reply.resolved_source_capability().empty()
          ? std::nullopt
          : std::optional<std::string>(proto_reply.resolved_source_capability()),
  };
}

} // namespace tensorcast::daemon::routed_authority_wire
