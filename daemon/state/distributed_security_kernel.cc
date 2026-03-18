// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/distributed_security_kernel.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "daemon/util/grpc_peer_utils.h"
#include "grpcpp/security/auth_context.h"

namespace tensorcast::daemon {

namespace {

int auth_class_rank(DaemonHopAuthClass auth_class) {
  switch (auth_class) {
    case DaemonHopAuthClass::kLegacyUnauthenticated:
      return 0;
    case DaemonHopAuthClass::kDeploymentTrustedChannel:
      return 1;
    case DaemonHopAuthClass::kDaemonMutualAuth:
      return 2;
  }
  return -1;
}

DaemonHopAuthClass daemon_hop_auth_class_from_proto(v2::RoutedDaemonHopAuthClass auth_class) {
  switch (auth_class) {
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DAEMON_MUTUAL_AUTH:
      return DaemonHopAuthClass::kDaemonMutualAuth;
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_DEPLOYMENT_TRUSTED_CHANNEL:
      return DaemonHopAuthClass::kDeploymentTrustedChannel;
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_LEGACY_UNAUTHENTICATED:
    case v2::ROUTED_DAEMON_HOP_AUTH_CLASS_UNSPECIFIED:
    default:
      return DaemonHopAuthClass::kLegacyUnauthenticated;
  }
}

AuthorityKind authority_kind_from_proto(v2::RoutedAuthorityKind authority_kind) {
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
    case v2::ROUTED_AUTHORITY_KIND_INTERNAL_AUTHORITY:
    case v2::ROUTED_AUTHORITY_KIND_UNSPECIFIED:
    default:
      return AuthorityKind::kInternalAuthority;
  }
}

FencingPrincipalKind fencing_principal_kind_from_proto(v2::RoutedFencingPrincipalKind principal_kind) {
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
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_INTERNAL_AUTHORITY:
    case v2::ROUTED_FENCING_PRINCIPAL_KIND_UNSPECIFIED:
    default:
      return FencingPrincipalKind::kInternalAuthority;
  }
}

AuthorityRef authority_ref_from_proto(const v2::AuthorityRef& authority_ref) {
  AuthorityRef result{
      .authority_kind = authority_kind_from_proto(authority_ref.authority_kind()),
      .authority_id = authority_ref.authority_id(),
  };
  if (authority_ref.has_fencing_context()) {
    result.fencing_context = FencingContext{
        .principal_kind = fencing_principal_kind_from_proto(authority_ref.fencing_context().principal_kind()),
        .principal_id = authority_ref.fencing_context().principal_id(),
        .epoch = authority_ref.fencing_context().epoch(),
    };
  }
  return result;
}

bool authority_kind_supports_direct_daemon_binding(AuthorityKind authority_kind) {
  return authority_kind == AuthorityKind::kShardHomeDaemon || authority_kind == AuthorityKind::kIssuerDaemon ||
      authority_kind == AuthorityKind::kInternalAuthority;
}

bool request_contains_sensitive_forwarded_claims(const v2::RoutedAuthorityRequest& routed_request) {
  return routed_request.forwarded_claims_size() > 0;
}

bool policy_allows_payload_kind(const StageDisclosurePolicy& disclosure_policy, DelegationPayloadKind payload_kind) {
  return std::find(
             disclosure_policy.allowed_payload_kinds.begin(),
             disclosure_policy.allowed_payload_kinds.end(),
             payload_kind) != disclosure_policy.allowed_payload_kinds.end();
}

absl::flat_hash_set<ForwardedClaimProvenance> make_claim_provenance_set(
    const StageDisclosurePolicy& disclosure_policy) {
  return absl::flat_hash_set<ForwardedClaimProvenance>(
      disclosure_policy.accepted_claim_provenance.begin(), disclosure_policy.accepted_claim_provenance.end());
}

ForwardedClaimProvenance claim_provenance_from_proto(v2::RoutedForwardedClaimProvenance provenance) {
  switch (provenance) {
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_AUTHORITY_AUTHENTICATED:
      return ForwardedClaimProvenance::kAuthorityAuthenticated;
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_RELAY_AUTHENTICATED:
      return ForwardedClaimProvenance::kRelayAuthenticated;
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_INGRESS_LOCAL:
    case v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_UNSPECIFIED:
    default:
      return ForwardedClaimProvenance::kIngressLocal;
  }
}

DelegationPayloadKind delegation_payload_kind_from_proto(v2::RoutedDelegationPayloadKind payload_kind) {
  switch (payload_kind) {
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDABLE_EVIDENCE:
      return DelegationPayloadKind::kForwardableEvidence;
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDED_CLAIM:
      return DelegationPayloadKind::kForwardedClaim;
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_UNSPECIFIED:
    case v2::ROUTED_DELEGATION_PAYLOAD_KIND_PORTABLE_CREDENTIAL:
    default:
      return DelegationPayloadKind::kPortableCredential;
  }
}

DelegationClass delegation_class_from_proto(v2::RoutedDelegationClass delegation_class) {
  switch (delegation_class) {
    case v2::ROUTED_DELEGATION_CLASS_OWNER_SCOPED_SENSITIVE:
      return DelegationClass::kOwnerScopedSensitive;
    case v2::ROUTED_DELEGATION_CLASS_ISSUER_SECRET:
      return DelegationClass::kIssuerSecret;
    case v2::ROUTED_DELEGATION_CLASS_UNSPECIFIED:
    case v2::ROUTED_DELEGATION_CLASS_BOOTSTRAP_SAFE:
    default:
      return DelegationClass::kBootstrapSafe;
  }
}

absl::StatusOr<absl::Time> time_from_proto_timestamp(const google::protobuf::Timestamp& timestamp) {
  if (timestamp.nanos() < 0 || timestamp.nanos() >= 1000000000) {
    return absl::InvalidArgumentError("delegation timestamp nanos is out of range");
  }
  return absl::FromUnixSeconds(timestamp.seconds()) + absl::Nanoseconds(timestamp.nanos());
}

absl::Status validate_delegation_envelope(
    const v2::DelegationEnvelope& delegation_envelope,
    const AuthorityRef& audience_authority_ref,
    std::string_view root_request_id,
    std::string_view path_family,
    const std::optional<std::string>& edge_ref,
    DelegationPayloadKind expected_payload_kind,
    absl::Time now) {
  if (!delegation_envelope.has_audience_authority_ref()) {
    return absl::FailedPreconditionError("delegation envelope audience_authority_ref is required");
  }
  if (authority_ref_from_proto(delegation_envelope.audience_authority_ref()) != audience_authority_ref) {
    return absl::PermissionDeniedError("delegation envelope audience does not match requested authority");
  }
  if (delegation_envelope.bound_root_request_id() != root_request_id) {
    return absl::FailedPreconditionError("delegation envelope root_request_id mismatch");
  }
  if (delegation_payload_kind_from_proto(delegation_envelope.payload_kind()) != expected_payload_kind) {
    return absl::FailedPreconditionError("delegation envelope payload_kind mismatch");
  }
  if (delegation_envelope.has_bound_path_family() && delegation_envelope.bound_path_family() != path_family) {
    return absl::FailedPreconditionError("delegation envelope path_family mismatch");
  }
  if (edge_ref.has_value()) {
    if (!delegation_envelope.has_bound_edge() || delegation_envelope.bound_edge() != *edge_ref) {
      return absl::FailedPreconditionError("delegation envelope edge scope mismatch");
    }
  } else if (delegation_envelope.has_bound_edge()) {
    return absl::FailedPreconditionError("delegation envelope unexpectedly scoped to an undeclared edge");
  }
  if (delegation_envelope.has_not_before()) {
    auto not_before_or = time_from_proto_timestamp(delegation_envelope.not_before());
    if (!not_before_or.ok()) {
      return not_before_or.status();
    }
    if (*not_before_or > now) {
      return absl::FailedPreconditionError("delegation envelope is not yet valid");
    }
  }
  if (delegation_envelope.has_expires_at()) {
    auto expires_at_or = time_from_proto_timestamp(delegation_envelope.expires_at());
    if (!expires_at_or.ok()) {
      return expires_at_or.status();
    }
    if (*expires_at_or <= now) {
      return absl::FailedPreconditionError("delegation envelope expired");
    }
  }
  return absl::OkStatus();
}

int delegation_class_rank(DelegationClass delegation_class) {
  switch (delegation_class) {
    case DelegationClass::kBootstrapSafe:
      return 0;
    case DelegationClass::kOwnerScopedSensitive:
      return 1;
    case DelegationClass::kIssuerSecret:
      return 2;
  }
  return -1;
}

absl::Status require_delegation_binding(
    const v2::DelegationEnvelope& delegation_envelope,
    const std::optional<AuthorityBindingProof>& authority_binding_proof) {
  const auto delegation_class = delegation_class_from_proto(delegation_envelope.delegation_class());
  if (delegation_class_rank(delegation_class) > delegation_class_rank(DelegationClass::kBootstrapSafe) &&
      !authority_binding_proof.has_value()) {
    return absl::FailedPreconditionError("sensitive delegated disclosure requires authority binding");
  }
  return absl::OkStatus();
}

absl::Status validate_request_delegation_envelopes(
    const v2::RoutedAuthorityRequest& routed_request,
    const StageDisclosurePolicy& disclosure_policy,
    const std::optional<AuthorityBindingProof>& authority_binding_proof) {
  const AuthorityRef audience_authority_ref = authority_ref_from_proto(routed_request.authority_ref());
  const std::string root_request_id =
      routed_request.has_request_metadata() ? routed_request.request_metadata().root_request_id() : std::string();
  const absl::Time now = absl::Now();

  if (!routed_request.has_portable_credential_envelope()) {
    return absl::FailedPreconditionError("portable_credential delegation envelope is required");
  }
  auto portable_status = validate_delegation_envelope(
      routed_request.portable_credential_envelope(),
      audience_authority_ref,
      root_request_id,
      routed_request.path_family(),
      disclosure_policy.edge_ref,
      DelegationPayloadKind::kPortableCredential,
      now);
  if (!portable_status.ok()) {
    return portable_status;
  }
  auto portable_binding_status =
      require_delegation_binding(routed_request.portable_credential_envelope(), authority_binding_proof);
  if (!portable_binding_status.ok()) {
    return portable_binding_status;
  }

  if (routed_request.has_forwardable_evidence()) {
    if (!routed_request.has_forwardable_evidence_envelope()) {
      return absl::FailedPreconditionError("forwardable_evidence delegation envelope is required");
    }
    auto forwardable_status = validate_delegation_envelope(
        routed_request.forwardable_evidence_envelope(),
        audience_authority_ref,
        root_request_id,
        routed_request.path_family(),
        disclosure_policy.edge_ref,
        DelegationPayloadKind::kForwardableEvidence,
        now);
    if (!forwardable_status.ok()) {
      return forwardable_status;
    }
    auto forwardable_binding_status =
        require_delegation_binding(routed_request.forwardable_evidence_envelope(), authority_binding_proof);
    if (!forwardable_binding_status.ok()) {
      return forwardable_binding_status;
    }
  } else if (routed_request.has_forwardable_evidence_envelope()) {
    return absl::FailedPreconditionError("forwardable_evidence delegation envelope present without evidence payload");
  }

  if (!routed_request.forwarded_claims().empty()) {
    if (!routed_request.has_forwarded_claims_envelope()) {
      return absl::FailedPreconditionError("forwarded_claims delegation envelope is required");
    }
    auto forwarded_claims_status = validate_delegation_envelope(
        routed_request.forwarded_claims_envelope(),
        audience_authority_ref,
        root_request_id,
        routed_request.path_family(),
        disclosure_policy.edge_ref,
        DelegationPayloadKind::kForwardedClaim,
        now);
    if (!forwarded_claims_status.ok()) {
      return forwarded_claims_status;
    }
    auto forwarded_claims_binding_status =
        require_delegation_binding(routed_request.forwarded_claims_envelope(), authority_binding_proof);
    if (!forwarded_claims_binding_status.ok()) {
      return forwarded_claims_binding_status;
    }
  } else if (routed_request.has_forwarded_claims_envelope()) {
    return absl::FailedPreconditionError("forwarded_claims delegation envelope present without claims");
  }

  return absl::OkStatus();
}

absl::Status validate_continuation_disclosure(
    const v2::RoutedAuthorityRequest& routed_request,
    const v2::OwnerStageReply& owner_stage_reply,
    std::string_view edge_ref) {
  if (!(routed_request.path_family() == "gate_continue_then_adopt" && routed_request.stage_ref() == "workflow_gate" &&
        edge_ref == "workflow_to_issuer")) {
    return absl::FailedPreconditionError("continuation edge is not declared for this stage");
  }
  if (!owner_stage_reply.has_continuation()) {
    return absl::DataLossError("continue_with_authority reply omitted continuation");
  }
  if (!owner_stage_reply.continuation().has_next_authority_ref()) {
    return absl::DataLossError("continuation.next_authority_ref is required");
  }
  if (owner_stage_reply.continuation().edge_ref() != edge_ref) {
    return absl::FailedPreconditionError("continuation.edge_ref mismatch");
  }
  if (!owner_stage_reply.continuation().forwarded_claims().empty()) {
    if (!owner_stage_reply.continuation().has_forwarded_claims_envelope()) {
      return absl::FailedPreconditionError("continuation forwarded_claims delegation envelope is required");
    }
    return validate_delegation_envelope(
        owner_stage_reply.continuation().forwarded_claims_envelope(),
        authority_ref_from_proto(owner_stage_reply.continuation().next_authority_ref()),
        routed_request.request_metadata().root_request_id(),
        routed_request.path_family(),
        std::optional<std::string>(std::string(edge_ref)),
        DelegationPayloadKind::kForwardedClaim,
        absl::Now());
  }
  if (owner_stage_reply.continuation().has_forwarded_claims_envelope()) {
    return absl::FailedPreconditionError("continuation forwarded_claims delegation envelope present without claims");
  }
  return absl::OkStatus();
}

} // namespace

TransportSecurityContext DistributedSecurityKernel::transport_security_context_from_server_context(
    const grpc::ServerContext& server_context) {
  TransportSecurityContext transport_security_context{
      .transport_peer = server_context.peer(),
  };
  const std::shared_ptr<const grpc::AuthContext> auth_context = server_context.auth_context();
  if (auth_context == nullptr) {
    return transport_security_context;
  }
  transport_security_context.peer_authenticated = auth_context->IsPeerAuthenticated();
  transport_security_context.peer_identity_property_name = auth_context->GetPeerIdentityPropertyName();
  for (const grpc::string_ref& identity_value : auth_context->GetPeerIdentity()) {
    transport_security_context.peer_identity_values.emplace_back(identity_value.data(), identity_value.size());
  }
  for (auto it = auth_context->begin(); it != auth_context->end(); ++it) {
    const auto [key, value] = *it;
    transport_security_context.auth_properties.emplace_back(
        std::string(key.data(), key.size()), std::string(value.data(), value.size()));
  }
  return transport_security_context;
}

TransportSecurityContext DistributedSecurityKernel::transport_security_context_from_client_context(
    const grpc::ClientContext& client_context) {
  TransportSecurityContext transport_security_context{
      .transport_peer = client_context.peer(),
  };
  const std::shared_ptr<const grpc::AuthContext> auth_context = client_context.auth_context();
  if (auth_context == nullptr) {
    return transport_security_context;
  }
  transport_security_context.peer_authenticated = auth_context->IsPeerAuthenticated();
  transport_security_context.peer_identity_property_name = auth_context->GetPeerIdentityPropertyName();
  for (const grpc::string_ref& identity_value : auth_context->GetPeerIdentity()) {
    transport_security_context.peer_identity_values.emplace_back(identity_value.data(), identity_value.size());
  }
  for (auto it = auth_context->begin(); it != auth_context->end(); ++it) {
    const auto [key, value] = *it;
    transport_security_context.auth_properties.emplace_back(
        std::string(key.data(), key.size()), std::string(value.data(), value.size()));
  }
  return transport_security_context;
}

AuthenticatedPeerIdentity DistributedSecurityKernel::derive_authenticated_peer_identity(
    const TransportSecurityContext& transport_security_context) {
  AuthenticatedPeerIdentity authenticated_peer_identity{
      .transport_peer = transport_security_context.transport_peer,
  };
  if (transport_security_context.peer_authenticated) {
    authenticated_peer_identity.peer_kind = AuthenticatedPeerKind::kDaemon;
    authenticated_peer_identity.peer_id = transport_security_context.peer_identity_values.empty()
        ? transport_security_context.transport_peer
        : transport_security_context.peer_identity_values.front();
    authenticated_peer_identity.auth_class = DaemonHopAuthClass::kDaemonMutualAuth;
    if (!transport_security_context.peer_identity_property_name.empty()) {
      authenticated_peer_identity.channel_binding_id = absl::StrCat(
          transport_security_context.peer_identity_property_name, ":", authenticated_peer_identity.peer_id);
    }
    for (const auto& identity_value : transport_security_context.peer_identity_values) {
      constexpr std::string_view kAuthorityPrefix = "spiffe://tensorcast/authority/";
      if (absl::StartsWith(identity_value, kAuthorityPrefix)) {
        authenticated_peer_identity.presented_authority_ref = AuthorityRef{
            .authority_kind = AuthorityKind::kInternalAuthority,
            .authority_id = std::string(identity_value.substr(kAuthorityPrefix.size())),
        };
        break;
      }
    }
    return authenticated_peer_identity;
  }
  if (is_loopback_grpc_peer(transport_security_context.transport_peer)) {
    authenticated_peer_identity.peer_kind = AuthenticatedPeerKind::kLocalProcess;
    authenticated_peer_identity.peer_id =
        transport_security_context.transport_peer.empty() || transport_security_context.transport_peer == "unknown"
        ? "loopback"
        : transport_security_context.transport_peer;
    authenticated_peer_identity.auth_class = DaemonHopAuthClass::kDeploymentTrustedChannel;
    return authenticated_peer_identity;
  }
  authenticated_peer_identity.peer_kind = AuthenticatedPeerKind::kAnonymousTransport;
  authenticated_peer_identity.peer_id =
      transport_security_context.transport_peer.empty() ? "unknown" : transport_security_context.transport_peer;
  authenticated_peer_identity.auth_class = DaemonHopAuthClass::kLegacyUnauthenticated;
  return authenticated_peer_identity;
}

absl::Status DistributedSecurityKernel::validate_sender_hop_auth_projection(
    const v2::RoutedAuthorityRequest& routed_request,
    const AuthenticatedPeerIdentity& authenticated_peer_identity) {
  if (!routed_request.has_hop_auth_context()) {
    return absl::OkStatus();
  }
  const auto reported_auth_class = daemon_hop_auth_class_from_proto(routed_request.hop_auth_context().auth_class());
  if (auth_class_rank(reported_auth_class) > auth_class_rank(authenticated_peer_identity.auth_class)) {
    return absl::FailedPreconditionError("sender-reported hop auth exceeds transport-derived peer auth");
  }
  if (routed_request.hop_auth_context().has_authenticated_peer_daemon_id()) {
    if (authenticated_peer_identity.auth_class != DaemonHopAuthClass::kDaemonMutualAuth ||
        routed_request.hop_auth_context().authenticated_peer_daemon_id() != authenticated_peer_identity.peer_id) {
      return absl::FailedPreconditionError("sender-reported authenticated_peer_daemon_id does not match transport");
    }
  }
  if (routed_request.hop_auth_context().has_transport_peer() &&
      routed_request.hop_auth_context().transport_peer() != authenticated_peer_identity.transport_peer) {
    return absl::FailedPreconditionError("sender-reported transport_peer does not match transport");
  }
  return absl::OkStatus();
}

StageDisclosurePolicy DistributedSecurityKernel::default_stage_disclosure_policy(
    const v2::RoutedAuthorityRequest& routed_request) {
  return StageDisclosurePolicy{
      .path_family = routed_request.path_family(),
      .stage_ref = routed_request.stage_ref(),
      .allowed_payload_kinds = {DelegationPayloadKind::kPortableCredential},
      .minimum_auth_class = DaemonHopAuthClass::kLegacyUnauthenticated,
      .accepted_claim_provenance = {ForwardedClaimProvenance::kIngressLocal},
      .continuity_class = HandoffContinuityClass::kFailClosedEphemeral,
  };
}

StageDisclosurePolicy DistributedSecurityKernel::declared_stage_disclosure_policy(
    std::string_view path_family,
    std::string_view stage_ref,
    std::optional<std::string_view> edge_ref) {
  if (path_family == "gate_continue_then_adopt" && stage_ref == "workflow_gate" && !edge_ref.has_value()) {
    return StageDisclosurePolicy{
        .path_family = std::string(path_family),
        .stage_ref = std::string(stage_ref),
        .allowed_payload_kinds =
            {
                DelegationPayloadKind::kPortableCredential,
                DelegationPayloadKind::kForwardableEvidence,
            },
        .minimum_auth_class = DaemonHopAuthClass::kDeploymentTrustedChannel,
        .accepted_claim_provenance = {ForwardedClaimProvenance::kIngressLocal},
        .continuity_class = HandoffContinuityClass::kFailClosedEphemeral,
    };
  }
  if (path_family == "gate_continue_then_adopt" && stage_ref == "issuer_validate" && edge_ref.has_value() &&
      *edge_ref == "workflow_to_issuer") {
    return StageDisclosurePolicy{
        .path_family = std::string(path_family),
        .stage_ref = std::string(stage_ref),
        .edge_ref = std::optional<std::string>(std::string(*edge_ref)),
        .allowed_payload_kinds =
            {
                DelegationPayloadKind::kPortableCredential,
                DelegationPayloadKind::kForwardableEvidence,
                DelegationPayloadKind::kForwardedClaim,
            },
        .minimum_auth_class = DaemonHopAuthClass::kDeploymentTrustedChannel,
        .accepted_claim_provenance = {ForwardedClaimProvenance::kAuthorityAuthenticated},
        .continuity_class = HandoffContinuityClass::kFailClosedEphemeral,
    };
  }
  if (path_family == "gate_continue_then_adopt" && stage_ref == "issuer_validate" && !edge_ref.has_value()) {
    return StageDisclosurePolicy{
        .path_family = std::string(path_family),
        .stage_ref = std::string(stage_ref),
        .allowed_payload_kinds =
            {
                DelegationPayloadKind::kPortableCredential,
                DelegationPayloadKind::kForwardableEvidence,
                DelegationPayloadKind::kForwardedClaim,
            },
        .minimum_auth_class = DaemonHopAuthClass::kDeploymentTrustedChannel,
        .accepted_claim_provenance = {ForwardedClaimProvenance::kAuthorityAuthenticated},
        .continuity_class = HandoffContinuityClass::kFailClosedEphemeral,
    };
  }
  if (path_family == "immediate_lowering" && stage_ref == "issuer_validate") {
    return StageDisclosurePolicy{
        .path_family = std::string(path_family),
        .stage_ref = std::string(stage_ref),
        .edge_ref = edge_ref.has_value() ? std::optional<std::string>(std::string(*edge_ref)) : std::nullopt,
        .allowed_payload_kinds =
            {
                DelegationPayloadKind::kPortableCredential,
                DelegationPayloadKind::kForwardableEvidence,
            },
        .minimum_auth_class = DaemonHopAuthClass::kDeploymentTrustedChannel,
        .accepted_claim_provenance = {ForwardedClaimProvenance::kIngressLocal},
        .continuity_class = HandoffContinuityClass::kFailClosedEphemeral,
    };
  }
  if (path_family == "gate_continue_then_adopt" && stage_ref == "workflow_gate" && edge_ref.has_value() &&
      *edge_ref == "workflow_to_issuer") {
    return StageDisclosurePolicy{
        .path_family = std::string(path_family),
        .stage_ref = std::string(stage_ref),
        .edge_ref = std::optional<std::string>(std::string(*edge_ref)),
        .allowed_payload_kinds =
            {
                DelegationPayloadKind::kPortableCredential,
                DelegationPayloadKind::kForwardedClaim,
            },
        .minimum_auth_class = DaemonHopAuthClass::kDaemonMutualAuth,
        .accepted_claim_provenance = {ForwardedClaimProvenance::kAuthorityAuthenticated},
        .continuity_class = HandoffContinuityClass::kSameAuthoritySuccessorVerified,
    };
  }
  v2::RoutedAuthorityRequest routed_request;
  routed_request.set_path_family(std::string(path_family));
  routed_request.set_stage_ref(std::string(stage_ref));
  auto disclosure_policy = default_stage_disclosure_policy(routed_request);
  disclosure_policy.edge_ref = edge_ref.has_value() ? std::optional<std::string>(std::string(*edge_ref)) : std::nullopt;
  return disclosure_policy;
}

StageDisclosurePolicy DistributedSecurityKernel::declared_stage_disclosure_policy(
    const v2::RoutedAuthorityRequest& routed_request,
    std::optional<std::string_view> edge_ref) {
  return declared_stage_disclosure_policy(routed_request.path_family(), routed_request.stage_ref(), edge_ref);
}

absl::StatusOr<AuthorityBindingProof> DistributedSecurityKernel::verify_authority_binding(
    const v2::AuthorityRef& authority_ref,
    const AuthenticatedPeerIdentity& authenticated_peer_identity,
    const std::optional<AuthorityLocatorResult>& locator_result,
    HandoffContinuityClass continuity_class) {
  return verify_authority_binding(
      authority_ref_from_proto(authority_ref), authenticated_peer_identity, locator_result, continuity_class);
}

absl::StatusOr<AuthorityBindingProof> DistributedSecurityKernel::verify_local_authority_binding(
    const v2::AuthorityRef& authority_ref,
    std::string_view local_authority_id,
    const AuthenticatedPeerIdentity& authenticated_peer_identity,
    HandoffContinuityClass continuity_class) {
  if (authority_ref.authority_id().empty() || authority_ref.authority_id() != local_authority_id) {
    return absl::UnauthenticatedError("requested authority does not match the receiving daemon");
  }
  if (continuity_class != HandoffContinuityClass::kFailClosedEphemeral) {
    return AuthorityBindingProof{
        .authority_ref = authority_ref_from_proto(authority_ref),
        .peer_identity = authenticated_peer_identity,
        .proof_kind = continuity_class == HandoffContinuityClass::kReplicatedAuthority
            ? AuthorityBindingProofKind::kAuthorityAttestedBinding
            : AuthorityBindingProofKind::kSuccessorVerifiedBinding,
        .proof_payload = std::string("receiver_local_authority_binding"),
        .issued_at = absl::Now(),
    };
  }
  return AuthorityBindingProof{
      .authority_ref = authority_ref_from_proto(authority_ref),
      .peer_identity = authenticated_peer_identity,
      .proof_kind = AuthorityBindingProofKind::kDirectPeerBinding,
      .proof_payload = std::string("receiver_local_authority_binding"),
      .issued_at = absl::Now(),
  };
}

absl::StatusOr<AuthorityBindingProof> DistributedSecurityKernel::verify_authority_binding(
    const AuthorityRef& authority_ref,
    const AuthenticatedPeerIdentity& authenticated_peer_identity,
    const std::optional<AuthorityLocatorResult>& locator_result,
    HandoffContinuityClass continuity_class) {
  if (locator_result.has_value() && locator_result->authority_ref != authority_ref) {
    return absl::InvalidArgumentError("locator result authority_ref does not match requested authority_ref");
  }
  if (authenticated_peer_identity.auth_class != DaemonHopAuthClass::kDaemonMutualAuth) {
    return absl::UnauthenticatedError("authority binding requires daemon_mutual_auth transport");
  }
  if (!authority_kind_supports_direct_daemon_binding(authority_ref.authority_kind)) {
    return absl::FailedPreconditionError("requested authority does not support direct peer binding");
  }
  if (!authenticated_peer_identity.peer_id.empty() &&
      authenticated_peer_identity.peer_id == authority_ref.authority_id) {
    return AuthorityBindingProof{
        .authority_ref = authority_ref,
        .peer_identity = authenticated_peer_identity,
        .proof_kind = AuthorityBindingProofKind::kDirectPeerBinding,
        .proof_payload = std::string("direct_peer_binding"),
        .issued_at = absl::Now(),
    };
  }
  if (continuity_class == HandoffContinuityClass::kSameAuthoritySuccessorVerified &&
      authenticated_peer_identity.presented_authority_ref.has_value() &&
      *authenticated_peer_identity.presented_authority_ref == authority_ref) {
    return AuthorityBindingProof{
        .authority_ref = authority_ref,
        .peer_identity = authenticated_peer_identity,
        .proof_kind = AuthorityBindingProofKind::kSuccessorVerifiedBinding,
        .proof_payload = std::string("presented_authority_successor_binding"),
        .issued_at = absl::Now(),
    };
  }
  if (continuity_class == HandoffContinuityClass::kReplicatedAuthority &&
      authenticated_peer_identity.presented_authority_ref.has_value() &&
      *authenticated_peer_identity.presented_authority_ref == authority_ref) {
    return AuthorityBindingProof{
        .authority_ref = authority_ref,
        .peer_identity = authenticated_peer_identity,
        .proof_kind = AuthorityBindingProofKind::kAuthorityAttestedBinding,
        .proof_payload = std::string("replicated_authority_binding"),
        .issued_at = absl::Now(),
    };
  }
  return absl::UnauthenticatedError("authenticated peer does not match requested authority");
}

absl::Status DistributedSecurityKernel::enforce_pre_disclosure_policy(
    const v2::RoutedAuthorityRequest& routed_request,
    const StageDisclosurePolicy& disclosure_policy,
    const std::optional<AuthorityBindingProof>& authority_binding_proof) {
  if (auth_class_rank(disclosure_policy.minimum_auth_class) > 0 && !authority_binding_proof.has_value()) {
    return absl::FailedPreconditionError("stage requires authority binding before disclosure");
  }

  auto delegation_status =
      validate_request_delegation_envelopes(routed_request, disclosure_policy, authority_binding_proof);
  if (!delegation_status.ok()) {
    return delegation_status;
  }

  if (routed_request.has_forwardable_evidence()) {
    if (!policy_allows_payload_kind(disclosure_policy, DelegationPayloadKind::kForwardableEvidence)) {
      return absl::FailedPreconditionError("stage disclosure policy forbids forwarding evidence on this edge");
    }
  }

  if (request_contains_sensitive_forwarded_claims(routed_request)) {
    if (!policy_allows_payload_kind(disclosure_policy, DelegationPayloadKind::kForwardedClaim)) {
      return absl::FailedPreconditionError("stage disclosure policy forbids forwarded claims on this edge");
    }
    const auto accepted_claim_provenance = make_claim_provenance_set(disclosure_policy);
    for (const auto& claim : routed_request.forwarded_claims()) {
      if (!accepted_claim_provenance.contains(claim_provenance_from_proto(claim.provenance()))) {
        return absl::FailedPreconditionError("forwarded claim provenance is not allowed by stage disclosure policy");
      }
    }
  }

  if (!policy_allows_payload_kind(disclosure_policy, DelegationPayloadKind::kPortableCredential)) {
    return absl::FailedPreconditionError("stage disclosure policy must allow portable credential bootstrap");
  }
  return absl::OkStatus();
}

absl::Status DistributedSecurityKernel::admit_reply(
    const v2::RoutedAuthorityRequest& routed_request,
    const v2::OwnerStageReply& owner_stage_reply,
    const AuthenticatedPeerIdentity& authenticated_peer_identity,
    const std::optional<AuthorityLocatorResult>& initial_locator_result,
    const std::optional<AuthorityLocatorResult>& current_locator_result,
    HandoffContinuityClass continuity_class) {
  if (!routed_request.has_authority_ref()) {
    return absl::InvalidArgumentError("routed_request.authority_ref is required for reply admission");
  }
  if (!owner_stage_reply.has_answered_by()) {
    return absl::DataLossError("owner_stage_reply.answered_by is required for reply admission");
  }
  if (owner_stage_reply.answered_by().authority_id() != routed_request.authority_ref().authority_id()) {
    return absl::FailedPreconditionError("owner_stage_reply answered_by mismatch");
  }
  if (owner_stage_reply.path_family() != routed_request.path_family()) {
    return absl::FailedPreconditionError("owner_stage_reply path_family mismatch");
  }
  if (owner_stage_reply.stage_ref() != routed_request.stage_ref()) {
    return absl::FailedPreconditionError("owner_stage_reply stage_ref mismatch");
  }

  if (!current_locator_result.has_value()) {
    return absl::UnavailableError("issuer route unavailable during reply admission");
  }
  if (continuity_class == HandoffContinuityClass::kFailClosedEphemeral && initial_locator_result.has_value() &&
      (initial_locator_result->target_daemon_id != current_locator_result->target_daemon_id ||
       initial_locator_result->target_address != current_locator_result->target_address)) {
    return absl::UnavailableError("issuer continuity lost during reply admission");
  }

  if (authenticated_peer_identity.auth_class == DaemonHopAuthClass::kDaemonMutualAuth) {
    auto binding_or = verify_authority_binding(
        routed_request.authority_ref(), authenticated_peer_identity, current_locator_result, continuity_class);
    if (!binding_or.ok()) {
      return binding_or.status();
    }
    if (owner_stage_reply.reply_kind() == v2::ROUTED_OWNER_STAGE_REPLY_KIND_CONTINUE_WITH_AUTHORITY) {
      auto continuation_status = validate_continuation_disclosure(
          routed_request, owner_stage_reply, owner_stage_reply.continuation().edge_ref());
      if (!continuation_status.ok()) {
        return continuation_status;
      }
    }
    return absl::OkStatus();
  }

  if (continuity_class == HandoffContinuityClass::kFailClosedEphemeral &&
      authenticated_peer_identity.auth_class == DaemonHopAuthClass::kDeploymentTrustedChannel &&
      is_loopback_grpc_peer(authenticated_peer_identity.transport_peer) &&
      grpc_peer_matches_address(authenticated_peer_identity.transport_peer, current_locator_result->target_address)) {
    if (owner_stage_reply.reply_kind() == v2::ROUTED_OWNER_STAGE_REPLY_KIND_CONTINUE_WITH_AUTHORITY) {
      auto continuation_status = validate_continuation_disclosure(
          routed_request, owner_stage_reply, owner_stage_reply.continuation().edge_ref());
      if (!continuation_status.ok()) {
        return continuation_status;
      }
    }
    return absl::OkStatus();
  }

  return absl::UnauthenticatedError("reply peer identity does not satisfy authority binding");
}

v2::BatchItemStatus batch_item_status_from_absl_status(const absl::Status& status) {
  switch (status.code()) {
    case absl::StatusCode::kInvalidArgument:
      return v2::BATCH_ITEM_STATUS_INVALID_ARGUMENT;
    case absl::StatusCode::kNotFound:
      return v2::BATCH_ITEM_STATUS_MISS;
    case absl::StatusCode::kUnavailable:
      return v2::BATCH_ITEM_STATUS_UNAVAILABLE;
    case absl::StatusCode::kInternal:
      return v2::BATCH_ITEM_STATUS_INTERNAL_ERROR;
    case absl::StatusCode::kUnauthenticated:
    case absl::StatusCode::kPermissionDenied:
    case absl::StatusCode::kFailedPrecondition:
    default:
      return v2::BATCH_ITEM_STATUS_FAILED_PRECONDITION;
  }
}

} // namespace tensorcast::daemon
