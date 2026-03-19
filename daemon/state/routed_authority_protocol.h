// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "daemon/state/lifecycle_kernel.h"

namespace tensorcast::daemon {

struct ResolvedSourceCapability;

enum class AuthorityKind : std::uint8_t {
  kCatalogOwner = 0,
  kShardHomeDaemon = 1,
  kWorkflowOwner = 2,
  kOperationOwner = 3,
  kIssuerDaemon = 4,
  kInternalAuthority = 5,
};

struct AuthorityRef {
  AuthorityKind authority_kind{AuthorityKind::kInternalAuthority};
  std::string authority_id;
  std::optional<FencingContext> fencing_context;

  bool operator==(const AuthorityRef&) const = default;
};

struct AuthorityLocatorResult {
  AuthorityRef authority_ref;
  std::string target_daemon_id;
  std::string target_address;
  absl::Time resolved_at{absl::InfinitePast()};
  absl::Duration staleness_budget{absl::ZeroDuration()};

  bool operator==(const AuthorityLocatorResult&) const = default;
};

class AuthorityLocator {
 public:
  virtual ~AuthorityLocator() = default;

  virtual absl::Status warm_authorities(
      absl::Span<const AuthorityRef> authority_refs,
      absl::Time now,
      absl::Duration staleness_budget) = 0;

  [[nodiscard]] virtual absl::StatusOr<AuthorityLocatorResult> resolve_authority(
      const AuthorityRef& authority_ref,
      absl::Time now,
      absl::Duration staleness_budget) = 0;
};

enum class DaemonHopAuthClass : std::uint8_t {
  kLegacyUnauthenticated = 0,
  kDeploymentTrustedChannel = 1,
  kDaemonMutualAuth = 2,
};

struct DaemonHopAuthContext {
  DaemonHopAuthClass auth_class{DaemonHopAuthClass::kLegacyUnauthenticated};
  std::optional<std::string> authenticated_peer_daemon_id;
  std::optional<std::string> transport_peer;

  bool operator==(const DaemonHopAuthContext&) const = default;
};

enum class ForwardedClaimProvenance : std::uint8_t {
  kAuthorityAuthenticated = 0,
  kRelayAuthenticated = 1,
  kIngressLocal = 2,
};

struct ForwardedClaim {
  std::string claim_kind;
  ForwardedClaimProvenance provenance{ForwardedClaimProvenance::kIngressLocal};
  std::string claim_payload;
  AuthorityRef minted_by_authority_ref;
  AuthorityRef audience_authority_ref;
  std::string bound_root_request_id;
  std::optional<std::string> bound_credential_binding_digest;
  std::optional<std::string> bound_path_family;
  std::optional<std::string> bound_edge;
  std::optional<absl::Time> claim_expires_at;
  std::optional<std::string> claim_authenticator;

  bool operator==(const ForwardedClaim&) const = default;
};

enum class DelegationPayloadKind : std::uint8_t {
  kPortableCredential = 0,
  kForwardableEvidence = 1,
  kForwardedClaim = 2,
};

enum class DelegationClass : std::uint8_t {
  kBootstrapSafe = 0,
  kOwnerScopedSensitive = 1,
  kIssuerSecret = 2,
};

enum class HandoffContinuityClass : std::uint8_t {
  kFailClosedEphemeral = 0,
  kSameAuthoritySuccessorVerified = 1,
  kReplicatedAuthority = 2,
};

struct DelegationEnvelope {
  AuthorityRef audience_authority_ref;
  std::string bound_root_request_id;
  std::optional<std::string> bound_path_family;
  std::optional<std::string> bound_edge;
  DelegationPayloadKind payload_kind{DelegationPayloadKind::kPortableCredential};
  DelegationClass delegation_class{DelegationClass::kBootstrapSafe};
  std::optional<absl::Time> not_before;
  std::optional<absl::Time> expires_at;
  std::optional<std::string> authenticator;

  bool operator==(const DelegationEnvelope&) const = default;
};

struct StageDisclosurePolicy {
  std::string path_family;
  std::string stage_ref;
  std::optional<std::string> edge_ref;
  std::vector<DelegationPayloadKind> allowed_payload_kinds;
  DaemonHopAuthClass minimum_auth_class{DaemonHopAuthClass::kLegacyUnauthenticated};
  std::vector<ForwardedClaimProvenance> accepted_claim_provenance;
  HandoffContinuityClass continuity_class{HandoffContinuityClass::kFailClosedEphemeral};

  bool operator==(const StageDisclosurePolicy&) const = default;
};

struct RoutedRequestMetadata {
  std::string root_request_id;
  std::optional<absl::Time> deadline;
  std::optional<std::string> trace_context;
  std::optional<std::string> idempotency_key;
  std::optional<std::string> credential_binding_digest;
  std::uint32_t hop_budget_remaining{0};
  std::uint32_t retry_attempt{0};

  bool operator==(const RoutedRequestMetadata&) const = default;
};

struct RoutedAuthorityRequest {
  AuthorityRef authority_ref;
  std::string path_family;
  std::string stage_ref;
  PortableParsedCredential portable_credential;
  std::optional<ForwardableCredentialEvidence> forwardable_evidence;
  std::optional<DelegationEnvelope> portable_credential_envelope;
  std::optional<DelegationEnvelope> forwardable_evidence_envelope;
  DaemonHopAuthContext hop_auth_context;
  std::vector<ForwardedClaim> forwarded_claims;
  std::optional<DelegationEnvelope> forwarded_claims_envelope;
  RoutedRequestMetadata request_metadata;

  bool operator==(const RoutedAuthorityRequest&) const = default;
};

struct AuthorityAttachmentRef {
  AuthorityRef authority_ref;
  std::string attachment_kind;
  std::string attachment_id;
  std::optional<FencingContext> fencing_context;

  bool operator==(const AuthorityAttachmentRef&) const = default;
};

struct AuthorityContinuation {
  AuthorityRef next_authority_ref;
  std::string edge_ref;
  std::vector<ForwardedClaim> forwarded_claims;
  std::optional<DelegationEnvelope> forwarded_claims_envelope;
  std::optional<std::string> continuation_reason;

  bool operator==(const AuthorityContinuation&) const = default;
};

enum class TerminalProjectionKind : std::uint8_t {
  kSemanticSuccess = 0,
  kSemanticReject = 1,
  kStatusSnapshot = 2,
  kFamilyDefined = 3,
};

struct TerminalProjection {
  TerminalProjectionKind projection_kind{TerminalProjectionKind::kStatusSnapshot};
  std::optional<std::string> status_code;
  std::optional<std::string> family_payload;

  bool operator==(const TerminalProjection&) const = default;
};

enum class OwnerStageReplyKind : std::uint8_t {
  kReadyForLowering = 0,
  kContinueWithAuthority = 1,
  kRetryLater = 2,
  kAttachExisting = 3,
  kTerminal = 4,
};

struct OwnerStageReply {
  AuthorityRef answered_by;
  std::string path_family;
  std::string stage_ref;
  OwnerStageReplyKind reply_kind{OwnerStageReplyKind::kRetryLater};
  std::shared_ptr<ResolvedSourceCapability> resolved_source_capability;
  std::optional<AuthorityContinuation> continuation;
  std::optional<AuthorityAttachmentRef> attachment_ref;
  std::optional<TerminalProjection> terminal_projection;
  std::optional<std::string> diagnostics;
};

} // namespace tensorcast::daemon
