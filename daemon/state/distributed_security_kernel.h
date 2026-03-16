// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "daemon/state/routed_authority_protocol.h"
#include "grpcpp/client_context.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

enum class AuthenticatedPeerKind : std::uint8_t {
  kAnonymousTransport = 0,
  kLocalProcess = 1,
  kDaemon = 2,
};

enum class AuthorityBindingProofKind : std::uint8_t {
  kDirectPeerBinding = 0,
  kAuthorityAttestedBinding = 1,
  kSuccessorVerifiedBinding = 2,
};

struct TransportSecurityContext {
  std::string transport_peer;
  bool peer_authenticated{false};
  std::string peer_identity_property_name;
  std::vector<std::string> peer_identity_values;
  std::vector<std::pair<std::string, std::string>> auth_properties;
};

struct AuthenticatedPeerIdentity {
  AuthenticatedPeerKind peer_kind{AuthenticatedPeerKind::kAnonymousTransport};
  std::string peer_id;
  DaemonHopAuthClass auth_class{DaemonHopAuthClass::kLegacyUnauthenticated};
  std::optional<std::string> channel_binding_id;
  std::optional<AuthorityRef> presented_authority_ref;
  std::string transport_peer;

  bool operator==(const AuthenticatedPeerIdentity&) const = default;
};

struct AuthorityBindingProof {
  AuthorityRef authority_ref;
  AuthenticatedPeerIdentity peer_identity;
  AuthorityBindingProofKind proof_kind{AuthorityBindingProofKind::kDirectPeerBinding};
  std::optional<std::string> proof_payload;
  std::optional<absl::Time> issued_at;
  std::optional<absl::Time> expires_at;

  bool operator==(const AuthorityBindingProof&) const = default;
};

class DistributedSecurityKernel {
 public:
  [[nodiscard]] static TransportSecurityContext transport_security_context_from_server_context(
      const grpc::ServerContext& server_context);

  [[nodiscard]] static TransportSecurityContext transport_security_context_from_client_context(
      const grpc::ClientContext& client_context);

  [[nodiscard]] static AuthenticatedPeerIdentity derive_authenticated_peer_identity(
      const TransportSecurityContext& transport_security_context);

  [[nodiscard]] static absl::Status validate_sender_hop_auth_projection(
      const v2::RoutedAuthorityRequest& routed_request,
      const AuthenticatedPeerIdentity& authenticated_peer_identity);

  [[nodiscard]] static StageDisclosurePolicy default_stage_disclosure_policy(
      const v2::RoutedAuthorityRequest& routed_request);

  [[nodiscard]] static StageDisclosurePolicy declared_stage_disclosure_policy(
      std::string_view path_family,
      std::string_view stage_ref,
      std::optional<std::string_view> edge_ref = std::nullopt);

  [[nodiscard]] static StageDisclosurePolicy declared_stage_disclosure_policy(
      const v2::RoutedAuthorityRequest& routed_request,
      std::optional<std::string_view> edge_ref = std::nullopt);

  [[nodiscard]] static absl::StatusOr<AuthorityBindingProof> verify_authority_binding(
      const v2::AuthorityRef& authority_ref,
      const AuthenticatedPeerIdentity& authenticated_peer_identity,
      const std::optional<AuthorityLocatorResult>& locator_result,
      HandoffContinuityClass continuity_class);

  [[nodiscard]] static absl::StatusOr<AuthorityBindingProof> verify_local_authority_binding(
      const v2::AuthorityRef& authority_ref,
      std::string_view local_authority_id,
      const AuthenticatedPeerIdentity& authenticated_peer_identity,
      HandoffContinuityClass continuity_class);

  [[nodiscard]] static absl::StatusOr<AuthorityBindingProof> verify_authority_binding(
      const AuthorityRef& authority_ref,
      const AuthenticatedPeerIdentity& authenticated_peer_identity,
      const std::optional<AuthorityLocatorResult>& locator_result,
      HandoffContinuityClass continuity_class);

  [[nodiscard]] static absl::Status enforce_pre_disclosure_policy(
      const v2::RoutedAuthorityRequest& routed_request,
      const StageDisclosurePolicy& disclosure_policy,
      const std::optional<AuthorityBindingProof>& authority_binding_proof);

  [[nodiscard]] static absl::Status admit_reply(
      const v2::RoutedAuthorityRequest& routed_request,
      const v2::OwnerStageReply& owner_stage_reply,
      const AuthenticatedPeerIdentity& authenticated_peer_identity,
      const std::optional<AuthorityLocatorResult>& initial_locator_result,
      const std::optional<AuthorityLocatorResult>& current_locator_result,
      HandoffContinuityClass continuity_class);
};

[[nodiscard]] v2::BatchItemStatus batch_item_status_from_absl_status(const absl::Status& status);

} // namespace tensorcast::daemon
