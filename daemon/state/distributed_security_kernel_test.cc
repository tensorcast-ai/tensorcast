// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/distributed_security_kernel.h"

#include <optional>

#include <catch2/catch_test_macros.hpp>

namespace tensorcast::daemon {

namespace {

AuthorityRef make_test_authority_ref(std::string authority_id) {
  return AuthorityRef{
      .authority_kind = AuthorityKind::kIssuerDaemon,
      .authority_id = std::move(authority_id),
  };
}

v2::RoutedAuthorityRequest make_test_routed_request(std::string authority_id) {
  v2::RoutedAuthorityRequest routed_request;
  routed_request.mutable_authority_ref()->set_authority_kind(v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  routed_request.mutable_authority_ref()->set_authority_id(std::move(authority_id));
  routed_request.set_path_family("immediate_lowering");
  routed_request.set_stage_ref("issuer_validate");
  routed_request.mutable_portable_credential()->mutable_address()->set_binding_key("payload-ref:test");
  routed_request.mutable_request_metadata()->set_root_request_id("root-req-1");
  auto* portable_envelope = routed_request.mutable_portable_credential_envelope();
  portable_envelope->mutable_audience_authority_ref()->CopyFrom(routed_request.authority_ref());
  portable_envelope->set_bound_root_request_id("root-req-1");
  portable_envelope->set_bound_path_family("immediate_lowering");
  portable_envelope->set_payload_kind(v2::ROUTED_DELEGATION_PAYLOAD_KIND_PORTABLE_CREDENTIAL);
  portable_envelope->set_delegation_class(v2::ROUTED_DELEGATION_CLASS_BOOTSTRAP_SAFE);
  return routed_request;
}

v2::OwnerStageReply make_test_owner_stage_reply(std::string authority_id) {
  v2::OwnerStageReply owner_stage_reply;
  owner_stage_reply.mutable_answered_by()->set_authority_kind(v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  owner_stage_reply.mutable_answered_by()->set_authority_id(std::move(authority_id));
  owner_stage_reply.set_path_family("immediate_lowering");
  owner_stage_reply.set_stage_ref("issuer_validate");
  owner_stage_reply.set_reply_kind(v2::ROUTED_OWNER_STAGE_REPLY_KIND_READY_FOR_LOWERING);
  owner_stage_reply.set_resolved_source_capability("{}");
  return owner_stage_reply;
}

} // namespace

TEST_CASE("loopback transport derives deployment trusted peer identity", "[daemon][security_kernel]") {
  const auto peer_identity = DistributedSecurityKernel::derive_authenticated_peer_identity(
      TransportSecurityContext{
          .transport_peer = "unix:/tmp/tensorcast.sock",
          .peer_authenticated = false,
      });

  CHECK(peer_identity.peer_kind == AuthenticatedPeerKind::kLocalProcess);
  CHECK(peer_identity.auth_class == DaemonHopAuthClass::kDeploymentTrustedChannel);
  CHECK(peer_identity.peer_id == "unix:/tmp/tensorcast.sock");
  CHECK_FALSE(peer_identity.channel_binding_id.has_value());
}

TEST_CASE("authority binding uses authenticated peer identity instead of locator output", "[daemon][security_kernel]") {
  const AuthorityRef authority_ref = make_test_authority_ref("daemon-a");
  const AuthorityLocatorResult locator_result{
      .authority_ref = authority_ref,
      .target_daemon_id = "daemon-a",
      .target_address = "10.0.0.1:50051",
      .resolved_at = absl::Now(),
  };
  const AuthenticatedPeerIdentity peer_identity{
      .peer_kind = AuthenticatedPeerKind::kDaemon,
      .peer_id = "daemon-b",
      .auth_class = DaemonHopAuthClass::kDaemonMutualAuth,
      .channel_binding_id = "spiffe://daemon-b",
      .transport_peer = "ipv4:10.0.0.2:50051",
  };

  auto binding_or = DistributedSecurityKernel::verify_authority_binding(
      authority_ref, peer_identity, locator_result, HandoffContinuityClass::kFailClosedEphemeral);
  REQUIRE_FALSE(binding_or.ok());
  CHECK(binding_or.status().code() == absl::StatusCode::kUnauthenticated);
}

TEST_CASE("direct peer binding succeeds only for exact daemon authority match", "[daemon][security_kernel]") {
  const AuthorityRef authority_ref = make_test_authority_ref("daemon-a");
  const AuthenticatedPeerIdentity peer_identity{
      .peer_kind = AuthenticatedPeerKind::kDaemon,
      .peer_id = "daemon-a",
      .auth_class = DaemonHopAuthClass::kDaemonMutualAuth,
      .channel_binding_id = "spiffe://daemon-a",
      .transport_peer = "ipv4:10.0.0.1:50051",
  };

  auto binding_or = DistributedSecurityKernel::verify_authority_binding(
      authority_ref, peer_identity, std::nullopt, HandoffContinuityClass::kFailClosedEphemeral);
  REQUIRE(binding_or.ok());
  CHECK(binding_or->proof_kind == AuthorityBindingProofKind::kDirectPeerBinding);
  CHECK(binding_or->authority_ref == authority_ref);
  CHECK(binding_or->peer_identity == peer_identity);
}

TEST_CASE("sensitive routed disclosure is blocked before authority binding", "[daemon][security_kernel]") {
  v2::RoutedAuthorityRequest routed_request;
  routed_request.mutable_authority_ref()->set_authority_kind(v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  routed_request.mutable_authority_ref()->set_authority_id("daemon-a");
  routed_request.set_path_family("gate_continue_then_adopt");
  routed_request.set_stage_ref("issuer_validate");
  routed_request.mutable_portable_credential()->mutable_address()->set_binding_key("payload-ref:test");
  routed_request.mutable_request_metadata()->set_root_request_id("root-req-sensitive");
  auto* portable_envelope = routed_request.mutable_portable_credential_envelope();
  portable_envelope->mutable_audience_authority_ref()->CopyFrom(routed_request.authority_ref());
  portable_envelope->set_bound_root_request_id("root-req-sensitive");
  portable_envelope->set_bound_path_family("gate_continue_then_adopt");
  portable_envelope->set_payload_kind(v2::ROUTED_DELEGATION_PAYLOAD_KIND_PORTABLE_CREDENTIAL);
  portable_envelope->set_delegation_class(v2::ROUTED_DELEGATION_CLASS_BOOTSTRAP_SAFE);
  routed_request.mutable_forwardable_evidence()->set_evidence_kind(v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_RAW_CREDENTIAL);
  routed_request.mutable_forwardable_evidence()->set_raw_credential_bytes("payload-ref-token");
  auto* evidence_envelope = routed_request.mutable_forwardable_evidence_envelope();
  evidence_envelope->mutable_audience_authority_ref()->CopyFrom(routed_request.authority_ref());
  evidence_envelope->set_bound_root_request_id("root-req-sensitive");
  evidence_envelope->set_bound_path_family("gate_continue_then_adopt");
  evidence_envelope->set_payload_kind(v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDABLE_EVIDENCE);
  evidence_envelope->set_delegation_class(v2::ROUTED_DELEGATION_CLASS_OWNER_SCOPED_SENSITIVE);

  const StageDisclosurePolicy disclosure_policy =
      DistributedSecurityKernel::default_stage_disclosure_policy(routed_request);
  auto disclosure_status =
      DistributedSecurityKernel::enforce_pre_disclosure_policy(routed_request, disclosure_policy, std::nullopt);
  REQUIRE_FALSE(disclosure_status.ok());
  CHECK(disclosure_status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE(
    "delegated routed disclosure succeeds only when envelopes match audience and binding",
    "[daemon][security_kernel]") {
  auto routed_request = make_test_routed_request("daemon-a");
  routed_request.mutable_forwardable_evidence()->set_evidence_kind(v2::ROUTED_CREDENTIAL_EVIDENCE_KIND_RAW_CREDENTIAL);
  routed_request.mutable_forwardable_evidence()->set_raw_credential_bytes("payload-ref-token");
  auto* evidence_envelope = routed_request.mutable_forwardable_evidence_envelope();
  evidence_envelope->mutable_audience_authority_ref()->CopyFrom(routed_request.authority_ref());
  evidence_envelope->set_bound_root_request_id("root-req-1");
  evidence_envelope->set_bound_path_family("immediate_lowering");
  evidence_envelope->set_payload_kind(v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDABLE_EVIDENCE);
  evidence_envelope->set_delegation_class(v2::ROUTED_DELEGATION_CLASS_OWNER_SCOPED_SENSITIVE);

  const AuthenticatedPeerIdentity peer_identity{
      .peer_kind = AuthenticatedPeerKind::kDaemon,
      .peer_id = "daemon-a",
      .auth_class = DaemonHopAuthClass::kDaemonMutualAuth,
      .transport_peer = "ipv4:10.0.0.1:50051",
  };
  auto binding_or = DistributedSecurityKernel::verify_authority_binding(
      make_test_authority_ref("daemon-a"), peer_identity, std::nullopt, HandoffContinuityClass::kFailClosedEphemeral);
  REQUIRE(binding_or.ok());

  const auto disclosure_policy = DistributedSecurityKernel::declared_stage_disclosure_policy(routed_request);
  auto disclosure_status =
      DistributedSecurityKernel::enforce_pre_disclosure_policy(routed_request, disclosure_policy, *binding_or);
  REQUIRE(disclosure_status.ok());

  routed_request.mutable_forwardable_evidence_envelope()->mutable_audience_authority_ref()->set_authority_id(
      "daemon-b");
  auto bad_disclosure_status =
      DistributedSecurityKernel::enforce_pre_disclosure_policy(routed_request, disclosure_policy, *binding_or);
  REQUIRE_FALSE(bad_disclosure_status.ok());
  CHECK(bad_disclosure_status.code() == absl::StatusCode::kPermissionDenied);
}

TEST_CASE("reply admission fails closed when shape matches but peer binding is invalid", "[daemon][security_kernel]") {
  const auto routed_request = make_test_routed_request("daemon-a");
  const auto owner_stage_reply = make_test_owner_stage_reply("daemon-a");
  const AuthenticatedPeerIdentity peer_identity{
      .peer_kind = AuthenticatedPeerKind::kAnonymousTransport,
      .peer_id = "10.0.0.9:50051",
      .auth_class = DaemonHopAuthClass::kLegacyUnauthenticated,
      .transport_peer = "ipv4:10.0.0.9:50051",
  };
  const AuthorityLocatorResult locator_result{
      .authority_ref = make_test_authority_ref("daemon-a"),
      .target_daemon_id = "daemon-a",
      .target_address = "10.0.0.1:50051",
      .resolved_at = absl::Now(),
  };

  auto admission_status = DistributedSecurityKernel::admit_reply(
      routed_request,
      owner_stage_reply,
      peer_identity,
      locator_result,
      locator_result,
      HandoffContinuityClass::kFailClosedEphemeral);
  REQUIRE_FALSE(admission_status.ok());
  CHECK(admission_status.code() == absl::StatusCode::kUnauthenticated);
}

TEST_CASE("reply admission fails closed when continuity changes after route resolution", "[daemon][security_kernel]") {
  const auto routed_request = make_test_routed_request("daemon-a");
  const auto owner_stage_reply = make_test_owner_stage_reply("daemon-a");
  const AuthenticatedPeerIdentity peer_identity{
      .peer_kind = AuthenticatedPeerKind::kLocalProcess,
      .peer_id = "ipv4:127.0.0.1:50051",
      .auth_class = DaemonHopAuthClass::kDeploymentTrustedChannel,
      .transport_peer = "ipv4:127.0.0.1:50051",
  };
  const AuthorityLocatorResult initial_locator_result{
      .authority_ref = make_test_authority_ref("daemon-a"),
      .target_daemon_id = "daemon-a",
      .target_address = "127.0.0.1:50051",
      .resolved_at = absl::Now(),
  };
  const AuthorityLocatorResult current_locator_result{
      .authority_ref = make_test_authority_ref("daemon-a"),
      .target_daemon_id = "daemon-a",
      .target_address = "127.0.0.1:50052",
      .resolved_at = absl::Now(),
  };

  auto admission_status = DistributedSecurityKernel::admit_reply(
      routed_request,
      owner_stage_reply,
      peer_identity,
      initial_locator_result,
      current_locator_result,
      HandoffContinuityClass::kFailClosedEphemeral);
  REQUIRE_FALSE(admission_status.ok());
  CHECK(admission_status.code() == absl::StatusCode::kUnavailable);
}

TEST_CASE("successor continuity accepts presented authority proof", "[daemon][security_kernel]") {
  const AuthorityRef authority_ref = make_test_authority_ref("authority-a");
  const AuthenticatedPeerIdentity peer_identity{
      .peer_kind = AuthenticatedPeerKind::kDaemon,
      .peer_id = "daemon-successor-1",
      .auth_class = DaemonHopAuthClass::kDaemonMutualAuth,
      .presented_authority_ref = authority_ref,
      .transport_peer = "ipv4:10.0.0.9:50051",
  };

  auto binding_or = DistributedSecurityKernel::verify_authority_binding(
      authority_ref, peer_identity, std::nullopt, HandoffContinuityClass::kSameAuthoritySuccessorVerified);
  REQUIRE(binding_or.ok());
  CHECK(binding_or->proof_kind == AuthorityBindingProofKind::kSuccessorVerifiedBinding);
}

TEST_CASE("continuation edge widening is rejected", "[daemon][security_kernel]") {
  v2::RoutedAuthorityRequest routed_request;
  routed_request.mutable_authority_ref()->set_authority_kind(v2::ROUTED_AUTHORITY_KIND_WORKFLOW_OWNER);
  routed_request.mutable_authority_ref()->set_authority_id("workflow-owner-a");
  routed_request.set_path_family("gate_continue_then_adopt");
  routed_request.set_stage_ref("workflow_gate");
  routed_request.mutable_portable_credential()->mutable_address()->set_binding_key("payload-ref:test");
  routed_request.mutable_request_metadata()->set_root_request_id("root-req-edge");
  auto* portable_envelope = routed_request.mutable_portable_credential_envelope();
  portable_envelope->mutable_audience_authority_ref()->CopyFrom(routed_request.authority_ref());
  portable_envelope->set_bound_root_request_id("root-req-edge");
  portable_envelope->set_bound_path_family("gate_continue_then_adopt");
  portable_envelope->set_payload_kind(v2::ROUTED_DELEGATION_PAYLOAD_KIND_PORTABLE_CREDENTIAL);
  portable_envelope->set_delegation_class(v2::ROUTED_DELEGATION_CLASS_BOOTSTRAP_SAFE);

  v2::OwnerStageReply owner_stage_reply;
  owner_stage_reply.mutable_answered_by()->CopyFrom(routed_request.authority_ref());
  owner_stage_reply.set_path_family("gate_continue_then_adopt");
  owner_stage_reply.set_stage_ref("workflow_gate");
  owner_stage_reply.set_reply_kind(v2::ROUTED_OWNER_STAGE_REPLY_KIND_CONTINUE_WITH_AUTHORITY);
  owner_stage_reply.mutable_continuation()->mutable_next_authority_ref()->set_authority_kind(
      v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  owner_stage_reply.mutable_continuation()->mutable_next_authority_ref()->set_authority_id("daemon-a");
  owner_stage_reply.mutable_continuation()->set_edge_ref("undeclared_edge");
  auto* claim = owner_stage_reply.mutable_continuation()->add_forwarded_claims();
  claim->set_claim_kind("workflow_gate");
  claim->set_provenance(v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_AUTHORITY_AUTHENTICATED);
  claim->set_claim_payload("approved");
  claim->mutable_minted_by_authority_ref()->CopyFrom(routed_request.authority_ref());
  claim->mutable_audience_authority_ref()->CopyFrom(owner_stage_reply.continuation().next_authority_ref());
  claim->set_bound_root_request_id("root-req-edge");
  auto* claims_envelope = owner_stage_reply.mutable_continuation()->mutable_forwarded_claims_envelope();
  claims_envelope->mutable_audience_authority_ref()->CopyFrom(owner_stage_reply.continuation().next_authority_ref());
  claims_envelope->set_bound_root_request_id("root-req-edge");
  claims_envelope->set_bound_path_family("gate_continue_then_adopt");
  claims_envelope->set_bound_edge("undeclared_edge");
  claims_envelope->set_payload_kind(v2::ROUTED_DELEGATION_PAYLOAD_KIND_FORWARDED_CLAIM);
  claims_envelope->set_delegation_class(v2::ROUTED_DELEGATION_CLASS_OWNER_SCOPED_SENSITIVE);

  const AuthenticatedPeerIdentity peer_identity{
      .peer_kind = AuthenticatedPeerKind::kDaemon,
      .peer_id = "workflow-owner-a",
      .auth_class = DaemonHopAuthClass::kDaemonMutualAuth,
      .transport_peer = "ipv4:10.0.0.4:50051",
  };
  const AuthorityLocatorResult locator_result{
      .authority_ref =
          AuthorityRef{
              .authority_kind = AuthorityKind::kWorkflowOwner,
              .authority_id = "workflow-owner-a",
          },
      .target_daemon_id = "workflow-owner-a",
      .target_address = "10.0.0.4:50051",
      .resolved_at = absl::Now(),
  };

  auto admission_status = DistributedSecurityKernel::admit_reply(
      routed_request,
      owner_stage_reply,
      peer_identity,
      locator_result,
      locator_result,
      HandoffContinuityClass::kSameAuthoritySuccessorVerified);
  REQUIRE_FALSE(admission_status.ok());
  CHECK(admission_status.code() == absl::StatusCode::kFailedPrecondition);
}

} // namespace tensorcast::daemon
