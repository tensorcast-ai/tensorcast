// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/lifecycle_kernel.h"
#include "daemon/state/routed_authority_protocol.h"
#include "daemon/state/routed_authority_wire.h"

#include <catch2/catch_test_macros.hpp>

namespace tensorcast::daemon {

TEST_CASE("LifecycleKernel admits address-derived credentials", "[daemon][lifecycle_kernel]") {
  LifecycleKernel kernel("daemon-test");

  LifecycleSubjectRecord subject;
  subject.subject_id = "inline-snapshot:test";
  subject.epochs.subject_generation = 1;
  subject.subject_kind = LifecycleSubjectKind::kInlineSnapshot;
  subject.created_at = absl::Now();
  subject.last_observed_at = subject.created_at;
  subject.artifact_id = "artifact-a";
  subject.semantic_ref_id = "payload-a";

  MintCapabilityRequest request{
      .subject = subject,
      .address =
          CapabilityBindingAddress{
              .route_principal = make_issuer_route_principal("daemon-test"),
              .family = LifecycleCapabilityFamily::kServe,
              .binding_space = LifecycleBindingSpace::kPayload,
              .binding_key_kind = BindingKeyKind::kPayloadId,
              .binding_key = "payload-a",
              .epochs = subject.epochs,
          },
      .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
      .capability_id = "payload-ref:payload-a",
      .lease_id = 11,
      .capability_expires_at = absl::Now() + absl::Minutes(1),
      .carriage_kind = CredentialCarriageKind::kSelfDescribing,
      .binding_mode = LifecycleBindingMode::kAddressDerived,
      .constraint_claims = ConstraintClaims{.artifact_id = "artifact-a"},
  };
  auto mint_or = kernel.mint_capability(request);
  REQUIRE(mint_or.ok());

  auto admitted_or = kernel.admit_redemption(
      ParsedCredential{
          .address = request.address,
          .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
          .credential_expires_at = absl::Now() + absl::Minutes(1),
          .carriage_kind = CredentialCarriageKind::kSelfDescribing,
          .binding_mode = LifecycleBindingMode::kAddressDerived,
          .constraint_claims = ConstraintClaims{.artifact_id = "artifact-a"},
      });
  REQUIRE(admitted_or.ok());
  CHECK(admitted_or->capability.capability_id == "payload-ref:payload-a");

  REQUIRE(kernel.release_use_guard(admitted_or->use_guard).ok());
  REQUIRE(kernel.release_capability("payload-ref:payload-a").ok());

  auto stale_or = kernel.admit_redemption(
      ParsedCredential{
          .address = request.address,
          .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
          .credential_expires_at = absl::Now() + absl::Minutes(1),
          .carriage_kind = CredentialCarriageKind::kSelfDescribing,
          .binding_mode = LifecycleBindingMode::kAddressDerived,
          .constraint_claims = ConstraintClaims{.artifact_id = "artifact-a"},
      });
  CHECK_FALSE(stale_or.ok());
}

TEST_CASE("LifecycleKernel admits binding-record credentials", "[daemon][lifecycle_kernel]") {
  LifecycleKernel kernel("daemon-test");

  LifecycleSubjectRecord subject;
  subject.subject_id = "backing:artifact-b";
  subject.epochs.subject_generation = 42;
  subject.subject_kind = LifecycleSubjectKind::kBacking;
  subject.created_at = absl::Now();
  subject.last_observed_at = subject.created_at;
  subject.artifact_id = "artifact-b";
  subject.semantic_ref_id = "backing:artifact-b";

  MintCapabilityRequest request{
      .subject = subject,
      .address =
          CapabilityBindingAddress{
              .route_principal = make_issuer_route_principal("daemon-test"),
              .family = LifecycleCapabilityFamily::kExport,
              .binding_space = LifecycleBindingSpace::kExportHandle,
              .binding_key_kind = BindingKeyKind::kOpaqueLocalToken,
              .binding_key = "token-b",
              .epochs = subject.epochs,
              .binding_id = "token-b",
          },
      .front_door_kind = LifecycleFrontDoorKind::kLocalCpuMemfdExport,
      .capability_id = "export:token-b",
      .lease_id = 12,
      .capability_expires_at = absl::InfiniteFuture(),
      .carriage_kind = CredentialCarriageKind::kOpaqueLocalCompat,
      .binding_mode = LifecycleBindingMode::kBindingRecord,
      .constraint_claims = ConstraintClaims{.artifact_id = "artifact-b", .local_only = true},
      .credential_expires_at = absl::InfiniteFuture(),
      .binding_id = "token-b",
      .local_only = true,
  };
  auto mint_or = kernel.mint_capability(request);
  REQUIRE(mint_or.ok());

  auto admitted_or = kernel.admit_redemption(
      ParsedCredential{
          .address = request.address,
          .front_door_kind = LifecycleFrontDoorKind::kLocalCpuMemfdExport,
          .credential_expires_at = absl::InfiniteFuture(),
          .carriage_kind = CredentialCarriageKind::kOpaqueLocalCompat,
          .binding_mode = LifecycleBindingMode::kBindingRecord,
          .constraint_claims = ConstraintClaims{.artifact_id = "artifact-b", .local_only = true},
      });
  REQUIRE(admitted_or.ok());
  CHECK(admitted_or->binding_record.has_value());
  CHECK(admitted_or->binding_record->binding_id == "token-b");

  REQUIRE(kernel.release_use_guard(admitted_or->use_guard).ok());
}

TEST_CASE(
    "FrontDoorCredentialContext keeps forwardable evidence separate from local observations",
    "[daemon][front_door]") {
  FrontDoorCredentialContext context{
      .parsed_credential =
          ParsedCredential{
              .address =
                  CapabilityBindingAddress{
                      .route_principal = make_issuer_route_principal("daemon-test"),
                      .family = LifecycleCapabilityFamily::kServe,
                      .binding_space = LifecycleBindingSpace::kPayload,
                      .binding_key_kind = BindingKeyKind::kPayloadId,
                      .binding_key = "payload-ref:test",
                      .epochs = LifecycleEpochs{.subject_generation = 7},
                  },
              .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
              .credential_expires_at = absl::Now() + absl::Seconds(30),
              .carriage_kind = CredentialCarriageKind::kSelfDescribing,
              .binding_mode = LifecycleBindingMode::kAddressDerived,
              .constraint_claims = ConstraintClaims{.artifact_id = "artifact-front-door"},
          },
      .forwardable_evidence =
          ForwardableCredentialEvidence{
              .evidence_kind = CredentialEvidenceKind::kRawCredential,
              .raw_credential_bytes = "payload-ref-token",
          },
      .local_observations =
          LocalObservationSet{
              .observations = {LocalObservation{
                  .observation_kind = "peer_pid",
                  .observation_payload = "1234",
              }},
          },
  };

  REQUIRE(context.forwardable_evidence.has_value());
  CHECK(context.forwardable_evidence->evidence_kind == CredentialEvidenceKind::kRawCredential);
  CHECK(context.forwardable_evidence->raw_credential_bytes == std::optional<std::string>("payload-ref-token"));
  REQUIRE_FALSE(context.local_observations.empty());
  CHECK(context.local_observations.observations.front().observation_kind == "peer_pid");
  CHECK(context.parsed_credential.address.binding_key == "payload-ref:test");
}

TEST_CASE("Opaque local compatibility front doors remain non-forwardable by default", "[daemon][front_door]") {
  FrontDoorCredentialContext context{
      .parsed_credential =
          ParsedCredential{
              .address =
                  CapabilityBindingAddress{
                      .route_principal = make_issuer_route_principal("daemon-test"),
                      .family = LifecycleCapabilityFamily::kExport,
                      .binding_space = LifecycleBindingSpace::kExportHandle,
                      .binding_key_kind = BindingKeyKind::kOpaqueLocalToken,
                      .binding_key = "compat-local-token",
                      .epochs = LifecycleEpochs{.subject_generation = 1},
                  },
              .front_door_kind = LifecycleFrontDoorKind::kLocalCpuMemfdExport,
              .credential_expires_at = absl::Now() + absl::Seconds(30),
              .carriage_kind = CredentialCarriageKind::kOpaqueLocalCompat,
              .binding_mode = LifecycleBindingMode::kBindingRecord,
              .constraint_claims = ConstraintClaims{.artifact_id = "artifact-local-compat", .local_only = true},
          },
      .forwardable_evidence = std::nullopt,
      .local_observations =
          LocalObservationSet{
              .observations = {LocalObservation{
                  .observation_kind = "holder_pid",
                  .observation_payload = "4321",
              }},
          },
  };

  CHECK(context.parsed_credential.carriage_kind == CredentialCarriageKind::kOpaqueLocalCompat);
  CHECK_FALSE(context.forwardable_evidence.has_value());
  REQUIRE_FALSE(context.local_observations.empty());
  CHECK(context.local_observations.observations.front().observation_kind == "holder_pid");
}

TEST_CASE("Queue wait and fencing outcomes stay in workflow vocabulary", "[daemon][workflow][queue]") {
  const WorkflowCompanionRef queue_workflow_ref{
      .owner_kind = WorkflowOwnerKind::kQueue,
      .workflow_id = "queue-workflow-a",
      .currentness_key = std::optional<std::string>("queue-key-a"),
      .operation_id = std::nullopt,
      .fencing_context =
          FencingContext{
              .principal_kind = FencingPrincipalKind::kQueueLeader,
              .principal_id = "queue-leader-a",
              .epoch = 17,
          },
  };

  const WorkflowObservationResult not_ready_wait{
      .observation_kind = WorkflowObservationKind::kWait,
      .resolved_workflow_ref = queue_workflow_ref,
      .outcome_projection = std::nullopt,
      .ready = false,
      .diagnostics = std::string("queue_or_visibility_wait"),
  };

  const WorkflowGateDecision fenced_gate{
      .decision_class = WorkflowDecisionClass::kFenced,
      .resolved_workflow_ref = queue_workflow_ref,
      .binding_projection = std::nullopt,
      .outcome_projection = std::nullopt,
      .diagnostics = std::string("queue_epoch_mismatch"),
  };

  CHECK(not_ready_wait.resolved_workflow_ref.owner_kind == WorkflowOwnerKind::kQueue);
  REQUIRE(not_ready_wait.resolved_workflow_ref.fencing_context.has_value());
  CHECK(not_ready_wait.resolved_workflow_ref.fencing_context->principal_kind == FencingPrincipalKind::kQueueLeader);
  CHECK(not_ready_wait.ready == std::optional<bool>(false));
  CHECK(not_ready_wait.diagnostics == std::optional<std::string>("queue_or_visibility_wait"));

  CHECK(fenced_gate.decision_class == WorkflowDecisionClass::kFenced);
  CHECK(fenced_gate.resolved_workflow_ref.owner_kind == WorkflowOwnerKind::kQueue);
  CHECK(fenced_gate.diagnostics == std::optional<std::string>("queue_epoch_mismatch"));
}

TEST_CASE("Routed authority protocol carriers expose canonical request and reply skeleton", "[daemon][routing]") {
  const AuthorityRef authority_ref{
      .authority_kind = AuthorityKind::kIssuerDaemon,
      .authority_id = "issuer-daemon-a",
      .fencing_context =
          FencingContext{
              .principal_kind = FencingPrincipalKind::kIssuerDaemon,
              .principal_id = "issuer-daemon-a",
              .epoch = 9,
          },
  };

  RoutedAuthorityRequest request{
      .authority_ref = authority_ref,
      .path_family = "gate_continue_then_adopt",
      .stage_ref = "issuer_validate",
      .portable_credential =
          PortableParsedCredential{
              .address =
                  CapabilityBindingAddress{
                      .route_principal = make_issuer_route_principal("issuer-daemon-a"),
                      .family = LifecycleCapabilityFamily::kServe,
                      .binding_space = LifecycleBindingSpace::kPayload,
                      .binding_key_kind = BindingKeyKind::kPayloadId,
                      .binding_key = "payload-ref:test",
                      .epochs = LifecycleEpochs{.subject_generation = 7},
                  },
              .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
              .credential_expires_at = absl::Now() + absl::Minutes(1),
              .binding_mode = LifecycleBindingMode::kAddressDerived,
              .portable_constraint_claims = ConstraintClaims{.artifact_id = "artifact-front-door"},
          },
      .hop_auth_context =
          DaemonHopAuthContext{
              .auth_class = DaemonHopAuthClass::kDaemonMutualAuth,
              .authenticated_peer_daemon_id = "daemon-b",
              .transport_peer = "10.0.0.2:12345",
          },
      .request_metadata =
          RoutedRequestMetadata{
              .root_request_id = "root-req-1",
              .idempotency_key = "idem-1",
              .hop_budget_remaining = 3,
          },
  };

  request.forwarded_claims.push_back(
      ForwardedClaim{
          .claim_kind = "workflow_gate",
          .provenance = ForwardedClaimProvenance::kIngressLocal,
          .claim_payload = "approved",
          .minted_by_authority_ref = authority_ref,
          .audience_authority_ref = authority_ref,
          .bound_root_request_id = "root-req-1",
      });

  OwnerStageReply reply{
      .answered_by = authority_ref,
      .path_family = request.path_family,
      .stage_ref = request.stage_ref,
      .reply_kind = OwnerStageReplyKind::kRetryLater,
      .attachment_ref =
          AuthorityAttachmentRef{
              .authority_ref = authority_ref,
              .attachment_kind = "operation",
              .attachment_id = "op-123",
              .fencing_context = authority_ref.fencing_context,
          },
      .diagnostics = "owner still warming",
  };

  CHECK(request.authority_ref.authority_id == "issuer-daemon-a");
  CHECK(request.forwarded_claims.size() == 1);
  CHECK(request.hop_auth_context.auth_class == DaemonHopAuthClass::kDaemonMutualAuth);
  CHECK(reply.reply_kind == OwnerStageReplyKind::kRetryLater);
  REQUIRE(reply.attachment_ref.has_value());
  CHECK(reply.attachment_ref->attachment_id == "op-123");
  CHECK(reply.diagnostics == std::optional<std::string>("owner still warming"));
}

TEST_CASE("Routed authority wire helpers round-trip request and reply algebra", "[daemon][routing][wire]") {
  const AuthorityRef authority_ref{
      .authority_kind = AuthorityKind::kIssuerDaemon,
      .authority_id = "issuer-daemon-a",
      .fencing_context =
          FencingContext{
              .principal_kind = FencingPrincipalKind::kIssuerDaemon,
              .principal_id = "issuer-daemon-a",
              .epoch = 12,
          },
  };

  const RoutedAuthorityRequest request{
      .authority_ref = authority_ref,
      .path_family = "gate_continue_then_adopt",
      .stage_ref = "issuer_validate",
      .portable_credential =
          PortableParsedCredential{
              .address =
                  CapabilityBindingAddress{
                      .route_principal = make_issuer_route_principal("issuer-daemon-a"),
                      .family = LifecycleCapabilityFamily::kServe,
                      .binding_space = LifecycleBindingSpace::kPayload,
                      .binding_key_kind = BindingKeyKind::kPayloadId,
                      .binding_key = "payload-ref:test",
                      .epochs = LifecycleEpochs{.subject_generation = 8},
                  },
              .front_door_kind = LifecycleFrontDoorKind::kPayloadRef,
              .credential_expires_at = absl::Now() + absl::Minutes(1),
              .binding_mode = LifecycleBindingMode::kAddressDerived,
              .portable_constraint_claims =
                  ConstraintClaims{
                      .artifact_id = "artifact-roundtrip",
                      .direction = "get",
                      .operation_id = "op-roundtrip",
                  },
          },
      .forwardable_evidence =
          ForwardableCredentialEvidence{
              .evidence_kind = CredentialEvidenceKind::kRawCredential,
              .raw_credential_bytes = "payload-ref-token",
          },
      .hop_auth_context =
          DaemonHopAuthContext{
              .auth_class = DaemonHopAuthClass::kDeploymentTrustedChannel,
          },
      .forwarded_claims =
          {
              ForwardedClaim{
                  .claim_kind = "workflow_gate",
                  .provenance = ForwardedClaimProvenance::kIngressLocal,
                  .claim_payload = "approved",
                  .minted_by_authority_ref =
                      AuthorityRef{
                          .authority_kind = AuthorityKind::kInternalAuthority,
                          .authority_id = "front-daemon",
                      },
                  .audience_authority_ref = authority_ref,
                  .bound_root_request_id = "root-req-roundtrip",
                  .bound_path_family = "gate_continue_then_adopt",
              },
          },
      .request_metadata =
          RoutedRequestMetadata{
              .root_request_id = "root-req-roundtrip",
              .idempotency_key = "idem-roundtrip",
              .hop_budget_remaining = 2,
          },
  };

  tensorcast::daemon::v2::RoutedAuthorityRequest proto_request;
  routed_authority_wire::populate_proto_routed_authority_request(request, &proto_request);
  auto request_roundtrip_or = routed_authority_wire::routed_authority_request_from_proto(proto_request);
  REQUIRE(request_roundtrip_or.ok());
  CHECK(*request_roundtrip_or == request);

  const OwnerStageReply continuation_reply{
      .answered_by = authority_ref,
      .path_family = "gate_continue_then_adopt",
      .stage_ref = "issuer_validate",
      .reply_kind = OwnerStageReplyKind::kContinueWithAuthority,
      .continuation =
          AuthorityContinuation{
              .next_authority_ref =
                  AuthorityRef{
                      .authority_kind = AuthorityKind::kWorkflowOwner,
                      .authority_id = "workflow-owner-a",
                  },
              .edge_ref = "issuer_to_workflow",
              .forwarded_claims = request.forwarded_claims,
              .continuation_reason = "await_workflow_gate",
          },
  };
  tensorcast::daemon::v2::OwnerStageReply proto_continuation_reply;
  REQUIRE(
      routed_authority_wire::populate_proto_owner_stage_reply_shell(
          continuation_reply, std::nullopt, &proto_continuation_reply)
          .ok());
  auto continuation_roundtrip_or = routed_authority_wire::owner_stage_reply_shell_from_proto(proto_continuation_reply);
  REQUIRE(continuation_roundtrip_or.ok());
  CHECK(continuation_roundtrip_or->reply.reply_kind == OwnerStageReplyKind::kContinueWithAuthority);
  REQUIRE(continuation_roundtrip_or->reply.continuation.has_value());
  CHECK(continuation_roundtrip_or->reply.continuation->edge_ref == "issuer_to_workflow");
  CHECK(
      continuation_roundtrip_or->reply.continuation->forwarded_claims.front().provenance ==
      ForwardedClaimProvenance::kIngressLocal);

  const OwnerStageReply attach_reply{
      .answered_by = authority_ref,
      .path_family = "attach_existing",
      .stage_ref = "issuer_validate",
      .reply_kind = OwnerStageReplyKind::kAttachExisting,
      .attachment_ref =
          AuthorityAttachmentRef{
              .authority_ref = authority_ref,
              .attachment_kind = "operation",
              .attachment_id = "op-attach-1",
              .fencing_context = authority_ref.fencing_context,
          },
  };
  tensorcast::daemon::v2::OwnerStageReply proto_attach_reply;
  REQUIRE(
      routed_authority_wire::populate_proto_owner_stage_reply_shell(attach_reply, std::nullopt, &proto_attach_reply)
          .ok());
  auto attach_roundtrip_or = routed_authority_wire::owner_stage_reply_shell_from_proto(proto_attach_reply);
  REQUIRE(attach_roundtrip_or.ok());
  CHECK(attach_roundtrip_or->reply.reply_kind == OwnerStageReplyKind::kAttachExisting);
  REQUIRE(attach_roundtrip_or->reply.attachment_ref.has_value());
  CHECK(attach_roundtrip_or->reply.attachment_ref->attachment_id == "op-attach-1");

  const OwnerStageReply terminal_reply{
      .answered_by = authority_ref,
      .path_family = "immediate_terminal",
      .stage_ref = "issuer_validate",
      .reply_kind = OwnerStageReplyKind::kTerminal,
      .terminal_projection =
          TerminalProjection{
              .projection_kind = TerminalProjectionKind::kSemanticReject,
              .status_code = "permission_denied",
              .family_payload = "terminal-payload",
          },
      .diagnostics = "issuer rejected",
  };
  tensorcast::daemon::v2::OwnerStageReply proto_terminal_reply;
  REQUIRE(
      routed_authority_wire::populate_proto_owner_stage_reply_shell(terminal_reply, std::nullopt, &proto_terminal_reply)
          .ok());
  auto terminal_roundtrip_or = routed_authority_wire::owner_stage_reply_shell_from_proto(proto_terminal_reply);
  REQUIRE(terminal_roundtrip_or.ok());
  CHECK(terminal_roundtrip_or->reply.reply_kind == OwnerStageReplyKind::kTerminal);
  REQUIRE(terminal_roundtrip_or->reply.terminal_projection.has_value());
  CHECK(
      terminal_roundtrip_or->reply.terminal_projection->status_code == std::optional<std::string>("permission_denied"));
  CHECK(terminal_roundtrip_or->reply.diagnostics == std::optional<std::string>("issuer rejected"));
}

} // namespace tensorcast::daemon
