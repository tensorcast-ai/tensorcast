// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/lifecycle_kernel.h"

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

} // namespace tensorcast::daemon
