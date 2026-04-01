// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/time/time.h"
#include "core/common/memory/memory_location.h"
#include "core/common/selection_identity.h"
#include "core/store/runtime/ingestion/artifact_truth.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

enum class BodyAccessClass : std::uint8_t {
  kHomeDefault = 0,
  kLocalGpuHot = 1,
  kTransientForward = 2,
  kSmallObject = 3,
};

enum class BodyRouteRole : std::uint8_t {
  kHomeAuthority = 0,
  kTransientForwarder = 1,
};

enum class BodyConsumerLocality : std::uint8_t {
  kLocalOnly = 0,
  kRemoteOrMixed = 1,
};

enum class BodyAccessPattern : std::uint8_t {
  kDefault = 0,
  kLocalGpuHot = 1,
  kTransientForward = 2,
  kSmallObject = 3,
};

enum class BodyPreferredResidency : std::uint8_t {
  kCpu = 0,
  kGpu = 1,
};

enum class BodyRetentionIntent : std::uint8_t {
  kEphemeral = 0,
  kRetained = 1,
};

enum class BodyStableRetentionRequirement : std::uint8_t {
  kNone = 0,
  kPreferStable = 1,
  kRequireStable = 2,
};

enum class BodySharingIntent : std::uint8_t {
  kPrivateLocal = 0,
  kLocalReadMostly = 1,
  kRemoteShareable = 2,
};

struct BodyBackingIntent {
  BodyPreferredResidency preferred_residency{BodyPreferredResidency::kCpu};
  BodyRetentionIntent retention_intent{BodyRetentionIntent::kRetained};
  BodyStableRetentionRequirement stable_retention_requirement{BodyStableRetentionRequirement::kNone};
  BodySharingIntent sharing_intent{BodySharingIntent::kPrivateLocal};
};

struct BodyPlacementContext {
  BodyRouteRole route_role{BodyRouteRole::kHomeAuthority};
  BodyConsumerLocality locality{BodyConsumerLocality::kRemoteOrMixed};
  BodyAccessPattern access_pattern{BodyAccessPattern::kDefault};
  std::uint64_t size_bytes{0};
  std::uint32_t expected_fanout{1};
};

enum class BodyStableRetentionState : std::uint8_t {
  kUnknown = 0,
  kNotRequested = 1,
  kHeld = 2,
  kSkipped = 3,
};

enum class BodyCommunicatorExportState : std::uint8_t {
  kUnknown = 0,
  kNotExported = 1,
  kExported = 2,
};

enum class BodyCapabilityResolutionMode : std::uint8_t {
  kLocalBodyHandle = 0,
  kLoader = 1,
  kChunkRpcFallback = 2,
};

enum class AuthorityVisibilityKind : std::uint8_t {
  kNone = 0,
  kReadyBacking = 1,
  kPolicyBackedPath = 2,
};

enum class AuthorityClaimState : std::uint8_t {
  kUnclaimed = 0,
  kClaimedVisible = 1,
  kClaimedInvisible = 2,
  kClaimDeleted = 3,
};

enum class BackingLifecycleState : std::uint8_t {
  kActive = 0,
  kInvalidated = 1,
  kSuperseded = 2,
  kDraining = 3,
  kRetired = 4,
};

enum class PolicyVisibilityPathKind : std::uint8_t {
  kUnspecified = 0,
  kSharedDisk = 1,
  kRemoteStable = 2,
};

enum class ServingCapabilitySubjectKind : std::uint8_t {
  kBacking = 0,
  kCopiedPayload = 1,
  kPolicyBackedPath = 2,
};

enum class LifecycleOwnerKind : std::uint8_t {
  kInlineCopyWindow = 0,
  kPayloadRefToken = 1,
  kRetentionHandle = 2,
  kPersistenceTask = 3,
};

struct BodyDescriptor {
  std::string physical_artifact_id;
  std::string layout_id;
  std::uint64_t size_bytes{0};
  std::string payload_digest_alg;
  std::string payload_digest_hex;
  v2::ByteArtifactVerificationMode verification_mode{
      v2::BYTE_ARTIFACT_VERIFICATION_MODE_STRICT_SHA256,
  };
  absl::Time created_at{absl::InfinitePast()};
  absl::Time verified_at{absl::InfinitePast()};

  bool operator==(const BodyDescriptor&) const = default;
};

struct ResolvedBodyCapability {
  BodyCapabilityResolutionMode mode{BodyCapabilityResolutionMode::kLoader};
  bool local{true};
  BodyHandle body_handle;
  BodyDescriptor descriptor;
};

struct PolicyVisibilityRef {
  std::string path_id;
  PolicyVisibilityPathKind path_kind{PolicyVisibilityPathKind::kUnspecified};
  store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
  std::string control_ref;
  absl::Time expires_at{absl::InfinitePast()};

  bool operator==(const PolicyVisibilityRef&) const = default;
};

struct LifecycleOwnerRef {
  LifecycleOwnerKind owner_kind{LifecycleOwnerKind::kInlineCopyWindow};
  std::string owner_id;

  bool operator==(const LifecycleOwnerRef&) const = default;
};

struct ServingCapability {
  std::string capability_id;
  absl::Time expires_at{absl::InfinitePast()};
  BodyCapabilityResolutionMode mode{BodyCapabilityResolutionMode::kLoader};
  bool local{true};
  ServingCapabilitySubjectKind subject_kind{ServingCapabilitySubjectKind::kBacking};
  LifecycleOwnerRef lifecycle_owner_ref;
  std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
  std::uint64_t backing_instance_generation{0};
  std::optional<PolicyVisibilityRef> policy_visibility_ref;
};

struct ResolvedSourceCapability {
  tensorcast::common::SelectionIdentity selection_identity;
  store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
  ServingCapability serving_capability;
  std::optional<store::runtime::ingestion::BackingIdentity> backing_identity;
  store::loading::MaterializationSource source_kind{store::loading::MaterializationSource::kUnspecified};
  std::optional<ResolvedBodyCapability> body_capability;
  std::shared_ptr<const std::string> inline_payload;
  std::string payload_ref;
  std::optional<PolicyVisibilityRef> policy_source_ref;
};

struct AuthorityRecord {
  std::string artifact_id;
  std::uint64_t shard_id{0};
  std::uint64_t lease_generation{0};
  std::uint64_t routing_epoch{1};
  absl::Time expires_at{absl::InfinitePast()};
  AuthorityVisibilityKind visibility_kind{AuthorityVisibilityKind::kNone};
  AuthorityClaimState claim_state{AuthorityClaimState::kUnclaimed};
  std::optional<store::runtime::ingestion::BackingIdentity> retained_backing_identity;
  std::optional<PolicyVisibilityRef> policy_visibility_ref;
  bool visible{false};
};

struct BodyBackingObservation {
  std::string physical_artifact_id;
  common::memory::MemoryLocation memory_location{common::memory::MemoryLocation::CPU};
  std::uint64_t size_bytes{0};
  bool cpu_memfd_available{false};
  bool cuda_ipc_available{false};
  BodyCommunicatorExportState communicator_export_state{BodyCommunicatorExportState::kUnknown};
  BodyStableRetentionState stable_retention_state{BodyStableRetentionState::kUnknown};
  absl::Time observed_at{absl::InfinitePast()};
};

struct BackingRecord {
  store::runtime::ingestion::BackingIdentity identity;
  std::uint64_t instance_generation{0};
  store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
  store::runtime::ingestion::VerificationRecord verification_record;
  BodyDescriptor descriptor;
  BodyBackingObservation last_observation;
  BodyHandle retained_body_handle;
  BackingLifecycleState lifecycle_state{BackingLifecycleState::kActive};
};

inline std::string normalize_body_digest_value(std::string_view value) {
  std::string normalized(value);
  absl::AsciiStrToLower(&normalized);
  return normalized;
}

inline v2::ByteArtifactVerificationMode normalize_byte_artifact_verification_mode(
    v2::ByteArtifactVerificationMode mode) {
  if (mode == v2::BYTE_ARTIFACT_VERIFICATION_MODE_UNSPECIFIED) {
    return v2::BYTE_ARTIFACT_VERIFICATION_MODE_STRICT_SHA256;
  }
  return mode;
}

inline v2::ByteArtifactVerificationMode invariant_verification_mode(const v2::PutIfAbsentInvariant& invariant) {
  return normalize_byte_artifact_verification_mode(invariant.verification_mode());
}

inline bool verification_mode_requires_payload_digest(v2::ByteArtifactVerificationMode mode) {
  return normalize_byte_artifact_verification_mode(mode) == v2::BYTE_ARTIFACT_VERIFICATION_MODE_STRICT_SHA256;
}

inline const char* byte_artifact_verification_mode_label(v2::ByteArtifactVerificationMode mode) {
  switch (normalize_byte_artifact_verification_mode(mode)) {
    case v2::BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY:
      return "layout_and_size_only";
    case v2::BYTE_ARTIFACT_VERIFICATION_MODE_STRICT_SHA256:
    default:
      return "strict_sha256";
  }
}

inline BodyDescriptor normalized_body_descriptor(BodyDescriptor descriptor) {
  descriptor.payload_digest_alg = normalize_body_digest_value(descriptor.payload_digest_alg);
  descriptor.payload_digest_hex = normalize_body_digest_value(descriptor.payload_digest_hex);
  descriptor.verification_mode = normalize_byte_artifact_verification_mode(descriptor.verification_mode);
  return descriptor;
}

inline v2::PutIfAbsentInvariant body_descriptor_to_invariant(const BodyDescriptor& descriptor) {
  v2::PutIfAbsentInvariant invariant;
  invariant.set_layout_id(descriptor.layout_id);
  invariant.set_byte_length(descriptor.size_bytes);
  invariant.set_payload_digest_alg(descriptor.payload_digest_alg);
  invariant.set_payload_digest_hex(descriptor.payload_digest_hex);
  invariant.set_verification_mode(normalize_byte_artifact_verification_mode(descriptor.verification_mode));
  return invariant;
}

inline store::runtime::ingestion::VerifiedContentDescriptor body_descriptor_to_verified_content_descriptor(
    const BodyDescriptor& descriptor) {
  store::runtime::ingestion::VerifiedContentDescriptor verified;
  verified.content_identity.semantic_layout_identity.kind =
      store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId;
  verified.content_identity.semantic_layout_identity.value = descriptor.layout_id;
  verified.content_identity.logical_size_bytes = descriptor.size_bytes;
  verified.content_identity.digest_alg = descriptor.payload_digest_alg;
  std::string digest_bytes;
  if (!absl::HexStringToBytes(descriptor.payload_digest_hex, &digest_bytes)) {
    digest_bytes = descriptor.payload_digest_hex;
  }
  verified.content_identity.digest_bytes = std::move(digest_bytes);
  return verified;
}

inline store::runtime::ingestion::VerificationRecord body_descriptor_to_verification_record(
    const BodyDescriptor& descriptor) {
  return store::runtime::ingestion::VerificationRecord{
      .verification_method = verification_mode_requires_payload_digest(descriptor.verification_mode)
          ? store::runtime::ingestion::VerificationMethod::kSharedExecutorFullReadDigest
          : store::runtime::ingestion::VerificationMethod::kLayoutAndSizeContract,
      .verified_at = descriptor.verified_at,
      .layout_proof_kind = store::runtime::ingestion::LayoutProofKind::kNamedLayoutId,
      .layout_proof_value = descriptor.layout_id,
  };
}

inline store::runtime::ingestion::BackingIdentity body_descriptor_to_backing_identity(
    const BodyDescriptor& descriptor,
    const BodyHandle& body_handle) {
  return store::runtime::ingestion::BackingIdentity{
      .physical_artifact_id = descriptor.physical_artifact_id,
      .replica_key = body_handle.replica_handle().key(),
  };
}

inline store::runtime::ingestion::VerifiedContentDescriptor body_descriptor_to_verified_content_descriptor_with_layout(
    std::string_view layout_id,
    std::uint64_t size_bytes,
    std::string_view digest_alg,
    std::string_view digest_hex) {
  store::runtime::ingestion::VerifiedContentDescriptor verified;
  if (!layout_id.empty()) {
    verified.content_identity.semantic_layout_identity.kind =
        store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId;
    verified.content_identity.semantic_layout_identity.value = std::string(layout_id);
  }
  verified.content_identity.logical_size_bytes = size_bytes;
  verified.content_identity.digest_alg = normalize_body_digest_value(digest_alg);
  std::string digest_bytes;
  if (!absl::HexStringToBytes(normalize_body_digest_value(digest_hex), &digest_bytes)) {
    digest_bytes = std::string(digest_hex);
  }
  verified.content_identity.digest_bytes = std::move(digest_bytes);
  return verified;
}

inline std::size_t resolved_source_capability_form_count(const ResolvedSourceCapability& capability) {
  std::size_t forms = 0;
  forms += capability.body_capability.has_value() ? 1 : 0;
  forms += capability.inline_payload ? 1 : 0;
  forms += capability.payload_ref.empty() ? 0 : 1;
  forms += capability.policy_source_ref.has_value() ? 1 : 0;
  return forms;
}

inline absl::Status validate_resolved_source_capability(const ResolvedSourceCapability& capability) {
  if (capability.selection_identity.artifact_id.empty()) {
    return absl::InvalidArgumentError("ResolvedSourceCapability requires selection_identity.artifact_id");
  }
  if (resolved_source_capability_form_count(capability) != 1) {
    return absl::InvalidArgumentError("ResolvedSourceCapability requires exactly one concrete source form");
  }
  if (capability.serving_capability.capability_id.empty()) {
    return absl::InvalidArgumentError("ResolvedSourceCapability requires serving_capability");
  }
  if (capability.body_capability.has_value()) {
    if (!capability.body_capability->local || capability.body_capability->body_handle.empty()) {
      return absl::InvalidArgumentError("ResolvedSourceCapability local body source requires a local body_handle");
    }
    if (capability.source_kind == store::loading::MaterializationSource::kUnspecified) {
      return absl::InvalidArgumentError("ResolvedSourceCapability local body source requires source_kind");
    }
  }
  if (capability.inline_payload && capability.source_kind == store::loading::MaterializationSource::kUnspecified) {
    return absl::InvalidArgumentError("ResolvedSourceCapability inline payload source requires source_kind");
  }
  if (!capability.payload_ref.empty() &&
      capability.source_kind == store::loading::MaterializationSource::kUnspecified) {
    return absl::InvalidArgumentError("ResolvedSourceCapability payload_ref source requires source_kind");
  }
  if (capability.policy_source_ref.has_value() && capability.policy_source_ref->path_id.empty()) {
    return absl::InvalidArgumentError("ResolvedSourceCapability policy-backed source requires policy_source_ref");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::daemon
