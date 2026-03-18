// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>

#include "absl/hash/hash.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/time/time.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store::runtime::ingestion {

enum class SemanticLayoutKind : std::uint8_t {
  kUnspecified = 0,
  kNamedLayoutId = 1,
  kCanonicalIndexDigest = 2,
  kFixedProfileLayout = 3,
  kCanonicalizedProjectionDigest = 4,
};

enum class LayoutProofKind : std::uint8_t {
  kUnspecified = 0,
  kCanonicalIndexDigest = 1,
  kSelectedIndexDigest = 2,
  kFixedProfileLayout = 3,
  kProjectionOnly = 4,
  kNamedLayoutId = 5,
};

enum class VerificationMethod : std::uint8_t {
  kUnspecified = 0,
  kSharedExecutorFullReadDigest = 1,
  kSharedExecutorStreamDigest = 2,
  kRegistrationCommit = 3,
  kSealCommit = 4,
};

struct ResolvedSourceDescriptor {
  std::string source_id;
  std::uint64_t exact_size_bytes{0};
  bool size_is_authoritative{true};
  bool resolved_locally{false};
  bool resolved_remotely{false};
  bool already_verified{false};
  loading::MaterializationSource source_kind{loading::MaterializationSource::kUnspecified};

  bool operator==(const ResolvedSourceDescriptor&) const = default;
};

struct SemanticLayoutIdentity {
  SemanticLayoutKind kind{SemanticLayoutKind::kUnspecified};
  std::string value;

  bool operator==(const SemanticLayoutIdentity&) const = default;
};

struct ContentIdentity {
  SemanticLayoutIdentity semantic_layout_identity;
  std::uint64_t logical_size_bytes{0};
  std::string digest_alg;
  std::string digest_bytes;

  bool operator==(const ContentIdentity&) const = default;
};

struct VerifiedContentDescriptor {
  ContentIdentity content_identity;

  bool operator==(const VerifiedContentDescriptor&) const = default;
};

struct VerificationRecord {
  VerificationMethod verification_method{VerificationMethod::kUnspecified};
  absl::Time verified_at{absl::InfinitePast()};
  LayoutProofKind layout_proof_kind{LayoutProofKind::kUnspecified};
  std::string layout_proof_value;

  bool operator==(const VerificationRecord&) const = default;
};

struct BackingIdentity {
  std::string physical_artifact_id;
  loading::ReplicaKey replica_key;

  bool operator==(const BackingIdentity&) const = default;
};

struct BackingIdentityHash {
  size_t operator()(const BackingIdentity& identity) const {
    return absl::HashOf(
        identity.physical_artifact_id,
        identity.replica_key.artifact_id,
        identity.replica_key.view_id,
        static_cast<int>(identity.replica_key.device.type),
        identity.replica_key.device.ordinal,
        identity.replica_key.device.uuid,
        identity.replica_key.replica);
  }
};

inline bool backing_identity_matches_replica_key(const BackingIdentity& identity) {
  return !identity.physical_artifact_id.empty() && identity.physical_artifact_id == identity.replica_key.artifact_id;
}

inline std::string normalize_content_digest_alg(std::string_view digest_alg) {
  std::string normalized(digest_alg);
  absl::AsciiStrToLower(&normalized);
  return normalized;
}

inline std::string content_digest_bytes_to_hex(std::string_view digest_bytes) {
  return absl::AsciiStrToLower(absl::BytesToHexString(digest_bytes));
}

} // namespace tensorcast::store::runtime::ingestion
