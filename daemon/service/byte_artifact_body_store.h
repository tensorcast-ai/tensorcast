// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "daemon/service/body_backing_types.h"
#include "daemon/service/byte_artifact_body_handle.h"
#include "daemon/service/byte_artifact_runtime_state.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ByteArtifactBodyStore {
 public:
  struct EntrySnapshot {
    BodyDescriptor descriptor;
    store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
    store::runtime::ingestion::VerificationRecord verification_record;
    BackingRecord backing_record;
    AuthorityRecord authority_record;
    absl::Time expires_at{absl::InfinitePast()};
  };

  struct AuthoritySnapshot {
    BodyDescriptor descriptor;
    store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
    store::runtime::ingestion::VerificationRecord verification_record;
    AuthorityRecord authority_record;
    absl::Time expires_at{absl::InfinitePast()};
  };

  struct PersistenceSourceSnapshot {
    std::string source_artifact_id;
    std::uint64_t size_bytes{0};
    store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
  };

  enum class PutOutcome {
    kCreated,
    kJoined,
    kConflict,
  };

  struct PutResult {
    PutOutcome outcome{PutOutcome::kConflict};
  };

  explicit ByteArtifactBodyStore(ByteArtifactRuntimeState& state);

  [[nodiscard]] bool exists(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now);

  [[nodiscard]] std::optional<EntrySnapshot> get(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now);

  [[nodiscard]] std::optional<AuthoritySnapshot> inspect_authority(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now);

  [[nodiscard]] std::optional<BackingRecord> inspect_backing(
      const store::runtime::ingestion::BackingIdentity& identity) const;

  [[nodiscard]] std::optional<PersistenceSourceSnapshot> inspect_persistence_source(std::string_view artifact_id) const;

  [[nodiscard]] bool install_policy_visibility(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now,
      const PolicyVisibilityRef& policy_visibility_ref);

  [[nodiscard]] bool clear_policy_visibility(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now,
      std::string_view reason);

  [[nodiscard]] PutResult put_if_absent(
      std::string_view artifact_id,
      const v2::PutIfAbsentInvariant& invariant,
      const BodyDescriptor& descriptor,
      const store::runtime::ingestion::VerifiedContentDescriptor& verified_content_descriptor,
      const store::runtime::ingestion::VerificationRecord& verification_record,
      const store::runtime::ingestion::BackingIdentity& backing_identity,
      const BodyBackingObservation& observation,
      const BodyHandle& body_handle,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now,
      const std::optional<std::uint64_t>& ttl_ms);

  [[nodiscard]] bool touch_ttl(
      std::string_view artifact_id,
      std::uint64_t shard_id,
      std::uint64_t lease_generation,
      std::uint64_t routing_epoch,
      absl::Time now,
      std::uint64_t ttl_ms);

  void invalidate_artifact_visibility(std::string_view artifact_id, absl::Time now, std::string_view reason);

  void invalidate_replica_visibility(
      const store::loading::ReplicaKey& replica_key,
      absl::Time now,
      std::string_view reason);

 private:
  ByteArtifactRuntimeState& state_;
};

} // namespace tensorcast::daemon
