// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "daemon/service/body_backing_types.h"
#include "daemon/service/byte_artifact_body_handle.h"

namespace tensorcast::daemon {

struct ByteArtifactRuntimeState {
  struct ShardRouteCacheEntry {
    std::uint64_t lease_generation{0};
    std::string holder_daemon_id;
    std::uint64_t routing_epoch{1};
    absl::Time expires_at{absl::UnixEpoch()};
    absl::Time refreshed_at{absl::UnixEpoch()};
  };

  struct OwnedShardLeaseEntry {
    std::uint64_t lease_generation{0};
    std::string lease_token;
    std::uint64_t routing_epoch{1};
    absl::Time expires_at{absl::UnixEpoch()};
    absl::Time last_keepalive{absl::UnixEpoch()};
  };

  struct DaemonEndpointCacheEntry {
    std::string address;
    absl::Time refreshed_at{absl::UnixEpoch()};
  };

  struct PublishPreregEntry {
    store::runtime::ingestion::BackingIdentity backing_identity;
    std::uint64_t instance_generation{0};
    common::memory::MemoryLocation memory_location{common::memory::MemoryLocation::NONE};
    std::shared_ptr<void> keepalive;
    std::uint64_t size_bytes{0};
    absl::Time activated_at{absl::UnixEpoch()};
    absl::Time expires_at{absl::UnixEpoch()};
  };

  struct AuthorityEntry {
    BodyDescriptor claim_descriptor;
    store::runtime::ingestion::VerifiedContentDescriptor verified_content_descriptor;
    store::runtime::ingestion::VerificationRecord verification_record;
    std::optional<store::runtime::ingestion::BackingIdentity> retained_backing_identity;
    std::optional<PolicyVisibilityRef> policy_visibility_ref;
    absl::Time expires_at{absl::InfinitePast()};
    std::uint64_t shard_id{0};
    std::uint64_t lease_generation{0};
    std::uint64_t routing_epoch{1};
    AuthorityVisibilityKind visibility_kind{AuthorityVisibilityKind::kNone};
    AuthorityClaimState claim_state{AuthorityClaimState::kUnclaimed};
  };

  mutable absl::Mutex mu;
  absl::flat_hash_map<std::string, AuthorityEntry> authority_entries ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<
      store::runtime::ingestion::BackingIdentity,
      BackingRecord,
      store::runtime::ingestion::BackingIdentityHash>
      backing_entries ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<
      store::runtime::ingestion::BackingIdentity,
      absl::flat_hash_set<std::string>,
      store::runtime::ingestion::BackingIdentityHash>
      backing_authority_index ABSL_GUARDED_BY(mu);
  absl::flat_hash_set<store::runtime::ingestion::BackingIdentity, store::runtime::ingestion::BackingIdentityHash>
      orphan_backing_candidates ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<
      store::loading::ReplicaKey,
      absl::flat_hash_set<store::runtime::ingestion::BackingIdentity, store::runtime::ingestion::BackingIdentityHash>,
      store::loading::ReplicaKeyHash>
      replica_visibility_index ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::uint64_t, std::uint64_t> shard_authority_refcounts ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::uint64_t, ShardRouteCacheEntry> shard_routes ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::uint64_t, OwnedShardLeaseEntry> owned_shard_leases ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, DaemonEndpointCacheEntry> daemon_endpoints ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<
      store::runtime::ingestion::BackingIdentity,
      PublishPreregEntry,
      store::runtime::ingestion::BackingIdentityHash>
      publish_prereg_entries ABSL_GUARDED_BY(mu);
};

} // namespace tensorcast::daemon
