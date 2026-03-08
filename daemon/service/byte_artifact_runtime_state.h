// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
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

  mutable absl::Mutex mu;
  absl::flat_hash_map<std::string, BodyHandle> body_handles ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, BodyDescriptor> body_descriptors ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, BodyBackingObservation> body_observations ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, absl::Time> expires_at ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, std::uint64_t> entry_shard_id ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, std::uint64_t> entry_lease_generation ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, std::uint64_t> entry_routing_epoch ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::uint64_t, ShardRouteCacheEntry> shard_routes ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::uint64_t, OwnedShardLeaseEntry> owned_shard_leases ABSL_GUARDED_BY(mu);
  absl::flat_hash_map<std::string, DaemonEndpointCacheEntry> daemon_endpoints ABSL_GUARDED_BY(mu);
};

} // namespace tensorcast::daemon
