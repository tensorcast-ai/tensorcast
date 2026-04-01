// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/store/components/global_store_client.h"
#include "daemon/service/byte_artifact_runtime_state.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ByteArtifactRouteResolver {
 public:
  struct Options {
    absl::Duration route_staleness_budget{absl::Milliseconds(500)};
    absl::Duration route_refresh_timeout{absl::Milliseconds(500)};
    absl::Duration lease_ttl{absl::Seconds(5)};
    absl::Duration keepalive_interval{absl::Seconds(1)};
    std::uint64_t routing_epoch{1};
    bool shard_home_eligible{true};
  };

  struct RouteDecision {
    bool ok{false};
    std::uint64_t lease_generation{0};
    std::string holder_daemon_id;
    std::string message;
  };

  struct HomeLeaseDecision {
    enum class Kind {
      kOwned,
      kRedirect,
    };

    Kind kind{Kind::kOwned};
    std::uint64_t lease_generation{0};
    std::string message;
    v2::RouteFence redirect;
  };

  ByteArtifactRouteResolver(
      ByteArtifactRuntimeState& state,
      std::shared_ptr<store::components::IGlobalStoreClient> global_store_client,
      std::string local_daemon_id,
      Options options);

  [[nodiscard]] const std::string& local_daemon_id() const;
  [[nodiscard]] bool is_local_only() const;

  [[nodiscard]] absl::flat_hash_map<std::uint64_t, RouteDecision> resolve_routes(
      absl::Span<const std::uint64_t> shard_ids,
      absl::Time now);

  [[nodiscard]] RouteDecision resolve_route(std::uint64_t shard_id, absl::Time now);

  [[nodiscard]] RouteDecision refresh_route_from_redirect(
      std::uint64_t shard_id,
      const v2::RouteFence& redirect,
      absl::Time now);

  void populate_redirect(std::uint64_t shard_id, v2::RouteFence* redirect, absl::Time now);

  [[nodiscard]] absl::StatusOr<HomeLeaseDecision> ensure_home_lease(const v2::RouteFence& fence, absl::Time now);

  void keepalive_owned_shard_leases_once(absl::Time now);

 private:
  [[nodiscard]] absl::StatusOr<std::string> expected_shard_home_owner_daemon_id(
      std::uint64_t shard_id,
      const store::components::RpcOptions& rpc_options) const;

  [[nodiscard]] absl::StatusOr<store::components::AcquireShardHomeLeaseResult> acquire_shard_home_lease_with_policy(
      std::uint64_t shard_id,
      std::uint64_t ttl_ms,
      const store::components::RpcOptions& rpc_options) const;

  void cache_route_from_lookup(
      std::uint64_t shard_id,
      const store::components::ShardHomeRouteInfo& route,
      absl::Time now);

  void cache_route_from_lease(
      std::uint64_t shard_id,
      const store::components::ShardHomeLeaseDescriptor& lease,
      absl::Time now);

  ByteArtifactRuntimeState& state_;
  std::shared_ptr<store::components::IGlobalStoreClient> global_store_client_;
  std::string local_daemon_id_;
  Options options_;
};

} // namespace tensorcast::daemon
