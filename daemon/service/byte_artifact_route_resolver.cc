// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/byte_artifact_route_resolver.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace tensorcast::daemon {

namespace {

void set_route_redirect(
    std::uint64_t shard_id,
    std::uint64_t lease_generation,
    std::string_view holder_daemon_id,
    std::uint64_t routing_epoch,
    v2::RouteFence* redirect) {
  if (redirect == nullptr) {
    return;
  }
  redirect->set_shard_id(shard_id);
  redirect->set_lease_generation(lease_generation);
  redirect->set_holder_daemon_id(std::string(holder_daemon_id));
  redirect->set_routing_epoch(routing_epoch);
}

absl::Status validate_home_fence_locked(
    const v2::RouteFence& fence,
    std::string_view local_daemon_id,
    std::uint64_t expected_routing_epoch) {
  if (fence.lease_generation() == 0) {
    return absl::InvalidArgumentError("fence.lease_generation must be > 0");
  }
  if (fence.holder_daemon_id().empty()) {
    return absl::InvalidArgumentError("fence.holder_daemon_id is required");
  }
  if (fence.routing_epoch() == 0) {
    return absl::InvalidArgumentError("fence.routing_epoch must be > 0");
  }
  if (fence.routing_epoch() != expected_routing_epoch) {
    return absl::FailedPreconditionError("fence routing_epoch mismatch");
  }
  if (fence.holder_daemon_id() != local_daemon_id) {
    return absl::FailedPreconditionError("fence holder is not this daemon");
  }
  return absl::OkStatus();
}

std::uint64_t shard_home_eligible_flag_mask() {
  return 1ULL << tensorcast::global_store::v1::WORKER_CAPABILITY_FLAG_SHARD_HOME_ELIGIBLE;
}

std::uint64_t shard_home_hrw_score(std::uint64_t shard_id, std::string_view daemon_id) {
  const std::string key = absl::StrCat("byte-artifact-home:", shard_id, ":", daemon_id);
  const auto digest =
      common::sha256_digest_bytes(absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(key.data()), key.size()));
  std::uint64_t score = 0;
  for (std::size_t i = 0; i < sizeof(std::uint64_t); ++i) {
    score |= static_cast<std::uint64_t>(digest[i]) << (8U * i);
  }
  return score;
}

bool is_owned_lease_usable(
    const ByteArtifactRuntimeState::OwnedShardLeaseEntry& entry,
    absl::Time now,
    std::uint64_t routing_epoch) {
  return entry.routing_epoch == routing_epoch && (entry.expires_at == absl::UnixEpoch() || entry.expires_at > now);
}

bool is_route_fresh(absl::Time now, absl::Time refresh_at, absl::Duration staleness_budget) {
  return now - refresh_at <= staleness_budget;
}

bool is_cached_route_usable(
    const ByteArtifactRuntimeState::ShardRouteCacheEntry& entry,
    absl::Time now,
    absl::Duration staleness_budget,
    std::uint64_t routing_epoch) {
  return entry.routing_epoch == routing_epoch && entry.lease_generation != 0 && !entry.holder_daemon_id.empty() &&
      (entry.expires_at == absl::UnixEpoch() || entry.expires_at > now) &&
      is_route_fresh(now, entry.refreshed_at, staleness_budget);
}

} // namespace

ByteArtifactRouteResolver::ByteArtifactRouteResolver(
    ByteArtifactRuntimeState& state,
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client,
    std::string local_daemon_id,
    Options options)
    : state_(state),
      global_store_client_(std::move(global_store_client)),
      local_daemon_id_(local_daemon_id.empty() ? "daemon-local" : std::move(local_daemon_id)),
      options_(std::move(options)) {}

const std::string& ByteArtifactRouteResolver::local_daemon_id() const {
  return local_daemon_id_;
}

bool ByteArtifactRouteResolver::is_local_only() const {
  return global_store_client_ == nullptr;
}

absl::flat_hash_map<std::uint64_t, ByteArtifactRouteResolver::RouteDecision> ByteArtifactRouteResolver::resolve_routes(
    absl::Span<const std::uint64_t> shard_ids,
    absl::Time now) {
  absl::flat_hash_map<std::uint64_t, RouteDecision> routes;
  routes.reserve(shard_ids.size());

  if (is_local_only()) {
    for (const auto shard_id : shard_ids) {
      routes[shard_id] = RouteDecision{
          .ok = true,
          .lease_generation = 1,
          .holder_daemon_id = local_daemon_id_,
      };
    }
    return routes;
  }

  std::vector<std::uint64_t> shards_to_refresh;
  shards_to_refresh.reserve(shard_ids.size());
  {
    absl::MutexLock lock(&state_.mu);
    for (const auto shard_id : shard_ids) {
      const auto it = state_.shard_routes.find(shard_id);
      if (it != state_.shard_routes.end() &&
          is_cached_route_usable(it->second, now, options_.route_staleness_budget, options_.routing_epoch)) {
        routes[shard_id] = RouteDecision{
            .ok = true,
            .lease_generation = it->second.lease_generation,
            .holder_daemon_id = it->second.holder_daemon_id,
        };
      } else {
        shards_to_refresh.push_back(shard_id);
      }
    }
  }

  if (shards_to_refresh.empty()) {
    return routes;
  }
  if (global_store_client_ == nullptr || !global_store_client_->is_connected()) {
    for (const auto shard_id : shards_to_refresh) {
      routes[shard_id] = RouteDecision{
          .ok = false,
          .message = "global store unavailable for routing refresh",
      };
    }
    return routes;
  }

  store::components::RpcOptions rpc_options;
  rpc_options.timeout = options_.route_refresh_timeout;
  auto batch_or = global_store_client_->batch_get_shard_home_leases(shards_to_refresh, rpc_options);
  if (!batch_or.ok()) {
    for (const auto shard_id : shards_to_refresh) {
      routes[shard_id] = RouteDecision{
          .ok = false,
          .message = std::string(batch_or.status().message()),
      };
    }
    return routes;
  }

  absl::flat_hash_map<std::uint64_t, store::components::ShardHomeRouteInfo> gs_routes;
  gs_routes.reserve(batch_or->size());
  for (const auto& route : *batch_or) {
    gs_routes.emplace(route.shard_id, route);
    cache_route_from_lookup(route.shard_id, route, now);
    routes[route.shard_id] = RouteDecision{
        .ok = route.lease_generation != 0 && !route.holder_daemon_id.empty(),
        .lease_generation = route.lease_generation,
        .holder_daemon_id = route.holder_daemon_id,
    };
  }

  const auto ttl_ms = static_cast<std::uint64_t>(std::max<int64_t>(1, absl::ToInt64Milliseconds(options_.lease_ttl)));
  for (const auto shard_id : shards_to_refresh) {
    if (routes.contains(shard_id) && routes.at(shard_id).ok) {
      continue;
    }
    auto acquired_or = acquire_shard_home_lease_with_policy(shard_id, ttl_ms, rpc_options);
    if (!acquired_or.ok()) {
      routes[shard_id] = RouteDecision{
          .ok = false,
          .message = std::string(acquired_or.status().message()),
      };
      continue;
    }
    if (acquired_or->lease.lease_generation == 0 || acquired_or->lease.holder_daemon_id.empty()) {
      routes[shard_id] = RouteDecision{
          .ok = false,
          .message = "shard home lease unavailable",
      };
      continue;
    }
    cache_route_from_lease(shard_id, acquired_or->lease, acquired_or->acquired, now);
    routes[shard_id] = RouteDecision{
        .ok = true,
        .lease_generation = acquired_or->lease.lease_generation,
        .holder_daemon_id = acquired_or->lease.holder_daemon_id,
    };
  }
  return routes;
}

ByteArtifactRouteResolver::RouteDecision ByteArtifactRouteResolver::resolve_route(
    std::uint64_t shard_id,
    absl::Time now) {
  auto routes = resolve_routes(absl::MakeSpan(&shard_id, 1), now);
  const auto it = routes.find(shard_id);
  if (it == routes.end()) {
    return RouteDecision{
        .ok = false,
        .message = "route resolution returned no decision",
    };
  }
  return it->second;
}

ByteArtifactRouteResolver::RouteDecision ByteArtifactRouteResolver::refresh_route_from_redirect(
    std::uint64_t shard_id,
    const v2::RouteFence& redirect,
    absl::Time now) {
  if (redirect.routing_epoch() != options_.routing_epoch) {
    return RouteDecision{
        .ok = false,
        .message = "home routing_epoch mismatch",
    };
  }

  if (global_store_client_ != nullptr && global_store_client_->is_connected()) {
    store::components::RpcOptions rpc_options;
    rpc_options.timeout = options_.route_refresh_timeout;
    auto route_or = global_store_client_->get_shard_home_lease(shard_id, rpc_options);
    if (route_or.ok() && route_or->lease_generation != 0 && !route_or->holder_daemon_id.empty()) {
      cache_route_from_lookup(shard_id, *route_or, now);
      return RouteDecision{
          .ok = true,
          .lease_generation = route_or->lease_generation,
          .holder_daemon_id = route_or->holder_daemon_id,
      };
    }
  }

  {
    absl::MutexLock lock(&state_.mu);
    state_.shard_routes[shard_id] = ByteArtifactRuntimeState::ShardRouteCacheEntry{
        .lease_generation = redirect.lease_generation(),
        .holder_daemon_id = redirect.holder_daemon_id(),
        .routing_epoch = options_.routing_epoch,
        .expires_at = absl::UnixEpoch(),
        .refreshed_at = now,
    };
    if (redirect.holder_daemon_id() != local_daemon_id_) {
      state_.owned_shard_leases.erase(shard_id);
    }
  }
  return RouteDecision{
      .ok = redirect.lease_generation() != 0 && !redirect.holder_daemon_id().empty(),
      .lease_generation = redirect.lease_generation(),
      .holder_daemon_id = redirect.holder_daemon_id(),
  };
}

void ByteArtifactRouteResolver::populate_redirect(std::uint64_t shard_id, v2::RouteFence* redirect, absl::Time now) {
  std::optional<ByteArtifactRuntimeState::ShardRouteCacheEntry> cached;
  {
    absl::MutexLock lock(&state_.mu);
    const auto it = state_.shard_routes.find(shard_id);
    if (it != state_.shard_routes.end()) {
      cached = it->second;
    }
  }
  if (cached.has_value() && cached->routing_epoch == options_.routing_epoch && cached->lease_generation != 0 &&
      !cached->holder_daemon_id.empty()) {
    set_route_redirect(shard_id, cached->lease_generation, cached->holder_daemon_id, options_.routing_epoch, redirect);
    return;
  }
  if (global_store_client_ == nullptr || !global_store_client_->is_connected()) {
    return;
  }
  auto route_or = global_store_client_->get_shard_home_lease(shard_id);
  if (!route_or.ok()) {
    return;
  }
  set_route_redirect(
      route_or->shard_id, route_or->lease_generation, route_or->holder_daemon_id, options_.routing_epoch, redirect);
  cache_route_from_lookup(shard_id, *route_or, now);
}

absl::StatusOr<ByteArtifactRouteResolver::HomeLeaseDecision> ByteArtifactRouteResolver::ensure_home_lease(
    const v2::RouteFence& fence,
    absl::Time now) {
  const auto fence_status = validate_home_fence_locked(fence, local_daemon_id_, options_.routing_epoch);
  if (!fence_status.ok()) {
    if (absl::IsInvalidArgument(fence_status)) {
      return fence_status;
    }
    HomeLeaseDecision decision;
    decision.kind = HomeLeaseDecision::Kind::kRedirect;
    decision.message = std::string(fence_status.message());
    populate_redirect(fence.shard_id(), &decision.redirect, now);
    return decision;
  }

  std::uint64_t owned_generation = 0;
  std::string owned_token;
  absl::Time owned_expires_at = absl::UnixEpoch();
  absl::Time owned_last_keepalive = absl::UnixEpoch();
  {
    absl::MutexLock lock(&state_.mu);
    if (is_local_only()) {
      owned_generation = 1;
    } else {
      const auto it = state_.owned_shard_leases.find(fence.shard_id());
      if (it != state_.owned_shard_leases.end()) {
        owned_generation = it->second.lease_generation;
        owned_token = it->second.lease_token;
        owned_expires_at = it->second.expires_at;
        owned_last_keepalive = it->second.last_keepalive;
        if (!is_owned_lease_usable(it->second, now, options_.routing_epoch)) {
          state_.owned_shard_leases.erase(it);
          owned_generation = 0;
          owned_token.clear();
        }
      }
    }
  }

  if (!is_local_only() && owned_generation == 0 && global_store_client_ != nullptr &&
      global_store_client_->is_connected()) {
    store::components::RpcOptions rpc_options;
    rpc_options.timeout = options_.route_refresh_timeout;
    const auto ttl_ms = static_cast<std::uint64_t>(std::max<int64_t>(1, absl::ToInt64Milliseconds(options_.lease_ttl)));
    auto acquired_or = acquire_shard_home_lease_with_policy(fence.shard_id(), ttl_ms, rpc_options);
    if (acquired_or.ok() && acquired_or->lease.lease_generation != 0 && !acquired_or->lease.holder_daemon_id.empty()) {
      cache_route_from_lease(fence.shard_id(), acquired_or->lease, acquired_or->acquired, now);
      if (acquired_or->lease.holder_daemon_id == local_daemon_id_ && acquired_or->acquired) {
        owned_generation = acquired_or->lease.lease_generation;
        owned_token = acquired_or->lease.lease_token;
        owned_expires_at = acquired_or->lease.expires_at;
        owned_last_keepalive = now;
      }
    }
  }

  if (owned_generation == 0 || fence.lease_generation() != owned_generation) {
    HomeLeaseDecision decision;
    decision.kind = HomeLeaseDecision::Kind::kRedirect;
    decision.message = "stale shard lease fence";
    populate_redirect(fence.shard_id(), &decision.redirect, now);
    return decision;
  }

  if (!is_local_only()) {
    const bool keepalive_due = owned_last_keepalive == absl::UnixEpoch() ||
        now - owned_last_keepalive >= options_.keepalive_interval ||
        (owned_expires_at != absl::UnixEpoch() && owned_expires_at - now <= options_.keepalive_interval);
    if (keepalive_due) {
      const auto ttl_ms =
          static_cast<std::uint64_t>(std::max<int64_t>(1, absl::ToInt64Milliseconds(options_.lease_ttl)));
      auto lease_or = global_store_client_->keepalive_shard_home_lease(owned_token, ttl_ms);
      if (!lease_or.ok() || lease_or->lease_generation != owned_generation) {
        {
          absl::MutexLock lock(&state_.mu);
          state_.owned_shard_leases.erase(fence.shard_id());
        }
        HomeLeaseDecision decision;
        decision.kind = HomeLeaseDecision::Kind::kRedirect;
        decision.message = "shard lease keepalive failed";
        populate_redirect(fence.shard_id(), &decision.redirect, now);
        return decision;
      }
      cache_route_from_lease(fence.shard_id(), *lease_or, /*acquired=*/true, now);
    }
  }

  HomeLeaseDecision decision;
  decision.kind = HomeLeaseDecision::Kind::kOwned;
  decision.lease_generation = owned_generation;
  return decision;
}

absl::StatusOr<bool> ByteArtifactRouteResolver::is_expected_shard_home_owner(
    std::uint64_t shard_id,
    const store::components::RpcOptions& rpc_options) const {
  if (global_store_client_ == nullptr || !global_store_client_->is_connected()) {
    return false;
  }
  auto workers_or = global_store_client_->list_active_workers(
      /*include_unavailable=*/false, shard_home_eligible_flag_mask(), rpc_options);
  if (!workers_or.ok()) {
    return workers_or.status();
  }

  std::string best_daemon_id;
  std::uint64_t best_score = 0;
  for (const auto& worker : *workers_or) {
    if (worker.daemon_id.empty()) {
      continue;
    }
    const auto score = shard_home_hrw_score(shard_id, worker.daemon_id);
    if (best_daemon_id.empty() || score > best_score || (score == best_score && worker.daemon_id < best_daemon_id)) {
      best_daemon_id = worker.daemon_id;
      best_score = score;
    }
  }
  if (best_daemon_id.empty()) {
    return absl::FailedPreconditionError("no shard-home eligible daemons available");
  }
  return best_daemon_id == local_daemon_id_;
}

absl::StatusOr<store::components::AcquireShardHomeLeaseResult> ByteArtifactRouteResolver::
    acquire_shard_home_lease_with_policy(
        std::uint64_t shard_id,
        std::uint64_t ttl_ms,
        const store::components::RpcOptions& rpc_options) const {
  if (global_store_client_ != nullptr && global_store_client_->is_connected()) {
    auto current_route_or = global_store_client_->get_shard_home_lease(shard_id, rpc_options);
    if (current_route_or.ok() && current_route_or->lease_generation != 0 &&
        !current_route_or->holder_daemon_id.empty()) {
      if (current_route_or->holder_daemon_id != local_daemon_id_) {
        return absl::FailedPreconditionError("local daemon is not the current shard-home holder");
      }
      if (!options_.shard_home_eligible) {
        return absl::FailedPreconditionError("local daemon is not shard-home eligible");
      }
      return global_store_client_->acquire_shard_home_lease(shard_id, local_daemon_id_, ttl_ms, rpc_options);
    }
    if (!current_route_or.ok() && !absl::IsNotFound(current_route_or.status())) {
      return current_route_or.status();
    }
  }
  if (!options_.shard_home_eligible) {
    return absl::FailedPreconditionError("local daemon is not shard-home eligible");
  }
  auto expected_or = is_expected_shard_home_owner(shard_id, rpc_options);
  if (!expected_or.ok()) {
    return expected_or.status();
  }
  if (!*expected_or) {
    return absl::FailedPreconditionError("local daemon is not the expected shard-home owner");
  }
  return global_store_client_->acquire_shard_home_lease(shard_id, local_daemon_id_, ttl_ms, rpc_options);
}

void ByteArtifactRouteResolver::cache_route_from_lookup(
    std::uint64_t shard_id,
    const store::components::ShardHomeRouteInfo& route,
    absl::Time now) {
  absl::MutexLock lock(&state_.mu);
  state_.shard_routes[shard_id] = ByteArtifactRuntimeState::ShardRouteCacheEntry{
      .lease_generation = route.lease_generation,
      .holder_daemon_id = route.holder_daemon_id,
      .routing_epoch = options_.routing_epoch,
      .expires_at = route.expires_at,
      .refreshed_at = now,
  };
  if (route.holder_daemon_id != local_daemon_id_) {
    state_.owned_shard_leases.erase(shard_id);
  }
}

void ByteArtifactRouteResolver::cache_route_from_lease(
    std::uint64_t shard_id,
    const store::components::ShardHomeLeaseDescriptor& lease,
    bool acquired,
    absl::Time now) {
  absl::MutexLock lock(&state_.mu);
  state_.shard_routes[shard_id] = ByteArtifactRuntimeState::ShardRouteCacheEntry{
      .lease_generation = lease.lease_generation,
      .holder_daemon_id = lease.holder_daemon_id,
      .routing_epoch = options_.routing_epoch,
      .expires_at = lease.expires_at,
      .refreshed_at = now,
  };
  if (lease.holder_daemon_id != local_daemon_id_) {
    state_.owned_shard_leases.erase(shard_id);
    return;
  }
  if (acquired) {
    state_.owned_shard_leases[shard_id] = ByteArtifactRuntimeState::OwnedShardLeaseEntry{
        .lease_generation = lease.lease_generation,
        .lease_token = lease.lease_token,
        .routing_epoch = options_.routing_epoch,
        .expires_at = lease.expires_at,
        .last_keepalive = now,
    };
  }
}

} // namespace tensorcast::daemon
