// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/instance_execution_directory_cache.h"

#include <algorithm>
#include <utility>

namespace tensorcast::daemon {

namespace {

absl::Duration rpc_timeout(absl::Duration staleness_budget) {
  return std::max(absl::Milliseconds(1), staleness_budget);
}

bool is_routable(const store::components::ActiveInstanceInfo& instance) {
  return !instance.instance_id.empty() && !instance.daemon_id.empty() && !instance.execution_endpoint.empty();
}

std::string execution_host_kind_for(const store::components::ActiveInstanceInfo& instance) {
  return instance.execution_host_kind.empty() ? std::string("node_agent_grpc") : instance.execution_host_kind;
}

} // namespace

InstanceExecutionDirectoryCache::InstanceExecutionDirectoryCache(
    std::shared_ptr<store::components::IGlobalStoreClient> global_store_client)
    : global_store_client_(std::move(global_store_client)) {}

bool InstanceExecutionDirectoryCache::is_state_fresh(
    const CacheState& state,
    absl::Time now,
    absl::Duration staleness_budget) {
  return state.refreshed_at != absl::UnixEpoch() && now - state.refreshed_at <= staleness_budget;
}

InstanceExecutionDirectoryCache::Snapshot InstanceExecutionDirectoryCache::filtered_snapshot(
    const CacheState& state,
    uint64_t required_capability_flags) {
  Snapshot snapshot;
  snapshot.refreshed_at = state.refreshed_at;
  snapshot.cache_epoch = state.cache_epoch;
  snapshot.entries.reserve(state.entries.size());
  for (const auto& entry : state.entries) {
    if ((entry.capability_flags & required_capability_flags) != required_capability_flags) {
      continue;
    }
    snapshot.entries.push_back(entry);
  }
  return snapshot;
}

InstanceExecutionDirectoryCache::CacheState& InstanceExecutionDirectoryCache::state_for(bool include_unavailable) {
  return include_unavailable ? include_unavailable_state_ : active_state_;
}

const InstanceExecutionDirectoryCache::CacheState& InstanceExecutionDirectoryCache::state_for(
    bool include_unavailable) const {
  return include_unavailable ? include_unavailable_state_ : active_state_;
}

absl::StatusOr<InstanceExecutionDirectoryCache::Snapshot> InstanceExecutionDirectoryCache::list_instances(
    bool include_unavailable,
    uint64_t required_capability_flags,
    absl::Time now,
    absl::Duration staleness_budget) {
  {
    absl::MutexLock lock(&mu_);
    const auto& state = state_for(include_unavailable);
    if (is_state_fresh(state, now, staleness_budget)) {
      return filtered_snapshot(state, required_capability_flags);
    }
  }

  if (global_store_client_ == nullptr || !global_store_client_->is_connected()) {
    return absl::UnavailableError("Global Store client unavailable for instance directory refresh");
  }

  store::components::RpcOptions rpc_opts;
  rpc_opts.timeout = rpc_timeout(staleness_budget);
  auto instances_or = global_store_client_->list_active_instances(
      include_unavailable,
      /*required_capability_flags=*/0,
      rpc_opts);
  if (!instances_or.ok()) {
    return instances_or.status();
  }

  absl::MutexLock lock(&mu_);
  auto& state = state_for(include_unavailable);
  update_from_instances(*instances_or, state, now);
  return filtered_snapshot(state, required_capability_flags);
}

absl::StatusOr<InstanceExecutionDirectoryCache::Entry> InstanceExecutionDirectoryCache::resolve_instance_execution(
    std::string_view instance_id,
    absl::Time now,
    absl::Duration staleness_budget) {
  {
    absl::MutexLock lock(&mu_);
    const auto& state = state_for(/*include_unavailable=*/false);
    if (is_state_fresh(state, now, staleness_budget)) {
      const auto it = state.index_by_instance_id.find(std::string(instance_id));
      if (it != state.index_by_instance_id.end()) {
        return state.entries[it->second];
      }
    }
  }

  auto snapshot_or = list_instances(
      /*include_unavailable=*/false,
      /*required_capability_flags=*/0,
      now,
      staleness_budget);
  if (!snapshot_or.ok()) {
    return snapshot_or.status();
  }

  for (const auto& entry : snapshot_or->entries) {
    if (entry.instance_id == instance_id) {
      return entry;
    }
  }
  return absl::NotFoundError("instance execution endpoint not found");
}

void InstanceExecutionDirectoryCache::update_from_instances(
    const std::vector<store::components::ActiveInstanceInfo>& instances,
    CacheState& state,
    absl::Time now) {
  state.entries.clear();
  state.index_by_instance_id.clear();
  state.entries.reserve(instances.size());
  for (const auto& instance : instances) {
    if (!is_routable(instance)) {
      continue;
    }
    Entry entry{
        .instance_id = instance.instance_id,
        .daemon_id = instance.daemon_id,
        .execution_host_kind = execution_host_kind_for(instance),
        .execution_endpoint = instance.execution_endpoint,
        .engine = instance.engine,
        .capability_flags = instance.capability_flags,
    };
    const auto it = state.index_by_instance_id.find(instance.instance_id);
    if (it != state.index_by_instance_id.end()) {
      state.entries[it->second] = std::move(entry);
      continue;
    }
    state.index_by_instance_id.emplace(instance.instance_id, state.entries.size());
    state.entries.push_back(std::move(entry));
  }
  state.refreshed_at = now;
  state.cache_epoch += 1;
}

} // namespace tensorcast::daemon
