// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/worker_directory_cache.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {

namespace {

bool address_is_routable(const store::components::ActiveWorkerInfo& worker) {
  return !worker.daemon_id.empty() && !worker.node_address.empty() && worker.grpc_port != 0;
}

absl::Duration rpc_timeout(absl::Duration staleness_budget) {
  return std::max(absl::Milliseconds(1), staleness_budget);
}

} // namespace

WorkerDirectoryCache::WorkerDirectoryCache(std::shared_ptr<store::components::IGlobalStoreClient> global_store_client)
    : global_store_client_(std::move(global_store_client)) {}

bool WorkerDirectoryCache::is_state_fresh(const CacheState& state, absl::Time now, absl::Duration staleness_budget) {
  return state.refreshed_at != absl::UnixEpoch() && now - state.refreshed_at <= staleness_budget;
}

WorkerDirectoryCache::Snapshot WorkerDirectoryCache::filtered_snapshot(
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

WorkerDirectoryCache::CacheState& WorkerDirectoryCache::state_for(bool include_unavailable) {
  return include_unavailable ? include_unavailable_state_ : active_state_;
}

const WorkerDirectoryCache::CacheState& WorkerDirectoryCache::state_for(bool include_unavailable) const {
  return include_unavailable ? include_unavailable_state_ : active_state_;
}

bool WorkerDirectoryCache::is_fresh(std::string_view daemon_id, absl::Time now, absl::Duration staleness_budget) const {
  absl::MutexLock lock(&mu_);
  const auto& state = state_for(/*include_unavailable=*/false);
  if (!is_state_fresh(state, now, staleness_budget)) {
    return false;
  }
  const auto it = state.index_by_daemon_id.find(std::string(daemon_id));
  return it != state.index_by_daemon_id.end() && !state.entries[it->second].address.empty();
}

absl::Status WorkerDirectoryCache::warm_for_daemons(
    const std::vector<std::string>& daemon_ids,
    absl::Time now,
    absl::Duration staleness_budget) {
  bool needs_refresh = false;
  {
    absl::MutexLock lock(&mu_);
    const auto& state = state_for(/*include_unavailable=*/false);
    if (!is_state_fresh(state, now, staleness_budget)) {
      needs_refresh = true;
    }
    for (const auto& daemon_id : daemon_ids) {
      if (daemon_id.empty()) {
        continue;
      }
      const auto it = state.index_by_daemon_id.find(daemon_id);
      if (it == state.index_by_daemon_id.end()) {
        needs_refresh = true;
        break;
      }
    }
  }
  if (!needs_refresh) {
    return absl::OkStatus();
  }
  if (global_store_client_ == nullptr || !global_store_client_->is_connected()) {
    return absl::UnavailableError("Global Store client unavailable for worker directory refresh");
  }

  store::components::RpcOptions rpc_opts;
  rpc_opts.timeout = rpc_timeout(staleness_budget);
  auto workers_or = global_store_client_->list_active_workers(
      /*include_unavailable=*/false, /*required_capability_flags=*/0, rpc_opts);
  if (!workers_or.ok()) {
    return workers_or.status();
  }

  absl::MutexLock lock(&mu_);
  update_from_workers(*workers_or, active_state_, now);
  return absl::OkStatus();
}

absl::StatusOr<WorkerDirectoryCache::Snapshot> WorkerDirectoryCache::list_workers(
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
    return absl::UnavailableError("Global Store client unavailable for worker directory refresh");
  }

  store::components::RpcOptions rpc_opts;
  rpc_opts.timeout = rpc_timeout(staleness_budget);
  auto workers_or = global_store_client_->list_active_workers(
      include_unavailable,
      /*required_capability_flags=*/0,
      rpc_opts);
  if (!workers_or.ok()) {
    return workers_or.status();
  }

  absl::MutexLock lock(&mu_);
  auto& state = state_for(include_unavailable);
  update_from_workers(*workers_or, state, now);
  return filtered_snapshot(state, required_capability_flags);
}

absl::StatusOr<std::string> WorkerDirectoryCache::resolve_daemon_address(
    std::string_view daemon_id,
    absl::Time now,
    absl::Duration staleness_budget) {
  {
    absl::MutexLock lock(&mu_);
    const auto& state = state_for(/*include_unavailable=*/false);
    if (is_state_fresh(state, now, staleness_budget)) {
      const auto it = state.index_by_daemon_id.find(std::string(daemon_id));
      if (it != state.index_by_daemon_id.end() && !state.entries[it->second].address.empty()) {
        return state.entries[it->second].address;
      }
    }
  }

  std::vector<std::string> daemon_ids;
  daemon_ids.emplace_back(daemon_id);
  auto warm_status = warm_for_daemons(daemon_ids, now, staleness_budget);
  if (!warm_status.ok()) {
    return warm_status;
  }

  absl::MutexLock lock(&mu_);
  const auto& state = state_for(/*include_unavailable=*/false);
  const auto it = state.index_by_daemon_id.find(std::string(daemon_id));
  if (!is_state_fresh(state, now, staleness_budget) || it == state.index_by_daemon_id.end() ||
      state.entries[it->second].address.empty()) {
    return absl::NotFoundError("daemon endpoint not found");
  }
  return state.entries[it->second].address;
}

void WorkerDirectoryCache::update_from_workers(
    const std::vector<store::components::ActiveWorkerInfo>& workers,
    CacheState& state,
    absl::Time now) {
  state.entries.clear();
  state.index_by_daemon_id.clear();
  state.entries.reserve(workers.size());
  for (const auto& worker : workers) {
    if (!address_is_routable(worker)) {
      continue;
    }
    Entry entry{
        .daemon_id = worker.daemon_id,
        .worker_id = worker.worker_id,
        .address = absl::StrCat(worker.node_address, ":", worker.grpc_port),
        .capability_flags = worker.capability_flags,
    };
    const auto it = state.index_by_daemon_id.find(worker.daemon_id);
    if (it != state.index_by_daemon_id.end()) {
      state.entries[it->second] = std::move(entry);
      continue;
    }
    state.index_by_daemon_id.emplace(worker.daemon_id, state.entries.size());
    state.entries.push_back(std::move(entry));
  }
  state.refreshed_at = now;
  state.cache_epoch += 1;
}

} // namespace tensorcast::daemon
