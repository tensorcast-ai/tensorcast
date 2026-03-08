// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/worker_directory_cache.h"

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {

namespace {

bool is_entry_fresh(const WorkerDirectoryCache::Entry& entry, absl::Time now, absl::Duration staleness_budget) {
  return !entry.address.empty() && entry.refreshed_at != absl::UnixEpoch() &&
      now - entry.refreshed_at <= staleness_budget;
}

} // namespace

WorkerDirectoryCache::WorkerDirectoryCache(std::shared_ptr<store::components::IGlobalStoreClient> global_store_client)
    : global_store_client_(std::move(global_store_client)) {}

bool WorkerDirectoryCache::is_fresh(std::string_view daemon_id, absl::Time now, absl::Duration staleness_budget) const {
  absl::MutexLock lock(&mu_);
  const auto it = entries_.find(std::string(daemon_id));
  return it != entries_.end() && is_entry_fresh(it->second, now, staleness_budget);
}

absl::Status WorkerDirectoryCache::warm_for_daemons(
    const std::vector<std::string>& daemon_ids,
    absl::Time now,
    absl::Duration staleness_budget) {
  bool needs_refresh = false;
  {
    absl::MutexLock lock(&mu_);
    for (const auto& daemon_id : daemon_ids) {
      const auto it = entries_.find(daemon_id);
      if (it == entries_.end() || !is_entry_fresh(it->second, now, staleness_budget)) {
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
  rpc_opts.timeout = std::max(absl::Milliseconds(1), staleness_budget);
  auto workers_or = global_store_client_->list_active_workers(
      /*include_unavailable=*/false, /*required_capability_flags=*/0, rpc_opts);
  if (!workers_or.ok()) {
    return workers_or.status();
  }

  absl::flat_hash_set<std::string> requested_daemon_ids;
  requested_daemon_ids.reserve(daemon_ids.size());
  for (const auto& daemon_id : daemon_ids) {
    if (!daemon_id.empty()) {
      requested_daemon_ids.insert(daemon_id);
    }
  }

  absl::MutexLock lock(&mu_);
  update_from_workers(*workers_or, requested_daemon_ids, now);
  return absl::OkStatus();
}

absl::StatusOr<std::string> WorkerDirectoryCache::resolve_daemon_address(
    std::string_view daemon_id,
    absl::Time now,
    absl::Duration staleness_budget) {
  {
    absl::MutexLock lock(&mu_);
    const auto it = entries_.find(std::string(daemon_id));
    if (it != entries_.end() && is_entry_fresh(it->second, now, staleness_budget)) {
      return it->second.address;
    }
  }

  std::vector<std::string> daemon_ids;
  daemon_ids.emplace_back(daemon_id);
  auto warm_status = warm_for_daemons(daemon_ids, now, staleness_budget);
  if (!warm_status.ok()) {
    return warm_status;
  }

  absl::MutexLock lock(&mu_);
  const auto it = entries_.find(std::string(daemon_id));
  if (it == entries_.end() || !is_entry_fresh(it->second, now, staleness_budget)) {
    return absl::NotFoundError("daemon endpoint not found");
  }
  return it->second.address;
}

void WorkerDirectoryCache::update_from_workers(
    const std::vector<store::components::ActiveWorkerInfo>& workers,
    const absl::flat_hash_set<std::string>& requested_daemon_ids,
    absl::Time now) {
  absl::flat_hash_set<std::string> refreshed_daemon_ids;
  refreshed_daemon_ids.reserve(requested_daemon_ids.size());
  for (const auto& worker : workers) {
    if (worker.daemon_id.empty() || worker.node_address.empty() || worker.grpc_port == 0) {
      continue;
    }
    if (!requested_daemon_ids.empty() && !requested_daemon_ids.contains(worker.daemon_id)) {
      continue;
    }
    entries_[worker.daemon_id] = Entry{
        .address = absl::StrCat(worker.node_address, ":", worker.grpc_port),
        .refreshed_at = now,
    };
    refreshed_daemon_ids.insert(worker.daemon_id);
  }
  for (const auto& daemon_id : requested_daemon_ids) {
    if (!refreshed_daemon_ids.contains(daemon_id)) {
      entries_.erase(daemon_id);
    }
  }
}

} // namespace tensorcast::daemon
