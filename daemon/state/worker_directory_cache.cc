// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/worker_directory_cache.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "core/communicator/misc/utils.h"

namespace tensorcast::daemon {

namespace {

bool address_is_routable(const store::components::ActiveWorkerInfo& worker) {
  return !worker.daemon_id.empty() && !worker.node_address.empty() && worker.grpc_port != 0;
}

absl::Duration rpc_timeout(absl::Duration staleness_budget) {
  return std::max(absl::Milliseconds(1), staleness_budget);
}

std::string canonicalize_same_host_address(std::string_view address, std::string_view local_default_ip) {
  if (local_default_ip.empty()) {
    return std::string(address);
  }
  const std::size_t port_sep = address.rfind(':');
  if (port_sep == std::string_view::npos) {
    return std::string(address);
  }
  const std::string_view host = address.substr(0, port_sep);
  if (host != local_default_ip) {
    return std::string(address);
  }
  return absl::StrCat("127.0.0.1", address.substr(port_sep));
}

} // namespace

WorkerDirectoryCache::WorkerDirectoryCache(std::shared_ptr<store::components::IGlobalStoreClient> global_store_client)
    : global_store_client_(std::move(global_store_client)) {}

void WorkerDirectoryCache::update_local_entry(Entry entry) {
  const std::string local_default_ip = communicator::misc::get_default_ip();
  if (entry.address.empty() && !entry.node_address.empty() && entry.grpc_port != 0) {
    entry.address = absl::StrCat(entry.node_address, ":", entry.grpc_port);
  }
  entry.address = canonicalize_same_host_address(entry.address, local_default_ip);

  absl::MutexLock lock(&mu_);
  local_entry_ = std::move(entry);
  overlay_local_entry(active_state_);
  overlay_local_entry(include_unavailable_state_);
}

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

std::optional<WorkerDirectoryCache::Entry> WorkerDirectoryCache::local_entry_for(std::string_view daemon_id) const {
  if (!local_entry_.has_value() || local_entry_->daemon_id != daemon_id || local_entry_->address.empty()) {
    return std::nullopt;
  }
  return *local_entry_;
}

void WorkerDirectoryCache::overlay_local_entry(CacheState& state) {
  if (!local_entry_.has_value() || local_entry_->daemon_id.empty() || local_entry_->address.empty()) {
    return;
  }
  const auto it = state.index_by_daemon_id.find(local_entry_->daemon_id);
  if (it != state.index_by_daemon_id.end()) {
    state.entries[it->second] = *local_entry_;
    return;
  }
  state.index_by_daemon_id.emplace(local_entry_->daemon_id, state.entries.size());
  state.entries.push_back(*local_entry_);
}

bool WorkerDirectoryCache::is_fresh(std::string_view daemon_id, absl::Time now, absl::Duration staleness_budget) const {
  absl::MutexLock lock(&mu_);
  if (local_entry_for(daemon_id).has_value()) {
    return true;
  }
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
      bool has_remote_daemon = false;
      for (const auto& daemon_id : daemon_ids) {
        if (daemon_id.empty() || local_entry_for(daemon_id).has_value()) {
          continue;
        }
        has_remote_daemon = true;
        break;
      }
      if (has_remote_daemon) {
        needs_refresh = true;
      }
    }
    for (const auto& daemon_id : daemon_ids) {
      if (daemon_id.empty() || local_entry_for(daemon_id).has_value()) {
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
  const std::string local_default_ip = communicator::misc::get_default_ip();

  absl::MutexLock lock(&mu_);
  update_from_workers(*workers_or, active_state_, now, local_default_ip);
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
  const std::string local_default_ip = communicator::misc::get_default_ip();

  absl::MutexLock lock(&mu_);
  auto& state = state_for(include_unavailable);
  update_from_workers(*workers_or, state, now, local_default_ip);
  return filtered_snapshot(state, required_capability_flags);
}

absl::StatusOr<std::string> WorkerDirectoryCache::resolve_daemon_address(
    std::string_view daemon_id,
    absl::Time now,
    absl::Duration staleness_budget) {
  auto entry_or = resolve_daemon_entry(daemon_id, now, staleness_budget);
  if (!entry_or.ok()) {
    return entry_or.status();
  }
  return entry_or->address;
}

absl::StatusOr<WorkerDirectoryCache::Entry> WorkerDirectoryCache::resolve_daemon_entry(
    std::string_view daemon_id,
    absl::Time now,
    absl::Duration staleness_budget) {
  const absl::Time resolve_started_at = absl::Now();
  bool cache_hit_before_refresh = false;
  bool local_fast_path = false;
  std::optional<Entry> resolved_entry;
  {
    absl::MutexLock lock(&mu_);
    resolved_entry = local_entry_for(daemon_id);
    if (resolved_entry.has_value()) {
      cache_hit_before_refresh = true;
      local_fast_path = true;
    } else {
      const auto& state = state_for(/*include_unavailable=*/false);
      if (is_state_fresh(state, now, staleness_budget)) {
        const auto it = state.index_by_daemon_id.find(std::string(daemon_id));
        if (it != state.index_by_daemon_id.end() && !state.entries[it->second].address.empty()) {
          cache_hit_before_refresh = true;
          resolved_entry = state.entries[it->second];
        }
      }
    }
  }
  if (resolved_entry.has_value()) {
    LOG(INFO) << "worker_directory_cache.resolve_daemon_entry_summary"
              << " daemon_id=" << daemon_id << " cache_hit=true"
              << " refreshed=false"
              << " local_fast_path=" << local_fast_path << " refresh_ms=0"
              << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - resolve_started_at);
    return *resolved_entry;
  }

  const absl::Time refresh_started_at = absl::Now();
  std::vector<std::string> daemon_ids;
  daemon_ids.emplace_back(daemon_id);
  auto warm_status = warm_for_daemons(daemon_ids, now, staleness_budget);
  const absl::Duration refresh_elapsed = absl::Now() - refresh_started_at;
  if (!warm_status.ok()) {
    LOG(INFO) << "worker_directory_cache.resolve_daemon_entry_summary"
              << " daemon_id=" << daemon_id << " cache_hit=" << cache_hit_before_refresh << " refreshed=true"
              << " local_fast_path=false"
              << " refresh_ms=" << absl::ToDoubleMilliseconds(refresh_elapsed)
              << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - resolve_started_at)
              << " status=" << warm_status;
    return warm_status;
  }

  {
    absl::MutexLock lock(&mu_);
    resolved_entry = local_entry_for(daemon_id);
    if (!resolved_entry.has_value()) {
      const auto& state = state_for(/*include_unavailable=*/false);
      const auto it = state.index_by_daemon_id.find(std::string(daemon_id));
      if (!is_state_fresh(state, now, staleness_budget) || it == state.index_by_daemon_id.end() ||
          state.entries[it->second].address.empty()) {
        LOG(INFO) << "worker_directory_cache.resolve_daemon_entry_summary"
                  << " daemon_id=" << daemon_id << " cache_hit=" << cache_hit_before_refresh << " refreshed=true"
                  << " local_fast_path=false"
                  << " refresh_ms=" << absl::ToDoubleMilliseconds(refresh_elapsed)
                  << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - resolve_started_at)
                  << " status=NOT_FOUND";
        return absl::NotFoundError("daemon endpoint not found");
      }
      resolved_entry = state.entries[it->second];
    }
  }
  LOG(INFO) << "worker_directory_cache.resolve_daemon_entry_summary"
            << " daemon_id=" << daemon_id << " cache_hit=" << cache_hit_before_refresh << " refreshed=true"
            << " local_fast_path=false"
            << " refresh_ms=" << absl::ToDoubleMilliseconds(refresh_elapsed)
            << " total_ms=" << absl::ToDoubleMilliseconds(absl::Now() - resolve_started_at);
  return *resolved_entry;
}

void WorkerDirectoryCache::update_from_workers(
    const std::vector<store::components::ActiveWorkerInfo>& workers,
    CacheState& state,
    absl::Time now,
    std::string_view local_default_ip) {
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
        .node_id = worker.node_id,
        .node_address = worker.node_address,
        .grpc_port = worker.grpc_port,
        .p2p_port = worker.p2p_port,
        .address =
            canonicalize_same_host_address(absl::StrCat(worker.node_address, ":", worker.grpc_port), local_default_ip),
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
  overlay_local_entry(state);
  state.refreshed_at = now;
  state.cache_epoch += 1;
}

} // namespace tensorcast::daemon
