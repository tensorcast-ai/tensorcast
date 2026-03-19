// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/derived_view_export_manager.h"

#include <unistd.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {

namespace {

constexpr uint32_t kRetireDrainTimeoutMs = 30000;
constexpr std::string_view kResidentViewRouteKind = "resident_view";
constexpr uint64_t kDerivedViewBudgetFloorBytes = 16ULL << 30;
constexpr uint64_t kDerivedViewBudgetCeilBytes = 128ULL << 30;
constexpr uint64_t kDerivedViewHeadroomFloorBytes = 64ULL << 30;
constexpr uint64_t kDerivedViewHeadroomCeilBytes = 128ULL << 30;

struct DerivedBudgetWindow {
  uint64_t base_budget_bytes{0};
  uint64_t headroom_bytes{0};
  uint64_t resident_derived_bytes{0};
  uint64_t non_derived_stable_bytes{0};
  uint64_t effective_budget_bytes{0};
};

uint64_t clamp_derived_budget_bytes(uint64_t stable_total_bytes) {
  if (stable_total_bytes == 0) {
    return 0;
  }
  const uint64_t quarter_budget = stable_total_bytes / 4;
  return std::min(kDerivedViewBudgetCeilBytes, std::max(kDerivedViewBudgetFloorBytes, quarter_budget));
}

DerivedBudgetWindow compute_budget_window(
    const store::MemoryTierBudget::Snapshot& snapshot,
    uint64_t resident_derived_bytes,
    const DerivedViewExportManager::Options& options) {
  DerivedBudgetWindow window;
  window.base_budget_bytes =
      options.budget_override_bytes.value_or(clamp_derived_budget_bytes(snapshot.stable_total_bytes));
  if (snapshot.stable_total_bytes == 0) {
    window.headroom_bytes = options.headroom_override_bytes.value_or(0);
    window.resident_derived_bytes = resident_derived_bytes;
    window.effective_budget_bytes = window.base_budget_bytes;
    return window;
  }

  window.headroom_bytes = options.headroom_override_bytes.value_or(
      std::min(
          kDerivedViewHeadroomCeilBytes, std::max(kDerivedViewHeadroomFloorBytes, snapshot.stable_total_bytes / 8)));
  window.resident_derived_bytes = resident_derived_bytes;
  window.non_derived_stable_bytes =
      snapshot.stable_used_bytes > resident_derived_bytes ? snapshot.stable_used_bytes - resident_derived_bytes : 0;
  if (window.non_derived_stable_bytes + window.headroom_bytes >= snapshot.stable_total_bytes) {
    window.effective_budget_bytes = 0;
    return window;
  }

  const uint64_t headroom_limited_budget =
      snapshot.stable_total_bytes - window.non_derived_stable_bytes - window.headroom_bytes;
  window.effective_budget_bytes = std::min(window.base_budget_bytes, headroom_limited_budget);
  return window;
}

} // namespace

DerivedViewExportManager::DerivedViewExportManager(store::StoreEngine& engine, SessionLifecycleManager& lifecycle)
    : DerivedViewExportManager(engine, lifecycle, Options{}) {}

DerivedViewExportManager::DerivedViewExportManager(
    store::StoreEngine& engine,
    SessionLifecycleManager& lifecycle,
    Options options)
    : engine_(engine), lifecycle_(lifecycle), options_(std::move(options)), owner_pid_(::getpid()) {
  if (options_.ttl <= absl::ZeroDuration()) {
    options_.ttl = absl::Minutes(10);
  }
  if (options_.retry_retire_ttl <= absl::ZeroDuration()) {
    options_.retry_retire_ttl = absl::Seconds(30);
  }
}

void DerivedViewExportManager::set_global_store_client(std::shared_ptr<store::components::IGlobalStoreClient> client) {
  global_store_client_ = std::move(client);
}

bool DerivedViewExportManager::can_acquire_prepare_budget_locked(const PrepareBudgetWaitContext* ctx) {
  return ctx != nullptr && ctx->manager != nullptr &&
      (ctx->manager->pending_prepare_bytes_ + ctx->reserved_bytes) <= ctx->pending_budget_bytes;
}

bool DerivedViewExportManager::can_finish_local_drain_locked(const LocalDrainWaitContext* ctx) {
  if (ctx == nullptr || ctx->manager == nullptr) {
    return true;
  }
  auto it = ctx->manager->entries_.find(ctx->key);
  return it == ctx->manager->entries_.end() || it->second.generation != ctx->generation ||
      it->second.active_fetches == 0;
}

absl::Status DerivedViewExportManager::acquire_prepare_budget(uint64_t reserved_bytes) {
  if (reserved_bytes == 0) {
    return absl::OkStatus();
  }

  for (;;) {
    const auto snapshot_opt = engine_.get_memory_tier_snapshot();
    if (!snapshot_opt.has_value()) {
      return absl::OkStatus();
    }

    absl::MutexLock lock(&mu_);
    const DerivedBudgetWindow budget_window =
        compute_budget_window(*snapshot_opt, current_ready_derived_bytes_locked(), options_);
    const uint64_t pending_budget_bytes = std::max<uint64_t>(reserved_bytes, budget_window.effective_budget_bytes);
    PrepareBudgetWaitContext wait_ctx{
        .manager = this,
        .reserved_bytes = reserved_bytes,
        .pending_budget_bytes = pending_budget_bytes,
    };
    if (can_acquire_prepare_budget_locked(&wait_ctx)) {
      pending_prepare_bytes_ += reserved_bytes;
      VLOG(1) << "DerivedViewExportManager: acquired prepare budget reserved_bytes=" << reserved_bytes
              << " pending_prepare_bytes=" << pending_prepare_bytes_ << " pending_budget_bytes=" << pending_budget_bytes
              << " effective_budget_bytes=" << budget_window.effective_budget_bytes
              << " non_derived_stable_bytes=" << budget_window.non_derived_stable_bytes;
      return absl::OkStatus();
    }

    VLOG(1) << "DerivedViewExportManager: waiting for prepare budget reserved_bytes=" << reserved_bytes
            << " pending_prepare_bytes=" << pending_prepare_bytes_ << " pending_budget_bytes=" << pending_budget_bytes
            << " effective_budget_bytes=" << budget_window.effective_budget_bytes
            << " non_derived_stable_bytes=" << budget_window.non_derived_stable_bytes;
    mu_.Await(absl::Condition(&DerivedViewExportManager::can_acquire_prepare_budget_locked, &wait_ctx));
  }
}

void DerivedViewExportManager::release_prepare_budget(uint64_t reserved_bytes) {
  if (reserved_bytes == 0) {
    return;
  }

  absl::MutexLock lock(&mu_);
  pending_prepare_bytes_ = pending_prepare_bytes_ > reserved_bytes ? pending_prepare_bytes_ - reserved_bytes : 0;
  VLOG(1) << "DerivedViewExportManager: released prepare budget reserved_bytes=" << reserved_bytes
          << " pending_prepare_bytes=" << pending_prepare_bytes_;
}

std::optional<ArtifactDeviceKey> DerivedViewExportManager::to_entry_key(const store::loading::ReplicaKey& key) {
  if (!key.view_id.has_value() || key.view_id->empty()) {
    return std::nullopt;
  }
  return ArtifactDeviceKey{
      .artifact_id = key.artifact_id,
      .view_id = *key.view_id,
      .device_id = key.device.type == DeviceType::CPU ? -1 : key.device.ordinal,
  };
}

absl::Status DerivedViewExportManager::install_entry(
    const ArtifactDeviceKey& key,
    const store::loading::ReplicaKey& replica_key) {
  Entry entry;
  entry.replica_key = replica_key;
  entry.state = EntryState::kPending;
  entry.generation = next_generation_++;
  entry.ttl = options_.ttl;
  entry.active_fetches = 0;
  entry.accept_new_fetches = true;
  auto resident_bytes_or = engine_.get_replica_size(replica_key);
  if (!resident_bytes_or.ok()) {
    LOG(WARNING) << "DerivedViewExportManager: failed to query replica size for artifact_id=" << key.artifact_id
                 << " view_id=" << key.view_id << " generation=" << entry.generation << ": "
                 << resident_bytes_or.status();
    entry.resident_bytes = 0;
  } else {
    entry.resident_bytes = *resident_bytes_or;
  }
  entry.last_access_time = absl::Now();
  entry.expiry_time = absl::InfiniteFuture();
  entries_[key] = std::move(entry);
  return activate_reserved_entry(key, entries_[key]);
}

absl::Status DerivedViewExportManager::activate_reserved_entry(const ArtifactDeviceKey& key, Entry& entry) {
  auto use_lease_or = lifecycle_.create_use_lease(entry.replica_key, owner_pid_);
  if (!use_lease_or.ok()) {
    return use_lease_or.status();
  }

  auto retention_lease_or = lifecycle_.create_retention_lease(
      options_.ttl,
      {[this, key, generation = entry.generation]() { return this->on_retention_expired(key, generation); }});
  if (!retention_lease_or.ok()) {
    lifecycle_.release_lease(*use_lease_or);
    return retention_lease_or.status();
  }

  entry.use_lease_id = *use_lease_or;
  entry.retention_lease_id = *retention_lease_or;
  entry.state = EntryState::kReady;
  entry.ttl = options_.ttl;
  entry.active_fetches = 0;
  entry.accept_new_fetches = true;
  auto resident_bytes_or = engine_.get_replica_size(entry.replica_key);
  if (!resident_bytes_or.ok()) {
    LOG(WARNING) << "DerivedViewExportManager: failed to query replica size for artifact_id=" << key.artifact_id
                 << " view_id=" << key.view_id << " generation=" << entry.generation << ": "
                 << resident_bytes_or.status();
  } else {
    entry.resident_bytes = *resident_bytes_or;
  }
  entry.last_access_time = absl::Now();
  entry.expiry_time = entry.last_access_time + entry.ttl;
  VLOG(1) << "DerivedViewExportManager: installed entry artifact_id=" << key.artifact_id << " view_id=" << key.view_id
          << " device_id=" << key.device_id << " generation=" << entry.generation
          << " resident_bytes=" << entry.resident_bytes << " event=create"
          << " route_kind=" << kResidentViewRouteKind;
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::reserve(const store::loading::ReplicaKey& key, uint64_t reserved_bytes) {
  auto entry_key = to_entry_key(key);
  if (!entry_key.has_value()) {
    return absl::InvalidArgumentError("derived view export reservation requires ReplicaKey.view_id");
  }

  const auto snapshot_opt = engine_.get_memory_tier_snapshot();
  if (!snapshot_opt.has_value()) {
    return absl::OkStatus();
  }
  const auto resident_derived_bytes_locked = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    uint64_t total_bytes = 0;
    for (const auto& [entry_key_ignored, entry] : entries_) {
      (void)entry_key_ignored;
      if (entry.state == EntryState::kPending) {
        continue;
      }
      total_bytes += entry.resident_bytes;
    }
    return total_bytes;
  };

  std::vector<std::pair<ArtifactDeviceKey, Entry>> victims;
  uint64_t reserved_generation = 0;
  DerivedBudgetWindow budget_window;
  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(*entry_key);
    if (it != entries_.end()) {
      if (it->second.state == EntryState::kReady) {
        VLOG(1) << "DerivedViewExportManager: reuse hit artifact_id=" << entry_key->artifact_id
                << " view_id=" << entry_key->view_id << " device_id=" << entry_key->device_id
                << " generation=" << it->second.generation << " event=reuse_hit"
                << " route_kind=" << kResidentViewRouteKind;
        return renew_entry(*entry_key, it->second);
      }
      return absl::FailedPreconditionError("derived view export already pending or draining");
    }

    Entry entry;
    entry.replica_key = key;
    entry.state = EntryState::kPending;
    entry.generation = next_generation_++;
    entry.ttl = options_.ttl;
    entry.resident_bytes = reserved_bytes;
    entry.active_fetches = 0;
    entry.accept_new_fetches = true;
    entry.last_access_time = absl::Now();
    entry.expiry_time = absl::InfiniteFuture();
    entries_[*entry_key] = std::move(entry);
    reserved_generation = entries_[*entry_key].generation;

    budget_window = compute_budget_window(*snapshot_opt, resident_derived_bytes_locked(), options_);
    const uint64_t budget_bytes = budget_window.effective_budget_bytes;
    if (budget_bytes == 0) {
      entries_.erase(*entry_key);
      return absl::ResourceExhaustedError(
          absl::StrCat(
              "derived-view budget exhausted by non-derived stable residency: stable_total_bytes=",
              snapshot_opt->stable_total_bytes,
              " stable_used_bytes=",
              snapshot_opt->stable_used_bytes,
              " resident_derived_bytes=",
              budget_window.resident_derived_bytes,
              " non_derived_stable_bytes=",
              budget_window.non_derived_stable_bytes,
              " headroom_bytes=",
              budget_window.headroom_bytes));
    }

    uint64_t total_bytes = current_derived_bytes_locked();
    if (total_bytes <= budget_bytes) {
      VLOG(1) << "DerivedViewExportManager: reserved pending entry artifact_id=" << entry_key->artifact_id
              << " view_id=" << entry_key->view_id << " device_id=" << entry_key->device_id
              << " generation=" << entries_[*entry_key].generation << " reserved_bytes=" << reserved_bytes
              << " budget_bytes=" << budget_bytes << " base_budget_bytes=" << budget_window.base_budget_bytes
              << " headroom_bytes=" << budget_window.headroom_bytes
              << " non_derived_stable_bytes=" << budget_window.non_derived_stable_bytes
              << " resident_derived_bytes=" << budget_window.resident_derived_bytes;
      return absl::OkStatus();
    }

    std::vector<std::pair<ArtifactDeviceKey, Entry>> candidates;
    candidates.reserve(entries_.size());
    for (const auto& [candidate_key, candidate_entry] : entries_) {
      if (candidate_entry.state != EntryState::kReady || candidate_entry.active_fetches > 0) {
        continue;
      }
      if (candidate_key == *entry_key) {
        continue;
      }
      candidates.emplace_back(candidate_key, candidate_entry);
    }

    std::sort(candidates.begin(), candidates.end(), [entry_key](const auto& lhs, const auto& rhs) {
      const bool lhs_cross_artifact = lhs.first.artifact_id != entry_key->artifact_id;
      const bool rhs_cross_artifact = rhs.first.artifact_id != entry_key->artifact_id;
      if (lhs_cross_artifact != rhs_cross_artifact) {
        return lhs_cross_artifact > rhs_cross_artifact;
      }
      return lhs.second.last_access_time < rhs.second.last_access_time;
    });

    for (const auto& candidate : candidates) {
      const bool cross_artifact = candidate.first.artifact_id != entry_key->artifact_id;
      if (!cross_artifact && total_bytes <= budget_bytes) {
        break;
      }
      auto candidate_it = entries_.find(candidate.first);
      if (candidate_it == entries_.end() || candidate_it->second.generation != candidate.second.generation ||
          candidate_it->second.state != EntryState::kReady) {
        continue;
      }
      candidate_it->second.state = EntryState::kDraining;
      total_bytes = total_bytes > candidate.second.resident_bytes ? total_bytes - candidate.second.resident_bytes : 0;
      victims.push_back(candidate);
    }

    if (total_bytes > budget_bytes) {
      entries_.erase(*entry_key);
      return absl::ResourceExhaustedError(
          absl::StrCat(
              "unable to free enough derived-view budget before export: budget_bytes=",
              budget_bytes,
              " base_budget_bytes=",
              budget_window.base_budget_bytes,
              " headroom_bytes=",
              budget_window.headroom_bytes,
              " non_derived_stable_bytes=",
              budget_window.non_derived_stable_bytes,
              " current_derived_bytes=",
              total_bytes));
    }
  }

  if (!victims.empty()) {
    VLOG(1) << "DerivedViewExportManager: reservation eviction start victims=" << victims.size()
            << " budget_bytes=" << budget_window.effective_budget_bytes
            << " base_budget_bytes=" << budget_window.base_budget_bytes
            << " headroom_bytes=" << budget_window.headroom_bytes
            << " non_derived_stable_bytes=" << budget_window.non_derived_stable_bytes
            << " reserved_bytes=" << reserved_bytes;
  }
  for (const auto& [victim_key, victim_entry] : victims) {
    const absl::Status retire_status =
        retire_entry(victim_key, victim_entry, "pre_reserve_budget", /*retry_on_failure=*/false);
    if (!retire_status.ok()) {
      cancel_reserved(key).IgnoreError();
      return retire_status;
    }
  }

  VLOG(1) << "DerivedViewExportManager: reserved pending entry artifact_id=" << entry_key->artifact_id
          << " view_id=" << entry_key->view_id << " device_id=" << entry_key->device_id
          << " generation=" << reserved_generation << " reserved_bytes=" << reserved_bytes
          << " budget_bytes=" << budget_window.effective_budget_bytes
          << " base_budget_bytes=" << budget_window.base_budget_bytes
          << " headroom_bytes=" << budget_window.headroom_bytes
          << " non_derived_stable_bytes=" << budget_window.non_derived_stable_bytes;
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::commit_reserved(const store::loading::ReplicaKey& key) {
  auto entry_key = to_entry_key(key);
  if (!entry_key.has_value()) {
    return absl::InvalidArgumentError("derived view export commit requires ReplicaKey.view_id");
  }

  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(*entry_key);
    if (it == entries_.end()) {
      return absl::NotFoundError("derived view export reservation not found");
    }
    if (it->second.state == EntryState::kReady) {
      return renew_entry(*entry_key, it->second);
    }
    if (it->second.state != EntryState::kPending) {
      return absl::FailedPreconditionError("derived view export reservation is not pending");
    }
    return activate_reserved_entry(*entry_key, it->second);
  }
}

absl::Status DerivedViewExportManager::cancel_reserved(const store::loading::ReplicaKey& key) {
  auto entry_key = to_entry_key(key);
  if (!entry_key.has_value()) {
    return absl::InvalidArgumentError("derived view export cancel requires ReplicaKey.view_id");
  }

  absl::MutexLock lock(&mu_);
  auto it = entries_.find(*entry_key);
  if (it == entries_.end()) {
    return absl::NotFoundError("derived view export reservation not found");
  }
  if (it->second.state != EntryState::kPending) {
    return absl::FailedPreconditionError("derived view export reservation is not pending");
  }
  entries_.erase(it);
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::renew_entry(const ArtifactDeviceKey& key, Entry& entry) {
  if (entry.state != EntryState::kReady) {
    return absl::FailedPreconditionError("derived view export is draining");
  }
  absl::Status renew_status = lifecycle_.renew_retention(entry.retention_lease_id, entry.ttl);
  if (renew_status.ok()) {
    entry.last_access_time = absl::Now();
    entry.expiry_time = entry.last_access_time + entry.ttl;
    VLOG(2) << "DerivedViewExportManager: renewed entry artifact_id=" << key.artifact_id << " view_id=" << key.view_id
            << " device_id=" << key.device_id << " generation=" << entry.generation << " event=ttl_refresh"
            << " route_kind=" << kResidentViewRouteKind;
  }
  return renew_status;
}

absl::Status DerivedViewExportManager::retain_or_refresh(const store::loading::ReplicaKey& key) {
  auto entry_key = to_entry_key(key);
  if (!entry_key.has_value()) {
    return absl::InvalidArgumentError("derived view export retention requires ReplicaKey.view_id");
  }

  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(*entry_key);
    if (it != entries_.end()) {
      VLOG(1) << "DerivedViewExportManager: retain_or_refresh hit artifact_id=" << entry_key->artifact_id
              << " view_id=" << entry_key->view_id << " device_id=" << entry_key->device_id
              << " generation=" << it->second.generation << " event=reuse_hit"
              << " route_kind=" << kResidentViewRouteKind;
      return renew_entry(*entry_key, it->second);
    }
    absl::Status install_status = install_entry(*entry_key, key);
    if (!install_status.ok()) {
      return install_status;
    }
  }
  return maybe_evict_for_budget(*entry_key);
}

absl::Status DerivedViewExportManager::begin_fetch(const store::loading::ReplicaKey& key, std::string_view fetch_id) {
  auto entry_key = to_entry_key(key);
  if (!entry_key.has_value()) {
    return absl::InvalidArgumentError("derived view export begin_fetch requires ReplicaKey.view_id");
  }
  if (fetch_id.empty()) {
    return absl::InvalidArgumentError("derived view export begin_fetch requires fetch_id");
  }

  absl::MutexLock lock(&mu_);
  auto it = entries_.find(*entry_key);
  if (it == entries_.end()) {
    return absl::NotFoundError("derived view export not managed");
  }
  auto fetch_it = active_fetches_.find(std::string(fetch_id));
  if (fetch_it != active_fetches_.end()) {
    if (fetch_it->second.key == *entry_key && fetch_it->second.generation == it->second.generation) {
      VLOG(2) << "DerivedViewExportManager: begin fetch idempotent artifact_id=" << entry_key->artifact_id
              << " view_id=" << entry_key->view_id << " generation=" << it->second.generation
              << " fetch_id=" << fetch_id;
      return absl::OkStatus();
    }
    return absl::AlreadyExistsError(
        absl::StrCat("derived view export fetch_id already bound to another entry: fetch_id=", fetch_id));
  }
  if (it->second.state == EntryState::kPending) {
    return absl::FailedPreconditionError(
        "derived view export fetch rejected: entry is still preparing; fallback_reason=entry_pending");
  }
  if (!it->second.accept_new_fetches) {
    return absl::FailedPreconditionError(
        "derived view export fetch rejected: draining entry no longer accepts attaches; "
        "fallback_reason=drain_attach_closed");
  }

  if (it->second.state == EntryState::kReady) {
    const absl::Status renew_status = renew_entry(*entry_key, it->second);
    if (!renew_status.ok()) {
      LOG(WARNING) << "DerivedViewExportManager: data-plane TTL refresh failed for artifact_id="
                   << entry_key->artifact_id << " view_id=" << entry_key->view_id
                   << " generation=" << it->second.generation << ": " << renew_status;
    }
  } else {
    it->second.last_access_time = absl::Now();
    VLOG(1) << "DerivedViewExportManager: serving in-flight fetch on draining entry artifact_id="
            << entry_key->artifact_id << " view_id=" << entry_key->view_id << " generation=" << it->second.generation;
  }

  it->second.active_fetches += 1;
  active_fetches_.emplace(std::string(fetch_id), ActiveFetch{.key = *entry_key, .generation = it->second.generation});
  VLOG(2) << "DerivedViewExportManager: begin fetch artifact_id=" << entry_key->artifact_id
          << " view_id=" << entry_key->view_id << " generation=" << it->second.generation
          << " state=" << static_cast<int>(it->second.state) << " active_fetches=" << it->second.active_fetches
          << " fetch_id=" << fetch_id << " event=fetch_begin"
          << " route_kind=" << kResidentViewRouteKind;
  return absl::OkStatus();
}

void DerivedViewExportManager::end_fetch(std::string_view fetch_id, std::string_view reason) {
  if (fetch_id.empty()) {
    return;
  }

  absl::MutexLock lock(&mu_);
  auto fetch_it = active_fetches_.find(std::string(fetch_id));
  if (fetch_it == active_fetches_.end()) {
    VLOG(2) << "DerivedViewExportManager: end fetch ignored unknown fetch_id=" << fetch_id << " reason=" << reason;
    return;
  }
  const ArtifactDeviceKey entry_key = fetch_it->second.key;
  const uint64_t generation = fetch_it->second.generation;
  active_fetches_.erase(fetch_it);

  auto it = entries_.find(entry_key);
  if (it == entries_.end()) {
    return;
  }
  if (it->second.generation != generation) {
    VLOG(2) << "DerivedViewExportManager: end fetch ignored generation mismatch artifact_id=" << entry_key.artifact_id
            << " view_id=" << entry_key.view_id << " fetch_id=" << fetch_id << " reason=" << reason
            << " expected_generation=" << generation << " actual_generation=" << it->second.generation;
    return;
  }
  if (it->second.active_fetches == 0) {
    VLOG(1) << "DerivedViewExportManager: end fetch observed zero active_fetches artifact_id=" << entry_key.artifact_id
            << " view_id=" << entry_key.view_id << " reason=" << reason << " fetch_id=" << fetch_id;
    return;
  }

  it->second.active_fetches -= 1;
  VLOG(2) << "DerivedViewExportManager: end fetch artifact_id=" << entry_key.artifact_id
          << " view_id=" << entry_key.view_id << " generation=" << it->second.generation
          << " state=" << static_cast<int>(it->second.state) << " active_fetches=" << it->second.active_fetches
          << " reason=" << reason << " fetch_id=" << fetch_id << " event=fetch_end"
          << " route_kind=" << kResidentViewRouteKind;
}

std::optional<DerivedViewExportManager::EntrySnapshot> DerivedViewExportManager::find_entry(
    const store::loading::ReplicaKey& key) const {
  auto entry_key = to_entry_key(key);
  if (!entry_key.has_value()) {
    return std::nullopt;
  }

  absl::MutexLock lock(&mu_);
  auto it = entries_.find(*entry_key);
  if (it == entries_.end()) {
    return std::nullopt;
  }
  return EntrySnapshot{
      .key = *entry_key,
      .replica_key = it->second.replica_key,
      .use_lease_id = it->second.use_lease_id,
      .retention_lease_id = it->second.retention_lease_id,
      .state = it->second.state,
      .generation = it->second.generation,
      .ttl = it->second.ttl,
      .resident_bytes = it->second.resident_bytes,
      .active_fetches = it->second.active_fetches,
      .accept_new_fetches = it->second.accept_new_fetches,
      .last_access_time = it->second.last_access_time,
      .expiry_time = it->second.expiry_time,
  };
}

absl::Status DerivedViewExportManager::safe_retire_published_replica(
    const store::loading::ReplicaKey& key,
    std::string_view reason) const {
  engine_.set_replica_publish_state(key, store::StoreEngine::ReplicaPublishState::kRetiring);

  const auto replica_id = engine_.get_replica_global_store_id(key);
  if (!replica_id.has_value() || replica_id->empty()) {
    return absl::OkStatus();
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("Global Store client unavailable while retiring derived view export");
  }

  auto mark_or = global_store_client_->mark_replica_unavailable(key.artifact_id, *replica_id, std::string(reason));
  if (!mark_or.ok() && !absl::IsNotFound(mark_or.status())) {
    return mark_or.status();
  }

  auto drain_or = global_store_client_->wait_replica_drain(*replica_id, kRetireDrainTimeoutMs);
  if (!drain_or.ok() && !absl::IsNotFound(drain_or.status())) {
    return drain_or.status();
  }
  if (drain_or.ok() && !drain_or->drained) {
    return absl::DeadlineExceededError("derived view export drain timed out");
  }
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::unregister_published_replica(const store::loading::ReplicaKey& key) const {
  const auto replica_id = engine_.get_replica_global_store_id(key);
  if (!replica_id.has_value() || replica_id->empty()) {
    return absl::OkStatus();
  }
  if (!global_store_client_ || !global_store_client_->is_connected()) {
    return absl::FailedPreconditionError("Global Store client unavailable while unregistering derived view export");
  }
  absl::Status unregister_status = global_store_client_->unregister_replica(key.artifact_id, *replica_id);
  if (!unregister_status.ok() && !absl::IsNotFound(unregister_status)) {
    return unregister_status;
  }
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::arm_retry_retention(const ArtifactDeviceKey& key, uint64_t generation) {
  absl::MutexLock lock(&mu_);
  auto it = entries_.find(key);
  if (it == entries_.end() || it->second.generation != generation) {
    return absl::OkStatus();
  }

  auto retention_lease_or = lifecycle_.create_retention_lease(
      options_.retry_retire_ttl, {[this, key, generation]() { return this->on_retention_expired(key, generation); }});
  if (!retention_lease_or.ok()) {
    return retention_lease_or.status();
  }

  it->second.retention_lease_id = *retention_lease_or;
  it->second.state = EntryState::kReady;
  it->second.ttl = options_.ttl;
  it->second.accept_new_fetches = true;
  it->second.expiry_time = absl::Now() + options_.ttl;
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::arm_retry_retire(const ArtifactDeviceKey& key, uint64_t generation) {
  absl::MutexLock lock(&mu_);
  auto it = entries_.find(key);
  if (it == entries_.end() || it->second.generation != generation) {
    return absl::OkStatus();
  }

  auto retention_lease_or = lifecycle_.create_retention_lease(
      options_.retry_retire_ttl, {[this, key, generation]() { return this->on_retry_retire(key, generation); }});
  if (!retention_lease_or.ok()) {
    return retention_lease_or.status();
  }

  it->second.retention_lease_id = *retention_lease_or;
  it->second.expiry_time = absl::Now() + options_.retry_retire_ttl;
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::on_retry_retire(const ArtifactDeviceKey& key, uint64_t generation) {
  Entry snapshot;
  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(key);
    if (it == entries_.end() || it->second.generation != generation) {
      return absl::OkStatus();
    }
    if (it->second.state != EntryState::kDraining) {
      return absl::OkStatus();
    }
    snapshot = it->second;
  }
  return retire_entry(key, snapshot, "retry_retire", /*retry_on_failure=*/true);
}

uint64_t DerivedViewExportManager::current_derived_bytes_locked() const {
  uint64_t total_bytes = 0;
  for (const auto& [entry_key, entry] : entries_) {
    (void)entry_key;
    if (entry.state == EntryState::kDraining) {
      continue;
    }
    total_bytes += entry.resident_bytes;
  }
  return total_bytes;
}

uint64_t DerivedViewExportManager::current_ready_derived_bytes_locked() const {
  uint64_t total_bytes = 0;
  for (const auto& [entry_key, entry] : entries_) {
    (void)entry_key;
    if (entry.state != EntryState::kReady) {
      continue;
    }
    total_bytes += entry.resident_bytes;
  }
  return total_bytes;
}

absl::Status DerivedViewExportManager::maybe_evict_for_budget(const std::optional<ArtifactDeviceKey>& protected_key) {
  const auto snapshot_opt = engine_.get_memory_tier_snapshot();
  if (!snapshot_opt.has_value()) {
    return absl::OkStatus();
  }
  const auto resident_derived_bytes_locked = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    uint64_t total_bytes = 0;
    for (const auto& [entry_key_ignored, entry] : entries_) {
      (void)entry_key_ignored;
      if (entry.state == EntryState::kPending) {
        continue;
      }
      total_bytes += entry.resident_bytes;
    }
    return total_bytes;
  };
  DerivedBudgetWindow budget_window;
  uint64_t budget_bytes = 0;
  uint64_t total_bytes = 0;
  std::vector<std::pair<ArtifactDeviceKey, Entry>> victims;
  {
    absl::MutexLock lock(&mu_);
    budget_window = compute_budget_window(*snapshot_opt, resident_derived_bytes_locked(), options_);
    budget_bytes = budget_window.effective_budget_bytes;
    total_bytes = current_derived_bytes_locked();
    if (total_bytes <= budget_bytes) {
      return absl::OkStatus();
    }

    std::vector<std::pair<ArtifactDeviceKey, Entry>> candidates;
    candidates.reserve(entries_.size());
    for (const auto& [entry_key, entry] : entries_) {
      if (entry.state != EntryState::kReady || entry.active_fetches > 0) {
        continue;
      }
      if (protected_key.has_value() && *protected_key == entry_key) {
        continue;
      }
      candidates.emplace_back(entry_key, entry);
    }

    const absl::Time now = absl::Now();
    std::sort(candidates.begin(), candidates.end(), [now](const auto& lhs, const auto& rhs) {
      const bool lhs_expired = lhs.second.expiry_time <= now;
      const bool rhs_expired = rhs.second.expiry_time <= now;
      if (lhs_expired != rhs_expired) {
        return lhs_expired > rhs_expired;
      }
      return lhs.second.last_access_time < rhs.second.last_access_time;
    });

    for (const auto& candidate : candidates) {
      if (total_bytes <= budget_bytes) {
        break;
      }
      auto it = entries_.find(candidate.first);
      if (it == entries_.end() || it->second.generation != candidate.second.generation ||
          it->second.state != EntryState::kReady) {
        continue;
      }
      it->second.state = EntryState::kDraining;
      total_bytes = total_bytes > candidate.second.resident_bytes ? total_bytes - candidate.second.resident_bytes : 0;
      victims.push_back(candidate);
    }
  }

  if (victims.empty()) {
    VLOG(1) << "DerivedViewExportManager: derived-view budget exceeded but no idle victims available"
            << " total_bytes=" << total_bytes << " budget_bytes=" << budget_bytes
            << " base_budget_bytes=" << budget_window.base_budget_bytes
            << " headroom_bytes=" << budget_window.headroom_bytes
            << " non_derived_stable_bytes=" << budget_window.non_derived_stable_bytes;
    return absl::OkStatus();
  }

  VLOG(1) << "DerivedViewExportManager: pressure eviction start victims=" << victims.size()
          << " budget_bytes=" << budget_bytes << " base_budget_bytes=" << budget_window.base_budget_bytes
          << " headroom_bytes=" << budget_window.headroom_bytes
          << " non_derived_stable_bytes=" << budget_window.non_derived_stable_bytes << " event=eviction_start"
          << " route_kind=" << kResidentViewRouteKind;
  for (const auto& [victim_key, victim_entry] : victims) {
    const absl::Status retire_status =
        retire_entry(victim_key, victim_entry, "pressure_budget", /*retry_on_failure=*/false);
    if (!retire_status.ok()) {
      LOG(WARNING) << "DerivedViewExportManager: pressure eviction failed for artifact_id=" << victim_key.artifact_id
                   << " view_id=" << victim_key.view_id << " generation=" << victim_entry.generation << ": "
                   << retire_status;
    }
  }
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::retire_entry(
    const ArtifactDeviceKey& key,
    const Entry& snapshot,
    std::string_view reason,
    bool retry_on_failure) {
  VLOG(1) << "DerivedViewExportManager: retiring entry artifact_id=" << key.artifact_id << " view_id=" << key.view_id
          << " device_id=" << key.device_id << " generation=" << snapshot.generation << " reason=" << reason
          << " resident_bytes=" << snapshot.resident_bytes << " active_fetches=" << snapshot.active_fetches
          << " event=drain_start"
          << " route_kind=" << kResidentViewRouteKind;

  bool route_withdrawn = false;
  auto handle_retire_failure = [&](const absl::Status& status) -> absl::Status {
    LOG(WARNING) << "DerivedViewExportManager: retire step failed for artifact_id=" << key.artifact_id
                 << " view_id=" << key.view_id << " generation=" << snapshot.generation << " reason=" << reason
                 << " route_withdrawn=" << route_withdrawn << ": " << status;
    if (!route_withdrawn) {
      {
        absl::MutexLock lock(&mu_);
        auto it = entries_.find(key);
        if (it != entries_.end() && it->second.generation == snapshot.generation) {
          it->second.state = EntryState::kReady;
          it->second.accept_new_fetches = true;
        }
      }
      if (retry_on_failure) {
        const absl::Status retry_status = arm_retry_retention(key, snapshot.generation);
        if (!retry_status.ok()) {
          LOG(WARNING) << "DerivedViewExportManager: retry retention arm failed for artifact_id=" << key.artifact_id
                       << " view_id=" << key.view_id << " generation=" << snapshot.generation << ": " << retry_status;
        }
      }
      engine_.set_replica_publish_state(snapshot.replica_key, store::StoreEngine::ReplicaPublishState::kPublished);
      return status;
    }

    {
      absl::MutexLock lock(&mu_);
      auto it = entries_.find(key);
      if (it != entries_.end() && it->second.generation == snapshot.generation) {
        it->second.state = EntryState::kDraining;
        it->second.accept_new_fetches = false;
      }
    }
    const absl::Status retry_status = arm_retry_retire(key, snapshot.generation);
    if (!retry_status.ok()) {
      LOG(WARNING) << "DerivedViewExportManager: retry retire arm failed for artifact_id=" << key.artifact_id
                   << " view_id=" << key.view_id << " generation=" << snapshot.generation << ": " << retry_status;
    }
    return status;
  };

  const absl::Status retire_status = safe_retire_published_replica(snapshot.replica_key, reason);
  if (!retire_status.ok()) {
    return handle_retire_failure(retire_status);
  }
  route_withdrawn = true;

  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(key);
    if (it != entries_.end() && it->second.generation == snapshot.generation) {
      it->second.accept_new_fetches = false;
    }
  }

  LocalDrainWaitContext wait_ctx{
      .manager = this,
      .key = key,
      .generation = snapshot.generation,
  };
  bool local_drain_complete = false;
  {
    absl::MutexLock lock(&mu_);
    local_drain_complete = mu_.AwaitWithTimeout(
        absl::Condition(&DerivedViewExportManager::can_finish_local_drain_locked, &wait_ctx),
        absl::Milliseconds(kRetireDrainTimeoutMs));
  }
  if (!local_drain_complete) {
    return handle_retire_failure(
        absl::DeadlineExceededError(
            absl::StrCat(
                "derived view export local drain timed out: artifact_id=",
                key.artifact_id,
                " view_id=",
                key.view_id,
                " generation=",
                snapshot.generation)));
  }
  VLOG(1) << "DerivedViewExportManager: local drain complete artifact_id=" << key.artifact_id
          << " view_id=" << key.view_id << " generation=" << snapshot.generation << " event=drain_complete"
          << " route_kind=" << kResidentViewRouteKind;

  const absl::Status unregister_status = unregister_published_replica(snapshot.replica_key);
  if (!unregister_status.ok()) {
    return handle_retire_failure(unregister_status);
  }

  lifecycle_.release_lease(snapshot.retention_lease_id);
  lifecycle_.release_lease(snapshot.use_lease_id);
  const absl::Status retire_status_runtime = engine_.retire_replica_status(snapshot.replica_key);
  if (!retire_status_runtime.ok() && !absl::IsNotFound(retire_status_runtime)) {
    LOG(WARNING) << "DerivedViewExportManager: runtime retire failed for artifact_id=" << key.artifact_id
                 << " view_id=" << key.view_id << " generation=" << snapshot.generation << " reason=" << reason << ": "
                 << retire_status_runtime;
  }

  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(key);
    if (it != entries_.end() && it->second.generation == snapshot.generation) {
      entries_.erase(it);
    }
  }

  VLOG(1) << "DerivedViewExportManager: retired entry artifact_id=" << key.artifact_id << " view_id=" << key.view_id
          << " device_id=" << key.device_id << " generation=" << snapshot.generation << " reason=" << reason
          << " event=retire_complete"
          << " route_kind=" << kResidentViewRouteKind;
  return absl::OkStatus();
}

absl::Status DerivedViewExportManager::on_retention_expired(const ArtifactDeviceKey& key, uint64_t generation) {
  Entry snapshot;
  {
    absl::MutexLock lock(&mu_);
    auto it = entries_.find(key);
    if (it == entries_.end() || it->second.generation != generation) {
      return absl::OkStatus();
    }
    if (it->second.state != EntryState::kReady) {
      VLOG(2) << "DerivedViewExportManager: skip retention expiry for artifact_id=" << key.artifact_id
              << " view_id=" << key.view_id << " device_id=" << key.device_id << " generation=" << generation
              << " because state=" << static_cast<int>(it->second.state);
      return absl::OkStatus();
    }
    it->second.state = EntryState::kDraining;
    it->second.accept_new_fetches = true;
    snapshot = it->second;
  }
  return retire_entry(key, snapshot, "ttl_expired", /*retry_on_failure=*/true);
}

} // namespace tensorcast::daemon
