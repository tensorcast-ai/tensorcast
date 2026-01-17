// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/session_lifecycle.h"

#include <utility>

#include <unistd.h>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "core/store/device_registry.h"
#include "core/store/replica/memory_state.h"
#include "core/store/store_engine.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/pid_monitor.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/replica_session_manager.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon {

SessionLifecycleManager::SessionLifecycleManager(
    ReplicaSessionManager& sessions,
    RefTracker& refs,
    LipManager& lip,
    store::StoreEngine& engine)
    : sessions_(sessions), refs_(refs), lip_(lip), engine_(&engine) {}

SessionLifecycleManager::SessionLifecycleManager(ReplicaSessionManager& sessions, RefTracker& refs, LipManager& lip)
    : sessions_(sessions), refs_(refs), lip_(lip), engine_(nullptr) {}

void SessionLifecycleManager::sweep_once() {
  // Ensure we're watching all currently-referenced PIDs for exit notifications
  sync_pid_watches_();
  // Legacy sweeps removed: sessions TTL handled by session guards; join TTL via TTL UseLeases.
  // If pidfd monitor is active, event-driven handle_pid_exit() will drop refs;
  // keep a minimal fallback scan for environments without pidfd.
  PidMonitor* mon = nullptr;
  {
    absl::MutexLock lock(&mu_);
    mon = monitor_;
  }
  if (!mon || !mon->using_pidfd()) {
    sweep_pid_liveness();
  }
  expire_due(absl::Now());
}

void SessionLifecycleManager::set_eviction_notify(std::function<void(const ReplicaSubject&)> fn) {
  absl::MutexLock lock(&mu_);
  eviction_notify_ = std::move(fn);
}

void SessionLifecycleManager::set_schedule_hook(std::function<void(absl::Time)> fn) {
  absl::MutexLock lock(&mu_);
  schedule_hook_ = std::move(fn);
}

absl::StatusOr<SessionLifecycleManager::LeaseId> SessionLifecycleManager::create_use_lease(
    const ReplicaSubject& subj,
    pid_t pid,
    std::vector<std::function<absl::Status()>> extra_finalizers) {
  PidMonitor* mon = nullptr;
  LeaseId created_id = 0;
  {
    absl::MutexLock lock(&mu_);
    LeaseRec r;
    r.id = next_id_++;
    r.kind = LeaseKind::kUse;
    r.subj = subj;
    r.pid = pid;
    // Guards: PID liveness
    GuardRec g;
    g.id = next_guard_id_++;
    g.kind = GuardKind::kPidLiveness;
    g.lease = r.id;
    g.generation = 1;
    g.pid = pid;
    guard_by_id_.emplace(g.id, g);
    r.guards.push_back(g.id);
    pid_index_[pid].insert(g.id);
    mon = monitor_;
    // Finalizer: drop ref for this pid+subject, then attempt immediate reclaim if eligible
    r.finalizers.emplace_back([this, subj, pid]() -> absl::Status {
      refs_.drop_ref(subj, pid);
      // Attempt immediate reclaim if no active uses or pins remain and a daemon-owned GPU replica exists.
      if (subj.device.type == DeviceType::GPU) {
        maybe_unload_daemon_replica_(subj);
      }
      return absl::OkStatus();
    });
    for (auto& f : extra_finalizers) {
      r.finalizers.emplace_back(std::move(f));
    }
    created_id = r.id;
    by_id_[r.id] = std::move(r);
    inc_use_(subj);
  }
  if (mon) {
    mon->watch(pid);
  }
  return created_id;
}

absl::StatusOr<SessionLifecycleManager::LeaseId> SessionLifecycleManager::create_ttl_use_lease(
    const ReplicaSubject& subj,
    pid_t pid,
    absl::Duration ttl,
    std::vector<std::function<absl::Status()>> extra_finalizers) {
  if (ttl <= absl::ZeroDuration())
    return absl::InvalidArgumentError("ttl must be > 0");
  PidMonitor* mon = nullptr;
  LeaseId created_id = 0;
  std::optional<absl::Time> notify_when;
  {
    absl::MutexLock lock(&mu_);
    LeaseRec r;
    r.id = next_id_++;
    r.kind = LeaseKind::kUse;
    r.subj = subj;
    r.pid = pid;
    // PID liveness guard
    GuardRec g1;
    g1.id = next_guard_id_++;
    g1.kind = GuardKind::kPidLiveness;
    g1.lease = r.id;
    g1.generation = 1;
    g1.pid = pid;
    guard_by_id_.emplace(g1.id, g1);
    r.guards.push_back(g1.id);
    pid_index_[pid].insert(g1.id);
    // TTL deadline guard
    GuardRec g2;
    g2.id = next_guard_id_++;
    g2.kind = GuardKind::kDeadline;
    g2.lease = r.id;
    g2.generation = 1;
    g2.deadline = absl::Now() + ttl;
    guard_by_id_.emplace(g2.id, g2);
    r.guards.push_back(g2.id);
    notify_when = push_deadline_(g2.id, g2.deadline, g2.generation);

    mon = monitor_;
    // Finalizer: drop RefTracker and attempt immediate reclaim
    r.finalizers.emplace_back([this, subj, pid]() -> absl::Status {
      refs_.drop_ref(subj, pid);
      if (subj.device.type == DeviceType::GPU) {
        maybe_unload_daemon_replica_(subj);
      }
      return absl::OkStatus();
    });
    for (auto& f : extra_finalizers) {
      r.finalizers.emplace_back(std::move(f));
    }
    created_id = r.id;
    by_id_[r.id] = std::move(r);
    inc_use_(subj);
  }
  notify_schedule_if_earlier_(notify_when);
  if (mon)
    mon->watch(pid);
  return created_id;
}

absl::StatusOr<SessionLifecycleManager::LeaseId> SessionLifecycleManager::create_placement_lease(
    const ReplicaSubject& subj,
    /*spec*/ absl::Duration ttl) {
  LeaseId created_id = 0;
  std::optional<absl::Time> notify_when;
  {
    absl::MutexLock lock(&mu_);
    LeaseRec r;
    r.id = next_id_++;
    r.kind = LeaseKind::kPlacement;
    r.subj = subj;
    r.pid = -1;
    if (ttl > absl::ZeroDuration()) {
      GuardRec g;
      g.id = next_guard_id_++;
      g.kind = GuardKind::kDeadline;
      g.lease = r.id;
      g.generation = 1;
      g.deadline = absl::Now() + ttl;
      guard_by_id_.emplace(g.id, g);
      r.guards.push_back(g.id);
      notify_when = push_deadline_(g.id, g.deadline, g.generation);
    } else {
      GuardRec g;
      g.id = next_guard_id_++;
      g.kind = GuardKind::kManual;
      g.lease = r.id;
      g.generation = 1;
      guard_by_id_.emplace(g.id, g);
      r.guards.push_back(g.id);
    }
    // Finalizer: attempt immediate reclaim if this was the last pin and no active uses remain
    r.finalizers.emplace_back([this, subj]() -> absl::Status {
      maybe_unload_daemon_replica_(subj);
      return absl::OkStatus();
    });
    created_id = r.id;
    by_id_[r.id] = std::move(r);
    inc_pin_(subj);
  }
  notify_schedule_if_earlier_(notify_when);
  return created_id;
}

absl::StatusOr<SessionLifecycleManager::LeaseId> SessionLifecycleManager::create_commit_lease(
    const CommitSubject& subj,
    pid_t pid) {
  PidMonitor* mon = nullptr;
  LeaseId created_id = 0;
  {
    absl::MutexLock lock(&mu_);
    LeaseRec r;
    r.id = next_id_++;
    r.kind = LeaseKind::kCommit;
    r.subj = store::loading::ReplicaKey{
        .artifact_id = subj.artifact_id,
        .device = store::DeviceRegistry::instance().gpu_key(subj.device_id),
        .replica = 0,
    };
    r.pid = pid;
    // Guards: PID liveness
    GuardRec g;
    g.id = next_guard_id_++;
    g.kind = GuardKind::kPidLiveness;
    g.lease = r.id;
    g.generation = 1;
    g.pid = pid;
    guard_by_id_.emplace(g.id, g);
    r.guards.push_back(g.id);
    pid_index_[pid].insert(g.id);
    mon = monitor_;
    // Finalizer: remove device-unique commit index in LipManager for matching owner
    r.finalizers.emplace_back([this, r]() -> absl::Status {
      const bool revoked = lip_.revoke_commit_lease_if_owner_matches(r.subj.artifact_id, r.subj.device.ordinal, r.pid);
      if (revoked) {
        VLOG(2) << "Commit lease revoked for artifact=" << r.subj.artifact_id << " dev=" << r.subj.device.ordinal
                << " owner_pid=" << r.pid;
      }
      return absl::OkStatus();
    });
    created_id = r.id;
    by_id_[r.id] = std::move(r);
  }
  if (mon) {
    mon->watch(pid);
  }
  return created_id;
}

absl::StatusOr<SessionLifecycleManager::GuardId> SessionLifecycleManager::add_deadline_guard_for_test(
    LeaseId id,
    absl::Duration ttl) {
  if (ttl <= absl::ZeroDuration())
    return absl::InvalidArgumentError("ttl must be > 0");
  std::optional<absl::Time> notify_when;
  GuardId new_gid = 0;
  {
    absl::MutexLock lock(&mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end())
      return absl::NotFoundError("lease not found");
    GuardRec g;
    g.id = next_guard_id_++;
    g.kind = GuardKind::kDeadline;
    g.lease = id;
    g.generation = 1;
    g.deadline = absl::Now() + ttl;
    guard_by_id_.emplace(g.id, g);
    it->second.guards.push_back(g.id);
    notify_when = push_deadline_(g.id, g.deadline, g.generation);
    new_gid = g.id;
  }
  notify_schedule_if_earlier_(notify_when);
  return new_gid;
}

bool SessionLifecycleManager::has_pid_guard_for_test(pid_t pid) const {
  absl::MutexLock lock(&mu_);
  auto it = pid_index_.find(pid);
  return it != pid_index_.end() && !it->second.empty();
}

absl::Status SessionLifecycleManager::keepalive_session(std::string sid, absl::Duration ttl) {
  if (ttl <= absl::ZeroDuration())
    return absl::InvalidArgumentError("ttl must be > 0");
  std::optional<absl::Time> notify_when;
  {
    absl::MutexLock lock(&mu_);
    auto it = session_by_sid_.find(sid);
    if (it == session_by_sid_.end()) {
      SessionGuardRec sg;
      sg.id = next_guard_id_++;
      sg.sid = std::move(sid);
      sg.generation = 1;
      sg.deadline = absl::Now() + ttl;
      session_by_sid_[sg.sid] = sg.id;
      session_guard_by_id_[sg.id] = sg;
      notify_when = push_deadline_(sg.id, sg.deadline, sg.generation);
    } else {
      auto& sg = session_guard_by_id_[it->second];
      sg.generation++;
      sg.deadline = absl::Now() + ttl;
      notify_when = push_deadline_(sg.id, sg.deadline, sg.generation);
    }
  }
  notify_schedule_if_earlier_(notify_when);
  return absl::OkStatus();
}

absl::Status SessionLifecycleManager::renew_placement(LeaseId id, absl::Duration ttl) {
  std::optional<absl::Time> notify_when;
  {
    absl::MutexLock lock(&mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end())
      return absl::NotFoundError("lease not found");
    if (ttl <= absl::ZeroDuration())
      return absl::InvalidArgumentError("ttl must be > 0");
    // Find the deadline guard and renew
    for (GuardId gid : it->second.guards) {
      auto git = guard_by_id_.find(gid);
      if (git == guard_by_id_.end())
        continue;
      if (git->second.kind == GuardKind::kDeadline || git->second.kind == GuardKind::kHeartbeatTtl) {
        git->second.generation++;
        git->second.deadline = absl::Now() + ttl;
        auto n = push_deadline_(git->second.id, git->second.deadline, git->second.generation);
        if (n.has_value()) {
          if (!notify_when.has_value() || *n < *notify_when)
            notify_when = n;
        }
      }
    }
  }
  notify_schedule_if_earlier_(notify_when);
  return absl::OkStatus();
}

void SessionLifecycleManager::release_lease(LeaseId id) {
  retire_lease_(id, /*reason=*/"manual_release");
}

absl::Status SessionLifecycleManager::release_use_lease(const ReplicaSubject& subj, pid_t pid) {
  LeaseId match = 0;
  {
    absl::MutexLock lock(&mu_);
    for (const auto& kv : by_id_) {
      const auto& rec = kv.second;
      if (rec.kind != LeaseKind::kUse)
        continue;
      if (rec.pid != pid)
        continue;
      if (!(rec.subj == subj))
        continue;
      match = kv.first;
      break;
    }
  }
  if (match == 0) {
    return absl::NotFoundError("use lease not found");
  }
  retire_lease_(match, /*reason=*/"manual_release_use");
  return absl::OkStatus();
}

void SessionLifecycleManager::release_by_pid(pid_t pid) {
  // Retire all leases bound to this pid (Use/Commit kinds)
  std::vector<LeaseId> to_retire;
  {
    absl::MutexLock lock(&mu_);
    to_retire.reserve(by_id_.size());
    for (const auto& kv : by_id_) {
      if (kv.second.pid == pid)
        to_retire.push_back(kv.first);
    }
  }
  for (LeaseId id : to_retire)
    retire_lease_(id, /*reason=*/"pid_release");
}

void SessionLifecycleManager::release_session(const std::string& sid) {
  GuardId gid = 0;
  {
    absl::MutexLock lock(&mu_);
    auto it = session_by_sid_.find(sid);
    if (it == session_by_sid_.end())
      return;
    gid = it->second;
    session_by_sid_.erase(it);
    session_guard_by_id_.erase(gid);
  }
  {
    const size_t erased = sessions_.erase(sid);
    if (erased == 0) {
      VLOG(2) << "release_session: session not found for sid=" << sid;
    }
  }
}

absl::Time SessionLifecycleManager::next_deadline() const {
  absl::MutexLock lock(&mu_);
  return (deadlines_.empty() ? absl::InfiniteFuture() : deadlines_.top().when);
}

void SessionLifecycleManager::expire_due(absl::Time now) {
  std::vector<LeaseId> to_retire;
  std::vector<std::string> sessions_to_erase;
  absl::Time next_after = absl::InfiniteFuture();
  {
    absl::MutexLock lock(&mu_);
    while (!deadlines_.empty() && deadlines_.top().when <= now) {
      const auto d = deadlines_.top();
      deadlines_.pop();
      // Session guard?
      auto sit = session_guard_by_id_.find(d.guard);
      if (sit != session_guard_by_id_.end()) {
        if (sit->second.generation != d.generation)
          continue;
        sessions_to_erase.push_back(sit->second.sid);
        session_by_sid_.erase(sit->second.sid);
        session_guard_by_id_.erase(sit);
        continue;
      }
      // Lease guard
      auto git = guard_by_id_.find(d.guard);
      if (git == guard_by_id_.end())
        continue; // already removed
      if (git->second.generation != d.generation)
        continue; // stale entry
      // Mark guard failed and schedule lease retirement
      git->second.failed = true;
      to_retire.push_back(git->second.lease);
    }
    next_after = (deadlines_.empty() ? absl::InfiniteFuture() : deadlines_.top().when);
  }
  for (LeaseId id : to_retire) {
    retire_lease_(id, /*reason=*/"deadline_expired");
  }
  for (const auto& sid : sessions_to_erase) {
    const size_t erased = sessions_.erase(sid);
    if (erased == 0) {
      VLOG(2) << "expire_due: session not found for sid=" << sid;
    }
  }
  // Reschedule according to the new earliest deadline
  notify_schedule_if_earlier_(next_after);
}

void SessionLifecycleManager::handle_pid_exit(pid_t pid) {
  // Mark all pid guards failed and retire their leases
  std::unordered_set<GuardId> guards;
  {
    absl::MutexLock lock(&mu_);
    auto it = pid_index_.find(pid);
    if (it != pid_index_.end())
      guards = it->second;
    for (GuardId gid : guards) {
      auto git = guard_by_id_.find(gid);
      if (git != guard_by_id_.end()) {
        git->second.failed = true;
      }
    }
  }
  size_t lease_count = 0;
  {
    std::unordered_set<LeaseId> leases_to_retire;
    {
      absl::MutexLock lock(&mu_);
      for (GuardId gid : guards) {
        auto git = guard_by_id_.find(gid);
        if (git != guard_by_id_.end()) {
          leases_to_retire.insert(git->second.lease);
        }
      }
    }
    lease_count = leases_to_retire.size();
    for (LeaseId id : leases_to_retire) {
      retire_lease_(id, /*reason=*/"pid_exit");
    }
  }
  // Best-effort: drop RefTracker references for this pid
  auto keys = refs_.keys();
  VLOG(1) << "SessionLifecycle: pid_exit pid=" << pid << " guards=" << guards.size() << " leases=" << lease_count
          << " ref_keys=" << keys.size();
  for (const auto& key : keys) {
    refs_.drop_ref(key, pid);
  }
  // Cleanup LIP leases if owner_pid matches (delegated)
  lip_.sweep_expired_and_dead_pids();
}

void SessionLifecycleManager::attach_pid_monitor(PidMonitor* mon) {
  absl::MutexLock lock(&mu_);
  monitor_ = mon;
}

size_t SessionLifecycleManager::use_count_for(const store::loading::ReplicaKey& key) const {
  if (key.device.type != DeviceType::GPU) {
    return 0;
  }
  absl::MutexLock lock(&mu_);
  auto it = counters_.find(key);
  return (it == counters_.end() ? 0 : static_cast<size_t>(it->second.use_count));
}

size_t SessionLifecycleManager::placement_pin_count_for(const store::loading::ReplicaKey& key) const {
  if (key.device.type != DeviceType::GPU) {
    return 0;
  }
  absl::MutexLock lock(&mu_);
  auto it = counters_.find(key);
  return (it == counters_.end() ? 0 : static_cast<size_t>(it->second.placement_pins));
}

void SessionLifecycleManager::inc_use_(const ReplicaSubject& s) {
  auto& c = counters_[s];
  ++c.use_count;
}

void SessionLifecycleManager::dec_use_(const ReplicaSubject& s) {
  auto it = counters_.find(s);
  if (it != counters_.end() && it->second.use_count > 0)
    --it->second.use_count;
}

void SessionLifecycleManager::inc_pin_(const ReplicaSubject& s) {
  auto& c = counters_[s];
  ++c.placement_pins;
}

void SessionLifecycleManager::dec_pin_(const ReplicaSubject& s) {
  auto it = counters_.find(s);
  if (it != counters_.end() && it->second.placement_pins > 0)
    --it->second.placement_pins;
}

void SessionLifecycleManager::sync_pid_watches_() {
  PidMonitor* mon = nullptr;
  {
    absl::MutexLock lock(&mu_);
    mon = monitor_;
  }
  if (!mon)
    return;
  // RefTracker PIDs
  auto keys = refs_.keys();
  for (const auto& key : keys) {
    auto plist = refs_.pids(key);
    for (int32_t pid : plist) {
      mon->watch(static_cast<pid_t>(pid));
    }
  }
  // Note: LIP leases are handled by LipManager; we rely on RefTracker cover for now.
}

void SessionLifecycleManager::sweep_pid_liveness() {
  auto keys = refs_.keys();
  for (const auto& key : keys) {
    auto plist = refs_.pids(key);
    for (int32_t pid : plist) {
      std::string proc_path = absl::StrCat("/proc/", pid);
      if (::access(proc_path.c_str(), F_OK) != 0) {
        refs_.drop_ref(key, pid);
      }
    }
  }
  // Delegate LIP lease expiry and dead PID cleanup to LipManager
  lip_.sweep_expired_and_dead_pids();
}

std::optional<absl::Time> SessionLifecycleManager::push_deadline_(GuardId gid, absl::Time when, uint64_t gen) {
  const absl::Time before = (deadlines_.empty() ? absl::InfiniteFuture() : deadlines_.top().when);
  deadlines_.push(Due{.when = when, .guard = gid, .generation = gen});
  const absl::Time after = deadlines_.top().when;
  if (after < before)
    return after;
  return std::nullopt;
}

void SessionLifecycleManager::notify_schedule_if_earlier_(std::optional<absl::Time> when) {
  if (!when.has_value())
    return;
  std::function<void(absl::Time)> hook;
  {
    absl::MutexLock lock(&mu_);
    hook = schedule_hook_;
  }
  if (hook)
    hook(*when);
}

void SessionLifecycleManager::notify_schedule_if_earlier_(absl::Time when) {
  std::function<void(absl::Time)> hook;
  {
    absl::MutexLock lock(&mu_);
    hook = schedule_hook_;
  }
  if (hook)
    hook(when);
}

void SessionLifecycleManager::retire_lease_(LeaseId id, const char* /*reason*/) {
  std::vector<std::function<absl::Status()>> finalizers;
  std::vector<pid_t> pids_to_unwatch;
  std::optional<ReplicaSubject> subj_opt;
  std::optional<ReplicaSubject> subj_to_notify;
  {
    absl::MutexLock lock(&mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end())
      return;
    if (it->second.state == LeaseState::kRetired)
      return;
    // Transition to retiring → retired
    it->second.state = LeaseState::kRetiring;
    subj_opt = it->second.subj;
    // Move finalizers out; call them after unlock
    finalizers = std::move(it->second.finalizers);
    // Update counters once
    if (it->second.kind == LeaseKind::kUse)
      dec_use_(it->second.subj);
    if (it->second.kind == LeaseKind::kPlacement)
      dec_pin_(it->second.subj);
    // If this was the last protection for the subject, capture subject for eviction notify
    if (subj_opt.has_value()) {
      auto cit = counters_.find(*subj_opt);
      if (cit != counters_.end() && cit->second.use_count == 0 && cit->second.placement_pins == 0) {
        subj_to_notify = *subj_opt;
      }
    }
    // Cleanup guard indices
    for (GuardId gid : it->second.guards) {
      auto git = guard_by_id_.find(gid);
      if (git != guard_by_id_.end()) {
        if (git->second.kind == GuardKind::kPidLiveness && git->second.pid > 0) {
          auto pit = pid_index_.find(git->second.pid);
          if (pit != pid_index_.end()) {
            pit->second.erase(gid);
            if (pit->second.empty()) {
              pids_to_unwatch.push_back(git->second.pid);
              pid_index_.erase(pit);
            }
          }
        }
        guard_by_id_.erase(git);
      }
    }
    it->second.state = LeaseState::kRetired;
    by_id_.erase(it);
  }
  for (auto& f : finalizers) {
    absl::Status st = f();
    if (!st.ok()) {
      LOG(WARNING) << "SessionLifecycle finalizer failed: " << st;
    }
  }
  // Perform unwatch outside the lock
  PidMonitor* mon = nullptr;
  {
    absl::MutexLock lock(&mu_);
    mon = monitor_;
  }
  if (mon) {
    for (pid_t p : pids_to_unwatch)
      mon->unwatch(p);
  }
  if (subj_to_notify.has_value()) {
    std::function<void(const ReplicaSubject&)> cb;
    {
      absl::MutexLock lock(&mu_);
      cb = eviction_notify_;
    }
    if (cb)
      cb(*subj_to_notify);
  }
}

void SessionLifecycleManager::maybe_unload_daemon_replica_(const ReplicaSubject& subj) {
  // Engine may be absent in tests or minimal setups
  if (engine_ == nullptr)
    return;
  if (subj.device.type != DeviceType::GPU) {
    return;
  }
  // Snapshot counters under lock
  int uses = 0;
  int pins = 0;
  {
    absl::MutexLock lock(&mu_);
    auto it = counters_.find(subj);
    if (it != counters_.end()) {
      uses = it->second.use_count;
      pins = it->second.placement_pins;
    }
  }
  if (uses != 0 || pins != 0)
    return;
  // Check residency; only unload if GPU memory is present.
  auto state = engine_->get_replica_state(subj, DeviceType::GPU);
  if (state <= store::replica::MemoryState::UNALLOCATED) {
    return; // not allocated/loaded on GPU
  }
  int rc = engine_->unload_replica(subj);
  if (rc != 0) {
    LOG(WARNING) << "maybe_unload_daemon_replica: unload_replica rc=" << rc << " key=" << subj;
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto ctr = meter->CreateDoubleCounter("tc_unload_failed_total");
      ctr->Add(1.0);
    } catch (...) {
    }
  }
}

SessionLifecycleTask::SessionLifecycleTask(SessionLifecycleManager& mgr) : mgr_(mgr) {}

void SessionLifecycleTask::poll() {
  mgr_.sweep_once();
}

const char* SessionLifecycleTask::name() {
  return "SessionLifecycleTask";
}

} // namespace tensorcast::daemon
