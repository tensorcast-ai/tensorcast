// Copyright (c) 2025, TensorCast Team.

// Unified session lifecycle: consolidates session TTL, registration join TTL,
// and PID-based liveness into a single manager and background task.

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <array>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "daemon/lip_manager.h"
#include "daemon/ref_tracker.h"
#include "daemon/registration_manager.h"
#include "daemon/replica_session_manager.h"

namespace tensorcast::daemon {

// Lightweight PID liveness monitor.
// Uses pidfd + epoll when available; falls back to periodic /proc polling otherwise.
class PidMonitor final {
 public:
  using ExitCallback = std::function<void(pid_t)>;

  explicit PidMonitor(ExitCallback cb) : cb_(std::move(cb)) {}
  ~PidMonitor() {
    stop();
  }

  void start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true))
      return;
    // Try to create epoll and eventfd; if either fails, use polling fallback
    epoll_fd_ = ::epoll_create1(EPOLL_CLOEXEC);
    event_fd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (epoll_fd_ >= 0 && event_fd_ >= 0) {
      epoll_event ev{};
      ev.events = EPOLLIN;
      ev.data.fd = event_fd_;
      (void)::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, event_fd_, &ev);
      use_pidfd_.store(true);
    } else {
      // cleanup partially created
      if (epoll_fd_ >= 0) {
        ::close(epoll_fd_);
        epoll_fd_ = -1;
      }
      if (event_fd_ >= 0) {
        ::close(event_fd_);
        event_fd_ = -1;
      }
      use_pidfd_.store(false);
    }
    th_ = std::thread([this]() { this->run_loop_(); });
  }

  void stop() {
    if (!running_.load())
      return;
    running_.store(false);
    // Wake thread
    if (event_fd_ >= 0) {
      uint64_t one = 1;
      (void)::write(event_fd_, &one, sizeof(one));
    }
    if (th_.joinable())
      th_.join();
    // Close fds
    if (epoll_fd_ >= 0) {
      ::close(epoll_fd_);
      epoll_fd_ = -1;
    }
    if (event_fd_ >= 0) {
      ::close(event_fd_);
      event_fd_ = -1;
    }
    // Close all pidfds
    {
      std::lock_guard<std::mutex> g(mu_);
      for (auto& kv : pidfds_) {
        if (kv.second >= 0)
          ::close(kv.second);
      }
      pidfds_.clear();
      watched_.clear();
    }
  }

  // Start watching a PID; safe to call repeatedly.
  void watch(pid_t pid) {
    if (pid <= 0)
      return;
    std::lock_guard<std::mutex> g(mu_);
    if (!watched_.insert(pid).second)
      return; // already watching
    if (use_pidfd_.load()) {
      int pfd = static_cast<int>(::syscall(SYS_pidfd_open, pid, 0));
      if (pfd >= 0) {
        pidfds_[pid] = pfd;
        epoll_event ev{};
        ev.events = EPOLLIN | EPOLLRDHUP | EPOLLHUP;
        ev.data.u32 = static_cast<uint32_t>(pid);
        (void)::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, pfd, &ev);
      } else {
        // pidfd unavailable for this pid (maybe it already died); leave in watched_ for poll fallback
      }
    }
  }

  bool using_pidfd() const {
    return use_pidfd_.load();
  }

 private:
  void run_loop_() {
    if (use_pidfd_.load()) {
      run_epoll_();
    } else {
      run_poll_();
    }
  }
  void run_epoll_() {
    constexpr int kMaxEvents = 32;
    std::array<epoll_event, kMaxEvents> events{};
    while (running_.load()) {
      int n = ::epoll_wait(epoll_fd_, events.data(), kMaxEvents, 1000 /*ms*/);
      if (n < 0)
        continue;
      for (int i = 0; i < n; ++i) {
        const auto& ev = events[static_cast<size_t>(i)];
        if (ev.data.fd == event_fd_) {
          // drain eventfd
          uint64_t tmp;
          (void)::read(event_fd_, &tmp, sizeof(tmp));
          continue;
        }
        pid_t pid = static_cast<pid_t>(ev.data.u32);
        // Cleanup pidfd and mapping
        int pfd = -1;
        {
          std::lock_guard<std::mutex> g(mu_);
          auto it = pidfds_.find(pid);
          if (it != pidfds_.end()) {
            pfd = it->second;
            pidfds_.erase(it);
          }
          watched_.erase(pid);
        }
        if (pfd >= 0)
          ::close(pfd);
        // Notify
        if (cb_)
          cb_(pid);
      }
    }
  }
  void run_poll_() {
    while (running_.load()) {
      std::vector<pid_t> to_drop;
      {
        std::lock_guard<std::mutex> g(mu_);
        to_drop.reserve(watched_.size());
        for (pid_t pid : watched_) {
          std::string proc_path = absl::StrCat("/proc/", pid);
          if (::access(proc_path.c_str(), F_OK) != 0) {
            to_drop.push_back(pid);
          }
        }
        for (pid_t pid : to_drop) {
          watched_.erase(pid);
        }
      }
      for (pid_t pid : to_drop) {
        if (cb_)
          cb_(pid);
      }
      // Sleep ~1s
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  }

  ExitCallback cb_;
  std::atomic<bool> running_{false};
  std::atomic<bool> use_pidfd_{false};
  int epoll_fd_{-1};
  int event_fd_{-1};
  std::thread th_;
  std::mutex mu_;
  std::unordered_set<pid_t> watched_;
  std::unordered_map<pid_t, int> pidfds_;
};

class SessionLifecycleManager {
 public:
  // Public API skeleton (to be adopted by controllers)
  using LeaseId = uint64_t;
  using GuardId = uint64_t;
  struct ReplicaSubject {
    std::string artifact_id;
    int device_id{-1};
  };
  struct CommitSubject {
    std::string artifact_id;
    int device_id{-1};
  };

  enum class LeaseKind : uint8_t { kPlacement, kUse, kCommit };
  enum class GuardKind : uint8_t { kHeartbeatTtl, kDeadline, kPidLiveness, kManual };
  enum class LeaseState : uint8_t { kActive, kRetiring, kRetired };

  struct GuardRec {
    GuardId id{0};
    GuardKind kind{GuardKind::kManual};
    LeaseId lease{0};
    // Generation to prevent ABA: renewing a guard bumps generation; stale heap entries are ignored
    uint64_t generation{0};
    // Only for time-based guards
    absl::Time deadline{absl::InfinitePast()};
    // Only for pid guard
    int pid{-1};
    bool failed{false};
  };

  struct LeaseRec {
    LeaseId id{0};
    LeaseKind kind{LeaseKind::kUse};
    ReplicaSubject subj;
    int pid{-1};
    // Guard set associated with this lease
    std::vector<GuardId> guards;
    LeaseState state{LeaseState::kActive};
    // Finalizers: idempotent cleanup (called once on transition to kRetired)
    std::vector<std::function<absl::Status()>> finalizers;
  };

  SessionLifecycleManager(ReplicaSessionManager& sessions, RefTracker& refs, LipManager& lip, RegistrationManager& reg)
      : sessions_(sessions), refs_(refs), lip_(lip), reg_(reg) {}

  // One-pass sweep that combines the prior SessionTtlTask, RegJoinTtlTask,
  // and PidWatchTask behaviors into a single unified pass.
  void sweep_once() {
    // Ensure we're watching all currently-referenced PIDs for exit notifications
    sync_pid_watches_();
    sweep_sessions_ttl();
    sweep_reg_join_ttl();
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

  // API shape (skeleton)
  absl::StatusOr<LeaseId> create_use_lease(const ReplicaSubject& subj, pid_t pid) ABSL_LOCKS_EXCLUDED(mu_) {
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
      // Finalizer: drop ref for this pid+subject
      r.finalizers.emplace_back([this, subj, pid]() -> absl::Status {
        store::DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = subj.device_id, .uuid = ""};
        store::loading::ReplicaKey key{.artifact_id = subj.artifact_id, .device = dev_key, .replica = 0};
        refs_.drop_ref(key, pid);
        return absl::OkStatus();
      });
      by_id_[r.id] = std::move(r);
      inc_use_(subj);
      created_id = by_id_[next_id_ - 1].id;
    }
    if (mon) {
      mon->watch(pid);
    }
    return created_id;
  }
  absl::StatusOr<LeaseId> create_placement_lease(const ReplicaSubject& subj, /*spec*/ absl::Duration ttl)
      ABSL_LOCKS_EXCLUDED(mu_) {
    LeaseId created_id = 0;
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
        push_deadline_(g.id, g.deadline, g.generation);
      } else {
        GuardRec g;
        g.id = next_guard_id_++;
        g.kind = GuardKind::kManual;
        g.lease = r.id;
        g.generation = 1;
        guard_by_id_.emplace(g.id, g);
        r.guards.push_back(g.id);
      }
      // Finalizer: drop placement pin counters only; actual memory reclaim is handled elsewhere
      r.finalizers.emplace_back([subj]() -> absl::Status {
        (void)subj;
        return absl::OkStatus();
      });
      by_id_[r.id] = std::move(r);
      inc_pin_(subj);
      created_id = by_id_[next_id_ - 1].id;
    }
    return created_id;
  }
  absl::StatusOr<LeaseId> create_commit_lease(const CommitSubject& subj, pid_t pid) ABSL_LOCKS_EXCLUDED(mu_) {
    PidMonitor* mon = nullptr;
    LeaseId created_id = 0;
    {
      absl::MutexLock lock(&mu_);
      LeaseRec r;
      r.id = next_id_++;
      r.kind = LeaseKind::kCommit;
      r.subj = ReplicaSubject{.artifact_id = subj.artifact_id, .device_id = subj.device_id};
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
      // Finalizer: remove commit metadata is handled by LipManager sweep; no explicit engine work here.
      r.finalizers.emplace_back([]() -> absl::Status { return absl::OkStatus(); });
      by_id_[r.id] = std::move(r);
      created_id = by_id_[next_id_ - 1].id;
    }
    if (mon) {
      mon->watch(pid);
    }
    return created_id;
  }

#if defined(TC_ENABLE_TEST_HOOKS)
  // Test-only: attach an additional deadline guard to an existing lease to simulate
  // mixed-guard scenarios (AND semantics). Returns the new GuardId.
  absl::StatusOr<GuardId> add_deadline_guard_for_test(LeaseId id, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_) {
    if (ttl <= absl::ZeroDuration())
      return absl::InvalidArgumentError("ttl must be > 0");
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
    push_deadline_(g.id, g.deadline, g.generation);
    return g.id;
  }
#endif

  static absl::Status keepalive_session(/*sid*/ const std::string& /*sid*/, absl::Duration /*ttl*/) {
    return absl::OkStatus();
  }

  absl::Status renew_placement(LeaseId id, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_) {
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
        push_deadline_(git->second.id, git->second.deadline, git->second.generation);
      }
    }
    return absl::OkStatus();
  }

  void release_lease(LeaseId id) ABSL_LOCKS_EXCLUDED(mu_) {
    retire_lease_(id, /*reason=*/"manual_release");
  }

  // Retire a single Use lease bound to the given subject and pid.
  // If multiple Use leases exist for the same (subject, pid), retires one of them.
  // Returns NotFound when no matching Use lease exists.
  absl::Status release_use_lease(const ReplicaSubject& subj, pid_t pid) ABSL_LOCKS_EXCLUDED(mu_) {
    LeaseId match = 0;
    {
      absl::MutexLock lock(&mu_);
      for (const auto& kv : by_id_) {
        const auto& rec = kv.second;
        if (rec.kind != LeaseKind::kUse)
          continue;
        if (rec.pid != pid)
          continue;
        if (rec.subj.artifact_id != subj.artifact_id || rec.subj.device_id != subj.device_id)
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

  void release_by_pid(pid_t pid) ABSL_LOCKS_EXCLUDED(mu_) {
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

  void release_session(/*sid*/ const std::string& /*sid*/) {}

  [[nodiscard]] absl::Time next_deadline() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    return (deadlines_.empty() ? absl::InfiniteFuture() : deadlines_.top().when);
  }

  void expire_due(absl::Time now) ABSL_LOCKS_EXCLUDED(mu_) {
    std::vector<LeaseId> to_retire;
    {
      absl::MutexLock lock(&mu_);
      while (!deadlines_.empty() && deadlines_.top().when <= now) {
        const auto d = deadlines_.top();
        deadlines_.pop();
        auto git = guard_by_id_.find(d.guard);
        if (git == guard_by_id_.end())
          continue; // already removed
        if (git->second.generation != d.generation)
          continue; // stale entry
        // Mark guard failed and schedule lease retirement
        git->second.failed = true;
        to_retire.push_back(git->second.lease);
      }
    }
    for (LeaseId id : to_retire) {
      retire_lease_(id, /*reason=*/"deadline_expired");
    }
  }

  void handle_pid_exit(pid_t pid) ABSL_LOCKS_EXCLUDED(mu_) {
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
      for (LeaseId id : leases_to_retire) {
        retire_lease_(id, /*reason=*/"pid_exit");
      }
    }
    // Best-effort: drop RefTracker references for this pid
    auto keys = refs_.keys();
    for (const auto& key : keys) {
      refs_.drop_ref(key, pid);
    }
    // Cleanup LIP leases if owner_pid matches (delegated)
    lip_.sweep_expired_and_dead_pids();
  }

  // Attach a PID monitor (owned by the caller) for event-driven PID exit handling
  void attach_pid_monitor(PidMonitor* mon) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    monitor_ = mon;
  }

  // Query counters for eviction and status
  [[nodiscard]] size_t use_count_for(const store::loading::ReplicaKey& key) const ABSL_LOCKS_EXCLUDED(mu_) {
    auto subj = subj_from_key_(key);
    if (!subj)
      return 0;
    absl::MutexLock lock(&mu_);
    auto it = counters_.find(subject_key_(*subj));
    return (it == counters_.end() ? 0 : static_cast<size_t>(it->second.use_count));
  }
  [[nodiscard]] size_t placement_pin_count_for(const store::loading::ReplicaKey& key) const ABSL_LOCKS_EXCLUDED(mu_) {
    auto subj = subj_from_key_(key);
    if (!subj)
      return 0;
    absl::MutexLock lock(&mu_);
    auto it = counters_.find(subject_key_(*subj));
    return (it == counters_.end() ? 0 : static_cast<size_t>(it->second.placement_pins));
  }

 private:
  struct Counts {
    int use_count{0};
    int placement_pins{0};
  };

  static std::string subject_key_(const ReplicaSubject& s) {
    return absl::StrCat(s.artifact_id, "#", s.device_id);
  }

  void inc_use_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto& c = counters_[subject_key_(s)];
    ++c.use_count;
  }

  void dec_use_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = counters_.find(subject_key_(s));
    if (it != counters_.end() && it->second.use_count > 0)
      --it->second.use_count;
  }

  void inc_pin_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto& c = counters_[subject_key_(s)];
    ++c.placement_pins;
  }

  void dec_pin_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = counters_.find(subject_key_(s));
    if (it != counters_.end() && it->second.placement_pins > 0)
      --it->second.placement_pins;
  }

  static std::optional<ReplicaSubject> subj_from_key_(const store::loading::ReplicaKey& key) {
    if (key.device.type != DeviceType::GPU)
      return std::nullopt;
    return ReplicaSubject{.artifact_id = key.artifact_id, .device_id = key.device.ordinal};
  }

  void sync_pid_watches_() ABSL_LOCKS_EXCLUDED(mu_) {
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

  void sweep_sessions_ttl() {
    for (const auto& k : sessions_.keys()) {
      (void)sessions_.remove_if_expired(k);
    }
  }

  void sweep_reg_join_ttl() {
    auto ids = reg_.keys();
    for (const auto& id : ids) {
      auto meta_opt = reg_.get_meta(id);
      if (!meta_opt.has_value())
        continue;
      const auto& m = *meta_opt;
      if (!m.joined_existing)
        continue;
      if (reg_.expire_if_ttl_elapsed(id)) {
        store::DeviceKey dev_key{
            .type = (m.plan == RegistrationManager::RegPlan::DVMP ? DeviceType::CPU : DeviceType::GPU),
            .ordinal = (m.plan == RegistrationManager::RegPlan::DVMP ? -1 : m.device_id),
            .uuid = ""};
        store::loading::ReplicaKey key{.artifact_id = m.artifact_id_mi2, .device = dev_key, .replica = 0};
        refs_.drop_ref(key, m.owner_pid);
      }
    }
  }

  void sweep_pid_liveness() {
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

  ReplicaSessionManager& sessions_;
  RefTracker& refs_;
  LipManager& lip_;
  RegistrationManager& reg_;

  // ==== Minimal ExpirationQueue (min-heap by deadline) ====
  struct Due {
    absl::Time when;
    GuardId guard;
    uint64_t generation;
    bool operator<(const Due& other) const {
      return when > other.when;
    } // min-heap
  };
  void push_deadline_(GuardId gid, absl::Time when, uint64_t gen) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    deadlines_.push(Due{.when = when, .guard = gid, .generation = gen});
  }

  // Mutex guarding state below
  mutable absl::Mutex mu_;

  LeaseId next_id_ ABSL_GUARDED_BY(mu_){1};
  GuardId next_guard_id_ ABSL_GUARDED_BY(mu_){1};
  absl::flat_hash_map<LeaseId, LeaseRec> by_id_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<GuardId, GuardRec> guard_by_id_ ABSL_GUARDED_BY(mu_);
  std::priority_queue<Due> deadlines_ ABSL_GUARDED_BY(mu_);
  PidMonitor* monitor_ ABSL_GUARDED_BY(mu_){nullptr};
  absl::flat_hash_map<std::string, Counts> counters_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<pid_t, std::unordered_set<GuardId>> pid_index_ ABSL_GUARDED_BY(mu_);

  void retire_lease_(LeaseId id, const char* /*reason*/) ABSL_LOCKS_EXCLUDED(mu_) {
    std::vector<std::function<absl::Status()>> finalizers;
    {
      absl::MutexLock lock(&mu_);
      auto it = by_id_.find(id);
      if (it == by_id_.end())
        return;
      if (it->second.state == LeaseState::kRetired)
        return;
      // Transition to retiring → retired
      it->second.state = LeaseState::kRetiring;
      // Move finalizers out; call them after unlock
      finalizers = std::move(it->second.finalizers);
      // Update counters once
      if (it->second.kind == LeaseKind::kUse)
        dec_use_(it->second.subj);
      if (it->second.kind == LeaseKind::kPlacement)
        dec_pin_(it->second.subj);
      // Cleanup guard indices
      for (GuardId gid : it->second.guards) {
        auto git = guard_by_id_.find(gid);
        if (git != guard_by_id_.end()) {
          if (git->second.kind == GuardKind::kPidLiveness && git->second.pid > 0) {
            auto pit = pid_index_.find(git->second.pid);
            if (pit != pid_index_.end()) {
              pit->second.erase(gid);
              if (pit->second.empty())
                pid_index_.erase(pit);
            }
          }
          guard_by_id_.erase(git);
        }
      }
      it->second.state = LeaseState::kRetired;
      by_id_.erase(it);
    }
    for (auto& f : finalizers) {
      (void)f();
    }
  }
};

class SessionLifecycleTask final {
 public:
  explicit SessionLifecycleTask(SessionLifecycleManager& mgr) : mgr_(mgr) {}
  void poll() {
    mgr_.sweep_once();
  }
  [[nodiscard]] static const char* name() {
    return "SessionLifecycleTask";
  }

 private:
  SessionLifecycleManager& mgr_;
};

} // namespace tensorcast::daemon
