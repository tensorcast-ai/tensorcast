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
#include <cerrno>
#include <csignal>
#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "daemon/lip_manager.h"
#include "daemon/ref_tracker.h"
#include "daemon/replica_session_manager.h"

namespace tensorcast::daemon {

// Lightweight PID liveness monitor.
// Uses pidfd + epoll when available; falls back to periodic /proc polling otherwise.
class PidMonitor final {
 public:
  using ExitCallback = std::function<void(pid_t)>;

  // poll_interval controls the fallback polling cadence when pidfd is
  // unavailable, and also the epoll wait timeout when pidfd is available.
  explicit PidMonitor(ExitCallback cb, std::chrono::milliseconds poll_interval = std::chrono::milliseconds(1000))
      : cb_(std::move(cb)), poll_interval_(poll_interval) {}

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
      ev.data.u64 = kEpollTagWake;
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
        ev.data.u64 = (kEpollTagPid | static_cast<uint64_t>(pid));
        (void)::epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, pfd, &ev);
      } else {
        // pidfd unavailable for this pid (maybe it already died); leave in watched_ for poll fallback
      }
    }
    // Wake the epoll loop so fallback polling notices newly watched PIDs promptly
    if (event_fd_ >= 0) {
      uint64_t one = 1;
      (void)::write(event_fd_, &one, sizeof(one));
    }
  }

  // Stop watching a PID. Best-effort: removes any pidfd from epoll and closes it.
  void unwatch(pid_t pid) {
    if (pid <= 0)
      return;
    int pfd = -1;
    {
      std::lock_guard<std::mutex> g(mu_);
      watched_.erase(pid);
      auto it = pidfds_.find(pid);
      if (it != pidfds_.end()) {
        pfd = it->second;
        pidfds_.erase(it);
      }
    }
    if (use_pidfd_.load() && epoll_fd_ >= 0 && pfd >= 0) {
      (void)::epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, pfd, nullptr);
      ::close(pfd);
    }
    // Wake the loop to apply changes promptly
    if (event_fd_ >= 0) {
      uint64_t one = 1;
      (void)::write(event_fd_, &one, sizeof(one));
    }
  }

  bool using_pidfd() const {
    return use_pidfd_.load();
  }

#if defined(TC_ENABLE_TEST_HOOKS)
  bool is_watching_for_test(pid_t pid) {
    std::lock_guard<std::mutex> g(mu_);
    return watched_.find(pid) != watched_.end();
  }
#endif

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
      int timeout_ms = static_cast<int>(poll_interval_.count());
      if (timeout_ms <= 0)
        timeout_ms = 1;
      int n = ::epoll_wait(epoll_fd_, events.data(), kMaxEvents, timeout_ms);
      if (n < 0)
        continue;
      for (int i = 0; i < n; ++i) {
        const auto& ev = events[static_cast<size_t>(i)];
        const uint64_t data = ev.data.u64;
        if ((data & kEpollTagMask) == kEpollTagWake) {
          // drain eventfd
          uint64_t tmp;
          (void)::read(event_fd_, &tmp, sizeof(tmp));
          continue;
        }
        if ((data & kEpollTagMask) != kEpollTagPid) {
          // Unknown tag; ignore
          continue;
        }
        pid_t pid = static_cast<pid_t>(data & kEpollValueMask);
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
      // Poll fallback for PIDs without pidfd
      poll_missing_pidfds_once_();
    }
  }

  void run_poll_() {
    while (running_.load()) {
      std::vector<pid_t> to_drop;
      {
        std::lock_guard<std::mutex> g(mu_);
        to_drop.reserve(watched_.size());
        for (pid_t pid : watched_) {
          if (::kill(pid, 0) == -1 && errno == ESRCH) {
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
      // Sleep for configured interval
      std::this_thread::sleep_for(poll_interval_);
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
  std::chrono::milliseconds poll_interval_;
  // Epoll data tagging to disambiguate sources
  static constexpr uint64_t kEpollTagMask = 0xFFFF000000000000ULL;
  static constexpr uint64_t kEpollTagWake = 0xFFFF000000000000ULL;
  static constexpr uint64_t kEpollTagPid = 0xEEEE000000000000ULL;
  static constexpr uint64_t kEpollValueMask = 0x0000FFFFFFFFFFFFULL;

  void poll_missing_pidfds_once_() {
    if (!running_.load())
      return;
    std::vector<pid_t> to_drop;
    {
      std::lock_guard<std::mutex> g(mu_);
      to_drop.reserve(watched_.size());
      for (pid_t pid : watched_) {
        if (pidfds_.find(pid) != pidfds_.end())
          continue; // handled by epoll via pidfd
        if (::kill(pid, 0) == -1 && errno == ESRCH) {
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
  }
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

  SessionLifecycleManager(
      ReplicaSessionManager& sessions,
      RefTracker& refs,
      LipManager& lip,
      store::StoreEngine& engine)
      : sessions_(sessions), refs_(refs), lip_(lip), engine_(&engine) {}

  // Backward-compatible ctor for tests or when immediate reclaim is not desired.
  SessionLifecycleManager(ReplicaSessionManager& sessions, RefTracker& refs, LipManager& lip)
      : sessions_(sessions), refs_(refs), lip_(lip), engine_(nullptr) {}

  // One-pass sweep that combines the prior SessionTtlTask, RegJoinTtlTask,
  // and PidWatchTask behaviors into a single unified pass.
  void sweep_once() {
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

  // Optional: register a callback invoked when a subject becomes fully
  // unprotected (use_count==0 && placement_pins==0). Used to notify eviction
  // logic for faster reclaim. The subject is provided to the callback.
  void set_eviction_notify(std::function<void(const ReplicaSubject&)> fn) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    eviction_notify_ = std::move(fn);
  }

  // Optional: register a scheduling hook to tighten deadline-driven wake-ups.
  // The hook is invoked whenever the earliest time-based guard deadline moves
  // earlier (or after pops) so the BackgroundScheduler can reschedule the
  // lifecycle task to run sooner.
  void set_schedule_hook(std::function<void(absl::Time)> fn) ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    schedule_hook_ = std::move(fn);
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
      // Finalizer: drop ref for this pid+subject, then attempt immediate reclaim if eligible
      r.finalizers.emplace_back([this, subj, pid]() -> absl::Status {
        store::DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = subj.device_id, .uuid = ""};
        store::loading::ReplicaKey key{.artifact_id = subj.artifact_id, .device = dev_key, .replica = 0};
        refs_.drop_ref(key, pid);
        // Attempt immediate reclaim if no active uses or pins remain and a daemon-owned replica exists
        maybe_unload_daemon_replica_(subj);
        return absl::OkStatus();
      });
      created_id = r.id;
      by_id_[r.id] = std::move(r);
      inc_use_(subj);
    }
    if (mon) {
      mon->watch(pid);
    }
    return created_id;
  }

  // Create a Use lease bound to pid with an additional Deadline guard (TTL-based join).
  absl::StatusOr<LeaseId> create_ttl_use_lease(const ReplicaSubject& subj, pid_t pid, absl::Duration ttl)
      ABSL_LOCKS_EXCLUDED(mu_) {
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
        store::DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = subj.device_id, .uuid = ""};
        store::loading::ReplicaKey key{.artifact_id = subj.artifact_id, .device = dev_key, .replica = 0};
        refs_.drop_ref(key, pid);
        maybe_unload_daemon_replica_(subj);
        return absl::OkStatus();
      });
      created_id = r.id;
      by_id_[r.id] = std::move(r);
      inc_use_(subj);
    }
    notify_schedule_if_earlier_(notify_when);
    if (mon)
      mon->watch(pid);
    return created_id;
  }

  absl::StatusOr<LeaseId> create_placement_lease(const ReplicaSubject& subj, /*spec*/ absl::Duration ttl)
      ABSL_LOCKS_EXCLUDED(mu_) {
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
      // Finalizer: remove device-unique commit index in LipManager for matching owner
      r.finalizers.emplace_back([this, r]() -> absl::Status {
        (void)lip_.revoke_commit_lease_if_owner_matches(r.subj.artifact_id, r.subj.device_id, r.pid);
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

#if defined(TC_ENABLE_TEST_HOOKS)
  // Test-only: attach an additional deadline guard to an existing lease to simulate
  // mixed-guard scenarios (AND semantics). Returns the new GuardId.
  absl::StatusOr<GuardId> add_deadline_guard_for_test(LeaseId id, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_) {
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

  // Test-only: check if a PID still has any liveness guards tracked by the manager.
  bool has_pid_guard_for_test(pid_t pid) const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    auto it = pid_index_.find(pid);
    return it != pid_index_.end() && !it->second.empty();
  }
#endif

  // Session principal keepalive: create or renew a deadline for the given session id.
  absl::Status keepalive_session(std::string sid, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_) {
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

  absl::Status renew_placement(LeaseId id, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_) {
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

  void release_session(const std::string& sid) ABSL_LOCKS_EXCLUDED(mu_) {
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
    (void)sessions_.erase(sid);
  }

  [[nodiscard]] absl::Time next_deadline() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    return (deadlines_.empty() ? absl::InfiniteFuture() : deadlines_.top().when);
  }

  void expire_due(absl::Time now) ABSL_LOCKS_EXCLUDED(mu_) {
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
      (void)sessions_.erase(sid);
    }
    // Reschedule according to the new earliest deadline
    notify_schedule_if_earlier_(next_after);
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

  // sweep_sessions_ttl and sweep_reg_join_ttl removed (legacy code path replaced by lifecycle guards)

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
  store::StoreEngine* engine_;

  // ==== Minimal ExpirationQueue (min-heap by deadline) ====
  struct Due {
    absl::Time when;
    GuardId guard;
    uint64_t generation;

    bool operator<(const Due& other) const {
      return when > other.when;
    } // min-heap
  };

  // Push a new deadline and return the earliest deadline after insertion when
  // it moves earlier (caller should notify scheduler outside the lock).
  std::optional<absl::Time> push_deadline_(GuardId gid, absl::Time when, uint64_t gen)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const absl::Time before = (deadlines_.empty() ? absl::InfiniteFuture() : deadlines_.top().when);
    deadlines_.push(Due{.when = when, .guard = gid, .generation = gen});
    const absl::Time after = deadlines_.top().when;
    if (after < before)
      return after;
    return std::nullopt;
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
  std::function<void(const ReplicaSubject&)> eviction_notify_ ABSL_GUARDED_BY(mu_);
  std::function<void(absl::Time)> schedule_hook_ ABSL_GUARDED_BY(mu_);

  // Session principal guards (deadline-driven)
  struct SessionGuardRec {
    GuardId id{0};
    std::string sid;
    uint64_t generation{0};
    absl::Time deadline{absl::InfinitePast()};
  };

  absl::flat_hash_map<GuardId, SessionGuardRec> session_guard_by_id_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, GuardId> session_by_sid_ ABSL_GUARDED_BY(mu_);

  // Notify scheduler if a new earliest deadline was inserted. Safe to call
  // without holding the manager mutex.
  void notify_schedule_if_earlier_(std::optional<absl::Time> when) ABSL_LOCKS_EXCLUDED(mu_) {
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

  // Notify scheduler with the given earliest deadline value (used after pops).
  void notify_schedule_if_earlier_(absl::Time when) ABSL_LOCKS_EXCLUDED(mu_) {
    std::function<void(absl::Time)> hook;
    {
      absl::MutexLock lock(&mu_);
      hook = schedule_hook_;
    }
    if (hook)
      hook(when);
  }

  void retire_lease_(LeaseId id, const char* /*reason*/) ABSL_LOCKS_EXCLUDED(mu_) {
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
        auto cit = counters_.find(subject_key_(*subj_opt));
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
      (void)f();
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

  // Attempt to unload a daemon-owned GPU replica for the given subject if both
  // the use_count and placement_pins are zero at the time of the check. This is
  // best-effort and runs off-lock. It only targets daemon-owned memory: we
  // consult the engine state and skip when no GPU residency is present.
  void maybe_unload_daemon_replica_(const ReplicaSubject& subj) {
    // Engine may be absent in tests or minimal setups
    if (engine_ == nullptr)
      return;
    // Snapshot counters under lock
    int uses = 0;
    int pins = 0;
    {
      absl::MutexLock lock(&mu_);
      auto it = counters_.find(subject_key_(subj));
      if (it != counters_.end()) {
        uses = it->second.use_count;
        pins = it->second.placement_pins;
      }
    }
    if (uses != 0 || pins != 0)
      return;
    // Build key and check residency; only unload if GPU memory is present
    store::DeviceKey dev_key{.type = DeviceType::GPU, .ordinal = subj.device_id, .uuid = ""};
    store::loading::ReplicaKey key{.artifact_id = subj.artifact_id, .device = dev_key, .replica = 0};
    auto state = engine_->get_replica_state(key, DeviceType::GPU);
    if (state <= store::replica::MemoryState::UNALLOCATED) {
      return; // not allocated/loaded on GPU
    }
    (void)engine_->unload_replica(key);
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
