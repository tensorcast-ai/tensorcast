// Copyright (c) 2025-2026, TensorCast Team.

// Unified session lifecycle: consolidates session TTL, registration join TTL,
// and PID-based liveness into a single manager and background task.

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sys/types.h>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/store/materialization/contracts/loading_spec.h"

namespace tensorcast::store {
class StoreEngine;
} // namespace tensorcast::store

namespace tensorcast::daemon {

class LipManager;
class PidMonitor;
class RefTracker;
class ReplicaSessionManager;

class SessionLifecycleManager {
 public:
  // Public API skeleton (to be adopted by controllers)
  using LeaseId = uint64_t;
  using GuardId = uint64_t;

  using ReplicaSubject = store::loading::ReplicaKey;

  struct CommitSubject {
    std::string artifact_id;
    int device_id{-1};
  };

  enum class LeaseKind : uint8_t { kPlacement, kUse, kCommit, kRetention };
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
      store::StoreEngine& engine);

  // Backward-compatible ctor for tests or when immediate reclaim is not desired.
  SessionLifecycleManager(ReplicaSessionManager& sessions, RefTracker& refs, LipManager& lip);

  // One-pass sweep that combines the prior SessionTtlTask, RegJoinTtlTask,
  // and PidWatchTask behaviors into a single unified pass.
  void sweep_once();

  // Optional: register a callback invoked when a subject becomes fully
  // unprotected (use_count==0 && placement_pins==0). Used to notify eviction
  // logic for faster reclaim. The subject is provided to the callback.
  void set_eviction_notify(std::function<void(const ReplicaSubject&)> fn) ABSL_LOCKS_EXCLUDED(mu_);

  // Optional: register a scheduling hook to tighten deadline-driven wake-ups.
  // The hook is invoked whenever the earliest time-based guard deadline moves
  // earlier (or after pops) so the BackgroundScheduler can reschedule the
  // lifecycle task to run sooner.
  void set_schedule_hook(std::function<void(absl::Time)> fn) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::StatusOr<LeaseId> create_use_lease(
      const ReplicaSubject& subj,
      pid_t pid,
      std::vector<std::function<absl::Status()>> extra_finalizers = {}) ABSL_LOCKS_EXCLUDED(mu_);

  // Create a Use lease bound to pid with an additional Deadline guard (TTL-based join).
  [[nodiscard]] absl::StatusOr<LeaseId> create_ttl_use_lease(
      const ReplicaSubject& subj,
      pid_t pid,
      absl::Duration ttl,
      std::vector<std::function<absl::Status()>> extra_finalizers = {}) ABSL_LOCKS_EXCLUDED(mu_);

  absl::StatusOr<LeaseId> create_placement_lease(const ReplicaSubject& subj, /*spec*/ absl::Duration ttl)
      ABSL_LOCKS_EXCLUDED(mu_);

  absl::StatusOr<LeaseId> create_placement_lease(
      const ReplicaSubject& subj,
      /*spec*/ absl::Duration ttl,
      std::vector<std::function<absl::Status()>> extra_finalizers) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::StatusOr<LeaseId> create_retention_lease(
      absl::Duration ttl,
      std::vector<std::function<absl::Status()>> finalizers) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::StatusOr<LeaseId> create_commit_lease(const CommitSubject& subj, pid_t pid)
      ABSL_LOCKS_EXCLUDED(mu_);

  // Test-only: attach an additional deadline guard to an existing lease to simulate
  // mixed-guard scenarios (AND semantics). Returns the new GuardId.
  absl::StatusOr<GuardId> add_deadline_guard_for_test(LeaseId id, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_);

  // Test-only: check if a PID still has any liveness guards tracked by the manager.
  bool has_pid_guard_for_test(pid_t pid) const ABSL_LOCKS_EXCLUDED(mu_);

  // Session principal keepalive: create or renew a deadline for the given session id.
  [[nodiscard]] absl::Status keepalive_session(std::string sid, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::Status renew_placement(LeaseId id, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::Status renew_retention(LeaseId id, absl::Duration ttl) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::Status add_finalizer(LeaseId id, std::function<absl::Status()> finalizer)
      ABSL_LOCKS_EXCLUDED(mu_);

  void release_lease(LeaseId id) ABSL_LOCKS_EXCLUDED(mu_);

  // Retire a single Use lease bound to the given subject and pid.
  // If multiple Use leases exist for the same (subject, pid), retires one of them.
  // Returns NotFound when no matching Use lease exists.
  [[nodiscard]] absl::Status release_use_lease(const ReplicaSubject& subj, pid_t pid) ABSL_LOCKS_EXCLUDED(mu_);

  void release_by_pid(pid_t pid) ABSL_LOCKS_EXCLUDED(mu_);

  void release_session(const std::string& sid) ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] absl::Time next_deadline() const ABSL_LOCKS_EXCLUDED(mu_);

  void expire_due(absl::Time now) ABSL_LOCKS_EXCLUDED(mu_);

  void handle_pid_exit(pid_t pid) ABSL_LOCKS_EXCLUDED(mu_);

  // Attach a PID monitor (owned by the caller) for event-driven PID exit handling.
  void attach_pid_monitor(PidMonitor* mon) ABSL_LOCKS_EXCLUDED(mu_);

  // External PID liveness watches (for resources that are not modeled as leases,
  // e.g. long-lived VRAM region registrations with ttl_ms==0).
  //
  // These are reference-counted: callers should pair watch/unwatch. When the
  // last external watch is removed and no lease PID guards remain, the PID is
  // unwatched from the monitor.
  void watch_pid(pid_t pid) ABSL_LOCKS_EXCLUDED(mu_);
  void unwatch_pid(pid_t pid) ABSL_LOCKS_EXCLUDED(mu_);

  // Query counters for eviction and status.
  [[nodiscard]] size_t use_count_for(const store::loading::ReplicaKey& key) const ABSL_LOCKS_EXCLUDED(mu_);

  [[nodiscard]] size_t placement_pin_count_for(const store::loading::ReplicaKey& key) const ABSL_LOCKS_EXCLUDED(mu_);

 private:
  struct Counts {
    int use_count{0};
    int placement_pins{0};
  };

  void inc_use_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void dec_use_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void inc_pin_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void dec_pin_(const ReplicaSubject& s) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void sync_pid_watches_() ABSL_LOCKS_EXCLUDED(mu_);
  void sweep_pid_liveness();

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
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Mutex guarding state below.
  mutable absl::Mutex mu_;

  LeaseId next_id_ ABSL_GUARDED_BY(mu_){1};
  GuardId next_guard_id_ ABSL_GUARDED_BY(mu_){1};
  absl::flat_hash_map<LeaseId, LeaseRec> by_id_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<GuardId, GuardRec> guard_by_id_ ABSL_GUARDED_BY(mu_);
  std::priority_queue<Due> deadlines_ ABSL_GUARDED_BY(mu_);
  PidMonitor* monitor_ ABSL_GUARDED_BY(mu_){nullptr};
  absl::flat_hash_map<store::loading::ReplicaKey, Counts, store::loading::ReplicaKeyHash> counters_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<pid_t, std::unordered_set<GuardId>> pid_index_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<pid_t, uint32_t> external_pid_watches_ ABSL_GUARDED_BY(mu_);
  std::function<void(const ReplicaSubject&)> eviction_notify_ ABSL_GUARDED_BY(mu_);
  std::function<void(absl::Time)> schedule_hook_ ABSL_GUARDED_BY(mu_);

  // Session principal guards (deadline-driven).
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
  void notify_schedule_if_earlier_(std::optional<absl::Time> when) ABSL_LOCKS_EXCLUDED(mu_);

  // Notify scheduler with the given earliest deadline value (used after pops).
  void notify_schedule_if_earlier_(absl::Time when) ABSL_LOCKS_EXCLUDED(mu_);

  void retire_lease_(LeaseId id, const char* /*reason*/) ABSL_LOCKS_EXCLUDED(mu_);

  // Attempt to unload a daemon-owned GPU replica for the given subject if both
  // the use_count and placement_pins are zero at the time of the check. This is
  // best-effort and runs off-lock. It only targets daemon-owned memory: we
  // consult the engine state and skip when no GPU residency is present.
  void maybe_unload_daemon_replica_(const ReplicaSubject& subj);

  // Retry unload attempts that previously failed while references were already
  // drained (for example, unload raced with an in-flight LOADING transition).
  void retry_pending_gpu_unloads_();

  // GPU subjects that should be retried by sweep_once().
  absl::flat_hash_set<ReplicaSubject, store::loading::ReplicaKeyHash> pending_gpu_unloads_ ABSL_GUARDED_BY(mu_);
};

class SessionLifecycleTask final {
 public:
  explicit SessionLifecycleTask(SessionLifecycleManager& mgr);

  void poll();

  [[nodiscard]] static const char* name();

 private:
  SessionLifecycleManager& mgr_;
};

} // namespace tensorcast::daemon
