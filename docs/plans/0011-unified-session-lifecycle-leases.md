---
id: plan-20250911-unified-session-lifecycle-leases
slug: 0011-unified-session-lifecycle-leases
title: Plan — Unified Session Lifecycle via Leases, Guards, and Finalizers
status: in_progress
owners: ["daemon"]
reviewers: ["core", "store", "infra"]
created: 2025-09-11
last_updated: 2025-09-11
links:
  design: ../designs/0011-unified-session-lifecycle-leases.md
---

# Objective

Deliver a unified lifecycle system in the daemon that replaces `SessionTtlTask`, `RegJoinTtlTask`, and `PidWatchTask` with a single Lease/Guard/Finalizer pipeline and a `SessionLifecycleTask`, optimizing for correctness and simplicity. Remove redundant code; no compatibility layer or dual‑run — execute the best design directly.

# Phases & Milestones

- Milestone 1 — Core Infrastructure
  - Implement Lease/Guard/Finalizer, sharded SessionLifecycleManager, ExpirationQueue (min‑heap), and PidMonitor (pidfd + polling fallback).

- Milestone 2 — Single Task Integration
  - Replace legacy sweepers with `SessionLifecycleTask` in `start_sweepers()`; delete `SessionTtlTask`, `RegJoinTtlTask`, `PidWatchTask`.

- Milestone 3 — Ownership Semantics
  - Implement PlacementLease/UseLease/CommitLease invariants; enforce device‑unique CommitIndex for VRAM_LEASED; implement TTL prefetch semantics.

- Milestone 4 — Eviction & Metrics
  - Update `EvictionTask` to consult Manager counters (UseCount/PlacementPins); remove the `keep_for_global` behavior; integrate/extend metrics.

- Milestone 5 — Tests & Docs & Cleanup
  - Unit/integration/load tests; update/remove old tests; update `daemon/README.md`, internal docs and metrics guide; clean up redundant code and paths.

# Tasks

- Core: Lease, Guard variants, Finalizer, ExpirationQueue, with tests.
- Manager: SessionLifecycleManager (sharding + indices + thread-safety).
- PidMonitor: pidfd + polling fallback, simulated PID tests.
- Reclaimer: Idempotent integration with `refs_`, `lip_mgr_`, `engine_`.
- Integration: Register a single `SessionLifecycleTask` in `start_sweepers()`; delete the three legacy sweepers.
- Controllers: materialize/register/commit paths create/renew/release Leases; convert joined lightweight refs to PlacementLease; remove use of `keep_for_global`.
- VRAM_LEASED: CommitIndex (unique on (artifact_id, device_id)); duplicate commit returns AlreadyExists; maintain keepalive/expiry handling.
- Prefetch: PlacementLease (DeadlineGuard) and reclaim on expiry if unused.
- Eviction: Use Manager's UseCount/PlacementPins for decisions.
- Metrics/Tracing: use_count, placement_pins, vram_leased_commits, commit_denied, prefetch_expired_without_use, etc.
- Docs: Update `daemon/README.md`, `docs/internals/model-loading.md`, and the metrics guide.

## Integration Details

- Register a single lifecycle task in `start_sweepers()` and remove legacy sweepers:

```cpp
// Construct once during service startup
auto lifecycle_mgr = std::make_shared<SessionLifecycleManager>(sessions_, refs_, *lip_mgr_, *reg_mgr_, *engine_);
auto lifecycle_task = std::make_shared<SessionLifecycleTask>(*lifecycle_mgr);

// Prefer dynamic scheduling (sleep until next deadline or signal).
scheduler_->add_task(TaskKind::kSessionLifecycle, /*period=*/std::chrono::milliseconds(0), [lifecycle_task]() {
  lifecycle_task->poll();
});

// Remove: SessionTtlTask, RegJoinTtlTask, PidWatchTask
// Keep: VerificationTask, LockTtlTask (until optionally unified), EvictionTask (optional)
```

## Internal APIs (shape)

```cpp
class SessionLifecycleManager {
 public:
  // Lease creation
  absl::StatusOr<LeaseId> create_use_lease(const ReplicaSubject& subj, pid_t pid);
  absl::StatusOr<LeaseId> create_placement_lease(const ReplicaSubject& subj, PlacementSpec spec);
  absl::StatusOr<LeaseId> create_commit_lease(const CommitSubject& subj, pid_t pid);

  // Renewal and release
  absl::Status keepalive_session(SessionId sid, absl::Duration ttl);
  absl::Status renew_placement(LeaseId id, absl::Duration ttl);
  void release_lease(LeaseId id);
  void release_by_pid(pid_t pid);
  void release_session(SessionId sid);

  // Scheduling hooks
  absl::Time next_deadline() const;
  void expire_due(absl::Time now);
  void handle_pid_exit(pid_t pid);
};
```

## Code Removals & Simplifications

- Delete sweeper classes: `SessionTtlTask`, `RegJoinTtlTask`, `PidWatchTask`.
- Remove `keep_for_global`; express pins as PlacementLease with TTL/Manual guards.
- Replace commit "existed=true" lightweight refs with PlacementLease management (TTL/Manual unified).
- Route all PID liveness via `PidMonitor` (pidfd + poll), not periodic `/proc` loops.
- Session TTL becomes a Session principal HeartbeatGuard; no separate session sweeper.

# Test & Cutover

- Unit: renew/expire race conditions, PID exit precedence, finalizer idempotence.
- Integration: synthetic clients creating/joining/crashing; verify reclamation latency and resource state.
- Load: scale to 1e6 mixed operations; observe CPU, lock contention, queue sizes, and `expiry_drift_ms`.
- Cutover: single cutover (no dual-run/rollback); remove legacy sweepers and `keep_for_global` paths together; enforce via CI tests and docs-check guards.

# Risks & Tracking

- pidfd unavailability → fallback correctness; tracked by `pid_monitor_fallback` metric.
- Finalizer failures → retry budgets and error counters; alerts on sustained errors.
- Drift in cleanup latency → SLO on `expiry_drift_ms`; trigger dynamic scheduling if exceeded.

---

Status Update — 2025-09-11

- Implemented unified lifecycle wiring and task:
  - Added `SessionLifecycleManager` and `SessionLifecycleTask` (daemon/session_lifecycle.h).
  - Replaced `SessionTtlTask`, `RegJoinTtlTask`, and `PidWatchTask` with unified lifecycle scheduling in `start_sweepers()`.
  - Updated `BackgroundScheduler::TaskKind` to include `kSessionLifecycle` and removed legacy kinds.
  - Kept Lock TTL, Verification, and optional Eviction tasks unchanged.

- Behavior parity in first cut:
  - Session TTL, joined‑registration TTL, and PID liveness sweeps consolidated into one pass.
  - `LipManager::sweep_expired_and_dead_pids()` invoked from unified task.
  - TTL prefetch pins: `MaterializeReplica(keep_for_global=true)` now creates a PlacementLease with a default TTL (10m) on GPU replicas; pin expires automatically if no activity.

- Docs synced:
  - Updated daemon README to describe `SessionLifecycleTask`.
  - Updated model loading internals to reference the unified task for joined‑TTL cleanup.
  - Updated daemon README to note pidfd-based PidMonitor with `/proc` fallback.

- Next steps (Milestones 1/3/4):
  - Introduce explicit Lease/Guard/Finalizer primitives and ExpirationQueue to shift from scan‑based to deadline‑driven expirations.
  - Replace `keep_for_global` with PlacementLease pins and plumb Manager counters into Eviction policy.
  - Add CommitIndex uniqueness enforcement for VRAM_LEASED and associated metrics.

Incremental progress (scaffolding & invariants):

- Added initial Lease/Guard/Finalizer scaffolding and an in‑manager ExpirationQueue:
  - SessionLifecycleManager contains `LeaseRec`, basic create/renew/release APIs, and a min‑heap for deadlines. Currently used internally by the unified task (periodic scheduling); next iteration will drive dynamic scheduling.
- VRAM_LEASED uniqueness:
  - `LipManager::commit_lease_in_place()` now enforces device‑unique commits for `(artifact_id, device_id)` and returns `AlreadyExists` when an active lease exists for the same subject.
- Pid liveness:
  - Introduced `PidMonitor` (pidfd + epoll) with `/proc` polling fallback. Unified lifecycle attaches the monitor and handles PID exits via callbacks (`handle_pid_exit`), dropping leases and refs immediately. We retain a minimal `/proc` sweep only when pidfd is unavailable.

- Dynamic scheduling:
  - Added `BackgroundScheduler::set_next_due()` and prepared the lifecycle task to update its next wake based on `SessionLifecycleManager::next_deadline()` (using time deltas). This shifts the task toward deadline‑driven execution rather than fixed intervals.

- Eviction alignment:
  - Integrated `SessionLifecycleManager` counters (UseCount/PlacementPins) into `EvictionTask` decisions; eviction now skips when there are active use refs or placement pins. The legacy `keep_for_global` flag is ignored.

- Controllers integration (initial):
  - `MaterializationController` now creates a `UseLease` for the caller PID on successful GPU materialization or LIP fast path and releases leases on `UnloadReplica` (by PID). When `keep_for_global` is requested on MaterializeReplica, it is expressed as a `PlacementLease` with a default TTL (10m) rather than a sticky flag.
  - `RegistrationController` creates a `CommitLease` on successful LIP in‑place commits (VRAM_LEASED), enforcing device‑unique ownership at the manager+LIP layer.

## Milestone Progress (2025-09-11)

- Milestone 1 — Core Infrastructure
  - [x] Initial Lease/Guard/Finalizer scaffolding (data model + APIs)
  - [x] ExpirationQueue (min‑heap) integrated into lifecycle manager
  - [x] PidMonitor (pidfd + poll fallback) integrated

- Milestone 2 — Single Task Integration
  - [x] Replace legacy sweepers with `SessionLifecycleTask` in `start_sweepers()`
  - [x] Remove `SessionTtlTask`, `RegJoinTtlTask`, `PidWatchTask` from runtime and header

- Milestone 3 — Ownership Semantics
  - [x] Placement/Use/Commit lease paths: UseLease on load; CommitLease on LIP in‑place
  - [x] Enforce device‑unique commit for VRAM_LEASED (returns AlreadyExists)
  - [x] TTL prefetch semantics — PlacementLease(DeadlineGuard) via `keep_for_global` on MaterializeReplica (10m default)

- Milestone 4 — Eviction & Metrics
  - [x] Eviction consults lifecycle counters (UseCount/PlacementPins) — integrated
  - [x] Integrate Manager counters (UseCount/PlacementPins) — EvictionTask uses lifecycle manager queries
  - [x] Metrics (minimal): commit_denied and pid_monitor_fallback added

- Milestone 5 — Tests & Docs & Cleanup
  - [x] Updated daemon README and internals doc references
  - [x] Plan doc kept in sync
  - [x] Unit tests for lifecycle (TTL prefetch expiry, PID exit precedence, deadline guard generation renewal, manual releases)
  - [x] Integration tests for lifecycle — validated key paths
  - [x] Load tests for lifecycle — completed
  - [x] Remove old tests/paths tied to legacy sweepers — audit found none remaining

Execution summary (2025-09-11):
- Verified unified lifecycle wiring and behavior with targeted tests:
  - bazel test //daemon:session_lifecycle_test → PASSED
  - bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true → PASSED
  - bazel test //daemon:grpc_service_impl_lip_ttl_expiry_test --define=use_fake_cuda=true → PASSED
  - bazel test //daemon:session_lifecycle_load_test → PASSED
- start_sweepers(): switched SessionLifecycleTask to 0ms base interval and rely on deadline-driven rescheduling via next_deadline().
- Controllers: UseLease created on load paths; PlacementLease TTL pin created when keep_for_global=true; CommitLease created on LIP in-place commits with device-unique enforcement.

Completed guard/finalizer broadening and deadline-driven scheduling:
- Introduced Guard records (PidLiveness, Deadline, Manual) with generations and a min-heap keyed by guard deadlines.
- Leases now track guard sets and run idempotent finalizers on retirement.
- PidMonitor exit events fail pid guards and retire associated leases immediately.
- ExpirationQueue processes guard generations to avoid ABA and retires leases on due deadlines.
