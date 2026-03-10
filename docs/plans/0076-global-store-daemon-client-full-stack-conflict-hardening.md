---
slug: global-store-daemon-client-full-stack-conflict-hardening
title: Global Store / Daemon / Client Full-Stack Conflict Hardening Plan
status: draft
areas: ["global_store", "daemon", "core", "sdk"]
related_code:
  - tensorcast/global_store/repositories/base.py
  - tensorcast/global_store/repositories/placement_repository.py
  - tensorcast/global_store/rpc/placement_persistence_rpc_handler.py
  - tensorcast/global_store/services/placement_service.py
  - tensorcast/global_store/services/worker_control_reducer.py
  - daemon/ha/worker_lifecycle_manager.cc
  - daemon/state/persistence_manager.cc
  - core/store/components/global_store_client.cc
  - tests/python/global_store/test_grpc_service.py
  - daemon/ha/worker_lifecycle_manager_sync_test.cc
---

# Objective

Deliver a root-cause fix for conflict/retry storms across the full control-plane path:

- Global Store write conflicts must not amplify into repeated internal exceptions.
- daemon must not spin aggressive reconcile/report loops when Global Store is down.
- client/daemon report paths must become conflict-aware and rate-bounded.

# Current State and Grounding

Observed failure signatures:

- `ReportPersistenceStatus` commit conflict on `artifact_placement_targets` under concurrent updates.
- transaction rollback branch throws `UnboundLocalError` and masks primary conflict error.
- daemon `ReconcileWorkerState` shows repeated `attempt 1/4...4/4` cycles after Global Store shutdown due to immediate re-queue.

Code-level grounding:

- Transaction wrapper and rollback handling: `tensorcast/global_store/repositories/base.py`
- Placement status write path: `tensorcast/global_store/repositories/placement_repository.py`, `tensorcast/global_store/services/placement_service.py`, `tensorcast/global_store/rpc/placement_persistence_rpc_handler.py`
- Daemon state sync retry behavior: `daemon/ha/worker_lifecycle_manager.cc`
- Daemon persistence report emission cadence: `daemon/state/persistence_manager.cc`
- C++ RPC retry/status mapping: `core/store/components/global_store_client.cc`

# Scope

In scope:

- Placement/persistence write serialization and conflict handling in Global Store.
- Daemon reconcile/report retry shaping under connectivity loss.
- Client/daemon status handling alignment for persistence/report RPCs.
- End-to-end acceptance gates for conflict, retry pressure, and recovery latency.

Out of scope:

- Distributed consensus redesign.
- Storage engine swap away from DuckDB.
- New external user-facing API surface in Python SDK.

# Baseline and Entry Criteria

- [ ] Reproduce with deterministic test and archived logs:
  - [ ] placement conflict path reproduces without process crash.
  - [ ] daemon reconcile loop under GS shutdown reproduces high retry pressure.
- [ ] Capture baseline metrics:
  - [ ] per-minute reconcile attempts during GS outage.
  - [ ] per-minute `ReportPersistenceStatus` failures.
  - [ ] conflict/error class distribution in Global Store.

# Phases and Milestones

- [x] Phase 0: Immediate correctness guardrails (P0)
  - [x] Milestone 0.1: fix `BaseRepository.transaction()` rollback variable lifetime bug so primary errors are preserved.
  - [x] Milestone 0.2: add regression test proving no `UnboundLocalError` on commit conflict + failed rollback.
  - [x] Milestone 0.3: classify placement persistence transient tx conflict as retryable class (`ABORTED`) instead of generic `INTERNAL`.

- [x] Phase 1: Global Store placement single-writer hardening
  - [x] Milestone 1.1: enforce per-key serialized write lane for placement target updates (`plan_id` + `shard_idx` + `node_id` family).
  - [x] Milestone 1.2: remove/avoid parallel write paths that can touch same target rows outside the selected lane.
  - [x] Milestone 1.3: add bounded conflict retry only inside the owning lane; no cross-thread blind retries.

- [x] Phase 2: Daemon outage-mode retry shaping
  - [x] Milestone 2.1: in `perform_state_sync` failure path, replace immediate `request_state_sync()` with bounded exponential backoff + jitter.
  - [x] Milestone 2.2: add connectivity-aware cooldown state so `UNAVAILABLE/connection refused` enters outage mode.
  - [x] Milestone 2.3: restore normal cadence quickly after first successful reconcile.

- [x] Phase 3: Persistence report de-amplification (daemon + client alignment)
  - [x] Milestone 3.1: send persistence status on change (state/progress delta threshold) rather than every tick.
  - [x] Milestone 3.2: add minimum report interval guard during repeated failures.
  - [x] Milestone 3.3: ensure C++ `ReportPersistenceStatus` maps retryable transient conflicts distinctly from hard errors.

- [ ] Phase 4: End-to-end resilience and verification
  - [ ] Milestone 4.1: add GS-down chaos test (shutdown/restart) validating bounded daemon retry QPS.
  - [ ] Milestone 4.2: add placement conflict stress test validating no error-masking and eventual convergence.
  - [ ] Milestone 4.3: soak run with multiple workers and persistence traffic for at least 30 minutes.

- [ ] Phase 5: Rollout and stabilization
  - [ ] Milestone 5.1: dev rollout with metrics gates.
  - [ ] Milestone 5.2: canary rollout with rollback checkpoints.
  - [ ] Milestone 5.3: production rollout after two stable canary windows.

# Detailed Implementation Checklist

- [ ] Global Store
  - [x] Patch `tensorcast/global_store/repositories/base.py` transaction exception path.
  - [x] Add/reuse transient conflict classifier for placement persistence RPC handler.
  - [x] Introduce placement write-lane serialization strategy (service/reducer/repository boundary).
  - [x] Add bounded retry in placement status lane (`PlacementService.record_status`) for transient tx conflicts.
  - [ ] Add structured logs with keys:
    - [ ] `plan_id`
    - [ ] `task_id`
    - [ ] `worker_id`
    - [ ] `daemon_id`
    - [ ] `error_class`
    - [ ] `attempt`

- [ ] daemon
  - [x] Implement reconcile outer-loop cooldown/backoff in `daemon/ha/worker_lifecycle_manager.cc`.
  - [x] Add metrics for:
    - [x] outage-mode active
    - [x] reconcile enqueue suppressed count
    - [x] reconnect recovery latency
  - [x] Add persistence report emission throttling in `daemon/state/persistence_manager.cc`.

- [ ] core client
  - [x] Refine `core/store/components/global_store_client.cc` status mapping for `ReportPersistenceStatus`.
  - [ ] Ensure retry policy distinguishes:
    - [x] transaction conflict
    - [ ] connectivity outage
    - [ ] identity/configuration errors

- [ ] tests
  - [x] Python regression for transaction wrapper correctness on commit conflict.
  - [x] Python/GS regression for `ReportPersistenceStatus` transient conflict classification.
  - [x] daemon sync test for GS outage retry cap behavior.
  - [ ] integration stress for placement updates on same shard targets.

# Acceptance Gates

- [x] No `UnboundLocalError` in transaction handling under conflict/failure paths.
- [x] During GS outage, daemon reconcile attempt rate is bounded (no immediate tight re-queue loop).
- [x] `ReportPersistenceStatus` conflict path surfaces consistent retryable class; no generic internal spam loop.
- [x] Reconcile transient transaction conflicts degrade to `RETRY_LATER` instead of surfacing as fatal/internal RPC failures.
- [ ] Placement target updates converge without repeated same-key write conflicts under stress.
- [ ] After GS recovery, daemon returns to healthy state without manual restart.

# Test Plan

- Python:
  - `source .venv/bin/activate && uv run pytest tests/python/global_store/test_grpc_service.py`
  - `source .venv/bin/activate && uv run pytest tests/python/global_store/test_worker_control_plane_stage1.py`
  - `source .venv/bin/activate && uv run pytest tests/python/global_store/test_worker_control_plane_stage23.py`
- Daemon/C++:
  - `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
- Additional stress:
  - dedicated GS shutdown/restart chaos run with persistence reporting enabled.
  - same-placement-key concurrent update stress.

# Rollout Plan

- [ ] Stage A (dev): enable all patches and verify acceptance gates with synthetic load.
- [ ] Stage B (canary): limited worker subset, monitor retry/conflict dashboards.
- [ ] Stage C (production): full rollout after canary stability and no regression signatures.

# Backout Plan

- [ ] Backout order:
  - [ ] disable new daemon outage-mode throttling logic first if it causes stale behavior.
  - [ ] revert placement write-lane changes if throughput/latency regressions are severe.
  - [ ] keep transaction error-path fix in place (never back out unless proved incorrect).
- [ ] Preserve new metrics/log keys during rollback for continued diagnosis.

# Risks and Tracking

- [ ] Risk: stricter serialization reduces peak placement update throughput.
  - [ ] Mitigation: per-key sharding and queue-depth observability.
- [ ] Risk: over-conservative daemon backoff delays recovery.
  - [ ] Mitigation: fast-reset on first successful reconcile and bounded max cooldown.
- [ ] Risk: change-only persistence reporting may hide intermediate state transitions.
  - [ ] Mitigation: force terminal-state immediate report and periodic heartbeat report floor.

# Progress Log

| Date | Phase | Status | Notes |
| --- | --- | --- | --- |
| 2026-02-12 | Plan Draft | Completed | Initial full-stack root-fix plan created from live failure signatures (`placement` conflict + transaction wrapper masking + daemon outage retry storm). |
| 2026-02-12 | Phase 0 | Completed | Fixed transaction wrapper rollback error lifetime in `tensorcast/global_store/repositories/base.py` and added regression `test_base_repository_transaction_keeps_primary_error_on_rollback_noop` in `tests/python/global_store/test_repositories.py`; placement persistence conflict path now returns `ABORTED` with explicit conflict metric/logging in `tensorcast/global_store/rpc/placement_persistence_rpc_handler.py`. |
| 2026-02-12 | Phase 1 | Completed | Added artifact-scoped single-writer lane in `tensorcast/global_store/services/placement_service.py` for both `plan_placement` and `record_status`, and added bounded transient conflict retry in-lane (3 attempts with backoff). This replaces reliance on coarse global transaction serialization and keeps contention isolated to hot artifacts. |
| 2026-02-12 | Phase 2 | Completed | Added state sync outer-loop backoff/cooldown for transport/connectivity failures in `daemon/ha/worker_lifecycle_manager.cc` (failure classification, bounded backoff, fast reset on success) and added daemon regression `WorkerLifecycleManager backs off reconcile retries on transport failures` in `daemon/ha/worker_lifecycle_manager_sync_test.cc`. |
| 2026-02-12 | Phase 3 | Completed | Implemented persistence report de-amplification in `daemon/state/persistence_manager.cc` using signature-based change detection + minimum unchanged interval; validated via `//daemon:persistence_manager_test` (includes `PersistenceManager throttles unchanged status reports`). |
| 2026-02-12 | Reconcile Conflict Hardening | Completed | Updated `tensorcast/global_store/services/recovery_service.py` so exhausted transient reconcile tx conflicts return `RETRY_LATER` with bounded retry delay (500ms) and explicit conflict/retry metrics instead of bubbling as `INTERNAL`; added regression `test_transient_reconcile_conflict_exhausted_returns_retry_later` in `tests/python/global_store/test_recovery_service_checksum.py` (`8 passed`). |
| 2026-02-12 | Validation | Completed | Ran `source .venv/bin/activate && uv run ruff check tensorcast/global_store/repositories/base.py tensorcast/global_store/repositories/placement_repository.py tensorcast/global_store/rpc/placement_persistence_rpc_handler.py tests/python/global_store/test_repositories.py tests/python/global_store/test_placement_persistence.py` (passed); `source .venv/bin/activate && uv run pytest tests/python/global_store/test_placement_persistence.py tests/python/global_store/test_repositories.py` (`21 passed`); `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=streamed --noshow_progress --noshow_loading_progress` (PASSED). |
| 2026-02-12 | Validation (incremental) | Completed | Ran `source .venv/bin/activate && uv run ruff check tensorcast/global_store/services/recovery_service.py tests/python/global_store/test_recovery_service_checksum.py` (passed); `source .venv/bin/activate && uv run pytest tests/python/global_store/test_recovery_service_checksum.py` (`8 passed`); `source .venv/bin/activate && uv run pytest tests/python/global_store/test_worker_control_plane_stage1.py tests/python/global_store/test_worker_control_plane_stage23.py` (`4 passed`); `bazel test //daemon:persistence_manager_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors` (PASSED, cached). |
| 2026-02-12 | Phase 1 Validation Update | Completed | Added `test_record_status_retries_transient_conflict` in `tests/python/global_store/test_placement_service.py`; ran `source .venv/bin/activate && uv run ruff check tensorcast/global_store/repositories/base.py tensorcast/global_store/services/placement_service.py tests/python/global_store/test_placement_service.py tests/python/global_store/test_repositories.py` (passed) and `source .venv/bin/activate && uv run pytest tests/python/global_store/test_placement_service.py tests/python/global_store/test_placement_persistence.py tests/python/global_store/test_repositories.py tests/python/global_store/test_recovery_service_checksum.py tests/python/global_store/test_worker_control_plane_stage1.py tests/python/global_store/test_worker_control_plane_stage23.py` (`37 passed`). |
| 2026-02-12 | Daemon Metrics Hardening | Completed | Added reconcile outage/suppression/reconnect metrics in `daemon/ha/worker_lifecycle_manager.cc` (`tc_daemon_ha_outage_mode_active`, `tc_daemon_ha_reconcile_enqueue_suppressed_total`, `tc_daemon_ha_reconnect_latency_ms`) with new state accessors in `daemon/ha/worker_lifecycle_manager.h`; added regression `WorkerLifecycleManager tracks outage mode and suppresses duplicate reconcile enqueue` in `daemon/ha/worker_lifecycle_manager_sync_test.cc`; validated with `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors` (PASSED). |
