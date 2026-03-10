---
slug: global-store-control-plane-reconcile-v2
title: Global Store Worker Control-Plane Reconcile V2 (Execution Plan and Progress)
status: draft
areas: ["global_store", "daemon", "core"]
related_code:
  - tensorcast/global_store/repositories/worker_repository.py
  - tensorcast/global_store/services/worker_service.py
  - tensorcast/global_store/services/recovery_service.py
  - tensorcast/global_store/services/worker_control_reducer.py
  - tensorcast/global_store/rpc/worker_rpc_handler.py
  - tensorcast/global_store/rpc/worker_state_sync_rpc_handler.py
  - tensorcast/global_store/maintenance_coordinator.py
  - daemon/ha/worker_lifecycle_manager.cc
  - core/store/components/global_store_client.cc
  - schema.sql
---

# Objective

Eliminate recurring worker control-plane conflict storms and retry loops by enforcing a true single-writer topology per worker, correcting reconcile sequencing semantics, and hardening retry behavior across Global Store and daemon.

# Current State and Grounding

Observed production failure pattern (2026-02-12):

- `WorkerHeartbeat(enhanced)` repeatedly fails with `TransactionContext Error: Conflict on tuple deletion`.
- daemon enters re-registration path after heartbeat failures.
- `ReconcileWorkerState` repeatedly returns `RETRY_LATER` and daemon keeps retrying.

Code-level grounding:

- Heartbeat writes worker liveness in `tensorcast/global_store/repositories/worker_repository.py`.
- Reconcile request gap returns `RETRY_LATER` in `tensorcast/global_store/services/recovery_service.py`.
- Daemon keeps issuing new reconcile tokens and retries in `daemon/ha/worker_lifecycle_manager.cc`.
- Client-side RPC retries are applied in `core/store/components/global_store_client.cc`.

# Scope

In scope:

- Worker control-plane write paths: register, heartbeat, unregister, reconcile, maintenance, recovery.
- Reconcile sequencing (`generation`, `request_seq`) and retry semantics.
- Control-plane retry and backoff behavior between daemon and Global Store.
- Test coverage and observability for conflict/retry loops.

Out of scope:

- Multi-node consensus redesign.
- Unrelated artifact query or placement policy redesign.

# Phases and Milestones

- [x] Stage 0: Incident triage and root-cause capture
  - [x] Milestone 0.1: map failure logs to concrete code paths.
  - [x] Milestone 0.2: identify retry amplification chain across client, daemon, and server.
  - [x] Milestone 0.3: produce staged execution plan with measurable gates.

- [x] Stage 1: Reproducibility and instrumentation baseline
  - [x] Milestone 1.1: add deterministic stress tests for daemon restart + register/heartbeat/reconcile concurrency.
  - [x] Milestone 1.2: add per-worker tracing fields (`worker_id`, `daemon_id`, reducer shard, operation kind) across control-plane logs.
  - [x] Milestone 1.3: establish baseline conflict-rate and retry-loop metrics.

- [x] Stage 2: Single-writer boundary hardening
  - [x] Milestone 2.1: enforce one reducer lane per logical worker identity across register/heartbeat/unregister/reconcile/maintenance.
  - [x] Milestone 2.2: remove worker-row write bypasses from background maintenance paths.
  - [x] Milestone 2.3: verify no same-worker concurrent writes can occur outside reducer topology.

- [x] Stage 3: Reconcile sequencing and idempotent progression
  - [x] Milestone 3.1: make daemon `request_seq` progression ack-driven (no increment on transport failure or `RETRY_LATER`).
  - [x] Milestone 3.2: accept replay of same generation + same request sequence idempotently in server path.
  - [x] Milestone 3.3: prevent persistent request-gap loops.

- [x] Stage 4: Retry policy and fail-safe behavior
  - [x] Milestone 4.1: split transient transaction conflict handling from connectivity faults.
  - [x] Milestone 4.2: avoid immediate re-registration on conflict-class heartbeat failures.
  - [x] Milestone 4.3: cap retry storms with bounded exponential backoff and clear stop conditions.

- [ ] Stage 5: Validation, rollout, and stabilization
  - [ ] Milestone 5.1: run long-duration soak tests with `worker_control_reducer.shard_count > 1`.
  - [ ] Milestone 5.2: pass targeted Python and daemon regression suites.
  - [ ] Milestone 5.3: complete staged rollout gates and production monitoring checks.

# Detailed Task Checklist

- [x] Add a dedicated Python stress test file for worker control-plane conflict reproduction under restart and concurrent heartbeats.
- [ ] Add explicit metrics for:
  - [x] conflict/reducer intent counts by operation kind and reducer shard.
  - [ ] `RETRY_LATER` count and duration by worker.
  - [ ] re-registration trigger reason breakdown.
- [x] Refactor control reducer keying so worker lifecycle writes always map to one stable key.
- [x] Refactor maintenance worker cleanup path to submit per-worker intents, not direct/bulk writes against worker rows.
- [x] Update daemon reconcile token management to prevent seq gaps after retries.
- [ ] Add regression tests for:
  - [x] `RETRY_LATER` does not create unbounded request sequence growth.
  - [x] heartbeat conflict does not directly trigger re-registration storm.
  - [x] same-worker control-plane writes serialize correctly under load.

# Progress Log

| Date | Stage | Status | Notes |
| --- | --- | --- | --- |
| 2026-02-12 | Stage 0 | Completed | Root cause and amplification chain identified from daemon/global-store logs and code paths. |
| 2026-02-12 | Stage 1 | Completed | Added `tests/python/global_store/test_worker_control_plane_stage1.py` with restart+heartbeat+reconcile stress and log-context assertions; added metrics `tc_worker_control_reducer_intents_total` and `tc_reconcile_retry_later_total`; baseline from stress run: `conflict_gap_delta=3`, `retry_later_delta=3`, `retry_later_reason_delta=3`, `heartbeat_submitted_delta=13`; regression checks passed: `uv run pytest tests/python/global_store/test_worker_control_plane_stage1.py -s` (2 passed), `uv run pytest tests/python/global_store/test_grpc_service.py` (42 passed), `uv run pytest tests/python/global_store/test_recovery_service_checksum.py` (7 passed). |
| 2026-02-12 | Stage 2/3 | Completed | Implemented stable control-lane resolution (`daemon_id`-keyed) across register/heartbeat/unregister/reconcile and switched maintenance cleanup from bulk timeout write to per-worker reducer intents; implemented daemon ACK-driven reconcile sequence (`request_seq` no longer advances on transport failure/`RETRY_LATER`) and server idempotent replay (`same generation + same request_seq` returns `NOOP`); added tests `tests/python/global_store/test_worker_control_plane_stage23.py` and updated `tests/python/global_store/test_recovery_service_checksum.py`; validation passed: `source .venv/bin/activate && uv run pytest tests/python/global_store/test_worker_control_plane_stage23.py tests/python/global_store/test_recovery_service_checksum.py tests/python/global_store/test_grpc_service.py` (51 passed), `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors` (PASSED). |
| 2026-02-12 | Stage 4 | Completed | Heartbeat conflict path now returns `ABORTED` (server-side classification), daemon heartbeat failures are classified (`conflict`/`identity`/`connectivity`) and only identity failures can trigger re-registration with bounded exponential cooldown, and heartbeat loop applies bounded backoff to suppress retry storms; client heartbeat response now maps `STATUS_NOT_FOUND`/timeout/resource-exhausted to typed statuses; added regression tests in `tests/python/global_store/test_grpc_service.py` (heartbeat tx conflict fast-fail) and `daemon/ha/worker_lifecycle_manager_sync_test.cc` (conflict suppresses re-registration, identity failure re-registration bounded). |
| 2026-02-12 | Stage 4 Validation | Completed | Re-ran Stage4 regression checks on latest code: `source .venv/bin/activate && uv run ruff check tensorcast/global_store/rpc/worker_rpc_handler.py tests/python/global_store/test_grpc_service.py` (passed), `source .venv/bin/activate && uv run pytest tests/python/global_store/test_grpc_service.py::TestGRPCService::test_worker_heartbeat_tx_conflict_fails_fast` (1 passed), `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error` (PASSED). |

# Validation and Acceptance Gates

- [ ] No recurring `Conflict on tuple deletion` loop for the same worker during restart/reconnect stress.
- [ ] No sustained `RETRY_LATER` loop with fixed short retry interval for a single worker.
- [x] No unbounded `request_seq` drift caused by retry paths.
- [ ] Daemon readiness reaches healthy state without repeated re-register storms.
- [ ] Soak run completes with stable conflict and retry metrics.

# Test and Rollout Plan

Test plan:

- Python:
  - `source .venv/bin/activate && uv run pytest tests/python/global_store/...`
- Daemon:
  - `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Additional stress:
  - dedicated restart + concurrent heartbeat/reconcile scenario with shard count greater than 1.

Rollout:

- Stage rollout by environment (dev -> canary -> wider production).
- Gate each promotion on conflict/retry dashboards and log signatures.

Backout:

- Revert reconcile sequence behavior changes and reducer keying changes independently if a regression appears.
- Keep new observability metrics in place during rollback for root-cause continuity.

# Risks and Tracking

- [ ] Risk: behavior changes in reconcile sequencing may expose hidden assumptions in daemon lifecycle state machine.
- [ ] Risk: stricter serialization can reduce throughput if keying is too coarse.
- [ ] Risk: retries become too conservative and delay real recovery under transient network faults.

# Update Protocol

When progress changes:

- Update stage and milestone checkboxes.
- Append one line to `Progress Log` with date, stage, status, and evidence.
- Keep acceptance gates aligned with actual observed behavior and test results.
