---
slug: global-store-conflict-resilience-and-idempotent-control-plane
title: Global Store Conflict Resilience and Idempotent Control-Plane (MVP Plan)
areas: ["global_store", "daemon", "core", "proto", "sdk"]
related_code:
  - tensorcast/global_store/repositories/base.py
  - tensorcast/global_store/repositories/artifact_repository.py
  - tensorcast/global_store/repositories/artifact_index_repository.py
  - tensorcast/global_store/rpc/replica_registration_rpc_handler.py
  - tensorcast/global_store/rpc/worker_rpc_handler.py
  - tensorcast/global_store/services/worker_control_reducer.py
  - core/store/components/global_store_client.cc
  - core/store/runtime/metadata/metadata_gateway.cc
  - daemon/ha/worker_lifecycle_manager.cc
  - proto/tensorcast/global_store/v1/global_store.proto
  - schema.sql
  - tests/python/global_store/test_grpc_service.py
  - tests/python/global_store/test_concurrency.py
links:
  design: ../designs/0075-global-store-conflict-resilience-and-idempotent-control-plane.md
---

# Objective

Deliver a minimal, high-impact fix for conflict storms by changing only two RPC paths:

- `RegisterReplica`
- `UnregisterWorker`

MVP principles:

- fail-fast on transactional conflict,
- deterministic replay by `client_request_id`,
- no Python SDK public API changes,
- no new runtime config surface for strict/TTL toggles.

# Current State and Grounding

Hotspots seen in logs map to:

- `RegisterReplica` metadata writes (`artifacts` / `artifact_indices`) under concurrent publish.
- `UnregisterWorker` commit conflict on worker key under duplicated lifecycle calls.

Relevant current behavior:

- register-replica reducer routing currently uses worker key.
- unregister path can be triggered more than once in stop/recovery edges.
- tx conflict handling still leaves room for hidden retry-style behavior in some paths.

# Status Update (2026-02-12)

- Implemented all planned code changes for MVP scope (`RegisterReplica` + `UnregisterWorker` only).
- Added proto + schema changes and regenerated Python proto bindings.
- Added Python idempotency repository and handler integration.
- Added C++ client deterministic `client_request_id` generation and metadata/daemon wiring.
- Added regression tests for replay/mismatch/fail-fast behavior.
- Verified:
  - `source .venv/bin/activate && uv run pytest tests/python/global_store/test_grpc_service.py` (`42 passed`)
  - `source .venv/bin/activate && uv run pytest tests/python/global_store/test_concurrency.py` (`9 passed`)
  - `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake` (`PASSED`)
  - `bazel test //core/store/runtime/metadata:metadata_gateway_test --test_env=TENSORCAST_CUDA_BACKEND=fake` (`PASSED`)

# Baseline and Entry Criteria

- [x] Baseline Python global-store tests pass.
- [x] Baseline daemon lifecycle sync test passes.
- [ ] Team accepts MVP scope lock (only two RPCs).

# Phases and Milestones

- [x] Phase 1: Protocol and schema minimum
  - [x] Milestone 1: add `client_request_id` only to `RegisterReplicaRequest` and `UnregisterWorkerRequest`.
  - [x] Milestone 2: add minimal `control_plane_idempotency` table in `schema.sql`.
  - [x] Milestone 3: regenerate proto bindings.

- [x] Phase 2: Server path hardening (two RPCs only)
  - [x] Milestone 1: route `RegisterReplica` by `artifact:{artifact_id}` when available.
  - [x] Milestone 2: implement idempotency reserve/replay/conflict in the two handlers.
  - [x] Milestone 3: enforce fail-fast tx conflict behavior for the two paths.

- [x] Phase 3: Daemon and C++ client alignment
  - [x] Milestone 1: generate and attach deterministic `client_request_id` in C++ Global Store client for these two RPCs.
  - [x] Milestone 2: use publish context as register-replica identity source in metadata gateway.
  - [x] Milestone 3: add unregister single-flight in worker lifecycle manager.

- [ ] Phase 4: Tests and rollout
  - [x] Milestone 1: add replay/mismatch/fast-fail regressions for two paths.
  - [ ] Milestone 2: run TP=4 equivalent stress focused on register-replica.
  - [ ] Milestone 3: staged rollout and log/metric gate check.

# Detailed Implementation Checklist

- [x] Proto and schema
  - [x] Edit `proto/tensorcast/global_store/v1/global_store.proto` for the two `client_request_id` fields.
  - [x] Edit `schema.sql` with minimal idempotency table and index.
  - [x] Run `bash tools/build_proto_python.sh`.

- [x] Server
  - [x] Update `tensorcast/global_store/rpc/replica_registration_rpc_handler.py`:
    - [x] reducer key = artifact key when artifact id exists.
    - [x] idempotency gate using `client_request_id`.
    - [x] fail-fast on tx conflict.
  - [x] Update `tensorcast/global_store/rpc/worker_rpc_handler.py`:
    - [x] idempotency gate for unregister.
    - [x] fail-fast on tx conflict.
  - [x] Update `tensorcast/global_store/repositories/base.py`:
    - [x] keep primary commit error dominant over rollback secondary noise.
  - [x] Update metadata repositories:
    - [x] `tensorcast/global_store/repositories/artifact_repository.py` immutable-first behavior.
    - [x] `tensorcast/global_store/repositories/artifact_index_repository.py` immutable-first behavior.

- [x] Daemon and C++ client
  - [x] Update `core/store/components/global_store_client.cc` to auto-generate `client_request_id` for:
    - [x] register-replica,
    - [x] unregister-worker.
  - [x] Update `core/store/runtime/metadata/metadata_gateway.cc` to pass publish-context-derived request identity.
  - [x] Update `daemon/ha/worker_lifecycle_manager.cc` for unregister single-flight.

- [x] Explicit non-goal enforcement
  - [x] Do not add strict/TTL runtime config knobs for this MVP.
  - [x] Do not change Python SDK public API signatures.

# Test Matrix

- [x] Python checks
  - [x] `source .venv/bin/activate && uv run pytest tests/python/global_store/test_grpc_service.py`
  - [x] `source .venv/bin/activate && uv run pytest tests/python/global_store/test_concurrency.py`

- [x] Daemon regression
  - [x] `bazel test //daemon:worker_lifecycle_manager_sync_test --test_env=TENSORCAST_CUDA_BACKEND=fake`

- [x] New required regressions
  - [x] same `client_request_id` register-replica replay returns same result.
  - [x] same `client_request_id` unregister replay is idempotent.
  - [x] same `client_request_id` with mismatched payload returns failed precondition.
  - [x] tx conflict in these paths fails immediately (no hidden internal retry loop).

# Rollout Plan

- [ ] Stage 0: Deploy server support (backward compatible)
  - [ ] old clients without `client_request_id` keep working.
  - [ ] new logs/metrics visible for idempotency and conflict context.

- [ ] Stage 1: Deploy daemon/C++ client changes
  - [ ] daemon emits `client_request_id` for two RPCs.
  - [ ] verify replay-hit and mismatch signals.

- [ ] Stage 2: Stabilization
  - [ ] run focused TP=4 publish/unregister scenarios.
  - [ ] verify no conflict storm reappearance in targeted paths.

# Rollout Gates

- [ ] no hidden retry loops in `RegisterReplica` / `UnregisterWorker` error paths.
- [ ] no primary transaction errors dominated by rollback-no-active-tx.
- [ ] idempotency replay behavior is deterministic in production-like traffic.

# Backout Plan

- [ ] Keep proto fields optional; old clients remain compatible.
- [ ] Revert artifact-key routing for register-replica if severe side effects appear.
- [ ] Temporarily disable idempotency checks in handlers only if incident response requires it.

# Risks and Tracking

- [ ] Risk: fail-fast exposes latent races early
  - [ ] track failure rate and mean time to fix.
- [ ] Risk: idempotency storage growth
  - [ ] track table size and cleanup behavior.
- [ ] Risk: daemon duplicate unregister edge cases persist
  - [ ] track duplicate submit frequency and single-flight hit rate.

# Owner Checklist

- [ ] Scope freeze respected (two RPC paths only).
- [ ] Design review approved by Global Store + daemon owners.
- [ ] Proto/schema changes reviewed.
- [ ] Runbook updated for first-failure triage in these two paths.
