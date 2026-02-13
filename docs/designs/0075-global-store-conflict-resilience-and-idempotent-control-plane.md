---
slug: global-store-conflict-resilience-and-idempotent-control-plane
title: Global Store Conflict Resilience and Idempotent Control-Plane (MVP Design)
status: draft
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
links:
  plan: ../plans/0075-global-store-conflict-resilience-and-idempotent-control-plane.md
---

# Summary

This MVP narrows the fix scope to the two paths that currently generate the highest operational risk:

- `RegisterReplica`
- `UnregisterWorker`

The goal is to remove conflict storms with the smallest effective change set:

- deterministic reducer routing for hot keys,
- `client_request_id` idempotency on the two RPCs,
- fail-fast policy for transactional conflicts in these paths.

# Scope

In scope:

- `RegisterReplica` server path and its client callsites.
- `UnregisterWorker` server path and daemon stop-path callsites.
- Minimal idempotency storage for replay safety.
- Transaction conflict fail-fast behavior for these two paths.

Out of scope for this MVP:

- full worker lifecycle key unification (`register/heartbeat/reconcile/maintenance`).
- broad conflict policy redesign for all RPCs.
- config-surface expansion for strict mode or TTL tuning.

# Problem Statement

Observed failures (2026-02-12) show conflict hotspots concentrated in:

1. `RegisterReplica` metadata persistence (`artifacts` / `artifact_indices`).
2. `UnregisterWorker` commit conflicts on `workers.worker_id`.

Retries in these paths can mask topology issues and amplify write pressure. For MVP, these conflicts are treated as correctness failures and surfaced immediately.

# Goals and Non-Goals

## Goals

- Make duplicate requests deterministic on the two hot RPCs.
- Prevent hidden internal retries from masking transaction conflicts.
- Keep change set small enough for fast rollout and rollback.

## Non-Goals

- Re-architecting all control-plane lanes in one release.
- Adding runtime flags for strict/fail-fast behavior.
- Exposing idempotency retention TTL as user config.

# Architecture and Interfaces

## A. Lane ownership in MVP

### A1. RegisterReplica lane

`RegisterReplica` MUST route reducer intents by artifact identity when available:

- key: `artifact:{artifact_id}`
- fallback key (legacy no artifact id): `worker:{worker_id}`

This serializes concurrent TP publish writes touching the same metadata rows.

### A2. UnregisterWorker lane

`UnregisterWorker` remains worker-keyed:

- key: `worker:{worker_id}`

MVP strengthens this path through idempotency and client-side single-flight, not through full lifecycle key redesign.

## B. `client_request_id` protocol (MVP)

Add optional fields only for two requests in `proto/tensorcast/global_store/v1/global_store.proto`:

- `UnregisterWorkerRequest.client_request_id = 3`
- `RegisterReplicaRequest.client_request_id = 10`

No new field is added in `RegisterWorkerRequest` for this MVP.

Semantics:

- same `client_request_id` + same payload => replay prior result,
- same `client_request_id` + different payload => `FAILED_PRECONDITION`.

## C. Idempotency storage

Add minimal table in `schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS control_plane_idempotency (
    client_request_id TEXT PRIMARY KEY,
    operation_kind TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    response_status TEXT NOT NULL,
    response_proto BLOB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_control_plane_idempotency_created_at
    ON control_plane_idempotency(created_at);
```

Notes:

- retention is implementation-owned (fixed in code), not user-configurable in MVP.
- schema remains additive.

## D. Conflict policy for MVP paths

For `RegisterReplica` and `UnregisterWorker` only:

- transaction conflicts are fail-fast,
- server does not apply internal tx retries,
- conflict is emitted as primary error with full context.

`rollback-no-active-tx` remains secondary noise and must not replace the primary commit error.

# Client and SDK Changes

## C++ and daemon

In `core/store/components/global_store_client.cc`:

- generate deterministic `client_request_id` for:
  - `register_replica` / `register_memory_replica`
  - `unregister_worker`
- reuse the same `client_request_id` for one logical attempt chain.

In `core/store/runtime/metadata/metadata_gateway.cc`:

- derive register-replica request identity from publish context when available.

In `daemon/ha/worker_lifecycle_manager.cc`:

- enforce unregister single-flight in stop flow to avoid duplicate logical unregister operations.

## Python SDK surface

Python SDK public API signatures remain unchanged in this MVP.

# Observability

Required for the two MVP paths:

- conflict log includes: `operation_kind`, `worker_id`, `artifact_id`, `client_request_id`.
- replay-hit and replay-conflict counters for idempotency path.
- existing conflict and reducer queue metrics remain unchanged.

# Trade-offs and Risks

- Smaller scope means other control-plane paths keep existing behavior for now.
- Fail-fast can initially increase visible errors before all race edges are removed.

Mitigation:

- keep rollout focused to two hottest paths,
- track first-failure diagnostics and fix topology issues rapidly.

# Compatibility and Acceptance Criteria

## Compatibility

- old clients still work without `client_request_id`.
- new clients gain deterministic replay on the two MVP RPCs.

## Acceptance Criteria

1. `RegisterReplica` duplicate replays with same `client_request_id` return stable result.
2. `UnregisterWorker` duplicate replays with same `client_request_id` are idempotent.
3. Same `client_request_id` with mismatched payload is rejected.
4. Transaction conflicts in these two paths are surfaced immediately (no hidden server retry loop).

# Naming Compliance

- Python functions: `begin_idempotent_operation`, `is_transient_tx_conflict`.
- Python classes: `IdempotencyRepository`, `IdempotencyRecord`.
- C++ methods: `build_client_request_id`, `register_replica_idempotent`, `unregister_worker_idempotent`.
- constants: `IDEMPOTENCY_RETENTION_MS`.

# References

- `tensorcast/global_store/rpc/replica_registration_rpc_handler.py`
- `tensorcast/global_store/rpc/worker_rpc_handler.py`
- `tensorcast/global_store/repositories/base.py`
- `tensorcast/global_store/repositories/artifact_repository.py`
- `tensorcast/global_store/repositories/artifact_index_repository.py`
- `core/store/components/global_store_client.cc`
- `core/store/runtime/metadata/metadata_gateway.cc`
- `daemon/ha/worker_lifecycle_manager.cc`
- `proto/tensorcast/global_store/v1/global_store.proto`
- `schema.sql`
