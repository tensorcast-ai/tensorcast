---
slug: persistent-worker-state-version
title: Persistent Worker State Versioning and Checksums (Design)
status: draft
areas: ["global_store", "daemon"]
related_code:
  - tensorcast/global_store/services/recovery_service.py
  - tensorcast/global_store/repositories/worker_repository.py
  - tensorcast/global_store/models/worker.py
  - tensorcast/global_store/grpc_service.py
  - schema.sql
links:
  plan: ../plans/0047-persistent-worker-state-version.md
---

# Summary

Persist `state_version` and `state_checksum` in the Global Store database and make state synchronization transactional. This prevents version drift on partial failures, removes per-heartbeat full-table checksum scans, and keeps state version stable across restarts.

# Problem Statement

The current recovery flow keeps `worker_state_versions` in memory and increments the version even if some state changes fail. On restart, versions reset and checksums are recomputed by scanning all replicas on each heartbeat. This allows silent drift where the version advances without fully applying state changes.

# Goals / Non-Goals

## Goals

- Persist `state_version` and `state_checksum` per worker.
- Apply state changes transactionally and only bump version on full success.
- Avoid full-table checksum recomputation on every heartbeat.
- Keep checksum format aligned with the daemon checksum algorithm.

## Non-Goals

- Changing the checksum algorithm or protocol fields.
- Rewriting replica data model or ownership rules.
- Introducing a new storage engine or migration system.

# Architecture & Interfaces

## Data model

Store state version and checksum in the `workers` table. This keeps identity and HA state colocated and avoids extra joins.

## Recovery and sync flow

- `RecoveryService` reads and writes `state_version` and `state_checksum` via `WorkerRepository`.
- `synchronize_worker_state` runs in a single transaction:
  - Apply all state changes.
  - If any operation fails, rollback and return error without bumping version.
  - On success, compute checksum once and update `(state_version, state_checksum)` together.
- `get_worker_state_checksum` returns the stored checksum. If empty or missing (legacy row), compute once and persist.
- `ensure_worker_state_version` loads the stored value and initializes it to 1 if missing.

## Repository updates

- Extend `Worker` model with `state_version` and `state_checksum` fields.
- Add `WorkerRepository` helpers:
  - `get_state_version`, `set_state_version`
  - `get_state_checksum`, `set_state_checksum`
  - `update_state_version_and_checksum` (transaction-safe)
- Add transaction-friendly replica helpers as needed (for delete/update paths) to keep `_apply_state_changes` atomic.

## Naming Compliance (Python APIs)

- `get_state_version`, `set_state_version`, `get_state_checksum`, `set_state_checksum` use snake_case.
- `update_state_version_and_checksum` uses snake_case.
- `state_version` and `state_checksum` model fields use snake_case.

# Schema Changes

Add columns to the canonical schema.

```sql
ALTER TABLE workers ADD COLUMN IF NOT EXISTS state_version BIGINT NOT NULL DEFAULT 1;
ALTER TABLE workers ADD COLUMN IF NOT EXISTS state_checksum TEXT NOT NULL DEFAULT '';
```

# Trade-offs & Risks

- Additional writes to `workers` on sync success; expected to be low frequency compared to heartbeats.
- Migrations must handle existing databases; `ALTER TABLE ... IF NOT EXISTS` mitigates this.
- Stored checksum could become stale if external writers bypass recovery service; mitigate by routing changes through recovery service or validating on demand.

# Compatibility & Acceptance Criteria

- Existing workers default to `state_version = 1` and empty checksum on upgrade.
- Version only increments when all state changes apply successfully.
- Heartbeat checksum comparisons use stored checksum, not full-table scans.
- Restart retains the last known version and checksum.

# References

- `tensorcast/global_store/services/recovery_service.py`
- `schema.sql`
