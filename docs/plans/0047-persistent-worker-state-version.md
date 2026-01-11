---
slug: persistent-worker-state-version
title: Persistent Worker State Versioning and Checksums (Plan)
areas: ["global_store", "daemon"]
related_code:
  - tensorcast/global_store/services/recovery_service.py
  - tensorcast/global_store/repositories/worker_repository.py
  - tensorcast/global_store/models/worker.py
  - tensorcast/global_store/grpc_service.py
  - schema.sql
  - tensorcast/global_store/init.sql
  - tests/python/global_store/test_recovery_service_checksum.py
  - tests/python/global_store/test_grpc_service.py
links:
  design: ../designs/0047-persistent-worker-state-version.md
---

# Objective

Persist worker state version and checksum in the Global Store database and make state synchronization transactional so versions cannot advance on partial failures and heartbeats no longer scan all replicas every time.

# Current State & Grounding

- `RecoveryService` keeps `worker_state_versions` in memory and clears it on recovery.
  - `tensorcast/global_store/services/recovery_service.py`
- `_apply_state_changes` logs and continues on failure, then `synchronize_worker_state` bumps the version.
  - `tensorcast/global_store/services/recovery_service.py`
- Heartbeat checksum validation recomputes checksum by scanning all replicas.
  - `tensorcast/global_store/services/recovery_service.py`
- Canonical schema is in `schema.sql`; `init_db` re-applies it at startup.
  - `tensorcast/global_store/db_utils.py`

# Phases & Milestones

- [ ] Phase 1: Schema and model updates
  - [ ] Milestone: add `state_version` and `state_checksum` columns to `schema.sql` (with `ALTER TABLE ... IF NOT EXISTS`).
  - [ ] Milestone: keep `tensorcast/global_store/init.sql` in sync.
  - [ ] Milestone: extend `Worker` model and `WorkerRepository` SELECTs to include the new fields.

- [ ] Phase 2: Transactional sync and checksum caching
  - [ ] Milestone: update `RecoveryService` to read/write version and checksum via `WorkerRepository`.
  - [ ] Milestone: make `_apply_state_changes` transactional and return failure on any error.
  - [ ] Milestone: update checksum handling to read stored values and persist on miss.

- [ ] Phase 3: Tests and documentation
  - [ ] Milestone: update tests that assume in-memory versions.
  - [ ] Milestone: add regression coverage for partial failure (no version bump).
  - [ ] Milestone: update `tensorcast/global_store/README.md` with new state version semantics.

# Tasks

- Add `state_version` and `state_checksum` columns in `schema.sql` and `tensorcast/global_store/init.sql`.
- Update `Worker` dataclass to include `state_version` and `state_checksum`.
- Update `WorkerRepository._row_to_model` and `_WORKER_SELECT` to read new columns.
- Add repository helpers for reading and updating state version and checksum.
- Update `RecoveryService.ensure_worker_state_version` to consult the DB (cache optional).
- Update `synchronize_worker_state` to run changes in a transaction and only bump version on success.
- Update `get_worker_state_checksum` to use the stored checksum and backfill when empty.
- Update `grpc_service.py` to use the new recovery service APIs if signatures change.

# Test / Rollout / Backout

- Tests (Python):
  - `uv run pytest tests/python/global_store/test_recovery_service_checksum.py`
  - `uv run pytest tests/python/global_store/test_grpc_service.py`
- Rollout:
  - Startup re-applies `schema.sql`; new columns are added with `ALTER TABLE ... IF NOT EXISTS`.
  - Stored checksum is backfilled on first access when empty.
- Backout:
  - Revert recovery service logic to in-memory versions; new columns remain unused.

# Risks & Tracking

- Incorrect migration could leave columns missing in existing DBs; mitigate with explicit `ALTER TABLE` statements in `schema.sql` and startup logging.
- Transaction failures could mask partial updates; mitigate with clear error returns and metrics for sync failures.
