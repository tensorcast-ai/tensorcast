---
slug: ha-replica-visibility-and-retire
title: HA Replica Visibility and Safe Retire Flow (Plan)
links:
  design: ../designs/0048-ha-replica-visibility-and-retire.md
---

# Objective

Implement publish-state-backed HA inventory and a safe retire pipeline so state sync only advertises resident, publishable replicas and GS-triggered removals never unload in-use replicas. This plan assumes the heartbeat/sync decoupling (0046) and persistent state versioning (0047) are already in place or landing first.

# Current State & Grounding

- `daemon/worker_lifecycle_manager.cc:perform_state_sync` builds `WorkerLocalState` from `engine_->get_all_replicas_info()` and maps non-resident entries to DISK; `compute_state_checksum` uses the same list.
- `daemon/worker_lifecycle_manager.cc:heartbeat_loop` sends `registered_ids` derived from `get_all_replicas_info()`, which feeds `tensorcast/global_store/services/recovery_service.py:get_obsolete_artifacts`.
- `core/store/runtime/replica/replica_runtime.cc:get_all_replicas_info` iterates registry keys even when CPU/GPU states are NONE; there is no publish state tracking.
- `core/store/runtime/metadata/metadata_gateway.cc:handle_ingestion_result` registers replicas on success but does not record publish state or retry when GS is disconnected.
- `tensorcast/global_store/services/recovery_service.py:_compute_state_changes` diffs local vs global replicas and issues `CHANGE_TYPE_REMOVE_REPLICA`; the daemon currently unloads immediately in `apply_obsolete_replicas` and `perform_state_sync`.
- Safety gates exist in `daemon/sweep_tasks.h:EvictionTask` using `RefTracker`, `SessionLifecycleManager::use_count_for`, and `SessionLifecycleManager::placement_pin_count_for`; `TransportLockManager` has no per-replica query.
- 0046/0047 provide a background sync loop and worker `state_version`/`state_checksum`, so retire processing belongs in `state_sync_loop`, not heartbeat.

# Phases & Milestones

- [x] Phase 1: Publishable resident inventory
  - [x] Milestone 1: Track local publish state in core runtime and update it from ingestion + GS registration outcomes.
  - [x] Milestone 2: Emit an HA inventory snapshot (publishable + resident only) and consume it in heartbeat/sync.

- [x] Phase 2: Safe retire pipeline
  - [x] Milestone 1: Route `REMOVE_REPLICA` and full-sync diffs into a retire queue (no immediate unload).
  - [x] Milestone 2: Process retire queue with safety gates and best-effort remote access disable.
  - [x] Milestone 3: Treat `obsolete_replicas` as diagnostic-only; no direct unloads.

- [x] Phase 3: Validation + doc updates
  - [x] Milestone 1: Add unit tests for inventory filtering, publish state transitions, and retire gating.
  - [x] Milestone 2: Update HA and module docs to reflect the new semantics.

# Implementation Tasks

## Phase 1: Publishable resident inventory

- Add `ReplicaPublishState` and per-replica storage in `core/store/runtime/replica/replica_runtime.{h,cc}` (keyed by `loading::ReplicaKey` with `ReplicaKeyHash`).
- Extend `ReplicaRuntime` with:
  - `set_replica_publish_state(...)` / `get_replica_publish_state(...)`.
  - `get_ha_inventory()` returning a lightweight `ReplicaInventoryEntry` (artifact_id, device_key, size_bytes, memory_type, is_available, publish_state, remote_memory_keys, buffer_sizes).
- Update ingestion/registration paths to advance publish state:
  - `ReplicaRuntime::record_ingestion_result` marks `LOCAL_ONLY` when `publish_to_global_store=false`, `PUBLISH_PENDING` on successful resident ingestion.
  - `core/store/runtime/metadata/metadata_gateway.cc:handle_ingestion_result` marks `PUBLISHED` on successful registration; if registration fails or GS disconnected, keep `PUBLISH_PENDING` for sync to reconcile.
  - `core/store/materialization/control/materialize_orchestrator.cc` or `core/store/runtime/ingestion/materialization_facade.cc` already carries `IngestionResultEvent.publish_to_global_store`; reuse it without new flags.
- Surface HA inventory through `StoreEngine` (e.g., `StoreEngine::get_ha_inventory()` forwarding to `ReplicaRuntime`) so daemon code does not reach into runtime internals.

## Phase 1: Worker lifecycle integration

- Update `daemon/worker_lifecycle_manager.cc`:
  - Build `registered_ids` from `get_ha_inventory()` (only publishable artifacts).
  - Build `WorkerLocalState.local_replicas` from `get_ha_inventory()`; never map non-resident entries to DISK.
  - Update `compute_state_checksum` to accept inventory entries or filter `ReplicaInfo` by residency/publishability (align with GS checksum semantics).
- Ensure publish-state changes can trigger `request_state_sync()` (e.g., via a callback in `StoreDaemonServiceImpl` that listens for `RuntimeEventType::kReplicaLoaded` and queries publish state).

## Phase 2: Safe retire pipeline

- Replace direct unloads with retire transitions:
  - `perform_state_sync` handling of `CHANGE_TYPE_REMOVE_REPLICA` enqueues retire instead of `unload_replica_keys`.
  - `apply_full_state` computes obsolete selectors using HA inventory and enqueues retire instead of unloading.
  - `apply_obsolete_replicas` becomes diagnostic-only: log + `request_state_sync()`; no unload.
- Add retire queue data structures to `WorkerLifecycleManager` (e.g., `pending_retire_keys_`, `retire_pending_`), processed inside `state_sync_loop` (keeps heartbeat lightweight per 0046).
- Implement gating checks using existing modules:
  - `RefTracker::ref_count` and `SessionLifecycleManager::{use_count_for,placement_pin_count_for}` must be zero.
  - Add `TransportLockManager::has_lock_for_key(const ReplicaKey&)` (or equivalent) to check active transport locks without iterating tokens in callers.
- On retire enqueue:
  - Mark publish state `RETIRING`.
  - Best-effort `engine_->disable_remote_replica_access(...)` for GPU replicas to stop routing while data remains resident.
- On retire processing:
  - If all gates pass, unload with `engine_->unload_replica` (use `async_runtime->blocking_executor()` if unload may block).
  - If any gate fails, keep in queue and recheck next cycle; log ref/lease/lock counts on sustained stalls.

## Phase 3: Validation + docs

- Tests (C++):
  - `daemon/worker_lifecycle_manager_sync_test.cc`: verify `obsolete_replicas` no longer unloads and REMOVE/full-sync diffs only mark retire.
  - `daemon/worker_lifecycle_manager_sync_test.cc`: assert retire gating honors ref/lock blockers and releases after gates clear.
  - `core/store/runtime/replica/replica_runtime_test.cc`: validate `get_ha_inventory` filters non-resident and publish state transitions.
  - `core/store/runtime/metadata/metadata_gateway_test.cc`: verify registration failures leave `PUBLISH_PENDING`.
- Docs:
  - Update `docs/architecture/high-availability-design.md`.
  - Update module docs: `daemon/README.md`, `core/store/README.md` (and any linked sub-docs if behavior changes are described).

# Acceptance Checks

- HA inventory advertises only resident CPU/GPU replicas and only when publishable.
- `obsolete_replicas` never triggers unload; sync is requested instead.
- `CHANGE_TYPE_REMOVE_REPLICA` and full-sync diffs only unload after ref/lease/lock gates clear.
- `state_version`/`state_checksum` remain consistent with publishable inventory (per 0047).

# Test / Rollout / Backout

- Tests
  - `bazel test //daemon:worker_lifecycle_manager_sync_test`
  - `bazel test //daemon:eviction_task_behavior_test`
  - `bazel test //core/store/runtime/replica:replica_runtime_test`
  - `bazel test //core/store/runtime/metadata:metadata_gateway_test`
- Rollout
  - Land Phase 1 first (inventory/publish state), then Phase 2 (retire pipeline) once tests pass; no new config flags.
  - Validate by forcing GS restart and ensuring replicas are not unloaded while still referenced.
- Backout
  - Revert inventory filtering and retire queue changes; keep 0046/0047 intact.

# Risks & Tracking

- Risk: publish state regressions hide replicas from GS. Mitigation: log publish state transitions and expose counters for PUBLISH_PENDING/PUBLISHED/RETIRING.
- Risk: retire queue backlog if refs never drop. Mitigation: emit warnings with ref/lease/lock counts and add manual operator tooling if needed.
- Risk: checksum mismatch after filtering. Mitigation: compute checksum from the same inventory used for sync/heartbeat.

# Owner Checklist

- [x] Code changes mapped to modules and tests above
- [x] HA docs and module READMEs updated per doc sync rule
- [x] No new configs introduced outside unified config design (0004)
- [x] C++ naming conventions verified for new APIs

# Status

- Completed Phases 1–3; HA inventory now publish-state backed and retire queue enforces ref/use/pin/lock gates.
- Tests added/updated for publish-state transitions, HA inventory filtering, retire gating, and obsolete heartbeat behavior.
- Tests: `bazel test //daemon:worker_lifecycle_manager_sync_test`, `bazel test //daemon:eviction_task_behavior_test`, `bazel test //core/store/runtime/replica:replica_runtime_test`, `bazel test //core/store/runtime/metadata:metadata_gateway_test` (pass).
