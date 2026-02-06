---
slug: standalone-daemon-disk-hints
title: Standalone Daemon Disk Imports (Tokenless) And Removal Of hints.disk_path (Plan)
areas: ["daemon", "core", "sdk"]
related_code:
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_controller.cc
  - daemon/state/daemon_kernel.{h,cc}
  - daemon/util/path_utils.{h,cc}
  - daemon/util/grpc_peer_utils.{h,cc}
  - core/store/materialization/contracts/loading_spec.h
  - core/store/materialization/contracts/materialization_request.{h,cc}
  - core/store/materialization/control/materialize_orchestrator.{h,cc}
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/materialization/runtime/pipeline/source_adapter.cc
  - tensorcast/api/store/__init__.py
  - tests/python/**
  - daemon/service/**_test.cc
links:
  design: ../designs/0073-standalone-daemon-disk-hints.md
---

# Objective

Make `tc.from_disk(...)` usable without Global Store by storing a daemon-local identity→disk-location binding at import time, and remove `hints.disk_path` from core to enforce strict identity/location separation.

# Current State & Grounding

- `ResolveArtifactFromDisk` validates a disk directory but can return a path-like `artifact_id` when the descriptor is missing:
  - `daemon/service/controllers/materialization_controller.cc` (`resolve_artifact_from_disk`)
- Core request construction can treat disk paths as implicit identities (identity/location conflation):
  - `core/store/materialization/contracts/materialization_request.cc`
- Core materialization requires Global Store connectivity unless `hints.disk_path` is set:
  - `core/store/materialization/control/materialize_orchestrator.cc`
  - `core/store/runtime/ingestion/materialization_facade.cc`
- `normalize_disk_path` enforces “under storage_path” when configured, which is correct for managed disk, but too strict for `from_disk` imports:
  - `daemon/util/path_utils.cc` (`normalize_disk_path`)
- Python `from_disk()` does not retain `disk_path` (good); it only caches `artifact_id` and `canonical_index_bytes`:
  - `tensorcast/api/store/__init__.py` (`Store.from_disk`)

# Phases & Milestones

- [x] Phase 1: Fix disk import identity + path semantics
  - [x] Milestone 1: Add `normalize_disk_import_path` and use it in `ResolveArtifactFromDisk`
  - [x] Milestone 2: `ResolveArtifactFromDisk` always returns `mi2:` (compute missing multihashes; never return path)
  - [x] Milestone 3: Gate `ResolveArtifactFromDisk` to loopback/UDS peers

- [x] Phase 2: Add daemon-local import catalog (always enabled)
  - [x] Milestone 1: Implement `LocalDiskImportCatalog` in daemon state
  - [x] Milestone 2: Insert/update catalog on `ResolveArtifactFromDisk`
  - [x] Milestone 3: On materialization, consult catalog when GS is disconnected (loopback only)

- [x] Phase 3: Core refactor (remove `hints.disk_path`)
  - [x] Milestone 1: Delete `MaterializeHints.disk_path` and all call sites
  - [x] Milestone 2: Stop deriving `canonical_artifact_id` from any path
  - [x] Milestone 3: Thread a typed `DiskSource` from daemon to core (`materialize_replica` and `materialize_into_target`)

- [x] Phase 4: Tests + docs cleanup
  - [x] Milestone 1: Bazel test: `from_disk` then materialize with GS disabled succeeds (standalone)
  - [x] Milestone 2: Bazel negative test: non-loopback peer cannot import or use standalone disk
  - [x] Milestone 3: Python test: `tc.from_disk(...)` then `tensor_dict(...)` works with GS disabled
  - [x] Milestone 4: Update `core/store/README.md` to remove `hints.disk_path` semantics and reflect typed disk sources

## Latest Status (2026-02-06)

- 0073 implementation is complete across daemon/core/sdk paths, including standalone disk-import materialization without Global Store.
- Final standalone fix landed in core runtime ingestion: AUTO materialization now directly uses disk ingestion when Global Store is unavailable and a disk source is present.
- Validation run:
  - `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //daemon:resolve_artifact_from_disk_test //core/store/runtime/ingestion:materialization_service_test //daemon:materialize_into_target_validation_test //daemon:materialize_into_mapped_target_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `source .venv/bin/activate && TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/test_shared_storage.py -q`

# Tasks

## Phase 1 tasks

- Add `normalize_disk_import_path(...)` in `daemon/util/path_utils.{h,cc}`:
  - relative paths require `storage_path` and resolve under it
  - absolute paths are allowed even if `storage_path` is set
  - canonicalize with lexical fallback (match `normalize_disk_path` robustness)

- Update `MaterializationController::resolve_artifact_from_disk`:
  - enforce loopback/UDS peer gating
  - ensure `tensor_index.json` exists (safetensors backfill as today)
  - compute `index_multihash` from canonical index bytes (`common::compute_index_multihash`)
  - compute `data_multihash` from disk dir (`store::loader::compute_data_multihash_from_disk_dir`)
  - build `mi2:<index>:<data>` and set `resp.artifact_id`
  - optionally backfill `artifact_descriptor.json`

## Phase 2 tasks

- Implement `LocalDiskImportCatalog`:
  - store `artifact_id -> normalized_disk_path` (+ optional metadata)
  - add minimal thread-safety (`absl::Mutex` or similar) appropriate for controller usage

- Wire catalog into the daemon kernel/controller:
  - update catalog on `ResolveArtifactFromDisk`
  - in `MaterializeReplica` / `MaterializeIntoTarget`, when Global Store is disconnected:
    - if loopback/UDS peer: look up `artifact_id` in catalog and use disk as the source
    - otherwise: return `FAILED_PRECONDITION`

## Phase 3 tasks

- Remove `MaterializeHints.disk_path` from `core/store/materialization/contracts/loading_spec.h`
- Update:
  - `MaterializationRequest::Create` to require a real identifier (mi2/cgid)
  - `MaterializeOrchestrator` and `MaterializationFacade` to accept an optional `DiskSource` rather than reading a string from hints
  - any pipeline adapters that currently read `hints.disk_path`

## Phase 4 tasks

- Tests
  - C++: add a focused daemon/controller test that:
    1) imports a prepared artifact dir via `ResolveArtifactFromDisk`
    2) simulates Global Store disconnected
    3) materializes via `MaterializeReplica` and verifies disk source success
  - Negative: simulate non-loopback peer and assert disk import / standalone disk use is rejected
  - Python: add a minimal test around `Store.from_disk` and `Artifact.tensor_dict` without starting Global Store

- Docs
  - Update `core/store/README.md` to remove `hints.disk_path` terminology and describe typed disk sources / daemon-driven disk authority.

# Test / Rollout / Backout

## Tests

- C++:
  - `bazel test //daemon:...`
  - `bazel test //core/store:...` (targets touched by refactor)
- Python:
  - `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/...`

## Rollout

- Pre-launch: land as a coordinated breaking refactor (daemon + core) in one change set.

## Backout

- Revert by restoring `hints.disk_path` plumbing and the old import behavior (not recommended once tests are updated).

# Risks & Tracking

- Hidden dependencies on `hints.disk_path` across core: mitigate by refactoring in one sweep and adding compilation-breaking removals early.
- Security regressions: add explicit non-loopback negative tests (do not rely on `peer=="unknown"` shortcuts).
