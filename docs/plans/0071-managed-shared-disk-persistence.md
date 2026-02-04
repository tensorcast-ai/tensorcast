---
slug: managed-shared-disk-persistence
title: Plan - Managed Shared-Disk Persistence (Async) And Internal Disk Fallback
links:
  design: ../designs/0071-managed-shared-disk-persistence.md
areas: ["core", "daemon", "global_store", "sdk"]
related_code:
  - schema.sql
  - proto/tensorcast/global_store/v1/global_store.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/state/background_scheduler.h
  - daemon/state/persistence_manager.cc
  - daemon/service/controllers/materialization_controller.cc
  - core/store/materialization/dataplane/loaders/disk_loader.cc
  - core/store/materialization/dataplane/metadata/disk_dir_hash.cc
  - core/checkpoint/tensor_writer.cc
  - tensorcast/global_store/db_utils.py
  - tensorcast/global_store/services/worker_service.py
  - tensorcast/global_store/repositories/replica_repository.py
  - tensorcast/global_store/services/artifact_service.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/types.py
---

# Objective

Deliver a real asynchronous shared-disk persistence pipeline and make disk fallback fully internal (no user-supplied disk paths), while:
- Using a single disk-location mechanism: `artifact_disk_locations` (new Global Store entity).
- Allowing multi-part disk data files (`tensor.data_<n>`) aligned with persistence shards.
- Removing `key_mappings.disk_path` entirely.
- Keeping `from_disk` as an explicit import/registration API (no fallback path injection).

# Status (2026-02-04)

- Implemented the end-to-end managed shared-disk persistence flow, disk location discovery, internal disk fallback, and SDK surface cleanup.
- Implemented managed shared-disk GC on `DeregisterArtifact` (tombstone disk locations + delete bytes by default; opt out via `keep_shared_disk_copy=true`).
- Outstanding work is test coverage: cluster_id stability + key-mapping behavior (Phase 1/M4), disk-location RPC tests (Phase 2/M3), multi-shard writer test (Phase 4/M4), and disk-fallback integration (Phase 6/M3).
- The persistence writer uses the daemon blocking executor directly; a `pump_ranges` + partition sink implementation is still optional (Phase 4/M2).
- Phase 8 complete: added explicit retrieval wait `wait_for_shared_disk_ms` (daemon+SDK) so callers can block until managed shared-disk is ready without delaying publication, plus Bazel/PyTest coverage.

# Current State & Grounding

- Previously: persistence was simulated in `daemon/state/persistence_manager.cc`; now disk writes are real and registered via `artifact_disk_locations`.
- Previously: disk fallback required explicit disk paths; now the daemon resolves managed disk locations internally.
- Disk reading still relies on the same on-disk format (`artifact_descriptor.json` + `tensor_index.(json|cbor)`), which remains valid.
- Daemon background work still runs on the single-threaded scheduler, so disk IO must stay on blocking executors.
- Schema changes now re-apply on startup with targeted migrations (legacy `key_mappings.disk_path` drop).
- Multi-part partition ordering is now numeric to avoid `tensor.data_10` vs `tensor.data_2` misordering.
- Gap: a key/artifact can be published before shared-disk persistence completes. Retrieval defaults to fail-fast, but callers can opt into waiting via `wait_for_shared_disk_ms`.

# Phases & Milestones

- [ ] Phase 1: Global Store schema and migrations (cluster_id + disk locations + key mapping cleanup)
  - [x] Milestone 1: Add `cluster_info` and `artifact_disk_locations` to `schema.sql`
    - Add `cluster_info` single-row table holding `cluster_id` + `created_at`.
    - Add `artifact_disk_locations` table as the single authoritative disk-location mechanism.
    - Remove `key_mappings.disk_path` column from the schema.
  - [x] Milestone 2: Make schema changes actually apply on existing DB files
    - Decide and implement one of:
      - Re-apply idempotent `schema.sql` on every startup (recommended since schema uses `CREATE ... IF NOT EXISTS`), or
      - Add a migration runner (and wire it into startup) that applies ordered migrations from `tensorcast/global_store/migrations/`.
  - [x] Milestone 3: Update Global Store repositories/models for key_mappings without disk_path
    - Remove `disk_path` from:
      - `KeyMappingRepository` (SQL, return dict)
      - key mapping proto messages (`UpsertKeyMappingRequest`, `ResolveKeyMappingResponse`)
      - any call sites (daemon + SDK)
  - [ ] Milestone 4: Tests
    - Unit/integration tests for:
      - `cluster_id` stability across restarts (when using a persisted db_file)
      - `key_mappings` swap/resolve behavior unchanged except disk_path removed

- [ ] Phase 2: Global Store RPCs for cluster identity and disk locations
  - [x] Milestone 1: Expose `cluster_id` to daemons
    - Decision: add `cluster_id` field to `GetServerInfoResponse` (Global Store).
    - Proto hygiene:
      - Run `bazel run @rules_buf_toolchains//:buf -- format ./proto -w`
      - Run `bash tools/build_proto_python.sh`
  - [x] Milestone 2: Add disk location RPCs
    - Add proto + server + repository methods for:
      - Upsert: add/update a disk location for `(artifact_id, relative_path, kind)` (v1: kind is always `MANAGED`)
      - List: list disk locations for `artifact_id`
      - Tombstone via soft-delete fields on the same upsert/list RPCs:
        - `UpsertArtifactDiskLocationRequest.is_deleted` (sticky)
        - `ListArtifactDiskLocationsRequest.include_deleted`
    - Add C++ GlobalStoreClient helpers used by the daemon.
  - [ ] Milestone 3: Tests
    - Global Store test: upsert then list returns deterministic ordering and expected fields.

- [x] Phase 3: Core correctness for multi-part disk directories (required)
  - [x] Milestone 1: Fix partition ordering to be numeric
    - Update `DiskLoader` to sort `tensor.data_<n>` by numeric suffix, not lexicographic filename.
    - Update `compute_data_multihash_from_disk_dir` similarly.
    - Keep legacy support for `tensor.data` and `.safetensors`.
  - [x] Milestone 2: Add tests for >=10 partitions
    - Construct a disk dir with `tensor.data_0..tensor.data_12` and verify:
      - DiskLoader reads a byte stream equal to numeric concatenation order.
      - Disk-dir hash matches the expected hash computed from numeric concatenation.

- [ ] Phase 4: Core/daemon disk persistence writer (multi-part, data plane)
  - [x] Milestone 1: Implement a disk persistence writer that supports multi-part
    - Inputs include shard byte ranges (start/length) and output directory.
    - Output:
      - Writes `tensor.data_<shard_idx>` for each shard.
      - Writes `tensor_index.json`.
      - Writes `artifact_descriptor.json` last (commit marker).
      - Optionally writes `verification.json`.
  - [ ] Milestone 2: Reuse pump infrastructure with a partition sink
    - Add `FilePartitionSink` (or equivalent) and reuse `pump_ranges` to bound memory and avoid blocking.
  - [x] Milestone 3: Atomic/Crash-safe semantics
    - Ensure partial directories are not considered loadable:
      - Decision: write `artifact_descriptor.json` last and require descriptor for managed loads.
  - [ ] Milestone 4: Tests
    - Bazel test writes >=10 shard parts, then loads via DiskLoader and validates:
      - artifact size, descriptor presence, and content hash identity.

- [x] Phase 5: Daemon persistence manager integration (real shared disk + disk locations)
  - [x] Milestone 1: Pass `storage_path` + `cluster_id` into `PersistenceManager`
    - On daemon startup, fetch `cluster_id` from Global Store and cache it.
    - Require `server.storage_path` when shared disk is requested.
  - [x] Milestone 2: Keep persistence key-aware for friendly naming (no key-mapping disk_path)
    - Extend `StartPersistenceRequest` to carry `key_hint` (optional).
    - Daemon uses `key_hint` to create `by_key/<key>/<timestamp>_<short_artifact>/` symlink only.
  - [x] Milestone 3: Replace shared-disk simulation with real writes
    - For each shard disk target:
      - resolve persistence source (stable DRAM replica)
      - write shard part `tensor.data_<shard_idx>` into `<storage_path>/clusters/<cluster_id>/objects/<artifact_id>/...`
      - keep the shard disk target in `COPYING` until the artifact directory is committed
    - Finalize commit (artifact-scoped, once):
      - validate all expected `tensor.data_<n>` exist and sizes match the plan
      - write `artifact_descriptor.json` last (commit marker)
      - upsert `artifact_disk_locations`
      - only then mark all shard disk targets `COMPLETE` (so spill gating and durability index are not satisfied early)
  - [x] Milestone 4: Progress + error reporting
    - Track bytes written and set shard/task progress.
    - MUST failures fail task; SHOULD failures degrade.
  - [x] Milestone 5: Tests
    - Extend `daemon/state/persistence_manager_test.cc` to assert that:
      - disk directories are created with expected part files
      - Global Store receives `artifact_disk_locations` upsert

- [x] Phase 6: Internal disk fallback in materialization (no user paths)
  - [x] Milestone 1: Remove user disk-path fallback request parameters
    - Update daemon protos and SDK call sites to stop sending disk paths for materialization fallbacks.
    - Keep `ResolveArtifactFromDiskRequest.disk_path` for explicit import only.
  - [x] Milestone 2: Resolve disk path internally via `artifact_disk_locations`
    - In `MaterializeByKey`:
      - resolve key -> artifact_id (no disk path in key mapping)
      - when disk allowed, query disk locations and pick one; set `hints.disk_path`
    - In `MaterializeReplica` by artifact_id:
      - when disk allowed, query disk locations and set `hints.disk_path`
  - [ ] Milestone 3: Tests
    - Integration-style test: persist to shared disk, then materialize with disk allowed and with P2P disabled; ensure disk is used.

- [x] Phase 7: SDK surface cleanup (breaking) + from_disk refactor
  - [x] Milestone 1: Remove disk_path as a retrieval fallback surface
    - Remove `FallbackOptions.for_disk` and `FallbackOptions.disk_path`.
    - Remove `Store.artifact(disk_path=...)` / `tc.artifact(disk_path=...)`.
  - [x] Milestone 2: Keep `from_disk` as import/registration only
    - Redefine `from_disk(path)` to:
      - call daemon `ResolveArtifactFromDisk` to resolve artifact_id + canonical index (optional verify)
      - optionally upsert a key mapping (if the user also provides a key)
      - return an `Artifact` referencing `artifact_id` (no disk_path fallback injection)
  - [x] Milestone 3: Update docs and examples
    - Instruct users to use disk policy/allow_disk rather than passing paths.
    - Document from_disk as import/registration (not fallback).
  - [x] Milestone 4: Python tests
    - Update `tests/python/...` to match the new fallback API and from_disk semantics.

- [x] Phase 8: Explicit retrieval wait for managed shared-disk (`wait_for_shared_disk_ms`)
  - [x] Milestone 1: Proto surface (daemon v2)
    - Add `wait_for_shared_disk_ms` to:
      - `MaterializeReplicaRequest`
      - `MaterializeByKeyRequest`
    - Run `bash tools/build_proto_python.sh`.
  - [x] Milestone 2: Daemon behavior (fail then wait then disk-only retry)
    - In `MaterializationController::materialize_replica` and `materialize_by_key`:
      - attempt normal materialize first (P2P/LIP/local as usual)
      - on failure and when `wait_for_shared_disk_ms > 0`, wait for `artifact_disk_locations` to contain a location with:
        - `cluster_id == local cluster_id`
        - `kind == DISK_LOCATION_KIND_MANAGED`
      - once ready, retry materialization from disk with `prefer=disk` and `allow_p2p=false`
      - if disk load fails after ready, fail the request (no P2P fallback)
    - Add metrics + backoff to avoid Global Store polling storms.
  - [x] Milestone 3: SDK surface + semantics
    - Add `GetArtifactOptions.wait_for_shared_disk_ms` (default 0).
    - Expose `options: GetArtifactOptions | None = None` on:
      - `Artifact.tensor_dict`, `Artifact.tensor`, `Artifact.tensor_dict_into`, `Artifact.tensor_into`
      - keep `Store.get(..., options=...)` unchanged
    - Implement "after one normal materialize failure" semantics:
      - first attempt is unchanged (uses existing retry policy)
      - if it still fails and `wait_for_shared_disk_ms > 0`, issue a follow-up request that includes the wait budget
      - set timeouts for the follow-up request to include the wait budget (+ small buffer)
  - [x] Milestone 4: Tests
    - Bazel: simulate "disk location appears later" and verify wait succeeds; verify "disk ready but disk read fails" fails.
    - Pytest: option validation + `Artifact.tensor_dict(..., options=...)` passthrough + timeout behavior.

# Acceptance Checks

- `policy="durable"`:
  - persistence task reaches SUCCESS
  - disk replica directory exists under `server.storage_path/clusters/<cluster_id>/objects/<artifact_id>/`
  - Global Store contains an `artifact_disk_locations` entry for `artifact_id`
- Get/materialize:
  - with disk allowed, daemons can load from disk without any user path parameters
  - with `prefer="local"` (disk disallowed), disk is not used even if present
- Get/materialize with wait:
  - if `wait_for_shared_disk_ms > 0` and managed shared-disk becomes ready within the budget, retrieval can succeed after an initial failure
  - if disk is ready but disk loading fails, the retrieval fails (no P2P fallback)
- Failure modes:
  - missing `server.storage_path` causes `shared_disk=MUST` persistence to fail fast with a clear error

# Rollout / Backout

- Rollout:
  - Deploy Global Store schema + migrations + RPC first (Phase 1–2).
  - Deploy daemons with cluster-id + persistence writes + disk location upserts next (Phase 5–6).
  - Deploy SDK breaking changes last (Phase 7).
- Backout:
  - If disk persistence causes instability, disable `shared_disk` requirements in policy (operationally) and rely on local stable/p2p until fixed.
  - Preserve the on-disk directories; they are append-only and safe to ignore if not referenced by Global Store.

# Risks & Tracking

- Risk: `wait_for_shared_disk_ms` can create Global Store polling storms under load; mitigate with exponential backoff,
  jitter, bounded budgets, and metrics to observe and tune behavior.
- Risk: SDK retry loops can multiply server-side waits; mitigate by implementing "normal attempt first, then a single
  follow-up wait attempt" semantics in the SDK.
- Risk: "disk ready but disk read fails -> fail" can surprise users; mitigate with clear errors and an operator playbook
  (re-run persistence, verify shared filesystem health, and inspect committed directories).

# Owner Checklist

- [x] Global Store: schema/migrations + RPCs for cluster_id + disk locations + key mapping cleanup
- [x] Daemon: cluster_id wiring + persistence write path + disk location upsert + internal disk resolution
- [ ] Core: multi-part ordering fix + disk writer + tests (writer test pending)
- [x] SDK: remove disk-path fallback APIs + from_disk import refactor + update docs/tests
- [x] Daemon+SDK: explicit retrieval wait (`wait_for_shared_disk_ms`) + tests
