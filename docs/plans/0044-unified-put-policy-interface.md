---
slug: 0044-unified-put-policy-interface
title: Unified Put Policy Interface Test Plan (Plan)
links:
  design: ../designs/0044-unified-put-policy-interface.md
areas: ["sdk", "daemon", "global_store", "proto"]
related_code:
  - docs/designs/0041-distributed-persistence-placement.md
  - docs/designs/0004-unified-runtime-config.md
  - tensorcast/api/store/__init__.py
  - tensorcast/api/_config.py
  - tensorcast/api/_register.py
  - tensorcast/api/store/registration.py
  - daemon/grpc_service_impl.*
  - daemon/persistence_manager.*
  - daemon/service/controllers/registration_controller.cc
  - proto/tensorcast/daemon/v1/store_daemon.proto
  - proto/tensorcast/global_store/v1/global_store.proto
  - tensorcast/global_store/services/placement_service.py
  - tensorcast/global_store/init.sql
  - tensorcast/global_store/*
  - tests/python/README.md
  - tests/python/api/test_config_models.py
  - tests/python/api/test_persistence_wireup.py
  - tests/python/global_store/test_placement_service.py
  - tests/python/global_store/test_placement_persistence.py
  - daemon/persistence_manager_test.cc
---

# Objective

Define the test coverage required to validate the unified `policy` interface for `put`, including profile mapping, daemon-side policy resolution, lowering onto the existing 0041 persistence pipeline, and the stable-DRAM persistence source needed to make `Store.put(...)` persistence work in practice. No backward compatibility is required; after implementation all call sites must use the new interface.

# Current State & Grounding
- SDK API surface: `RegisterArtifactOptions` exposes `policy` as the single durability/placement control; `Store.put` uses `PlanType.DRAM_STABLE` (`tensorcast/api/_config.py`, `tensorcast/api/store/registration.py`).
- Persistence wire-up: the SDK calls `start_persistence(...)` post-commit when the resolved `StorePolicy` requires it, forwarding the `policy` to the daemon (`tensorcast/api/store/registration.py`).
- Daemon persistence: `StoreDaemonServiceImpl::StartPersistence` resolves `StorePolicy` and delegates to `PersistenceManager::start_task(...)`, which can start from either an active LIP lease or a locally committed stable-DRAM CPU replica (`daemon/grpc_service_impl.cc`, `daemon/persistence_manager.cc`).
- Global Store placement: `PlacementService.plan_placement()` always includes the source node, attempts a remote stable target for replicated/sharded, and degrades to local-only with reason `insufficient_remote_capacity` when none is available (`tensorcast/global_store/services/placement_service.py`).
- Schema constraint: `artifact_placements.policy` is constrained to `local_only|replicated|sharded`; profile names must not be stored there (`tensorcast/global_store/init.sql`).
- Existing tests to adapt:
  - SDK wire-up: `tests/python/api/test_persistence_wireup.py`
  - Global Store placement/degraded behavior: `tests/python/global_store/test_placement_service.py`, `tests/python/global_store/test_placement_persistence.py`
  - Daemon persistence state machine and sharding: `daemon/persistence_manager_test.cc`

# Test Strategy (Contracts to Verify)
- `must/should/may` semantics are evaluated on the persistence task (`QueryPersistenceStatus`), not on the synchronous `put()` return.
- Policy lowering only affects existing 0041 fields (`placement_policy`, `persist_to_shared_disk`); profile strings never leak into Global Store schema.
- “Repo-aligned HA” (`ha` profile) is disk `must` + remote stable `should`; lack of remote capacity must yield `degraded` (not `failed`).
- Explicit strict remote (remote stable in `must`) must yield `failed` when Global Store placement returns a degraded plan.
- Persistence can start from either a LIP lease or a locally-committed stable-DRAM replica (fix the current LIP-only coupling).
- Stable DRAM cache uses `engine.memory_tiers.stable_bytes` as the only capacity knob (no `stable_cache_*` config); eviction is demand-driven on `put` admission pressure (no watermarks/periodic loops).

# Scope & Assumptions
- Breaking change: legacy `persist` and `placement_policy` options are removed (no compatibility tests).
- Storage tiers in scope: stable DRAM and shared disk only.
- Global Store placement policy values remain `local_only|replicated|sharded`; tests must not assume a new DB schema.
- Sharding thresholds follow 0041 (128MB threshold, 64–256MB shard caps) unless explicitly moved to config.
- No stable-cache-specific daemon config fields are added; stable tier eviction behavior must be exercised via admission failures during `put`.

# Phases & Milestones

- [x] Phase 0: Proto + contract invariants
  - [x] Add/replace proto fields to accept `StorePolicy`; remove legacy per-call options.
  - [x] Prove `must/should/may` map to persistence task states (`success/degraded/failed`) via `QueryPersistenceStatus`.
  - [x] Ensure policy metadata is not written into `artifact_placements.policy` (lowered values only).
  - [x] Add a repo check (CI or `tools/lint/*`) that fails if `PutPolicy` appears anywhere under `docs/` or `proto/` to prevent naming drift (single policy surface is `StorePolicy`).

- [x] Phase 1: SDK policy model and wire-up (Python)
  - [x] Policy shape validation in the SDK model (`tests/python/api/test_config_models.py`)
    - [x] Reject `profile` when `must/should/may` are set.
    - [x] Reject `overflow_policy=spill` unless `shared_disk` is requested.
    - [x] Validate `shared_disk` tier constraints (`scope=any`, `min_replicas=1`).
    - [x] Validate `ttl` requires `retention_ttl_ms`.
  - [x] Registration pipeline triggers persistence based on resolved policy (`tests/python/api/test_persistence_wireup.py`)
    - [x] `cache`/`pinned`: do not call `start_persistence(...)`.
    - [x] `durable`/`cold`: call `start_persistence(...)` with disk persistence enabled unless remote stable is requested.
    - [x] `ha`: call `start_persistence(...)` with remote stable + shared disk requirements and tolerate degraded remote placement.
  - [x] Status mapping remains correct (`tests/python/api/test_persistence_wireup.py` Query path).

- [x] Phase 2: Daemon policy resolution + lowering (C++)
  - [x] Unit-test profile expansion and lowering at the daemon boundary (new test colocated with resolver, or extend `daemon/persistence_manager_test.cc`)
    - [x] Profiles lower to `{placement_policy,persist_to_shared_disk}` exactly as specified in the design.
    - [x] `layout=auto` follows 0041 sharding thresholds; `layout=unsharded` and `layout=sharded` override when explicitly set.
    - [x] Explicit `must` remote stable treats a degraded placement plan as task-level `failed`.
    - [x] `should` remote stable treats a degraded placement plan as task-level `degraded`.
  - [x] Policy enforcement matches error model (INVALID_ARGUMENT/FAILED_PRECONDITION/RESOURCE_EXHAUSTED).
  - [x] Hard validation (no silent downgrade) aligned to updated design:
    - [x] SDK (`tensorcast/api/_config.py`): reject retention fields on `shared_disk` tiers; reject `stable_dram.min_replicas != 1`; reject retention fields when `stable_dram.scope=remote`; reject/normalize `must` local stable to pinned-only (if user specifies `best_effort`/`ttl` for a `must` local tier, raise `InvalidPlan` / `ValueError`).
    - [x] Daemon (`daemon/store_policy_resolver.cc`): reject retention fields on `shared_disk` tiers; reject `stable_dram.min_replicas != 1`; reject retention fields when `scope=remote`; enforce `must` local stable ⇒ pinned retention and “never evict” semantics (and ensure contradictory explicit retention is `INVALID_ARGUMENT`).
    - [x] Tests: add explicit cases in `tests/python/api/test_config_models.py` and `daemon/store_policy_resolver_test.cc` to keep SDK/daemon validation behavior identical.

- [x] Phase 3: Persistence source coverage (fix LIP-only coupling) (C++)
  - [x] StartPersistence succeeds for stable-DRAM puts with no LIP lease (exercise production path, not `start_task_for_test`).
  - [x] StartPersistence continues to succeed for LIP-based registrations (existing behavior preserved).
  - [x] When neither source is available, StartPersistence returns FAILED_PRECONDITION with an actionable message.

- [x] Phase 4: Global Store placement + degraded propagation (Python + C++)
  - [x] Global Store plan degradation is covered (`tests/python/global_store/test_placement_service.py`)
    - [x] `replicated` + insufficient capacity yields degraded reason `insufficient_remote_capacity` and only local targets.
  - [x] Daemon consumes plan degraded flags and reports `degraded_reason` in `QueryPersistenceStatus` (`daemon/persistence_manager_test.cc`).
    - [x] Extend `RecordingGlobalStoreClient` to simulate `plan.degraded=true` (needed to avoid depending on worker capacity state).
  - [x] DB rows store only lowered policy values (`tests/python/global_store/test_placement_persistence.py` and `tensorcast/global_store/init.sql` constraint).

- [x] Phase 5: Stable DRAM retention / cache manager (C++)
  - [x] Admission and eviction behavior is covered with deterministic tests (extend `core/store/components/eviction_service_test.cc` or add a dedicated cache manager test target).
    - [x] `overflow_policy=evict` evicts best-effort entries until admission succeeds.
    - [x] `overflow_policy=reject` fails admission with `RESOURCE_EXHAUSTED`.
    - [x] `ttl` entries are not evicted before expiry; `pinned` entries are never evicted.
  - [x] `spill` requires shared disk and fails closed if disk is unavailable.
  - [x] No watermark-driven eviction: eviction occurs only when a new `put` needs space (demand-driven admission path).
  - [x] Redefine `spill` to be durability-gated (no hidden writes), per updated design:
    - [x] Add a queryable durability index/state (daemon-owned) that answers “is this artifact durable enough to allow spill-eviction?” and is updated only by the persistence pipeline:
      - [x] Update on shared-disk completion/dedup-hit (per shard) and on remote target completion (per shard).
      - [x] Track tier-level completion (`shared_disk_complete`, `remote_stable_complete`) and/or expose a derived `spill_evictable(artifact_id)` that incorporates must-tier gating (disk must ⇒ require `shared_disk_complete`; remote must ⇒ require `remote_stable_complete`).
    - [x] Update `core/store/components/stable_dram_cache_manager.{h,cc}` so `overflow_policy=spill` only evicts entries that are durable per the index; if no durable candidates exist, return `RESOURCE_EXHAUSTED` (backpressure) rather than evicting non-durable entries.
    - [x] Update `daemon/persistence_manager.*` to publish durability transitions (artifact becomes durable) to the durability index.
    - [x] Tests:
      - [x] Add `core/store/components/stable_dram_cache_manager_test.cc` cases proving spill evicts only durable entries and fails when none are durable.
      - [x] Add a daemon-level test (extend `daemon/persistence_manager_test.cc`) that drives a task to disk-complete/remote-complete and asserts the durability index flips, enabling spill-eviction.
  - [x] Enforce “`must` is long-lived” for local stable DRAM:
    - [x] Treat `must` local stable as pinned-for-life: never eligible for eviction, and admission must fail (`RESOURCE_EXHAUSTED`) if the stable budget cannot fit the new artifact without violating pinned entries.
    - [x] Tests: add cases covering “must local stable cannot be evicted by spill/evict overflow” and “must admission fails under pressure”.

- [x] TODO: Eventual consistency contract for eviction vs Global Store routing (system-level)
  - [x] Document and test the “stale routing window”: Global Store may briefly still reference an evicted local replica until chunk/state sync updates propagate.
  - [x] Ensure daemon materialization treats placement/routing as a hint and falls back when local replica is missing (e.g., retry remote/disk, and/or emit an explicit degraded reason/metric).
  - [x] Add a deterministic test that simulates “GS says local exists but it was evicted” and verifies `get()` succeeds via fallback (or returns a clearly retryable error) without hangs.

- [ ] Phase 6: End-to-end smoke (optional, if harness exists)
  - [ ] Minimal “put with policy → StartPersistence → QueryPersistenceStatus” flow using fakes/deterministic tick advancement to avoid flakiness.

# Acceptance Checks
- [x] Legacy `persist`/`placement_policy` surfaces removed; all call sites use `policy`.
- [x] Profile mapping matches design (esp. `ha` is disk `must`, remote stable `should`).
- [x] `must/should/may` semantics reflected in persistence task state and `degraded_reason`.
- [x] Persistence starts for stable-DRAM puts (no LIP lease dependency).
- [x] Global Store continues to store only lowered placement policy values.
- [x] Stable DRAM cache introduces no new `stable_cache_*` config knobs; capacity comes from `engine.memory_tiers.stable_bytes` and eviction is demand-driven.

# Test Plan
- Python:
  - `uv run pytest tests/python/api/test_config_models.py`
  - `uv run pytest tests/python/api/test_persistence_wireup.py`
  - `uv run pytest tests/python/global_store/test_placement_service.py`
  - `uv run pytest tests/python/global_store/test_placement_persistence.py`
- C++:
  - `bazel test //daemon:persistence_manager_test`
  - `bazel test //daemon:store_policy_resolver_test`
  - `bazel test //core/store:stable_dram_cache_manager_test`
  - `bazel test //core/store:eviction_service_test`

# Rollout / Backout
- Rollout: land proto + daemon resolution + SDK migration, then run the full test plan.
- Backout: revert to the previous `persist`/`placement_policy` interface (not supported long-term by design).

# Risks & Tracking
- Risk: SDK and daemon policy resolution diverge; mitigate with daemon-side resolution and shared fixtures/golden tests.
- Risk: degraded-vs-failed semantics regress because Global Store uses “degrade, don’t fail” placement; tests must cover both `should` and explicit `must` remote stable.
- Risk: persistence remains LIP-only; Phase 3 is the correctness gate for stable-DRAM `put(...)` persistence.

# Owner Checklist
- [x] Policy tests cover profiles and validation paths.
- [x] Daemon and Global Store tests cover required and degraded outcomes.
- [ ] End-to-end tests validate status and error reporting.
- [x] Stable-DRAM puts can start persistence without a LIP lease.
- [x] Global Store schema continues to store lowered placement policy values only.
