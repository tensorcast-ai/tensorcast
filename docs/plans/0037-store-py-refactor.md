---
slug: 0037-store-py-refactor
title: tensorcast.api.store refactor (Plan)
links:
  design: ../designs/0037-store-py-refactor.md
areas: ["sdk"]
related_code:
  - tensorcast/api/store.py
  - tensorcast/api/_register.py
  - tensorcast/api/_materialize.py
  - tensorcast/api/_view_ops.py
  - tensorcast/store_session_registry.py
---

# Objective

Deliver the modular split proposed in Design 0037 so that `tensorcast/api/store.py` becomes a thin façade over explicit runtime, registration, materialization, view, and async orchestration modules without changing the public API or behavior.

# Current State & Grounding
- `tensorcast/api/store.py` (~3.2K lines) mixes session lifecycle, retry logic, fallback resolution, view building, and async orchestration; sync/async verbs duplicate error mapping and cancellation handling.
- Lease tracking and session-record writes are intertwined with executor bookkeeping and `_ensure_client` (e.g., `_track_future`, `_persist_session_record_locked`), making correctness hard to isolate in tests.
- Fallback and view handling span multiple helpers (`_resolve_key_mapping_cached`, `_resolve_disk_path`, `_materialize`, `_resolve_view_inputs`), with implicit invariants about placement defaults and disk compatibility.
- Async flows (`register_async`, `put_async`, `get_async`, `get_into_async`) each manage their own cancellation hooks, risking drift from the sync code paths.
- Recent designs (0014, 0033, 0035, 0036) expect clearer runtime boundaries and typed view orchestration than the current monolith provides.

# Phases & Milestones

- [x] Phase 1: Types + error/retry primitives
  - [x] Milestone 1: Add `tensorcast/api/store/types.py` for dataclasses/aliases (`RetryPolicy`, `FallbackOptions`, `StoreOptions`, `ArtifactError`, `CanonicalIndex*`, `ReplicaInfo`, `LeaseHandle`, `TensorDict`) and re-export from `tensorcast/api/store/__init__.py` and legacy `store.py`.
  - [x] Milestone 1b: Add `tensorcast/api/store/handles.py` for behavioral wrappers (`RegisteredArtifact`, `RegisteredLease`), keeping `types.py` pure.
  - [x] Milestone 1c: Add shared retry/error/metrics helpers (e.g., `retry.py` or module-level utilities) to be consumed by registration and materialization; port existing mapping logic from `store.py`.
- [x] Phase 2: Runtime context
  - [x] Milestone 2a: Introduce `tensorcast/api/store/runtime.py` as the process-global singleton runtime to own daemon client creation, at-fork refresh, session-record persistence (via `tensorcast/store_session_registry.py`), capability caching, key cache, and process-wide store management; adapt existing helpers to delegate.
- [x] Phase 3: Registration pipeline
  - [x] Milestone 3a: Create `tensorcast/api/store/registration.py` with option normalization, device selection, shared retry/error mapping utilities, and lease tracking; wire sync/async register/put/register_view through it while delegating to `_register._register_artifact_core`.
  - [x] Milestone 3b: Add focused tests for registration retry/cancellation and lease bookkeeping (e.g., new unit targets plus `tests/python/test_register_*.py` coverage).
- [x] Phase 4: Materialization and fallback pipeline
  - [x] Milestone 4a: Build `tensorcast/api/store/materialization.py` + `FallbackResolver` to unify get/get_into/get_view flows, disk fallback, canonical index parsing, target validation; reuse shared retry/error helpers; keep span/metric attribution centralized; call into `_materialize.materialize_artifact` for the core retrieval primitive.
  - [x] Milestone 4b: Cover disk/P2P fallback, key-cache TTL, and view placement defaults with tests (`tests/python/test_store_view_api.py`, new unit cases for fallback resolution and `_validate_targets`).
- [x] Phase 5: View orchestration
  - [x] Milestone 5a: Add `tensorcast/api/store/views.py` for view parsing/placement/plan metadata; ensure it depends only on types/runtime and delegates spec building to `_view_ops.py` (no dependency on registration/materialization).
- [x] Phase 6: Async façade cleanup
  - [x] Milestone 6a: Add shared async helpers (`tensorcast/api/store/async_ops.py` with `TrackedExecutor`, cancellation hooks, session-record updates) and route all async verbs through them; design helpers with injectable callbacks for cancel/confirm so they can be unit-tested with fakes.
  - [x] Milestone 6b: Add targeted async tests (unit or fake-daemon) that assert: (1) cancel triggers pipeline abort and releases leases/materialized artifacts; (2) session record pending_futures/active_leases counts are updated on submit/complete/cancel; (3) confirm triggers daemon-side confirm before result delivery.
  - [x] Milestone 6c: Slim `tensorcast/api/store.py` to delegation and maintain module-level helpers (`register`, `get`, etc.) via re-exports in `tensorcast/api/store/__init__.py`; update any affected docs/README notes.

Status: Core refactor and façade wiring are complete. New unit coverage added in `tests/python/test_store_refactor_unit.py` for registration retries, fallback validation, target validation, and async cancellation. Full compatibility sweep executed (`uv run pytest tests/python/test_store_session_api.py tests/python/test_store_view_api.py tests/python/test_register_* tests/python/test_store_region_registration.py`). Remaining follow-up: optional import-DAG lint if desired.

# Tasks
- Document new module ownership in `tensorcast/api` README or inline docstrings where appropriate; add `tensorcast/api/store/__init__.py` re-export notes for compatibility.
- Validate dependency DAG with a lint/pytest check (e.g., import-linter or custom guard): expected ordering is `types -> handles -> runtime`; `views -> (types, runtime)`; `registration -> (types, handles, runtime, views)`; `materialization -> (types, handles, runtime, views)`; `async_ops -> (types, handles, runtime)`; façade depends on pipelines only.
- Keep `tensorcast/store_session_registry.py` as a standalone dependency consumed by `runtime.py` (no inlining); add a note in runtime docs/comments to document the relationship.
- Add lightweight unit tests per pipeline (registration/materialization/runtime/fallback) to reduce reliance on daemon integration tests.
- Ensure OTEL spans/attributes and metrics counters are preserved across the refactor with a shared attribution helper.
- Verify process-store lifecycle (`store()`, `shutdown_process_store()`, `force_recreate`) remains thread/fork-safe after delegation.

# Test / Rollout / Backout
- Targeted tests: `uv run pytest tests/python/test_store_session_api.py tests/python/test_store_view_api.py tests/python/test_register_* tests/python/test_store_region_registration.py`.
- Regression sweep: `uv run pytest tests/python` (time permitting).
- Backout strategy: revert to the pre-refactor `tensorcast/api/store.py` and re-run the above tests; no schema or protocol changes require rollback steps.

# Risks & Tracking
- Risk of behavioral drift in fallback resolution and lease tracking while splitting code; mitigated by unit seams and existing integration tests.
- Potential import cycles between new modules (runtime/view/registration/materialization); enforce acyclic dependencies in design and verify with incremental refactors.
- Async cancellation hooks may miss session-record updates; validate with tests and shared helper coverage.
