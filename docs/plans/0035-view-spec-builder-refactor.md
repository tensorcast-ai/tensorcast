---
slug: 0035-view-spec-builder-refactor
title: ViewSpec Builder Refactoring (Plan)
links:
  design: ../designs/0035-view-spec-builder-refactor.md
areas: ["sdk"]
related_code:
  - tensorcast/api/store.py
  - tensorcast/api/_register.py
  - tests/python/test_store_view_api.py
---

# Objective

Refactor view-spec construction and resolution to use the typed models from the paired design, remove tuple-based returns and legacy compatibility surfaces, and land targeted tests while keeping current SDK APIs and wire semantics unchanged.

# Current State & Grounding
- `_build_view_spec` mixes validation, normalization, and proto assembly, returning a 3-tuple `(ViewSpec | None, bool, MutableNormalizedViewOps)`; error handling is intertwined with protobuf mutation (`tensorcast/api/store.py:1485`).
- `_resolve_view_inputs` returns a 6-tuple and repeats identity/view_id branching plus canonical index fetching (`tensorcast/api/store.py:1766`).
- View registration depends on the legacy `normalized_ops` dict for plan computation and tensor materialization (`tensorcast/api/_register.py:600`).
- Tests assert tuple ordering directly (`tests/python/test_store_view_api.py:77`) and rely on `normalized_ops` shape for registration/get flows.
- Design intent captured in `docs/designs/0035-view-spec-builder-refactor.md` (typed ops, structured results, adapters for legacy consumers).

# Phases & Milestones

- [x] Phase 1: Scaffold typed ops module
  - [x] Milestone 1: Add `_view_ops.py` with `NarrowOp`, `TransposeOp`, `TensorViewOp`, `ViewSpecBuildResult`, `ResolvedViewInputs`, and helpers (`validate_narrow`, `validate_transpose`, `build_view_spec`, `_coerce_slice_spec`).
  - [x] Milestone 2: Add focused unit tests for the helpers and invariants (identity folding, swap-to-place canonicalization, mutual exclusivity).
- [x] Phase 2: Store integration
  - [x] Milestone 1: Refactor `Store._build_view_spec` to delegate to `build_view_spec`, returning structured results while preserving `ArtifactError` wrapping.
  - [x] Milestone 2: Refactor `Store._resolve_view_inputs` to emit `ResolvedViewInputs`; update `get_view`, `get_view_into`, and related helpers to consume properties instead of tuple slots.
  - [x] Milestone 3: Remove tuple/`normalized_ops` compatibility in Store paths; ensure all consumers use structured results directly.
- [x] Phase 3: Registration alignment and legacy removal
  - [x] Milestone 1: Refactor `_compute_view_plan_metadata`, `_materialize_canonical_tensors`, and registration paths to consume typed ops/structured results; delete `normalized_ops` dict surfaces.
  - [x] Milestone 2: Refresh tests in `tests/python/test_store_view_api.py` and add new cases covering mutual exclusivity and legacy removal (assert old tuple/dict paths are gone).
  - [x] Milestone 3: Add doc cross-links or notes in the design/plan index if required by CI.

# Tasks
- Implement `_view_ops.py` per design, including identity singleton handling and deterministic ordering.
- Wire `Store._build_view_spec` and `_resolve_view_inputs` to structured results, keeping `ArtifactError` messages consistent.
- Update call sites in `Store.get_view`, `Store.get_view_into`, and registration flows to consume `ResolvedViewInputs` and `ViewSpecBuildResult`.
- Remove legacy tuple returns and `normalized_ops` dict adapters; ensure registration helpers accept typed ops instead of legacy dicts.
- Extend or add tests for validation helpers and end-to-end Store flows using the new return types; assert legacy surfaces are absent.
- Keep logging/debug output consistent with existing verbosity; add only minimal new debug hooks if needed.
- Update any relevant docs or indexes if required after code changes (no schema or proto updates expected).

# Test / Rollout / Backout
- Unit tests: `uv run pytest tests/python/test_store_view_api.py` plus any new helper-specific test modules.
- Lint/format: `uv run ruff check .` and `uv run ruff format .` on touched files.
- Backout: revert the refactor commit set if needed; do not reintroduce tuple/adapter surfaces—fix forward in the structured path.

# Risks & Tracking
- Regression risk in view placement logic when swapping tuple indices for properties; mitigate with targeted tests on `has_transpose`/placement selection.
- Potential mismatch when converting registration helpers to typed ops; validate via round-trip tests against `_compute_view_plan_metadata`.
- Typed helper adoption may miss edge cases (negative indices, empty transpose); ensure test matrix covers legacy scenarios before rollout.
