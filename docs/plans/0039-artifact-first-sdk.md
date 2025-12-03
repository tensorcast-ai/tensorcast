---
title: Plan – Artifact-First TensorCast SDK API
links:
  design: ../designs/0039-artifact-first-sdk.md
areas: ["sdk"]
related_code:
  - tensorcast/startup.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/registration.py
  - tensorcast/api/_config.py
  - tensorcast/api/store/types.py
  - tensorcast/__init__.py
---

# Summary

Implement the artifact-first SDK surface defined in Design 0039: embed a default daemon config for `tc.init()`, simplify public options/enums, add symmetric tensor materialization helpers, remove KV-specific helpers, and prune legacy eager verbs in favor of handle-centric APIs. Update docs and examples to match, and add tests to lock behavior.

# Current State

- `tensorcast/startup.py`: `init` raises when no daemon config path is provided or discoverable; no embedded defaults.
- `tensorcast/api/store/__init__.py`: legacy eager helpers (`get`, `get_into`, `get_view`, `get_view_into`, async variants) coexist with handle APIs; includes `register_kv_block`.
- `tensorcast/api/store/artifact.py`: lazy `Artifact` with `tensor_dict`, `tensor`, `tensor_dict_into`; no `tensor_into` helper; view derivation is lazy but not explicitly documented in code comments.
- `tensorcast/api/_config.py`, `tensorcast/api/store/types.py`: public `PlanType` and options use internal naming (`vram_coalesced`/`vram_leased`, `prefer` flags).
- Docs/examples still mention legacy eager flows and KV-specific helper.

# Goals

- `tc.init()` always works out-of-box via embedded minimal daemon config when no config is supplied or discoverable.
- Public API surface is handle-first; legacy eager verbs and KV-only helpers removed from public exports.
- Symmetric materialization convenience via `Artifact.tensor_into`.
- User-friendly option strings (`plan="lease" | "copy"`, fallback string shortcuts) with validation mapped to existing enums.
- Clear, lazy semantics and error surfaces documented and covered by tests.
- Repository docs/README/AGENTS reflect the new surface; legacy examples removed.

# Phases & Tasks

## Phase 1: Bootstrap defaults for tc.init
- [x] Embed minimal daemon config in SDK (loopback bind, ephemeral/standard port, default cache path) inside `tensorcast/startup.py` (inline YAML/JSON proto). Ensure cache path is writable (e.g., `tempfile.mkdtemp` fallback) and ports avoid conflicts.
- [x] Update `startup.py:init` to fall back to embedded config when no path is provided and none is discoverable; preserve current resolve/connect code paths.
- [x] Add logging/telemetry markers for “embedded default” path (single INFO, no spam).
- [x] Document behavior in docstrings (`startup.py`) and user-facing docs (`README.md`, module README if present).

## Phase 2: Public option ergonomics
- [x] Accept user-friendly `plan="lease"|"copy"` at public surfaces (`tensorcast/api/store/__init__.py`, `tensorcast/api/store/registration.py`, `tensorcast/api/_config.py`, `tensorcast/api/store/types.py`); coerce to `PlanType` internally; retain enum access for power users.
- [x] Add fallback string shortcut parsing (e.g., `fallback="disk:/tmp/foo"` → `FallbackOptions.for_disk`) in `tensorcast/api/store/types.py` or helper; keep explicit `FallbackOptions`.
- [x] Update `RegisterArtifactOptions`/`GetArtifactOptions` docstrings/type hints to reflect user-friendly inputs; ensure `ArtifactError` messages are actionable on bad inputs.
- [x] Identify and adjust affected tests in `tests/python` that assert enum strings.

## Phase 2.5: Surface audit & deprecation mapping
- [x] Audit imports/exports in `tensorcast/api/__init__.py`, `tensorcast/api/store/__init__.py`, `tensorcast/__init__.py` to align with handle-first surface and removed helpers.
- [x] Search for legacy helpers in code/tests/examples (`rg "register_kv_block"`, `rg "get_view"`, etc.) and build a migration checklist.

## Phase 3: API surface cleanup
- [x] Remove `register_kv_block` from public exports and docs (`tensorcast/api/store/__init__.py`, `tensorcast/__init__.py`, examples/tests); rely on standard `register` with region reuse.
- [x] Remove module-level eager getters (`get`, `get_into`, `get_view`, `get_view_into`, async variants) from public exports (`__all__`, imports). Provide migration note in docs.
- [x] Ensure `tensorcast/__init__.py` mirrors the new export set; fix any star-import expectations in tests/examples.
- [x] Keep `store()` available but de-emphasized in docs/README.

## Phase 4: Artifact conveniences & clarity
- [x] Add `Artifact.tensor_into(name, target_tensor, device=None)` in `tensorcast/api/store/artifact.py` as a thin helper over `tensor_dict_into`; export via `tensorcast/api/store/__init__.py` and `tensorcast/__init__.py`.
- [x] Clarify lazy view chaining in code comments/docstrings within `artifact.py` (no RPC/data until `tensor*`/`tensor_dict*`/`exists()`).
- [x] Clarify error surfaces: `tc.artifact()` construction is non-throwing for `NOT_FOUND`; materialization and `exists()` surface `ArtifactError`. Add short docstring/comment.

## Phase 5: Docs & Examples
- [x] Update design doc 0039 references if needed; add concise user-facing guide snippets in README or module README under `tensorcast/api/store/`.
- [x] Remove legacy eager examples; add handle-first examples (sync/async, view, batch, prefetch, tensor_into).
- [x] Update AGENTS/README where they mention public APIs or init behavior (root README, tests/python/README if applicable).

## Phase 6: Tests & Validation
- [x] Add/adjust tests (likely under `tests/python/api` or nearest suites) for: embedded default init path; `plan` string coercion and error handling; fallback string shortcut; `tensor_into`; lazy view chaining (no RPC mocked); error surfaces (`ArtifactError` on materialization, not on handle construction).
- [ ] Run lint/format/type checks: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright ./tensorcast`.
- [ ] Run relevant Python tests: `uv run pytest tests/python/...` (targeted modules).
  - Test execution pending locally (`uv` not available in current environment); rerun in CI or with `uv` installed.
- [x] Added regression coverage for `tc.artifact` exports, `Artifact.prefetch()` clone semantics, `tensor_into` subset copy, and plan defaults via `tests/python/api/test_public_surface.py`, `tests/python/api/test_artifact_handle.py`, and `tests/python/api/test_config_models.py`.

# Phase 7: Test suite cleanup for interface removal
- [x] Audit and update tests under `tests/python` that reference removed eager helpers (`get`, `get_into`, `get_view`, `get_view_into`, `register_kv_block`) to handle-first APIs (`artifact().tensor_dict`, `.tensor_dict_into`, `.tensor`, views via `.view()`, region reuse via standard `register`).
  - Likely files: `tests/python/test_store_session_api.py`, `tests/python/test_store_view_api.py`, `tests/python/api/test_public_exports.py`, `tests/python/api/test_artifact_handle.py`, `tests/python/api/test_materialization_pipeline_v2.py`, `tests/python/test_store_pipelines_unit.py`, `tests/python/test_store_region_registration.py`, `tests/python/test_register_cgid.py`.
- [x] Update fixtures/mocks relying on older option strings (`vram_coalesced`/`vram_leased`, legacy fallback strings) to new user-friendly inputs in `tests/python/test_config_enum_normalization.py`, `tests/python/api/test_fallback_options.py`, `tests/python/test_store_pipelines_unit.py`, and any registration/materialization tests.
- [x] Remove or rewrite deprecated helper shims introduced during migration; ensure coverage reflects only the new surface.
- [ ] Re-run targeted test suite after updates: `uv run pytest tests/python/...`.

# Acceptance Criteria

- `tc.init()` succeeds with no user-provided config by using embedded defaults; existing explicit-config paths unchanged.
- Public API exports exclude `register_kv_block` and legacy eager getters; handle-first surface is canonical.
- `Artifact` exposes `tensor_into`; lazy view chaining and lazy error model documented and validated.
- User-facing `plan` and fallback shortcuts accepted and validated; internal enums remain intact.
- Docs/README/AGENTS updated; examples show handle-first flows.
- Tests and linters pass for updated areas.

# Rollout / Backout

- Rollout: land feature branches per phase; merge when tests/docs updated. No external compatibility constraints (pre-GA).
- Backout: revert individual phase changes if regressions appear (e.g., revert embedded default config if it impacts existing deployments), keeping design intent for subsequent iterations.

# Risks / Mitigations

- Removing eager helpers and KV helper may break imports/tests: audit and update all references; add migration note; consider a short-lived shim raising clear errors if needed.
- Embedded default daemon config must choose writable paths and non-conflicting ports: use `tempfile` paths and ephemeral ports; log chosen path/port.
- String coercions for `plan`/fallback risk silent typos: validate strictly and surface `ArtifactError` with actionable messages.
- `tensor_into` duplication risk: keep it as a thin wrapper over `tensor_dict_into` and cover with tests to avoid divergent behavior.
