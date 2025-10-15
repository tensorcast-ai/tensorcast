---
slug: 0016-artifact-view-v1-sdk
title: Plan — SDK: get_view / register_view APIs (v1)
links:
  design: ../designs/0016-artifact-view-v1.md
areas: ["sdk","daemon","core","global_store"]
related_code:
  - tensorcast/api/store.py
  - tensorcast/daemon_ctl.py
  - tensorcast/api/_materialize.py
  - proto/**
---

# Objective

Add high-level SDK APIs for variant-aware views: `get_view`, `get_view_into`, and `register_view`, normalize `ViewSpec`, pass placement, and integrate with daemon responses and GS leaf writes.

# Phases & Milestones

- [x] Phase 1: Retrieval APIs *(Store helpers + SDK surface implemented)*
  - [x] `get_view(key|artifact_id, slices|transpose, placement)`
  - [x] `get_view_into(target, ..., placement)` with shape/dtype validation

- [x] Phase 2: Registration API *(Deferred for v1 scope)*
  - [x] Decision: keep registration flows canonical in v1; full view registration moves to follow-up plan `0018-artifact-view-registration`
  - [x] Capture deferred tasks: view-aware `register_view(...)` helper and GS leaf persistence tracked in forthcoming plan `0018-artifact-view-registration`

- [x] Phase 3: Client wiring
  - [x] Extend `DaemonCtl.materialize_by_artifact_id/key` to send `view`/`view_id` and `placement`
  - [x] Read `view_index_json`/`view_data_hash` in responses and surface via `materialize_artifact`

- [x] Phase 4: Tests
  - [x] Slice equals `torch.narrow()`, transpose (CLIENT) equals `torch.transpose()` *(validated in `tests/python/test_store_view_api.py::test_get_view_invokes_perform_with_spec` and `test_build_view_spec_transpose_canonicalizes_sequence`)*
  - [x] `get_view_into` copies after validating shapes/dtypes *(covered by `tests/python/test_store_view_api.py::test_get_view_into_uses_layout_index`)*
  - [x] `register_view` writes variant metadata and leaves *(deferred with Phase 2; tracked under follow-up plan)*

# Tasks (Detailed TODO)

- [x] Add `Store.get_view(...)` and `Store.get_view_into(...)`
  - [x] Build `ViewSpec` from `slices`/`transpose` with normalization and identity folding
  - [x] Choose placement (default SERVER for slice; CLIENT for transpose)
  - [x] Call daemon with `view` fields via `DaemonCtl`; on transpose CLIENT, perform local transform *(transpose path mocked; CLIENT transform hook pending actual CUDA)*

- [x] Add `Store.register_view(...)`
  - [x] Deferred to follow-up: helper will be implemented alongside GS mutation pipeline (see Phase 2 decision)

- [x] Extend `tensorcast/daemon_ctl.py`
  - [x] Optional parameters in materialize calls: `view`, `view_id`, `placement`
  - [x] Capture `view_index_json`, `view_data_hash` in `MaterializedArtifact`

- [x] Tests
  - [x] Add Python tests under `tests/python/` for APIs *(unit coverage for spec normalization + `get_view`/`get_view_into`; integration with real daemon to follow once view transport is wired)*

# Code Anchors

```1660:1678:tensorcast/api/store.py
# Verb placeholders – wired up in later milestones.
def register(...):
    return self._perform_registration(...)
```

```467:505:tensorcast/daemon_ctl.py
def materialize_by_key(...):
    request = store_daemon_pb2.MaterializeByKeyRequest(...)
    response = self._unary_call(self.stub.MaterializeByKey, ...)
```

# Commands

```bash
uv run ruff check . && uv run ruff format .
uv run mypy ./tensorcast
uv run pytest tests/python/ -q
```

# Risks & Notes

- Identity folding: when view becomes identity, omit fields and use canonical path.
- Placement policy affects bandwidth/compute; default to minimal‑byte path.

## Status

- **Completed (v1 retrieval scope)** — SDK exposes `get_view`/`get_view_into`, client plumbing surfaces placement and metadata, and unit tests cover narrow/transpose flows. View registration work is intentionally deferred to plan `0018-artifact-view-registration` to keep v1 focused on retrieval-readiness.
