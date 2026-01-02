---
slug: 0036-disk-artifact-variant
title: Plan – Lazy Artifact Handle Phase 4 (Disk Variant)
links:
  design: ../designs/0036-04-disk-artifact-variant.md
areas: ["sdk", "daemon"]
related_code:
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/runtime.py
  - tensorcast/api/store/README.md
  - tensorcast/daemon_ctl.py
  - daemon/service/controllers/materialization_controller.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tests/python/api/test_artifact_handle.py
---

# Objective

Ship the disk-backed `Artifact` variant (`tc.from_disk`) defined in the design so disk paths resolve through the daemon, hydrate `ArtifactCache` with canonical metadata + generation, and materialize via the unified iterator pipeline with disk-first source selection and view/subset parity.

# Current State & Grounding
- `Store.from_disk` and `Artifact` already accept `disk_path` hints and default to `FallbackOptions.for_disk`; `Artifact._ensure_identified()` calls `resolve_artifact_from_disk_v2` and seeds `ArtifactCache` with canonical bytes/generation/disk_path on success (tensorcast/api/store/artifact.py).
- `MaterializationPipeline.materialize_subset()` threads `disk_path_hint` and fallback through selective materialization/view inputs and unloads, but does not validate the canonical index format returned by the disk resolver or enforce checksum policy (tensorcast/api/store/materialization.py).
- Daemon `ResolveArtifactFromDisk` returns `{artifact_id, canonical_index_bytes, generation}` using JSON index bytes from `read_from_artifact_dir`; whitelist gating exists, but `verify_checksums` is unused and no Bazel test covers the RPC (daemon/service/controllers/materialization_controller.cc, proto/tensorcast/daemon/v2/store_daemon.proto).
- Client wrapper `DaemonCtl.resolve_artifact_from_disk_v2` is present and used by Python tests; coverage is limited to stubbed resolution/cache seeding (`tests/python/api/test_artifact_handle.py::test_from_disk_resolves_once_and_caches_generation`) with no daemon-backed integration or view/materialization cases.
- Docs mention `from_disk` and cache seeding in tensorcast/api/store/README.md, but architecture docs and rollout/backout levers for disk-path materialization are not yet captured.

# Phases & Milestones

- [x] Phase 1: RPC correctness and proto hygiene
  - [x] Ensure `ResolveArtifactFromDisk` emits canonical index bytes in the v2 format (not JSON), propagates `generation`, honors `verify_checksums`, and records whitelist/validation telemetry.
  - [x] Add daemon/Bazel coverage for success, whitelist rejection, bad path, checksum failure, and generation computation; regenerate Buf/Python stubs after proto/build rule updates.

- [x] Phase 2: SDK resolution, caching, and materialization parity
  - [x] Harden `Artifact` disk-path construction and cache seeding: single resolution per path, TTL/LRU alignment with `ArtifactCache`, and invalidation when generation/disk path mismatch is detected.
  - [x] Ensure `MaterializationPipeline` always threads `disk_path_hint`/`FallbackOptions.prefer="disk"` through selective fetch, view materialization, batch/prefetch release, and unload calls.

- [x] Phase 3: Validation, docs, and rollout
  - [x] Add Python tests covering `from_disk` end-to-end (resolution, subset materialization, view composition, release/unload) and cache reuse; include regression cases for stale paths and checksum toggle.
  - [x] Update docs (`tensorcast/api/store/README.md`, `docs/architecture/architecture-overview.md`, design cross-links) and define rollout/backout toggles with metrics to monitor disk-path usage; publish verification steps.

# Tasks
- Align `ResolveArtifactFromDisk` implementation with the design: use canonical index bytes, enforce checksum flag in the loader, surface whitelist/validation errors with structured status codes, and add tracing/metrics tags for disk paths (bounded cardinality).
- Update proto/service definitions if new fields are required (e.g., `view_index_bytes` hint) and regenerate artifacts via `bash tools/build_proto_python.sh`; adjust Bazel deps for any new includes.
- Strengthen `Artifact` disk-path resolution: debounce repeated RPCs, hydrate `ArtifactCache` with generation/disk_path, reconcile cache entries when disk path differs from the hint, and ensure serialization (`to_dict`) preserves disk hints.
- Ensure `MaterializationPipeline`/`FallbackResolver` prefer disk automatically when `_disk_path_hint` is set, propagate `verify_checksums`, and pass disk_path to replica unload paths to keep daemon cleanup deterministic.
- Expand Python tests under `tests/python/api/` for disk flows (resolution, selective materialization, view, release/unload, cache TTL/LRU, checksum flag) and add a Bazel daemon test for `ResolveArtifactFromDisk`; keep tests GPU-free by setting `TENSORCAST_CUDA_BACKEND=fake`.
- Document behavior and operational knobs (env flags, whitelist expectations, metrics) and outline rollout/backout steps for enabling disk-path materialization in production.

# Acceptance Checks
- [x] `ResolveArtifactFromDisk` returns canonical index bytes + generation in v2 format, respects whitelist/checksum flags, and is covered by `bazel test //daemon:resolve_artifact_from_disk_test --test_env=TENSORCAST_CUDA_BACKEND=fake`.
- [x] `tc.from_disk(path)` resolves once, seeds `ArtifactCache` with `{artifact_id, canonical_index_bytes, generation, disk_path}`, and subsequent `tensor_dict(names=...)` calls use `SourcePreference=DISK` with correct unload calls (validated via Python tests).
- [x] View/batch/prefetch flows operate identically for disk-backed artifacts (tensor subset, view index hints) without extra daemon index RPCs; cache invalidation triggers on NOT_FOUND/FAILED_PRECONDITION for disk paths.
- [x] Docs updated and linked (store README, architecture overview, design), and rollout/backout instructions captured; metrics/telemetry validated for disk-path usage.
- [x] Test suite passes: `uv run pytest tests/python/api/test_artifact_handle.py -k disk`, `uv run pytest tests/python/api/test_materialization_pipeline_v2.py`, and relevant new cases; proto/stub regeneration executed if definitions change.

# Test / Rollout / Backout
- **Unit/Integration**: `uv run pytest tests/python/api/test_artifact_handle.py -k disk`, `uv run pytest tests/python/api/test_materialization_pipeline_v2.py`, plus new disk/view/batch prefetched cases. For daemon RPC coverage run `bazel test //daemon:resolve_artifact_from_disk_test --test_env=TENSORCAST_CUDA_BACKEND=fake`. Regenerate protos with `bash tools/build_proto_python.sh` if proto files change.
- **Rollout**: Keep `tc.from_disk` behind default enablement but validate in staging with disk whitelist configured; monitor disk resolution/materialization metrics and cache hit/miss counters before enabling broadly.
- **Backout**: Disable disk-path materialization by removing disk_path hints (force standard artifact_id/key flows) or tightening whitelist entries; invalidate `ArtifactCache` entries for disk artifacts to avoid stale metadata, and fall back to P2P/local replicas.

# Risks & Tracking
- Disk contents drifting after caching could produce stale metadata; mitigate with generation validation and cache invalidation hooks.
- Checksum enforcement and whitelist gating may block previously working paths; provide clear errors and an opt-out toggle for development.
- Canonical index format mismatch (JSON vs. binary) could break the iterator pipeline; add assertions/tests to lock format.
- Disk-path cardinality in telemetry could explode; sanitize/hash labels and sample where necessary.
