---
slug: 0036-materialization-pipeline-v2
title: Plan – Lazy Artifact Handle Phase 1 (Materialization Pipeline v2)
links:
  design: ../designs/0036-01-materialization-pipeline-v2.md
areas: ["sdk", "daemon"]
related_code:
  - tensorcast/api/store/materialization.py
  - tensorcast/api/_materialize.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/materialization/**
---

# Objective

Deliver the streaming `MaterializationPipeline` described in the design by (1) revving the daemon RPCs/protos to v2 with descriptor streams, (2) updating the Python pipeline to operate on iterators of `(TensorPayloadDescriptor, memoryview)`, and (3) routing disk fallback through the daemon so eager and future lazy clients share the same plumbing. Each phase below maps directly to concrete files/functions that exist in the repo today so implementation work can begin immediately.

# Current State & Grounding

- `MaterializationPipeline` now streams `MaterializationPayload` from the daemon v2 surface by default. Public `get*` methods accept `tensor_names` and filter payloads; the v1 path and `TC_ENABLE_MATERIALIZE_V2` flag have been removed.
- `materialize_artifact_v2` calls the v2 gRPC RPCs, builds descriptors from daemon responses, and yields tensors lazily off the CUDA IPC handle. Canonical/view index bytes are trimmed to requested tensors; lifetimes remain guarded via `_release_materialized`.
- The v2 gRPC service is always registered; descriptor construction uses UMA view plans for view requests, falling back to canonical/view index JSON for non-view paths.
- Disk fallback now travels through `DiskFallbackHint` + `SourcePreference` and no longer invokes SDK disk helpers; daemon disk descriptor parity tests and docs remain outstanding.
- Added `tests/python/api/test_materialization_pipeline_v2.py` covering selective fetch, iterator cancellation, and copy semantics for streaming payloads.

# Phases & Milestones

- [x] **Phase 1: Proto + Daemon Surface Rev**
  - [x] Add `proto/tensorcast/daemon/v2/store_daemon.proto` with `MaterializeReplicaRequest/Response` v2, `MaterializeByKeyRequest/Response` v2, `TensorPayloadDescriptor`, `DiskFallbackHint`, and `ViewSubset`. Update `MODULE.bazel`, `buf.gen.yaml`, and Bazel BUILD rules so Buf + Bazel generate the new package.
- [x] Extend the daemon’s `StoreDaemonServiceImpl` (under `daemon/materialization/`) to populate descriptor arrays using UMA metadata (`core/store/materialization/dataplane/view/view_plan_source.h`). V2 services are always registered alongside v1 for legacy compatibility.
  - [x] Remove materialize capability probing from SDK handshakes.

- [x] **Phase 2: Python Streaming Payloads**
  - [x] Introduce `MaterializationPayload` + `materialize_artifact_v2()` in `tensorcast/api/_materialize.py`. Reuse `get_cuda_memory_ptr` but return a lazy iterator that yields `(TensorPayloadDescriptor, memoryview)`; wrap lifetimes with `contextlib.ExitStack` so `_release_materialized` semantics carry over.
  - [x] Update `MaterializationPipeline` (all `get*` paths plus async variants) to call the streaming iterator path by default; the v1 materialization path and flag have been removed.
  - [x] Create `tests/python/api/test_materialization_pipeline_v2.py` that exercises selective fetch (`tensor_names=["foo"]`), iterator early-stop cleanup, and `get_into` copy semantics without intermediate dicts.

- [ ] **Phase 3: Daemon-only Disk Path**
  - [x] Replace direct SDK disk reads by plumbing `FallbackOptions.for_disk()` through the v2 RPC via `DiskFallbackHint` and `SourcePreference=SOURCE_PREFERENCE_PREFER_DISK`. Delete or guard any call sites that touch disk directly (search for `FallbackResolver.resolve_disk_path` usages).
  - [ ] Ensure daemon `DiskLoader` + `SelectionPlan` + `ViewPlanSource` cover disk materialization by default; extend `//daemon:disk_loader_materialization_test` to assert descriptor correctness for disk payloads.
  - [x] Update `docs/architecture/high-availability-design.md`, `docs/designs/0038-daemon-only-disk-materialization.md`, and `tensorcast/api/store/README.md` to record “daemon-only disk” as an enforced invariant.

- [x] **Phase 4: Observability & Rollout Cleanup**
  - [x] Attach per-descriptor metrics/spans in `tensorcast/api/store/materialization.py` by augmenting the existing `store_metrics.materialization_latency_ms` instrumentation; include differentiators for `names_subset` vs `full`.
  - [x] Remove `MaterializedArtifact` once all clients depend on `MaterializationPayload`; delete unused `materialize_artifact` call sites, update `tensorcast/api/_materialize.py` exports, and drop the flag.

# Tasks

- Diff the new proto files against `proto/tensorcast/daemon/v1/store_daemon.proto`, run `bazel run @rules_buf_toolchains//:buf -- format ./proto -w`, and regenerate bindings via `bash tools/build_proto_python.sh`.
- Share UMA descriptor serialization helpers between `core/store/materialization/dataplane/view/*` and the daemon gRPC layer so both network and disk payloads use identical stride/offset calculations.
- Introduce iterator-safe resource guards (`contextlib.ExitStack`) in Python so early termination still unmaps CUDA IPC; add regression tests that cancel `get_async()` futures midway.
- Update `tensorcast/api/store/README.md`, `daemon/README.md`, and `docs/designs/0038-daemon-only-disk-materialization.md` per the doc-sync rule, highlighting the streaming iterator contract and daemon-only disk requirement.
- Prepare migration notes (release.md + internal runbook) for SDK consumers that call out the removal of `TC_ENABLE_MATERIALIZE_V2`, fallback behavior, and expected perf improvements for selective tensors.

# Status (client scaffolding)

- `materialize_artifact_v2` now calls the daemon v2 RPCs, streams descriptors off CUDA IPC handles, and trims descriptors/canonical index bytes when `tensor_names` are supplied. SDK always uses the v2 path; the v1 helper has been removed from exports.
- `MaterializationPipeline` uses streaming payloads for all `get*` paths; `get_into*` copies consume iterators directly and async flows still unload replicas on completion/cancel.
- The v2 gRPC service is always registered in `server_main`. Descriptor construction now uses UMA view plans when a view is requested, falling back to canonical/view index JSON for non-view calls.
- Disk fallbacks are routed through `DiskFallbackHint`/`SourcePreference`; SDK disk helpers are no longer used in the v2 path. Daemon disk descriptor tests are still pending; docs were refreshed to reflect the daemon-only invariant and checksum hint.
- Per-descriptor telemetry is attached to the client span/metrics, including tensor counts/bytes and a subset/full selector.
- Buf format/regeneration has been run (`bash tools/build_proto_python.sh`); Python stubs for `tensorcast.proto.daemon.v2` are generated. New suite `tests/python/api/test_materialization_pipeline_v2.py` is passing under `uv run pytest`.

# Test / Rollout / Backout

- **Unit / Integration**
  - `uv run pytest tests/python/api/test_materialization_pipeline_v2.py`
  - `uv run pytest tests/python/api/test_disk_materialization_v2.py`
  - `bazel test //daemon:materialization_v2_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //daemon:disk_loader_materialization_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- **Soak / Observability**
  - Enable the v2 RPC flag on a staging daemon pool; verify OTLP traces show descriptor-level attributes and metrics differentiate subset vs. full loads.
  - Run mixed SDK versions to ensure v1 clients continue to work until the fleet is flipped.
- **Rollout**
  - Sequence anchored to real artifacts: (1) land proto + Bazel changes; (2) deploy daemon binaries (v2 RPCs always on); (3) ship SDK with streaming-only pipeline; (4) verify staging and production metrics for descriptor streaming; (5) delete v1 RPC dependencies after a soak window.
- **Backout**
  - SDK no longer ships a v1 code path; daemon always serves v2 alongside v1. Backout requires reverting to a prior SDK/daemon that restored the dual-path guard.

# Risks & Tracking

- **Dual-stack drift**: Maintaining `StoreDaemonService` v1/v2 handlers risks divergence. Mitigation: add shared Catch2 fixture that hits both RPC versions inside `//daemon:materialization_v2_test`.
- **Iterator leaks**: Callers that stop consuming descriptors early could leak IPC handles. Mitigation: enforce context-managed iterators with aggressive finalizers and add stress tests that cancel midway.
- **Disk path regression**: Routing all disk IO through the daemon may expose latent daemon bugs or increase latency. Mitigation: reuse existing DiskLoader code paths, add disk-focused performance benchmarks, and keep the previous direct-SDK path behind an emergency flag for one release.
- **Telemetry volume**: Per-tensor tracing attributes could inflate span sizes. Mitigation: cap descriptor telemetry to top-N tensors by size when `tensor_count > 64` and document the sampling knob.
- **Rollout sequencing**: SDK must not send `tensor_names`/`view_subset_hash` before daemon rollout. Mitigation: add a lightweight `GetServerConfig` extension or new `MaterializeCapabilities` RPC and gate the Python flag on the response.
