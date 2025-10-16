---
slug: 0018-artifact-view-registration-plan
title: Plan — Variant View Registration (v1.5)
links:
  design: ../designs/0018-artifact-view-registration.md
areas: ["core","daemon","global_store","sdk"]
related_code:
  - core/store/**
  - daemon/**
  - tensorcast/api/**
  - tensorcast/global_store/**
status: in_progress
---

# Objective

Bridge the current codebase to the design in `../designs/0018-artifact-view-registration.md` so SDK callers can register slice/transpose views while the daemon and Store Engine reconstruct canonical ByteSpace, persist hashes, and publish variant metadata to Global Store without regressing canonical registration flows.

# Baseline & Gaps

- SDK client now supports explicit `SERVER` and `CLIENT` placements; remaining work is better retry guidance and surfacing daemon errors to orchestrators.
- Daemon publishes view descriptors and now pushes variant leaf digests plus canonical coverage into Global Store; metrics/CLI surfacing remain.
- Integration coverage is limited to unit plumbing; we still lack end-to-end registration tests in Python exercising fake CUDA + daemon loops.

# Progress Update (current)

- Phase 0 complete: proto surfaces, feature flag plumbing, and Buf regeneration landed.
- Phase 1 now wired end-to-end: StoreEngine stores bidirectional plans, ingests view bytes, and computes variant hashes; daemon plumbing is underway.
- Global Store integration online: StoreEngine publishes `UpdateArtifactViewState` via the new client helper, now including variant leaf digests and canonical coverage. SDK emits canonical index v3 bytes and captures view metadata from commit responses.
- Test coverage run during this iteration: `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`, `bazel test //core/store:registration_memory_replica_test --define=use_fake_cuda=true`, `uv run pytest tests/python/global_store/test_services.py`.
- Global Store metrics expose `tc_view_registration_total`/`tc_view_partial_backlog_bytes`, and SDK now surfaces actionable placement guidance (new pytest coverage for both paths).
- Open items before feature completion: capture rollout/backout SOP, finalize owner reviews, and confirm staging telemetry matches expectations.

# Phases & Milestones

- [x] Phase 0: Protocol surfaces
  - [x] Extend `proto/tensorcast/daemon/v1/store_daemon.proto` with `ViewRegistrationOptions` (fields: `string view_id`, `ViewSpec spec`, `TransformPlacement placement`, `uint64 canonical_size_bytes`, repeated `CanonicalRange ranges`, `bool allow_partial`) and add optional `view` field to `BeginRegisterArtifactRequest`.
  - [x] Add `ViewUploadChunk` (offset/length) branch to `FeedRegisterArtifactStreamRequest.feed` while retaining canonical lease segments.
  - [x] Regenerate stubs via `bash tools/build_proto_python.sh` and update Bazel deps; adjust `tensorcast/proto/__init__` exports if needed.
  - [x] (Rolled back) Removed the temporary `enable_view_registration` feature flag; the daemon now always accepts view registrations.

- [x] Phase 1: Core bidirectional planning & ingestion
  - [x] Extend `core/store/loader/view_planner.{h,cc}` with `ViewWritePlan` + `BidirectionalViewPlan` that invert selection/permutations for registration.
  - [x] Add `view_ingest_executor.{h,cc}` to apply inverse transforms and write view chunks into canonical GPU memory; include Catch2 tests (`//core/store/loader:view_ingest_executor_test`).
  - [x] Update `StoreEngine::begin_register_artifact` / `commit_registered_artifact` to accept optional `BidirectionalViewPlan`, stream view chunks, compute canonical + view hashes, and persist verification metadata.
  - [x] Ensure canonical hash computation remains guarded by `schema_version="v3"`; replace any lingering `IndexV2` assumptions discovered during implementation.

- [x] Phase 2: Daemon registration pipeline
  - [x] Plumb new proto fields through `RegistrationController::begin/feed/commit`, requesting bidirectional plans from `StoreEngine` when `req.has_view()`.
  - [x] Implement streaming handler that copies uploaded view byte ranges into the pending replica using the core executor while tracking canonical coverage.
  - [x] Enforce explicit placement honoring: daemon surfaces `FAILED_PRECONDITION` when server placement is unavailable.
  - [x] Collect telemetry (bytes transferred, view ids, coverage) and surface via existing metrics registry.

- [x] Phase 3: Global Store publication
  - [x] Add Global Store client helper that wraps `UpdateArtifactViewState` with retries so StoreEngine can publish variant metadata.
  - [x] Populate `UpdateArtifactViewState` requests with variant leaf digests and canonical coverage; add retry/backoff logging for failures.
  - [x] Extend Global Store CLI/metrics to expose `tc_view_registration_total` and `tc_view_partial_backlog_bytes` counters so operators can track partial coverage backlog.

- [x] Phase 4: SDK API + tests
  - [x] Replace `IndexV2.build_bytes` path in `_register._prepare_build` with canonical index v3 emission (reuse `_C.build_canonical_index_from_safetensors` or new helper) and ensure `schema_version="v3"` is sent.
  - [x] Implement `Store.register_view(...)` in `tensorcast/api/store.py`, reusing `_resolve_view_inputs` and honoring caller-specified placement.
  - [x] Update `_register_artifact_core` and uploader paths to stream view chunks when placement is SERVER, and reconstruct canonical tensors on the client path before invoking the legacy upload loop.
  - [x] Add SDK unit + integration tests: extend `tests/python/test_store_view_api.py` with registration failure guidance coverage and expand `tests/python/global_store/test_services.py` for metrics/backlog handling.

- [ ] Phase 5: Rollout & migration cleanup
  - [x] Staged rollout in CI: exercised view registration using fake CUDA env, captured metrics baseline, and confirmed the feature is always enabled post-flag removal.
  - [x] Backfill documentation cross-links (docs/architecture/p2p-transfer-strategies.md, module READMEs) to describe registration data path.

# Tasks

- [x] Proto & build
  - [x] Modify `proto/tensorcast/daemon/v1/store_daemon.proto` structures as described; add Bazel deps for new messages.
  - [x] Run `bash tools/build_proto_python.sh` and update generated code references.
  - [x] Adjust `tensorcast/daemon/v1/store_daemon_pb2.pyi` export expectations if type hints break *(regen confirmed no deltas needed)*.

- [x] Core implementation
  - [x] Define `ViewWritePlan` structs alongside `SelectionPlan` in `view_planner.h`; ensure serialization stays deterministic.
  - [x] Implement inverse permutation logic in `view_planner.cc`, covering mixed narrow+transpose cases.
  - [x] Create `view_ingest_executor.{h,cc}` to materialize canonical buffers from view uploads (CPU + GPU paths); integrate into Bazel BUILD.
  - [x] Extend `StoreEngine::PendingRegistrationEntry` to store bidirectional plans + coverage bookkeeping; update commit path hashing to use canonicalized memory post-ingest.

- [x] Daemon changes
  - [x] Update `daemon/service/controllers/registration_controller.cc` begin/feed/commit to propagate view descriptors, call new engine entrypoints, and manage partial coverage accounting in `RegistrationManager`.
  - [x] Extend `RegistrationManager` metadata structs to track view id/spec, placement, canonical coverage map, and feature flag state.
  - [x] Wire new telemetry into OTEL counters (`tc_register_view_bytes_total`, `tc_register_view_partials_total`).

- [x] Global Store integration
  - [x] Add helper utility for `UpdateArtifactViewState` (Global Store service now exposes `record_variant_registration` for daemon and SDK callers).
  - [x] Update daemon commit path to populate variant leaf digests and canonical coverage via the helper (with retries + logging).
  - [x] Expand `tests/python/global_store/test_services.py` to assert variant registration via new helper.

- [x] SDK + docs
  - [x] Improve client retry guidance when daemon rejects a placement (surface actionable `ArtifactError`).
  - [x] Update `tensorcast/api/_register.py` to interpret new response fields and store them on `RegistrationResult`.
  - [x] Document registration flow changes in `tensorcast/api/README.md` and `docs/architecture/p2p-transfer-strategies.md`.

# Test / Rollout / Backout

- **Unit / component tests**
  - [x] `bazel test //core/store/loader:view_ingest_executor_test --define=use_fake_cuda=true`
  - [x] `bazel test //core/store:store_engine_test --define=use_fake_cuda=true` (add view registration cases)
  - [x] `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`
  - [x] `bazel test //core/store:registration_memory_replica_test --define=use_fake_cuda=true`
  - [x] `uv run pytest tests/python/test_store_view_api.py`
  - [x] `uv run pytest tests/python/global_store/test_services.py`

- **Integration**
  - End-to-end registration harness in `tests/python` using fake daemon + fake CUDA to assert canonical hash parity and Global Store updates.
  - Benchmark canonical vs view registration throughput (target ≥50% byte reduction for narrow-only uploads; ≤5% CPU regression).

- **Rollout**
  - Exercise view registration flows in staging and monitor OTEL counters / Global Store metrics before production rollout.

- **Backout**
  - Publish operational SOP for temporarily blocking view registrations (e.g., via config hotfix) until a full fix is deployed.
  - Provide script to prune inconsistent variant metadata if ingestion fails mid-rollout.

# Risks & Tracking

| Risk | Impact | Mitigation |
|------|--------|------------|
| Inverse plan calculation corrupts canonical ByteSpace | High | Comprehensive Catch2 coverage + invariant checks in `StoreEngine::commit_registered_artifact` |
| GPU transpose requires large scratch buffers | Medium | Capability probe; document when callers should choose CLIENT placement |
| Proto change breaks legacy clients | Medium | Keep daemon tolerant of missing view fields and document SDK upgrade path |
| Global Store metadata growth from partial uploads | Medium | Add TTL/retention in helper and monitor metrics |
| SDK placement negotiation regress canonical flows | Low | Expand integration tests covering canonical + view registration |

# Owner Checklist

- [ ] Secure reviews from core, daemon, SDK, and Global Store owners.
- [ ] Confirm documentation updates land alongside code changes.
- [ ] Capture rollout/backout runbook in internal wiki once staging validation passes.
