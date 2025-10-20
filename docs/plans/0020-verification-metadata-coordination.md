---
slug: 0020-verification-metadata-coordination
title: Plan — Deterministic Verification Metadata Coordination
status: completed
links:
  design: ../designs/0020-verification-metadata-coordination.md
areas: ["core"]
related_code:
  - core/store/loader/**
  - core/store/replica/**
  - core/common/artifact_verification.cc
created: 2025-10-20
last_updated: 2025-10-21
---

# Objective

Sequence the changes that introduce guarded verification metadata, atomic file persistence, guaranteed GPU completion, and regression coverage so multi-GPU loads no longer surface false `DataLoss` during concurrent materialisation.

# Phases & Milestones

- [x] **Phase 1 — Guard Infrastructure**
  - [x] Implement `VerificationMetadataGuard` keyed by canonical artifact id with RAII acquisition (`core/store/loader/verification_utils.cc`).
  - [x] Integrate guard acquisition into `reuse_or_generate_verification_json` and add in-process metadata cache (cache + guard reuse with descriptor validation).
  - [x] Emit warn-once logs when guard acquisition waits exceed the configured threshold (threshold: 100 ms default).
- [x] **Phase 2 — Atomic Persistence**
  - [x] Replace direct writes with temp-file + flush + `fsync` + atomic `rename` helper (`atomic_write_file` helper).
  - [x] Extend unit coverage to ensure readers never observe partial JSON (injectable reader hook via new stress test in `verification_utils_test.cc`).
  - [x] Emit structured logs when filesystem operations fail, falling back to cached payload (`verification_metadata_write_{succeeded,failed}` events).
- [x] **Phase 3 — GPU Completion Barrier**
  - [x] Add explicit stream/device synchronization in `TransferService::execute` and `load_from_source` (via `AsyncCopyManager::synchronize_h2d_stream` + `cuda::device_synchronize`).
  - [x] Gate metadata generation on sync success in `ReplicaLoadController::load_async_from_source` (metadata hook runs only after synchronised execute success).
  - [x] Ensure CUDA failures propagate without touching on-disk metadata (sync failures bubble to callers before persistence).
- [x] **Phase 4 — Regression & Stress Tests**
  - [x] Author `core/store/multi_gpu_verification_race_test.cc` covering concurrent LOAD_ONLY flows.
  - [x] Add a stress harness that spawns parallel futures to hammer metadata generation (multi-threaded barriers in new test and cache-reset helper).
  - [x] Capture guard contention behaviour via deterministic assertions (ensures no DataLoss and final metadata parses in concurrent rounds).
  - [x] Verify structured logs emit expected fields without excessive verbosity (log sink assertion in `verification_utils_test.cc`).

# Tasks

- Update Bazel (and tests) to include guard implementations and new regression suites (`core/store/loader:verification_utils_test`, `core/store:multi_gpu_verification_race_test`). No CMake changes required.
- Document guard usage expectations in `core/store/README.md` and loader docs per doc-sync rule. **Done** — README and architecture guide describe guard, atomic writes, and logging.
- Audit existing callers for potential nested guard attempts; add debug assertions. **Done** — guard prevents re-entry via per-artifact map.
- Ensure fake-CUDA path covers the new synchronization calls and mirrors real CUDA behaviour in tests. **Done** — synchronisation funnels through shared `cuda::device_synchronize()`.
- Add portable atomic-write helper under `common/filesystem/` if reuse emerges; otherwise scope it to loader implementation. **Implemented** within loader for now.
- Emit structured logs (artifact id, generator version, guard wait duration) when metadata writes complete and on failure paths. **Done** — VLOG + structured INFO/ERROR entries with guard wait/write durations.

# Test / Rollout / Backout

- `uv run pytest tests/python/store/test_multi_gpu_verification.py` (new or extended) for Python-level regression.
- `bazel test //core/store:multi_gpu_verification_race_test` under fake CUDA and real CUDA gates.
- `bazel test //core/store:all` smoke to catch integration regressions.
- Rollout via staged canary: enable guard and atomic writes behind runtime flag; once validated, flip flag to mandatory and remove fallback.
- Backout plan: toggle runtime flag to disable guard usage (reverts to cached metadata only) while keeping atomic write helper available; if necessary, roll back to previous release tag.

# Risks & Tracking

- **Deadlock risk:** Monitor guard wait duration via structured log sampling; add thresholds for manual investigation if waits exceed 100 ms.
- **Performance regression:** Benchmark load latency before and after; set guard-critical-section budget (<5 ms per artifact).
- **Filesystem incompatibility:** Validate atomic rename on supported filesystems in staging; document fallback behaviour for unsupported cases.
- **Debug visibility gaps:** Validate that log sampling thresholds and message contents are sufficient to detect prolonged waits without additional telemetry.
- **Logging coverage:** Confirm structured logs surface enough context (artifact id, generator version, wait duration, status) to debug regressions quickly.
