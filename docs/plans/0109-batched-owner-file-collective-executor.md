---
slug: batched-owner-file-collective-executor
title: Batched Owner-File Collective Executor Plan
status: completed
areas: ["core", "daemon", "proto", "docs", "tests", "benchmarks", "serving"]
created: 2026-03-28
last_updated: 2026-03-28
related_code:
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0109-batched-owner-file-collective-executor.md
  - docs/plans/0108-01-pre-109-strategy-plane-convergence.md
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/replica.cc
  - core/store/replica/replica_load_controller.cc
  - core/store/store_engine_options.h
  - proto/tensorcast/config/v1/daemon_config.proto
  - core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc
  - docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md
  - docs/internals/disk-load-strategy.md
links:
  design: ../designs/0109-batched-owner-file-collective-executor.md
  related:
    - ../plans/0107-retrieval-policy-plane-cleanup.md
    - ../plans/0108-01-pre-109-strategy-plane-convergence.md
---

# Objective

Implement `OwnerFileBatchedCollectiveExecutor` as the bounded, tensor-aware,
shared-FS collective executor described by `0109`, and make it the default
collective executor for eligible shared-filesystem, same-host TP cold starts.

This plan is now closed for the phase-1 scope described by
`docs/designs/0109-batched-owner-file-collective-executor.md`.

# Sequencing Note

- This is phase 3 of the overall execution chain, not the next plan to start.
- The `0107` runtime prerequisite and all acceptance checks of
  `docs/plans/0108-01-pre-109-strategy-plane-convergence.md` are now complete.
- This plan should start from the converged common-runtime seams rather than
  reopening request-normalization or replica-layer strategy ownership.
- Later SDK/public cleanup from `0107` may continue in parallel only if it does
  not change the internal request and strategy contracts consumed here.
- Because the project is pre-launch, the plan should optimize for the final
  architecture and end by removing rollout-only or prototype-only code that is
  no longer needed.

# Ready Seams

The following prerequisite seams are now available and should be treated as the
fixed starting point for `0109`:

- `NormalizedMaterializationRequestContext` carries retrieval policy and
  execution-topology context separately.
- `ExecutionEnvironmentFacts`, `ExecutionStrategyPlan`, and richer
  `ExecutionCommitReport` exist in
  `core/store/runtime/ingestion/materialization_strategy_types.h`.
- Ordinary `GPU <- DISK` startup now builds a common-runtime strategy plan in
  `MaterializationFacade` before executor choice.
- `Replica` / `ReplicaLoadController` execute the selected plan and emit commit
  diagnostics instead of owning `AUTO`.
- Typed `engine.materialization_strategy` config already includes:
  - owner peak budget
  - batch bytes
  - dim1 staging bytes
  - max inflight batches
  - shared-FS-only gating
  - owner skew threshold
  - dedup saving threshold
  - group-assemble timeout
  - mixed-residual allowance
  - planner cache entries
- The current eager owner-file preload and root whole-source preload paths are
  isolated as non-default prototype collective scaffolding with explicit cleanup
  targets.

# Closure

This plan is closed with the following implementation state:

- ordinary `GPU <- DISK` strategy selection now requires explicit shared-source
  proof before `OwnerFileCollectiveExecutor` becomes eligible under `AUTO`;
  `source_locality=AUTO` no longer silently promotes the collective route.
- `MaterializationFacade` now prefers owner-file collective for eligible
  shared-source TP startup and keeps host-local or unproven-source requests on
  `TensorBatchedLocalExecutor` or generic fallback.
- `collective_disk_load` now uses a bounded owner-batched execution path when
  owner-file collective is selected:
  - weighted file ownership replaces file-index round robin,
  - hot-file skew can fall back to job-level segment splitting,
  - replicated, dim0, and dim1 tensor jobs execute through owner-local staging
    plus NCCL peer distribution,
  - owner staging is batch-local and released after use.
- the old eager `owned_payload` preload is no longer used on the owner-file
  collective execution path.
- root whole-source preload is skipped whenever owner-file batched execution is
  active and remains only as a legacy collective fallback shape.
- the steady-state phase-1 rollout remains intentionally zero-residual-only:
  collective eligibility still rejects requests with residual fallback bytes,
  so owner-file collective never invents generic fallback at runtime.

# Verification

Passed:

- `bazel test //core/store/replica:collective_disk_loader_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //core/store/runtime/ingestion:materialization_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

Focused coverage added:

- shared-source eligibility versus `AUTO` / `SHARED_SOURCE` routing in
  `materialization_facade_test`
- bounded owner-file execution regression coverage through
  `collective_disk_loader_test`

# Follow-On

The remaining work after this closure is rollout evidence rather than core
executor bring-up:

- rerun the shared-FS versus host-local benchmark matrix from
  `docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`
- re-run TP serving validation on the target workload family
- broaden owner-file rollout only after those benchmark and serving gates are
  recaptured
