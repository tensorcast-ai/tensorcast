---
slug: batched-owner-file-collective-executor
title: Batched Owner-File Collective Executor Plan
status: in_progress
areas: ["core", "daemon", "proto", "docs", "tests", "benchmarks", "serving"]
created: 2026-03-28
last_updated: 2026-03-30
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

Phase-1 executor bring-up described by
`docs/designs/0109-batched-owner-file-collective-executor.md` is complete.

This plan remains active until the
remaining rollout evidence, graduation decisions, and any required follow-up
implementation for default shared-FS routing are complete.

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

# Current State & Grounding

Implemented baseline:

- ordinary `GPU <- DISK` strategy selection now requires explicit shared-source
  proof before `OwnerFileCollectiveExecutor` becomes eligible under `AUTO`;
  `source_locality=AUTO` no longer silently promotes the collective route
- `MaterializationFacade` now prefers owner-file collective for eligible
  shared-source TP startup and keeps host-local or unproven-source requests on
  `TensorBatchedLocalExecutor` or generic fallback
- `collective_disk_load` now uses a bounded owner-batched execution path when
  owner-file collective is selected:
  - weighted file ownership replaces file-index round robin
  - hot-file skew can fall back to job-level segment splitting
  - replicated, dim0, and dim1 tensor jobs execute through owner-local staging
    plus NCCL peer distribution
  - owner staging is batch-local and released after use
- eager `owned_payload` preload is no longer used on the owner-file collective
  execution path
- root whole-source preload is skipped whenever owner-file batched execution is
  active and remains only as a legacy collective fallback shape

Still open for full `0109` sign-off:

- the steady-state implementation remains intentionally zero-residual-only;
  collective eligibility still rejects requests with residual fallback bytes
- collective candidate estimation in `MaterializationFacade` is still coarse and
  is not yet backed by executor-measured owner bytes, peer bytes, or planner
  overhead
- benchmark and serving evidence have not yet been recaptured after the
  batched-owner implementation landed
- rollout-default decisions and post-evidence cleanup of legacy collective
  scaffolding remain open

Code paths the follow-on owner should treat as primary:

- planning and routing:
  - `core/store/runtime/ingestion/materialization_facade.cc`
  - `core/store/runtime/ingestion/materialization_strategy_types.h`
- executor runtime:
  - `core/store/replica/collective_disk_loader.cc`
- verification and focused tests:
  - `core/store/runtime/ingestion/materialization_facade_test.cc`
  - `core/store/replica/collective_disk_loader_test.cc`
- benchmark harness:
  - `core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc`
- benchmark baseline and comparison context:
  - `docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`

# Phases & Milestones

- [x] Phase 1: Phase-1 Executor Bring-Up
  - [x] Milestone 1.1: Reuse the converged common-runtime strategy seam rather
    than replica-layer branch order
  - [x] Milestone 1.2: Require explicit shared-source proof for `AUTO`
    eligibility
  - [x] Milestone 1.3: Replace eager owner preload with bounded owner-batched
    execution for replicated, dim0, and dim1 tensor work

- [ ] Phase 2: Close Remaining Executor Gaps
  - [ ] Milestone 2.1: Decide whether default graduation requires explicit mixed
    residual handling; if yes, implement preplanned collective-plus-generic
    residual execution rather than zero-residual-only gating
  - [ ] Milestone 2.2: Tighten owner-file collective candidate estimation using
    real owner unique bytes, peer-transfer bytes, owner skew, and planner
    overhead rather than only coarse request-size heuristics
  - [ ] Milestone 2.3: Add focused unit coverage for weighted ownership,
    hot-file split behavior, dim1 owner-batched execution, and failure
    accounting

- [ ] Phase 3: Benchmark Evidence Recapture
  - [ ] Milestone 3.1: Re-run the host-local versus shared-FS benchmark matrix
    used by
    `docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`
  - [ ] Milestone 3.2: Record peak temporary bytes, selection reasons, owner
    skew, and effective source-read amplification for each run
  - [ ] Milestone 3.3: Produce one comparable result table for:
    `fastsafetensors`, TensorCast local-batched, TensorCast owner-file batched
    collective, and any remaining legacy collective fallback used for reference

- [ ] Phase 4: TP Serving Validation
  - [ ] Milestone 4.1: Re-run TP=8 `vllm serve` validation on host-local SSD
  - [ ] Milestone 4.2: Re-run TP=8 `vllm serve` validation on shared-FS / JFS
    model roots
  - [ ] Milestone 4.3: Capture `/v1/completions` correctness, digest-compare
    output, startup wall time, and selected executor diagnostics

- [ ] Phase 5: Graduation And Cleanup
  - [ ] Milestone 5.1: Decide whether owner-file collective can become the
    default collective executor for eligible shared-FS TP startup
  - [ ] Milestone 5.2: Update benchmark / internals docs with the final
    evidence package and rollout decision
  - [ ] Milestone 5.3: If default graduation is accepted, demote or delete the
    remaining eager owner-file and whole-source legacy fallback scaffolding

# Tasks

- [ ] Implementation follow-up
  - [ ] Add focused `collective_disk_loader_test` coverage for:
    - weighted ownership
    - owner-skew split behavior
    - dim1 owner-batched execution
    - runtime failure accounting
  - [ ] Add focused `materialization_facade_test` coverage for:
    - collective candidate rejection due to dedup threshold
    - host-local rejection under `owner_file_collective_shared_fs_only`
    - collective selection after local rejection for eligible shared-source
      requests
  - [ ] Decide whether mixed residual support is required before default
    graduation. If yes:
    - add explicit planner output for collective-covered versus residual ranges
    - keep residual work visible in `ExecutionCommitReport`
    - prohibit runtime-synthesized fallback

- [ ] Benchmark execution
  - [ ] Build `//core/store/materialization/benchmarks:safetensors_load_strategy_benchmark`
  - [ ] Use the same `loading-meta.json` families and storage classes called out
    in `docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`
  - [ ] Capture at least:
    - owner-file batched collective candidate
    - TensorCast local-batched baseline
    - `fastsafetensors`
    - host-local and shared-FS/JFS variants
  - [ ] Save raw logs, summary table, and exact flags used for each run

- [ ] Serving execution
  - [ ] Reuse the full `vllm serve` command shapes documented in
    `docs/plans/0108-tensor-aware-materialization-strategy-plane.md`
    “Full vLLM validation”
  - [ ] Run host-local SSD `auto`, `fastsafetensors`, and `tensorcast`
    references
  - [ ] Run shared-FS / JFS `tensorcast` candidate with the same TP shape
  - [ ] Record:
    - startup wall time
    - daemon strategy logs
    - `/v1/completions` output
    - digest-compare output

- [ ] Documentation and handoff
  - [ ] Append the new benchmark and serving results to a benchmark note or a
    follow-up plan doc
  - [ ] Update `docs/internals/disk-load-strategy.md` if the final rollout
    decision changes default collective behavior again
  - [ ] Record the final rollout decision and cleanup scope back into
    `docs/designs/0109-batched-owner-file-collective-executor.md`

# Acceptance Checks

- [x] `source_locality=AUTO` is never treated as sufficient proof for default
  shared-FS routing
- [x] owner-file collective no longer depends on eager `owned_payload`
  residency
- [x] owner-file collective skips root whole-source preload on the steady-state
  batched path
- [ ] shared-FS TP cold-start wall time matches or beats `fastsafetensors` on
  the target workload family
- [ ] host-local default path remains no worse than the current best
  local-batched path
- [ ] no OOM is observed on the current Step3p5 TP=8 workload
- [ ] TP=8 serving correctness passes on both host-local SSD and shared-FS /
  JFS model roots
- [ ] if default graduation is requested, the collective route either:
  - [ ] supports explicit mixed residual planning, or
  - [ ] remains clearly documented and approved as zero-residual-only for the
    graduated workload family
- [ ] the final rollout decision is backed by one committed evidence package,
  not by ad-hoc local judgment

# Evidence Package

The owner finishing `0109` should attach or produce all of the following:

- [ ] exact commit SHA and daemon config used for benchmark and serving runs
- [ ] raw benchmark logs for host-local and shared-FS runs
- [ ] one summary table with:
  - strategy
  - storage class
  - wall time
  - source bytes read
  - peak temporary bytes
  - selected executor
  - owner skew
- [ ] daemon logs containing:
  - `materialize_replica strategy_plan`
  - `collective_owner_file_batched plan`
  - `collective_disk_load timings`
- [ ] TP=8 serving logs and `/v1/completions` outputs
- [ ] digest-compare output versus the selected reference baseline
- [ ] one written graduation recommendation:
  - keep off by default
  - enable for explicit shared-FS experiments only
  - promote to default collective executor for eligible shared-FS TP startup

# Test / Rollout / Backout

Already passed on the landed phase-1 implementation:

- [x] `bazel test //core/store/replica:collective_disk_loader_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //core/store/runtime/ingestion:materialization_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

Remaining rollout steps:

- [ ] Re-run the benchmark matrix after any follow-up code change
- [ ] Re-run TP serving validation after any routing-default or residual-policy
  change
- [ ] Do not enable owner-file collective as the default shared-FS route unless
  the benchmark and serving evidence package is complete

Backout rules:

- [ ] If shared-FS startup regresses or memory spikes exceed budget, revert
  default selection to local-batched or generic fallback
- [ ] Do not back out by restoring eager `owned_payload` as the preferred
  steady-state execution model
- [ ] Keep host-local bias explicit even if shared-FS rollout is expanded

# Risks & Tracking

- [ ] Risk: the phase-1 zero-residual-only implementation is mistaken for full
  `0109` completion and default rollout happens without explicit residual-policy
  sign-off
  - Mitigation: keep this plan `in_progress` until the evidence package and
    graduation decision are complete
- [ ] Risk: benchmark wins are measured only on one storage class and hide
  regressions on host-local SSD
  - Mitigation: require both host-local and shared-FS recapture in one result
    table
- [ ] Risk: serving startup looks healthy but semantic outputs drift
  - Mitigation: require both `/v1/completions` correctness and digest-compare
    output
- [ ] Risk: legacy collective scaffolding remains in the hot path long after the
  new route is proven
  - Mitigation: make cleanup a graduation milestone rather than an optional
    later tidy-up
