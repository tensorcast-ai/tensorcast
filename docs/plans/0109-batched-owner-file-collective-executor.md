---
slug: batched-owner-file-collective-executor
title: Batched Owner-File Collective Executor Plan
status: proposed
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

This plan starts only after the prerequisite convergence work in
`docs/plans/0108-01-pre-109-strategy-plane-convergence.md` lands.

# Sequencing Note

- This is phase 3 of the overall execution chain, not the next plan to start.
- Start only after `docs/plans/0107-retrieval-policy-plane-cleanup.md` Phases 1
  through 3 and all acceptance checks of
  `docs/plans/0108-01-pre-109-strategy-plane-convergence.md` complete.
- Later SDK/public cleanup from `0107` may continue in parallel only if it does
  not change the internal request and strategy contracts consumed here.
- Because the project is pre-launch, the plan should optimize for the final
  architecture and end by removing rollout-only or prototype-only code that is
  no longer needed.

# Current State & Grounding

- `core/store/replica/collective_disk_loader.cc` still contains the current
  owner-file prototype:
  - file ownership computed by `compute_file_segments(...)`,
  - eager owner preload into `OwnedFileRankState::owned_payload`,
  - optional root whole-source preload through
    `maybe_load_whole_source_to_root_buffer(...)`,
  - collective and local-batched execution in the same file.
- Current file ownership is file-index round-robin, not weighted by the current
  request's selected bytes.
- `core/store/replica/collective_disk_loader.cc` already has useful building
  blocks that should be reused rather than rediscovered:
  - shared `build_tensor_jobs(...)`
  - local-batched direct segment lowering
  - local and collective dim1 2D-pack execution shapes
  - cached NCCL clique construction and same-host group assembly
- `core/store/runtime/ingestion/materialization_facade.cc` already emits
  `ExecutionCommitReport` for mapped-target execution and has the right place to
  become the long-term owner of default executor routing after pre-109
  convergence.
- the shared request-normalization boundary and the shared
  `ExecutionEnvironmentFacts` / `ExecutionStrategyPlan` contracts are still
  prerequisite work; this plan must consume those seams rather than recreate
  them locally.
- `proto/tensorcast/config/v1/daemon_config.proto` and
  `core/store/store_engine_options.h` currently expose booleans and preference
  enums for `materialization_strategy`, but `0109` needs typed budgets and
  thresholds before it can be a default strategy.
- The benchmark evidence in
  `docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`
  already identifies the winning execution shape:
  - avoid tiny fragmented reads,
  - keep dim1 on row-block staging plus 2D pack,
  - reduce planner overhead,
  - overlap IO and pack/scatter more aggressively.

# Phases & Milestones

- [ ] Phase 1: Extract Planner Inputs And Ownership Logic
  - [ ] Milestone 1.1: Reuse the pre-109 shared strategy contracts and ordinary
    replica convergence seams instead of adding a second prototype path.
  - [ ] Milestone 1.2: Implement weighted owner assignment from selected source
    bytes and cap owner skew with typed policy.
  - [ ] Milestone 1.3: Define the executor-private `OwnerBatchPlan` /
    `OwnerBatchOp` lowering from shared strategy inputs.

- [ ] Phase 2: Implement The Batched Executor
  - [ ] Milestone 2.1: Add the direct contiguous batched path for replicated and
    dim0-dominant work without eager whole-owner preload.
  - [ ] Milestone 2.2: Add the staged dim1 batched path using bounded row-block
    staging and 2D pack.
  - [ ] Milestone 2.3: Make batch lifetime explicit and remove the need for
    persistent `owned_payload` in the new executor path.

- [ ] Phase 3: Mixed Residual And Commit Semantics
  - [ ] Milestone 3.1: Support explicit mixed collective-plus-generic execution
    where residual ranges are planned before execution starts.
  - [ ] Milestone 3.2: Emit batch-level commit accounting through
    `ExecutionCommitReport`.
  - [ ] Milestone 3.3: Define failure handling for read, pack, send/recv, and
    group-assembly failure cases without inventing residual fallback at runtime.

- [ ] Phase 4: Default Routing And Observability
  - [ ] Milestone 4.1: Implement the cost-model and typed-policy gating used by
    `AUTO` to choose this executor only for eligible shared-source TP startup.
  - [ ] Milestone 4.2: Keep host-local SSD and low-dedup cases on
    `TensorBatchedLocalExecutor`.
  - [ ] Milestone 4.3: Emit executor-selection reasons, owner skew, batch
    counts, peak temporary bytes, and planner-overhead diagnostics.

- [ ] Phase 5: Promote And Clean Up
  - [ ] Milestone 5.1: Make this executor the default collective executor for
    eligible shared-FS same-host TP startup.
  - [ ] Milestone 5.2: Demote or delete the eager `owned_payload` path and the
    coarse root whole-source preload path.
  - [ ] Milestone 5.3: Keep the old collective path only as an explicit
    low-priority fallback while migration evidence is still being gathered.

- [ ] Phase 6: Hard-Cut Final Cleanup
  - [ ] Milestone 6.1: Delete temporary rollout-only fallbacks once the default
    route is proven.
  - [ ] Milestone 6.2: Remove redundant owner-file prototype helpers that are no
    longer used by the steady-state executor.
  - [ ] Milestone 6.3: Ensure no temporary compatibility or transition-only path
    remains in the hot path when this plan closes.

# Tasks

- [ ] Planner and lowering
  - [ ] Add owner-file batched planning structures under the shared strategy
    seam introduced by the pre-109 work.
  - [ ] Reuse `build_tensor_jobs(...)` and shared representation work where
    possible; do not build a second semantic recovery stack.
  - [ ] Add weighted owner assignment and optional hot-file segment split logic.

- [ ] Executor implementation
  - [ ] Add direct read-to-final execution for replicated and dim0 work in
    bounded batches.
  - [ ] Add staged dim1 row-block execution with bounded staging and 2D pack.
  - [ ] Add double-buffering or equivalent overlap so read of batch N+1 can
    overlap pack/scatter of batch N when resources permit.
  - [ ] Remove eager `owned_payload` residency from the new path.

- [ ] Mixed residual support
  - [ ] Integrate explicit residual `FallbackByteRangeOp` planning for requests
    that are not zero-residual.
  - [ ] Ensure executor code consumes only the ranges assigned to it and does not
    synthesize fallback ranges at runtime.
  - [ ] Wire commit/failure accounting into the shared
    `ExecutionCommitReport`.

- [ ] Cost model and routing
  - [ ] Implement candidate estimation using:
    - selected source bytes
    - unique owner bytes after dedup
    - dim1 staging amplification
    - peer-transfer bytes
    - owner skew
    - planner overhead
    - peak temporary bytes
  - [ ] Add typed config fields and defaults for these thresholds and budgets.
  - [ ] Regenerate protobuf outputs with `bash tools/build_proto_python.sh`
    whenever config proto fields change.
  - [ ] Integrate the routing decision into the common-runtime strategy owner,
    not into integration-local code.

- [ ] Cleanup and migration
  - [ ] Remove the need for `compute_file_segments(...)` round-robin ownership in
    the default path.
  - [ ] Demote eager owner preload and whole-source preload to migration fallback
    only.
  - [ ] Update `docs/internals/disk-load-strategy.md` once the new default
    routing behavior lands.

# Test / Rollout / Backout

## Acceptance checks

- [ ] Eligible shared-FS same-host TP cold starts choose
  `OwnerFileBatchedCollectiveExecutor` by default once rollout is enabled.
- [ ] Host-local SSD startup remains on `TensorBatchedLocalExecutor` unless
  explicit policy overrides it.
- [ ] No eager whole-owner `owned_payload` residency is required on the new
  execution path.
- [ ] Mixed residual execution is explicit and preserves fallback correctness.
- [ ] No new branch-order policy is added to `Replica`,
  `ReplicaLoadController`, or `collective_disk_loader.cc` to make this executor
  work.
- [ ] The new executor meets the memory and performance gates below.
- [ ] Temporary rollout-only and prototype-only code introduced during landing
  is removed before the plan is considered complete.

## Test plan

- [ ] `bash tools/build_proto_python.sh`
- [ ] `bazel test //core/store/replica:collective_disk_loader_test --test_output=errors`
- [ ] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_output=errors`
- [ ] `bazel test //core/store:store_engine_test --test_output=errors`
- [ ] `bazel test //daemon:materialize_into_mapped_target_test --test_output=errors`
- [ ] Add focused unit tests for:
  - [ ] weighted owner assignment and skew rejection
  - [ ] batched dim0 direct execution
  - [ ] batched dim1 staged execution
  - [ ] mixed residual planning
  - [ ] batch commit/failure accounting
- [ ] Run the benchmark driver used by
  `docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md` for
  shared-FS and host-local variants.
- [ ] Re-run TP serving validation on the target workload family once the
  executor reaches rollout phase.

## Rollout

- [ ] Stage 1: typed config off by default, explicit experiment enable only.
- [ ] Stage 2: explicit shared-FS TP experiments with benchmark and serving
  evidence.
- [ ] Stage 3: make it the default collective executor for eligible shared-FS
  same-host TP startup.
- [ ] Stage 4: remove or demote eager owner preload once the new path is stable.
- [ ] Stage 5: delete rollout-only fallbacks and temporary implementation
  scaffolding that are no longer needed in the target state.

## Backout

- [ ] Revert the new executor selection in `AUTO` if shared-FS startup regresses
  or memory spikes exceed budget.
- [ ] Do not back out by reintroducing eager `owned_payload` as the default
  execution shape.
- [ ] Keep local-batched and generic fallback viable throughout the rollout.
- [ ] Do not back out by skipping the common-runtime strategy owner and moving
  policy back into replica-layer branch order.

# Risks & Tracking

- [ ] Risk: the implementation lands as another branch inside
  `collective_disk_loader.cc` without really separating planner and executor.
  - Mitigation: require the pre-109 convergence seam before feature
    implementation and review file ownership carefully.
- [ ] Risk: weighted ownership is skipped and hot-file skew erases the benefit of
  dedup.
  - Mitigation: make skew measurement and rejection part of the eligibility gate
    and the validation matrix.
- [ ] Risk: dim1 batches still serialize `read -> pack -> sync` too tightly and
  leave benchmark headroom on the table.
  - Mitigation: include overlap work as a first-class milestone, not a future
    tuning note.
- [ ] Risk: default routing flips too early and regresses host-local startup.
  - Mitigation: keep host-local bias explicit in the cost model and require
    no-regression evidence before default promotion.
