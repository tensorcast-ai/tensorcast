---
slug: tp-source-window-collective-realization-plan
title: TP Source-Window Collective Realization Plan
status: draft
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests", "benchmarks"]
created: 2026-06-15
last_updated: 2026-06-15
related_code:
  - docs/designs/0122-tp-source-window-collective-realization.md
  - docs/plans/0122-01-source-window-collective-feasibility-validation.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0109-batched-owner-file-collective-executor.md
  - docs/designs/0115-composite-materialization-and-vectored-direct-write.md
  - docs/designs/0121-unified-artifact-realization-kernel.md
  - docs/internals/disk-load-strategy.md
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/replica/collective_disk_loader.h
  - core/store/replica/collective_disk_loader.cc
  - core/store/materialization/dataplane/sources/source_window_scheduler.h
  - core/store/materialization/dataplane/sources/source_window_scheduler.cc
  - core/store/materialization/dataplane/sinks/target_layout_gpu_sink.h
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - proto/tensorcast/config/v1/daemon_config.proto
links:
  design: ../designs/0122-tp-source-window-collective-realization.md
  feasibility: ./0122-01-source-window-collective-feasibility-validation.md
  dependencies:
    - ../designs/0108-tensor-aware-materialization-strategy-plane.md
    - ../designs/0109-batched-owner-file-collective-executor.md
    - ../designs/0110-artifact-representation-contract-and-transform-unification.md
    - ../designs/0115-composite-materialization-and-vectored-direct-write.md
    - ../designs/0121-unified-artifact-realization-kernel.md
---

# Objective

Implement the `SourceWindowCollectiveExecutor` described in
`docs/designs/0122-tp-source-window-collective-realization.md`.

The implementation must make TensorCast competitive with InstantTensor on TP
cold start while preserving TensorCast's artifact-centered design:

- no vLLM-private data path;
- no public SDK source-window API;
- no compatibility obligation for old experimental knobs or executor behavior;
- all source, target, layout, strategy, lifecycle, and diagnostics facts flow
  through the existing TensorCast realization and materialization trunk.

# Grounding In Current Code

Current relevant code state:

- `core/store/runtime/ingestion/materialization_strategy_types.h`
  defines:
  - `ResolvedMaterializationPlan`
  - `SourceBoundLanePlan`
  - `ExecutionStrategyPlan`
  - `ExecutionStrategyExecutor`
  - `ExecutionCommitReport`
- `core/store/runtime/ingestion/materialization_facade.cc`
  already includes `source_window_scheduler.h` and owns selection between
  generic, local typed, local mapped, and collective paths.
- `core/store/replica/collective_disk_loader.cc`
  already owns same-host NCCL clique setup, owner-file collective, mapped
  collective, and local mapped target execution.
- `core/store/materialization/dataplane/sources/source_window_scheduler.cc`
  already implements source-ordered window merging and scatter for
  `ByteRangeMap`, but it is single-rank and not TP collective aware.
- `core/store/replica/collective_disk_loader.h`
  exposes:
  - `try_collective_disk_load`
  - `try_collective_mapped_target_load`
  - `try_local_mapped_target_load`
  - `summarize_local_batched_disk_load`
- `proto/tensorcast/config/v1/daemon_config.proto` already owns typed daemon
  materialization strategy config.

Measured current regression:

- 30B mapped target reads `25.15GB` source per rank to produce `7.53GB`
  destination bytes per rank, `3.34x` read amplification.
- 235B mapped target reads `195.97GB` source per rank to produce `57.97GB`
  destination bytes per rank, `3.38x` read amplification.
- owner-file collective has `collective_eligible=0` for the Qwen3 TP=8 mapped
  path, so the current collective lane does not attack the main work.

# Target End State

The target implementation has one strategy-selected TP source-window executor:

```text
MaterializationFacade
  -> ExecutionStrategyPlan or SourceBoundStrategyPlan
  -> SourceWindowCollectiveExecutor
  -> source-window plan
  -> TP rank striping read
  -> NCCL distribution
  -> scatter into target layout
  -> ExecutionCommitReport
```

The executor must support:

- ordinary disk `GPU <- DISK` TensorDict target;
- mapped target / binding target through `IntoTargetLayout`;
- safetensors source index and canonical source offsets;
- full, dim0, dim1, rect2d, concat-compatible typed work;
- local pad/fill lanes;
- explicit generic residual only in mixed mode.

Production code must not use the Qwen tensor-name classifier from
`tools/experiments/source_window_collective_feasibility.py`. That script is a
validation tool only. The production plan builder consumes
`RepresentationWorkPlan`, `DiskArtifactContext`, and `IntoTargetLayout`.

# Implementation Phases

## Phase 0: Document And Benchmark Guardrail

Status: completed for the pre-implementation feasibility gate. Production
executor implementation has not started.

Tasks:

1. Add design document:
   - `docs/designs/0122-tp-source-window-collective-realization.md`
2. Add plan document:
   - `docs/plans/0122-tp-source-window-collective-realization.md`
3. Update developer docs index.
4. Preserve the benchmark report:
   - `/data/tc/qwen3_30b_a3b_instruct_tensorcast_loader_report.md`
5. Define the acceptance benchmark matrix:
   - 30B SSD cold, JFS cold, tmpfs resident;
   - 235B SSD cold;
   - TP=8, bf16, `max_model_len=1024`, `HF_HUB_OFFLINE=1`;
   - daemon startup excluded from `LOAD_DONE`;
   - correctness generation check.
6. Add and run pre-implementation feasibility validation:
   - `tools/experiments/source_window_collective_feasibility.py`
   - `docs/plans/0122-01-source-window-collective-feasibility-validation.md`

Exit criteria:

- design and plan are committed to docs;
- existing benchmark numbers and target metrics are recorded;
- no code behavior changes yet.

Validation result:

- metadata-only planning matches existing TensorCast read profiles within about
  `0.5%` on read bytes for both 30B and 235B;
- estimated source-window read volume is `0.302x` of current local mapped for
  30B and `0.299x` for 235B;
- rank-striped SSD reads with `posix_fadvise(DONTNEED)` achieved `6.13GB/s`
  on 30B and `6.58GB/s` on 235B;
- the read-side feasibility gate is passed, with remaining risk in NCCL
  distribution, scatter, bounded buffering, and strategy integration.

## Phase 0.5: Collective And Scatter Feasibility Gate

Status: required before treating the runtime MVP as performance-relevant.

The completed Phase 0 validation proves the read-side shape. It deliberately
does not prove NCCL distribution or scatter overhead. Because the 235B read-only
result is already close to InstantTensor, this extra gate prevents a naive
runtime implementation from consuming the full performance margin.

Tasks:

1. Add a small synthetic source-window collective microbench:
   - TP=8 same-host clique;
   - 512MiB windows;
   - equal padded all-gather;
   - target scatter using `cudaMemcpyAsync` and `cudaMemcpy2DAsync`;
   - optional no-op scatter mode to isolate collective cost.
2. Feed it plan summaries derived from:
   - 30B feasibility metadata;
   - 235B feasibility metadata;
   - synthetic dim1-heavy and rect2d-heavy layouts.
3. Measure:
   - all-gather bandwidth;
   - peer bytes and peer waste;
   - scatter bytes/sec;
   - host pinned and GPU staging peak bytes;
   - one-window vs two-window overlap.
4. Decide the first runtime distribution mode:
   - `FullWindowAllGather` if overhead is within the acceptance budget;
   - `ConsumerRouted` or `HybridWindow` if all-gather peer waste consumes too
     much of the 235B margin.

Exit criteria:

- all-gather plus scatter overhead is quantified before Qwen3 mapped-target
  integration;
- first implementation has an explicit distribution mode decision;
- the plan records which follow-up optimization is mandatory if the MVP mode
  cannot meet the 235B target.

## Phase 1: Strategy Type And Config Cut

Files:

- `core/store/runtime/ingestion/materialization_strategy_types.h`
- `core/store/store_engine_options.h`
- `proto/tensorcast/config/v1/daemon_config.proto`
- `daemon/app/server_main.cc`
- `docs/internals/disk-load-strategy.md`

Tasks:

1. Add `ExecutionStrategyExecutor::kSourceWindowCollective`.
2. Add source-bound modes:
   - `kSourceWindowCollective`
   - `kSourceWindowCollectiveMixed`
3. Add cost estimate fields:
   - `source_window_bytes`
   - `rank_read_bytes_max`
   - `read_amplification_x1000` or equivalent typed metric
   - `scatter_op_count`
   - `source_window_count`
4. Add `CollectiveExecutionMetrics` extensions or a nested source-window
   metrics struct:
   - actual source window bytes
   - actual useful payload bytes
   - actual target write bytes
   - actual read amplification
   - actual scatter op count
5. Add typed daemon config:
   - `enable_source_window_collective`
   - `source_window_collective_batch_bytes`
   - `source_window_collective_max_gap_bytes`
   - `source_window_collective_max_amplification`
   - `source_window_collective_max_scatter_ops_per_window`
   - `source_window_collective_peak_bytes_budget`
   - `source_window_collective_min_saving_bytes`
   - `source_window_collective_max_peer_to_read_ratio`
   - `source_window_collective_distribution`
   - `source_window_collective_allow_mixed_residual`
6. Delete or deprecate any older environment-only toggles that duplicate the
   new typed config.

Tests:

- proto compile tests;
- C++ build for `//daemon:tensorcast_daemon`;
- unit tests for enum names and default config mapping.

Exit criteria:

- strategy config can represent the new executor without selecting it;
- all old behavior remains unchanged until the executor is wired.

## Phase 2: Source-Window Plan Builder

Files:

- new `core/store/replica/source_window_collective_plan.h`
- new `core/store/replica/source_window_collective_plan.cc`
- `core/store/replica/BUILD`
- tests under `core/store/replica`

Inputs:

- `RepresentationWorkPlan`
- `DiskArtifactContext`
- `IntoTargetLayout`
- TP group size and rank facts
- strategy config

Outputs:

- `SourceWindowCollectivePlan`
- `SourceWindowPlanSummary`

Minimal C++ shape:

```cpp
struct SourceWindowScatterOp {
  uint32_t consumer_rank;
  uint32_t source_index;
  uint64_t window_offset;
  uint32_t target_storage_index;
  uint64_t target_offset;
  uint64_t bytes;
  uint64_t src_pitch_bytes;
  uint64_t dst_pitch_bytes;
  uint64_t rows;
};

struct SourceWindowConsumerSpan {
  uint32_t consumer_rank;
  uint64_t window_offset;
  uint64_t bytes;
};

enum class SourceWindowDistributionMode {
  kAuto,
  kFullWindowAllGather,
  kConsumerRouted,
  kHybridWindow,
  kLocalOnly,
};

struct SourceWindowEntry {
  uint32_t source_index;
  uint64_t source_offset;
  uint64_t source_bytes;
  uint64_t useful_payload_bytes;
  SourceWindowDistributionMode distribution_mode;
  std::vector<SourceWindowConsumerSpan> consumer_spans;
  std::vector<SourceWindowScatterOp> scatter_ops;
};

struct SourceWindowCollectivePlan {
  std::vector<SourceWindowEntry> windows;
  uint64_t source_window_bytes;
  uint64_t useful_payload_bytes;
  uint64_t target_write_bytes;
  uint64_t peak_temporary_bytes;
  uint64_t rank_read_bytes_max;
  uint64_t peer_transfer_bytes_estimate;
  uint64_t peer_useful_bytes_estimate;
  uint64_t peer_waste_bytes_estimate;
  uint64_t residual_bytes;
  std::string plan_hash;
};
```

Tasks:

1. Lower full and replicated tensor work into contiguous scatter ops.
2. Lower dim0 tensor work into contiguous scatter ops.
3. Lower dim1 tensor work into source windows plus pitched scatter ops.
4. Lower rect2d work into source windows plus pitched scatter ops.
5. Lower concat-compatible fragments when source and target geometry is
   representable without generic fallback.
6. Preserve local pad/fill as typed local work, not residual data.
7. Produce true residual map only for unsupported bytes.
8. Produce consumer spans per window and rank from the work plan and target
   layout, without using tensor names.
9. Select an initial distribution mode per plan:
   - `FullWindowAllGather` for MVP-compatible dense windows;
   - `ConsumerRouted` or `HybridWindow` when peer waste dominates;
   - `LocalOnly` for windows consumed only by the producing rank.
10. Merge windows by source offset using the existing `SourceWindowScheduler`
   rules as a starting point:
   - max gap,
   - max amplification,
   - window cap.
11. Reject plans that exceed scatter op count, temporary memory, peer waste,
   or residual policy.
12. Compute a stable `plan_hash` from source windows, distribution modes,
   consumer spans, scatter ops, and residual summary. Runtime must verify this
   hash across ranks before target mutation.

Tests:

- plan builder full tensor case;
- dim0 case;
- dim1 narrow case with no full-row source read accounting;
- rect2d case with coalesced source windows;
- adjacent window merge;
- max amplification rejection;
- max scatter-op rejection;
- residual map correctness;
- no tensor-name hardcoding;
- plan hash changes when any window, distribution, or scatter field changes;
- distribution-mode selection for all-gather, routed, hybrid, and local-only
  synthetic cases.

Exit criteria:

- plan summaries predict read amplification near `1.0x-1.2x` for synthetic
  TP dim1/rect2d cases;
- Qwen3 trace-derived plan summary shows materially lower source bytes than
  the current local mapped plan;
- Qwen3 trace-derived plan summary includes peer useful/waste estimates, not
  only disk read estimates.

## Phase 3: Strategy Planner Integration

Files:

- `core/store/runtime/ingestion/materialization_facade.cc`
- `core/store/runtime/ingestion/source_bound_strategy_planner.cc`
- `core/store/replica/collective_disk_loader.h`
- new plan summarizer API from Phase 2

Tasks:

1. Add `summarize_source_window_collective(...)`.
2. Call it from ordinary disk strategy planning when:
   - GPU target;
   - safetensors or canonical source index present;
   - TP group present;
   - representation work plan present.
3. Call it from source-bound mapped strategy planning when:
   - `IntoTargetLayout` present;
   - disk source present;
   - collective group present;
   - local mapped plan would otherwise be selected.
4. Add cost-model comparison:
   - reject if residual policy fails;
   - reject if source-window read amplification exceeds threshold;
   - reject if peer transfer dominates expected disk savings;
   - reject or downgrade if peer waste dominates useful peer bytes;
   - reject if peak temporary bytes exceed budget;
   - prefer source-window collective when it reduces source bytes and task
     count compared with local mapped.
5. Make strict source-window collective fail before any local mapped or generic
   execution starts.
6. Record rejected candidate reasons in existing candidate diagnostics.

Tests:

- ordinary strategy chooses source-window when its estimate wins;
- ordinary strategy rejects source-window without group hint;
- mapped strategy chooses source-window over local mapped when read
  amplification is much lower;
- strict mode fails fast when residual exists;
- mixed mode retains residual in the lane plan;
- host-local SSD can still choose local/generic when cost model says so.
- all-gather can lose to routed distribution in a synthetic sparse-consumer
  plan without changing the public executor selection.

Exit criteria:

- executor can be selected in dry-run summary mode;
- no data movement path uses it yet unless explicitly enabled for tests.

## Phase 4: Collective Runtime MVP

Files:

- `core/store/replica/collective_disk_loader.h`
- `core/store/replica/collective_disk_loader.cc`
- new `source_window_collective_executor.cc` if the file needs to be split
- tests under `core/store/replica`

Runtime shape:

1. Join same-host clique using existing `NcclClique`.
2. Build or receive one `SourceWindowCollectivePlan`.
3. Allocate bounded per-rank host pinned buffers and GPU staging buffers.
4. For each window:
   - divide into aligned read stripes;
   - each rank reads its stripe from `SeekableSource`;
   - execute the selected distribution mode:
     - full-window all-gather,
     - routed send/recv or peer copy,
     - hybrid distribution,
     - local-only;
   - execute scatter ops targeting the current rank's `IntoTargetLayout`;
   - release temporary buffers before next window or batch.
5. Record timings and actual byte counters.

First implementation choices:

- Use buffered I/O by default because current profiling shows direct I/O is
  slower for this workload.
- Use NCCL all-gather for initial simplicity only when Phase 0.5 shows it
  preserves the acceptance margin. Keep the executor API distribution-mode
  aware from the first patch.
- Use `cudaMemcpyAsync` and `cudaMemcpy2DAsync` for scatter before adding a
  custom GPU scatter kernel.
- Keep one window in flight until correctness is stable, then add prefetch
  depth. A benchmarkable implementation should overlap read, distribution, and
  scatter with at least a double-buffer pipeline if the single-window pipeline
  misses the 235B target.

Tests:

- fake or CUDA test for 2-rank all-gather window assembly;
- fake or CUDA test for routed consumer-span distribution;
- single-window full tensor scatter;
- dim1 scatter with expected bytes;
- rect2d scatter with expected bytes;
- multiple windows and merge boundaries;
- short read failure propagates;
- mismatched group plan fails before target mutation where possible.
- rank plan-hash mismatch invalidates the target and fails before publication.

Exit criteria:

- small CUDA correctness test passes for full, dim0, dim1, rect2d;
- logs include `source_window_collective_plan` and
  `source_window_collective_timings`;
- actual source read bytes match plan accounting.

## Phase 5: Mapped Target Wiring For Qwen3

Files:

- `core/store/runtime/ingestion/materialization_facade.cc`
- `core/store/replica/collective_disk_loader.cc`
- `daemon/service/controllers/materialization_target_plan_utils.cc`

Tasks:

1. Wire selected source-window plan into
   `try_collective_mapped_target_load`.
2. Preserve local typed pad/fill execution in the existing target substrate.
3. Replace the current local mapped rect2d default when the source-window plan
   wins.
4. Ensure residual bytes are explicit and bounded.
5. Emit commit report:
   - `dominant_executor=SourceWindowCollectiveExecutor`
   - actual source read bytes
   - actual peer transfer bytes
   - actual peer useful bytes
   - actual peer waste bytes
   - actual target write bytes
   - read amplification
6. Keep old local mapped path available only as a losing candidate or explicit
   fallback until source-window correctness is proven.

Tests:

- mapped target unit tests with a synthetic `IntoTargetLayout`;
- existing collective disk loader tests;
- existing materialization target plan utils tests;
- source-bound planner tests.

Exit criteria:

- Qwen3-30B TensorCast mapped target selects source-window collective with
  expected source byte estimate;
- Qwen3-30B and 235B mapped target reports include source-window distribution
  mode and peer waste estimates;
- end-to-end load succeeds.

## Phase 6: Ordinary TensorDict / Runtime Realization Convergence

Files:

- `MaterializationFacade` ordinary disk path
- runtime attachment or binding realization path if still separate
- vLLM TensorCast adapter only for capability facts, not data path

Tasks:

1. Make ordinary TensorDict and mapped binding consume the same source-window
   strategy candidate.
2. Remove duplicate selection or fallback logic that conflicts with `0121`.
3. Ensure vLLM only contributes:
   - runtime host capabilities;
   - placement facts;
   - collective group facts;
   - source catalog facts;
   - target layout facts.
4. Do not add a vLLM-owned source-window loader or private daemon RPC.

Tests:

- TensorDict projection test;
- runtime attachment or binding projection test;
- vLLM loader config tests updated for new typed config;
- no old compatibility branch required.

Exit criteria:

- source-window collective is a realization strategy, not a separate loader.

## Phase 7: Benchmark, Profile, And Cleanup

Benchmarks:

- 30B:
  - JFS cold;
  - SSD cold;
  - tmpfs resident.
- 235B:
  - SSD cold.

Configuration:

- TP=8
- bf16
- `max_model_len=1024`
- `HF_HUB_OFFLINE=1`
- daemon startup excluded from `LOAD_DONE`
- fadvise/fincore for cold cases as before

Compare:

- vLLM default
- InstantTensor
- TensorCast local mapped old path
- TensorCast source-window collective

Acceptance targets:

- 30B SSD cold TensorCast weight load should approach InstantTensor `10.43s`.
- 235B SSD cold TensorCast weight load should approach InstantTensor `70.69s`.
- source-window lane read amplification should be no more than `1.2x` for the
  tested Qwen3 workloads.
- TensorCast must not regress generation correctness.

Correctness:

- 30B deterministic prompt must match default and InstantTensor output.
- 235B TensorCast generation must be coherent.
- Tensor parity tests must cover all lowerings in the source-window plan.

Cleanup:

1. Remove obsolete packed per-row rect2d experiment if source-window scatter
   supersedes it.
2. Narrow or delete owner-file code paths that no longer have a selected use.
3. Delete old env toggles and compatibility aliases.
4. Update `docs/internals/disk-load-strategy.md` with the new default matrix.
5. Update the benchmark report.

# Implementation Order

Recommended order:

1. Phase 1 config and strategy type patch.
2. Phase 2 source-window plan builder and tests.
3. Phase 3 dry-run strategy integration and diagnostics.
4. Phase 0.5 collective and scatter microbench using Phase 2 plan summaries.
5. Phase 4 runtime MVP on synthetic CUDA tests.
6. Phase 5 Qwen3 mapped target wiring.
7. Phase 7 benchmark loop on 30B.
8. Phase 5 and 7 hardening on 235B.
9. Phase 6 convergence into the unified realization target-set model.
10. Cleanup old behavior after benchmark evidence.

# Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| all-gather padding wastes too much bandwidth | add irregular send/recv or peer-copy window distribution after MVP |
| full-window all-gather wastes too many peer bytes on sparse consumer spans | keep consumer spans in the first plan representation and allow `ConsumerRouted` or `HybridWindow` before production defaulting |
| scatter op count is too high | merge scatter ops, add GPU scatter kernel, cap plan by cost model |
| source windows reintroduce read amplification | enforce max amplification and report actual vs planned bytes |
| temporary GPU memory is too high | bounded window bytes, one-window-in-flight MVP, explicit peak budget |
| single-window pipeline fails 235B target despite good read-side feasibility | add double-buffered read/distribute/scatter overlap before declaring the executor benchmark-ready |
| rank plans diverge and corrupt mapped targets | verify `plan_hash` across ranks before target mutation and invalidate targets on failure |
| host-local SSD collective loses to local read | keep cost model storage-class aware and do not force collective |
| mixed residual hides performance regressions | report residual bytes and keep strict mode for validation |
| old paths remain and confuse behavior | delete old experimental toggles after source-window evidence |

# Validation Commands

Expected local validation after each implementation phase:

```bash
bazel build //daemon:tensorcast_daemon
```

```bash
bazel test //core/store/replica:collective_disk_loader_test \
  //daemon:materialization_target_plan_utils_test \
  //core/store/runtime/ingestion:source_bound_strategy_planner_test
```

Add new targets as implementation lands:

```bash
bazel test //core/store/replica:source_window_collective_plan_test
bazel test //core/store/replica:source_window_collective_executor_test
```

End-to-end benchmark scripts should reuse the existing runners:

```bash
python /data/tc/run_tensorcast_predaemon_bench.py \
  --model-path /mnt/host0/vllm-loader-bench/qwen3-30b-a3b-instruct-2507 \
  --case-name qwen3-30b-a3b-instruct-source-window \
  --storage-label ssd_cold \
  --tp 8 \
  --max-model-len 1024
```

```bash
python /data/tc/run_vllm_generation_correctness.py \
  --model-path /mnt/host0/vllm-loader-bench/qwen3-30b-a3b-instruct-2507 \
  --load-format tensorcast \
  --tp 8 \
  --max-model-len 1024
```

# Closeout Criteria

This plan is complete only when all of the following are true:

1. `SourceWindowCollectiveExecutor` is selected by the strategy plane, not by
   vLLM or an environment-only switch.
2. Qwen3-30B and Qwen3-235B TP=8 benchmarks show source read amplification at
   or below `1.2x` for the source-window lane.
3. TensorCast weight-load time is in the same performance class as
   InstantTensor on SSD cold reads.
4. Correctness generation checks pass.
5. Diagnostics explain selected and rejected executor candidates.
6. Obsolete experimental compatibility paths are deleted or narrowed.
7. `docs/internals/disk-load-strategy.md` reflects the new default behavior.
