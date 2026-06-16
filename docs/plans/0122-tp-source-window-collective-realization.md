---
slug: tp-source-window-collective-realization-plan
title: TP Source-Window Collective Realization Plan
status: draft
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests", "benchmarks"]
created: 2026-06-15
last_updated: 2026-06-16
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
  The new source-window executor should add a separate
  `try_source_window_collective_mapped_target_load(...)` entry instead of
  overloading the current mapped collective request.
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

Status: completed for the pre-implementation feasibility gate and the first
runtime validation loop. The current production implementation slice now has
typed config, group planner, dry-run/pre-admission, group final admission,
facade integration, strict fail-closed execution, and a `FullWindowAllGather`
mapped-target runtime MVP plus `LocalOnly` owner-only execution. 30B/235B TP=8
source-window benchmarks have been run with daemon startup excluded. The MVP
plus persistent readers, direct-to-pinned source-window reads, 128MiB chunk
tuning, and multi-slot host read-ahead improves TensorCast local mapped
substantially but does not yet reach InstantTensor-class performance. The
current best source-window 30B weight-load log is `12.97s` from the prepared
full-window path after source-handle reuse, first-load source-window plan-cache
disablement, and ref-backed group input. The run measured `LOAD_DONE=52.81s`,
`key_sec=0`, `input_sec=0.00033s`, `build_sec=0.862s`, data-plane
`total=9.50s`, and daemon startup was excluded. Prepared ConsumerRouted
deferred-2D-pack measured `14.99s` / `51.32s`; it reduced routed scheduling
overhead but did not beat full-window. The best 235B source-window weight-load
run is now `79.67s` / `121.57s`; the best 235B `LOAD_DONE` sample remains
`87.17s` / `118.95s`. All are TP=8 with daemon startup excluded.
The latest compiled routed program + batched scatter path, with CPU-parallel
compiled-program construction, measures `13.71s` / `48.53s` on 30B SSD
first-load and `11.54s` / `33.92s` on 30B SSD same-daemon hot load. The
matching 235B SSD run measures `80.90s` / `116.98s` first-load and the prior
same-daemon hot load remains `76.84s` / `104.74s`. Correctness was revalidated
on the real 30B TP=8 model with the same prompt through vLLM default and
TensorCast, both producing the same coherent answer with `finish_reason=stop`.
The real WorkPlan staged-copy gate has also run on 30B and 235B; both reported
zero operation-count reduction, so the simple tensor-staged adjacent-coalescing
runtime MVP is rejected. The follow-up batched-scatter feasibility gate ran on
real 30B/235B plans and estimates that current copy launches can drop from
`151,685` to `3,605` on 30B and from `303,223` to `28,181` on 235B, so the
launch-count target is real. The first NVRTC descriptor-copy runtime experiment
was nevertheless rejected: it reduced launches but regressed 30B
`scatter_issue` from `1.12s` to `18.09s`. The default path keeps
`cudaMemcpyAsync/cudaMemcpy2DAsync`; any future batched scatter/pack primitive
must first prove copy-engine-equivalent bandwidth and host issue time in a
microbenchmark. A later routed final-scatter diagnostic fixed the runtime
cached-device state after clique synchronization and extended the default-off
batched kernel to ConsumerRouted final scatter. It reduced 30B JFS
ConsumerRouted `scatter_issue` from `6.36s` to `5.77s` and runtime total from
`9.43s` to `8.51s`, but `routed_pack_ops` stayed at `154k`; this confirms
batched final scatter is useful but not sufficient. The current high-leverage
work is group-level prepared realization/control and lower-waste routed/hybrid
distribution. The latest
prepared-key component diagnostic measured `13.09s` / `49.59s` on 30B TP=8 SSD
cold. It showed one shared target-layout-template hash and one shared
target-index hash across the group, but eight distinct
`BindingRealizationPlan` hashes. Therefore first-load prepared realization
cannot be fixed by trimming rank-local cache keys; the group builder must
collect rank-specific realization plans as member facts and normalize them into
one group template plus per-rank views.

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

# Current Implementation Slice

Implemented in the current code slice:

- typed daemon and engine config for `enable_source_window_collective`,
  selection mode, distribution mode, window sizing, amplification gates, peak
  budget, and residual policy;
- source-window metrics in `CollectiveExecutionMetrics` and
  `MaterializeIntoTargetResult`;
- group-level planner in `core/store/replica/source_window_collective_plan.*`
  consuming `RepresentationWorkPlan`, `DiskArtifactContext`,
  `IntoTargetLayout`, source index digest, and TP group facts;
- dry-run/pre-admission in `source_bound_strategy_planner.cc` that marks a
  source-window candidate without mutating lane mode;
- independent required-collective semantics for source-window. Under
  `COLLECTIVE_POLICY_REQUIRE_COLLECTIVE`, source-window strict/auto candidates
  now select `SourceWindowCollectiveExecutor` and reach group final admission
  instead of being rejected by the legacy owner-file `pure_collective` preflight
  gate;
- fast source work-plan rebind for source-window strict paths. When physical
  safetensors metadata is layout-compatible with the immutable representation
  contract, daemon planning now builds the work plan from the original contract
  and patches only work-plan source specs, avoiding a deep copy and patch of
  the whole `RepresentationTransformContract`;
- group final admission in `collective_disk_loader.cc` with group consistency
  checks, plan hash, strict/auto behavior, and explicit fallback diagnostics;
- mapped-target facade hook before local mapped fallback;
- correctness-first rank-striped `FullWindowAllGather` runtime in
  `try_source_window_collective_mapped_target_load(...)`: each rank reads a
  padded stripe from a bounded source window, stages it to GPU, NCCL
  all-gathers the assembled window to every rank, and scatters into each rank's
  `IntoTargetLayout`;
- multi-slot host read-ahead for source-window runtime. When the pinned pool can
  provide per-rank host slots, multiple future chunks can be read while earlier
  chunks are doing H2D, all-gather, and scatter. The path logs
  `requested_pipeline_slots`, `active_pipeline_slots`, read wait time, and
  summed read-job wall time. The current cap is 8 slots;
- persistent per-rank reader workers for source-window runtime. This removes
  per-chunk read thread creation and improves the 235B strict/direct read-ahead
  run from `115.08s` to `109.88s` max rank weight load;
- persistent read-ahead pump for source-window runtime. This removes
  per-chunk `std::async` coordinator creation. It improves 30B to `20.72s`
  max rank weight load and is neutral/noisy on 235B (`109.96s` versus
  `109.88s`);
- direct-to-pinned source-window reads for `DirectAlignedSafetensorsSource`.
  When a stripe stays within one safetensors payload segment, the runtime reads
  the aligned `O_DIRECT` file range directly into pinned host staging and
  adjusts the H2D source offset. This removes the scratch-buffer copy caused by
  unaligned safetensors payload starts, improving 30B to `18.94s` and 235B to
  `94.28s` max rank weight load;
- direct pinned-read fallback diagnostics in the source-window runtime. The
  executor now reports attempts, successes, fallback bytes, and fallback
  reasons (`unaligned_host`, `outside_segment`, `cross_segment`, `file_edge`,
  and `capacity`). On the current 235B full-window all-gather control run,
  `28194/28312` direct pinned reads succeeded, with `1.96GB` falling back at
  safetensors segment/file boundaries;
- explicit `ConsumerRouted` planner/runtime support behind the distribution
  config. Correctness passes and peer waste falls to zero. The initial 30B
  experiment regressed to `23.10s` because pack/scatter issue time rose to
  `9.76s`; after CUDA device caching, the forced ConsumerRouted rerun measures
  `17.61s`/`50.07s` with only `53.7GB` peer transfer, but
  `routed_pack_ops=147585`, `routed_remote_pieces=126207`, and
  `scatter_issue=5.59s` still erase the peer-byte saving;
- source-window runtime CUDA device-switch caching and lazy scatter device
  selection. On 30B, the earlier `ConsumerRouted` profile improved to `19.09s`
  max-rank weight load with `scatter_issue=5.80s`; full-window all-gather
  scatter issue improves to `1.22s` and `LOAD_DONE=49.84s`. The 235B
  lazy-device rerun measured
  `98.87s`/`133.99s` because read time regressed to `80.05s`, so it does not
  replace the `94.28s`/`128.41s` direct-to-pinned best. A producer-batched
  routed-pack experiment was measured and rejected because it regressed
  end-to-end time. Runtime chunk consumer-span prefiltering was also measured
  and reverted: on 30B SSD cold it produced `19.16s` max-rank weight load,
  `53.39s` `LOAD_DONE`, `scatter_issue=1.24s`, and no improvement over the
  lazy-device all-gather profile;
- `LocalOnly` planner/runtime support for windows consumed only by the owner
  rank. The group planner rejects a local-only window with remote consumers,
  and the runtime reads only the owner rank's source chunk, stages it to that
  rank's GPU buffer, and scatters only into owner-owned target spans. This is
  primarily a consistency building block for later hybrid/residual plans, not a
  new 30B/235B performance result;
- planner-level scatter span coalescing for adjacent linear spans, horizontally
  adjacent 2D column spans, and vertically adjacent 2D row-band spans. The
  coalesced consumer spans become the source-window scatter program used for
  plan hash, admission, and runtime execution, and
  `source_window_scatter_op_count` now reflects coalesced consumer spans
  instead of raw WorkPlan fragments. The row-band extension has a focused unit
  test and is retained for correctness, but the 30B HybridWindow 64MB rerun
  shows it is nearly neutral on Qwen3 (`17.43s`/`49.04s`, runtime scatter ops
  about `153714` versus the previous `153754`). This does not yet replace the
  runtime with a custom GPU scatter kernel or a coalesced routed/hybrid data
  distribution path;
- distribution `auto` now selects `LocalOnly` when every planned source window
  has consumers from exactly one rank. Otherwise `auto` remains
  `FullWindowAllGather`; `ConsumerRouted` still requires explicit selection
  until the routed pack/scatter path is cost-gated and benchmark-proven;
- `HybridWindow` now has a first cost-gated per-window implementation: the
  group plan records each window's distribution mode, owner-only windows
  execute as `LocalOnly`, dense multi-rank windows stay on
  `FullWindowAllGather`, and sparse multi-rank windows execute as
  `ConsumerRouted` only when spans are linear/contiguous and the routed peer
  saving clears `source_window_collective_min_routed_peer_saving_bytes`
  (default `64MB`). The runtime dispatches each chunk from its window
  distribution mode. This establishes the mixed-mode plan/runtime shape while
  keeping `auto` on the benchmark-proven all-gather path until routed is
  benchmark-proven for the real Qwen3 TP=8 plans. The latest explicit
  HybridWindow reruns route only `1/115` windows on 30B (`17.60s`/`48.49s`) and
  `4/885` windows on 235B (`89.10s`/`126.17s`); the same-code 235B
  full-window all-gather control with 8-slot read-ahead is faster in `LOAD_DONE`
  at `87.17s`/`118.95s`;
- daemon-side canonical/source index parsing now uses a bounded shared parsed
  index cache in `materialization_layout_utils`. This is an artifact metadata
  cache shared by source-window, local mapped, and other daemon planning paths,
  not a vLLM-private loader branch. On the 30B SSD cold historical HybridWindow 4GB-gate
  profile it reduced `prepare_source_bound_plan` mean time from `1.197s` to
  `0.805s` and `parse_source_tables` mean time from `0.626s` to `0.043s`.
  Max-rank weight load improved only from `19.22s` to `19.07s`, so this
  removes a real planner cost but does not change the main data-plane and
  orchestration bottlenecks;
- the parsed-index cache now coalesces concurrent same-key parses, and
  source-bound prepare starts physical source-index parsing before execution
  plan construction. This keeps the optimization in TensorCast's artifact
  metadata layer while overlapping the hot execution-planning tail. On the 30B
  SSD cold historical HybridWindow 4GB-gate rerun, `physical_source_table_preparsed=1`,
  `parse_source_index_sec=0`, max `physical_source_index_wait_sec=0.000135s`,
  and `resolved_mapped_execution_plan total_sec` max fell to `0.279s`. The fair
  end-to-end result was `18.79s`/`49.28s`, so the change is retained for
  consistency and planner latency but does not replace the latest `17.29s`
  best;
- the Python artifact runtime now builds `SourceCatalog` directly from
  daemon-attested canonical-index bytes for manifest/selected-safetensors
  paths, instead of round-tripping through `CanonicalIndexEntry` dataclasses
  when only catalog facts are needed. A synthetic 20k-entry canonical-index
  microbench measured old avg `0.153s` versus new avg `0.141s` (`1.08x`), and
  `tests/python/artifact_runtime/test_source.py` validates parity with the
  dataclass path plus malformed JSON rejection. Keep this as low-risk
  artifact-boundary cleanup and fail-fast validation, not as a root fix for the
  TensorCast/InstantTensor gap;
- `AdminLocalSourceBootstrap` now carries a prepared `source_catalog` through
  `_LocalReadyBootstrap`, and `_local_ready_source_catalog` validates and uses
  it before consulting `IntegrationHost.source_catalog`. This exposes the
  existing internal prepared-catalog hook at the admin/prewarm lifecycle
  boundary, preserves the public `LocalSourceBootstrap` surface, and keeps
  source identity validation mandatory. Targeted tests prove the prepared
  catalog skips the provider and still drives local-ready recipe/materialization
  preparation;
- local-ready bootstrap now validates prepared recipe identity before
  materialization starts. A prepared recipe's `source_artifact_ref` must match
  the request source ref and any `SourceSubject.artifact_ref`; when a prepared
  source catalog is present, its artifact identity and metadata fingerprint must
  match the recipe. Targeted tests prove mismatches fail before
  `realize_local_ready_binding_from_source`, giving future prepared
  source/catalog/recipe flows a fail-before-mutation guard;
- `ArtifactRuntimeIntegration.prepare_local_ready_recipe(...)` now exposes the
  same local-ready admission, source resolution, source catalog, recipe cache,
  and identity-preflight path without target materialization. vLLM
  `prime_model_load()` uses this API for `source_bootstrap_to_binding` only,
  skips durable artifact and retained-binding modes before planner authority
  validation, and connects through the configured runtime before source
  resolution. Source-subject resolution expects the TensorCast daemon/RPC
  endpoint to be available, so benchmark runs should keep using the
  prestarted-daemon protocol and configure runtime connect mode when daemon
  startup time must stay outside loader timing. This makes recipe prewarm an
  artifact-runtime lifecycle capability rather than a vLLM-private loader
  branch;
- mounted-source model-runtime realization can now accept the prepared
  local-ready result through the artifact realization path. The core
  `realize_mounted_source_model_runtime(...)` reuses the prepared recipe,
  source catalog, cache config, source subject, and placement only after the
  normal mounted-source artifact identity is known, so the existing prepared
  recipe/source/catalog preflight still gates target mutation. vLLM stores a
  prime result on the loader instance and passes it to
  `artifact.realize(..., runtime_prepared_local_ready=...)` only when source
  path, model hash, and target device match;
- TensorCast store now exposes `from_resolved_public_disk_source(...)` for
  daemon-attested `PublicDiskSourceHandle` values. vLLM uses it only when a
  matching prepared local-ready result carries a public disk source subject
  whose path and checksum policy match the current source-bootstrap request.
  This avoids a second `ResolvePublicDiskSource` RPC while keeping the
  artifact-centered `Artifact.realize(...)` boundary and prepared identity
  preflight. The 30B TP=8 SSD cold prepared full-window rerun measured
  `14.63s` weight load and `47.27s` `LOAD_DONE`, with
  `startup.from_prepared_disk_source` avg `0.093s`;
- source-window group final admission now reports `input_sec`, `key_sec`,
  `lookup_sec`, `build_sec`, and `cache_store_sec`. The first profile showed
  `plan_sec=2.18s`, with `key_sec=1.07s` from re-walking 150k WorkPlan
  fragments for a prebuild cache lookup. Compact key encoding only reduced
  that to `1.01s`, so first-load source-window plan cache is now disabled by
  default and only enabled explicitly for repeated-load tests. The 30B rerun
  measured `key_sec=0`, `plan_sec=1.108s`, and `13.78s` weight load;
- source-window group input is now ref-backed in runtime. The group planner can
  consume `shared_ptr<const RepresentationWorkPlan>` and
  `shared_ptr<const IntoTargetLayout>` while tests can still construct value
  inputs. Runtime final admission no longer copies eight large WorkPlans into
  the group input; `input_sec` fell from `0.255s` to `0.00033s`, `plan_sec` to
  `0.862s`, and the 30B SSD cold prepared full-window rerun measured `12.97s`
  weight load;
- prepared local-ready was validated on 30B TP=8 SSD cold with the prestarted
  daemon harness. Full-window all-gather improved the vLLM `Loading weights
  took` max to `14.69s` and `LOAD_DONE=48.59s`, with
  `prime.source_bootstrap.prepare_recipe=3.14s` average still counted in
  `LOAD_DONE`. ConsumerRouted reduced peer transfer from `427.45GB` to
  `53.75GB` with zero peer waste, but regressed to `15.59s`/`49.93s` because
  routed execution still issues about `143k` pack operations, `104k` remote
  pieces, and `61029` device switches. A deferred-2D-pack scheduling follow-up
  reduced device switches to `42622` and runtime total from `10.31s` to
  `9.81s`, but the visible result was only `14.99s`/`51.32s`. HybridWindow
  selected 114/115 full-window windows and measured `15.04s`/`55.71s`. This
  proves prepared source/recipe reuse is a valid TensorCast lifecycle step,
  while the next performance lever must be group-level prepared realization
  plus coalesced routed/batched scatter, not more local parser/cache tuning or
  reorder-only routed pack tweaks;
- a canonical-index preparse future inside rank-local binding realization was
  benchmarked and rejected. It reported `canonical_index_table_preparsed=1` and
  reduced `prepare_plan_sec` max to `0.906s`, but the builder still waited
  `0.23-0.28s` on the future and the fair 30B SSD cold result regressed to
  `17.62s`/`50.69s`. The code was reverted. The next control-plane cut should
  remove duplicated per-rank representation/materialization planning with a
  group-level artifact realization plan, not add more per-rank preparse futures;
- rank-local mapped-plan/execution-template cache singleflight was measured on
  30B TP=8 SSD cold. The run completed with `17.06s` max-rank weight load and
  `50.40s` `LOAD_DONE`, while every rank logged
  `mapped_plan_cache_hit=0`, `mapped_plan_cache_waited=0`,
  `execution_template_cache_hit=0`, and `execution_template_cache_waited=0`.
  This is expected because the current cache keys include rank-local target
  layout/index and TP topology facts. Treat singleflight as a same-key
  diagnostic guard, not the Qwen3 source-window performance path;
- binding realization now reuses the canonical parsed source table when no
  non-identity view changes the source byte-space. The 30B HybridWindow 64MB
  rerun confirmed `source_table_reused=1` on all TP ranks, but measured
  `17.83s`/`49.31s`, with runtime data-plane total `10.19s` and vLLM-visible
  startup max `17.82s`. Keep this as a safe artifact-metadata cleanup; it is
  not a new acceptance baseline;
- a stream-ordered all-gather/scatter experiment was measured and reverted. It
  removed the explicit post-all-gather `synchronize_all()` and relied on CUDA
  stream ordering before scatter. The 30B HybridWindow 64MB run measured
  `17.81s`/`50.62s` with runtime total `10.04s`; the 235B full-window control
  measured `89.37s`/`122.39s` with runtime total `74.14s`. The wait moved from
  `collective_sync` into final `scatter_sync`, so this is not the runtime
  bottleneck;
- source-window plan hashing now reserves the expected payload size before
  appending member, window, and consumer-span facts. This preserves the stable
  plan-hash inputs while trimming 30B `hash_sec` from about `0.294s` to
  `0.278s`; the benchmark-level result stayed inside run-to-run noise;
- source-window plan hashing now has a v2 execution-fact payload. It hashes
  source identity, group world size, member target layout, summary metrics, rank read
  bytes, windows, per-window distribution mode, and consumer scatter spans,
  instead of also serializing the full original `RepresentationWorkPlan` after
  those facts have been lowered into windows. On the 30B SSD cold HybridWindow
  rerun, `hash_sec` fell to `0.126s`, planner `total_sec` fell to `0.934s`,
  and group `plan_sec` fell to `1.198s`. The fair run measured
  `17.74s`/`51.40s`, so this is retained as a control-plane improvement but
  does not by itself solve the total startup tail;
- source-window plan hash no longer includes the runtime `group_id`. The
  `group_id` is still used for group assembly and log correlation, but it is
  not an execution fact and must not make identical source/target/layout plans
  hash differently across runs. Unit coverage now verifies source index digest,
  member order, target layout, distribution mode, and span changes remain
  hash-sensitive while changing only `group_id` keeps the same hash;
- strict source-window mapped execution-template cache keys now also exclude
  runtime `group_id` while retaining source locality, sharing domain, TP world
  size, rank, strategy config, source metadata, and target/materialization
  plan identity. This does not change group assembly; it only makes daemon
  prepared execution templates reusable across runs or explicit prebuilds when
  the actual source/target/layout facts are identical. Unit coverage verifies
  the stable key ignores only `group_id`; including runtime group id still
  changes the key, and rank changes remain key-sensitive;
- a compact binary plan-hash payload was benchmarked and rejected because it
  regressed 30B `hash_sec` to `0.565s` and plan build total to `1.388s`.
  Keep the text payload with explicit reserve for now; future work should
  reduce the number of hashed facts or introduce incremental structural
  digests instead of only changing serialization format;
- replacing the planner metrics interval buckets with `absl::flat_hash_map`
  was benchmarked and rejected because it regressed 30B `metrics_sec` from
  `0.278s` to `0.430s` and plan build total from `1.104s` to `1.408s`.
  Future planner work should combine or cache interval reductions instead of
  only changing associative container implementations;
- source-window participant copy-elision is kept. The leader moves assembled
  participants out of group state, and source-window runtime avoids copying
  `RepresentationWorkPlan`/`IntoTargetLayout` into the scatter-only runtime
  participant view. The 30B SSD cold run improved to `17.75s` max-rank weight
  load with `runtime_total=9.83s`;
- source-window window building now avoids the intermediate grouped-span vector
  while preserving exponential `consumer_spans` capacity growth. The first
  one-pass rewrite was rejected because exact-size repeated `reserve()` calls
  regressed `windows_sec` to `1.329s`. The retained reserve-fix run measured
  `windows_sec=0.368s`, planner `total_sec=0.877s`, group `plan_sec=1.129s`,
  and `17.29s`/`49.93s` fair 30B SSD cold timing;
- a borrowed planner member-input experiment was benchmarked and reverted. It
  reduced `plan_sec` but regressed the vLLM-visible run to `23.30s`, so the
  group-plan input remains value-based while the runtime copy-elision stays;
- strict source-window fallback behavior: strict request, missing candidate,
  unsupported source type, invalid target layout, group timeout, or plan hash
  mismatch fail before target publication instead of silently falling back to
  local mapped;
- strict source-window fast-prep path in daemon planning: skip byte-range map
  construction, skip source layout map composition, use compact coverage, and
  key realization-plan cache entries by source-window config;
- production WorkPlan and runtime validation on 30B and 235B TP=8 using the
  real vLLM TensorCast mapped-target path. 30B strict/direct reports `1.0x`
  read amplification, `61,064,245,248` group disk-read bytes,
  `8,047,610,880` max rank-read bytes, `25,296,662,528` local mapped physical
  baseline bytes, about `150k` coalesced scatter ops, zero residual bytes, and
  a best measured `16.81s` max rank weight load after direct-to-pinned reads,
  source-window participant copy-elision, plan-hash v2, the window-builder
  reserve fix, and multi-slot read-ahead. 235B
  strict/direct reports `1.0x` read amplification, `470,187,269,120` group
  disk-read bytes, `59,054,668,288` max rank-read bytes,
  `196,952,218,624` local mapped physical baseline bytes, and zero residual
  bytes. The current same-code full-window all-gather control with 128MB chunks
  and 8-slot read-ahead measures `87.17s` max rank weight load and `118.95s`
  `LOAD_DONE`, with runtime direct-read throughput near the measured SSD
  direct-read ceiling;
- vLLM-side boundary tests in `/data/workspace/internal-vllm` that keep the
  integration artifact-centered: TP placement facts map into TensorCast runtime
  placement, unavailable collective context is surfaced to TensorCast and does
  not request a collective execution context, and `materialization.collective`
  typed config round-trips into the runtime profile without adding a
  vLLM-private source-window branch;
- ordinary `tensor_dict` projection report coverage that keeps execution
  topology facts in the shared realization report model. The test
  `test_tensor_dict_realization_reports_execution_topology_policy` verifies
  that `GetArtifactOptions.execution_topology` reaches the materialization
  pipeline and is projected into `strategy_plan.source_policy`, rather than
  being handled through a separate TensorDict-only control path;
- runtime attachment projection report coverage for the same facts. Direct
  runtime bind and reload paths now pass the materialization options used for
  daemon binding into `report_for_runtime_attachment(...)`. The test
  `test_runtime_attachment_resolved_report_projects_execution_topology_options`
  verifies that runtime attachment reports project retrieval, collective
  policy, source locality, and source sharing domain into
  `strategy_plan.source_policy`;
- admission baseline correction so source-window is compared against current
  local mapped physical reads derived from WorkPlan geometry, not an ideal
  local payload estimate;
- tests for config parsing, planner geometry/hash behavior, local-only
  admission/rejection, source-bound pre-admission, facade selection, group
  final admission, and optional two-GPU correctness tests for all-gather,
  routed, and local-only source-window runtime paths.

Not yet completed:

- deeper async IO and GPU stream overlap beyond the current 2-slot host
  read-ahead, persistent reader workers, persistent read-ahead pump, and
  direct-to-pinned read path;
- runtime/GPU scatter coalescing beyond the current planner-level adjacent span
  coalescing and `cudaMemcpy2DAsync` path;
- coalesced ConsumerRouted performance path and routed/hybrid auto admission
  beyond the owner-only `LocalOnly` auto case and explicit cost-gated
  `HybridWindow` implementation;
- deeper runtime attachment execution convergence beyond the currently
  validated vLLM capability-facts boundary and TensorDict/runtime-attachment
  execution-topology report projection;
- overlap/double-buffer pipeline beyond the current single chunk/window
  execution path.

## Phase 0.25: Real WorkPlan Dump Validation

Status: complete for 30B and 235B TP=8 strict source-window validation.

The Phase 0 feasibility script intentionally used safetensors headers and
Qwen/vLLM naming conventions to validate the hypothesis quickly. Production
code must not use that classifier. Before runtime benchmarking, validate
against real TensorCast `RepresentationWorkPlan` data from the existing vLLM
TensorCast path.

30B production strict/direct result:

- model: `/mnt/host0/vllm-loader-bench/qwen3-30b-a3b-instruct-2507`
- TP=8, daemon startup excluded;
- `LOAD_DONE`: `55.428s`;
- max rank weight load: `23.404s`;
- runtime data-plane total: `14.527s`;
- runtime read time: `11.730s`;
- group disk read bytes: `61,064,245,248`;
- max rank disk read bytes: `8,047,610,880`;
- local mapped physical read baseline, max rank: `25,296,662,528`;
- rank read saving: `17,249,051,648`;
- target write bytes: `61,444,685,824`;
- read amplification: `1.0x`;
- scatter ops: `154,057`;
- residual bytes: `0`.

30B production strict/direct read-ahead result:

- `LOAD_DONE`: `52.814s`;
- max rank weight load: `21.269s`;
- runtime data-plane total: `12.458s`;
- runtime read time: `11.946s`;
- active pipeline slots: `2`;
- chunk bytes: `67,108,864`;
- chunks: `913`.

235B production strict/direct result:

- model: `/mnt/host0/vllm-loader-bench/qwen3-235b-a22b-instruct-2507`
- TP=8, daemon startup excluded;
- `LOAD_DONE`: `154.284s`;
- max rank weight load: `121.379s`;
- runtime data-plane total: `103.272s`;
- runtime read time: `92.025s`;
- group disk read bytes: `470,187,269,120`;
- max rank disk read bytes: `59,054,668,288`;
- local mapped physical read baseline, max rank: `196,952,218,624`;
- rank read saving: `137,897,550,336`;
- target write bytes: `471,676,936,192`;
- read amplification: `1.0x`;
- scatter ops: `313,709`;
- residual bytes: `0`.

235B production strict/direct read-ahead result:

- `LOAD_DONE`: `146.933s`;
- max rank weight load: `115.080s`;
- runtime data-plane total: `96.484s`;
- runtime read time: `94.545s`;
- active pipeline slots: `2`;
- chunk bytes: `67,108,864`;
- chunks: `7,074`.

The implementation progression was:

- naive fragment expansion: about `262s` and not viable;
- guarded residual fallback: about `1s`, but about `20.535GB/group` residual;
- compressed 2D plus 2D-source to 3D-target expert spans: about `1.1-1.4s`,
  guarded residual eliminated, residual reduced to about `1.208GB/group`.
- typed local pad/fill work and strict source-window coverage-only planning:
  residual reduced to zero for 30B and 235B strict runs;
- skip byte-range map construction for strict source-window: 30B max rank
  weight load improved from `52.87s` to `25.53s`;
- skip source layout map composition and use the configured direct source
  factory: 30B max rank weight load improved to `23.40s`.
- 2-slot host read-ahead: 30B improved to `21.27s`; 235B improved from
  `121.38s` to `115.08s`.
- persistent per-rank readers: 235B improved from `115.08s`/`146.93s` to
  `109.88s`/`141.43s`; 30B was neutral/noisy at `21.43s`/`53.86s`.
- persistent read-ahead pump: 30B improved to `20.72s`/`52.86s`; 235B was
  neutral/noisy at `109.96s`/`141.77s`.
- direct-to-pinned source-window reads: 30B improved to `18.94s`/`50.67s`;
  235B improved to `94.28s`/`128.41s`; 235B runtime read throughput reached
  about `6.29GB/s`, close to the measured SSD O_DIRECT upper bound.
- direct pinned-read fallback diagnostics: current 235B full-window all-gather
  control succeeds on `28194/28312` pinned direct reads, with the remaining
  `1.96GB` fallback bytes explained by safetensors segment and file edges.
- explicit ConsumerRouted: 30B peer transfer fell from `427GB` to `53.7GB`.
  The initial run regressed to `23.10s`; the latest forced rerun reaches
  `17.61s`/`50.07s`, but `routed_pack_ops=147585` and `scatter_issue=5.59s`
  keep it behind the all-gather/hybrid best.
- HybridWindow 64MB routed gate: routed windows are now reachable, but the
  latest 30B and 235B runs route only `1/115` and `4/885` windows respectively.
  30B measures `17.60s`/`48.49s`; 235B measures `89.10s`/`126.17s`. The
  same-code 235B full-window all-gather control remains faster at
  `88.52s`/`119.79s`.
- 128MiB engine slice/chunk tuning was measured and rejected for now: 30B
  regressed to `21.56s`/`54.41s`, and 235B regressed to `117.28s`/`149.68s`.

Latest feasibility refinement, based on real production artifacts:

- The 30B TP=8 warm-cache trace was re-parsed from the cached
  `BindingRealizationPlan` protobuf, the production `trace_plan` dumps, and
  safetensors headers. This avoids Qwen naming heuristics and verifies the
  geometry that the daemon actually consumes.
- Every TP rank has `18,867` copy entries, `0` fill entries, `18,626`
  source slices, `435` destination tensors, and about `7.68GB` of copy bytes.
  Across the group this is `150,936` copy entries and about `61.45GB` of
  realized bytes. Source slice bytes are about `7.655GB` per rank:
  `5.138GB` from dim0 slices, `2.517GB` from dim1 slices, plus about `26MB`
  of full-source small tensors.
- The group source-window plan reports `work_items=150176`,
  `compressed_2d_spans=148240`, `expanded_work_items=2696`,
  `candidate_spans=150936`, `windows=115`, `group_disk_read_bytes=61.064GB`,
  `read_amplification_x1000=1000`, and no residual bytes. Runtime reports
  `scatter_ops=151985`, `read=7.41679s`, `collective_issue=0.143561s`,
  `collective_sync=0.682566s`, `scatter_issue=1.18386s`, and total
  `9.49568s`.
- This corrects the earlier working hypothesis: current scatter count is
  essentially recipe/source-tensor granularity plus storage-boundary effects,
  not an accidental `O(window_count * rows)` explosion. A simple tensor-staged
  adjacent-span coalescer is therefore not enough; the winning shape must
  execute the real `~150k` tensor/rectangle copies with far fewer GPU launches
  and reduce full-window all-gather peer waste (`427GB` peer bytes for
  `53.8GB` useful peer bytes), not just merge window fragments.
- Control-plane breakdown for the same fair warm-cache run:
  `store.from_disk` mean `1.60s`, `build_source_catalog` mean `0.46s`,
  `build_recipe` mean `0.48s`, `source_window plan_sec=1.12s`,
  `local_ready.realize.realize_from` mean `12.87s`, and
  vLLM-visible `startup.artifact_runtime.start` mean `16.12s`.
  Optimizing all repeated source/catalog/recipe/group-plan work would likely
  move the 30B run toward the low-`12s`, but matching InstantTensor also
  requires a lower-waste distribution/materialization shape.
- Prepared source-handle reuse updates the source acquisition portion of that
  breakdown: the same source-bootstrap path now reports
  `startup.from_prepared_disk_source` avg `0.093s` instead of the earlier
  `store.from_disk`/`startup.from_disk` second resolve. The remaining
  `startup.artifact_runtime.start` time is therefore dominated by realization
  and group runtime, not mounted-source acquisition.

Optimization boundary from this refinement:

| Class | Examples | Plan decision |
|---|---|---|
| Easy and already worth keeping | Direct-to-pinned reads, auto direct IO policy, one-time source-window IO decision, source catalog/fingerprint fast paths, target-storage fast path, request shared refs, ref-backed source-window group input, default first-load skip of source-window prebuild cache lookup. | Keep as correctness-preserving cleanups. Further variants in this class need a profile showing they move more than noise. |
| Medium and lifecycle-dependent | Explicit source-window group plan cache, warm recipe/template reuse, `tc.from_disk` resolve/import caching. | Continue only when it strengthens explicit prepared source/catalog/recipe/group-plan lifecycle. It mostly helps repeated loads or user-visible prewarm, not first cold data-plane time. Cache-key re-encoding alone is not worth further work after the compact-key experiment, and the prepared-key component diagnostic shows rank-local source-window keys still split because `BindingRealizationPlan` differs per rank. |
| High-return structural work | Group-prepared artifact realization that collects rank-specific `BindingRealizationPlan` member facts and derives one group template plus per-rank views; lower-waste ConsumerRouted/HybridWindow with batched pack/scatter; InstantTensor-like deep ring scheduling inside the TensorCast daemon while preserving target-layout realization. | Prioritize. These are the only paths that can plausibly close the remaining 30B/235B gap without breaking TensorCast abstractions. |
| Conditional work | Batched final scatter or routed pack/scatter. | Implement only after a microbench proves copy-engine-equivalent bandwidth and host issue time. The rejected NVRTC descriptor copy kernel proves launch-count reduction alone is insufficient. |
| Low-return or fixed work | Daemon startup, group join, source setup/open, Python attachment, binding creation, CUDA handle restore/finalize, target pointer lookup, raw full-window NCCL bandwidth. | Stop optimizing unless new profiles put them on the critical path. Daemon startup is excluded from the benchmark by design. |
| Destructive or invalid work | vLLM-private source/load branches, bypassing artifact identity or recipe preflight, non-contiguous source views that vLLM loaders cannot consume, full-target coverage shortcuts that skip typed local pad/fill. | Do not do. These would trade TensorCast consistency or correctness for benchmark-only gains. |

The latest ConsumerRouted deferred-2D-pack result belongs in the "easy but
bounded" class: it reduces device switches and routed runtime total, but does
not improve end-to-end load. That confirms the next routed step must reduce
the dominant 2D pack/final-scatter operation shape, not add another local
ordering pass.

Tasks:

1. Keep the debug/profile path that records compact summaries of the real
   `RepresentationWorkPlan` for Qwen3 30B and 235B TP=8 mapped target runs:
   - item kind;
   - `WorkPartitionKind`;
   - source index and source offsets;
   - source fragment geometry;
   - destination geometry and target storage spans;
   - residual fallback bytes.
2. Use the production `summarize_source_window_collective(...)` API in dry-run
   or strict mode against those dumps.
3. Verify that source-window read/rank estimates reproduce the Phase 0
   metadata estimates within the same tolerance:
   - about `8.05GB` max read/rank for the current 30B production strict plan;
   - about `59.05GB` max read/rank for the current 235B production strict
     plan.
4. Explicitly validate rect2d inference from
   `RepresentationWorkSourceFragment` geometry. `WorkPartitionKind` currently
   has replicated, dim0, and dim1; there is no explicit `kRect2d` marker to
   rely on.

Exit criteria:

- production summarization can explain the Qwen3 read amplification without
  tensor-name rules;
- residual bytes and unsupported fragment shapes are visible in diagnostics;
- Phase 0.5 collective/scatter microbench can consume production plan
  summaries, not only script-derived metadata.

## Phase 0.5: Collective And Scatter Feasibility Gate

Status: initial production-shaped microbench implemented and measured for the
full-window all-gather lower bound; production-plan ingestion into the
microbench remains future work.

The completed Phase 0 validation proves the read-side shape. It deliberately
does not prove NCCL distribution or scatter overhead. Because the 235B read-only
result is already close to InstantTensor, this extra gate prevents a naive
runtime implementation from consuming the full performance margin.

Tasks:

1. Add a small synthetic source-window collective microbench:
   - TP=8 same-host clique;
   - 512MiB windows;
   - rank-striped equal padded all-gather;
   - target scatter using `cudaMemcpyAsync` and `cudaMemcpy2DAsync`;
   - optional no-op scatter mode to isolate collective cost.
2. Feed it production plan summaries derived from:
   - Phase 0.25 30B real WorkPlan dumps;
   - Phase 0.25 235B real WorkPlan dumps;
   - synthetic dim1-heavy and rect2d-heavy layouts.
3. Measure:
   - all-gather bandwidth;
   - peer bytes and peer waste;
   - scatter bytes/sec;
   - host pinned and GPU staging peak bytes;
   - one-window vs two-window overlap.
4. Decide the first runtime distribution mode:
   - rank-striped `FullWindowAllGather` if overhead is within the acceptance
     budget;
   - coalesced/cost-gated `ConsumerRouted` or `HybridWindow` if all-gather
     peer waste consumes too much of the 235B margin.

Current evidence:

- `bazel build //core/store/materialization/benchmarks:collective_transform_microbench`
  passes after adding `source_window_allgather_no_scatter` and
  `source_window_allgather_scatter`.
- Correctness passes for TP=2 and TP=8 small cases with `--check=true`.
- TP=8, BF16, 512MiB synthetic window, H800 same-host:
  - no-scatter: `nccl_avg_sec=0.00325537`, unique-window throughput
    `153.6GiB/s`, aggregate receiver throughput about `1.23TiB/s`;
  - scatter: `nccl_avg_sec=0.00302353`,
    `receiver_transform_avg_sec=0.000112516`, `total_avg_sec=0.00313622`.
- The microbench also supports `--source_window_scatter_ops_per_rank` to model
  production scatter launch count. With `165` scatter ops/rank/window
  (`1320` group scatter ops/window, close to the 30B production
  `151985 / 115` group-op density), the same 512MiB run measures
  `receiver_transform_avg_sec=0.00391022` and `total_avg_sec=0.00693844`.
- Interpretation: the NCCL full-window all-gather primitive and one coarse
  `cudaMemcpy2DAsync` scatter per rank are not the current InstantTensor gap.
  Scaling the production-op-count synthetic result to the real 30B `115`
  windows gives roughly `0.8s`, which is close to the measured production
  `scatter_issue+sync=1.19s` and confirms that launch granularity is a real
  but bounded cost. The remaining gap is therefore production execution shape:
  disk/read and host staging, repeated planning/preparation, and the
  `150k`-piece scatter program, not the raw NCCL all-gather bandwidth alone.

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
3. Add a separate collective executor selector:
   - `SourceBoundCollectiveExecutor::kNone`
   - `SourceBoundCollectiveExecutor::kOwnerFile`
   - `SourceBoundCollectiveExecutor::kSourceWindow`
4. Keep `ResolvedSourceBinding::collective_eligible` as owner-file state.
   Source-window candidate eligibility must be represented by typed
   source-window fields and diagnostics, not by overloading that boolean.
5. Add source-window pre-admission fields:
   - `source_window_collective_candidate`
   - `SourceWindowCollectiveCandidateSummary`
   - `source_window_group_reject_reason`
   - `source_window_selection_mode`
6. Add cost estimate fields:
   - `source_window_group_disk_read_bytes`
   - `rank_read_bytes_max`
   - `source_window_unique_payload_bytes`
   - `source_window_target_write_bytes`
   - `source_window_read_amplification_x1000`
   - `scatter_op_count`
   - `source_window_count`
7. Add `CollectiveExecutionMetrics` extensions or a nested source-window
   metrics struct:
   - `source_window_group_disk_read_bytes`
   - `source_window_rank_read_bytes_max`
   - `source_window_unique_payload_bytes`
   - `source_window_target_write_bytes`
   - `source_window_peer_transfer_bytes`
   - `source_window_peer_useful_bytes`
   - `source_window_peer_waste_bytes`
   - `source_window_scatter_op_count`
   - `source_window_window_count`
   - `source_window_read_amplification_x1000`
   - `source_window_distribution_mode`
8. Add typed daemon config with disabled dry-run defaults:
   - `enable_source_window_collective`
   - `source_window_collective_selection_mode`
   - `source_window_collective_window_bytes`
   - `source_window_collective_max_gap_bytes`
   - `source_window_collective_max_window_amplification_x1000`
   - `source_window_collective_max_plan_read_amplification_x1000`
   - `source_window_collective_max_scatter_ops_per_window`
   - `source_window_collective_peak_bytes_budget`
   - `source_window_collective_min_rank_read_saving_bytes`
   - `source_window_collective_max_peer_to_read_ratio_x1000`
   - `source_window_collective_min_routed_peer_saving_bytes`
   - `source_window_collective_distribution_mode`
   - `source_window_collective_allow_mixed_residual`
   - `enable_source_window_batched_scatter_kernel`
   - `enable_source_window_compiled_routed_program`
   - `source_window_compiled_program_build_threads`
   - `enable_source_window_scatter_cuda_graph`
9. Use integer byte fields and `_x1000` fixed-point ratios in proto/C++.
   Human-readable YAML size literals may be parsed by the unified config
   layer, but stored strategy values should be integer typed values.
10. Delete or deprecate any older environment-only toggles that duplicate the
   new typed config.

Tests:

- proto compile tests;
- C++ build for `//daemon:tensorcast_daemon`;
- unit tests for enum names and default config mapping;
- default config maps to `enable_source_window_collective=false` and
  `source_window_collective_selection_mode=dry_run`.

Exit criteria:

- strategy config can represent the new executor without selecting it;
- all old behavior remains unchanged until the executor is wired.

## Phase 2: Source-Window Group Plan Builder

Files:

- new `core/store/replica/source_window_collective_plan.h`
- new `core/store/replica/source_window_collective_plan.cc`
- `core/store/replica/BUILD`
- tests under `core/store/replica`

Inputs:

- `SourceWindowCollectiveGroupInput`
- all member `RepresentationWorkPlan` values
- all member `IntoTargetLayout` values
- all member target storage spans
- shared `DiskArtifactContext`
- `CollectiveLoadGroupHint`
- strategy config

Outputs:

- `SourceWindowCollectivePlan`
- `SourceWindowCollectiveCandidateSummary`
- `SourceWindowCollectiveGroupSummary`

Minimal C++ shape:

```cpp
struct SourceWindowCollectiveMemberInput {
  uint32_t rank;
  int device_id;
  RepresentationWorkPlan work_plan;
  loading::IntoTargetLayout target_layout;
  std::vector<TargetStorageSpan> storage_spans;
};

struct SourceWindowCollectiveGroupInput {
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  std::vector<SourceWindowCollectiveMemberInput> members;
  SourceWindowCollectiveConfig config;
};

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
  uint64_t group_disk_read_bytes;
  uint64_t useful_payload_bytes;
  uint64_t unique_payload_bytes;
  uint64_t target_write_bytes;
  uint64_t peak_temporary_bytes;
  uint64_t rank_read_bytes_max;
  uint64_t peer_transfer_bytes_estimate;
  uint64_t peer_useful_bytes_estimate;
  uint64_t peer_waste_bytes_estimate;
  uint64_t read_amplification_x1000;
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
8. Produce consumer spans per window and rank from the group work plans and
   target layouts, without using tensor names.
9. Build windows over group source intervals, not rank-local source intervals.
   Source bytes should be counted once per group window and then striped across
   producer ranks.
10. Infer rect2d from `RepresentationWorkSourceFragment` geometry and target
    layout. Do not require an explicit `WorkPartitionKind::kRect2d`.
11. Select an initial distribution mode per plan:
   - `FullWindowAllGather` for MVP-compatible dense windows;
   - `LocalOnly` for windows consumed only by the producing rank;
   - coalesced/cost-gated `ConsumerRouted` or `HybridWindow` as plan-level
     options when peer waste dominates. Naive ConsumerRouted is already
     implemented and measured negative, so admission must model pack/scatter
     operation count.
12. Merge windows by source offset using the existing `SourceWindowScheduler`
   rules as a starting point:
   - max gap,
   - max amplification,
   - window cap.
13. Reject plans that exceed scatter op count, temporary memory, peer waste,
   or residual policy.
14. Compute a stable `plan_hash` from source windows, distribution modes,
   consumer spans, scatter ops, and residual summary. Runtime must verify this
   hash across ranks before target mutation.

Tests:

- plan builder full tensor case;
- dim0 case;
- 2-rank dim1 narrow case with group read bytes close to source bytes, not
  per-rank full-row source reads;
- rect2d case with coalesced source windows;
- consumer span covering multiple ranks;
- target storage boundary split;
- adjacent window merge;
- max amplification rejection;
- max scatter-op rejection;
- residual map correctness;
- no tensor-name hardcoding;
- plan hash changes when source index digest, member order, target layout,
  distribution mode, window, consumer span, or scatter field changes;
- plan hash does not change when only the runtime group id changes;
- distribution-mode selection for all-gather, routed, hybrid, and local-only
  synthetic cases;
- host-local source is not rejected by owner-file shared-fs gates; only the
  source-window cost model can reject it.

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
2. In dry-run mode, produce `SourceWindowCollectiveCandidateSummary` and
   diagnostics only. Do not change lane mode or executor selection.
3. In enabled auto mode, mark
   `source_window_collective_candidate=true` and let the group executor perform
   final admission after all participants arrive.
4. In strict mode, record that source-window is required; final group admission
   failure must fail before target mutation.
5. Call it from ordinary disk strategy planning when:
   - GPU target;
   - safetensors or canonical source index present;
   - TP group present;
   - representation work plan present.
6. Call it from source-bound mapped strategy planning when:
   - `IntoTargetLayout` present;
   - disk source present;
   - collective group present;
   - local mapped plan would otherwise be selected.
7. Add cost-model comparison:
   - reject if residual policy fails;
   - reject if source-window read amplification exceeds threshold;
   - compare source-window rank-read bytes against the current local mapped
     physical read estimate or profile, not against ideal local payload bytes;
   - reject if peer transfer dominates expected disk savings;
   - reject or downgrade if peer waste dominates useful peer bytes;
   - reject if peak temporary bytes exceed budget;
   - prefer source-window collective when it reduces source bytes and task
     count compared with local mapped.
8. Make strict source-window collective fail before any local mapped or generic
   execution starts.
9. Record rejected candidate reasons in existing candidate diagnostics. Auto
   fallback must report `source_window_group_rejected:<reason>` when group
   final admission fails.

Tests:

- ordinary strategy chooses source-window when its estimate wins;
- ordinary strategy rejects source-window without group hint;
- mapped strategy chooses source-window over local mapped when read
  amplification is much lower;
- default config performs dry-run only and does not change executor;
- enabled auto attempts source-window candidate before local mapped;
- auto group rejection falls back explicitly with
  `source_window_group_rejected:<reason>`;
- strict mode fails fast when residual exists;
- strict mode group rejection fails before target mutation;
- mixed mode retains residual in the lane plan;
- host-local SSD can still choose local/generic when cost model says so;
- all-gather can lose to routed distribution in a synthetic sparse-consumer
  plan without changing the public executor selection;
- owner-file tests keep their existing semantics and are not rewritten to pass
  through source-window gates.

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
   - execute rank-striped `FullWindowAllGather` or `LocalOnly`;
   - execute scatter ops targeting the current rank's `IntoTargetLayout`;
   - release temporary buffers before next window or batch.
5. Record timings and actual byte counters.

The current code uses rank-striped all-gather plus multi-slot host read-ahead,
persistent per-rank reader workers, a persistent read-ahead pump, and
direct-to-pinned reads for direct safetensors source stripes. It proves strict
publication, removes read amplification, and overlaps part of GPU work with
future CPU reads, but it is not the final performance shape. TP=8 profiling
shows the current path still misses InstantTensor: prepared 30B source-window
loads weights in a best measured `12.97s` versus InstantTensor `10.43s`, and
prepared 235B improves to `79.67s` versus InstantTensor `70.69s`.

`LocalOnly` is implemented as an owner-only runtime branch. It is admitted only
when every consumer span in the window targets the selected owner rank. In that
case the owner rank reads the full chunk, stages it to the owner GPU, and
scatters into owner target storage without NCCL peer transfer. This keeps
distribution semantics explicit and prevents `LocalOnly` from silently
standing in for all-gather or routed distribution.

The explicit ConsumerRouted experiment is intentionally not the default. The
initial implementation reduced 30B peer transfer from `427GB` to `53.7GB`, but
regressed weight load to `23.10s` because pack/scatter issue cost dominated.
Caching CUDA device selection improved this to `19.09s`, close to the
full-window path but still not clearly better. A later forced rerun measured
`17.61s`/`50.07s` with `routed_pack_ops=147585` and `scatter_issue=5.59s`.
The next routed step must reduce pack/scatter operation count or choose a
hybrid policy only when peer traffic is the real bottleneck.

Planner-level scatter span coalescing is now implemented before admission. It
merges adjacent 1D spans and horizontally adjacent 2D column spans only when
rank, source index, target storage, and row geometry are identical and the
source and target offsets are exactly adjacent. This makes the plan hash,
`max_scatter_ops_per_window`, `source_window_scatter_op_count`, and runtime
scatter program agree on the same coalesced consumer span list. It is a safe
first step toward lower scatter issue count, but the larger ConsumerRouted /
HybridWindow performance path still needs coalesced pairwise packing or a GPU
scatter kernel.

The first `auto` distribution admission rule is implemented for owner-only
windows. When every window has consumer spans for one rank only, `auto` selects
`LocalOnly`, assigns each window owner to that consumer rank, and reports zero
peer transfer. When any window has consumers from multiple ranks, `auto` keeps
`FullWindowAllGather`. This preserves the conservative Qwen3 TP=8 behavior
while avoiding unnecessary all-gather on pure local windows.

`HybridWindow` now uses the same per-window distribution representation. The
planner marks owner-only windows as `LocalOnly`, dense multi-rank windows as
`FullWindowAllGather`, and sparse multi-rank windows as `ConsumerRouted` only
when the spans are linear/contiguous and the routed peer saving clears
`source_window_collective_min_routed_peer_saving_bytes` (default `64MB`). The
selected window mode is included in the plan hash, peer metrics are computed
from the per-window mode, and the runtime accepts `HybridWindow` by dispatching
each chunk according to its window mode. This is the first correct mixed-mode
executor. The earlier 4GB default made the routed gate unreachable for the
default 512MB windows, so the default is now deliberately below one TP=8
window's maximum all-gather peer bytes while remaining large enough to avoid
tiny-window routing. On the real Qwen3 TP=8 plans, however, the 64MB gate only
routes `1/115` 30B windows and `4/885` 235B windows; 235B full-window
all-gather remains faster in `LOAD_DONE` than explicit HybridWindow. The
remaining performance step is making routed pack/scatter coalesced enough, or
GPU-kernel backed enough, to become benchmark-proven and eligible for `auto`.

First implementation choices:

- Use the configured safetensors source factory so `buffered`, `auto`, and
  `direct_aligned_edges` apply equally to local mapped and source-window
  runtime paths.
- Use NCCL all-gather for the default correctness/performance path today. Keep
  consumer spans and distribution-mode fields in the plan so coalesced
  `ConsumerRouted` or `HybridWindow` can replace the naive routed experiment
  without changing artifact or vLLM boundaries.
- Use `cudaMemcpyAsync` and `cudaMemcpy2DAsync` for scatter before adding a
  custom GPU scatter kernel.
- Cache CUDA device selection inside the executor and only switch devices when
  an actual H2D, pack, or scatter operation is about to be issued. This keeps
  the runtime consistent with existing CUDA stream ownership while avoiding
  millions of redundant `cudaSetDevice` calls during chunk/span scans.
- Keep one window in flight until correctness is stable, then add prefetch
  depth. A benchmarkable implementation should overlap read, distribution, and
  scatter with at least a double-buffer pipeline if the single-window pipeline
  misses the 235B target.
- Benchmark and tune the explicit cost-gated hybrid runtime next. The planner
  can now mix `LocalOnly`, `ConsumerRouted`, and `FullWindowAllGather` windows,
  but the 30B naive ConsumerRouted run shows that peer-byte reduction alone is
  not enough; route planning must also reduce pack/scatter operation count. The
  235B strict run still shows rank-striped full-window all-gather consumes
  performance margin:
  it transfers `3.29TB` across peers, of which `2.88TB` is waste.
- Treat post-all-gather stream synchronization as not worth relaxing in the
  current runtime. Removing it did not reduce 30B/235B end-to-end time and only
  shifted wait into final scatter synchronization.
- Treat further per-rank parse prefetch as exhausted for now. The rejected
  canonical-index preparse future and the neutral source-table reuse rerun show
  that the meaningful TensorCast-consistent control-plane optimization is a
  group-level artifact realization builder that constructs source/target work
  once for the TP group and derives member views from that shared plan.

Add a separate request/result API:

```cpp
struct SourceWindowCollectiveMappedTargetLoadRequest {
  std::string artifact_id;
  loading::CollectiveLoadGroupHint group;
  std::shared_ptr<const loader::DiskArtifactContext> disk_context;
  RepresentationWorkPlan representation_work_plan;
  loading::IntoTargetLayout target_layout;
  SourceWindowCollectiveCandidateSummary candidate_summary;
  std::string source_index_digest;
  int device_id{-1};
};

struct SourceWindowCollectiveMappedTargetLoadResult {
  bool handled{false};
  absl::Status status{absl::OkStatus()};
  runtime::ingestion::strategy::CollectiveExecutionMetrics metrics;
  std::string skip_reason;
  std::string plan_hash;
};
```

Expose this as `try_source_window_collective_mapped_target_load(...)` or an
equivalent source-window-specific executor. It may reuse clique, pinned pool,
and target span helpers from `collective_disk_loader.cc`, but it must not be a
flag on `CollectiveMappedTargetLoadRequest` because that request is
`ByteRangeMap` and owner-file lane shaped.

Tests:

- fake or CUDA test for 2-rank all-gather window assembly;
- plan-level test keeps routed consumer spans and rejects local-only windows
  with remote consumers;
- plan-level tests coalesce adjacent linear spans and adjacent 2D column spans
  before `max_scatter_ops_per_window` admission;
- plan-level tests verify `auto` selects `LocalOnly` for owner-only windows and
  keeps `FullWindowAllGather` for multi-rank consumers;
- plan-level and loader tests verify `HybridWindow` admits mixed
  LocalOnly/ConsumerRouted/FullWindowAllGather plans and does not fall through
  `distribution_unsupported`;
- local-only CUDA correctness test keeps bytes on the owner rank and leaves
  non-consumer participants untouched;
- single-window full tensor scatter;
- dim1 scatter with expected bytes;
- rect2d scatter with expected bytes;
- multiple windows and merge boundaries;
- short read failure propagates;
- mismatched group plan fails before target mutation where possible;
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
   `try_source_window_collective_mapped_target_load(...)`.
2. Preserve local typed pad/fill execution in the existing target substrate.
3. Replace the current local mapped rect2d default when the source-window plan
   wins.
4. Ensure residual bytes are explicit and bounded.
5. Emit commit report:
   - `dominant_executor=SourceWindowCollectiveExecutor`
   - `source_window_group_disk_read_bytes`
   - `source_window_rank_read_bytes_max`
   - `source_window_unique_payload_bytes`
   - `source_window_target_write_bytes`
   - `source_window_peer_transfer_bytes`
   - `source_window_peer_useful_bytes`
   - `source_window_peer_waste_bytes`
   - `source_window_scatter_op_count`
   - `source_window_window_count`
   - `source_window_read_amplification_x1000`
   - `source_window_distribution_mode`
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

- TensorDict projection test. Covered for the public API report boundary by
  `test_tensor_dict_realization_reports_execution_topology_policy`, which
  proves ordinary `tensor_dict` keeps collective/source-locality facts in the
  same realization strategy report model;
- runtime attachment or binding projection test. Covered for direct runtime
  attachment report projection by
  `test_runtime_attachment_resolved_report_projects_execution_topology_options`;
- vLLM loader tests cover placement facts for TP rank/world size. Covered by
  `test_tensorcast_vllm_runtime_placement_maps_tp_pp_dp_identity`;
- vLLM loader tests cover source-window not selected when collective context is
  unavailable. Covered by
  `test_tensorcast_vllm_execution_facts_mark_unavailable_collective_context`
  and
  `test_tensorcast_execution_context_does_not_request_collective_when_unavailable`;
- vLLM loader config tests verify typed config reaches the daemon runtime
  config. Covered by
  `test_tensorcast_loader_carries_materialization_policy_to_runtime_profile`;
- test or code search guard confirms there is no vLLM-private source-window
  loader or RPC. Covered by
  `test_tensorcast_vllm_adapter_has_no_private_source_window_data_path`, which
  scans `vllm/tensorcast/**/*.py` for source-window private-path tokens while
  leaving daemon YAML/config as TensorCast-owned policy surface;
- no old compatibility branch required.

Exit criteria:

- source-window collective is a realization strategy, not a separate loader.

Current validation:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  pytest -q tests/model_executor/model_loader/test_tensorcast_loader_config.py -q
```

Result: 39 passed in `/data/workspace/internal-vllm`. The command uses
`internal-vllm/.venv`; `tensorcast` resolves to the current
`/data/workspace/tensorcast-280` source checkout. vLLM still calls
`artifact.realize(...)`, `from_disk(...).realize(...)`, or retained-claim
realization and does not introduce a private source-window loader/RPC.

Latest rerun after the window-builder reserve fix:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  python -m pytest -q tests/model_executor/model_loader/test_tensorcast_loader_config.py -q
```

Result: 40 passed. The interpreter was
`/data/workspace/internal-vllm/.venv/bin/python`, `pytest` was loaded from
`internal-vllm/.venv`, `torch` was `2.8.0+cu128`, and `tensorcast` resolved to
`/data/workspace/tensorcast-280/tensorcast/__init__.py`. The no-private-loader
guard intentionally scans only Python adapter code under `vllm/tensorcast`;
source-window daemon configuration remains TensorCast-owned.

TensorDict projection validation:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  export TORCH_SITE=/data/workspace/internal-vllm/.venv/lib/python3.10/site-packages && \
  export LD_LIBRARY_PATH="$TORCH_SITE/torch/lib:$TORCH_SITE/nvidia/nvshmem/lib:$TORCH_SITE/nvidia/cublas/lib:$TORCH_SITE/nvidia/cuda_runtime/lib:$TORCH_SITE/nvidia/cudnn/lib:$TORCH_SITE/nvidia/cufft/lib:$TORCH_SITE/nvidia/cufile/lib:$TORCH_SITE/nvidia/cuda_cupti/lib:$TORCH_SITE/nvidia/curand/lib:$TORCH_SITE/nvidia/cusparse/lib:$TORCH_SITE/nvidia/cusparselt/lib:$TORCH_SITE/nvidia/nccl/lib:$TORCH_SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}" && \
  pytest -q tests/python/api/test_artifact_handle.py -q
```

Result: 34 passed. Latest rerun after the window-builder reserve fix also passed
with `python -m pytest` under `/data/workspace/internal-vllm/.venv`. Without the
venv CUDA/NVSHMEM library path, existing subset/view tests that import
`tensorcast._C` fail to load `libnvshmem_host.so.3`; the added TensorDict
topology test itself passed in the smaller targeted run before the full-file
environment rerun.

Runtime attachment projection validation:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  export TORCH_SITE=/data/workspace/internal-vllm/.venv/lib/python3.10/site-packages && \
  export LD_LIBRARY_PATH="$TORCH_SITE/torch/lib:$TORCH_SITE/nvidia/nvshmem/lib:$TORCH_SITE/nvidia/cublas/lib:$TORCH_SITE/nvidia/cuda_runtime/lib:$TORCH_SITE/nvidia/cudnn/lib:$TORCH_SITE/nvidia/cufft/lib:$TORCH_SITE/nvidia/cufile/lib:$TORCH_SITE/nvidia/cuda_cupti/lib:$TORCH_SITE/nvidia/curand/lib:$TORCH_SITE/nvidia/cusparse/lib:$TORCH_SITE/nvidia/cusparselt/lib:$TORCH_SITE/nvidia/nccl/lib:$TORCH_SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}" && \
  pytest -q tests/python/api/test_realization_kernel.py -q
```

Result: full file passed. Latest rerun after the window-builder reserve fix also
passed with `python -m pytest` under `/data/workspace/internal-vllm/.venv`. The
added test verifies that runtime attachment resolved reports project the same
retrieval and execution-topology options used for daemon materialization into
the shared `strategy_plan.source_policy`.

Planner scatter coalescing validation:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel test //core/store/replica:source_window_collective_plan_test \
    //core/store/replica:collective_disk_loader_test

source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel build //daemon:tensorcast_daemon
```

Result: both tests passed and daemon build completed successfully. The new
planner tests cover adjacent linear span coalescing and adjacent 2D column span
coalescing before scatter-op admission.

Binding source-table reuse validation:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel test //daemon:materialization_target_plan_utils_test \
    //core/store/replica:source_window_collective_plan_test \
    //core/store/replica:collective_disk_loader_test \
    //core/common:daemon_config_io_test && \
  bazel build //daemon:tensorcast_daemon
```

Result: all targeted tests passed and the daemon build completed successfully.
The 30B HybridWindow 64MB rerun
`/data/tc/qwen3-source-window-runtime/20260615-source-table-reuse/20260615-213059-30b/qwen3_30b_ssd_source_window_source_table_reuse_hybrid64`
measured `17.83s` weight load and `49.31s` `LOAD_DONE`; daemon profile logs
reported `source_table_reused=1` on every TP rank.

Stream-ordered scatter experiment:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel test //core/store/replica:collective_disk_loader_test \
    //core/store/replica:source_window_collective_plan_test && \
  bazel build //daemon:tensorcast_daemon
```

Result: tests and daemon build passed, but the runtime change was rejected and
reverted after benchmarks. 30B HybridWindow 64MB measured `17.81s` weight load,
`50.62s` `LOAD_DONE`, runtime `total=10.04s`, `collective_sync=0.002s`,
`scatter_sync=0.028s`. 235B full-window all-gather measured `89.37s`,
`122.39s`, runtime `total=74.14s`, `collective_sync=0.000s`,
`scatter_sync=1.734s`.

Auto owner-only distribution validation:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel test //core/store/replica:source_window_collective_plan_test

source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel test //core/store/replica:collective_disk_loader_test && \
  bazel build //daemon:tensorcast_daemon
```

Result: planner test, collective loader test, and daemon build passed. New
planner tests cover `auto -> LocalOnly` for owner-only windows and
`auto -> FullWindowAllGather` when a window has multi-rank consumers.

Hybrid LocalOnly/Routed/AllGather distribution validation:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel test //core/store/replica:source_window_collective_plan_test \
    //core/store/replica:collective_disk_loader_test

source /data/workspace/internal-vllm/.venv/bin/activate && \
  bazel build //daemon:tensorcast_daemon
```

Result: both tests passed and daemon build completed successfully. The planner
tests cover a mixed hybrid plan with one owner-only `LocalOnly` window and one
sparse multi-rank `ConsumerRouted` window, plus a dense multi-rank window that
stays on `FullWindowAllGather`; loader admission coverage verifies
`HybridWindow` reaches runtime fallback instead of being rejected as
unsupported, including the routed subpath.

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

1. Phase 1 config and strategy type patch. Done in the current code slice.
2. Phase 2 source-window group plan builder and tests. Done for the 30B/235B
   compressed geometry paths in the current code slice.
3. Phase 3 dry-run candidate integration and diagnostics. Done far enough to
   collect rejected-candidate metrics without changing publication.
4. Phase 0.25 real WorkPlan validation for 30B and 235B. Done.
5. Strict zero-residual source-window execution for 30B and 235B. Done.
6. Phase 0.5/Phase 7 TP=8 profile with daemon startup excluded. Done for the
   current FullWindowAllGather MVP and explicit ConsumerRouted experiment.
7. Profile and optimize the rank-striped runtime: parallel stripe reads,
   H2D/all-gather overlap, scatter coalescing, and double buffering. First
   2-slot host read-ahead, persistent per-rank readers, and persistent
   read-ahead pump are done; direct-to-pinned reads are done for direct
   safetensors stripes; planner index parsing, hash allocation, storage-span
   reuse, and member/span projection have been tightened; compact binary hash
   serialization, flat-hash interval buckets, and representation builder
   reserve variants were measured negative and reverted; source-window
   participant copy-elision and the one-pass window-builder reserve fix are
   done; deeper async IO and orchestration reduction remain.
8. Add coalesced `ConsumerRouted` or `HybridWindow` distribution and re-run
   30B/235B. Naive ConsumerRouted is implemented and measured negative;
   `HybridWindow` is implemented with a conservative routed-saving gate and
   currently keeps Qwen3 TP=8 on full-window all-gather by default. The 64MB
   gate makes routed windows reachable, but current Qwen3 runs route too few
   windows to beat the full-window all-gather control.
9. Continue control-plane compression after source-index preparse: execution
   plan build no longer reparses the physical source index, but group
   admission and artifact resolve still show up in the measured first-load
   tail. Public mounted-source resolve now warms the shared
   `DiskArtifactContext` cache for later materialization, with the cache key
   aligned to the mounted-source fingerprint fields (`inode`, `size`,
   `mtime_ns`). The 30B SSD cold rerun confirmed resolve-time cache warm and
   rank-local cache hits, but max-rank weight load stayed in the same band at
   `17.38s`/`49.60s`.
10. Continue group-plan compression after plan-hash v2 and the one-pass
    window-builder reserve fix: group `plan_sec` is down to about 1.13s on 30B,
    but metrics/admission, artifact resolve, and rank-local preparation still
    show up in the first-load tail. The reserve-fix profile has max
    `store.from_disk=1.693s`, `realize_from=13.546s`, daemon
    `prepare_plan_sec=0.957s`, `prepare_execution_sec=0.778s`,
    `materialize_sec=11.840s`, source-window runtime `9.906s`, and Python
    finalize only `0.116s`. The follow-up storage-span member cache/span-view
    cleanup reduced `append_sec` to about `0.091s` and measured
    `17.36s`/`49.87s`; it is a retained control-plane cleanup. The follow-up
    local-read metrics inline change collects the local mapped physical-read
    intervals while scanning window consumer spans, reducing `metrics_sec` to
    about `0.261s`, group `plan_sec` to about `1.095s`, and measuring
    `17.36s`/`48.99s`; the weight-load best remains `17.29s`. A later
    `representation_transform_builder` reserve experiment measured
    `17.70s`/`50.83s` for a manual group-reserve variant and `17.68s`/`49.21s`
    for a simple reserve variant; both were reverted because they did not move
    `representation_realization_plan` below the existing noise band.
    A later mapped-plan/execution-template singleflight validation measured
    `17.06s`/`50.40s` on 30B TP=8 SSD cold with
    `mapped_plan_cache_hit=0`, `mapped_plan_cache_waited=0`,
    `execution_template_cache_hit=0`, and `execution_template_cache_waited=0`
    on every rank. This confirms the current per-rank cache key shape cannot
    share Qwen3 TP preparation work; the next useful cut is a group-level
    prepared plan, not more rank-local cache coalescing.
11. Phase 6 convergence into the unified realization target-set model.
12. Cleanup old behavior after benchmark evidence.

# Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| all-gather padding wastes too much bandwidth | add coalesced irregular send/recv or peer-copy window distribution after MVP |
| rank-striped full-window all-gather wastes too many peer bytes on sparse consumer spans | keep consumer spans in the first plan representation; require routed/hybrid admission to account for pack/scatter op count before production defaulting |
| owner-read plus broadcast gets reintroduced as a shortcut | keep benchmark and admission tied to rank-striped all-gather metrics |
| scatter op count is too high | merge scatter ops, add GPU scatter kernel, cap plan by cost model |
| source windows reintroduce read amplification | enforce max amplification and report actual vs planned bytes |
| source-window is rejected despite reducing physical reads | compare against local mapped physical read estimate/profile, not ideal local payload bytes |
| temporary GPU memory is too high | bounded window bytes, one-window-in-flight MVP, explicit peak budget |
| single-window pipeline fails 235B target despite good read-side feasibility | add double-buffered read/distribute/scatter overlap before declaring the executor benchmark-ready |
| production plan diverges from the Qwen-name feasibility model | require Phase 0.25 WorkPlan dump validation before runtime benchmarking |
| per-rank planning admits inconsistent source windows | build final plan from `SourceWindowCollectiveGroupInput` and validate `plan_hash` across ranks |
| rank plans diverge and corrupt mapped targets | verify `plan_hash` across ranks before target mutation and invalidate targets on failure |
| owner-file collective gates reject valid source-window cases | use `SourceBoundCollectiveExecutor` and source-window candidate diagnostics instead of overloading `collective_eligible` |
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

Current closeout status after the typed-config ConsumerRouted reruns:

- Criteria 1, 2, 4, 5, and 7 have direct evidence from strategy-plane tests,
  daemon metrics, 30B/235B generation checks, structured source-window metrics,
  and the updated disk-load strategy docs.
- Criterion 3 is now satisfied at the "same performance class" level for the
  accepted typed-config path. 30B SSD same-daemon TensorCast reaches
  `11.736s` weight load and `35.059s LOAD_DONE` versus InstantTensor
  `10.43s`/`33.06s`; 235B SSD same-daemon TensorCast reaches `77.225s` and
  `102.132s` versus InstantTensor `70.69s`/`105.86s`.
- Criterion 6 is satisfied for the execution switches that matter to this
  plan: the old `TENSORCAST_SOURCE_WINDOW_*` fast-path environment gates are no
  longer present in the source tree, and the accepted fast path is controlled by
  typed `materialization_strategy` config. Remaining diagnostic-only paths such
  as CUDA graph scatter stay default-off and documented as rejected/diagnostic
  evidence, not compatibility behavior.

Optimization triage for the remaining work:

- Good optimization space: group-level prepared realization/control and
  coalesced routed/hybrid distribution. These directly attack the remaining
  TensorCast-specific overhead while preserving the artifact-centered boundary.
  The prepared local-ready result is useful but not sufficient: it moves about
  `3.14s` of source/recipe work before the weight-load log, while `LOAD_DONE`
  remains around `48.6s`.
- Already near fixed limits: 235B source-window disk reads are about
  `6.51GB/s` against a measured `6.72GB/s` SSD direct-read ceiling, so deeper
  read-loop tuning alone should not be the main project.
- Low-return space: daemon startup is excluded, Python finalize and CUDA IPC
  restore are below the main critical path, and per-rank cache/preparse
  micro-tweaks have repeatedly measured inside noise or regressed. The
  data-only strict coverage proof is safe and retained, but its 30B rerun only
  moved `build_resolved_plan_sec` inside the noise band. The target-storage
  fast path is also safe and retained, but it hit all `151985` scatter pieces
  and still measured only `16.83s`/`49.61s`, proving target storage re-resolution
  is not the root gap. The delayed-pack ConsumerRouted experiment was rejected:
  it touched only `4275` pending linear pack pieces (`732MB`) and left total
  `routed_pack_ops` near `143k`; the prepared forced ConsumerRouted rerun still
  had `scatter_issue=5.78s` and `cuda_device_switches=61029`, so the routed
  bottleneck is the 2D pack/scatter primitive and operation density, not
  linear-piece ordering. Do not make disruptive abstraction
  changes for these costs. The source-window group call-profile rerun also
  shows group assemble skew is only `0.035-0.232s`; group join mechanics are
  not a meaningful optimization target. The direct canonical-index-bytes source
  catalog path is similarly worth keeping for correctness and object-churn
  cleanup, but the verified `1.08x` synthetic speedup means further local
  parser tuning should stop in favor of prepared source/catalog state.
- Not the root primitive bottleneck: a TP=8 H800 synthetic source-window
  microbench moves a 512MiB full window through NCCL all-gather in about
  `0.0033s` and all-gather plus one 2D scatter per rank in about `0.0031s`.
  Raw full-window collective bandwidth is therefore healthy. Raising the
  microbench to production-like scatter launch density (`165` ops/rank/window)
  raises per-window time to about `0.0069s`, or roughly `0.8s` when scaled to
  30B's `115` windows. The real runtime gap comes from disk/read staging,
  repeated planning/preparation, and production recipe scatter operations
  rather than the NCCL primitive itself.
- Partially optimized but not root cause: source catalog construction now uses
  the daemon-attested canonical index and canonical-hash fingerprint, reducing
  the warm-cache stage from `1.06-1.12s` to `0.46-0.50s` per rank. The first
  run after changing the fingerprint key exposed cold recipe compile cost
  (`5.18s` per rank), so future work should make prepared recipe/cache
  lifecycle explicit rather than continue small per-rank catalog tweaks.
- Newly tightened control-plane cache boundary: the daemon execution-template
  cache now uses a stable runtime-group key for source-window selectable
  auto/strict paths. Runtime `group_id` is a rendezvous instance, not a
  semantic input to rank-local `RepresentationWorkPlan` or strategy candidate
  construction; rank, world size, source metadata, target layout, policy, and
  source-window config remain in the key. This is a correct prepared/cache
  convergence step and is verified by `//daemon:owned_binding_service_test`,
  but it is not expected to move first-load strict SSD-cold data-plane numbers
  by itself. The follow-up prepared-realization key split now validates the
  intended identity: one prepared group-template key across TP=8, one target
  layout template hash, one target-index hash, and eight member/realization-plan
  hashes. That makes further rank-local key trimming a dead end. The useful next
  abstraction is group member fact collection: collect the per-rank
  `BindingRealizationPlan`s, normalize the shared source/target template once,
  then lower per-rank member views from that group template.

Decision boundary for future implementation:

| Area | Keep optimizing? | Reason |
|---|---|---|
| Group-level prepared realization/control | Yes | Removes repeated rank-local target plan and execution preparation while preserving artifact identity and strict admission. It must collect rank-specific `BindingRealizationPlan` member facts instead of trying to share one rank-local realization key. |
| Source catalog and recipe/template preparation | Yes, but only structurally | Source catalog fast paths reduced warm-cache catalog work to `0.46-0.50s`; admin bootstrap now accepts a prepared source catalog, prepared recipe/source/catalog identity is checked before mutation, and vLLM prime can now call TensorCast `prepare_local_ready_recipe()` then pass matching prepared facts back into mounted-source artifact realization. Cold recipe compile can still cost `5.18s` per rank, so solve through prepared source/recipe artifact lifecycle, not more rank-local micro-tweaks. |
| Execution-template runtime group key | Done for source-window selectable paths | Stable across runtime `group_id` while remaining sensitive to rank/world/source/target/config. Prepared-realization identity now has one group-template key and rank-specific member keys, so this is correctness/cache hygiene for repeated loads and group-prepared semantics, not a root first-load speed lever. |
| Group source-window plan prebuild/cache | Maybe | Can save about `1s`, but only if keyed by source index digest, member target layouts, topology facts, and config. |
| Batched routed/hybrid pack/scatter | Yes | This is the remaining path to reduce peer bytes without exploding operation count. |
| Raw NCCL all-gather primitive | No for now | TP=8 512MiB microbench is about `0.0033s`; production overhead is above this primitive. |
| Production scatter launch density | Yes, structurally | Real 30B batched-scatter gate estimates `151,685 -> 3,605` copy launches; real 235B estimates `303,223 -> 28,181`. This can recover scatter issue (`1.11s` on 30B, `3.64s` on 235B) but not the full InstantTensor gap. Coarsen/batch through TensorCast realization shape. |
| Direct IO/read-ahead loop | Not primary | Current 235B SSD runtime is already near measured direct-read ceiling. |
| Python binding/finalize | No for now | Latest profile shows tens of milliseconds for binding and sub-`0.5s` finalize/view work. |
| CUDA IPC restore | No for now | About `0.23-0.25s`, async-overlapped in the current path. |
| Group wait/condition variable mechanics | No | Follower assemble skew is below `0.24s`; followers wait for leader execution. |
| Target storage re-resolution | Done, do not iterate | Fast path hit all `151985` scatter pieces but did not move end-to-end time materially. |
| Full-target coverage proof | No | It can skip typed local pad/fill and would weaken correctness. |
| Routed linear-piece ordering tweaks | No | The rejected delayed-pack run proved this touches too little data and not the dominant op count. |

InstantTensor root-cause comparison and revised optimization target:

- `/data/workspace/InstantTensor` uses physical safetensors order to build
  `8MiB` world chunks. Each rank reads its padded rank slice, H2D copies it,
  and runs `ncclAllGather`; the Python iterator then exposes contiguous tensor
  views and, with default `copy=True`, clones them as coarse tensor-sized GPU
  copies before vLLM copies into model parameters. The measured SSD debug run
  reports `open=0.21s`, `load=8.89s`, and total InstantTensor loader time
  around `10.46s`.
- TensorCast now uses the same high-level ingredients (rank-striped reads,
  direct-to-pinned staging, all-gather, and daemon-owned target tensors), but
  the realization shape differs: source-window publishes the production
  recipe's tensor/rectangle copies into the target binding layout. Dim1/rect2d
  TP shards become many 2D scatter pieces, and full-window all-gather transfers
  far more peer bytes than are useful. The latest target-storage fast path
  proves pointer lookup is no longer the bottleneck; the bottleneck is the
  distribution and scatter/pack operation shape.
- The current 30B best weight-load log is `14.69s` with prepared full-window,
  but `LOAD_DONE` remains `48.59s` because the `3.14s` prime preparation is
  still in the end-to-end path. Removing remaining rank-local prepared
  `from_disk`/catalog/recipe/group-plan work would help, but it is still
  insufficient by itself; InstantTensor-class end-to-end performance requires a
  layout/data-plane realization change.

Next implementation target:

1. Add a TensorCast-owned batched GPU pack/scatter or lower-waste routed
   materialization mode under the source-window collective executor. It should
   keep group final admission, plan hashing, strict failure-before-mutation,
   and artifact/source/target identity keys, but it should execute the existing
   real tensor/rectangle fragments in batches instead of independently issuing
   every recipe/storage scatter piece.
2. Use the real 30B/235B WorkPlan summaries as the gate. The simple
   tensor-staged adjacent-span coalescer reports zero operation-count reduction
   on both models, so do not build that runtime MVP. The batched-scatter gate
   reports `151,685 -> 3,605` estimated copy launches on 30B and
   `303,223 -> 28,181` on 235B, so the first CUDA/runtime prototype should
   target one batched final-scatter launch per runtime chunk/rank for
   FullWindowAllGather.
3. After FullWindowAllGather scatter batching is correct and measured, extend
   the same descriptor/kernel shape to ConsumerRouted/HybridWindow pack plus
   final scatter. Do not revive linear-piece ordering tweaks; they touched only
   a small tail and left the dominant 2D pack/scatter cost.
4. Keep vLLM on the public `tc.from_disk(...).realize(...)` /
   `artifact.realize(...)` boundary. If source acquisition becomes a product
   prewarm step, express it as a TensorCast prepared-source/recipe lifecycle
   that the daemon can own before model load; do not introduce a vLLM-private
   source path.
5. Stop investing in fixed or low-return costs unless a new profile moves them
   onto the critical path: daemon startup is excluded, group wait is
   sub-`0.24s`, CUDA handle restore is mostly async, target-storage lookup is
   solved, and full-target coverage shortcuts are semantically invalid.

2026-06-16 tensor-staged/coarse-copy feasibility gate implementation: the
source-window planner now exposes
`summarize_source_window_tensor_staged_copy(...)`. It consumes the same
`SourceWindowCollectiveGroupInput`/`RepresentationWorkPlan` geometry as the
production group planner, computes raw tensor/source-fragment copy ops,
coalesces adjacent rank-local tensor-staged linear and 2D copy ops by
source/destination tensor and target storage, and reports eligible bytes,
ineligible bytes, source fragment count, source/destination tensor counts,
coalesced op count, max staged ops per rank, and estimated op reduction. The
API is intentionally not called on the production hot path yet, so it cannot
slow current source-window loads. Unit coverage proves adjacent fragments can
coalesce from four raw copy ops to two rank-local staged ops, and unsupported
fragments such as `prefix_count != 1` fail the feasibility gate instead of
silently entering the future fast path. Validation passed `git diff --check`,
`bazel test //core/store/replica:source_window_collective_plan_test`,
`bazel test //core/store/replica:collective_disk_loader_test`, and
`bazel build //daemon:tensorcast_daemon`.

The production gate has now run against real WorkPlans. It rejected the simple
tensor-staged runtime MVP: the real summaries did not reduce operation count
on either 30B or 235B, so the next data-plane effort should go straight to a
batched GPU pack/scatter primitive or a different routed distribution shape.

2026-06-16 real WorkPlan diagnostic wiring: the source-window collective
loader now calls `summarize_source_window_tensor_staged_copy(...)` during group
final admission when materialization strategy diagnostics are `verbose`. This
uses the already-built production `SourceWindowCollectiveGroupInput`, so it
observes the real daemon-side `RepresentationWorkPlan`, target layout, storage
spans, source index digest, and TP member set. The default/basic path does not
call the summary and therefore does not add first-load overhead. Verbose logs
now include `source_window_tensor_staged_feasibility` with
`raw_copy_ops`, `tensor_staged_copy_ops`, `linear_copy_ops`, `copy_2d_ops`,
eligible/ineligible bytes, max ops per rank, estimated op reduction, and
summary runtime. Targeted validation passed `git diff --check`, explicit
`--no-index --check` on the currently untracked source-window planner files,
`bazel test //core/store/replica:source_window_collective_plan_test
//core/store/replica:collective_disk_loader_test`, and
`bazel build //daemon:tensorcast_daemon`.

2026-06-16 real WorkPlan staged-copy gate result: 30B and 235B source-window
were rerun with daemon startup excluded and
`engine.materialization_strategy.diagnostics_verbosity` set to
`MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_VERBOSE`. The 30B run at
`/data/tc/qwen3-source-window-runtime/20260616-staged-copy-verbose/20260616-043249-30b-staged-copy-verbose-slots8`
completed with `fincore_resident=0`, `weight_load_max=20.00s`, and
`LOAD_DONE=51.17s`. Its verbose summary reported
`raw_copy_ops=150936`, `tensor_staged_copy_ops=150936`,
`linear_copy_ops=2696`, `copy_2d_ops=148240`,
`max_tensor_staged_copy_ops_per_rank=18867`, and
`estimated_op_reduction_x1000=0`. The 235B run at
`/data/tc/qwen3-source-window-runtime/20260616-staged-copy-verbose/20260616-043450-235b-staged-copy-verbose-slots8`
completed with `fincore_resident=0`, `weight_load_max=102.44s`, and
`LOAD_DONE=137.21s`. Its verbose summary reported
`raw_copy_ops=295560`, `tensor_staged_copy_ops=295560`,
`linear_copy_ops=5272`, `copy_2d_ops=290288`,
`max_tensor_staged_copy_ops_per_rank=36945`, and
`estimated_op_reduction_x1000=0`. These verbose runs include extra diagnostic
summary overhead and are not new best-performance baselines. They are
decision gates: do not build the simple tensor-staged adjacent-coalescing
runtime.

2026-06-16 batched-scatter feasibility gate implementation: the source-window
planner now exposes `summarize_source_window_batched_scatter(...)`. It consumes
the admitted `SourceWindowCollectivePlan` plus the runtime chunk size and
estimates current scatter/pack copy launches versus a future batched GPU
program with one final-scatter launch per runtime chunk/rank and one routed
pack launch per producer-consumer pair. The summary is diagnostic only: it does
not affect admission, fallback, plan hash, target mutation order, or non-verbose
runtime overhead. Verbose loader logs now include
`source_window_batched_scatter_feasibility` next to the tensor-staged summary.
Targeted validation passed `git diff --check`, explicit `--no-index --check`
on the currently untracked source-window planner files,
`bazel test //core/store/replica:source_window_collective_plan_test
//core/store/replica:collective_disk_loader_test`, and
`bazel build //daemon:tensorcast_daemon`.

2026-06-16 real batched-scatter gate result: 30B and 235B source-window were
rerun with daemon startup excluded, TP=8, SSD cold, `fincore_resident=0`, and
verbose diagnostics. The 30B run at
`/data/tc/qwen3-source-window-runtime/20260616-batched-scatter-summary/20260616-045803-30b-batched-scatter-summary/qwen3_30b_ssd_source_window_batched_scatter_summary`
completed with `weight_load_max=18.24s`, `LOAD_DONE=50.77s`,
`daemon_ready_sec_excluded=10.03s`, and runtime `chunks=457`,
`scatter_ops=151985`, `read=7.47s`, `collective_sync=0.688s`,
`scatter_issue=1.110s`, `total=9.47s`. Its batched-scatter summary reported
`estimated_current_copy_launches=151685`, `batched_total_copy_launches=3605`,
`max_descriptors_per_batched_scatter=47`, and
`estimated_copy_launch_reduction_x1000=977`. The 235B run at
`/data/tc/qwen3-source-window-runtime/20260616-batched-scatter-summary/20260616-050012-235b-batched-scatter-summary/qwen3_235b_ssd_source_window_batched_scatter_summary`
completed with `weight_load_max=90.00s`, `LOAD_DONE=121.80s`,
`daemon_ready_sec_excluded=9.72s`, and runtime `chunks=3539`,
`scatter_ops=304795`, `read=62.58s`, `collective_sync=5.288s`,
`scatter_issue=3.643s`, `total=72.43s`. Its summary reported
`estimated_current_copy_launches=303223`, `batched_total_copy_launches=28181`,
`max_descriptors_per_batched_scatter=15`, and
`estimated_copy_launch_reduction_x1000=908`. This gate proves batched scatter is
a real structural target, while also bounding it: it mostly recovers scatter
issue time and must be paired with group-level prepared realization and
lower-waste routed/hybrid distribution to reach InstantTensor-class load time.

2026-06-16 experimental FullWindowAllGather batched-scatter runtime result
(rejected): I implemented an NVRTC descriptor copy kernel for source-window
final scatter, added daemon prewarm, validated linear and 2D copy correctness
with a focused CUDA test, and then profiled it on the same 30B TP=8 SSD-cold
source-window run. With the kernel enabled, runtime issued only `7172` kernel
launches for `153647` descriptors, but `scatter_issue` regressed from
`1.12s` to `18.09s`; vLLM-visible timing regressed to
`weight_load_max=28.11s`, `LOAD_DONE=61.12s`. Moving descriptors through
per-rank pinned host scratch did not improve the result, so pageable descriptor
copy was not the root cause. The default path now leaves the kernel behind the
typed config gate `enable_source_window_batched_scatter_kernel=true` and uses
the existing `cudaMemcpyAsync/cudaMemcpy2DAsync` scatter path. The recovery 30B run
reported `weight_load_max=18.32s`, `LOAD_DONE=50.56s`,
`batched_scatter_kernel_enabled=0`, `scatter_issue=1.12s`, and runtime
`total=9.52s`. The 235B recovery run reported `weight_load_max=89.95s`,
`LOAD_DONE=123.02s`, `batched_scatter_kernel_enabled=0`, `read=62.61s`,
`collective_sync=5.23s`, `scatter_issue=3.55s`, and runtime `total=72.56s`.
Do not spend more implementation time on a naive custom D2D copy kernel. A
future batched scatter/pack primitive must first prove copy bandwidth and host
issue time close to CUDA memcpy/copy-engine behavior in a standalone
microbenchmark before it is allowed onto the default runtime path. Until then,
the high-leverage runtime work is lower-waste ConsumerRouted/HybridWindow and
group-level prepared realization/control.

2026-06-16 routed final-scatter batched-kernel diagnostic: I extended the
default-off batched scatter path to ConsumerRouted local pieces and packed
remote final scatter. During the first 30B JFS run, the kernel succeeded for a
few launches and then failed with `CUDA_ERROR_INVALID_HANDLE`. The minimal
two-GPU alternating-launch test passed, so the root was not the NVRTC module
cache. The real issue was TensorCast runtime state: `set_device_cached` could
be stale after `NcclClique::synchronize_all()`, because the clique synchronizer
switches CUDA devices internally. The source-window runtime now invalidates the
cached current device after every clique-wide synchronize before issuing later
GPU work.

Validation on `ws-7681b3683947089e-worker-spdxq`:

```bash
bazel test //core/store/replica:source_window_batched_scatter_kernel_test \
  //core/store/replica:collective_disk_loader_test
bazel build //daemon:tensorcast_daemon
```

The focused two-GPU kernel test and collective loader test both passed. The
real 30B JFS forced-ConsumerRouted comparison used TP=8, source-window strict,
8 read-ahead slots, daemon startup excluded, and `fincore_resident=0`:

| Run | Kernel | Weight load | LOAD_DONE | Runtime total | Scatter issue | Kernel/fallback descriptors |
|---|---|---:|---:|---:|---:|---:|
| `/data/tc/qwen3-source-window-runtime/20260616-routed-batched-scatter/20260616-152636-20260616-routed-no-batched-jfs-devicecache-fix/30b-jfs-consumer-routed-no-batched-devicecache-fix` | off | 13.06s | 45.97s | 9.43s | 6.36s | 0 / 121,464 |
| `/data/tc/qwen3-source-window-runtime/20260616-routed-batched-scatter/20260616-152428-20260616-routed-batched-scatter-jfs-devicecache-fix/30b-jfs-consumer-routed-batched-scatter-devicecache-fix` | on | 12.83s | 48.63s | 8.51s | 5.77s | 120,728 / 736 |

The kernel path no longer fails and covers almost all routed final scatter
descriptors. It is still not a default path: `routed_pack_ops=154436` and
`routed_deferred_2d_pack_ops=147550` were unchanged, so the main routed
limiter remains producer-side 2D pack and operation density. The result
supports reusing the descriptor/kernel machinery for future routed prototypes,
but the next performance cut must batch or change the pack shape, not only the
final scatter.

InstantTensor source-code comparison: `/data/workspace/InstantTensor` confirms
that its advantage is structural, not a hidden per-copy trick. Python reads
safetensors metadata, warms NCCL, and passes filenames/tensor offsets plus
chunk/concurrency/io-depth parameters into a thin C++ loader. The C++ side
builds a contiguous safetensors ring-buffer layout, uses aligned chunks, and
pipelines file read, `cudaMemcpyAsync`, and NCCL all-gather through dedicated
async executors and non-blocking streams. AIO uses `O_DIRECT` and CUDA-registered
host buffers; cuFile reads into a registered device buffer; mmap/tmpfs uses
host memcpy plus H2D. It returns DLPack tensor views into that device buffer.
TensorCast cannot reach the same shape by shaving isolated milliseconds from
target lookup or group join. The artifact-centered path needs explicit
prepared source/catalog/recipe/group-plan lifecycle plus lower-waste
source-window distribution, while keeping vLLM at the existing artifact
boundary.

2026-06-16 local-ready prepare profiling update: the latest profile exposed a
remaining startup/control-plane gap outside the source-window group call. Rank 7
spent `16.47s` in `startup.artifact_runtime.start`, while
`local_ready.bootstrap.realize_binding` accounted for `12.90s` and the source
window group call accounted for about `10.82s`. `store.from_disk` was `1.45s`,
but the JSONL profile did not include `recipe.summary` or other recipe build
events, leaving about `1.7s` between source resolution and contract preflight.
To avoid misclassifying that time as fixed cost, the lifecycle now records
`local_ready.prepare.*` profile scopes around source-subject resolution,
artifact-ref resolution, framework placement, source catalog construction,
recipe cache config, and recipe build. The follow-up vLLM outer-profile rerun
below used these events to separate source/catalog/recipe preparation from
lower-return framework startup work.

2026-06-16 vLLM outer-profile validation: the follow-up 30B TP=8 SSD cold rerun
at
`/data/tc/qwen3-source-window-runtime/20260616-vllm-profile/20260616-022622-30b-vllm-profile-slots8/qwen3_30b_ssd_source_window_vllm_profile_slots8`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=16.80s`, and `LOAD_DONE=49.80s`. The new vLLM `startup.*`
scopes show host/context/placement/spec, source selector, plan selection, and
attachment are not meaningful optimization targets. The real outer control
cost is `startup.from_disk=1.53-1.86s`,
`local_ready.prepare.build_source_catalog=1.06-1.12s` CPU per rank, and
`local_ready.prepare.build_recipe=0.46-0.48s` CPU per rank. These should be
folded into the group-level prepared realization plan through shared artifact
source catalog facts and recipe/template reuse. They should not be optimized by
adding a vLLM-private loader/source path.

2026-06-16 source-catalog and fingerprint fast-path validation: vLLM now builds
its TensorCast source catalog from the daemon-attested
`PublicDiskSourceHandle.canonical_index_bytes` when the source subject is a
public disk source. This preserves the real artifact id and avoids per-rank
safetensors rescans. The rerun at
`/data/tc/qwen3-source-window-runtime/20260616-source-catalog-fastpath/20260616-023627-30b-source-catalog-fastpath-slots8/qwen3_30b_ssd_source_catalog_fastpath_slots8`
measured `weight_load_max=16.36s`, `LOAD_DONE=50.09s`, and
`local_ready.prepare.build_source_catalog=0.49-0.52s`. TensorCast source
metadata fingerprinting now treats the canonical index hash plus selected-file
facts as authoritative when canonical identity is present. The first run after
that key change measured `22.06s` because it intentionally missed the old
recipe cache and exposed full recipe compile (`build_recipe=5.18s`). The
warm-cache rerun at
`/data/tc/qwen3-source-window-runtime/20260616-metadata-fingerprint-fastpath-warm/20260616-024315-30b-metadata-fingerprint-fastpath-warm-slots8/qwen3_30b_ssd_metadata_fingerprint_fastpath_warm_recipe_slots8`
measured `weight_load_max=16.23s`, `LOAD_DONE=51.90s`,
`build_source_catalog=0.46s`, and `build_recipe=0.48s`. Retain these as
artifact-consistent cleanup, but treat the cold recipe compile as the stronger
design signal.

2026-06-15 required-policy validation: the strict source-window path now passes
the old failing `COLLECTIVE_POLICY_REQUIRE_COLLECTIVE` case. The 30B TP=8 SSD
cold rerun completed with daemon startup excluded, `admitted=1`,
`handled=true`, `read_amplification_x1000=1000`, `weight_load_max=21.33s`, and
`LOAD_DONE=53.22s`. The slower timing is explained by a slower data-plane read
sample (`13.16s` versus the recent `9.64-9.78s` best runs), not by a fallback to
local mapped or owner-file collective.

2026-06-15 fast work-plan rebind validation: the 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260615-fast-work-plan-rebind/20260615-222714-30b/ssd_cold`
completed with `work_plan_source_rebind_fast_path=1` on all ranks,
`work_plan_sec=0.202-0.206s`, `admitted=1`, `read_amplification_x1000=1000`,
`weight_load_max=19.88s`, and `LOAD_DONE=51.41s`. This is a measured
control-plane reduction but not a new best end-to-end result because the data
plane read sample was `11.91s`.

2026-06-15 auto direct IO policy validation: the previous fast work-plan
rebind run exposed that the `auto` safetensors IO policy still chose buffered
IO for large cold local files. The production policy now selects
`direct_aligned_edges` for large regular safetensors on direct-friendly local
filesystems when page-cache residency is cold or partial, and keeps buffered IO
for small sources, non-direct-friendly filesystems, and hot page-cache probes.
The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260615-auto-direct-policy/20260615-224847-30b-auto-direct/ssd_cold`
used no benchmark IO override and logged
`decision=direct_aligned_edges reason=page_cache_cold_or_partial_direct`,
`weight_load_max=17.76s`, `LOAD_DONE=49.71s`, `read=9.60s`,
`runtime_total=10.20s`, and `direct_pinned_read_successes=7288/7304`. The 235B
TP=8 SSD cold rerun at
`/data/tc/qwen3-235b-source-window-runtime/20260615-auto-direct-policy/20260615-225123-235b-auto-direct/qwen3_235b_ssd_source_window_auto_direct_default`
likewise used no IO override and logged direct auto selection,
`weight_load_max=90.83s`, `LOAD_DONE=123.54s`, `read=73.86s`,
`runtime_total=75.47s`, and `direct_pinned_read_successes=28194/28312`.

2026-06-15 source-window auto IO decision-once validation: source-window group
execution now resolves the local safetensors auto IO mode once and reuses the
resolved strategy for all per-rank `SeekableSource` instances. This keeps local
mapped semantics unchanged while avoiding repeated `statfs` and `mincore`
probes in a TP group. The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260615-auto-decision-once/20260615-230538-30b-auto-once/ssd_cold`
logged `auto_decision_count=1`, `direct_source_count=8`, `weight_load_max=17.70s`,
`LOAD_DONE=50.85s`, `read=9.73s`, and `runtime_total=10.19s`. The 235B TP=8
SSD cold rerun at
`/data/tc/qwen3-235b-source-window-runtime/20260615-auto-decision-once/20260615-230737-235b-auto-once/qwen3_235b_ssd_source_window_auto_decision_once`
logged `auto_decision_count=1`, `direct_source_count=8`, `weight_load_max=89.89s`,
`LOAD_DONE=122.12s`, `read=74.26s`, and `runtime_total=74.70s`, improving over
the prior auto-default `90.83s`/`123.54s` run while preserving the same
artifact-centered source-window path.

2026-06-16 rank-local cache singleflight validation: the 30B TP=8 SSD cold
rerun at
`/data/tc/qwen3-source-window-runtime/20260616-cache-singleflight/20260616-000150-30b-slots8/qwen3_30b_ssd_source_window_cache_singleflight_slots8`
completed with daemon startup excluded, `weight_load_max=17.06s`,
`LOAD_DONE=50.40s`, runtime `read=7.32s`, and runtime `total=9.47s`.
Every rank logged `mapped_plan_cache_hit=0`, `mapped_plan_cache_waited=0`,
`execution_template_cache_hit=0`, and `execution_template_cache_waited=0`.
Because the cache key includes rank-local target layout/index and TP topology
facts, this direction does not remove duplicated Qwen3 TP preparation work.
No 235B rerun is needed for this point; the structural no-share result is
already visible in the 30B logs.

2026-06-16 stable source-window cache-key follow-up: runtime `group_id` was
removed from the source-window plan hash and from the daemon mapped execution
template cache key when strict source-window preparation is active. The cache
key still includes rank/world-size topology facts and target/materialization
identity, so it does not make different rank-local target layouts alias. This
is a consistency prerequisite for explicit prepared realization or repeated
same-daemon runs; it is not counted as a first-load throughput fix until a
benchmark demonstrates template reuse on the hot path. Validation:
`bazel test //core/store/replica:source_window_collective_plan_test` and
`bazel test //daemon:owned_binding_service_test` both pass.

2026-06-16 source-window group-plan cache implementation: the collective disk
loader now keeps a bounded in-process source-window group-plan cache. The key
excludes runtime `group_id` and target buffer pointers, but includes artifact
identity/path facts, source index digest, world size, member rank/device,
target layout size, target storage boundaries, the full
`RepresentationWorkPlan`, and every source-window config field that affects
admission or distribution. On hit, the cached plan is copied and rebound to
the current `CollectiveLoadGroupHint`; the plan hash remains stable because it
already excludes runtime group id. Logs now include `plan_cache_hit` and
`plan_cache_key`, and the internal result carries `plan_cache_hit` for tests.
Targeted validation passed `git diff --check` and
`bazel test //core/store/replica:collective_disk_loader_test --test_output=errors`.
The new test runs the same source/work-plan/layout under two different runtime
group ids and verifies the first group misses, the second hits, and both
produce the same source-window plan hash.

Optimization judgment for this cache: it is correct and worth keeping for
repeated same-artifact/same-layout loads, but it is not the root InstantTensor
gap. The first load still builds the group input and verifies an O(work-plan)
strict key, and the best current profile shows only about `1.13s` of
source-window group planning. This is a medium-return control-plane cleanup;
the high-return work remains explicit prepared source/recipe/group
realization lifecycle and a lower-waste/coarser data-plane realization shape.

2026-06-16 source-window request shared-ref validation: the runtime request now
can pass immutable shared references for `RepresentationWorkPlan` and
`IntoTargetLayout`, while retaining value fields as a fallback for focused tests
and older call sites inside the unfinished branch. `materialization_facade`
builds these refs for source-window mapped target loads; the collective loader
stores refs in `SourceWindowMappedParticipant` and only materializes value
copies at the group planner boundary, where the current public group-plan input
still intentionally owns member values. This removes one large
request-to-participant copy without weakening group final admission semantics.
The targeted test
`try_source_window_collective_mapped_target_load performs group final admission
before runtime fallback` now leaves the value fields empty and exercises the ref
path. Validation passed:
`bazel test //core/store/replica:source_window_collective_plan_test
//core/store/replica:collective_disk_loader_test`,
`bazel test //core/store/runtime/ingestion:materialization_facade_test
//core/store/runtime/ingestion:source_bound_strategy_planner_test
//daemon:owned_binding_service_test`, and `bazel build //daemon:tensorcast_daemon`.
The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260616-shared-request-refs/20260616-002238-30b-slots8/qwen3_30b_ssd_source_window_shared_request_refs_slots8`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=16.83s`, `LOAD_DONE=48.59s`, runtime `read=7.42s`, and runtime
`total=9.49s`. This is retained as a control-plane ownership cleanup, but it is
not a root performance lever; it does not replace the remaining structural work
on group-level prepared realization/control and coalesced routed/hybrid
distribution.

2026-06-16 bounded routed packing validation: ConsumerRouted runtime now packs
small remote producer-to-consumer pieces into per-pair producer staging, sends
one packed NCCL pair to consumer staging, and scatters from that receive staging
to the final target spans. Large or capacity-overflow pieces keep the previous
direct piece send/recv fallback. This keeps the source-window group plan as the
single source of truth and changes only the runtime distribution primitive. The
targeted validation passed:
`bazel test //core/store/replica:collective_disk_loader_test`,
`bazel test //core/store/replica:source_window_collective_plan_test
//core/store/runtime/ingestion:materialization_facade_test`, and
`bazel build //daemon:tensorcast_daemon`; `git diff --check` is clean.
The 30B TP=8 SSD cold HybridWindow rerun at
`/data/tc/qwen3-source-window-runtime/20260616-packed-routed/20260616-004211-30b-hybrid-packed-slots8/qwen3_30b_ssd_source_window_hybrid_packed_routed_slots8`
completed with daemon startup excluded, `weight_load_max=17.02s`,
`LOAD_DONE=49.23s`, runtime `read=7.34s`, `collective_issue=0.49s`,
`scatter_issue=1.23s`, `routed_pack_ops=32`, and
`routed_packed_pairs=32`. It routed only `1/115` windows, so the impact is
necessarily limited.
The forced ConsumerRouted rerun at
`/data/tc/qwen3-source-window-runtime/20260616-packed-routed/20260616-004426-30b-consumer-routed-packed-slots8/qwen3_30b_ssd_source_window_consumer_routed_packed_slots8`
completed with `weight_load_max=17.53s`, `LOAD_DONE=47.81s`, runtime
`read=3.03s`, `collective_issue=0.58s`, `collective_sync=0.07s`,
`scatter_issue=6.18s`, `peer_transfer_bytes=53.75GB`,
`routed_pack_ops=142917`, `routed_packed_pairs=25105`, and
`routed_remote_pieces=104259`. This improves the best observed TensorCast 30B
`LOAD_DONE`, but it does not improve the weight-load target because routed mode
has shifted the bottleneck to GPU pack/scatter issue count and device switching.
Do not make ConsumerRouted the default on this evidence; the next routed work
should coalesce or kernelize final scatter, or route only windows whose saved
peer bytes exceed the measured pack/scatter cost.
Packed ConsumerRouted correctness was validated with
`/data/tc/run_vllm_generation_correctness.py` after adding a
`--source-window-distribution-mode` switch to that local harness. The run at
`/data/tc/qwen3-generation-correctness-20260616-packed-routed/20260616-005124-consumer-routed-packed/tensorcast`
completed with `rc=0`, daemon startup excluded, `load_done_sec=50.10s`, and
generated: `张量并行通过将大模型的张量（如权重矩阵）切分到多个设备上，实现模型参数的分布式计算，从而提升大规模模型的训练和推理效率。`

2026-06-16 ConsumerRouted deferred-2D-pack scheduling validation: the routed
executor now reserves remote pack offsets first, groups pending 2D pack copies
by producer, switches CUDA device once per producer group, then issues the
queued `cudaMemcpy2DAsync` operations before the matching NCCL send on the same
stream. Targeted validation passed `git diff --check`,
`bazel build //core/store/replica:collective_disk_loader`,
`bazel test //core/store/replica:collective_disk_loader_test`, and
`bazel build //daemon:tensorcast_daemon`. The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260616-prime-prepared/20260616-074838-prime-prepared-consumer-routed-deferred2d/qwen3_30b_ssd_prime_prepared_consumer_routed_deferred2d`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=14.99s`, and `LOAD_DONE=51.32s`. Runtime metrics improved
device switches from `61029` to `42622`, read time to `3.11s`,
`collective_issue` to `0.56s`, and total routed runtime from `10.31s` to
`9.81s`, but `scatter_issue` remained high at `5.93s` and the visible
end-to-end result did not beat prepared full-window. Keep this as a small
internal scheduling cleanup, not a default-selection reason. The next routed
work must change the dominant 2D pack/final-scatter operation shape.

2026-06-16 strict source-window data-only coverage-proof validation: strict
source-window prepare now skips owner-file collective compact coverage and
keeps only a data coverage proof for the generic fallback gate. A first
full-target proof attempt was rejected after a 30B TP=8 SSD cold run failed
before target mutation with `merged byte range map contains overlapping
segments`; the full target data segment overlapped local pad/fill bytes in the
facade map merge. The retained implementation builds compact data coverage
from `RepresentationWorkPlan`, preserving typed local pad/fill boundaries.
Targeted validation passed:
`bazel test //daemon:materialization_target_plan_utils_test
//core/store/runtime/ingestion:source_bound_strategy_planner_test
//core/store/runtime/ingestion:materialization_facade_test
//core/store/replica:source_window_collective_plan_test
//core/store/replica:collective_disk_loader_test`, followed by
`bazel build //daemon:tensorcast_daemon`. The successful 30B rerun at
`/data/tc/qwen3-source-window-runtime/20260616-fast-coverage/20260616-011134-30b-data-proof-slots8/qwen3_30b_ssd_source_window_data_proof_slots8`
completed with daemon startup excluded, `weight_load_max=16.74s`,
`LOAD_DONE=49.54s`, `compact_collective_coverage_map_used=0`,
`source_window_coverage_proof_map_used=1`, runtime `read=7.41s`,
`scatter_issue=1.18s`, and source-window runtime `total=9.48s`. This is safe
to retain, but it is a low-return control-plane cleanup; further full-target or
generic proof shortcuts would be destructive or noise-level.

2026-06-16 source-window target-storage fast-path validation: runtime scatter
now resolves destination pointers from the planner-proven consumer
`storage_index` instead of scanning every target storage span for each copy.
This makes the runtime consistent with the group plan's source/target layout
facts and fails fast if a planned span crosses its storage. Targeted validation
passed: `bazel test //core/store/replica:collective_disk_loader_test
//core/store/replica:source_window_collective_plan_test --test_output=errors`,
followed by `bazel build //daemon:tensorcast_daemon` and `git diff --check`.
The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260616-target-storage-fastpath/20260616-012915-30b-target-storage-fastpath-slots8/qwen3_30b_ssd_source_window_target_storage_fastpath_slots8`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=16.83s`, and `LOAD_DONE=49.61s`. Profile counters showed
`target_storage_fast_path_pieces=151985` and
`target_storage_fast_path_bytes=61444685824`, with runtime `read=7.47s`,
`scatter_issue=1.12s`, and total `9.48s`. The fast path is retained as a
consistency cleanup, but it proves target storage re-resolution is not a
high-return optimization area.

2026-06-16 delayed-pack ConsumerRouted experiment (rejected): I tested delaying
small linear remote pack pieces until after routed span collection, then sorting
and coalescing them before writing the producer-consumer pack stage. Targeted
tests and `bazel build //daemon:tensorcast_daemon` passed before the run. The
30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260616-delayed-pack/20260616-015434-30b-consumer-routed-delayed-pack-slots8/qwen3_30b_ssd_source_window_consumer_routed_delayed_pack_slots8`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=17.24s`, and `LOAD_DONE=50.36s`. Runtime total stayed
`9.95s`; `scatter_issue` improved from `6.18s` to `5.67s`, but `read` moved
from `3.03s` to `3.54s` and the end-to-end result regressed. The diagnostic
counters showed `routed_pending_pack_pieces=4275`,
`routed_pending_pack_bytes=732213856`,
`routed_pending_pack_coalesced_pieces=4212`, and
`routed_pack_ops=142854`. The code was reverted after measurement. This
establishes that simple delayed linear coalescing is not the missing
InstantTensor-class lever; future routed work should use a batched GPU
pack/scatter kernel or a different distribution shape.

2026-06-16 source-window group call-profile validation: the collective loader
now logs per-rank `source_window_collective_mapped_target call_profile` entries
for request preparation, group-state lookup, assemble wait, leader execution,
follower result wait, and total call time. Targeted validation passed:
`bazel test //core/store/replica:collective_disk_loader_test
//core/store/replica:source_window_collective_plan_test --test_output=errors`,
followed by `bazel build //daemon:tensorcast_daemon` and `git diff --check`.
The 30B TP=8 SSD cold profile rerun at
`/data/tc/qwen3-source-window-runtime/20260616-call-profile/20260616-020616-30b-call-profile-slots8/qwen3_30b_ssd_source_window_call_profile_slots8`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=16.77s`, and `LOAD_DONE=49.29s`. The leader reported
`leader_execute_sec=10.8221`, consisting of `plan_sec=1.13s` and runtime
`total=9.49s`; followers reported `result_wait_sec=10.8223`, with assemble
wait only `0.035-0.232s`. `source_setup` was about `0.0004s` and local typed
continuation was at most `0.022s`. This proves source-window group assembly is
not the remaining bottleneck. The next control-plane implementation should
remove repeated rank-local source-bound realization before group execution, and
the next data-plane implementation should reduce leader runtime or routed
scatter operation count.

2026-06-16 mapped-plan preparation split: `prepare_source_bound_plan` now logs
`mapped_plan_cache_key_sec`, `mapped_plan_cache_lookup_sec`,
`mapped_plan_cache_build_sec`, and `mapped_plan_cache_store_sec` in addition to
the aggregate `target_plan_sec`. This is profiling only; it does not change
admission, cache policy, or realization behavior. `bazel build
//daemon:tensorcast_daemon` passed after the instrumentation.

The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260616-mapped-plan-profile/20260616-091815-30b-mapped-plan-profile-rerun/qwen3_30b_ssd_mapped_plan_profile`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=13.56s`, and `LOAD_DONE=47.33s`. The split shows
`mapped_plan_cache_key_sec=0.022-0.023s`, lookup near zero, no cache hits or
waits, and `mapped_plan_cache_build_sec=0.682-0.729s`. Cache store ranges from
`0.026s` to `0.196s` because each rank copies its large local plan into the
bounded cache. This makes the optimization boundary clearer: compacting cache
keys and cache lookup mechanics are not first-load root levers; the meaningful
control-plane work is to remove repeated rank-local plan/build work through a
group-level prepared realization/template keyed by artifact identity, source
index digest, TP layout facts, target layout digest, and materialization
strategy.

2026-06-16 identity-keyed canonical-index cache slice: C++ canonical-index
parsing now has `parse_canonical_index_shared_with_identity(...)`, an entry
point for daemon-validated artifact/source-index identities that avoids using
the full index JSON as the cache identity. The binding realization path uses it
only when `DiskMetadata.index_multihash` matches the canonical bytes being
planned, and the cache fails closed if the same identity is presented with
different index bytes. This keeps the optimization inside TensorCast artifact
identity instead of introducing a vLLM-owned source path.

Validation:

```bash
bazel test //daemon:materialization_layout_utils_test \
  //daemon:materialization_target_plan_utils_test \
  //core/store/replica:collective_disk_loader_test \
  //core/store/replica:source_window_collective_plan_test --test_output=errors
bazel build //daemon:tensorcast_daemon
```

The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260616-identity-index-cache/20260616-093331-30b-identity-index-cache/qwen3_30b_ssd_identity_index_cache`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=13.05s`, and `LOAD_DONE=48.91s`. All ranks logged
`canonical_index_identity_parse_key=1`, but `parse_source_tables_sec` remained
`0.275-0.320s`: the first thread still parses the canonical index and
concurrent ranks wait for that in-flight parse. Therefore this slice is useful
for explicit prepared source/catalog/group-realization prewarm and repeated
loads, but it is not a first-load root fix. The next meaningful first-load
control-plane step remains group-level prepared realization/template reuse.

2026-06-16 source-window prepared-realization key diagnostics: the daemon
refill boundary now records diagnostic-only source-window prepared realization
keys. The split is:

- `source_window_prepared_group_key`: artifact/source/strategy/topology plus
  realization facts intended to describe a group-template candidate;
- `source_window_prepared_member_key`: the group key plus rank/member target
  facts;
- `source_window_realization_plan_hash`;
- `source_window_target_layout_template_hash`;
- `source_window_target_index_hash`.

Validation passed:

```bash
bazel test //daemon:owned_binding_service_test
bazel build //daemon:tensorcast_daemon
```

The 30B TP=8 SSD cold rerun at
`/data/tc/qwen3-source-window-runtime/20260616-prepared-key-components/20260616-101153-prepared-key-components/30b-prepared-key-components`
completed with daemon startup excluded, `fincore_resident=0`,
`weight_load_max=13.09s`, and `LOAD_DONE=49.59s`. The diagnostic counts are
the important result: `source_window_target_layout_template_hash` has one
unique value, `source_window_target_index_hash` has one unique value, but
`source_window_realization_plan_hash` has eight unique values across the TP
group. Target layout and target index are already group-common; the remaining
rank split is the `BindingRealizationPlan` itself. Therefore the next
implementation should not keep tightening rank-local cache keys. It should
collect per-rank realization plans as group member facts, derive one normalized
group template plus per-rank member views, and build the source-window group
plan from that normalized template.

2026-06-16 prepared source-window plan-cache key landed: the collective mapped
source-window executor can now use prepared realization facts as the opt-in
group plan-cache identity. The new key is only considered when
`enable_source_window_plan_cache` is explicitly true and every member supplies
complete prepared facts. It includes artifact id/path, source index digest,
strategy config, TP world size, and each member's prepared group key, member
key, realization-plan hash, target-layout-template hash, target-index hash,
target storage lengths, and target storage spans. Member facts are sorted by
rank/device/member key, so the identity is independent of participant arrival
order. Missing or incomplete facts fall back to the existing full-WorkPlan
cache key.

The first implementation matched the then-current production diagnostic shape
where the 30B TP=8 run reported eight values in the field then called the
prepared group key, alongside eight realization-plan hashes. The follow-up
daemon key split fixes that semantic debt: the group key now describes the
shared artifact/source/strategy and target-layout template, while the member key
carries rank-local realization-plan and target-geometry facts. The cache
remains usable with sorted member facts, but its group identity now matches the
planned group-template abstraction. Validation on
`ws-7681b3683947089e-worker-spdxq` with `/data/workspace/internal-vllm/.venv`
activated:

```bash
git diff --check -- core/store/replica/collective_disk_loader.cc \
  core/store/replica/collective_disk_loader_test.cc
bazel test //core/store/replica:collective_disk_loader_test \
  --nocache_test_results \
  --test_output=errors \
  --test_env=LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 \
  --test_env=NCCL_SOCKET_IFNAME=host0 \
  --test_env=GLOO_SOCKET_IFNAME=host0
```

The target passed. The original cache test verifies same prepared facts hit
across different runtime group ids and verifies changed prepared member facts
miss. The follow-up daemon test verifies group-template key stability across
rank-local runtime group id/rank/device changes and member-key sensitivity to
realization-plan, rank, and target-geometry changes. Existing vLLM benchmarks
still log `plan_cache_enabled=0` unless the explicit option is used, so this is
a correctness/control-plane foundation for prepared repeated loads rather than
a new 30B/235B performance row.

2026-06-16 group-template prepared-realization key normalization validation:
the real 30B TP=8 JFS run at
`/data/tc/qwen3-source-window-runtime/20260616-group-template-key/20260616-150156-20260616-group-template-key-jfs/30b-jfs-group-template-key`
used a prestarted daemon, strict `FullWindowAllGather`,
`enable_source_window_plan_cache=true`, and `fincore_resident=0`. The run
measured `weight_load_max=30.85s` and `LOAD_DONE=65.51s`; because JFS cold-read
latency varied heavily, this run is an identity validation rather than a
performance row. The daemon diagnostics are the acceptance signal:
`prepared_realization_members=8`, `prepared_realization_group_key_unique=1`,
`prepared_realization_member_key_unique=8`,
`prepared_realization_plan_hash_unique=8`,
`prepared_realization_target_layout_template_hash_unique=1`, and
`prepared_realization_target_index_hash_unique=1`.

2026-06-16 CUDA Graph scatter microbench: a benchmark-only mode
`source_window_allgather_scatter_graph` now validates a copy-engine-equivalent
scatter batching direction. The production source-window executor is unchanged.
The mode captures each rank's source-window `cudaMemcpy2DAsync` scatter sequence
as a CUDA graph, then replays it after the usual NCCL all-gather.

Validation on `ws-7681b3683947089e-worker-spdxq` used TP=8, bf16, 8x H800,
`/data/workspace/internal-vllm/.venv`, and a 128MiB synthetic window
(`rows=4096`, `cols=16384`) with correctness checks enabled. Build passed:

```bash
bazel build //core/store/materialization/benchmarks:collective_transform_microbench
```

Sweep directory:
`/data/tc/source_window_scatter_graph_sweep-20260616-132925`.

| Scatter ops/rank | Baseline receiver | Graph receiver | Receiver speedup | Baseline total | Graph total | Total speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0546ms | 0.0451ms | 1.21x | 0.886ms | 0.871ms | 1.02x |
| 42 | 1.015ms | 0.132ms | 7.70x | 1.848ms | 0.956ms | 1.93x |
| 165 | 3.683ms | 0.401ms | 9.19x | 4.507ms | 1.247ms | 3.62x |
| 512 | 11.119ms | 1.173ms | 9.48x | 11.948ms | 2.009ms | 5.95x |

A follow-up build-cost run at
`/data/tc/source_window_scatter_graph_buildcost-20260616-133143` measured 8
rank graphs, 1320 copy nodes, graph build/instantiate time `7.58ms`, and replay
receiver time `0.403ms` for 165 ops/rank. This changes the batched
pack/scatter boundary: CUDA Graph replay is a credible copy-engine-equivalent
primitive and should replace the rejected NVRTC copy-kernel direction as the
next microarchitectural candidate. However, graph capture/instantiate is too
expensive to blindly perform for every cold runtime chunk/window. The production
path should first design a prepared source-window execution template that can
own graph/node-template construction ahead of target mutation, or prove that
`cudaGraphExecMemcpyNodeSetParams` style updates can cheaply retarget a stable
node template. Until that is proven, keep the production executor on the
existing memcpy path and treat graph replay as an admitted feasibility gate, not
as a default runtime change.

2026-06-16 CUDA Graph memcpy-node update gate: the benchmark-only
`source_window_allgather_scatter_graph_update` mode now tests whether a prepared
CUDA graph can cheaply retarget source-window scatter copies. The first 2D-node
attempt failed with `cudaGraphExecMemcpyNodeSetParams(...): invalid argument`.
CUDA 12.8 documents this limitation: executable memcpy-node updates require
both the original and new memory operands to be one-dimensional, and reject
multidimensional operands.

The follow-up valid path expands every 2D scatter into row-level 1D memcpy graph
nodes, then retargets them with `cudaGraphExecMemcpyNodeSetParams1D`. This
proves retargeting semantics but rejects the row-expanded implementation as a
production path. For the same TP=8, bf16, 128MiB synthetic window and 165
scatter ops/rank:

| Mode | Nodes | Build | Receiver | Update portion | Total |
|---|---:|---:|---:|---:|---:|
| 2D static graph replay | 1,320 | 7.62ms | 0.402ms | 0 | 1.238ms |
| 1D row-expanded graph update | 32,768 | 875.6ms | 79.62ms | 28.38ms | 80.55ms |

Run directory:
`/data/tc/source_window_scatter_graph_update1d-20260616-134011`.

This narrows the production direction again. Static or exact-shape CUDA Graph
replay remains a promising prepared execution-template primitive. Retargetable
2D scatter graphs are not viable through `cudaGraphExecMemcpyNodeSetParams` as
of CUDA 12.8, and row-expanded 1D graph updates should not be implemented in
the TensorCast runtime. The next feasibility step, if graph reuse is pursued,
should be `cudaGraphExecUpdate` with a replacement 2D graph or an exact-shape
prepared graph cache keyed by admitted source-window plan/chunk geometry.

2026-06-16 CUDA Graph whole-exec update gate: a benchmark-only
`source_window_allgather_scatter_graph_exec_update` mode now tests whether a
mutable 2D memcpy graph plus `cudaGraphExecUpdate(...)` can retarget an
instantiated graph. The mode keeps topology fixed, updates 2D memcpy nodes in
the non-executable graph with `cudaGraphMemcpyNodeSetParams(...)`, calls
`cudaGraphExecUpdate(...)`, and launches the exec graph. Each measured
iteration updates to a scratch target and then back to the real target, so the
update really changes pointer parameters while correctness still checks the
real target.

Run directory:
`/data/tc/source_window_scatter_graph_exec_update-20260616-134616`.

CUDA rejects this path with `cudaGraphExecUpdateErrorParametersChanged`:

```text
cudaGraphExecUpdate(source-window scatter graph): the graph update was not
performed because it included changes which violated constraints specific to
instantiated graph update result=5
```

This closes the CUDA Graph retargeting branch for 2D source-window scatter on
CUDA 12.8. The remaining graph-compatible production option is exact-shape
static replay prepared for a specific admitted realization/chunk geometry. If
TensorCast needs retargetable scatter batching, it should use a non-graph
copy-engine primitive or change the routed/hybrid plan shape to reduce operation
count, not row-expanded 1D graph updates and not 2D exec updates.

2026-06-16 production exact-shape graph gate: a typed-config runtime
diagnostic path now tests exact-shape CUDA Graph scatter in the real
source-window loader: `enable_source_window_scatter_cuda_graph=true`. The
default is off. The path consumes the same production `SourceWindowBatchedScatterDescriptor`
vectors, builds one graph per rank/chunk with 1D or 2D memcpy nodes, launches
the graph, and falls back to the memcpy path on graph build/launch failure. It
does not change source-window admission, plan hashing, target layout semantics,
or vLLM integration.

Validation on `ws-7681b3683947089e-worker-spdxq`:

```bash
bazel test //core/store/replica:collective_disk_loader_test \
  --nocache_test_results \
  --test_output=errors \
  --test_env=LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 \
  --test_env=NCCL_SOCKET_IFNAME=host0 \
  --test_env=GLOO_SOCKET_IFNAME=host0
bazel build //daemon:tensorcast_daemon
```

The unit target passed and the daemon binary was rebuilt. The worker did not
have the earlier `/mnt/host0` SSD model copy, so the production gate was run on
JFS with
`/mnt/step3-alignment/zane/opensources_model/Qwen3-30B-A3B-Instruct-2507`.
Both control and graph-on runs used TP=8, bf16, `max_model_len=1024`,
source-window strict full-window all-gather, 8 read-ahead slots, 128MiB engine
slices, 10GB engine pool, `HF_HUB_OFFLINE=1`, `posix_fadvise(DONTNEED)`, and
`fincore_resident=0`. Daemon ready time was excluded.

| JFS 30B run | Weight load | LOAD_DONE | Runtime total | Scatter issue+sync | Graph build | Launch shape |
|---|---:|---:|---:|---:|---:|---|
| Control memcpy scatter | 25.22s | 58.62s | 21.29s | 1.286s | 0 | 151,985 memcpy launches |
| Exact-shape graph scatter | 26.26s | 59.63s | 22.70s | 2.149s | 1.106s | 3,589 graph launches + 16 memcpy launches |

Run directories:

- Control:
  `/data/tc/qwen3-source-window-runtime/20260616-scatter-cuda-graph/20260616-140349-20260616-140349-30b-jfs-scatter-control/30b-jfs-scatter-control`
- Graph-on:
  `/data/tc/qwen3-source-window-runtime/20260616-scatter-cuda-graph/20260616-140142-20260616-140142-30b-jfs-scatter-cuda-graph/30b-jfs-scatter-cuda-graph`

This rejects the hot-path exact graph implementation. It proves correctness and
cuts host launches by about 42x, but graph build/instantiate in the materialize
loop adds `1.106s` and regresses the end-to-end run by about `1.0s`. The
implementation should remain default-off diagnostic evidence. The plan should
not spend more time on CUDA Graph scatter unless one of these higher-level
preconditions becomes true:

- TensorCast owns or can stabilize target allocation addresses before the hot
  realization path, so exact graph instances can be prepared and reused.
- A future CUDA/runtime path supports cheap 2D memcpy graph retargeting, which
  CUDA 12.8 does not.
- A prepared artifact lifecycle can hide graph build without delaying
  `realize`, and can validate graph state against the admitted source-window
  plan hash and target storage spans before mutation.

Given the current vLLM-owned target allocation and CUDA 12.8 constraints, this
branch is not a path to InstantTensor-level first-load latency. The next
performance work should prioritize fixed TensorCast/vLLM orchestration costs,
prepared artifact/recipe lifecycle, source-window plan caching in real runs,
and reducing operation density through source-window distribution/layout
planning rather than graph retargeting.

2026-06-16 source-window plan cache wiring: the prepared group plan cache is now
reachable from real daemon/vLLM configuration instead of only loader tests.
Added `enable_source_window_plan_cache` to daemon materialization strategy
config, normalized it to `false` by default, mapped it into
`StoreEngineOptions::MaterializationStrategyConfig`, and propagated it through
`MaterializationFacade` into `CollectiveMappedTargetLoadOptions`. The benchmark
runner also accepts `--enable-source-window-plan-cache` and `--repeat-loads` so
same-daemon repeated realization can be measured without including daemon
startup.

Validation on `ws-7681b3683947089e-worker-spdxq`:

```bash
bazel test //core/common:daemon_config_io_test \
  //core/store/replica:collective_disk_loader_test \
  --nocache_test_results \
  --test_output=errors \
  --test_env=LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 \
  --test_env=NCCL_SOCKET_IFNAME=host0 \
  --test_env=GLOO_SOCKET_IFNAME=host0
bazel build //daemon:tensorcast_daemon
```

Both tests passed and the daemon rebuilt. A two-load same-daemon JFS 30B run
used TP=8, bf16, `max_model_len=1024`, source-window strict full-window
all-gather, 8 read-ahead slots, 128MiB engine slices, 10GB engine pool,
`HF_HUB_OFFLINE=1`, and `posix_fadvise(DONTNEED)` before each load.

Run directory:
`/data/tc/qwen3-source-window-runtime/20260616-source-window-plan-cache/20260616-141609-20260616-141609-30b-jfs-plan-cache-repeat2/30b-jfs-plan-cache-repeat2`.

| Same daemon load | plan cache | plan key | plan_sec | build_sec | runtime total | weight load | LOAD_DONE |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | miss | prepared | 0.863s | 0.857s | 26.12s | 30.11s | 64.65s |
| 2 | hit | prepared | 0.0098s | 0 | 24.97s | 26.96s | 50.19s |

The cache removes the repeated source-window group-plan build, about `0.85s`
for this 30B TP=8 plan. It is now a real TensorCast prepared/reuse mechanism.
It is not a first-load data-path optimization by itself: the first realization
still builds and stores the plan. To affect first-load latency, the admitted
plan must be built in an earlier prepare/prime lifecycle, or the vLLM side must
reuse a daemon whose prepared source-window plan cache has already been primed
for the same artifact/source-index/member realization facts.

2026-06-16 batched scatter kernel diagnostic: the default-off
`enable_source_window_batched_scatter_kernel=true` runtime path was validated
on `ws-7681b3683947089e-worker-spdxq`. The CUDA unit test
`//core/store/replica:source_window_batched_scatter_kernel_test` passed, and
`//daemon:tensorcast_daemon` rebuilt. Real 30B JFS source-window runs used TP=8,
bf16, `max_model_len=1024`, strict full-window all-gather, 8 read-ahead slots,
128MiB engine slices, 10GB pinned pool, `HF_HUB_OFFLINE=1`, and
`posix_fadvise(DONTNEED)` with `fincore_resident=0`; daemon startup and daemon
prewarm were excluded.

| JFS 30B run | Weight load | LOAD_DONE | Runtime total | Scatter issue+sync | Key scatter detail |
|---|---:|---:|---:|---:|---|
| Memcpy control | 25.22s | 58.62s | 21.29s | 1.286s | 151,985 memcpy launches |
| Batched kernel, uncached submit | 30.34s | 64.46s | 26.53s | 9.849s | 3,589 kernel launches; `kernel_submit=9.093s` |
| Batched kernel, prepared launch cache | 28.69s | 68.64s | 24.93s | 0.762s | 3,589 kernel launches; `kernel_submit=0.038s`, descriptor build `0.701s` |

The first batched-kernel profile exposed a real bug in the experimental host
wrapper: every kernel submit repeated CUDA device-property/context/cache work,
so `batched_scatter_kernel_submit` dominated scatter issue. A per-device
prepared launch-state cache, populated by daemon prewarm, reduced submit from
`9.093s` to `0.038s` and made the path correct enough as an experimental
diagnostic. The resulting scatter issue is now lower than the memcpy control.

This still does not change the main optimization plan. The maximum visible
30B gain from this branch is only the scatter-side delta, roughly `0.5-0.8s`
for the current full-window plan. It does not address prime/local-ready
recipe work, group-prepared realization, full-window peer waste, or vLLM
model-construction tail latency. Keep the batched kernel default-off and
diagnostic; continue prioritizing group-prepared lifecycle/cache and lower-waste
routed/hybrid distribution with a copy-engine-equivalent primitive.

2026-06-16 cache-hide-latency validation: I reran a same-daemon three-load
30B JFS benchmark on `ws-7681b3683947089e-worker-spdxq` with
`enable_source_window_plan_cache=true`. The worker still does not expose the
SSD `/mnt/host0` copy, so this is a cache/lifecycle validation rather than an
SSD replacement row. Each load used a new vLLM child process, the same
daemon/global store, TP=8, bf16, `max_model_len=1024`, strict source-window
full-window all-gather, 8 read-ahead slots, 128MiB engine slices, 10GB engine
pool, `HF_HUB_OFFLINE=1`, and `posix_fadvise(DONTNEED)` with
`fincore_resident=0`. Daemon startup stayed excluded.

Run directory:
`/data/tc/qwen3-source-window-runtime/20260616-cache-hide-latency/20260616-144608-20260616-cache-hide-jfs-repeat3/30b-jfs-cache-hide-repeat3`.

| Same daemon load | plan cache | plan_sec | daemon runtime total | daemon read | weight load | LOAD_DONE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | miss | 0.889s | 27.50s | 25.27s | 31.28s | 64.20s |
| 2 | hit | 0.0196s | 25.03s | 22.97s | 27.02s | 50.78s |
| 3 | hit | 0.0093s | 24.84s | 22.77s | 26.58s | 49.53s |

The source-window plan cache now answers the "can TensorCast hide prime/realize
latency with cache?" question narrowly: yes for repeated group-plan
construction, about `0.88s` on this 30B TP=8 plan; no for the data-plane read
or the vLLM engine startup tail. On the third cache-hit load,
`startup.source_bootstrap.realize` was `26.44s` max across ranks,
`local_ready.realize.realize_from` was `26.01s`, and
`startup.from_prepared_disk_source` was only `0.093s`. The daemon source-window
runtime was `24.84s`, dominated by `read=22.77s`. The vLLM log then spent
`10.29s` in `init engine (profile, create kv cache, warmup model)` after
weight loading. This cost is outside the TensorCast daemon loader and should
not be optimized by changing source-window copy mechanics.

Re-reading the earlier SSD diagnostic row (`13.09s` weight load /
`49.59s` `LOAD_DONE`) shows the same boundary. The preserved TensorCast profile
for
`/data/tc/qwen3-source-window-runtime/20260616-prepared-key-components/20260616-101153-prepared-key-components/30b-prepared-key-components`
has `startup.source_bootstrap.realize=max 12.88s`,
`local_ready.realize.realize_from=max 12.45s`, daemon data-plane
`total=9.48s`, `read=7.38s`, and
`prime.source_bootstrap.prepare_recipe=max 3.11s`. The 49s number is therefore
full vLLM engine startup latency with TensorCast included, not the daemon
source-window load time.

Updated expected optimization envelope:

- Prepared source/catalog/recipe and source-window group-plan cache can hide
  about `1-2s` in repeated same-daemon loads, and up to roughly `3s` if recipe
  construction is moved fully into a real pre-realize prime lifecycle.
- Group-prepared realization should remove roughly `1-1.5s` of first-load
  rank-local plan/execution construction and make prepared cache state useful
  before target mutation, but it is not a 10s lever.
- Full-window scatter batching is a small current-plan lever, about
  `0.5-0.8s` on 30B. Keep it default-off until it passes broader correctness
  and storage tests.
- With the current SSD daemon data-plane already around `9.5s`, the near-term
  30B SSD weight-load target is roughly `10.5-12s`. Going below that needs a
  lower-waste routed/hybrid distribution that avoids full-window peer waste
  without reintroducing routed pack/scatter overhead.
- `LOAD_DONE` can approach InstantTensor-class `33-38s` only if the
  TensorCast-specific prime/realize overhead is moved before the measured
  serving startup or amortized by a long-lived daemon, and vLLM's own fixed
  initialization tail stays comparable to the default/InstantTensor runs.
  A source-window plan cache alone cannot turn `49s` into `33s`.

## 2026-06-16 ConsumerRouted Pack/Scatter Follow-Up

I continued on the same H800 worker
`ws-7681b3683947089e-worker-spdxq`, using the activated
`/data/workspace/internal-vllm/.venv`. All real runs used TP=8, bf16,
`max_model_len=1024`, strict source-window `ConsumerRouted`, 8 read-ahead
slots, 4GiB engine pool, JFS 30B model, `HF_HUB_OFFLINE=1`,
`posix_fadvise(DONTNEED)`, and `fincore_resident=0`. Daemon/global-store
startup time stayed excluded.

Validation before real runs:

- `bazel test //core/store/replica:collective_disk_loader_test`
- `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`
- `bazel build //daemon:tensorcast_daemon`

Results:

| Run | Path | Weight load | LOAD_DONE | Daemon runtime | Scatter issue | Notes |
|---|---|---:|---:|---:|---:|---|
| `/data/tc/qwen3-source-window-runtime/20260616-routed-batched-pack-control/20260616-154253-20260616-routed-batched-pack-control-jfs/30b-jfs-consumer-routed-pack-control` | kernel off control | 14.93s | 47.77s | 10.12s | 6.20s | fallback descriptors 269,014 |
| `/data/tc/qwen3-source-window-runtime/20260616-routed-breakdown/20260616-154736-20260616-routed-breakdown-jfs/30b-jfs-consumer-routed-breakdown` | producer pack + final scatter batched | 11.09s | 50.11s | 7.23s | 5.28s | `routed_span_plan=5.13s` |
| `/data/tc/qwen3-source-window-runtime/20260616-routed-prefilter/20260616-155248-20260616-routed-prefilter-jfs/30b-jfs-consumer-routed-prefilter` | plus runtime chunk span prefilter | 10.86s | 45.74s | 7.09s | 5.20s | span refs 1,202,960 -> 153,007, but `routed_span_plan=5.02s` |
| `/data/tc/qwen3-source-window-runtime/20260616-routed-local2d-batched/20260616-155740-20260616-routed-local2d-batched-jfs/30b-jfs-consumer-routed-local2d-batched` | plus local 2D batched descriptors | 10.46s | 44.83s | 6.75s | 4.86s | `routed_local_2d_pieces=21,107`; `routed_span_plan=4.62s` |

This changes the near-term plan:

- Keep descriptor batching default-off for now, but it is no longer merely a
  final-scatter diagnostic. It now covers producer 2D pack, packed final
  scatter, and local grouped 2D copies, and it is a useful runtime primitive for
  routed/hybrid prototypes.
- Runtime chunk span prefilter is correct and worth keeping, but not a major
  performance lever. The remaining `4-5s` is per-effective-span geometry and
  target assembly.
- The next high-leverage implementation step is a compiled scatter/pack
  program: derive per-window/chunk/rank descriptors from the source-window group
  plan and target storage spans once, cache them with the prepared realization,
  and let runtime only bind stage pointers and issue descriptor batches.
- With compiled scatter/pack, this route plausibly moves 30B daemon runtime
  from `6.75s` toward `2.5-3.5s` on the measured JFS worker. That implies a
  `6-8s` weight-load target if vLLM and storage behavior stay similar, while
  `LOAD_DONE` remains bounded by vLLM fixed startup.

### Runtime-Compiled Program Slice

I implemented a first ConsumerRouted compiled scatter/pack program behind the
typed config gate `enable_source_window_compiled_routed_program=true`. It compiles
per-chunk local descriptor templates, producer pack descriptor templates,
packed transfers, packed final-scatter templates, and direct remote pieces.
The normal batched descriptor path remains default because this first compiled
program is still built synchronously inside `realize`.

Validation:

- `bazel test //core/store/replica:collective_disk_loader_test`
- `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`
- `bazel build //daemon:tensorcast_daemon`

30B JFS TP=8 forced-ConsumerRouted result:

| Run | Compiled program | Weight load | LOAD_DONE | Daemon total | read | scatter issue | routed span plan | compiled build |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `/data/tc/qwen3-source-window-runtime/20260616-routed-compiled-program/20260616-161111-20260616-routed-compiled-program-jfs/30b-jfs-consumer-routed-compiled-program` | on | 15.08s | 50.80s | 11.39s | 5.23s | 0.248s | 0s | 4.77s |
| `/data/tc/qwen3-source-window-runtime/20260616-routed-compiled-gate-off/20260616-161500-20260616-routed-compiled-gate-off-jfs/30b-jfs-consumer-routed-compiled-gate-off` | off | 14.19s | 50.63s | 10.14s | 3.96s | 5.29s | 5.08s | 0s |

This is useful evidence but not a promotion candidate. It proves that the
compiled shape can shrink runtime scatter issue to sub-second scale, but it
also shows that compiling raw target-pointer templates during `realize` just
moves the `4-5s` cost from `routed_span_plan` to
`routed_compiled_program_build`. The next implementation step should move the
stable part of this program into prepared group realization or a prime cache.
Runtime should bind only stage pointers and target base pointers.

### Target-Relative Compiled Program Cache

Implemented follow-up:

- Converted the compiled ConsumerRouted program from raw target-pointer
  templates to target-relative templates:
  `(rank, storage_index, logical_offset)`.
- Added a process-local compiled routed program cache keyed by artifact id,
  artifact path, source-window `plan_hash`, runtime chunk sizing, participant
  rank/device, source-index digest, and target storage span geometry. The key
  deliberately excludes target allocation base pointers.
- Execution now resolves current target base pointers from
  `mapped_participants` immediately before issuing local scatter, direct remote
  receives, or packed final scatter. A mismatched target layout fails before
  target mutation.
- The compiled program remains gated by
  typed `materialization_strategy.enable_source_window_compiled_routed_program`.
  Earlier benchmark-only environment gates have been removed from the source
  tree and are superseded by the typed config convergence described below.

Validation on `ws-7681b3683947089e-worker-spdxq`:

- `bazel test //core/store/replica:collective_disk_loader_test`
- `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`
- `bazel build //daemon:tensorcast_daemon`

30B JFS TP=8 repeat-load evidence:

| Iteration | Plan cache | Compiled cache | Weight load max | LOAD_DONE | Daemon total | read | compiled build | scatter issue |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | miss | miss | 15.698s | 50.342s | 11.893s | 5.715s | 4.613s | 0.352s |
| 2 | hit | hit | 8.216s | 29.242s | 6.248s | 5.364s | 0s | 0.348s |

Run root:
`/data/tc/qwen3-source-window-runtime/20260616-routed-compiled-cache/20260616-162946-20260616-routed-compiled-cache-jfs`.
Daemon/global-store ready time was excluded (`10.03s`), both iterations used
`posix_fadvise(DONTNEED)`, and `fincore_resident=0`.

This changes the implementation plan:

- Keep the current env-gated process-local cache as the feasibility path.
- Promote the cache boundary into prepared group realization next, so prime can
  construct the stable group program before target mutation.
- Keep target pointer binding in runtime execution. Prepared/cache artifacts
  should contain only layout-relative facts plus validated plan identity.
- Add focused tests for cache key sensitivity to target storage span geometry,
  source index digest, runtime chunk sizing, and plan hash before making this
  configurable outside the experimental gate.

Correctness evidence:

- Request: `请用一句话说明张量并行的作用。`
- TensorCast output:
  `张量并行是一种通过多路并行处理来提升计算效率的技术。`
- Run root:
  `/data/tc/qwen3-source-window-runtime/20260616-routed-compiled-cache-correctness/20260616-163258-30b-tensorcast-compiled-cache-correctness`.
- `rc=0`, `finish_reason=stop`.

Follow-up test coverage landed:

- Added a testing-only routed program cache-key helper that builds the key from
  public `SourceWindowCollectiveMappedTargetLoadRequest` inputs and the public
  `SourceWindowCollectivePlan`, while keeping runtime participant details
  private.
- Added a unit test proving the key excludes target allocation base pointers:
  same target span geometry with different `base_ptr` values produces the same
  key.
- The same test proves the key changes when source-index digest, runtime chunk
  sizing, target storage span geometry, or `plan_hash` changes, and remains
  stable under participant vector reordering for the same ranks/devices.
- Validation on `ws-7681b3683947089e-worker-spdxq`:
  `bazel test //core/store/replica:collective_disk_loader_test`,
  `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`,
  and `bazel build //daemon:tensorcast_daemon`.

Fresh-target runtime parity also landed:

- Added a CUDA/NCCL correctness test that runs two source-window
  ConsumerRouted loads with the experimental compiled program cache enabled.
- The second load reuses the cached compiled program but binds a fresh pair of
  target allocations. The first load's target buffers remain allocated and are
  overwritten with guard bytes before the second load, so any stale cached
  target pointer would either mutate the old buffers or leave the new buffers
  unchanged.
- The test asserts first-run routed program cache miss, second-run hit,
  identical plan hash, correct data in the fresh targets, and unchanged guard
  bytes in the old targets.
- Validation on `ws-7681b3683947089e-worker-spdxq`:
  `bazel test //core/store/replica:collective_disk_loader_test`,
  `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`,
  and `bazel build //daemon:tensorcast_daemon`.

Prepared-realization identity follow-up landed:

- The compiled routed program cache key now uses a `v2` schema and prefers
  `SourceWindowPreparedRealizationFacts` when every group participant carries a
  complete and consistent prepared group key plus member facts.
- The prepared key still includes the source-window `plan_hash`, runtime chunk
  sizing, participant rank/device, source-index digest, and target storage span
  geometry. Prepared facts therefore establish artifact/prepared-realization
  identity, while runtime facts still validate the current execution geometry.
- Incomplete or inconsistent prepared facts explicitly fall back to the runtime
  identity path instead of producing a partially prepared key.
- Added unit coverage for prepared group key, member key, target layout
  template hash, source-index digest, target span geometry, and incomplete facts.
- Validation on `ws-7681b3683947089e-worker-spdxq`:
  `bazel test //core/store/replica:collective_disk_loader_test`,
  `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`,
  and `bazel build //daemon:tensorcast_daemon`.

Pure CPU compiled-program prepare boundary landed:

- Extracted source-window runtime chunk construction and ConsumerRouted compiled
  program construction out of the hot execution lambda into reusable helpers.
  The hot runtime cache-miss path and the prepare test path now compile routed
  programs through the same function.
- Added a target geometry resolver so compiled program construction validates
  target storage span geometry without producing or storing target pointers.
  Runtime execution remains the only place that binds volatile target base
  pointers.
- Added `prepare_source_window_routed_program_cache_for_testing(...)` as a
  narrow prewarm boundary: given a built `SourceWindowCollectivePlan`, public
  mapped-target requests, and runtime chunk sizing, it compiles the
  target-relative ConsumerRouted program and stores it in the same process-local
  cache used by hot `realize`.
- Added unit coverage proving the cache can be prepared before runtime
  execution and then hit with a new set of target pointers carrying the same
  target storage geometry.
- Validation on `ws-7681b3683947089e-worker-spdxq`:
  `bazel test //core/store/replica:collective_disk_loader_test`,
  `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`,
  and `bazel build //daemon:tensorcast_daemon`.

Production prepared-group prewarm hook landed:

- Added
  `prepare_source_window_collective_routed_program_cache(...)`, a production
  core helper that accepts the same public
  `SourceWindowCollectiveMappedTargetLoadRequest` shape used by hot runtime,
  performs group final admission/plan-cache lookup, builds the source-window
  group plan when needed, and then prepares the target-relative ConsumerRouted
  compiled program cache.
- Added unit coverage that prepares the routed program cache from group
  requests, then repeats with fresh target pointers and verifies both group plan
  cache and routed program cache hits.
- Added a daemon-side prepared-group collector in `OwnedBindingService`.
  Prepared source-bound execution contributes each TP member's prepared
  realization facts, resolved work plan, target storage layout, disk context,
  source-index digest, and candidate summary. When all ranks for the current
  runtime group arrive, the daemon submits a best-effort async prewarm task to
  the blocking executor.
- The hook is gated by existing
  `materialization_strategy.enable_source_window_plan_cache`; default paths do
  not pay the extra collector/prewarm work.
- The collector does not change vLLM boundaries. vLLM still only provides
  TensorCast facts/config; TensorCast owns group plan admission, program cache,
  and runtime execution.

Validation on `ws-7681b3683947089e-worker-spdxq`:

- `bazel test //core/store/replica:collective_disk_loader_test`
- `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`
- `bazel test //daemon:owned_binding_service_test`
- `bazel build //daemon:tensorcast_daemon`

Remaining before promotion: profile the production prepared-group hook on 30B
and 235B. The current hook can hide compile latency when prime/prepared
realization runs early enough or when the daemon cache is reused. In an
immediate `prepare -> realize` flow it is intentionally best-effort and may race
with hot runtime cache lookup; if that race misses too often, add an inflight
program-cache coordination primitive instead of moving pointer-bound work back
into runtime.

Singleflight and correctness follow-up landed:

- Added an in-flight state to the process-local compiled routed program cache.
  A runtime realization that reaches the same cache key while prewarm is
  building now waits for the build instead of compiling the same program again.
  The cache remains target-relative; target pointers are still rebound only at
  execution time.
- 30B JFS TP=8 evidence with daemon prestarted and `posix_fadvise(DONTNEED)`:

  | Run | Weight load | LOAD_DONE | Runtime program state | Runtime total | read | scatter issue |
  |---:|---:|---:|---|---:|---:|---:|
  | 1 | 15.154s | 52.840s | waited 4.68s on prewarm build | 11.44s | 4.78s | 1.08s |
  | 2 | 7.965s | 30.532s | cache hit, wait 0s | 5.73s | 4.27s | 1.11s |

  Run root:
  `/data/tc/qwen3-source-window-runtime/20260616-compiled-safe/20260616-184115-30b-jfs-consumer-routed-compiled-safe`.
  This shows the stable target after the compile program is cached: the
  remaining daemon materialization is roughly one cold read plus about `1.4s`
  of routing/scatter/control overhead. First-load can only approach that number
  if prime/prepared realization starts this CPU program build earlier than the
  current immediate `prepare -> realize` sequence.

- Correctness testing found that the combined experimental path
  `enable_source_window_compiled_routed_program=true` plus
  `enable_source_window_batched_scatter_kernel=true` can corrupt
  generation, even though consumer-routed baseline, compiled-only, and
  batched-only runs are each correct. Runtime now rejects that unsafe
  combination by disabling the batched scatter kernel whenever compiled routed
  program is enabled, and logs a warning. The safe guard was validated with the
  same prompt as the default loader:
  `请用一句话说明张量并行的作用。`
- Correctness run roots:
  `/data/tc/qwen3-generation-correctness-20260616-singleflight/20260616-183321-30b-tensorcast-consumer-routed-no-experimental`,
  `/data/tc/qwen3-generation-correctness-20260616-singleflight/20260616-183447-30b-tensorcast-consumer-routed-compiled-only`,
  `/data/tc/qwen3-generation-correctness-20260616-singleflight/20260616-183613-30b-tensorcast-consumer-routed-batched-only`,
  and
  `/data/tc/qwen3-generation-correctness-20260616-singleflight/20260616-183924-30b-tensorcast-combo-guard-correctness`.
- The next batched-scatter task should not re-enable the combined path until it
  has a large-model parity test that exercises compiled program templates,
  descriptor batching, packed remote pieces, direct remote pieces, and
  multi-storage target spans together. The current small CUDA unit test proves
  target-pointer rebinding, but it is not enough to cover the compiled+batched
  descriptor interaction seen on Qwen3-30B.
- The attempted 235B rerun in this intermediate step failed before loading with
  an empty safetensors list. That status is superseded by the later SSD/235B
  coverage and typed-config reruns below on the same worker, where
  `/mnt/host0/vllm-loader-bench/qwen3-235b-a22b-instruct-2507` was available.

2026-06-16 compiled+batched staging fix:

- The corruption in the combined experimental path was traced to host
  descriptor staging lifetime, not to the source-window plan or target-relative
  compiled program abstraction. The batched scatter launcher performs an async
  H2D copy from pinned host descriptors; the caller reused one pinned descriptor
  buffer per rank and could overwrite it before the previous copy consumed it.
- Runtime now uses a 16-slot per-rank pinned descriptor ring. Each slot records
  a CUDA event on the rank stream after launch and is only reused after
  `cudaEventQuery/Synchronize` confirms the stream has consumed the descriptor
  data. Device descriptor buffers remain one per rank because stream ordering
  safely serializes device-buffer reuse.
- The temporary guard that disabled batched scatter when compiled routed
  program was enabled has been removed. The CUDA/NCCL compiled program cache
  correctness test now enables both
  `enable_source_window_compiled_routed_program=true` and
  `enable_source_window_batched_scatter_kernel=true`.
- Validation on `ws-7681b3683947089e-worker-spdxq`:
  `bazel test //core/store/replica:source_window_batched_scatter_kernel_test`,
  `bazel test //core/store/replica:collective_disk_loader_test`,
  `bazel test //daemon:owned_binding_service_test`, and
  `bazel build //daemon:tensorcast_daemon`.
- 30B correctness with both experimental paths enabled now produces normal text
  for the same prompt that previously exposed repeated-token corruption. Run
  root:
  `/data/tc/qwen3-generation-correctness-20260616-compiled-batched-fixed/20260616-185726-30b-tensorcast-compiled-batched-fixed`.

2026-06-16 direct-I/O benchmark update after the fix:

- A buffered rerun was intentionally kept as a diagnostic: without
  `local_mapped_safetensors_io_mode=direct`, JFS auto mode selected buffered
  I/O and daemon read grew to `21-24s`. This is not the intended TensorCast
  source-window performance profile.
- The official direct-I/O rerun used daemon-prestarted TP=8, strict
  ConsumerRouted, plan cache, compiled routed program, batched scatter,
  `posix_fadvise(DONTNEED)`, and `fincore_resident=0`.
- Run root:
  `/data/tc/qwen3-30b-a3b-instruct-tensorcast-predaemon-20260614-compare/20260616-190427-30b-jfs-consumer-routed-compiled-batched-direct-fixed`.

  | Iteration | Weight load max | LOAD_DONE | Daemon total | Read | Scatter issue |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 14.107s | 47.904s | 10.517s | 4.359s | 0.428s |
  | 2 | 8.620s | 36.336s | 6.545s | 5.595s | 0.424s |

- Compared with the previous compiled-safe guard run, batched scatter reduces
  scatter issue from roughly `1.1s` to `0.42s`. End-to-end hot iteration did
  not improve in that sample because direct-read time increased by about
  `1.3s`; the daemon runtime is still dominated by physical read variance.
- Promotion status: compiled+batched is correct on 30B and should remain
  experimental until it has SSD and 235B coverage. The next root optimization
  is lifecycle timing: make prepared group prewarm deterministic and early
  enough that first `realize` observes the same plan/program cache state as the
  second iteration.

2026-06-16 SSD and 235B coverage update:

- Fresh SSD copies were created on
  `ws-7681b3683947089e-worker-spdxq` under `/mnt/host0/vllm-loader-bench`.
  The 30B copy has 16 safetensors and the 235B copy has 118 safetensors. All
  benchmark runs below used `internal-vllm/.venv`, TP=8, bf16,
  `max_model_len=1024`, source-window strict ConsumerRouted, plan cache,
  compiled routed program, batched scatter, and daemon/global-store prestart.
  `DAEMON_READY` was recorded and excluded from `LOAD_DONE`.
- 30B SSD cold, run root:
  `/data/tc/qwen3-30b-a3b-instruct-tensorcast-predaemon-20260614-compare/20260616-192930-30b-ssd-consumer-routed-compiled-batched-fixed`.
  Auto I/O selected `direct_aligned_edges`, both iterations used
  `posix_fadvise(DONTNEED)`, and both recorded `fincore_resident=0`.

  | Iteration | Weight load max | LOAD_DONE | Daemon total | Read | Program wait | Scatter issue |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | 19.704s | 58.150s | 14.658s | 8.708s | 4.581s | 0.375s |
  | 2 | 11.537s | 33.916s | 9.616s | 8.798s | 0s | 0.340s |

- 235B SSD cold, run root:
  `/data/tc/qwen3-235b-a22b-instruct-tensorcast-predaemon-20260616-compare/20260616-194048-235b-ssd-consumer-routed-compiled-batched-fixed`.
  Auto I/O selected `direct_aligned_edges`, both iterations used
  `posix_fadvise(DONTNEED)`, and both recorded `fincore_resident=0`.

  | Iteration | Weight load max | LOAD_DONE | Daemon total | Read | Program wait | Scatter issue |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | 100.436s | 148.581s | 92.481s | 68.108s | 18.526s | 1.988s |
  | 2 | 76.838s | 104.737s | 72.968s | 67.173s | 0s | 2.094s |

- 235B confirms the group source-window objective at scale:
  `bytes_read=470.187GB` for `470.192GB` of safetensors payload, so effective
  read amplification remains about 1.0x. The hot daemon path is dominated by
  direct SSD read time (`67.2s` of `73.0s`).
- The compiled+batched backend now has 30B correctness, 30B SSD, and 235B SSD
  coverage. It should still remain default-off in typed strategy config until
  the prepared-realization/prime scheduling is improved: first-load still waits
  on program prewarm (`4.6s` on 30B and `18.5s` on 235B), while same-daemon hot
  loads show the intended performance envelope.

2026-06-16 parallel compiled-program build update:

- Implemented bounded CPU-parallel construction of the target-relative
  ConsumerRouted compiled program. The implementation parallelizes independent
  source-window chunks and records `routed_compiled_program_build_threads` in
  runtime metrics. It is controlled by typed config
  `source_window_compiled_program_build_threads`; `0` keeps auto mode with a
  conservative 16-thread cap for large plans.
- Validation on `ws-7681b3683947089e-worker-spdxq`:
  `bazel test //core/store/replica:collective_disk_loader_test`,
  `bazel test //daemon:owned_binding_service_test`, and
  `bazel build //daemon:tensorcast_daemon`.
- 30B SSD first-load A/B:

  | Build threads | Weight load max | LOAD_DONE | Prewarm program build | Runtime program wait | Daemon total | Read |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | 17.952s | 53.442s | 4.876s | 4.544s | 14.433s | 8.516s |
  | 16 | 13.708s | 48.527s | 0.328s | 0s | 9.957s | 8.561s |

- 235B SSD first-load with 16 build threads:

  | Weight load max | LOAD_DONE | Prewarm program build | Runtime program wait | Daemon total | Read |
  |---:|---:|---:|---:|---:|---:|
  | 80.903s | 116.977s | 1.260s | 0.512s | 73.913s | 67.392s |

- Correctness was revalidated after this change with the same prompt through
  vLLM default and TensorCast. Both returned the same coherent Chinese answer
  with `finish_reason=stop`. Run root:
  `/data/tc/qwen3-30b-a3b-instruct-generation-correctness-20260616/20260616-200500-default-vs-tensorcast-parallel-build`.
- Plan status: CPU program construction is no longer a primary blocker. The
  remaining first-load work is lifecycle placement and vLLM integration
  timing: prime/prepared realization should start the TensorCast-owned
  target-relative program earlier, and `LOAD_DONE` remains sensitive to vLLM
  post-weight initialization that is outside daemon materialization.

2026-06-16 typed config convergence update:

- Replaced the remaining source-window fast-path environment gates with typed
  `materialization_strategy` fields:
  `enable_source_window_batched_scatter_kernel`,
  `enable_source_window_compiled_routed_program`,
  `source_window_compiled_program_build_threads`, and
  `enable_source_window_scatter_cuda_graph`.
- `StoreEngineOptions`, daemon YAML/proto normalization, and `server_main`
  now carry these fields into both runtime materialization and deferred
  startup prewarm. Routed-program prewarm skips cleanly when
  `enable_source_window_compiled_routed_program=false`, so the daemon no
  longer builds an unused program cache.
- `/data/tc/run_qwen3_tensorcast_tp8_profile.py` and
  `/data/tc/run_tensorcast_predaemon_bench.py` now express best-practice
  compiled ConsumerRouted runs through daemon config rather than
  `TENSORCAST_SOURCE_WINDOW_*` process environment variables.
- Validation:
  `bazel test //core/common:daemon_config_io_test
  //core/store/replica:source_window_collective_plan_test
  //core/store/replica:collective_disk_loader_test --test_output=errors`
  passed locally.
- vLLM boundary validation:
  `pytest -q tests/model_executor/model_loader/test_tensorcast_loader_config.py
  tests/model_executor/model_loader/test_tensorcast_adapter.py
  tests/model_executor/model_loader/test_tensorcast_family_registry.py
  tests/model_executor/model_loader/test_tensorcast_source_catalog.py
  tests/model_executor/model_loader/test_tensorcast_qwen3_moe_semantics.py`
  passed with `79 passed, 1 warning` in `32.81s` from
  `/data/workspace/internal-vllm/.venv`.
- GPU-dependent daemon validation was rerun on
  `ws-7681b3683947089e-worker-spdxq` via `brainctl exec -n shai-core` as
  `luoyuchu`, with `/data/workspace/internal-vllm/.venv` activated. The worker
  reported torch `2.8.0+cu128`, `cuda_available=True`, and `device_count=8`.
  `bazel test //daemon:owned_binding_service_test --test_output=errors
  --test_env=LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64`
  passed in `50.4s`. This target is intentionally not judged from the local
  host because the local host has no CUDA driver.
- Completion-audit regression sweep:
  `bazel test //core/common:daemon_config_io_test
  //core/store/replica:source_window_collective_plan_test
  //core/store/materialization/dataplane:source_window_scheduler_test
  //core/store/runtime/ingestion:source_bound_strategy_planner_test
  --test_output=errors` passed locally. The same sweep's
  `//core/store/runtime/ingestion:materialization_facade_test` member fails on
  the local no-GPU host at `cudaHostRegister`, then passes on
  `ws-7681b3683947089e-worker-spdxq` with the internal-vllm venv and CUDA 12.8
  `LD_LIBRARY_PATH` propagated through Bazel `--test_env`, in `171.9s`.

2026-06-16 typed config remote runtime validation:

- Remote worker: `ws-7681b3683947089e-worker-spdxq`, user `luoyuchu`,
  `internal-vllm/.venv`, TP=8, bf16, `max_model_len=1024`,
  `HF_HUB_OFFLINE=1`.
- 30B profile run root:
  `/data/tc/qwen3-30b-tp8-profile-sourcewindow-typedconfig-20260616-213232-20260616-213232`.
  Daemon startup was excluded (`7.291s`). The run reported
  `load_model.summary max=6.742s`, `/health=79.126s`, and
  `local_ready.bootstrap.realize_binding max=6.224s`.
- The effective daemon config, not environment variables, enabled the fast path:
  `enable_source_window_plan_cache=true`,
  `enable_source_window_batched_scatter_kernel=true`,
  `enable_source_window_compiled_routed_program=true`, and
  `source_window_compiled_program_build_threads=16`.
- Daemon profile confirmed the intended source-window shape:
  `build_byte_range_maps=0`, `routed_compiled_program_enabled=1`,
  `routed_compiled_program_cache_hit=1`, `batched_scatter_kernel_enabled=1`,
  prewarm `program_build_threads=16`, runtime `total=2.935s`,
  `read=0.646s`, and `scatter_issue=0.848s`.
- Correctness run root:
  `/data/tc/qwen3-30b-typedconfig-correctness-20260616/20260616-213532-tensorcast-typedconfig`.
  The prompt `请用一句话说明张量并行的作用。` returned a normal coherent answer
  with `finish_reason=stop`; `LOAD_DONE=40.540s`, generation `2.814s`.
- 30B loader benchmark rerun with the same typed config:
  `/data/tc/qwen3-30b-a3b-instruct-tensorcast-typedconfig-20260616-logfix/20260616-215644-typedconfig-consumer-routed-logfix`.
  Both loads used `posix_fadvise(DONTNEED)`, recorded `fincore_resident=0`,
  and kept daemon startup excluded (`daemon_ready_sec_excluded=9.324s`).
  The rerun also validates that vLLM attempt logs are persisted under the shared
  case `logs/` directory rather than worker-local `/tmp`.

  | Iteration | Weight load max | LOAD_DONE | Plan cache | Plan sec | Program prepare | Runtime total | Read | Scatter issue |
  |---:|---:|---:|---|---:|---:|---:|---:|---:|
  | 1 | 13.961s | 47.804s | miss | 0.885s | 1.225s | 9.960s | 8.599s | 0.401s |
  | 2 | 11.736s | 35.059s | hit | 0.006s | 0.107s | 9.560s | 8.751s | 0.419s |

  Daemon metrics confirmed `read_amplification_x1000=1000`,
  `peer_waste_bytes=0`, `batched_scatter_kernel_enabled=1`,
  `routed_compiled_program_cache_hit=1`, and
  `routed_compiled_program_build=0s` on runtime execution. This is the clearest
  current evidence for the cache/lifecycle boundary: TensorCast can hide most
  source-window plan and compiled-program construction in a same-daemon flow,
  bringing 30B SSD to `11.736s` weight load and `35.059s` `LOAD_DONE`, close to
  the InstantTensor sample (`10.43s` / `33.06s`) without adding a vLLM-private
  loader path. The remaining gap is primarily cold read/runtime and vLLM engine
  tail, not source-window group-plan construction.
- 30B JFS/tmpfs typed-config matrix:
  JFS run root:
  `/data/tc/qwen3-30b-a3b-instruct-tensorcast-typedconfig-20260616-jfs-tmpfs/20260616-220522-typedconfig-jfs`.
  tmpfs run root:
  `/data/tc/qwen3-30b-a3b-instruct-tensorcast-typedconfig-20260616-jfs-tmpfs/20260616-220914-typedconfig-tmpfs-buffered`.
  JFS used `posix_fadvise(DONTNEED)` and `fincore_resident=0`; tmpfs used the
  resident `/dev/shm` copy and buffered safetensors I/O.

  | Storage | Iteration | Weight load max | LOAD_DONE | Plan sec | Program prepare | Runtime total | Read | Scatter issue |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | JFS cold | 1 | 11.685s | 44.974s | 0.850s | 1.234s | 7.719s | 6.279s | 0.429s |
  | JFS cold | 2 | 7.440s | 31.855s | 0.006s | 0.108s | 5.393s | 4.494s | 0.446s |
  | tmpfs resident buffered | 1 | 7.618s | 42.674s | 0.849s | 1.225s | 3.873s | 2.129s | 0.630s |
  | tmpfs resident buffered | 2 | 4.280s | 28.065s | 0.007s | 0.115s | 1.999s | 0.470s | 0.810s |

  A deliberately misconfigured tmpfs direct-I/O attempt failed before target
  publication with `O_DIRECT open failed ... Invalid argument`; the run is
  preserved at
  `/data/tc/qwen3-30b-a3b-instruct-tensorcast-typedconfig-20260616-jfs-tmpfs/20260616-220741-typedconfig-tmpfs`.
  This is the expected filesystem capability boundary: tmpfs resident
  benchmarks should use buffered or auto I/O, while SSD/JFS cold benchmarks use
  direct aligned safetensors I/O.

2026-06-16 typed config 235B rerun:

- Remote worker: `ws-7681b3683947089e-worker-spdxq`, user `luoyuchu`,
  `internal-vllm/.venv`, TP=8, bf16, `max_model_len=1024`,
  `HF_HUB_OFFLINE=1`.
- Model path:
  `/mnt/host0/vllm-loader-bench/qwen3-235b-a22b-instruct-2507`.
  It has 118 safetensors and `470191875040` bytes.
- Run root:
  `/data/tc/qwen3-235b-a22b-instruct-tensorcast-typedconfig-20260616/20260616-214151-typedconfig-consumer-routed`.
  Both loads used `posix_fadvise(DONTNEED)` and recorded
  `fincore_resident=0`; daemon startup was excluded
  (`daemon_ready_sec_excluded=9.423s`).
- Effective config matched the intended TensorCast best-practice path:
  source-window strict `ConsumerRouted`, direct local mapped safetensors I/O,
  plan cache, compiled routed program, batched scatter, and
  `source_window_compiled_program_build_threads=16`.

  | Iteration | Weight load max | LOAD_DONE | Prepare total | Program build | Runtime total | Read | Scatter issue |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 1 | 80.754s | 117.073s | 3.009s | 1.264s | 73.683s | 67.392s | 1.998s |
  | 2 | 77.225s | 102.132s | 0.284s | 0s | 73.320s | 68.405s | 1.977s |

- Daemon metrics confirmed `build_byte_range_maps=0`,
  `read_amplification_x1000=1000`, `bytes_read=470187269120`,
  `peer_transfer_bytes=413119274016`, `routed_compiled_program_enabled=1`,
  `routed_compiled_program_cache_hit=1`, and
  `batched_scatter_kernel_enabled=1`.
- Interpretation: for 235B, TensorCast now reads the model once and sustains
  about SSD-direct-class runtime throughput. Same-daemon load 2 reaches
  `77.225s` weight load and `102.132s` `LOAD_DONE`, slightly better than the
  existing 235B InstantTensor `105.86s` end-to-end sample while still slower on
  the weight-load section (`70.69s` for InstantTensor). First-load remains
  behind because it still pays prepare/program/cache placement and vLLM-visible
  model initialization tail that is not fully hidden before `realize`.
- Latest 235B typed-config generation correctness:
  `/data/tc/qwen3-235b-typedconfig-generation-correctness-20260616/20260616-221522-tensorcast-typedconfig-consumer-routed/tensorcast`.
  The run used TP=8, bf16, `max_model_len=1024`, SSD `/mnt/host0`, direct
  safetensors I/O, strict source-window `ConsumerRouted`, plan cache, compiled
  routed program, batched scatter, and daemon/global-store prestart. It returned
  `rc=0`, `finish_reason=stop`, `LOAD_DONE=118.298s`, generation time
  `3.715s`, and the coherent answer:
  `张量并行通过将大型张量计算任务拆分到多个设备上并行执行，从而加速深度学习模型的训练和推理过程。`
  Daemon logs confirmed `local_mapped_safetensors_io_mode=direct_aligned_edges`,
  `read_amplification_x1000=1000`, `batched_scatter_kernel_enabled=1`, and
  `routed_compiled_program_cache_hit=1`.

2026-06-16 remote worker completion audit:

- Local host has no usable GPU, so final GPU validation was run through
  `brainctl exec process/ws-7681b3683947089e-worker-spdxq -n shai-core` as
  `luoyuchu`, after activating `/data/workspace/internal-vllm/.venv`. The worker
  reported torch `2.8.0+cu128`, CUDA available, 8 visible H800 GPUs, and
  TensorCast imported from `/data/workspace/tensorcast-280`.
- The worker does not have `eth0`; it exposes `brainpf*`, `brainvf*`, and
  `ens20np0`. A `collective_disk_loader_test` attempt that inherited
  `NCCL_SOCKET_IFNAME=eth0` failed before TensorCast execution with
  `ncclCommInitAll` bootstrap errors. Rerunning with
  `--test_env=NCCL_SOCKET_IFNAME=ens20np0` passed:
  `//core/store/replica:collective_disk_loader_test PASSED in 22.6s`.
- GPU-only facade coverage remains validated on the worker, not on the local
  no-GPU host: `//core/store/runtime/ingestion:materialization_facade_test`
  passed remotely in `171.9s` with CUDA 12.8 `LD_LIBRARY_PATH` propagated via
  Bazel `--test_env`.
- The latest remote 30B generation correctness run used the typed-config
  strict `ConsumerRouted` path with compiled routed program and batched scatter:
  `/data/tc/qwen3-30b-typedconfig-generation-correctness-remote-20260616/20260616-230232-tensorcast-typedconfig-consumer-routed-remote`.
  It returned `rc=0`, `finish_reason=stop`, `load_done_sec=47.346s`,
  generation `2.260s`, and a coherent Chinese answer. Daemon ready time
  (`9.625s`) was recorded separately and excluded from `load_done_sec`.
- vLLM loader boundary tests were rerun with the explicit interpreter
  `/data/workspace/internal-vllm/.venv/bin/python -m pytest` and passed:
  `79 passed, 1 warning in 32.73s`.
- TensorCast Python API/runtime tests were also rerun with the explicit
  internal-vLLM Python. The initial failures were caused by missing
  `libnvshmem_host.so.3` on the dynamic loader path. Adding
  `/data/workspace/internal-vllm/.venv/lib/python3.10/site-packages/nvidia/nvshmem/lib`
  to `LD_LIBRARY_PATH` made the selected Python suites complete with exit code
  0.
- Reproducibility warning: `internal-vllm/.venv/bin/pytest` currently has a
  stale shebang pointing at `/data/workspace/tensorcast-280/.venv/bin/python3`.
  Use `/data/workspace/internal-vllm/.venv/bin/python -m pytest` for all
  internal-vLLM-environment validation.
