---
slug: batched-owner-file-collective-executor
title: Batched Owner-File Collective Executor
status: implemented
areas: ["core", "daemon", "sdk", "serving", "benchmarks"]
created: 2026-03-24
last_updated: 2026-04-15
related_code:
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/replica.cc
  - core/store/replica/replica_load_controller.cc
  - core/store/store_engine_options.h
  - core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc
  - daemon/service/controllers/replica_materialization_service.cc
  - daemon/app/server_main.cc
  - tensorcast/api/store/artifact.py
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/internals/disk-load-strategy.md
  - docs/internals/model-loading.md
  - docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md
links:
  dependencies:
    - ./0108-tensor-aware-materialization-strategy-plane.md
  related:
    - ./0107-retrieval-policy-plane-cleanup.md
    - ./0112-binding-native-serving-realization-and-publication.md
    - ../benchmarks/20260415-qwen2.5-32b-mounted-collective-first-v4-serving-evidence.md
    - ../internals/disk-load-strategy.md
    - ../internals/model-loading.md
    - ../benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md
  predecessors:
    - ./0108-tensor-aware-materialization-strategy-plane.md
    - ./0107-retrieval-policy-plane-cleanup.md
    - ./0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
---

# Summary

Introduce a new common-runtime executor:

- `OwnerFileBatchedCollectiveExecutor`

This executor is designed to replace the current eager owner-file collective
prototype for shared-filesystem, same-host tensor-parallel cold starts.

It is intended to become the default collective executor for that workload
family only after the pre-109 strategy-plane convergence work is complete.

The design is inspired by the parts of `fastsafetensors` that are currently
winning in practice:

- owner-file source ownership,
- bounded file batching,
- release-after-use memory behavior,
- GPU-side cross-rank distribution.

But it is not a framework-specific clone of `fastsafetensors`. It remains
strictly inside TensorCast's selection-first architecture:

- `ArtifactSelection` remains the only public selection contract,
- `MaterializationFacade` remains the lowering boundary,
- `ByteRangeMap` remains the canonical fallback IR,
- source acquisition stays separate from semantic truth and executor choice,
- execution environment facts stay separate from retrieval policy and semantic
  truth,
- mixed execution remains allowed.

The key idea is:

- keep the good part of file ownership and cross-rank source dedup,
- remove the bad part of eager whole-owner payload preloading,
- preserve tensor and target-layout semantics all the way to executor lowering,
- use bounded batches that read only necessary source spans and release owner
  staging immediately after commit.

This design exists to close the remaining shared-filesystem gap against
`fastsafetensors` without violating TensorCast's internal layering and without
reverting to vLLM- or model-specific special cases.

Normative source rule:

- TensorCast does not need to preserve broad production compatibility while this
  executor lands,
- the repo-owned benchmark notes, strategy-plane designs, and this design are
  the normative record for what TensorCast adopts,
- external `fastsafetensors` code is non-normative implementation reference
  only,
- temporary migration fallbacks are acceptable only to stage rollout evidence
  and must be removed once the new executor is proven.

# Implementation Status

The phase-1 scope of this design is now landed in the repository:

- ordinary disk startup requires explicit shared-source proof before
  owner-file collective becomes eligible under `AUTO`
- `MaterializationFacade` prefers owner-file collective for eligible
  shared-source TP startup and keeps host-local or unproven-source requests on
  local-batched or generic execution
- `collective_disk_load` now builds a bounded owner batch plan with weighted
  file ownership, optional job-level split for skew control, and batch-local
  owner staging
- replicated, dim0, and dim1 collective tensor work now execute through the
  owner rank without eager `owned_payload` residency
- the selected owner-file batched route no longer drops back to root
  whole-source preload scaffolding, and execution diagnostics now surface
  actual unique-source bytes, peer-transfer bytes, peak temporary bytes, batch
  count, and dedup savings

The implemented steady-state scope remains intentionally zero-residual-only:

- owner-file collective eligibility still rejects requests with true generic
  residual bytes
- mixed execution remains strategy-owned:
  - local typed work must already be peeled off before this executor is chosen,
  - typed work that is not admitted to the collective lane must remain typed in
    shared planning rather than being silently recast as residual inside this
    executor
- collective runtime therefore remains collective-lane-only rather than
  becoming the owner of mixed-execution semantics

Residual execution tracking is now closed back into the surviving design and
benchmark record:

- the standalone `0109-01` rollout plan is retired,
- the current mounted collective evidence lives in
  `docs/benchmarks/20260415-qwen2.5-32b-mounted-collective-first-v4-serving-evidence.md`,
- `0108` remains the sole strategy-plane owner,
- `0112` remains the same-binding serving-path owner,
- and this design now records only the executor-specific policy and runtime
  semantics that survived closure.

## Closure Evidence

The `2026-04-15` mounted qwen2.5 TP4 rerun on the current `v4` contract shows:

- `bootstrap_source_bound_contract_version=4`
- `bootstrap_source_bound_contract_path=collective_first_v4`
- `bootstrap_realize_collective_policy=collective_first`
- `planned_collective_admitted_bytes=16382928896`
- `actual_collective_committed_bytes=16382928896`
- `actual_generic_backend_bytes=0`
- `dominant_executor=OwnerFileCollectiveExecutor`
- `collective_unique_source_bytes=75762083840`
- `collective_peer_transfer_bytes=49148786688`
- `collective_peak_temporary_bytes=268435456`
- `collective_batch_count=437`
- `collective_dedup_saving_bytes=0`
- and serving correctness probes `/health`, `/v1/models`, and
  `/v1/completions` all returned `200`.

The companion benchmark note also records the same-host TP4 `safetensors`
baseline:

- weight loading `6.26s` to `6.28s`,
- ready wall time about `31s`,
- versus TensorCast mounted ready wall time about `59s`.

That comparison closes the remaining policy ambiguity:

- host-local workloads do **not** graduate to collective default,
- shared-source workloads remain collective-eligible only under explicit
  shared-source proof plus the typed daemon policy,
- and the current qwen2.5 mounted packet is sufficient to validate the
  collective-dominant executor path without granting a broader "all local loads
  should default to collective" conclusion.

# Sequencing Note

This design is downstream of two prerequisites:

- the runtime-critical request-normalization boundary from `0107`,
- the common-runtime strategy convergence work now folded back into `0108`.

Those prerequisites are now landed in this repository. `0109` should start from
the already-converged seams:

- normalized retrieval-policy versus execution-topology inputs,
- ordinary-disk `ExecutionStrategyPlan` ownership in `MaterializationFacade`,
- executor-runner-only replica/runtime ownership,
- typed owner-file budget and timeout fields under
  `engine.materialization_strategy`.

It must not be used to justify adding more branch-order policy to `Replica`,
`ReplicaLoadController`, or `collective_disk_loader.cc`. The executor belongs on
top of the converged strategy-plane boundary, not inside today's prototype
ownership seams.

# Non-Normative External Reference Material

This design intentionally uses `fastsafetensors` as an implementation
reference, but only for the parts that are actually winning in practice.

Those references are illustrative only. The absorbed invariants are restated in
this design so implementation and review do not depend on an external workspace
being present.

Relevant local workspace references:

- code:
  - `/data/workspace/fastsafetensors/fastsafetensors/loader.py`
  - `/data/workspace/fastsafetensors/fastsafetensors/file_buffer.py`
  - `/data/workspace/fastsafetensors/fastsafetensors/tensor_factory.py`
- docs:
  - `/data/workspace/fastsafetensors/docs/architecture.md`
  - `/data/workspace/fastsafetensors/docs/safetensors-load-optimization.md`

The most important implementation behaviors are:

- owner-file batching and orchestration:
  - `BaseSafeTensorsFileLoader.add_filenames(...)`
  - `BaseSafeTensorsFileLoader.copy_files_to_device(...)`
  - `/data/workspace/fastsafetensors/fastsafetensors/loader.py`
- per-batch device-buffer lifetime:
  - `FilesBufferOnDevice`
  - `fb.close()`
  - `/data/workspace/fastsafetensors/fastsafetensors/file_buffer.py`
- owner-mediated GPU-side distribution:
  - `LazyTensorFactory.shuffle(...)`
  - `LazyTensorFactory.push(...)`
  - `/data/workspace/fastsafetensors/fastsafetensors/tensor_factory.py`

The key lesson from those files is not merely “use owner rank”.
The key lesson is:

- bound temporary owner memory by batch,
- distribute from the owner immediately,
- release current batch buffers after use,
- never require a whole-model extra owner payload on GPU.

# Problem Statement

The current repository has three relevant disk-backed execution shapes for
ordinary `tensor_dict(...)` startup:

- `collective_disk_load`
- `local_batched_disk_load`
- generic residual `ByteRangeMap` fallback

The current shared-filesystem regression shows that these shapes are not enough.

Observed behavior on the Step3p5 TP=8 workload:

- Current eager collective path can read far more source bytes than the actual
  selected TP-local target workload.
  - Example observed run:
    - selected `dim0` workload about `23.14 GiB`
    - current collective `dim0_source_bytes` about `198.76 GiB`
    - `dim0_exec` about `115.61s`
- Current eager owner-file collective prototype avoids part of that amplification
  but still preloads whole owner payloads into GPU memory.
  - For the current 44-shard model, that means roughly `21 GiB` to `30 GiB`
    extra owner payload per rank before final target bytes are counted.
  - In practice this hits GPU OOM.
- Current `local_batched_disk_load` is a good stopgap and already much better
  than the old collective path, but it still leaves performance on the table on
  shared filesystems:
  - ranks still duplicate source reads,
  - `dim1` currently reads full row blocks and can amplify source bytes by `8x`
    for TP=8 column partitioning,
  - it does not perform cross-rank source dedup.

At the same time, experiments already show that TensorCast's core data plane is
not fundamentally slower than `fastsafetensors`:

- exact single-rank `879` source-tensor subset:
  - TensorCast common path: about `7.1s`
  - `fastsafetensors`: about `7.5s`
- exact large `dim0` subsets on host-local SSD:
  - TensorCast non-collective subset materialization already meets or exceeds
    the same `fastsafetensors` subset runs

These results imply:

- the main remaining gap is not single-rank H2D or basic tensor materialization,
- the main gap is cross-rank execution shape on shared source media,
- current eager collective and eager owner-file collective are both wrong
  execution models for this workload.

# Goals / Non-Goals

## Goals

- Match or beat `fastsafetensors` on shared-filesystem TP cold start for the
  same end-to-end workload.
- Become the default collective executor for shared-filesystem, same-host TP
  cold start once the prerequisite strategy-plane convergence work is landed.
- Preserve TensorCast's public selection-first model:
  - `ArtifactSelection`
  - `tensor_dict`
  - `materialize_view`
  - `bind_into`
  - `MaterializeIntoMappedTarget`
- Keep strategy selection inside the common runtime, not in vLLM.
- Replace eager whole-owner-payload VRAM preloading with bounded batching.
- Deduplicate cross-rank source reads when source media is shared.
- Preserve mixed execution:
  - tensor-aware direct reads,
  - staged 2D dim1 packing,
  - owner-file peer distribution,
  - residual generic fallback.
- Keep `ByteRangeMap` as the canonical fallback and explainability surface.
- Fit within typed daemon config under `engine.materialization_strategy`.

## Non-Goals

- Add model-name or tensor-name hardcoding.
- Add vLLM-only fast paths.
- Replace `ArtifactSelection` with a new public contract.
- Force owner-file collective on host-local SSD where local-batched may still
  be the right executor.
- Replace `TensorBatchedLocalExecutor` as the global default disk executor.
- Encode executor-private plans into public SDK or proto request fields.
- Depend on whole-owner-payload persistent VRAM residency.
- Skip the pre-109 convergence work and wire this executor directly into the
  current replica-layer prototype boundaries.

# Prior Constraints Reviewed

- `0108` correctly requires strategy selection to happen before generic lowering,
  but it did not yet make ordinary replica convergence, cost-model routing, and
  coordinator boundaries explicit enough for a default collective executor.
- `0107` correctly separates retrieval policy from rollout policy, but it needed
  a clearer execution-topology boundary so collective context would not be
  modeled as retrieval policy.
- The current repository prototype proved that file ownership and peer
  distribution are useful, but the eager `owned_payload` and optional root
  whole-source preload shapes are not acceptable long-term defaults.
- The host-local/local-batched bias remains correct and is kept. This design
  narrows its default claim to shared-source collective startup only.

# Current State

## 1. Current eager collective is too coarse

`collective_disk_load` today is transport-first.

It is good at:

- building same-host TP collectives,
- using GPU-side communication,
- reusing a shared root stage buffer.

It is bad at:

- preserving tensor-level locality all the way to execution,
- limiting source reads to only the necessary TP-local tensor bytes,
- handling shared-filesystem cold starts without large read amplification.

In particular, current collective execution still tends to collapse a
selection-aware tensor workload into:

- root-side reads,
- coarse owner payload preloads,
- or generic source windows and piece plans.

That loses too much information too early.

## 2. Current eager owner-file prototype is too memory-hungry

The current owner-file code path in `collective_disk_loader.cc` is close in
shape to the benchmark's eager Strategy A:

- assign files to owner ranks,
- preload all owner files into `owned_payload` on the owner's GPU,
- then run local pack plus NCCL `send/recv`.

This proves that file ownership is a useful idea, but it also proves that eager
whole-owner preload is not viable for large TP workloads. It creates a second
GPU-resident copy of a large fraction of the model.

This is the key reason the current owner-file path OOMs.

This is also the most important difference from `fastsafetensors`:

- current TensorCast eager owner-file path:
  - precomputes large owner payload totals,
  - allocates a long-lived `owned_payload`,
  - and fills it before distribution
- `fastsafetensors`:
  - batches files in `loader.py`,
  - materializes one batch with `copy_files_to_device(...)`,
  - distributes through `file_buffer.py` / `tensor_factory.py`,
  - then releases the batch via `fb.close()`

That difference in lifetime and peak temporary memory is the direct reason
`fastsafetensors` avoids the OOM currently seen in TensorCast's eager owner-file
prototype.

## 3. Current local-batched path is a good stopgap but not the end state

`local_batched_disk_load` is already selection-aware and is often the best
current ordinary disk executor.

It is the right default stopgap because it:

- avoids the worst collective read amplification,
- does not require whole-owner payloads,
- is already correct and stable enough for production startup.

But it remains single-rank and does not solve the cross-rank shared source
duplication problem. On shared filesystems that still leaves real performance on
the table.

Current runtime starting point after the converged `0108` strategy seam:

- ordinary disk startup already builds a common-runtime `ExecutionStrategyPlan`,
- `AUTO` currently prefers `TensorBatchedLocalExecutor` before the collective
  prototype,
- the owner-file collective candidate is non-default and still dispatches into
  the current `collective_disk_load` prototype runtime when explicitly selected.

## 4. `fastsafetensors` provides the right memory lesson, not the final shape

`fastsafetensors` currently wins because its runtime behavior is bounded:

- files are assigned to owner ranks,
- files are processed in batches,
- current batch buffers are closed and freed after use,
- cross-rank distribution is done on GPU with framework collectives.

This gives it two practical advantages over current TensorCast eager owner-file:

- it avoids whole-model extra VRAM preload,
- it keeps peak temporary owner memory bounded by a small batch.

TensorCast should adopt that memory discipline.

But TensorCast should not stop there.

Unlike `fastsafetensors`, TensorCast also knows:

- resolved selection identity,
- resolved view identity,
- exact TP-local slice hull,
- target layout,
- copy-contract structure,
- residual coverage and fallback accounting.

So the long-term TensorCast executor should be:

- at least as memory-safe as `fastsafetensors`,
- more target-layout-aware than `fastsafetensors`.

## 5. Concept mapping to `fastsafetensors`

The implementation relationship should be explicit.

### 5.1 What TensorCast should preserve that `fastsafetensors` does not model

TensorCast must preserve:

- `ArtifactSelection`
- `view_id`
- `ResolvedMaterializationPlan`
- `RepresentationTransformContract`
- residual coverage and `ExecutionCommitReport`

`fastsafetensors` does not expose these abstractions. Its effective planning
state is the combination of:

- file metadata and tensor frames in `loader.py`
- owner-file routing in `loader.py`
- per-tensor access order in `file_buffer.py`
- collective semantics in `tensor_factory.py`

This is why TensorCast should not directly embed `fastsafetensors` as a second
loading stack. TensorCast has richer semantic truth and should keep it.

### 5.2 What TensorCast should borrow directly

TensorCast should borrow the following runtime behaviors:

- owner-file assignment as an execution strategy
- batch-bounded temporary owner memory
- release-after-use batch lifetime
- owner-mediated GPU-side distribution

### 5.3 What TensorCast should improve beyond `fastsafetensors`

TensorCast should go further by using:

- direct writes into final target layout
- staged dim1 packing only when required
- mixed execution with residual generic fallback
- selection-aware and contract-aware batching

That is the point where TensorCast should stop “copying a good idea” and start
using the additional semantics that only TensorCast already has.

# Design Principles

## 1. Stay inside the 0108 strategy plane

This design is a follow-on to `0108`, not a competing architecture.

It keeps the same layering:

- controller safety boundary stays in daemon controllers,
- semantic resolution stays in `ResolvedMaterializationPlan`,
- source acquisition stays separate,
- executor lowering stays in `MaterializationFacade`,
- residual fallback stays in generic byte-range execution.

The new executor is therefore not:

- a vLLM-specific loader trick,
- a benchmark-only path,
- or a new public contract.

It is one additional executor choice inside the existing strategy plane.

It is explicitly **not** the future topology-scoped reshard executor.

- This design optimizes disk-backed, shared-source, tensor-aware materialization.
- Future communicator-aware group reshard execution may later reuse the same
  semantic core, but it should be specified as a separate executor design rather
  than folded into owner-file collective behavior.

## 2. File ownership is an execution strategy, not semantic truth

Owner assignment must not leak into:

- `ArtifactSelection`,
- `view_id`,
- `selection_hash`,
- or copy-contract truth.

File ownership is strictly an executor-private planning artifact derived after:

- semantic truth is resolved,
- `RepresentationWorkPlan` is derived,
- source capabilities are known,
- target placement is known.

## 3. TensorCast should adopt bounded owner batching, not eager owner preload

The main lesson from `fastsafetensors` is not “use owner rank”.
The main lesson is:

- bound peak temporary memory,
- process source in batches,
- free temporary owner state immediately after commit.

That is the minimum requirement for a usable owner-file collective path.

## 4. TensorCast should use its extra semantic knowledge to go beyond

TensorCast should outperform `fastsafetensors` by exploiting:

- direct writes into final target layout,
- staged 2D pack only when necessary,
- source-slice dedup across ranks,
- residual fallback only for the small irregular tail.

This is exactly the kind of shape described by the benchmark's Strategy C.

## 5. Default routing must be cost-model driven

This executor must not become default merely because collective context is
present.

`AUTO` policy should choose it only when the planner can show that:

- shared-source dedup is real for the current request,
- the estimated owner skew stays within policy,
- peak temporary bytes stay within typed budgets,
- the expected collective and planner overhead is repaid by lower source reads,
- and the residual tail is explicit and acceptable for the current rollout
  stage.

# Proposed Architecture

## 1. Introduce `OwnerFileBatchedCollectiveExecutor`

Add a new common-runtime executor:

- `OwnerFileBatchedCollectiveExecutor`

This executor replaces the current eager owner-file prototype for shared-FS
collective startup.

It is intended for requests where:

- the source is disk-backed and shared across ranks,
- the request is TP collective eligible,
- cross-rank source dedup is expected to beat fully independent local reads,
- and the request has enough tensor structure to avoid generic fallback for the
  bulk of bytes.

### 1.1 Phase-1 scope

The first production wave is intentionally narrower than the full long-term
executor model.

Phase-1 scope:

- ordinary disk-backed `tensor_dict(...)` startup,
- same-host TP collective groups,
- safetensors sources with complete metadata,
- replicated, dim0, and 2D dim1 tensor work as the dominant bytes.

Deferred follow-on scope:

- cross-topology reshard,
- executor-owned concat irregular tails,
- arbitrary mapped-target execution as part of the default rollout claim.

## 2. Execution model

The executor runs in bounded batches, not one eager preload pass.

### 2.1 Strategy inputs

The planner should consume:

- `ResolvedMaterializationPlan`
- `RepresentationWorkPlan`
- collective-lane-local lowering derived by the `0108` strategy plane
- `ResolvedSourceBinding`
- `ExecutionEnvironmentFacts`
- typed materialization-strategy policy

This keeps:

- semantic truth in `0108`,
- topology/locality context in the `0107` execution-topology plane,
- and executor-private batching inside this executor only.

Normative rule:

- this executor must not recover its primary request graph from
  `residual_fallback_map`;
- it consumes collective-admitted work only, while residual fallback remains
  outside the executor as a separate lane.

### 2.2 Default routing and eligibility

This executor is eligible only when all of the following hold:

- collective group context is present and same-host,
- the source is disk-backed and shared across ranks,
- shared-source locality is explicit or otherwise proven by runtime-owned
  evidence; `source_locality=AUTO` is not sufficient proof for default routing,
- metadata is complete enough to build tensor-aware work,
- estimated owner peak temporary bytes fit configured budgets,
- estimated owner skew stays below configured thresholds,
- estimated dedup savings exceed planner and collective overhead,
- and residual handling satisfies the current rollout stage.

Normative rule:

- when these conditions do not hold, `AUTO` must prefer
  `TensorBatchedLocalExecutor` or generic fallback rather than forcing
  owner-file collective.
- the initial `0109` rollout must require either:
  - explicit `ExecutionTopologyContext.source_locality = SHARED_SOURCE`, or
  - daemon-managed shared-disk provenance that proves ranks share the same
    backing source domain.

Each batch contains a subset of source work such that:

- owner-side temporary GPU bytes stay below a configured peak budget,
- staging bytes stay below a configured staging budget,
- NCCL or peer-transfer group size stays below a configured launch budget,
- batch lifetime ends as soon as all destinations for that batch are committed.

At a high level:

1. lower the request into tensor-aware work items,
2. group those items by owner file and source locality,
3. split them into batches under memory budgets,
4. for each batch:
   - owner rank reads only necessary source spans,
   - owner rank writes local destination bytes directly into final layout,
   - owner rank distributes peer bytes via collective send or scatter,
   - dim1 and other strided cases use bounded staging and pack,
   - batch staging is released immediately after commit,
5. residual irregular work falls back to generic byte-range execution.

This means no persistent `owned_payload` equal to “all owner files for this
rank”.

This is the main point where the design intentionally follows the
`fastsafetensors` memory model while still keeping TensorCast's richer planner
shape.

## 3. Planner output

The planner stays executor-private and is derived inside the strategy plane.

It should not become SDK or proto surface.

It is also intentionally downstream of shared typed work and shared lane
allocation:

- `0110` owns the executor-neutral typed work inventory,
- `0108` owns lane allocation and residual accounting,
- `0109` owns only the executor-private plan for the collective-admitted lane.

The planner should construct an internal batch plan family, for example:

- `OwnerBatchPlan`
  - owner rank
  - source file
  - source span set
  - batch bytes
  - committed destination ranges
  - estimated temporary bytes
  - estimated peer bytes
  - op list
- `OwnerBatchOp`
  - `DirectReadToFinalOp`
  - `StagedDim1PackOp`
  - `ReplicatedSendOp`
  - `Dim0ScatterOp`
  - `DedupPeerCopyOp`
  - `ResidualFallbackOp`

This is aligned with `0108`:

- executor-private ops stay executor-private,
- semantic truth remains in `ResolvedMaterializationPlan` and
  `RepresentationTransformContract`,
- residual byte accounting is still reported through `ExecutionCommitReport`.

## 4. Direct, staged, and residual paths

### 4.1 Direct contiguous path

For replicated and dim0-contiguous slices:

- read only the required contiguous source span,
- write owner-local bytes directly into final owner output,
- distribute peer-local bytes directly from the read span,
- do not materialize a whole tensor payload if the request only needs a slice.

### 4.2 Staged dim1 path

For dim1 or other non-contiguous tensor layouts:

- compute the minimum row-range staging windows that cover the selected slices,
- read bounded row-blocks into a staging slab,
- use 2D or structured GPU pack into final layout,
- distribute peer slices from staged or packed results as appropriate.

This is specifically meant to replace the current “read full rows for dim1”
behavior when collective owner execution is chosen.

### 4.3 Residual fallback

Residual ranges that cannot be safely handled by the batched owner-file plan
continue to use the generic byte-range path.

This keeps:

- correctness,
- explainability,
- and rollout safety.

Current phase rule:

- owner-file collective must not partially execute a request and then implicitly
  derive residual generic fallback at runtime,
- residual generic work must already be explicit in the emitted work plan before
  execution begins,
- typed work that is not admitted to this executor must remain typed in the
  shared lane plan; it must not be relabeled as true residual by collective
  code,
- Phase 1 implementation may start with zero-residual-only eligibility,
- but default graduation for shared-FS `AUTO` requires explicit mixed
  collective-plus-generic execution with preplanned residual coverage.

## 5. Source ownership model

The ownership concept should remain file-oriented by default because that is the
best match for:

- current safetensors shard structure,
- shared-filesystem cold start locality,
- and the current benchmark results.

But ownership must be used as a scheduling hint, not as a requirement to preload
entire owner files.

Normative rule:

- owner-file assignment determines which rank is responsible for first reading a
  source span,
- it does not imply that all bytes from that file must become resident in GPU
  memory at once.

Default ownership should be file-oriented, but it must be weighted by selected
bytes, not assigned by raw file index round-robin alone.

Normative implementation rules:

- ownership should be computed from the current request's selected source spans,
- planner policy should cap owner skew with an explicit threshold,
- hot files may be split into source segments when file-level ownership would
  violate that threshold,
- the planner must be able to reject owner-file collective if the resulting skew
  is still too high.

Future extensions may refine ownership from file-level to source-segment-level,
but this design does not require that on day one.

Normative implementation note:

- owner-file state should be batch-local and short-lived,
- it should not become long-lived GPU residency comparable to the current
  `owned_payload` field in `collective_disk_loader.cc`,
- its lifetime should be closer to the current batch/file-buffer lifetime in:
  - `/data/workspace/fastsafetensors/fastsafetensors/loader.py`
  - `/data/workspace/fastsafetensors/fastsafetensors/file_buffer.py`

# Relation To Existing Executors

## 1. Relation to `TensorBatchedLocalExecutor`

`TensorBatchedLocalExecutor` remains useful and should stay.

It is still the best common path for:

- host-local SSD,
- single-rank loads,
- and workloads where cross-rank dedup is not worth the coordination cost.

The new owner-file batched collective executor is not meant to replace local
batched execution globally.

Instead:

- host-local ordinary disk startup should continue to bias toward local batched,
- shared-FS TP cold start should gain a better collective executor than the
  current eager collective path.

## 2. Relation to current `collective_disk_load`

The current eager collective path should become legacy or internal fallback for
the subset of cases where:

- metadata is incomplete,
- batched owner-file planning is not possible,
- or rollout explicitly prefers the old behavior during migration.

Long-term, the current coarse root-first collective path should no longer be the
default strategy for shared-FS TP startup.

Hard-cut rule:

- once the new executor is proven for its intended workload family, temporary
  migration fallbacks and prototype-only preload shapes should be deleted or
  isolated away from the steady-state hot path rather than preserved as equal
  citizens.

This relation is still limited to materialization strategy.

- It does not imply that owner-file collective owns future cross-topology
  representation transforms.
- It only says owner-file collective is the preferred shared-FS execution
  strategy when the semantic core has already been resolved.

## 3. Relation to current eager owner-file path

The current eager owner-file collective path is a useful prototype but should be
retired after migration.

Its two key ideas survive:

- owner-file source ownership,
- peer distribution from owner rank.

Its eager memory behavior does not survive:

- no whole-owner-payload VRAM preload,
- no second full copy of large file ownership on GPU.

## 4. Closure status (2026-04-27)

The current mounted TP8 closure packet is
`docs/benchmarks/20260427-step3p5-fp8-mounted-tp8-cold-start-evidence.md`.

That packet freezes the `0109` policy conclusion:

- `owner_file_collective_shared_fs_only=true` is the intended steady-state
  policy, not temporary rollout debt;
- `owner_file_collective_allow_mixed_residual=false` is the intended
  fail-closed steady-state policy, not a provisional blocker waiting to be
  relaxed;
- current TP8 startup still reaches
  `dominant_executor=OwnerFileCollectiveExecutor` with
  `actual_generic_backend_bytes=0`, so the mounted case does not justify
  widening mixed residual merely to keep startup alive; and
- the remaining open item is performance quality in the TensorCast bootstrap
  runtime, not policy ambiguity in `0109`.

# Configuration

This design should not add new environment-only toggles.

Typed daemon rollout should remain under:

- `engine.materialization_strategy`

The converged repository already has the relevant typed fields:

- `enable_owner_file_collective`
- `owner_file_collective_peak_bytes_budget`
- `owner_file_collective_batch_bytes`
- `owner_file_collective_dim1_staging_bytes`
- `owner_file_collective_max_inflight_batches`
- `owner_file_collective_shared_fs_only`
- `owner_file_collective_max_owner_skew_ratio`
- `owner_file_collective_min_dedup_saving_bytes`
- `owner_file_collective_group_assemble_timeout`
- `owner_file_collective_allow_mixed_residual`
- `owner_file_collective_planner_cache_entries`

These are the actual field names under the current
`proto/tensorcast/config/v1/daemon_config.proto` and
`StoreEngineOptions::MaterializationStrategyConfig`. `0109` should consume
these landed controls rather than introducing a second naming family.

Normative rules:

- rollout must be typed and config-backed,
- env variables may still exist for diagnosis but not as the primary
  configuration mechanism,
- executor preference remains a policy choice and must not alter semantic truth,
- budget and threshold fields are required for default routing; boolean enable
  flags alone are not sufficient.

# Correctness Model

The executor must preserve the same correctness guarantees as current common
materialization.

## 1. No semantic widening

The executor may batch and reorder for source locality, but it may not widen
the requested semantic result.

In particular:

- owner batches must read only spans required by the resolved selection, except
  for explicit staging windows whose widened source bytes are accounted for as
  executor-private amplification,
- output bytes must still match the resolved target layout exactly,
- residual ranges must remain visible and correctly accounted for.

## 2. Ownership is not permission to alias semantics

Owner-file execution must not change:

- `artifact_id`
- `view_id`
- `selection_hash`
- `logical_layout_hash`
- target coverage

It is only a transport and execution optimization.

## 3. Release-after-commit

Temporary owner staging must be freed after the batch is committed and before
the next batch requires that memory.

This is both:

- a memory-safety rule,
- and a correctness rule against accidental stale aliasing.

## 4. Atomic batch commit and failure handling

Each batch is a commit unit for this executor.

Normative rules:

- a batch is committed only after all destination bytes covered by that batch
  are visible and semantically complete,
- owner-local direct writes, peer sends, and staged pack results belong to the
  same commit unit,
- if any part of the batch fails, the batch is reported as uncommitted and must
  follow explicit retry, poison, or fallback policy from the strategy plane,
- `ExecutionCommitReport` must be able to account for committed and uncommitted
  batch coverage without inventing residual work after the fact.

# Naming Compliance

| Proposed symbol | Kind | Required style | Result |
| --- | --- | --- | --- |
| `OwnerFileBatchedCollectiveExecutor` | C++ class | `PascalCase` | pass |
| `OwnerBatchPlan` | C++ struct | `PascalCase` | pass |
| `OwnerBatchOp` | C++ variant/struct family | `PascalCase` | pass |
| `build_owner_batch_plan` | C++ function | `snake_case` | pass |
| `estimate_owner_collective_cost` | C++ function | `snake_case` | pass |
| `owner_file_collective_peak_bytes_budget` | config field | `snake_case` | pass |

# Why This Is More Consistent Than Reusing `fastsafetensors` Directly

Directly embedding a `fastsafetensors`-style loader into vLLM or into the SDK
surface would be inconsistent with TensorCast because it would bypass:

- `ArtifactSelection`
- `view_id`
- strategy-plane lowering
- residual byte accounting
- target-layout-aware mixed execution

By contrast, this design uses `fastsafetensors` as an implementation reference
for:

- bounded batching,
- owner-source dedup,
- release-after-use memory behavior,
- GPU-side peer distribution.

Those ideas are then re-expressed as a TensorCast executor under the common
runtime.

That is the consistent form for this project.

Practical reference points:

- batch orchestration:
  - `/data/workspace/fastsafetensors/fastsafetensors/loader.py`
- current batch lifetime and release behavior:
  - `/data/workspace/fastsafetensors/fastsafetensors/file_buffer.py`
- owner-rank collective distribution behavior:
  - `/data/workspace/fastsafetensors/fastsafetensors/tensor_factory.py`
- higher-level architecture rationale:
  - `/data/workspace/fastsafetensors/docs/architecture.md`
  - `/data/workspace/fastsafetensors/docs/safetensors-load-optimization.md`

# Rollout Plan

## Phase 0: Pre-109 convergence dependency

- completed:
  - the runtime-critical request-normalization work from
    `docs/designs/0107-retrieval-policy-plane-cleanup.md`,
  - the strategy-plane convergence work folded back into
    `docs/designs/0108-tensor-aware-materialization-strategy-plane.md`,
  - especially ordinary replica convergence, execution-environment facts, typed
    strategy budgets, and prototype cleanup seams.

## Phase 1: Add the executor beside existing paths

- implement `OwnerFileBatchedCollectiveExecutor` behind typed config,
- keep current local-batched and generic fallback unchanged,
- keep current eager collective available only as a lower-priority fallback.
- require explicit shared-source proof for initial routing; do not let
  `source_locality=AUTO` silently behave as “shared filesystem”.

## Phase 2: Route explicit shared-FS TP experiments into it

- ordinary disk startup on shared filesystems may choose the new executor when:
  - source metadata is complete,
  - tensor-aware planning covers the bulk of bytes,
  - estimated owner peak bytes fit the configured budget,
  - owner skew and dedup thresholds pass policy.

## Phase 3: Make it the default collective executor for shared-FS TP startup

- once explicit mixed residual handling, cost-model routing, and observability
  are proven,
- make this executor the default collective choice for eligible shared-FS,
  same-host TP cold starts.

## Phase 4: Retire eager owner-file preload

- once correctness and performance are proven,
- delete or demote the current eager `owned_payload` collective path.

# Validation Plan

This design must be validated with both correctness and performance evidence.

## 1. Correctness

- exact subset tensor digest compare against:
  - TensorCast local-batched,
  - `fastsafetensors`,
  - and reference `auto` startup where applicable
- TP=8 `vllm serve` startup and `/v1/completions` correctness
- mixed workloads including:
  - replicated
  - dim0
  - dim1
  - concat / residual fallback

## 2. Peak memory

- record per-rank temporary owner bytes
- prove that peak extra owner staging stays within configured budget
- show that current owner-file OOM cases no longer OOM

## 3. Planner and routing evidence

- record planner overhead and cache-hit behavior
- record estimated versus actual owner skew
- record executor selection reasons for chosen and rejected candidates
- validate group-assemble timeout and fail-open behavior under partial rank join

## 4. Performance

Required comparisons:

- shared filesystem cold start:
  - current collective
  - current local-batched
  - new owner-file batched collective
  - `fastsafetensors`
- host-local SSD:
  - ensure no regression relative to local-batched default
- exact benchmark subsets:
  - exact `879` source-tensor subset
  - exact `614` sliced subset
  - large dim0 and dim1 focused subsets

Primary success criteria:

- shared-FS TP cold-start wall time matches or beats `fastsafetensors`
- no OOM on the current Step3p5 TP=8 workload
- shared-FS eligible requests choose this executor by default once rollout is
  enabled
- host-local default path remains no worse than current best local-batched path

# Expected Outcome

If implemented correctly, this design should deliver:

- the memory safety lessons of `fastsafetensors`,
- the semantic rigor of TensorCast's selection-first architecture,
- a practical replacement for the current eager owner-file collective path,
- and a shared-filesystem TP startup path that is at least competitive with
  `fastsafetensors`, with room to surpass it by using direct final-layout writes
  and mixed execution.

In short:

- `fastsafetensors` shows the right memory behavior,
- TensorCast already has the richer semantic model,
- this design combines the two in a form that matches TensorCast's long-term
  architecture.

# References

- [`docs/designs/0107-retrieval-policy-plane-cleanup.md`](/data/workspace/tensorcast-280/docs/designs/0107-retrieval-policy-plane-cleanup.md)
- [`docs/designs/0108-tensor-aware-materialization-strategy-plane.md`](/data/workspace/tensorcast-280/docs/designs/0108-tensor-aware-materialization-strategy-plane.md)
- [`docs/designs/0112-binding-native-serving-realization-and-publication.md`](/data/workspace/tensorcast-280/docs/designs/0112-binding-native-serving-realization-and-publication.md)
- [`docs/benchmarks/20260415-qwen2.5-32b-mounted-collective-first-v4-serving-evidence.md`](/data/workspace/tensorcast-280/docs/benchmarks/20260415-qwen2.5-32b-mounted-collective-first-v4-serving-evidence.md)
- [`docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`](/data/workspace/tensorcast-280/docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md)
- [`docs/internals/disk-load-strategy.md`](/data/workspace/tensorcast-280/docs/internals/disk-load-strategy.md)
- [`core/store/replica/collective_disk_loader.cc`](/data/workspace/tensorcast-280/core/store/replica/collective_disk_loader.cc)
