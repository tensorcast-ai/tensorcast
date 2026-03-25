---
title: Disk Load Strategy
description: Decision tree and executor selection for TensorCast disk-backed model loading
areas: ["sdk", "daemon", "core", "serving"]
---

# Disk Load Strategy

This document describes the current `load from disk` strategy for TensorCast-backed
model loading, with emphasis on:

- TP-aware rank-local slicing in the current internal-vLLM TensorCast loader,
- local SSD versus shared filesystem behavior,
- daemon-side executor selection under the `0108` strategy plane,
- the difference between ordinary `tensor_dict` startup and mapped `into-target` paths.

Related docs:
- `docs/internals/model-loading.md`
- `docs/architecture/api/materialization-flow.md`
- `docs/internals/byte-range-mapping-and-execution.md`
- `docs/designs/0108-tensor-aware-materialization-strategy-plane.md`

## Scope

In scope:

- Ordinary disk-backed startup through `Artifact.tensor_dict(...)`.
- TP-aware request shaping before materialization.
- Automatic collective versus non-collective selection.
- Daemon-side choice between collective, local-batched, and generic fallback executors.
- Disk-backed `MaterializeIntoTarget` / mapped binding strategy.

Out of scope:

- P2P-first loads from remote replicas.
- Serving-artifact startup and reload-by-swap beyond their interaction with disk fallback.
- Canonical index and byte-range compiler details beyond what is needed to explain strategy choice.

## Layered View

The current strategy is split into two layers:

1. The integration loader decides what the current rank actually needs and,
   if desired, whether to synthesize an explicit per-call collective hint.
2. The daemon chooses the actual disk executor based on metadata completeness,
   target/source type, and typed `engine.materialization_strategy` config.

This separation is intentional after `0108`:

- integration side handles TP-local request shaping and any integration-owned
  collective hint policy,
- daemon side owns executor rollout and data-plane choice.

Important boundary:

- the generic TensorCast SDK does not auto-infer collective mode from ambient
  environment or filesystem state,
- integrations such as internal-vLLM may choose to synthesize an explicit
  `CallContext.collective`,
- the daemon and common runtime still treat collective as an explicit request
  hint plus typed strategy config.

Executor note:

- local-batched disk load and owner-file collective now consume prebuilt
  `RepresentationWorkPlan` inputs,
- executor-local code no longer reconstructs semantic tensor-job truth from
  source/view index JSON at execution time.

## Typed Daemon Defaults

The default `0108` strategy entry point is daemon config, not ad-hoc environment
variables:

```yaml
engine:
  materialization_strategy:
    enable_local_batched_disk_load: true
    enable_tensor_aware_mapped_executor: true
    allow_mixed_execution: true
    enable_owner_file_collective: false
    executor_preference: MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_AUTO
```

See `examples/config/store_daemon_config.yaml`.

Operationally, this means:

- local-batched disk load is enabled by default,
- tensor-aware mapped execution remains available for eligible mapped paths,
- owner-file collective exists but is not the default rollout path.

## End-To-End Decision Tree For Ordinary Disk Startup

The main startup path is ordinary disk-backed `tensor_dict(...)`, not mapped
runtime binding.

```mermaid
flowchart TD
    A["TensorCast loader startup"] --> B{"explicit source artifact key?"}
    B -- "yes" --> B1["not ordinary disk path\nuse artifact-key source"]
    B -- "no" --> C["ordinary disk path\nartifact_ref = disk:<hf_folder>"]

    C --> D["build TP-local trace plan\nmaterialize_names + tensorcast_slices"]
    D --> E{"TP world size > 1?"}
    E -- "no" --> F["no collective hint"]
    E -- "yes" --> G{"integration collective policy\nenabled?"}
    G -- "no" --> F
    G -- "yes" --> H["attach explicit CollectiveLoadGroup hint"]

    F --> I["daemon disk materialization"]
    H --> I

    I --> J{"collective eligible?"}
    J -- "yes" --> K["collective_disk_load"]
    J -- "no" --> L{"local_batched eligible?"}
    L -- "yes" --> M["local_batched_disk_load"]
    L -- "no" --> N["generic byte-range / mapped fallback"]
```

## Step 1: TP-Aware Request Shaping In The internal-vLLM Loader

The current internal-vLLM TensorCast loader does not send a full-model load
request per rank. Instead, each rank first traces its own checkpoint access
pattern and builds a rank-local plan:

- `materialize_names`: the checkpoint tensors this rank needs,
- `tensorcast_slices`: the rank-local slice hull used to build the view.

The disk request is then:

1. `artifact.subset(materialize_names)`
2. `.view(slices=tensorcast_slices)`
3. `.tensor_dict(...)`

This is the first TP-specific optimization. It reduces disk work before the
daemon chooses any executor.

This step is integration-owned, not a generic TensorCast SDK behavior. The
current implementation lives in the internal-vLLM loader, which traces
`model.load_weights(...)`, builds `TracePlan`, derives `tensorcast_slices`, and
caches the result per TP rank and world size.

Important consequences:

- TP handling starts in the loader, not only in the daemon.
- The trace cache is per `tp_rank` and `tp_world_size`, so each rank carries its
  own stable trace-plan cache entry.
- `TP=1` naturally degenerates to a non-collective single-rank request.

## Step 2: Integration-Level Collective Hinting

For ordinary `disk:<model_path>` startup, collective remains an explicit
TensorCast API contract. The generic SDK does not attach a collective hint on
its own.

The current internal-vLLM loader may synthesize that explicit
`CollectiveLoadGroup` on behalf of the caller by constructing
`CallContext.collective`, but this is an integration policy above the API
boundary rather than a TensorCast-core SDK default.

Current rule:

- local filesystems:
  - `xfs`
  - `ext2`
  - `ext3`
  - `ext4`
  - `btrfs`
  - `zfs`
  - `tmpfs`
  - default to `collective = false`
- anything else:
  - currently also defaults to `collective = false`
  - reason: current shared-fs ordinary disk startup still regresses badly when
    the loader eagerly supplies a collective hint and the request lands on the
    existing `collective_disk_load` path
  - explicit overrides remain available for collective experiments while the
    owner-file collective strategy is not yet production-ready

This is not a hard-coded `JuiceFS` or `JFS` special case by name. The current
internal-vLLM loader treats them through filesystem type detection:

- host-local SSD paths usually resolve to one of the local filesystem types and
  therefore default to non-collective startup,
- JuiceFS/JFS/shared filesystems usually do not fall into that local whitelist,
  but ordinary disk-backed startup is currently still kept on the
  non-collective path by default as a temporary regression mitigation,
- if mount type detection fails, the code currently also biases toward
  non-collective.

Current internal-vLLM behavior therefore is:

- local filesystems default to no collective hint,
- shared/non-local filesystems also default to no collective hint today,
- explicit integration override remains available for collective experiments.

This choice only affects whether the request carries a collective hint. The
daemon still performs a second-stage eligibility check before it actually uses a
collective executor.

## Why Local SSD And Shared Filesystems Split Here

The split exists because the best-known disk strategy differs by storage class:

- on host-local SSD, direct per-rank reading plus local batching is usually
  better than coordinating a TP collective read,
- on shared filesystems, letting TP ranks cooperate often reduces redundant IO
  and improves effective source-side locality.

This is why the common TensorCast runtime no longer uses ambient environment as
the control surface for collective selection. The current intended behavior is:

- keep the generic SDK contract explicit,
- allow integrations to synthesize explicit per-call hints when they have the
  right TP topology context,
- keep daemon-side executor policy centralized in typed strategy config,
- avoid forcing shared-fs collective by default until the owner-file path is
  production-ready.

## Step 3: Daemon Executor Selection For `tensor_dict`

Once the disk request reaches the daemon, executor selection happens in this
order:

1. try `collective_disk_load`,
2. otherwise try `local_batched_disk_load`,
3. otherwise fall back to generic byte-range/mapped execution.

### Collective Eligibility

`collective_disk_load` is only considered when all of the following hold:

- target is `GPU <- DISK`,
- the request carries a `collective_load_group`,
- `canonical_index_json` is available,
- `source_index_json` is available,
- `variant_identity` is available,
- the requested view does not require extra server-side materialization
  transform,
- the source is backed by a `DiskLoader` with usable shared context.

If any of these fail, the daemon logs why collective was skipped and continues
to the next strategy.

### Local-Batched Eligibility

If collective is not used, the daemon next tries `local_batched_disk_load`.

It requires:

- target is GPU,
- `canonical_index_json` is available,
- `source_index_json` is available,
- `variant_identity` is available,
- `DiskLoader` shared context is available,
- `enable_local_batched_disk_load = true`.

If the metadata is incomplete or the loader cannot expose the required disk
context, the request falls through again.

### Generic Fallback

If neither collective nor local-batched can run, the daemon builds a generic
`ByteRangeMap` path:

- canonical index and source index are composed when available,
- view selection is composed if needed,
- the request is lowered through `ByteRangeMappedSource`,
- transfer runs through the generic pump path.

This fallback remains the correctness backstop, but it is no longer the primary
intended hot path for disk-backed startup.

## Internal Shape Of `local_batched_disk_load`

`local_batched_disk_load` is still tensor-aware, but it is single-rank and
local. It splits disk jobs into:

- `replicated` jobs,
- `dim0 partitioned` jobs,
- `dim1 partitioned` jobs.

```mermaid
flowchart TD
    A["local_batched_disk_load"] --> B["build tensor jobs"]
    B --> C["replicated + dim0\n-> direct segments -> batched pump"]
    B --> D["dim1\n-> execute_local_dim1_jobs"]
```

Operationally:

- replicated and dim0-partitioned regions are lowered into direct source
  segments and pumped efficiently,
- dim1-partitioned tensors use a dedicated execution path,
- the daemon emits `local_batched_disk_load timings` with job counts and
  timing breakdowns for verification.

## Internal Shape Of `collective_disk_load`

`collective_disk_load` is designed for same-host TP groups. The ranks form a
collective clique and execute tensor jobs cooperatively.

```mermaid
flowchart TD
    A["collective_disk_load"] --> B{"owner-file collective enabled?"}
    B -- "yes" --> C["owner-file path\noptimized for replicated + dim0"]
    B -- "no" --> D["regular collective path"]

    C --> E["peer distribution via NCCL send/recv or broadcast"]
    D --> F{"whole-source preload enabled?"}
    F -- "yes" --> G["root rank preloads source\nthen distributes"]
    F -- "no" --> H["streaming pinned buffer\nexecute jobs directly"]
```

Current defaults matter here:

- `enable_owner_file_collective` exists,
- but the default daemon config keeps it `false`,
- so owner-file collective is not the default production path today.

As with local-batched, collective execution is still tensor-aware:

- replicated tensors,
- dim0-partitioned tensors,
- dim1-partitioned tensors

are handled separately with different execution routines.

## Filesystem-Specific Summary

### Host-Local SSD

Expected behavior:

- no collective hint from the loader,
- TP ranks still perform rank-local `subset + slice` shaping,
- daemon usually lands on `local_batched_disk_load`,
- generic fallback is only the residual safety path.

This is the current best-known default path for ordinary host-local startup.

### JuiceFS / JFS / Shared Filesystems

Expected behavior:

- current internal-vLLM loader still defaults to no collective hint,
- daemon therefore usually stays on the non-collective local-batched or generic
  path,
- explicit collective experiments can still attach a group and exercise
  `collective_disk_load` when eligibility is satisfied.

The important point is that these filesystems are currently handled by storage
class inference, not by a product-specific special case table, but current
default policy still biases to non-collective startup.

## Mapped `into-target` / `bind_into` Disk Path

Mapped disk execution follows a related but different strategy tree. This is
used by `MaterializeIntoTarget`, `bind(...)`, and `bind_into(...)` flows.

```mermaid
flowchart TD
    A["disk-backed mapped into-target request"] --> B{"disk + collective hint + representation work plan?"}
    B -- "yes" --> C["try collective mapped executor"]
    B -- "no" --> D{"source-ordered path available?"}
    D -- "yes" --> E["GenericByteRangeExecutor(source_ordered)"]
    D -- "no" --> F["ByteRangeMappedSource + GenericByteRangeExecutor"]
```

The key differences from ordinary `tensor_dict` startup are:

- the request already arrives as a mapped target layout,
- the runtime considers source-ordered execution using source layout metadata,
- collective mapped execution is only possible when a derived representation
  work plan and
  collective group are both present,
- owner-file or collective mapped execution is only eligible when the emitted
  work plan already has zero residual generic fallback,
- runtime execution must not partially execute mapped work and then implicitly
  derive new residual fallback at execution time,
- otherwise the residual executor is still the generic byte-range engine.

## Runtime Binding And Ordinary Disk Startup

Ordinary disk startup should not be confused with serving-artifact runtime
binding.

Current rule:

- `tensorcast_enable_runtime_binding=true` does not by itself turn ordinary
  `disk:<model_path>` startup into the preferred mapped startup path,
- startup source binding is only attempted when an explicit
  `tensorcast_source_artifact_key` is provided,
- otherwise ordinary disk startup stays on the `tensor_dict` strategy described
  above.

This guard exists because disk-backed mapped source binding is not currently the
best-known startup path for plain local checkpoint directories.

## Practical Verification Checklist

When validating which disk path actually executed, use the following signals.

For host-local SSD:

- loader should not attach a collective hint,
- daemon should emit `local_batched_disk_load timings`,
- materialization diagnostics should not show `source=local_replica` unless the
  request was intentionally satisfied from an already-local source.

For shared filesystem startup:

- do not expect a collective group by default today,
- if you explicitly enable collective experiments, then the loader should attach
  a collective group,
- in that explicit-collective case, the daemon should log collective
  eligibility and successful runs should emit `collective_disk_load timings`.

For mapped `into-target` requests:

- inspect `materialize_mapped_into_target execution_commit`,
- check:
  - `collective_handled`,
  - `direct_write_supported`,
  - `source_ordered`,
  - `dominant_executor`.

## Current Intended Defaults

The intended operator-facing behavior today is:

- ordinary host-local disk startup:
  - use `tensor_dict` path,
  - rely on rank-local trace slicing,
  - let the daemon select local-batched disk load,
- shared filesystem startup:
  - still default to non-collective startup today,
  - reserve collective for explicit experiments until owner-file collective is
    production-ready,
  - let the daemon decide whether collective eligibility is satisfied once a
    collective hint is actually present,
- mapped runtime binding:
  - reserve for explicit serving/source-artifact workflows, not plain disk
    startup.

This gives TensorCast a single consistent strategy stack:

1. TP-aware request shaping in the loader,
2. integration-owned explicit collective hint synthesis,
3. typed daemon strategy selection,
4. byte-range fallback only for residual coverage.
