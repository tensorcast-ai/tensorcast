---
title: API Design
description: Public SDK surface and contracts
---

# API Design

This document describes the public Python SDK surface for artifact registration
and materialization, along with the contracts and invariants callers can rely
on.

This page intentionally prioritizes **What/Why** (semantics and rationale) over
**How** (internal mechanics). For internal flows, see:

- [Registration Flow](./registration-flow.md)
- [Materialization Flow](./materialization-flow.md)
- [Policy & Persistence](./policy-persistence.md)
- [Region-Backed](./region-backed.md)
- [Error, Retry, Observability](./error-retry-observability.md)

## Navigation

- [Goals](#goals)
- [Core Concepts](#core-concepts)
- [Entry Points](#entry-points)
- [Registration APIs](#registration-apis)
- [Artifact Handles and Retrieval APIs](#artifact-handles-and-retrieval-apis)
- [Policy and Persistence Hooks](#policy-and-persistence-hooks)
- [Region APIs](#region-apis)
- [Contracts and Invariants](#contracts-and-invariants)
- [Code Map](#code-map)

## Goals

The SDK aims to provide a small, stable surface that can scale from “single
process cache” to “cluster-wide durable artifact distribution” without forcing
applications to rewrite I/O paths.

Why the API is shaped the way it is:

- **Artifact handles over ad-hoc getters**: An `Artifact` is a stable unit of
  identity + metadata + fallbacks; it allows the SDK to evolve internal
  materialization strategies without breaking callers.
- **Explicit policy**: `StorePolicy` is the single durability and placement
  declaration, so callers can reason about “where the bytes must end up” rather
  than “which RPC to call”.
- **Daemon-owned data plane**: materialization and disk reads remain daemon-owned
  so that transport locks, verification, and P2P orchestration stay consistent.

## Core Concepts

These terms show up across docs and APIs:

- **Artifact**: a logical collection of named tensors with a canonical index
  describing dtypes/shapes/strides and a canonical byte layout.
- **Artifact ID**: a content-addressed identifier (e.g. `mi2:...`) returned by
  registration, used for retrieval and persistence.
- **Key**: a human-friendly string name mapped to an artifact ID (key mapping
  lives outside the caller process).
- **Replica**: a concrete materialization of an artifact on a specific device
  (local VRAM/DRAM, remote VRAM/DRAM, disk-backed, etc.).
- **Lease / LIP (Lease-In-Place)**: a daemon-tracked lifetime for a replica that
  is backed by client-owned VRAM; the daemon exports it via CUDA IPC handles.
- **Fallback**: a caller-provided hint for preferred source selection (`local`,
  `p2p`, `disk`) and optional disk path hints.

## Entry Points

TensorCast exposes a process-wide store session and module-level helpers.

- `tensorcast.init(...)`: establishes a runtime (connect to an existing daemon
  or launch services). Implementation: [tensorcast/startup.py](../../../tensorcast/startup.py).
- `tensorcast.store(...)`: returns the process-wide `Store` (lazy initialization).
- Module-level helpers (`tensorcast.register`, `tensorcast.put`,
  `tensorcast.register_view`, `tensorcast.artifact`, `tensorcast.from_disk`) are
  thin wrappers around the process `Store`.

Advanced usage: directly construct a store instance if you want multiple stores
in one process (uncommon).

- `tensorcast.api.store.Store(daemon_endpoint, opts=StoreOptions)`

### StoreOptions (process/store configuration)

`StoreOptions` configures client-side behavior such as default fallbacks and
retry policy overrides.

- Type: [tensorcast/api/store/types.py](../../../tensorcast/api/store/types.py) (`StoreOptions`, `RetryPolicy`)
- Where to pass: `tensorcast.store(opts=...)` or `Store(..., opts=...)`

Supported fields:

| Field | Type | Default | What it does | Why it exists |
|---|---|---:|---|---|
| `fallback` | `FallbackOptions \| str \| None` | `None` | Default fallback hints applied by the store runtime when materializing artifacts. | Provide consistent source preference for a whole process (e.g. “local-only”). |
| `retry_overrides` | `Mapping[str, RetryPolicy] \| None` | `None` | Override the built-in retry policies per verb (keys: `register`, `put`, `get`, `get_into`). | Tune latency vs. resilience for different deployments. |

Example:

```python
import tensorcast
from tensorcast.api.store.types import RetryPolicy

tensorcast.init(mode="connect", address="127.0.0.1:50051")
store = tensorcast.store(
    opts=tensorcast.StoreOptions(
        fallback="local",
        retry_overrides={"get": RetryPolicy(20.0, 2, 0.1, 2.0, 0.5)},
    )
)
```

## Store And Entry Points

TensorCast exposes a process-wide Store session and module-level helpers that
bind to it.

This section is kept for compatibility with older links; see [Entry Points](#entry-points).

## Registration And Upload APIs

Primary registration verbs:

- `Store.register(...)` registers existing GPU memory using lease-in-place (LIP)
  when supported.
- `Store.put(...)` uploads tensors into daemon-owned stable DRAM.
- `Store.register_view(...)` registers a slice or transpose view and allows the
  daemon to rebuild the canonical artifact from a partial upload.

Why there are multiple verbs:

- `register` is the **fastest** path when tensors already live in GPU memory and
  you want to export them without a deep copy; it requires careful lifetime
  management (leases/regions).
- `put` trades some overhead for **stability** by uploading into daemon-owned
  stable memory (less coupling to client process lifetime).
- `register_view` is for **structural reuse** (views/slices/transposes) and can
  reduce redundant data movement when the canonical artifact can be derived from
  a view payload.

`register` and `put` accept an optional `policy` and an optional `options`:

- `policy: StorePolicy | str | None` is the preferred surface.
- `options.policy` is an advanced escape hatch.
- If both are provided they must normalize to the same policy.

`register_view` does not currently expose a first-class `policy=` parameter; use
`options=RegisterArtifactOptions(policy=...)` when you need durability/placement
control for view registrations.

### RegisterArtifactOptions (advanced registration tuning)

`RegisterArtifactOptions` controls plan selection and payload sizing. It is an
advanced surface: most applications should start with `policy=` only.

- Type: [tensorcast/api/_config.py](../../../tensorcast/api/_config.py) (`RegisterArtifactOptions`, `PlanType`)
- Where to pass: `Store.register(..., options=...)`, `Store.put(..., options=...)`, `Store.register_view(..., options=...)`

Key fields (selected; see source for full list):

| Field | Default | What it does | Typical use |
|---|---:|---|---|
| `plan` | `DRAM_STABLE` | Selects daemon registration plan (`coalesced`, `lease`, `stable_dram`, etc.). | Force a plan when debugging performance or compatibility. |
| `max_inflight_bytes` | `512 MiB` | Upper-bounds coalesced upload inflight bytes. | Prevent large registrations from monopolizing pinned buffers. |
| `lease_in_place` | `False` | Opt-in to LIP flows (client-owned VRAM exported via lease). | Register already-loaded model weights without copying. |
| `min_tensor_bytes` | `64 KiB` | LIP segmentation threshold. | Reduce per-tensor overhead for tiny tensors. |
| `max_tensor_count` | `8192` | Safety cap for pathological tensor dicts. | Guard against high fan-out metadata. |
| `stage_on_gpu` | `True` | For stable DRAM plans, stage uploads via GPU buffers. | Improve throughput when GPU→DRAM path is faster. |
| `disk_path` | `None` | Optional disk hint to attach to the artifact. | Enable disk-first fallback for later reads. |

Example (LIP opt-in):

```python
import tensorcast

artifact = tensorcast.register(
    {"w": some_cuda_tensor},
    options=tensorcast.RegisterArtifactOptions(lease_in_place=True),
    policy="cache",
)
```

### Store.register (register tensors already in VRAM)

Signature (module-level is identical): `tensorcast.register(tensors, *, artifact_id=None, key=None, policy=None, options=None, ttl_ms=None)`

| Parameter | What it means | Why/when to use it |
|---|---|---|
| `tensors` | `Mapping[str, torch.Tensor]` of CUDA tensors to register. | The logical tensor dict that forms the artifact. |
| `artifact_id` | Optional client-provided identifier (not the content-addressed id). | Useful for diagnostics/idempotency tagging; the daemon still returns the canonical `mi2:...` id. |
| `key` | Optional human-friendly name to publish for later lookup. | Use when you want consumers to fetch by name rather than by `artifact_id`. |
| `policy` | `StorePolicy \| str \| None`. | Declare durability/placement; see [Policy & Persistence](./policy-persistence.md). |
| `options` | `RegisterArtifactOptions \| None`. | Advanced tuning (plan selection, inflight sizing). |
| `ttl_ms` | Optional lease TTL override (ms). | Mainly relevant for lease/LIP flows. |

Typical scenarios:

- “I already have model weights on GPU, export them without copying”: use `register` (often with region-backed LIP).
- “I want fast local caching but don’t need durability”: `policy="cache"`.
- “I need durability after registration”: `policy="durable"` or `policy="ha"` (starts persistence).

### Key mapping (why `key` exists and how to use it)

`key` is a human-friendly name that resolves to an `artifact_id`. Use it when:

- producers and consumers are decoupled (different processes/nodes)
- you want a stable name (“model:v7”) but don’t want to pass around content ids

Key mapping is resolved by the daemon so SDK clients do not need direct Global
Store knowledge:

- Resolve: `ResolveKeyMapping` RPC (daemon → mapping store)
- Publish: `PublishReplicaKey` RPC during/after registration

Contracts:

- A key maps to at most one active artifact id at a time.
- Publishing a key that already maps to a different artifact id fails with
  `FAILED_PRECONDITION`.
- Prefer `artifact_id` for fully deterministic reads; use `key` when you want
  indirection/versioning.

See:

- Key materialization: [Materialization Flow](./materialization-flow.md#materialize-by-key-and-by-replica)
- v1 proto: [proto/tensorcast/daemon/v1/store_daemon.proto](../../../proto/tensorcast/daemon/v1/store_daemon.proto)

### Store.put (upload into daemon-owned stable DRAM)

Signature: `tensorcast.put(tensors, *, artifact_id=None, key=None, policy=None, options=None, device=None)`

`put` uploads tensors and commits a stable DRAM-backed replica. This reduces
coupling to client process lifetime at the cost of an upload.

Key parameter:

- `device`: optional target device selection for upload planning (e.g. pin to `cuda:1`).

### Store.register_view (register a view-derived artifact)

Signature: `tensorcast.register_view(tensors, *, artifact_id=None, key=None, slices=None, transpose=None, view_id=None, placement=None, ttl_ms=None, allow_partial=False, options=None)`

`register_view` is for cases where the canonical artifact can be derived from a
view/slice/transpose of an existing tensor dict.

View inputs:

- `slices`: mapping of tensor name → a single narrow slice spec.
  - Supported forms: a `slice` (defaults to `dim=0`), or `(dim, slice)`.
  - Only one narrow op per tensor; `slice.step` must be `1`.
- `transpose`: mapping of tensor name → non-empty sequence of `(dim0, dim1)` swaps.
- `view_id`: alternate identity; when provided, do not provide `slices`/`transpose`.
- `placement`: `"SERVER"` or `"CLIENT"`.
  - Defaults: registration chooses `"CLIENT"` when transpose is present; otherwise `"SERVER"`.
  - If the daemon rejects `"SERVER"` placement for a view, the SDK surfaces a
    `FAILED_PRECONDITION` with guidance to retry `"CLIENT"`.
- `allow_partial`: when `True`, allows partial coverage uploads (daemon returns canonical coverage ranges).

Example:

```python
import tensorcast

reg = tensorcast.register_view(
    {"w": w, "proj": proj},
    slices={"w": [(0, slice(0, 1024))]},
    transpose={"proj": [(0, 1)]},
    placement="CLIENT",
    options=tensorcast.RegisterArtifactOptions(policy="cache"),
)
```

## Artifact Handles And Materialization

Retrieval is centered on Artifact handles, not on `Store.get` or `Store.get_into`.

- `tensorcast.artifact(...)` and `Store.artifact(...)` return an `Artifact` handle
  that exposes metadata and materialization helpers.
- `Artifact.tensor_dict(...)` and `Artifact.tensor_dict_into(...)` provide the
  primary read surface.
- `tensorcast.from_disk(path)` creates a handle backed by a disk path and
  configures disk-first fallback.

### Why handles?

Handles separate **identity** (artifact id/key/disk hint) from **execution**
(materialize local/P2P/disk, batch, prefetch, verify). This makes it possible to:

- add new materialization sources without changing call sites
- attach per-artifact fallbacks (`with_fallback(...)`)
- preserve cached canonical index metadata across calls

## Fallback Selection

Materialization behavior is controlled by `FallbackOptions` and
`GetArtifactOptions` (advanced):

- `FallbackOptions.prefer` selects `auto`, `local`, `p2p`, or `disk`.
- `disk_path` pins a specific disk location for fallbacks.
- `replica_uuid` hints the daemon to reuse a prefetched replica.
- `verify_checksums` controls descriptor validation on disk reads.

`GetArtifactOptions` is used by the materialization pipeline; most applications
won’t pass it directly today, but it is important for understanding behavior
like region-backed `get_into`, pinned allocation timeouts, and “wait for
completion”.

- Type: [tensorcast/api/_config.py](../../../tensorcast/api/_config.py) (`GetArtifactOptions`, `RegionBackedMode`)

Examples:

```python
import tensorcast

# Disk-first handle with checksum verification.
handle = tensorcast.from_disk("/mnt/models/model_a")
weights = handle.tensor_dict(device="cuda:0")

# Local-only reads (no P2P, no disk fallback).
handle = tensorcast.artifact(artifact_id="mi2:...", fallback="local")
weights = handle.tensor_dict(device="cuda:0")

# Prefer P2P (allow remote replica selection).
handle = tensorcast.artifact(artifact_id="mi2:...", fallback="p2p")
weights = handle.tensor_dict(device="cuda:0")
```

### Store.artifact / from_disk (build handles)

Signature: `tensorcast.artifact(*, artifact_id=None, key=None, disk_path=None, fallback=None)`

Use `artifact(...)` to build a reusable handle that carries identity and fallback
hints. You typically provide exactly one of:

- `artifact_id`: content-addressed id (preferred)
- `key`: mapped to an artifact id via key mapping
- `disk_path`: for disk-backed/disk-hinted materialization

`tensorcast.from_disk(path)` is a convenience wrapper that sets a disk-first
fallback automatically (equivalent to `artifact(disk_path=..., fallback="disk:/path")`).

## StorePolicy And Persistence Hooks

`StorePolicy` is the single durability and placement declaration for `register`
and `put`. It supports:

- Profiles (`cache`, `durable`, `ha`, `cold`, `warm`, `pinned`).
- Explicit tiers with `must`, `should`, `may`.
- `overflow_policy` and `layout` overrides.

When policy includes shared disk or remote stable tiers, the SDK triggers
`StartPersistence` after registration and stores the returned
`persistence_task_id` on `RegisteredArtifact`.

`Store.query_persistence_status(...)` exposes daemon task state by task id or
artifact id.

For the full policy model and examples, see [Policy & Persistence](./policy-persistence.md).

## Region APIs

For region-backed registration and quiesced cleanup:

- `Store.register_vram_region(...)` and `Store.unregister_vram_region(...)` manage
  reusable CUDA IPC regions.
- `Store.deregister_artifact(...)` quiesces and drains active exports, then
  revokes the lease and performs best-effort Global Store cleanup.

Region-backed APIs are primarily about **making LIP safe** and **reducing CUDA IPC
churn** by reusing stable region handles. See [Region-Backed](./region-backed.md)
for request/response fields and lifecycle details.

## Contracts And Invariants

- The SDK owns the process Store and its runtime caches.
- All public methods raise `ArtifactError` with structured status codes.
- Registration and materialization errors map to retryable or non-retryable
  categories; the SDK applies bounded retries for transient errors.
- Policy resolution is authoritative on the daemon; the SDK only validates the
  policy shape.

## Code Map

- Public store facade: [tensorcast/api/store/__init__.py](../../../tensorcast/api/store/__init__.py)
- Store runtime: [tensorcast/api/store/runtime.py](../../../tensorcast/api/store/runtime.py)
- Registration pipeline: [tensorcast/api/store/registration.py](../../../tensorcast/api/store/registration.py)
- Materialization pipeline: [tensorcast/api/store/materialization.py](../../../tensorcast/api/store/materialization.py)
- Policy + options model: [tensorcast/api/_config.py](../../../tensorcast/api/_config.py)
