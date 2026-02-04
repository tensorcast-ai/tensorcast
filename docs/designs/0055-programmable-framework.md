---
slug: 0055-programmable-framework
title: Programmable API Design (Artifact-First) (Design)
description: Extend the existing Artifact/Store handle API with programmable control-plane primitives (context, operations, plans) while preserving TensorCast’s single data-plane and consistency model.
status: implemented
areas:
  - sdk
  - daemon
  - global_store
  - proto
created: 2026-01-23
last_updated: 2026-02-04
related_code:
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/batch_context.py
  - tensorcast/api/_materialize.py
  - tensorcast/api/plan/plan.py
  - tensorcast/api/plan/targets.py
  - tensorcast/api/plan/transforms.py
  - tensorcast/engine_adapter/adapter.py
  - tensorcast/node_agent/executor.py
  - tensorcast/node_agent/server.py
  - daemon/service/controllers/materialization_controller.cc
  - daemon/state/replica_session_manager.h
  - daemon/state/session_lifecycle.h
  - core/common/capability_token.h
  - core/common/capability_token.cc
  - tensorcast/common/capability_token.py
  - proto/tensorcast/common/v1/common.proto
  - proto/tensorcast/common/v1/capability_token.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/global_store/v1/global_store.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/config/v1/node_agent_config.proto
  - schema.sql
related_docs:
  - docs/architecture/api/api-design.md
  - docs/architecture/api/materialization-flow.md
  - docs/architecture/api/policy-persistence.md
  - docs/architecture/api/region-backed.md
  - docs/architecture/p2p-transfer-strategies.md
  - docs/internals/model-loading.md
  - docs/designs/0001-docs-system-design.md
  - docs/designs/0004-unified-runtime-config.md
  - docs/designs/0011-unified-session-lifecycle-leases.md
  - docs/architecture/artifact-views-and-retrieval.md
  - docs/designs/0039-artifact-first-sdk.md
  - docs/architecture/api/region-backed.md
links:
  plan: ../plans/0055-programmable-framework.md
  predecessors:
    - ./0039-artifact-first-sdk.md
    - ./0011-unified-session-lifecycle-leases.md
  dependencies:
    - ./0001-docs-system-design.md
    - ./0004-unified-runtime-config.md
  schema: ../../schema.sql
---

# Programmable API Design (Artifact-First)

This document proposes a **programmable** extension to the existing public Python SDK surface. The key design choice is:

> **Weights, KVCache, checkpoints, etc. are all Artifacts (tensor dicts).**  
> **Programmability must not introduce a parallel “data object” abstraction.**

Instead, programmability is expressed by **composing existing Artifact primitives** (prefetch, into, register/put, persistence, view/subset) with a small set of new control-plane primitives:

- `CallContext`: QoS + deadline + idempotency + tracing tags (per request/task).
- `Operation[T]`: unified operation handle (`status/wait/cancel`) for long-tail control-plane actions (Phase-0: sync/blocking only).
- `Plan`: a composable orchestration IR that dispatches actions across daemons and instances via node-local agents/engine adapters (no “shipping arbitrary Python” like Ray).

This document prioritizes **What/Why** (semantics and rationale) and defines contracts/invariants that must hold from Phase-0.

Planned advanced extensions (Controller execution, app/instance agents, signals, and engine-agnostic KV-cache integration)
are specified in `docs/designs/0056-programmable-framework-adv.md`.

---

## Scope & Relationship

This design extends and refines the artifact-first SDK (`docs/designs/0039-artifact-first-sdk.md`) with **programmable**
control-plane primitives while keeping TensorCast’s data plane and consistency model unified.

Scope choices (long-term, repo-wide):

- **Artifact-first**: programmability composes `Artifact`/`View`/`StorePolicy`; it does not introduce a parallel “data
  object” abstraction.
- **Control-plane programmable, data-plane fixed**: programmability may add control RPCs (idempotency, pinning,
  wait/cancel/status), but does not fork transport/loading semantics.
- **Node-local safety boundary**: any action that touches PID/IPC/regions must run at the node-local boundary
  (Engine Adapter / node agent), not from a central controller.

## Prior State (Grounding)

This section captures the repo state that motivated the design and clarifies what is true **as of 0055** (implemented)
vs what was true **before 0055** (historical motivation). This avoids ambiguity given `status: implemented`.

**Before 0055 (historical gaps; motivation)**

- **Local-only process-context paths**: `wait_for_completion` materialization and region-backed
  `MaterializeIntoTarget` were loopback-only (PID/UDS coupled). Remote orchestration required dispatching work to
  node-local agents / engine adapters (`daemon/service/controllers/materialization_controller.cc`).
- **PID coupling**: handle export paths relied on PID-derived liveness/leases for local IPC safety
  (`daemon/service/controllers/materialization_controller.cc` + `daemon/state/session_lifecycle.h`). Warm pools needed
  a PID-independent mode so retries/prefetch did not accidentally become per-PID residency.
- **`replica_uuid` reuse hazards**: `replica_uuid` reuse could overwrite or cross-wire operation/session bookkeeping,
  leading to unsafe joins across different `ReplicaKey`s.
- **Placement pinning API gap**: the daemon tracked placement pins internally but did not expose a stable, capability
  token-based RPC surface for pin/renew/release.
- **No engine instance registry**: Global Store tracked workers and replicas, but did not expose a first-class “engine
  instance” registry for routing instance-scoped actions.

**As of 0055 (implemented; current state)**

- **Prefetch is a first-class operation**: `Artifact.prefetch(device=...)` returns `Operation[PrefetchedReplica]` with
  `wait/cancel/status` semantics backed by daemon-side operation RPCs; the legacy `PrefetchTicket` wrapper is removed.
- **PID-independent orchestration exists**: daemon-owned warm actions default to `NO_LEASE` for retry safety and for
  decoupling from PID-bound handle export semantics.
- **Safe `replica_uuid` joins are enforced**: the daemon enforces PutIfAbsent/JoinIfMatch semantics for `replica_uuid`
  joins and fails fast on mismatched reuse (`daemon/state/replica_session_manager.h`).
- **Placement pins are exposed**: placement pinning is a first-class programmable primitive (`pin_device_residency`)
  backed by daemon RPCs and capability tokens.
- **Instance registry exists**: Global Store exposes instance registration/heartbeat/listing (`RegisterInstance`,
  `InstanceHeartbeat`, `ListActiveInstances`) for `instance_id` discovery.

## Navigation

- [Scope & Relationship](#scope--relationship)
- [Prior State (Grounding)](#prior-state-grounding)
- [Goals](#goals)
- [Non-Goals](#non-goals)
- [Core Concepts](#core-concepts)
- [Entry Points](#entry-points)
- [Public API Surface](#public-api-surface)
  - [CallContext](#callcontext)
  - [Operation](#operation)
  - [Artifact Enhancements](#artifact-enhancements)
  - [Residency Pinning (Placement Pin)](#residency-pinning-placement-pin)
  - [Plan (Programmable Orchestration)](#plan-programmable-orchestration)
  - [Targets & Engine Adapter](#targets--engine-adapter)
  - [Transforms (Reshard / KV Layout)](#transforms-reshard--kv-layout)
- [Feature Mapping (Legacy → Programmable)](#feature-mapping-legacy--programmable)
- [Contracts and Invariants](#contracts-and-invariants)
- [Error, Retry, Deadline, Cancel Semantics](#error-retry-deadline-cancel-semantics)
- [Proto Requirements](#proto-requirements)
- [Schema Changes](#schema-changes)
- [Trade-offs & Risks](#trade-offs--risks)
- [Examples](#examples)
- [Compatibility & Migration](#compatibility--migration)
- [Code Map](#code-map)
- [Naming Compliance](#naming-compliance)
- [Appendix](#appendix-design-rationale-why-this-shape)

---

## Goals

1. **Artifact-first consistency**
   - Keep `Artifact` handles as the single unit of identity + metadata + fallbacks.
   - Keep `StorePolicy` as the single durability/placement declaration.
   - Keep daemon-owned data plane and existing materialization pipeline semantics.

2. **User-friendly programmability**
   - Most users should only learn:
     - `tensorcast.artifact(...)` / `Store.artifact(...)`
     - `Artifact.prefetch(...)`, `Artifact.tensor_dict_into(...)`
     - `Artifact.pin_device_residency(...)` (new)
     - `CallContext` (new) and `Plan` (new, advanced)
   - Domain-specific data (e.g., weights, KV cache, checkpoints) is expressed as artifacts using stable key/view conventions; higher-level helper APIs are layered on top (see `docs/designs/0056-programmable-framework-adv.md`).

3. **Unified operation semantics (Phase-0: sync-only)**
   - Prefetch, persistence, and device residency pinning expose a unified `Operation[T]` interface with
     `status/wait/cancel`.
   - Phase-0 does **not** introduce any new `async def` / `await` SDK surface. (Internal concurrency is allowed.)

4. **At-least-once orchestration with safe idempotency**
   - When a stable `CallContext.idempotency_key` is provided, repeated submissions of deterministic-operation-id
     actions (e.g., daemon-owned materialization operations) MUST reuse the same operation id and MUST NOT leak daemon
     operation records.
   - Placement pin creation is capability-based in 0055 and is **not idempotent**: callers MUST avoid implicit retries
     on unknown outcomes and SHOULD always use a finite `ttl_ms`. (Idempotent pin creation is a planned extension; see
     `docs/designs/0056-programmable-framework-adv.md`.)

5. **Deterministic operation id (`replica_uuid`) for joinable actions**
   - Use `replica_uuid` as a joinable operation/session id for wait/cancel/telemetry for daemon-owned materialization
     actions (not replica identity).
   - Placement pins are joinable via capability tokens (`PlacementPin.capability_token`) and do not have deterministic
     operation ids in 0055.
   - Avoid duplicate daemon-owned VRAM replicas via the existing engine invariant: join on `ReplicaKey`.

6. **Durability vs residency are orthogonal**
   - `StorePolicy` describes durable tiers (stable_dram/shared_disk), not “GPU never evict”.
   - Residency is expressed via placement pins (device residency) and/or engine-owned memory.

---

## Non-Goals

- No new data-plane implementation: RDMA/MTCP/FlowCredit and transport locks remain unchanged.
- No requirement to build a full distributed function execution runtime (Ray-like). `Plan` dispatches **actions**, not arbitrary Python code.
- No requirement for Central Executor state recovery in v1.
- No hard dependency on view-aware routing in Global Store (Phase-0 can be canonical-only).

---

## Core Concepts

### Artifact (existing)

- A logical collection of named tensors with canonical index and byte layout.
- Identity: `artifact_id` (content-addressed, e.g. `mi2:...`) or `key` (human-friendly name).
- Retrieval and execution are centered on `Artifact` handles.

### View (existing)

A view is a derived artifact shape/layout (subset, slice, transpose, etc.) whose identity must be stable (`view_id`) and participates in cache/memory semantics.

**Programmability requirement**

- Any non-identity view MUST resolve to a stable `view_id` (`docs/architecture/artifact-views-and-retrieval.md`).
- Any view that changes bytes MUST participate in cache identity (`view_id`), `ReplicaKey`, and the selection identity
  used for idempotent operations.

### Selection identity (new; required)

Programmability needs a stable identity for **“which bytes”** that is independent of **“where they are written”**. This
identity must be used consistently across:

- action-level idempotency (`CallContext.idempotency_key` → deterministic `operation_id`)
- cache keys (daemon/engine)
- plan step fingerprinting (controller/agent)
- validation/debug (detect mismatched joins)

We standardize two hashes (see `docs/architecture/api/region-backed.md` for canonical definitions):

1) **`logical_layout_hash`** (base ByteSpace identity)  
   Identity of the base logical byte space: canonical/view index bytes + index kind. Excludes any physical binding
   (regions, offsets, addresses).

2) **`selection_hash`** (selection identity on top of the base)  
   Identity of “which bytes to produce”: resolved `view_id` (if any) + subset identity (sorted/unique `tensor_names`
   and/or `view_subset_hash` as raw digest bytes). This MUST exclude any placement/target binding.

**Rule (required):** placement (tier/device) is part of `ReplicaKey` (cache entity identity) and part of the
action-level idempotency fingerprint, but it MUST NOT be part of `selection_hash`.

**Design rule:** any joinable or retryable action MUST derive its stable fingerprint from `(artifact_id,
logical_layout_hash, selection_hash)` (or an equivalent legacy tuple that is provably collision-free). It MUST NOT
include any physical bindings (addresses, region ids, buffer handles). Stable target identity (e.g., `daemon_id` /
`instance_id`) and placement (tier/device) are action-scoped inputs layered on top of this selection identity.

Canonical encoding (required):

- `logical_layout_hash` and `selection_hash` are raw digest bytes (wire/proto: `bytes`). Any time digest bytes are
  embedded into a string fingerprint, they MUST be encoded as lowercase hex (e.g., `layout_hex =
  logical_layout_hash.hex()`).
- Fingerprint strings MUST be encoded as UTF-8 prior to UUID/digest derivation.
- `|` is a structural delimiter in fingerprints. Any interpolated field value MUST be delimiter-safe (MUST NOT contain
  `|`).
- `CallContext.idempotency_key` is treated as opaque user input. When it participates in a structured UUIDv5 “name”
  string, it MUST be canonicalized as `idempotency_key_hex = sha256(utf8(idempotency_key)).hexdigest()` (lowercase
  hex) to ensure delimiter-safety and bounded size.

### Replica (existing)

A concrete materialization of an artifact (or view) on a tier/device.

### ReplicaKey (existing; StoreEngine identity)

In the C++ StoreEngine, the identity of a daemon-owned replica is the `ReplicaKey`:

- `artifact_id`
- `view_id` (optional; absent means canonical view)
- `device` (`DeviceType`, `ordinal`, `uuid`)
- `replica` (uint32; Phase-0 uses `replica=0`)

This is the key that determines which underlying daemon-owned VRAM/DRAM replica is being loaded/kept around.

### ReplicaKey vs operation id (`replica_uuid`) (distinct; required)

TensorCast has two different identities that must not be conflated:

1) **ReplicaKey (replica identity; existing)**  
   The identity of a daemon-owned materialization in the StoreEngine, derived from `(artifact_id, view_id, device_id,
   tier, ...)`.  
   **Engine invariant**: repeated materialization on the same daemon MUST join on `ReplicaKey` to avoid duplicate VRAM
   allocations; programmability must preserve and rely on this behavior.

2) **`replica_uuid` (operation/session id; used for wait/cancel/telemetry)**  
   A joinable handle for “this operation” (e.g., prefetch/pin/persistence wait), **not** “the replica identity”.
   Multiple `replica_uuid`s may legitimately point to the same `ReplicaKey` (different callers/operations joining the
   same underlying load).

This separation is what makes programmable orchestration safe: cancellation can be scoped to an operation id without
introducing cross-talk between unrelated callers that happen to touch the same `ReplicaKey`.

### Two ownership modes (must be explicit)

1. **Daemon-owned replica (cache warm / shared cache)**
   - Created by `Artifact.prefetch(...)` (daemon materializes and owns VRAM/DRAM).
   - Optional: kept resident by `pin_device_residency(...)` (placement pin).

2. **Caller-owned buffer (engine arena / KV buffers)**
   - Filled by `Artifact.tensor_dict_into(...)` / `deferred_loader.commit(...)`.
   - Uses region-backed `MaterializeIntoTarget` when enabled.
   - Does **not** create daemon-owned VRAM replica.

### CallContext (new)

A per-request/task context carrying:

- QoS class (realtime/interactive/background)
- hard deadline
- idempotency key / request_id
- optional debug tags

### Operation[T] (new unified operation handle)

A unified interface for long-tail workflows (Phase-0: sync/blocking only):

- prefetch wait
- persistence task wait
- placement pin lifecycle
- status/cancel unify across subsystems

#### Operation backend selection (integration with operation.proto)

`Operation[T]` is an SDK-level abstraction. The **backend protocol** can differ by workload:

1. **Daemon-local replica operations**: `QueryReplicaStatus` / `WaitReplicaStatus` with `ReplicaOperationStatus`
   (high-frequency, short-lived, daemon-local actions like prefetch/materialize).
2. **Global Store-backed operations**: `tensorcast.operation.v1` (`Acquire/Keepalive/Release/Get/UpdateOperation`)
   for long-tail, globally coordinated workflows.
3. **Polling wrappers**: legacy status RPCs (e.g., persistence) wrapped into `Operation[T]` when a dedicated operation
   backend does not exist yet.

**Use `operation.proto` when ALL of the following are true:**

- **Global coordination required**: multiple daemons could race, or the action must be serialized across the cluster.
- **Lease + fencing needed**: correctness depends on a single active coordinator with a monotonic generation token.
- **Durable progress matters**: status/progress/results must survive daemon restarts and be queryable without the
  originating daemon.
- **Low write rate is acceptable**: operation updates can be throttled (avoid per-chunk/per-shard high-frequency writes).

**Do NOT use `operation.proto` when any of the following hold:**

- **High-frequency or short-lived** actions (e.g., prefetch, replica readiness) where GS write amplification would be
  prohibitive.
- **Daemon-local semantics dominate**: progress is tied to local lifecycle or PID/IPC resources.
- **Rich domain-specific status** is required (e.g., shard-level persistence details) that does not map cleanly to a
  single `OperationStatus`.

**RPC expectations when using `operation.proto`:**

- The coordinator acquires a lease (`AcquireOperationLease`) and **must** include the fencing token
  (`lease_generation`) on all updates.
- `UpdateOperation` writes are **throttled** to bounded rates; snapshots/results are stored in `Any`.
- Clients query via `GetOperation` / `WaitOperation` and **must not** assume daemon-local state.

This split keeps the **user-facing `Operation[T]` API uniform** while avoiding unnecessary Global Store load for
high-frequency or local-only actions.

### Plan (new orchestration IR)

A composable plan expressing:

- which node/instance executes which action
- dependencies and concurrency (Phase-0: no plan-level reschedule retries; future phases may add bounded retries)
- derived idempotency via context + canonical inputs

Plan is an **IR**, not “send Python to remote”.

### TargetSpec (new engine integration abstraction)

For remote `into(...)` operations, targets cannot be raw `torch.Tensor` across machines. A `TargetSpec` is a serializable reference resolved by the node’s engine adapter to actual buffers.

### Runtime topology and safety boundary (required)

Programmability must respect the existing TensorCast process/runtime boundaries:

- **Global Store (central, required)**: metadata & coordination (worker registry, instance registry, persistence metadata).
- **Store Daemon / Worker (node-local, required)**: daemon-owned data plane (`StoreEngine`, replica lifecycles, P2P transfers, placement pins).
- **Node Agent (node-local, required for instance steps)**: executes actions that must run on the target node (including actions that require process context via an Engine Adapter).
- **Engine Adapter (in-process, required for PID/IPC/region binding)**: executes process-context actions (region-backed `into`, LIP registration, target resolution, transforms).

Remote-safety rules:

- Any action requiring PID/IPC/region references MUST run via the Engine Adapter on the target instance.
- Any daemon-owned action that should be safe under retries and should not couple to a PID MUST run in `NO_LEASE` mode
  (see Prefetch/Pinning).
- Remote callers must not directly call PID/IPC-binding RPCs across nodes; they dispatch instance-scoped plan steps to Node Agents, which enforce process-context safety.

---

## Entry Points

### Existing (unchanged)

- `tensorcast.init(...)`
- `tensorcast.store(...)`
- `tensorcast.register / put / register_view / artifact / from_disk` (module-level wrappers)
- `Store.register / put / register_view / artifact / query_persistence_status / region APIs`

### New (additive)

- `tensorcast.context(...) -> CallContext`
- `tensorcast.plan(ctx: CallContext) -> Plan`

Example:

```python
import tensorcast

tensorcast.init(mode="connect", address="127.0.0.1:50051")

ctx = tensorcast.context(
    qos="realtime",
    deadline_ms=50,
    tags={"tenant": "prod"},
)

art = tensorcast.artifact(key="/prod/weights/llama/v7", fallback="p2p")

# Issue (non-blocking) a daemon-owned warm replica on some worker:
op = art.prefetch(device="cuda:0", ctx=ctx)  # Operation[PrefetchedReplica]

# Blocking wait:
replica = op.result(timeout_s=60.0)
```

---

## Public API Surface

### CallContext

`CallContext` is a pure per-call data container: it does not construct handles, and it does not participate in identity.

```python
from dataclasses import dataclass
from typing import Literal, Mapping

QosClass = Literal["realtime", "interactive", "background"]

@dataclass(frozen=True, slots=True)
class CallContext:
    request_id: str | None = None
    qos: QosClass = "interactive"
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    tags: Mapping[str, bool | int | float | str] | None = None
```

**Notes**

- `request_id` is optional. If omitted, the SDK generates one so every operation is traceable.
- If the caller already has a stable idempotency key (stable across retries and across agents), it should be set as
  `idempotency_key`. Otherwise, `request_id` provides traceability, not cross-retry idempotency.
- `deadline_ms` is an end-to-end budget (relative), enforced as a hard cutoff (see Deadline semantics).
 - Rule of thumb: **handles come from `tensorcast` / `Store` factories; `ctx` appears only on action execution (or on `tensorcast.plan(ctx)`).**

**Module-level helper**

```python
def context(
    *,
    request_id: str | None = None,
    qos: QosClass = "interactive",
    deadline_ms: int | None = None,
    idempotency_key: str | None = None,
    tags: Mapping[str, bool | int | float | str] | None = None,
) -> CallContext: ...
```

#### Signature consistency (required)

- **Handle factories are context-free**: `tensorcast.artifact(...)` / `Store.artifact(...)` never accept `ctx`.
- **All action APIs accept** `*, ctx: CallContext | None = None` as a keyword-only parameter.
- **`ctx` does not participate in artifact/view/selection identity**; for deterministic `operation_id`, only `ctx.idempotency_key` participates (deadline/qos/tags must never affect identity).
- **Plan is the only exception**: `tensorcast.plan(ctx)` binds the context once; plan steps intentionally do not accept `ctx` to avoid repetition.

Why this is the most consistent:

- `Artifact` represents **what** (identity); `ctx` represents **how this time** (deadline/qos/idempotency/tags).
- It prevents mixed patterns like `ctx.artifact(...)`, `tensorcast.artifact(..., ctx=...)`, and `art.prefetch(..., ctx=...)` from co-existing.
- It makes it easy to enforce a repo-wide signature rule: every RPC/scheduling action takes the same keyword-only `ctx=`.

### Operation

Unified operation interface. New programmable control-plane actions should return an `Operation[T]`; existing long-tail surfaces (persistence task ids) should be adapted to this interface for consistency.

**Phase-0 design rule:** public APIs are sync/blocking only (no `async def`, no `await`).

```python
from dataclasses import dataclass
from typing import Generic, Literal, Mapping, TypeVar

T = TypeVar("T")

OperationState = Literal[
    "pending",
    "running",
    "success",
    "failed",
    "cancelled",
    "degraded",
]

@dataclass(frozen=True, slots=True)
class TimeoutErrorDetails:
    """Additional detail for DEADLINE_EXCEEDED/timeout outcomes."""

    kind: Literal["ctx_deadline", "wait_timeout"]
    deadline_ms: int | None = None
    timeout_s: float | None = None
    elapsed_ms: int | None = None

@dataclass(frozen=True, slots=True)
class OperationError:
    """
    Structured error surfaced by Operation status APIs.

    Notes:
    - `status_code` uses gRPC canonical code names (e.g. DEADLINE_EXCEEDED).
    - `context` is a small, safe-to-log dictionary (e.g. request_id, worker_id, artifact_id, operation_id).
    """

    status_code: str
    message: str
    retryable: bool
    timeout: TimeoutErrorDetails | None = None
    context: Mapping[str, str] | None = None

@dataclass(frozen=True, slots=True)
class OperationStatus:
    state: OperationState
    message: str | None = None
    progress: float | None = None
    as_of_ms: int | None = None
    error: OperationError | None = None

class Operation(Generic[T]):
    operation_id: str
    def done(self) -> bool: ...
    def status(self) -> OperationStatus: ...
    def result(self, *, timeout_s: float | None = None) -> T: ...
    def wait(self, *, timeout_s: float | None = None) -> T: ...
    def cancel(self) -> bool: ...
```

**Timestamp conventions**

- `as_of_ms` and `*_at_ms` fields use Unix epoch milliseconds (wall-clock) so they can be correlated across machines.
- `deadline_ms` and `timeout_s` are budgets (relative), not timestamps.

`Operation` is always scoped to an execution target (a specific daemon/worker for daemon-owned actions, or a specific
engine instance for in-process actions). The SDK MUST retain enough target information to implement
`status/wait/cancel` without assuming `operation_id` is globally unique.

#### Operation is a system primitive (not just an SDK wrapper)

For correctness, daemon-owned actions MUST have daemon-side, operation-scoped state:

- A daemon MUST treat `replica_uuid` as an **operation id** (join/wait/cancel/telemetry), not as replica identity.
- Multiple `replica_uuid`s may map to the same `ReplicaKey`; cancel MUST NOT affect other `replica_uuid`s.
- If a request attempts to reuse an existing `replica_uuid` but resolves to a different `ReplicaKey`, the daemon MUST
  fail fast with `FAILED_PRECONDITION` (never silent overwrite).

#### Operation lifecycle and completion conditions (normative)

`Operation.wait()` resolves when the underlying action reaches its completion condition:

- `prefetch`: the daemon-owned replica is **ready for serving** (loaded on the requested device/tier). This does not
  imply IPC handle export, and does not imply verification unless explicitly enabled.
- `pin_device_residency`: the placement pin is active and a valid `PlacementPin` capability is returned.
- `persistence`: the persistence task reaches a terminal state (`success|failed|degraded`).
- `into` / transforms: the in-process Engine Adapter has completed the buffer writes / produced the output artifact.

**`degraded` is used when**

- The caller’s deadline is exhausted.
- Best-effort cancellation has been initiated.
- The underlying action may still complete in the background.

**Failure reporting (required)**

- `Operation.status()` MUST return structured failure detail in `OperationStatus.error` when `state in {"failed", "degraded"}`.
- `Operation.wait()` MUST raise `ArtifactError` (or a subclass) on failure; implementations SHOULD provide a dedicated timeout subtype (e.g., `OperationTimeoutError`) for `timeout_s` expirations and MUST include enough context (`request_id`, `operation_id`, and the failing target identifiers) to make user-driven retries feasible.

### Artifact Enhancements

The existing `Artifact` handle remains the center of retrieval and execution. Programmability is expressed as handle methods and views, not parallel APIs.

#### Prefetch (daemon-owned cache warm)

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PrefetchedReplica:
    artifact_id: str
    view_id: str
    operation_id: str
    device_id: int
    daemon_id: str
    source: str | None  # "p2p" | "disk" | "local" | None

class Artifact:
    def prefetch(
        self,
        *,
        device: torch.device | str | int,
        ctx: CallContext | None = None,
        options: "GetArtifactOptions | None" = None,
    ) -> Operation[PrefetchedReplica]:
        """
        Ensure a daemon-owned replica exists (and optionally begins loading immediately).

        Returns an `Operation[PrefetchedReplica]` (Phase-0: sync/blocking only):
        - `op = art.prefetch(...)`
        - `replica = op.result(timeout_s=...)`

        Semantics:
        - Daemon-owned residency (VRAM/DRAM is owned by the daemon).
        - Defaults to `NO_LEASE` (does not create PID UseLeases; does not mint IPC handle leases).
        - CPU/DRAM prefetch is allowed under `NO_LEASE` because `prefetch` does not export any CPU memfd / IPC handle
          to the caller. Any API that returns a handle (`mem_handle`) remains PID/lease-bound and is separate from
          daemon-owned warm replicas.
        - `device` may refer to:
          - GPU VRAM (e.g., `"cuda:0"` or `0`), or
          - daemon-owned host DRAM (e.g., `"dram"`, `"cpu"`, or `-1`).
        - Uses a deterministic **operation id** (sent as `replica_uuid`) when `ctx.idempotency_key` is provided.
        - Cache identity follows StoreEngine `ReplicaKey` (engine dedup is `ReplicaKey`-based; operation ids are not).
        - Prefetch does not imply pinning. To keep a replica resident, callers must use `pin_device_residency(...)`.
        """
```

**Deterministic operation id (wire: `replica_uuid`) (when `ctx.idempotency_key` is provided)**

The on-wire field is historically named `replica_uuid`, but semantically it is an **operation id** (joinable wait/cancel handle), not replica identity.

When `ctx.idempotency_key` is provided, the SDK derives:

- `idempotency_key_hex = sha256(utf8(idempotency_key)).hexdigest()` (lowercase hex; see canonical encoding rules above)
- `layout_hex = logical_layout_hash.hex()` and `selection_hex = selection_hash.hex()` (lowercase hex; see canonical encoding rules above)
- `action_fingerprint = f"prefetch|daemon={daemon_id}|artifact={artifact_id}|layout={layout_hex}|selection={selection_hex}|device={device_id}|lease=NO_LEASE|v1"`
- `op_namespace_v1 = UUID5(NAMESPACE_DNS, "tensorcast.op.v1")` (namespace derivation; RFC 4122 UUIDv5; `NAMESPACE_DNS = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")`)
- `operation_id = UUID5(op_namespace_v1, utf8(f"{idempotency_key_hex}|{action_fingerprint}"))`
- `replica_uuid = operation_id` (wire name)

Properties:

- **Safe retries**: resubmitting the same action with the same `idempotency_key` reuses the same operation id.
- **No cross-talk cancellation**: cancelling an operation id MUST NOT cancel other operations, even if they join the same
  underlying `ReplicaKey`.
- **VRAM dedup remains ReplicaKey-based**: the daemon/engine MUST join repeated loads on `ReplicaKey`, independent of
  operation ids.
- **Canonical UUID form**: the on-wire `replica_uuid` MUST use RFC 4122 canonical string form (lowercase hex with `-`).

Canonicalization rules (required):

- `selection_hash` MUST be derived from canonicalized inputs (stable `view_id`, sorted/unique subset identity).
- `artifact_id` MUST be the resolved artifact identity string (`mi2:...` or `cgid:...`), not an unstable alias/key.
- `logical_layout_hash` MUST be computed from canonical/view index bytes (see Selection identity) and MUST NOT be derived
  from an unstable alias/key. If an `Artifact` handle is key-only (unresolved `artifact_id` / missing index bytes), the
  SDK MUST resolve key→artifact_id and fetch canonical/view index bytes before deriving deterministic operation ids; if
  index bytes cannot be resolved, deterministic operation ids MUST NOT be produced silently (the call MUST fail rather
  than “fall back” to a non-deterministic operation id when `ctx.idempotency_key` is provided).
- `device` MUST canonicalize to an integer `device_id` before fingerprinting (e.g., `"cpu"`/`"dram"`/`-1` → `-1`,
  `"cuda:N"`/`N` → `N`).
- `daemon_id` MUST be a stable daemon identity (derived from unified config). If only a network address is available,
  deterministic operation ids remain deterministic but are HA-unsafe; SDKs SHOULD fail fast when
  `ctx.idempotency_key` is provided but `daemon_id` cannot be resolved.
- `ctx.deadline_ms`, `ctx.qos`, and `ctx.tags` MUST NOT affect operation identity.

#### Into (caller-owned buffers; engine arena / KV buffers)

This is the existing primary read surface (`tensor_dict_into`). Programmable extensions add optional per-call context and remote target indirection.

```python
class Artifact:
    def tensor_dict_into(
        self,
        target: "dict[str, torch.Tensor] | Mapping[str, torch.Tensor] | TargetSpec",
        *,
        device: torch.device | str | None = None,
        ctx: CallContext | None = None,
        options: "GetArtifactOptions | None" = None,
    ) -> None:
        """
        Fill caller-owned buffers.

        Notes:
        - When region-backed is enabled and selected, uses MaterializeIntoTarget (loopback/UDS only).
        - Region-backed `into` does not allocate daemon-owned VRAM replicas.
        - If region-backed cannot be used and the SDK falls back to the standard materialization path, normal daemon-side
          materialization/caching semantics apply (the replica may be created transiently and released after copy).
        - For remote execution, `target` MUST be a `TargetSpec` capability minted by the engine adapter on the target
          instance. Controllers must not mint targets directly.
        - Cross-process/remote `into(...)` orchestration is supported via `Plan` instance steps when the target is a
          `TargetSpec` minted by the engine adapter on the target instance.
        """
```

For advanced usage, deferred loader remains the recommended path.

```python
with art.deferred_loader(device="cuda:0") as loader:
    q = loader.tensor("layers.0.attn.q_proj.weight")
    ...
    loader.commit()
```

#### Persistence (publish durability; existing)

Persistence remains expressed by `StorePolicy` (durable tiers) and is primarily triggered by publish flows
(`register/put`). Programmability requires that any long-tail persistence work is surfaced as an `Operation[...]`:

- Phase-0: wrap the existing `persistence_task_id` + `query_persistence_status(...)` into an `Operation` (polling).
- Phase-1: add a daemon `WaitPersistenceStatus` RPC to avoid polling storms (mirrors `WaitReplicaStatus`).

### Residency Pinning (Placement Pin)

This is the missing “device residency intent” primitive (e.g., GPU residency), orthogonal to `StorePolicy`.

**Important:** `StorePolicy(profile="pinned")` already exists today and expresses **stable DRAM retention** semantics. The programmable primitive introduced here is **device residency pinning** for daemon-owned replicas and must not be conflated with policy-tier retention.

Implementation note (as of 2026-02-04): `pin_device_residency` targets GPU devices only (`device_id >= 0`). Host DRAM
retention remains expressed via `StorePolicy(profile="pinned")` and daemon cache policy.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PlacementPin:
    pin_id: int
    capability_token: str     # opaque, unforgeable capability minted by the daemon
    daemon_id: str            # stable target identity (derived from daemon config)
    artifact_id: str
    view_id: str
    device_id: int
    expires_at_ms: int | None

    def renew(self, *, ttl_ms: int, ctx: CallContext | None = None) -> "PlacementPin": ...
    def release(self, *, ctx: CallContext | None = None) -> None: ...

class Artifact:
    def pin_device_residency(
        self,
        *,
        device: str | int,
        ttl_ms: int | None,
        ctx: CallContext | None = None,
    ) -> Operation[PlacementPin]:
        """
        Keep a daemon-owned replica resident via placement_pins.

        Semantics:
        - Does not create PID UseLease (pinning is not process-liveness-coupled).
        - Best-effort: daemon restart loses leases; callers must tolerate and rebuild.
        - `capability_token` is required for renew/release; it is daemon-scoped and time-bounded.
        - Idempotency (implemented constraint): pin creation is capability-based and **not idempotent**. Callers MUST
          NOT automatically retry on unknown outcomes (it may create multiple pins). Callers SHOULD always provide a
          finite `ttl_ms`. To extend TTL, use `PlacementPin.renew(...)` instead of re-issuing a new pin.
        """
```

### Plan (Programmable Orchestration)

Plan is a programmable orchestration IR that composes artifact actions across nodes.

Naming note: `Plan` here is orchestration IR, distinct from the existing `PlanType` enum used for registration plans (`tensorcast/api/_config.py`).

#### Why Plan (vs `submit_to(fn=...)`)

- Avoids “distributed Python execution” complexity.
- Supports at-least-once semantics with idempotent action submission (Phase-0 retries are user-driven).
- Provides a composable DAG of actions: worker steps run on Store Daemons, and instance steps run via Node Agents / Engine Adapters.

#### Plan is a serializable IR (normative)

For long-term correctness, `Plan` MUST be serializable and versioned (proto preferred) so that:

- controller/agent can compute identical action fingerprints from identical inputs
- retries and at-least-once execution remain safe under partial failures
- plan steps can be audited and replayed (without “shipping arbitrary Python”)

**Implementation note (as of 2026-02-04)**:

- `PlanSpec` proto (`proto/tensorcast/plan/v1/plan.proto`) and `Plan.to_spec()` provide a versioned, deterministic IR.
- A Node Agent implementation exists for node-local execution and Engine Adapter dispatch.
- `Plan.run()` executes in the caller process and dispatches actions to Store Daemons and Node Agents.

#### Node identity: Worker vs Instance

For correctness, Plan distinguishes:

- **Worker** (daemon identity; cache/prefetch/pin run here)
- **Instance** (engine process identity; into/targets resolved here)

Instance registry is exposed via the Global Store (`RegisterInstance`,
`InstanceHeartbeat`, `ListActiveInstances`) so callers can resolve stable
`instance_id` targets.

#### Target identity rules (required)

- `daemon_id` and `instance_id` are required stable identities derived from unified config (not from network address).
- `worker_id` and `daemon_address` are operational routing details and may change under HA re-registration.

#### Minimal types

```python
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class Worker:
    worker_id: str
    daemon_address: str
    daemon_id: str
    p2p_port: int | None = None
    labels: Mapping[str, str] | None = None

@dataclass(frozen=True, slots=True)
class Instance:
    instance_id: str
    worker_id: str
    engine: str
    signals_endpoint: str | None = None
    labels: Mapping[str, str] | None = None
```

Note: the implemented `Instance` type does not carry `daemon_id`. `worker_id` is a routing attribute and may change
under HA re-registration; callers that require HA-safe co-location information should resolve `daemon_id` via Global
Store instance listings keyed by `instance_id`.

#### Plan API

`Plan` binds a single `CallContext` at construction time (`tensorcast.plan(ctx)`). To keep call sites uniform and avoid repetition, plan step builders do not accept `ctx`.

```python
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Sequence, TypeVar

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class PlanStepRef(Generic[T]):
    """A typed reference to a planned step (IR), not an executing operation."""

    step_id: str

class Plan:
    def __init__(self, ctx: CallContext): ...

    def on_worker(self, worker: Worker) -> "WorkerStepBuilder": ...
    def on_instance(self, inst: Instance) -> "InstanceStepBuilder": ...

    def to_spec(self) -> "PlanSpec": ...

    def run(
        self,
        *,
        concurrency: int = 16,
        raise_on_error: bool = True,
    ) -> "PlanResult": ...


class PlanFailedError(ArtifactError):
    """Raised when `Plan.run()` observes any step failure."""

    result: "PlanResult"

@dataclass(frozen=True, slots=True)
class PlanStepResult:
    step_id: str
    target_id: str  # daemon_id or instance_id (stable)
    action: str
    status: OperationStatus
    value: Any | None = None

@dataclass(frozen=True, slots=True)
class PlanResult:
    ok: bool
    request_id: str
    steps: Mapping[str, PlanStepResult]

    def step(self, ref: PlanStepRef[Any]) -> PlanStepResult: ...
```

`Plan.run(raise_on_error=True)` raises `PlanFailedError` by default to prevent silently ignoring failures; callers can opt into “always return a result” via `raise_on_error=False`.

#### Worker steps (daemon-owned)

```python
class WorkerStepBuilder:
    def prefetch(
        self,
        art: "Artifact",
        *,
        device: str | int,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef["PrefetchedReplica"]: ...

    def pin_device_residency(
        self,
        art: "Artifact",
        *,
        device: str | int,
        ttl_ms: int | None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef["PlacementPin"]: ...

    def unpin_device_residency(
        self,
        pin: "PlacementPin",
        *,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[None]: ...
```

#### Instance steps (engine-owned; via Node Agent)

Instance steps are dispatched to the target instance’s Node Agent, which calls the engine adapter.

```python
class InstanceStepBuilder:
    def transform_register(
        self,
        src: "Artifact",
        *,
        spec: TransformSpec,
        out_key: str,
        policy: "StorePolicy | None" = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef["Artifact"]: ...

    def transform_into(
        self,
        src: "Artifact",
        *,
        spec: TransformSpec,
        target: TargetSpec,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[None]: ...
```

#### Execution

`Plan.run()` enforces:

- bounded concurrency
- deadline propagation
- action-level idempotency derived from `(ctx.idempotency_key, action_name, selection identity (artifact_id, logical_layout_hash, selection_hash), placement (tier/device), stable target identity, and action-specific stable inputs such as ttl_ms)`
- atomic success reporting: `PlanResult.ok` is `True` only if all steps succeed

#### Failure, retry, and partial success (Phase-0)

Phase-0 adopts a deliberately simple contract:

- **Atomic reporting (not rollback)**: any single-step failure causes the overall plan result to be marked failed (`PlanResult.ok=False`).
- **Plan atomicity is reporting atomicity only**: side effects may remain even when `ok=False` (e.g., a replica may have been prefetched successfully before a later step failed).
- **No rollback**: Phase-0 does not attempt to “undo” side effects that already happened (e.g., a replica that was prefetched before a later step failed may still exist).
- **Best-effort cancellation**: once a failure is observed, remaining not-yet-started steps are skipped and in-flight steps receive a best-effort cancel; some steps may still succeed concurrently.
- **No plan-level reschedule retries**: Phase-0 does not automatically reschedule failed steps; callers can retry by re-running the whole plan with the same `CallContext.idempotency_key` and by using `OperationStatus.error.retryable` + `OperationStatus.error.status_code` to decide what to retry vs discard.

### Targets & Engine Adapter

#### TargetSpec (serializable)

```python
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class TargetSpec:
    """
    A serializable reference to engine-owned buffers.
    Resolved by the engine adapter on the target instance.
    """

    instance_id: str  # target engine instance identity
    name: str  # e.g. "weights.params", "kv.req:123"
    tensors: Mapping[str, str]  # tensor_name -> buffer_handle_id (opaque to controller)
    layout_hash: str | None = None
    expires_at_ms: int  # hard expiry for replay safety
    capability_token: str  # opaque, unforgeable, instance-scoped capability
```

Note: `TargetSpec.layout_hash` is an **engine buffer layout/shape contract hash** (to prevent mis-writes). It is not the
same concept as `logical_layout_hash` (artifact ByteSpace identity) and MUST NOT be substituted for selection identity.

#### Capability model (required)

`TargetSpec` MUST be treated as a **capability**, not as an ambient identifier:

- `buffer_handle_id` values are not trusted on their own. The engine adapter MUST require and validate
  `capability_token` before resolving any target buffers.
- The capability MUST be instance-scoped (`instance_id`) and time-bounded (`expires_at_ms`) to prevent replay and
  cross-instance confusion.
- The capability MUST be unforgeable (high-entropy token and/or MAC/signature) and SHOULD bind to `layout_hash` and the
  declared tensor set to prevent silent mis-writes.

#### Engine adapter contract (minimal)

Each engine instance exposes:

- `mint_target(name, tensors, *, layout_hash=None, ttl_ms=None) -> TargetSpec`
- `resolve_targets(spec: TargetSpec) -> Mapping[str, torch.Tensor]`

Applications may layer domain-specific helpers (e.g., “mint targets for weights/KV”) on top of the base engine adapter,
but those helpers are not part of the implemented core surface. Planned signals and engine-integration extensions are
specified in `docs/designs/0056-programmable-framework-adv.md`.

This preserves the “process context boundary”: no cross-process raw tensor pointers.

### Transforms (Reshard / KV Layout)

TensorCast already supports **view transforms** (subset/narrow/transpose/etc.) as part of the artifact/view model.
However, some workflows require **compute transforms** that cannot be expressed as a pure view:

- **Reshard** (checkpoint/weights): all-to-all/concat/reshape across parallelism.
- **KV layout transform**: page/stride/schema changes that must execute compute.

Design rule: compute transforms are **control-plane programmable** but MUST execute at the **node-local safety
boundary** (Engine Adapter or node-local transform worker), never as a “central controller direct daemon RPC”.

Minimal typed contract:

```python
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class TransformSpec:
    name: str                      # e.g. "reshard.tp8_to_tp4.v1", "kvcache.layout_v1_to_v2.v1"
    args: Mapping[str, str | int]  # small, JSON-serializable, versioned
    layout_hash: str | None = None
```

Note: `TransformSpec.layout_hash` is a transform-specific **output layout contract hash** (used for validation/debug).
It is not selection identity and MUST NOT be substituted for `logical_layout_hash` / `selection_hash`.

Two common execution shapes:

1) **transform → register** (produces a new Artifact)

```python
class InstanceStepBuilder:
    def transform_register(
        self,
        src: "Artifact",
        *,
        spec: TransformSpec,
        out_key: str,
        policy: "StorePolicy | None" = None,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef["Artifact"]: ...
```

2) **transform → into** (fills engine-owned buffers; used by KV delta + layout)

```python
class InstanceStepBuilder:
    def transform_into(
        self,
        src: "Artifact",
        *,
        spec: TransformSpec,
        target: TargetSpec,
        depends_on: Sequence[PlanStepRef[Any]] | None = None,
    ) -> PlanStepRef[None]: ...
```

The Engine Adapter owns the plugin registry and rejects unknown `spec.name` or incompatible `layout_hash`.

Signals and engine integration extensions (including LLM KV-cache orchestration) are specified in `docs/designs/0056-programmable-framework-adv.md`.

---

## Feature Mapping (Legacy → Programmable)

This section maps today’s SDK surfaces (and common workflows) onto the programmable primitives introduced here.

| Legacy surface / workflow | Programmable primitive | Notes |
|---|---|---|
| `Artifact.prefetch()` | `Artifact.prefetch() -> Operation[PrefetchedReplica]` | Phase-0 is sync/blocking: use `op.result(...)` / `op.wait(...)`. |
| warm pool | `prefetch + pin_device_residency` | Daemon-owned cache warm (shared). |
| execute-while-load | `Artifact.view(...).tensor_dict_into(...)` / `DeferredLoader` | Caller-owned buffers; remote requires Engine Adapter. |
| persistence status polling | `Operation[PersistenceOutcome]` | Phase-0 wraps polling; Phase-1 adds `WaitPersistenceStatus`. |
| weights broadcast | `Plan` over workers calling prefetch/pin | Source selection remains Global Store-owned. |
| KV delta transfer | `Artifact.view(...)` (slice/subset) + `tensor_dict_into(...)` | Requires domain-specific tensor naming/layout conventions. |
| reshard tasks | `transform_register(...)` | Produces a new artifact + versioned key; orchestrated via `Plan`. |

## Contracts and Invariants

1. **Single-world artifact model**
   - Weights/KVCache/checkpoints are artifacts (tensor dicts).
   - All data-plane actions are performed via `Store`/`Artifact` primitives.

2. **Selection identity is explicit (required)**
   - Any joinable/retryable action MUST have a stable fingerprint based on `(artifact_id, logical_layout_hash, selection_hash)`.
   - Physical binding (addresses, region ids, buffer handles) MUST NOT be part of selection identity.
   - Non-identity views MUST resolve to a stable `view_id` and participate in `selection_hash`.

3. **ReplicaKey is the replica identity (engine join)**
   - Daemon-owned VRAM replicas are identified by `ReplicaKey` (artifact_id, view_id, device, tier, ...).
   - The daemon/StoreEngine MUST join repeated loads on the same `ReplicaKey` and must not allocate duplicate VRAM
     replicas.

4. **Deterministic operation id (`replica_uuid`) for joinable actions**
   - When `ctx.idempotency_key` is provided, the SDK MUST derive a deterministic operation id and send it as
     `replica_uuid` for joinable actions (prefetch/pin/waits).
   - Cancel/status are scoped to `replica_uuid` (operation id), not `ReplicaKey` identity.
   - Multiple operation ids may map to the same `ReplicaKey`; cancel MUST NOT affect other operation ids.
   - When `ctx` is absent, the SDK MUST still generate a per-call operation id once and reuse it across internal retries,
     but does not promise cross-call idempotency.

5. **Prefetch/pin are `NO_LEASE` (process-independent)**
   - Prefetch and residency pinning MUST NOT create PID-bound UseLeases by default.
   - PID-bound leases are reserved for IPC-handle export flows that actually return or depend on IPC handles.

6. **Capabilities are required for unsafe references**
   - `TargetSpec` MUST carry an unforgeable, instance-scoped capability token before an engine adapter will resolve
     buffers.
   - `PlacementPin` MUST carry an unforgeable, daemon-scoped capability token for renew/release.

7. **Stable target identity is config-derived**
   - `daemon_id` and `instance_id` MUST be derived from the unified runtime config (no ad-hoc env vars).
   - Controllers MUST NOT assume `worker_id` or `daemon_address` are stable across HA re-registration.

8. **Durability vs residency orthogonal**
   - `StorePolicy` does not express “GPU never evict”.
   - Residency is expressed by `pin_device_residency` (placement pin) or engine-owned memory.

9. **Key mapping immutable; alias is CAS**
   - Key mapping remains conflict-by-default (`FAILED_PRECONDITION` on overwrite).
   - Movable alias must be a separate service/table with CAS and linearizable read.

10. **Namespace as first-class concept (Phase-0: encoded)**
    - Phase-0: key prefixes encode namespace (e.g. `/<ns>/weights/...`).
    - All programmable loops (plans, caches, catalog) must treat namespace as a first-class scope.

---

## Error, Retry, Deadline, Cancel Semantics

### Error model

- All public methods raise `ArtifactError` (or subclasses) with canonical status codes.
- Errors must be classified into retryable/non-retryable categories, consistent with existing SDK.
- For operation workflows, `Operation.status()` MUST expose structured failure detail (`OperationStatus.error`) and `Operation.wait()` MUST raise a typed exception that preserves `status_code` + `retryable` (timeouts SHOULD use a dedicated subtype such as `OperationTimeoutError`).

### Deadline semantics (hard cutoff)

- If deadline expires before dispatch, operation fails with `DEADLINE_EXCEEDED` and no side effects.
- If deadline expires in flight, SDK triggers best-effort cancel and marks the operation `degraded`.
- The SDK MUST clamp any internal retry budgets (e.g., StoreOptions retry policy deadlines) to the remaining `ctx.deadline_ms` so per-call context remains the true end-to-end budget.
- `Operation.wait(timeout_s=...)` is a client-side wait budget; if it expires, `wait()` SHOULD raise a dedicated timeout exception and include `TimeoutErrorDetails(kind="wait_timeout", timeout_s=..., ...)` in status reporting.
- gRPC deadline is `min(remaining_budget, per_call_timeout)`.

### Cancel semantics

- Cancel is best-effort.
- Cancel is scoped to the **operation id** (`Operation.operation_id`, typically `replica_uuid` for daemon-owned actions).
  Cancelling an operation id MUST NOT cancel other operations, even if they map to the same underlying `ReplicaKey`.
- Cancellation MUST be implemented against operation-scoped daemon state (not `ReplicaKey` state) to avoid cross-talk.
- For in-flight loads, cancellation does not require canceling underlying IO; it only ensures the caller is no longer
  blocked and the operation record can be released/expired. The underlying load may still complete and remain cached.
- Implementations SHOULD avoid treating `cancelled` as a permanent “poison pill” for deterministic operation ids:
  subsequent retries with the same `CallContext.idempotency_key` SHOULD be able to re-issue/join the action when the
  underlying `ReplicaKey` is still compatible (e.g., by erasing the cancelled operation record or by allowing joins to
  proceed when background completion continues).

### Idempotency

If `ctx.idempotency_key` is provided:

- `action_id = hash(utf8(idempotency_key) + action_name + canonicalized_inputs)`
- canonicalized inputs MUST include selection identity (`artifact_id`, `logical_layout_hash`, `selection_hash`) and target identity, and
  MUST exclude physical bindings and deadline/qos/tags
- repeated submissions of deterministic-operation-id actions MUST return the same underlying operation semantics (no
  duplicate replicas, no operation record leakage)

Placement pins (implemented constraint):

- `CreatePlacementLease` is capability-based and **not idempotent** in 0055. If the caller times out or loses the
  response, retrying may create multiple pins; `ctx.idempotency_key` does not prevent this.
- Callers SHOULD always provide a finite `ttl_ms` and MUST avoid implicit retries on unknown outcomes. To extend
  residency, prefer `PlacementPin.renew(...)` once a token is available.

If no idempotency key is provided:

- semantics are at-least-once best-effort; callers must tolerate duplicates

---

## Proto Requirements

This design assumes proto can evolve. The following are required to make the user-facing contracts real (`NO_LEASE`,
operation-scoped wait/cancel, placement pins, and stable daemon identity).

### StoreDaemonService (additive)

1) **MaterializeReplica / MaterializeByKey**

- add `lease_mode` with `NO_LEASE` option (`prefetch` MUST default to `NO_LEASE`)
  - in `NO_LEASE`, the daemon must ignore PID use-lease creation even for loopback peers
  - in `NO_LEASE`, the daemon MUST NOT mint IPC handle leases and SHOULD omit `mem_handle` entirely
- clarify semantics of `replica_uuid` as an operation/session id:
  - multiple `replica_uuid`s may map to the same `ReplicaKey`
  - cancel/status are scoped to `replica_uuid` (operation id), not global replica identity
- (optional but recommended) include `replica_key_hash` fields for validation/debug so the daemon can detect accidental
  mismatches (same `replica_uuid` used for different `ReplicaKey`s)

2) **ReleaseReplica**

- define release semantics as operation-scoped (by `replica_uuid`)
- implement `QueryReplicaStatus` + `ReleaseReplica` (or return `UNIMPLEMENTED`) so SDK `Operation.wait/cancel` can be
  correct and observable

3) **WaitReplicaStatus (recommended)**

- avoid polling storms; required for low-latency operation waits
- MUST reflect the daemon-side operation record state machine (not just `ReplicaKey` state)
- SHOULD be server-streaming (or long-poll) to support low-latency waits without polling loops

4) **WaitPersistenceStatus (recommended)**

- avoid polling storms for persistence task waits (mirrors `WaitReplicaStatus`)

5) **Placement pin RPCs** (wire uses legacy “lease” naming)

- `CreatePlacementLease`, `RenewPlacementLease`, `ReleasePlacementLease`
- map to daemon’s existing `placement_pins` semantics (`SessionLifecycleManager::create_placement_lease`)
- MUST return and require a `lease_token` capability for renew/release (daemon-scoped, time-bounded, unforgeable). The Python SDK should surface this as a generic `capability_token` to avoid overloading “lease” terminology.

6) **Unified capability token envelope (implemented)**

Several subsystems require daemon-issued capability tokens (placement pins, retention handles, and future broker-issued
tokens like queue work leases). To prevent token-shape drift and token-type confusion, daemon-issued capabilities use a
single versioned envelope with issuer binding, audience binding, and expiry.

Authoritative proto: `proto/tensorcast/common/v1/capability_token.proto` (package `tensorcast.common.v1`).

Verification model (normative, v1):

- Tokens MUST be validated by the issuing daemon (issuer-only validation).
- Tokens MUST embed `issuer_daemon_id`, and the issuer MUST reject tokens whose issuer does not match `self.daemon_id`.
- SDKs SHOULD treat tokens as opaque `bytes` and never parse them.

Token requirements (normative):

- **Unforgeable**: authenticated (MAC/signature) with a key loaded from unified runtime config
  (`docs/designs/0004-unified-runtime-config.md`); no ad-hoc env vars.
- **Versioned**: tokens MUST carry a `token_version` (key id / format version) to support rotation.
- **Audience binding**: include an `audience` field to prevent token type confusion.
- **Scope binding**: bind tightly to an audience-specific scope payload. `scope` MUST be the deterministic Protobuf
  serialization of an audience-specific scope message so the issuer can parse/validate consistently across languages.
- **Expiry**: include absolute `expires_at_ms`; issuer MUST validate expiry.
- **Optional fencing**: when a token authorizes writes to volatile control-plane state, embed a fencing principal +
  epoch so stale tokens fail fast after leader changes.

Schema (excerpt; see proto for full details):

```proto
// Audiences prevent token type confusion.
enum CapabilityAudience {
  CAPABILITY_AUDIENCE_UNSPECIFIED = 0;
  CAPABILITY_AUDIENCE_PLACEMENT_LEASE = 1;
  CAPABILITY_AUDIENCE_RETENTION_HANDLE = 2;
  CAPABILITY_AUDIENCE_QUEUE_WORK_LEASE = 3;
}

message CapabilityTokenEnvelope {
  uint32 token_version = 1;        // key id / format version
  string issuer_daemon_id = 2;     // stable identity (not address)
  CapabilityAudience audience = 3; // prevents token type confusion
  bytes scope = 4;                // deterministic audience-specific payload
  uint64 expires_at_ms = 5;        // absolute expiry

  oneof fencing {
    QueueEpochFencing queue_epoch = 10;
  }

  bytes auth_tag = 100;           // MAC/signature over fields above
}

message QueueEpochFencing {
  string queue_operation_id = 1;  // canonical queue principal; see `docs/designs/0060-tensor-work-queue.md`
  uint64 queue_epoch = 2;         // Global Store operation lease_generation
}
```

Key rotation (required):

- Issuers MUST support bounded-overlap rotation: accept the active key and a bounded set of previous keys for
  verification; `token_version` selects which key to use.
- Keys are configured in `DaemonConfig.capability_tokens` (active + bounded previous).

Migration/compatibility (required):

- Placement lease tokens support dual-format verification during migration (legacy token format vs the envelope), and
  issuers SHOULD keep this window bounded. The token format MUST be unambiguously detectable by the issuer so clients
  can roll independently.
- Retention handles require the unified envelope (issuer must be configured with capability token keys); they do not
  have a legacy token format.

### Node Agent / Engine Adapter RPCs (Phase-4; implemented)

Node-local execution is provided by:

- `proto/tensorcast/node_agent/v1/node_agent.proto` (Plan execution RPCs)
- `tensorcast/node_agent/executor.py` + `tensorcast/node_agent/server.py`
- `tensorcast/engine_adapter` (transform registry + target capability enforcement)

### PlanSpec (Phase-4; implemented)

`PlanSpec` is defined in `proto/tensorcast/plan/v1/plan.proto` and embeds
`tensorcast.common.v1.ArtifactSelection` from `proto/tensorcast/common/v1/common.proto`. It includes:

- `CallContext` metadata (deadline, idempotency, tags)
- `ArtifactSelection` identity (`artifact_id`, `logical_layout_hash`, `selection_hash`) plus optional `tensor_names` for subset execution
- Worker/instance targets with ordered dependencies
- `TransformSpec` and `TargetSpec` for instance-scoped transforms

- Engine adapters MUST expose a target-minting surface that returns a `TargetSpec` with a capability token
  (instance-scoped, time-bounded). Controllers must not mint targets directly.

### GlobalStoreService (recommended)

- Worker registration/heartbeat requires stable `daemon_id` (derived from daemon config, not from address);
  address/port are routing attributes and endpoint conflicts are rejected regardless of `node_id`.
- `ListActiveWorkers` should return `daemon_id` alongside `worker_id` so controllers/node agents can reconcile state
  across HA re-registrations.
- Instance registry exposes `RegisterInstance`, `InstanceHeartbeat`, and `ListActiveInstances` for stable
  `instance_id` discovery.
- Capability directory (low-frequency): worker/instance discovery includes a bounded capability bitset (routing hints
  only, not correctness) in `proto/tensorcast/global_store/v1/global_store.proto`, for example:
  - workers: `WORKER_CAPABILITY_FLAG_QUEUE_BROKER_ENABLED`, `WORKER_CAPABILITY_FLAG_RETENTION_HANDLES_ENABLED`,
    `WORKER_CAPABILITY_FLAG_CAPABILITY_TOKENS_V2_ENABLED`
  - instances: `INSTANCE_CAPABILITY_FLAG_EXECUTION_SIGNALS_ENABLED`, `INSTANCE_CAPABILITY_FLAG_NODE_AGENT_ENABLED`
  - listing supports filters (`required_capability_flags`) so clients/controllers can find broker-enabled endpoints
  - daemons publish worker capability flags when `DaemonConfig.capability_directory.enabled=true`
  - writes SHOULD be update-on-change (do not rewrite identical flags every heartbeat)
  - clients MUST cache results and MUST NOT query Global Store per control-plane action; refresh is bounded by explicit
    staleness budgets and backoff

---

## Schema Changes

Stable target identity requires Global Store persistence beyond ephemeral `worker_id` / address tuples:

- Add `daemon_id` to the Global Store worker registry (`schema.sql`: `workers` table), **NOT NULL** with a uniqueness
  constraint (stable identity; address/port remain unique but are not identity).
- Plumb `daemon_id` through `RegisterWorker`, heartbeat, and `ListActiveWorkers`, so controllers/node agents can
  reconcile worker restarts and address changes without guessing.

---

## Trade-offs & Risks

- **Operation-scoped wait/cancel requires daemon work**: tracking operation ids independently from `ReplicaKey` adds
  state; mitigate with bounded retention (TTL) + explicit `ReleaseReplica(replica_uuid)` + server-streaming
  `WaitReplicaStatus`.
- **`NO_LEASE` default changes expectations**: warm pools stop being per-process by default; mitigate by allowing
  explicit PID-bound lease mode for IPC-handle export flows only.
- **Placement pins are best-effort**: daemon restart loses pins; mitigate by treating pinning as rebuildable intent
  and making controllers tolerant (reconcile + re-pin).
- **Alias and KV catalog scale/availability**: linearizable alias reads and large KV catalogs can hurt availability;
  mitigate with explicit fallback policy (fixed-version keys) + admission control + bounded candidate LPM.
- **Transform plugins risk semantic drift**: mitigate with versioned `TransformSpec.name` + `layout_hash` gating and
  fail-fast on mismatch.

---

## Examples

### Example 1: Warm a daemon-owned replica

```python
import tensorcast

tensorcast.init(mode="connect", address="127.0.0.1:50051")

ctx = tensorcast.context(request_id="warm:001", qos="background", deadline_ms=120_000)
art = tensorcast.artifact(key="/prod/weights/llama-70b/v7", fallback="p2p")

# Warm pool (daemon-owned replica on the requested device).
art.prefetch(device="cuda:0", ctx=ctx).result(timeout_s=60.0)
pin = art.pin_device_residency(device="cuda:0", ttl_ms=300_000, ctx=ctx).result(timeout_s=5.0)
```

### Example 2: Load a subset view into preallocated buffers

```python
import tensorcast

tensorcast.init(mode="connect", address="127.0.0.1:50051")

ctx = tensorcast.context(request_id="subset:001", qos="interactive", deadline_ms=10_000)
art = tensorcast.artifact(key="/prod/weights/llama-70b/v7", fallback="p2p")

hot = art.view(
    names=[
        "layers.0.attn.q_proj.weight",
        "layers.0.attn.k_proj.weight",
    ]
)

hot.tensor_dict_into(target=param_tensors, device="cuda:0", ctx=ctx)
```

### Example 3: Orchestrate a warm pool with Plan

```python
import tensorcast

tensorcast.init(mode="connect", address="127.0.0.1:50051")

cap = tensorcast.CapabilityDirectoryClient(
    tensorcast.CapabilityDirectoryOptions(target="127.0.0.1:50051")
)
worker_infos = cap.list_workers(include_unavailable=False)
workers = [
    tensorcast.Worker(
        worker_id=w.worker_id,
        daemon_address=f"{w.node_address}:{w.grpc_port}",
        daemon_id=w.daemon_id,
        p2p_port=int(w.p2p_port) if w.p2p_port else None,
    )
    for w in worker_infos
]

ctx = tensorcast.context(request_id="warm-pool:v7", qos="background", deadline_ms=120_000)
plan = tensorcast.plan(ctx)

art = tensorcast.artifact(key="/prod/weights/llama-70b/v7", fallback="p2p")

for w in workers:
    steps = plan.on_worker(w)
    prefetch_step = steps.prefetch(art, device=0)
    steps.pin_device_residency(art, device=0, ttl_ms=300_000, depends_on=[prefetch_step])

try:
    plan.run(concurrency=16)
except tensorcast.PlanFailedError as exc:
    # Result still contains completed step statuses for debugging.
    result = exc.result
    raise
```

---

## Compatibility & Migration

- Existing artifact retrieval calls (`tensor_dict`, `tensor_dict_into`, `DeferredLoader`) continue to work.
- This design intentionally introduces **meaningful breaking changes** where needed to make programmability correct:
  - `Artifact.prefetch(...)` returns `Operation[PrefetchedReplica]` (instead of a `(handle, ticket)` tuple).
  - `Artifact.prefetch(...)` defaults to `NO_LEASE` so it does not create PID-bound UseLeases.
  - When `CallContext.idempotency_key` is provided, `prefetch()` derives a deterministic operation id (`replica_uuid`) so
    retries are idempotent and cancellation is safe.
- `pin_device_residency()` is new; deployments without placement pin RPCs SHOULD fail with `FAILED_PRECONDITION` plus a clear
  remediation message.

---

## Code Map

Suggested code locations:

- Context + operation protocols:
  - `tensorcast/api/context.py` (`CallContext`, `tensorcast.context`)
  - `tensorcast/api/operation.py` (`Operation`, `OperationStatus`)
- Artifact handle extensions:
  - `tensorcast/api/store/artifact.py` (ctx-aware `prefetch`/`pin_device_residency`, ctx plumbing for into)
- Control-plane clients (ground truth today):
  - `tensorcast/daemon_ctl.py` (daemon ticket/status RPCs)
  - `tensorcast/dashboard/gs_client.py` (Global Store client; can be extracted into a shared module)
- Plan orchestration:
  - `tensorcast/api/plan/__init__.py`
  - `tensorcast/api/plan/plan.py` (IR + execution)
  - `tensorcast/api/plan/README.md`
  - `proto/tensorcast/plan/v1/plan.proto`
  - `proto/tensorcast/node_agent/v1/node_agent.proto`
  - `tensorcast/node_agent/executor.py`
  - `tensorcast/node_agent/server.py`
  - `tensorcast/node_agent/__main__.py`
  - `proto/tensorcast/config/v1/node_agent_config.proto`
  - `examples/config/node_agent_config.yaml`
  - `tensorcast/api/plan/targets.py` (`TargetSpec`)
- Transforms:
  - `tensorcast/api/plan/transforms.py` (`TransformSpec`)
  - `tensorcast/engine_adapter/adapter.py` (transform registry + target resolution)
- Instance registry:
  - `proto/tensorcast/global_store/v1/global_store.proto`
  - `tensorcast/global_store/services/instance_service.py`
  - `tensorcast/global_store/repositories/instance_repository.py`
  - `schema.sql` (instances table)
- Capability directory (worker/instance discovery):
  - `tensorcast/capability_directory.py` (`CapabilityDirectoryClient`)

---

## Naming Compliance

This design’s proposed public interfaces follow repository naming conventions:

- Python classes use `PascalCase`: `CallContext`, `Operation`, `OperationStatus`, `OperationError`, `TimeoutErrorDetails`,
  `PrefetchedReplica`, `PlacementPin`, `TargetSpec`, `TransformSpec`, `Plan`, `PlanResult`, `PlanStepResult`,
  `PlanStepRef`, `PlanFailedError`, `Worker`, `Instance`.
- Python functions/methods use `snake_case`: `tensorcast.context`, `tensorcast.plan`, `Artifact.prefetch`, `Artifact.tensor_dict_into`,
  `Artifact.pin_device_residency`, `Operation.status`, `Operation.wait`, `Operation.result`, `Operation.cancel`, `Plan.to_spec`,
  `Plan.run`.
- Plan step builders use `snake_case`: `WorkerStepBuilder.prefetch`, `WorkerStepBuilder.pin_device_residency`,
  `WorkerStepBuilder.unpin_device_residency`, `InstanceStepBuilder.transform_register`, `InstanceStepBuilder.transform_into`.
- Protobuf RPC names remain `PascalCase` (`MaterializeReplica`, `QueryReplicaStatus`, `ReleaseReplica`); new fields use
  `snake_case` (`lease_mode`, `replica_uuid`, `lease_token`).

---

## Appendix: Design Rationale (Why this shape?)

- Handles over ad-hoc getters: matches the existing SDK principle and keeps room for evolving internal strategies.
- No parallel “programmability object model”: avoids fracturing users into two APIs.
- Plan as IR: delivers programmability without building a distributed Python runtime.
- Deterministic operation id (`replica_uuid`): makes retries and cancel semantics correct without building a second
  waiter model; VRAM dedup remains `ReplicaKey`-based.
- Pinning as first-class: makes warm pools and serverless viable without abusing PID leases.
