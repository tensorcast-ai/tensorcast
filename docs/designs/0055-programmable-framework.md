---
slug: 0055-programmable-framework
title: Programmable API Design (Artifact-First) (Design)
status: draft
areas: ["sdk", "daemon", "global_store", "proto"]
related_code:
  - docs/architecture/api/api-design.md
  - docs/architecture/api/materialization-flow.md
  - docs/architecture/api/policy-persistence.md
  - docs/architecture/api/region-backed.md
  - docs/architecture/p2p-transfer-strategies.md
  - docs/internals/model-loading.md
  - docs/designs/0011-unified-session-lifecycle-leases.md
  - docs/designs/0039-artifact-first-sdk.md
  - daemon/README.md
  - daemon/service/grpc_service_impl.{h,cc}
  - daemon/state/replica_session_manager.h
  - daemon/state/session_lifecycle.{h,cc}
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/global_store/v1/global_store.proto
  - tensorcast/api/_config.py
  - tensorcast/api/store/README.md
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/async_ops.py
  - tensorcast/api/store/batch_context.py
  - tensorcast/daemon_ctl.py
  - tensorcast/__init__.py
links:
  predecessors:
    - ./0039-artifact-first-sdk.md
    - ./0011-unified-session-lifecycle-leases.md
---

# Programmable API Design (Artifact-First)

This document proposes a **programmable** extension to the existing public Python SDK surface. The key design choice is:

> **Weights, KVCache, checkpoints, etc. are all Artifacts (tensor dicts).**  
> **Programmability must not introduce a parallel “data object” abstraction.**

Instead, programmability is expressed by **composing existing Artifact primitives** (prefetch, into, register/put, persistence, view/subset) with a small set of new control-plane primitives:

- `CallContext`: QoS + deadline + idempotency + tracing tags (per request/task).
- `Operation[T]`: unified async wait/cancel/status (prefetch/persistence/pin).
- `Plan`: a composable orchestration IR that can dispatch actions to node-local agents (without “shipping arbitrary Python” like Ray).

This document prioritizes **What/Why** (semantics and rationale) and defines contracts/invariants that must hold from Phase-0.

---

## Current State (as of 2026-01)

This design extends the existing artifact-first SDK and daemon APIs:

- Prefetch exists today as `Artifact.prefetch(device=...) -> (prefetched_handle, PrefetchTicket)` (see `tensorcast/api/store/artifact.py`). `PrefetchTicket.wait()` polls `QueryReplicaStatus` and `PrefetchTicket.cancel()` calls `ReleaseReplica` via `tensorcast/daemon_ctl.py`; however, `StoreDaemonServiceImpl` does not currently override these RPCs (`daemon/service/grpc_service_impl.h`), so wait/cancel are best-effort today (may return `False` / no-op).
- The daemon already tracks `replica_uuid` sessions via `ReplicaSessionManager` (`daemon/state/replica_session_manager.h`) and consults lifecycle counters (`use_count`, `placement_pins`) for eviction (see `daemon/README.md`).
- The Global Store tracks workers and replicas (by `worker_id`), but there is no first-class “engine instance” registry surface today (`proto/tensorcast/global_store/v1/global_store.proto`).

## Navigation

- [Current State](#current-state-as-of-2026-01)
- [Goals](#goals)
- [Non-Goals](#non-goals)
- [Core Concepts](#core-concepts)
- [Entry Points](#entry-points)
- [Public API Surface](#public-api-surface)
  - [CallContext](#callcontext)
  - [Operation](#operation)
  - [Artifact Enhancements](#artifact-enhancements)
  - [Residency Pinning (Placement Lease)](#residency-pinning-placement-lease)
  - [Plan (Programmable Orchestration)](#plan-programmable-orchestration)
  - [Targets & Engine Adapter](#targets--engine-adapter)
  - [Signals (Control-Loop Inputs)](#signals-control-loop-inputs)
  - [Domain Helpers: weights / kvcache](#domain-helpers-weights--kvcache)
- [Contracts and Invariants](#contracts-and-invariants)
- [Error, Retry, Deadline, Cancel Semantics](#error-retry-deadline-cancel-semantics)
- [Proto Requirements](#proto-requirements)
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
     - `Artifact.pin_residency(...)` (new)
     - `CallContext` (new) and `Plan` (new, advanced)
   - Weights/KVCache are expressed as artifacts with **key conventions + view helpers**.

3. **Unified async semantics**
   - Prefetch, persistence, and residency pinning expose a unified `Operation[T]` interface (without breaking existing `PrefetchTicket` / `ArtifactFuture` surfaces), with `wait/cancel/status` and `await op` support.

4. **At-least-once orchestration with safe idempotency**
   - When a `CallContext.idempotency_key` (or `request_id`) is provided, repeated submissions must not cause duplicate daemon-owned replicas or leaked waiters.

5. **Deterministic `replica_uuid` (no EnsureReplica RPC)**
   - Avoid duplicate daemon-owned VRAM replicas by joining on a deterministic `replica_uuid` derived from `(worker_id, artifact_id, view_id, tier, device)` (where `worker_id` is the daemon identity tracked by the Global Store).

6. **Durability vs residency are orthogonal**
   - `StorePolicy` describes durable tiers (stable_dram/shared_disk), not “GPU never evict”.
   - Residency is expressed via placement leases (pinning) and/or engine-owned memory.

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

**Programmability requirement**: any view that changes bytes must be included in the cache identity (`view_id`) and deterministic `replica_uuid` derivation.

### Replica (existing)

A concrete materialization of an artifact (or view) on a tier/device.

### Two ownership modes (must be explicit)

1. **Daemon-owned replica (cache warm / shared cache)**
   - Created by `Artifact.prefetch(...)` (daemon materializes and owns VRAM/DRAM).
   - Optional: kept resident by `pin_residency(...)` (placement lease).

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

### Operation[T] (new unified async)

A unified interface for async workflows:

- prefetch wait
- persistence task wait
- residency pin lifecycle (optional async)
- status/cancel unify across subsystems

### Plan (new orchestration IR)

A composable plan expressing:

- which node/instance executes which action
- dependencies and concurrency (Phase-0: no plan-level reschedule retries; future phases may add bounded retries)
- derived idempotency via context + canonical inputs

Plan is an **IR**, not “send Python to remote”.

### TargetSpec (new engine integration abstraction)

For remote `into(...)` operations, targets cannot be raw `torch.Tensor` across machines. A `TargetSpec` is a serializable reference resolved by the node’s engine adapter to actual buffers.

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
    request_id="req-123",
    qos="realtime",
    deadline_ms=50,
    tags={"tenant": "prod"},
)

art = ctx.artifact(key="/prod/weights/llama/v7", fallback="p2p")

# Issue (non-blocking) + sync wait:
prefetched, ticket = art.prefetch(device="cuda:0", ctx=ctx)
ticket.wait(timeout=60.0)

# Or issue + async wait (awaitable result):
prefetched, ticket = await art.prefetch(device="cuda:0", ctx=ctx)
```

---

## Public API Surface

### CallContext

`CallContext` binds per-call semantics without changing the process-wide `StoreOptions`.

```python
from dataclasses import dataclass
from typing import Literal, Mapping

QosClass = Literal["realtime", "interactive", "background"]

@dataclass(frozen=True, slots=True)
class CallContext:
    request_id: str
    qos: QosClass = "interactive"
    deadline_ms: int | None = None
    idempotency_key: str | None = None
    tags: Mapping[str, str] | None = None

    def artifact(self, *, artifact_id=None, key=None, disk_path=None, fallback=None):
        """Build an Artifact handle bound to this context (identity unchanged)."""
```

**Notes**

- `request_id` is required. If the caller already has a stable idempotency key, it should be set as `idempotency_key`; otherwise `request_id` serves as the default.
- `deadline_ms` is a hard cutoff (see Deadline semantics).

**Module-level helper**

```python
def context(
    *,
    request_id: str,
    qos: QosClass = "interactive",
    deadline_ms: int | None = None,
    idempotency_key: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> CallContext: ...
```

### Operation

Unified async interface. New programmable control-plane actions should return an `Operation[T]`; existing long-tail surfaces (`PrefetchTicket`, `ArtifactFuture`, persistence task ids) should be adapted to this interface for consistency.

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
    - `context` is a small, safe-to-log dictionary (e.g. request_id, worker_id, artifact_id, replica_uuid).
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
    async def status(self) -> OperationStatus: ...
    async def wait(self, *, timeout_s: float | None = None) -> T: ...
    async def cancel(self) -> bool: ...
    def __await__(self): ...
```

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
from typing import NamedTuple

class PrefetchResult(NamedTuple):
    """
    Tuple-like return value for `Artifact.prefetch(...)`.

    Compatibility:
    - Supports tuple unpacking: `prefetched, ticket = art.prefetch(...)`.
    - Is awaitable: `prefetched, ticket = await art.prefetch(...)`.
    """

    artifact: "Artifact"
    ticket: "PrefetchTicket"

    async def wait(self, *, timeout_s: float | None = None) -> "PrefetchResult": ...
    def __await__(self): ...

class Artifact:
    def prefetch(
        self,
        *,
        device: torch.device | str,
        ctx: CallContext | None = None,
        options: "GetArtifactOptions | None" = None,
    ) -> PrefetchResult:
        """
        Ensure a daemon-owned replica exists (and optionally begins loading immediately).

        Returns:
        - `PrefetchResult(artifact=<prefetched_handle>, ticket=<PrefetchTicket>)`
        - `prefetched_handle`: a clone whose fallback carries the issued ticket's `replica_uuid`
          (so subsequent reads can hint reuse without mutating the original handle).
        - `PrefetchTicket`: a helper to `wait()` / `cancel()` the staged replica.
          It is backed by `ReplicaTicket` + `QueryReplicaStatus` / `ReleaseReplica` RPCs.

        Semantics:
        - Daemon-owned residency (VRAM/DRAM is owned by the daemon).
        - Uses deterministic `replica_uuid` + join when `ctx` has an `idempotency_key`/`request_id`.
        - Prefetch does not require creating per-process UseLeases (no mapping).
        - Cache identity includes `(artifact_id, view_id, tier, device)`.

        Await semantics (recommended in async code):
        - `await art.prefetch(...)` waits for the ticket to become ready and then returns the same tuple-like
          `PrefetchResult`. On timeout/failure it SHOULD raise a typed exception consistent with the
          `OperationError`/timeout model.
        - `ticket.wait(timeout=...)` remains a synchronous, best-effort boolean helper for compatibility.
        """
```

**Deterministic `replica_uuid` (default when `ctx` provided)**

- Derived from a canonical tuple: `(worker_id, artifact_id, view_id, tier, device_id, algo_version)`.
- Uses UUIDv5 (or explicit algo enum).
- Ensures “same replica key on same worker” joins instead of creating multiple daemon-owned replicas.

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
        - When region-backed is enabled, uses MaterializeIntoTarget (loopback/UDS only).
        - Does not create daemon-owned VRAM replica.
        - For remote execution, `target` must be a TargetSpec (serializable reference).
        """
```

For advanced usage, deferred loader remains the recommended path.

```python
with art.deferred_loader(device="cuda:0") as loader:
    q = loader.tensor("layers.0.attn.q_proj.weight")
    ...
    loader.commit()
```

### Residency Pinning (Placement Lease)

This is the missing “GPU residency intent” primitive, orthogonal to `StorePolicy`.

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PlacementLease:
    lease_id: int
    artifact_id: str
    view_id: str
    device_id: int
    expires_at_ms: int | None

    async def renew(self, ttl_ms: int, ctx: CallContext | None = None) -> None: ...
    async def release(self, ctx: CallContext | None = None) -> None: ...

class Artifact:
    async def pin_residency(
        self,
        *,
        device: str | int,
        ttl_ms: int | None,
        ctx: CallContext | None = None,
    ) -> PlacementLease:
        """
        Keep a daemon-owned replica resident via placement_pins.

        Semantics:
        - Does not create PID UseLease.
        - Best-effort: daemon restart loses leases; callers must tolerate and rebuild.
        - Idempotent per (ReplicaKey, owner_id/request_id) depending on implementation.
        """
```

### Plan (Programmable Orchestration)

Plan is a programmable orchestration IR that composes artifact actions across nodes.

Naming note: `Plan` here is orchestration IR, distinct from the existing `PlanType` enum used for registration plans (`tensorcast/api/_config.py`).

#### Why Plan (vs `submit_to(fn=...)`)

- Avoids “distributed Python execution” complexity.
- Supports at-least-once semantics with idempotent action submission (Phase-0 retries are user-driven).
- Central/controller can build a plan; node-local agent executes actions safely.

#### Node identity: Worker vs Instance

For correctness, Plan distinguishes:

- **Worker** (daemon identity; cache/prefetch/pin run here)
- **Instance** (engine process identity; into/targets resolved here). Phase-0 note: TensorCast does not currently expose an instance registry in the control plane; `Instance` steps are future work.

#### Minimal types

```python
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True, slots=True)
class Worker:
    worker_id: str
    daemon_address: str
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

#### Plan API

```python
class Plan:
    def __init__(self, ctx: CallContext): ...

    def on_worker(self, worker: Worker) -> "WorkerStepBuilder": ...
    def on_instance(self, inst: Instance) -> "InstanceStepBuilder": ...

    async def run(self, *, concurrency: int = 16) -> "PlanResult": ...
```

#### PlanResult (Phase-0)

`Plan.run()` returns a `PlanResult` summarizing per-step outcomes. Phase-0 treats the plan as failed if any step fails; the result still includes all completed step statuses for debugging.

```python
from dataclasses import dataclass
from typing import Any, Sequence

@dataclass(frozen=True, slots=True)
class PlanStepResult:
    step_id: str
    target_id: str  # worker_id or instance_id
    action: str
    status: OperationStatus
    value: Any | None = None

@dataclass(frozen=True, slots=True)
class PlanResult:
    ok: bool
    request_id: str
    steps: Sequence[PlanStepResult]
```

#### Worker steps (daemon-owned)

```python
class WorkerStepBuilder:
    def prefetch(self, art: "Artifact", *, device: str | int) -> Operation["PrefetchResult"]: ...
    def pin_residency(self, art: "Artifact", *, device: str | int, ttl_ms: int | None) -> Operation["PlacementLease"]: ...
    def unpin_residency(self, lease: "PlacementLease") -> Operation[None]: ...
```

#### Instance steps (engine-owned)

```python
class InstanceStepBuilder:
    def into(self, art: "Artifact", *, target: "TargetSpec", device: str | None = None) -> Operation[None]: ...
```

#### Execution

`Plan.run()` enforces:

- bounded concurrency
- deadline propagation
- action-level idempotency derived from `(ctx.idempotency_key/request_id, action_name, canonical inputs)`
- atomic success reporting: `PlanResult.ok` is `True` only if all steps succeed

#### Failure, retry, and partial success (Phase-0)

Phase-0 adopts a deliberately simple contract:

- **Atomic reporting (not rollback)**: any single-step failure causes the overall plan result to be marked failed (`PlanResult.ok=False`).
- **No rollback**: Phase-0 does not attempt to “undo” side effects that already happened (e.g., a replica that was prefetched before a later step failed may still exist).
- **Best-effort cancellation**: once a failure is observed, remaining not-yet-started steps are skipped and in-flight steps receive a best-effort cancel; some steps may still succeed concurrently.
- **No plan-level reschedule retries**: Phase-0 does not automatically reschedule failed steps; callers can retry by re-running the whole plan with the same `CallContext.idempotency_key`/`request_id` and by using `OperationStatus.error.retryable` + `OperationStatus.error.status_code` to decide what to retry vs discard.

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

    name: str  # e.g. "weights.params", "kv.req:123"
    tensors: Mapping[str, str]  # tensor_name -> buffer_handle_id (opaque to controller)
    layout_hash: str | None = None
```

#### Engine adapter contract (minimal)

Each engine instance exposes:

- `resolve_targets(spec: TargetSpec) -> Mapping[str, torch.Tensor]`
- optionally: `kv_targets(request_id) -> TargetSpec`, `weights_targets(model_id) -> TargetSpec`
- execution signals endpoint (queue, inflight, etc.)

This preserves the “process context boundary”: no cross-process raw tensor pointers.

### Signals (Control-Loop Inputs)

Signals are low-cardinality, cacheable inputs for policies. TensorCast distinguishes two different signal families:

- **TensorCastSignals**: TensorCast-owned signals sourced from the Global Store + Store Daemons (health, memory pressure, replica residency, etc.). These are authoritative for storage state.
- **ExecutionSignals**: engine-owned signals sourced from external execution frameworks (LLM inference engines, training runtimes, schedulers). TensorCast does not collect these itself; they are provided via a pluggable adapter.

#### Snapshot semantics

All signals MUST include:

- `as_of_ms` (timestamp of observation)
- `staleness_ms` (age at read time)

Default staleness budgets by QoS:

- `realtime`: ≤ 250ms
- `interactive`: ≤ 1s
- `background`: ≤ 5–10s

Recommended representation:

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

@dataclass(frozen=True, slots=True)
class SignalSnapshot(Generic[T]):
    value: T
    as_of_ms: int
    staleness_ms: int
```

Policies MUST enforce staleness budgets: if a snapshot’s `staleness_ms` exceeds the budget for the current QoS, the signal is treated as unavailable (ignored), triggering policy fallback.

#### TensorCastSignals (daemon/GS)

Semantics:

- **Source**: TensorCast control/data plane RPCs (Global Store + Store Daemon).
- **Authority**: canonical for TensorCast-owned state (worker availability, loaded replicas, store capacity).
- **Intended use**: correctness-sensitive placement decisions and safe optimization (e.g., “pick a healthy worker with enough free mem”).

```python
class TensorCastSignals:
    async def list_workers(
        self, *, include_unavailable: bool = False
    ) -> SignalSnapshot[list[Worker]]: ...

    async def get_worker_status(self, worker: Worker) -> SignalSnapshot["GetWorkerStatusResponse"]: ...

    async def get_loaded_replicas(
        self,
        worker: Worker,
        *,
        artifact_id: str | None = None,
        device_id: int | None = None,
    ) -> SignalSnapshot["GetLoadedReplicasV2Response"]: ...

    async def list_instances(
        self, *, labels=None
    ) -> SignalSnapshot[list[Instance]]: ...  # future
```

#### ExecutionSignals (engine-owned via adapter)

Semantics:

- **Source**: third-party execution engines (LLM inference/training frameworks). These signals are *not* available from TensorCast itself.
- **Authority**: advisory only. They must not be used for correctness-critical decisions (durability, identity, leases); only for performance scheduling.
- **Extensibility**: provided by `ExecutionSignalsAdapter` implementations that can be shipped independently (e.g., `tensorcast-vllm`, `tensorcast-torch`, `tensorcast-ray`).
- **Selection**: adapters are selected by matching `Instance.engine` to `ExecutionSignalsAdapter.engine`. If no adapter matches (or the engine has no signals endpoint), `ExecutionSignals.profile(...)` returns `None` and `batch_profile(...)` omits that instance.

```python
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

@dataclass(frozen=True, slots=True)
class InstanceProfile:
    engine: str
    queue_time_ms_p50: float | None = None
    queue_time_ms_p90: float | None = None
    inflight_total: int | None = None
    inflight_by_stage: Mapping[str, int] | None = None
    metrics: Mapping[str, float] | None = None  # bounded, engine-documented keys

class ExecutionSignalsAdapter(Protocol):
    """Pluggable provider for engine execution profiles (not TensorCast-owned)."""

    engine: str  # matches Instance.engine

    async def profile(
        self,
        inst: Instance,
        *,
        ctx: CallContext | None = None,
    ) -> SignalSnapshot[InstanceProfile] | None: ...

    async def batch_profile(
        self,
        insts: Sequence[Instance],
        *,
        ctx: CallContext | None = None,
    ) -> dict[Instance, SignalSnapshot[InstanceProfile]]: ...

class ExecutionSignals:
    def __init__(self, adapters: Sequence[ExecutionSignalsAdapter]): ...

    async def profile(
        self,
        inst: Instance,
        *,
        ctx: CallContext | None = None,
    ) -> SignalSnapshot[InstanceProfile] | None: ...

    async def batch_profile(
        self,
        insts: Sequence[Instance],
        *,
        ctx: CallContext | None = None,
    ) -> dict[Instance, SignalSnapshot[InstanceProfile]]: ...
```

### Domain Helpers: weights / kvcache

#### Principle

Domain helpers MUST:

- only define key conventions and view helpers
- return `Artifact` handles or `TargetSpec`s
- avoid creating a parallel data-plane API

Public surface:

- `from tensorcast import weights`
- `from tensorcast import kvcache`

Implementation lives in `tensorcast/domain/weights.py` and `tensorcast/domain/kvcache.py`, but these helpers are re-exported from the top-level `tensorcast` package for discoverability.

#### weights

```python
# tensorcast/domain/weights.py (new module)
def key(
    *,
    namespace: str,
    model_id: str,
    version: str,
    dtype: str,
    layout: str,
) -> str: ...

def layers(art: "Artifact", *, first_n: int) -> "Artifact":
    """Return a subset view (tensor_names) for execute-while-load."""
```

Usage:

```python
import tensorcast
from tensorcast import weights

ctx = tensorcast.context(request_id="warm", qos="background", deadline_ms=60_000)
art = ctx.artifact(
    key=weights.key(
        namespace="prod",
        model_id="llama-70b",
        version="v7",
        dtype="bf16",
        layout="tp8pp1",
    )
)

# warm pool (daemon-owned)
prefetched, ticket = art.prefetch(device="cuda:0", ctx=ctx)
ticket.wait(timeout=60.0)
await prefetched.pin_residency(device="cuda:0", ttl_ms=300_000, ctx=ctx)

# execute-while-load (engine-owned)
hot = weights.layers(art, first_n=8)
hot.tensor_dict_into(target=params_tensors, device="cuda:0", ctx=ctx)
```

#### kvcache

```python
# tensorcast/domain/kvcache.py (new module)
def key_to_str(key: "KvCacheKey") -> str: ...

def delta(
    art: "Artifact",
    *,
    tensor_names: list[str],
    seq_dim: int,
    start: int,
    length: int,
) -> "Artifact":
    """Return a delta view via slices/subset suitable for into()."""
```

Control plane (optional) returns keys/handles, not new objects:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class KvLookupHit:
    key: "KvCacheKey" | None
    hit_len: int
    locations: list[Worker]  # sources, optional
    est_fetch_ms_p90: float | None
```

---

## Contracts and Invariants

1. **Single-world artifact model**
   - Weights/KVCache/checkpoints are artifacts (tensor dicts).
   - All data-plane actions are performed via `Store`/`Artifact` primitives.

2. **Deterministic `replica_uuid` + join (no EnsureReplica)**
   - When `ctx.request_id` (or `ctx.idempotency_key`) is present, prefetch MUST use deterministic `replica_uuid` derived from `(worker_id, artifact_id, view_id, tier, device)` and algo version.
   - Daemon MUST join on identical `replica_uuid` instead of allocating multiple daemon-owned replicas.

3. **Waiter idempotency: `request_id` is mandatory in proto for joinable ops**
   - Materialize/prefetch requests must include `request_id` (waiter id).
   - `ReleaseReplica` must release by `(artifact_id, replica_uuid, request_id)` to avoid cancel cross-talk.

4. **Durability vs residency orthogonal**
   - `StorePolicy` does not express “GPU never evict”.
   - Residency is expressed by `pin_residency` (placement lease) or engine-owned memory.

5. **Key mapping immutable; alias is CAS**
   - Key mapping remains conflict-by-default (`FAILED_PRECONDITION` on overwrite).
   - Movable alias must be a separate service/table with CAS and linearizable read.

6. **Namespace as first-class concept (Phase-0: encoded)**
   - Phase-0: key prefixes encode namespace (e.g. `/<ns>/weights/...`).
   - All programmable loops (plans, caches, catalog) must treat namespace as a first-class scope.

---

## Error, Retry, Deadline, Cancel Semantics

### Error model

- All public methods raise `ArtifactError` (or subclasses) with canonical status codes.
- Errors must be classified into retryable/non-retryable categories, consistent with existing SDK.
- For async workflows, `Operation.status()` MUST expose structured failure detail (`OperationStatus.error`) and `Operation.wait()` MUST raise a typed exception that preserves `status_code` + `retryable` (timeouts SHOULD use a dedicated subtype such as `OperationTimeoutError`).

### Deadline semantics (hard cutoff)

- If deadline expires before dispatch, operation fails with `DEADLINE_EXCEEDED` and no side effects.
- If deadline expires in flight, SDK triggers best-effort cancel and marks the operation `degraded`.
- `Operation.wait(timeout_s=...)` is a client-side wait budget; if it expires, `wait()` SHOULD raise a dedicated timeout exception and include `TimeoutErrorDetails(kind="wait_timeout", timeout_s=..., ...)` in status reporting.
- gRPC deadline is `min(remaining_budget, per_call_timeout)`.

### Cancel semantics

- Cancel is best-effort.
- For joinable operations (prefetch), cancel MUST be scoped to the caller’s waiter (`request_id`), not global to the `replica_uuid`, unless explicitly owned.

### Idempotency

If `ctx.idempotency_key` is provided:

- `action_id = hash(idempotency_key + action_name + canonicalized_inputs)`
- repeated action submissions must return the same underlying operation semantics (no duplicate replicas, no waiter leakage)

If no idempotency key is provided:

- semantics are at-least-once best-effort; callers must tolerate duplicates

---

## Proto Requirements

This design assumes proto can evolve. The following are required to make the user-facing contracts real (deterministic `replica_uuid`, join, waiter-scoped cancel):

### StoreDaemonService (additive)

**MaterializeReplica / MaterializeByKey**

- add `request_id` (waiter id) so joins/cancels are scoped per caller
- add `join_if_exists` (idempotent join on existing deterministic `replica_uuid`)
- extend `MaterializeReplicaStatus` (or introduce a dedicated ticket enum) to represent a terminal success state (e.g., `READY`) so tickets can be waited/polled safely
- include `replica_key` and `replica_uuid_algo` for validation/debug (optional but recommended)

**QueryReplicaStatus / ReleaseReplica (existing RPCs; must be wired)**

- implement handlers in `StoreDaemonServiceImpl` (currently not overridden) and define semantics:
  - `QueryReplicaStatus` returns ticket state (`ALLOCATED`/`READY`/`FAILED`) for a given `ReplicaTicket`
  - `ReleaseReplica` is waiter-scoped by `(replica_uuid, request_id)`; releasing one waiter must not cancel the underlying replica globally unless explicitly owned

**WaitReplicaStatus (recommended)**

- add a long-polling wait RPC to avoid client-side polling loops (especially for low-latency QoS)

**Placement lease RPCs**

- `CreatePlacementLease`, `RenewPlacementLease`, `ReleasePlacementLease`
- map to daemon’s existing placement_pins semantics

### GlobalStoreService (recommended)

- Worker registration already assigns a stable `worker_id`; treat `worker_id` as the daemon identity for deterministic `replica_uuid` derivation.
- `RegisterReplica`/`UpdateReplica` should be upsert-safe for deterministic replicas (reject collisions across different daemons/workers).

---

## Examples

### Example 1: Weights warm pool + execute-while-load

```python
import tensorcast
from tensorcast import weights

ctx = tensorcast.context(request_id="scaleout:001", qos="background", deadline_ms=120_000)
model_key = weights.key(namespace="prod", model_id="llama-70b", version="v7", dtype="bf16", layout="tp8pp1")
art = ctx.artifact(key=model_key, fallback="p2p")

# warm pool (daemon-owned)
prefetched, ticket = art.prefetch(device="cuda:0", ctx=ctx)
ticket.wait(timeout=60.0)
lease = await prefetched.pin_residency(device="cuda:0", ttl_ms=300_000, ctx=ctx)

# execute-while-load (engine-owned)
hot = weights.layers(art, first_n=8)
hot.tensor_dict_into(target=params, device="cuda:0", ctx=ctx)
```

### Example 2: KV delta transfer via view + into

```python
import tensorcast
from tensorcast import kvcache

ctx = tensorcast.context(request_id="req-123", qos="realtime", deadline_ms=50)

kv_key = kvcache.key_to_str(hit_key)
src = ctx.artifact(key=kv_key, fallback="p2p")

delta = kvcache.delta(
    src,
    tensor_names=layer_tensor_names,
    seq_dim=2,
    start=node_hit,
    length=delta_len,
)

delta.tensor_dict_into(target=kv_targets, device="cuda:0", ctx=ctx)
```

### Example 3: Orchestrate a warm pool with Plan

```python
import tensorcast

ctx = tensorcast.context(request_id="warm-pool:v7", qos="background", deadline_ms=120_000)
plan = tensorcast.plan(ctx)

art = ctx.artifact(key="/prod/weights/llama-70b/v7", fallback="p2p")

for w in workers:
    plan.on_worker(w).prefetch(art, device=0)
    plan.on_worker(w).pin_residency(art, device=0, ttl_ms=300_000)

await plan.run(concurrency=16)
```

---

## Compatibility & Migration

- All new APIs are additive.
- Existing calls to `tensorcast.artifact(...).tensor_dict(...)` / `tensor_dict_into(...)` continue to work.
- `prefetch()` behavior becomes safer when `CallContext` is used:
  - without context: existing behavior (random `replica_uuid`) remains for compatibility
  - with context: deterministic `replica_uuid` + join semantics prevent duplicate daemon-owned replicas
- `pin_residency()` is new; existing deployments without placement lease RPC will raise a clear `FAILED_PRECONDITION` with guidance.

---

## Code Map

Suggested code locations:

- Context + operation protocols:
  - `tensorcast/api/store/context.py` (`CallContext`, `tensorcast.context`)
  - `tensorcast/api/store/operation.py` (`Operation`, `OperationStatus`)
  - `tensorcast/api/store/async_ops.py` (existing `ArtifactFuture` adaptor)
  - `tensorcast/api/store/batch_context.py` (existing `PrefetchTicket` adaptor)
- Artifact handle extensions:
  - `tensorcast/api/store/artifact.py` (add ctx-aware prefetch, ctx plumbing for into)
- Control-plane clients (ground truth today):
  - `tensorcast/daemon_ctl.py` (daemon ticket/status RPCs)
  - `tensorcast/dashboard/gs_client.py` (async Global Store client; can be extracted into a shared module)
- Plan orchestration:
  - `tensorcast/api/plan/__init__.py`
  - `tensorcast/api/plan/plan.py` (IR + execution)
  - `tensorcast/api/plan/targets.py` (`TargetSpec`)
- Signals:
  - `tensorcast/api/signals/tensorcast.py`
  - `tensorcast/api/signals/execution.py`
- Domain helpers:
  - `tensorcast/__init__.py` (public re-exports: `weights`, `kvcache`)
  - `tensorcast/domain/__init__.py`
  - `tensorcast/domain/weights.py`
  - `tensorcast/domain/kvcache.py`
  - `tensorcast/domain/kvcache_catalog.py` (optional)

---

## Naming Compliance

This design’s proposed public interfaces follow repository naming conventions:

- Python classes use `PascalCase`: `CallContext`, `Operation`, `OperationStatus`, `OperationError`, `TimeoutErrorDetails`, `PlacementLease`, `TargetSpec`, `Plan`, `PlanResult`, `PlanStepResult`, `PrefetchResult`, `SignalSnapshot`, `TensorCastSignals`, `ExecutionSignals`, `ExecutionSignalsAdapter`, `InstanceProfile`.
- Python functions/methods use `snake_case`: `tensorcast.context`, `CallContext.artifact`, `Artifact.prefetch`, `Artifact.tensor_dict_into`, `Artifact.pin_residency`, `Operation.status`, `Operation.wait`, `Operation.cancel`.
- Domain helper functions use `snake_case`: `weights.key`, `weights.layers`, `kvcache.key_to_str`, `kvcache.delta`.
- Protobuf RPC names remain `PascalCase` (`MaterializeReplica`, `QueryReplicaStatus`, `ReleaseReplica`); new fields use `snake_case` (`request_id`, `join_if_exists`).

---

## Appendix: Design Rationale (Why this shape?)

- Handles over ad-hoc getters: matches the existing SDK principle and keeps room for evolving internal strategies.
- No parallel “programmability object model”: avoids fracturing users into two APIs.
- Plan as IR: delivers programmability without building a distributed Python runtime.
- Deterministic `replica_uuid`: fixes real-world VRAM duplication issues under retries, multi-agent deployments, and at-least-once orchestration.
- Pinning as first-class: makes warm pools and serverless viable without abusing PID leases.
