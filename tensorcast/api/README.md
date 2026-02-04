# TensorCast Python API

The `tensorcast.api` package exposes the high-level registration and loading
helpers that SDK integrations use during artifact lifecycle management.

## Store module layout

Design 0037 refactored `tensorcast.api.store` into a structured subpackage:

- `store/types.py` and `store/handles.py` keep immutable dataclasses and handle wrappers importable from `tensorcast.api.store`.
- `store/runtime.py` owns the process-wide daemon client, session record writes, key/capability caches, and fork-aware executor lifecycle.
- `store/registration.py` and `store/materialization.py` orchestrate register/put/view and artifact materialization flows with shared retry/error mapping.
- `store/views.py` keeps view-spec parsing, placement defaults, and canonical index lookups isolated from the pipelines.
- `store/async_ops.py` centralizes async helpers (`ArtifactFuture`, `TrackedExecutor`) so cancellation/confirm semantics are consistent across verbs.
- `store/__init__.py` is the public façade; it now eagerly wires runtime/registration/materialization without monkeypatch/override hooks or lazy rebuilds.

Module-level helpers (`tensorcast.api.store.register`, `get`, etc.) reuse a process-scoped `Store`. If you close that store (or invoke `shutdown_process_store()`), the next helper invocation transparently reinitializes a fresh instance instead of reusing the closed handle.

## Store Policy, Local Stable Tier, and Persistence

- `Store.register`/`register_async` and `Store.put`/`put_async` accept a first-class `policy: StorePolicy | str | None`; `RegisterArtifactOptions(policy=...)` remains as an advanced escape hatch. See `../../docs/architecture/api/policy-persistence.md`.
- If both `policy` and `options.policy` are provided, the SDK rejects conflicts after normalization to avoid silent divergence.
- Policies can be simple profiles (`cache`, `durable`, `ha`, `cold`, `pinned`, `warm`) or explicit `must`/`should`/`may` tier lists with `overflow_policy` and `layout` overrides.
- Tier constraints are enforced: `shared_disk` forbids retention fields, `stable_dram` supports only `min_replicas=1`, remote-only stable tiers disallow retention settings, and `must` local stable tiers require `pinned` retention.
- `CommitRegisteredArtifact` returns a `local_stable_tier` result (`ready`/`degraded`/`skipped`) on `RegisteredArtifact` when the resolved policy requests `stable_dram(scope=local)`. `must` failures raise commit errors; `should` failures are surfaced as `degraded` while the commit remains successful.
- The registration pipeline invokes daemon RPC `StartPersistence` only when the resolved policy requires shared disk or remote stable DRAM, and records the returned `persistence_task_id` on `RegisteredArtifact` (persistence prefers a daemon-owned local stable DRAM source when present).
- Use `Store.query_persistence_status` (or module helper `tensorcast.api.store.query_persistence_status`) with a `task_id` or `artifact_id` to fetch daemon-side task state; the SDK does not poll automatically.

## Artifact Handles & Metadata Cache

- `tensorcast.artifact(...)` / `Store.artifact(...)` provide lazy handles that
  expose metadata (`tensor_names`, `tensor_meta`, `describe`) and selective
  materialization (`tensor_dict(names=...)`, `tensor(name, ...)`, `tensor_into(...)`) as the canonical retrieval surface.
- Handles accept whichever identifiers are available (`artifact_id`, key, or
  disk path). At least one identifier is required, but resolved handles keep all
  known hints so `with_fallback(...)` and `to_dict()/from_dict()` remain valid
  even after key resolution.
- Handles are bound to the originating `Store`; materialization after
  `Store.close()` or `Artifact.release()` raises
  `ArtifactError(status_code="FAILED_PRECONDITION")` while cached metadata
  remains readable.
- The runtime now maintains an `ArtifactCache` for canonical indices. Defaults
  can be tuned with `TENSORCAST_STORE_INDEX_CACHE_TTL_SECONDS` (600s) and
  `TENSORCAST_STORE_CACHE_MAX_ENTRIES` (1000). Metrics are emitted for cache
  hits, misses, evictions (`reason=ttl|lru`), and invalidations (`reason=*`).

## Programmable Control-Plane Primitives

- `tensorcast.context(...) -> CallContext` is a pure per-call container for deadline/idempotency/tags. Handle factories
  remain context-free (`tensorcast.artifact(...)` / `Store.artifact(...)`).
- Action APIs accept `*, ctx: CallContext | None = None` as a keyword-only parameter; `ctx` does not participate in
  artifact/view identity, but `ctx.idempotency_key` seeds deterministic operation ids for joinable actions.
- Long-tail control-plane actions return `Operation[T]` (sync/blocking): use `status()` / `result()` / `cancel()` to
  implement wait/cancel without ad-hoc polling loops.
- `Artifact.prefetch(...)` warms a **daemon-owned** replica and supports both GPU and CPU/DRAM placement:
  - GPU VRAM: `device="cuda:0"` or `0`
  - daemon-owned host DRAM: `device="cpu"`, `"dram"`, or `-1`
  Prefetch defaults to `NO_LEASE` and does not export handles to the caller; handle-exporting APIs remain PID/lease-bound.
- `ctx.deadline_ms` is enforced end-to-end: materialization retries and polling operations clamp their budgets to the
  remaining deadline, and worker/agent RPCs inherit the same timeout budget.
- `tensorcast.plan(ctx)` builds a programmable orchestration plan. Plan steps target stable worker identities
  (`daemon_id`) and return `PlanStepRef` handles; `Plan.run()` executes with bounded concurrency and returns a
  `PlanResult` that aggregates per-step `OperationStatus`.

## Materialization v2 (descriptor streaming)

- `MaterializationPipeline` streams `TensorPayloadDescriptor` + tensor pairs from the daemon v2 surface (`tensorcast.proto.daemon.v2`) by default; the v1 path and `TC_ENABLE_MATERIALIZE_V2` flag have been removed.
- Selective fetch (`tensor_names`) trims descriptors and canonical index bytes; iterator cancellation still routes through `_release_materialized` so CUDA IPC handles are unmapped even on early exit. `tensor_dict_into` / `tensor_into` copies consume descriptors directly without building intermediate dicts.
- Telemetry attaches per-descriptor attributes (`tc.tensor.count`/`tc.tensor.bytes`) and a subset/full selector to the materialization span, and the client metrics surface attaches the same selector to latency/error/retry series.
- Disk fallbacks are forwarded to the daemon through `DiskFallbackHint` + `SourcePolicy` (preference + allow flags). Disk‑first sets `preference=PREFER_DISK` and all disk reads stay in the daemon data path.
- `MemCopyHandle` is lease-aware: the daemon returns a handle (`cuda_ipc_handle` for GPU or `cpu_memfd` for CPU) plus an opaque `lease_token`. The SDK binds that lease token to the returned `torch.Tensor` lifetimes and releases it over the daemon’s local handle plane (UDS).

## Region-backed get_into

`tensor_dict_into` / `tensor_into` / `get_into` can stream bytes directly into
caller-owned CUDA regions via the v2 `MaterializeIntoTarget` RPC. The SDK
computes a coalesced `TargetLayout` over either the canonical index or a
view-indexed ByteSpace (including packed subsets via `tensor_names`), validates
dtype/shape/stride against the selected index, and requests the daemon to
materialize into the mapped region. Multi-storage layouts are supported via
ordered concatenation across registered regions. For non-identity views, the
SDK resolves a deterministic `view_id` and attaches it to the layout. The
daemon does not allocate VRAM, and the SDK does not call `UnloadReplica` for
this path.

The `region_backed_mode` default in the unified runtime config controls the
behavior:

- `auto`: try region-backed first; fall back to the legacy replica path on
  validation failures.
- `require`: enforce region-backed and surface errors.

`get` / `get_view` always use the daemon-owned replica path and ignore
`region_backed_mode`.

## Device requirements

`Artifact.tensor*`/`tensor_dict*` default to materializing replicas onto CUDA
devices. On hosts without `torch.cuda.is_available()`, callers must either
provide disk fallback options (so bytes can be streamed from disk) or select a
CPU target (`device="cpu"`) when the daemon is configured with CPU shared-memory
materialization (`engine.cpu_shared_memory.enabled=true`). If
`lifecycle.handle_leases.local_handle_socket_path` is empty, the daemon
auto-selects `<daemon_state_dir>/local_handle.sock` for same-pod/local SDKs
(daemon_state_dir defaults to `$TENSORCAST_HOME/hosts/<host_id>/sessions/<session_id>/session`
or `~/.tensorcast/hosts/<host_id>/sessions/<session_id>/session`, auto-discovery
relies on `TENSORCAST_INSTANCE`); set it
explicitly when daemon and client SDK run in different pods.
When connecting to the current local session and the daemon does not advertise
the socket path, the SDK falls back to the same daemon_state_dir location.
Otherwise the API raises `ArtifactError("CUDA device required for retrieval")`
immediately, keeping callers from running deeper into retry loops that can never succeed.

## View Retrieval

`Artifact.view(...).tensor*` defaults to executing transforms on the daemon so
transpose views return buffers in the expected orientation. Client-side
execution is intentionally disabled until a local transform engine exists; the
pipeline still accepts `placement="CLIENT"` explicitly for forward
compatibility.

## View Registration

`Store.register_view()` mirrors retrieval semantics so trainers can upload only
the bytes required for a narrow or transpose view while the daemon rebuilds the
canonical artifact. Key behaviours:

- Placement defaults to `SERVER` for pure narrow views and `CLIENT` when any
  transpose is present. Users can override via the `placement` keyword.
- When the daemon lacks GPU support for a server-side transpose it returns a
  `FAILED_PRECONDITION` status. The client surfaces a clear `ArtifactError`
  instructing callers to retry with `placement="CLIENT"`.
- `registration_kind="piece"` (or `register_piece`) registers dense view pieces
  for partial coverage. Piece registration is selection-only (narrow only), does
  not allow transpose, and requires server placement.
- `allow_partial` is deprecated and maps to `registration_kind="piece"`. Sparse
  canonical zero-fill semantics are removed.
- `canonical_index_bytes` can be supplied to bootstrap a new assembly without
  requiring prior Global Store state.

The API returns a `RegisteredArtifact` whose `registration_result` carries the
view identifier, canonical coverage ranges, and view hash for downstream
automation.
See the [View Registration Telemetry](../../docs/architecture/p2p-transfer-strategies.md#view-registration-telemetry) guide for the full daemon ↔ Global Store ↔ SDK flow.

## Tensor Storage Graph Helper

`build_tensor_storage_graph()` inspects a `dict[str, torch.Tensor]` and returns
a `TensorStorageGraph` containing:

- **storages** – one `StorageEntry` per unique `torch.Storage` (deduped by
  device id, base pointer, and storage length).
- **aliases** – per logical tensor metadata (shape, stride, dtype, storage
  offset, logical byte length) keyed by tensor name.
- **tensor_meta_index / tensor_source_index** – canonical metadata reused by
  both disk persistence and lease-in-place registration.

Invariants guaranteed by the helper:

- All CUDA tensors in the input must reside on the same device; the helper
  records the device id for each storage and raises on mismatches.
- Storage identifiers are deterministic and stable for the lifetime of the
  process, enabling clients to sort and reference storage groups.
- The aliases map preserves every tensor key from the input dict; consumers
  can rebuild canonical index JSON by combining alias metadata with the device
  offsets supplied by layout planners.

Clients must invoke this helper before feeding lease segments so they can send
the deduplicated storage table alongside per-tensor aliases. The daemon uses
these structures to rebuild canonical indices and avoid repeated CUDA IPC
opens for shared storages.
