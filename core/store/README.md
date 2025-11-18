---
title: Store Engine Internals
description: Deep-dive of the C++ StoreEngine, data paths, memory model, and P2P orchestration
sidebar_position: 1
---

# Store Engine Internals (core/store)

This document explains the internal implementation of the C++ Store Engine based on the code in core/store. It focuses on:

- Public API surface and data contracts used by Python bindings and the Daemon
- Data flows for disk and P2P loading, and for in-memory registration
- Memory model (VS/UMA), state machines, and eviction
- Component responsibilities and how they collaborate

Key files:
- StoreEngine: core/store/store_engine.h, core/store/store_engine.cc
- Replica + ReplicaLoadController: core/store/replica/*
- Loaders and pump: core/store/materialization/dataplane/*
- View planning + execution: core/store/materialization/dataplane/view/{view_planner,view_plan_source}.{h,cc}
- Service helpers:
  - IndexService: core/store/materialization/dataplane/metadata/index_reader.{h,cc}
  - VerificationService: core/store/materialization/dataplane/verification/verification_utils.{h,cc}
  - EvictionService: core/store/components/eviction_service.{h,cc}
  - View spec helpers: core/store/view_utils.{h,cc}
- Components: core/store/components/*
- Types: core/store/materialization/contracts/loading_spec.h, core/store/device_types.h, core/store/communication_types.h

## High-level Architecture

```mermaid
graph TB
  subgraph API
    SE[StoreEngine]
    MO[MaterializeOrchestrator]
  end

  subgraph Components
    RR[ReplicaRegistry]
    DM[DeviceManager]
    GSC[GlobalStoreClient]
    MC[MetricsCollector]
    CMN[CommunicationManager]
  end

  subgraph Replica Runtime
    REP[Replica]
    MM[ReplicaLoadController]
    UMA[UnifiedMemoryAuthority]
  end

  subgraph Memory
    PMP[PinnedBufferPool]
    SPB[StreamingPinnedBuffer]
    CUMEM[GpuDeviceMemory]
  end

  subgraph IO & Transfer
    LD[IArtifactLoader]
    DLD[DiskLoader]
    P2L[P2PLoader]
    SRC[SeekableSource]
    PUMP[pump_ranges]
    TS[TransferService]
  end

  SE --> MO
  SE --> RR
  SE --> DM
  SE --> GSC
  SE --> MC
  SE --> CMN

  RR --> REP
  REP --> MM
  MM --> UMA
  MM --> PMP
  MM --> SPB
  MM --> CUMEM

  REP --> LD
  LD --> DLD
  LD --> P2L
  LD --> SRC
  SRC --> PUMP
  TS --> PUMP
```

## UMA V3 Cutover (single ledger)

- Final cutover (V3): VirtualAddressSpace has been removed; UnifiedMemoryAuthority owns CPU virtual address reservation, chunk telemetry, and export bookkeeping. Incremental rollout flags are gone; transactional Plan→Execute→Commit and UMA-ledger authority are always enabled.
- Canonical Bazel targets: `//core/store/replica:unified_memory_authority` and `//core/store/replica:memory_export_registry`.

### Release & Eviction (GPU)

- GPU unload path now explicitly informs UMA to drop per-device residency and optional allocation via `UnifiedMemoryAuthority::release_gpu_device(ReplicaKey, device_id, drop_allocation=true)`. This keeps the UMA ledger authoritative and ensures VRAM is actually reclaimed. Subsequent `plan_load(GPU)` will correctly compute missing chunks instead of returning an empty set.

## Public API Surface (StoreEngine)

- Construction: `StoreEngine::StoreEngine(const StoreEngineOptions& opts)`
  - Configures `storage_path`, pinned pool size/chunk size, UMA chunk size, and optional `CommunicationManager` and `GlobalStore` address.

- Materialization (multi-device):
  - `absl::StatusOr<loading::ReplicaHandle> materialize_replica(const DeviceKey&, MaterializeMode, const MaterializeHints&)`
  - Modes:
    - `AUTO`: Uses `MaterializeOrchestrator` to request a P2P transport from Global Store. If `hints.disk_path` is populated it will fall back to disk; when `disk_path` is empty the orchestrator now returns the transport status directly (no implicit fallback).
    - `LOAD_ONLY`: Loads from disk only and requires `hints.disk_path` (rejects content-addressed IDs without that path).
    - `COPY_ONLY`: GPU→GPU copy from an already-loaded GPU instance; requires `hints.artifact_id` and a GPU target.
  - Safetensors disk fallback rebuilds canonical index JSON bytes and re-hashes them via `common::compute_index_multihash` so `mi2` identities stay stable even when the original `tensor_index.json` is absent.
  - Returns `ReplicaHandle { ReplicaKey, ready_future, cpu_state, gpu_state, gpu_base_ptr, cuda_ipc_handle, view_index_json?, view_data_hash? }`.
  - Variant-aware hints: populate `MaterializeHints::variant` (canonical id, optional view id/spec, placement) to request a view. The resulting `ReplicaKey` includes `view_id` so the registry differentiates canonical and variant replicas on the same device.

  Note: In the key-based client flow (RFC‑0014), the Store Daemon is responsible for resolving the human key via Global Store and supplying `hints.artifact_id` and, when applicable, `hints.disk_path` (derived from key mapping). Clients do not pass `disk_path` directly; fallback is orchestrated entirely inside the daemon/engine.

- In-memory registration (RFC-0006/0007):
  - `begin_register_artifact(const ArtifactRegistration&) -> RegistrationBeginResult`
    - Allocates target GPU memory via a temporary `Replica` and returns a CUDA IPC handle so the caller can write directly.
  - `commit_registered_artifact(string_view registration_id) -> RegistrationCommitResult`
    - Computes `mi2:<index_multihash>:<data_multihash>` from GPU memory and optional canonical index bytes, optionally exports remote keys via communicator, and registers with Global Store.
  - `abort_registered_artifact(string_view registration_id)`
  - Implementation is provided by `components::ArtifactRegistrationManager`, which owns pending registration state, TTL enforcement, view ingestion, and Global Store publication so `StoreEngine` only orchestrates dependencies.

- Replica queries and management (ReplicaKey-centric):
  - `wait_replica_ready`, `unload_replica`, `get_replica_state`, `get_replica_gpu_ptr`, `get_replica_size`
  - `get_resident_devices(artifact_id)`, `list_device_replicas(DeviceKey)`
  - UMA telemetry (authoritative):
    - `get_chunk_states_telemetry(artifact_id)` returns the UMA CPU snapshot (identical to `get_chunk_states_cpu_uma`)
    - GPU: `get_chunk_states_for_device(artifact_id, device_id)`
    - CPU (artifact-level convenience): `get_chunk_states_cpu_uma(artifact_id)`
      - Selection rule when multiple replicas exist: prefer a CPU instance if present; otherwise choose the GPU instance with the smallest device ordinal.

- View registration (v1.5):
  - `begin_register_artifact` accepts optional `ViewRegistration` payloads and requests a `BidirectionalViewPlan` from `core/store/materialization/dataplane/view::ViewPlanner`.
  - `ingest_view_registration_chunk` streams view bytes (SERVER placement) into canonical memory using `ViewIngestExecutor`; `commit_registered_artifact` publishes canonical + variant hashes and canonical coverage.
  - See [Variant View Registration Telemetry](../../docs/architecture/p2p-transfer-strategies.md#variant-view-registration-telemetry) for the end-to-end flow across daemon and Global Store.

- Remote access and registration helpers:
  - `enable_remote_replica_access/disable_remote_replica_access`
  - `register_replica_with_global_store(ReplicaKey[, canonical_mi2_override])`

### Variant-Aware Views (v1)

- `StoreEngine::compute_view_plan(canonical_index_json, ViewSpec)` constructs a deterministic `ViewPlan` by delegating to the loader `ViewPlanner`. v1 supports single-dimension `narrow` (slice) operations.
- `StoreEngine::view_plan_allows_alias(const ViewPlan&)` exposes the loader selection analysis so callers can short-circuit to zero-copy aliasing when (and only when) the selection is contiguous, segment-aligned, and no transforms are required.
- `StoreEngine::compute_view_data_hash_from_source(SeekableSource&, ViewPlan, leaf_bytes)` streams the canonical byte space through a loader `ViewPlanSource`, materializes the view in CPU memory when transforms are present, and computes the variant `view_data_hash` using the standard TreeHash pipeline.

These helpers intentionally operate on generic `SeekableSource` instances; the daemon sends either a GPU-backed `LinearizedGpuPlanSource` or a disk-backed source to the hash helper, which keeps verification logic unified across transports.

// VS lock APIs have been removed in UMA V3 final state.
// Use UMA plan/commit and VS pin_range for CPU residency leasing.

### Granularity and Invariants

- Canonical sizes:
  - Artifact chunk (layout/state): `artifact_chunk_bytes`
  - Transfer slice/window (pump, pinned blocks): `tx_slice_bytes`
  - Hash leaf (content-addressed leaf): `hash_leaf_bytes`
- Invariants enforced at startup:
  - `artifact_chunk_bytes % tx_slice_bytes == 0`
  - Pinned buffer pool slice size aligned to 512 B and page size
- Transfer ranges never cross artifact chunk boundaries; pump slices always align to `tx_slice_bytes`.

## Data Paths

### AUTO Materialize (P2P-first)

```mermaid
sequenceDiagram
  participant Client
  participant SE as StoreEngine
  participant MO as MaterializeOrchestrator
  participant GS as GlobalStoreClient
  participant REP as Replica/ReplicaLoadController
  participant P2L as P2PLoader

  Client->>SE: materialize_replica(target, AUTO, hints)
  SE->>MO: run(hints.artifact_id, target, hints)
  MO->>GS: request_replica_transport()
  alt transport granted
    GS-->>MO: TransportSession(remote replica)
    MO->>SE: ingest_from_p2p(...)
    SE->>REP: ensure_loaded_async(GPU/CPU)
    REP->>REP: load_async_from_source(P2P)
    REP-->>SE: ready_future (LOADED)
    SE->>GS: complete_replica_transport()
    SE->>GS: register_replica_with_global_store()
  else no transport
    MO-->>SE: fallback to disk
    SE->>SE: ingest_from_disk(...)
  end
  SE-->>Client: ReplicaHandle
```

### Disk Loading

```mermaid
sequenceDiagram
  participant SE as StoreEngine
  participant REP as Replica
  participant MM as ReplicaLoadController
  participant DLD as DiskLoader
  participant TS as TransferService

  SE->>REP: get_or_create_replica(DiskSource)
  REP->>MM: ensure_loaded_async(target)
  MM->>MM: allocate_memory(target)
  MM->>DLD: open_source()
  DLD-->>MM: SeekableSource (FilePartition/MultiSafetensors)
  MM->>TS: load_from_source(..., ranges)
  TS->>PUMP: pump_ranges (concurrency)
  PUMP->>MM: write into VS or GPU via sinks
  MM->>MM: finalize_load and set_state(LOADED)
```

### In-memory Registration (Begin → Write → Commit)

```mermaid
sequenceDiagram
  participant Client
  participant SE as StoreEngine
  participant REP as Replica/ReplicaLoadController
  participant CMN as CommunicationManager
  participant GS as GlobalStoreClient

  Client->>SE: begin_register_artifact(reg)
  SE->>REP: allocate GPU (GpuDeviceMemory + VS)
  SE-->>Client: CUDA IPC handle + registration_id
  Client->>REP: write tensor bytes via IPC handle
  Client->>SE: commit_registered_artifact(registration_id)
  SE->>SE: compute index_multihash + data_multihash
  alt comm enabled
    SE->>CMN: export GPU memory keys
  end
  SE->>GS: register_memory_replica(mi2:..., keys, sizes)
  SE-->>Client: RegistrationCommitResult (ids, sizes, hashes)
```

## Memory Model and State Machines

Store Engine relies on UMA (UnifiedMemoryAuthority) for contiguous per-replica CPU virtual address space, chunk bookkeeping, and export management. ReplicaLoadController orchestrates allocation, state transitions, and transfers.

### ReplicaLoadController Location States

```mermaid
stateDiagram-v2
  [*] --> UNINITIALIZED
  UNINITIALIZED --> UNALLOCATED: ctor
  UNALLOCATED --> ALLOCATED: allocate_memory(loc)
  ALLOCATED --> LOADING: load_from_source / copy_data_async
  LOADING --> LOADED: finalize_load_state(OK)
  LOADING --> FAILED: finalize_load_state(error)
  FAILED --> UNALLOCATED: release_memory
  LOADED --> UNALLOCATED: release_memory
```


- GPU allocations are lazily created via UMA on first use; VS CPU region is reserved at construction.
- Transfers and loading are pipelined via `TransferService` and `pump_ranges`, using a per-session `StreamingPinnedBuffer` backed by the shared `PinnedBufferPool`. `TransferService` now synchronises the per-device H2D stream via `AsyncCopyManager::synchronize_h2d_stream()` followed by `cuda::device_synchronize()` so GPU residency is fully committed before verification and metadata persistence run.

### Async Copy Manager Integration

- H2D/D2H transfers submit via `AsyncCopyManager` (ACM), which wraps `cudaMemcpyAsync` with traced host callbacks and forwards completions onto a dedicated CPU worker to avoid making CUDA Runtime calls inside `cudaLaunchHostFunc`.
- StreamingPinnedBuffer slots remain owned by the pump consumer. ACM callbacks now only publish their completion status (and may request a `BufferPool::shutdown()` on failure); the consumer thread waits for that status and then returns the slot. Callbacks must not mutate `PumpState` or hand the slot back directly.
- Use `StreamingChunkGuard` (from `core/common/memory/streaming_chunk_guard.h`) when staging chunks for AsyncCopyManager. The guard acquires slots, promotes them to consumer ownership, and guarantees cleanup if submission fails, so callers cannot forget the `promote_producer_slot_to_consumer` transition.
- For H2D, UMA state should advance from the completion callback (if the caller needs it) using the `absl::Status` argument as proof of success; per-chunk `stream_synchronize` remains unnecessary.
- ACM owns per-device non-blocking streams per direction (H2D/D2H/D2D) and routes all copies through them; external stream injection has been removed.

#### ACM Usage Quick Guide

- Submit one async copy per chunk via `AsyncCopyManager`, and return SPB slots (or wake dependents) inside the `on_copy_done` callback. Do not `cudaStreamSynchronize()` per chunk; the callback runs on a CPU worker after the CUDA stream callback fires. Pump consumers dealing with non-owning state (e.g., `pump_ranges`) instead poll handles and finalize completions inline so the callback does not outlive stack-owned context.
- H2D example (return slot + optional UMA advancement in callback):
  ```cpp
  using common::AsyncCopyManager;
  common::HostRegion h{.base = host_ptr, .length = bytes, .pinned = true};
  common::DeviceRegion d{.device_id = device_id, .dev_ptr = dst_dev, .length = bytes};
  auto hdl_or = AsyncCopyManager::instance().submit_h2d(
      h,
      d,
      {.tracing_stage = "H2D/Copy",
       .callbacks = {.on_copy_done = [spb, slot_id, maybe_uma](absl::Status st) {
         if (st.ok()) {
           (void)spb->return_chunk(slot_id);
           if (maybe_uma) {
             maybe_uma();
           }
         } else {
           LOG(ERROR) << "async copy failed, slot=" << slot_id << ": " << st;
         }
       }}} );
  RETURN_IF_ERROR(hdl_or.status());
  ```
- D2H example (stage into SPB, mark ready in callback for writer thread):
  ```cpp
  auto hdl_or = AsyncCopyManager::instance().submit_d2h(
      src_dev,
      dst_host,
      {.tracing_stage = "D2H/Copy",
       .callbacks = {.on_copy_done = [spb, slot_id, gid, bytes](absl::Status st) {
         if (!st.ok()) {
           LOG(ERROR) << "async copy failed, gid=" << gid << ": " << st;
           return;
         }
         (void)spb->mark_chunk_ready(slot_id, gid, bytes);
       }}} );
  ```
- Same-device D2D: use `submit_d2d` and wait on handles as needed. Cross-device D2D remains under `ReplicaLoadController` (peer or staged fallback).
- Tests and CPU-only flows can use `submit_h2h` to exercise the async pump path without CUDA; the copy is dispatched on ACM’s callback worker so CopyHandles complete asynchronously.

### Chunk States (UMA Authority)

UMA is the sole ledger for chunk residency and export flags. VS no longer performs
transfer locking; instead, it provides CPU pin leases via `pin_range()` and basic
IO helpers (`write_at`, `map_file_segments`).

## Component Responsibilities

- StoreEngine
  - Coordinates materialization, memory registration and eviction
  - Owns `DeviceManager`, `ReplicaRegistry`, `MetricsCollector`, optional `GlobalStoreClient`, and optional `CommunicationManager`
  - When Global Store is configured, store registrations now fail if the daemon cannot determine a routable (non-loopback) IP to advertise; operators must set `--advertise_host` or a non-loopback `listen_addr`
  - The `GlobalStoreClient` channel and stub are constructed once at engine bring-up; `initialize()` performs only the health-check handshake so retry/backoff state stays immutable at runtime
  - Implements helpers for VS chunk locks, remote access registration, and Global Store registration

- Replica
  - Facade for a single instance bound to a specific `DeviceKey`
  - Provides `ensure_loaded_async`, `copy_from`, `release_memory`, verification, and remote access enable/disable

- ReplicaLoadController
  - Manages CPU/GPU memory allocation and state; exposes pointers and CUDA IPC handle
  - Performs copy and load via `copy_data_async` and `load_async_from_source`
  - Uses UMA + VS for chunk metadata and virtual address ownership

- Loaders (`DiskLoader`, `P2PLoader`)
  - Abstracted via `IArtifactLoader` to produce a `SeekableSource`
  - P2P path muxes a remote source with an optional on-disk fallback (`MuxSeekableSource`)

- TransferService
  - Builds target sinks (GPU or VS region) and computes ranges from chunk indices
  - Enforces per-GPU concurrency (1 active session per GPU)
  - Streams data using a per-session `StreamingPinnedBuffer`

- ReplicaRegistry
  - Thread-safe multi-index: by-instance (`ReplicaKey`), by-artifact, by-device
  - LRU ordering feeds GPU eviction

- DeviceManager
  - GPU discovery, uuid↔ordinal mapping, per-device stream, and free/total memory metrics

- CommunicationManager
  - Wraps `Communicator`; registers memory for P2P and provides shared engine to loaders

## Content-Addressed Identity and Verification (RFC-0007)

During disk ingestion or commit of in-memory registration:
- `index_multihash` is derived from canonical index bytes (safetensors or tensor_index.json). When absent, it is computed.
- `data_multihash` is computed by linearizing the canonical SegmentPlan (PAD=0) over the loaded buffer. When canonical index bytes are available, hashing injects zeroes for PAD gaps to ensure RP‑A/B/C equivalence; otherwise it falls back to contiguous GPU hashing using the runtime-compiled NVRTC SHA256 kernel (with the old CPU streaming path as automatic fallback). The GPU lane now ingests 64-byte message blocks directly, auto-tunes leaf chunking down to 512 KiB to keep ≥4K leaves resident for large tensors, and returns digests via pinned host memory with async copies to overlap compute and transfer.
- For disk ingestion, missing descriptors are materialized under the artifact directory:
  - `artifact_descriptor.json` (index/data multihash, sizes)
  - `tensor_index.json` (canonical, when needed)
- In-memory commit returns `mi2:<index_multihash>:<data_multihash>` and optionally registers with Global Store.

## Memory Eviction (GPU)

When GPU allocation fails or during registration when memory is tight, StoreEngine executes an LRU-based, device-aware eviction:

```mermaid
flowchart TD
  Start[need bytes on GPU d] --> Query[query free memory]
  Query -->|enough| Done
  Query -->|not enough| LRU[iterate LRU instances on device d]
  LRU --> Release["release GPU memory of LOADED instances (safe)"]
  Release --> Recheck[re-check free memory]
  Recheck -->|enough| Done
  Recheck -->|not enough| LRU
  Done --> Result{OK or ResourceExhausted}
```

Implementation: `try_evict_gpu_memory_impl()` in store_engine.cc. CPU VS memory is not freed here.

## Concurrency and Error Semantics

- ReplicaLoadController uses a single `absl::Mutex` with per-location condition variables to gate state transitions.
- `ready_future` from `ReplicaHandle` resolves with the final `absl::Status` of the load/copy.
- Unsafe releases during LOADING mark the destination `FAILED` with last error preserved for diagnostics.
- TransferService limits one active transfer per GPU; disk read/write concurrency is configured via `MaterializeHints::pipeline_concurrency` (propagated to `pump_ranges`).

## Metrics

`MetricsCollector` gathers:
- Pinned pool utilization and CPU available bytes
- Replica counts and sizes
- GPU metrics (free/total) via `DeviceManager`
- Operation latencies (disk/P2P) and P2P byte counters (exposed via `tc_artifact_load_seconds` with `view_scope`, `view_id_prefix` labels so canonical vs variant loads can be distinguished)

## Key Types and Contracts

- `DeviceKey` (core/store/device_types.h): logical device id `{type, ordinal, uuid}` used across APIs.
- `ReplicaKey` (core/store/materialization/contracts/loading_spec.h): `{artifact_id, view_id?, device, replica}` uniquely identifies an instance; `view_id` captures optional variant byte-space residency.
- `MemoryLocation` (core/common/memory/memory_location.h): `GPU`, `CPU`, `DISK`, `REMOTE`.
- `ReplicaHandle` (loading_spec.h): conveys instance key, states, CUDA IPC handle, optional view metadata, and a `ready_future`.

## Test Coverage (selected)

- Disk → GPU: core/store/store_engine_test.cc
- P2P (TCP) → GPU with communicator: core/store/store_engine_p2p_loader_test.cc
- Streaming and sinks: core/store/materialization/dataplane/**/tests/*_test.cc
- UMA/VS chunk semantics and locking: exercised through ReplicaLoadController and loaders

## Notes and Limitations

- `InlineBufferSource` ingestion path is currently unimplemented for general loading; it is used internally for memory-only allocations (registration, COPY_ONLY source/dest creation).
- Content-addressed `mi2:...` artifact IDs require Global Store routing; disk fallback must specify `hints.disk_path`.
- Per-GPU transfer concurrency is limited to 1 active session by design to reduce VRAM fragmentation and pressure.
- Canonical tensor indices are expected in schema version `"v3"`; the engine emits v3 descriptors and rejects older schemas on write paths.
- Variant replica registration still publishes canonical residency to Global Store. The engine issues a best-effort `record_variant_residency` call so the daemon can start wiring the future RPC; until Global Store implements it the call returns `UNIMPLEMENTED` and is logged at `VLOG(1)`.

For broader architectural context, see docs/architecture.md and docs/state-management.md.
- Verification metadata: canonical replicas reuse `verification.json`. Variant views persist per-view metadata under `verification.view_<sanitized_view_id>.json`; each record carries the `byte_space_id` so canonical metadata is never reused for a variant. Every persisted payload now embeds a `metadata_signature` (canonical SHA-256 of the serialized payload). The loader re-reads the on-disk JSON on every materialization, validates the signature, and compares the payload fingerprint against any cached entry before reuse. Tampered or truncated files trigger `DataLoss` (cache is invalidated) and force regeneration, while cached entries are only reused when the file is absent and a fresh persistence will rewrite it. P2P senders may still provide inline `verification_json`; the backend `ingest_from_p2p()` path (StoreEngine delegates to `ingest_from_p2p_internal()`) parses it and performs fast KEY_POINTS verification of the loaded replica (CPU/GPU). Verification failure returns a `DataLoss` error and aborts materialization. All metadata reads/writes are serialized through `VerificationMetadataGuard`, persisted via an atomic write helper (`open` → `write` → `fsync` → `rename` + directory `fsync`), and accompanied by structured `verification_metadata_write_{succeeded,failed}` logs that surface artifact, byte-space, guard wait, and write durations.
- Debug visibility: enabling `--v=1` (or higher) now emits the key-point triplet and artifact size whenever verification metadata is regenerated or reused. These logs include the active CUDA device id so stale metadata vs. stale GPU residency issues can be distinguished quickly when diagnosing DataLoss failures.
## Granularity Terminology and Invariants

- Artifact layout chunk: `artifact_chunk_bytes` (VS/UMA)
- Transfer slice/window: `tx_slice_bytes` (pinned buffer block)
- Invariant: `artifact_chunk_bytes % tx_slice_bytes == 0`
- Pinned pool block is aligned to 512 B and 4096 B (page) and validated at startup.

Defaults are defined in `core/common/const/granularity.h` and exposed to Python via the extension module.
