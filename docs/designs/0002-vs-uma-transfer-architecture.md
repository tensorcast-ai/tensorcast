---
id: design-0002-vs-uma-transfer-architecture
slug: 0002-vs-uma-transfer-architecture
title: VS (Virtual Address Space), UMA Ownership, TransferService, and Unified Loader (Design)
status: accepted
links:
  plan: ../plans/0002-dvmp-uma-transfer-architecture.md
areas: ["core", "daemon"]
related_code:
  - core/common/memory/virtual_address_space.*
  - core/store/replica/unified_memory_authority.*
  - core/store/replica/transfer_service.*
  - core/store/replica/memory_export_registry.*
  - core/store/replica/replica_load_controller.*
supersedes: ["rfc-0002", "rfc-0004"]
created: 2025-08-10
last_updated: 2025-09-09
---

# Summary

This design consolidates the memory and data‑movement architecture across VS (Virtual Address Space, CPU VA/IO), UMA (UnifiedMemoryAuthority), ReplicaLoadController (façade), a new TransferService, and a MemoryExportRegistry. It formalizes VS/UMA ownership boundaries, a unified loader sink contract, UMA’s sole ownership of VRAM and chunk state, and centralizes P2P export/unexport. The design aims for correctness under concurrency, clear ownership, low contention, and robust observability.

Key outcomes:
- VS: per‑artifact region handles, VS‑owned IO (`write_at`, `map_file_segments`), and pin leases (`pin_range`) for safe external exposure; VS carries telemetry only (non‑authoritative).
- Unified loader contract: all data paths pump via `pump_ranges(...)` into `PositionedSink::write_at(offset, ...)` sinks; optional capability negotiation via `DirectWriteCapable` with automatic staged fallback.
- UMA authority: single source of truth for DRAM/VRAM base pointers and chunk states; sole owner of VRAM allocations via `get_or_create_gpu_allocation`.
- ReplicaLoadController becomes a thin orchestration façade (capture → delegate → finalize) without long‑running work under its mutex.
- TransferService: owns CPU↔GPU copies and DISK/REMOTE→CPU/GPU pumping, plus StreamingPinnedBuffer (SPB) lifecycle, sizing, and alignment.
- ChunkExportService: centralizes P2P export/unexport with UMA‑provided pin/translation helpers; maintains registrations and leases.

```mermaid
flowchart LR
  subgraph Loaders
    S1[FilePartitionSource]
    S2[RemoteKeySource]
    MUX[MuxSeekableSource]
    S1 --> MUX
    S2 --> MUX
    PUMP[Pump Ranges]
    BUF["StreamingPinnedBuffer<br>(BufferPool via Adapter)"]
    MUX --> PUMP
    PUMP -.producers/consumer.-> BUF
  end

  subgraph Sinks
    DVS["CpuVaSink<br>(PositionedSink + DirectWriteCapable)"]
    GPS["GPUMemorySink<br>(PositionedSink)"]
  end

  subgraph Memory
    UMA[UnifiedMemoryAuthority]
    MM[ReplicaLoadController]
    VS[Virtual Address Space]
  end

  PUMP -->|Positioned writes| DVS
  PUMP -->|Positioned writes| GPS
  DVS -->|VS write_at| VS
  MM <-->|alloc/export/evict| VS
  MM -->|provide VS region/base| DVS
  MM -->|GPU ptr/stream| GPS
  PUMP -->|"after execute: UMA::commit(session)"| MM
```

# Goals / Non‑Goals

Goals
- Unified, coherent memory and IO policy centered on VS and UMA.
- Clear separation: façade (orchestration) vs. execution (VS/UMA/transfer/export).
- Correctness for partial/range loads under concurrency via positioned writes.
- Safe external exposure with pin leases and deterministic lifetimes.
- Low contention via per‑artifact handles and local locking.
- Preserve external `Artifact` API and enable incremental migration.

Non‑Goals / Constraints
- No new “runtime engine” layers beyond TransferService and ChunkExportService.
- No public loader API changes beyond the unified sink contract.
- No separate HostStagingManager; staging resides in TransferService.

# Architecture & Interfaces

Ownership boundaries
- VS (Virtual Address Space)
  - CPU VA reservation and per‑chunk telemetry.
  - Provides `write_at` and `map_file_segments` (VS‑owned IO) and exposes pin leases via `pin_range`.
  - Exposes per‑artifact `VaRegion` handles for low‑contention operations.
- UMA (UnifiedMemoryAuthority)
  - Sole owner of VRAM allocations on each device; returns base pointers and streams.
  - Owns chunk‑level state for CPU/GPU and direct‑write grants for sinks.
  - No synchronization from VS: UMA is authoritative; VS carries telemetry only.
- ReplicaLoadController (façade)
  - Capture inputs and set destination LOADING under lock, delegate out‑of‑lock to services, finalize LOADED/FAILED.
  - Provides small pass‑through helpers to UMA/Transfer/Export (no long operations under `mutex_`).
- TransferService
  - Owns SPB lifecycle: sizing, alignment (SPB chunk divides VS chunk; 4KiB multiple), back‑pressure.
  - Executes data paths:
    - DISK/REMOTE → CPU: `CpuVaSink` + `pump_ranges` into `write_at`.
    - DISK/REMOTE → GPU: ensure UMA GPU allocation; `GPUMemorySink` + `pump_ranges`.
    - CPU↔GPU streaming copies.
- MemoryExportRegistry
  - Centralizes CPU/GPU export/unexport for P2P, coalesces indices into VS‑aligned ranges.
  - Acquires UMA leases/grants and registers ranges with the Communicator; retains keepalives.

Unified loader contract
- A single pumping API: `pump_ranges(...)`.
- All sinks implement `PositionedSink::write_at(offset, data)` to guarantee placement.
- `DirectWriteCapable` capability negotiates remote→destination direct writes with automatic fallback to staging.
- ReplicaLoadController orchestrates `plan_load → execute → UMA::commit` and transitions facade state to LOADED.

# VS Preemptibility (CPU memory)

Semantics
- VS regions/chunks can be marked preemptible to allow the system to reclaim CPU RSS without giving up the virtual address reservation or artifact metadata.
- Preemption converts a resident, writable anonymous mapping into a non‑resident or file‑backed, read‑only placeholder for the affected range. VA, chunk layout, and offsets remain stable.
- Any consumer write (e.g., via `write_at`) transparently re‑hydrates the affected pages by remapping to writable anonymous memory for the minimal subrange needed.

State and transitions (per chunk)
- ResidentHot (anon RW) → MarkPreemptible → EligiblePreemptible
- EligiblePreemptible → Preempt → PlaceholderRO (file‑mapped) | Unmapped (if no durable file); remains logically present in VS
- PlaceholderRO/Unmapped → RehydrateOnAccess → ResidentHot (anon RW)
- Pinned (lease count > 0) is an attribute that forbids Preempt while active; it coexists with ResidentHot and PlaceholderRO but blocks transitions to Preempt.

Pin leases and safety
- Pin leases (issued by VS) prevent preemption of leased ranges. UMA and services must acquire leases for external exposure (P2P export, IPC) and direct‑write windows.
- `preempt(ranges)` skips or errors on pinned subranges; the operation is idempotent on already‑preempted ranges.
- UMA is responsible for acquiring/releasing leases when orchestrating flows that require stability.

Triggers and policy (owned by UMA)
- After GPU load: default policy is `EvictCPU`; alternative is `MarkPreemptible` to keep a low‑cost rehydrate path. Policy is configurable per workload.
- Memory pressure: UMA may invoke `preempt(ranges, reason=MEM_PRESSURE)` based on RSS budget, high‑water marks, or aging (LRU) tracked via VS telemetry (chunk touch time).
- Fairness: UMA enforces per‑artifact and global budgets; preemption candidates are chosen among `EligiblePreemptible` chunks by age/size.

Interactions with IO and loaders
- `write_at` on PlaceholderRO re‑maps the specific subrange to anonymous RW and updates chunk heat/timestamps.
- `map_file_segments` may materialize RO placeholders; `ensure_writable_mapping` upgrades on demand.
- TransferService DISK/REMOTE→CPU writes simply call `write_at`; demand rehydrate occurs inside VS.

APIs (illustrative)
- VS
  - `mark_preemptible(region, ranges)` → mark chunks eligible; does not reclaim RSS by itself.
  - `evict_tail_bytes(region, bytes)` → reclaim from tail in chunk granularity; advisory via madvise
  - Rehydration occurs on write via VS `write_at` path.
- UMA
  - `post_gpu_load_policy`: `EvictCPU` (call `preempt`), or `MarkPreemptible`.
  - `ensure_cpu_resident(region, ranges)` when a caller requires CPU residency prior to an operation.

Observability
- Metrics（VS）
  - `tc_va_map_bytes_total`, `tc_va_write_bytes_total`
  - `tc_va_pin_leases_total{reason}`
- Logging
  - VLOG(1) for policy decisions; VLOG(2) for reclaimed ranges; LOG(ERROR) on partial failures with affected ranges.

Failure modes and guarantees
- Preemption never breaks leases: leased ranges remain resident; operations return `FAILED_PRECONDITION` when a forced preempt is attempted on pinned chunks.
- If no durable file‑backing exists for a range, UMA must not preempt unless it guarantees re‑materialization from a source (disk/remote). Otherwise the chunk must remain ResidentHot or be explicitly reloaded.
- VA stability is guaranteed: addresses and offsets do not change due to preemption/rehydration.

Concurrency and locking
- VS: per‑artifact mutex for region operations.
- ReplicaLoadController: never hold its mutex while invoking UMA/VS/Communicator operations that may block.
- UMA: avoid holding VS locks across external calls; if unavoidable, acquire UMA → VS and release in reverse.
- RAII finalizer ensures exactly‑once LOADING→LOADED/FAILED transition per async path.

Observability and metrics
- VS: `tc_va_write_bytes_total`, `tc_va_map_bytes_total`, `tc_va_pin_leases_total{reason}`.
- Loader/transfer: `loader_bytes_total{source,location,mode∈{staged,direct}}`, `transfer_bytes_total{direction}`, `transfer_latency_ms{path}`.
- Export: `chunk_exports_total{location}`; registration keys logged at VLOG(2).
- UMA state telemetry: `artifact{location,state}` and transition logs.

Representative interfaces (illustrative)
- UMA
  - `get_or_create_gpu_allocation(device)`
  - `sync_cpu_chunk_states(ranges)`
  - `update_chunk_states(..., COPIED_GPU)`
- ReplicaLoadController
  - `load_async_from_source(source, target, concurrency, chunk_indices)`
  - `finalize_load(location, ranges)`
- TransferService
  - `ensure_streaming_buffer(capacity_chunks)` / `release_streaming_buffer()`
  - `load_from_source(source, Dst)` / `copy_cpu_to_gpu_streaming` / `copy_gpu_to_cpu_streaming`
- ChunkExportService
  - `export_chunks(location, indices)` / `unexport_chunks(location, indices)`

# Schema Changes

None. This design covers in‑process memory management, IO paths, and P2P export; it does not introduce or alter persisted schemas.

# Trade‑offs & Risks

Risks and mitigations
- Hidden VS semantic dependencies → Centralize VS‑affecting ops in UMA; IO in TransferService; export in MemoryExportRegistry.
- Missed async finalization → RAII finalizer enforces LOADING→LOADED/FAILED exactly once.
- Performance regressions from additional layers → Services are in‑process; zero‑copy preserved; positioned writes avoid rework.

Alternatives considered
- Keeping staging/export logic in ReplicaLoadController increased lock coupling and deadlock risk; rejected in favor of clear service ownership.

# Compatibility & Acceptance Criteria

Compatibility
- External `Artifact` API remains unchanged.
- Internal façades keep method signatures while delegating to UMA/Transfer/Export.

Acceptance criteria
- All loader paths pump via `pump_ranges(...)` into positioned sinks with capability negotiation.
- UMA exclusively owns VRAM allocations and chunk state updates; VS never writes authoritative states.
- TransferService manages SPB lifecycle and executes DISK/REMOTE→CPU/GPU and CPU↔GPU copies.
- ChunkExportService centralizes export/unexport and maintains leases/registrations.
- Concurrency rules and metrics are implemented as specified.

# References

- Owning code paths: see `related_code` in frontmatter.
- Architecture docs: [Architecture Overview](../architecture/architecture-overview.md), [P2P Transfer Strategies](../architecture/p2p-transfer-strategies.md), [Artifact Loading Workflow](../internals/model-loading.md).
 

## Appendix

DISK/REMOTE → GPU (streaming)

```mermaid
sequenceDiagram
  participant MM as ReplicaLoadController
  participant TS as TransferService
  participant UMA as UMA
  participant VS as VS
  MM->>TS: load_from_source(source, GPU, ...)
  TS->>TS: ensure_streaming_buffer()
  TS->>UMA: get_or_create_gpu_allocation
  UMA-->>TS: GpuDeviceMemory
  TS->>VS: pump ranges (read→write_at / device copy)
  TS-->>MM: status
  MM->>UMA: update_chunk_states(..., COPIED_GPU)
  MM->>UMA: post_gpu_load_policy(..., EvictCPU)
  MM->>MM: finalize LOADING→LOADED/FAILED
```

P2P Export

```mermaid
sequenceDiagram
  participant MM as ReplicaLoadController
  participant CES as MemoryExportRegistry
  participant UMA as UMA
  participant CE as Communicator
  MM->>CES: export_chunks(loc, indices)
  CES->>UMA: grant_direct_write(ranges)
  UMA-->>CES: grant (leases held)
  CES->>CES: coalesce indices → ranges
  CES->>CE: register ranges (keys)
  CE-->>CES: ok
  CES-->>MM: registration (keepalive retained)
```
