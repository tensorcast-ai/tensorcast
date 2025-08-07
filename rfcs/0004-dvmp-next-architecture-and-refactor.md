# 0004 — DVMP Next: Single-Owner Memory, Per‑Model Concurrency, and Fault‑Aware IO

Status: Proposed
Date: 2025‑08‑07
Owner: StepCast Core
Depends on: 0001 (DVMP), 0002 (Unified Loader), 0003 (Unified IO + Pin Leases)
Supersedes: Parts of 0003 (IO path ownership and safety), clarifies 0001

## 0. Executive Summary

This RFC consolidates DRAM ownership in the Distributed Virtual Memory Pool (DVMP), removes legacy fast paths that bypass DVMP control, introduces a per‑model concurrency model that scales under multi‑tenant load, and completes the chunk‑aware P2P→Disk fallback story. The design is grounded in the current code:

- DVMP interface and implementation: `core/common/memory/distributed_memory_pool.{h,cc}`
- Memory orchestration: `core/store/model/memory_manager.{h,cc}`, `core/store/model/unified_model_memory.{h,cc}`
- Unified loaders and sinks: `core/store/loader/*` including `DVMPRegionSink`, `UnifiedMemorySink`, `pump{,_ranges}()`

Key changes:

- DVMP becomes the sole owner of DRAM writes/mappings; loader‑side `mmap` fast‑path is removed by default.
- Per‑model sharded synchronization in DVMP eliminates the global mutex bottleneck in `write_at()`, `map_file_segments()`, and state transitions.
- Pin Leases are required for externally visible memory (Communicator registration, zero‑copy readers), including whole‑region CPU registrations.
- IO is positioned and chunk‑aware end‑to‑end; per‑chunk P2P→Disk fallback is mandatory behavior, not best‑effort.
- Optional Linux `userfaultfd` integration enables true demand paging for PAGEABLE_CPU.

The outcome is a simpler mental model, higher throughput under contention, and stronger safety guarantees for external access, while keeping RFC 0001’s performance goals.


## 1. What the Code Does Today (from code, not just docs)

- DVMP allocates a contiguous VA per model and tracks per‑chunk state via atomics (`ChunkMeta`):
  - allocate(): `mmap(PROT_READ|PROT_WRITE, MAP_ANONYMOUS)` and initialize states `COLD`.
  - lock/unlock: transitions to/from `LOCKED_TX` with best‑effort `mlock/munlock`.
  - mark_preemptible(): transitions to `PREEMPTIBLE` and issues `madvise(MADV_FREE|DONTNEED)`.
  - evict_tail_bytes(): tail‑first eviction to `EVICTED` + `madvise(PAGEOUT|DONTNEED)`.
  - ensure_chunk_resident(): returns `kErrChunkRemote` when `EVICTED`.
  - write_at(): memcpy under the DVMP mutex, marks affected chunks `HOT`.
  - map_file_segments(): `mmap(MAP_FIXED)` file windows into DVMP VA, marks `HOT`.

- MemoryManager coordinates DVMP and GPU, plus external exposure:
  - Allocates DVMP region and (optionally) unified per‑chunk tracking via `UnifiedModelMemory`.
  - Uses DVMP lock/unlock for transfers and updates per‑chunk states after copies.
  - Exposes remote access registration to CommunicateEngine; uses `pin_range()` for chunk exports but not for full CPU region registration.

- Unified loader pipeline is already in place:
  - Disk/P2P loaders pump through `UnifiedMemorySink` into `DVMPRegionSink` (CPU) or `GPUMemorySink` (GPU).
  - `DVMPMappedSink` still exists but the disk path now prefers `DVMPRegionSink` positioned writes.
  - P2P can mux to disk (`MuxSeekableSource`) for per‑chunk fallback.

Observed issues and bottlenecks:

- Global DVMP mutex serializes large `write_at()` copies and `mmap` operations across models, penalizing multi‑tenant throughput.
- `write_at()` holds the global mutex during memcpy and metadata updates; long O(N) copies block all other models.
- Loader safety: `DVMPMappedSink` model‑side `mmap(MAP_FIXED)` bypasses DVMP’s write/lock semantics and should be disabled by default.
- External safety: CPU whole‑region registration doesn’t hold a Pin Lease; mapped memory might be evicted or reclaimed while exported.
- Eviction/preemption is manual and tail‑only; there’s no pressure‑driven background service (front‑guard, MaxTotal utilization, etc.).
- Remote fetch path is not completed in code: `UnifiedModelMemory::get_best_source_for_chunk()` doesn’t actually query a GlobalStore.


## 2. Design Goals (trade‑offs)

- Generality: One IO surface (positioned) for all sources, one ownership layer for DRAM.
- Maintainability: Eliminate dual fast paths; reduce interleaved responsibilities; adopt per‑model locking.
- Performance: Remove global DVMP bottlenecks; preserve zero‑copy where safe; enable background preemption/eviction and optional fault‑driven fetch.
- Safety: Memory visible to external agents must be pinned with explicit lifetimes; internal preemption never violates exported contracts.


## 3. Proposed Architecture Changes

### 3.1 DVMP: Per‑Model Concurrency and Ownership

- Replace the single DVMP `mutex_` with a 2‑level scheme:
  - `models_mutex_` protects the `models_` map (insert/erase/lookup only).
  - Each `ModelInfo` gains its own `std::mutex model_mutex` (or `std::shared_mutex` if needed).
  - `ChunkMeta[]` stays atomic; state flips require only the model‑level lock for compound operations.

- `allocate()` creates `shared_ptr<ModelInfo>` stored in the map. Spans returned by `chunk_snapshot()` remain valid; lifetime is tied to the shared_ptr.

- `write_at()` is split into fast and slow phases:
  1) With `models_mutex_`, find `ModelInfo` shared_ptr and drop `models_mutex_`.
  2) Acquire `model_mutex` (short) to snapshot pointers and compute affected chunks.
  3) Release `model_mutex` and do the memcpy without holding any DVMP locks.
  4) Reacquire `model_mutex` and update per‑chunk state and timestamps (atomics), optionally coalescing cache line updates.

- `map_file_segments()` follows the same pattern; `mmap(MAP_FIXED)` is done without holding any DVMP mutex, and metadata is updated under `model_mutex`.

- New optional `begin_write(range) / end_write(range)` internal helpers can set chunks to `LOCKED_TX` around multi‑range writes. `write_at()` uses them if configured to avoid eviction races during long copies.

- State transitions remain single source of truth inside DVMP. `UnifiedModelMemory` mirrors for queries but does not attempt to author states contradicting DVMP. It calls DVMP to lock/unlock and preempt.

### 3.2 Loader/IO: DVMP‑Owned, Positioned, and Unified

- Remove `DVMPMappedSink` usage from production paths (keep behind a debug flag for benchmarking only). The default CPU path uses `DVMPRegionSink::write_at()`.
- Keep `map_file_segments()` as an optimization but call it through DVMP itself at the end of planning, not via a loader‑side `mmap`. Loader requests become a list of `FileSegment`s; DVMP applies them.
- `UnifiedMemorySink` remains the coordinator that updates chunk states post‑IO, delegating CPU state ownership to DVMP.
- Ensure `pump_ranges()` is the default for all partial loads; no sequential fallbacks if a `SeekableSource` is available.

### 3.3 External Visibility: Mandatory Pin Leases

- For CPU whole‑region registration (`MemoryManager::enable_remote_memory_access(PAGEABLE_CPU)`), acquire a `PinLease` covering the entire DVMP VA span and store it in `MemoryManager` for the registration lifetime.
- For chunk‑scoped exports, continue using `export_chunks_for_p2p()` which already acquires pin leases per coalesced range.
- DVMP prevents eviction or preemption of pinned chunks while a lease is active; `evict_tail_bytes()` and `mark_preemptible()` must skip pinned chunks (already implemented).

### 3.4 Background Pressure Engine (DVMP Evictor)

- Introduce a DVMP “Evictor” thread with tunables (front_guard_ratio, max_total_ratio, interval_ms), configurable via `dvmp:` in the runtime config. The Evictor:
  - Reads host memory pressure (MemAvailable / MemTotal) and triggers `evict_tail_bytes()` opportunistically.
  - Applies `mark_preemptible()` to aging chunks per model‑specific watermarks.
  - Respects Pin Leases and `LOCKED_TX`.

- Expose a small public API on DVMP for Evictor control: `start_eviction_loop(cfg)`, `stop_eviction_loop()`, `update_config(cfg)`.

### 3.5 Demand Paging (Optional, Linux‑Only)

- Add an optional `userfaultfd` module for PAGEABLE_CPU to translate page faults into chunk fetches:
  - On `EVICTED` page faults within a model VA, schedule a read via `P2PLoader::pull_chunk()` with Mux fallback to Disk.
  - Integrate with DVMP states: fault handler transitions chunk to `LOCKED_TX`, performs the fetch, then `unlock_chunks()` to `HOT`.
  - Keep disabled by default behind a feature flag due to elevated complexity and kernel requirements.

### 3.6 UnifiedModelMemory: Single Source of Truth and Queries

- Preserve UnifiedModelMemory as a query/tracking helper for GPU state and orchestrating chunk locks before transfers. CPU states are synchronized from DVMP (`sync_cpu_chunk_states()` is already present) and not independently authored.
- Complete `get_best_source_for_chunk()` by allowing an injected `GlobalStoreClient` to resolve remote locations. Prefer LOCAL→REMOTE→DISK, configurable per policy.


## 4. API Changes

### DVMP (`DistributedMemoryPool`)

- Backward‑compatible, but with clarified semantics:
  - `write_at(model, offset, src, bytes)`: Non‑blocking with respect to other models; may set chunks to `LOCKED_TX` internally if configured; guarantees affected chunks become `HOT` on success.
  - `map_file_segments(model, segs)`: Applies segments atomically per‑segment; returns first failure. Metadata updated after mapping.
  - `pin_range(model, offset, bytes, reason)`: Unchanged; documented requirement for external visibility.
  - New (optional): `configure_io_guard({lock_during_write: bool, lock_chunk_on_write: bool})`.
  - New: `start_eviction_loop(cfg)`, `stop_eviction_loop()`, `update_eviction_config(cfg)`.

### MemoryManager

- CPU Comm registration acquires/stores a whole‑region `PinLease` during registration and releases it on unregistration.
- Expose a `dvmp_configure_io_guard(...)` pass‑through for deployments needing stricter write locking.
- Keep current coarse `MemoryState` transitions and unify chunk state interactions through `UnifiedModelMemory` as today.

### Loaders

- `DiskLoader` and `P2PLoader` already use `UnifiedMemorySink` + positioned sinks; ensure `DVMPMappedSink` is not constructed in production code paths.
- For disk zero‑copy, loaders submit `FileSegment` lists to DVMP rather than calling `mmap` locally.


## 5. Performance Impact

- Contention: Per‑model locking removes a global bottleneck, raising multi‑tenant throughput near‑linearly with model count (until IO saturates).
- Copy path: Non‑blocking memcpy in `write_at()` improves overlap; metadata updates are cheap and coalesced.
- Zero‑copy: When `map_file_segments()` applies, throughput matches or exceeds legacy `DVMPMappedSink` while keeping DVMP ownership.
- Eviction: Background evictor smooths tail latency during bursts; prevents OOM thrash.


## 6. Safety and Correctness

- External access safety: Pin Leases become mandatory for any exported memory range; eviction/preemption respect lease refcounts.
- State machine: Only DVMP mutates CPU chunk states; higher levels request transitions via DVMP APIs.
- IO ordering: Positioned writes remain mandatory; `pump_ranges()` is the default for partial loads to avoid corruption under concurrency.


## 7. Migration Plan

- Phase A (no behavior change):
  - Introduce per‑model locks internally; keep existing signatures. Split `write_at()`/`map_file_segments()` into phases without changing call sites.
  - Add whole‑region Pin Lease in `enable_remote_memory_access(PAGEABLE_CPU)`.
  - Guard `DVMPMappedSink` behind a debug flag; switch DiskLoader to DVMP‑owned `map_file_segments()` when page‑aligned.

- Phase B (feature add):
  - Add Evictor thread and config. Default disabled.
  - Inject `GlobalStoreClient` into `UnifiedModelMemory` and finish remote resolution.

- Phase C (optional):
  - Add `userfaultfd` module, flag‑gated.

Each phase is independently releasable and testable.


## 8. Concrete Edits Suggested (code pointers)

- DVMP concurrency refactor:
  - File: `core/common/memory/distributed_memory_pool.cc`
    - Replace class `mutex_` with `models_mutex_` + per‑`ModelInfo::model_mutex`.
    - Rework `write_at()` and `map_file_segments()` to drop locks during memcpy/mmap.
  - File: `core/common/memory/distributed_memory_pool.h`
    - Change `models_` values to `std::shared_ptr<ModelInfo>`; add `model_mutex`.

- MemoryManager safety fixes:
  - File: `core/store/model/memory_manager.cc`
    - In `enable_remote_memory_access(PAGEABLE_CPU)`, acquire a whole‑region `PinLease` and store into `cpu_pin_leases_`.
    - Expose `dvmp_configure_io_guard` pass‑through.

- Loader ownership alignment:
  - File: `core/store/loader/disk_loader.cc`
    - For page‑aligned partitions, build `DVMP::FileSegment` list and call `dvmp->map_file_segments()` instead of any `mmap` in the loader.
  - Remove production usage of `DVMPMappedSink` (keep for benchmarks under a build flag).

- GlobalStore integration:
  - File: `core/store/model/unified_model_memory.{h,cc}`
    - Add injectable `GlobalStoreClient*` and implement `get_best_source_for_chunk()` lookup.

- Evictor:
  - New files under `core/common/memory/dvmp_eviction_{h,cc}` wired into DVMP.

- Optional `userfaultfd` module:
  - New files under `core/common/memory/dvmp_uffd_{h,cc}` plus feature flag plumbing.


## 9. Testing Strategy

- Unit: DVMP concurrency (parallel `write_at()` across distinct models), state transitions, pin lease refcounts, Evictor policy application.
- Integration: Disk→CPU positioned writes; P2P→CPU with mux fallback; GPU copy and CPU auto‑preemption policies.
- Soak: Multi‑model concurrent loads with and without Evictor; verify throughput and lack of global lock contention.
- Fault injection: `userfaultfd` path behind flag, verifying demand‑loaded chunks and correct state transitions under stress.


## 10. Risk Analysis and Mitigations

- Memory visibility races during unlocked memcpy: mitigated via optional `LOCKED_TX` around writes and atomic state updates after copies; pin leases prevent eviction on exported ranges.
- Complexity of `userfaultfd`: kept optional; strict isolation from default paths.
- Behavior drift vs 0003: This RFC formalizes DVMP as the single owner for IO and external safety while keeping all positioned write benefits from 0003.


## 11. Summary

This proposal finishes the DVMP unification story by:
- Making DRAM IO unequivocally DVMP‑owned,
- Removing global lock contention via per‑model concurrency,
- Enforcing external safety with mandatory Pin Leases,
- Completing chunk‑aware P2P→Disk fallback and remote resolution,
- Adding an optional demand‑paging path for future growth.

It strikes a deliberate balance: the architecture is simpler to reason about, easier to maintain, and demonstrably higher‑performance under real multi‑tenant workloads.