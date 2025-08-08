# RFC 0003 — DVMP‑Unified IO, Pin Leases, and Chunk‑Level P2P→Disk Fallback

Status: Proposed
Date: 2025‑08‑08
Owner: StepCast Core
Depends on: 0001 (DVMP), 0002 (Unified Loader Architecture)
Supersedes: none

## 0. Motivation

We need one authoritative layer to manage all model DRAM operations — mapping, residency, eviction, and safe external visibility. Today, DiskLoader’s `dvmp_mapped_sink` maps partition files directly into the DVMP‑reserved VA (`mmap(MAP_FIXED)`), which cedes practical paging control to the kernel. That breaks three core needs:

- P2P‑first policy: We want to try P2P for chunks and only fallback to disk per failed chunk; kernel faults bypass this policy.
- External safety: Memory registered to Communicator (RDMA/TCP) must remain valid; DVMP cannot guarantee this when OS paging is in control.
- Coherent lifecycle: Eviction/preemption must be DVMP‑governed, not implicit via a page cache.

Additionally, the unified loader (RFC 0002) has a correctness gap: `pump_ranges` writes without a destination offset, so concurrent producers may corrupt output ordering. Chunk‑level P2P→disk fallback is not end‑to‑end in code.

This RFC folds all DRAM writes and file mappings under DVMP, introduces “Pin Leases” to protect externally visible memory, fixes range writes via positioned sinks, and completes chunk‑level P2P→disk fallback. As requested, we do not preserve backward compatibility; we implement “the best” design.

## 1. Summary of Changes

- DVMP becomes the single owner of DRAM writes and file mappings (no loader‑side `mmap`).
- New DVMP APIs: `map_file_segments`, `write_at`, `pin_range` (Pin Lease RAII), and contiguous chunk export helpers.
- Replace `dvmp_mapped_sink` with a DVMP‑backed `DVMPRegionSink` that writes at offsets; remove mmap fast‑path by default.
- Fix `pump_ranges` with `PositionedSink::write_at(offset, ...)` to guarantee correct placement under concurrency.
- Enable per‑chunk P2P→disk fallback using a muxing `SeekableSource` and a planner driven by `UnifiedModelMemory`.
- Export only the pinned subset of chunk ranges to Communicator; unexport unlocks and depins cleanly.

## 2. Scope and Non‑Goals

In scope:
- DVMP‑owned mapping/writes; Pin Leases.
- Positioned sinks and corrected `pump_ranges` contract.
- Per‑chunk P2P→disk fallback implemented in loaders.
- Chunk‑scoped external export/unexport in `MemoryManager`.

Out of scope:
- Changes to `CommunicateEngine` wire protocol.
- Transparent remote page‑fault refill (userfaultfd) in this iteration.

## 3. Current State (Code References)

- DVMP: `core/common/memory/distributed_memory_pool.{h,cc}`
  - Reserves contiguous VA; tracks `ChunkMeta` (`HOT`, `LOCKED_TX`, `COPIED_GPU`, `COLD`, `EVICTED`, `PREEMPTIBLE`).
  - `lock_chunks`, `unlock_chunks`, `evict_tail_bytes`, `mark_preemptible`, `ensure_chunk_resident`.
- Unified Memory: `core/store/model/unified_model_memory.{h,cc}`
  - Tracks per‑chunk state across CPU/GPU; `get_missing_chunks`, `get_best_source_for_chunk`, `lock_chunks_for_transfer`, `update_chunk_states`.
- MemoryManager: `core/store/model/memory_manager.{h,cc}`
  - Owns DVMP region, streaming pinned buffer, CUDA stream; registers CPU/GPU memory with Communicator.
- Loaders: `core/store/loader/*`
  - Sources: `file_partition_source`, `remote_key_source`
  - Sinks: `gpu_memory_sink`, `dvmp_mapped_sink`, `unified_memory_sink`
  - Orchestration: `pump.{h,cc}`
- Communicator: `core/communicator/engine/engine.h`
  - Registers partitions/buffers and serves P2P reads.

## 4. Design

### 4.1 DVMP‑Owned IO APIs

Add authoritative DVMP methods; DVMP updates chunk timestamps/states as it maps or writes:

- `map_file_segments(model_id, segs)`: map a set of file segments into the reserved VA using `MAP_FIXED|MAP_PRIVATE` and `PROT_READ` (optional `MAP_POPULATE`). Updates `last_touch_s`; sets relevant chunks `HOT`.
- `write_at(model_id, va_offset, src, bytes)`: copy bytes into anonymous pages (`PROT_READ|PROT_WRITE`), update `last_touch_s`; set affected chunks `HOT`.

API sketch (header-level only):

- File: `core/common/memory/distributed_memory_pool.h`

```
namespace stepcast::memory {

struct FileSegment {
  std::filesystem::path path;
  uint64_t file_offset;  // file offset in bytes
  uint64_t va_offset;    // destination VA offset (within model region)
  uint64_t length;       // bytes to map
  bool populate{false};  // MAP_POPULATE hint
};

class DistributedMemoryPool {
 public:
  absl::Status map_file_segments(std::string_view model_id, absl::Span<const FileSegment> segs);
  absl::Status write_at(std::string_view model_id, uint64_t va_offset, const void* src, size_t bytes);
  // ... existing API ...
};

} // namespace stepcast::memory
```

Rules:
- DVMP verifies segment ranges fall within the reserved VA and chunk boundaries are respected; it may segment a range internally across chunks to update per‑chunk metadata.
- Only DVMP is permitted to call `mmap(MAP_FIXED)` into model VA.

Why: Centralize all VA mappings/writes to ensure policy and state coherence.

### 4.2 Pin Leases (DVMP‑Level)

Separate transient transfer locks from external visibility pins:

- `LOCKED_TX`: short‑lived locks used for H→D or P2P transfer concurrency control (`lock_chunks`/`unlock_chunks` remain).
- Pin Leases: protect chunks while exposed to peers; pinned chunks are never evicted or marked PREEMPTIBLE regardless of state.

API sketch:

```
class PinLease {
 public:
  ~PinLease();
  PinLease(PinLease&&) noexcept;
  PinLease& operator=(PinLease&&) noexcept;
  PinLease(const PinLease&) = delete;
  PinLease& operator=(const PinLease&) = delete;
};

absl::StatusOr<PinLease> pin_range(std::string_view model_id,
                                   uint64_t va_offset,
                                   uint64_t bytes,
                                   std::string_view reason /* "ExternalShare" | "InternalIO" */);
```

Semantics:
- Best‑effort `mlock` per chunk; if `mlock` fails (`ENOMEM`/`EPERM`), DVMP still increments a per‑chunk pin refcount and blocks DVMP evictions for those chunks. TCP can fault demand pages; RDMA typically pins via verbs during registration.
- `evict_tail_bytes` and `mark_preemptible` skip chunks with pin refcount > 0.
- Lease destructor decrements refcounts and `munlock`s as needed.

### 4.3 Positioned Sinks and Pump Ranges

Fix `pump_ranges` correctness by moving to destination‑offset‑aware writes:

- Add `PositionedSink`:

```
class PositionedSink {
 public:
  virtual ~PositionedSink() = default;
  virtual absl::Status write_at(uint64_t offset, const void* src, size_t bytes) = 0;
};
```

- Change `pump_ranges(SeekableSource&, PositionedSink&, ...)` to call `dst.write_at(offset, ptr, bytes)` for each range.
- Extend `gpu_memory_sink` to implement `PositionedSink` (compute `gpu_base + offset`, issue H2D `memcpy_async`).
- Replace the default CPU sink with `DVMPRegionSink` (compute `dvmp_cpu_base + offset`, `memcpy`).

This ensures concurrent producers cannot corrupt destination order; consumer simply writes at the designated global offset.

### 4.4 Chunk‑Level P2P→Disk Fallback

Planner:
- For a target (CPU/GPU), get missing chunks: `unified_memory.get_missing_chunks(key, target, device_id)`
- Rank source per chunk: `get_best_source_for_chunk` (prefer local, else P2P, else disk).
- Group chunks by source type and coalesce into contiguous byte ranges to amortize I/O calls.

Sources:
- Reuse `RemoteKeySource` and `FilePartitionSource`.
- Add `MuxSeekableSource` wrapper that tries P2P on `read_at`, and if short or non‑OK, completes the remainder with disk; returns full requested bytes or error.

Execution per group:
1) `unified_memory.lock_chunks_for_transfer(key, source, target, chunks)`
2) `pump_ranges(mux_or_single_source, positioned_sink, buffer_pool, ranges, concurrency)`
3) `unified_memory.update_chunk_states(key, target, chunks, new_state, device_id)`
4) On partial failures, retry failed chunks with disk source only

Outcome:
- Only the failing chunks fall back to disk; healthy chunks take the fast P2P path.
- Works equivalently for CPU and GPU targets (sinks differ).

### 4.5 Safe External Access (Export / Unexport)

MemoryManager gains chunk‑scoped export helpers using DVMP Pin Leases:

- `export_chunks_for_p2p(location, chunks, ce)`:
  - DVMP pin leases over the contiguous byte ranges for the requested chunks.
  - For each contiguous block, register a Communicator tensor key (CPU: multiple keys; GPU: typically single key).
  - Return `CommRegistrationInfo` with key list and sizes.

- `unexport_chunks_for_p2p(location, chunks, ce)`:
  - Unregister keys for the chunks’ blocks; release pin leases.

Policy:
- CPU default moves from whole‑region registration to chunk‑scoped export.
- GPU remains single contiguous registration, but we can still add chunk granularity later if needed.

### 4.6 State Machine Integration

- DVMP chunk states remain as in RFC‑0001; `write_at`/`map_file_segments` set modified chunks `HOT` and refresh timestamps.
- `lock_chunks_for_transfer` transitions source chunks as needed; after GPU copy, `update_chunk_states(..., COPIED_GPU)` and call DVMP `unlock_chunks(..., copied_gpu = true)`.
- PREEMPTIBLE remains a hint; pinned chunks are immune to PREEMPTIBLE/EVICT.

## 5. Breaking Changes (Intentional)

- `pump_ranges` signature change: requires a `PositionedSink` instead of simple `Sink`, and passes offsets. Existing sinks must implement `write_at`.
- `dvmp_mapped_sink` removed from default flow; replaced by `DVMPRegionSink` calling DVMP `write_at` (no implicit `mmap`).
- CPU external exposure defaults to chunk‑scoped export; whole‑region registration is removed or left as an explicit compatibility shim (off by default).

## 6. Detailed API Changes (By File)

- `core/common/memory/distributed_memory_pool.h/.cc`
  - Add `struct FileSegment { path, file_offset, va_offset, length, populate }`
  - Add `absl::Status map_file_segments(std::string_view, absl::Span<const FileSegment>)`
  - Add `absl::Status write_at(std::string_view, uint64_t va_offset, const void* src, size_t bytes)`
  - Add Pin Lease API:
    - `class PinLease { ... }`
    - `absl::StatusOr<PinLease> pin_range(std::string_view, uint64_t va_offset, uint64_t bytes, std::string_view reason)`
  - Enforce eviction/preemption skip for pinned chunks.

- `core/store/loader/sink.h`
  - Add `class PositionedSink` with `write_at`.

- `core/store/loader/pump.h/.cc`
  - Change `pump_ranges(SeekableSource&, PositionedSink&, ...)` to feed `write_at(offset, ...)`.
  - Keep `pump(Source&, Sink&, ...)` unchanged for purely sequential flows.

- `core/store/loader/gpu_memory_sink.{h,cc}`
  - Implement `PositionedSink::write_at(offset, ...)` (compute `gpu_base + offset`, H2D copy).
  - Keep `write()` for `pump(...)`.

- `core/store/loader/dvmp_region_sink.{h,cc}` (new)
  - Implements `PositionedSink`, computing `dvmp_cpu_base + offset`, performing `memcpy`.
  - Reports errors upon overrun or null base.

- `core/store/loader/dvmp_mapped_sink.{h,cc}`
  - Remove from default loader paths; optionally delete file (no compatibility required).

- `core/store/loader/mux_seekable_source.{h,cc}` (new)
  - Wrap `RemoteKeySource` and `FilePartitionSource`; on `read_at`, try P2P then disk for any remainder.

- `core/store/loader/disk_loader.cc`, `core/store/loader/p2p_loader.cc`
  - For CPU/GPU targets:
    - Query `unified_memory.get_missing_chunks(...)`
    - Plan per‑chunk sources using `get_best_source_for_chunk`
    - Group contiguous byte ranges; run `pump_ranges` with appropriate `PositionedSink`
    - Surround with `lock_chunks_for_transfer`/`update_chunk_states`
  - Remove direct `dvmp_mapped_sink` usage.

- `core/store/model/memory_manager.{h,cc}`
  - Add:
    - `absl::StatusOr<CommRegistrationInfo> export_chunks_for_p2p(ModelLocation, absl::Span<const uint32_t>, CommunicateEngine&)`
    - `absl::Status unexport_chunks_for_p2p(ModelLocation, absl::Span<const uint32_t>, CommunicateEngine&)`
  - Internally: compute VA ranges per contiguous chunk group, acquire DVMP Pin Leases, register keys, and store handles for unexport.

## 7. Data Flow (Typical)

Disk → CPU:
- FilePartitionSource (O_DIRECT) → pump_ranges → DVMPRegionSink (write_at) → UnifiedMemorySink (update chunks HOT)

P2P → CPU:
- RemoteKeySource → (via Mux for fallback) → pump_ranges → DVMPRegionSink → UnifiedMemorySink (update chunks HOT)

CPU → GPU:
- UnifiedModelMemory: determine missing chunks on GPU
- lock_chunks_for_transfer(CPU→GPU)
- StreamingPinnedBuffer + pump_ranges with `gpu_memory_sink::write_at(offset, ...)`
- update_chunk_states(..., COPIED_GPU); DVMP unlock(copied_gpu=true)

Export CPU to peers:
- MemoryManager.export_chunks_for_p2p:
  - DVMP.pin_range(VA group) → Communicator.register_tensor per group → peers read safely
- Unexport: unregister → release pin leases

## 8. Implementation Plan (No Compatibility Mode)

1) Positioned Writes (correctness first)
- Add `PositionedSink`; update `pump_ranges`.
- Implement in `gpu_memory_sink` and add `dvmp_region_sink`.
- Remove default usage of `dvmp_mapped_sink`.

2) DVMP IO & Pins
- Implement `map_file_segments`, `write_at`.
- Implement `PinLease` (`pin_range`) with per‑chunk refcounts and best‑effort `mlock`.

3) Chunk Export/Unexport
- Add APIs to `MemoryManager`; integrate DVMP pin leases; register coalesced VA groups to `Communicator`.

4) Fallback and Loader Integration
- Add `MuxSeekableSource`; update both loaders to per‑chunk planning and to call positioned sinks.

5) Validation
- Unit: `pump_ranges` with out‑of‑order producers writes correctly at offsets.
- E2E: partial P2P failures fallback to disk per chunk; final chunk set present; states updated (CPU: HOT, GPU: COPIED_GPU).
- P2P safety: only exported chunks readable; unexport blocks remote reads.
- Perf: regression guard vs RFC‑0002 throughput numbers.

## 9. Risks & Mitigations

- Increased DVMP surface (mapping + pins):
  - Keep API small, focused; avoid general I/O logic in DVMP (no read paths).
- More registrations for chunk exports:
  - Coalesce adjacent chunks into larger VA groups; cache registrations over session if access patterns are stable.
- Loader change complexity:
  - RFC‑0002 abstractions remain; we only add positioned writes and a muxed source.

## 10. Acceptance Criteria

- Correctness: concurrent `pump_ranges` never misplaces output; all writes at correct offsets.
- Fallback robustness: P2P failure on any chunk triggers disk fallback only for those chunks; whole load completes if disk available.
- External safety: Remote peers can read only while DVMP holds Pin Leases; PREEMPTIBLE/EVICTED never invalidates exported pages.
- Performance: Throughput ≥ 98% of RFC‑0002 for 10–50 GB models on NVMe + PCIe Gen4 GPU; no additional steady‑state host memory.
- Observability: Metrics
  - `pin_leases_total{reason}`
  - `chunk_exports_total{location}`
  - `fallback_chunks_total{reason}`
  - `loader_bytes_total{source,location}`

## 11. References (In‑Repo)

- DVMP: `core/common/memory/distributed_memory_pool.{h,cc}`
- Unified memory: `core/store/model/unified_model_memory.{h,cc}`
- Memory manager: `core/store/model/memory_manager.{h,cc}`
- Loaders: `core/store/loader/*.cc,*.h` (pump, sources, sinks)
- Communicator: `core/communicator/engine/engine.h`
- Model façade: `core/store/model/model.{h,cc}`

## 12. Notes on Future Work

- Transparent remote/CPU page fault handling (userfaultfd) to auto‑refill EVICTED chunks from P2P/disk without prefetch.
- Adaptive export coalescing guided by peer access telemetry.
- Zero‑copy CPU→GPU with UVA detection, extending `gpu_memory_sink::write_at` to sometimes pass host pointers directly when legal.

## 13. Appendix — Offset ↔ Chunk Mapping

- DVMP chunk size: `DistributedMemoryPool::kChunk` (currently 256 MiB).
- Given `va_offset` and `bytes`:
  - First chunk index: `i0 = va_offset / kChunk`
  - Last chunk index: `i1 = (va_offset + bytes - 1) / kChunk`
  - Per affected chunk, update timestamp and state (`HOT`) on successful write or mapping.
- Export grouping: coalesce consecutive chunk indices into maximal contiguous VA ranges to minimize registration count.

---

## Execution Status (2025-08-08)

- Overall: Major features are implemented and integrated; metrics (pin leases, chunk exports, mux fallbacks, loader bytes) are wired; one new unit test validates mux fallback; remaining work is optional planner refinement and additional tests. See mapping below.

- Implemented (by RFC section):
  - 4.1 DVMP‑Owned IO: `DistributedMemoryPool::{write_at,map_file_segments}` added; writes update per‑chunk state/timestamps; only DVMP uses `MAP_FIXED`.
  - 4.2 Pin Leases: `pin_range(...)` with per‑chunk refcounts + best‑effort `mlock`; eviction/preemptible skips pinned chunks; RAII release wired.
  - 4.3 Positioned Sinks: `PositionedSink` introduced; `pump_ranges(...)` uses destination offsets; `gpu_memory_sink::write_at(...)` and new `DVMPRegionSink` implemented; default CPU path now uses `DVMPRegionSink` (no implicit `mmap`).
  - 4.4 P2P→Disk fallback: `MuxSeekableSource` added and integrated in `P2PLoader` (full and chunked). Reads try P2P first and complete any remainder from disk.
  - 4.6 State integration: DVMP writes/maps mark chunks HOT and refresh timestamps; preemptible/evict paths skip pinned; existing lock/unlock semantics preserved.

- Partially implemented:
  - 4.5 Safe external access: Added `MemoryManager::{export_chunks_for_p2p,unexport_chunks_for_p2p}` (CPU). These coalesce requested chunks, acquire DVMP pin leases, and register per‑range keys. Default `enable_remote_memory_access(PAGEABLE_CPU)` still registers the whole region; flipping the default to chunk‑scoped export is a follow‑up.
  - Planner refinement (4.4): We provide per‑range P2P→disk mux at the source level. A full plan that ranks sources per chunk and coalesces by source type can be layered on top; not required for functional correctness.

- Clean‑ups (compat removals):
  - Removed `UnifiedMemorySink` fallback to sequential writes when `PositionedSink` is unavailable; positioned writes are now required for correctness.
  - Removed `P2PLoader::pull_chunk()` which performed DVMP‑bypassing direct writes; all CPU writes go through `DVMPRegionSink`/`DVMP::write_at`.

- Usage notes:
  - Disk fallback opt‑in: set `SCSTORE_FALLBACK_MODEL_DIR=/path/to/model_dir` where `tensor.data*` partitions exist. P2P pipelines will automatically fallback per‑range on short/error reads.
  - Export/unexport API: callers can export only the needed chunk groups on CPU, then unexport to release leases and unregister keys.

- Code map (high‑level):
  - DVMP IO & pins: `core/common/memory/distributed_memory_pool.{h,cc}`
  - Positioned sinks & pump: `core/store/loader/sink.h`, `core/store/loader/pump.{h,cc}`
  - GPU sink: `core/store/loader/gpu_memory_sink.{h,cc}`
  - CPU DVMP sink: `core/store/loader/dvmp_region_sink.{h,cc}`
  - Mux source & integration: `core/store/loader/mux_seekable_source.{h,cc}`, `core/store/loader/p2p_loader.cc`
  - CPU export APIs: `core/store/model/memory_manager.{h,cc}`

- Build/Test status:
  - Bazel: a workspace external repo issue (`@hedron_compile_commands`) may still block a full `//...` build; targeted builds for loaders, DVMP, and unit tests pass.
  - Added unit test `tests/cpp/unit:mux_seekable_source_test` covering primary‑error fallback to disk; further tests for `pump_ranges` offsets and pin‑lease eviction immunity are planned.

- Metrics & acceptance:
  - Metrics hooked:
    - `pin_leases_total{reason}` in DVMP `pin_range()`
    - `chunk_exports_total{location}` in `MemoryManager::export_chunks_for_p2p()`
    - `fallback_chunks_total{reason}` in `MuxSeekableSource::read_at()` on `primary_error|short_read`
    - `loader_bytes_total{source,location}` in DiskLoader and P2PLoader (full + chunked)
  - Acceptance targets (throughput/regression guard) to be validated post‑build with 10–50 GB test artifacts.

- Deviations from “no compatibility mode” (intentional/pragmatic):
  - We kept the existing whole‑region CPU registration in `enable_remote_memory_access` for now and added chunk‑scoped export as an explicit API. Default flip can be done after consumers are updated.
  - `map_file_segments` exists in DVMP but is not used by default loader paths (which stream via `write_at`). It remains available for optional “fast import” paths if needed.

- Next steps (tracked):
  - Flip CPU default exposure to chunk‑scoped export once downstreams adopt the new API.
  - Add unit tests for `pump_ranges` offset correctness and pin‑lease behavior; add perf regression guard against RFC‑0002 numbers.
  - Optional: introduce a small ranking/planning layer to choose P2P/disk per chunk ahead of pumping (current mux suffices functionally).
