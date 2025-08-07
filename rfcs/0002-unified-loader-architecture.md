# 0002 — Unified Loader Architecture (Final Revision 2025-08-06)

> **Status**: Accepted – supersedes all previous drafts.

## 1  Background & Motivation

The existing `DiskLoader` (≈ 1 200 LoC) and `P2PLoader` (≈ 650 LoC) are **feature-rich but monolithic**.  Each loader:

* hand-rolls its own producer/consumer logic
* embeds details about I/O alignment, CUDA streams, DVMP, chunk state transitions, etc.
* is hard to reason about and harder to extend (zero-copy, adaptive buffering, progress hooks …)

The goal of this RFC is to present **production-grade abstractions** that **completely replace** today’s implementation without legacy fall-backs.

Key principles (from the StepCast Common Guidelines) remain unchanged:

1. **Occam’s Razor** – minimum concepts.
2. **Data-Flow First** – loading is a *stream* from *some storage* into *model memory*.
3. **No Hidden Work** – all concurrency and resource ownership is explicit.
4. **Stable Public API** – `Model::ensure_loaded_async()` keeps its signature.

Compatibility shims that existed in earlier drafts (dual code paths, etc.) are removed – we are doing a **clean-cut refactor**.

---

## 2  High-Level Architecture

```
Storage / Remote ─► Source ─► Pump ─► Sink ─► UnifiedModelMemory
                        ▲         │
                        │         └──► MemoryManager (state / DVMP / pool)
                        │
             BufferPool (owned by Pump)
```

### 2.1  Core Concepts

```cpp
// source.h — *zero ownership* over buffers
class Source {
 public:
  virtual ~Source() = default;
  // Read **≤ pool.chunk_size()** bytes into `dst`.
  // Returns {bytes_read > 0} or {0 == EOF}.
  virtual absl::StatusOr<size_t> read(void* dst) = 0;
};

// SeekableSource — optional extension for random / chunk-wise access
class SeekableSource : public Source {
 public:
  // Same semantics as pread().
  virtual absl::StatusOr<size_t> read_at(uint64_t offset,
                                         void*   dst,
                                         size_t  bytes) = 0;
};

// sink.h
class Sink {
 public:
  virtual ~Sink() = default;
  // Write `bytes` starting at the *current* logical position.
  virtual absl::Status write(const void* src, size_t bytes) = 0;
  // Flush / finalise – idempotent, must leave target in valid state on error.
  virtual absl::Status close() { return absl::OkStatus(); }
};

// buffer_pool.h – owns *host-pinned* or *pageable* chunks
class BufferPool {
 public:
  virtual ~BufferPool() = default;
  virtual size_t        chunk_size() const = 0;
  virtual int           capacity()   const = 0;  // #chunks
  // Obtain exclusive chunk for producer.
  virtual absl::StatusOr<int> get_free_chunk()   = 0;
  virtual void               return_chunk(int)  = 0;
  // Producer marks chunk ready; consumer later reclaims.
  virtual absl::Status mark_chunk_ready(int slot,
                                        uint64_t global_chunk_idx,
                                        size_t   valid_bytes) = 0;
  virtual absl::StatusOr<ReadyChunk> get_ready_chunk() = 0;  // blocks
  virtual void signal_production_complete() = 0;             // non-blocking
};

// ---------------------------------------------------------------------------
// NOTE ON STREAMINGPINNEDBUFFER
// ---------------------------------------------------------------------------
// `StreamingPinnedBuffer` already exists in production code today and owns a
// ring-buffer of host-pinned chunks while exposing two FIFO queues (free ↔ ready)
// plus an EOS flag.  The refactor makes this type the *sole* concrete
// implementation of `BufferPool` – **all other pinned-memory helpers (`PinnedMemory`,
// `PinnedChunkPool`, etc.) are deleted.**
//
// Key consequences:
//   • Loaders never call `MemoryManager::get_pinned_memory()` or any "raw"
//     pinned-memory API.  They *only* interact with the abstract `BufferPool`.
//   • `StreamingPinnedBuffer` is used *exclusively* as a *transient* staging area
//     to optimise H2D/D2H bandwidth.  Model data is never considered "resident"
//     inside the buffer – it always flows Source→Pool→Sink in a single pass.
//   • No caller retains pointers obtained from `BufferPool` after the corresponding
//     slot is returned.  This allows future allocator changes (e.g. slab vs. VM-based)
//     without ABI breakage.
// ---------------------------------------------------------------------------
```

`ReadyChunk` is a trivial POD (`void* data_ptr; uint64_t global_chunk_id; size_t bytes_in_chunk; int slot_id;`).

### 2.2  Pump – The Only Piece Owning Threads

```cpp
absl::Status pump(Source&      src,
                  Sink&        dst,
                  BufferPool&  pool,
                  int          concurrency = 2);

absl::Status pump_ranges(SeekableSource& src,
                         Sink&          dst,
                         BufferPool&    pool,
                         absl::Span<const std::pair<uint64_t, size_t>> ranges,
                         int concurrency = 2);
```

Responsibilities:

1. Spawn **`concurrency` producers + 1 consumer**.
2. Handle back-pressure via `BufferPool`.
3. Respect cancellation via propagated `absl::Status`.
4. Guarantee that `dst.close()` is called exactly once.

Pump never calls CUDA – that belongs to `GPUMemorySink`.

### 2.3  Specialised Implementations

| New type | Layer | Key duties |
|----------|-------|------------|
| `FilePartitionSource`        | Source          | DIRECT IO alignment, multi-file mapping, supports `SeekableSource`. |
| `RemoteKeySource`            | Source          | Wraps CommunicateEngine (`read_tensor`) for RDMA/TCP. |
| `StreamingPinnedBuffer`      | BufferPool      | The sole host-pinned queue used for both Disk and P2P loaders. 64–512 KiB aligned. |
| `GPUMemorySink`              | Sink            | Async H2D on per-pump CUDA stream. |
| `DVMPMappedSink`             | Sink            | `mmap` partitions into DVMP VA using `MAP_FIXED`, then `close()`: lock/unlock chunks to HOT. |
| `UnifiedMemorySink`          | Sink wrapper    | After successful copy/update, writes new `ChunkState` via `UnifiedModelMemory`. |

### 2.4  Fast-Path Bypasses

* **Disk → PAGEABLE_CPU** (all partitions ≥ page size):
  `DiskLoader` may directly construct a `DVMPMappedSink` and call its `map_partitions()` helper – **no Pump** necessary.
* **Zero-copy CPU→GPU** (future): detect at runtime that Source & Sink share VA space, skip intermediate host pool.

---

## 3  Mapping to Current Codebase

| Loader call-site | Today | After refactor |
|------------------|-------|---------------|
| `load_async(GPU)` (disk) | 520 LoC custom pipeline | `FilePartitionSource` → `StreamingPinnedBuffer` → Pump → `GPUMemorySink` → `UnifiedMemorySink` |
| `load_async(PAGEABLE_CPU)` (disk, ≥ page size) | mmap loop | `DVMPMappedSink::map_partitions()` (≈ 40 LoC) |
| `load_async(GPU)` (remote) | CommunicateEngine loop | `RemoteKeySource` → Pump → `GPUMemorySink` → `UnifiedMemorySink` |
| `load_async(PAGEABLE_CPU)` (remote) | manual loop | `RemoteKeySource` → Pump → `DVMPMappedSink` |
| `load_chunks_async(..)` | duplicated logic | `pump_ranges()` + `SeekableSource` |

All corner-cases handled by today’s code (DIRECT IO, page misalignment, CUDA streams, DVMP hot-locking, chunk state propagation) are preserved but **moved into specialised classes**.

---

## 4  Detailed Behavioural Contracts

1. **Alignment & I/O flags** – `FilePartitionSource` decides whether to open files with `O_DIRECT` based on (`total_size > 5 GiB` && `blk_size ≥ 512`).  It also performs the 512 B padding dance already present in `FilePartitionReader`.
2. **CUDA Stream Ownership** – `GPUMemorySink` creates **one** non-blocking stream per pump invocation and destroys it in `close()` after `cudaStreamSynchronize()`.
3. **MemoryManager Interaction**
   * Loaders never mutate allocations; they **delegate** to `MemoryManager`.
   * `UnifiedMemorySink` is the only place updating `ChunkState` for *partial* loads (`load_chunks_async`).
4. **Error Propagation** – The first non-OK `absl::Status` aborts all producer threads, waits on them, then aborts consumer, then flushes `dst.close()` and unwinds.
5. **Thread Count Heuristic** – default: `min(hw_concurrency, 4)` – same as today.
6. **Chunk Size Source of Truth** – `BufferPool::chunk_size()` is supplied by `MemoryManager`; all calculations use that single value.

---

## 5  Reference LOC Comparison

* Source/Sink/Pool interfaces       ≃ 120 LoC
* Helpers (`pump*`)                 ≃ 180 LoC
* Disk specialisations              ≃ 250 LoC (vs 1 200)
* P2P specialisations               ≃ 190 LoC (vs 650)

Overall **–60 % reduction** inside `core/store/loader`, yet behaviour-equivalent.

---

## 6  Migration Plan

Because **no backwards-compatibility is required**, the refactor is executed in
just *two* atomic PRs, each leaving the tree green:

1.  **Core Abstractions + Disk Path (≈ –900 LoC).**
   • Add `Source`, `SeekableSource`, `Sink`, `BufferPool`, Pump helpers.
   • Replace all logic inside `disk_loader.*` with the new pipeline.
   • Delete `FilePartitionReader`, custom producer/consumer loops, and *all*
     pinned-memory helpers except `StreamingPinnedBuffer`.

2.  **Remote Path (≈ –300 LoC).**
   • Implement `RemoteKeySource` and port `p2p_loader.*` onto Pump.
   • Remove bespoke CommunicateEngine loops and any residual chunk staging
     code.

Both PRs include exhaustive unit-tests and docs updates; no intermediate
compatibility shims or runtime flags are introduced.

---

## 7  Open Extensions

1. **Zero-copy CPU→GPU** – detect if host & device share UVA; `GPUMemorySink` can then simply hand the pointer to CUDA.
2. **Adaptive buffer sizing** – Pump could monitor throughput and ask `BufferPool` to grow/shrink.
3. **Progress hooks / tracing** – trivial: add `std::function<void(size_t bytes)>` callbacks in Pump.

---

## 8  Success Criteria

* **Zero business logic in concrete loaders** – each `*Loader::load_*` is ≤ 20 LoC and delegates to Pump.
* **Unit-test coverage ≥ 95 %** for `core/store/loader` (pump variants, sources, sinks).
* **Throughput ≥ 98 %** of baseline for a 10 GB GPT-2 model on NVMe + PCIe Gen4 GPU.
* **No additional host memory** vs 2025-06-01 HEAD (verified with /proc/self/smaps).
* **Binary size –50 %** for `libstore_loader.so` thanks to code deletion.

---

## 9  Appendix – Why This Covers Current Edge-Cases

* **DIRECT IO alignment** – isolated in `FilePartitionSource`.
* **Pinned buffer reuse** – handled by `PinnedChunkPool`, reusing `MemoryManager` allocations.
* **CUDA stream lifecycle** – local to `GPUMemorySink`, eliminating duplication.
* **DVMP fast-path** – small 40 LoC helper, maintained for page-aligned partitions.
* **Chunk-aware loads** – `pump_ranges()` + `SeekableSource` replace ~350 LoC across both loaders.
* **CommunicateEngine plumbing** – encapsulated by `RemoteKeySource`, so networking stays outside core logic.

---

## 10  Implementation Phases

#### Phase 1: Core Abstractions (~120 LoC)
**New files:**
- `core/store/loader/source.h` - Base interfaces
- `core/store/loader/sink.h` - Sink interface
- `core/store/loader/buffer_pool.h` - BufferPool interface
- `core/store/loader/pump.{h,cc}` - Orchestration logic

#### Phase 2: Disk Path (~250 LoC, removes ~900 LoC)
**New files:**
- `core/store/loader/file_partition_source.{h,cc}` - O_DIRECT handling
- `core/store/loader/gpu_memory_sink.{h,cc}` - CUDA transfers
- `core/store/loader/dvmp_mapped_sink.{h,cc}` - mmap fast-path
- `core/store/loader/unified_memory_sink.{h,cc}` - ChunkState wrapper

**Modifications:**
- Adapt `StreamingPinnedBuffer` via adapter
- Reduce `disk_loader.cc` to ~20 LoC
- Delete `FilePartitionReader` entirely

#### Phase 3: Remote Path (~190 LoC, removes ~300 LoC)
**New files:**
- `core/store/loader/remote_key_source.{h,cc}` - CommunicateEngine wrapper

**Modifications:**
- Reduce `p2p_loader.cc` to ~20 LoC
- Remove custom transfer loops

#### Key Pitfalls to Avoid

1. **O_DIRECT Misalignment** - Preserve 512B alignment logic from `FilePartitionReader`
2. **BufferPool Slot Leaks** - Return slots immediately after use
3. **Cancellation Deadlocks** - Check `should_stop` in all blocking operations
4. **Mixed Concerns** - Keep Source/Sink/BufferPool interfaces pure

---

## 11  Execution Status



