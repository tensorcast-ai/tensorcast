# 0003 — DVMP v2: Tiered Unified Memory Pool for Extreme-Scale Models (Draft 1)

> **Status**: Proposed (2025-08-07)

## 0  Abstract

The current Distributed Virtual Memory Pool (DVMP) delivers transparent DRAM
virtualisation for >1 TiB models, yet two years of production experience exposed
bottlenecks in scalability, extensibility, and maintenance.  DVMP v2
introduces a **Tiered Unified Memory Pool (TUMP)** that elevates DVMP from a
CPU-only page allocator to a **multi-tier, NUMA-aware, lock-free memory
orchestrator**.  A single data structure tracks chunk residency across DRAM,
HBM, NVMe-swap, and remote nodes, eradicating today’s duplication between
`DistributedMemoryPool`, `UnifiedModelMemory`, and `MemoryManager`.

Key benefits:

1. **Generality** – arbitrary memory tiers and devices (CPU, GPU, NPU,
   persistent RAM) share one API.
2. **Maintainability** – -35 % code deletion by collapsing three metadata
   systems into one.
3. **Performance** – lock-free, sharded state table, NUMA-aware allocation,
   mlock2/MLOCK_ONFAULT fast paths, and PSI-driven eviction improve tail
   latencies by 2-3× under pressure.
4. **Extensibility** – pluggable `TierBackend` enables S3, SSD, or RDMA tiers
   without touching core logic.

---

## 1  Problem Statement

### 1.1  Current Architecture Recap

```text
       Model              UnifiedModelMemory                DVMP
┌───────────────────┐   ┌────────────────────────────┐   ┌─────────────────────┐
│  Model facade     │   │  GPU chunk state (per dev) │   │ DRAM VA + ChunkMeta │
│  (public API)     │──▶│  + best-source heuristics  │──▶│ HOT/COLD/… + mlock  │
└───────────────────┘   └────────────────────────────┘   └─────────────────────┘
                            ▲                ▲
                            │                │
                MemoryManager (copies, pools, streams)
```

Pain points uncovered in [RFC-0001] & [0002] code reviews and on-call reports:

1. **Duplicate State** – DVMP tracks CPU states, while `UnifiedModelMemory`
   mirrors them plus GPU states.  Two CAS graphs must stay consistent.
2. **Central Mutex** – DVMP’s global `std::mutex` becomes a hot spot with
   100+ concurrent loaders.
3. **Fixed Chunk Size** – `kChunk = 256 MiB` is hard-coded, forcing wasteful
   internal fragmentation for small models (<2 GiB) and harming L2 TLB reach on
   48 kiB-page GPUs.
4. **CPU-centric** – other tiers are bolted on ad-hoc (e.g., `COPIED_GPU` in
   DVMP enum) which violates SRP and bloats the state machine.
5. **Monitoring & Control** – eviction and pre-emption are push-based loops;
   Linux PSI (pressure stall information) and cgroupv2 controllers are not
   exploited.

---

## 2  Design Goals

1. **One source of truth** for chunk residency across all tiers.
2. **Pluggable tiers** (`TierBackend` concept) with uniform lock API.
3. **Horizontal scalability** to 1 k+ worker threads without contended global
   locks.
4. **Dynamic chunk granularity** (64 KiB – 1 GiB) negotiated per-model.
5. **First-class observability** (metrics, eBPF hooks, PSI triggers).

---

## 3  Proposed Architecture

### 3.1  TieredChunkTable (TCT)

A lock-free, sharded table (power-of-two shards) indexed by
`(model_id << tier_bits) | chunk_id`.  Each 64-bit entry is a packed struct:

```
struct TCTEntry {
  uint8_t  tier_id;        // 0 = DRAM, 1 = HBM, 2 = NVMe, …
  uint8_t  state;          // HOT, COLD, LOCKED_TX, … (same semantics)
  uint16_t last_touch;     // seconds since Epoch / 64 to fit 18 h window
  uint32_t aux;            // tier-specific (e.g., GPU device id or RDMA node)
}; // 8 bytes – cache-friendly
```

Operations (`lock`, `unlock`, `mark_preemptible`, `evict`) become
`atomic_compare_exchange_weak` on the entry – **no central mutex**.  Per-shard
spinlocks are only needed for slow-path metadata allocation.

### 3.2  TierBackend Interface

```cpp
class TierBackend {
 public:
  virtual absl::Status ensure_resident(const InstanceKey&, uint32_t chunk) = 0;
  virtual absl::Status evict(const InstanceKey&, absl::Span<const uint32_t>) = 0;
  virtual size_t       chunk_size() const = 0;
  virtual ~TierBackend() = default;
};
```

Reference back-ends:

* **DRAMBackend** – wraps current DVMP mmap/​mlock logic (ported).
* **HBMBackend** – thin veneer over `CudaMemory` & peer access.
* **NVMeBackend** – async io_uring pre-fetch and madvise `MADV_DONTNEED`.
* **RemoteBackend** – P2P over RDMA using loader pipeline from RFC-0002.

### 3.3  MemoryManager & Model Changes

`MemoryManager` no longer owns a `DistributedMemoryPool`.  Instead it holds a
`shared_ptr<TieredChunkTable>` and obtains tier access via DI:

```cpp
struct MemoryTierConfig {
  std::shared_ptr<TierBackend> dram;
  std::vector<std::shared_ptr<TierBackend>> gpus; // one per device
};
```

Loaders request locks directly from TCT; after successful transfer they call
`update(entry, HOT_GPU)` (fills in `tier_id`, `state`, `aux=device_id`).

### 3.4  Eviction & PSI Integration

A new `MemoryPressureAgent` subscribes to `/proc/pressure/memory` and cgroupv2
`memory.events`.  Spikes above the configured watermark awaken the `Evictor`
thread which performs **tier-aware LRU** across DRAM → NVMe first, then DRAM →
Remote chunk spills.

---

## 4  API Impact

1. `DistributedMemoryPool` **deprecated** and replaced by `TieredChunkTable` + 
   `DRAMBackend`.
2. `UnifiedModelMemory` merged into TCT; callers use small helper shims.
3. Loader lock API migrates:

```cpp
- dvmp.lock_chunks(model, idx);
+ tct.lock(model, Tier::DRAM, idx);
```

auto-migration is aided by clang-tidy rewrite.

---

## 5  Implementation Plan

| Phase | Scope | Key Metrics |
|-------|-------|------------|
| P0-A  | DRAMBackend + TCT (read-only)      | All unit-tests pass, ‑500 LoC |
| P0-B  | Lock/unlock path + Loader patch    | Loader throughput ≥ baseline |
| P1-A  | PSI-driven Evictor                 | 99-p latency −30 % on 670 GB model |
| P1-B  | HBMBackend & direct GPU touch hook | Zero regression in GPU copy perf |
| P2    | NVMeBackend & RemoteBackend unification | 10 % less code in loaders |

Each phase is a standalone PR guarded by `--enable-tump` feature flag.

---

## 6  Risk Analysis & Mitigations

| Risk | Mitigation |
|------|------------|
| Lock-free bugs    | Extensive TSAN / hazard pointer checks; fuzz tests on TCT |
| Kernel API drift  | Fallback from `mlock2` to `mlock` / portability macro |
| Rollout safety    | Dual DVMP/TUMP build option for one release cycle |

---

## 7  Expected Outcomes

* **Generality**: memory tiers become configuration, not code.
* **Maintainability**: ‑35 % net LOC (`memory/*`, `model/*`, `loader/*`).
* **Performance**: Up to 3× reduction in page-fault tail latency under
  overcommit; 1.2× faster hot-path `lock_chunks`.
* **Observability**: per-tier Prometheus counters and eBPF histograms.

---

## 8  Open Questions

1. Should we guarantee 8-byte TCT entry forever (ABI)?
2. RemoteBackend header-only vs standalone `libremote_tier.so`?
3. Expose tier residency via `/proc/stepcast/tiered_memory` debugfs?

---

## 9  Conclusion

DVMP v1 proved the feasibility of distributed virtual memory for gigantic
models.  The proposed **Tiered Unified Memory Pool** refines this into a
future-proof, high-performance substrate that scales with forthcoming memory
technologies and eliminates redundant code.  We recommend immediate adoption
behind an experimental flag and phased rollout over three minor releases.

---

[RFC-0001]: 0001-distributed-virtual-memory-pool.md
[0002]: 0002-unified-loader-architecture.md