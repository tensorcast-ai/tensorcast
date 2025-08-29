# 0009 — Unified MemoryStager and Staged P2P (DVMP-safe, RDMA/TCP)

## 1. Overview

### Problem
In TCP mode, GPU tensors cannot be directly registered, so we stage via `GpuTcpStager` with a small pinned buffer. For CPU tensors, current RDMA export registers DVMP VA ranges directly which pins large pageable regions (e.g., 600+ GB) and violates DVMP’s design (eviction/preemption, pin-lease scope). With upcoming CUDA VMM for GPU, registering entire GPU tensors is also unacceptable.

### Goals
- CPU tensors: segmented copy from DVMP VA into small pinned buffers; immediately release DVMP pin leases after copy.
- Unify a staging interface that hides CPU/GPU and TCP/RDMA differences.
- Avoid large-region MR registration. Pre-register MRs on stable pinned buffers and reuse, never registering multiple times in one transfer.
- Clarify and coordinate two registrations:
  - Application-level “user tensor” registration (keys, ranges, lifetimes)
  - Transport-level RNIC MR registration (stable buffers only)
- NUMA-aware pinned pools and policy for NIC/GPU proximity.
- Backward-compatible, incremental rollout.

### High-level Solution
Introduce a `MemoryStager` interface with concrete implementations that produce host-pinned slices suitable for network transport. For RDMA, we return an already-registered MR on a stable pool buffer; for TCP, we return a host buffer for socket I/O. Server-side READ responses use staged buffers (not DVMP/GPU memory) and require a lightweight ACK to release the buffer.

## 2. Current Architecture Analysis

### Where DVMP windows are exported and registered
```69:99:/workspace/core/store/replica/chunk_export_service.cc
rec.cpu_tokens.reserve(ranges.size());
for (const auto& [start, end] : ranges) {
  uint64_t va_off = static_cast<uint64_t>(start) * kChunk;
  uint64_t va_end = std::min<uint64_t>(info.artifact_size, (static_cast<uint64_t>(end) + 1) * kChunk);
  uint64_t length = (va_end > va_off) ? (va_end - va_off) : 0;
  if (length == 0)
    continue;
  const uint64_t addr = reinterpret_cast<uint64_t>(static_cast<char*>(base) + va_off);
  auto tensor_key = absl::StrFormat("%s_CPU_chunk_%zu", key.artifact_id, range_idx++);
  auto ret = comm_engine.register_tensor(tensor_key, addr, length, info.comm_dev_type, info.device_id);
  if (!ret.ok()) { /* ... */ }
}
```
- Registers contiguous DVMP ranges as individual "tensors" in communicator. In RDMA mode, this results in MR registration over page-backed DVMP VA, pinning those pages.

### Communicator tensor registration pins RDMA MRs
```134:189:/workspace/core/communicator/engine/engine.cc
auto tensor = std::make_shared<PartitionTensor>(tensor_key, addr, bytes, dev_type, net_dev);
...
if (enable_rdma_) {
  net_dev->reg_async(tensor); // ibv_reg_mr on (addr, bytes)
  ...
}
```

### RDMA READ path uses the tensor’s MR directly
```456:466:/workspace/core/communicator/engine/engine.cc
if (enable_rdma_ && req->transport_type == ENGINE_TRANSPORT_RDMA) {
  payload->transport_type = ENGINE_TRANSPORT_RDMA;
  auto dev = tensor->get_dev();
  STRNCPY(payload->nic_name, dev->get_name(), kMaxDevName);
  tensor->wait_read_ready();
  auto* mr = tensor->get_mr();
  payload->addr = tensor->get_uint64_addr() + req->offset;
  payload->rkey = mr->rkey;
  payload->bytes = req->bytes;
} else {
  // using mtcp transport
  ...
}
```
- Directly advertises the remote MR of the registered DVMP window or GPU allocation.

### GPU staging for TCP already exists
```33:92:/workspace/core/communicator/engine/gpu_tcp_stager.cc
GpuTcpStager::GpuTcpStager(size_t chunk_size, size_t num_chunks, std::shared_ptr<store::PinnedMemoryPool> pool)
...
VLOG(1) << "GpuTcpStager initialized with ...";
```
- TCP sends use a staged D2H buffer; pinned pool is created in engine when RDMA is disabled.

### Pinned pool (host) today (no RDMA MR pre-reg yet)
```26:67:/workspace/core/common/memory/pinned_memory_pool.cc
PinnedMemoryPool::PinnedMemoryPool(size_t total_size, size_t chunk_size) : chunk_size_(chunk_size) {
  // aligned_alloc(...) + cuda::host_register(...)
  pool_.insert(buffer); free_list_.insert(buffer);
}
~PinnedMemoryPool() { cuda::host_unregister(buffer); free(buffer); }
```
- No RDMA MR registration cache yet; owns CUDA host pin only.

## 3. Proposed Solution

### 3.1 MemoryStager: unified staging contract
```mermaid
classDiagram
class MemoryStager {
  +StageToken stage(uint64_t va_offset, size_t bytes)  // blocking or async-wait
}
class StageToken {
  +void* host_ptr
  +size_t bytes
  +optional RdmaMr mr     // if RDMA is used
  +void complete()        // RAII or explicit; returns buffer to pool and releases DVMP pin
}
class DRAMStager
class GpuNetStager
MemoryStager <|.. DRAMStager
MemoryStager <|.. GpuNetStager
```
- `DRAMStager`: DVMP VA (CPU) -> host pinned buffer (memcpy). Acquires DVMP pin lease for `[va_off, len]`, copies to pool buffer, immediately releases DVMP pin (lease kept only during memcpy), returns pool slice; if RDMA, also provides pre-registered MR.
- `GpuNetStager`: VRAM -> host pinned (cudaMemcpyAsync D2H + event). Returns pool slice; if RDMA, also MR. Reuses logic from `GpuTcpStager`.

Key properties:
- Small, bounded buffers; DVMP pins are held just during copy, then released.
- Transport-agnostic: MTCP uses `host_ptr`, RDMA uses `mr`.
- NUMA-aware pool selection (see §3.4).

### 3.2 Transport behavior with staging
- Server READ handling:
  - RDMA path: server stages requested slice via stager; returns RDMA `addr/rkey/bytes` of the pool MR in READ_RESPONSE; waits for client ACK to release buffer.
  - MTCP path: server stages (CPU and GPU) and enqueues staged buffer chunks to `MTcpTransport` send queue; RAII returns buffer when send completes.
- Client remains unchanged for MTCP; for RDMA, after completion, send ACK.

Updated control:
```mermaid
sequenceDiagram
  participant Client
  participant Server
  participant CE as CommunicateEngine
  participant Stager as MemoryStager
  participant NIC as RDMA

  Client->>Server: READ_REQUEST (offset, bytes, prefer=RDMA)
  Server->>Stager: stage(offset, bytes)
  Stager-->>Server: StageToken{host_ptr, bytes, mr}
  Server->>Client: READ_RESPONSE{addr, rkey, bytes}
  Client->>NIC: RDMA READ
  NIC-->>Client: complete
  Client->>Server: RDMA_READ_DONE (req_key)
  Server->>Stager: token.complete()
```

### 3.3 Permanent MR registration on pool buffers
- Each pool buffer is registered once per RNIC PD; handle cached until pool destruction.
- No per-transfer `ibv_reg_mr` churn. MR lookup by `(pd, buffer_ptr)` maps to a cached `ibv_mr*`.
- For multi-NIC: per-PD MR registries.

### 3.4 NUMA-aware pinned pools
- Create per-domain pools: `pool[by_nic]`, `pool[by_gpu]`, or `pool[fixed_node]` per policy.
- Selection:
  - GPU staging: pool near GPU (minimize D2H)
  - CPU staging for RDMA: pool near NIC PD
  - CPU staging for MTCP: pool near sending CPU cores or NIC

### 3.5 Registration layering and coordination
- Application-level registration (PartitionTensor in store): Identify logical source (DVMP VA or GPU base), key, size, device type. No requirement to have an MR.
- Transport-level MR registration: Only on stable pool buffers, owned by pool/registry; reused across transfers.
- `register_tensor(...)` extended with options:
  - `register_mr=false` for DVMP/GPU logical windows that are not to be RDMA-read directly.
  - `needs_staging=true` for GPU, and for CPU when policy mandates staged RDMA.

### 3.6 CPU segmented copy + immediate DVMP pin release
- DRAMStager flow:
  1) Acquire DVMP pin lease for `[va_off, len]` (via UMA: `create_direct_write_token`).
  2) memcpy into pool buffer slice.
  3) Drop lease immediately (DVMP is free to evict).
  4) Return token carrying host_ptr and (if RDMA) MR.

This guarantees large artifacts are never pinned as a whole.

### 3.7 Protocol additions
- New op: `ENGINE_OP_RDMA_READ_DONE` (Client ➜ Server) with `{tensor_key, offset, bytes}` or `req_key`.
- Server releases staged resources on ACK. If ACK missing, apply TTL reap (`STEPCAST_COMM_RDMA_STAGE_TTL_MS`).

Backward compatibility: If server responds with a direct MR (legacy), client behavior unchanged. The ACK is only required for staged RDMA responses (server advertises a flag in READ_RESPONSE reserved bits).

## 4. Implementation Approach

### 4.1 APIs
- MemoryStager (new): `StageToken stage(tensor, offset, bytes)` where `tensor` is `PartitionTensor` or a typed source descriptor.
- StageToken: RAII completion.
- RegisterTensorOptions (new): `{ bool register_mr; bool needs_staging; int device_id; }`.

### 4.2 Component changes

- ChunkExportService: stop registering DVMP windows for direct RDMA MR; keep application-level keys but call `register_tensor(..., options{register_mr=false, needs_staging=(CPU_policy)})`.
- CommunicateEngine:
  - Add `RegisterTensorOptions` overload.
  - Maintain `std::shared_ptr<MemoryStager>`; provide stager to MTCP and RDMA paths.
  - RDMA READ_RESPONSE path uses stager if `needs_staging==true` or policy requires; otherwise legacy direct MR (for small hot CPU allocations if desired).
  - Track inflight staged tokens keyed by `req_key`; release on `RDMA_READ_DONE`.
- MTcpTransport: replace GPU-only staging with MemoryStager usage for both CPU and GPU.
- PinnedMemoryPool: add NUMA placement and per-PD MR registry (cache). Export `get_or_register_mr(pd, buffer_ptr)`.
- RdmaTransport / RdmaThread: on client completion, send `RDMA_READ_DONE`.

### 4.3 Configuration
- `STEPCAST_COMM_EXPORT_MAX_WINDOW_MB` (default: min(DVMP chunk, 64MiB)).
- `STEPCAST_COMM_PINNED_NUMA_POLICY=auto|nic|gpu|fixed:<node>`
- `STEPCAST_COMM_RDMA_STAGE_TTL_MS=30000`
- `STEPCAST_COMM_STAGE_CPU_FOR_RDMA=true|false` (default true)

### 4.4 Trade-offs
- Staged RDMA for CPU adds one memcpy but removes massive pinning and MR churn.
- Server ACK requirement adds one small control message per range.
- Requires careful pool sizing and back-pressure.

## 5. Code References (current)

- CPU export registering DVMP ranges as communicator tensors: 69:99:/workspace/core/store/replica/chunk_export_service.cc
- RDMA register on registration: 134:189:/workspace/core/communicator/engine/engine.cc
- RDMA READ uses tensor MR directly: 456:466:/workspace/core/communicator/engine/engine.cc
- GPU TCP staging exists: 33:92:/workspace/core/communicator/engine/gpu_tcp_stager.cc
- Pinned pool (host) without RDMA MR cache: 26:67:/workspace/core/common/memory/pinned_memory_pool.cc

## 6. Implementation Plan

### Phase 1 — Windowized CPU staging (TCP + immediate DVMP release)
- Introduce `DRAMStager` using pooled pinned buffers; integrate into MTCP send path for CPU.
- In `ChunkExportService`, keep keys but set `register_mr=false` for CPU DVMP ranges.
- Metrics: staged_bytes_total{cpu}, dvmp_pin_lease_duration_ms, pool_pressure.

### Phase 2 — RDMA: staged responses and ACK
- Add `ENGINE_OP_RDMA_READ_DONE` and ACK handling on both sides.
- Server RDMA READ path uses MemoryStager for CPU and GPU; responds with pool MR.
- Add MR cache in pinned pool per PD; pre-register MRs on pool init for registered NICs.

### Phase 3 — NUMA-aware pools
- Add NUMA placement and policy to `PinnedMemoryPool`.
- Create per-NIC and per-GPU pools; selection policy in MemoryStager.

### Phase 4 — Cleanup and policy knobs
- Optional: enable direct MR for small CPU slabs (disable staging) via policy; default remains staged.
- Expand tests across all combinations.

## 7. File Modification Table

| Area | File(s) | Change |
|---|---|---|
| Memory staging | `core/communicator/engine/memory_stager.{h,cc}` | NEW interface + StageToken |
| CPU stager | `core/communicator/engine/dram_stager.{h,cc}` | NEW: DVMP->pinned memcpy + DVMP lease RAII |
| GPU stager | `core/communicator/engine/gpu_net_stager.{h,cc}` | NEW: Adapt from `GpuTcpStager` with MR support |
| Pool MR cache | `core/common/memory/pinned_memory_pool.{h,cc}` | Add NUMA + MR registry per PD |
| Registration opts | `core/communicator/engine/engine.{h,cc}` | Add `RegisterTensorOptions`; avoid MR for DVMP windows |
| Server RDMA path | `core/communicator/engine/engine.cc` | Use MemoryStager and ACK inflight tracking |
| Protocol | `core/communicator/engine/protocol.h` | Add `ENGINE_OP_RDMA_READ_DONE` |
| MTCP send | `core/communicator/transport/mtcp_transport.{h,cc}` | Use MemoryStager for CPU+GPU; RAII completion |
| Chunk export | `core/store/replica/chunk_export_service.{h,cc}` | Window tiling + register_mr=false for CPU |
| Loader source | `core/store/loader/remote_key_source.{h,cc}` | Unchanged externally; may set prefer=RDMA flag |
| Tests | `core/communicator/engine/*_test.cc`, `core/store/replica/*_test.cc` | Update assertions and add staged RDMA cases |

## 8. Detailed Flows

### 8.1 CPU (DVMP) → RDMA (staged)
```mermaid
sequenceDiagram
  participant Cl as Client
  participant Sv as Server
  participant CE as CommunicateEngine
  participant ST as DRAMStager
  participant DV as DVMP

  Cl->>Sv: READ_REQUEST{key, off, len, RDMA}
  Sv->>ST: stage(off,len)
  ST->>DV: pin_range(off,len)
  ST->>ST: memcpy(DVMP -> pool)
  ST-->>DV: release pin (immediate)
  ST-->>Sv: StageToken{mr,addr,len}
  Sv->>Cl: READ_RESPONSE{addr,rkey,len,flag=staged}
  Cl->>Sv: RDMA READ
  Cl-->>Sv: RDMA_READ_DONE
  Sv->>ST: complete()
```

### 8.2 GPU → RDMA (staged)
```mermaid
sequenceDiagram
  participant Cl as Client
  participant Sv as Server
  participant ST as GpuNetStager

  Cl->>Sv: READ_REQUEST{key, off, len, RDMA}
  Sv->>ST: stage(off,len)
  ST->>ST: cudaMemcpyAsync(D2H)
  ST-->>Sv: StageToken{mr,addr,len}
  Sv->>Cl: READ_RESPONSE{addr,rkey,len,flag=staged}
  Cl->>Sv: RDMA READ
  Cl-->>Sv: RDMA_READ_DONE
  Sv->>ST: complete()
```

### 8.3 CPU/GPU → TCP (staged)
#### CPU (DVMP) → TCP (staged)
```mermaid
sequenceDiagram
  participant Sv as Server
  participant ST as DRAMStager
  participant DV as DVMP
  participant MTCP as MTcpTransport

  Sv->>ST: stage(offset,len)
  ST->>DV: pin_range(offset,len)
  DV-->>ST: lease_token
  ST->>ST: memcpy(DVMP VA → pinned_pool)
  ST-->>DV: release lease_token
  Note over DV,ST: Lease held only during memcpy
  ST-->>Sv: StageToken{host_ptr,len}
  Sv->>MTCP: enqueue(host_ptr,len, token)
  MTCP->>MTCP: send over sockets
  MTCP-->>ST: send_done(token) → complete()
```

#### GPU → TCP (staged)
```mermaid
sequenceDiagram
  participant Sv as Server
  participant ST as GpuNetStager
  participant CU as CUDA
  participant MTCP as MTcpTransport

  Sv->>ST: stage(offset,len)
  ST->>CU: cudaMemcpyAsync(D2H, stream)
  CU-->>ST: event_complete
  ST-->>Sv: StageToken{host_ptr,len}
  Sv->>MTCP: enqueue(host_ptr,len, token)
  MTCP->>MTCP: send over sockets
  MTCP-->>ST: send_done(token) → complete()
  Note over ST,MTCP: RAII returns buffer to pool on send completion
```

## 9. Alternatives Considered
- Keep direct RDMA on DVMP windows: rejected due to large-region pinning and incompatibility with DVMP eviction policy.
- GPU direct RDMA: rejected for CUDA VMM and portability.
- Per-request MR registration on DVMP pages: rejected due to registration latency and kernel pressure.

## 10. Success Metrics
- No DVMP leases held beyond memcpy duration (CPU staged).
- No RDMA MR registrations on DVMP VA; MRs only on pool buffers.
- Throughput ≥ current (±5%) for 10–50 GB workloads; scalable beyond 600 GB without pinning blowups.
- Bounded pool memory: configurable; back-pressure when exhausted instead of pinning system memory.

## 11. Compatibility & Rollout
- Phase 1 can ship without protocol changes (TCP only).
- Phase 2 introduces a backward-compatible ACK op. If ACK not supported by peer, server falls back to MTCP.

## 12. Appendix — Proposed Interfaces (sketch)

```cpp
// memory_stager.h
struct StageToken {
  void* host_ptr;
  size_t bytes;
  struct ibv_mr* mr; // optional
  std::function<void()> complete; // RAII/explicit
};

class MemoryStager {
 public:
  virtual ~MemoryStager() = default;
  virtual absl::StatusOr<StageToken> stage(const communicator::tensor_t& t, uint64_t offset, size_t bytes) = 0;
};

// Register options
struct RegisterTensorOptions {
  bool register_mr = true;         // false for DVMP/GPU logical windows
  bool needs_staging = false;      // true for GPU; true for CPU when policy requires
  int device_id = -1;
};
```

## 13. Notes
- Keep `register_tensor()` fast; avoid blocking on MR registration when `register_mr=false`.
- Leverage existing UMA `create_direct_write_token(...)` for DVMP pin leases.
- MR cache must be per-PD and cleared on device shutdown.

## 14. P2P Throughput Optimizations (Framework-level)

This section refines the design with specific knobs and mechanisms to maximize end-to-end throughput while keeping CPU overhead and latency low.

### 14.1 Zero-register (DirectMR) vs Staged policy matrix

- Default policy:
  - CPU: staged for RDMA (to avoid DVMP pinning) and direct for TCP.
  - GPU: staged for both RDMA and TCP.
- Override for hot, small CPU slabs (≤ `STEPCAST_COMM_DIRECT_MR_MAX_BYTES`): allow DirectMR to skip memcpy if UMA marks chunk range as HOT+Resident and total bytes small, bounded by `max_inflight_direct_mr`.

Knobs:
- `STEPCAST_COMM_DIRECT_MR_MAX_BYTES` (default 4 MiB)
- `STEPCAST_COMM_MAX_INFLIGHT_DIRECT_MR` (default 32)

Rationale: tiny transfers are memcpy-bound; avoiding staging can help if DVMP guarantee prevents eviction during the small window. Still bounded to avoid system pin growth.

### 14.2 Pipelining: triple-buffer and deep queues

- MemoryStager should support acquiring multiple StageTokens concurrently; recommended buffers per inflight request: 3–4 to overlap copy/D2H, network IO, and CPU processing.
- MTCP: keep `send_queue_` depth ≥ `conn_count × 2` chunks; use chunk size that matches NIC BDP.
- RDMA: allow multiple outstanding reads per request; `RdmaTransport` to post `k` WRs in advance (default 64) and coalesce READs up to MR boundary.

Knobs:
- `STEPCAST_COMM_STAGE_BUFFERS_PER_FLOW` (default 4)
- `STEPCAST_COMM_RDMA_OUTSTANDING_WR` (default 64)
- `STEPCAST_COMM_TCP_CONN_COUNT` (already exists)

### 14.3 Chunk sizing tuned to BDP

- Choose stage chunk size s.t. `chunk_bytes ≈ bandwidth × RTT`. Typical values:
  - 100 Gbps, RTT 80 µs → BDP ≈ 1 MB; with parallelism, 4–8 MB chunk performs well.
  - PCIe Gen4 GPU D2H prefers ≥ 16 MiB for high sustained rates; use larger staging chunks on GPU side when CPU can handle it.

Knobs:
- `STEPCAST_COMM_STAGE_CHUNK_MB_CPU` (default 4)
- `STEPCAST_COMM_STAGE_CHUNK_MB_GPU` (default 16)

Validation: expose dynamic autotune to increase/decrease chunk sizes based on observed throughput and completion latency percentiles.

### 14.4 NUMA and affinity

- Bind staging memcpy threads to cores local to the memory node of the pool buffer.
- For GPU staging, prefer pools local to the GPU’s closest NUMA node and bind D2H streams to a dedicated CUDA stream per GPU.
- For RDMA, prefer pools local to NIC’s NUMA node; pin `RdmaThread::poll_loop` to NIC-local cores.

Knobs:
- `STEPCAST_COMM_AFFINITY_ENABLE=true|false`
- `STEPCAST_COMM_RDMA_POLL_CORE_MASK` (bitmap)
- `STEPCAST_COMM_STAGE_CORE_MASK` (bitmap) per GPU/CPU domain

### 14.5 Copy engines and async completion

- DRAMStager: use `memcpy` for now; optionally switch to `memcpy_nt`/`std::memcpy` tuned by platform. Consider `io_uring` `IORING_OP_READ/WRITE` when source is file-mapped in future paths (not DVMP path).
- GpuNetStager: keep `cudaMemcpyAsync` + per-buffer CUDA event; always reuse one stream per stager to avoid stream creation overhead.
- Completion: use eventfd or lock-free queues for StageToken completions to minimize contention.

### 14.6 RDMA control overhead minimization

- Batch READ_RESPONSE messages when a single READ_REQUEST enumerates multiple segments; the server can return a vector of staged segments (addr,rkey,bytes) for pipelined posting by the client.
- Use inline data for small READ_RESPONSE payloads to reduce latency.
- ACK (`RDMA_READ_DONE`) batching: client can ACK a group of segments with a bitmap.

Protocol additions (optional, backward compatible):
- `ENGINE_OP_READ_RESPONSE_EX` with repeated segments.
- `ENGINE_OP_RDMA_READ_DONE_EX` with ranges/bitmap.

### 14.7 Pre-registration and warmup

- On engine init or first RDMA enable, pre-register MRs for all pool buffers across discovered PDs; record rkeys.
- Warmup path to stage N dummy chunks to calibrate chunk size autotune and prime caches.

### 14.8 Back-pressure and fairness

- If pool pressure is high, prefer smaller chunks to improve fairness across flows; if pressure is low and completion is fast, increase chunk size.
- Implement token-bucket per remote peer to prevent head-of-line blocking.

### 14.9 Observability for tuning

Counters and histograms:
- `stager_copy_bytes_total{type=cpu|gpu}`
- `stager_copy_seconds_sum{type=cpu|gpu}` & P50/95 latency histograms
- `rdma_wr_inflight`, `rdma_wr_posted_total`, `rdma_wr_completed_total`
- `mtcp_send_q_depth`, `mtcp_recv_q_depth`
- `pool_buffers_total`, `pool_buffers_free`
- `ack_pending_total`, `ack_batch_size`

### 14.10 Concrete defaults per transport

- TCP (MTCP): `conn_count=8`, stage chunk CPU=4 MiB, GPU=16 MiB, buffers per flow=4.
- RDMA: outstanding WR=64, stage chunk CPU=4–8 MiB, GPU=16–32 MiB; pre-registered MR per pool buffer.

### 14.11 Mapping to code changes

- `PinnedMemoryPool`: add per-PD MR registry; add NUMA placement; expose `reserve_buffers(n)` for preallocation.
- `GpuTcpStager` → `GpuNetStager`: add MR lookup, configurable chunk size, buffers per flow.
- `DRAMStager`: support multi-buffer stage with memcpy and RAII completion queue.
- `CommunicateEngine`:
  - Add segment array in READ_RESPONSE_EX; posting and ACK batching.
  - Add autotune engine that observes throughput and tail latencies and adjusts chunk size/buffers.
- `RdmaTransport`: pipeline multiple WRs per response; add ACK batching.
- `MTcpTransport`: ensure deep send queue; avoid frequent heap allocations by pooling `MTcpTransportChunk`.