---
slug: 0005-async-copy-manager
title: Async Copy Manager (Design)
related_code:
  - core/common/async_copy_manager.*
  - core/common/memory/streaming_pinned_buffer.*
  - core/store/replica/transfer_helpers.*
  - core/store/replica/memory_manager.*
  - core/checkpoint/streaming_tensor_writer.*
  - core/communicator/transport/mtcp_transport.*
links:
  plan: ../plans/0005-async-copy-manager.md
---# SummaryIntroduce a process-wide Async Copy Manager (ACM) that centralizes GPU copy submission and completion handling across H2D, D2H, D2D, and H2H. ACM provides a minimal, uniform interface to submit a single async copy and receive a lightweight host callback upon completion; it does not own chunking, throttling, fairness, or UMA/VS state. The design aligns copy boundaries with VS chunk boundaries, integrates with StreamingPinnedBuffer (SPB) for pinned staging, eliminates per-chunk stream synchronizations, and establishes consistent observability via tracing and metrics.Target outcomes (same hardware, same datasets):
- End-to-end MTCP receive→GPU-ready throughput improves by ≥20%.
- Checkpoint GPU→disk throughput improves by ≥15%.
- 10GB H2D load achieves ≥80% overlap between I/O and GPU copies.
- Pinned slot utilization averages <80% with P95 wait <5ms.# Goals / Non‑GoalsGoals
- Single entry point to submit H2D/D2H/D2D/H2H copies with uniform completion callbacks.
- Align copy planning and completion with VS chunk boundaries for UMA/VS state transitions.
- Unify with StreamingPinnedBuffer: slots act as the flow-control window; callbacks return slots without busy-waiting.
- First-class observability: OTel + Chrome trace spans including direction, bytes, latency, throughput, and error codes.Non‑Goals / Constraints
- No fairness, priority scheduling, or global throttling in ACM; planners control inflight windows.
- ACM does not manage SPB capacity or UMA/VS state machines.
- No external CUDA stream injection; ACM uses per-device, per-direction internal non-blocking streams.
- No cancel() of in-flight DMA; cancellation is achieved by stopping new submissions.# Architecture & InterfacesTerminology and invariants
- VS block: minimum UMA/VS management unit. Copies are planned on VS-aligned boundaries; single submissions must not cross VS boundaries.
- SPB slot: single pinned chunk from the process-level pinned pool used for staging; every slot is returned exactly once.
- Pageable↔Pinned↔Device: All H2D/D2H paths explicitly stage through pinned memory. GPU DMA source/destination must be pinned; planners handle pageable→pinned staging via SPB.
- Direction streams: For each device, ACM keeps three non-blocking streams (H2D/D2H/D2D same-device). Submissions on the same direction stream are serialized; different directions overlap per hardware.
- Callback visibility: `cudaLaunchHostFunc` guarantees all prior work on the stream is completed when the callback runs. Callbacks may safely return SPB slots and advance UMA/VS state.Minimal interface (conceptual)
```cpp
namespace tensorcast::common {struct HostRegion { const void* base; size_t length; bool pinned; };
struct DeviceRegion { int device_id; void* dev_ptr; size_t length; };struct CopyOptions {
  // room for tracing labels, alignment intents, etc.
};class CopyHandle {
 public:
  // Blocks until the device completes the copy; returns final status.
  absl::Status wait();
};class AsyncCopyManager {
 public:
  static AsyncCopyManager& instance();  // Each submit wraps exactly one cudaMemcpyAsync on ACM’s internal stream
  // and attaches a lightweight host callback to finalize resources/state.
  absl::StatusOr<CopyHandle> submit_h2d(const HostRegion& src,
                                        const DeviceRegion& dst,
                                        const CopyOptions& opts = {});  absl::StatusOr<CopyHandle> submit_d2h(const DeviceRegion& src,
                                        const HostRegion& dst,
                                        const CopyOptions& opts = {});  absl::StatusOr<CopyHandle> submit_d2d(const DeviceRegion& src,
                                        const DeviceRegion& dst,
                                        const CopyOptions& opts = {});  // H2H path is synchronous memcpy with immediate handle completion to
  // exercise pipeline logic in environments without CUDA.
  absl::StatusOr<CopyHandle> submit_h2h(const HostRegion& src,
                                        const HostRegion& dst,
                                        const CopyOptions& opts = {});  void shutdown();
};} // namespace tensorcast::common
```Submission semantics
- Exactly one device copy per call; chunking is handled by planners before submission.
- The host callback is attached via `cudaLaunchHostFunc` and is intentionally lightweight: return SPB slots, notify UMA/VS chunk completion, enqueue follow-on work. To comply with CUDA's restriction on host callbacks, it performs no CUDA runtime calls and simply records completion metadata. If `cudaLaunchHostFunc` is unavailable (older drivers, restricted runtimes), ACM falls back to a detached thread that `cudaStreamSynchronize`s before firing the callback, ensuring data is resident before slots recycle. Runtime errors surface through `CopyHandle::wait()`; submission-time errors surface via `StatusOr`.
- Callbacks receive an `absl::Status` describing the stream completion result; `CopyHandle::wait()` (or `ok()`) runs the CUDA runtime checks (`cudaSetDevice`, `cudaGetLastError`) on the caller thread before returning, so asynchronous failures still surface even when host callbacks execute on restricted threads.
- FAKE CUDA backend enqueues copies on a lightweight worker thread and fires callbacks asynchronously to mirror stream ordering while still running entirely on CPU memory. This keeps staging credit and slot recycling behaviour aligned with the real runtime.

SPB interaction patterns
- H2D (VS/Loader/Network→GPU): producers fill SPB slots (memcpy from VS view, file I/O, or `recv` directly into pinned). ACM submits H2D; the callback returns the slot and optionally advances UMA state.
- D2H (GPU→Disk/VS): ACM submits D2H into SPB; callbacks either mark the slot ready for consumer threads (disk writer) or notify higher layers to `vs->write_at(...)`, after which the slot is returned.Observability
- All submissions are wrapped by the existing tracing helpers (`core/common/trace/trace_cuda_async_fn.h`) with spans covering queue→DMA→callback. Prometheus metrics export per-direction throughput and SPB slot usage (e.g., `tc_spb_slots_in_use{device}`).# Schema Changes (if any)None. This design is an internal C++ runtime component and does not modify persistent schemas.# Trade‑offs & RisksTrade‑offs
- Simplicity over policy: ACM avoids fairness/priorities and global rate control; planners own inflight windows. This keeps ACM small and predictable but shifts policy elsewhere.
- Internal streams only: no external stream injection. This standardizes semantics at the cost of flexibility for advanced users.Risks and mitigations
- Timing shifts and races: UMA/VS transitions now occur in callbacks, not after per-chunk synchronizations. Mitigation: keep callbacks lightweight and idempotent; validate with consistency tests.
- Resource leaks: Every SPB slot must be returned exactly once, including FAKE CUDA. Mitigation: slot counters and assertions; last-resort return in error paths.
- Callback nesting: Disallow heavy work or recursive submissions inside callbacks. Mitigation: callbacks only notify; heavier work is dispatched to planner-owned threads.
- P2P/NUMA variability: Cross-device D2D might require staged D2H→H2D fallback. Mitigation: retain peer-access probing and fallback paths in ReplicaLoadController.
- Cancellation semantics: In-flight DMA is not cancelable; planners must stop new submissions and drain outstanding handles. Document clearly.# Compatibility & Acceptance CriteriaCompatibility
- Works with FAKE CUDA: copies execute on the simulated stream worker, callbacks still fire, and ordering matches the real runtime so tests remain deterministic.
- Unifies all H2D/D2H/D2D/H2H through ACM; removes per-chunk `stream_synchronize()` sites and ad‑hoc slot returns.
- Same-device D2D is handled by ACM; cross-device peer or staged fallback remains in ReplicaLoadController.Acceptance and success criteria
- Throughput improves by the stated targets for MTCP receive→GPU and Checkpoint GPU→disk.
- Overlap between I/O and GPU copies ≥80% for a 10GB H2D load.
- SPB slot utilization <80% on average with P95 wait <5ms.
- No slot leaks or handle leaks under 1‑hour stress; error paths remain observable and consistent.# ReferencesPrimary components and call sites
- SPB: `core/common/memory/streaming_pinned_buffer.{h,cc}`
- UMA/VS: `core/common/memory/virtual_address_space.{h,cc}`, `core/store/replica/unified_memory_authority.{h,cc}`
- CUDA wrappers: `core/common/cuda_api.h`
- Tracing: `core/common/trace/trace_cuda_async_fn.h`
- Store load path (VS→GPU, H2D): `core/store/replica/transfer_helpers.*`, `core/store/replica/transfer_service.*`
- Checkpoint write (GPU→disk, D2H): `core/checkpoint/streaming_tensor_writer.*`
- MTCP GPU receive (Network→GPU, H2D): `core/communicator/transport/mtcp_transport.*`
- D2D (same device): `core/store/replica/replica_load_controller.*`# Open Questions- Where to draw the line between per-connection streams vs. per-device direction streams for MTCP (isolation vs. overlap)?
- Should UMA/VS advancement be strictly in-callback or signal-only followed by thread-pool execution?
- Do we need a thin CopyGroup for aggregated waiting/fail‑fast, or should this remain in planners?
