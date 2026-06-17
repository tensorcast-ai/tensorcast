---
slug: 0013-rdma-staging-flow-control
title: Unified Staging Flow Control for RDMA and MTCP
related_code:
  - core/communicator/engine/engine.cc
  - core/communicator/engine/engine.h
  - core/communicator/engine/channel.h
  - core/communicator/engine/host_pinned_gpu_stager.h
  - core/common/memory/streaming_pinned_buffer.*
  - core/communicator/transport/rdma_transport.cc
  - core/communicator/transport/mtcp_transport.cc
  - core/communicator/transport/mtcp_transport.h
  - proto/tensorcast/communicator/v1/communicator_config.proto
created: 2025-09-26
last_updated: 2026-06-17
links:
  plan: ../plans/0013-rdma-staging-flow-control.md
---

# Summary

The Communicator dedicates a shared staging infrastructure (`HostPinnedGpuStager`, `HostPinnedCpuStager`, `StreamingPinnedBuffer`) to serve both RDMA and MTCP transfers. Today, RDMA responses require pre-staging every segment before the server emits a single `READ_RESPONSE_EX`; if a request spans more bytes than the pool capacity (`buffers_per_flow * chunk_size`), the server blocks inside `HostPinnedGpuStager::stage()`, preventing the control channel from responding and the client from issuing reads—deadlocking the transfer. MTCP avoids this deadlock only because its send path streams staged buffers as soon as they are available, but RDMA and MTCP compete for the same pool without explicit arbitration, so one flow can starve the other under heavy load.

This design introduces a **unified staging flow controller** that manages credit across transports, slices responses into windows bounded by available credit, and continuously refills both RDMA and MTCP pipelines as acknowledgements (RDMA) or send completions (MTCP) arrive. The approach aligns with the Communicator architecture in `core/communicator/README.md`, preserves staged-only RDMA semantics, improves fairness across concurrent flows, and adds observability and configuration validation for staging pressure. We assume a simultaneous rollout of all daemons/clients, so no backward-compatibility shims are required.

# Goals / Non-Goals

## Goals
- Remove staging deadlocks irrespective of replica size by decoupling request size from pool capacity.
- Apply a single flow-control policy across RDMA and MTCP so neither transport can monopolize staging buffers.
- Maintain bounded pinned-memory usage while sustaining throughput for multi-gigabyte GPU replicas.
- Expose metrics/logging consistent with the README to help operators debug staging back-pressure.

## Non-Goals
- Autotuning chunk sizes or buffer counts (tracked separately in `design-0006`).
- Changing the `tensorcast.api` surface or Store Engine contracts.
- Replacing staging primitives (`PinnedBufferPool`, `StreamingPinnedBuffer`)—we extend them but keep their core API.

# Architecture & Interfaces

## Communicator Alignment

The README highlights three invariants we must keep:
1. **Staging is mandatory for RDMA**; buffers release on ACK.
2. **`buffers_per_flow` caps rotating chunks** per flow through `StreamingPinnedBuffer`.
3. **Channels multiplex multiple requests**; fairness is enforced at the channel level.

We retain these invariants while adding transport-agnostic flow credit and windowing.

## New Core Abstractions

### FlowCreditLedger (Channel-scoped)
- Maintains `total_credit = buffers_per_flow` and grants leases to active flows (RDMA or MTCP).
- Enforces fairness by limiting the sum of outstanding segments across transports on the same channel.
- API:
  ```cpp
  struct FlowCreditLease {
    int granted_segments;
    std::function<void()> return_fn;
  };
  absl::StatusOr<FlowCreditLease> acquire(int requested_segments);
  void release(int segments);
  ```
- Keeps waitlists for requests that cannot obtain credit immediately.

### StageLease
- Wraps a staged buffer (exposed pointer, bytes, staging metadata) and remembers the owning FlowCredit lease.
- Carries RDMA registration state: `{ibv_mr*, bool deregister_on_release}` so the unified controller—not ad hoc maps—owns MR lifecycle.
- Provides `void release()` which invokes the stored `release_fn` (stager release + optional MR deregistration) and returns the credit back to `FlowCreditLedger`.
- Uniform across RDMA and MTCP: RDMA returns the lease when ACK arrives; MTCP returns it on send completion.

### StageLeaseRegistry (Channel-scoped)
- Replaces the existing `staged_segments_` map.
- Stores entries keyed by `SegmentKey{request_key, window_seq, offset}` so ACK handlers (which still report offsets) and the GC thread can locate leases unambiguously.
- Exposes lookup APIs used by:
  - RDMA ACK handler (`ENGINE_OP_RDMA_READ_DONE_EX`) → release lease + credit.
  - TTL reaper → reap and log overdue leases (per transport) with `[staging_credit]` annotations.
- Thread-safe (channel-scoped mutex) and enforces invariants: registry size must match `FlowCreditLedger::inflight_credit()`; double-release attempts are logged and ignored to guard against duplicate ACKs.

### StagingWindow (Request-scoped)
- Encapsulates segmentation logic, window sequence, outstanding StageLeases, and transport hand-offs.
- Responsible for staging up to `min(window_size, available_credit)` segments per iteration, where `window_size` defaults to `buffers_per_flow` but can be overridden via config.

### Unified Flow Controller
- Manages StagingWindow lifecycle: attempts staging when credit is granted, emits transport messages, tracks completions, and triggers refill when leases return.
- Keeps per-transport policies pluggable: RDMA windows send `READ_RESPONSE_EX`, MTCP windows hand staged buffers to `MTcpTransport::enqueue_stage_lease`.
- Owns MR lifecycle for staged RDMA segments by wrapping registration/deregistration inside StageLeases (see "RDMA MR Lifecycle").

### RDMA MR Lifecycle
- After staging, the controller registers each buffer via `MrCache::get_or_register()` (or NUMA-specific `NetDev::reg_mr()`), storing the resulting `ibv_mr*` inside the StageLease.
- Deregistration runs exclusively through `StageLease::release()` so both ACK handlers and the TTL reaper reclaim registrations consistently.
- Registration failures immediately unwind the partially created lease (return staging buffer, release credit) and propagate a `READ_FAILED` response so callers observe the same error semantics as today.

## Server Pipeline

```mermaid
sequenceDiagram
  participant RQ as RequestThread
  participant CH as Channel
  participant LED as FlowCreditLedger
  participant WIN as StagingWindow
  participant ST as MemoryStager
  participant TX as Transport (RDMA/MTCP)

  RQ->>CH: get_or_create_channel()
  RQ->>WIN: init(request, transport)
  loop until WIN.done()
    WIN->>LED: acquire(window_budget)
    alt credit granted
      loop for each granted segment
        WIN->>ST: stage(offset, bytes, StageMode::kTry)
        alt staged
          ST-->>WIN: StageLease (with MR metadata)
          WIN->>REG: register(window_seq, offset, StageLease)
          WIN->>TX: dispatch(window_seq, StageLease)
        else unavailable
          WIN->>LED: release(unused_credit)
          WIN->>WIN: park()
          break
        end
      end
    else waitlist
      WIN->>WIN: park()
    end
  end
```

### Transport-specific Behavior

- **RDMA**: `dispatch()` serializes a `READ_RESPONSE_EX` per window (`window_seq`, `more_segments`, `credit_granted`). StageLeases remain outstanding until `RDMA_READ_DONE_EX(window_seq, offsets)` returns them.
- **MTCP**: `dispatch()` hands StageLeases to the refactored MTCP path (`MTcpTransport::enqueue_stage_lease(...)`). The transport now treats leases as immutable send descriptors (socket write + completion callback) instead of staging internally; its previous staging pipeline (chunk batching, async release) is lifted into `StagingWindow` so the unified controller remains the single owner of staging buffers.

### MTCP Transport Adaptation

- Remove per-transport staging logic from `MTcpTransport::send_loop()`/`recv_loop()`; instead those loops consume lease descriptors provided by the controller.
- Add a lightweight `StageLeaseHandle` that transfers ownership into worker threads and triggers the lease’s `release_fn` on completion.
- Preserve batching and multi-connection striping by allowing the transport to queue multiple leases while the controller throttles outstanding leases via credit.

### Non-blocking Staging

- `HostPinnedGpuStager::stage()` gains a `StageMode` enum:
  ```cpp
  enum class StageMode { kBlocking, kTry };
  absl::StatusOr<StageLease> stage(..., StageMode mode);
  ```
- `StreamingPinnedBuffer::try_get_free_chunk()` is used when `StageMode::kTry` to avoid blocking the staging loop. If no buffer is available, the window parks until a lease returns.
- `MemoryStager::stage()` gains an overload accepting `StageMode`; existing callers continue to pass `StageMode::kBlocking`, while implementations add a non-blocking fast path for `StageMode::kTry`.
- `StreamingPinnedBuffer` tracks an explicit slot lifecycle (`Free → ProducerOwned → Ready → ConsumerOwned`) and rejects out-of-order transitions. This guards against premature slot recycling if async completions lag. The regression test `//core/common:streaming_pinned_buffer_test` exercises those invariants so pump/transport regressions surface quickly.

### ACK / Completion Handling

- **RDMA**: `ENGINE_OP_RDMA_READ_DONE_EX` looks up StageLeases in the registry by `(request_key, window_seq, offset)`, runs the lease’s release hook (MR deregistration + stager return), and calls `FlowCreditLedger::release(1)`. The window immediately tries to refill remaining bytes.
- **MTCP**: `MTcpTransport` invokes the StageLease callback once a send completes; credit returns automatically. No protocol changes are required—only the staging pipeline changes.

- The GC thread consumes the same registry: it walks overdue entries, emits `[staging_credit]` logs (including `window_seq` and transport), and invokes each lease’s `release()` so buffers and MRs are reclaimed even if ACKs never arrive.

## Client / Receiver Changes

### RDMA Client
- Processes `READ_RESPONSE_EX` windows iteratively; each window triggers `RdmaTransport::read_multi()` posting exactly the segments provided.
- `ReadRequest` tracks outstanding windows and posts ACKs after each completion (with optional coalescing). ACK payloads include `window_seq` and `final_window` flags.

### MTCP Receiver
- Already streams data as it arrives. No protocol changes required.
- To support back-pressure visibility, we tag MTCP send completions with window sequence metadata so server logs/metrics reflect unified flow control.

## Protocol & Config Updates

- `ProtoReadResponseExHeader` adds:
  - `uint32 window_seq`
  - `uint32 credit_granted`
  - `uint8 more_segments`
- `ProtoRdmaReadDoneExHeader` adds:
  - `uint32 window_seq`
  - `uint8 final_window`
- `CommunicatorConfig.stager` gains:
  - `max_window_segments` (default 0 → auto = `buffers_per_flow`)
- Diagnostics: at startup we validate `buffers_per_flow > 0` and that `PinnedBufferPool::total_slots >= buffers_per_flow`. Warnings emit when configured replica sizes exceed available credit.

## Observability

Metrics (tagged by transport, channel, window):
- `stager.credit_inflight{transport}` — gauge of outstanding segments.
- `stager.credit_wait_total{transport}` — counter for times a request parks waiting for credit/buffers.
- `rdma.window_refills_blocked_total` — increments when RDMA refill fails due to unavailable credit/buffers.
- `mtcp.window_refills_blocked_total` — same for MTCP.
- `stager.release_latency_ms` — histogram measuring time between StageLease creation and release (per transport).
- Recommended tuning loop: start with `buffers_per_flow = 4` and default chunk sizes; under load, monitor `credit_wait_total` and link throughput. If wait counters climb while throughput is below target, increment `buffers_per_flow` or chunk size and re-test; if throughput plateaus but `stager.credit_inflight` stays near the pool ceiling, reduce chunk size or window segments to ease memory pressure.

Logs:
- `[staging_credit] channel=<...> transport=rdma seq=7 granted=4 outstanding=12`
- `[staging_credit] channel=<...> transport=mtcp seq=3 complete latency_ms=18`

Traces:
- `load_replica.window_stage` and `load_replica.window_ack` spans annotated with channel id, transport, window seq, credit granted, and bytes.

# Schema Changes (if any)

None.

# Trade-offs & Risks

- **Additional Control Messages**: Windowing increases RDMA control-plane chatter. Coalesced ACKs and dynamic window sizing mitigate this overhead.
- **Implementation Complexity**: Unified flow control introduces new abstractions; scoped unit and integration tests reduce regression risk.
- **Potential Throughput Dips**: If `max_window_segments` is too low relative to bandwidth, throughput may fall. Metrics and config validation help operators tune defaults.
- **MTCP Back-pressure**: MTCP send completions now influence credit availability. We must ensure transport threads surface completion callbacks promptly to avoid starvation.

# Compatibility & Acceptance Criteria

- No deadlocks when `replica_bytes >> buffers_per_flow * chunk_size`; both RDMA and MTCP flows progress.
- FlowCreditLedger never grants credit beyond `buffers_per_flow`; metrics confirm enforcement.
- MTCP throughput remains within ±5% of baseline under typical workloads.
- Cross-transport soak coverage provided by `bazel test //core/communicator:cross_transport_soak_test`; run on verbs-enabled hosts to validate StageLease credit recycling across RDMA and MTCP.
- Documentation (`core/communicator/README.md`, `docs/architecture/p2p-transfer-strategies.md`) updated to describe unified flow control before rollout.

# Addendum (2026-06-17): MTCP Receive-Side Recv-Path Decoupling

The original design removed **send-side** staging deadlocks (request size decoupled
from pool capacity; credit returns on send completion). The MTCP GPU **receive** path now
carries the complementary decoupling: the network *producer* and the H2D *consumer* run on
separate threads over the `StreamingPinnedBuffer` slot lifecycle, so the producer never
blocks on the H2D copy. This addendum documents that recv-path design as it currently
stands.

## Background: the coupling this removed

The risk was already flagged in *Trade-offs & Risks → MTCP Back-pressure*: completion
callbacks must surface promptly or flows starve. On the receive side the "completion" that
frees a staging slot is the **H2D copy**. The previous recv path held the slot across that
copy — it acquired a slot with a blocking `get_free_chunk`, handed it to a lane reader,
spawned one `std::async` thread per sub-chunk to run the H2D and free the slot in its
`on_copy_done` callback, and then blocked the request on an aggregate `future.get()` over
all sub-chunks. Three coupled effects made this deadlock-prone:

1. **Slot held across H2D.** Once every slot was stuck in H2D, the next `get_free_chunk`
   blocked, the recv loop stopped feeding lane readers, the sockets stopped draining, TCP
   back-pressure stalled the source's `socket-write`, and the source's staging credit (freed
   on send completion) never returned — so the source `StagingWindow` parked on credit.
   `AsyncCopyManager` is a process **singleton** shared across transports, so one H2D stall
   correlated across the tp views (more views → more likely to trip).
2. **Reaper masked the starvation.** `staging_wait_timeout_` and `ack_ttl_ms_` were both
   30s. When they shared a deadline, the channel-GC reaper could reap the stale lease and
   "rescue" the source loop (`release()` credit + `finish_source_transfer_progress` with a
   `DeadlineExceededError`) **without** sending `READ_FAILED` to the target — leaving the
   target to hang to its full ~180s read budget → `DEADLINE_EXCEEDED`.
3. **Per-request buffer reset race.** The per-request `reset_for_new_production()` assumed
   the buffer was idle between requests (true only because of the aggregate `future.get()`).
   Because `return_chunk` fired from the async `on_copy_done` callback, a return could still
   be in flight when the next request's reset ran, clobbering it — the same use-after-reset
   family as the RDMA `raw_hash_set RAW: Use of destroyed hash table` warning.

The failure mode was a **pipeline head-of-line coupling / flow-control** stall (the source
and target serialize requests A-then-B, so each socket carries ordered bytes; data
corruption was ruled out), and it was non-deterministic — sensitive to the timing of the
target draining its sockets.

## Current recv path: network producer + dedicated H2D consumer

The recv path routes through the native `StreamingPinnedBuffer` ready-queue
(`Free → ProducerOwned → Ready → ConsumerOwned`,
`core/common/memory/streaming_pinned_buffer.h`: `get_free_chunk`, `mark_chunk_ready`,
`get_ready_chunk`, `return_chunk`). The network producer never blocks on H2D:

```
recv_queue_ (per request)
   │ recv_loop: producer-orchestrator (never blocks on H2D)
   ▼
acquire_recv_slot()  ── non-blocking try_get_free_chunk + poll, shutdown-aware
   │  register slot_jobs_[slot] = {gpu_ptr, bytes, chunk_idx, request_state}
   │  chunk.on_complete = on_recv_chunk_complete(slot, status)
   │  tasks_[lane]->push_recv(chunk)        // returns immediately; recv_loop pops next request
   ▼
lane reader fills slot, set_result() → fires on_complete → mark_chunk_ready
   ▼
ready_queue_  ──►  h2d_consumer_loop (one dedicated thread per transport)
                     get_ready_chunk(); submit_h2d + wait (skip if recv failed);
                     return_chunk(slot)          // the ONLY return path
                     request_state.on_chunk_done(status, bytes)  → last chunk fires msg->set_result()
```

- **`MTcpTransportChunk`** (`mtcp_transport.h`) carries an optional
  `std::function<void(misc::result_t)> on_complete_`, invoked at the tail of `set_result()`.
  The per-lane reader thread already calls `set_result()` after the socket read, so this
  fires `mark_chunk_ready` with **no extra waiter thread**.
- **`recv_loop`** (`mtcp_transport.cc`) is producer-only: per request it plans the
  sub-chunks, then for each one `acquire_recv_slot` → register `slot_jobs_` → set
  `on_complete` → `push_recv`, and returns to pop the next request. There is no aggregate
  `future.get()`. The CPU branch routes through the same per-request accounting for uniform
  completion. `acquire_recv_slot` uses the non-blocking `try_get_free_chunk` with a poll so
  it wakes promptly on shutdown and returns `Cancelled` when stopping.
- **`on_recv_chunk_complete`** (`mtcp_transport.cc`) fires from the lane reader thread:
  on success `mark_chunk_ready(slot, chunk_index, bytes)`; on failure it marks the slot's
  job failed and aborts/marks the slot so the consumer still observes it exactly once.
- **`h2d_consumer_loop`** (`mtcp_transport.cc`), one dedicated thread per transport, is the
  sole consumer: `get_ready_chunk` → look up `slot_jobs_` → set the CUDA device →
  `submit_h2d` + `wait` (skipped when the recv failed) → `return_chunk(slot)` synchronously
  on this thread → `request_state.on_chunk_done`. It exits when `get_ready_chunk` reports
  the queue is drained after `signal_production_complete`.
- The buffer is reset once at receive-side init rather than per request; `slot_id` is the
  authoritative key (`global_chunk_id` is diagnostic only), so the per-request reset race is
  gone.

The `slot_jobs_` side table (sized to the slot count, guarded by a small mutex) links a slot
to its in-flight job `{gpu_ptr, bytes, device_id, chunk_index, request, recv_status}`; only
the slot's current owner writes it.

Invariant: **every slot acquired via `acquire_recv_slot` is returned by exactly one
`return_chunk` in the consumer** — success and failure both route through a ready chunk
(failure carries a poison flag) — so accounting lives in one place and double-free is
impossible. H2D concurrency is bounded to one consumer thread per transport instead of one
`std::async` thread per sub-chunk, removing the thread explosion that aggravated the shared
`AsyncCopyManager` stall.

## Completion accounting: `MTcpRecvRequestState`

A single per-request `MTcpRecvRequestState` (`mtcp_transport.h`) accounts for completion
with one atomic counter seeded with a producer "issuing" token (`outstanding_ = 1`). The
producer calls `add_chunk()` per issued sub-chunk and `seal()` once issuing is done; the
consumer calls `on_chunk_done(status, bytes)` per sub-chunk. Whichever thread decrements the
count to zero finalizes the request exactly once and sets the result (failed if any
sub-chunk failed). The issuing token closes the check-then-act race between the last
completing chunk and `seal()`, so the producer never has to block waiting for completions.

## Timeout ordering and reaper notification (defense-in-depth)

Two engine-side guards bound the residual case where H2D genuinely cannot keep up:

1. **Timeout ordering** (`engine.cc`, transport init): `ack_ttl_ms_` is raised at startup to
   at least `staging_wait_timeout_ + 15s` when it would otherwise be lower. Ordering the
   lease-TTL reaper strictly more patient than the source staging-credit wait lets the source
   `fail_mtcp_read_task` path — which already sends `READ_FAILED` — fire first, leaving the
   reaper as a pure memory backstop.
2. **Reaper notifies the target** (`engine.cc`, channel-GC reaper): when the reaper takes a
   stale lease it `release()`s its credit, finishes the source transfer progress, and sends a
   `READ_FAILED` (reason `TENSORCAST_READ_FAILED_MEM_MISMATCH`) to the target over the control
   transport. The target's `ENGINE_OP_READ_FAILED` handler then fails the read immediately
   instead of waiting out its full read budget. A duplicate notification is benign — the
   target tolerates a missing/already-completed request.

## Why it holds / residual risk

The consumer drains `ready_queue_` independently and continuously, so slots cycle and the
lane readers keep draining sockets whenever any slot is free → the source `socket-write`
completes → source credit returns → the reaper does not fire under H2D-comparable-to-net
conditions (H2D ≈ PCIe gen4 ~25 GB/s ≈ one 200GbE link). If H2D genuinely cannot keep up,
the system degrades to **bounded back-pressure (rate-limited), not deadlock**, and the two
guards above make the extreme-stall case fail fast rather than hang. The separate
non-deterministic mutex-assert / `destroyed hash table` concurrency crash on the RDMA path is
not addressed here (the per-request reset race is removed on the MTCP path, but the RDMA-side
variant is tracked separately).

## Implementation

- Primary: `core/communicator/transport/mtcp_transport.{h,cc}` — chunk completion callback,
  dedicated H2D consumer thread, `slot_jobs_` side table, `MTcpRecvRequestState`, and the
  producer-only `recv_loop`. No protocol, source-staging, or credit changes; no new pinned
  pool; no extra copy.
- Secondary: `core/communicator/engine/engine.cc` — startup `ack_ttl_ms_` ordering and the
  reaper `READ_FAILED` notification to the target.

# References

- `docs/designs/0006-unified-memory-stager-and-staged-p2p.md`
- `core/communicator/README.md`
- `docs/architecture/p2p-transfer-strategies.md`
- `core/common/memory/streaming_pinned_buffer.cc`
- `core/communicator/engine/engine.cc`
- `core/communicator/transport/rdma_transport.cc`
- `core/communicator/transport/mtcp_transport.cc`
