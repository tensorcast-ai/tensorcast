---
slug: communicator-small-packet-latency-and-throughput-optimization
title: Communicator Small-Packet Latency and Throughput Optimization (Plan)
links:
  design: ../designs/0105-communicator-small-packet-latency-and-throughput-optimization.md
areas:
  - core
  - proto
  - docs
  - benchmarks
related_code:
  - docs/benchmarks/20260316-communicator-vs-transfer-engine-comparison.md
  - docs/benchmarks/20260315-communicator-rdma-nic-gdr-progress.md
  - core/communicator/engine/engine.cc
  - core/communicator/engine/engine.h
  - core/communicator/engine/staging_flow_controller.cc
  - core/communicator/transport/rdma_transport.cc
  - core/communicator/transport/request.h
  - core/communicator/transport/mtcp_transport.cc
  - proto/tensorcast/communicator/v1/communicator_config.proto
---

# Objective

Deliver the `0105` design as an execution path centered on
"non-transfer-cost quantification + targeted optimization." The first objective
is not a single bandwidth number. It is to clearly split non-data-plane cost
inside task execution into one-time amortizable cost and recurring
per-transfer cost. After attribution is clear, we then optimize
small/medium-packet behavior while keeping `256 MiB+` throughput regression-free.

# Current State & Grounding

## Baseline (2026-03-16)

From [20260316-communicator-vs-transfer-engine-comparison.md](../benchmarks/20260316-communicator-vs-transfer-engine-comparison.md):

| Logical request size | Communicator | Transfer Engine | Gap |
| --- | --- | --- | --- |
| `32 MiB` | `18.56 GB/s` | `22.49 GB/s` | `-3.93 GB/s` (`-17.5%`) |
| `64 MiB` | `20.56 GB/s` | `22.75 GB/s` | `-2.19 GB/s` (`-9.6%`) |
| `256 MiB` | `23.17 GB/s` | `23.02 GB/s` | same class |
| `1 GiB` | `23.96 GB/s` | `23.69 GB/s` | same class |

## Confirmed code-side bottleneck anchors

1. Request control path is currently single-thread serialized.
   - request thread creation:
     [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - loop entry `do_read_request_loop()`:
     [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - queue/inflight structures `request_queue_`, `pending_requests_`:
     [core/communicator/engine/engine.h](../../core/communicator/engine/engine.h)
2. RDMA refill/ACK cadence is conservative.
   - `resume_rdma_reads()` breaks immediately after `made_progress`:
     [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - refill is re-triggered after ACK inside `ENGINE_OP_RDMA_READ_DONE_EX`:
     [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
3. RDMA path bookkeeping is too fine-grained at per-WR level.
   - in `read_multi()`, each posted WR executes `enqueue_completion_bytes` and
     `per_qp_inflight_queues_.push(...)`:
     [core/communicator/transport/rdma_transport.cc](../../core/communicator/transport/rdma_transport.cc)
   - in `do_process_wc()`, each completion performs `pop` and request-state
     updates:
     [core/communicator/transport/rdma_transport.cc](../../core/communicator/transport/rdma_transport.cc)
   - ACK/completion tracking structures in `ReadRequest`:
     [core/communicator/transport/request.h](../../core/communicator/transport/request.h)
4. MTCP staged path has heavy wait chains.
   - staging thread serially consumes `mtcp_staging_queue_`:
     [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - MTCP send/recv path includes many `std::async` and future `wait/get` calls:
     [core/communicator/transport/mtcp_transport.cc](../../core/communicator/transport/mtcp_transport.cc)
5. Hot-path `LOG(INFO)` frequency is high.
   - typical hot spots include `read_tensor` enqueue,
     `resume_rdma_reads`, and `read_multi` posted/completion paths
     (same files above).

## Constraints and boundaries

1. Configuration must use unified runtime config. No temporary env-var
   feature switches are allowed on the main path (see
   [0004-unified-runtime-config.md](../designs/0004-unified-runtime-config.md)
   and
   [communicator_config.proto](../../proto/tensorcast/communicator/v1/communicator_config.proto)).
2. `pending_requests_` lifecycle semantics, error-propagation semantics, and
   idempotent `ReadRequest::set_result()` behavior must remain intact.
3. `FlowCreditLedger` credit limits and release path must keep their safety
   constraints
   ([core/communicator/engine/staging_flow_controller.cc](../../core/communicator/engine/staging_flow_controller.cc)).

# Phases & Milestones

- [ ] Phase 0: Baseline freeze and metric-definition alignment
  - [ ] Milestone 0.1: Freeze the comparison setup for this round
    (`32/64/256 MiB`, `threads=1`, `batch_size=1`, single-NIC strict direct
    RDMA).
  - [ ] Milestone 0.2: Freeze benchmark output fields and parser rules
    (prioritize `amortizable_*` and `recurring_*` while keeping
    `bw_GBps`/`bw_gbps` compatibility).
  - [ ] Milestone 0.3: Record baseline case metadata (host pair,
    GPU/NIC mapping, qp/outstanding settings).

- [ ] Phase 1: Instrumentation first (observe before optimize)
  - [ ] Milestone 1.1: Add a request-level stats snapshot container in
    `ReadRequest` (without changing completion semantics).
  - [ ] Milestone 1.2: Add instrumentation on benchmark/engine/transport hot
    paths (queue wait, control send, window/ACK rounds, WR posted/completion,
    MTCP wait times).
  - [ ] Milestone 1.3: Extend benchmark output with amortizable
    (init/warmup) and recurring (clear/issue/wait/verify) fields, with both
    per-iteration and per-request amortized views.
  - [ ] Milestone 1.4: Split `verify` cost into allocation/copy/checksum to
    produce an amortizable vs non-amortizable attribution report.

- [ ] Phase 2: Control path and window-granularity optimization
  - [ ] Milestone 2.1: Evolve from single request thread to
    "per-peer sharding + fixed worker pool" with `worker_count=1` as
    compatible default.
  - [ ] Milestone 2.2: Introduce per-pass budget (window or bytes) in
    `resume_rdma_reads()`, replacing "wait for ACK after any progress."
  - [ ] Milestone 2.3: Land ACK batching strategy
    (max windows + max delay constraints) while guaranteeing timely credit
    recovery.
  - [ ] Milestone 2.4: Complete functional regression and low-traffic canary
    verification.

- [ ] Phase 3: RDMA/MTCP data-plane micro-optimizations
  - [ ] Milestone 3.1: Converge RDMA inflight bookkeeping from "per-WR records"
    to "batch record + remaining counter."
  - [ ] Milestone 3.2: Add configurable signaled-WR interval
    (default remains backward-compatible).
  - [ ] Milestone 3.3: Reduce serial future waits and active sleep backoff in
    MTCP staged path, moving toward event-driven behavior.
  - [ ] Milestone 3.4: Complete hot-path log noise reduction
    (`INFO -> VLOG` or sampled logging).

- [ ] Phase 4: Acceptance, canary rollout, and default convergence
  - [ ] Milestone 4.1: Run the full validation matrix
    (functionality, performance, non-regression, stability).
  - [ ] Milestone 4.2: Produce recommended config defaults after acceptance
    thresholds are met.
  - [ ] Milestone 4.3: Complete doc sync (design, plan, benchmark result pages).

# Tasks

## Module task breakdown

| Module | Task | Key files |
| --- | --- | --- |
| Engine scheduler layer | Request sharded queues and worker pool, RDMA refill budget, ACK-batch trigger policy | [engine.cc](../../core/communicator/engine/engine.cc), [engine.h](../../core/communicator/engine/engine.h) |
| Flow-control layer | Credit-boundary checks, budget progression and release safety regression | [staging_flow_controller.cc](../../core/communicator/engine/staging_flow_controller.cc) |
| RDMA transport layer | Inflight batching, signaled interval, completion-path adaptation | [rdma_transport.cc](../../core/communicator/transport/rdma_transport.cc) |
| Request state layer | Request-level stats aggregation, ACK-batch metadata | [request.h](../../core/communicator/transport/request.h) |
| MTCP transport layer | Remove high-cost wait chains, reduce active backoff waits | [mtcp_transport.cc](../../core/communicator/transport/mtcp_transport.cc) |
| Config and protocol | Add new tuning config fields with backward-compatible defaults | [communicator_config.proto](../../proto/tensorcast/communicator/v1/communicator_config.proto) |
| Benchmark and docs | Re-run benchmarks, archive results, sync design/plan docs | [20260316 comparison](../benchmarks/20260316-communicator-vs-transfer-engine-comparison.md), [20260315 progress](../benchmarks/20260315-communicator-rdma-nic-gdr-progress.md) |

## Execution-order constraints

1. Phase 1 observability must finish before entering Phase 2/3 to avoid blind
   tuning.
2. Phase 2 and Phase 3 may run in parallel branches, but each optimization must
   be justified by Phase 1 metric improvements.
3. Add config fields before enabling new strategies, and keep default values
   backward-compatible.

# Test / Rollout / Backout

## Validation Matrix

| Dimension | Content | Pass criteria |
| --- | --- | --- |
| Unit/component regression | `//core/communicator:request_test`, `:rdma_engine_test`, `:staging_flow_controller_test`, `:mtcp_transport_lane_test`, `:mtcp_transfer_completion_tracker_test`, `:tcp_engine_test` | All pass, no new flaky tests |
| Config regression | `//core/communicator:config_io_test` + new config field read/write/default tests | New field defaults are equivalent to old behavior |
| Attribution completeness | `ITER`/`SUMMARY` output amortizable + recurring fields together; `verify` split into alloc/copy/checksum | Fields are complete and stably parseable |
| Small-packet structural metrics | `32 MiB`, `64 MiB` strict direct-RDMA single-request baseline comparison | Can quantify dominant recurring costs and identify top bottlenecks |
| Large-packet non-regression | `256 MiB`, `1 GiB` retest under same setup | Regression no more than `3%` vs baseline |
| Cost classification | `amortized_per_iteration_us`, `amortized_per_request_us`, and recurring per-request metrics | Can determine whether one-time cost is amortizable over subsequent transfers |
| Stability | long-run stress + error paths (ACK delay, partial post failure, MTCP fallback) | No deadlock, leak, or silent failure |

## Execution commands (examples)

1. C++ regression:
   - `bazel test //core/communicator:request_test`
   - `bazel test //core/communicator:rdma_engine_test`
   - `bazel test //core/communicator:staging_flow_controller_test`
   - `bazel test //core/communicator:mtcp_transport_lane_test`
   - `bazel test //core/communicator:mtcp_transfer_completion_tracker_test`
   - `bazel test //core/communicator:tcp_engine_test`
   - `bazel test //core/communicator:config_io_test`
2. Benchmark toolchain:
   - `tools/communicator/run_bench_case.py`
   - `tools/communicator/launch_remote_bench_case.py`
   - `tools/communicator/render_te_comparison_charts.py`

## Rollout

1. Step 1: ship instrumentation only (all new strategies disabled by default).
2. Step 2: canary control-granularity optimizations
   (request workers, refill budget, ACK batch).
3. Step 3: canary RDMA/MTCP micro-optimizations
   (bookkeeping, signaled interval, wait-chain optimizations).
4. Step 4: converge recommended defaults based on acceptance and update docs.

## Backout

1. Each new strategy can be independently disabled via config and rolled back to
   old behavior.
2. If `256 MiB+` regresses or stability issues appear, back out Phase 3 first,
   then Phase 2.
3. Keep Phase 1 observability available for post-backout diagnosis.

# Risks & Tracking

- [ ] Risk 1: request-worker parallelization regresses ordering semantics
  - Tracking: preserve order via per-peer sharding and add ordering consistency
    tests.
- [ ] Risk 2: ACK batching slows credit recovery or creates backlog
  - Tracking: enforce dual thresholds (`ack_batch_max_windows`,
    `ack_batch_max_delay_us`) and add timeout alerts.
- [ ] Risk 3: unsignaled-WR interval affects error visibility
  - Tracking: keep default `1`, scale up gradually in canary, and monitor
    completion anomaly rate.
- [ ] Risk 4: MTCP de-blocking changes concurrency behavior and introduces
  latent deadlocks
  - Tracking: add stress tests and timeout-path tests to ensure worker threads
    are not blocked waiting.
- [ ] Risk 5: hot-path log reduction hurts troubleshooting
  - Tracking: keep sampled logs and debug switches while preserving
    `WARNING/ERROR` for critical failures.

# Owner Checklist

- [ ] Design and plan links are bidirectional and resolvable.
- [ ] Milestones are verifiable outcomes, not implementation action labels.
- [ ] All config increments follow unified runtime config.
- [ ] Acceptance matrix covers uplift, non-regression, stability, and rollback.
- [ ] Benchmark-result docs are synced after each key milestone.
