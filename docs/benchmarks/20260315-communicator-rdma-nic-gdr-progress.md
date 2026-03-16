# Communicator RDMA NIC / GDR Structured Summary

Date: 2026-03-15  
Last Updated: 2026-03-16  
Status: Active

## 1. Scope

This document summarizes the TensorCast communicator work to reproduce the main
Mooncake Transfer Engine experiment families:

1. RDMA NIC selection / affinity
2. GPU Direct RDMA locality
3. single-large-read throughput
4. full-8 lane mapping and aggregate throughput

This is no longer a chronological lab notebook. It is a structured summary of:

1. tuning methods
2. performance data
3. code and tooling changes
4. framework-level implications for TensorCast

## 2. Executive Summary

The main outcomes are now:

1. communicator supports strict GPU Direct RDMA with explicit NIC selection and
   no implicit fallback
2. one large strict-direct `read_tensor()` can now reach about `20.25 GB/s`
3. explicit full-8 lane communicator tests can now reach about
   `1204 ~ 1234 Gbps` data-plane aggregate on a good host pair
4. communicator and Mooncake agree that local `GPU <-> NIC` pairing is the
   first-order rule
5. communicator currently shows weaker sensitivity to remote one-to-one lane
   permutation than Mooncake
6. current TensorCast framework can fill all NICs only when the higher layer
   already exposes many independent GPU-to-GPU transfers
7. current TensorCast framework does not automatically stripe one artifact over
   `8` source GPUs/NICs

## 3. Scope and Alignment Rules

### 3.1 Primary Communicator Modes

The communicator side has two GPU RDMA modes:

1. `Direct RDMA`
   - GPU memory is registered directly for RDMA
   - expected signal: `zero_copy=1`
2. `Stage RDMA`
   - GPU memory is copied into a staging buffer before RDMA
   - expected signal: `zero_copy=0`

Execution policy for this study:

1. Direct RDMA is the main GPU experiment track
2. Stage RDMA is kept as a control group
3. DRAM RDMA NIC experiments are tracked separately from GPU Direct RDMA

### 3.2 Comparison Rules

To compare communicator against Mooncake cleanly, the following rules are used:

1. strict NIC selection must be enabled
2. strict direct RDMA must be enabled for direct-path cases
3. only `zero_copy=1` cases count as direct-GDR throughput evidence
4. communicator full-8 aggregate uses lane-aligned data-plane wall:
   - `sum(lane bytes) / max(lane wall_us)`
5. whole-case wall is recorded too, but it includes orchestration and target
   lane bring-up, so it is not the primary throughput metric

### 3.3 Topology Constraint

The H800 nodes in this cluster are not fully topology-identical.

Observed differences:

1. `gpu-h800-0496` has a very clean one-to-one `PIX` pattern
2. `gpu-h800-0550` is not a simple `GPU4 -> mlx5_4` machine
3. mapping and locality experiments must therefore record the actual host pair
   and actual `nvidia-smi topo -m` output

For the key same-pair comparison work in this summary, the most useful pair was:

1. target:
   - `gpu-h800-0360.host.platform.shaipower.com`
2. initiator:
   - `gpu-h800-0496.host.platform.shaipower.com`

On this pair:

1. `GPU0 <-> mlx5_0` is preferred
2. `GPU4 <-> mlx5_4` is preferred
3. `GPU5 <-> mlx5_6`
4. `GPU6 <-> mlx5_7`
5. `GPU7 <-> mlx5_8`

### 3.4 Metric Semantics

This study mixes several benchmark layers, so the metric meaning must stay
explicit.

#### 3.4.1 Communicator Bench Output

The communicator bench prints:

1. `ITER ... total_us=...`
2. `SUMMARY ... wall_us=... avg_us=... p50/p95/p99 ... bw_GBps=...`
3. legacy compatibility:
   - `bw_gbps=...`

The important implementation detail is that the bench computes:

1. `total_bytes` from completed requests
2. `wall_us` from the whole measured interval
3. `bw_GBps = (total_bytes / 1e9) / (wall_us / 1e6)`

New runs now emit the explicit field `bw_GBps`. The historical field name
`bw_gbps` is still emitted as a legacy alias for compatibility with older
parsers and older result files.

#### 3.4.2 Communicator Full-8 Metrics

For full-8 mapping runs, there are two different denominators:

1. `case-wall aggregate`
   - from the runner's outer wall clock
   - includes target lane startup and orchestration
2. `data-plane aggregate`
   - `sum(lane bytes) / max(lane wall_us)`
   - this is the primary metric for communicator capability

The `data-plane aggregate` is the metric used for framework-comparison plots.

#### 3.4.3 Raw RDMA, Mooncake, and Communicator Are Not All The Same Benchmark

The main benchmark layers in this report are:

1. raw verbs micro-benchmark
   - hardware / transport ceiling reference
2. Mooncake Transfer Engine benchmark
   - framework benchmark, primarily write-path in the full-8 best-practice doc
3. communicator bench
   - direct `read_tensor()` based benchmark

So the strongest comparisons in this document are:

1. same-system communicator A/B comparisons
2. same-pair same-path trend comparisons

The weakest comparisons are:

1. cross-framework full-8 absolute comparisons when one side is write and the
   other side is read

### 3.5 Evidence and Confidence Levels

Not every conclusion in this report has the same weight.

| Level | Meaning | Typical Example In This Report |
|---|---|---|
| `L1: Functional proof` | proves the path exists and has the intended properties | strict direct RDMA with `zero_copy=1` and strict NIC validation |
| `L2: Same-workload A/B` | strongest optimization evidence | single-big-read `before -> after` and repeated-direct `before -> after` |
| `L3: Same-pair reference comparison` | useful for framework comparison, but not always apples-to-apples | raw verbs vs Mooncake vs communicator on the same host pair |
| `L4: Cross-pair or doc-derived reference` | directionally useful, but least strict | Mooncake full-8 best-practice ceiling references |

When reading the result tables:

1. performance attribution claims should preferably come from `L2`
2. framework comparison claims are usually `L3`
3. high-level product positioning often relies on `L3 + L4`

## 4. Tuning Methods Summary

### 4.1 Functional Safety and No-Fallback Policy

The first set of changes established hard validation rather than permissive
fallback:

1. strict NIC selection was added to the bench and validation path
2. strict direct RDMA was added so a requested direct path fails instead of
   silently falling back
3. RDMA responses now surface enough signals to verify the path:
   - `local_nic`
   - `remote_nic`
   - `local_rail`
   - `remote_rail`
   - `rdma`
   - `staged`
   - `zero_copy`

Functional baseline already proven:

1. case:
   - `/data/tc/comm-rdma-strict-20260315-040015`
2. key signal:
   - `local_nic=mlx5_0 remote_nic=mlx5_0 rdma=1 zero_copy=1`

This baseline proves:

1. explicit NIC choice works
2. direct GPU MR registration works
3. communicator can complete a strict no-fallback GPU Direct RDMA read

### 4.2 Correctness and Protocol Fixes

Several real bugs had to be fixed before throughput numbers were trustworthy.

#### 4.2.1 Staged RDMA Multi-window Protocol

Root issues:

1. `READ_RESPONSE_EX` lookup used the wrong offset for later windows
2. `ReadRequest` could complete after the first window instead of the final
   one
3. successful requests were not always cleaned out of `pending_requests_`

Fixes:

1. `ProtoReadResponseExHeader` carries `request_offset`
2. client-side lookup uses the original request offset
3. `ReadRequest` now waits for the final RDMA window
4. `ReadRequest::set_result()` is idempotent
5. successful completion removes pending request state

#### 4.2.2 Failure-path Cleanup

Root issue:

1. some RDMA error paths tried to delete already-removed pending keys and could
   `LOG(FATAL)`

Fix:

1. cleanup paths now use `erase_if_present` where the operation is logically
   idempotent

#### 4.2.3 Request Identity and Repeated Reads

Root issue:

1. multi-threaded repeated reads could collide on request identity

Fix:

1. every request now has a unique `request_id`

#### 4.2.4 Rail Semantics for Mapping Cases

Root issue:

1. target RDMA request handling incorrectly treated initiator `rail_id` as a
   target-side rebind instruction
2. identity mapping hid this bug
3. `within-half` and `cross-half` mapping exposed it immediately

Fix:

1. target now keeps its own local tensor/NIC rail instead of reusing the
   initiator rail

### 4.3 Single-Lane Throughput Tuning

The main throughput bottlenecks were not only in RDMA itself.

#### 4.3.1 Request-side MR Reuse

Change:

1. repeated reads of the same local buffer now prefer `MrCache`

Why it matters:

1. otherwise local MR registration cost repeats on every read

#### 4.3.2 Direct-Path Credit Leak

Root issue:

1. direct zero-copy `StageLease` credit was not explicitly released
2. after one iteration, later requests could stall on exhausted credit

Fix:

1. direct-path leases are explicitly released after successful send and on
   send-failure cleanup

#### 4.3.3 Refill Race

Root issue:

1. a late-arriving read could miss the next refill trigger while
   `rdma_refill_in_progress` was already true

Fix:

1. explicit refill request bookkeeping was added so a second pass is forced
   after late arrivals

#### 4.3.4 Multi-QP and QP Send Depth

Root issues:

1. single-QP defaults were too conservative for high direct-RDMA inflight
2. `RdmaConfig.outstanding_wr` existed but was not wired into QP creation
3. `read_multi()` originally posted one entire window to one QP

Fixes:

1. `qp_count` is now an explicit tuning input in the bench and transport
2. `outstanding_wr` is now wired into `max_send_wr`
3. a multi-segment window is now spread across multiple QPs

#### 4.3.5 Direct Request-scoped Window Expansion

Problem:

1. direct zero-copy requests were still limited by channel-shared direct window
   sizing

Fix:

1. request-scoped direct ledgers can temporarily expand direct in-flight
   segments beyond the old shared limit

#### 4.3.6 Adaptive Direct Chunking

Problem:

1. one large `64 MiB` direct read was under-segmented
2. it originally produced only `4 x 16 MiB` RDMA READ WRs

Fix:

1. direct RDMA chunking is now adaptive for large requests
2. the same `64 MiB` request now uses `16 x 4 MiB`

#### 4.3.7 Control TCP Latency

Problem:

1. communicator control TCP sockets did not enable `TCP_NODELAY`
2. single-request latency was consistent with delayed ACK / Nagle behavior

Fix:

1. communicator control sockets now enable low-latency TCP settings
   - `TCP_NODELAY`
   - `TCP_QUICKACK`

### 4.4 Full-8 Mapping Methodology

Mooncake's full-8 mapping logic was translated into communicator with one
important modeling choice:

1. one lane = one communicator process on each host
2. each lane is pinned to exactly one GPU and one NIC
3. aggregate throughput is built by running `8` explicit lane pairs in parallel

This produced two reusable tools:

1. `tools/communicator/run_mapping_host.py`
2. `tools/communicator/launch_remote_mapping_case.py`

Runner hardening that was needed:

1. remote shell composition in the mapping launcher
2. NUMA wrapper invocation using the correct Python executable
3. sequential target lane startup
   - each target lane reaches `READY` before the next is launched

### 4.5 Correctness Validation Strategy

Correctness was verified at three levels:

1. deterministic local unit tests
2. same-host real hardware smoke
3. cross-host strict direct benchmark validation

New local deterministic tests added in this work:

1. `ReadRequest completion path is concurrency-safe`
2. `StageLease concurrent release is idempotent`
3. `StagingWindow supports request-scoped direct credit expansion`

Same-host real hardware smoke conclusions:

1. aggressive same-host concurrent load can fail with default send queue depth
2. the same verified workload passes once `outstanding_wr=256`

### 4.6 Optimization Impact Summary

The most important missing view is not just *what* we changed, but *how much*
each change moved the result.

Two notes are important before reading the table:

1. the current communicator bench field name `bw_gbps` is historically named
   but the reported value is used here as `GB/s`
2. only the entries below with a stable before/after pair are presented as
   quantitative A/B comparisons

#### 4.6.1 Communicator A/B Impact Table

| Optimization Track | Workload | Before | After | Delta | Main Reading | Evidence |
|---|---|---:|---:|---:|---|---|
| direct repeated-read path hardening | `0078 <- 0496`, strict direct, `GPU0 + mlx5_0`, `threads=8` | `1.78507 GB/s` | `11.5755 GB/s` | `+548.5%` (`6.48x`) | repeated direct reads stopped wasting work on request collisions, repeated MR registration, direct credit leaks, and refill races | `/data/tc/comm-bench-20260315-gpu-direct-mlx5_0-threads8-fixed-open-user-132524/result.json` -> `/data/tc/comm-postopt-0078-0496-manual-rerun4-141623/initiator.log` |
| single large `read_tensor()` cumulative gain | `0360 <- 0496`, `64 MiB`, `threads=1`, `batch=1`, `qp_count=4` | `1.43004 GB/s` | `20.2483 GB/s` | `+1315.9%` (`14.16x`) | large single-request throughput only reached line-rate class after the transport and control-path fixes were combined | `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-attempt1-020457/result.json` -> `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json` |
| adaptive direct chunking plus `outstanding_wr` alone | same single-large-read workload as above | `1.43004 GB/s` | `1.29556 GB/s` | `-9.4%` | chunking the request into `16 x 4 MiB` and raising QP send depth did not help on their own because the control TCP path was still dominant | `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-attempt1-020457/result.json` -> `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-adaptive2-attempt1-024552/result.json` |
| low-latency control TCP on top of the chunked direct path | same single-large-read workload as above | `1.29556 GB/s` | `20.2483 GB/s` | `+1462.9%` (`15.63x`) | once `TCP_NODELAY` and `TCP_QUICKACK` were enabled, the already-chunked direct path finally translated into end-to-end throughput | `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-adaptive2-attempt1-024552/result.json` -> `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json` |

#### 4.6.2 Parameter and Topology Sensitivity Table

These are also important, but they are not all pure code A/Bs.

| Topic | Setting A | Setting B | Effect | Reading | Evidence |
|---|---|---|---:|---|---|
| Mooncake `MC_IB_PCI_RELAXED_ORDERING` | default ordering: `7.21 GB/s` | `RO=1`: `10.12 GB/s` | `+40.4%` | relaxed ordering is a first-order knob for Mooncake on this host pair | `/data/tc/mooncake-tuning-0360-0496-003649/moon_default_initiator_exec.json` -> `/data/tc/mooncake-tuning-0360-0496-003649/moon_ro1_initiator_exec.json` |
| Mooncake `MC_SLICE_SIZE` on top of `RO=1` | `RO=1`: `10.12 GB/s` | `RO=1 + SLICE=1MiB`: `8.69 GB/s` | `-14.1%` | larger slice size is not automatically better | `/data/tc/mooncake-tuning-0360-0496-003649/moon_ro1_initiator_exec.json` -> `/data/tc/mooncake-tuning-0360-0496-003649/moon_ro1_slice1m_initiator_exec.json` |
| communicator full-8 local affinity | `map8_id`: `150.56 GB/s` data-plane aggregate | `map8_target_bad_local`: `83.61 GB/s` data-plane aggregate | `-44.3%` | keeping each host's local `GPU <-> NIC` pairing correct matters far more than the remote permutation itself | `/data/tc/comm-map8-id-rerun-attempt1-043514/result.json` -> `/data/tc/comm-map8-target-bad-local-attempt1-052344/result.json` |

#### 4.6.3 Cases Without A Clean Numeric A/B

Some changes were still important even though they do not have a fair
"same-workload before/after" number:

1. `qp_count=4` plus larger `outstanding_wr` turned the high-inflight
   `4 MiB x 16 x 8` workload from an unstable / queue-depth-limited path into a
   stable `19.8634 GB/s` point
2. strict direct RDMA and strict NIC selection are functional guardrails rather
   than throughput optimizations, but they were required to ensure the measured
   path was the intended RDMA path
3. staged-RDMA multi-window protocol fixes were correctness fixes first; their
   main value is that repeated runs stopped aborting and became benchmarkable

## 5. Performance Data Summary

### 5.1 Reference Hardware and Mooncake Baselines

Key same-pair reference numbers on `0360 <- 0496`:

| Metric | Result | Evidence |
|---|---:|---|
| raw verbs single-NIC ceiling | `127.79 Gb/s` = `15.97 GB/s` | `/data/tc/mooncake-tuning-0360-0496-003649/verbs` |
| Mooncake preferred GDR (`RO=1`) | `10.12 GB/s` | `/data/tc/mooncake-tuning-0360-0496-003649/moon_ro1_initiator_exec.json` |
| Mooncake far GDR (`RO=1`) | `7.92 GB/s` | `/data/tc/mooncake-far-0360-0496-004020` |

Mooncake full-8 framework best-practice reference:

1. raw full-8 RDMA:
   - `1568.75 Gbps`
2. full-8 Transfer Engine best:
   - `1482.64 Gbps`
3. source:
   - `/data/workspace/Mooncake/docs/source/performance/transfer-engine-full8-rdma-best-practices-20260316.md`

### 5.2 CPU RDMA Summary

CPU staged-RDMA evidence:

| Case | Host Pair | Result |
|---|---|---|
| `mlx5_0 -> mlx5_0` | `0426 -> 0550` | `0.772268 GB/s` |
| `mlx5_5 -> mlx5_5` | `0496 -> 0426` | `0.994633 GB/s` |
| `mlx5_4 -> mlx5_4` | one tested host pair | RDMA connect timeout |

Current reading:

1. communicator no longer has a protocol bug that prevents repeated CPU RDMA
   experiments
2. CPU NIC results are still host-pair-sensitive
3. communicator does not yet support a Mooncake-style PF heatmap conclusion for
   CPU DRAM RDMA

### 5.3 Single-Lane GPU RDMA Summary

Important communicator milestones:

| Case | Setting | Result | Notes |
|---|---|---:|---|
| strict direct baseline | `mlx5_0`, `GPU0` | functional success | `zero_copy=1` |
| early preferred case | `0360 <- 0496`, preferred | `8.43 GB/s` | before final single-request tuning |
| early far-NIC case | `0360 <- 0496`, far | `7.35 GB/s` | weaker than Mooncake far penalty |
| tuned large single read | `64 MiB`, `threads=1`, `batch=1`, `qp_count=4`, `outstanding_wr=256` | `20.25 GB/s` | strict direct `read_tensor()` |

Key final single-request case:

| Case | Result | Evidence |
|---|---:|---|
| communicator single big read | `20.25 GB/s` | `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json` |

This final result is important because:

1. it is a single large strict-direct `read_tensor()` path
2. it is close to the hardware-class target for one lane
3. it came from engine behavior improvements rather than only bench-level
   concurrency tricks

### 5.4 Full-8 Mapping Summary

#### 5.4.1 Case-wall Aggregate

This metric includes orchestration overhead and target lane startup.

| Case | Aggregate |
|---|---:|
| `map8_id` | `68.35 GB/s` |
| `map8_within_swap` | `69.56 GB/s` |
| `map8_half_swap` | `69.37 GB/s` |
| `map8_target_bad_local` | `38.04 GB/s` |

Evidence:

1. `map8_id`
   - `/data/tc/comm-map8-id-rerun-attempt1-043514/result.json`
2. `map8_within_swap`
   - `/data/tc/comm-map8-within-swap-clean-attempt1-050919/result.json`
3. `map8_half_swap`
   - `/data/tc/comm-map8-half-swap-attempt2-052059/result.json`
4. `map8_target_bad_local`
   - `/data/tc/comm-map8-target-bad-local-attempt1-052344/result.json`

#### 5.4.2 Data-plane Aggregate

This is the better metric for communicator full-8 capability:

1. `sum(lane bytes) / max(lane wall_us)`

Using that lane-aligned wall:

| Case | Aggregate |
|---|---:|
| `map8_id` | `150.56 GB/s` = `1204.45 Gbps` |
| `map8_within_swap` | `153.94 GB/s` = `1231.49 Gbps` |
| `map8_half_swap` | `154.21 GB/s` = `1233.69 Gbps` |
| `map8_target_bad_local` | `83.61 GB/s` = `668.89 Gbps` |

Interpretation:

1. among the three cases that preserve good local GPU/NIC pairs, communicator
   varies only about `1.8%`
2. but once destination local pairing is broken, aggregate throughput drops by
   about `44.3%`
3. communicator therefore strongly agrees with Mooncake on local-affinity
   importance
4. communicator does not currently reproduce Mooncake's stronger sensitivity to
   remote permutation after local pairing is already correct

### 5.5 Communicator vs Mooncake Mapping Reading

On the same conceptual mapping axis:

1. Mooncake says:
   - good local pairing first
   - then within-half is better than identity
   - cross-half is worse
2. communicator says:
   - good local pairing first
   - once good local pairing is preserved, identity / within-half / cross-half
     are very close
   - but breaking destination local pairing is extremely expensive

So the shared rule is:

1. local topology correctness matters first

The remaining difference is:

1. Mooncake's aggregated write engine is more sensitive to remote permutation
   than communicator's current `8` decoupled direct-read model

## 6. Code and Tooling Changes Summary

### 6.1 Communicator Protocol and Request Lifecycle

Files:

1. `core/communicator/engine/protocol.h`
2. `core/communicator/engine/engine.cc`
3. `core/communicator/transport/request.h`
4. `core/communicator/transport/request.cc`
5. `core/communicator/misc/map.h`

Changes:

1. request-scoped RDMA response lookup
2. final-window completion semantics
3. idempotent `set_result()`
4. pending-request cleanup made safe for concurrent failure paths
5. unique request IDs for repeated multi-threaded reads

### 6.2 Direct RDMA Transport Tuning

Files:

1. `core/communicator/transport/rdma_context.h`
2. `core/communicator/transport/rdma_transport.h`
3. `core/communicator/transport/rdma_transport.cc`
4. `core/communicator/engine/engine.cc`

Changes:

1. `qp_count` exposed and used
2. `outstanding_wr` wired into QP creation
3. one window's WRs spread across multiple QPs
4. request-scoped direct ledgers for larger zero-copy windows
5. adaptive direct chunking for large single-request reads
6. direct credit leak fixed
7. refill race fixed
8. target rail semantics fixed for mapping cases

### 6.3 Control TCP Latency

File:

1. `core/communicator/transport/tcp_context.cc`

Changes:

1. control sockets enable:
   - `TCP_NODELAY`
   - `TCP_QUICKACK`

This was essential for lifting single large `read_tensor()` latency from
`~45 ms` request time down to sub-millisecond control-path behavior.

### 6.4 Benchmark and Mapping Infrastructure

Files:

1. `core/communicator/bench/communicator_bench.cc`
2. `tools/communicator/run_bench_case.py`
3. `tools/communicator/launch_remote_bench_case.py`
4. `tools/communicator/run_with_numa_policy.py`
5. `tools/communicator/run_mapping_host.py`
6. `tools/communicator/launch_remote_mapping_case.py`

Changes:

1. strict direct RDMA and strict NIC validation
2. output signals for `local_nic/remote_nic/rail/staged/zero_copy`
3. support for:
   - `threads`
   - `batch_size`
   - `qp_count`
   - `outstanding_wr`
4. detached worker control with explicit host validation
5. full-8 multi-lane mapping runner

### 6.5 Correctness Tests

Files:

1. `core/communicator/transport/request_test.cc`
2. `core/communicator/engine/staging_flow_controller_test.cc`
3. `core/communicator/engine/rdma_engine_test.cc`

New or updated coverage:

1. concurrent `ReadRequest` completion and idempotent result publication
2. concurrent `StageLease` release
3. direct request-scoped credit expansion
4. RDMA connect-failure cleanup tolerance when pending-request state is already
   gone

## 7. TensorCast Framework-Level Summary

### 7.1 What The Current Framework Naturally Does

Current TensorCast P2P loading is still fundamentally single-source-session
oriented.

Evidence:

1. `RemoteKeySource` issues `read_tensor()` to one fixed `ip:port`
   - `core/store/materialization/dataplane/sources/remote_key_source.cc`
2. Global Store transport assignment is one source replica per transport
   request
3. one artifact load does not automatically stripe itself across multiple
   source replicas

### 7.2 Current Framework Limits

Three defaults shape real framework behavior:

1. artifact chunk size default:
   - `256 MiB`
   - `core/common/const/granularity.h`
2. materialization pipeline concurrency default:
   - `4`
   - `core/store/materialization/contracts/loading_spec.h`
3. per-GPU gate:
   - one active transfer session per target GPU
   - `core/store/replica/transfer_service.cc`

### 7.3 Can Current TensorCast Fill Every NIC?

Yes, but only when the workload shape already exposes enough independent
transfers.

What the current framework can do:

1. if there are many independent tensors / shards targeting different GPUs, the
   shared communicator can run those lanes concurrently
2. in that regime, communicator now has evidence for near-full-8-NIC data-plane
   use

What the current framework does not naturally do:

1. one artifact load does not auto-stripe over `8` source GPUs/NICs
2. one artifact to one target GPU will not fill all `8` NICs
3. a single source replica also has a transport concurrency budget

Practical product reading:

1. communicator is no longer the main blocker for per-lane line rate
2. current TensorCast can fill all NICs only when the upper layer already
   exposes many independent GPU-to-GPU transfers
3. automatic full-8 striping of one artifact would require a new framework
   feature above communicator

### 7.4 Capability vs Current Product Behavior

This distinction is important enough to state in table form.

| Layer | What Is Proven Today | What Is Not Automatically True |
|---|---|---|
| communicator transport capability | one large strict-direct `read_tensor()` can reach about `20.25 GB/s`; explicit full-8 lane runs can reach about `150 ~ 154 GB/s` data-plane aggregate | this does not imply one product artifact load automatically uses all lanes |
| current TensorCast framework behavior | if the workload already exposes many independent GPU-to-GPU transfers, communicator can be driven into a near-full-8 regime | one artifact still does not auto-stripe across `8` source GPUs/NICs |
| product-level feature set | current product can benefit directly from per-lane communicator gains | automatic multi-source striping for one artifact would still need a new orchestration feature above communicator |

## 8. Current Best-Practice Configuration

For communicator direct RDMA on the tested H800 class machines, the most useful
configuration today is:

1. strict direct RDMA
2. explicit NIC selection
3. `qp_count=4`
4. `outstanding_wr=256`
5. adaptive direct chunking
6. low-latency control TCP
7. target and initiator local preferred GPU/NIC pairing
8. for full-8 explicit lane tests:
   - run one lane per process
   - bind each process to the NIC's NUMA node

## 9. Remaining Gaps and Bottlenecks

The main remaining gaps are now framework-level and product-level rather than
raw communicator transport correctness.

| Gap / Bottleneck | Evidence | Current Reading | Best Next Step |
|---|---|---|---|
| communicator is still modeled as `8` decoupled reads, not one aggregated submission engine | full-8 local-pair cases are strong, but remote-permutation sensitivity is weaker than Mooncake | communicator may be leaving some cross-lane scheduling benefit on the table | add an engine-level batched / readv-style submission path |
| TensorCast artifact loading is still single-source-session oriented | `RemoteKeySource` reads from one fixed `ip:port`; `pipeline_concurrency` defaults to `4`; per-target-GPU gate is `1` | communicator capability is ahead of current product orchestration | design framework-level striping / multi-source orchestration for one artifact |
| CPU staged-RDMA PF ranking is incomplete | only a small subset of CPU PF cases was validated in communicator | communicator CPU NIC conclusion is not yet equivalent to Mooncake's `9x9` PF heatmap | add a dedicated communicator CPU PF sweep harness if CPU RDMA remains important |
| `rdma_engine_test` is not a clean environment-independent signal | fake-CUDA local environment still hits older environment-dependent failures first | current regression confidence relies more on targeted unit tests and hardware smoke | either isolate the failing legacy setup path or split a narrower RDMA regression test target |

## 10. Repro and Evidence Anchors

The most important claims in this report are tied to the following anchor
cases.

| Claim | Evidence Class | Case / Source |
|---|---|---|
| strict direct RDMA works with no fallback | `L1` | `/data/tc/comm-rdma-strict-20260315-040015` |
| repeated direct-read path improved from `1.78507` to `11.5755 GB/s` | `L2` | `/data/tc/comm-bench-20260315-gpu-direct-mlx5_0-threads8-fixed-open-user-132524/result.json` and `/data/tc/comm-postopt-0078-0496-manual-rerun4-141623/initiator.log` |
| one large strict-direct `read_tensor()` improved to `20.2483 GB/s` | `L2` | `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-attempt1-020457/result.json`, `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-adaptive2-attempt1-024552/result.json`, `/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json` |
| communicator full-8 good-local aggregate reaches about `1204 ~ 1234 Gbps` | `L3` | `/data/tc/comm-map8-id-rerun-attempt1-043514/result.json`, `/data/tc/comm-map8-within-swap-clean-attempt1-050919/result.json`, `/data/tc/comm-map8-half-swap-attempt2-052059/result.json` |
| destination local affinity break causes a large aggregate drop | `L3` | `/data/tc/comm-map8-target-bad-local-attempt1-052344/result.json` |
| Mooncake full-8 best-practice reaches `1482.64 Gbps` | `L4` | `/data/workspace/Mooncake/docs/source/performance/transfer-engine-full8-rdma-best-practices-20260316.md` |
| Mooncake mapping prefers within-half over identity in full-8 write | `L4` | `/data/workspace/Mooncake/docs/source/performance/transfer-engine-gdr-mapping-20260315.md` |

## 11. Related Documents

1. communicator vs Transfer Engine chart set:
   - `docs/benchmarks/20260316-communicator-vs-transfer-engine-comparison.md`
2. Mooncake full-8 best-practice entry:
   - `/data/workspace/Mooncake/docs/source/performance/transfer-engine-full8-rdma-best-practices-20260316.md`
3. Mooncake mapping study:
   - `/data/workspace/Mooncake/docs/source/performance/transfer-engine-gdr-mapping-20260315.md`
