# Global Store Control Plane HA/Resolve Load Analysis (2026-02-22)

## Scope

This report isolates the Global Store control-plane behavior under mixed:

- `ResolveKeyMapping` read traffic from receiver polling.
- HA `WorkerHeartbeat` traffic.
- HA `ReconcileWorkerState` traffic.
- HA `BatchUpdateChunkStates` traffic.

Focus:

- Why the control plane can saturate with relatively small node counts.
- Whether there is optimization headroom in framework-level design (not retry masking).
- Whether new observability can show executor pressure (`busy/idle/queue`) clearly.

## Workload Generator

`examples/cross_host/global_store_control_plane_benchmark.py`

Added in this round:

- Progress log with control-plane telemetry:
  - `inflight`, `inflight_peak`, `busy`, `idle`, `queue`, `reducer_queue`, `max_workers`.
- Prometheus scrape and final JSON summary of control-plane metrics.
- Reducer knobs for stress tests:
  - `--reducer-shards`
  - `--reducer-queue-capacity`
  - `--reducer-coalesce-window-ms`

## Core Improvements Evaluated

Implemented before/within this analysis cycle:

1. Key mapping hot-read cache inside GS (`KeyMappingRepository`) to avoid DB hit on every resolve.
2. Chunk-directory batch set-based upsert (`ChunkDirectoryRepository`) to reduce lock hold and SQL overhead.
3. Daemon chunk sync changed from full periodic snapshot to delta publish (`WorkerLifecycleManager::chunk_sync_loop`).
4. Control-plane telemetry gauges added:
   - `tc_control_plane_executor_{max,live,busy,idle,queue_depth}`
   - `tc_grpc_server_inflight_requests`
   - `tc_grpc_server_inflight_peak_requests`
5. Metrics-port bugfix for `metrics_port=0` (ephemeral port) so benchmark scraping works.

## Test Matrix

All tests are single-process GS local benchmark runs using the same synthetic shape and explicit settings.

| Case | workers | resolver_threads x qps | hb interval | reconcile interval | chunk sync interval | gs_max_workers | reducer_shards |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_pre_fix | 16 | 32 x 20 | 0.5s | 2.0s | 2.0s | 10 | 1 |
| baseline_post_fix | 16 | 32 x 20 | 0.5s | 2.0s | 2.0s | 10 | 1 |
| ha_mix_mw10_rs1 | 16 | 32 x 20 | 0.1s | 0.5s | 0.5s | 10 | 1 |
| ha_mix_mw10_rs8 | 16 | 32 x 20 | 0.1s | 0.5s | 0.5s | 10 | 8 |
| ha_only_mw10 | 16 | 0 | 0.1s | 0.5s | 0.5s | 10 | 1 |
| ha_mix_mw24_rs1 | 16 | 32 x 20 | 0.1s | 0.5s | 0.5s | 24 | 1 |
| ha_mix_mw24_rs8 | 16 | 32 x 20 | 0.1s | 0.5s | 0.5s | 24 | 8 |
| resolve_only_mw10 | 0 | 32 x 20 | N/A | N/A | N/A | 10 | 1 |

## Key Results

### 0) This round step-by-step execution (requested)

We executed the two framework-level actions in sequence under identical workload:

- Workload: `workers=16`, `resolver_threads=32`, `resolve_qps_per_thread=20`,
  `heartbeat=0.1s`, `reconcile=0.5s`, `chunk_sync=0.5s`,
  `gs_max_workers=10`, `reducer_shards=1`.

`before` (`ha_mix_mw10_rs1`) -> `step1` (heartbeat lane decouple) -> `step2` (heartbeat write-throttle):

- `ResolveKeyMapping`
  - rps: `135.73 -> 184.15 -> 202.90`
  - p95: `282.52ms -> 208.89ms -> 202.89ms`
- `WorkerHeartbeat`
  - rps: `28.04 -> 34.14 -> 35.84`
  - p95: `530.79ms -> 424.63ms -> 417.15ms`
- `ReconcileWorkerState`
  - rps: `18.11 -> 19.90 -> 20.62`
  - p95: `439.88ms -> 343.88ms -> 357.67ms`
- `BatchUpdateChunkStates`
  - rps: `21.16 -> 23.12 -> 23.70`
  - p95: `295.62ms -> 222.52ms -> 230.72ms`
- Control-plane queue mean:
  - `46.04 -> 43.80 -> 41.78`
- Reducer queue mean:
  - `6.34 -> 2.00 -> 1.98`

### 1) Before/After optimization impact (same workload)

`baseline_pre_fix` -> `baseline_post_fix`

- `ResolveKeyMapping`: `31.53 rps / p95 1322.71ms` -> `610.68 rps / p95 24.23ms`
- `WorkerHeartbeat`: `8.31 rps / p95 2070.93ms` -> `25.49 rps / p95 266.55ms`
- `ReconcileWorkerState`: `4.93 rps / p95 1386.81ms` -> `7.61 rps / p95 231.46ms`
- `BatchUpdateChunkStates`: `3.32 rps / p95 3284.97ms` -> `7.61 rps / p95 182.61ms`

Conclusion: previously observed saturation was largely due to control-plane hot paths repeatedly hitting serialized DB work. The implemented cache + batch/delta reductions removed the worst amplification.

### 2) Why still saturates under high-frequency HA

`ha_mix_mw10_rs1` (`hb=0.1s`, `reconcile=0.5s`, `chunk=0.5s`, + resolve load):

- `busy_mean=9.8/10`, `queue_depth_mean=46.04` (`p95=49`), `inflight_peak_mean=9.8`
- Progress shows near-constant full occupancy:
  - `inflight_peak=10 busy=10 idle=0 queue≈45-49`

Even with only 16 workers, request rate is already high:

- heartbeat: `16 / 0.1 = 160 rps`
- reconcile: `16 / 0.5 = 32 rps`
- chunk-sync: `16 / 0.5 = 32 rps`
- resolve target: `32 * 20 = 640 rps`

Total attempted control-plane calls ~= `864 rps`.

### 3) HA traffic alone can saturate (without resolve)

`ha_only_mw10`:

- `busy_mean=9.6/10`, `queue_depth_mean=13.0`
- `WorkerHeartbeat p95=461.96ms`
- `ReconcileWorkerState p95=379.14ms`
- `BatchUpdateChunkStates p95=245.82ms`

Conclusion: high-frequency HA periodic sync itself can saturate a 10-thread control plane; resolve traffic is an amplifier, not the sole cause.

### 4) Increasing thread pool alone is insufficient

`ha_mix_mw24_rs1`:

- Resolve recovered (`626.18 rps`, `p95 66.47ms`), but
- Heartbeat still regressed vs `mw10` (`18.21 rps`, `p95 920.01ms`)
- Reconcile worsened (`14.09 rps`, `p95 757.69ms`)
- `busy_mean=23.52/24`, `reducer_queue_mean=19.12`

Interpretation: raising `max_workers` reduces one queue but shifts pressure downstream to serialized worker-control mutation lanes and DB write contention.

### 5) Reducer sharding helps reconcile/queue, not heartbeat fairness

`ha_mix_mw24_rs8`:

- `reducer_queue_mean`: `19.12 -> 7.90` (improved)
- `ReconcileWorkerState`: `14.09 -> 20.44 rps` (improved)
- `ResolveKeyMapping p95`: `66.47 -> 34.94ms` (improved)
- `WorkerHeartbeat`: worsened (`18.21 -> 14.69 rps`, `p95 ~1.27s`)

Interpretation: reducer sharding increases aggregate throughput but heartbeat is still serialized behind per-worker reconcile work on the same worker-key lane.

### 6) Under `mw10`, reducer sharding improves reconcile and reducer backlog

`ha_mix_mw10_rs8` vs `ha_mix_mw10_rs1`:

- `reducer_queue_mean`: `6.34 -> 2.44`
- `ReconcileWorkerState`: `18.11 -> 20.75 rps`, `p95 439.88ms -> 325.44ms`
- `ResolveKeyMapping`: `135.73 -> 159.26 rps`, `p95 282.52ms -> 241.73ms`
- `WorkerHeartbeat`: roughly flat (`28.04 -> 27.18 rps`, p95 slightly worse)

Interpretation: reducer sharding is useful, but executor/DB pressure still dominates and heartbeat fairness remains unresolved.

## ResolveKeyMapping Chain (for root-cause context)

Receiver polling path:

1. SDK: `StoreRuntime.resolve_key_mapping_cached(key)`
2. Daemon RPC: `ResolveKeyMapping`
3. Daemon `KeyMappingController` local cache check
4. StoreEngine `resolve_key_mapping` -> MetadataGateway -> GlobalStoreClient
5. GS `ArtifactCatalogService.ResolveKeyMapping` -> `KeyMappingRpcHandler` -> `KeyMappingRepository.get`

Critical semantic detail:

- Alias key mappings now return a **configurable short TTL** from GS
  (`worker_policy.key_mapping.alias_cache_ttl`, default `1s`).
- This keeps alias freshness near real-time while reducing tight polling storms.

## Root Cause Summary

The saturation is not due to node count alone; it is due to `request_rate x per-request service_time` and serial bottlenecks.

Primary causes:

1. High HA periodic rates produce substantial write load even at 16 workers.
2. GS DB execution path still has strong serialization characteristics (global execution lock + write critical sections).
3. Worker control reducer serializes heartbeat and reconcile intents per worker key; heavy reconcile can delay heartbeat significantly.
4. Resolve polling (especially when alias TTL is too short for workload) adds sustained read pressure and head-of-line contention when HA writes are busy.

## Observability Outcome

Now we can directly observe:

- executor capacity and saturation (`busy/idle/max/live`)
- queue depth (`tc_control_plane_executor_queue_depth`)
- request occupancy (`inflight` + `inflight_peak`)
- reducer backlog (`tc_worker_control_reducer_queue_depth`)

This provides concrete evidence when a run is thread-saturated versus reducer-saturated.

## 2026-02-23 Follow-up (P0-1/P0-2 landed)

Landed changes in this round:

1. Daemon `ResolveKeyMapping` now inherits client deadline budget and propagates cancellation to upstream GS RPCs.
2. GS alias key mapping TTL is configurable (default `1s`) instead of fixed `0`.

Re-ran workload (same shape as step-by-step matrix):

- `workers=16`
- `resolver_threads=32`, `resolve_qps_per_thread=20`
- `heartbeat=0.1s`, `reconcile=0.5s`, `chunk_sync=0.5s`
- `gs_max_workers=10`, `reducer_shards=1`
- duration: `30s`

Observed (single run):

- `ResolveKeyMapping`: `225.31 rps`, `p95 192.51ms`
- `WorkerHeartbeat`: `41.07 rps`, `p95 355.11ms`
- `ReconcileWorkerState`: `19.89 rps`, `p95 381.84ms`
- `BatchUpdateChunkStates`: `24.10 rps`, `p95 208.19ms`
- control-plane queue depth mean: `41.70`
- reducer queue depth mean: `2.93`
- executor busy mean: `9.67 / 10`

Compared to prior `step2` sample in this doc, resolve/heartbeat/chunk are better in this run while reconcile is within same range with slight p95 regression. Because this is not a strict A/B same-host repeat with multiple runs, treat this as directional validation, not a statistical claim.

## Optimization Space (Framework-level, long-term)

Priority order from observed data:

1. **Decouple heartbeat from heavy reconcile lane per worker**
   - Current same-key serialization makes heartbeat tail latency sensitive to reconcile cost.
   - Move to priority scheduling or separate intent lanes with conflict-safe write strategy.

2. **Reduce HA write amplification**
   - Keep reconcile snapshot semantics, but allow stricter diff/incremental modes for stable periods.
   - Consider batching heartbeat persistence (`batch_update_heartbeats`) under bounded flush interval.

3. **Further reduce DB critical-section time**
   - Keep set-based SQL pattern for all high-frequency updates.
   - Audit long transactions in reconcile path and split read/compute/write segments where safe.

4. **Resolve path pressure control**
   - Keep correctness semantics for alias freshness, but explore bounded short TTL + generation guard,
     or push/subscribe key mapping updates to receivers to avoid tight polling.

## Validation Status

- Python tests:
  - `pytest tests/python/global_store/test_binding_key_mapping_rpc.py`
  - Result: `7 passed`
  - `pytest tests/python/global_store/test_configuration.py`
  - Result: `16 passed`
  - `pytest tests/python/global_store/test_grpc_service.py`
  - Result: `45 passed`
  - `pytest tests/python/global_store/test_worker_control_plane_stage23.py`
  - Result: `4 passed`
  - `pytest tests/python/global_store/test_worker_control_plane_stage1.py`
  - Result: `2 passed`
- C++ tests:
  - `bazel test //daemon:grpc_service_impl_publish_replica_key_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
  - Result: `PASSED`
- Benchmark:
  - `python examples/cross_host/global_store_control_plane_benchmark.py ... --duration-s 30 --output-json /tmp/gs_control_plane_step12_20260222.json`
  - Result: completed; summary included above
- Build:
  - `bazel build //daemon:tensorcast_daemon`
  - Result: success
