---
slug: cpu-memfd-zero-copy-publish
title: CPU Memfd Zero-Copy Stable DRAM Publish Plan
status: in_progress
areas: ["core", "daemon", "sdk", "proto", "tests", "docs"]
created: 2026-02-24
last_updated: 2026-02-25
related_code:
  - docs/designs/0082-cpu-memfd-zero-copy-publish.md
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/registration_controller.cc
  - daemon/state/local_handle_server.h
  - daemon/state/local_handle_server.cc
  - daemon/state/handle_lease_registry.h
  - daemon/state/handle_lease_registry.cc
  - core/store/store_engine.h
  - core/store/runtime/metadata/metadata_types.h
  - core/store/runtime/metadata/registration_backend.h
  - core/store/runtime/metadata/registration_backend.cc
  - tensorcast/types.py
  - tensorcast/daemon_ctl.py
  - tensorcast/api/_register.py
  - tensorcast/csrc/checkpoint_py.cc
  - tests/python/api/test_register_stable_dram_streaming.py
  - core/store/runtime/metadata/metadata_gateway_test.cc
links:
  design: ../designs/0082-cpu-memfd-zero-copy-publish.md
---

# Objective

Implement and ship `stable_dram + stage_on_gpu=false` publish over CPU memfd shared memory so publish cost approaches the memory-copy bound, with no daemon-side payload memcpy and no additional model-sized memory footprint.

# Current State and Grounding

Current baseline behavior:

- stable DRAM stream path sends payload bytes over gRPC (`FeedRegisterArtifactStream`).
- daemon ingests chunks and memcpys into stable DRAM.
- commit validates stream coverage counters.
- local handle and CPU memfd lease primitives already exist for materialization.

Grounding references:

- proto registration surface: `proto/tensorcast/daemon/v2/store_daemon.proto`
- feed handling and mode routing: `daemon/service/controllers/registration_controller.cc`
- stable DRAM ingestion/commit checks: `core/store/runtime/metadata/registration_backend.cc`
- local handle FD exchange: `daemon/state/local_handle_server.{h,cc}`
- existing SDK upload path: `tensorcast/api/_register.py`, `tensorcast/daemon_ctl.py`

Constraints:

- SDK must not connect directly to Global Store.
- config/API behavior must stay within unified runtime config and explicit protocol, no ad-hoc env-only control.
- backward compatibility required for existing clients.

# Baseline Measurements and Targets

Before code changes, record on identical hardware/profile:

- publish latency (40GB and 320GB)
- key mapping absent duration
- publish throughput GiB/s
- daemon-side memcpy counters and wall time
- peak RSS of publisher and daemon

Target outcomes:

- daemon payload memcpy removed for CPU memfd publish path
- 40GB publish latency materially reduced vs baseline stream path
- 320GB publish latency reduced with the same test topology
- no model-sized extra memory occupancy introduced

# Phases and Milestones

- [ ] Phase 0: Baseline and Measurement Harness
  - [ ] Milestone 0.1: Freeze baseline measurements for stream path (40GB then 320GB).
  - [ ] Milestone 0.2: Add or validate publish-only microbenchmark for copy-bound reference.
  - [ ] Milestone 0.3: Define standard metrics table template for before/after comparison.

- [x] Phase 1: Protocol and Type Model Extension
  - [x] Milestone 1.1: Extend stable DRAM handshake to represent CPU memfd publish handle.
  - [x] Milestone 1.2: Extend feed stream with stable DRAM write-progress range message.
  - [x] Milestone 1.3: Regenerate Python protobuf stubs (`bash tools/build_proto_python.sh`).
  - [x] Milestone 1.4: Update typed SDK models (`tensorcast/types.py`) for handshake decoding.

- [x] Phase 2: Daemon and Core Publish Path
  - [x] Milestone 2.1: Begin registration mints publish lease token and returns CPU memfd publish handshake when eligible.
  - [x] Milestone 2.2: Add range-only ingestion API in StoreEngine/RegistrationBackend.
  - [x] Milestone 2.3: Keep strict range validation (bounds, overlap, full coverage).
  - [x] Milestone 2.4: Commit path for CPU memfd publish performs validation only, no payload memcpy.
  - [x] Milestone 2.5: Add path-level metrics and logs (`cpu_memfd_publish`, fallback reasons).

- [ ] Phase 3: SDK Upload Path and Copy Engine
  - [x] Milestone 3.1: SDK handshake branch selects CPU memfd publish when offered.
  - [x] Milestone 3.2: SDK obtains fd over local handle and maps with `MAP_SHARED`.
  - [ ] Milestone 3.3: Implement efficient tensor-to-mapped-target copy path (C++ extension preferred) with no model-sized temporary payload.
  - [x] Milestone 3.4: SDK sends write-progress ranges and commits.
  - [x] Milestone 3.5: deterministic release/abort cleanup for publish lease token.

- [ ] Phase 4: Correctness and Regression Tests
  - [x] Milestone 4.1: Add C++ unit tests for range ingestion and commit coverage behavior.
  - [x] Milestone 4.2: Add daemon controller tests for handshake and feed routing.
  - [x] Milestone 4.3: Add Python API tests for CPU memfd publish path and fallback path.
  - [ ] Milestone 4.4: Add concurrency correctness tests (multiple ranks, bounded overlap rules).

- [ ] Phase 5: Performance Validation and Benchmark Replay
  - [x] Milestone 5.1: Re-run 40GB publish-only benchmarks for fast iteration.
  - [x] Milestone 5.2: Re-run same-parameter TP8-320GB publish and compare with baseline.
  - [ ] Milestone 5.3: Publish comparison table: publish latency, key_mapping_absent, 8-rank swap latencies.
  - [ ] Milestone 5.4: Capture residual gap vs copy-bound microbenchmark and list next-step optimizations.

- [ ] Phase 6: Rollout and Documentation Closure
  - [x] Milestone 6.1: Keep fallback behavior verified for older clients.
  - [x] Milestone 6.2: Update benchmark report and relevant architecture docs.
  - [ ] Milestone 6.3: Mark design/plan status based on evidence and owner signoff.

# Tasks

- [x] Proto + model tasks
  - [x] Modify `proto/tensorcast/daemon/v2/store_daemon.proto` for handshake and progress message.
  - [x] Run `bash tools/build_proto_python.sh`.
  - [x] Update `tensorcast/types.py` handshake models.

- [x] Core/daemon tasks
  - [x] Add `ingest_registration_written_range` style API into `core/store/store_engine.h` and backend.
  - [x] Implement mode split in `RegistrationBackend` commit path (`staging_gpu`, `cpu_stream`, `cpu_memfd_publish`).
  - [x] Route new feed message in `registration_controller.cc`.
  - [x] Reuse or minimally extend local handle lease behavior for publish token FD exchange.

- [x] SDK tasks
  - [x] Add handshake branching in `tensorcast/api/_register.py`.
  - [x] Add fd request and mapping helper reuse from local handle utilities.
  - [x] Implement efficient copy engine without model-sized temporary payload.
  - [x] Send progress ranges in bounded batches.

- [ ] Test tasks
  - [x] C++ tests under daemon/core targets for new protocol and backend logic.
  - [x] Python tests under `tests/python/` for path correctness and fallback.
  - [ ] Stress/concurrency checks for multi-rank publish overlap and tail behavior.

- [ ] Docs tasks
  - [x] Update `docs/benchmarks/20260222-weight-publisher-multihost-p2p-report.md` with finalized analysis and before/after tables.
  - [x] Update architecture/API docs if public handshake semantics change.

# Test, Rollout, and Backout

## Test Plan

C++ tests:

- `bazel test //core/store/runtime/metadata:metadata_gateway_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //daemon:grpc_service_impl_registration_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- add and run dedicated new tests for range-only publish path

Python tests:

- `source .venv/bin/activate && pytest tests/python/api/test_register_stable_dram_streaming.py`
- add and run dedicated tests for CPU memfd publish handshake and upload path

Benchmark verification:

- run 40GB publish-only loop for iterative tuning
- run TP8-320GB same-parameter replay after optimization
- capture before/after metrics table

## Rollout

1. Land behind handshake feature detection with automatic fallback.
2. Validate on local GS + local daemon first.
3. Validate in multihost benchmark environment.
4. Enable by default when evidence meets acceptance metrics.

## Backout

- Keep stream upload path intact.
- If regressions occur, daemon can stop advertising CPU memfd publish handshake and clients automatically fall back.
- Revert only new handshake branch while preserving compatibility fields.

# Execution Evidence (2026-02-25)

## Code + Test Status

- Proto/C++/daemon/SDK path landed and wired end-to-end.
- Local regression tests passed:
  - `pytest tests/python/api/test_register_stable_dram_streaming.py`
  - `bazel test //core/store/runtime/metadata:metadata_gateway_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  - `bazel test //daemon:grpc_service_impl_cpu_memfd_stable_budget_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  - `bazel test //daemon:grpc_service_impl_cpu_memfd_e2e_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  - `bazel test //daemon:grpc_service_impl_registration_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

## Remote Benchmark (brainctl worker, local GS + local daemon)

Workers and base path:

- workers:
  - `ws-7681b3683947089e-worker-djjw2` (earlier A/B runs)
  - `ws-7681b3683947089e-worker-m7fmp` (320GB fix replay + correctness rerun)
- results root: `/data/tc_cross_20260225/publish_only_0082`

Parameters:

- `payload_mode=tp_ranked`
- `tp_world_size=8`
- `keep_last=1`
- `publish_device=cpu`
- `TENSORCAST_FEED_VIEW_UPLOAD_WORKERS=4`
- `TENSORCAST_FEED_VIEW_CHUNK_BYTES=4MB`
- A/B switch: daemon `engine.cpu_shared_memory.enabled=false|true`

Measured results (`-r2`):

| Case | Path | `publish_latency_s` | `put_s` | `stream_s` | `put_bw_gibs` |
| --- | --- | ---: | ---: | ---: | ---: |
| `tp8-4gb-stream-c4m-w4-r2` | `cpu_stream` | `4.164` | `4.162` | `3.793` | `0.961` |
| `tp8-4gb-memfd-c4m-w4-r2` | `cpu_memfd_publish` | `2.564` | `2.562` | `0.669` | `1.561` |
| `tp8-40gb-stream-c4m-w4-r2` | `cpu_stream` | `41.975` | `41.972` | `39.170` | `0.953` |
| `tp8-40gb-memfd-c4m-w4-r2` | `cpu_memfd_publish` | `25.560` | `25.558` | `7.783` | `1.565` |
| `tp8-320gb-stream-c4m-w4-r1` | `cpu_stream` | `333.275` | `333.272` | `311.533` | `0.960` |
| `tp8-320gb-memfd-c4m-w4-r4-fix` | `cpu_memfd_publish` | `77.685` | `77.681` | `77.234` | `4.119` |

Observed improvement:

- 4GB `put_s`: `4.162s -> 2.562s` (about `38.4%` lower).
- 40GB `put_s`: `41.972s -> 25.558s` (about `39.1%` lower).
- 40GB `publish_latency_s`: `41.975s -> 25.560s` (about `39.1%` lower).
- 320GB `put_s`: `333.272s -> 77.681s` (about `76.7%` lower).

## Cross-host Runner Validation (40GB, report-style topology)

Topology and params:

- GS (local control host): `100.97.246.95:50051`
- publisher: `ws-7681b3683947089e-worker-wx4zv` (1 GPU)
- receiver: `ws-7681b3683947089e-worker-wl6dz` (8 GPU)
- `payload_mode=tp_ranked`, `tp_world_size=8`, `tp_total_bytes=40GiB`, `num_versions=2`, `keep_last=1`,
  `receiver_apply_mode=tp_bind_into_swap`, `publish_interval_s=60`, `poll_interval_s=1.5`, `max_concurrency=1`

Outputs:

- stream:
  `/tmp/tc_cross_20260225/results_weight_publisher_mainline_tp8_40gb/tp8-40gb-r1-v2-stream-hb60-p1p5-mc1-k1-r4-1771975029-1771975030/tp8-40gb-r1-v2-stream-hb60-p1p5-mc1-k1-r4-1771975029.json`
- memfd:
  `/tmp/tc_cross_20260225/results_weight_publisher_mainline_tp8_40gb/tp8-40gb-r1-v2-memfd-hb60-p1p5-mc1-k1-r4-1771974795-1771974795/tp8-40gb-r1-v2-memfd-hb60-p1p5-mc1-k1-r4-1771974795.json`

Measured deltas:

| Metric | stream | memfd | Delta |
| --- | ---: | ---: | ---: |
| publish latency mean | `46.428s` | `13.258s` | `-71.4%` |
| publish `put_s` mean | `44.681s` | `10.788s` | `-75.9%` |
| publish `put_bw` (40GiB/put_mean) | `0.895 GiB/s` | `3.708 GiB/s` | `4.14x` |
| v2 `key_mapping_absent` max wait | `105.4s` | `63.3s` | `-39.9%` |
| v2 8-rank `swap` total | `6139.9ms` | `8056.7ms` | `+31.2%` |

Correctness signals:

- both runs: receiver completed all versions (`v1 bind_into`, `v2 swap`), no skip, pointer stable.
- runner top-level `passed=false` in both runs is due immediate `cluster_probe` window (`materializable_versions_within_window=false`);
  `global_store_probe` checks pass (`old_versions_released=true`, version-window constraints pass).

Execution notes:

- TP8 receiver must provide 8 GPUs; using 1-GPU receiver reproduces `invalid device ordinal` and is not a valid TP8 run.
- stream baseline is stable with:
  `TENSORCAST_FEED_VIEW_UPLOAD_WORKERS=4` and `TENSORCAST_FEED_VIEW_CHUNK_BYTES=4MB`.

## 320GB Commit-Budget Failure and Fix Verification

Failure before fix:

- case: `tp8-320gb-memfd-c4m-w4-r3`
- error: `CommitRegisteredArtifact failed: Insufficient stable bytes: requested=343597383680 used=343597383680 total=343597383680`
- root cause: temporary publish handle lease consumed stable budget, then commit attempted a second stable admission for the same payload.

Fix:

- release publish lease before `commit_registered_artifact(...)` admission in `registration_controller`.
- return `stable_cache_admitted` from backend commit result and skip duplicate local stable-tier admission in controller.
- added regression coverage for exact-budget pinned commit success.

After fix:

- case: `tp8-320gb-memfd-c4m-w4-r4-fix-1771972466`
- result: success with `stable_dram upload path=cpu_memfd_publish ... stream_seconds=77.234`
- no `RESOURCE_EXHAUSTED` / `Insufficient stable bytes` in publish or daemon logs.

## Runtime Correctness Spot Check

Single-host memfd run:

- case: `tp8-4gb-singlehost-memfd-correctness-r1`
- summary: `/data/tc_cross_20260225/publish_only_0082/tp8-4gb-singlehost-memfd-correctness-r1/single_host_summary.json`
- log confirms publish path: `stable_dram upload path=cpu_memfd_publish ... stream_seconds=0.678`
- receiver outcome:
  - `apply_mode=tensor_dict`
  - `apply_operation=materialize`
  - `materialize_latency_s=0.082`

Post-fix rerun:

- case: `tp8-4gb-singlehost-memfd-correctness-r2-1771972682`
- summary: `/data/tc_cross_20260225/publish_only_0082/tp8-4gb-singlehost-memfd-correctness-r2-1771972682/single_host_summary.json`
- publish: `publish_latency_s=1.060`, `put_s=1.058`
- receiver: `materialize_latency_s=0.035` (`tensor_dict`, `apply_operation=materialize`)

Conclusion at this stage:

- CPU memfd direct publish path is functionally correct on the exercised cases.
- Performance is materially better than stream baseline at 4GB, 40GB, and 320GB.
- Remaining performance/coverage work is tracked in unchecked milestones (full TP8 key_mapping/swap table, residual gap analysis, stress/concurrency hardening).

# Risks and Tracking

| Risk | Impact | Detection | Mitigation |
| --- | --- | --- | --- |
| Lease token expiry during long publish | Commit failure, `not found` | publish logs + feed keepalive metrics | bounded keepalive cadence and clearer TTL diagnostics |
| Range accounting bugs | false pass or false fail | unit tests + overlap/out-of-range fuzz | strict invariant tests and commit-time full-span verification |
| Python-side overhead remains high | throughput below target | microbenchmark vs end-to-end gap analysis | move copy hot loop to C++ extension and release GIL |
| Unexpected RSS growth | memory pressure and OOM risk | process RSS telemetry | block merges without memory profile evidence |
| Compatibility break for old clients | rollout risk | mixed-version integration tests | handshake feature detection with stream fallback |

# Owner Checklist

- [x] Baseline and after metrics captured with the same topology and parameters.
- [x] No daemon payload memcpy on CPU memfd publish path.
- [x] No model-sized extra memory introduced.
- [ ] Correctness holds under concurrency and fault scenarios.
- [x] Benchmark report updated with quantitative comparison table.
- [ ] Design and plan status updated with final decision record.
