# Cross-Host Benchmark Helpers

这个目录集中放置“多机 put/get benchmark”脚本：

- `cross_host_matrix_runner.py`：统一矩阵 runner（负责重启两侧 daemon、循环 put/get、收集结果、清理 artifact）。
- `cross_host_fanout_runner.py`：多机 fanout/cascade runner（验证“get 节点成为新 P2P 源”，并支持 wave 并发扩容性能测试）。
- `cross_host_iperf3_probe.py`：多机 `iperf3` 双向链路探针（产出单链路硬件上限参考，供 early gate 自动对照）。
- `cross_host_chaos_runner.py`：chaos 编排 runner（按 event spec 定时注入故障，输出可回放 timeline，并支持 expected-failure pass 语义）。
- `cross_host_weight_publisher_runner.py`：多机权重发布 runner（publisher 连续发布，receiver 通过 `binding.swap` 持续更新，含 `keep_last` 去注册验证）。
- `cross_host_put_once.py`：单轮 put helper。
- `cross_host_get_once.py`：单轮 get helper（含可见性等待与 comm bytes delta 采样）。
- `cross_host_deregister_once.py`：单轮 deregister helper（`wait=true`，用于验证释放收敛）。
- `run_multihost_chaos_suite.sh`：统一 chaos 执行入口（含 brainctl preflight/predict/launch/cleanup 元数据输出）。
- `run_multihost_weight_publisher_suite.sh`：WeightPublisher 多机套件入口（single-host 功能验证 -> 2 节点 -> 3 节点）。

## 1. 前置条件

1. 本机（CPU 节点）启动 Global Store（blocking）。
2. 两台 GPU worker 已通过 `brainctl` 启动，能拿到：
- `put_proc` / `get_proc`（process id）
- 两侧 daemon advertise IP（推荐使用 worker `Pod IP`，通常是 `100.x`，不要用节点 `Host IP` 的 `10.x`）
- 两侧 daemon gRPC 地址（`host:port`）
3. 两侧 worker 都有同一份代码目录（默认 `/data/workspace/tensorcast-280`）和 `.venv`。

## 2. 标准运行方式

示例（forward link，A put -> B get）：

```bash
source .venv/bin/activate

python examples/cross_host/cross_host_matrix_runner.py \
  --case-name compact_fwd_put_to_get_c20b16w16_g0 \
  --conn 20 --buffers 16 --maxw 16 --expected-gpu-channels 0 \
  --size-mib 1024 --warmup 1 --iterations 4 \
  --put-proc <PUT_PROCESS_ID> \
  --get-proc <GET_PROCESS_ID> \
  --put-adv-ip <PUT_WORKER_IP> \
  --get-adv-ip <GET_WORKER_IP> \
  --put-daemon-addr <PUT_WORKER_IP>:62001 \
  --get-daemon-addr <GET_WORKER_IP>:62011 \
  --gs-addr <GS_HOST_IP>:50051 \
  --daemon-config examples/config/store_daemon_config_cross_host_bench.yaml \
  --out-dir /tmp/tc_cross_20260221/results
```

反向链路（reverse link）只需要交换 put/get 两侧参数（`--put-*` 与 `--get-*`）。

## 2.1 多机 fanout/cascade（推荐用于扩容与 VRAM 源验证）

`cascade` 用于强验证：每个 hop 成功后先退役上游源（默认 `--source-retire-mode=deregister`），再验证下一跳仍能 get，证明“get 节点成为新 P2P 源”。

```bash
source .venv/bin/activate

python examples/cross_host/cross_host_fanout_runner.py \
  --mode cascade \
  --case-name fanout_cascade_vram_chain_4n \
  --seed-proc <SEED_PROCESS_ID> \
  --seed-adv-ip <SEED_IP> \
  --get-procs <GET1_PROCESS_ID>,<GET2_PROCESS_ID>,<GET3_PROCESS_ID> \
  --get-adv-ips <GET1_IP>,<GET2_IP>,<GET3_IP> \
  --gs-addr <GS_IP>:50051 \
  --conn 20 --buffers 16 --maxw 16 --expected-gpu-channels 0 \
  --size-mib 1024 \
  --source-retire-mode deregister \
  --deregister-device-id 0 \
  --source-stop-settle-sec 2 \
  --require-p2p \
  --require-vram-source \
  --daemon-config examples/config/store_daemon_config_cross_host_bench.yaml \
  --daemon-start-timeout-sec 600 \
  --remote-timeout-sec 900 \
  --stop-timeout-sec 240 \
  --out-dir /tmp/tc_cross_20260221/results_multi_host
```

如果你要做“严格 VRAM 源”验证（上一跳必须不可用），可以使用：

```bash
source .venv/bin/activate

python examples/cross_host/cross_host_fanout_runner.py \
  --mode cascade \
  --case-name fanout_cascade_3n_vram_source_after_seed_stop \
  --seed-proc <SEED_PROCESS_ID> \
  --seed-adv-ip <SEED_POD_IP> \
  --get-procs <GET1_PROCESS_ID>,<GET2_PROCESS_ID> \
  --get-adv-ips <GET1_POD_IP>,<GET2_POD_IP> \
  --gs-addr <GS_IP>:50051 \
  --conn 20 --buffers 16 --maxw 16 --expected-gpu-channels 0 \
  --size-mib 1024 \
  --source-retire-mode deregister_then_stop \
  --source-stop-settle-sec 2 \
  --require-p2p \
  --daemon-config examples/config/store_daemon_config_cross_host_bench.yaml \
  --out-dir /tmp/tc_cross_20260222/results_multi_host
```

说明：
- `--source-retire-mode=deregister` 是推荐模式：要求单次 `deregister(wait=true)` 成功，并由 daemon 驱逐 replica。
- `--deregister-device-id` 默认会从 `--get-device` 推断（如 `cuda:0 -> 0`），用于在无 active lease 场景下仍能触发 GS 侧按 worker+device 清理。
- `--source-retire-mode=stop` 更容易暴露“stale source / 心跳收敛窗口”问题，一般用于故障复现。
- 若传了 `--gs-db-file` 但 DuckDB 文件被 GS 写锁占用，runner 会自动退化为功能链路验证模式，不会直接失败。

`fanout` 用于扩容性能：一轮 put 后按 wave1/wave2 并发 get，对比 wave 吞吐和 cluster 吞吐。

```bash
source .venv/bin/activate

python examples/cross_host/cross_host_fanout_runner.py \
  --mode fanout \
  --case-name fanout_perf_6n_c20b16w16 \
  --seed-proc <SEED_PROCESS_ID> \
  --seed-adv-ip <SEED_IP> \
  --get-procs <G1_PROCESS_ID>,<G2_PROCESS_ID>,<G3_PROCESS_ID>,<G4_PROCESS_ID>,<G5_PROCESS_ID> \
  --get-adv-ips <G1_IP>,<G2_IP>,<G3_IP>,<G4_IP>,<G5_IP> \
  --gs-addr <GS_IP>:50051 \
  --conn 20 --buffers 16 --maxw 16 --expected-gpu-channels 0 \
  --size-mib 1024 --warmup 1 --iterations 4 --wave-size 2 \
  --require-p2p \
  --daemon-config examples/config/store_daemon_config_cross_host_bench.yaml \
  --daemon-start-timeout-sec 600 \
  --remote-timeout-sec 900 \
  --stop-timeout-sec 240 \
  --out-dir /tmp/tc_cross_20260221/results_multi_host
```

## 2.2 标准化多机套件（一键跑）

提供统一脚本：`examples/cross_host/run_multihost_benchmark_suite.sh`。

```bash
source .venv/bin/activate

export TC_SEED_PROC=<SEED_PROCESS_ID>
export TC_SEED_IP=<SEED_IP>
export TC_GET_PROCS=<G1_PROCESS_ID>,<G2_PROCESS_ID>,<G3_PROCESS_ID>,<...>
export TC_GET_IPS=<G1_IP>,<G2_IP>,<G3_IP>,<...>
export TC_GS_ADDR=<GS_IP>:50051

# 可选：
# export TC_DAEMON_CONFIG=examples/config/store_daemon_config_cross_host_bench.yaml
# 大负载（>=8GiB）建议：
# export TC_DAEMON_CONFIG=examples/config/store_daemon_config_cross_host_bench_large_payload.yaml
# export TC_OUT_DIR=/data/tc_cross_rerun/results_multi_host_scaleout
# export TC_RUNTIME_ROOT=/data/tc_cross_rerun/runtime
# export TC_FAILURE_DIAG=1
# export TC_FAILURE_DIAG_TIMEOUT_SEC=45
# export TC_VISIBILITY_TIMEOUT_SEC=120
# export TC_EARLY_GATE_ENABLE=1
# export TC_EARLY_GATE_HARD_FAIL=1
# export TC_EARLY_GATE_ROUND_WORKERS=3,4
# export TC_EARLY_GATE_TARGET_SIZE_MIB=1024
# export TC_EARLY_GATE_MIN_CLUSTER_SCALE_RATIO=1.10
# export TC_EARLY_GATE_MIN_WAVE2_OVER_WAVE1=0.75
# export TC_IPERF3_PROBE_ENABLE=1
# export TC_IPERF3_AUTOFILL_EARLY_BASELINE=1
# export TC_IPERF3_PROBE_HARD_FAIL=0
# export TC_EARLY_GATE_BASELINE_LINK_GIBPS=<micro_baseline_gibps>
# export TC_EARLY_GATE_MIN_BASELINE_RATIO=0.60
# export TC_QUOTA_PREFLIGHT_ENABLE=1
# export TC_QUOTA_CHARGED_GROUP=tensorcast_dev
# export TC_XLARGE_STABLE_PREFLIGHT_ENABLE=1
# export TC_XLARGE_STABLE_OVERLAP_VERSIONS=2
# export TC_XLARGE_STABLE_PREFLIGHT_MARGIN_RATIO=1.05
# export TC_WAVE_ASSIGNMENT=rotate
# export TC_WAVE_ASSIGNMENT_SEED=20260227
# export TC_CLEANUP_LEAK_SENTINEL=1
# export TC_CLEANUP_LEAK_THRESHOLD_BYTES=0
# export TC_CLEANUP_LEAK_STREAK_THRESHOLD=2
# export TC_TEARDOWN_STRICT=1
# export TC_TEARDOWN_VERIFY_TIMEOUT_SEC=30
# export TC_RUN_ID=manual-20260222
# export TC_PORT_BASE=62800
# export TC_SCALE_WORKERS=2,4,8,16,32
# export TC_SCALE_SIZES_MIB=1024,8192

# 全阶段（small -> medium -> large -> xlarge）
bash examples/cross_host/run_multihost_benchmark_suite.sh --phase all

# 仅小规模功能阶段（3/4 节点）
bash examples/cross_host/run_multihost_benchmark_suite.sh --phase small

# 仅中规模阶段（6 节点）
bash examples/cross_host/run_multihost_benchmark_suite.sh --phase medium

# 仅大规模阶段（8/9 节点）
bash examples/cross_host/run_multihost_benchmark_suite.sh --phase large

# 仅超大规模阶段（16/32 节点，要求 getter >= 15）
bash examples/cross_host/run_multihost_benchmark_suite.sh --phase xlarge
```

阶段说明：
1. `phase=small`
- 需要至少 2 个 getter。
- 执行 3/4 节点功能验证（含 cascade 与 fanout）。
- 若 getter >= 3，阶段结束后自动执行 early gate（默认 round workers=3,4）。
2. `phase=medium`
- 需要至少 5 个 getter。
- 执行 6 节点 fanout 收敛与吞吐验证。
3. `phase=large`
- 需要至少 7 个 getter。
- 执行 8 节点负载 case；当 getter >= 8 时追加 9 节点 case。
4. `phase=xlarge`
- 根据 `TC_SCALE_WORKERS`（默认 `2,4,8,16,32`）筛选 `>=16` 的目标规模并执行。
- 默认 payload 使用 `TC_SCALE_SIZES_MIB`（默认 `1024,8192`）。
- 配额不足（如 32-worker 无法补齐）会直接 skip，并写入 skip manifest。
- 开启 `TC_XLARGE_STABLE_PREFLIGHT_ENABLE=1` 时，会按 `size_mib * overlap_versions * margin_ratio` 预算 `stable_bytes`，预算不足的 case 会提前 skip，避免运行时 OOM / insufficient bytes。
5. `phase=all`
- 自动按 `small -> medium -> large -> xlarge` 顺序执行，节点不足的阶段会自动 skip 并打印原因。
- `small` 跑完会自动执行 early gate；当 `TC_EARLY_GATE_HARD_FAIL=1` 且 gate 失败时，流程立即中止。
- 默认会先执行 `iperf3` 探针（固定 profile：前 2 个 getter、`20s`、`P=20`），并自动回填 early gate baseline。
- 探针输出：`suite_<run_id>_iperf3_probe.json/md`；early gate 输出：`suite_<run_id>_early_gate.json/md`。

手工复核 early gate：

```bash
source .venv/bin/activate

python examples/cross_host/scaleout_early_gate.py \
  --fanout-dir /data/tc_cross_rerun/results_multi_host_scaleout \
  --run-id <run_id> \
  --round-workers 3,4 \
  --target-size-mib 1024 \
  --min-cluster-scale-ratio 1.10 \
  --baseline-link-gibps <micro_baseline_gibps> \
  --min-baseline-ratio 0.60 \
  --out-json /data/tc_cross_rerun/results_multi_host_scaleout/suite_<run_id>_early_gate.json \
  --out-md /data/tc_cross_rerun/results_multi_host_scaleout/suite_<run_id>_early_gate.md
```

## 2.3 Chaos 套件（事件时间线 + expected-failure 门禁）

`cross_host_chaos_runner.py` 使用 case schema 运行多个 case，每个 case 支持：

- `chaos_events`：
  - `offset_sec`
  - `target_role`（`seed/getter/all_getters/custom`）
  - `action`（`daemon_stop/daemon_kill/command/sleep`）
  - `duration_sec`
  - `expected_impact`
- 负向 case 门禁：
  - `expected_outcome=success|failure`
  - `expected_error_pattern`（failure case 可选）
  - 输出状态区分 `expected_failure_pass` / `unexpected_failure` / `unexpected_success`
- 失败路径强制清理：
  - 当 case 失败（含 timeout/执行异常）时，runner 会对 seed/getter 执行 best-effort daemon cleanup
  - cleanup 结果写入 run/case 的 `events.jsonl` 与 `result.json.cleanup_records`

示例 schema：
- `examples/cross_host/case_schemas/chaos_suite_example.json`

统一入口脚本：

```bash
source .venv/bin/activate

export TC_CASE_SCHEMA=examples/cross_host/case_schemas/chaos_suite_example.json
export TC_OUT_DIR=/tmp/tc_cross_20260222/results_chaos
export TC_RUN_ID=chaos-20260222

# 可选：让脚本托管 brainctl 生命周期（技能默认 charged-group=tensorcast_dev）
# export TC_BRAINCTL_ENABLE_LAUNCH=true
# export TC_BRAINCTL_GPU=1
# export TC_BRAINCTL_CPU=4
# export TC_BRAINCTL_MEMORY=106400
# export TC_BRAINCTL_POSITIVE_TAGS=L40S,H200,H800,H100
# export TC_BRAINCTL_MOUNT='juicefs+s3://xxx:/mnt/xxx'
# 可选：门禁评审阈值（默认开启 all_get_complete/source_cardinality/expected_failure/comm_errors gate）
# export TC_GATE_MAX_RECOVER_TIME_SEC=180
# export TC_GATE_REQUIRE_SOURCE_CARDINALITY=true

bash examples/cross_host/run_multihost_chaos_suite.sh
```

输出目录（run 级）：

- `<out_dir>/<run_id>/summary.json`
- `<out_dir>/<run_id>/events.jsonl`
- `<out_dir>/<run_id>/cases/<case_name>/result.json`
- `<out_dir>/<run_id>/cases/<case_name>/metrics.json`
- `<out_dir>/<run_id>/cases/<case_name>/classification.json`
- `<out_dir>/<run_id>/meta/brainctl_steps.jsonl`
- `<out_dir>/<run_id>/gate_review.json`
- `<out_dir>/<run_id>/gate_review.md`

也可以手工复核（不依赖入口脚本）：

```bash
source .venv/bin/activate

python examples/cross_host/chaos_gate_review.py \
  --run-dir <out_dir>/<run_id> \
  --max-recover-time-sec 180
```

Phase-level aggregation review for `small/medium/large`:

```bash
source .venv/bin/activate

python examples/cross_host/chaos_phase_gate_review.py \
  --small-run-dir <out_dir>/<run_id_small> \
  --medium-run-dir <out_dir>/<run_id_medium> \
  --large-run-dir <out_dir>/<run_id_large>
```

Recommended profiles:

- fast-failover profile: `examples/config/global_store_config_cross_host_bench_fast_failover.yaml`
- slow-cleanup profile: `examples/config/global_store_config_cross_host_bench_slow_cleanup.yaml`

## 2.4 Fixed 0081 One-Command Entry (Recommended)

This is the fixed execution entrypoint for `docs/designs/0081-multihost-put-get-chaos-stability.md`:

```bash
source .venv/bin/activate

# Default behavior:
# - launch 8 workers with --private-machine group
# - run phases in order: small -> medium -> large
# - generate phase gate aggregation artifacts
# - cleanup temporary workers after completion
bash examples/cross_host/run_0081_multihost_chaos.sh --phase all
```

Common parameters:

```bash
bash examples/cross_host/run_0081_multihost_chaos.sh \
  --phase all \
  --gs-addr 100.97.246.95:50051 \
  --run-label 0081-manual-$(date +%Y%m%d-%H%M%S) \
  --out-root /tmp/tc_cross_20260222/results_chaos_0081_fixed \
  --charged-group tensorcast_dev \
  --private-machine group \
  --keep-workers
```

Notes:

- If `--gs-addr` is not provided, the script resolves it from `tensorcast-cli global status --json`.
- The script automatically handles: `brainctl predict`, worker launch/readiness wait, GPU + repo health checks, schema generation, phase execution, and phase-gate aggregation.
- Output layout:
  - `<out-root>/<run-label>/schemas/` (small/medium/large schemas)
  - `<out-root>/<run-label>/results/` (per-phase runs + gate artifacts + phase gate review)
  - `<out-root>/<run-label>/meta/launcher_meta.json` (full launcher metadata)

## 2.5 WeightPublisher Multi-host (staged group_realization)

入口脚本：`examples/cross_host/run_multihost_weight_publisher_suite.sh`

```bash
source .venv/bin/activate

export TC_WP_PUBLISHER_PROC=<PUBLISHER_PROCESS_ID>
export TC_WP_RECEIVER_PROCS=<RECEIVER1_PROCESS_ID>,<RECEIVER2_PROCESS_ID>
export TC_GS_ADDR=<GS_IP>:50051
export TC_OUT_DIR=/data/tc_cross_rerun/results_weight_publisher
export TC_RUN_AS_USER=$(id -un)
export TC_PUBLISH_INTERVAL_S=60
export TC_RECEIVER_TIMEOUT_S=95
export TC_MAX_PUBLISH_TO_APPLY_S=30
export TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST=1
export TC_WP_SCALE_RECEIVER_COUNTS=1,2,4,8,16,31
export TC_SCALE_NUM_VERSIONS=10
export TC_WP_PAYLOAD_MODE=tp_ranked
export TC_WP_TP_WORLD_SIZE=4
export TC_WP_TP_TOTAL_BYTES=42949672960
export TC_WP_MAX_CONCURRENCY=1
export TC_WP_PRE_PUBLISH_TRIM_MARGIN=1
export TC_WP_RECEIVER_PREFLIGHT_TRANSIENT_OVERLAP=1
export TC_WP_P0_EARLY_STOP=1
export TC_WP_P0_EARLY_STOP_GRACE_S=20
export TC_LONG_RUN_ENABLE=1
export TC_LONG_RUN_NUM_VERSIONS=20
export TC_LONG_RUN_TARGET_DURATION_S=900
export TC_PROGRESS_POLL_S=10

# 可选：本地 SDK 连接地址（默认 127.0.0.1:50052）
# export TC_DAEMON_CONNECT_ADDRESS=127.0.0.1:50052

bash examples/cross_host/run_multihost_weight_publisher_suite.sh
```

流程与判定：
1. 在 publisher 节点先执行 single-host staged group realization 功能烟测。
2. 按 `TC_WP_SCALE_RECEIVER_COUNTS` 逐级执行 receiver 规模 case。
3. 每个规模固定执行 unified `group_realization` staged publish/acquire 路径。
4. 验证点包括：
- receiver 通过 staged publish/acquire 连续更新，接收操作为 `stage_acquire`；
- pointer 稳定；
- `keep_last=2` 时旧版本去注册后不可物化；
- 通过 GS RPC `BatchGetReplicaCounts` 审计，旧版本副本计数回到 0，且有副本的版本数不超过 2。
- runner 显式管理每台机器的 daemon 生命周期，role/probe 均通过 `tc.init(mode="connect", address=127.0.0.1:50052)` 仅连接本地 daemon。
- 建议将 `TC_RECEIVER_TIMEOUT_S` 设为大于发布间隔（例如 `publish=60s` 时用 `timeout=95s`），并通过
  `TC_MAX_PUBLISH_TO_APPLY_S` 约束“发布到应用”的时延上限；在 TP 扩容场景建议保持
  `TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST=1`，由 runner 按 receiver 数、TP 字节规模与 group realization staging 开销自动抬升阈值下限。
- 当 `allow_receiver_skips` 生效时，runner 会基于 receiver 日志中的显式
  `[receiver] skipped version=...` 记录做缺失版本记账：仅将“未被显式 skip 覆盖”的缺失版本计为 sequence failure，
  并在输出中给出 `timeout_analysis.waiting_timeout_reason_counts` 与
  `timeout_analysis.transport_timeout_reason_counts` 以区分等待超时和传输超时。
- 深度压测可通过 `TC_WP_SCALE_RECEIVER_COUNTS` 扩大到 32 worker 规模（1 publisher + 31 receivers），并用
  `TC_SCALE_NUM_VERSIONS`/`TC_LONG_RUN_NUM_VERSIONS` 与 `TC_LONG_RUN_TARGET_DURATION_S`
  组合出 10/20 版本、最长约 15 分钟的长跑 case。
- TP 场景可通过 `TC_WP_TP_WORLD_SIZE`、`TC_WP_TP_TOTAL_BYTES`
  固定为 TP4/TP8 接收规模；接收路径固定为 staged group realization publish/acquire。
- 运行中会周期打印 `[progress]` 聚合状态，便于外部 `tail -f`/log poll 观察实时进展。
- suite 输出目录包含 `suite_skips.jsonl`，用于记录无效 receiver count、long-run 跳过等结构化原因。

## 3. 输出说明

runner 会输出：

- `SUMMARY {...}`：当前 case 的聚合结果。
- `OUTPUT <path>.json`：完整结果文件，包含：
- `summary`：`put_sec_mean` / `e2e_gibps_mean` / `transfer_gibps_mean` / `visibility_wait_sec_p90` 等。
- `records`：逐轮明细（put/get/cleanup）。
- `events`：事件时间线（iteration start/end、get_failed 及分类）。
- 对 `cross_host_fanout_runner.py`：
- `mode=cascade`：`summary` 包含 `source_path_expected/observed`、`vram_exportable_failures`。
- `mode=fanout`：`summary` 额外包含
  - `all_get_complete` / `incomplete_workers`
  - `source_cardinality_timeline`
  - `recover_time_sec`
  - `put_success_rate` / `get_success_rate`
  - `retry_reason_buckets`
  - `failure_classification_counts`（`infra/product/unknown`）
  - payload sample hash 校验统计
- `params`：本次运行关键参数快照。
- `teardown`：每个 worker 的 stop + status verify 结果；当 `--teardown-strict` 生效且未完全退出时 case 失败。
- `summary.pre_case_seed_status` / `summary.cleanup_leak_*`：seed 端 cleanup 泄漏哨兵指标与阈值配置。

suite 额外输出：

- `suite_<run_id>_cases.jsonl`：计划执行的 case 清单。
- `suite_<run_id>_skips.jsonl`：结构化 skip 记录（含 `reason/detail`，用于复盘资源不足或策略跳过）。

chaos 报告模板：

- `docs/internals/chaos-report-template.md`

## 4. 常见问题

1. `no active lease found; proceeding with stateless retire`
- 表示该次 deregister 未命中 active lease，runner 仍会继续本地 retire。

2. `GlobalStoreClient requires a non-zero P2P port ...`
- 常见于 daemon 刚重启立即开始压测时，worker 尚未完成 GS 注册。
- 现在 runner 在每次重启后会等待 daemon+GS 连接就绪再继续；若仍出现，可直接重跑该 case。

3. daemon startup preflight 失败（内存不足）
- 1~2GiB 优先使用 `examples/config/store_daemon_config_cross_host_bench.yaml`。
- 8GiB/32GiB 建议使用 `examples/config/store_daemon_config_cross_host_bench_large_payload.yaml`。

4. `comm_gpu` 切片不足启动失败
- 调大 `pinned_memory.classes[name=comm_gpu].pool_bytes`，或降低 `conn/buffers/maxw` 组合。
- 例子：`conn=24, buffers=16` 在 `comm_gpu.pool=6GiB, slice=16MiB` 下会要求 `required_slices=400`，超出 `capacity_slices=384`，启动会失败。

5. `No daemon found at <ip:port>`
- 常见原因：误把节点 `Host IP(10.x)` 作为 `--seed-adv-ip/--get-adv-ips`。
- 处理：统一改用 worker `Pod IP(100.x)`。

6. `cannot exec into ... Stopped`
- 表示 worker 被平台自动回收。
- 处理：重新 `brainctl launch`，并更新 process id / pod ip 后重跑 case。

7. `cudaErrorNoDevice` / `device_count=0` 导致 daemon 首启崩溃
- 这是 worker 运行时 GPU 不可用（常见于异常节点），不是 case 参数问题。
- `cross_host_fanout_runner.py` 现在会在 daemon 重启前先执行 GPU preflight（`nvidia-smi -L`），无 GPU 会直接 fail-fast 并给出明确诊断。
- 处理：对目标 worker 执行 `nvidia-smi -L` 与 `torch.cuda.device_count()` 健康检查；下线坏 worker 并替换后重跑。

8. `startup memory preflight failed`（`available < total_required`）
- 常见于同一批 worker 上并发叠加执行多个 suite，导致瞬时可用内存不足。
- 处理：改为串行执行（`small -> medium -> large`），避免同组 worker 并发压测；必要时重启 worker 清理环境。
