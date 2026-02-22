# Cross-Host Benchmark Helpers

这个目录集中放置“多机 put/get benchmark”脚本：

- `cross_host_matrix_runner.py`：统一矩阵 runner（负责重启两侧 daemon、循环 put/get、收集结果、清理 artifact）。
- `cross_host_fanout_runner.py`：多机 fanout/cascade runner（验证“get 节点成为新 P2P 源”，并支持 wave 并发扩容性能测试）。
- `cross_host_put_once.py`：单轮 put helper。
- `cross_host_get_once.py`：单轮 get helper（含可见性等待与 comm bytes delta 采样）。
- `cross_host_deregister_once.py`：单轮 deregister helper（`wait=true`，用于验证释放收敛）。

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
export TC_GET_PROCS=<G1_PROCESS_ID>,<G2_PROCESS_ID>,<G3_PROCESS_ID>,<G4_PROCESS_ID>,<G5_PROCESS_ID>,<G6_PROCESS_ID>,<G7_PROCESS_ID>
export TC_GET_IPS=<G1_IP>,<G2_IP>,<G3_IP>,<G4_IP>,<G5_IP>,<G6_IP>,<G7_IP>
export TC_GS_ADDR=<GS_IP>:50051

# 可选：
# export TC_DAEMON_CONFIG=examples/config/store_daemon_config_cross_host_bench.yaml
# 大负载（>=8GiB）建议：
# export TC_DAEMON_CONFIG=examples/config/store_daemon_config_cross_host_bench_large_payload.yaml
# export TC_OUT_DIR=/tmp/tc_cross_20260222/results_multi_host
# export TC_RUN_ID=manual-20260222
# export TC_PORT_BASE=62800

bash examples/cross_host/run_multihost_benchmark_suite.sh
```

默认套件会顺序执行：
1. `cascade` 4 节点功能链路验证（deregister 退役模式）
2. `fanout` 6 节点（1GiB）
3. `fanout` 8 节点（1GiB，`20/16/16/g0`）
4. `fanout` 8 节点（1GiB，`24/12/24/g0`）
5. `fanout` 8 节点（1GiB，`20/16/16/g8`）
6. `fanout` 8 节点（2GiB，`20/16/16/g0`）
7. 当 `TC_GET_PROCS/TC_GET_IPS` 提供到第 8 个 getter（总 9 节点）时，额外执行 9n case（1GiB 与 2GiB）。

## 3. 输出说明

runner 会输出：

- `SUMMARY {...}`：当前 case 的聚合结果。
- `OUTPUT <path>.json`：完整结果文件，包含：
- `summary`：`put_sec_mean` / `e2e_gibps_mean` / `transfer_gibps_mean` / `visibility_wait_sec_p90` 等。
- `records`：逐轮明细（put/get/cleanup）。
- 对 `cross_host_fanout_runner.py`：
- `mode=cascade`：`summary` 包含 `source_path_expected/observed`、`vram_exportable_failures`。
- `mode=fanout`：`summary` 包含 `wave1/2 transfer_gibps` 与 `cluster_gibps`。
- `params`：本次运行关键参数快照。

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
