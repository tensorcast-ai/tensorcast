# TensorCast SDK P2P (MTCP) Put->Get（正向链路）/ Get->Put（反向链路）性能调优报告

日期：2026-02-21（含 2026-02-22 跨机双向与稳定性更新）  
代码版本：`29cb29a5`  
说明：
- 本文中的 `Get->Put（反向链路）` 表示 **角色互换后的 Put->Get**（Reverse Link），不是协议方向反转。
- 大规模多机 fanout/cascade 扩容实验与瓶颈修复详情见：`docs/benchmarks/20260221-multi-host-p2p-fanout-benchmark-report.md`。
- 2026-02-22 补充：多机扩展已推进到 9 节点（1 seed + 8 getters），1GiB 下 `cluster_gibps` 达 `1.012`，2GiB 下达 `1.783`，8GiB 下达 `3.907`，32GiB 下达 `6.976`；并完成 `deregister_then_stop` 严格链路验证（seed 下线后下一跳仍 `source=p2p`）。
- 2026-02-22 补充：已修复 32GiB 场景的 stable 预算重复计数问题（CPU export 重复申请 stable lease），对应复测 case `fanout_perf_9n_c20b16w16_g0_s32768_v4` 通过。

测试目标：
- 在不绕开 key/index 路径（`lookup_mode=key`）的前提下，评估 MTCP 在当前框架内可达到的吞吐上限。
- 对照本机 TCP 上限（`iperf3`）判断“是否打满”。
- 通过参数调优与必要框架修改，收敛到可复现的高性能配置。

---

## 1. 测试环境

### 1.1 硬件环境
- CPU：Intel(R) Xeon(R) Platinum 8468，2 sockets，96 cores / 192 threads（`lscpu`）
- 内存：885 GiB（`free -h`）
- GPU：8 × NVIDIA H800，单卡 81559 MiB（`nvidia-smi --query-gpu`）
- GPU Driver / CUDA：`535.161.08` / `12.2`（`nvidia-smi`）
- NUMA：2 NUMA nodes（`lscpu`）
- 存储：
  - `nvme0n1 INTEL SSDPF2KX038T1 3.5T` 挂载 `/mnt/host0`
  - 根盘 `sda 446.6G`
- 网络：
  - 多张网卡（`net1..net8`, `eth0`）
  - 主机地址样例：`10.196.6.154`

### 1.2 软件环境
- OS：Ubuntu 22.04.5 LTS，kernel `5.4.0-153-generic`
- Bazel：`8.4.1`
- uv：`0.9.6`
- TensorCast 代码：`git rev-parse --short HEAD = 29cb29a5`

### 1.3 Runtime 形态
- Global Store 1 个（本机）
- Put Daemon 1 个 + Get Daemon 1 个（本机双 daemon）
- SDK 基准脚本：
  - `examples/tensorcast_sdk_p2p_put_get_benchmark.py`
  - `examples/tensorcast_sdk_p2p_benchmark_driver.sh`
- 关键约束：
  - 全程 `lookup_mode=key`
  - 全程 `prefer=p2p`, `require_p2p=true`

---

## 2. 方法与指标

### 2.1 测试流程
1. 清理机器上残留 daemon/global/iperf 进程。
2. 启动 GS。
3. 启动 put/get daemon（通过 driver 注入 config 和 `--set` 覆盖）。
4. 每轮执行 `put -> get`，收集 JSON。
5. 结束后停止 GS，并进入下一轮参数。

### 2.2 关键指标
- `e2e_gibps_mean`：SDK get 端整体吞吐（包含可见性等待等）
- `transfer_gibps_mean`：materialization diagnostics 中 transfer 吞吐
- `visibility_wait_sec_p90`：可见性等待尾延迟
- `p2p_ratio`：P2P 命中率（目标 100%）
- `comm_errors_delta_sum`：通信错误增量（目标 0）

说明：
- `transfer_*` 是框架内部 transfer 时间窗口，不完全等同于 `iperf3` 的纯 socket 口径；出现高于 `iperf` 的值时，不表示物理链路超过 TCP 极限，而是计时边界不同。

---

## 3. 测试数据

### 3.1 TCP Ceiling（iperf3）

#### 3.1.1 首轮 ceiling（loopback + host IP）
数据文件：`/tmp/tc-bench-20260220/iter-next5/iperf/summary.tsv`

| target | host | P | receiver_GiBps |
|---|---:|---:|---:|
| loopback | 127.0.0.1 | 24 | 5.513647 |
| loopback | 127.0.0.1 | 32 | 5.415421 |
| hostip | 10.196.6.154 | 16 | 5.574272 |
| hostip | 10.196.6.154 | 32 | 5.547468 |

#### 3.1.2 复核 ceiling（loopback，`-w 8M/16M`）
数据文件：`/tmp/tc5/iperf_recheck/summary.tsv`

| P | window | receiver_GiBps |
|---:|---:|---:|
| 16 | 16M | 5.413034 |
| 20 | 16M | 5.394912 |
| 24 | 16M | 5.238580 |
| 28 | 16M | 5.098222 |
| 32 | 16M | 4.759825 |

结论：本机 TCP ceiling 合理区间约 `5.4~5.6 GiB/s`。

---

### 3.2 粗扫矩阵（早期低吞吐阶段，定位瓶颈）
数据文件：`/tmp/tc5/mc/summary.tsv`

关键样本（`expected_gpu_channels=1`）：

| case | conn/buf | chunk_mb | comm_slice_mb | engine_slice_mb | transfer_mean GiB/s |
|---|---:|---:|---:|---:|---:|
| c20_b20_m20_g1_ch1024_s8_e64 | 20/20 | 1024 | 8 | 64 | 2.0698 |
| c24_b24_m24_g1_ch1024_s8_e64 | 24/24 | 1024 | 8 | 64 | 2.0464 |
| c28_b24_m24_g1_ch1024_s8_e64 | 28/24 | 1024 | 8 | 64 | 2.0809 |
| c24_b24_m24_g1_ch1024_s8_e32 | 24/24 | 1024 | 8 | 32 | 1.3117 |

观察：
- 在 `engine_slice=64MB` 下，吞吐长期卡在约 `1.7~2.1 GiB/s`，与 conn 增减弱相关。
- `engine_slice=32MB` 更差。

---

### 3.3 `expected_gpu_channels` 与 `slice/chunk` 对照
数据文件：`/tmp/tc5/mc_e8/summary.tsv` 与 `/tmp/tc5/postfix_log_throttle/summary.tsv`

| case | expected_gpu_channels | comm_slice_mb | transfer_mean GiB/s |
|---|---:|---:|---:|
| c24_b24_m24_g8_ch1024_s8_e64 | 8 | 8 | 1.7590 |
| c24_b24_m24_g0_ch1024_s8_e64 | 0 | 8 | 2.3902 |
| c24_b24_m24_g8_ch1024_s4_e64 | 8 | 4 | 2.4270 |

观察：
- `expected_gpu_channels` 与 `comm_gpu.slice` 会影响表现，但提升幅度不足以解释与 ceiling 的差距。
- 主瓶颈仍不在 MTCP 连接数本身。

---

### 3.4 `tx_slice`（engine slice）放大实验（关键）
数据文件：
- `/tmp/tc5/txslice_scale/summary.tsv`
- `/tmp/tc5/txslice_scale_hi/summary.tsv`

| engine_slice_mb | engine_pool_gb | transfer_mean GiB/s |
|---:|---:|---:|
| 64 | 8 | 1.6951 |
| 128 | 16 | 2.8075 |
| 256 | 32 | 4.1588 |
| 512 | 64 | 4.3659 |
| 1024 | 128 | 5.1483 |

结论：
- 吞吐随 `engine_slice` 放大显著上升，证明之前主瓶颈是 request 粒度/调度固定开销，而不是 MTCP 物理链路。

---

### 3.5 最终调优矩阵（高吞吐阶段）
数据文件：`/tmp/tc5/final_tune/summary.tsv`

| case | conn | buf | maxw | comm_slice | e2e_mean | transfer_mean | vis_wait_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| c16b16s8 | 16 | 16 | 16 | 8MB | 5.6553 | 5.8010 | 0.0093 |
| c20b20s8 | 20 | 20 | 20 | 8MB | 5.6394 | 5.7821 | 0.0090 |
| c24b24s8 | 24 | 24 | 24 | 8MB | 4.8089 | 4.9462 | 0.0149 |
| c28b24s8 | 28 | 24 | 24 | 8MB | 5.9655 | 6.1273 | 0.0093 |
| c16b16s16 | 16 | 16 | 16 | 16MB | 4.3875 | 4.5657 | 0.0299 |
| c20b16s16 | 20 | 16 | 16 | 16MB | 5.7430 | 5.9447 | 0.0158 |

附加复测：
- `/tmp/tc5/final_recheck/summary.tsv`
  - `c20b16s16_rep`: transfer_mean `5.1600`
  - `c28b24s8_rep`: transfer_mean `4.9461`

长稳态复测（`warmup=2, iter=8`）：
- 配置：`conn=20, buf=16, maxw=16, comm_slice=16MB, engine_slice=1024MB, expected_gpu_channels=0`
- 结果：
  - `transfer_mean=5.0644 GiB/s`
  - `transfer_p90=5.7371 GiB/s`
  - `e2e_mean=4.9436 GiB/s`
  - `p2p_ratio=1`, `comm_errors=0`

### 3.6 主要配置与吞吐对照（简化）

这部分只保留“最主要要改的配置”和对应吞吐，便于快速调参。

#### 3.6.1 `tx_slice` / engine pool（决定性）
数据源：`/tmp/tc5/txslice_scale/summary.tsv` 与 `/tmp/tc5/txslice_scale_hi/summary.tsv`

| 主要配置 | 对应值 | transfer_mean GiB/s |
|---|---:|---:|
| `pinned_memory.classes[name=engine].slice_bytes` (`tx_slice`) | 64MB | 1.6951 |
| `pinned_memory.classes[name=engine].slice_bytes` (`tx_slice`) | 128MB | 2.8075 |
| `pinned_memory.classes[name=engine].slice_bytes` (`tx_slice`) | 256MB | 4.1588 |
| `pinned_memory.classes[name=engine].slice_bytes` (`tx_slice`) | 512MB | 4.3659 |
| `pinned_memory.classes[name=engine].slice_bytes` (`tx_slice`) | 1024MB | 5.1483 |

对应实验中的 engine pool：
- 64/128/256/512/1024MB `tx_slice` 分别对应 `8/16/32/64/128GB` pool。

#### 3.6.2 `conn / buffers / maxw / comm_slice`（次要但必要）
数据源：`/tmp/tc5/final_tune/summary.tsv`

| 主要配置组合 | 对应值 | transfer_mean GiB/s |
|---|---|---:|
| `tcp_conn_count / buffers_per_flow / max_window_segments / comm_gpu.slice` | `16/16/16/8MB` | 5.8010 |
| `tcp_conn_count / buffers_per_flow / max_window_segments / comm_gpu.slice` | `20/20/20/8MB` | 5.7821 |
| `tcp_conn_count / buffers_per_flow / max_window_segments / comm_gpu.slice` | `24/24/24/8MB` | 4.9462 |
| `tcp_conn_count / buffers_per_flow / max_window_segments / comm_gpu.slice` | `28/24/24/8MB` | 6.1273 |
| `tcp_conn_count / buffers_per_flow / max_window_segments / comm_gpu.slice` | `16/16/16/16MB` | 4.5657 |
| `tcp_conn_count / buffers_per_flow / max_window_segments / comm_gpu.slice` | `20/16/16/16MB` | 5.9447 |

长稳态推荐配置（复测）：
- `engine.artifact_chunk_bytes=1024MB`
- `pinned_memory.classes[name=engine].slice_bytes=1024MB`
- `pinned_memory.classes[name=engine].pool_bytes=128GB`
- `communicator.transport.tcp_conn_count=20`
- `communicator.stager.buffers_per_flow=16`
- `communicator.stager.max_window_segments=16`
- `pinned_memory.classes[name=comm_gpu].slice_bytes=16MB`
- `communicator.stager.expected_gpu_channels=0`
- 对应吞吐：`transfer_mean=5.0644 GiB/s`（`transfer_p90=5.7371 GiB/s`，`e2e_mean=4.9436 GiB/s`）

---

## 4. 分析与结论

### 4.1 是否打到瓶颈
- 以 `iperf3`（`5.4~5.6 GiB/s`）作为 TCP 对照，长稳态 `transfer_mean=5.064 GiB/s`，约为 ceiling 的 `~93.6%`。
- 在短窗最佳样本中，`transfer_p90` 已接近或超过 `5.6 GiB/s`，说明 MTCP 在当前框架内已进入“近饱和区间”。
- 因 `transfer` 与 `iperf` 计时口径不同，个别样本出现 `transfer > iperf` 是可预期的计量现象。

### 4.2 主瓶颈定位
- 决定性变量是 `engine.slice_bytes (tx_slice)`。
- `conn/buffer` 的收益远小于 `tx_slice`，且过高 conn 可能带来回退。
- 结论：当前阶段优先调 `tx_slice/artifact_chunk` 与 pipeline 粒度，比继续堆 conn 更有效。

### 4.3 当前最优建议（本机）
- 推荐首选：
  - `engine.slice_bytes=1024MB`
  - `engine.artifact_chunk_bytes=1024MB`
  - `engine.pool_bytes=128GB`
  - `communicator.stager.expected_gpu_channels=0`
  - `conn/buf/maxw=20/16/16`
  - `comm_gpu.slice=16MB`
- 预期：
  - 长稳态 transfer 均值约 `5.0 GiB/s`
  - P90 可达 `5.7 GiB/s` 左右

---

## 5. 本轮框架改动（与调优直接相关）

### 5.1 MTCP 可观测性日志降噪（减少高频 INFO 对吞吐干扰）
- 文件：`core/communicator/transport/mtcp_transport.cc`
- 变更：
  - `mtcp_lane` 从 `LOG(INFO)` 调整为 `VLOG(2)`（`core/communicator/transport/mtcp_transport.cc:1149`）
  - `mtcp_read_req` 改为分级日志（慢请求才 INFO/WARNING，常规 VLOG）（`core/communicator/transport/mtcp_transport.cc:1388`）

### 5.2 相关回归
- `bazel test //core/communicator:mtcp_transport_lane_test //core/communicator:tcp_transfer_test --test_env=TENSORCAST_CUDA_BACKEND=fake` 通过。

---

## 6. 过程问题清单（含复现线索）

### 6.1 已修复
1. 2GiB 跨 request lane 映射错位导致的超时卡死（此前已修复并纳入 lane test）。

### 6.2 待优化（性能与使用性）
1. `local_handle_socket_path` 自动路径过长导致 daemon 启动失败。
- 复现日志：
  - `/tmp/tc-bench-20260220/iter-next5/matrix_coarse/.../daemon.err`
- 关键报错：
  - `local_handle_socket_path too long even after shortening`

2. 配置联动可用性：`comm_gpu.slice=16MB + conn=24 + buffers=24` 触发容量校验失败。
- 复现日志：
  - `/tmp/tc5/mc/c24_b24_m24_g1_ch1024_s16_e64/.../daemon.err`
- 关键报错：
  - `pinned_memory.classes[name=comm_gpu] too small: capacity_slices=512 required_slices=600`

3. 默认 `stable_bytes=4GB` 对大轮次 benchmark 容易触发容量不足（已将默认配置提升为 `64GB`，该问题对当前默认配置已缓解）。
- 复现日志：
  - `/tmp/tc5/smoke/run.log`
- 关键报错：
  - `Insufficient stable bytes`

4. 大 pinned pool 下 daemon 就绪时间过长（常见 50~75s）。
- 复现日志：
  - `/tmp/tc5/final_tune/*/run.log`

5. 高性能配置下仍有波动（同配置跨轮次可出现明显回落）。
- 对比数据：
  - `/tmp/tc5/final_tune/summary.tsv`
  - `/tmp/tc5/final_recheck/summary.tsv`

---

## 7. 复现实验命令（关键路径）

### 7.1 清理与启动
```bash
source .venv/bin/activate
pkill -x tensorcast_daemon || true
TENSORCAST_HOME=/tmp/tc5/gs tensorcast-cli global stop || true
TENSORCAST_HOME=/tmp/tc5/gs tensorcast-cli global start --listen-host 127.0.0.1 --listen-port 50051 --json
```

### 7.2 推荐配置（示例）
```bash
BASE_DIR=/tmp/tc5/example \
GLOBAL_ADDR=127.0.0.1:50051 \
CONN_COUNT=20 \
BUFFERS_PER_FLOW=16 \
HEARTBEAT_SEC=5 \
SYNC_SEC=5 \
SIZE_MIB=2048 \
WARMUP=2 \
ITERATIONS=8 \
LOOKUP_MODE=key \
LOG_LEVEL=warn \
VISIBILITY_TIMEOUT_SEC=30 \
VISIBILITY_RETRY_INTERVAL_SEC=0.05 \
bash examples/tensorcast_sdk_p2p_benchmark_driver.sh \
  --daemon-config=/path/to/daemon_config.yaml \
  --set=communicator.stager.max_window_segments=16 \
  --set=communicator.stager.expected_gpu_channels=0
```

---

## 8. 总结

- 在严格 `key` 路径和本机双 daemon P2P 模式下，MTCP 已可逼近本机 TCP 上限。
- 当前最关键调优手柄是 `engine.slice_bytes(tx_slice)` 与对应 pool 容量，不是继续线性提升连接数。
- 后续若要进一步提升稳定性，优先处理：
  - daemon 启动慢（大池初始化）
  - 路径长度导致的 handle socket 失败
  - 超大轮次 benchmark 的 stable_bytes 自适配
  - 统一链路/transfer 计时口径，减少 ceiling 对比歧义。

---

## 9. 后续改进问题清单（完整）

以下问题来自本次完整“修改-测试-迭代”过程，包含性能、可用性和可观测性三个维度。

1. 请求粒度导致性能断崖。
- 现象：`engine.slice=64MB` 时吞吐长期落在 `~1.7-2.1 GiB/s`，扩大到 `1024MB` 才接近上限。
- 影响：默认配置下很难逼近链路能力。
- 建议：在 communicator/materialization 层做“多 request 聚合”或“更粗粒度窗口调度”，降低每 request 固定开销。

2. 高吞吐配置依赖超大 pinned pool，启动慢。
- 现象：`engine.pool=128GB` 等配置下，daemon ready 常见 `50~75s`。
- 影响：调试迭代成本高，线上弹性扩缩慢。
- 建议：优化 pinned pool 初始化策略（分批/惰性/并行预热），并提供启动阶段进度指标。

3. `local_handle_socket_path` 自动推导容易超出 AF_UNIX 长度限制。
- 现象：深路径 benchmark 目录下频繁启动失败。
- 影响：同机调试易踩坑，可复现性差。
- 建议：默认使用更短的固定目录（如 `/tmp/tc-uds/...`），并在 CLI 启动前预检路径长度。

4. `comm_gpu.slice` 与 `conn/buffers` 的容量耦合缺少预估工具。
- 现象：`comm_gpu.slice=16MB + conn=24 + buffers=24` 直接启动失败（容量不足）。
- 影响：参数调优效率低，失败成本高。
- 建议：CLI 增加 “capacity estimator / dry-run” 输出 required_slices 与建议 pool。

5. 默认 `stable_bytes` 在压测场景的适配性。
- 现象：历史默认 `4GB` 下中等轮次即出现 `Insufficient stable bytes`；当前默认已提升为 `64GB`，但更大轮次仍可能触发。
- 影响：当压测规模继续扩大时，仍可能出现容量误判。
- 建议：提供 benchmark profile（按 `size_mib * (warmup+iterations)` 自动建议 stable_bytes，或自动启用 cleanup-artifacts）。

6. `transfer` 与 `iperf` 口径不一致，导致“是否打满”判断困难。
- 现象：部分样本 `transfer_gibps` 高于 `iperf` ceiling。
- 影响：性能结论容易被误读。
- 建议：补充统一口径指标（wire-bytes / wire-time）并在 JSON 中明确字段定义。

7. 高吞吐配置下仍有明显抖动。
- 现象：同配置跨轮次可从 `~6.1` 回落到 `~4.9 GiB/s`。
- 影响：难以给出稳定 SLA。
- 建议：引入 scheduler/queue 深度指标和 stall reason 计数，结合 CPU 亲和性与 NUMA 策略做约束。

8. 连接数继续升高并不稳定增益，甚至退化。
- 现象：`conn=24` 有回落，`conn=28` 有时高有时低。
- 影响：盲目堆并发导致噪声和回退。
- 建议：增加自动调参建议逻辑（基于短跑探测收敛到最佳 conn/buffer 区间）。

9. benchmark 驱动缺少“环境健康预检”。
- 现象：GS 未启动时 run 直接失败；异常信息虽有，但流程中断明显。
- 影响：批量实验易中断，浪费时间。
- 建议：driver 开始前检查 GS 可达、端口冲突、容量配置可行性，并给出 fail-fast 报告。

10. benchmark 驱动缺少“一键清理模式”。
- 现象：需要手动 `pkill` + 多个 `TENSORCAST_HOME` 清理。
- 影响：多人共机时容易残留干扰进程。
- 建议：提供 `tensorcast-cli bench env clean`（清理 daemon/global/session/临时目录）。

11. daemon 启动等待缺少阶段化可见性。
- 现象：日志只表现为“Still waiting...”，不清楚卡在 pool init、GS 注册还是 gRPC ready。
- 影响：问题定位慢。
- 建议：暴露启动 phase（pool init / gs register / grpc ready）的时间分解。

12. key/index 路径在压力下虽然可用，但可见性等待仍有尾延迟。
- 现象：`visibility_wait_sec_p90` 在不同配置有明显波动。
- 影响：e2e 吞吐与时延稳定性受影响。
- 建议：补充 key mapping publish/observe 延迟指标，定位是 GS、daemon 缓存还是同步周期造成。

---

## 10. 跨机补充测试（2026-02-21，H800↔H800）

> 本节是本报告的“跨机增量”。目标是验证不同 worker 之间的真实 P2P `put/get` 调优区间，并沉淀一套可复用的标准化运行方式。

### 10.1 跨机拓扑与约束

- Global Store：本机 CPU 节点（`dev-yuchu`），地址 `100.97.246.95:55051`
- Put worker：`dev-yuchu-gpjsz-2551881-worker-0`（H800），daemon `100.101.240.71:62001`
- Get worker：`dev-yuchu-m2xvj-2551880-worker-0`（H800），daemon `100.99.166.45:62011`
- 统一约束：
  - `lookup_mode=key`
  - `prefer=p2p`
  - `allow_disk=false`
  - measured 轮次要求 `source=p2p`

### 10.2 关键方法变更（跨机必需）

单进程脚本 `put -> get`（同一 client 进程）在跨机时会触发 CUDA IPC 失败：
- 典型报错：`cudaIpcOpenMemHandle ... cudaErrorInvalidValue`
- 根因：CUDA IPC 句柄是**同机**语义，不能跨主机打开。

因此跨机 benchmark 必须改成“分离式”：
1. put 机本地 client 连接 put daemon 执行 `put(key=...)`
2. get 机本地 client 连接 get daemon 执行 `get(key=...)`
3. 每轮使用**唯一 key**，避免 get 端本地缓存把 source 变成 `local_replica`

### 10.3 标准化运行方式（v1）

本轮沉淀的标准化流程如下（已实跑）：

1. 固定拓扑：本机 GS + 双 worker daemon（put/get 分离）
2. 固定输入：
   - `size_mib=2048`
   - `warmup=1`
   - `iterations=4`
   - `conn/buffers/max_window/expected_gpu_channels` 作为矩阵变量
3. 固定采样单元（每轮）：
   - put 侧 `put_once.py`（本地 GPU）
   - get 侧 `get_once.py`（本地 GPU，带可见性重试）
4. 固定输出契约（JSON）：
   - 每轮：`source`, `e2e_sec`, `transfer_sec`, `e2e_gibps`, `transfer_gibps`, `visibility_wait_sec`
   - 汇总：`e2e_gibps_mean/p90`, `transfer_gibps_mean/p90`, `visibility_wait_sec_p90`, `p2p_ratio`
5. 固定失败门槛：
   - measured 轮次出现非 `p2p` 直接判失败
   - daemon 启动容量校验失败直接记录配置不可行（不“硬跑”）

本轮 runner 产物：
- `/tmp/tc_cross_20260221/run_cross_case.py`
- `/tmp/tc_cross_20260221/tools/put_once.py`
- `/tmp/tc_cross_20260221/tools/get_once.py`
- 结果目录：`/tmp/tc_cross_20260221/results/*.json`

### 10.4 跨机矩阵结果

数据源：`/tmp/tc_cross_20260221/results/*.json`

| case | conn/buf/maxw | expected_gpu_channels | e2e_mean GiB/s | transfer_mean GiB/s | vis_wait_p90 s | 结果 |
|---|---:|---:|---:|---:|---:|---|
| c16b16w16_g0 | 16/16/16 | 0 | 2.4107 | 2.4734 | 0.0219 | PASS |
| c20b16w16_g0 | 20/16/16 | 0 | **2.4417** | **2.4851** | 0.0151 | PASS |
| c20b20w20_g0 | 20/20/20 | 0 | 2.2755 | 2.3164 | 0.0149 | PASS |
| c24b16w16_g0 | 24/16/16 | 0 | 1.9209 | 1.9732 | 0.0758 | PASS（明显退化） |
| c20b16w16_g8 | 20/16/16 | 8 | 2.2317 | 2.2803 | 0.0419 | PASS（较 g0 退化） |
| c28b24w24_g0 | 28/24/24 | 0 | - | - | - | FAIL（容量校验） |

失败样本（关键日志）：
- `pinned_memory.classes[name=comm_gpu] too small: capacity_slices=512 required_slices=696`
- 说明 `conn/buffers` 超出当前 `comm_gpu` pinned pool 容量上限，不是瞬态网络问题。

### 10.5 跨机结论

1. 在当前跨机链路与默认 pinned 配置下，`20/16/16 + expected_gpu_channels=0` 是本轮最稳且最优组合。  
2. `conn` 提升到 24 后，均值与尾延迟明显变差（`vis_wait_p90` 上升到 `0.0758s`）。  
3. `expected_gpu_channels=8` 在本轮环境里不增益，反而降低吞吐并放大可见性等待。  
4. 与本机双 daemon 的 `~5 GiB/s` 相比，跨机本轮稳定区间在 `~2.3-2.5 GiB/s`。该差值来自跨机链路、可见性窗口与资源调度开销叠加，不属于单一参数可完全消除的问题。

### 10.6 本轮踩坑与解决办法（跨机专属）

1. `uv run` 在远端 worker 上触发 editable build 问题（`safe.directory` / PATH 中无 `uv`）。
- 解决：统一 `source .venv/bin/activate` 后直接 `python ...`。

2. 单进程跨机 `put/get` 触发 CUDA IPC `invalid argument`。
- 解决：改成 put/get 分离进程，分别 colocate 到各自 daemon。

3. `put` 后立即 `get` 偶发 `Artifact not found`。
- 解决：在 `get_once.py` 增加可见性重试（`VIS_TIMEOUT_SEC` + `VIS_RETRY_SEC`）。

4. 远端创建的调试目录属主为 root，导致本地清理失败。
- 解决：runtime/log 全部改落 `/tmp/tc_cross_20260221`，仓库目录不做跨机运行态落盘。

5. 高并发参数可直接导致 daemon 启动失败（非运行时抖动）。
- 解决：把“容量校验失败”纳入 benchmark 矩阵结果；后续需要 capacity estimator/dry-run 才能高效扫参。

### 10.7 标准化下一步（建议）

为把上述流程从“临时脚本”升级为长期可复用，建议下一步在仓库内固化：
1. `examples/cross_host/` 下提供官方 `cross_host_matrix_runner`（参数化 process id / daemon addr / matrix）。
2. 输出统一 JSON schema（与本节 `summary/records` 字段一致）。
3. 启动前增加三项 preflight：GS 可达、daemon 配置容量可行、端口冲突检查。
4. 在报告中固定使用 `warmup=1, iter=4`（快速调参）+ `warmup=2, iter=8`（稳态复核）双阶段。

---

## 11. 跨机瓶颈深挖与调优方法（2026-02-21，第二轮）

本节补充“如何系统定位跨机瓶颈”，并给出第二轮同链路复测结果（避免不同 worker 对造成偏差）。

### 11.1 分析方法（可复用）

将跨机性能拆成 4 层，逐层归因：

1. 网络层（理论/工具上限）
- 用 `iperf3` 双向测试并发吞吐，作为链路参考区间。
- 注意：`iperf3` 单进程在某些方向上可能低估上限，必须结合框架内通信计数器交叉验证。

2. 传输层（框架 data-plane）
- 以 `transfer_sec` 与 `transfer_gibps` 观察纯传输窗口。
- 用 daemon `communication_info.total_bytes_transferred` 前后差分校验真实线上字节（`comm_bytes_delta`）。

3. 控制层（可见性/重试）
- 观察 `visibility_wait_sec` 与 `visibility_attempts`，定位 key 映射可见性抖动。

4. 端到端层（用户体感）
- 观察 `e2e_gibps`，与 `transfer_gibps` 对比判断“网络/传输”与“控制等待”哪一层在吞吐上限前先成为瓶颈。

本轮使用的标准采样脚本（运行态）：
- `/tmp/tc_cross_20260221/run_cross_case.py`
- `/tmp/tc_cross_20260221/tools/put_once.py`
- `/tmp/tc_cross_20260221/tools/get_once.py`

### 11.2 同链路复测（pairB）

拓扑（同一对 worker）：
- put daemon: `100.99.166.46:62001`
- get daemon: `100.98.112.151:62011`
- GS: `100.97.246.95:55051`

结果（`size_mib=2048`, `warmup=1`, `iter=4`）：

| case | conn/buf/maxw | egc | put_mean sec | e2e_mean GiB/s | transfer_mean GiB/s | vis_wait_p90 s | attempts_p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pairB_c16b16w16_g0_v2 | 16/16/16 | 0 | 3.4507 | 2.6928 | 2.7449 | 0.0142 | 1 |
| pairB_c20b16w16_g0_v2 | 20/16/16 | 0 | 3.7164 | 2.2636 | 2.8941 | 2.2285 | 28 |
| pairB_c20b16w16_g8_v2 | 20/16/16 | 8 | 3.5491 | 2.7973 | 2.8586 | 0.0186 | 1 |
| pairB_c20b20w20_g0_v2 | 20/20/20 | 0 | 3.5471 | **2.9480** | **3.0099** | 0.0143 | 1 |
| pairB_c24b16w16_g0_v2 | 24/16/16 | 0 | 3.4200 | 2.7411 | 2.8795 | 0.0881 | 1 |

网络参考（`iperf3`, P=20）：
- `put->get`（`-w 16M`）：`2.1439 GiB/s`
- `get->put`（`-w 16M`）：`2.9823 GiB/s`

交叉校验样本（框架内计数器）：
- 单次 `get`：`comm_bytes_delta = 2147483648`（= 2GiB）
- 对应 `transfer_sec = 0.7132s`，`transfer_gibps = 2.8042`
- 说明框架确实传输了完整 2GiB，不是“少传字节”导致的假高吞吐。

### 11.3 瓶颈结论

1. 网络层存在明显方向性差异。
- `iperf3` 显示 `put->get` 与 `get->put` 差距明显（约 `2.14` vs `2.98 GiB/s`）。
- 结论：跨机链路并非“单值上限”，方向与 worker 组合会影响结果。

2. 框架控制路径可见性抖动会直接击穿 e2e。
- 例：`pairB_c20b16w16_g0_v2` 中 `transfer_mean=2.8941`，但 `e2e_mean=2.2636`。
- 根因：`visibility_attempts_p90=28`，`vis_wait_p90=2.2285s`，控制层等待主导了端到端退化。

3. data-plane 有“staging credit 满载”迹象。
- daemon 日志长期出现：
  - `[staging_credit] ... waiting for staging credit outstanding=16/16`
  - 高并发组合出现 `outstanding=20/20`
- 说明 stager credit 常态打满，继续堆并发并不保证收益，可能放大尾部延迟。

4. put 路径本身也是独立瓶颈。
- 各 case `put_mean` 基本在 `3.4~3.7s / 2GiB`，对应约 `0.54~0.60 GiB/s`。
- 即使 get 端更快，完整 `put+get` 管线总耗时仍受 put 阶段显著限制。

5. 高并发配置不仅有“运行时抖动”，还有“启动期容量门槛”。
- `conn=28, buffers=24, maxw=24` 在默认 pool 下启动失败（`required_slices > capacity_slices`）。
- 提升 `comm_gpu.pool_bytes` 时还会触发配置耦合约束（例如 `artifact_chunk_bytes` 与 engine slice 对齐）。

### 11.4 调优建议（当前跨机环境）

1. 推荐优先配置（稳定+性能平衡）：
- `conn/buf/maxw = 20/20/20`
- `expected_gpu_channels = 0`（或在本机复测后选 `8`，以 e2e 稳定性为准）

2. 评估配置时同时看 3 个指标，不只看 transfer：
- `transfer_gibps_mean`
- `e2e_gibps_mean`
- `visibility_wait_sec_p90 / visibility_attempts_p90`

3. 先控制可见性抖动，再追求更高并发。
- 若 `attempts_p90` 或 `vis_wait_p90` 拉高，优先排查 key 映射可见性链路，而不是盲目继续加 conn。

4. 高并发前先做容量可行性检查。
- 尤其是 `comm_gpu.pool_bytes` 与 `conn*buffers*max_window` 的联动。
- 建议在 runner 前置 `dry-run/capacity estimator`，避免“跑到启动期才失败”。

---

## 12. 跨机双向链路更新（2026-02-22）

### 12.1 术语澄清（必须统一）

为避免“get->put”歧义，本报告统一使用：
- `put->get（正向链路，forward link）`：A 节点执行 `put`，B 节点执行 `get`。
- `get->put（反向链路，reverse link）`：仍是先 `put` 再 `get`，但角色互换为 B 节点 `put`、A 节点 `get`。

`put` 的语义：
- `put(policy=pinned)` 是把 artifact 注册并落到 **put 侧 daemon 的本地稳定内存层（stable DRAM）**，随后通过 key/artifact_id 被远端 `get` 拉取。

`wave1/wave2` 的语义与倍数口径：
- 详见 `docs/benchmarks/20260221-multi-host-p2p-fanout-benchmark-report.md` 的 `2.3`（Wave 定义）与 `3.1`（`2x/4x` 理论差距量化）。
- 简化口径：`wave2/wave1 > 1` 说明 fanout 扩散生效；`~2x` 是工程可达目标；`~4x` 是理想上限。

本报告中 `compact` 的语义：
- `compact` 不是协议或传输模式，而是“**内存收缩版 daemon 配置档位**”的简称，用于 64GiB 且波动较大的 worker 上保证启动 preflight 和双向链路都可跑通。
- 对应配置文件：`examples/config/store_daemon_config_cross_host_64g_compact.yaml`
- 关键参数：`engine.pool=8GB`、`engine.slice=256MB`、`stable=8GB`、`comm_gpu.pool=6GB`（相对 baseline 64g 档位更保守）。

### 12.2 标准化 benchmark 运行方式（仓库内）

已在仓库内落地统一 runner：
- `examples/cross_host/cross_host_matrix_runner.py`
- `examples/cross_host/cross_host_put_once.py`
- `examples/cross_host/cross_host_get_once.py`
- `examples/cross_host/cross_host_deregister_once.py`

统一流程：
1. 本机 CPU 节点启动 GS（blocking）。
2. 两台 GPU worker 启 daemon（put/get 分离）。
3. 每轮唯一 key：`put_once -> get_once -> put/get 双侧 deregister`。
4. 输出统一 JSON（`summary + records`），并强制校验 measured 轮次 `source=p2p`。
5. 双阶段：
- 快速调参：`warmup=1, iter=4`
- 稳定性验证：更长轮次（例如 `iter>=24`）

补充修复：
- `cross_host/cross_host_get_once.py` 改为 `startup.current_client()` 读取 `get_detailed_status()`，`comm_bytes_delta` 与 `total_bytes` 可直接做字节一致性校验。
- 烟测样本 `compact_fwd_comm_delta_smoke`：`comm_bytes_delta=536870912` 与 `total_bytes=536870912` 一致。
- 运行过程中生成的 `.codex_brainctl_logs` / `.codex_brainctl_runs` 属于调试中间产物：保留可复盘，产出归档后可安全删除。

### 12.3 “stable 释放节奏滞后”问题：复现、根因、修复

#### 12.3.1 复现与现象

复现命令（旧行为）：
- `size_mib=2048, warmup=1, iterations=40, policy=pinned`
- 配置：`examples/config/store_daemon_config_cross_host_64g.yaml`（`stable_bytes=16GB`）

旧行为在第 7 轮附近稳定失败：
- `CommitRegisteredArtifact failed: Insufficient stable bytes: requested=2147483648 used=17179869184 total=17179869184`
- 同时 daemon 日志持续出现：
  - `stable_cache.release_lease_skipped_missing_replica ...`

#### 12.3.2 根因分析

根因不是“用户侧重试不足”，而是 runtime 内部事件顺序问题：
1. `retire_replica_status/clear_mem` 先发 `kReplicaEvicted` 事件。
2. `RuntimeContext` 的 stable cache 订阅先收到事件，但此时只拿到 key，拿不到 replica source。
3. 早期实现会在无 source 的情况下删 stable cache 条目，导致后续真正的 `deregister` 释放路径无法正确释放 lease。

涉及代码：
- `core/store/runtime/replica/replica_runtime.cc`
- `core/store/components/stable_dram_cache_manager.cc`

#### 12.3.3 修复内容

1. 调整顺序（先释放 stable，再发事件）：
- `core/store/runtime/replica/replica_runtime.cc`
  - `retire_replica_status`
  - `clear_mem`

2. 增强 stable cache 防护：
- `core/store/components/stable_dram_cache_manager.cc`
  - 当 `on_replica_evicted` 缺少 replica source 时，不再提前删除条目；等待后续带 source 的调用完成释放。

3. 回归测试：
- `core/store/components/stable_dram_cache_manager_test.cc`
  - 新增“runtime event 先到、deregister 后到”的竞态测试。

验证：
- `bazel test //core/store:stable_dram_cache_manager_test --test_env=TENSORCAST_CUDA_BACKEND=fake` 通过。
- 长轮次修复验证：`leak_regress_put_to_get_after_fix_1g_x24`（1GiB, iter=24）完整通过，未再出现 `Insufficient stable bytes`。
- 2GiB 验证（compact profile）：`leak_regress_compact_put_to_get_2g_x10`（iter=10）完整通过，`deregister` 全程单次完成。
- 对应结果：`/tmp/tc_cross_20260221/results/leak_regress_put_to_get_after_fix_1g_x24.json`
- 对应结果：`/tmp/tc_cross_20260221/results/leak_regress_compact_put_to_get_2g_x10.json`

### 12.4 双向链路矩阵结果（统一 compact 配置）

由于一台 worker 空闲内存不足（startup preflight 失败），双向对比统一采用：
- `examples/config/store_daemon_config_cross_host_64g_compact.yaml`
- `engine.pool=8GB, stable=8GB, comm_gpu.pool=6GB`

`size_mib=1024, warmup=1, iter=4`：

| case | conn/buf/maxw | egc | put_mean s | e2e_mean GiB/s | transfer_mean GiB/s | vis_wait_p90 s |
|---|---:|---:|---:|---:|---:|---:|
| compact_fwd_put_to_get_c16b16w16_g0 | 16/16/16 | 0 | 1.6329 | 2.4987 | 2.5998 | 0.0206 |
| compact_fwd_put_to_get_c20b16w16_g0 | 20/16/16 | 0 | 1.6019 | **2.7810** | **2.9797** | 0.0498 |
| compact_fwd_put_to_get_c20b16w16_g8 | 20/16/16 | 8 | 1.6051 | 2.4090 | 2.4942 | 0.0221 |
| compact_rev_link_get_to_put_c16b16w16_g0 | 16/16/16 | 0 | 1.5504 | 2.2632 | 2.3339 | 0.0138 |
| compact_rev_link_get_to_put_c20b16w16_g0 | 20/16/16 | 0 | 1.5276 | 2.3685 | 2.4620 | 0.0389 |
| compact_rev_link_get_to_put_c20b16w16_g8 | 20/16/16 | 8 | 1.5112 | **2.6972** | **2.8858** | 0.0420 |

数据文件：
- `/tmp/tc_cross_20260221/results/compact_fwd_put_to_get_c16b16w16_g0.json`
- `/tmp/tc_cross_20260221/results/compact_fwd_put_to_get_c20b16w16_g0.json`
- `/tmp/tc_cross_20260221/results/compact_fwd_put_to_get_c20b16w16_g8.json`
- `/tmp/tc_cross_20260221/results/compact_rev_link_get_to_put_c16b16w16_g0.json`
- `/tmp/tc_cross_20260221/results/compact_rev_link_get_to_put_c20b16w16_g0.json`
- `/tmp/tc_cross_20260221/results/compact_rev_link_get_to_put_c20b16w16_g8.json`

### 12.5 当前跨机调优结论（双向）

1. 链路方向确实影响最优点：
- 在 `g0` 条件下，反向链路最佳 transfer 比正向链路最佳 transfer 低约 `17.4%`（`2.462` vs `2.980 GiB/s`）。

2. `expected_gpu_channels` 对正反向影响不对称：
- 正向：`g8` 相比 `g0` 明显回退（`2.494` vs `2.980 GiB/s`）。
- 反向：`g8` 反而提升（`2.886` vs `2.462 GiB/s`）。

3. 控制面等待仍是关键观测维度：
- 例如正向 `c20b16w16_g0`，`vis_wait_p90=0.0498s`，虽吞吐最高但尾部等待更高。

建议：
- 若只追正向：优先 `c20/b16/w16 + g0`。
- 若做反向链路（`get->put`）：优先 `c20/b16/w16 + g8`。
- 若需要统一双向保守配置：可先用 `c16/b16/w16 + g0` 作为基线。

“Forward 最优 / Reverse 最优”的口径（本报告）：
- `Forward 最优`：在正向链路候选集中（A put, B get）`transfer_gibps_mean` 最高，且 `e2e_gibps_mean` 不出现反向退化的配置。
- `Reverse 最优`：在反向链路候选集中（B put, A get）按同一口径筛出的最优配置。
- 在本轮 compact 矩阵中：
  - `Forward 最优 = c20/b16/w16 + g0`（`transfer_mean=2.9797`, `e2e_mean=2.7810`）。
  - `Reverse 最优 = c20/b16/w16 + g8`（`transfer_mean=2.8858`, `e2e_mean=2.6972`）。

### 12.6 本轮新增踩坑与解决

1. `deregister` 外围重试会掩盖框架缺陷。
- 处理：改为单次 `deregister(wait=true)`，并直接修框架根因。

2. worker 内存波动导致 startup preflight 失败。
- 处理：提供 compact 配置；并把 preflight 失败记录为“配置/环境不可行”，不误判为网络抖动。

3. `comm_gpu` pool 切片不足导致 daemon 启动即失败。
- 现象：`capacity_slices=256 required_slices=272`
- 处理：compact 配置将 `comm_gpu.pool_bytes` 从 `4GB` 提升到 `6GB`。

4. 2GiB 长轮次压测触发 cgroup OOM（非 stable 计量报错）。
- 现象：`Killed process ... tensorcast_daem ... memory limit 67108864kB`
- 处理：把“长期稳定性验证”与“高负载吞吐验证”拆开；在 64GiB worker 上优先用 1GiB 长轮次验证释放路径。

---

## 13. RSS 分层（可操作定义）与完整修复方案

### 13.1 RSS 分层定义（怎么量、看什么）

为避免“只看一个 RSS 数字”的误判，本报告把内存观测拆成 3 层，并要求同一时间窗对齐采样：

1. **L0: cgroup 层（容器/进程组真实约束）**
- 核心字段：`memory.current`、`memory.max`、`memory.stat`（重点 `anon/file/shmem/inactive_file/slab`）。
- 建议派生量：
  - `effective_current = memory.current - inactive_file`
  - `cgroup_headroom = memory.max - effective_current`
- 用途：判断 OOM 风险是否来自 cgroup 预算耗尽，而非 TensorCast 内部单一池子。

2. **L1: 进程 RSS 层（daemon 进程自身画像）**
- 核心字段：`/proc/<pid>/status` 的 `VmRSS/RssAnon/RssFile/RssShmem/VmSwap`。
- 补充字段：`/proc/<pid>/smaps_rollup` 的 `Rss/Pss/Private_Dirty/Shared_Clean`。
- 用途：区分匿名内存增长（通常更危险）与文件页缓存增长（可回收部分）。

3. **L2: 框架账本层（TensorCast 可解释内存）**
- `stable`：`StoreEngine::get_memory_tier_snapshot()` 的 `stable_used_bytes/stable_total_bytes`。
- `stable cache`：`tc_stable_cache_bytes_used`。
- `pinned`：`tc_pinned_total_bytes`、`tc_pinned_committed_bytes`、`tc_pinned_class_*`。
- `comm`：`GetDetailedStatus.communication_info.total_bytes_transferred`（吞吐对照）+ staging credit 日志（瓶颈线索）。
- 用途：把“进程 RSS 增长”拆成可归因项，识别“未入账增长”。

统一判读口径：
- `unaccounted_rss = RssAnon - (stable_used_bytes + pinned_committed_bytes + comm_inflight_estimate)`
- 若 `unaccounted_rss` 在多轮后持续抬升，优先排查 lease/export/retire 清理路径和 allocator 碎片。

### 13.2 采样方法（可直接执行）

```bash
# 1) 定位 daemon pid
PID=$(pgrep -f tensorcast_daemon | head -n 1)

# 2) L1: 进程 RSS 分层
grep -E "VmRSS|RssAnon|RssFile|RssShmem|VmSwap" /proc/$PID/status
cat /proc/$PID/smaps_rollup

# 3) L0: cgroup 分层
CG=$(awk -F: '$1=="0"{print $3}' /proc/$PID/cgroup)
cat /sys/fs/cgroup$CG/memory.current
cat /sys/fs/cgroup$CG/memory.max
grep -E "anon|file|shmem|inactive_file|slab" /sys/fs/cgroup$CG/memory.stat
```

建议在每轮 `put_once -> get_once -> deregister` 的 4 个时刻采样：
- `T0`（put 前）
- `T1`（put 后）
- `T2`（get 后）
- `T3`（双侧 deregister 成功后）

### 13.3 与当前代码路径的对应关系

当前关键路径与风险点：
- `DeregisterArtifact`：`daemon/service/grpc_service_impl.cc`
  - 有 active lease 时会走 drain + revoke。
  - 无 active lease 时当前会落入 “stateless retire” 分支（消息：`no active lease found; proceeding with stateless retire`）。
- drain 逻辑：`daemon/service/artifact_retire_utils.cc`
  - `wait_exports_drained` 仅在进入 drain 流程时执行。
- lease/export 状态：`daemon/state/lip_manager.cc`
  - `quiesce_lease` / `wait_exports_drained` / `revoke_by_registration_id`。
- stable 释放：`core/store/runtime/replica/replica_runtime.cc` + `core/store/components/stable_dram_cache_manager.cc`
  - 已修复“事件顺序导致 lease 释放丢失”的竞态（先释放 stable，再发 `kReplicaEvicted`）。

### 13.4 完整修复方案（按实现阶段）

#### 阶段 A：观测补全（先把问题“量准”）

1. 新增 daemon 进程/cgroup 内存采样器（建议放在 daemon status/controller 层）：
- 周期采样并导出 `proc_rss_anon_bytes/proc_rss_file_bytes/proc_rss_shmem_bytes/cgroup_effective_current_bytes/cgroup_headroom_bytes`。
- 与现有 `stable/pinned` 指标同窗口输出，形成固定 dashboard。

2. 增加“未入账 RSS”指标：
- `tc_memory_unaccounted_bytes = rss_anon - stable_used - pinned_committed - comm_inflight_estimate`。
- 用于快速识别“框架外增长”或“清理滞后”。

3. benchmark runner 接入采样：
- 在 `examples/cross_host/cross_host_matrix_runner.py` 每轮写入 T0/T1/T2/T3 四点快照。
- 结果 JSON 增加 `memory_timeline`，用于自动判定泄漏/滞后。

#### 阶段 B：deregister 语义收敛（单次调用必须可收敛）

1. 消除“无 active lease 就不 drain”的语义缺口：
- 在 `DeregisterArtifact` 无 active lease 分支，仍按 `artifact/view/device` 解析本地 replica key 集合。
- 对每个 key 执行：`quiesce -> wait_exports_drained(timeout) -> retire_replica_status`。

2. 明确成功定义：
- `deregister(wait=true)` 返回成功时，必须满足：
  - 本地 key 已 retire（或 NotFound 视为已收敛）。
  - 导出引用计数归零（超时则返回 `DEADLINE_EXCEEDED`，不静默成功）。

3. 失败可诊断：
- 返回消息中附带 `drain_timeout_keys`、`retire_failed_keys`，便于直接定位卡点。

#### 阶段 C：运行期内存背压（不只靠启动 preflight）

1. 增加运行期 admission guard（commit/begin register 路径）：
- 复用 `startup_memory_preflight` 同类计算思路（`effective_current/headroom`）。
- 当 `cgroup_headroom < required_next_artifact + safety_margin` 时，快速失败 `RESOURCE_EXHAUSTED`。

2. 增加 in-flight 上限：
- 对 `put/get` 管道设置 `max_inflight_bytes` 与 `max_inflight_artifacts`。
- 超限时排队或快速失败，防止释放滞后期间继续放大 RSS 峰值。

3. 增加 staging credit 饱和指标：
- 补充 `staging_credit_outstanding/total`、`staging_credit_wait_seconds`、`staging_credit_timeout_total` 指标。
- 用于区分网络瓶颈与框架内部信用窗口瓶颈。

#### 阶段 D：标准化压测与门禁

1. 标准用例分层：
- `leak-regress`：1GiB，`iter>=24`（验证释放收敛）。
- `pressure-regress`：2GiB，`iter>=10`（验证高负载无 OOM）。
- `throughput-regress`：正向/反向链路各 3 组参数，输出 transfer/e2e/vis_wait。

2. 结果统一产物：
- 每个 case 输出 `summary + records + memory_timeline + daemon_log_digest`。
- 作为后续变更的固定回归基线。

### 13.5 验证标准（每次修复必须满足）

功能正确性：
1. `deregister(wait=true)` 对同一 key 单次调用可收敛，不依赖外围重试。
2. 不再出现 `stable_cache.release_lease_skipped_missing_replica`。
3. 无 active lease 场景下，若存在本地 replica，仍会执行 drain/retire 路径。

内存收敛性：
1. 单轮结束（T3）后 `stable_used_bytes` 回到基线（允许 ≤ 1 个 chunk 抖动）。
2. 连续 24 轮后 `RssAnon` 相对 T0 漂移不超过 `max(512MiB, 0.08 * stable_total_bytes)`。
3. `unaccounted_rss` 不得呈单调上升趋势（允许短窗抖动，不允许阶梯累积）。

性能稳定性：
1. 吞吐回归不低于当前基线的 `95%`（forward/reverse 分别对比）。
2. `visibility_wait_sec_p90` 不得较基线恶化超过 `30%`。
3. `comm_errors_delta_sum == 0`。

### 13.6 回归门槛（CI/准入建议）

建议把以下阈值固化为回归门槛：
1. **OOM 门槛**：所有标准用例 `oom_kill_count == 0`。
2. **释放门槛**：`deregister_single_call_success_rate == 100%`（在规范用例中）。
3. **内存门槛**：`rss_anon_drift_after_24_iter <= 512MiB`（或按 8% 相对阈值，取更宽者）。
4. **吞吐门槛**：`transfer_gibps_mean >= 0.95 * baseline`（正/反向分别考核）。
5. **控制面门槛**：`visibility_wait_sec_p90 <= 1.3 * baseline_p90`。
