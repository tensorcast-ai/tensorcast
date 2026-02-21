# TensorCast SDK P2P (MTCP) Put/Get 极限性能调优测试报告

日期：2026-02-21  
代码版本：`400c81a6`  
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
- TensorCast 代码：`git rev-parse --short HEAD = 400c81a6`

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
