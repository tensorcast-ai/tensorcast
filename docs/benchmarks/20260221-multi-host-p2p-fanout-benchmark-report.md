# TensorCast 多机 P2P Put/Get 扩容验证与调优（一体化计划与执行报告）

日期：2026-02-22  
代码版本：`29cb29a5`

## 1. 目标与范围

本文将原 `plan` 与 `report` 合并为一份可执行、可验收、可复盘的统一文档，覆盖以下目标：

1. 验证 Seed 节点 `put(policy=pinned)` 后，artifact 落在 Seed daemon 的 stable DRAM，并可被其他 GPU 节点 `get(device=cuda:0)`。
2. 验证每个成功 `get` 的节点能成为新的 P2P 可导出源（fanout/cascade 语义成立）。
3. 验证 `get(device=cuda:0)` 形成的 VRAM 副本可继续作为后续 hop 的源。
4. 量化扩容趋势（2 -> 4 -> 6/8/9 节点）与瓶颈，给出修复与调优结论。

## 2. 术语与判定口径

1. `put->get（forward）`：A 节点 put，B/C/... 节点 get。
2. `Reverse Link`：角色互换后仍是 put->get，不是协议方向反转。
3. `put(policy=pinned)`：artifact 注册并落在 put 侧 daemon 的 stable DRAM，可被远端 P2P 拉取。
4. `P2P 正常`：measured 轮 `source` 为 `p2p`，且在上游退役后链路仍可继续。
5. `VRAM 源正常`：上游应可观测到 `GPU + EXPORTABLE`，且下一跳 `transport.source_address` 符合预期链路。

## 3. 测试环境与前置

1. 本机 CPU 节点启动 Global Store。
2. 通过 `brainctl launch` 拉起多台 GPU worker（统一 `--charged-group=tensorcast_dev`）。
3. 每台 worker 使用同一代码目录 `/data/workspace/tensorcast-280` 与 `.venv`。
4. Python 命令统一在 `source .venv/bin/activate` 后执行。

## 4. 计划设计（执行前）

### 4.1 Phase A：功能正确性（2~4 节点）

目标：强证明“get 后副本可作为下一跳源”。

步骤：
1. Seed 节点 S0 put 1GiB 唯一 key。
2. S1 get（期望 `source=p2p`）。
3. 退役 S0（deregister 或 stop）。
4. S2 get 同一 artifact。
5. 重复级联到 S3，验证链路持续。

判定：
1. 所有 get 成功。
2. `source=p2p`。
3. `comm_bytes_delta == total_bytes`。
4. 上游退役后链路仍可继续。

### 4.2 Phase B：扩容性能（4~9 节点）

目标：验证 fanout 扩散是否带来吞吐提升。

流程（每个 case）：
1. S0 put 唯一 key（1/2/8/32GiB）。
2. Wave1：前半 getters 并发 get。
3. Wave2：后半 getters 并发 get（此时应出现多源扩散）。
4. 记录每 wave 与集群吞吐。

### 4.3 参数调优矩阵

1. `tcp_conn_count`: 16, 20, 24
2. `buffers_per_flow`: 8, 16
3. `max_window_segments`: 8, 16, 24
4. `expected_gpu_channels`: 0, 8
5. `size_mib`: 1024, 2048, 8192, 32768

## 5. 观测指标与回归门槛

### 5.1 关键指标

1. `put_sec`, `e2e_sec`, `transfer_sec`, `visibility_wait_sec`
2. `comm_transfers_delta`, `comm_bytes_delta`, `comm_errors_delta`
3. `p2p_ratio`
4. `wave1/2 transfer_gibps` 与 `cluster_gibps`

### 5.2 回归门槛（Pass/Fail）

1. 功能门槛：cascade 停源链路持续成功。
2. 正确性门槛：`p2p_ratio=1.0` 且 `comm_errors_delta=0`。
3. 一致性门槛：`comm_bytes_delta == total_bytes`。
4. 扩散门槛：`wave2_over_wave1_transfer_ratio >= 1.0`。
5. VRAM 源门槛：`vram_exportable_failures` 与 `chain_source_match_failures` 为空。

## 6. 执行状态（截至 2026-02-22）

1. Phase A（4n cascade）已通过，`source-retire-mode=deregister` 下单次 `deregister(wait=true)` 即可完成退役。
2. Phase B 已完成 6n/8n/9n，多组 `1GiB/2GiB` 与大负载 `8GiB/32GiB`。
3. 新增标准套件：`examples/cross_host/run_multihost_benchmark_suite.sh`。
4. 严格 VRAM 源链路验证通过：`deregister_then_stop` 下 seed 停止后下一跳仍 `source=p2p`。
5. 关键框架修复已落地并回归通过（见第 10 节）。

## 7. 功能验证结果（Cascade）

Case：
1. `fanout_cascade_4n_deregister_chain_v3`
2. `fanout_cascade_4n_deregister_chain_v4_fix`
3. `fanout_cascade_4n_deregister_chain_v7_readyfix`
4. `fanout_cascade_9n_deregister_chain_v9_podip`
5. `fanout_cascade_3n_vram_source_after_seed_stop_v1`

结果：
1. 级联 hop 全部成功，`p2p_ratio=1.0`。
2. 退役后仍能继续从新源拉取，功能链路成立。
3. `comm_bytes_delta == total_bytes`，`comm_error=0`。
4. `deregister_then_stop` 严格模式也可通过。

结果目录：`/tmp/tc_cross_20260222/results_multi_host`

## 8. 扩容性能结果（Fanout）

| Case | getters | size(MiB) | conn/buf/maxw/egc | wave1 xfer (GiB/s) | wave2 xfer (GiB/s) | wave2/wave1 | cluster (GiB/s) | p2p_ratio |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `fanout_perf_6n_c20b16w16_g0_s1024_v1` | 5 | 1024 | 20/16/16/0 | 2.432 | 2.579 | 1.060 | 0.686 | 1.0 |
| `fanout_perf_8n_c20b16w16_g0_s1024_v1` | 7 | 1024 | 20/16/16/0 | 1.940 | 2.426 | 1.251 | 0.877 | 1.0 |
| `fanout_perf_8n_c24b12w24_g0_s1024_v1` | 7 | 1024 | 24/12/24/0 | 2.037 | 2.480 | 1.218 | 0.876 | 1.0 |
| `fanout_perf_8n_c20b16w16_g8_s1024_v1` | 7 | 1024 | 20/16/16/8 | 2.419 | 2.377 | 0.983 | 0.886 | 1.0 |
| `fanout_perf_8n_c20b16w16_g0_s2048_v1` | 7 | 2048 | 20/16/16/0 | 2.647 | 3.341 | 1.262 | 1.629 | 1.0 |
| `fanout_perf_9n_c20b16w16_g0_s1024_v1` | 8 | 1024 | 20/16/16/0 | 1.926 | 2.297 | 1.193 | 1.012 | 1.0 |
| `fanout_perf_9n_c24b12w24_g0_s1024_v1` | 8 | 1024 | 24/12/24/0 | 1.822 | 2.297 | 1.261 | 0.958 | 1.0 |
| `fanout_perf_9n_c20b16w16_g8_s1024_v1` | 8 | 1024 | 20/16/16/8 | 1.611 | 2.216 | 1.376 | 0.975 | 1.0 |
| `fanout_perf_9n_c20b16w16_g0_s2048_v1` | 8 | 2048 | 20/16/16/0 | 2.306 | 3.304 | 1.433 | 1.783 | 1.0 |
| `fanout_perf_9n_c20b16w16_g0_s8192_v3` | 8 | 8192 | 20/16/16/0 | 2.244 | 5.202 | 2.318 | 3.907 | 1.0 |
| `fanout_perf_9n_c20b16w16_g0_s32768_v4` | 8 | 32768 | 20/16/16/0 | 2.613 | 5.055 | 1.935 | 6.976 | 1.0 |

大负载结果文件：
1. `/tmp/tc_cross_20260222/results_multi_host/fanout_perf_9n_c20b16w16_g0_s8192_v3.json`
2. `/tmp/tc_cross_20260222/results_multi_host/fanout_perf_9n_c20b16w16_g0_s32768_v4.json`

启动/历史失败样本（用于定位，不计入最终通过集）：
1. `fanout_perf_8n_c24b16w24_g0_s1024_v1`：`comm_gpu` slices 不足导致 seed 启动失败。
2. `fanout_perf_9n_c20b16w16_g0_s32768_v3`：stable 预算重复计数导致 put 失败（后续修复为 v4 通过）。
3. `fanout_perf_9n_c20b16w16_g0_s8192_v2`：cleanup 早期参数缺陷（后续修复为 v3）。

## 9. Wave 口径与理论差距量化

`wave1/wave2` 语义：
1. Wave1 主要反映“单源 -> 多接收者”首波能力。
2. Wave2 反映“多源扩散”能力。
3. `wave2/wave1 > 1` 表示 fanout 生效。

量化口径（9n, `20/16/16/0`）：
1. `obs_cluster_ratio = wave2_cluster_gibps / wave1_cluster_gibps`
2. `obs_transfer_ratio = wave2_transfer_gibps_mean / wave1_transfer_gibps_mean`
3. `gap_to_2x = obs/2 - 1`
4. `gap_to_4x = obs/4 - 1`

| size | obs_cluster_ratio | obs_transfer_ratio | gap_to_2x(cluster) | gap_to_4x(cluster) |
|---|---:|---:|---:|---:|
| 1GiB (`s1024_v1`) | 1.185 | 1.193 | -40.8% | -70.4% |
| 2GiB (`s2048_v1`) | 1.516 | 1.433 | -24.2% | -62.1% |
| 8GiB (`s8192_v3`) | 1.998 | 2.318 | -0.1% | -50.1% |
| 32GiB (`s32768_v4`) | 2.231 | 1.935 | +11.6% | -44.2% |

结论：
1. 相对 2x 工程目标：8GiB 基本打平，32GiB 超过；1/2GiB 仍有差距。
2. 相对 4x 理想上限：仍有 `44%~70%` 差距，符合多机真实拓扑与控制面开销预期。

## 10. 推荐配置与执行顺序

推荐参数（当前基线）：
1. `conn/buffers/maxw/egc = 20/16/16/0`
2. `size_mib=1024/2048` 用于常规回归；`8192/32768` 用于大负载压测。
3. cascade 默认 `source-retire-mode=deregister`。

建议执行顺序：
1. 先跑 4n cascade 功能链路。
2. 再跑 6n/8n/9n fanout 扩容。
3. 最后跑 8GiB/32GiB 大负载并做 cleanup 回归。

## 11. 产出物

1. 统一文档（本文档）：`docs/benchmarks/20260221-multi-host-p2p-fanout-benchmark-report.md`
2. 标准化脚本：`examples/cross_host/cross_host_fanout_runner.py`
3. 一键套件：`examples/cross_host/run_multihost_benchmark_suite.sh`
4. 原始结果目录：`/tmp/tc_cross_20260222/results_multi_host`
