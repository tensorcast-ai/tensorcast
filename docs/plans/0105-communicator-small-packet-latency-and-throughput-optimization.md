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

围绕 `0105` 设计落地一套“非数据传输代价量化 + 定向优化”执行路径。第一目标不是直接追求单一带宽数字，而是把任务过程中的非数据面成本拆分清楚：哪些属于一次性可均摊成本，哪些是每次传输都要重复支付的成本；在此基础上再推进小包/中包优化，同时保持 `256 MiB+` 吞吐不回退。

# Current State & Grounding

## Baseline（2026-03-16）

来自 [20260316-communicator-vs-transfer-engine-comparison.md](../benchmarks/20260316-communicator-vs-transfer-engine-comparison.md)：

| Logical request size | Communicator | Transfer Engine | Gap |
| --- | --- | --- | --- |
| `32 MiB` | `18.56 GB/s` | `22.49 GB/s` | `-3.93 GB/s` (`-17.5%`) |
| `64 MiB` | `20.56 GB/s` | `22.75 GB/s` | `-2.19 GB/s` (`-9.6%`) |
| `256 MiB` | `23.17 GB/s` | `23.02 GB/s` | 同一性能等级 |
| `1 GiB` | `23.96 GB/s` | `23.69 GB/s` | 同一性能等级 |

## 已确认的代码侧瓶颈锚点

1. 请求控制面当前是单线程串行消费
   - 请求线程创建: [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - 循环处理入口 `do_read_request_loop()`: [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - 队列/在途表定义 `request_queue_`、`pending_requests_`: [core/communicator/engine/engine.h](../../core/communicator/engine/engine.h)
2. RDMA refill/ACK 推进节奏偏保守
   - `resume_rdma_reads()` 在 `made_progress` 后立即 break: [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - `ENGINE_OP_RDMA_READ_DONE_EX` 中 ACK 后再触发 refill: [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
3. RDMA 路径每 WR bookkeeping 粒度较细
   - `read_multi()` 每个 posted WR 都会 `enqueue_completion_bytes` 和 `per_qp_inflight_queues_.push(...)`: [core/communicator/transport/rdma_transport.cc](../../core/communicator/transport/rdma_transport.cc)
   - `do_process_wc()` 每个 completion 都 `pop` 并更新请求完成态: [core/communicator/transport/rdma_transport.cc](../../core/communicator/transport/rdma_transport.cc)
   - ACK/完成跟踪结构在 `ReadRequest`: [core/communicator/transport/request.h](../../core/communicator/transport/request.h)
4. MTCP staged 路径存在较重等待链
   - staging 线程串行消费 `mtcp_staging_queue_`: [core/communicator/engine/engine.cc](../../core/communicator/engine/engine.cc)
   - MTCP 发送/接收路径包含大量 `std::async` 与 future `wait/get`: [core/communicator/transport/mtcp_transport.cc](../../core/communicator/transport/mtcp_transport.cc)
5. 热路径 `LOG(INFO)` 频率偏高
   - 典型热点包括 `read_tensor` 入队、`resume_rdma_reads`、`read_multi` posted/completion 等路径（同上文件）。

## 约束与边界

1. 配置必须走统一 runtime config，不引入临时环境变量开关（见 [0004-unified-runtime-config.md](../designs/0004-unified-runtime-config.md) 与 [communicator_config.proto](../../proto/tensorcast/communicator/v1/communicator_config.proto)）。
2. `pending_requests_` 生命周期语义、错误传播语义、`ReadRequest::set_result()` 幂等行为不得破坏。
3. `FlowCreditLedger` 的 credit 上限与释放路径必须保持安全约束（[core/communicator/engine/staging_flow_controller.cc](../../core/communicator/engine/staging_flow_controller.cc)）。

# Phases & Milestones

- [ ] Phase 0: 基线冻结与观测口径统一
  - [ ] Milestone 0.1: 固化本轮优化对比口径（`32/64/256 MiB`, `threads=1`, `batch_size=1`，单 NIC strict direct RDMA）。
  - [ ] Milestone 0.2: 固化 benchmark 产出字段和解析规则（优先 `amortizable_*` 与 `recurring_*`，并保留 `bw_GBps`/`bw_gbps` 兼容）。
  - [ ] Milestone 0.3: 记录基线 case 元数据（host pair、GPU/NIC 对应、qp/outstanding 配置）。

- [ ] Phase 1: Instrumentation First（先观测后优化）
  - [ ] Milestone 1.1: 在 `ReadRequest` 增加 request 级统计快照容器（不改变现有完成语义）。
  - [ ] Milestone 1.2: 在 benchmark/engine/transport 关键路径补齐统计埋点（queue wait、control send、window/ack 轮次、WR posted/completion、MTCP 等待时间）。
  - [ ] Milestone 1.3: benchmark 输出提供 amortizable（init/warmup）与 recurring（clear/issue/wait/verify）字段，并支持 per-iteration/per-request 均摊视角。
  - [ ] Milestone 1.4: `verify` 开销拆分为内存分配、数据拷贝、校验计算三个子项，形成“可均摊/不可均摊”归因报表。

- [ ] Phase 2: 控制路径与窗口粒度优化
  - [ ] Milestone 2.1: 把单请求线程演进为“按 peer 分片 + 固定 worker 池”，默认 `worker_count=1` 保持旧行为。
  - [ ] Milestone 2.2: `resume_rdma_reads()` 引入 per-pass budget（window 或 bytes 预算），替代“有进展即等待 ACK”策略。
  - [ ] Milestone 2.3: ACK 批处理策略落地（最大窗口数 + 最大延迟约束），保证 credit 能及时回收。
  - [ ] Milestone 2.4: 完成功能回归与小流量灰度验证。

- [ ] Phase 3: RDMA/MTCP 数据面微优化
  - [ ] Milestone 3.1: RDMA inflight bookkeeping 从“每 WR 记录”收敛到“批记录 + remaining 计数”。
  - [ ] Milestone 3.2: 增加可控 signaled WR 间隔能力（默认兼容旧行为）。
  - [ ] Milestone 3.3: MTCP staged 路径减少 future 串行等待与主动 sleep 退避，向事件驱动推进。
  - [ ] Milestone 3.4: 热路径日志降噪完成（`INFO -> VLOG` 或采样）。

- [ ] Phase 4: 验收、灰度与默认值收敛
  - [ ] Milestone 4.1: 跑通完整验证矩阵（功能、性能、非退化、稳定性）。
  - [ ] Milestone 4.2: 达成验收阈值后形成推荐配置默认值。
  - [ ] Milestone 4.3: 完成文档同步（设计、计划、benchmark 结果页）。

# Tasks

## 模块任务拆分

| 模块 | 任务 | 关键文件 |
| --- | --- | --- |
| Engine 调度层 | 请求分片队列与 worker 池、RDMA refill budget、ACK 批处理触发策略 | [engine.cc](../../core/communicator/engine/engine.cc), [engine.h](../../core/communicator/engine/engine.h) |
| Flow 控制层 | credit 边界校验、预算推进与释放安全性回归 | [staging_flow_controller.cc](../../core/communicator/engine/staging_flow_controller.cc) |
| RDMA 传输层 | inflight 批化、signaled 间隔化、completion 处理路径适配 | [rdma_transport.cc](../../core/communicator/transport/rdma_transport.cc) |
| Request 状态层 | request 级统计聚合、ACK 队列批处理元数据 | [request.h](../../core/communicator/transport/request.h) |
| MTCP 传输层 | 去除高成本等待链、减少主动退避等待 | [mtcp_transport.cc](../../core/communicator/transport/mtcp_transport.cc) |
| 配置与协议 | 新增优化配置项并保持默认兼容 | [communicator_config.proto](../../proto/tensorcast/communicator/v1/communicator_config.proto) |
| 基准与文档 | benchmark 复测、结果归档、设计/计划同步更新 | [20260316 comparison](../benchmarks/20260316-communicator-vs-transfer-engine-comparison.md), [20260315 progress](../benchmarks/20260315-communicator-rdma-nic-gdr-progress.md) |

## 执行顺序约束

1. 必须先完成 Phase 1 观测再进入 Phase 2/3，避免“盲调”。
2. Phase 2 与 Phase 3 可分支并行，但每项优化都必须以 Phase 1 指标下降为前提。
3. 配置项先加字段再开策略，默认值必须保持旧行为。

# Test / Rollout / Backout

## Validation Matrix

| 维度 | 内容 | 通过标准 |
| --- | --- | --- |
| 单元/组件回归 | `//core/communicator:request_test`, `:rdma_engine_test`, `:staging_flow_controller_test`, `:mtcp_transport_lane_test`, `:mtcp_transfer_completion_tracker_test`, `:tcp_engine_test` | 全通过，无新增 flaky |
| 配置回归 | `//core/communicator:config_io_test` + 新增配置字段读写/默认值测试 | 新字段默认值等价旧行为 |
| 归因完整性 | `ITER`/`SUMMARY` 同时产出 amortizable + recurring 字段；`verify` 拆分 alloc/copy/checksum | 字段齐全，可稳定解析 |
| 小包结构性指标 | `32 MiB`、`64 MiB` strict direct RDMA 单请求对比 baseline | 能量化 recurring 主项并识别 top bottleneck |
| 大包非退化 | `256 MiB`、`1 GiB` 同口径复测 | 相对 baseline 回退不超过 `3%` |
| 成本分类 | `amortized_per_iteration_us`、`amortized_per_request_us` 与 recurring per-request 指标 | 可判断一次性成本是否可被后续传输均摊 |
| 稳定性 | 长时压测 + error path（ACK 延迟、部分 post 失败、MTCP fallback） | 无死锁/泄漏/静默失败 |

## 执行命令（示例）

1. C++ 回归：
   - `bazel test //core/communicator:request_test`
   - `bazel test //core/communicator:rdma_engine_test`
   - `bazel test //core/communicator:staging_flow_controller_test`
   - `bazel test //core/communicator:mtcp_transport_lane_test`
   - `bazel test //core/communicator:mtcp_transfer_completion_tracker_test`
   - `bazel test //core/communicator:tcp_engine_test`
   - `bazel test //core/communicator:config_io_test`
2. benchmark 工具链：
   - `tools/communicator/run_bench_case.py`
   - `tools/communicator/launch_remote_bench_case.py`
   - `tools/communicator/render_te_comparison_charts.py`

## Rollout

1. 第一步只上线观测埋点（策略默认关闭）。
2. 第二步灰度启用控制粒度优化（请求 worker、refill budget、ACK batch）。
3. 第三步灰度启用 RDMA/MTCP 微优化（bookkeeping/signaled/等待链优化）。
4. 第四步根据验收结果收敛推荐默认值并更新文档。

## Backout

1. 所有新策略均可通过配置项单独关闭并回退到旧行为。
2. 若出现 `256 MiB+` 回退或稳定性问题，优先回退 Phase 3，再回退 Phase 2。
3. 保留 Phase 1 观测能力，用于回退后的二次定位。

# Risks & Tracking

- [ ] 风险 1: 请求 worker 并行化引入顺序语义回归
  - 跟踪: 按 peer 分片保证顺序，补充顺序一致性测试。
- [ ] 风险 2: ACK 批处理导致 credit 回收变慢或堆积
  - 跟踪: 设置 `ack_batch_max_windows` 与 `ack_batch_max_delay_us` 双阈值，增加超时告警。
- [ ] 风险 3: unsignaled WR 间隔化影响错误可见性
  - 跟踪: 默认 `1`，灰度阶段逐步放大并监控 completion 异常率。
- [ ] 风险 4: MTCP 去阻塞化修改并发模型引入隐性死锁
  - 跟踪: 增加压力和超时失败路径测试，确保线程池不被阻塞等待。
- [ ] 风险 5: 热路径日志降噪影响问题定位
  - 跟踪: 保留采样日志与 debug 开关，确保关键错误仍保留 `WARNING/ERROR`。

# Owner Checklist

- [ ] 设计/计划双向链接已建立且可解析。
- [ ] 里程碑均为可验证结果，不是实现动作描述。
- [ ] 所有配置增量遵循统一 runtime config。
- [ ] 验收矩阵覆盖“性能提升 + 非退化 + 稳定性 + 可回滚”。
- [ ] benchmark 结果文档在每轮关键里程碑后同步更新。
