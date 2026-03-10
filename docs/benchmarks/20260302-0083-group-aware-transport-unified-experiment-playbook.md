# 0083 Group-Aware Transport 实验总览（Why / What / How / Results / Analysis）

日期：2026-03-02
状态：Active（0083 实验设计与结果的唯一维护入口）

## 1. Why（为什么做）

0083 的核心问题不是“能不能跑通一次”，而是回答以下可验证问题：

1. 统一链路是否稳定闭环：`Request -> enqueue -> dispatch -> lease -> complete(outcome)`。
2. group-aware 是否真实参与调度，而非只在接口上传参。
3. 在统一口径下，group 与 non-group 的性能差异是否可复现。
4. 在多个 group 并发竞争时，group-aware 是否能系统性改善 tail 与公平性。

## 2. What（做什么实验）

实验拆成两条轨道，分别回答“基础可靠性”与“并发收益”。

## 2.1 Track A：基线链路 + 容量边界

目标：验证链路闭环、group 接通、稳定性、容量行为。

矩阵：

1. `A0-non-group-mc1`: TP8, 40GiB, `group=none`, `max_concurrency=1`
2. `A1-group-mc1`: TP8, 40GiB, `group=tp_version`, `max_concurrency=1`
3. `B0-non-group-mc2`: TP8, 40GiB, `group=none`, `max_concurrency=2`
4. `B1-group-mc2`: TP8, 40GiB, `group=tp_version`, `max_concurrency=2`
5. `C0-non-group-mc1`: TP8, 320GiB, `group=none`, `max_concurrency=1`
6. `C1-group-mc1`: TP8, 320GiB, `group=tp_version`, `max_concurrency=1`

固定参数：

1. `receiver_preflight_transient_overlap=1`
2. `tp_world_size=8`
3. `num_versions=4`
4. `keep_last=2`
5. 320GiB 强制 `--receiver-apply-mode tp_bind_into_swap`

## 2.2 Track B：并发 group 收益

目标：验证多个 group 同时竞争时的 tail/fairness 改善。

矩阵：

1. Stage 0（门禁）
   - `S0-N`: `K=1`, `TP=2`, `16GiB`, `group=none`
   - `S0-G`: `K=1`, `TP=2`, `16GiB`, `group=tp_version`
2. Stage 1（主实验）
   - `M1-N`: `K=3`, `TP=2`, `16GiB`, `group=none`
   - `M1-G`: `K=3`, `TP=2`, `16GiB`, `group=tp_version`
   - `M2-N`: `K=3`, `TP=2`, `24GiB`, `group=none`
   - `M2-G`: `K=3`, `TP=2`, `24GiB`, `group=tp_version`
3. Stage 2（可选增强）
   - `E1-N`: `K=2`, `TP=4`, `32GiB`, `group=none`
   - `E1-G`: `K=2`, `TP=4`, `32GiB`, `group=tp_version`

固定参数：

1. `num_versions=3`
2. `keep_last=1`
3. `publish_interval_s=180`
4. `receiver_timeout_s=1800`
5. `tp_materialize_deadline_s=1800`
6. `max_publish_to_apply_s=1800`
7. `receiver_preflight_transient_overlap=1`
8. receiver 固定 `--receiver-apply-mode tp_bind_into_swap`
9. 每个 `(N,G)` 配对按 `ABBA` 执行，主实验至少 2 轮

## 3. How（怎么做与怎么判）

## 3.1 统一观测字段

group 元数据统一检查：

1. `tc.transport.group.kind`
2. `tc.transport.group.id`
3. `tc.transport.group.total_parts`
4. `tc.transport.group.part_id`
5. `tc.transport.group.priority`
6. `tc.transport.group.epoch`
7. `tc.transport.request_id`

## 3.2 门禁规则（统一）

结论仅以 case summary JSON 为准，必须满足：

1. `summary.transport_checks.group_metadata_probe.window_has_transports=true`
2. `summary.transport_checks.group_metadata_probe.requester_tagged_complete=true`
3. `summary.transport_checks.group_metadata_probe.group_mode_consistent=true`
4. `summary.transport_checks.group_metadata_probe.group_contract_consistent=true`
5. `summary.stability.all_receivers_completed=true`
6. `summary.stability.binding_pointer_stable=true`
7. `summary.retention_checks.global_store_probe.old_versions_released=true`
8. `summary.passed=true`

补充约束：

1. group case 必须 `grouped_transports > 0`
2. non-group case 必须 `grouped_transports = 0`

## 3.3 指标定义

基础指标：

1. E2E：`publish_to_apply_s.p95`、`apply_latency_s.mean`
2. 吞吐：`peak/p95/mean_active_throughput_gib_s`
3. 扩散（必须按 `artifact_transports.replica_id`）：`top1_share`、`hhi`、`unique_sources`
4. 发布侧：`publish_bandwidth_gib_s.*`、`put_bandwidth_gib_s.*`

并发收益指标（Track B）：

1. `group_completion_p95_s`
2. `publish_to_apply_p95_s`
3. 公平性：`completion_span_s`、`completion_cv`
4. 连续性：`create_gap_mean/p95`

## 3.4 统计判定（Track B）

按 run-order 做 paired 对照，使用中位数差值 + `95% bootstrap CI`。

宣称“group 收益成立”需同时满足：

1. `group_completion_p95_s` 改善 `>=20%`
2. `publish_to_apply_p95_s` 改善 `>=20%`
3. 两项 CI 均不跨 0
4. `mean_active_throughput` 回退不超过 `5%`

## 4. Results（当前实验结果）

记录规则：若同一 case 存在“修复前/修复后”两版数据，本节只保留修复后的最终有效数据与分析。

当前已完成结果：`A0-non-group-mc1`（2026-02-26）

有效产物：
`/data/tc_cross_20260226/results_weight_publisher_0083_unified_chain/A0-non-group-mc1-1772120803/A0-non-group-mc1.json`

关键结果：

| 指标 | 数值 |
|---|---:|
| `summary.passed` | `true` |
| `summary.performance.publish_latency_s.mean` | `12.104s` |
| `summary.performance.publish_bandwidth_gib_s.mean` | `3.319 GiB/s` |
| `summary.performance.put_bandwidth_gib_s.mean` | `4.216 GiB/s` |
| `summary.performance.apply_latency_s.mean` | `15.132s` |
| `summary.performance.publish_to_apply_s.p95` | `19.624s` |
| `summary.performance.transport_throughput_gib_s.peak_active_throughput_gib_s` | `4.515 GiB/s` |
| `summary.performance.transport_throughput_gib_s.mean_active_throughput_gib_s` | `4.058 GiB/s` |
| `summary.distribution.transport_diffusion.top1_share` | `0.25` |
| `summary.distribution.transport_diffusion.hhi` | `0.25` |
| `summary.distribution.transport_diffusion.unique_sources` | `4` |
| `summary.transport_checks.group_metadata_probe.grouped_transports` | `0`（non-group 预期） |

## 5. Analysis（对结果的深入分析）

## 5.1 链路与门禁层面

1. `summary.passed=true` 且门禁字段齐全，说明 A0 在 non-group 基线下完成了“链路闭环 + 稳定性 + retention”验证。
2. `grouped_transports=0` 与 case 设定一致，说明对照组语义干净，可作为后续 group case 的比较基线。

## 5.2 性能结构层面

1. 发布与应用的平均时延差为 `3.028s`（`15.132 - 12.104`），表明 receiver 侧仍存在可观处理开销。
2. `publish_to_apply_p95=19.624s` 相对 `apply_latency.mean=15.132s` 高 `29.7%`，存在可见 tail，但尚不属于离散异常级别。
3. `transport.mean_active=4.058 GiB/s`、`transport.peak_active=4.515 GiB/s`，峰均比 `1.113`，传输阶段波动中等，未出现明显抖动失控。
4. `publish_bandwidth.mean=3.319 GiB/s` 低于 `put_bandwidth.mean=4.216 GiB/s`（比值 `0.787`），说明 publish 端端到端阶段开销大于 put 子阶段，系统瓶颈不在纯 put 写入。

## 5.3 负载扩散层面

1. `top1_share=0.25`、`hhi=0.25`、`unique_sources=4` 组合呈现近似均匀扩散。
2. 在该基线样本中，source 侧不存在明显热点倾斜，为后续 group/non-group 差异分析提供了稳定底噪背景。

## 5.4 当前可下结论与不可下结论

可下结论：

1. non-group 基线已形成可复用基准点。
2. 指标口径、门禁、产物结构可支撑后续严格 A/B。

不可下结论：

1. 尚不能判断 group-aware 是否带来收益（缺少 `A1/B1/C1` 与 Track B 的并发样本）。
2. 尚不能判断收益是否具备统计显著性（缺少 ABBA 重复与 bootstrap CI）。

## 6. 输出要求（后续批次）

每个 case：

1. role JSON（publisher + receiver/cohort）。
2. case summary JSON（含门禁、吞吐、扩散、retention/stability）。
3. per-transport 记录（`bytes/duration/throughput`）。

每个批次：

1. group/non-group 对照汇总表。
2. ABBA 配对统计表（每轮差值 + CI）。
3. 结论页（Track A 门禁是否通过、Track B 收益是否成立）。

## 7. 维护规则

1. 0083 的设计、结果与分析只维护本文件。
2. 历史文档只保留归档索引，不再承载结论。
3. 同一实验若有修复前与修复后数据，正文仅保留修复后版本；修复前数据不进入结论与分析区。
