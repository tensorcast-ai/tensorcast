# 0083 Group-Aware Transport 多机复测设计（v3）

日期：2026-02-25（2026-02-26 修订，含 P0/P1 自动化）

参考：

1. `docs/designs/0083-group-aware-transport-scheduling.md`
2. `docs/benchmarks/20260226-0083-unified-chain-transport-experiment-report.md`
3. `docs/benchmarks/20260222-weight-publisher-multihost-p2p-report.md`

## 1. 本轮目标

本轮复测只回答三件事，并按同一口径输出：

1. 统一链路是否闭环：`Request -> enqueue -> dispatch -> lease -> complete(outcome)`。
2. `group-aware` 元数据是否真实落库并参与调度（不是“跑了 group 参数但 DB 里没有 group 字段”）。
3. 在一致口径下，`group=none` 与 `group=tp_version` 的 E2E、吞吐、扩散指标差异。

## 2. 上一轮经验与强制修正

### 2.1 Receiver 内存预检口径

`receiver preflight` 不再把瞬时窗口直接绑到 `max_concurrency`，统一使用独立参数：

- `--receiver-preflight-transient-overlap`（默认 `1`）

含义：统一链路当前默认按串行 apply 估算瞬时窗口；`max_concurrency` 变化不应改变预检口径。

### 2.2 Group hint 注入链路

group 元数据只通过 TP materialize 请求上下文注入，统一字段：

- `tc.transport.group.kind`
- `tc.transport.group.id`
- `tc.transport.group.total_parts`
- `tc.transport.group.part_id`
- `tc.transport.group.priority`
- `tc.transport.group.epoch`
- `tc.transport.request_id`

runner 对外只暴露：

- `--transport-group-mode {none,tp_version}`

其余字段自动推导：

- `group_kind = "tp_version"`
- `group_total_parts = receiver_count * tp_world_size`
- `group_priority = 0`
- `group_namespace = <case_name>-<run_tag>:receiver`

### 2.3 统计口径一致性

1. source 扩散必须按 `artifact_transports.replica_id` 统计，不能用 `source_node_id`。
2. retention 必须在 `retention_timeout_s` 窗口内轮询，不做单点瞬时判断。
3. `group-aware` 生效必须满足：
   - `grouped_transports > 0`
   - `group_id/group_kind/group_part_id/group_total_parts` 契约完整

### 2.4 P0 / P1 自动化策略

1. `P0`（早停）：
   - group case 默认启用：`--p0-early-stop`（默认 true）
   - 宽限窗口：`--p0-early-stop-grace-s`（默认 20）
   - 若在窗口内仍不满足 group 元数据门禁，runner 立即终止角色并标记失败。
2. `P1`（自动指标）：
   - runner 自动查询 `artifact_transports`（join `artifact_replicas.memory_size`）并产出吞吐/扩散指标；
   - 自动记录每个 transport 的 `bytes/duration/throughput_gib_s(bandwidth_gib_s)`；
   - 采样参数：`--throughput-sample-interval-s`（默认 1.0）与 `--throughput-max-samples`（默认 20000）。

## 3. 复测矩阵（最新）

### 3.1 TP8 40GB（主 A/B）

1. `A0-non-group-mc1`: `group=none`, `max_concurrency=1`
2. `A1-group-mc1`: `group=tp_version`, `max_concurrency=1`
3. `B0-non-group-mc2`: `group=none`, `max_concurrency=2`
4. `B1-group-mc2`: `group=tp_version`, `max_concurrency=2`

统一参数：

- `tp_world_size=8`
- `num_versions=4`
- `keep_last=2`
- `receiver_preflight_transient_overlap=1`

### 3.2 TP8 320GB（容量边界）

1. `C0-non-group-mc1`
2. `C1-group-mc1`

说明：320GB 先固定 `mc=1` 做元数据接通与稳定性验证，`mc=2` 另开容量专项。

## 4. 执行方式（runner）

### 4.1 Non-group

```bash
python examples/cross_host/cross_host_weight_publisher_runner.py \
  --case-name <case> \
  ... \
  --max-concurrency <1|2> \
  --receiver-preflight-transient-overlap 1 \
  --transport-group-mode none \
  --throughput-sample-interval-s 1.0
```

### 4.2 Group

```bash
python examples/cross_host/cross_host_weight_publisher_runner.py \
  --case-name <case> \
  ... \
  --max-concurrency <1|2> \
  --receiver-preflight-transient-overlap 1 \
  --transport-group-mode tp_version \
  --p0-early-stop \
  --p0-early-stop-grace-s 20 \
  --throughput-sample-interval-s 1.0
```

## 5. 门禁（与 runner 状态字段 1:1 对齐）

### 5.0 P0 早停门禁（group case）

读取 `summary.transport_checks.p0_early_stop`：

1. `enabled = true`
2. `triggered = false`
3. `reasons = []`

### 5.1 功能门禁（group-aware 接通）

读取 `summary.transport_checks.group_metadata_probe`：

1. `window_has_transports = true`
2. `requester_tagged_complete = true`
3. group case：`group_mode_consistent = true`（即 `grouped_transports > 0`）
4. non-group case：`group_mode_consistent = true`（即 `grouped_transports = 0`）
5. group case：`group_contract_consistent = true`

### 5.2 稳定性门禁

读取 `summary.stability`：

1. `all_receivers_completed = true`
2. `binding_pointer_stable = true`
3. `receiver_mode_consistent = true`
4. `publish_to_apply_within_limit = true`

### 5.3 retention 门禁

读取 `summary.retention_checks.global_store_probe`：

1. `old_versions_released = true`
2. `replica_versions_within_window = true`
3. `replica_version_count_within_limit = true`

### 5.4 量化输出（必产）

1. E2E：
   - `summary.performance.publish_to_apply_s.p95`
   - `summary.performance.apply_latency_s.mean`
2. 吞吐（按 transport 采样）：
   - `summary.performance.transport_throughput_gib_s.peak_active_throughput_gib_s`
   - `summary.performance.transport_throughput_gib_s.p95_active_throughput_gib_s`
   - `summary.performance.transport_throughput_gib_s.mean_active_throughput_gib_s`
   - `summary.performance.transport_throughput_gib_s.active_transport_peak`
   - `summary.performance.transport_throughput_gib_s.active_transport_mean`
3. 扩散（`replica_id` 口径）：
   - `summary.distribution.transport_diffusion.top1_share`
   - `summary.distribution.transport_diffusion.hhi`
   - `summary.distribution.transport_diffusion.unique_sources`
4. 每 transport 带宽记录：
   - `transport_metrics.per_transport_records[*].bytes`
   - `transport_metrics.per_transport_records[*].duration_s`
   - `transport_metrics.per_transport_records[*].throughput_gib_s`
   - `transport_metrics.per_transport_records[*].bandwidth_gib_s`

## 6. 状态一致性约束（必须满足）

本轮所有结论仅以 runner 输出 JSON 为准；人工 SQL 仅用于复核，不作为主结论来源。

对应关系：

1. 文档“group 接通”结论 -> `summary.transport_checks.group_metadata_probe.*`
2. 文档“P0 早停”结论 -> `summary.transport_checks.p0_early_stop.*`
3. 文档“稳定性”结论 -> `summary.stability.*`
4. 文档“retention”结论 -> `summary.retention_checks.*`
5. 文档“吞吐/扩散”结论 -> `summary.performance.transport_throughput_gib_s.*` 与 `summary.distribution.transport_diffusion.*`
6. 文档“最终 pass/fail” -> `summary.passed`

若任一子门禁失败，`summary.passed` 必须为 `false`，并阻断该 case 进入对比汇总。

## 7. 建议执行顺序

1. 先跑 `A0` 与 `A1` 验证 group 元数据接通。
2. 再跑 `B0` 与 `B1` 做并发对比。
3. 最后跑 `C0` 与 `C1` 做容量边界验证。
4. 每个 case 完成后立即检查 `summary.transport_checks.group_metadata_probe`，避免带病继续跑后续矩阵。
5. 若 `summary.transport_checks.p0_early_stop.triggered=true`，必须先修链路再进入后续 case。

## 8. 风险与回退

1. 若 `group_mode=tp_version` 但 `grouped_transports=0`：立即判失败，先修链路再重跑，不做收益结论。
2. 若 `group_probe_error` 非空：优先修复 GS 状态/DB 可见性问题，再执行 A/B。
3. 若出现资源等待（如调度 preemption）：保持充足 `--max-wait-duration`，不要用短等待窗口误判超时。
4. 远端和本地产物统一放在 `/data/...`，避免 `/tmp` 被清理导致证据缺失。
