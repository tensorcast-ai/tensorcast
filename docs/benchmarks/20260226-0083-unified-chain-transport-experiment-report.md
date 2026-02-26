# 0083 Unified Request Chain 实验方案修订（待复测，P0/P1 已接入）

日期：2026-02-26

## 1. 状态说明

本次仅做**实验方案与口径修正**，不重跑整体测试，不生成新一轮结果表。

目标是把后续复测方案调整到可直接验证以下三件事：

1. 0083 统一链路有效（`Request -> enqueue -> global dispatch -> claim+create lease -> complete(outcome)`）。
2. group transport 真正接通，并能做 group / non-group A/B。
3. 指标口径一致，能正确评估 replica 扩散与 retention 一致性。

## 2. 已修正的口径与执行前置

### 2.1 Receiver 内存预检口径

修正点：receiver preflight 不再直接按 `max_concurrency` 放大瞬时窗口，改为独立参数：

- `--receiver-preflight-transient-overlap`（默认 `1`）

含义：在统一链路当前实现下，默认按串行 apply 估算瞬时窗口；因此 `max_concurrency=1/2` 在默认预检下应是同口径。

### 2.2 Group hint 注入路径（实验开关）

修正点：增加 receiver 侧可控 group hint 注入能力（通过 TP materialize 请求上下文）。

Runner 对外仅保留：

- `--transport-group-mode {none,tp_version}`

其余 group 字段由现有参数自动推导并固定：

- `group_kind = "tp_version"`（仅在 group 模式）
- `group_total_parts = receiver_count * tp_world_size`
- `group_priority = 0`
- `group_namespace = <case_name>-<run_tag>:receiver`

建议：

- non-group 基线：`--transport-group-mode none`
- group 实验：`--transport-group-mode tp_version`
- runner 自动输出 `summary.transport_checks.group_metadata_probe`，用于门禁判定 `grouped_transports/requester_tagged/group_contract` 一致性。

### 2.3 Retention 审计一致性

修正点：`old_versions_released` 不再单次瞬时判断，改为在 `retention_timeout_s` 窗口内轮询 GS replica_count 收敛。

### 2.4 P0 / P1 自动化状态

已接入 runner 自动化能力：

1. `P0` 早停：
   - 参数：`--p0-early-stop`（默认 true）、`--p0-early-stop-grace-s`（默认 20）；
   - 产物：`summary.transport_checks.p0_early_stop.*`；
   - 行为：group case 若探针窗口后仍不满足 group 元数据门禁则立即终止角色并失败。
2. `P1` 自动指标：
   - 参数：`--throughput-sample-interval-s`（默认 1.0）、`--throughput-max-samples`（默认 20000）；
   - 产物：
     - `summary.performance.transport_throughput_gib_s.*`
     - `summary.distribution.transport_diffusion.*`
     - `transport_metrics.per_transport_records[*]`（每 transport 带宽明细）。

## 3. 正确指标口径（统一）

### 3.1 Replica 扩散（必须用 `replica_id`）

`source_node_id` 是请求端字段，不能用于 source 扩散统计。

扩散指标统一改为基于 `artifact_transports.replica_id`：

1. `source_top1_share = max(cnt(replica_id))/sum(cnt(replica_id))`
2. `source_hhi = sum((cnt_i/total)^2)`
3. `unique_sources = count(distinct replica_id)`

### 3.2 Group 接通判据

仅当以下条件满足，才可宣称 group-aware 生效：

1. `grouped_transports > 0`
2. `group_id/group_kind/group_part_id/group_total_parts` 非空且契约一致

### 3.3 集群采样实时吞吐

保持以下定义：

1. `rate_i = bytes_i / (completed_at_i - created_at_i)`
2. `cluster_sampled_throughput(t) = Σ rate_i`（所有 in-flight transport）

输出：`peak/p95/mean_active_throughput` + `active_transport_peak/mean`。

### 3.4 Publisher 发布带宽（新增，A0 起强制记录）

针对每个发布版本，统一记录两类发布带宽：

1. `publish_throughput_gib_s = publish_payload_bytes / publish_latency_s`
2. `put_throughput_gib_s = publish_payload_bytes / publish_breakdown_s.put_s`

说明：

1. `publish_payload_bytes` 由发布张量逻辑字节数统计得到；
2. `publish_throughput_gib_s` 反映发布端端到端 publish 阶段吞吐；
3. `put_throughput_gib_s` 更贴近 daemon 注册写入阶段吞吐。

## 4. 复测矩阵（新方案）

## 4.1 TP8 40GB（主 A/B）

1. `A0-non-group-mc1`: group=none, mc=1
2. `A1-group-mc1`: group=tp_version, mc=1
3. `B0-non-group-mc2`: group=none, mc=2
4. `B1-group-mc2`: group=tp_version, mc=2

统一参数建议：

- `tp_world_size=8`
- `num_versions=4`
- `keep_last=2`
- `receiver_preflight_transient_overlap=1`
- `group_total_parts` 由 runner 自动推导为 `receiver_count * tp_world_size`

### 4.2 TP8 320GB（容量边界验证）

1. `C0-non-group-mc1`
2. `C1-group-mc1`

注：320GB 先固定 `mc=1` 完成 group 接通与口径验证；`mc=2` 仅作为后续容量实验项。

## 5. 执行命令模板（Runner）

### 5.1 Non-group

```bash
python examples/cross_host/cross_host_weight_publisher_runner.py \
  --case-name <case> \
  --publisher-daemon-config examples/config/store_daemon_config_cross_host_bench.yaml \
  ... \
  --max-concurrency <1|2> \
  --receiver-preflight-transient-overlap 1 \
  --transport-group-mode none
```

### 5.2 Group

```bash
python examples/cross_host/cross_host_weight_publisher_runner.py \
  --case-name <case> \
  --publisher-daemon-config examples/config/store_daemon_config_cross_host_bench.yaml \
  ... \
  --max-concurrency <1|2> \
  --receiver-preflight-transient-overlap 1 \
  --transport-group-mode tp_version
```

## 6. 验收门禁（复测后执行）

1. 功能门禁：
   - group case：`summary.transport_checks.group_metadata_probe.group_mode_consistent=true`
   - non-group case：`summary.transport_checks.group_metadata_probe.group_mode_consistent=true`
   - 两类 case 都要求 `requester_tagged_complete=true` 且 `group_contract_consistent=true`
2. P0 门禁：
   - `summary.transport_checks.p0_early_stop.enabled=true`（group case）
   - `summary.transport_checks.p0_early_stop.triggered=false`
   - `summary.transport_checks.p0_early_stop.reasons=[]`
3. 稳定性门禁：
   - `all_receivers_completed=true`
   - `binding_pointer_stable=true`
4. retention 门禁：
   - `old_versions_released=true`（轮询窗口内达成）
5. 量化门禁：
   - E2E：`summary.performance.publish_to_apply_s.p95`、`summary.performance.apply_latency_s.mean`
   - 吞吐：`summary.performance.transport_throughput_gib_s.*`
   - 扩散：`summary.distribution.transport_diffusion.*`
   - 每 transport 带宽：`transport_metrics.per_transport_records[*].throughput_gib_s`

## 7. 输出物清单（复测后）

1. case summary JSON（runner 输出）
   - 包含 `summary.performance.publish_bandwidth_gib_s.*`
   - 包含 `summary.performance.put_bandwidth_gib_s.*`
   - 包含每版本 `publisher_summary.published[*].publish_payload_bytes/publish_throughput_gib_s/put_throughput_gib_s`
2. load-balance SQL 回放表（使用 `replica_id` 口径）
3. cluster sampled throughput 时序与汇总（`transport_metrics.throughput.series`）
4. group vs non-group 对比汇总表（TP8 40GB + TP8 320GB）

## 8. 备注

旧版文档中的历史结果可作为背景参考，但不再用于当前结论。当前结论需以本方案复测结果为准。

## 9. Smoke 测试已通过
