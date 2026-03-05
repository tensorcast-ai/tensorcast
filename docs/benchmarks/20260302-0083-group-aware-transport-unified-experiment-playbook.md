# 0083 Group-Aware Transport 统一实验手册（Steady 口径）

日期：2026-03-04  
状态：Active（本文件为 0083 实验方法、最终数据与分析的唯一维护入口）

## 1. 实验目标

本实验回答三个问题：

1. 在统一口径下，`tp_bind_into_swap` 的 steady 路径是否稳定（`keep_last=2`，`publish_interval=40~60s`）。
2. `group=none` 与 `group=tp_version` 的 group contract 是否严格成立。
3. 在 ABBA 多轮执行下，最终可复用的性能与 timeout 画像是什么。

## 2. 实验方法（唯一执行口径）

## 2.1 执行入口

1. 主流程口径：`docs/benchmarks/20260302-0083-group-aware-transport-unified-experiment-playbook.md`（本文件）
2. 命令入口：`examples/cross_host/README.md` 第 2.5 节
3. 执行脚本：`examples/cross_host/run_multihost_weight_publisher_suite.sh`

## 2.2 固定配置

| 维度 | 值 |
|---|---|
| 拓扑 | 1 publisher + 7 receivers（r7） |
| payload | `tp_ranked` |
| TP | `tp_world_size=4` |
| 单版本数据量 | `tp_total_bytes=42949672960`（40GiB） |
| receiver apply | `tp_bind_into_swap` |
| group 对照 | `none` vs `tp_version` |
| 执行顺序 | `ABBA` |
| 轮次要求 | `>=3` |
| keep_last | `2` |
| publish interval | `60s`（处于 40~60s 口径内） |
| publisher daemon config | `examples/config/store_daemon_config_cross_host_bench_stable120.yaml` |
| receiver daemon config | `examples/config/store_daemon_config_cross_host_bench_stable80.yaml` |
| Global Store | `100.97.246.95:50051` |

本次最终样本的轮次版本配置为：`round1=10 versions`、`round2=10 versions`、`round3=4 versions`。

## 2.3 关键环境变量

```bash
export TC_WP_SCALE_RECEIVER_COUNTS=7
export TC_WP_TRANSPORT_GROUP_MODES=none,tp_version
export TC_WP_GROUP_PAIR_ORDER=abba
export TC_WP_GROUP_PAIR_ROUNDS=3

export TC_WP_PAYLOAD_MODE=tp_ranked
export TC_WP_TP_WORLD_SIZE=4
export TC_WP_TP_TOTAL_BYTES=42949672960
export TC_WP_RECEIVER_APPLY_MODE=tp_bind_into_swap
export TC_WP_KEEP_LAST=2
export TC_WP_PUBLISH_INTERVAL_S=60
export TC_WP_PRE_PUBLISH_TRIM_MARGIN=0
```

## 2.4 轮次测试内容定义（ABBA）

公共定义：

1. 每版本 artifact：`40GiB`（`tp_total_bytes=42949672960`）
2. TP 分片：`tp_world_size=4`，等价 `10GiB/rank`
3. 每轮固定 4 个 case，顺序：`o1=none`、`o2=tp_version`、`o3=tp_version`、`o4=none`

| round | order | mode | versions | artifact/版本 | artifact/TP rank | 单 case artifact 总量 |
|---:|---:|---|---:|---:|---:|---:|
| 1 | o1 | none | 10 | 40GiB | 10GiB | 400GiB |
| 1 | o2 | tp_version | 10 | 40GiB | 10GiB | 400GiB |
| 1 | o3 | tp_version | 10 | 40GiB | 10GiB | 400GiB |
| 1 | o4 | none | 10 | 40GiB | 10GiB | 400GiB |
| 2 | o1 | none | 10 | 40GiB | 10GiB | 400GiB |
| 2 | o2 | tp_version | 10 | 40GiB | 10GiB | 400GiB |
| 2 | o3 | tp_version | 10 | 40GiB | 10GiB | 400GiB |
| 2 | o4 | none | 10 | 40GiB | 10GiB | 400GiB |
| 3 | o1 | none | 4 | 40GiB | 10GiB | 160GiB |
| 3 | o2 | tp_version | 4 | 40GiB | 10GiB | 160GiB |
| 3 | o3 | tp_version | 4 | 40GiB | 10GiB | 160GiB |
| 3 | o4 | none | 4 | 40GiB | 10GiB | 160GiB |

按轮汇总：

1. round1 总 artifact 量：`1600GiB`
2. round2 总 artifact 量：`1600GiB`
3. round3 总 artifact 量：`640GiB`
4. 全部 3 轮总 artifact 量：`3840GiB`（`3.75TiB`）

## 2.5 数据纳入规则

仅将满足以下条件的 case 纳入最终报告：

1. 有完整 case JSON（`summary` 字段齐全）。
2. `summary.passed=true`。
3. `group_metadata_probe.group_mode_consistent=true`。
4. `group_metadata_probe.group_contract_consistent=true`。
5. `summary.stability.all_receivers_completed=true`。

排除项：

1. 无完整结果文件的 case（目录存在但无 case JSON）。
2. `single_host_functional_disabled` 等非目标 case。

## 3. 最终数据报告

## 3.1 数据来源

1. round1/round2（10 版本）：
   - `/data/tc_cross_20260304/results_weight_publisher_0083/rw03041440_r7_steady_k2_pi60_margin0_abba3_rerun3`
2. round3 增量（4 版本）：
   - `/data/tc_cross_20260304/results_weight_publisher_0083/rw03041740_incremental_round3_v4`
3. 汇总产物：
   - `/data/tc_cross_20260304/results_weight_publisher_0083/20260304-0083-steady-k2-pi60-incremental-v4-report.md`
   - `/data/tc_cross_20260304/results_weight_publisher_0083/20260304-0083-steady-k2-pi60-incremental-v4-case-metrics.json`
   - `/data/tc_cross_20260304/results_weight_publisher_0083/20260304-0083-steady-k2-pi60-incremental-v4-aggregate.json`

## 3.2 样本规模

- 最终纳入：12 个 case（8 个 10-version + 4 个 4-version）
- 通过率：`12/12`
- 发布版本总数：96
- artifact 总量（publisher 口径）：`3840GiB`（`3.75TiB`）
- apply 样本总数：672

## 3.3 聚合结果

| 维度 | case数 | 版本总数 | publish_mean_s | apply_mean_s | publish_to_apply_mean_s | waiting_timeout_cases | transport_timeout_cases | grouped/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 12 | 96 | 20.700 | 92.886 | 52.819 | 12 | 2 | 1344/2690 |
| bind=none | 6 | 48 | 20.705 | 92.745 | 52.792 | 6 | 2 | 0/1346 |
| bind=tp_version | 6 | 48 | 20.694 | 93.028 | 52.846 | 6 | 0 | 1344/1344 |
| versions=10 | 8 | 80 | 20.620 | 92.655 | 52.639 | 8 | 1 | 1120/2241 |
| versions=4 | 4 | 16 | 21.098 | 94.041 | 53.722 | 4 | 1 | 224/449 |

## 3.4 Case 明细（最终纳入口径）

| round | order | mode | versions | passed | publish_mean_s | apply_mean_s | publish_to_apply_p95_s | grouped/total | transport_timeout_reasons |
|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| 1 | o1 | none | 10 | true | 20.676 | 93.107 | 65.173 | 0/280 | {} |
| 1 | o2 | tp_version | 10 | true | 21.045 | 93.089 | 59.646 | 280/280 | {} |
| 1 | o3 | tp_version | 10 | true | 20.584 | 92.209 | 62.554 | 280/280 | {} |
| 1 | o4 | none | 10 | true | 20.927 | 93.252 | 109.198 | 0/281 | {"deadline_exceeded":1} |
| 2 | o1 | none | 10 | true | 20.214 | 91.765 | 60.314 | 0/280 | {} |
| 2 | o2 | tp_version | 10 | true | 20.338 | 92.455 | 64.776 | 280/280 | {} |
| 2 | o3 | tp_version | 10 | true | 20.265 | 92.972 | 66.173 | 280/280 | {} |
| 2 | o4 | none | 10 | true | 20.911 | 92.394 | 59.466 | 0/280 | {} |
| 3 | o1 | none | 4 | true | 18.574 | 90.913 | 62.332 | 0/112 | {} |
| 3 | o2 | tp_version | 4 | true | 21.418 | 94.867 | 61.944 | 112/112 | {} |
| 3 | o3 | tp_version | 4 | true | 21.330 | 94.657 | 61.225 | 112/112 | {} |
| 3 | o4 | none | 4 | true | 23.069 | 95.727 | 93.359 | 0/113 | {"deadline_exceeded":1} |

## 3.5 Timeout 汇总

1. waiting timeout：`12/12` case 观测到
   - `queue_or_visibility_wait=84`
   - `version_window_evicted=1`
2. transport timeout：`2/12` case 观测到
   - `deadline_exceeded=2`

## 3.5.1 Timeout 设置方式（脚本口径）

在 `examples/cross_host/run_multihost_weight_publisher_suite.sh` 中通过以下变量控制：

1. `TC_RECEIVER_TIMEOUT_S`
2. `TC_WP_RECEIVER_TIMEOUT_AUTO_ADJUST`
3. `TC_MAX_PUBLISH_TO_APPLY_S`
4. `TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST`
5. `TC_REMOTE_TIMEOUT_SEC`
6. `TC_WP_P0_EARLY_STOP`
7. `TC_WP_P0_EARLY_STOP_GRACE_S`
8. `TC_PROGRESS_POLL_S`
9. `TC_POLL_INTERVAL_S`

本轮实际生效（来自 case params）：

1. `receiver_timeout_s=287`
2. `max_publish_to_apply_s=173/177`（none/tp_version）
3. `remote_timeout_sec=2466`
4. `progress_poll_s=10`
5. `poll_interval_s=0.5`

## 3.5.2 下次测试建议（降低 timeout 干扰，提升统计准确性）

推荐采用“采数稳定配置”（直接在套件前 `export`）：

```bash
export TC_PUBLISH_INTERVAL_S=60

export TC_RECEIVER_TIMEOUT_S=360
export TC_WP_RECEIVER_TIMEOUT_AUTO_ADJUST=0

export TC_MAX_PUBLISH_TO_APPLY_S=210
export TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST=0

export TC_REMOTE_TIMEOUT_SEC=3000

export TC_WP_P0_EARLY_STOP=0
export TC_WP_P0_EARLY_STOP_GRACE_S=30

export TC_PROGRESS_POLL_S=10
export TC_POLL_INTERVAL_S=0.5
```

补充（直接调用 runner 时）：

1. `--tp-materialize-deadline-s 900`（套件脚本当前未暴露该参数，runner 默认 `600`）。

建议依据：

1. 本轮 `deadline_exceeded` 仅 2 次，集中在尾部 case，属于小概率超时尾部；
2. 提高 `receiver/max_publish_to_apply/remote` 三类预算可降低“预算触顶”引入的非业务抖动；
3. 关闭 `P0_EARLY_STOP` 可避免早停机制对正式采数的干扰（保留在 smoke/预检阶段即可）。

## 3.6 吞吐/带宽/扩散聚合（按 mode）

| mode | publish_bw_mean_gib_s | put_bw_mean_gib_s | transport_mean_active_gib_s | transport_p95_active_gib_s | transport_peak_active_gib_s | diffusion_top1_share_mean | diffusion_hhi_mean | diffusion_unique_sources_mean | grouped_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 1.971 | 2.534 | 17.243 | 21.046 | 24.151 | 0.150 | 0.145 | 12.167 | 0.000 |
| tp_version | 1.975 | 2.528 | 17.591 | 20.435 | 21.385 | 0.150 | 0.150 | 8.000 | 1.000 |

## 3.7 ABBA 逐轮配对（tp_version - none）

说明：每轮按 `none=(o1,o4)`、`tp_version=(o2,o3)` 求均值后对比。

| round | none_p95_s | tp_p95_s | delta_p95_s | none_mean_active | tp_mean_active | delta_mean_active | none_peak_active | tp_peak_active | delta_peak_active | delta_publish_bw | delta_put_bw |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.186 | 61.100 | -26.086 | 16.978 | 17.644 | 0.667 | 25.616 | 22.018 | -3.597 | 0.006 | 0.018 |
| 2 | 59.890 | 65.474 | 5.585 | 17.827 | 17.022 | -0.804 | 22.066 | 21.249 | -0.817 | 0.025 | -0.007 |
| 3 | 77.846 | 61.585 | -16.261 | 16.926 | 18.107 | 1.181 | 24.773 | 20.887 | -3.886 | -0.051 | -0.066 |

## 3.8 去除 transport-timeout 尾部后的对照

| mode | publish_to_apply_p95_mean_s | transport_mean_active_gib_s | transport_peak_active_gib_s | publish_bw_mean_gib_s | put_bw_mean_gib_s |
|---|---:|---:|---:|---:|---:|
| none(no-timeout) | 61.821 | 17.792 | 22.020 | 2.003 | 2.583 |
| tp_version(no-timeout) | 62.720 | 17.591 | 21.385 | 1.975 | 2.528 |

## 4. 数据分析

## 4.1 稳定性与一致性

1. 功能结果：`12/12 passed`。
2. receiver 行为一致：`all_receivers_completed=true`、`receiver_skips=0`。
3. group contract 全通过：
   - `tp_version` 全部 `grouped_transports = total_transports`。
   - `none` 全部 `grouped_transports = 0`。

结论：group 语义生效且边界清晰，没有“误分组/漏分组”现象。

## 4.2 吞吐与带宽分析

1. 发布带宽与写入带宽：`publish_bw` 和 `put_bw` 在两种 mode 下几乎重合（`publish_bw` 差异 `+0.224%`，`put_bw` 差异 `-0.259%`），说明 group 模式未改变 publisher 侧写入效率。
2. 传输活跃吞吐：`tp_version` 的 `mean_active` 相对 `none` 为 `+2.017%`，但 `p95_active` 为 `-2.905%`、`peak_active` 为 `-11.456%`，呈现“均值接近、峰值更平滑”的形态。
3. 逐轮配对看，`delta_mean_active` 在三轮内正负交替（`+0.667 / -0.804 / +1.181`），说明在当前口径下吞吐差异尚不稳定，不能宣称 group 带来确定吞吐增益。
4. 全量样本中 `publish_to_apply_p95` 的 mode 差异主要受少量 tail 事件影响；去除 transport-timeout 后，`none/tp_version` 的 p95 分别为 `61.821s/62.720s`，差异回到同一量级。

## 4.3 Group 作用分析

1. 语义作用（强）：`grouped_ratio` 在 `tp_version` 为 `1.000`，在 `none` 为 `0.000`，且 `group_mode_consistent/group_contract_consistent` 为 `12/12` 通过，说明 group contract 严格生效。
2. 性能作用（当前口径下有限）：在 `max_concurrency=1` 的 steady 设置里，活跃并发大多接近 1（`active_transport_peak=1` 的 case 为 `10/12`），调度器可发挥的并发重排空间有限，因此 group 的性能增益不显著。
3. 负载扩散：`diffusion_top1_share` 两组几乎一致（`0.150`），`diffusion_unique_sources` 在 `tp_version` 更低（`8.0` vs `12.167`），说明该口径下 group 更偏向“按组约束传输路径”，而不是追求 source 扩散数最大化。

## 4.4 Timeout 合理性与可信度

1. `queue_or_visibility_wait` 主要反映“可见性等待”状态，本轮不影响最终通过判定。
2. `deadline_exceeded` 为少量 tail（2/12），集中在 `none/o4`，会抬高对应轮次 `publish_to_apply_p95`。
3. 即便存在 timeout 观测，整体合同检查与通过率保持 100%，结论可信。

## 4.5 keep_last=2 的资源可行性

1. publisher 使用 `stable120 + 300Gi`（worker 规格实测 `CPU=4, Memory=300Gi, GPU=1`）。
2. receiver 使用 `stable80 + 200Gi`（worker 规格实测 `CPU=4, Memory=200Gi, GPU=1`）。
3. 最终纳入口径日志未出现 OOM/Killed。

结论：在 40Gi payload 下，`keep_last=2` 与该资源配置匹配，满足 steady 运行要求。

## 5. 最终结论与后续使用建议

## 5.1 最终结论

1. 本轮在你指定口径下完成了可复用的最终数据集：12 个 case 全通过。
2. `group=none` 与 `group=tp_version` 的合同语义完全符合预期。
3. steady 推荐口径可固定为：
   - `keep_last=2`
   - `publish_interval_s=60`
   - publisher `stable120/300Gi`，receiver `stable80/200Gi`

## 5.2 文档使用规则

1. 本文件只保留：实验方法、最终数据、分析结论。
2. 过程性记录（调试、临时执行细节）不进入正文。
3. 新批次更新时，仅追加“同口径最终纳入数据”，并更新本文件的聚合表与结论。
