# 多机 P2P Put/Get 扩容重跑报告（精简版，最终口径）

日期：2026-03-01（UTC）  
范围：主线 A（fanout/cascade 扩容）与主线 B（TP4 diffusion）重跑结果的最终汇总。

---

## 1. 执行摘要

1. 主线 A：执行批次共 12 个 run（含 `32n + 1024MiB` 补跑），最终全部通过；按去重后的目标矩阵统计为 11/11 通过。
2. 主线 A 关键点：`32n + 1024MiB` 最终口径为 `all_get_complete=True`、`get_success_rate=1.0`、`task_load_complete_sec_mean=12.812`、`cluster_gibps_mean=2.424`。
3. 主线 B：目标矩阵 6 个 case 全通过（`r1/r3/r7 × none/tp_version`），其中 `r3 + tp_version` 采用 `stable80` 版本作为最终口径。
4. 组模式对比结论保持不变：`r7` 规模下 `tp_version` 的传播长尾与集中度劣于 `none`。

---

## 2. 统一实验说明与参数口径

### 2.1 统一字段定义

| 字段 | 含义 | 单位/取值 |
|---|---|---|
| `worker_count` | fanout 扩容场景下参与 worker 数 | 节点数（n） |
| `receiver_count` | TP4 diffusion 场景下 receiver 数 | `1/3/7` |
| `artifact_size_mib` | 单次载荷规模 | MiB（`1024/2048/8192`） |
| `group_mode` | 组传输策略 | `none` / `tp_version` |
| `task_load_complete_sec_mean` | task 端完成时延均值（formal） | 秒 |
| `cluster_gibps_mean` | 集群吞吐均值（formal） | GiB/s |
| `publish_to_apply_p95` | 发布到应用的 p95 时延 | 秒 |
| `passed` | case 通过标记 | `True/False` |

说明：所有“最终口径”均以该实验的最后一次有效结果为准，不再保留同一 case 的旧结论。

### 2.2 主线 A（fanout/cascade 扩容）

1. 结果目录：`/data/tc_cross_rerun/results_multi_host_scaleout_r0301a`。
2. 批次：`r0301a`、`r0301m`、`r0301l`、`r0301x`，以及 `32n + 1024MiB` 的补跑 `r0301xrootfix10`。
3. 去重后目标矩阵：11 个 case；执行层面共 12 个 run（含同一 case 的补跑）。
4. 通过标准：以 case 最终汇总结果为准（`passed=True`，并结合 `all_get_complete/get_success_rate`）。

### 2.3 主线 B（TP4 diffusion）

1. 原始目录：`/data/tc_cross_rerun/results_weight_publisher_r0301`。
2. 汇总目录：`/data/tc_cross_rerun/results_tp4_r0301_final`。
3. 目标矩阵（6 个 case）：
   - `suite_rw0301a3_r1_tp_bind_into_swap_none`
   - `suite_rw0301a3_r1_tp_bind_into_swap_tp_version`
   - `suite_rw0301a3_r3_tp_bind_into_swap_none`
   - `suite_rw0301a3_r3_tp_bind_into_swap_tp_version_stable80`
   - `suite_rw0301b1_r7_tp_bind_into_swap_none`
   - `suite_rw0301b1_r7_tp_bind_into_swap_tp_version`
4. `r3 + tp_version` 的最终口径固定为 `stable80` case。

---

## 3. 主线 A 最终结果（fanout/cascade）

### 3.1 去重后 case 结果（最终）

| case | passed | get_success_rate | task_load_complete_sec_mean | cluster_gibps_mean |
|---|---:|---:|---:|---:|
| `3n + 1024MiB` | True | 1.000 | 7.262 | 0.280 |
| `4n + 1024MiB` | True | 1.000 | 6.865 | 0.437 |
| `6n + 1024MiB` | True | 1.000 | 7.189 | 0.696 |
| `8n + 1024MiB` | True | 1.000 | 8.308 | 0.845 |
| `8n + 2048MiB` | True | 1.000 | 9.002 | 1.556 |
| `9n + 1024MiB` | True | 1.000 | 8.252 | 0.973 |
| `9n + 2048MiB` | True | 1.000 | 9.268 | 1.726 |
| `16n + 1024MiB` | True | 1.000 | 9.194 | 1.633 |
| `16n + 8192MiB` | True | 1.000 | 22.801 | 5.297 |
| `32n + 1024MiB` | True | 1.000 | 12.812 | 2.424 |
| `32n + 8192MiB` | True | 1.000 | 25.484 | 9.742 |

### 3.2 量化结论（最终口径）

1. `1024MiB` 线性扩容：
   - `3n -> 16n`：`0.280 -> 1.633 GiB/s`（`+5.83x`），归一化效率约 `109.3%`。
   - `3n -> 32n`：`0.280 -> 2.424 GiB/s`（`+8.66x`），归一化效率约 `81.2%`。
   - `16n -> 32n`：`1.633 -> 2.424 GiB/s`（`+48.4%`）。
2. `8192MiB` 线性扩容：
   - `16n -> 32n`：`5.297 -> 9.742 GiB/s`（`+1.84x`），归一化效率约 `92.0%`。
3. `32n + 1024MiB`（专项对照，formal-only）：
   - `task_mean`: `59.017s -> 12.812s`（`-78.29%`，约 `4.61x`）。
   - `cluster_mean`: `1.674 -> 2.424 GiB/s`（`+44.81%`）。
   - `all_get_complete_rate`: `66.7% -> 100%`，`>120s` 慢尾次数：`1 -> 0`。

---

## 4. 主线 B 最终结果（TP4 diffusion）

### 4.1 目标矩阵结果（6/6）

| case | passed | receiver_count | group_mode | publish_to_apply_p95(s) | grouped_transports | receiver_skip_count |
|---|---:|---:|---:|---:|---:|---:|
| `r1 none` | True | 1 | none | 7.216 | 0 | 0 |
| `r1 tp_version` | True | 1 | tp_version | 7.744 | 40 | 0 |
| `r3 none` | True | 3 | none | 27.445 | 0 | 0 |
| `r3 tp_version stable80` | True | 3 | tp_version | 27.494 | 120 | 0 |
| `r7 none` | True | 7 | none | 49.507 | 0 | 7 |
| `r7 tp_version` | True | 7 | tp_version | 63.605 | 63 | 7 |

### 4.2 组模式对比（`tp_version` vs `none`）

1. `publish_to_apply_p95` 变化：
   - `r1`：`+7.3%`（`7.216s -> 7.744s`）。
   - `r3`：`+0.18%`（`27.445s -> 27.494s`，基本持平）。
   - `r7`：`+28.5%`（`49.507s -> 63.605s`）。
2. `r7` 集中度变化：
   - `top1_share`：`0.183 -> 0.444`（`+143.5%`）。
   - `HHI`：`0.079 -> 0.352`（`+346%`）。
3. 最终结论：`r7` 规模下，`tp_version` 未体现扩散收益，且传播长尾更明显。

---

## 5. 最终结论（统一口径）

1. 主线 A：执行层面 12/12 run 通过；去重后目标矩阵 11/11 通过。
2. `32n + 1024MiB` 的最终结果已收敛到稳定完成（formal 约 `12~14s` 区间）。
3. 主线 B：目标矩阵 6/6 通过，但 `r7` 下 `tp_version` 相对 `none` 仍存在明显时延与集中度劣化。

---

## 6. 关键产物（最终口径）

1. fanout 汇总目录：`/data/tc_cross_rerun/results_multi_host_scaleout_r0301a`
2. TP4 原始目录：`/data/tc_cross_rerun/results_weight_publisher_r0301`
3. TP4 纳入汇总目录：`/data/tc_cross_rerun/results_tp4_r0301_final`
4. 汇总 JSON：`/data/tc_cross_rerun/report/scaleout_summary_r0301.json`
5. 汇总 Markdown：`/data/tc_cross_rerun/report/scaleout_summary_r0301.md`
6. `32n + 1024MiB` 最终 case JSON：`/data/tc_cross_rerun/r0301xrootfix-20260301-230123/results_multi_host_scaleout_r10/suite_r0301xrootfix10_xlarge_fanout_32n_c20b16w16_g0_s1024.json`
