# 2026-02-22 0081 多机 Chaos 测试报告（中文）

## 1. 测试结论

本次按 `docs/plans/0081-multihost-put-get-chaos-stability.md` 执行 `small -> medium -> large` 三阶段 chaos 套件，阶段门禁全部通过。

- 阶段聚合结论：`passed=true`
- 结果文件：`/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase_gate_review_0081_exec.json`

同时，本轮已完成远端 worker 资源回收（仅保留 master）。

## 2. 目标与范围

本轮目标是执行 0081 计划中的阶段化 chaos 验证闭环：

- 严格顺序执行：`small -> medium -> large`
- 每阶段执行正向 case + 负向 expected-failure case
- 生成并校验 run 级 `gate_review.json`
- 生成并校验 phase 级 `phase_gate_review.json`

本轮不在范围内：

- 全量故障矩阵（daemon_kill/GS outage/netem/resource pressure）
- `ticket_replica_uuid` 主扩散标识链路落地

## 3. 测试环境

### 3.1 Global Store

- 地址：`100.97.246.95:50051`
- 会话：`gs-20260222-191016-c724`
- 配置：`examples/config/global_store_config_cross_host_bench_fast_failover.yaml`

### 3.2 Worker 启动策略

- 启动命令约束：`brainctl launch --charged-group=tensorcast_dev --gpu 1 --cpu 4 --memory 106400 --private-machine group --max-wait-duration=20m`
- 本轮 worker（已于测试后删除）：
  - `ws-7681b3683947089e-worker-hhnxx`
  - `ws-7681b3683947089e-worker-7dtq7`
  - `ws-7681b3683947089e-worker-dm7sk`
  - `ws-7681b3683947089e-worker-r4sh7`
  - `ws-7681b3683947089e-worker-5l5p6`
  - `ws-7681b3683947089e-worker-994zx`
  - `ws-7681b3683947089e-worker-sq7d2`
  - `ws-7681b3683947089e-worker-hldhs`

### 3.3 Case Schema

- `small`：`/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/small.json`
- `medium`：`/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/medium.json`
- `large`：`/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/large.json`

## 4. 执行命令（实际）

```bash
source .venv/bin/activate

TC_CASE_SCHEMA=/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/small.json \
TC_OUT_DIR=/tmp/tc_cross_20260222/results_chaos_phase_0081_exec \
TC_RUN_ID=phase-small-20260222g \
TC_CHAOS_SEED=7 TC_REMOTE_TIMEOUT_SEC=900 TC_GATE_MAX_RECOVER_TIME_SEC=300 \
bash examples/cross_host/run_multihost_chaos_suite.sh

TC_CASE_SCHEMA=/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/medium.json \
TC_OUT_DIR=/tmp/tc_cross_20260222/results_chaos_phase_0081_exec \
TC_RUN_ID=phase-medium-20260222f \
TC_CHAOS_SEED=7 TC_REMOTE_TIMEOUT_SEC=900 TC_GATE_MAX_RECOVER_TIME_SEC=300 \
bash examples/cross_host/run_multihost_chaos_suite.sh

TC_CASE_SCHEMA=/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/large.json \
TC_OUT_DIR=/tmp/tc_cross_20260222/results_chaos_phase_0081_exec \
TC_RUN_ID=phase-large-20260222g \
TC_CHAOS_SEED=7 TC_REMOTE_TIMEOUT_SEC=900 TC_GATE_MAX_RECOVER_TIME_SEC=300 \
bash examples/cross_host/run_multihost_chaos_suite.sh

python examples/cross_host/chaos_phase_gate_review.py \
  --small-run-dir /tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase-small-20260222g \
  --medium-run-dir /tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase-medium-20260222f \
  --large-run-dir /tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase-large-20260222g \
  --output /tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase_gate_review_0081_exec.json \
  --markdown-output /tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase_gate_review_0081_exec.md
```

## 5. 结果汇总

| 阶段 | run_id | case 通过 | all_get_complete | expected_failure_pass | comm_errors_delta | gate_review |
|---|---|---:|---|---|---:|---|
| small | `phase-small-20260222g` | 2/2 | true | true | 0 | passed |
| medium | `phase-medium-20260222f` | 2/2 | true | true | 0 | passed |
| large | `phase-large-20260222g` | 2/2 | true | true | 0 | passed |

Phase 聚合：

- `passed=true`
- 文件：`/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase_gate_review_0081_exec.json`

## 6. 关键观测

1. 三阶段均无 `unexpected_failure` / `unexpected_success`。
2. 三阶段 `budget_exit_reason_buckets` 均为 `success`（small=4, medium=10, large=14）。
3. 三阶段 `failure_classification_counts` 均为 `{infra:0, product:0, unknown:0}`。
4. 本轮 schema 使用的是 `sleep` no-op 事件和启动参数负向 case，不包含实质性故障注入（如 kill/outage/netem）。

## 7. 失败原因分析与修复状态

### 7.1 本轮执行失败情况

- 本轮三阶段均通过，没有新失败。

### 7.2 历史阻塞失败（已修复）

在本轮前的历史执行中，`large` 曾出现 `all_get_complete=false`，核心根因为 GS worker 重注册遇到“daemon_id 与 inactive endpoint 冲突”时被错误拦截，导致 P2P 端口异常路径。

- 修复文件：`tensorcast/global_store/services/worker_service.py`
- 回归测试：`tests/python/global_store/test_services.py` 中
  `test_worker_service_registration_reclaims_inactive_endpoint_conflict`
- 修复后已在多轮 small/medium/large 重跑中验证通过。

## 8. 与 0081 计划的符合性评估

已满足（本轮证据覆盖）：

- `small/medium/large` 阶段 gate 通过
- 正向 `all_get_complete` / 负向 `expected_failure_pass` / `comm_errors_delta==0`
- 阶段级 gate 归档产物齐全

未完全满足（仍待后续工作）：

1. `Recovery thresholds` 的全故障类型证据：本轮未覆盖 kill/outage/netem/resource pressure。
2. 扩散门禁的“多源重建”证据：当前 `source_cardinality_timeline` 仍为单源 `p2p`，未覆盖设计文档中更严格的多源扩散验收。
3. 计划中 Phase 1/2/3/4/6 的工程项（schema/gate 深化、mixed steady-state、request_id 因果链、自动化回归）未在本轮内完成。

## 9. 资源回收

测试完成后已清理本轮所有临时 worker：

- 已删除：`hhnxx, 7dtq7, dm7sk, r4sh7, 5l5p6, 994zx, sq7d2, hldhs`
- 当前仅保留：`ws-7681b3683947089e-master`

## 10. 产物清单

- 阶段输出目录：`/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/`
- 三阶段 gate：
  - `phase-small-20260222g/gate_review.json`
  - `phase-medium-20260222f/gate_review.json`
  - `phase-large-20260222g/gate_review.json`
- phase 聚合 gate：
  - `phase_gate_review_0081_exec.json`
  - `phase_gate_review_0081_exec.md`
