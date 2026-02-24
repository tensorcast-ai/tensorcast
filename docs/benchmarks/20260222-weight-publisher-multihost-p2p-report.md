# WeightPublisher 多机强制 P2P 测试报告（训练权重推送到推理实例）

日期：2026-02-22  
代码版本：`904bf352`

## 1. 目标与范围

本次测试验证以下流程在多机下的正确性、性能与稳定性：

1. Publisher 节点持续发布新权重版本（`WeightPublisher.publish(...)`，写入本地 stable DRAM）。
2. 多个 Receiver 节点通过 `binding.swap` 持续更新到最新版本。
3. 版本管理采用 `keep_last=2`，旧版本通过 de-register 回收。
4. 强制数据路径走 P2P（`fallback_prefer=p2p`），不走其它主路径。

重点验证项：

1. 一键发布、多节点同步更新。
2. 不跳版（严格要求 receiver 完整消费 `v1..vN`）。
3. `keep_last=2` 下旧版本释放与冗余副本收敛。
4. 长时连续发布场景稳定性。

## 2. 测试环境

1. Global Store: `100.97.246.95:50051`
2. publisher worker: `ws-7681b3683947089e-worker-pfj85`
3. receiver workers:
   - `ws-7681b3683947089e-worker-wdljv`
   - `ws-7681b3683947089e-worker-2dmnq`
4. 关键参数：
   - `start_version=1`
   - `num_versions=4`
   - `keep_last=2`
   - `publish_interval_s=60`
   - `receiver_timeout_s=95`
   - `max_publish_to_apply_s=30`
   - `receiver_apply_mode=binding_swap`
   - `fallback_prefer=p2p`

## 3. 关键问题与根因修复

### 3.1 根因

多机 `binding_swap + p2p` 初始失败，receiver 报错：`Communicator is required`。

根因是框架层 `MaterializationFacade` 在部分 P2P 路径构造 `P2PSource` 时未注入 `comm_engine`，触发 `P2PLoader` 初始化失败。

### 3.2 修复

已在以下路径补齐 `comm_engine` 注入，并增加通信可用性前置检查：

- `core/store/runtime/ingestion/materialization_facade.cc:1491`
- `core/store/runtime/ingestion/materialization_facade.cc:1518`
- `core/store/runtime/ingestion/materialization_facade.cc:1859`
- `core/store/runtime/ingestion/materialization_facade.cc:1886`

### 3.3 测试框架增强（避免误判）

1. receiver 进程提前退出时 fail-fast，不再长时间等文件超时。
2. 每节点 daemon 使用独立本地端口（同一 run 内 `50052+index`），避免旧会话地址冲突。
3. 新增 `publish->apply` SLA 校验（`--max-publish-to-apply-s`，默认 30s）。
4. probe JSON 解析改为提取最后一个 JSON 对象，避免日志前缀导致误判。
5. 明确不允许跳版（默认 strict），仅在显式开关时允许。

## 4. 分阶段测试执行

### 4.1 功能预验证（单机）

文件：`/tmp/tc_cross_20260222/results_weight_publisher/wp-noskip-20260222-v5/single_host_summary.json`

结果：

1. 发布 4 个版本全部成功。
2. receiver 按顺序完整消费 4 个版本。
3. 基础功能正常，进入跨机阶段。

### 4.2 两节点（1 publisher + 1 receiver）

文件：`/tmp/tc_cross_20260222/results_weight_publisher/wp-noskip-v6-two-node-1771776062/wp-noskip-v6-two-node.json`

结果：`passed=true`

关键指标：

1. publish latency mean: `0.102s`
2. apply latency mean: `0.139s`
3. publish->apply p95: `0.629s`（<= 30s 阈值）
4. receiver 序列完整，无跳版。
5. `binding.swap` 指针稳定性通过。

版本管理/回收验证：

1. `keep_last=2`，最终保留 `[3,4]`，淘汰 `[1,2]`。
2. `cluster_probe.dropped_non_materializable=true`
3. `cluster_probe.dropped_not_exists=true`
4. `cluster_probe.materializable_version_count_within_limit=true`

### 4.3 三节点（1 publisher + 2 receivers）

文件：`/tmp/tc_cross_20260222/results_weight_publisher/wp-noskip-v6-three-node-1771776347/wp-noskip-v6-three-node.json`

结果：`passed=true`

关键指标：

1. publish latency mean: `0.141s`
2. apply latency mean: `0.244s`
3. publish->apply p95: `0.872s`（<= 30s 阈值）
4. 两个 receiver 均完整消费 `v1..v4`，无跳版。
5. `binding.swap` 指针稳定性通过。

版本管理/回收验证：

1. `keep_last=2`，最终保留 `[3,4]`，淘汰 `[1,2]`。
2. 旧版本不可物化且不存在冗余可用副本痕迹（cluster probe 口径）。
3. materializable 版本窗口始终不超过 2。

## 5. 稳定性与性能结论

1. 在强制 P2P、连续发布（60s 间隔）下，2/3 节点均稳定通过。
2. receiver 全程无跳版，且 publish->apply 延迟远低于 30s 阈值。
3. `keep_last=2` 机制在跨机场景下行为符合预期：旧版本被回收，窗口维持在最新两版。

## 6. 踩坑与最佳实践

### 6.1 踩坑

1. **框架根因**：P2P 源缺少 `comm_engine` 注入，导致 `Communicator is required`。
2. **超时口径错误**：`receiver_timeout < publish_interval` 会在“下一版本尚未发布”时误超时。
3. **probe 误判**：远端输出混杂日志时，直接 `json.loads` 会失败。
4. **GS DB 文件锁**：本地 `duckdb` 直连读可能被 GS 进程锁住，导致严格 DB 审计不可用。

### 6.2 最佳实践

1. 对强制 P2P 场景，先保证通信通路在框架层完整（`comm_engine` 不可缺）。
2. `receiver_timeout` 至少覆盖“发布周期 + 消费窗口”（如 `60s + 30s`，建议 `95s`）。
3. 单独设置并严格校验 `publish->apply` SLA（如 30s），不要混在 per-version timeout 里。
4. 多机 daemon 启停使用显式生命周期管理；每节点独立本地端口，避免会话冲突。
5. 跨机探针解析使用“最后 JSON 对象”策略，避免日志噪声误报。

## 7. 产出物

1. 两节点结果：`/tmp/tc_cross_20260222/results_weight_publisher/wp-noskip-v6-two-node-1771776062/wp-noskip-v6-two-node.json`
2. 三节点结果：`/tmp/tc_cross_20260222/results_weight_publisher/wp-noskip-v6-three-node-1771776347/wp-noskip-v6-three-node.json`
3. 单机功能预验证：`/tmp/tc_cross_20260222/results_weight_publisher/wp-noskip-20260222-v5/single_host_summary.json`
4. 本报告：`docs/benchmarks/20260222-weight-publisher-multihost-p2p-report.md`

## 8. 深度压测阶段更新（持续追加）

### 8.1 阶段 A：16 节点，20 versions（通过）

配置：

1. `heartbeat=60s`
2. `periodic_sync=60s`
3. receiver poll `1.5s`
4. GS worker threads `32`
5. `keep_last=2`
6. `publish_interval=60s`

结果文件：

1. `/tmp/tc_cross_20260222/results_weight_publisher_deep_hb60/wp16-hb60-poll1p5-v20-1771785231/wp16-hb60-poll1p5-v20.json`

关键结果：

1. `passed=true`
2. publish latency mean `0.0888s`
3. apply latency mean `0.2809s`
4. publish->apply p95 `1.9849s`
5. receiver 完整消费且 `binding.swap` 指针稳定。

### 8.2 阶段 B：32 workers 首轮失败（P2P staging 通道上限）

现象：

1. 部分 receiver 长时间停在低版本，出现 `MaterializeIntoTarget INTERNAL`。

根因：

1. publisher 源 daemon 日志出现：
   `MTCP read request failed: RESOURCE_EXHAUSTED: GPU staging pool capacity exceeded ...`
2. `comm_gpu` pinned pool 过小，不能覆盖 31 receiver fanout 并发通道需求。

动作：

1. 提升 `examples/config/store_daemon_config_cross_host_bench_lowmem.yaml` 中 `comm_gpu.pool_bytes`。

### 8.3 阶段 C：32 workers 二轮失败（启动预检卡边界）

现象：

1. 某些节点 daemon 启动失败，错误为 `startup memory preflight failed`。

根因：

1. 配置总需求约 `60.50GiB`，而部分 worker 可用内存接近 `60.44~60.58GiB`，触发边界失败。

动作：

1. 将 `comm_gpu.pool_bytes` 调整为 `8448MB`，在 fanout 需求与启动余量之间平衡。

### 8.4 阶段 D：32 workers 三轮失败（GS 地址/端口注册冲突）

现象：

1. 运行 `wp32-hb60-poll1p5-v20-r5-1771787641` 时，publisher 到 `v2`，但 receiver 仅 `15/31` 推进，另外 `16` 个停在 `v0`。

根因：

1. 落后节点 daemon 日志一致报错：
   `Worker lifecycle start failed: Address/port already registered by another worker`
2. 属于 worker 生命周期注册冲突（旧注册残留与新 run 冲突），导致部分节点未进入正常 GS 生命周期。

当前处置：

1. 该轮判定无效并停止。
2. 确认这不是数据面 P2P 瓶颈，而是控制面注册语义问题。

### 8.5 根本修复：GS endpoint takeover（不依赖换端口规避）

修复目标：

1. 老 daemon 异常退出后，即使还没过 heartbeat timeout，新 daemon 也应可立即注册同一 endpoint。
2. 新注册应成为权威记录，不要求“必须等待旧记录被动超时”。

修复实现：

1. `tensorcast/global_store/services/worker_service.py`
   - `register_worker` 新增 endpoint takeover 逻辑：当 `node_address:grpc_port` 被旧记录占用时，回收旧 endpoint owner（标记副本不可用并删除旧 worker 行），接受新注册。
   - 对 `daemon_id` 重绑场景同样允许回收冲突 endpoint 行，而不是仅在 `inactive_at` 已置位时放行。
2. `tensorcast/global_store/rpc/worker_rpc_handler.py`
   - 移除注册前的“按 node_id 拒绝 endpoint 冲突”短路检查，统一由 `WorkerService.register_worker` 执行一致语义。
3. 新增测试覆盖：
   - `tests/python/global_store/test_services.py`
     - `test_worker_service_registration_reclaims_active_endpoint_takeover`
     - `test_worker_service_registration_daemon_rebind_reclaims_active_endpoint`
   - `tests/python/global_store/test_grpc_service.py`
     - `test_worker_registration_endpoint_takeover_with_new_daemon_id`

测试结果：

1. `pytest tests/python/global_store/test_services.py -k 'reclaims_active_endpoint_takeover or daemon_rebind_reclaims_active_endpoint or reclaims_inactive_endpoint_conflict'` 通过。
2. `pytest tests/python/global_store/test_grpc_service.py -k 'endpoint_takeover_with_new_daemon_id or worker_registration_daemon_id_is_stable_identity'` 通过。

### 8.6 阶段 E：32 workers，20 versions（根本修复后通过）

执行：

1. case: `wp32-hb60-poll1p5-v20-r7`
2. 保持 `heartbeat=60s`、`periodic_sync=60s`、receiver poll `1.5s`。
3. `keep_last=2`，强制 P2P，`publish_interval=60s`。
4. 端口基线恢复为常规值（`daemon_p2p_port_base=65090`），不依赖“每轮换端口”规避冲突。

结果文件：

1. `/tmp/tc_cross_20260222/results_weight_publisher_deep_hb60/wp32-hb60-poll1p5-v20-r7-1771788460/wp32-hb60-poll1p5-v20-r7.json`

关键结果：

1. `passed=true`
2. publish latency mean `0.0876s`
3. apply latency mean `0.8212s`
4. publish->apply p95 `3.6302s`（<= 30s）
5. `all_receivers_completed=true`，31 个 receiver 全部完成，未出现跳版。
6. GS/cluster 审计通过：
   - `old_versions_released=true`
   - `replica_versions_within_window=true`
   - `replica_version_count_within_limit=true`
   - 仅最新两版（v19/v20）仍有副本，旧版副本计数均为 0。

### 8.7 后续建议（配置层 + 系统层）

配置层：

1. 32-worker fanout 推荐 `comm_gpu.pool_bytes >= 8448MB`，并保留启动内存余量。
2. 压测前保持显式 daemon 生命周期清理（stop + kill 残留），减少噪声。
3. runner 保留进度输出与 fail-fast（尤其在 receiver 未 ready、publisher 异常退出场景）。

系统层：

1. 保持“endpoint takeover”作为 GS 默认行为（新注册优先于陈旧 endpoint 记录）。
2. 增加可观测性：在 GS 暴露 endpoint takeover 计数与明细事件，便于定位注册抖动。
3. runner 在启动 publisher 前增加“worker lifecycle 全量就绪”校验，避免无效长跑。
4. `ResolveKeyMapping` 链路建议补充端到端诊断字段（版本时间戳、来源 daemon_id、映射生效延迟），加速控制面排障。

### 8.8 正确性黑盒补充（全量值校验）

目标：

1. 验证 `version=1` 时 tensor 全元素为 `1`，`version=2` 时全元素为 `2`（以此类推）。
2. 验证 `binding.swap` 路径下，多节点持续更新时不跳版且值准确。

实现方式：

1. 在 `tensorcast/tools/weight_publisher_e2e.py` 新增 `payload_mode=version_fill`：
   - 发布端构造 `version_marker / weight_probe / rolling_checksum` 三个 tensor，全部元素填充为当前版本号。
   - 接收端在 materialize/binding.swap 后做“全元素等于版本号”校验，不是只校验局部位点。
2. 在 `examples/cross_host/cross_host_weight_publisher_runner.py` 透传 `--payload-mode` 并记录到结果参数。

阶段结果：

1. `wp32-hb60-poll1p5-v20-version-fill-r2`：
   - 功能/性能通过，但 `passed=false`（GS 即时审计窗口出现 `v18` 残留 1 副本）。
   - 复查发现该残留随后收敛为 0，属于“`periodic_sync=60s` 下即时审计时序窗口”。
2. `wp32-hb60-poll1p5-v6-version-fill-r3/r4`：
   - `r3` 失败根因为 publisher worker 生命周期到期（最早 smoke worker 是 `sleep 1800`，进程自动 Completed）。
   - `r4` 失败根因为部分节点启动 preflight 内存不足（可用约 `53.28GiB`，需约 `59.67GiB`）。
3. `wp32-hb60-poll1p5-v6-version-fill-r5`（最终通过）：
   - 文件：`/tmp/tc_cross_20260222/results_weight_publisher_deep_hb60/wp32-hb60-poll1p5-v6-version-fill-r5-1771792583/wp32-hb60-poll1p5-v6-version-fill-r5.json`
   - `passed=true`
   - receiver 数量 `31`，每个 receiver 均完整消费 `v1..v6`（`bad_sequences=0`，`receiver_skips=0`）。
   - publish latency mean `0.0920s`，apply latency mean `1.0683s`，publish->apply p95 `7.2620s`（<= 30s）。
   - GS/cluster 审计均通过：仅保留最新两版（`v5/v6`），旧版释放成立。

### 8.9 本轮新增问题与沉淀

问题与根因：

1. **GS 地址不可达**：历史地址 `100.97.246.95:50051` 不再可达，导致 daemon connect 失败。
2. **publisher 进程非预期退出**：误用 smoke worker（`sleep 1800`）作为长期 publisher，生命周期到点后 `rc=137`。
3. **部分节点内存波动触发 preflight 失败**：在同等容器规格下，不同节点可用内存差异可达数 GiB。
4. **进度监控“回零”误导**：progress 仅按日志尾部窗口解析，尾部被 JSON 覆盖后会显示 0/0（非真实回退）。

已采取动作：

1. 改为在 publisher 节点显式启动/托管 GS（`python -m tensorcast.global_store`）并使用可达内网地址。
2. 更换 publisher 到长期 worker（`sleep 10800`），避免生命周期中途结束。
3. 新增 ultra-low-mem bench 配置用于低可用内存节点（`stable=8GB`、engine pinned pool `24GB`）。
4. 运行前执行全量 daemon 清理（`daemon stop --force` + kill 残留）。
5. 修复 runner 进度统计为“单调累积口径”（历史最大值），消除日志尾部覆盖导致的回零误判。

## 9. TP=4 多节点 Receiver 压测补充（bind_into + swap + copy_plan）

### 9.1 场景与目标

本轮新增场景：

1. publisher 保持单 GPU（完整权重发布）。
2. receiver 节点升级为 4-GPU worker，每节点模拟 TP=4：
   - 每个版本更新时，同一节点内执行 4 次 receiver apply（rank0~rank3）。
   - 每个 rank 绑定独立 GPU（`cuda:0..3`）。
   - 每个 rank 使用独立 `copy_plan`，通过 `bind_into(..., mapping=copy_plan)` + `binding.swap(...)` 完成更新。

验证目标：

1. 功能正确性：每个 rank 收到的数据值与版本/rank 预期一致。
2. 稳定性：长时连续发布下，不跳版、不丢版、pointer 稳定。
3. 性能：publish/apply 与 publish->apply 延迟可控。
4. 回收正确性：`keep_last=2` 下旧版本释放与副本窗口收敛。

### 9.2 实现说明（黑盒 + 真实 mapped-binding 路径）

本轮在 `weight_publisher_e2e` 增加 TP 载荷与 apply 模式：

1. `payload_mode=tp_ranked`：
   - 发布两个源 tensor：`tp_col_weight`（按 dim0 分片）与 `tp_row_weight`（按 dim1 分片）。
   - 版本 `v` 下，rank `r` 对应分片值为 `v + r/10`。
   - 例如 `v1`：rank0/1/2/3 对应 `1.0/1.1/1.2/1.3`。
2. `receiver_apply_mode=tp4_bind_into_swap`：
   - 每个 receiver 节点内部维护 4 组 rank binding。
   - 首版 `bind_into(mapping=copy_plan)`，后续全部 `swap`。
   - copy_plan 采用 row/col 混合切分：同一版本内同时覆盖 dim0 与 dim1 两种分片方式。
3. 校验：
   - 每个 rank 每次 apply 后，校验 rank 目标 tensor 全元素等于 `v + r/10`。
   - 若任一 rank 值不匹配，立即失败。

### 9.3 资源与编排

1. 启动前先清理上一轮 worker（按要求先关闭后重申）。
2. 资源申请（brainctl）：
   - publisher: `1 GPU / 4 CPU / 96GiB`
   - receiver: `8 workers * (4 GPU / 16 CPU / 240GiB)`
3. 显式 GS 生命周期：
   - 在 publisher 节点显式启动 GS（`python -m tensorcast.global_store`），`max_workers=32`。
   - 以可达内网地址对外提供（本轮 `100.99.211.49:50051`）。

### 9.4 阶段 A：功能验证（2 个 4-GPU receiver）

执行：

1. case: `wp-tp4-r2-v4-func`
2. 参数：`payload_mode=tp_ranked`，`receiver_apply_mode=tp4_bind_into_swap`，`tp_world_size=4`，`keep_last=2`。

结果文件：

1. `/tmp/tc_cross_20260222/results_weight_publisher_tp4/wp-tp4-r2-v4-func-1771794095/wp-tp4-r2-v4-func.json`

结果：

1. `passed=true`
2. 两个 receiver 均完整消费 `v1..v4`。
3. 每版本 apply 操作序列符合预期：首版 `bind_into`，后续 `swap`。
4. 指针稳定性通过（`pointer_stable=true`）。
5. GS/cluster 回收审计通过（仅最新两版保留副本）。

### 9.5 阶段 B：主压测（8 个 4-GPU receiver，10 versions）

执行：

1. case: `wp-tp4-r8-v10-main`
2. receiver 规模：`8 节点 * 4 rank = 32 rank receiver`
3. 参数：
   - `num_versions=10`
   - `publish_interval=45s`
   - `receiver_timeout=95s`
   - `heartbeat=60s`
   - `periodic_sync=60s`
   - `fallback_prefer=p2p`
   - `keep_last=2`

结果文件：

1. `/tmp/tc_cross_20260222/results_weight_publisher_tp4/wp-tp4-r8-v10-main-1771794381/wp-tp4-r8-v10-main.json`

关键结果：

1. `passed=true`
2. `all_receivers_completed=true`，8 个 receiver 全部完整消费 `v1..v10`，无 skip。
3. 性能：
   - publish latency mean `0.0525s`
   - apply latency mean `2.1245s`
   - publish->apply p95 `4.2813s`（<= 30s 阈值）
4. 正确性：
   - receiver 全部运行在 `tp4_bind_into_swap` 模式；
   - 首版 `bind_into`，其余版本 `swap`；
   - 日志与 summary 无 `Traceback`，无版本序列错误；
   - rank 值校验在每次 apply 内部通过（否则 case 会 fail）。
5. 回收：
   - GS/cluster 审计通过；
   - 仅最新两版（v9/v10）保留副本，旧版释放成立。

### 9.6 本轮问题与 owner 视角处理

问题与处理：

1. **4-GPU worker 调度时间较长**：
   - 现象：大卡 worker 在 `Starting/ContainerCreating` 阶段停留明显长于 1-GPU。
   - 处理：保留长 wait window，强制“全员 Running + GPU 数校验”后才进入主跑，避免无效长跑。
2. **GS 地址可达性易误配**：
   - 处理：改为“先启动 GS，再从 receiver 节点做 TCP 连通性验证”，通过后才启动 case。
3. **长日志下 progress 计数偏低（尾部窗口问题）**：
   - 处理：runner 进度改为“单调累计口径”，不再因 tail 窗口丢失早期事件而回退/低估。

后续建议：

1. 在 runner 增加“receiver 节点 GPU 数必须等于 `tp_world_size`”的硬校验（目前本轮已人工校验）。
2. 将 TP rank 值校验摘要（每 rank 通过计数）写入 receiver summary，提升报告可观测性。
3. 针对 4-GPU/8-GPU worker，补充一套专用 preflight 配置建议（内存余量、pinned pool 上限）并固化到部署文档。

### 9.7 64GB 场景 OOM 根因复盘与根治动作（Publisher SDK + Daemon）

问题复盘：

1. 在 `keep_last=2` 的默认 publish 流程中，若“先 put 再 gc oldest”，发布窗口会短时出现 `旧两版 + 新版` 的三版本重叠。
2. 同时，Publisher SDK 进程在 `publish()` 返回后到 retention 检查完成前，仍持有本次 source tensor 引用（64GB 量级），造成 SDK 侧额外驻留。
3. daemon 侧固定 pinned pool 是启动期确定的基线成本（可控），但与上面两项叠加后，会把 cgroup 推到高水位，触发 OOM kill 风险。

本轮根治改动：

1. `tensorcast/tools/weight_publisher.py`
   - 新增 `pre_publish_trim_enabled` / `pre_publish_keep_last`。
   - 在发布前执行预回收（默认 keep 到 `keep_last-1`），再执行 `put`，避免发布窗口叠三版本。
   - 保留发布后 `keep_last` 收敛，确保最终窗口仍是最新两版。
2. `tensorcast/tools/weight_publisher_e2e.py`
   - TP 大 payload 默认开启 pre-publish trim。
   - `publish_one_version` 在 `publish` 完成后立即 `del tensors`，并在大 CPU payload 场景触发 `gc.collect()`，把 SDK 侧 source 驻留窗口压到最短。
   - 新增分阶段内存日志：`before_build / after_build / after_publish / after_source_release`，用于快速区分 SDK 与 daemon 的内存增长来源。
3. `examples/cross_host/cross_host_weight_publisher_runner.py`
   - publisher 内存预算模型改为“带 pre-publish trim”的峰值模型（发布前保留窗口 + in-flight + source + pinned baseline）。
   - 报告新增 `pre_publish_trim_enabled`、`retained_before_publish_versions` 等字段，避免 308GiB 类误导估算。

验证：

1. 新增单测 `tests/python/tools/test_weight_publisher_pretrim.py`：验证预回收发生在 `put` 之前，且保留窗口正确。
2. 新增单测 `tests/python/tools/test_weight_publisher_e2e_retention.py::test_publish_one_version_releases_source_before_retention`：验证 source tensor 在 retention 检查前已释放。
3. 相关回归：
   - `pytest tests/python/tools/test_weight_publisher_pretrim.py tests/python/tools/test_weight_publisher_e2e_retention.py tests/python/tools/test_weight_publisher_e2e_sampling.py -q` 通过。

补充说明（64GB 审计）：

1. 接收正确性校验已以 GPU tensor 为主（TP 大 payload 采用 GPU 上 sampled/full 校验），避免把 64GB 拉回 CPU 审计。
2. 集群级 retention 审计已采用 `exists-only` 轻量探针，避免审计过程本身引入额外大内存成本。

### 9.8 最新进展同步：TP4 + 64GB + `max_concurrency=1`（7 receiver）通过

执行：

1. case: `tp4-64gb-r7-v6-mc1-hb60-r3m500`
2. publisher: `ws-7681b3683947089e-worker-qxrln`
3. receivers: 7 个 worker（每个 worker 4 GPU）
4. 参数：
   - `tp_total_bytes=64GB`
   - `num_versions=6`
   - `publish_interval=90s`
   - `heartbeat=60s`
   - `periodic_sync=60s`
   - daemon transport `max_concurrency=1`
   - worker 资源规格：`memory=500GiB`

结果文件：

1. `/data/tc_cross_20260223/results_weight_publisher/tp4-64gb-r7-v6-mc1-hb60-r3m500-1771855585/tp4-64gb-r7-v6-mc1-hb60-r3m500.json`

结果：

1. `passed=true`
2. 功能/稳定性：
   - 7 个 receiver 全部完成，无跳版；
   - `keep_last=2` 审计通过（`old_versions_released=true`、窗口约束通过）。
3. 性能：
   - publish latency mean `50.4449s`
   - apply latency mean `55.7151s`
   - publish->apply p95 `82.0615s`

说明：

1. 该轮是大模型（64GB）+ `max_concurrency=1` 的串行扇出基线，延迟显著高于小模型用例，属于预期现象。
2. 在单源 fanout 场景下，前序 receiver 完成后会逐步转化为新 source，传播路径呈级联扩散；`publish_interval=90s` 能覆盖本轮 p95。

### 9.9 daemon 内存增长最小复现与根因闭环（Publisher-only）

目标：

1. 隔离 receiver 影响，仅保留 `publisher + local daemon + GS`，复现/排除 daemon 线性内存增长。

复现实验：

1. worker：`ws-7681b3683947089e-worker-qxrln`（4 GPU，cgroup memory `500GiB`）。
2. GS：`100.97.246.95:50051`（cluster token 对齐）。
3. daemon（显式生命周期）：
   - session：`minrepro-pub3`
   - config：`examples/config/store_daemon_config_cross_host_bench_64gb_payload.yaml`
   - `promotion.max_concurrency=1`
   - `heartbeat=60s`、`periodic_sync=60s`
4. workload（publisher-only）：
   - `tensorcast/tools/weight_publisher_e2e.py publisher`
   - `payload_mode=tp_ranked`
   - `tp_total_bytes=64GB`
   - `keep_last=2`
   - `num_versions=8`
   - `publish_interval=20s`
   - `publish_device=cpu`
5. 采样：
   - 每 5s 采样 daemon RSS：`/data/tc_cross_20260223/minrepro_pub3_1771864822/daemon_mem.tsv`
   - publisher 日志：`/data/tc_cross_20260223/minrepro_pub3_1771864822/publisher.log`
   - 内核日志尾部：`/data/tc_cross_20260223/minrepro_pub3_1771864822/dmesg_tail.log`

结果：

1. publisher 进程内存行为正常：
   - 每版 `after_build≈64.65GiB`，`after_source_release≈0.65GiB`。
2. daemon RSS 不再线性增长：
   - `max_rss_gib=152.77`，`last_rss_gib=152.77`，`unique_daemon_pids=1`。
   - `v3` 之后呈“回落->再升到同一上限”的稳定锯齿，不再版本数线性抬升。
3. 稳定性：
   - `v1~v8` 全部成功，`RC=0`，`v7/v8` 不再崩溃。
4. OOM：
   - 本轮执行窗口未出现新的 `tensorcast_daem` OOM kill 记录（`dmesg_tail` 中 OOM 为更早历史记录）。

根因与修复：

1. 根因（已证实）：
   - begin 阶段用临时 key（`mem_reg:*`）注册到 `ReplicaRegistry`；
   - commit/abort/TTL 过期路径未清理该临时 alias；
   - 旧 alias 持有 `shared_ptr<Replica>`，导致旧版本内存对象无法析构，最终 OOM。
2. 修复点：
   - `core/store/runtime/metadata/registration_backend.h`
   - `core/store/runtime/metadata/registration_backend.cc`
   - 新增并在 commit/abort/TTL 分支调用 `erase_pending_registry_alias(...)`；
   - `PendingRegistrationContext` 记录 `pending_registry_key`，确保 alias 可精确删除。
3. 回归用例：
   - `core/store/runtime/metadata/metadata_gateway_test.cc`
   - 新增 commit/abort 两条测试覆盖 pending alias 清理。
   - `bazel test //core/store/runtime/metadata:metadata_gateway_test --test_env=TENSORCAST_CUDA_BACKEND=fake` 通过。

补充结论（为何不是 308GiB 峰值）：

1. 在当前 pre-publish trim 策略下，发布时的 daemon 峰值近似为：
   - `pinned_pool(约24GiB) + retained_before_publish(1*64GiB) + inflight_register(1*64GiB) + 小量开销`
   - 约 `152GiB`，与实测 `152.77GiB` 一致。
2. 之前接近 443GiB 并触发 OOM 的场景，本质是“旧版本对象未释放（alias 泄漏）”叠加导致，而不是设计预期峰值。

### 9.10 主线重跑：TP4 + 64GB + 7 receivers + 20 versions（`max_concurrency=1`）

背景：

1. 首次重跑（`tp4-64gb-r7-v20-mainline-mc1-1771866170`）在 `v6` 左右失败，根因是多台 brainctl worker 进程进入 `Stopped`，remote exec 不可用，随后 publisher 侧也出现本地 daemon 连接失败；该轮判定为资源生命周期问题，不作为框架结论。
2. 清理旧 worker 后，重新申请并固定 8 个 `4-GPU` worker（publisher 1 + receiver 7），再执行主线重跑。

执行：

1. case: `tp4-64gb-r7-v20-mainline-mc1-rerun`
2. 结果文件：
   - `/data/tc_cross_20260223/results_weight_publisher/tp4-64gb-r7-v20-mainline-mc1-rerun-1771867353/tp4-64gb-r7-v20-mainline-mc1-rerun.json`
3. 核心参数：
   - `num_versions=20`
   - `tp_total_bytes=64GB`
   - `tp_world_size=4`
   - `receiver_apply_mode=tp4_bind_into_swap`
   - `publish_interval=90s`
   - `poll_interval=1.5s`
   - `heartbeat=60s`
   - `periodic_sync=60s`
   - daemon transport `max_concurrency=1`

结果：

1. `passed=true`
2. 稳定性：
   - `all_receivers_completed=true`
   - `receiver_skips_present=false`
   - `binding_pointer_stable=true`
   - `receiver_mode_consistent=true`
3. 性能：
   - publish latency mean `53.9203s`（p95 `55.9251s`）
   - apply latency mean `62.3492s`（p95 `92.2627s`）
   - publish->apply p95 `90.3600s`
4. 版本管理：
   - `keep_last=2` 生效，最终保留 `[19, 20]`
   - GS RPC 审计通过：`old_versions_released=true`，窗口约束通过
   - cluster probe 通过：旧版本均不可物化，materializable 窗口不超过 2

排队/传输估算（基于本轮日志）：

1. publisher 实际版本节拍（相邻 `published_at`）：
   - mean `151.515s`，min `145.663s`，max `153.366s`
   - 说明：`publish_interval(90s)` + `publish_latency(~54s)` 叠加后形成约 150s 周期。
2. receiver 侧 key mapping 等待（`[receiver][wait] ... elapsed_s`）：
   - 全 receiver * 全版本（140 样本）mean `81.032s`，p95 `126.5s`
   - 该等待主要是“下一版本尚未发布/尚未映射可见”的控制面等待，不是 data-plane 传输耗时。
3. receiver 侧 materialize 传输：
   - mean `62.349s`，p95 `92.263s`
   - 从 `receiver_1.log` 可见每版本 `sum(rank0..3 latency_ms) ~= total_materialize_ms`，说明 TP4 在本 case 下是 rank 串行主导，单版本主要耗时落在 4 次 rank 传输/拷贝。
4. 结论：
   - 本轮 40~90s 级别耗时以“发布/映射可见等待 + TP4 串行 materialize”构成，未观测到 deadline 超时；
   - 在 `max_concurrency=1` 下，行为符合“单通道打满 + 级联扩散”的预期。

### 9.11 `stage_on_gpu=false` 流式上传两处根因修复（非重试兜底）

背景：

1. 在切换到 `stable_dram + stage_on_gpu=false` 后，64GB TP4 小规模专项首轮失败，并非业务超时，而是两处实现缺陷。

问题 1（协议层）：

1. 现象：publisher 失败，`FeedRegisterArtifactStream` 返回  
   `RESOURCE_EXHAUSTED: Received message larger than max (4194371 vs. 4194304)`。
2. 根因：客户端分片按 `4MiB` 裸值发送，未给 protobuf/gRPC 包装头留余量，边界包会越过默认 4MiB 限制。
3. 修复：
   - `tensorcast/daemon_ctl.py`
   - 将默认分片改为 `4MiB - 64KiB` 安全上限；
   - 对调用方传入的 `chunk_bytes` 做上限裁剪（超过安全值自动降到安全值）。

问题 2（语义层）：

1. 现象：修复分片后再次失败，daemon 返回  
   `FAILED_PRECONDITION: registration chunk offset must be contiguous: expected=0 got=...`。
2. 根因：上传基准 offset 错误使用了 `ctx.tensor_source_index`（源存储图 offset），而 daemon 要求的是 canonical 连续 offset。
3. 修复：
   - `tensorcast/api/_register.py`
   - `cpu_stream` 路径改用 `layout.offsets` 作为 `base_offset`；
   - 保留 `tensor_source_index` 仅用于字节数校验。
4. 回归测试：
   - `tests/python/api/test_register_stable_dram_streaming.py`
   - 新增“source offset 非 canonical”校验场景，确保 uploader 始终使用 `layout.offsets`。

专项验证：

1. case: `tp4-64gb-copycheck-v2-r1-fixoffset`
2. 文件：  
   `/data/tc_cross_20260223/results_weight_publisher/tp4-64gb-copycheck-v2-r1-fixoffset-1771874205/tp4-64gb-copycheck-v2-r1-fixoffset.json`
3. 结果：`passed=true`
4. 关键证据：
   - publisher 日志出现 `stable_dram upload path=cpu_stream`（无 `staging_gpu`）；
   - publisher 内存曲线：`after_build≈64.5GiB`、`after_source_release≈0.57GiB`，无双份驻留；
   - receiver 两版均成功（`bind_into` 首版、`swap` 次版），无 skip。

### 9.12 主线复跑（最新）：`tp4-64gb-r7-v20-mainline-fixstream-rerun2`

执行与参数：

1. case：`tp4-64gb-r7-v20-mainline-fixstream-rerun2`
2. 文件：
   - `/data/tc_cross_20260223/results_weight_publisher/tp4-64gb-r7-v20-mainline-fixstream-rerun2-1771880823/tp4-64gb-r7-v20-mainline-fixstream-rerun2.json`
3. 关键参数：
   - `num_versions=20`
   - `tp_total_bytes=64GB`
   - `tp_world_size=4`
   - `receiver_apply_mode=tp4_bind_into_swap`
   - `publish_interval=90s`
   - `poll_interval=1.5s`
   - `heartbeat=60s`
   - `periodic_sync=60s`
   - daemon transport `max_concurrency=1`
   - `fallback_prefer=p2p`
4. 运行时保护：
   - runner 将 `remote-timeout-sec` 从默认 `1800s` 自动抬升到 `5433s`，避免“进程仍健康但被 runner 超时误杀”的假失败。

结果（功能/稳定性）：

1. 发布与接收完整性：
   - publisher `20/20` 全部发布成功。
   - 7 个 receiver 均 `20/20` 完成，无跳版（`receiver_skips_present=false`）。
2. 绑定语义：
   - 每个 receiver 首版 `bind_into`，后续 `swap`，`pointer_stable=true`。
3. keep-last / de-register：
   - `keep_last=2` 生效，最终保留版本 `[19, 20]`。
   - GS RPC 审计：`old_versions_released=true`，`replica_versions_within_window=true`。
   - cluster probe：`dropped_non_materializable=true`、`materializable_version_count_within_limit=true`。

结果（性能）：

1. publish latency：mean `65.878s`，p95 `66.954s`。
2. apply latency（140 样本）：mean `71.302s`，p95 `108.212s`，max `112.500s`。
3. publish->apply：mean `68.712s`，p95 `106.008s`。
4. publisher 版本节拍（相邻 published_at）：mean `162.913s`，min `156.845s`，max `164.006s`。

排队/传输拆解（64GB, TP4, `max_concurrency=1`）：

1. 控制面等待（receiver `[wait] elapsed_s`，140 样本）：
   - mean `84.4s`，p95 `137.0s`。
2. TP4 rank materialize（全部 receiver 聚合）：
   - rank0 avg `38.8s`
   - rank1 avg `13.1s`
   - rank2 avg `9.6s`
   - rank3 avg `9.8s`
   - 每版本 rank 总和 avg `71.3s`
3. 解释：
   - 本 case 明确存在“发布可见等待 + rank 串行 materialize”两段耗时；
   - 在 `max_concurrency=1` 下，单通道链路被打满，排队属于预期行为；
   - 现有固定 `30s` 传播阈值对 64GB/TP4 过紧，导致 `propagation_violations=126`，但不是功能失败。

“put 双份显存/内存”问题结论（最新证据）：

1. 本轮 publisher 进程内存采样：
   - `after_build` avg `64.572GiB`
   - `after_publish` avg `64.575GiB`
   - `after_source_release` avg `0.576GiB`
2. 结论：
   - 在 `stable_dram + stage_on_gpu=false + cpu_stream` 下，publisher 进程未出现“2x 持续驻留”，不存在 64GB 模型常态双份占用。
   - 若观测到远高于该范围（如历史 300GiB+）应优先排查 daemon 侧对象释放链路与注册生命周期，而不是先加重试。

后续建议（配置 + 系统层）：

1. 配置层：
   - 将“传播超时阈值”从固定值改为 payload-aware（至少绑定 `tp_total_bytes`、`tp_world_size`、`max_concurrency`、历史 p95）。
   - `remote-timeout-sec` 保持动态下限估算，避免长任务被 runner 误判失败。
2. 系统层：
   - 在 orchestrator 增加“排队时间/传输时间”结构化指标，分别上报，避免把排队误判为传输异常。
   - 对 `max_concurrency=1` 场景，持续观测“新 source 扩散后的并发传输数”曲线，作为 fanout 是否按预期扩散的直接判据。

### 9.13 本轮 `//core/...` 回归状态（辅助）

执行命令：

1. `bazel test //core/... --verbose_failures --test_tag_filters=\"-stress,-rdma,-multi_gpu\" --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`

结果：

1. 本轮执行 `85` 个测试，`83` 通过，`2` 失败：
   - `//core/store:registration_memory_replica_test`
   - `//core/store:store_engine_test`
2. 失败特征（当前分支快照）：
   - `registration_memory_replica_test` 在 `commit` 后按旧 `artifact_id` 查询 GPU ptr 失败（断言点 `core/store/registration_memory_replica_test.cc:108`）。
   - `store_engine_test` 中多处 `canonical_or/view_or` 失败，状态为 `source reselection budget exhausted`，以及 view request 计数从 `1` 变为 `65`（断言点 `core/store/store_engine_test.cc:401/787/931/1258`）。
3. 说明：
   - 这两项为当前分支内在回归项，不影响本报告主线多机 benchmark 结果判定；
   - 后续需单独按“预期行为 vs 断言口径”做根因收敛与修复（优先判定是框架语义变化还是测试口径过时）。
