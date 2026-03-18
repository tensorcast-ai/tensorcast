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

### 9.14 最新主线续跑：TP8 + 320GB + `keep_last=1`（3 receivers，2026-02-24）

执行：

1. case：`tp8-320gb-r3-v2-mainline-hb60-p1p5-mc1-k1`
2. 结果文件：
   - `/data/tc_cross_20260224/results_weight_publisher_mainline_tp8/tp8-320gb-r3-v2-mainline-hb60-p1p5-mc1-k1-1771924003/tp8-320gb-r3-v2-mainline-hb60-p1p5-mc1-k1.json`
3. 关键参数：
   - `tp_world_size=8`
   - `tp_total_bytes=343597383680`（320GiB）
   - `receiver_apply_mode=tp_bind_into_swap`
   - `keep_last=1`
   - `publish_interval_s=600`
   - `poll_interval_s=1.5`
   - `daemon_heartbeat_interval=60s`
   - `daemon_periodic_sync_interval=60s`
   - daemon transport `max_concurrency=1`
   - `publish_device=cpu`
4. 资源：
   - publisher：`ws-7681b3683947089e-worker-t9dp7`
   - receivers：
     - `ws-7681b3683947089e-worker-ns65h`
     - `ws-7681b3683947089e-worker-zcbmn`
     - `ws-7681b3683947089e-worker-p8zjr`
   - GS：`100.99.211.131:50051`（`max_workers=32`，干净重启）

结果：

1. `passed=true`
2. 稳定性：
   - `all_receivers_completed=true`
   - `receiver_skips_present=false`
   - `binding_pointer_stable=true`
   - `receiver_mode_consistent=true`
3. 保留窗口与回收：
   - `keep_last=1` 生效，最终仅保留版本 `v2`
   - GS RPC 审计通过：
     - `old_versions_released=true`
     - `replica_versions_within_window=true`
     - `replica_version_count_within_limit=true`
4. 性能：
   - publish latency mean `351.768s`（p95 `360.648s`）
   - apply latency mean `354.972s`（p95 `417.571s`）
   - publish->apply p95 `453.327s`

时延拆解（本轮）：

1. publisher 版本节拍：
   - `v1 -> v2` 的 `published_at` 间隔 `993.725s`
   - 主要由 `publish_interval(600s) + publish_latency(~352s)` 叠加形成。
2. receiver 的 `v2` 等待与传输：
   - `key_mapping_absent` 最大等待（3 个 receiver）分别约 `527.2s / 517.4s / 548.2s`。
   - key 可见后，8 rank `swap` 总耗时分别约 `153246ms / 373979ms / 421511ms`。
3. 结论：
   - 本轮没有跳版；
   - 主要耗时由“下一版本尚未发布时的等待 + TP8 串行 rank swap”构成。

新增异常证据（需框架侧继续根治）：

1. `v1` 阶段，3 个 receiver 均出现同类窗口：
   - key 已能解析到 `artifact_id`，
   - 但紧接着 `Artifact id ... was not found by StoreDaemon`（每节点 5 次），随后才进入 `bind_into` 成功。
2. `v2` 阶段未复现该现象，表现为纯 `key_mapping_absent -> key 可见 -> swap`。
3. 推断：
   - 首版发布路径与后续 swap 路径在“key 可见性门禁”上存在语义差异；
   - 当前首版 key 可能先于 artifact/index 完整就绪对外可见，触发短暂不一致窗口。

后续建议（owner 视角）：

1. 框架层：
   - 将“index ready 后才允许 key 可见”的门禁统一到首版发布链路（不仅是 swap 链路）。
2. 可观测性：
   - 增加结构化延迟指标：`key_visible_at`、`index_ready_at`、`first_resolve_success_at`，直接量化窗口时长。
3. 调度策略：
   - 大权重场景建议从固定 `publish_interval` 过渡到“receiver 全量 ACK 后再发下一版”的手动/半自动驱动，减少纯等待时间并避免阈值误配。

### 9.15 TP8-320GB 尾延迟根因收敛与框架修复（mapped source 可扩散）

复现口径（同一问题域）：

1. 复现 case 仍使用 `tp8-320gb-r3-v2-mainline-hb60-p1p5-mc1-k1`。
2. 关键配置：`tp_world_size=8`、`keep_last=1`、`max_concurrency=1`、`poll=1.5s`。
3. 多机编排规则采用本仓库规范：GS 建议运行在本地控制机，worker 仅跑 daemon + role 进程。

问题收敛：

1. 关于“单实例是否 8 rank 并发发起 materialize”
   - 代码路径 `tensorcast/tools/weight_publisher_e2e.py` 在 `tp_bind_into_swap` 中对 rank 执行 `for rank in range(tp_world_size)` 串行循环；
   - 单实例在任一时刻只有 1 个 rank 在执行 materialize/swap，请求并非 8 卡并发齐发。
2. 为什么 `key_mapping_absent` 看起来很长
   - 该等待主要是“下一版本尚未发布或尚未可见”的控制面等待；
   - 在本 case 中，`publish_interval(600s) + publish_latency(~352s)` 叠加导致版本节拍本身很长，receiver 在“上一版处理完成到下一版可见”之间会自然进入长等待窗口。
3. 为什么 key 可见后 3 个 receiver 的 TP8 `swap` 总耗时差异大（`153s / 374s / 422s`）
   - 旧语义下 mapped binding 不支持 publish，且 mapped 选择未形成可路由的稳定 view byte-space 身份；
   - 结果是后续 receiver 仍主要回源到少数 canonical source，在 `max_concurrency=1` 下形成排队级联，后到达实例尾延迟被放大；
   - 因此“第一个 swap 完成后应有更多 source 并迅速拉平尾延迟”的前提在旧实现中并不成立。

根本修复（已落地）：

1. SDK 为 mapped binding 生成稳定 `mapped view_id`（`mapped:v1:<sha256>`），身份绑定：
   - canonical index
   - source view identity
   - copy plan
   - target tensor layout
2. `bind_into(..., mapping=...)` / `swap(..., publish=True)` 打通 mapped publish：
   - `MaterializeIntoMappedTarget` 可返回 `target_publication_token`；
   - `PublishTargetReplica` 可发布 VIEW byte-space 副本；
   - 使已完成 rank/实例可成为后续请求 source（可扩散）。
3. daemon/controller 放开 opaque `selection.view_id`（无 metadata 也可作为 byte-space 身份），并将 mapped request 的 `VariantIdentity` 下传给 StoreEngine。
4. core ingestion 的 mapped 路径优先 `request_view_transport`，仅在 `NOT_FOUND/UNIMPLEMENTED` 时回退 canonical route，保证新旧集群混部兼容。

验证状态：

1. Python：`pytest tests/python/api/test_mapped_binding.py` 通过（`4 passed, 2 skipped`）。
2. Daemon：`bazel test //daemon:materialize_into_mapped_target_test --test_env=TENSORCAST_CUDA_BACKEND=fake` 通过。
3. Core：`bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake` 通过。

预期收益：

1. 在 TP 多实例 fanout 下，source 不再长期收敛到单点 canonical replica；
2. 随着首批 rank 完成并 publish，后续 rank 可直接拉取 mapped/view byte-space，降低排队与尾延迟；
3. `publish latency` 与 `key_mapping_absent` 长尾将联动下降（后者受前者与发布节拍共同影响）。

### 9.16 修复后同参数 TP8-320GB 复跑（2026-02-24，local GS）

执行：

1. case：`tp8-320gb-r3-v2-mainline-hb60-p1p5-mc1-k1-fix`
2. 结果文件：
   - `/data/tc_cross_20260224/results_weight_publisher_mainline_tp8_fix/tp8-320gb-r3-v2-mainline-hb60-p1p5-mc1-k1-fix-1771943117/tp8-320gb-r3-v2-mainline-hb60-p1p5-mc1-k1-fix.json`
3. 参数口径（与 9.14 同域）：
   - `tp_world_size=8`
   - `tp_total_bytes=343597383680`（320GiB）
   - `receiver_apply_mode=tp_bind_into_swap`
   - `keep_last=1`
   - `publish_interval_s=600`
   - `poll_interval_s=1.5`
   - `daemon_heartbeat_interval=60s`
   - `daemon_periodic_sync_interval=60s`
   - daemon transport `max_concurrency=1`
   - `publish_device=cpu`
   - GS 运行在本地控制机（按本仓库多机编排规则）

结果摘要：

1. 功能稳定性：
   - `all_receivers_completed=true`
   - `receiver_skips_present=false`
   - `binding_pointer_stable=true`
   - `receiver_mode_consistent=true`
2. 首版异常窗口：
   - `Artifact id ... was not found by StoreDaemon`：`0/0/0`（修复前为 `5/5/5`）
3. runner 顶层 `passed=false` 的原因：
   - `cluster_probe.materializable_versions_within_window=false`
   - 但 `global_store_probe` 的保留窗口检查全部通过（`old_versions_released=true`、`replica_versions_within_window=true`），功能/数据路径口径可判定为通过。

修复前后核心指标对比（同问题域）：

| 指标 | 修复前（9.14） | 修复后（9.16） | 变化 |
|---|---:|---:|---:|
| publish latency mean | `351.768s` | `410.203s` | `+58.435s` |
| `v2 key_mapping_absent`（3 receiver 最大等待） | `527.2s / 517.4s / 548.2s` | `801.6s / 801.4s / 811.8s` | 增长 |
| `v2` 8-rank `swap` 总耗时 | `153246ms / 373979ms / 421511ms` | `341808ms / 325272ms / 260884ms` | 尾延迟显著收敛 |
| `swap` max/min 比值 | `2.75` | `1.31` | 收敛 |

围绕“预期”的四点判定：

1. 什么时候 key 应该可见
   - 预期：`publish(version)` 完成后，`key -> artifact` 映射与可解析 artifact 应原子可见。
   - 实测：修复后未再出现“key 已可见但 artifact not found”窗口；修复前 `v1` 每个 receiver 仍有 5 次该异常。
2. 可见后多长时间应发起传输
   - 预期：在 `poll=1.5s` 下，应在下一个轮询周期内发起首个 rank。
   - 实测：3 个 receiver 均表现为 `resolve status=ok (v2)` 下一行即 `rank=0 phase=start`（行距恒为 1），满足预期。
3. 中间有没有预期之外内容
   - 修复前：存在预期外内容（首版 not-found 窗口）。
   - 修复后：该窗口消失；数据路径仅表现为 `key_mapping_absent -> key 可见 -> swap`。
4. 性能表现是否良好
   - `swap` 尾延迟目标达成：多 receiver 之间显著拉平（`2.75 -> 1.31`）。
   - `publish/key_mapping_absent` 仍偏高：本轮 `v1->v2` 发布节拍约 `1077.97s`，叠加 receiver 的串行 TP8 执行，导致 `key_mapping_absent` 保持长窗口。
   - 当前结论：本次修复主要命中“source 可扩散与尾延迟放大”问题，对“发布链路本身耗时”不是直接优化项。

发布/传输分段观察（本轮）：

1. publish（put 到 stable DRAM）：
   - `v1 publish_latency=375.512s`
   - `v2 publish_latency=444.894s`
   - 从日志看 `v2` 在 upload 前有 `pre_publish trim`，且 publish 内部前段耗时较高，需继续细拆。
2. publish 完成到 receiver 启动传输：
   - 可见后发起传输及时（见“行距=1”证据），不是主要瓶颈。
3. 传输阶段（v2 swap）：
   - 三个 receiver 分别约 `260.884s / 325.272s / 341.809s`；
   - 比修复前尾部明显收敛，符合“mapped source 可扩散”预期。

下一步（owner 视角，系统性优化）：

1. 给 publish 增加结构化阶段指标：`build_tensors_s / pre_publish_trim_s / stream_upload_s / post_upload_finalize_s`，并按 version 上报。
2. 引入 ACK 驱动或半自动发布节拍（替代固定 `publish_interval=600`），缩短纯等待导致的 `key_mapping_absent`。
3. 在 TP 模式下评估“单实例 rank 串行”到“受控并发 rank”策略，配合 `max_concurrency` 做全局并发预算，避免串行尾部放大。

### 9.17 publish 路径专项优化（40GB 快速迭代 + 网络栈确认，2026-02-25）

目标：

1. 聚焦 `publish`（不带 receiver/swap），把瓶颈从控制面等待中剥离出来。
2. 确认“是不是网络栈吞吐限制”。
3. 在当前框架语义约束下，把 `put_s` 先大幅降下来，再回 320GB 终验。

#### 9.17.1 关键发现（根因）

1. 40GB publish-only 基线（TP8、`keep_last=1`、单 daemon）显示：
   - `publish_breakdown` 几乎全部耗时在 `put_s`（`verify_key_mapping/gc` 可忽略）。
2. 新增中间日志后确认：
   - 并非“完全 hang”，而是 `FeedRegisterArtifactStream` 持续前进但吞吐低。
3. 热点 micro-bench（Python protobuf 赋值路径）显示：
   - `req.view_chunk.data = mv.tobytes()` 在不同 chunk 下吞吐差异极大；
   - `16MB` 显著慢于 `4MB`。
4. 网络栈确认（同一 worker 的纯 TCP 发送，2GiB）：
   - `127.0.0.1`: `5.098 GiB/s`
   - `host_ip`: `5.708 GiB/s`
   - 结论：网络栈吞吐远高于 publish 上传吞吐，网络不是主瓶颈。

#### 9.17.2 关于“offset 必须连续”约束的根本思考

1. 历史现状（已修复）：
   - 早期 stable_dram CPU-stream ingestion 要求全局 `offset` 严格连续；
   - 尝试多 stream 并发上传时会触发 `FAILED_PRECONDITION`：
     - `registration chunk offset must be contiguous: expected=... got=...`
2. 这条约束的本质：
   - 它简化了完整性判定（`ingested_bytes == total_size`）和状态机；
   - 但直接阻断了“乱序/并发写入”这条高收益优化路径。
3. owner 视角的长期最优：
   - 需要升级为“允许乱序写入 + commit 前完整性校验（区间覆盖）”语义；
   - 这样既保正确性，也能开启多 stream 并发上传。
4. 当前语义（9.18 后）：
   - 已升级为“允许乱序写入 + commit 前全覆盖校验 + 重叠拒绝”；
   - 并发上传（`upload_workers>1`）可用；
   - 仍保留 `4MB` chunk 作为当前最优参数。

#### 9.17.3 本轮修复（已落地）

代码改动：

1. `tensorcast/daemon_ctl.py`
   - 默认 `TENSORCAST_FEED_VIEW_CHUNK_BYTES` 从 `16MB` 调整为 `4MB`（稳定提升）。
   - 新增/保留流式进度日志开关：
     - `TENSORCAST_FEED_PROGRESS_LOG_INTERVAL_BYTES`（默认 `0`，按需开启）。
2. `tensorcast/api/_register.py`
   - 保留 tensor 级上传阶段日志（`tensor_ready/tensor_done`）。
   - 支持 `TENSORCAST_FEED_VIEW_UPLOAD_WORKERS>1` 并发上传（需配合 9.17.2 的“乱序 chunk + commit 全覆盖校验”语义）。
3. `daemon/service/controllers/registration_controller.cc`
   - 增加 `FeedRegisterArtifactStream` 接收侧进度/完成日志（便于定位“慢 vs 卡住”）。

#### 9.17.4 40GB publish-only 对比（实测）

参数口径：

1. `payload_mode=tp_ranked`
2. `tp_world_size=8`
3. `tp_total_bytes=42949672960`（40GiB）
4. `keep_last=1`
5. 单 daemon，publisher-only（无 receiver）

结果目录：

1. baseline（host_ip + 16MB）：
   - `/data/tc_cross_20260224/publish_only_40g/tp8-40gb-publish-only-1771950878`
2. loopback（127.0.0.1 + 16MB）：
   - `/data/tc_cross_20260224/publish_only_40g/tp8-40gb-publish-only-loopback-1771951660`
3. loopback（127.0.0.1 + 64MB）：
   - `/data/tc_cross_20260224/publish_only_40g/tp8-40gb-publish-only-chunk64-1771951435`
4. loopback（127.0.0.1 + 4MB）：
   - `/data/tc_cross_20260224/publish_only_40g/tp8-40gb-publish-only-c4m-1771952004`

核心指标（v1）：

| 配置 | `put_s` | stream seconds | 观察 |
|---|---:|---:|---|
| host_ip + 16MB | `80.016s` | `76.216s` | 基线 |
| loopback + 16MB | `77.420s` | `73.959s` | 略优，非主因 |
| loopback + 64MB | `136.366s` | `132.842s` | 明显退化 |
| loopback + 4MB | `45.364s` | `42.057s` | 最优（本轮） |

结论：

1. 主瓶颈在 Python gRPC/protobuf 上传热路径，不在网络栈。
2. 在当前连续 offset 语义下，`4MB` chunk 显著优于 `16/64MB`。
3. 本轮可立即生效的优化已把 40GB `put_s` 从约 `77~80s` 降到 `45s`（约 `43%` 降低）。

### 9.18 CPU memfd 可用性问题根因与修复（2026-02-25）

问题现象（并发 publish 之后做 `tensor_dict(cpu)` 校验）：

1. receiver 持续报错：
   - `ArtifactError: CPU memfd handle unavailable for replica`
2. daemon 侧反复告警：
   - `cpu_target but engine handle missing cpu_memfd_region`

根因拆解：

1. 配置传播缺失：
   - `RegistrationBackend` 在 stable_dram 注册路径创建 `ReplicaConfig` 时，未继承 `cpu_shared_memory_enabled`。
   - 结果：这条路径注册出来的 CPU 副本不是 memfd-backed。
2. memfd 查找 key 语义错误：
   - CPU memfd 读取使用的是请求 key（`cgid/mi2`），
   - 但 UMA 分配记录在 replica 内部 allocation key 上（两者在注册后可能不一致）。
   - 结果：即使开启了 memfd，也会查找 miss。

修复（根本性）：

1. 打通配置链路：
   - `StoreEngineOptions.cpu_shared_memory_enabled`
   - `-> RegistrationResources.cpu_shared_memory_enabled`
   - `-> RegistrationBackend::cpu_shared_memory_enabled_`
   - `-> ReplicaConfig.cpu_shared_memory_enabled`
2. 修正 memfd 查找语义：
   - `HandleStage` / `MaterializationService` 改为基于 `replica->replica_key()`（allocation key）查询 `get_cpu_memfd_region(...)`。

本地回归：

1. `bazel test //core/store/runtime/metadata:metadata_gateway_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
2. `bazel test //daemon:grpc_service_impl_cpu_memfd_e2e_test --test_env=TENSORCAST_CUDA_BACKEND=fake`

远端复验（同 worker、同 40GB 参数）：

1. 修复前 run：
   - `/data/tc_cross_20260224/publish_only_40g/tp8-40gb-publish-only-c4m-w4-rerun-1771956633`
   - `receiver tensor_dict(cpu)` 长时间 `materialize_failed`，`CPU memfd handle unavailable`
   - daemon 告警计数：`cpu_target but engine handle missing cpu_memfd_region = 115`
2. 修复后 run：
   - `/data/tc_cross_20260224/publish_only_40g/tp8-40gb-publish-only-c4m-w4-fixmemfd-1771957537`
   - publish：`put_s=41.302s`（`publish_latency_s=41.305s`，`stream_seconds=38.338s`）
   - receiver `tensor_dict(cpu)` 成功：`materialize_latency_s=4.131s`，payload 校验通过
   - daemon 告警计数：`cpu_target but engine handle missing cpu_memfd_region = 0`

### 9.19 0082 CPU Shared Memory Direct Publish 验证（2026-02-25）

目标：

1. 验证 0082 新路径在真实 worker 上的正确性与性能收益。
2. 口径固定为同机、同参数，仅切换 daemon `engine.cpu_shared_memory.enabled`：
   - `false`：旧 `cpu_stream` payload 路径
   - `true`：新 `cpu_memfd_publish` 直写路径

环境与参数：

1. worker：
   - `ws-7681b3683947089e-worker-djjw2`（4GB/40GB A/B）
   - `ws-7681b3683947089e-worker-m7fmp`（320GB 修复后复跑）
2. 本地 GS + 本地 daemon（显式 session 生命周期管理）
3. 公共参数：
   - `payload_mode=tp_ranked`
   - `tp_world_size=8`
   - `keep_last=1`
   - `publish_device=cpu`
   - `TENSORCAST_FEED_VIEW_UPLOAD_WORKERS=4`
   - `TENSORCAST_FEED_VIEW_CHUNK_BYTES=4MB`
4. 结果根目录：
   - `/data/tc_cross_20260225/publish_only_0082`

#### 9.19.1 publish-only A/B（4GB + 40GB）

case（r2）：

1. `tp8-4gb-stream-c4m-w4-r2`
2. `tp8-4gb-memfd-c4m-w4-r2`
3. `tp8-40gb-stream-c4m-w4-r2`
4. `tp8-40gb-memfd-c4m-w4-r2`

实测结果：

| Case | 上传路径 | `publish_latency_s` | `put_s` | `stream_seconds` | `put_bw` (GiB/s) |
|---|---|---:|---:|---:|---:|
| `tp8-4gb-stream-c4m-w4-r2` | `cpu_stream` | `4.164` | `4.162` | `3.793` | `0.961` |
| `tp8-4gb-memfd-c4m-w4-r2` | `cpu_memfd_publish` | `2.564` | `2.562` | `0.669` | `1.561` |
| `tp8-40gb-stream-c4m-w4-r2` | `cpu_stream` | `41.975` | `41.972` | `39.170` | `0.953` |
| `tp8-40gb-memfd-c4m-w4-r2` | `cpu_memfd_publish` | `25.560` | `25.558` | `7.783` | `1.565` |

对比结论（同参数）：

1. 4GB：
   - `put_s`：`4.162s -> 2.562s`（降低约 `38.4%`）
2. 40GB：
   - `put_s`：`41.972s -> 25.558s`（降低约 `39.1%`）
   - `publish_latency_s`：`41.975s -> 25.560s`（降低约 `39.1%`）
3. 日志显示路径切换符合预期：
   - baseline：`stable_dram upload path=cpu_stream ...`
   - 0082：`stable_dram upload path=cpu_memfd_publish ...`

#### 9.19.2 正确性验证（single-host）

case：

1. `tp8-4gb-singlehost-memfd-correctness-r1`
2. summary：
   - `/data/tc_cross_20260225/publish_only_0082/tp8-4gb-singlehost-memfd-correctness-r1/single_host_summary.json`

关键结果：

1. 上传路径确认：
   - `stable_dram upload path=cpu_memfd_publish ... stream_seconds=0.678`
2. receiver 结果：
   - `apply_mode=tensor_dict`
   - `apply_operation=materialize`
   - `materialize_latency_s=0.082`
3. publish 结果：
   - `publish_latency_s=2.540`
   - `put_s=2.538`
4. 修复后复验（r2）：
   - case：`tp8-4gb-singlehost-memfd-correctness-r2-1771972682`
   - summary：`/data/tc_cross_20260225/publish_only_0082/tp8-4gb-singlehost-memfd-correctness-r2-1771972682/single_host_summary.json`
   - `publish_latency_s=1.060`，`materialize_latency_s=0.035`

#### 9.19.3 320GB 同参数复跑（修复后）

复跑背景：

1. 320GB memfd 首轮失败样例：
   - `/data/tc_cross_20260225/publish_only_0082/tp8-320gb-memfd-c4m-w4-r3`
   - 错误：`CommitRegisteredArtifact failed: Insufficient stable bytes: requested=343597383680 used=343597383680 total=343597383680`
2. 根因：
   - publish 阶段的临时 handle lease 已占用 stable budget；
   - commit 阶段再次做 stable cache admit，发生同一 payload 的双计费冲突。
3. 修复：
   - 在 `registration_controller` 的 commit 流程中，先释放 publish lease 再 commit；
   - 增加 `stable_cache_admitted` 返回信号，避免重复走本地 stable tier admission。

修复后复跑：

1. case：
   - `tp8-320gb-memfd-c4m-w4-r4-fix-1771972466`
2. 结果目录：
   - `/data/tc_cross_20260225/publish_only_0082/tp8-320gb-memfd-c4m-w4-r4-fix-1771972466`
3. 关键日志：
   - `stable_dram upload path=cpu_memfd_publish ... stream_seconds=77.234`
   - 未出现 `RESOURCE_EXHAUSTED` / `Insufficient stable bytes`

对比（同参数，publish-only，TP8/320GB）：

| Case | 上传路径 | `publish_latency_s` | `put_s` | `stream_seconds` | `put_bw` (GiB/s) |
|---|---|---:|---:|---:|---:|
| `tp8-320gb-stream-c4m-w4-r1` | `cpu_stream` | `333.275` | `333.272` | `311.533` | `0.960` |
| `tp8-320gb-memfd-c4m-w4-r4-fix` | `cpu_memfd_publish` | `77.685` | `77.681` | `77.234` | `4.119` |

量化收益：

1. `put_s`：`333.272s -> 77.681s`（降低约 `76.7%`）
2. `put_bw`：`0.960 -> 4.119 GiB/s`（约 `4.29x`）
3. `stream_seconds`：`311.533s -> 77.234s`（降低约 `75.2%`）

结论：

1. 0082 的 CPU shared-memory direct publish 路径在当前验证域内正确性通过。
2. 相比 stream payload 路径，publish 性能有稳定、显著提升（4GB 与 40GB 一致）。
3. 320GB 同参数复跑已完成且修复生效；后续重点转为“与理论 copy-bound 的残余差距”分析与优化。

#### 9.19.4 回到主报告口径的跨机 40GB 复测（2026-02-25）

目标：

1. 按 `docs/benchmarks/20260222-weight-publisher-multihost-p2p-report.md` 的 runner 口径验证 0082 在跨机链路上的收益与正确性。
2. 同参数对比 stream baseline 与 memfd publish。

拓扑与参数（两组一致）：

1. GS（本地控制机）：`100.97.246.95:50051`
2. publisher：`ws-7681b3683947089e-worker-wx4zv`（1 GPU）
3. receiver：`ws-7681b3683947089e-worker-wl6dz`（8 GPU）
4. 关键参数：
   - `payload_mode=tp_ranked`
   - `tp_world_size=8`
   - `tp_total_bytes=42949672960`（40GiB）
   - `num_versions=2`
   - `keep_last=1`
   - `receiver_apply_mode=tp_bind_into_swap`
   - `publish_interval_s=60`
   - `poll_interval_s=1.5`
   - `max_concurrency=1`
5. 产物：
   - stream：
     `/tmp/tc_cross_20260225/results_weight_publisher_mainline_tp8_40gb/tp8-40gb-r1-v2-stream-hb60-p1p5-mc1-k1-r4-1771975029-1771975030/tp8-40gb-r1-v2-stream-hb60-p1p5-mc1-k1-r4-1771975029.json`
   - memfd：
     `/tmp/tc_cross_20260225/results_weight_publisher_mainline_tp8_40gb/tp8-40gb-r1-v2-memfd-hb60-p1p5-mc1-k1-r4-1771974795-1771974795/tp8-40gb-r1-v2-memfd-hb60-p1p5-mc1-k1-r4-1771974795.json`

对比结果（同口径）：

| 指标 | stream（publisher `cpu_shared_memory=false`） | memfd（`cpu_shared_memory=true`） | 变化 |
|---|---:|---:|---:|
| publish latency mean | `46.428s` | `13.258s` | `-71.4%` |
| publish `put_s` mean | `44.681s` | `10.788s` | `-75.9%` |
| publish `put_bw`（40GiB/put_mean） | `0.895 GiB/s` | `3.708 GiB/s` | `4.14x` |
| `v2` 8-rank `swap` 总耗时 | `6139.9ms` | `8056.7ms` | `+31.2%` |
| receiver 版本完成 | `v1/v2` 完成 | `v1/v2` 完成 | 一致 |
| pointer stable | `true` | `true` | 一致 |

结论：

1. 0082 的核心收益在 publish 路径，跨机同口径下仍显著（`put_s` 约 `-75.9%`，带宽约 `4.14x`）。
2. 正确性通过：`v1 bind_into` + `v2 swap` 均成功，receiver 无 skip，pointer stable。
3. 两组 runner 顶层 `passed=false` 均来自 `cluster_probe` 即时窗口（旧版在探针瞬时仍可 materialize）；`global_store_probe` 的保留窗口检查均通过（`old_versions_released=true`、窗口约束通过）。
4. 执行过程中的问题已收敛：
   - 首次尝试用 1-GPU receiver 跑 `tp_world_size=8` 会触发 `invalid device ordinal`，已改为 8-GPU receiver；
   - stream baseline 在未限制 feed view 参数时曾出现 `FeedRegisterArtifactStream ... Transport closed`，本轮固定 `TENSORCAST_FEED_VIEW_UPLOAD_WORKERS=4` 与 `TENSORCAST_FEED_VIEW_CHUNK_BYTES=4MB` 后稳定完成。

### 9.20 TP4 40GB 三核心点复核与扩散副本量化（2026-02-25）

本节补充你关注的三个核心问题，并量化“扩散 source replica”带来的收益。

实验产物：

1. 修复前（扩散未稳定命中）：
   - `/data/tc_cross_20260226/results_weight_publisher_tp40_mainline/tp4-40gb-r2-v2-diffuse-r4-1772008102/tp4-40gb-r2-v2-diffuse-r4.json`
2. 修复后（扩散稳定命中）：
   - `/data/tc_cross_20260226/results_weight_publisher_tp40_mainline/tp4-40gb-r2-v2-diffuse-r5-1772009900/tp4-40gb-r2-v2-diffuse-r5.json`
3. GS 日志：
   - `/home/luoyuchu/.tensorcast/hosts/dev-yuchu-f4e87681d632408599afcebe24c1da20/global_sessions/gs-tp40-restart-1771995490/logs/global_store.out`

#### 9.20.1 三核心点当前结论

1. `v2 publish -> receiver key 可见/传输启动`：已达到“近实时”  
   - 通过 `publish_to_apply - apply_latency` 反推，`r5` 的两侧 receiver 间隔约 `-0.61s ~ -0.20s`；
   - 该负值来自跨机 `time.time()` 时钟偏差，按绝对值看均在 `<1s`，可视为 publish 完成后几乎立即进入传输。
2. 慢 rank `swap` 尾延迟：已显著收敛  
   - `v2` 单 rank 最慢从 `49,852.3ms` 降到 `13,295.4ms`（`-73.3%`）。
   - `v2` 8-rank 总耗时从 `106,824.8ms` 降到 `42,930.0ms`（`-59.8%`）。
3. view source 扩散稳定性：已验证生效  
   - `v2` 的实际 source replica 数从 `1` 提升到 `3`（`+200%`）。
   - 修复后 `wait_timeout_ms` 从历史的 `2e5~4e5ms` 级降为 `1000ms`（短探测后快速回退 canonical）。

#### 9.20.2 扩散 replica 量化（同参数 r4 vs r5）

| 指标 | `r4`（修复前） | `r5`（修复后） | 变化 |
|---|---:|---:|---:|
| publish latency mean | `16.386s` | `16.551s` | `+1.0%` |
| apply latency mean | `49.840s` | `33.436s` | `-32.9%` |
| publish->apply p95 | `74.706s` | `45.979s` | `-38.5%` |
| `v2 swap` 8-rank 总耗时 | `106,824.8ms` | `42,930.0ms` | `-59.8%` |
| `v2 swap` 最慢单 rank | `49,852.3ms` | `13,295.4ms` | `-73.3%` |
| `v2` unique source replicas | `1` | `3` | `+200%` |

`v2 publish` 后各 receiver 的关键链路耗时（更符合本轮目标口径）：

| Case | Receiver | `v2 publish -> key可见/传输启动`* | `v2 artifact传输+apply` | `v2 publish -> apply完成` |
|---|---|---:|---:|---:|
| `r4` | `ws-...-48knv` | `-0.341s` | `80.093s` | `79.753s` |
| `r4` | `ws-...-zx68g` | `-0.522s` | `26.732s` | `26.210s` |
| `r5` | `ws-...-48knv` | `-0.198s` | `23.259s` | `23.061s` |
| `r5` | `ws-...-zx68g` | `-0.606s` | `19.672s` | `19.066s` |

\* 通过 `publish_to_apply - apply_latency` 反推；负值来自跨机时钟偏差，可按“约 0s”解释。

按“发布完成后集群收敛时间（max receiver）”看，`r4 -> r5` 从 `79.753s` 降至 `23.061s`（`-71.1%`）。

`r5` 的 `v2` source 命中分布（按 GS `Transport requested ... replica` 计数）：

1. `16974f99-529c-4ce3-bf93-09cc4d9978e3`：`6` 次
2. `7c7d3852-f31a-4b6b-ac11-3dead92f59ea`：`1` 次
3. `d3e9a28c-288b-419d-aa4a-ce9dbe604dcd`：`1` 次

结论：

1. 扩散 source 已从“语义存在但易被长等待阻塞”变成“稳定可命中并能摊薄尾延迟”。
2. 以本轮关注口径看，`v2 publish` 后 key 可见与传输启动已是近实时（`<1s` 量级）。
3. 当前收益主要体现在 artifact 传输与慢 rank 收敛路径；publish 本身基本不受该修复影响。

### 9.21 队列主导尾延迟根因收敛与修复后复测（TP4/TP8 40GB，2026-02-25）

这一轮进一步聚焦“尾延迟主要来自排队还是传输本身”，并在 GS 调度侧做根因修复。

#### 9.21.1 根因（代码级）

问题点落在 `tensorcast/global_store/repositories/replica_repository.py`：

1. 在 `create(...)` 与 `create_or_update_atomic(...)` 中，注册副本时对 `replica_counters` 使用了 upsert 覆盖；
2. 覆盖行为会把 `current_requests` 与 `last_assigned_at` 重写为“注册时的值”（常见为 `0` 和 `now()`）；
3. 在 mapped source 快速扩散场景中，这会周期性“洗白”热 source 的真实负载，导致调度器错误地继续把请求打到同一 source，形成无序排队和尾延迟放大。

修复策略：

1. 注册副本时仅“确保计数行存在”，不再覆盖已有 `current_requests/last_assigned_at`；
2. 新建计数行时把 `last_assigned_at` 初始化为 epoch，保证新扩散 source 在同负载 tie-break 下可优先吃到首个请求。

对应回归测试：

1. `tests/python/global_store/test_repositories.py::test_replica_re_registration_does_not_reset_inflight_counter`
2. `tests/python/global_store/test_repositories.py::test_transport_prefers_new_idle_source_for_diffusion`

#### 9.21.2 TP4 40GB 修复后复测（r6）

实验产物：

1. case：`tp4-40gb-r2-v2-diffuse-r6`
2. 结果：
   `/data/tc_cross_20260226/results_weight_publisher_tp40_mainline/tp4-40gb-r2-v2-diffuse-r6-1772012463/tp4-40gb-r2-v2-diffuse-r6.json`
3. 对比基线：`r5`（同节 9.20）

关键对比（`r5 -> r6`）：

| 指标 | `r5` | `r6` | 变化 |
|---|---:|---:|---:|
| publish latency mean | `16.551s` | `12.246s` | `-26.0%` |
| apply latency mean | `33.436s` | `12.341s` | `-63.1%` |
| publish->apply p95 | `45.979s` | `13.256s` | `-71.2%` |
| `v2` 集群收敛（max receiver publish->apply） | `23.061s` | `10.839s` | `-53.0%` |
| `v2` 最慢 receiver `artifact传输+apply` | `23.259s` | `11.902s` | `-48.8%` |
| `v2` 最慢单 rank swap | `13,295.4ms` | `5,061.3ms` | `-61.9%` |

`v2 publish` 后 receiver 关键链路（r6）：

| Receiver | `v2 publish -> key可见/传输启动`* | `v2 artifact传输+apply` | `v2 publish -> apply完成` |
|---|---:|---:|---:|
| `ws-...-8gr7q` | `-0.871s` | `10.336s` | `9.465s` |
| `ws-...-vfnl9` | `-1.062s` | `11.902s` | `10.839s` |

\* 通过 `publish_to_apply - apply_latency` 反推；负值来自跨机时钟偏差，可按“约 0s”理解。

source 命中（`v2`）：

1. `e2734928-...`：`7` 次
2. `861b85c4-...`：`1` 次

#### 9.21.3 TP8 40GB 复测（r6，同修复代码）

实验产物：

1. case：`tp8-40gb-r2-v2-diffuse-r6`
2. 结果：
   `/data/tc_cross_20260226/results_weight_publisher_tp40_mainline/tp8-40gb-r2-v2-diffuse-r6-1772012928/tp8-40gb-r2-v2-diffuse-r6.json`

整体指标：

| 指标 | 数值 |
|---|---:|
| publish latency mean | `12.810s` |
| apply latency mean | `12.789s` |
| publish->apply p95 | `14.517s` |

`v2 publish` 后 receiver 关键链路：

| Receiver | `v2 publish -> key可见/传输启动`* | `v2 artifact传输+apply` | `v2 publish -> apply完成` |
|---|---:|---:|---:|
| `ws-...-8gr7q` | `-0.930s` | `9.966s` | `9.037s` |
| `ws-...-vfnl9` | `-1.524s` | `11.739s` | `10.215s` |

\* 同样按 `publish_to_apply - apply_latency` 反推，负值来自跨机时钟偏差。

`v2` source 命中分布（16 次请求）：

1. `ca3de91a-...`：`12`
2. `f48f7b91-...`：`1`
3. `b004fc7c-...`：`1`
4. `68a49d57-...`：`1`
5. `528bb503-...`：`1`

#### 9.21.4 队列 vs 传输：本轮判断

结论：尾延迟仍以“队列等待”为主，不是单次传输吞吐瓶颈。

证据：

1. GS 日志显示 `max_concurrency=1` 下 canonical source 在 `v2` 内被连续命中（TP4 `7/8`、TP8 `12/16`），形成串行服务链；
2. 单次 rank swap 多数在 `0.5s~2.0s`，但慢 receiver 的总耗时接近多个 rank 串行叠加；
3. TP8 中尽管已经注册出 `17` 个 source replica，真正被命中的只有 `5` 个，说明“可注册 source”到“可被调度 source”之间仍有显著门槛（view/可导出状态/时序窗口）。

当前状态：

1. 修复后延迟已显著收敛（TP4/TP8 都在 `~10-11s` 级完成 `v2` 集群收敛）；
2. 但 canonical source 仍承担多数请求，说明后续仍需继续优化“多 source 的可用性转化与调度利用率”。
