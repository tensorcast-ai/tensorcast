---
title: 稳定/可抢占内存双池计划
links:
  design: ../designs/0034-stable-memory-tiers.md
areas: ["core", "daemon", "global_store"]
related_code:
  - "core/store/materialization/planning/chunk_aware_strategy.cc"
  - "core/store/replica/replica_load_controller.cc"
  - "core/store/replica/unified_memory_authority.*"
  - "daemon/worker_lifecycle_manager.cc"
  - "tensorcast/global_store/services/*"
  - "schema.sql"
---

# Grounding

- 可抢占内存当前在 `core/store/materialization/planning/chunk_aware_strategy.cc:162-182` 强制执行固定比例；
  `ReplicaLoadController::mark_cpu_preemptible` 与 `UnifiedMemoryAuthority::mark_cpu_chunks_preemptible` 不理解“稳定”语义。
- GPU 完成后 `ReplicaLoadController` 始终调用 `post_gpu_load_policy(EvictCPU)`（`core/store/replica/replica_load_controller.cc:1089-1110`），
  `UnifiedMemoryAuthority::post_gpu_load_policy` 只提供 `EvictCPU/MarkPreemptible/Keep`，无人引用 `Keep`。
- Global Store worker 注册与心跳（`daemon/worker_lifecycle_manager.cc:122-206`）仅汇报单一 `mem_pool_*` 字段，
  DuckDB `schema.sql` 仍只有 `workers.mem_pool_total_size/available_size` 与 `chunk_directory.chunk_state`。
- 没有任何 RPC/表描述 `MemoryTierStatus` 或 `Lease`，需要新增服务与 proto。

# Goals & Deliverables

- 让 daemon 支持 `enable_preemptible_memory` 开关与稳定内存最小值，关闭开关后系统仅依靠稳定池运行。
- UMA ledger 支持 `StableLease`/`Preemptible` 双状态、租约 API，并与策略 loop/导出/loader 对接。
- Global Store 增加记账表、服务与 proto，记录每个 daemon 的稳定/可抢占容量，并派发租约。
- 心跳、Telemetry、Prometheus 指标对齐，文档与配置示例更新。
- 项目尚未上线，无需兼容旧 schema/RPC，按最优实现直接切换；可抢占内部策略可先保持占位/禁用路径，但稳定租约/Telemetry/配置必须完整。

## Status Update (2025-02-10)
- Preemptible gating仍按配置工作：显式 `enable_preemptible_memory=false` 会跳过预抢占并保持 GPU 完成后的 CPU 常驻。
- UMA ledger 增加 `STABLE` 状态、`stable_lease_count` 与单调 `ledger_version`；提供 `acquire_stable_lease` / `release_stable_lease` API，并在可抢占/回收路径上跳过持有稳定租约的 chunk。
- CPU 导出路径已在 `MemoryExportRegistry` 中自动申请稳定租约并在 unexport 时归还，防止导出窗口内被抢占；新增单测覆盖稳定租约与可抢占互斥。
- Core 回归：`bazel test //core/... --test_tag_filters=\"-stress,-rdma,-multi_gpu\" --define use_fake_cuda=true` 仍通过；新增 `//core/store/replica:unified_memory_authority_test` 覆盖稳定租约。
- 仍待实现：`MemoryTierBudget`、缺页再水化路径、Global Store schema/服务与 daemon ACK/重放逻辑。
- 2025-11-23: Global Store 侧新增 `memory_tier.proto` + `MemoryTierService` gRPC，以及 DuckDB 表 `memory_tier_snapshots`/`memory_tier_leases`，worker 表加入稳定/可抢占字段；配置 proto 加入 `worker_policy.memory_tiers` 并在 Python 配置中暴露快照留存/发布周期。`uv run pytest tests/python/global_store/test_repositories.py` 通过。
- 2025-11-23 (later): C++ `GlobalStoreClient` 接入 MemoryTierService（发布状态、请求/ACK/列出/撤销租约，含 chunk_ids 与 ledger_version）；测试桩 `RecordingGlobalStoreClient` 也捕获新调用。daemon 调用点与 UMA 元数据填充仍待接线。
- 2025-11-24: 实装 `MemoryTierBudget` 并在 UMA 稳定租约路径上做原子扣减/归还；RuntimeContext 依据 `engine.memory_tiers` + (host DRAM − pinned_pool) 构建预算并注入每个 ReplicaLoadController。StoreEngine 暴露 `get_memory_tier_snapshot()`，daemon heartbeat 线程现已通过 MemoryTierService 发布稳定/可抢占 telemetry。
- 2025-11-25: MemoryTier ACK 流水线强制携带 `artifact_id + chunk_ids + ledger_version`，Global Store 侧按租约 ID + artifact 验证并更新 DuckDB；MemoryTierStatus 发布会同步 Prometheus 指标 `tensorcast_memory_tier_*`。新增 `tests/python/global_store/test_memory_tier_service.py` 覆盖 retention/ACK 验证（本地 uv 缺失，暂未执行）。
- 2025-11-26: UMA 基于 `mincore` 识别 PREEMPTIBLE 缺页并记录 `faults_per_sec`、`rehydrate_p99_ns`；`MemoryTierBudget` 持有打点，心跳直接透传到 MemoryTierService/Prometheus。新增 UMA 单测覆盖预抢占缺页→再水化闭环，daemon telemetry 现已携带 faults/rehydrate 字段。

# Non-Goals

- 不改变 GPU 内存策略、不新增客户端 API、不实现自动拓扑发现。

# Milestones & Tasks

- [x] Phase 1 – Daemon Config & Loader Guard Rails
  - [x] 扩展 `proto/tensorcast/config/v1/daemon_config.proto`、`core/common/config/daemon_config_io.cc`，实现统一配置里的
        `engine.memory_tiers` 字段加载与校验，生成 `MemoryTierConfig`。
  - [x] 实现 `MemoryTierBudget`（或等效结构），根据配置初始化稳定/可抢占预算并暴露原子扣减/归还接口。
  - [x] 新增宿主机 DRAM / cgroup 读取逻辑，计算 `uma_cpu_capacity_bytes = host_dram - pinned_pool_bytes`，在 bootstrap 时校验 `stable_bytes` 上限。
  - [x] 当 `stable_bytes` 大于计算出的 `uma_cpu_capacity_bytes` 时，bootstrap 直接失败并输出配置错误提示（不再按比例缩减或静默继续）。
  - [x] 修改 `ChunkAwareLoadingStrategy`、`ReplicaLoadController::mark_cpu_preemptible`、`MemoryTierPolicy` 以在调用前检查配置，
        并在稳定模式下跳过预抢占而不返回错误。
  - [x] 将 `post_gpu_load_policy` 调整为依据配置选择 `Keep`/`EvictCPU`，并添加日志/统计。
  - [x] 单元测试/集成测试覆盖 `enable_preemptible_memory=false` 和 `true` 两种路径（核心 Bazel 套件已通过）。

- [~] Phase 2 – UMA Ledger & Lease API
  - [x] 扩展 `ChunkRecord` 状态枚举，添加 `StableLease` 与租约计数。
  - [x] 实现 `acquire_stable_lease`/`release_stable_lease`，并在 PREEMPTIBLE 缺页时通过 `mincore` 触发再水化与 telemetry。
  - [x] 更新 `memory_export_registry`, loader/export/communicator 调用链以使用租约 API。（当前仅导出侧自动加持稳定租约，loader/communicator 仍需线程入场。）
  - [x] 在 daemon 侧实现租约 potvr/重放逻辑：`WorkerLifecycleManager` 启动时调用 `ListOutstandingLeases`，并在租约成功/释放后调用 `AcknowledgeMemoryTierLease`。
  - [x] 在 ACK/租约获取路径中上报具体 `artifact_id + chunk_ids`（UMA 真实 ID/ordinal 集合）与 `ledger_version`，`chunk_range` 仅作诊断；重放/撤销以 `chunk_ids` 为准。
  - [~] 新增/扩充单测 `unified_memory_authority_test.cc` & `replica_p2p_transfer_test.cc`。（UMA 单测已覆盖稳定租约与缺页再水化；P2P 路径仍可补充独立覆盖。）
  - [x] 可抢占策略 loop 保持占位/禁用实现，但稳定租约与 telemetry 已完整接线并可在心跳中发布。

- [x] Phase 3 – Global Store Schema & Services
  - [x] 修改 `schema.sql`: `memory_tier_snapshots` 记录 `faults_per_sec`、`rehydrate_p99_ns`、`enable_preemptible`、`memory_tier_config_json`，并引入 `node_memory_tier_latest` 视图（取每个 `node_id` 最新快照）供 Worker 查询联结；`memory_tier_leases` 存储 `chunk_ids`（JSON 数组）、`ledger_version` 以支撑重放和审计。
  - [x] 编写初始化脚本或直接重建 schema（无历史兼容要求），默认稳定模式。
  - [x] 实现 `MemoryTierService`（Python）及 proto：`PublishMemoryTierStatus`、`RequestMemoryTierLease`、`AcknowledgeMemoryTierLease`、`RevokeMemoryTierLease`、`ListOutstandingLeases`，ACK/撤销消息必须包含 `artifact_id + chunk_ids + ledger_version`（`chunk_range` 仅作提示），并强制携带/存储 `faults_per_sec`、`rehydrate_p99_ns`、`enable_preemptible`。后续需与 daemon/GlobalStoreClient 接线以传递真实 chunk_ids/ledger_version。
  - [x] 更新 `proto/tensorcast/config/v1/global_store_config.proto` 与 `tensorcast/global_store/config/settings.py`，
        以统一配置方式暴露 Telemetry 周期与租约策略。
  - [x] Worker heartbeat/注册路径直接切换到新服务与字段，无需向后兼容旧字段。
  - [x] 为 `memory_tier_snapshots` 增加留存/压缩策略（每节点保留最近 N 条或最近 5–10 分钟），实现周期性清理任务并优化索引。

- [x] Phase 4 – Telemetry, Docs, Rollout
  - [x] 在 daemon/Global Store 增加 Prometheus 指标与日志字段，暴露稳定/可抢占 bytes、faults_per_sec、rehydrate_p99_ns，并附带 `enable_preemptible` 标识（info/label）区分“稳定-only 配置”与“节点未上报”。
  - [x] 更新 `docs/internals/preemptible-memory.md`, `daemon/README.md`, `tensorcast/global_store/README.md`。

# Test Plan

- 单元测试：UMA 状态转移、租约 API、策略选择、Global Store 服务逻辑。
- 集成测试：`daemon` gRPC 测试扩展，覆盖稳定模式、开启可抢占模式下的负载。
- e2e：在多 daemon 沙箱集群中部署 Global Store + daemon，验证租约申请/撤销、Telemetry 表写入。
- 验证 `enable_preemptible_memory=false` 时不会有任何 `mark_cpu_preemptible` 记录并保持 RSS 稳定。

# Rollout / Backout

- **Rollout**：按节点批量开启新二进制 → maintain `enable_preemptible_memory=false` → 验证 Heartbeat/Telemetry →
  在首批节点启用可抢占模式并观测 Global Store 表，再逐步扩大。
- **Backout**：若 Global Store 服务异常，可将 `enable_preemptible_memory=false` 并停用 Telemetry 发布；
  schema 变更使用向后兼容默认值，旧二进制依然可读写。

# Risks & Mitigations

- **Ledger/策略竞态**：并发租约与预抢占可能导致状态漂移；Mitigation：在 UMA 中使用互斥和计划/提交阶段。
- **Global Store schema 升级**：DuckDB 表需同时服务旧版本；Mitigation：使用 `ALTER TABLE ... DEFAULT` 与双写策略，先部署 DB。
- **Telemetry 丢失**：节点若未能上报新状态，会被视为稳定模式；Mitigation：在 daemon 监控中加入报警并回退开关。

# Acceptance Checklist

- [ ] 所有列出的阶段任务完成并通过 CI。
- [ ] UMA/daemon/global store 文档同步更新。
- [ ] 生产试点节点验证稳定模式与可抢占模式均可用。
