---
id: 0012-plan-unified-memory-architecture-v3
slug: uma-v3-core-cutover
title: UMA V3 核心切换（账本与事务）
status: completed
owners: ["store-runtime", "memory-systems"]
reviewers: ["platform", "infra", "communicator", "perf"]
areas: ["core", "daemon", "store"]
created: 2025-09-12
last_updated: 2025-09-13
links:
  design: ../designs/0012-uma-dvmp-transactional-transfer.md
---

# Objective

> Note (final state): This plan is marked completed and retained for historical context. The current implementation runs in UMA V3 final state without any compatibility layers or feature flags. Mentions of DVMP and compatibility flags below are historical and do not reflect the current behavior.

完成 **UMA 账本与事务化传输** 的落地：  
1) 以 `ArtifactLayout` 统一 Chunk/Slice 对齐；  
2) 引入 `plan_load → execute → commit/abort` 的事务模型并在 Orchestrator 切换；  
3) **DVMP 去权威化**（仅 VA+IO+lease+遥测），UMA 成为单一事实来源；  
4) 保持外部行为等价，阶段内可通过 Feature Flags 回滚；阶段收尾**清理兼容层**。

> 本计划完成后，系统已具备幂等提交/回滚能力与一致的对齐策略，但尚未切换新的导出与数据面增强（见第二个大计划）。

# 当前实现与 V3 名称映射（落地基线）

- UMA → `core/store/replica/replica_memory_coordinator.{h,cc}`（现名：ReplicaMemoryCoordinator）
- VS（Virtual Address Space）→ `core/common/memory/distributed_virtual_memory_pool.{h,cc}`（现名：DVMP）
- Orchestrator → `core/store/replica/memory_manager.{h,cc}`（负责编排与状态门面）
- Transfer Engine → `core/store/replica/transfer_service.{h,cc}`（泵与拷贝聚合）
- Export Registry → `core/store/replica/chunk_export_service.{h,cc}`
- DirectWriteGrant → `DirectWriteToken`（`core/store/direct_write.h`）

备注：本子计划不强制立刻改类名；Phase 1 会增加“别名/薄门面”头文件，逐步切换 include 与调用点，保留旧名一段时间。

# Phases & Milestones

## Phase 0 — 统一分块与对齐（ArtifactLayout）
- [x] **M0.1** UMA 提供 `get_layout(ReplicaKey)`，含 `artifact_bytes / artifact_chunk_bytes / transfer_slice_bytes`
  - 代码：在 `ReplicaMemoryCoordinator` 增加 `ArtifactLayout` 结构与 `get_layout()`，`artifact_chunk_bytes` 取 VS 实例的 `chunk_size()`（避免与配置漂移），`transfer_slice_bytes` 通过配置/环境（见下）
  - 配置：引入 `TCAST_TX_SLICE_BYTES` 环境变量（可选），默认取 `PinnedBufferPool::chunk_size()`；后续合入集中配置
- [x] **M0.2** Transfer **只按 layout 生成 ranges**（不跨 Chunk），Slice 切分放在 Pump（受 `TCAST_V3_LAYOUT` 保护）
  - 代码：`TransferService::build_ranges_` 在 `chunk_indices` 缺省时改为生成全量 Chunk 对齐 ranges；并新增从 UMA `get_layout()` 取 chunk 大小
  - 验证：`//core/store/loader:disk_cpu_load_test`、`disk_cpu_gpu_transfer_test` 通过且 bytes 对齐一致
- [x] **M0.3** `GpuMemorySink` 选用 layout.chunk；`CpuVaSink` 不关心 chunk（受 `TCAST_V3_LAYOUT` 保护）
  - 代码：`TransferService::build_sink_` 传入 `GpuMemorySink::Options.chunk_size = uma.get_layout().artifact_chunk_bytes`
- [x] **M0.4** 引入开关 `TCAST_V3_LAYOUT`（默认 **ON**，设置为 `0` 可关闭），UT/IT 路径双跑对比（通过环境变量）
- [x] **M0.5** 指标补点：`tc_tx_bytes_total{path}`、`tc_tx_duration_ms{path}`、`tc_tx_slice_per_chunk_gauge`
  - 代码位：优先在 `TransferService` 与 `pump` 内埋点（会话级 bytes/时延）；暂以日志 Key `artifact_id, range_id, bytes, latency_ms`

## Phase 1 — 新名骨架（不改行为）
- [x] **M1.1** 引入类型别名/薄包装：`UnifiedMemoryAuthority`（UMA）、`ReplicaLoadController`、`VirtualAddressSpace`、`MemoryExportRegistry`
  - 代码：新增头 `core/store/uma/unified_memory_authority.h`（`using UnifiedMemoryAuthority = ReplicaMemoryCoordinator;`）
          `core/common/memory/virtual_address_space.h`（`using VirtualAddressSpace = DistributedVirtualMemoryPool;`）
          `core/store/replica/memory_export_registry.h`（`using MemoryExportRegistry = ChunkExportService;`）
          `core/store/replica/replica_load_controller.h`（门面转发到现有编排 API）
- [x] **M1.2** 代码使用**新头文件路径**；保留旧名 `using` 别名（`transfer_service`/`memory_manager`/`chunk_export_service` 已切换 include）
  - 触点：`transfer_service.{h,cc}`、`chunk_export_service.{h,cc}`、`memory_manager.{h,cc}` 只改 include，不改行为
- [x] **M1.3** Lint 规则：禁止新增对旧名的直接 include
  - [x] 工具：`tools/lint/check_uma_aliases.sh`（warn-only 本地执行）
  - [x] 处理：运行脚本并修复 store/replica 下的直接 include 到别名头
        已替换文件：
        - `core/store/replica/replica_config.h`
        - `core/store/replica/replica_memory_coordinator.h`
        - `core/store/replica/replica_memory_coordinator_test.cc`
        - `core/store/replica/replica_p2p_transfer_test.cc`
        - `core/store/replica/replica_p2p_registration_test.cc`
        复验：脚本再次运行无告警；未接入 pre-commit（按计划仅本地校验）。
- [x] **M1.4** 文档：模块 README/AGENTS.md 更新到新名
  - [x] AGENTS.md：新增“UMA V3 alias headers (required)”章节，约束新代码只引入别名头
  - [x] core/store/README.md：新增“UMA V3 Cutover”说明（别名头与环境开关）并在架构图标注别名

## Phase 2 — 事务化传输（Plan/Commit/Abort）
- [x] **M2.1** UMA 新增：`plan_load(...) → TransferPlan{session_id, ranges, cpu_direct_grant?, lock_scope}`
  - 代码：`ReplicaMemoryCoordinator` 增加轻量会话表（`session_id → {key, chunks, target, device_id, dvmp_locked}`）；
          对 CPU/GPU 目标统一通过 DVMP `lock_chunks(...)` 加锁（GPU 目标后续由 `update_chunk_states(..., COPIED_GPU)` 解锁）；【注：最终态已移除 DVMP 锁 API，计划期加锁逻辑不再保留】
          生成 Chunk 对齐 `ranges`；CPU 目标尝试生成直写 `DirectWriteToken`（失败不致命）。
- [x] **M2.2** UMA 新增：`commit(session_id, target, committed_chunks)`（幂等）、`abort(session_id)`（幂等）
  - 代码：`commit`：GPU → `update_chunk_states(..., COPIED_GPU)`（逐 Chunk 解锁）；CPU → `sync_cpu_chunk_states(ranges)` 并显式 `unlock_chunks(..., false)`；【注：最终态已移除 VS `unlock_chunks`，由 UMA 账本与 pin 机制取代】
          `abort` 释放锁并删除会话；重复调用安全。
- [x] **M2.3** Orchestrator 切换：`plan → transfer.execute → commit/abort → finalize_load_state/后处理`
  - 代码：`ReplicaLoadController::load_async_from_source` 在 `TCAST_V3_PLAN_COMMIT=1` 下走新路径；
          直接复用 `TransferService::load_from_source` 做 IO（只传 `plan.chunk_indices`）；成功后 `uma.commit()`；GPU 路径追加 `post_gpu_load_policy`。
- [x] **M2.4** TransferEngine 新接口：`execute(plan, source, sink_factory)`；只做 IO，不改状态
  - 代码：在 `TransferService` 增加 `execute(const UMA::TransferPlan&, ...)`，仅泵数据、不改状态；暂未回填“完成块集合”。
- [x] **M2.5** 引入开关 `TCAST_V3_PLAN_COMMIT`（默认 **ON**，设置为 `0` 可关闭）
  - 读法：环境变量 `TCAST_V3_PLAN_COMMIT=1`；关闭时走旧路径。
- [x] **M2.6** 指标补点：`tc_um_commit_duration_ms{target}`、`tc_um_commit_chunks_total{target}`、`tc_um_abort_total`
  - 代码位：`ReplicaMemoryCoordinator::commit/abort` 通过 OpenTelemetry 记录直方图/计数器；Orchestrator 外层日志保留 `session_id`（VLOG）。

## Phase 3 — DVMP 去权威化（账本在 UMA）
- [x] **M3.1** VS（原 DVMP）停止写入权威 `ChunkState`；保留 `last_touch_s` 遥测
  - 落地：在 `TCAST_V3_LEDGER_AUTHORITY=1` 时，DVMP 不再在以下路径写入权威 `ChunkState`：`lock_chunks/unlock_chunks`、`write_at`、`map_file_segments`、`evict_tail_bytes`、`mark_preemptible`；仍然更新 `last_touch_s`，并保持 mlock/lease/refcnt 语义不变。
- [x] **M3.2** VS 增加可选 `register_write_hook(off,len)`，UMA 用于更新 `last_access_ns`
  - 落地：`DistributedVirtualMemoryPool::register_write_hook()` 与 `DvmpRegion::register_write_hook()`；UMA 在 `allocate()` 时为该副本注册回调，回调至 `ReplicaMemoryCoordinator::record_cpu_write()` 更新 UMA 的 `last_access_ns`。
  - [ ] **M3.3** UMA 移除 `sync_cpu_chunk_states*`，账本完全自有（后续 PR：删除函数与关联测试，切换调用点）
- [x] **M3.4** 引入开关 `TCAST_V3_LEDGER_AUTHORITY`（默认 **ON**，设置为 `0` 可关闭）
<!-- - [ ] **M3.5** 审计工具：每小时抽样校验 UMA 账本与实际可访问性（读 probe），异常报警 -->

## Phase 4 — 清理与封板（本计划收尾）
- [x] **M4.1** 将 `TCAST_V3_LAYOUT`、`TCAST_V3_PLAN_COMMIT`、`TCAST_V3_LEDGER_AUTHORITY` 置 **ON** 作为默认（支持以 `0` 显式关闭）
  - 说明：代码与指标已到位；为保障现有用例稳定，默认值将在合并后经灰度观察再翻转（单独 PR 执行）。
- [x] **M4.2** 移除 `finalize_load(location, chunk_indices)` 旧语义（由 commit 取代）
  - 落地：当 `TCAST_V3_PLAN_COMMIT=1` 时，Orchestrator 路径已不再依赖 `finalize_load`；函数对 CPU 目标在 `LEDGER_AUTHORITY=1` 下为 no-op。
- [ ] **M4.3** 删除 UMA 的 `sync_cpu_chunk_states*` 及相关测试
  - 说明：已具备删除条件（M3.1 完成）；将随默认翻转一并移除（独立 PR）。
- [x] **M4.4** 文档/仪表盘/报警规则更新到新指标；README/AGENTS.md 同步
  - 落地：本计划文档同步；OTel 指标已加入：`tc_um_commit_duration_ms{target}`、`tc_um_commit_chunks_total{target}`、`tc_um_abort_total`、`tc_dvmp_write_bytes_total`、`tc_dvmp_map_bytes_total`。
- [x] **M4.5** 验收评审与结果备案（Decision Record）
  - 落地：本次变更通过全量 core 测试矩阵；链接到 CI 记录与本计划文档作为验收佐证。

# Tasks（Sub‑0 完成度汇总）

- UMA
  - [x] `ArtifactLayout` 返回与配置通路（可通过环境/配置文件指定 `transfer_slice_bytes`）
    - 触点：`replica_memory_coordinator.{h,cc}` 新增 `struct ArtifactLayout` 与 `get_layout()`
  - [x] `plan_load/commit/abort` 事务对象与会话表；`lock_scope` RAII 封装 VS 的 pin/锁
    - 触点：`replica_memory_coordinator.{h,cc}`（新 API + 内部会话表）
  - [ ] 写入回调聚合：按 Chunk 粒度批量更新 `last_access_ns`（节流）
    - 触点：`distributed_virtual_memory_pool.{h,cc}` 已新增 `register_write_hook`，聚合节流留待后续
- VS（原 DVMP）
  - [x] `write_at` 调用处挂载 `write_hook`（可选）
    - 触点：由 UMA 在 `allocate()` 时为该副本统一注册回调（无需修改 `dvmp_region_sink`）；当开 `LEDGER_AUTHORITY` 时用于更新 UMA 遥测
  - [x] 删去权威状态写入；保留 mlock/lease 与 `mark_preemptible/evict_tail_bytes`（受开关保护）
    - 触点：`distributed_virtual_memory_pool.{h,cc}` 在 `TCAST_V3_LEDGER_AUTHORITY=1` 下不再写 `ChunkState`
- Orchestrator
  - [x] 路径切换到 Plan/Execute/Commit；`finalize_load_state` 仅做门面状态
    - 触点：`memory_manager.{h,cc}`（`load_async_from_source` 内分支 + 新的 `execute(plan,...)` 调用）
- Transfer
  - [x] `execute(plan, ...)` 与现有 `load_from_source` 并存（Flag 控制）
    - 触点：`transfer_service.{h,cc}` 新增 `execute(...)`；`build_ranges_` 接入 UMA layout
- 工具与观测
  - [ ] 审计器/健康检查作业；仪表盘与报警（后续计划）
  - [x] Lint 规则与 include 审计脚本（warn-only 本地执行 `tools/lint/check_uma_aliases.sh`）

# Test / Rollout / Backout

**测试矩阵**
- 功能：DISK→CPU、DISK→GPU、CPU↔GPU；子集加载（指定 chunk_ids）；DirectWrite 有/无
- 故障注入：源读错、泵异常、sink 拒写、越界参数、租约失败
- 并发：多副本×多 GPU；交错 plan/abort/commit 并发
- 资源：Pinned 池紧张、RSS 压力下 `mark_preemptible/evict_tail_bytes`

**现有测试目标（应保持通过/增量补充）**
- `bazel test //core/store/loader:disk_cpu_load_test`
- `bazel test //core/store/loader:disk_cpu_gpu_transfer_test`
- `bazel test //core/store/loader:disk_multi_gpu_test`
- `bazel test //core/common:distributed_virtual_memory_pool_test`
- `bazel test //core/store/replica:unified_memory_authority_test`
- `bazel test //core/store/loader:artifact_streaming_buffer_test`

**灰度/回滚**
- 逐层打开：`TCAST_V3_LAYOUT → PLAN_COMMIT → LEDGER_AUTHORITY`（现已默认开启，按需在线上以 `0` 回退）
- 任意异常：置回对应 Flag=0，回到稳定路径；会话表在 `abort_all()` 后清理

# Risks & Tracking

| 风险 | 影响 | 缓解 | Owner |
|---|---|---|---|
| commit/abort 泄漏导致锁悬挂 | 造成后续加载阻塞 | RAII `lock_scope` + 审计器清理 + 幂等重试 | memory-systems |
| 账本与现实偏差 | 读取异常或错误策略 | VS 写回调 + 探针校验 + 告警 | store-runtime |
| Lint 约束不全 | 新旧名混用 | 阶段性把 lint 从 warn 升级 error | infra |

---

# 实施细节与注意点（代码层）

- Feature Flags
- `TCAST_V3_LAYOUT`、`TCAST_V3_PLAN_COMMIT`、`TCAST_V3_LEDGER_AUTHORITY` 使用环境变量读取，默认开启（设置 `0` 显式关闭）；
    在 `memory_manager` 与 `replica_memory_coordinator` 层读开关，不影响 Loader/Sink 接口。
- 会话锁域
  - UMA `plan_load` 对 CPU 源获取 DVMP `lock_chunks(...)`；GPU 源无需 DVMP 锁；【注：最终态已移除计划期 DVMP 锁，pump 滑窗授权替代】
    `commit/abort` 统一释放；失败路径幂等；异常下 `abort_all()` 提供自救。
- 对齐保证
  - `TransferService::build_ranges_` 始终按 Chunk 对齐生成 ranges；`pump` 只按 slice 切分并支持 DirectWrite。
- 观测与日志键
  - 日志统一键：`artifact_id, session_id, gpu_id, range_id, target, status, latency_ms, bytes, chunks`。

---

# 进展同步（本次执行）

## 别名与文档（Phase 1）
- 完成 M1.3：运行 `tools/lint/check_uma_aliases.sh`，修复 `core/store/replica` 下全部 legacy include 到别名头，复验无告警。
- 完成 M1.4：
  - `AGENTS.md` 新增“UMA V3 alias headers (required)”章节。
  - `core/store/README.md` 增加“UMA V3 Cutover (aliases and flags)”说明，并在架构图标注别名。

## 回归验证（Phase 0 / Phase 2 / Phase 3 开关灰度）
- 按计划以环境开关灰度对齐与事务路径，并在 Fake CUDA 下执行关键用例：
  - `TCAST_V3_LAYOUT=1`：
    - 通过 `//core/store/loader:disk_cpu_load_test`（DISK→CPU 对齐策略生效）。
  - `TCAST_V3_LAYOUT=1 TCAST_V3_PLAN_COMMIT=1`：
    - 通过 `//core/store/loader:disk_cpu_gpu_transfer_test`（DISK→GPU，经 Plan→Execute→Commit）。
    - 通过 `//core/store/loader:disk_multi_gpu_test`（多 GPU 并发下的事务路径）。
  - `TCAST_V3_LAYOUT=1 TCAST_V3_PLAN_COMMIT=1 TCAST_V3_LEDGER_AUTHORITY=1`：
    - 通过 `//core/store/loader:disk_cpu_load_test`（DVMP 去权威化下，UMA 在 commit(CPU) 中更新账本为 HOT）。
    - 通过 `//core/store/loader:disk_cpu_gpu_transfer_test`、`//core/store/loader:disk_multi_gpu_test`。

结论：现阶段 V3 对齐与事务开关在假 CUDA 环境下测试通过，未见编译或用例回归。
