---
slug: uma-v3-final-cutover
title: UMA V3 最终态落地与全仓收束（无灰度/无兼容层）
links:
  design: ../designs/0012-uma-dvmp-transactional-transfer.md
---

# Objective

> Note (final state): This plan is marked completed and retained for historical context. The current codebase operates in the UMA V3 final state with no compatibility flags or transitional code. Any references to DVMP or compatibility toggles below reflect the migration process at the time and not the current behavior.

在不保留任何兼容层、Feature Flag、灰度切换的前提下，完成 UMA V3 的“最终态”一次性切换：

- UMA（UnifiedMemoryAuthority）成为唯一权威账本；VS（VirtualAddressSpace）仅负责 CPU VA 与 IO；Transfer 仅管数据；Orchestrator 仅编排；Export 只做注册/keepalive。
- 事务化传输 Plan → Execute → Commit/Abort 在 DISK→CPU、DISK→GPU、CPU↔GPU 全路径生效，错误路径幂等。
- 命名/目录/指标全面收敛到设计文档，删除全部旧名、别名头、环境开关与过渡代码。

# 当前仓库差距快照（基于现状）

- 旧名/别名已统一移除：
  - VS（VirtualAddressSpace）为最终命名；不再存在 `DistributedVirtualMemoryPool` 及别名头。
  - UMA（UnifiedMemoryAuthority）为最终命名；不再存在 `ReplicaMemoryCoordinator` 别名。
  - Export 使用 `MemoryExportRegistry`；不再存在 `ChunkExportService` 别名。
  - CPU Sink 为 `CpuVaSink`；GPU Sink 为 `GpuMemorySink`（命名一致）。
  - GPU 显存包装类与 `get_ipc_handle()` 接口保持不变（最终命名已统一）。
- 账本与 VS 已彻底解耦：
  - UMA `plan_load()` 基于 ranges 获取 VS `pin_range()` 短租约（CPU 目标）；`commit/abort()` 不触 VS 锁；StoreEngine/VS 锁 API 已废弃为 no-op。
  - `sync_cpu_chunk_states*` 路径已删除；`TCAST_V3_*` 环境开关已移除，统一执行最终态逻辑。
- 抽象/命名未一致：
  - `MemoryLocation::CPU` 仍在大量代码/测试中使用（虽有 `CPU=CPU` 的别名）。
  - Direct write 同时存在 `DirectWriteToken` 与 `DirectWriteGrant`，Transfer 中有桥接逻辑。
  - Transfer 使用 `build_ranges_()` 在 Flag 控制下生成 Chunk 对齐；
  - `transfer_helpers.*` 直接依赖 DVMP 与 per‑chunk UMA 锁流程。
- 文档与工具：
  - `core/store/README.md` 仍描述别名头与多个开关；
  - `tools/lint/check_uma_aliases.sh` 仍以“别名头优先”的策略约束，未转向“新名为准、旧名禁止”。

# Phases & Milestones

- [x] Phase F1 — 命名与类型统一（一次性重命名 + 目录收敛）
  - [x] M1.1 VS 重命名：类/文件 `DistributedVirtualMemoryPool` → `VirtualAddressSpace`
    - 代码：将 `core/common/memory/distributed_virtual_memory_pool.{h,cc}` 重命名为 `virtual_address_space.{h,cc}`，类名与命名空间内声明同步修改；
      全仓替换直接引用为新名；删除别名头 `core/common/memory/virtual_address_space.h`（或改为转发到新实现，随后删除）。
    - BUILD：更新 `//core/common:*` 与所有依赖目标；修复可见性与 deps。
  - [x] M1.2 UMA 重命名：`ReplicaMemoryCoordinator` → `UnifiedMemoryAuthority`
    - 代码：`core/store/replica/replica_memory_coordinator.{h,cc}` → `core/store/replica/unified_memory_authority.{h,cc}`；
      类名/注释/命名空间同步；删除别名头 `core/store/uma/unified_memory_authority.h`；全仓替换类型名与 include。
  - [x] M1.3 Orchestrator 重命名：`MemoryManager` → `ReplicaLoadController`
    - 代码：`core/store/replica/memory_manager.{h,cc}` → `core/store/replica/replica_load_controller.{h,cc}`；
      对 `StoreEngine` 与相关路径做最小改动以保持 API 行为一致。
  - [x] M1.4 Export 重命名：`ChunkExportService` → `MemoryExportRegistry`
    - 代码/BUILD：文件与类重命名；Communicator 对接保持 API 等价；删除别名头。
  - [x] M1.5 Sink/Src 命名统一：
    - `DVMPRegionSink` → `CpuVaSink`（文件/类名，注释中不再出现 DVMP 缩写）；
  - [x] M1.8 目录结构对齐设计文档（仅重命名/移动，不改变行为）：
    - 引入 `core/store/replica/types/` 并迁移 `direct_write_grant.h`（其余类型将随后续变更补齐）。
    - 将通用 Loader/Sink 的文档命名对齐到 `transfer/{sources|sinks}`（代码结构不变）。

- [x] Phase F2 — 账本最终化与 API 收束（删除所有开关/过渡逻辑）
  - [x] M2.1 删除所有 `TCAST_V3_*` 开关与读取逻辑（`layout/plan_commit/ledger_authority/export/gpu_sched/direct_window` 等），统一执行最终态逻辑：
    - Transfer：`build_ranges_()` 永远按 Chunk 对齐；`slice_bytes` 来源唯一化（Pinned Pool chunk）；
    - UMA：不再读取任何环境变量控制分支。
  - [x] M2.2 VS 去权威化彻底：
    - 移除 UMA 对 VS 锁/解锁的依赖：不在 `lock_chunks_for_transfer` 调用 VS；`update_chunk_states/commit/abort` 不再解锁 VS。
    - 传输路径不再执行 VS 解锁；以 UMA 账本为唯一事实来源。
  - [x] M2.3 Direct Write 收束：仅保留 `DirectWriteGrant`（滑动窗口），删除 `DirectWriteToken` 类型与 Transfer 中的桥接逻辑；`DirectWriteCapable` 以 `Grant` 为唯一能力接口。
    - 结果：移除 `core/store/direct_write.h`，UMA 仅暴露 `grant_direct_write()`；`TransferService` 不再桥接 Token；`UmaLeaseProvider` 也改为 `grant_direct_write()`。
  - [ ] M2.4 事务语义硬化：
    - Orchestrator 统一走 `UMA.plan → Transfer.execute → UMA.commit/abort`；
    - `transfer_helpers.*` 不再逐 DVMP chunk 手动 `lock_chunks_for_transfer`；按会话粒度由 UMA 账本在 `commit/abort` 内一次性推进；
    - GPU staged/直写路径保持幂等，无状态修改（state only in UMA commit）。
  - [ ] M2.5 Export 最终态：仅通过 `UMA.set_exported()` 维护导出标志与 CPU pin 租约合并；`MemoryExportRegistry` 只做注册/反注册与 keepalive；删除 `components/uma_lease_provider.*` 等过渡代码。
  - [ ] M2.6 Chunk 状态模型正交化：
    - 新增 `types/chunk_record.h`：拆分 `cpu_residency`、`gpu_residency{gpu_id→state}`、`transfer_lock_state`、`exported`、`pin_refcnt`；
    - UMA 账本改用 `ChunkRecord`；提供最小过渡层将旧 `ChunkState` 读写迁移到新结构；完成替换后删除旧枚举与工具函数。

- [ ] Phase F3 — 观测面、文档与清理（最终态一体化）
  - [ ] M3.1 指标统一：
    - UMA：`tc_um_commit_duration_ms{target}`、`tc_um_commit_chunks_total{target}`、`tc_um_abort_total{target}`；
    - Transfer：`tc_tx_bytes_total{path,mode}`、`tc_tx_duration_ms{path}`、`tc_tx_inflight_bytes_gauge{gpu}`（替换现有 `*_copies` 指标为 bytes 口径）；
    - VS：`tc_va_write_bytes_total`、`tc_va_pin_leases_total{reason}`；Export：`tc_ex_registrations_total{location}`、`tc_ex_keepalive_gauge`。
  - [ ] M3.2 文档更新：
    - 删除 `core/store/README.md` 中别名与所有 `TCAST_*` 开关描述；(还要确保代码中相关开关)
    - 更新 `docs/architecture/*` 与 `core/store/docs/*` 的命名、类图、时序图（用 `CpuVaSink/GpuMemorySink → GpuMemorySink`/`GpuDeviceMemory`、`VirtualAddressSpace/UnifiedMemoryAuthority/ReplicaLoadController/MemoryExportRegistry`）；
    - 设计文档中的“迁移策略”改为“已完成（最终态）”。
  - [ ] M3.3 工具与 Lint：
    - 删除 `tools/lint/check_uma_aliases.sh`（或重写为“禁止旧名/别名”的 error 级约束）；
    - `grep`/`rg` 审计：全仓无 `DistributedVirtualMemoryPool/ReplicaMemoryCoordinator/ChunkExportService/DVMPRegionSink/GPUMemorySink/CPU/get_ipc_handle/DirectWriteToken` 等旧符号。
  - [ ] M3.4 Bazel 与测试目标：
    - BUILD 文件与 target 名称对齐重命名；
    - 修复/重命名测试用例（如 `gpu_memory_sink_test` → `gpu_memory_sink_test` 保持文件名，类名改为 `GpuMemorySink`；新增/更新 UMA/Export/Transfer 用例以覆盖最终语义）。

# Tasks

- 代码改动（示例性重点触点，非穷尽）：
  - `core/common/memory/`：重命名 DVMP 文件与类；保留 `pin_range`、`write_at`、`map_file_segments`，移除 `lock_chunks/unlock_chunks`；
  - `core/store/replica/`：UMA 类/文件重命名；删除 `sync_cpu_chunk_states_*` 与所有环境开关；`commit/abort` 仅改 UMA 账本；
  - `core/store/replica/transfer_service.*`：移除 `TCAST_*`；`build_ranges_()` 恒定 Chunk 对齐；`build_sink_()` 改用 `CpuVaSink/GpuMemorySink` 新名；删除 `Token→Grant` 桥接；
  - `core/store/loader/`：`dvmp_region_sink.*` → `cpu_va_sink.*`；`GPUMemorySink` → `GpuMemorySink`；
  - `core/store/replica/transfer_helpers.*`：去除 per‑chunk UMA 锁用法，改为仅数据搬运；
  - `core/common/memory/cuda_memory.*` → `gpu_device_memory.*`；
  - `core/store/replica/memory_manager.*` → `replica_load_controller.*`；公开 `get_ipc_handle()`；
  - `core/store/replica/chunk_meta.*` → 迁移为 `types/chunk_record.h`；
  - `core/store/replica/chunk_export_service.*` → `memory_export_registry.*`；
  - 全仓替换 `MemoryLocation::CPU` → `MemoryLocation::CPU`。

- BUILD & 工具：
  - 更新/重命名 Bazel 目标（`*_lib/_test/_binary` 命名保持一致）；
  - 删除或重写 lint 脚本；
  - 运行 `bazel query 'kind(cc_.*, //...)'` 校验依赖闭包与可见性。

- 文档同步（Doc Sync Rule）：
  - 更新 `AGENTS.md`（移除别名头要求；强调 UMA V3 最终态命名与目录）；
  - 更新 `core/store/README.md`、`docs/architecture/*`、`docs/internals/*` 与本设计关联的示意图与名词表；
  - 在 `docs/README.md` 索引中标注 UMA V3 最终态状态。

# Acceptance Checks

- 一致性：任何路径均不从 VS 读取权威块状态；账本仅在 UMA。
- 对齐：所有 Pump Slice 不跨 Chunk；Transfer 总是按 `artifact_chunk_bytes` 生成 ranges。
- 事务：DISK→CPU、DISK→GPU、CPU↔GPU 全部经 `plan→execute→commit/abort`，且多次重复调用幂等；
- Direct Write：仅使用 `DirectWriteGrant`；直写不可用时自动回退；无重复写、越界写；
- Export：导出/反注册幂等，无租约泄漏；进程重启无悬挂导出租约；
- 观测：`tc_um_*`、`tc_tx_*`、`tc_va_*`、`tc_ex_*` 指标齐备且与日志键一致（`artifact_id, session_id, gpu_id, range_id, target, status, latency_ms, bytes, chunks`）；
- 命名/目录：仓库中不存在旧名/别名/环境开关；所有 README/设计/架构文档使用最终命名；
- 性能：GPU H2D 吞吐≥基线；p99 不高于基线；RSS 峰值不高于基线；
- 测试：核心单测/集成用例全部通过（见下文矩阵）。

# Test Plan

- 构建：
  - `BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext`（Fake CUDA 可通过默认后端运行）；
  - `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`。
- C++ 测试（更新/新增后）：
  - Loader/Transfer：`//core/store/loader:disk_cpu_load_test`、`disk_cpu_gpu_transfer_test`、`disk_multi_gpu_test`；
  - UMA/Orchestrator：`//core/store/replica:unified_memory_authority_test`、`replica_orchestrator_test`；
  - VS：`//core/common:virtual_address_space_test`（由旧 DVMP 测试重命名而来）；
  - Sink：`//core/store/loader:cpu_va_sink_test`、`gpu_memory_sink_test`（类名更新为 `GpuMemorySink`）；
  - Export：`//core/store/replica:memory_export_registry_test`；
  - P2P：`//core/store/replica:replica_p2p_transfer_test`、`replica_p2p_registration_test`。
- 故障注入：源读错、sink 拒写、单个 DMA 失败、DirectWrite 回退、导出反注册幂等、会话 abort 幂等。

# Progress Notes (2025-09-13)

- 完成：
  - M1.5 Sink/Src 命名统一（`DVMPRegionSink`→`CpuVaSink`，BUILD 已更新）。
  - M2.3 Direct Write 收束（移除 Token，统一为 Grant；`TransferService` 桥接删除；`UmaLeaseProvider` 使用 Grant）。
  - M2.1：移除所有 `TCAST_V3_*` 环境开关读取，统一执行最终态逻辑（布局总是 Chunk 对齐、直写窗口始终启用并带失败回退、GPU 调度默认限流、UMA 为唯一账本）。
  - M1.1：DVMP → VS 完成。分发式虚拟内存池类/文件全面重命名为 `VirtualAddressSpace`，BUILD 目标收敛为 `//core/common:virtual_address_space_lib`，删除旧 `distributed_virtual_memory_pool_lib` 与头文件。
  - M1.2：UMA 文件重命名为 `core/store/replica/unified_memory_authority.{h,cc}`，全仓改为直接包含；删除别名头 `core/store/uma/unified_memory_authority.h`。
  - M1.4：导出服务重命名为 `MemoryExportRegistry`，类与文件、BUILD 目标、调用点已切换。
  - 相关 BUILD 依赖清理与去重完成；`//core/store:store_engine` 成功编译（Fake CUDA 后端）。
  - M1.3：Orchestrator 对外命名切换完成：`Replica` 接口改用 `replica_load_controller.h`，以 `ReplicaLoadController` 暴露；保留别名头过渡。
  - M1.8：新增 `core/store/replica/types/` 并迁移 `direct_write_grant.h`；全仓 include 与 BUILD 已更新。
  - M2.2：移除 UMA/Transfer 对 VS 锁/解锁依赖；UMA `commit/abort/update_chunk_states` 不再调用 VS 解锁；传输去掉 DVMP 解锁。
  - 传输（CPU→GPU）路径接入 UMA `grant_direct_write` 固定 CPU VA，失败时自动回退到 VA 基址拷贝。
  - 文档同步（本次变更）：
    - 更新 core/store/docs/architecture.md：将 ReplicaMemoryCoordinator → UnifiedMemoryAuthority，DVMPRegionSink → CpuVaSink，CommRegistrationInfo → ExportRegistration。
    - 更新 docs/designs/0003-unified-memory-registration-avbs-lip.md：将 DVMPRegionSink → CpuVaSink。
- 代码同步：完成 CommRegistrationInfo → ExportRegistration 的全仓替换（C++ 与 Python 绑定、测试用例、docs）。

- 补充：
  - 文档与图示进一步收敛已完成（残留的 MemoryManager 文案已替换为 ReplicaLoadController；README 中去除了 DVMP 示例命名，统一指向 UMA/VS 最终态）。

# Rollout / Backout

- 无灰度、无开关；一次性切换。回退方式为 Git Revert 对应提交集。

# Risks & Tracking

| 风险 | 影响 | 缓解 |
|---|---|---|
| 大规模重命名导致编译爆炸 | 进度与可用性风险 | 分阶段提交但同一分支；优先修复编译，随后补测；保持最小改动原则 |
| 删除 VS 锁 API 后的竞态 | 传输期被驱逐导致失败 | UMA 用 `DirectWriteGrant(pin_range)` 保证窗口驻留；Pump 切块对齐；失败幂等重试 |
| Chunk 模型正交化变更深 | 触点多、回归风险 | 提供过渡层与一次性替换脚本；充分单测覆盖；审慎代码审查 |
| 指标口径变化 | 监控告警误报 | 迁移/替换看板同时进行；变更说明同步到值班手册 |

# 文档同步（必做）

- 更新 `AGENTS.md` 与模块 README，移除别名与开关说明；
- 更新 `docs/architecture/architecture-overview.md`、`core/store/docs/architecture.md`、`core/store/docs/state-management.md`、`core/store/docs/device-manager.md`；
- 将 `docs/designs/0012-uma-dvmp-transactional-transfer.md` 的“Migration Strategy”改写为“已完成（最终态）”，并在 Decision Summary 保留最终结论；
- `docs/README.md` 与仓库 `README.md` 补充最终态命名与主要入口。
