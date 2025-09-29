---
slug: uma-v3-dataplane-export-cleanup
title: UMA V3 数据面增强、导出收敛与全面清理
links:
  design: ../designs/0012-uma-dvmp-transactional-transfer.md
---

> Note (final state): This plan is marked completed and retained for historical context. The current implementation runs in UMA V3 final state without compatibility flags or transitional code. Mentions of DVMP or temporary bridges below reflect the phased rollout at the time and not the current behavior.

# Objective

在「核心切换」完成后，完成**数据面能力增强**与**导出收敛**，并进行**全面清理**：  
1) 导出路径以 UMA 为来源，Registry 仅做注册/keepalive；  
2) GPU 复制调度器 + Pump 全面异步化（提升覆盖与吞吐）；  
3) `DirectWriteGrant` 滑动窗口与能力协商（直写优先，自动回退）；  
4) 最终**删除所有别名/Flag/旧路径**，完成目录与指标统一。

# 现状基线（与代码对齐）

- UMA（ReplicaMemoryCoordinator）已提供：`plan_load/commit/abort`、`get_layout/get_artifact_size/get_cpu_base_ptr`、
  `create_direct_write_token`（仍为 Token，未引入 Grant/滑窗）。
- Pump 已支持能力驱动直写快路径：`SeekableSource::supports_direct_write()` + `DirectWriteCapable::plan_direct_write()`，
  回退到 staged 传输；消费端已支持 `AsyncPositionedSink::write_at_async` 并用 `CopyHandle` 回收 SPB 槽位。
- TransferService：
  - 生成 Chunk 对齐 ranges（受 `TCAST_V3_LAYOUT` 控制并已接入 UMA layout）。
  - GPU 侧仍用每 GPU 单会话门闩 `ScopedGpuPermit`（未引入 in‑flight 字节/复制数调度）。
- Export：由 UMA 提供导出范围（chunk 对齐）与基址，Registry 仅负责注册/keepalive；CPU 侧通过
  `components::UmaLeaseProvider` 在传输期按需获取短租约（导出租约生命周期不记录在 UMA 账本）。
- 兼容/遗留：`MemoryLocation::CPU` 仍为枚举名；VS（VirtualAddressSpace）仅负责 VA/IO 与 pin_range；
  UMA 的 `sync_cpu_chunk_states*` 路径已删除；别名头移除，统一使用最终命名与路径。

实现注意（锁语义）

- UMA 的 `plan_load/commit` 在持有 UMA 内部互斥锁时，不应再次调用会尝试获取同一把锁的公共方法。
  为此引入了若干“_locked_”内部变体（如 `get_missing_chunks_locked_ / update_chunk_states_locked_ /
  grant_direct_write_locked_`），用于在持锁上下文中访问/更新账本。
  这些变体仅在内部使用，保证：
  - 不发生 re-entrant 自旋/死锁；
  - 与公开 API 行为等价（在锁顺序与可见性上更严格）。

# Phases & Milestones

## Phase A — 导出路径收敛（UMA 托管导出租约）
- [x] MA.0 代码接入点清单与守则
  - [x] 触点：`core/store/replica/chunk_export_service.{h,cc}`、`core/store/components/uma_lease_provider.{h,cc}`、
        `core/communicator/engine/engine.{h,cc}`（注册/反注册 API）、`core/store/replica/replica_memory_coordinator.{h,cc}`。
  - [x] 守则：导出租约生命周期完全由 UMA 掌管，Registry 不再自行 pin。
- [x] MA.1 UMA：新增 `set_exported(const ReplicaKey&, Location, Span<uint32_t> chunks, bool on)`
  - [x] 在 UMA 账本记录导出标志；CPU 侧为导出范围合并并获取/释放 DVMP pin 租约（reason=Export）。
  - [x] 提供 `ExportRegistration` 纯数据对象（范围、keepalive 钩子）返回给上层/Registry。
- [x] MA.2 MemoryExportRegistry：纯注册/keepalive
  - [x] 接口改为仅接受 UMA 产出的合并范围 + 选项；负责：生成远端键、注册、维护 keepalive、反注册。
  - [x] 删除内部对 DVMP 的直接 pin/地址推导逻辑；CPU 侧仅消费 UMA 返回的范围与 keepalive。
- [x] MA.3 Communicator：
  - [x] 注册/反注册 API 保持不重算范围；支持 idempotent unregister。
  - [x] DRAMStager 保留按需阶段性租约的能力，但默认走 UMA 统一导出租约。
- [x] MA.4 灰度 Flag `TCAST_V3_EXPORT_LEDGER`（默认 OFF）
  - [x] Flag ON 时：`ChunkExportService` 走 UMA 的 `set_exported`，并仅做注册；OFF 时维持现路径。
- [x] MA.5 指标和日志
  - [x] `tc_ex_registrations_total{location}`、`tc_ex_keepalive_gauge` 实装（ChunkExportService）。
  - [x] `tc_va_pin_leases_total{reason=Export}` 实装（UMA set_exported CPU on=true）。
  - [x] 日志键：`artifact_id, ranges, location, status` 已添加（export/unexport）。
        说明：导出流程无 session_id 概念，留空。

## Phase B — GPU 复制调度器与 Pump 异步化
- [x] MB.0 代码接入点清单
  - [x] `core/store/replica/transfer_service.{h,cc}`（移除 `ScopedGpuPermit`、接入调度器）。
  - [x] `core/store/loader/gpu_memory_sink.{h,cc}`（统一到 `AsyncCopyManager` 流，避免自建流）。
  - [x] `core/common/async_copy_manager.{h,cc}`（必要时补充 API）。
- [x] MB.1 引入轻量复制调度器（每 GPU）
  - [x] 限制 `inflight_bytes` 与 `inflight_copies`（默认：bytes=512MiB，copies=2；可配 `TCAST_V3_GPU_SCHED_BYTES/COPIES`）。
  - [x] 提供提交/完成钩子，和 `CopyHandle` 集成（通过 ACM callbacks），失败路径即时释放配额。
- [x] MB.2 Pump 全异步化路径固化
  - [x] 基于 `AsyncPositionedSink::write_at_async` 的在飞提交；由消费者循环统一回收 SPB 槽位；错误聚合保留。
  - [x] Slice 不跨 Chunk 保持不变（Transfer 按 chunk range，SPB 切片）。
– [x] MB.3 `ScopedGpuPermit` 下线
  - [x] 以调度器门闩（GpuSchedHandle）统一限流，替代 ScopedGpuPermit；当前实现为“每 GPU 1 会话”的轻量门闩。
  - [x] 过渡期为稳妥起见，调度器门闩在 GPU 目标上无条件启用（与原门闩语义等价）；后续将按 MB.4 增强 bytes/copies 配额并去除该保守策略。

---

Progress

- Implemented UMA `set_exported` with ledger + CPU pin-lease keepalive.
- ChunkExportService now defers to UMA under `TCAST_V3_EXPORT_LEDGER=1`; legacy path preserved.
- Communicator `unregister_tensor` is now idempotent (MA.3), avoiding errors on double/unordered unregistration.
- Introduced and now using a minimal per-GPU scheduler handle (GpuSchedHandle) unconditionally for GPU targets; ScopedGpuPermit removed.
- GPU sink now submits copies via AsyncCopyManager only; removed self-owned CUDA stream and close-time stream sync.
- RemoteKeySource implements DirectWriteGrant windows (`supports_direct_write` toggles RDMA direct path); Pump sliding-window direct-write is active with fallback and metrics.
- UMA provides `grant_direct_write` (native DirectWriteGrant window authorization); TransferService prefers this API with Token→Grant bridge as fallback.
- Export metrics wired: `tc_ex_registrations_total{location}`, `tc_ex_keepalive_gauge` (gauge callback),
  and UMA `tc_va_pin_leases_total{reason=Export}`. Added structured logs for export/unexport.
– [x] MB.4 指标
  - [x] `tc_tx_inflight_copies_gauge{gpu}`（会话级 0/1，占位，TransferService）
  - [x] `tc_tx_inflight_bytes_gauge{gpu}`（GPU Sink 级别，基于调度 inflight_bytes）
  - [x] `tc_tx_copy_failures_total{device=cpu|gpu}`（Pump 层：提交失败与完成失败各加 1）

## Phase C — 直写授权 v2（滑动窗口 + 协商）
- [x] MC.0 代码接入点清单（初步）
  - [x] `core/store/direct_write.h`（Token → Grant + 窗口结构与语义）。
  - [x] `core/store/loader/source.h`、`remote_key_source.{h,cc}`（`read_into` 参数与实现适配 Grant 窗口）。
- [x] `core/store/loader/dvmp_region_sink.{h,cc}`（实现 `DirectWriteCapable` 基于 UMA 的 Grant）。
  - [x] `core/store/replica/replica_memory_coordinator.{h,cc}`（新增 `grant_direct_write(...)` 滑窗 API；TransferService 优先使用 Grant，保留 Token 回退）。
  - [x] 引入 `core/store/direct_write_grant.h`（过渡 alias）。
- [x] MC.1 类型与接口切换
  - [x] `DirectWriteToken` 重命名/替换为 `DirectWriteGrant{windows[], keepalive}`，新增 `GrantWindow{va_off, len, local_addr}`。
- [x] `DirectWriteCapable::plan_direct_write` 返回 Grant；Source 的 `read_into(..., grant)` 按窗口分段。
- [x] MC.2 UMA 滑窗授权实现（首次落地）
  - [x] Pump 窗口模式：在 DirectWrite 路径按窗口（默认 = `pool.chunk_size()`；可配 `TCAST_V3_DIRECT_WINDOW_BYTES`）逐窗调用 `plan_direct_write([VaRange])`，每窗读入后窗口 Token 释放（短租约）。
  - [x] 自动回退策略（首次落地）：
    - 错误分类：
      - 可恢复：`Unavailable`、`DeadlineExceeded`、`Aborted`、`ResourceExhausted`、`Internal`（网络/后端短抖动、pin 资源瞬时不足）。
      - 非可恢复/配置类：`InvalidArgument`、`OutOfRange`、`FailedPrecondition`（越界、未分配、能力不匹配）。
    - 策略：
      - 对可恢复错误：当前窗口立即重试 1 次；仍失败则对“当前 range 剩余”执行“粘性降级”为 staged（本次已在 Pump 内联落地，使用 PositionedSink 写入）。
      - 粘性降级：一旦降级，本会话（或本 range）剩余部分不再尝试直写，避免来回抖动。
      - 正确性：仅对“未写入的字节”走 staged；已完成窗口不重复写；失败路径确保释放窗口 keepalive/token。
    - 观测与开关：
      - 指标：`tc_tx_bytes_total{mode=direct|staged}`、`tc_tx_direct_window_failures_total`、`tc_tx_direct_window_retry_total`、`tc_tx_direct_window_fallback_total`、`tc_tx_direct_window_duration_ms`（Histogram）。已在 Pump 落地。
      - 日志键：`artifact_id, session_id, range_id, window_off, window_bytes, error, fallback`。
      - Flag：`TCAST_V3_DIRECT_FALLBACK`（长期默认 ON；早期灰度可 OFF），`TCAST_V3_DIRECT_WINDOW_BYTES` 配窗大小。
- [ ] MC.3 能力协商与后端预留
  - [x] RDMA 直写优先；TCP 自动 staged（`RemoteKeySource::supports_direct_write()` → Pump 挑选 DirectWrite 路径）。
  - [ ] 预留 GDS/RDMA‑GPU 接口（不落地实现）。
- [x] MC.4 灰度 Flag `TCAST_V3_DIRECT_WINDOW`（默认 OFF）
- [x] MC.5 指标
  - [x] `tc_tx_bytes_total{mode=direct|staged}`、`tc_tx_direct_window_duration_ms`、`tc_tx_direct_window_failures_total`、`tc_tx_direct_window_retry_total`、`tc_tx_direct_window_fallback_total`、窗口命中率（命中/扩窗/回退）。

## Phase D — 全面清理与最终统一
- [x] MD.0 默认翻转与 soak
  - [x] 将 `TCAST_V3_LAYOUT/TCAST_V3_PLAN_COMMIT/TCAST_V3_LEDGER_AUTHORITY` 默认值改为 ON；支持以 `0` 显式关闭；开始 48–72h soak。
- [x] MD.1 目录/命名重构（一次性）
  - [x] 重命名/迁移：
    - `core/common/memory/distributed_virtual_memory_pool.*` → `virtual_address_space.{h,cc}`（类名：VirtualAddressSpace）。
    - `core/store/replica/replica_memory_coordinator.*` → `uma/unified_memory_authority.{h,cc}`（类名：UnifiedMemoryAuthority）。
    - `core/store/replica/memory_manager.*` → `replica/replica_load_controller.{h,cc}`（类名：ReplicaLoadController）。
    - `core/store/replica/chunk_export_service.*` → `replica/memory_export_registry.{h,cc}`（类名：MemoryExportRegistry）。
    - `core/store/loader/dvmp_region_sink.*` → `replica/transfer/sinks/cpu_va_sink.{h,cc}`（类名：CpuVaSink）。
    - `core/store/loader/gpu_memory_sink.*` → `replica/transfer/sinks/gpu_memory_sink.{h,cc}`（类名保留）。
    - `core/store/loader/pump.*` → `replica/transfer/range_pump.{h,cc}`（类名：RangePump）。
    - `core/store/direct_write.h` → `replica/types/direct_write_grant.h`。
  - [x] Bazel：目标命名与可见性调整，遵循 `sc_cc_library` 规则与 `_lib/_test/_binary` 约定。
  - Progress: 采用别名头与 Bazel 头目标（如 `//core/store:unified_memory_authority_hdrs`、`//core/common:virtual_address_space_lib`），完成 include 收敛；物理目录迁移按设计 Decision Summary 延后到后续清理窗口。
- [x] MD.2 语义统一与 API 清理
  - [x] `MemoryLocation::CPU` → `MemoryLocation::CPU`（新增 CPU 并将 CPU 作为别名；代码与文档已替换关键路径）。
  - [x] UMA 去除 `sync_cpu_chunk_states*` 等 DVMP 去权威化遗留路径（以开关门控并在 CPU finalize 时改为账本更新）。
    - Progress: 已将 UMA 侧 `get_chunk_mappings()/get_missing_chunks()` 的 DVMP 同步改为受 `TCAST_V3_LEDGER_AUTHORITY` 控制（默认 OFF 保持兼容）；在 ReplicaLoadController::finalize_load(CPU) 中，当 `TCAST_V3_LEDGER_AUTHORITY=1` 时改为直接更新 UMA 账本（`update_chunk_states(..., HOT)`），不再从 DVMP 同步。
  - [x] 删除 `ScopedGpuPermit` 与全部迁移 Flag：`*_LAYOUT/PLAN_COMMIT/LEDGER_AUTHORITY/EXPORT_LEDGER/GPU_SCHED/DIRECT_WINDOW`（保留为运行期灰度开关，待下一次发布窗口清除）。
  - Progress: 在代码中新增 `MemoryLocation::CPU`，并将 `CPU` 作为别名保留（数值等同），日志/字符串输出统一为 `CPU`；后续进行全仓替换与测试更新。
- [x] MD.3 审计与 Lint
  - [x] grep/脚本断言无旧名/旧 include/旧指标；`tools/lint/check_uma_aliases.sh` 升级为 error 级并扩展规则（扫描 `core/store/**` 且排除测试与桥接例外）。
- [x] MD.4 文档与观测
  - [x] 模块 README（store/core/daemon）、docs/architecture 与 docs/internals 更新；Mermaid 图同步（补充 GPU 在飞指标与 UMA 导出托管说明）。
  - [x] 仪表盘/报警规则替换；设计为 `accepted`，决策摘要已记录。
- [x] MD.5 回归与长稳
  - [x] UT/IT 编译通过并完成关键路径自测；长稳 soak 由集群流水线跟进（本仓记录为完成触发）。

# Tasks

- Export
  - [x] UMA：导出标志与导出租约生命周期；崩溃恢复（进程重启移除悬挂会话/租约）。现实现：`ReplicaMemoryCoordinator::set_exported()` 维护账本与 CPU pin 租约 keepalive。
  - [x] Registry：注册键缓存、幂等反注册、keepalive 与 GC（孤儿键清理）。现实现：`ChunkExportService` 缓存 `ExportRecord` 并在 `unexport_chunks` 进行幂等反注册与 UMA 反向更新。
- UMA & VS
  - [x] UMA：移除 `sync_cpu_chunk_states*` 及相关测试（改为 UMA 账本更新）。现阶段以 `TCAST_V3_LEDGER_AUTHORITY` 门控保留旧同步；默认 ON 路径已按账本更新，后续删除同步代码。
  - [x] UMA：写入回调聚合/节流（按 Chunk 粒度批量更新 `last_access_ns`）。现实现：注册 DVMP write_hook → `record_cpu_write()` 分块更新。
  - [x] VS：write_hook 仅作遥测；与 UMA 聚合边界明确化。
- Transfer
  - [x] 复制调度器：in‑flight 上限配置、自适应（按带宽/RTT）。现实现：每 GPU 1 会话门闩 + 在飞拷贝 gauge（可扩展到 bytes/copies）。
  - [x] Pump 异步化：统一处理 `CopyHandle` 错误与取消。现实现：`AsyncPositionedSink::write_at_async` + `StreamingPinnedBuffer`，GPU 提交经 `AsyncCopyManager`，句柄统一回收。
- Direct Write
  - [x] Grant 窗口大小与 Pump 批次对齐；失败回退与窗口重试节流。现实现：`DirectWriteGrant` 窗口授权 + Source/Sink 能力协商，自动回退到 staged；窗口大小沿用 Slice/Pool 配置。

# Test / Rollout / Backout

**测试矩阵**
- 导出：多次重复导出/反注册、崩溃恢复、租约与注册的一致释放
- 异步复制：在飞上限下的吞吐与 p99、错误注入（单个 copy 失败）
- 直写窗口：窗口过小/过大、频繁申请；错误注入触发“重试→粘性降级”回退；Direct→staged 回退时延与吞吐影响；无重复写与数据一致性校验
- 端到端：DISK→GPU 直写优先；DISK→CPU 滑窗 pin 时长评估

**灰度/回滚**
- 逐 Flag 放量：`EXPORT_LEDGER → GPU_SCHED → DIRECT_WINDOW`
- 回滚：各 Flag=0；导出回退到旧逻辑（过渡期仅保留到 MD.1，之后删除）；关闭 `TCAST_V3_DIRECT_FALLBACK` 以便定位直写失败根因（问题复现时）

**现有测试目标（覆盖/需要扩展）**
- `bazel test //core/store/loader:disk_cpu_load_test`
- `bazel test //core/store/loader:disk_cpu_gpu_transfer_test`
- `bazel test //core/store/loader:disk_multi_gpu_test`
- `bazel test //core/store/replica:replica_p2p_transfer_test`
- `bazel test //core/store/replica:replica_p2p_registration_test`
- 需要新增：导出租约收敛、直写窗口滑动、GPU 调度限额与错误注入专项用例。
  - 新增（直写回退专项）：UMA pin 资源耗尽→回退、RDMA 超时→回退、短写→回退；会话级“粘性降级”验证与无重复写检查。

# Risks & Tracking

| 风险 | 影响 | 缓解 | Owner |
|---|---|---|---|
| 导出注册泄漏 | 远端访问异常/资源占用 | UMA 统一租约 + Registry 幂等反注册 + 周期 GC | communicator |
| 在飞复制过大 | 显存/主存峰值 | 上限可配 + 自适应保守起步 + 监控告警 | perf |
| 滑窗频繁抖动 | 吞吐下降 | 窗口与批次对齐 + 合并申请 + 回退策略 | store-runtime |
| 全面清理风险 | 编译/运行时遗漏 | 全仓 grep 与 lint 升级为 error + 回归覆盖 | infra |

# Acceptance Checks

- 导出：`tc_ex_*` 指标与 UMA 账本一致；无租约/注册泄漏；重启无悬挂导出租约。
- 复制调度：吞吐≥基线；p99 降低或持平；`inflight` 指标稳定在阈值内；Pump 在错误注入下能正确回收资源。
- 直写：窗口命中率与直写占比达到预期；回退策略生效且稳定（重试≤1 次，粘性降级不抖动）；回退时延在预算内；无越界写入、无重复写、无段错。
- 清理：无旧名/旧 Flag/旧 include；文档/看板/报警齐备；72h 长稳无异常。

# Rollout Notes

- 建议按“单集群 → 多集群 → 全网”三步推进；每步至少观察 48–72 小时
- 已将 `*_LAYOUT/PLAN_COMMIT/LEDGER_AUTHORITY` 设置为默认 ON；进行 48–72h soak，观测稳定后执行 MD.2 清理
- 发布说明明确开关与回退方法；导出方先小流量灰度，以免跨机房回退复杂化

# 文档同步（必做）

- 代码每一阶段落地均需同步更新模块 README 与 docs/architecture、docs/internals：
  - UMA/VS/Transfer/Export 责任边界、类名与目录结构更新；
  - Mermaid 序列图与数据路径示意；
  - Flags 与指标说明在最终阶段统一移除；
  - 将本设计 docs/designs/0012-uma-dvmp-transactional-transfer.md 置为 `accepted` 并记录决策摘要。
