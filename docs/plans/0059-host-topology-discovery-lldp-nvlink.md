---
slug: host-topology-discovery-lldp-nvlink
title: Plan - Host Topology Discovery from LLDP and NVLINK
links:
  design: ../designs/0059-host-topology-discovery-lldp-nvlink.md
areas:
  - core
  - daemon
  - proto
related_code:
  - core/communicator/transport/net_dev.cc
  - core/communicator/topology/topology.h
  - core/communicator/topology/simple_numa_topology.cc
  - core/communicator/topology/discovery/host_topology_builder.cc
  - core/communicator/topology/discovery/host_topology_builder.h
  - core/communicator/topology/discovery/discovery_test.cc
  - core/communicator/topology/simple_numa_topology_tool.cc
  - core/communicator/routing/routing_context.cc
  - core/communicator/routing/types.h
  - core/store/materialization/dataplane/sources/remote_key_source.cc
  - proto/tensorcast/communicator/v1/communicator_config.proto
  - docs/designs/0004-unified-runtime-config.md
  - docs/deployment/store-daemon.md
---

# Objective

把当前“零散的 LLDP rail 读取”和“缺失的 NVLINK 拓扑输入”升级为统一的主机拓扑发现能力，输出可验证的 `Topology` 模型，并以最小风险逐步接入现有通信/路由路径。

# Current State & Grounding

- LLDP 目前仅在 `NetDev::read_rail_id()` 中通过环境变量文件解析 rail id，属于设备局部逻辑，未进入统一拓扑模型（`core/communicator/transport/net_dev.cc`）。
- `core/communicator/topology/` 已有 `Pool/Endpoint/Link` 与校验器，但主要来自显式输入，不含 host discovery（`core/communicator/topology/topology.h`）。
- `simple_numa` 已用于 stager 亲和与拓扑样例构建，但不足以表达真实 rail/NVLINK 联合拓扑（`core/communicator/engine/engine.cc`，`core/communicator/topology/simple_numa_topology.cc`）。
- `RoutingContext` 当前只支持 direct 1-hop channel；`NvlinkAdapter` 仍为占位实现（`core/communicator/routing/routing_context.cc`，`core/communicator/routing/adapter.cc`）。
- 配置治理要求运行时行为由统一配置文件驱动，不新增散落环境变量（`docs/designs/0004-unified-runtime-config.md`）。

# Phases & Milestones

- [x] Phase 1: 配置与数据模型定型
  - [x] Milestone 1.1: 在 `CommunicatorConfig` 新增 `topology_discovery` 配置段（LLDP/NVLINK/merge policy）。
  - [x] Milestone 1.2: 扩展 `config_io` 默认值与校验规则，保证配置缺省行为稳定。
  - [x] Milestone 1.3: 更新示例配置与文档，明确默认关闭与 required 策略。

- [x] Phase 2: LLDP 解析器模块化
  - [x] Milestone 2.1: 新增 `core/communicator/topology/discovery/lldp_source.*`，实现统一 parser（支持 `if=pci,nic,rail`）。
  - [x] Milestone 2.2: 增加 LLDP 单测（合法行、空行注释、重复 nic、非法 rail、非法 BDF）。
  - [x] Milestone 2.3: `NetDev::read_rail_id()` 改为复用 LLDP 解析库，移除重复解析逻辑。

- [x] Phase 3: NVLINK 数据源抽象与首版输入
  - [x] Milestone 3.1: 新增 `NvlinkSource` 抽象和 `SnapshotNvlinkSource` 实现（文件快照）。
  - [x] Milestone 3.2: 定义快照 schema 与 parser 校验（GPU 标识、重复边归一化、自环过滤）。
  - [x] Milestone 3.3: 实现 `RuntimeNvlinkSource`（基于 `nvidia-smi` 运行时探测）并补齐 required/degrade 语义与测试覆盖。

- [x] Phase 4: LLDP + NVLINK 合并建模
  - [x] Milestone 4.1: 新增 `HostTopologyBuilder`，输入 baseline + LLDP + NVLINK，输出 `Topology`。
  - [x] Milestone 4.2: 生成 rail switch endpoint 与 NVLINK P2P links，支持 DOT 导出。
  - [x] Milestone 4.3: 增加合并单测（完整输入、LLDP 缺失、NVLINK 缺失、冲突输入）。

- [x] Phase 5: 路由/调用链接入与兼容
  - [x] Milestone 5.1: 启动路径接入 discovery builder（配置开启时启用）。
  - [x] Milestone 5.2: 扩展 `EndpointBinding` 元数据字段（可选）并保持旧调用兼容。
  - [x] Milestone 5.3: 验证 `RemoteKeySource` routed-first + strict-fallback 行为不回归。

- [x] Phase 6: 收尾与弃用治理
  - [x] Milestone 6.1: 在部署文档中标记 `TENSORCAST_LLDP_FILE_NAME` 为 legacy 兼容入口。
  - [x] Milestone 6.2: 加入可观测性字段（发现源、记录数、降级原因）。
  - [x] Milestone 6.3: 形成最终 deprecation 时间表（仅在配置路径稳定后执行）。

# Implementation Notes

- 首版采用“配置驱动 + 快照优先”，保证开发机和 CI 可稳定复现拓扑构建。
- `simple_numa` 继续作为基础 CPU/GPU/NIC inventory；LLDP 与 NVLINK 负责增量信息补强。
- 对不完整输入优先 degrade（除非 `required=true`），避免影响现网可用性。
- 2026-03-04 增量增强：`HostTopologyBuilder` 新增 NIC↔GPU 动态亲和性推断（PCI path longest-common-prefix 评分），并新增 builder observability 计数与降级原因字段；当推断不完整时保留基线 `pool_ids`。

# Acceptance Checks

- 配置与 Proto
  - [x] `CommunicatorConfig` 新增 discovery 字段并通过 loader 校验。
  - [x] 若变更 `.proto`，执行 `bash tools/build_proto_python.sh`。

- C++ 单元测试
  - [x] `bazel test //core/communicator:config_io_test`
  - [x] `bazel test //core/communicator:topology_test`
  - [x] `bazel test //core/communicator:routing_context_test`
  - [x] 新增 discovery 相关测试 target（LLDP/NVLINK/merge）。
  - [x] 新增 NIC↔GPU 亲和性推断单测（成功收敛与降级保持基线两类场景）。

- 集成验证
  - [x] 使用仓库内 `lldp-info.txt` 与 NVLINK 快照构建 DOT，人工核对 rail 分组和 GPU 互联边。
    - 2026-03-04 验证命令：`bazel run //core/communicator:simple_numa_topology_tool -- /tmp/topology_discovery_validation.yaml`
    - 观测输出：`lldp_records=16`、`rail_switch_endpoints=8`、`nvlink_gpus=8`、`nvlink_edges=7`，且 `lldp_degraded=false`、`nvlink_degraded=false`。
  - [x] 在 routed-first 场景验证失败回退直连路径。
  - [x] 使用远程 GPU 资源验证 runtime probe + 动态亲和性推断观测字段（brainctl skill）。
    - 2026-03-04 远程命令：
      - `brainctl launch -d --charged-group=tensorcast_dev --gpu 1 --cpu 4 --memory 106400 --private-machine group --positive-tags L40S,H200,H800,H100 ...`
      - `brainctl exec process/<id> -n shai-core -- ... bazel run //core/communicator:simple_numa_topology_tool -- /tmp/topology_remote_affinity.yaml`
      - `brainctl exec process/<id> -n shai-core -- ... bazel test //core/communicator:discovery_test ...`
    - 远程观测行：`nvlink_source=runtime_probe nvlink_gpus=1 nvlink_edges=0 affinity_nic_candidates=2 affinity_nic_scored=2 affinity_nic_narrowed=0 affinity_degraded=false`。
    - 清理：`brainctl delete process <id> -n shai-core` 后 `brainctl get process <id> -n shai-core` 返回 `NotFound`。

# Rollout / Backout

- Rollout:
  - 默认 `topology_discovery.enable=false`。
  - 先灰度开启 LLDP（`required=false`），再开启 NVLINK 快照源。
  - 观察日志中的降级与冲突计数，再考虑提升到 `required=true`。

- Backout:
  - 关闭 `topology_discovery.enable`，恢复现有 `simple_numa + direct` 路径。
  - 保留新模块但不生效，便于后续二次灰度。

# Risks & Tracking

- 解析规则漂移风险：外部 LLDP 文件格式可能变化。
- 标识映射风险：GPU index/UUID 与 NIC 命名跨环境不一致。
- 接口演进风险：`RoutingContext` 当前 direct-only，需避免过早绑定多跳语义。

# Owner Checklist

- [x] 设计文档与计划文档 1:1 链接完成。
- [x] 配置变更遵循统一配置设计（0004）。
- [x] 所有新增行为均有单测和 degrade 覆盖。
- [x] 相关 README/部署文档同步更新。
