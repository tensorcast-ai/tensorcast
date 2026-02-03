---
slug: communicator-topology-model
title: Plan - Communicator Topology Model and Routing
links:
  design: ../designs/0058-communicator-topology-model.md
areas:
  - core
  - daemon
  - proto
related_code:
  - core/communicator/README.md
  - core/communicator/engine/engine.h
  - core/communicator/engine/channel.h
  - core/communicator/transport/rdma_transport.h
  - core/communicator/transport/mtcp_transport.h
  - core/communicator/docs/comm-read-tensor.md
  - docs/architecture/p2p-transfer-strategies.md
  - proto/tensorcast/communicator/v1/communicator_config.proto
---

# Objective

在现有 Communicator 之上引入可搜索拓扑模型（Pool / Endpoint / Link）与路径级抽象（Connection / Channel / Communicator），为多路径、拓扑选路与故障切换提供统一的数据结构与策略入口，并保持现有读写路径可兼容回退。

# Current State & Grounding

- 通信层以 `communicator::engine::Communicator` 为中心，面向 peer 的连接抽象是 `engine::Channel`，其内部管理 TCP/MTCP/RDMA 连接与握手状态（`core/communicator/engine/engine.h`，`core/communicator/engine/channel.h`）。
- 数据面传输直接由 Communicator 选择 RDMA 或 MTCP，路径选择缺乏明确的“拓扑图”，也不存在多跳或多路径抽象（`core/communicator/README.md`，`core/communicator/docs/comm-read-tensor.md`）。
- P2P 选源由 Global Store 协调，数据面仍由 Communicator 完成读写，当前没有拓扑级路由或自动 failover（`docs/architecture/p2p-transfer-strategies.md`）。
- 现有 `CommunicatorConfig` 只有 `simple_numa` 等轻量映射，缺少可表达交换结构的拓扑模型（`proto/tensorcast/communicator/v1/communicator_config.proto`）。

# Phases & Milestones

- [ ] Phase 1: 拓扑模型与配置骨架
  - [ ] Milestone 1.1: 新增拓扑数据结构（`Pool` / `Endpoint` / `Link`）与图构建器（建议落在 `core/communicator/topology/`），并提供最小可视化/调试导出。
  - [ ] Milestone 1.2: 扩展 `CommunicatorConfig` 引入拓扑描述（或等价独立拓扑配置段），并提供从 `simple_numa` 的兼容映射；遵循 `docs/designs/0004-unified-runtime-config.md`。
  - [ ] Milestone 1.3: 拓扑校验与单元测试（连通性、带宽/延迟字段、Switch Endpoint 合法性）。

- [ ] Phase 2: 执行层映射与兼容桥接
  - [ ] Milestone 2.1: 新增 `Connection` 与路径级 `Channel`（为避免与 `engine::Channel` 冲突，可在新命名空间引入 `RouteChannel`）并完成到现有传输对象的映射。
  - [ ] Milestone 2.2: 构建 `Communicator` 聚合器（src-dst 维度）与 Connection 复用缓存，支持多 hop。
  - [ ] Milestone 2.3: 增加基础统计与健康状态收集（Link/Connection 级别）用于后续路由。

- [ ] Phase 3: 路径搜索、multipath 与故障切换
  - [ ] Milestone 3.1: 实现 k-shortest 或最短路搜索（BW/latency/统计混合打分），产出多条候选 Channel。
  - [ ] Milestone 3.2: `mct_map` 统计闭环，支持按消息大小选择路径并实现 striping。
  - [ ] Milestone 3.3: 连接失败/握手失败时的自动 failover（回退到其它 Channel 或默认直连）。

- [ ] Phase 4: 业务集成与回归验证
  - [ ] Milestone 4.1: 在 P2P 读路径引入“路由选择”入口（保持默认回退到旧逻辑），逐步接入 `P2PLoader` / `RemoteKeySource`。
  - [ ] Milestone 4.2: 增加集成测试与压力回归（多 GPU / 多 NIC / 交换拓扑模拟），并完善文档更新。

# Tasks

- 更新 `core/communicator/README.md`，补充拓扑与多路径语义。
- 更新 `docs/architecture/p2p-transfer-strategies.md`，说明路由与 failover 在数据面的位置。
- 若修改 `.proto`，运行 `bash tools/build_proto_python.sh` 并通过 `bazel test //proto/...`。
- 增加基准或调试工具，便于输出拓扑图与路径选择日志。

# Test / Rollout / Backout

- **单元测试**：拓扑构建、路径搜索、Switch Endpoint 约束、连接健康状态更新。
- **集成测试**：复用现有 `tcp_engine_test` / `rdma` 相关测试，并新增多路径与 failover 场景。
- **回归验证**：对 `read_tensor` 关键路径做性能对比，确认无退化。
- **上线策略**：新增配置开关，默认关闭；先在影子模式输出路径选择日志。
- **回退策略**：禁用拓扑路由配置，回退到旧的直连通道选择。

# Risks & Tracking

- **复杂度上升**：需确保拓扑模型与现有 `engine::Channel` 的语义边界清晰。
- **配置兼容**：拓扑配置必须兼容 `simple_numa`，避免破坏现有部署。
- **性能不确定性**：路径搜索与统计更新需要限频与缓存。

# Owner Checklist

- [ ] 设计与计划双向链接已建立。
- [ ] 配置变更遵循统一配置设计（`docs/designs/0004-unified-runtime-config.md`）。
- [ ] 文档更新覆盖核心模块 README 与架构说明。
- [ ] 关键路径有基准/回归指标。
