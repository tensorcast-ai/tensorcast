# 分布式协调系列（收敛版）

本目录已收敛为 3 个文档（含索引），用于统一 TensorCast 在以下问题上的高层认知：

- 如何在“中心化全局最优”与“分布式高频执行”之间做可落地折中。
- 如何基于当前项目现状，把状态放置、协调机制、模式切换、治理流程形成闭环。
- 如何同时考虑生命周期、生产/消费频率、增长速率、对象规模和网络拓扑干扰。
- 如何区分并治理不同 KV 形态（PD 流式 vs Decoder 共享），并支持三轨数据路径并存。
- 如何评估不同架构决策对集群 scaling 潜力的影响，并形成分阶段扩展路线。
- 如何把框架泛化到更多场景（检查点、视图装配、滚动预热、恢复对账等）。

## 文档结构

1. `01-global-optimum-vs-distributed-execution-framework.md`
- 统一框架：模块级状态总表、数据/状态/控制三流、热冷路径拆分、中心化与分布式边界，并附 Contract-First 的最小契约集合（可验收）。

2. `02-mode-switching-workload-playbook-and-governance.md`
- 运行策略：画像建模、模式切换与防振荡、场景裁决、阶段路线、上线闸门与治理指标，并附统一 ADR-lite 决策清单与局部重构优先集。

## 状态分层约定（重要）

两篇主文统一采用三层语义，避免把提案写成既成事实：

1. `当前已实现（Current）`：以 `0055` 与仓库现有代码/proto/schema 为准。
2. `目标提案（Proposed）`：以 `0056`、`0060` 场景与问题为目标约束，可调整。
3. `待验证假设（To Validate）`：必须通过实验和灰度收敛的参数与机制。

## 一致性约束（本系列统一口径）

1. `契约复用优先于实现复用`：跨语言组件优先复用协议与语义，不跨语言直接复用实现（例如 SDK `capability_directory` 语义可复用，但 daemon 采用本地实现）。
2. `执行面单一主轴`：0056 以 `NodeAgent.ExecutePlan` 为执行主轴；gateway 仅做无状态接入，不形成平行执行语义。
3. `LaneContext 必须闭环`：不仅 Plan 路径，非 Plan 的 daemon 请求也必须携带 lane/policy 上下文，保证审计与重试语义一致。
4. `policy 不等同 StorePolicy`：本系列的 `policy/policy_version` 指治理策略版本（lane/mode/预算/陈旧性），不等同于 API/Proto 的 `StorePolicy`（持久化/驻留/冷暖/保活等存储语义）。
5. `审计键必须统一`：最低审计键：`request_id + idempotency_key + lane + policy_version`（其中 `request_id/idempotency_key` 只进日志/trace，禁止进入 Prometheus labels）。
6. `QueueDirectory 先于队列状态机`：0060 必须先落 `queue identity -> leader endpoint -> epoch fencing` 的最小契约，再推进本地高频状态机。
7. `去热化不等于放宽一致性`：GS 选源去热化时，必须保留 claim/reservation/fencing 等价语义，避免副本过载与冲突分配。
8. `mode ⟂ lane`：mode 只改变闸门强度，不改变正确性边界，也不把 per-item 推进推回 GS。
9. `动作幂等闸门`：所有在未知结果下可能被重试的动作必须具备 join key（幂等合流）或显式禁止重试（placement pin 等副作用动作尤需严格）。
10. `NodeAgentDirectory 先于执行统一`：要把 instance steps 统一到 Node Agent 主轴，必须先把 `instance_id -> node_agent_endpoint` 变成可验收的目录事实锚点（pilot 允许短期通过 reserved label 键承载，例如 `labels["tc.node_agent.endpoint"]`；长期必须升格为显式字段以避免 ad-hoc 约定）。

## 使用建议

- 架构评审先看 `01`，确认边界与职责。
- 策略与上线评审看 `02`，确认切换逻辑、稳定性、灰度与回滚。

## 质量承诺（针对“是否无中生有”的回答）

本系列不采用“凭空抽象推导”，而是以当前仓库设计与协议为锚点：

- 以 `0055/0056/0060/0034/0011/0004` 的既有约束为基础。
- 与 `schema.sql`、`global_store/README.md`、`daemon proto`、`operation proto` 保持语义一致。
- 在两篇主文中显式给出“问题覆盖矩阵”和“仍需实证项”，避免伪完备结论。

## 与现有设计文档关系

本系列建立在以下设计基础上：
- `docs/designs/0055-programmable-framework.md`
- `docs/designs/0056-programmable-framework-adv.md`
- `docs/designs/0060-tensor-work-queue.md`
- `docs/designs/0034-stable-memory-tiers.md`
- `docs/designs/0011-unified-session-lifecycle-leases.md`
- `docs/designs/0004-unified-runtime-config.md`
