---
slug: step3p5-binding-native-serving-bootstrap-performance-followup
title: Step3p5 Binding-Native Serving Bootstrap Performance Follow-Up
status: proposed
areas: ["core", "daemon", "sdk", "integrations", "proto", "docs", "tests"]
created: 2026-03-28
last_updated: 2026-03-28
related_code:
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/replica/collective_disk_loader.cc
  - core/store/runtime/metadata/registration_backend.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/target_materialization_service.cc
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/state/lip_manager.cc
  - daemon/state/lip_manager.h
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/api/context.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/README.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0109-batched-owner-file-collective-executor.md
  - docs/designs/0111-source-to-serving-builder-and-representation-publication.md
  - docs/designs/0112-binding-native-serving-realization-and-publication.md
links:
  design:
    - ../designs/0108-tensor-aware-materialization-strategy-plane.md
    - ../designs/0112-binding-native-serving-realization-and-publication.md
  integration:
    - /data/workspace/internal-vllm/docs/design/tensorcast_step3p5_from_disk_cold_start_performance_followup.md
---

# Objective

把 Step3p5 当前 mounted 8xH800 `from_disk -> same-binding -> serving` 冷启动从
“correctness green but slow” 收敛成：

- 只做一次最终必要的 rank-local serving identity minting，
- 不再为同一份 immutable binding current value 付出第二轮 full-data hash，
- 让 TP bootstrap 能自然进入 collective / tensor-aware 最优 executor，
- 并保持 `0112` 的 binding-native、serving-only runtime 主线不回退。

# Background

实机基线来自：

- TensorCast case:
  `/data/tc/s35-0112/tc-20260328-054200`
- default baseline:
  `/data/tc/s35-default/default-20260328-055149`

同机结果：

- TensorCast ready 约 `403.24s`
- default ready 约 `212.44s`
- 额外成本约 `190.80s`

白盒与日志证据已经明确定位了这 `~190s` 的主要来源：

1. `materialize_mapped_into_target` 平均约 `142.36s/rank`
2. `publication.wait_assembly_attempt` 平均约 `37.5s/rank`
3. closeout 分解为：
   - `seal_from_cut.compute_data_multihash` 平均约 `15.41s/rank`
   - `assembly_attempt.finalize_dependency_ready_closeout`
     平均约 `21.60s/rank`

当前这条 path 已经不是 correctness blocker，而是典型的
“对象模型已对，但执行形态仍停留在通用 fallback”。

# White-Box Findings

## 1. source-bound realization 仍未兑现 0108 的 executor 目标

当前 Step3p5 `BindingRealizationPlan -> RefillOwnedBinding(realization_plan=...)`
热路径虽然已经走 common runtime，但实机日志仍显示：

- `dominant_executor=GenericByteRangeExecutor(source_ordered)`
- `collective_handled=0`
- `direct_write_supported=0`

这说明：

- `RepresentationWorkPlan` 已进入 hot path，
- 但 same-binding source-bound materialization 仍主要 lower 成
  generic byte-range executor，
- `0108` 想要的 tensor-aware local / owner-file collective executor 还没有真正
  落到这条 path。

## 2. 当前存在双重 identity minting

对当前 same-binding `representation_publish + binding_value subject` path：

### 第一轮

`seal_assembly_from_cut(canonical_full)` 会对 canonical serving bytes 计算
`index_multihash` / `data_multihash`，产出 `seal_result.sealed_artifact_id`。

这是一个真实 artifact-plane identity。

### 第二轮

`finalize_binding_subject_closeout(...) -> commit_lease_in_place(...)`
又会对同一份 sealed binding current value 再生成一次 MI2 serving identity。

对当前 Step3p5 path，这两轮针对的本地 bytes 语义上是同一份 rank-local finalized
canonical serving bytes。

结论：

- 当前 path 不需要双重 artifact-plane minting。
- 即使对象模型里保留 source lineage 与 serving lineage，也不应再重算第二轮
  full-data hash。

## 3. 第二轮 hash 当前不是 GPU fast path

当前 second-stage closeout 通过 `LipManager::commit_lease_in_place(...)` 把 binding
segments 包装成 `SeekableSource` 后调用
`compute_data_multihash_from_seekable_source(...)`。

这条实现：

- 不是 `compute_data_multihash_from_gpu(...)` fast path，
- 对 GPU binding segments 会走 host read / D2H 式的 seekable-source hashing，
- 与 same-binding path“最终 bytes 已经在 GPU 上”的事实不相称。

对当前 path，这违反了应有的规则：

- 如果必须算 hash，只能在 GPU 上算；
- 必须在最终 bytes 已经到达 binding host 之后算；
- 不允许再引入第二轮 CPU/host 风格 full-data pass。

## 4. 当前 path 上独立 `seal artifact` 进入 GS 的价值已经极低

对 `internal-vllm` 当前这条 path：

- `source_version_key=None`
- runtime 最终只消费 serving artifact
- integration 不暴露中间 `seal artifact`

当前独立 `seal artifact` 写入 GS 的主要作用只剩：

- `assembly_id -> sealed_artifact_id` 幂等 binding
- 可选 source lineage

对这条 path，如果保留独立 source lineage 槽位会导致：

- 第二次 full-data hash 更难消除
- closeout 更难收敛到 “subject-first, single-host, single-mint” 模型

因此，对这条 path 应把独立 `seal artifact` 降为：

- 要么完全不落 GS；
- 要么只保留内存内/本地过程状态，不作为单独 artifact-plane object。

但这应被视为后续设计选项，而不是当前性能 follow-up 的前置条件。

当前应先达成的，是：

- second-stage serving promote 不再做第二轮 full-data hash；
- same-binding path 先实现 single-mint effect；
- 再决定是否进一步把独立 `seal artifact` 槽位从 artifact-plane 对象中移除。

## 5. source-bound path 目前吃不到 first-class collective hint

`target_materialization_service` 会从 `operation_id#tcg:...` 解析：

- `transport_scheduling_group`
- `collective_load_group`

但 `owned_binding_service` 当前 source-bound path 没有相同的 hint lower 逻辑。

结果是：

- 即使 SDK 层未来传了 collective context，
- current `RefillOwnedBinding` path 也无法稳定进入 collective mapped executor。

这是 API 设计和 daemon ingress 设计共同造成的问题。

# Decision

对当前 `internal-vllm` same-binding Step3p5 serving bootstrap path，采用以下收敛方向：

1. **single-mint effect**
   - 当前 path 的总 identity 成本必须等价于“只为最终 serving artifact 做一次最终必要的 rank-local minting”
   - 这可以先通过复用第一次 seal identity 达成
   - 是否彻底移除独立 `seal artifact` 槽位保留为后续设计选项
2. **单次 rank-local hash**
   - 每个 TP rank 只对自己的最终 canonical serving bytes 算一次 hash
   - TP8 场景总计 8 次，不是 1 次，也绝不是当前接近 16 次 full pass
3. **GPU-only hash**
   - 对当前 same-binding path，不允许 CPU/host 风格 second-stage hash
4. **collective ingress 和 policy 必须 first-class**
   - collective/topology 信息必须成为 source-bound API 的显式输入
   - 不再长期依赖 `operation_id` side-channel
   - collective 行为策略必须显式化：
     - `REQUIRE_COLLECTIVE`
     - `ALLOW_NOT_ELIGIBLE_FALLBACK`
     - `DISABLE_COLLECTIVE`
   - `ALLOW_NOT_ELIGIBLE_FALLBACK` 只允许对 typed `not_eligible` 原因降级；
     `execution_failed` 默认仍然 `fatal`

# Workstreams

## Track A: Collapse Redundant Second-Stage Identity Work

### Slice A1: 后续设计选项，去掉当前 path 的独立 `seal artifact` 槽位

目标：

- 对 `representation_publish + binding_value subject + source_version_key=None`
  的 same-binding path，不再把 `seal_result.sealed_artifact_id` 当作独立 artifact
  写入 GS。

这不是当前性能修复的前置条件。
它应在 A2 已经落地、single-mint effect 已经稳定后，再作为对象模型收敛选项推进。

建议实现：

1. 为 `seal_assembly_from_cut(...)` 增加 path-specific mode：
   - 保留 cut validation / readiness / canonical byte proof
   - 但不再执行 `upsert_artifact_binding(assembly_id -> sealed_artifact_id)`
2. 对该 mode，closeout 只生成最终 `serving_artifact_id`。
3. `PublishedModelVersion.source_artifact_id` 在此 path 上可退化为：
   - 原始 imported source artifact id，或
   - 显式 `None` / path-specific absent source lineage
   但不能继续伪造中间 seal artifact 为 integration truth。

涉及文件：

- [core/store/runtime/ingestion/materialization_facade.cc](/data/workspace/tensorcast-280/core/store/runtime/ingestion/materialization_facade.cc)
- [daemon/service/controllers/assembly_operation_service.cc](/data/workspace/tensorcast-280/daemon/service/controllers/assembly_operation_service.cc)
- [tensorcast/api/store/__init__.py](/data/workspace/tensorcast-280/tensorcast/api/store/__init__.py)

退出标准：

- 当前 path 的 final `PublishedModelVersion` 只带最终 serving artifact identity
  作为 artifact-plane 结果；
- GS 中不再新增“仅供本地 serving bootstrap 使用”的独立 seal artifact 绑定记录。

### Slice A2: 当前主线，复用第一次 identity

目标：

- 在过渡阶段，第二步 serving promote 复用第一步 seal 的
  `index_multihash/data_multihash`，不再做第二轮 full-data hash。

建议实现：

- 为 `commit_lease_in_place(...)` / promote path 增加 `artifact_id_override` 或等价
  typed identity proof；
- 前提是：
  - binding current value 的 canonical index 与 seal 时相同
  - binding current value 的 byte-space 仍等于 seal 确认过的 canonical byte-space

涉及文件：

- [daemon/state/lip_manager.h](/data/workspace/tensorcast-280/daemon/state/lip_manager.h)
- [daemon/state/lip_manager.cc](/data/workspace/tensorcast-280/daemon/state/lip_manager.cc)
- [daemon/service/controllers/assembly_operation_service.cc](/data/workspace/tensorcast-280/daemon/service/controllers/assembly_operation_service.cc)

退出标准：

- current same-binding path 仍可保留独立 `seal artifact` 过程状态，但 serving promote
  不再重算第二轮 full-data hash；
- profile / logs 能明确看到 second-stage hash 已消失；
- single-mint effect 已成立，即当前 path 的总 identity 成本与“一次最终必要 hash”
  等价。

## Track B: Enforce GPU-Only Single-Round Hashing

### Slice B1: same-binding serving promote 禁止 second-stage CPU/host hash

目标：

- 当前 path 的第二轮 identity 不能再通过 seekable-source host path 计算。

建议实现：

1. 如果 Track A 已落地，直接删除第二轮 hash。
2. 如果仍保留第二轮 identity 入口，则：
   - 对 contiguous canonical-full binding current value 直接走 GPU hash
   - 不允许 fallback 到 host seekable-source path
3. 若第二轮 identity 仍然存在，它只能作为过渡方案；
   终态仍然应收敛到 Track A 的 single-mint effect。

### Slice B2: 对 hash 成本做显式 profile

目标：

- 满足“如果一定要算 hash，必须显式打印 hash 成本”的要求。

必须新增 profile 字段：

- `hash_stage`
- `hash_location`
- `hash_bytes`
- `hash_round`
- `hash_wall_sec`

退出标准：

- mounted Step3p5 case 中，每个 rank 的 hash 轮数可直接从 profile/log 判断；
- 若 path 仍出现 CPU hash，测试直接失败。

## Track C: Explicit Collective Policy And Ingress

### Slice C0: 在 current source-bound path 先接通 collective policy 与 hint lower

目标：

- 不先等完整 proto surface 改完，先让 current `RefillOwnedBinding` path 真正吃到
  collective hints 与 policy；
- 尽快验证 hot path 是否能离开 generic executor。

建议实现：

1. 复用 ordinary target-materialization path 现有的 hint 解析 / lower 逻辑。
2. current source-bound path 至少要能显式区分：
   - `REQUIRE_COLLECTIVE`
   - `ALLOW_NOT_ELIGIBLE_FALLBACK`
   - `DISABLE_COLLECTIVE`
3. 默认 `REQUIRE_COLLECTIVE`：
   - collective 只要被请求，`not_eligible` 或 `execution_failed` 都直接 `fatal`
4. 仅在显式开启 `ALLOW_NOT_ELIGIBLE_FALLBACK` 时：
   - `not_eligible` 类 typed reason 可以降级
   - `execution_failed` 仍然 `fatal`
5. 不允许 silent fallback；
   若发生 typed fallback，必须把 policy、reason、error_class 打进 log/profile。

### Slice C1: 为 source-bound APIs 新增显式 collective 字段

目标：

- `Binding.realize_from(...)` / `RefillOwnedBinding` 不再依赖 `operation_id#tcg`
  side-channel 请求 collective。

建议实现：

1. 在 `store_daemon.proto` 的 source-bound request 中新增显式字段：
   - `CollectiveLoadGroup`
   - `TransportSchedulingGroupHint`
   - `CollectiveMode` 或等价 typed policy enum
   - 可选 request id
2. SDK `CallContext.collective=CollectiveLoadGroup(...)` 直接映射到这些字段。
3. side-channel 解析只保留短期兼容，不再作为主路径 contract。

涉及文件：

- [proto/tensorcast/daemon/v2/store_daemon.proto](/data/workspace/tensorcast-280/proto/tensorcast/daemon/v2/store_daemon.proto)
- [tensorcast/api/context.py](/data/workspace/tensorcast-280/tensorcast/api/context.py)
- [tensorcast/api/store/binding.py](/data/workspace/tensorcast-280/tensorcast/api/store/binding.py)
- [tensorcast/api/store/owned_binding_slot.py](/data/workspace/tensorcast-280/tensorcast/api/store/owned_binding_slot.py)

### Slice C2: `owned_binding_service` 必须真正 lower collective hints

目标：

- current same-binding source-bound path 能把 collective/topology hint 写入
  `MaterializeHints`。
- current same-binding source-bound path 能把 collective policy 也写入
  `MaterializeHints` 或等价 execution contract。

建议实现：

- 复用 `target_materialization_service.cc` 现有 hint lower 逻辑；
- 不允许 source-bound path 和 ordinary target-materialization path 各自维护两套
  collective hint 解释器。
- 同时复用统一的 policy / error-class lower 逻辑，保证：
  - `not_eligible` 与 `execution_failed` 语义一致
  - fallback 只在 `ALLOW_NOT_ELIGIBLE_FALLBACK` 下发生

涉及文件：

- [daemon/service/controllers/owned_binding_service.cc](/data/workspace/tensorcast-280/daemon/service/controllers/owned_binding_service.cc)
- [daemon/service/controllers/target_materialization_service.cc](/data/workspace/tensorcast-280/daemon/service/controllers/target_materialization_service.cc)

退出标准：

- current source-bound path 可以看到 `hints.collective_load_group.has_value()`
- daemon profile/log 明确显示 hint 被接收；
- daemon profile/log 明确显示 collective policy、fallback reason、error class。

## Track D: Make 0108 Executors Reach This Path

### Slice D1: same-binding mapped realization 优先 collective path

目标：

- 在 TP>1 且 hint 完整时，same-binding path 优先尝试 mapped collective executor。

建议实现：

- `materialize_mapped_into_target(...)` 这条 path 在 source-bound 请求中能进入
  `try_collective_mapped_target_load(...)`
- 若 collective preconditions 不满足，应返回 typed reason，而不是静默退回 generic path

### Slice D2: 若未命中 collective，继续补 tensor-aware local executor

目标：

- source-bound path 不应长期停在 `GenericByteRangeExecutor(source_ordered)`。

建议实现：

- 把 0108 计划中尚未完成的 dedicated common-runtime local tensor-aware executor
  真正接到 source-bound realization path。

## Track E: Docs and Tests

### Slice E1: 文档修正

必须修正的口径：

1. `fast_path_validated` 不能被表述成“性能快路径已验证”。
2. 若 mounted latest case 仍显示 `seal_from_cut.compute_data_multihash ~15.41s`，
   文档不得继续沿用“当前主路径已到 0.2ms”作为现态描述。
3. 当前 path 的 artifact model 要明确写成：
   - 1 shared source artifact
   - 8 rank-local serving artifact
   - “无独立 GS-visible seal artifact 槽位”仅作为目标终态，不应再被写成当前轮
     性能修复的前置条件

### Slice E2: 测试

必须补的测试：

1. same-binding representation publish path 在 `source_version_key=None` 时实现
   single-mint effect
   - 当前轮至少要求 second-stage full-data hash 不再发生
   - 不强制要求当前轮先删掉独立 `seal artifact` 槽位
2. source-bound `RefillOwnedBinding` 能消费 collective hints 与 collective policy
3. `REQUIRE_COLLECTIVE` 下，collective 没命中或执行失败会直接失败
4. `ALLOW_NOT_ELIGIBLE_FALLBACK` 下，仅 typed `not_eligible` 原因允许降级
5. mounted TP8 case 不再以 `GenericByteRangeExecutor(source_ordered)` 为 dominant
   executor

# Suggested Execution Order

1. 先做 Track C：
   - 先把 current source-bound path 的 collective policy / hint lower 接通
   - 再把 collective/topology ingress 逐步收敛成 first-class contract
2. 再做 Track A：
   - 先落 A2，复用第一次 identity
   - A1 保留为后续设计选项
3. 再做 Track B：
   - 强制 GPU-only single-round hash
4. 最后做 Track D：
   - 把 mapped collective / tensor-aware local executor 真正打进 same-binding path
5. 同步补 Track E：
   - docs / tests / observability

# Acceptance Criteria

## Functional

- Step3p5 mounted 8xH800 cold-start 仍 correctness green；
- runtime 仍只消费 `serving_artifact` / strict runtime policy；
- 无 runtime source fallback。

## Identity

- 当前 path 至少不再做第二轮 full-data hash；
- 若过渡期仍保留独立 GS-visible `seal artifact`，它也必须复用第一次 identity，
  不得再付出额外 full-data hash；
- 每个 rank 只做一次最终必要的 GPU hash；
- TP8 总计 8 次 rank-local hash，而不是接近 16 次 full-data pass。

## Transport / Executor

- source-bound path 能显式看到 collective hints 与 collective policy；
- 默认 `REQUIRE_COLLECTIVE` 下，不允许 silent fallback；
- `ALLOW_NOT_ELIGIBLE_FALLBACK` 下，typed fallback reason 必须显式可见；
- daemon 日志不再长期显示：
  `dominant_executor=GenericByteRangeExecutor(source_ordered)`；
- mounted TP8 case 至少能命中：
  - `collective_handled=1`，或
  - 非 generic dominant executor。

## Performance

必须至少满足以下阶段门槛：

### Phase 1 gate

- `publication.wait_assembly_attempt` 不再稳定在 `~37s/rank`
- second-stage hash 成本不再存在

### Phase 2 gate

- `materialize_mapped_into_target` 主要时间不再维持在 `~142s/rank`
- total ready 时间显著收敛，不再维持当前 `~190s` 额外差值

### Phase 3 sign-off

- 在 mounted 8xH800 case 上接近 default baseline，且 remaining delta 只来自 serving
  contract / manifest 的少量固定成本

# Risks

1. 如果直接删独立 seal artifact 槽位，但仍有通用 `source_publish_only`
   调用者依赖 `source_version_key`，会破坏 repo-wide source lineage 语义。
   解决方法：A1 只作为后续设计选项，对 same-binding serving bootstrap path 条件化收敛。
2. 如果 collective 仍靠 `operation_id` side-channel，第三方框架仍然难以自然命中
   最优 path。
3. 如果默认就自动 fallback，会把 collective ingress / executor bug 掩盖成“只是慢”。
   解决方法：默认 `REQUIRE_COLLECTIVE`，仅显式开启时允许 typed fallback。
4. 如果第二轮 hash 改成“保留但 GPU-only”，短期能减速但不能从对象模型上消除重复
   identity；这应被视为过渡方案，不是终态。

# Reference Evidence

- [/data/tc/s35-0112/tc-20260328-054200/status.json](/data/tc/s35-0112/tc-20260328-054200/status.json)
- [/data/tc/s35-0112/tc-20260328-054200/logs/daemon.log](/data/tc/s35-0112/tc-20260328-054200/logs/daemon.log)
- [/data/tc/s35-0112/tc-20260328-054200/logs/vllm.log](/data/tc/s35-0112/tc-20260328-054200/logs/vllm.log)
- [/data/tc/s35-default/default-20260328-055149/status.json](/data/tc/s35-default/default-20260328-055149/status.json)
- [/data/workspace/internal-vllm/docs/design/tensorcast_step3p5_from_disk_cold_start_performance_followup.md](/data/workspace/internal-vllm/docs/design/tensorcast_step3p5_from_disk_cold_start_performance_followup.md)
