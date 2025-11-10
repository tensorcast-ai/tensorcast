---
title: Core/Local 重构接入计划
description: 将 core/local 重构部分接入现有 StoreEngine 系统的详细计划
design_id: 0015
status: draft
created: 2025-01-XX
---

# Core/Local 重构接入计划

## 概述

本文档描述如何将 `core/local` 目录下的重构部分（LocalManager、Artifact、View、Replica、Chunk、DataChunk）接入现有的 StoreEngine 系统，替换或增强现有系统的关键组件。

## 现有系统架构分析

### 核心组件

1. **StoreEngine** (`core/store/store_engine.h`)
   - 核心引擎，管理 artifact 注册、加载、生命周期
   - 关键 API：`begin_register_artifact()`, `commit_registered_artifact()`, `materialize_replica()`
   - 使用 `ReplicaRegistry` 管理 replica 实例
   - 使用 `ReplicaLoadController` 管理内存和加载

2. **Replica** (`core/store/replica/replica.h`)
   - 管理单个 artifact replica 的生命周期
   - 封装 `ReplicaLoadController` 和 `IArtifactLoader`
   - 提供 `ensure_loaded_async()`, `release_memory()` 等接口

3. **ReplicaLoadController** (`core/store/replica/replica_load_controller.h`)
   - 管理内存分配、状态转换、加载协调
   - 与 `UnifiedMemoryAuthority` (UMA) 协作管理 chunk 状态
   - 与 `VirtualAddressSpace` (VS) 协作管理 DRAM 虚拟地址空间

4. **UnifiedMemoryAuthority** (`core/store/replica/unified_memory_authority.h`)
   - 管理跨设备的统一内存状态
   - 跟踪每个 chunk 在 CPU/GPU 上的状态（COLD/HOT/COPIED_GPU）
   - 提供 `plan_load()`, `commit_load()` 等接口

5. **VirtualAddressSpace** (`core/common/memory/virtual_address_space.h`)
   - 管理 DRAM 虚拟地址空间
   - 提供 pageable CPU 内存分配

### 关键流程：begin_register_artifact

```cpp
// 1. 创建 Replica 实例（通过 Replica::create()）
// 2. 分配 GPU 内存（replica->get_memory_manager().allocate_memory(GPU)）
// 3. 获取 CUDA IPC handle
// 4. 创建 PendingRegistrationEntry 并存储
```

### 关键流程：commit_registered_artifact

```cpp
// 1. 计算 content-addressed artifact_id (mi2:...)
// 2. 检查是否已存在（idempotent）
// 3. 注册到 replica_registry_
// 4. 导出远程内存 keys（如果启用 P2P）
// 5. 注册到 Global Store
```

## Core/Local 重构部分分析

### 核心组件

1. **LocalManager** (`core/local/meta/local_manager.h`)
   - 静态接口，管理 artifacts、views、replicas
   - 提供 `get_or_create_artifact()`, `create_view()`, `bind_replica_source()`, `load_replica()`

2. **Artifact** (`core/local/meta/artifact.h`)
   - 管理 artifact 元数据
   - 包含多个 View 实例

3. **View** (`core/local/meta/view.h`)
   - 管理视图元数据和 chunks 映射
   - 支持 Vanilla、Slice、Transpose 等类型

4. **Replica** (`core/local/meta/replica.h`)
   - **注意**：与现有系统的 `Replica` 同名但不同
   - 表示特定设备上的一个视图
   - 提供迭代器访问 DataChunk

5. **Chunk** (`core/local/chunk/chunk.h`)
   - chunk 元数据
   - 管理多个设备上的 DataChunk 实例

6. **DataChunk** (`core/local/chunk/data_chunk.h`)
   - 特定设备上的 chunk 数据
   - 管理加载状态、锁状态、loader 注册

7. **ChunkLoader** (`core/local/loader/chunk_loader.h`)
   - 数据加载器接口
   - 实现：`DiskChunkLoader`, `DramChunkLoader`

## 架构差异分析

### 关键差异

1. **抽象层次不同**
   - 现有系统：artifact 级别（`store::replica::Replica` 管理整个 artifact）
   - `core/local`：chunk 级别（细粒度管理每个 chunk）

2. **内存管理模型**
   - 现有系统：UMA + VS（统一内存管理 + 虚拟地址空间）
   - `core/local`：DataChunk 直接管理设备内存

3. **加载器接口**
   - 现有系统：`IArtifactLoader` / `SeekableSource`（artifact 级别）
   - `core/local`：`ChunkLoader`（chunk 级别）

4. **状态管理**
   - 现有系统：`MemoryState` (UNINITIALIZED/UNALLOCATED/ALLOCATED/LOADING/LOADED/FAILED)
   - `core/local`：`DataChunk::loaded_`, `DataChunk::lock_refcnt_`

## 接入策略

### 策略选择：渐进式集成

采用**适配层模式**，在保持现有系统稳定的同时，逐步将 `core/local` 的功能接入。

### 阶段 1：适配层实现

#### 1.1 LocalManagerAdapter

创建一个适配层，将 `LocalManager` 的接口映射到 `StoreEngine` 的内部实现：

```cpp
// core/store/components/local_manager_adapter.h
namespace tensorcast::store::components {

class LocalManagerAdapter {
 public:
  explicit LocalManagerAdapter(StoreEngine* engine);
  
  // 将 LocalManager 的接口适配到 StoreEngine
  absl::StatusOr<local::meta::Artifact*> get_or_create_artifact(
      const std::string& artifact_id);
  
  absl::StatusOr<local::meta::View*> create_view(
      local::meta::Artifact* artifact,
      const std::string& view_id,
      local::meta::View::ViewType view_type,
      size_t size);
  
 private:
  StoreEngine* engine_;
  // 维护 LocalManager 的 artifacts 映射到 StoreEngine 的 replicas
  std::unordered_map<std::string, loading::ReplicaKey> artifact_to_replica_;
};

} // namespace tensorcast::store::components
```

#### 1.2 ChunkStateSync

实现 `core/local` 的 `DataChunk` 状态与 UMA chunk 状态的同步：

```cpp
// core/store/components/chunk_state_sync.h
namespace tensorcast::store::components {

class ChunkStateSync {
 public:
  // 从 UMA 同步 chunk 状态到 DataChunk
  static absl::Status sync_uma_to_datachunk(
      const loading::ReplicaKey& replica_key,
      uint32_t chunk_idx,
      local::data::DataChunk* data_chunk,
      UnifiedMemoryAuthority* uma);
  
  // 从 DataChunk 同步状态到 UMA
  static absl::Status sync_datachunk_to_uma(
      const loading::ReplicaKey& replica_key,
      uint32_t chunk_idx,
      const local::data::DataChunk* data_chunk,
      UnifiedMemoryAuthority* uma);
};

} // namespace tensorcast::store::components
```

#### 1.3 ChunkLoaderBridge

将 `ChunkLoader` 桥接到现有的 `IArtifactLoader` 接口：

```cpp
// core/store/loader/chunk_loader_bridge.h
namespace tensorcast::store::loader {

class ChunkLoaderBridge : public IArtifactLoader {
 public:
  ChunkLoaderBridge(
      const loading::ReplicaKey& key,
      const local::meta::Replica& local_replica);
  
  absl::StatusOr<loading::ReplicaHandle> load(
      const loading::ReplicaTarget& target,
      const loading::MaterializeHints& hints) override;
  
 private:
  loading::ReplicaKey key_;
  local::meta::Replica local_replica_;
};

} // namespace tensorcast::store::loader
```

### 阶段 2：begin_register_artifact 重构

#### 2.1 使用 LocalManager 创建 Artifact 和 View

```cpp
// 在 begin_register_artifact 中：
// 1. 使用 LocalManager::get_or_create_artifact() 创建 artifact
// 2. 如果需要 view，使用 LocalManager::create_view()
// 3. 创建 local::meta::Replica 实例
```

#### 2.2 内存分配适配

```cpp
// 将 LocalManager 的内存分配请求映射到 ReplicaLoadController：
// - LocalManager 的 GPU 内存分配 → ReplicaLoadController::allocate_memory(GPU)
// - LocalManager 的 CPU 内存分配 → ReplicaLoadController::allocate_memory(CPU)
```

### 阶段 3：commit_registered_artifact 重构

#### 3.1 状态同步

```cpp
// 在 commit_registered_artifact 中：
// 1. 同步 DataChunk 状态到 UMA
// 2. 确保所有 chunks 都已加载
// 3. 计算 content-addressed artifact_id
```

#### 3.2 Global Store 注册

```cpp
// 保持现有逻辑，但使用 LocalManager 的 artifact/view 信息
```

### 阶段 4：materialize_replica 集成

#### 4.1 使用 ChunkLoaderBridge

```cpp
// 在 materialize_replica 中：
// 1. 检查是否可以使用 LocalManager 的 Replica
// 2. 如果可以，使用 ChunkLoaderBridge 包装
// 3. 否则回退到现有逻辑
```

## 关键实现细节

### 1. 命名空间冲突解决

**问题**：`core/local` 的 `Replica` 与现有系统的 `store::replica::Replica` 同名。

**解决方案**：
- 保持 `core/local` 的 `Replica` 为 `local::meta::Replica`
- 在适配层中明确区分两种类型
- 使用类型别名提高可读性：

```cpp
namespace tensorcast::store::components {
  using LocalReplica = local::meta::Replica;
  using StoreReplica = store::replica::Replica;
}
```

### 2. 内存管理统一

**问题**：`core/local` 的 `DataChunk` 直接管理内存，而现有系统使用 UMA + VS。

**解决方案**：
- `DataChunk` 的内存分配委托给 `ReplicaLoadController`
- `DataChunk::base_addr_` 从 UMA/VS 获取
- 状态同步通过 `ChunkStateSync` 维护一致性

### 3. Chunk 大小对齐

**问题**：`core/local` 使用 `LocalConfig::chunk_size`（默认 2MB），而现有系统使用 `artifact_chunk_bytes`。

**解决方案**：
- 在适配层中统一 chunk 大小
- 优先使用 `StoreEngine::get_artifact_chunk_bytes()`
- 如果 `core/local` 的 chunk 大小不同，进行分片或合并

### 4. 加载器优先级

**问题**：`core/local` 的 `ChunkLoader` 有优先级（High/Low），而现有系统没有。

**解决方案**：
- 在 `ChunkLoaderBridge` 中实现优先级逻辑
- High 优先级 loader 优先执行
- Low 优先级 loader 作为后备

## 需要实现的关键功能

### 1. LocalManagerAdapter（高优先级）

- [ ] 实现 `get_or_create_artifact()` 适配
- [ ] 实现 `create_view()` 适配
- [ ] 实现 artifact ↔ replica 映射管理
- [ ] 实现生命周期管理（与 StoreEngine 同步）

### 2. ChunkStateSync（高优先级）

- [ ] 实现 UMA chunk 状态 → DataChunk 状态同步
- [ ] 实现 DataChunk 状态 → UMA chunk 状态同步
- [ ] 处理状态不一致的恢复逻辑

### 3. ChunkLoaderBridge（中优先级）

- [ ] 实现 `IArtifactLoader` 接口
- [ ] 将 chunk 级别的加载转换为 artifact 级别
- [ ] 处理并发加载和错误恢复

### 4. begin_register_artifact 集成（高优先级）

- [ ] 使用 `LocalManager::get_or_create_artifact()`
- [ ] 创建 `local::meta::View`（如果需要）
- [ ] 创建 `local::meta::Replica`
- [ ] 内存分配适配

### 5. commit_registered_artifact 集成（高优先级）

- [ ] 状态同步（DataChunk → UMA）
- [ ] 确保所有 chunks 已加载
- [ ] 保持现有 Global Store 注册逻辑

### 6. materialize_replica 集成（中优先级）

- [ ] 检查是否可以使用 LocalManager
- [ ] 使用 `ChunkLoaderBridge` 包装
- [ ] 回退到现有逻辑

### 7. 测试和验证（高优先级）

- [ ] 单元测试：LocalManagerAdapter
- [ ] 单元测试：ChunkStateSync
- [ ] 集成测试：begin_register_artifact 流程
- [ ] 集成测试：commit_registered_artifact 流程
- [ ] 性能测试：对比现有实现

## 迁移路径

### Phase 1: 基础适配层（2-3 周）

1. 实现 `LocalManagerAdapter`
2. 实现 `ChunkStateSync`
3. 单元测试

### Phase 2: begin_register_artifact 集成（1-2 周）

1. 修改 `begin_register_artifact` 使用 `LocalManager`
2. 内存分配适配
3. 集成测试

### Phase 3: commit_registered_artifact 集成（1-2 周）

1. 状态同步实现
2. 修改 `commit_registered_artifact`
3. 集成测试

### Phase 4: materialize_replica 集成（1-2 周）

1. 实现 `ChunkLoaderBridge`
2. 修改 `materialize_replica`
3. 集成测试

### Phase 5: 优化和清理（1 周）

1. 性能优化
2. 代码清理
3. 文档更新

## 风险评估

### 高风险项

1. **状态不一致**：UMA 和 DataChunk 状态可能不同步
   - **缓解**：实现强一致性检查，定期同步

2. **内存泄漏**：两个系统管理内存可能导致泄漏
   - **缓解**：统一内存所有权，明确生命周期

3. **性能退化**：适配层可能引入开销
   - **缓解**：性能测试，必要时优化

### 中风险项

1. **命名空间冲突**：类型名称冲突
   - **缓解**：使用明确的命名空间和类型别名

2. **Chunk 大小不匹配**：可能导致数据错误
   - **缓解**：严格验证和测试

## 后续工作

1. **逐步替换**：在验证稳定后，逐步用 `core/local` 替换现有实现
2. **API 统一**：统一 `LocalManager` 和 `StoreEngine` 的 API
3. **文档更新**：更新架构文档和使用指南

## 参考

- [Store Engine Architecture](../architecture/architecture-overview.md)
- [State Management](../architecture/state-management.md)
- [UMA V3 Design](../../docs/designs/0012-uma-dvmp-transactional-transfer.md)

