# 内存张量字典（Tensor Dict）注册与远程加载设计方案

## 1. 需求背景

### 1.1 业务需求

在分布式模型推理场景中，常见的使用模式是：
- 某个节点已经在GPU显存中加载了模型权重（通过PyTorch等框架）
- 其他节点需要访问这些权重，但不希望重复从磁盘加载
- 需要一种机制能够将内存中的张量直接注册到系统，供其他节点通过P2P方式访问

### 1.2 技术需求

- 支持将 `unordered_map<string, torch::Tensor>` 格式的张量字典注册到 CheckpointStore
- 张量数据已经在GPU显存中，需要直接使用现有内存地址
- 注册的张量字典能够被全局存储（Global Store）感知
- 其他节点能够通过P2P机制远程加载这些张量
- 与现有的文件加载机制无缝集成

## 2. 设计目标

### 2.1 功能目标

1. **零拷贝注册**：直接使用PyTorch张量的内存地址，避免数据拷贝
2. **统一抽象**：与现有的 DiskSource、P2PSource 保持一致的加载接口
3. **全局可见**：注册的张量字典能够被Global Store索引和发现
4. **高效传输**：支持通过RDMA/TCP进行高效的P2P传输
5. **生命周期管理**：确保张量内存在被远程访问期间不被释放

### 2.2 非功能目标

1. **兼容性**：不破坏现有的文件格式和加载机制
2. **可扩展性**：设计应易于扩展到其他内存数据源
3. **安全性**：防止未授权的内存访问
4. **性能**：最小化额外开销，保持原有的传输性能

## 3. 总体设计

### 3.1 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
│  torch::Tensor dict ──► register_tensor_dict() API          │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     CheckpointStore                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              New: TensorDictSource                   │    │
│  │  - Wraps torch::Tensor references                   │    │
│  │  - Generates virtual tensor_index metadata          │    │
│  │  - Manages tensor lifecycle                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                │                              │
│                                ▼                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              New: TensorDictLoader                   │    │
│  │  - Implements IModelLoader interface                │    │
│  │  - Provides SeekableSource for tensor data          │    │
│  │  - Handles memory registration with CommEngine      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                      Global Store                            │
│  - Registers tensor dict as a model replica                 │
│  - Tracks chunk availability and memory keys                │
│  - Coordinates P2P transfers                                │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件设计

#### 3.2.1 新增数据类型

```cpp
// In core/store/loading/loading_spec.h

/**
 * @brief In-memory tensor dictionary source
 */
struct TensorDictSource {
  // 张量名称到张量的映射
  std::unordered_map<std::string, torch::Tensor> tensors;
  
  // 可选：模型元信息
  std::optional<ModelMetadata> metadata;
  
  // 内存位置（GPU或CPU）
  ModelLocation location;
  
  // GPU设备ID（如果在GPU上）
  int device_id = -1;
};

// 扩展 ModelSource variant
using ModelSource = std::variant<
    DiskSource,
    P2PSource,
    InlineBufferSource,
    TensorDictSource  // 新增
>;
```

#### 3.2.2 TensorDictLoader 实现

```cpp
// core/store/loader/tensor_dict_loader.h

class TensorDictLoader : public IModelLoader {
public:
  explicit TensorDictLoader(TensorDictSource source);
  
  absl::Status initialize() override;
  absl::StatusOr<uint64_t> get_model_size() override;
  absl::StatusOr<std::unique_ptr<SeekableSource>> open_source() override;

private:
  // 生成虚拟的 tensor_index.json 格式元数据
  absl::Status generate_tensor_index();
  
  // 计算张量布局和偏移
  absl::Status compute_tensor_layout();
  
private:
  TensorDictSource source_;
  std::vector<TensorMetadata> tensor_metadata_;
  uint64_t total_size_ = 0;
  bool initialized_ = false;
};
```

#### 3.2.3 TensorDictSeekableSource 实现

```cpp
// core/store/loader/tensor_dict_source.h

class TensorDictSeekableSource : public SeekableSource {
public:
  struct Options {
    std::vector<torch::Tensor> tensors;  // 按偏移顺序排列
    std::vector<TensorMetadata> metadata;
    uint64_t total_size;
  };
  
  explicit TensorDictSeekableSource(Options options);
  
  absl::StatusOr<size_t> read_at(
      uint64_t offset, void* dst, size_t bytes) override;
  
  bool supports_direct_write() const override { return true; }
  
  absl::StatusOr<size_t> read_into(
      uint64_t dest_va_offset, 
      size_t bytes, 
      const DirectWriteToken& token) override;

private:
  // 根据偏移找到对应的张量和局部偏移
  std::pair<size_t, size_t> find_tensor_at_offset(uint64_t offset);
  
private:
  Options options_;
};
```

### 3.3 API 设计

#### 3.3.1 注册接口

```cpp
// In CheckpointStore class

/**
 * @brief 注册内存中的张量字典
 * 
 * @param model_id 模型标识符
 * @param tensor_dict 张量字典
 * @param device_key 目标设备
 * @param enable_remote_access 是否启用远程访问
 * @return StatusOr<ModelHandle> 模型句柄
 */
absl::StatusOr<ModelHandle> register_tensor_dict(
    std::string_view model_id,
    const std::unordered_map<std::string, torch::Tensor>& tensor_dict,
    const DeviceKey& device_key,
    bool enable_remote_access = true);
```

#### 3.3.2 实现流程

```cpp
absl::StatusOr<ModelHandle> CheckpointStore::register_tensor_dict(
    std::string_view model_id,
    const std::unordered_map<std::string, torch::Tensor>& tensor_dict,
    const DeviceKey& device_key,
    bool enable_remote_access) {
    
  // 1. 验证输入
  if (tensor_dict.empty()) {
    return absl::InvalidArgumentError("Empty tensor dictionary");
  }
  
  // 2. 检查所有张量是否在同一设备上
  auto location = validate_and_get_location(tensor_dict, device_key);
  if (!location.ok()) {
    return location.status();
  }
  
  // 3. 创建 TensorDictSource
  TensorDictSource source{
    .tensors = tensor_dict,
    .location = *location,
    .device_id = device_key.ordinal
  };
  
  // 4. 创建 ModelConfig
  ModelConfig config{
    .source = std::move(source),
    .model_identifier = std::string(model_id),
    .device_type = device_key.type,
    .local_device_id = device_key.ordinal,
    .pinned_memory_pool = memory_pool_,
    .dvmp = dvmp_
  };
  
  // 5. 通过标准流程创建 Model
  auto model_or = Model::create(std::move(config));
  if (!model_or.ok()) {
    return model_or.status();
  }
  
  // 6. 注册到 ModelRegistry
  InstanceKey instance_key{std::string(model_id), device_key, 0};
  auto status = model_registry_->emplace(instance_key, std::move(*model_or));
  if (!status.ok()) {
    return status;
  }
  
  // 7. 如果启用远程访问，注册到 Global Store
  if (enable_remote_access) {
    auto reg_status = register_for_remote_access(instance_key);
    if (!reg_status.ok()) {
      LOG(WARNING) << "Failed to register for remote access: " << reg_status;
    }
  }
  
  // 8. 返回 ModelHandle
  return ModelHandle{
    .instance_key = instance_key,
    .ready_future = std::async(std::launch::deferred, []{ return absl::OkStatus(); })
  };
}
```

### 3.4 元数据生成

为了与现有系统兼容，TensorDictLoader 需要生成虚拟的 tensor_index.json 格式元数据：

```cpp
absl::Status TensorDictLoader::generate_tensor_index() {
  uint64_t current_offset = 0;
  
  for (const auto& [name, tensor] : source_.tensors) {
    // 获取张量信息
    auto shape = tensor.sizes();
    auto stride = tensor.strides();
    auto dtype = get_dtype_string(tensor.dtype());
    auto storage_size = tensor.storage().nbytes();
    auto storage_offset = tensor.storage_offset();
    
    // 对齐到8字节边界
    if (current_offset % 8 != 0) {
      current_offset += (8 - current_offset % 8);
    }
    
    // 创建元数据条目
    TensorMetadata meta{
      .name = name,
      .offset = current_offset,
      .size = storage_size,
      .shape = std::vector<int64_t>(shape.begin(), shape.end()),
      .stride = std::vector<int64_t>(stride.begin(), stride.end()),
      .dtype = dtype,
      .storage_offset = storage_offset * tensor.element_size()
    };
    
    tensor_metadata_.push_back(meta);
    current_offset += storage_size;
  }
  
  total_size_ = current_offset;
  return absl::OkStatus();
}
```

### 3.5 内存管理和生命周期

#### 3.5.1 张量引用管理

```cpp
class TensorDictHolder {
public:
  explicit TensorDictHolder(
      std::unordered_map<std::string, torch::Tensor> tensors)
      : tensors_(std::move(tensors)) {
    // 增加张量的引用计数，防止被释放
    for (auto& [name, tensor] : tensors_) {
      tensor_guards_.push_back(tensor);
    }
  }
  
  ~TensorDictHolder() {
    // 自动释放引用
  }
  
private:
  std::unordered_map<std::string, torch::Tensor> tensors_;
  std::vector<torch::Tensor> tensor_guards_;  // 保持引用
};
```

#### 3.5.2 与 MemoryManager 集成

TensorDictLoader 需要特殊处理，因为内存已经分配：

```cpp
// In Model::create() for TensorDictSource
if (auto* td_source = std::get_if<TensorDictSource>(&config.source)) {
  // 创建特殊的 MemoryManager，标记为外部管理的内存
  memory_manager->set_external_memory(true);
  memory_manager->set_memory_state(td_source->location, MemoryState::LOADED);
}
```

### 3.6 远程访问流程

#### 3.6.1 注册到 Global Store

```cpp
absl::Status register_for_remote_access(const InstanceKey& key) {
  auto model = model_registry_->find(key);
  if (!model.ok()) {
    return model.status();
  }
  
  // 获取内存信息并导出chunks
  auto& mem_manager = (*model)->get_memory_manager();
  auto comm_info = mem_manager.export_chunks_for_p2p(
      ModelLocation::GPU, 
      all_chunks, 
      *comm_manager_->get_engine());
  
  if (!comm_info.ok()) {
    return comm_info.status();
  }
  
  // 注册到 Global Store
  auto replica_id = global_store_client_->register_model_replica(
      key.model_id,
      worker_id_,
      key.device,
      ModelLocation::GPU,
      mem_manager.get_model_size(),
      max_concurrency_);
      
  return replica_id.status();
}
```

#### 3.6.2 远程加载流程

其他节点通过标准的 prepare() API 加载：

```cpp
// 远程节点
auto handle = checkpoint_store->prepare(
    "model_from_tensor_dict",
    target_device,
    PrepareMode::AUTO);
```

系统会自动：
1. 查询 Global Store 找到可用的副本
2. 建立 P2P 连接
3. 通过 RemoteKeySource 传输数据
4. 在本地重建张量

## 4. 实现计划

### 4.1 第一阶段：核心功能

1. 实现 TensorDictSource 数据结构
2. 实现 TensorDictLoader 和 TensorDictSeekableSource
3. 在 Model::create() 中添加对 TensorDictSource 的支持
4. 实现 register_tensor_dict() API

### 4.2 第二阶段：集成优化

1. 与 ChunkExportService 集成，支持细粒度的chunk导出
2. 优化元数据生成，支持稀疏张量和视图
3. 添加内存压力下的自动卸载机制
4. 实现张量字典的增量更新

### 4.3 第三阶段：高级特性

1. 支持动态张量注册/注销
2. 实现张量级别的访问控制
3. 添加压缩传输支持
4. 性能监控和调优工具

## 5. 兼容性考虑

### 5.1 与现有系统的兼容

1. **统一接口**：TensorDictSource 作为 ModelSource 的一个变体，与现有加载流程无缝集成
2. **元数据兼容**：生成的虚拟 tensor_index 与文件格式规范完全兼容
3. **P2P传输**：复用现有的 CommunicateEngine 和 RemoteKeySource 机制

### 5.2 潜在的破坏性变更

1. 需要扩展 Model 类以支持外部管理的内存
2. MemoryManager 需要新增"外部内存"模式
3. Global Store 可能需要新增字段以区分文件模型和内存模型

## 6. 测试策略

### 6.1 单元测试

- TensorDictLoader 的各项功能
- 元数据生成的正确性
- 内存管理和生命周期

### 6.2 集成测试

- 本地注册和访问流程
- 远程P2P加载
- 与现有加载器的互操作性

### 6.3 性能测试

- 注册开销
- P2P传输性能
- 内存使用效率

## 7. 安全考虑

1. **访问控制**：通过 Global Store 的认证机制控制访问
2. **内存保护**：确保只能访问注册的张量内存区域
3. **生命周期管理**：防止悬空指针和内存泄漏

## 8. 未来扩展

1. **多框架支持**：除PyTorch外，支持TensorFlow、JAX等框架的张量
2. **异构内存**：支持HBM、NVMe等新型存储介质
3. **智能调度**：基于网络拓扑和负载自动选择最优的数据源
4. **联邦学习**：支持模型参数的增量同步和聚合

## 9. 风险和缓解措施

### 9.1 技术风险

1. **内存泄漏**：通过 RAII 和智能指针管理生命周期
2. **性能退化**：通过充分的性能测试和优化
3. **兼容性问题**：渐进式实施，保持向后兼容

### 9.2 操作风险

1. **误用API**：提供清晰的文档和示例
2. **资源耗尽**：实现配额和限流机制
3. **调试困难**：添加详细的日志和监控指标