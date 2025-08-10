# StepCast Store 架构分析：从 Model 到 Memory

## 1. 整体架构概览

StepCast Store 采用分层架构设计，从高层的 Model API 到底层的内存管理，形成了清晰的职责划分：

```mermaid
graph TD
    subgraph Application Layer
        A[Model - 对外API门面]
    end
    
    subgraph Memory Management Layer
        B[MemoryManager - 内存管理协调器]
        C[UnifiedModelMemory - 统一内存状态]
    end
    
    subgraph Storage Layer
        D[DVMP - 分布式虚拟内存池]
        E[CudaMemory - GPU内存]
        F[StreamingPinnedBuffer - 流式固定内存]
    end
    
    subgraph Loader Layer
        G[IModelLoader - 加载器接口]
        H[DiskLoader - 磁盘加载器]
        I[P2PLoader - 点对点加载器]
        J[Source/Pump/Sink - 统一加载架构]
    end
    
    A --> B
    B --> C
    B --> D
    B --> E
    B --> F
    A --> G
    G --> H
    G --> I
    H --> J
    I --> J
    J --> D
    J --> E
```

## 2. 核心组件详解

### 2.1 Model (模型门面)

Model 类是整个系统的对外接口，提供了统一的模型生命周期管理：

```mermaid
classDiagram
    class Model {
        -InstanceKey key_
        -IModelLoader loader_
        -MemoryManager memory_manager_
        -ModelLocation original_source_type_
        -shared_future~Status~ cpu_load_future_
        -shared_future~Status~ gpu_load_future_
        
        +create(ModelConfig) StatusOr~Model~
        +model_id() string
        +ensure_loaded_async(location, concurrency) shared_future
        +copy_from(src) Status
        +release_memory(location) Status
        +get_memory_state(location) MemoryState
        +enable_remote_memory_access(location, engine) StatusOr
        +wait_until_loaded(location, timeout) Status
    }
    
    class MemoryState {
        <<enumeration>>
        UNINITIALIZED
        UNALLOCATED
        ALLOCATED
        LOADING
        LOADED
        FAILED
    }
    
    class ModelLocation {
        <<enumeration>>
        PAGEABLE_CPU
        GPU
    }
    
    Model --> MemoryState
    Model --> ModelLocation
```

### 2.2 MemoryManager (内存管理核心)

MemoryManager 是内存管理的协调中心，负责：
- CPU 和 GPU 内存的分配与释放
- 数据传输的协调
- 外部通信接口的管理

```mermaid
classDiagram
    class MemoryManager {
        <<Internal Pods>>
        -CpuPod cpu_pod_
        -GpuPod gpu_pod_
        -UnifiedModelMemory uma_
        -DistributedMemoryPool dvmp_
        
        +allocate(location, model_size) Status
        +borrow_existing(location, ptr, size) Status
        +copy_data_async(src_location, dst_location) shared_future
        +release(location) Status
        +get_state(location) MemoryState
        +finalize_load(location, chunks) Status
        +plan_direct_write(ranges) StatusOr~DirectWriteToken~
        +get_dvmp_region() StatusOr~DvmpRegion~
    }
    
    class CpuPod {
        MemoryState state
        void* base
        size_t bytes
        StreamingPinnedBuffer* buffer
        bool comm_registered
        CommRegistrationInfo comm_info
    }
    
    class GpuPod {
        MemoryState state
        CudaMemory* memory
        cudaStream_t stream
        bool stream_ready
        bool comm_registered
        CommRegistrationInfo comm_info
    }
```

### 2.3 DVMP (分布式虚拟内存池)

DVMP 是 CPU 内存管理的核心，提供了高效的虚拟内存管理机制：

```mermaid
graph LR
    subgraph DVMP Features
        A[连续虚拟地址空间<br/>1TB VA per model]
        B[Chunk-based管理<br/>256MB chunks]
        C[零拷贝mmap]
        D[Pin Lease保护机制]
        E[按需物理内存分配]
        F[智能驱逐策略]
    end
    
    subgraph Chunk States
        G[HOT - 活跃使用]
        H[LOCKED_TX - 传输中]
        I[COPIED_GPU - 已复制到GPU]
        J[COLD - 冷数据]
        K[PREEMPTIBLE - 可抢占]
        L[EVICTED - 已驱逐]
    end
```

DVMP 的状态机：

```mermaid
stateDiagram-v2
    [*] --> COLD: 初始化
    COLD --> HOT: write_at/map_file_segments
    HOT --> LOCKED_TX: lock_chunks
    LOCKED_TX --> HOT: unlock(copied=false)
    LOCKED_TX --> COPIED_GPU: unlock(copied=true)
    HOT --> PREEMPTIBLE: mark_preemptible
    COPIED_GPU --> PREEMPTIBLE: mark_preemptible
    PREEMPTIBLE --> EVICTED: madvise/pageout
    EVICTED --> HOT: ensure_resident/refill
    
    note right of LOCKED_TX: 传输期间保护
    note right of PREEMPTIBLE: MADV_FREE优化
    note right of EVICTED: 保留VA映射
```

## 3. 统一加载器架构 (Unified Loader)

### 3.1 Source→Pump→Sink 模式

```mermaid
sequenceDiagram
    participant S as Source
    participant P as Pump
    participant B as BufferPool
    participant K as Sink
    participant D as Destination
    
    rect rgb(200, 230, 255)
        note over S,D: 生产者线程 (×N)
        loop 每个范围
            P->>B: get_free_chunk()
            P->>S: read_at(offset, buffer)
            S-->>P: bytes_read
            P->>B: mark_chunk_ready(data)
        end
    end
    
    rect rgb(255, 230, 200)
        note over S,D: 消费者线程
        loop 直到完成
            P->>B: get_ready_chunk()
            B-->>P: ReadyChunk
            P->>K: write_at(offset, data)
            K->>D: 写入目标位置
            P->>B: return_chunk()
        end
    end
    
    P->>K: close()
```

### 3.2 能力驱动的直接写入

```mermaid
classDiagram
    class PositionedSink {
        <<interface>>
        +write_at(offset, data, size) Status
        +close() Status
    }
    
    class DirectWritableSink {
        <<interface>>
        +plan_direct_write(ranges) StatusOr~DirectWriteToken~
    }
    
    class DVMPRegionSink {
        -DvmpRegion region
        -uint64_t total_size
        +write_at(offset, data, size) Status
        +plan_direct_write(ranges) StatusOr~DirectWriteToken~
    }
    
    class GPUMemorySink {
        -CudaMemory* memory
        -cudaStream_t stream
        +write_at(offset, data, size) Status
    }
    
    PositionedSink <|-- DVMPRegionSink
    DirectWritableSink <|-- DVMPRegionSink
    PositionedSink <|-- GPUMemorySink
```

直接写入协商流程：

```mermaid
sequenceDiagram
    participant P as pump_ranges
    participant S as SeekableSource
    participant D as DirectWritableSink
    participant DVMP as DVMP
    
    P->>S: supports_direct_write()?
    P->>D: 是否实现DirectWritableSink?
    
    alt 都支持
        P->>D: plan_direct_write(ranges)
        D->>DVMP: 创建pin lease
        D-->>P: DirectWriteToken
        loop 每个范围
            P->>S: read_into(token, offset, size)
            S->>DVMP: 直接写入VA
        end
    else 回退到staged路径
        P->>P: 使用BufferPool中转
    end
```

## 4. 内存管理流程

### 4.1 模型加载流程

```mermaid
sequenceDiagram
    participant User
    participant Model
    participant MM as MemoryManager
    participant Loader
    participant DVMP
    participant GPU
    
    User->>Model: ensure_loaded_async(GPU)
    Model->>MM: allocate(PAGEABLE_CPU, size)
    MM->>DVMP: allocate(model_id, size)
    DVMP-->>MM: VA空间
    
    Model->>MM: allocate(GPU, size)
    MM->>GPU: cudaMalloc(size)
    
    Model->>Loader: load_to_target(GPU)
    
    alt CPU未加载
        Loader->>DVMP: 加载到CPU
        Loader->>MM: finalize_load(CPU)
    end
    
    Loader->>MM: copy_data_async(CPU→GPU)
    MM->>DVMP: lock_chunks()
    MM->>GPU: cudaMemcpyAsync()
    MM->>DVMP: unlock_chunks(copied=true)
    MM->>DVMP: evict_tail_bytes()
    MM->>MM: 释放CPU资源
    
    Model-->>User: future<Status>
```

### 4.2 CPU→GPU 复制与强制释放

```mermaid
flowchart TD
    A[CPU→GPU复制请求] --> B{CPU数据已加载?}
    B -->|否| C[先加载到CPU]
    B -->|是| D[锁定CPU chunks]
    C --> D
    D --> E[设置CUDA流]
    E --> F[异步H2D传输]
    F --> G[更新chunk状态为COPIED_GPU]
    G --> H{复制成功?}
    H -->|是| I[驱逐CPU物理内存]
    I --> J[释放StreamingPinnedBuffer]
    J --> K[CPU状态→UNALLOCATED]
    H -->|否| L[保持原状态]
```

### 4.3 外部内存访问 (RDMA/TCP)

```mermaid
sequenceDiagram
    participant MM as MemoryManager
    participant DVMP
    participant CE as CommunicateEngine
    participant Remote
    
    rect rgb(200, 255, 200)
        note over MM,CE: 导出内存
        MM->>MM: 合并连续chunks
        MM->>DVMP: pin_range(va_off, len, ExternalShare)
        DVMP-->>MM: PinLease (保护内存)
        MM->>CE: register_tensor(key, addr, len)
        CE-->>MM: handle
    end
    
    rect rgb(255, 200, 200)
        note over MM,Remote: 远程访问
        Remote->>CE: RDMA Read/Write
        CE->>DVMP: 访问pinned内存
    end
    
    rect rgb(200, 200, 255)
        note over MM,CE: 取消导出
        MM->>CE: unregister_tensor(key)
        MM->>DVMP: ~PinLease (释放保护)
        note over DVMP: 内存可被驱逐
    end
```

## 5. DVMP 高级特性

### 5.1 Pin Lease 机制

Pin Lease 提供了细粒度的内存保护：

```mermaid
classDiagram
    class PinLease {
        -string model_id
        -uint64_t va_offset
        -size_t length
        -string reason
        -vector~uint32_t~ affected_chunks
        +~PinLease() 释放保护
    }
    
    class ChunkMeta {
        -atomic~ChunkState~ state
        -atomic~uint32_t~ pin_count
        -atomic~int64_t~ last_touch_s
        +is_pinned() bool
    }
    
    PinLease --> ChunkMeta: 增加pin_count
```

### 5.2 内存驱逐策略

```mermaid
flowchart LR
    subgraph 驱逐决策
        A[内存压力检测] --> B{MemAvailable < 15%?}
        B -->|是| C[触发驱逐]
        B -->|否| D[等待]
    end
    
    subgraph 驱逐执行
        C --> E[选择可驱逐chunks]
        E --> F{是否pinned?}
        F -->|否| G[MADV_FREE/DONTNEED]
        F -->|是| H[跳过]
        G --> I[更新状态为EVICTED]
    end
    
    subgraph 保护策略
        J[前20%数据保护]
        K[GPU已复制优先驱逐]
        L[LRU based on last_touch]
    end
```

### 5.3 透明页面错误处理

```mermaid
sequenceDiagram
    participant App as 应用
    participant DVMP
    participant GS as Global Store
    participant Remote as 远程节点
    
    App->>DVMP: 访问已驱逐内存
    DVMP-->>App: kErrChunkRemote
    
    App->>GS: 查询chunk位置
    GS-->>App: 远程节点信息
    
    App->>Remote: P2P拉取数据
    Remote-->>App: chunk数据
    
    App->>DVMP: ensure_chunk_resident()
    DVMP->>DVMP: 更新状态为HOT
```

## 6. 性能优化要点

### 6.1 并发优化
- Per-model锁减少全局竞争
- 多生产者/消费者的Pump设计
- 异步GPU传输与CPU操作重叠

### 6.2 内存优化
- Zero-copy mmap for disk reads
- MADV_FREE for efficient memory reclaim
- Chunk-aligned buffer allocation

### 6.3 网络优化
- Direct RDMA write support
- Capability-based negotiation
- Pin lease保护避免传输中断

## 7. 配置与监控

### 7.1 关键配置项

```yaml
dvmp:
  chunk_size: 256MiB          # Chunk大小
  front_guard_ratio: 0.20     # 前20%数据保护
  max_total_ratio: 0.80       # 最大系统内存使用率
  
eviction:
  interval_ms: 50             # 驱逐检查间隔
  pressure_threshold: 0.15    # 内存压力阈值
  gpu_auto_release: true      # GPU复制后自动释放CPU

streaming:
  buffer_size: 16MiB          # 流式缓冲区大小
  buffer_count: 64            # 缓冲区数量
  alignment: 4096             # 对齐要求
```

### 7.2 关键指标

- `dvmp_write_bytes_total`: DVMP写入字节数
- `dvmp_map_bytes_total`: DVMP映射字节数
- `dvmp_pin_leases_total{reason}`: Pin lease计数
- `chunk_exports_total{location}`: Chunk导出计数
- `loader_bytes_total{source,location,mode}`: 加载字节数

## 8. 总结

StepCast Store 通过精心设计的分层架构，实现了：

1. **统一的内存抽象**：DVMP提供透明的虚拟内存管理
2. **高效的数据流水线**：Source→Pump→Sink架构支持多种加载路径
3. **智能的内存管理**：自动驱逐、页面错误处理、GPU自动释放
4. **安全的并发访问**：Pin Lease机制、per-model锁
5. **灵活的扩展性**：能力驱动的接口设计

这种设计使得系统能够高效处理超大规模模型（670GB+），同时保持良好的性能和可靠性。