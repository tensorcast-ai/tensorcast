---
title: Architecture Design
description: Layered architectural design with clear responsibility boundaries
sidebar_position: 2
---

# Architecture Design

## System Architecture

The Core Store module adopts a layered architectural design, with clear responsibility boundaries from external interfaces to low-level memory management.

## Overall Architecture Diagram

```mermaid
graph TB
    subgraph "External Interface Layer"
        CS[CheckpointStore]
    end

    subgraph "Model Management Layer"
        M[Model]
        MC[ModelConfig]
        MR[ModelRegistry]
        DM[DeviceManager]
    end

    subgraph "Data Loading Layer"
        IL[IModelLoader]
        DL[DiskLoader]
        RL[P2PLoader]
    end

    subgraph "Memory Management Layer"
        MM[MemoryManager]
        MS1[MemoryState]
        ML[ModelLocation]
    end

    subgraph "Memory Implementation Layer"
        PM[PinnedMemory]
        CM[CudaMemory]
        PMP[PinnedMemoryPool]
        BV[BatchVector]
    end

    subgraph "Communication Layer"
        CMN[CommunicationManager]
        CE[CommunicateEngine]
        CRI[CommRegistrationInfo]
    end

    CS --> MR
    CS --> DM
    MR --> M
    M --> IL
    M --> MM
    M --> MC
    DL --> IL
    RL --> IL
    MM --> PM
    MM --> CM
    MM --> MS1
    MM --> ML
    PM --> PMP
    MM --> BV
    MM --> CMN
    CMN --> CE
    CE --> CRI
```

## Layer Details

### 1. External Interface Layer

**CheckpointStore** is the entry point of the entire system, providing:

- Model registration and management
- GPU device management
- Global resource coordination
- High-level API encapsulation

```cpp
class CheckpointStore {
public:
    int load_model();
};
```

### 2. Model Management Layer

**Model** class encapsulates the complete lifecycle of a single model:

```mermaid
graph LR
    subgraph "Model Internal Architecture"
        M[Model] --> MM[MemoryManager]
        M --> IL[IModelLoader]
        M --> CF[CPU Future]
        M --> GF[GPU Future]

        MM --> PM[PinnedMemory]
        MM --> CM[CudaMemory]

        IL --> DL[DiskLoader]
        IL --> RL[P2PLoader]
    end
```

**Design Features**:
- Uses Factory pattern for instance creation
- Asynchronous operation management
- Intelligent source selection strategy

### 3. Data Loading Layer

Adopts strategy pattern design, supporting multiple data sources:

```mermaid
classDiagram
    class IModelLoader {
        <<interface>>
        +initialize() Status
        +get_model_size() StatusOr~uint64_t~
        +load_async() future~Status~
    }

    class DiskLoader {
        -partition_paths_: vector~path~
        -partition_sizes_: vector~size_t~
        +load_async() future~Status~
    }

    class P2PLoader {
        -source_config_: P2PModelSource
        -comm_engine_: shared_ptr~Engine~
        +load_async() future~Status~
    }

    IModelLoader <|-- DiskLoader
    IModelLoader <|-- P2PLoader
```

**DiskLoader Workflow**:
1. Scan partition files (`tensor.data_0`, `tensor.data_1`, ...)
2. Multi-threaded parallel reading
3. Write to PinnedMemory buffers
4. Synchronize status through BatchVector

**P2PLoader Workflow**:
1. Connect to remote CommunicationManager (internally wraps CommunicateEngine)
2. Read data based on configured memory keys
3. Support CPU-CPU and GPU-GPU transfers
4. Optional data integrity verification

### 4. Memory Management Layer

**MemoryManager** is the core of memory management, managing memory at both CPU and GPU locations:

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED
    UNINITIALIZED --> UNALLOCATED: init_with_pool
    UNALLOCATED --> ALLOCATED: allocate_memory
    ALLOCATED --> LOADING: start_load
    LOADING --> LOADED: load_success
    LOADING --> FAILED: load_failed
    LOADED --> UNALLOCATED: release_memory
    ALLOCATED --> UNALLOCATED: release_memory
    FAILED --> UNALLOCATED: release_memory
```

**Memory Transfer Support**:
- CPU ↔ GPU: Asynchronous cudaMemcpyAsync
- DISK → CPU: Multi-threaded file reading
- REMOTE → CPU/GPU: Direct P2P transfer

### 5. Memory Implementation Layer

```mermaid
graph TB
    subgraph "CPU Memory"
        PMP[PinnedMemoryPool]
        PM[PinnedMemory]
        BV[BatchVector]

        PMP --> PM
        PM --> BV
    end

    subgraph "GPU Memory"
        CM[CudaMemory]
        IPC[IPC Handle]

        CM --> IPC
    end

    subgraph "Allocation Types"
        DIRECT[Direct Allocation]
        POOLED[Pool Allocation]
    end
```

**Memory Pool Design**:
- Pre-allocate fixed-size memory blocks
- Support CUDA host pinned memory
- Automatic alignment and NUMA optimization

**GPU Memory Features**:
- Support direct allocation
- Automatic CUDA context management
- Cross-process memory sharing via IPC handles

## 内存传输机制详解

### CPU 与 GPU 内存布局差异

CPU 和 GPU 使用不同的内存布局策略来优化各自的访问模式：

```mermaid
graph TB
    subgraph "CPU Memory Layout"
        direction TB
        CPUModel[Model Data: 1GB]
        CPUChunk0[Chunk 0: 256MB]
        CPUChunk1[Chunk 1: 256MB]
        CPUChunk2[Chunk 2: 256MB]
        CPUChunk3[Chunk 3: 232MB]

        CPUModel -.-> CPUChunk0
        CPUModel -.-> CPUChunk1
        CPUModel -.-> CPUChunk2
        CPUModel -.-> CPUChunk3

        CPUChunk0 --> CPUAddr0[0x1000000]
        CPUChunk1 --> CPUAddr1[0x2000000]
        CPUChunk2 --> CPUAddr2[0x3000000]
        CPUChunk3 --> CPUAddr3[0x4000000]
    end

    subgraph "GPU Memory Layout"
        direction TB
        GPUModel[Model Data: 1GB]
        GPUBuffer[Contiguous Buffer]

        GPUModel -.-> GPUBuffer
        GPUBuffer --> GPUAddr[0x700000000]

        GPUOffset0[Offset 0: 0MB]
        GPUOffset1[Offset 256MB]
        GPUOffset2[Offset 512MB]
        GPUOffset3[Offset 768MB]
    end
```

**设计原理**:
- **CPU 分块设计**: 支持并行I/O，减少内存碎片，便于池化管理
- **GPU 连续设计**: 最大化GPU带宽利用率，简化kernel访问模式

### 内存状态管理

每个内存位置都有独立的状态管理：

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED

    UNINITIALIZED --> UNALLOCATED_CPU: CPU Pool Available
    UNINITIALIZED --> UNALLOCATED_GPU: GPU Device Available

    UNALLOCATED_CPU --> ALLOCATED_CPU: allocate_memory(CPU)
    UNALLOCATED_GPU --> ALLOCATED_GPU: allocate_memory(GPU)

    ALLOCATED_CPU --> LOADING_CPU: Loader.load_async(CPU)
    ALLOCATED_GPU --> LOADING_GPU: copy_data_async(CPU→GPU)

    LOADING_CPU --> LOADED_CPU: Load Complete
    LOADING_GPU --> LOADED_GPU: Copy Complete

    LOADING_CPU --> FAILED_CPU: Load Error
    LOADING_GPU --> FAILED_GPU: Copy Error

    LOADED_CPU --> UNALLOCATED_CPU: release_memory(CPU)
    LOADED_GPU --> UNALLOCATED_GPU: release_memory(GPU)

    FAILED_CPU --> UNALLOCATED_CPU: release_memory(CPU)
    FAILED_GPU --> UNALLOCATED_GPU: release_memory(GPU)
```

### 磁盘到CPU加载机制

从磁盘加载到CPU时，数据被分块并行处理：

```mermaid
sequenceDiagram
    participant DL as DiskLoader
    participant MM as MemoryManager
    participant PM as PinnedMemory
    participant BV as BatchVector
    participant T1 as Thread1
    participant T2 as Thread2
    participant TN as ThreadN

    Note over DL,TN: 多线程并行磁盘读取

    DL->>MM: get_pinned_memory()
    MM->>PM: return shared_ptr
    DL->>MM: get_host_chunk_queue()
    MM->>BV: return shared_ptr

    DL->>MM: set_state(CPU, LOADING)

    par Thread 1
        DL->>T1: read_task(chunk_0, chunk_N/4)
        T1->>PM: write to chunk_0...chunk_N/4
        T1->>BV: enqueue(chunk_id, batch_info)
    and Thread 2
        DL->>T2: read_task(chunk_N/4, chunk_N/2)
        T2->>PM: write to chunk_N/4...chunk_N/2
        T2->>BV: enqueue(chunk_id, batch_info)
    and Thread N
        DL->>TN: read_task(chunk_3N/4, chunk_N)
        TN->>PM: write to chunk_3N/4...chunk_N
        TN->>BV: enqueue(chunk_id, batch_info)
    end

    Note over T1,TN: 所有线程完成
    DL->>MM: finalize_load_state(CPU, LOADED)
```

### CPU到GPU传输机制

CPU到GPU的传输是将分块的CPU内存重新组合成GPU的连续内存：

```mermaid
graph TB
    subgraph "Transfer Flow"
        direction TB

        subgraph "Source: CPU Chunks"
            C0[Chunk 0<br/>256MB]
            C1[Chunk 1<br/>256MB]
            C2[Chunk 2<br/>256MB]
            C3[Chunk 3<br/>232MB]
        end

        subgraph "CUDA Operations"
            Stream[CUDA Stream]
            Copy0[cudaMemcpyAsync<br/>H2D, Chunk 0]
            Copy1[cudaMemcpyAsync<br/>H2D, Chunk 1]
            Copy2[cudaMemcpyAsync<br/>H2D, Chunk 2]
            Copy3[cudaMemcpyAsync<br/>H2D, Chunk 3]
            Sync[cudaStreamSynchronize]
        end

        subgraph "Destination: GPU Buffer"
            GPU[Contiguous GPU Buffer<br/>1GB]
            Offset0[GPU + 0MB]
            Offset1[GPU + 256MB]
            Offset2[GPU + 512MB]
            Offset3[GPU + 768MB]
        end

        C0 --> Copy0
        C1 --> Copy1
        C2 --> Copy2
        C3 --> Copy3

        Copy0 --> Offset0
        Copy1 --> Offset1
        Copy2 --> Offset2
        Copy3 --> Offset3

        Copy0 --> Stream
        Copy1 --> Stream
        Copy2 --> Stream
        Copy3 --> Stream
        Stream --> Sync
        Sync --> GPU
    end
```

### GPU to CPU Transfer Mechanism

GPU to CPU transfer splits contiguous GPU memory into CPU chunks by block size:

```mermaid
graph TB
    subgraph "Reverse Transfer Flow"
        direction TB

        subgraph "Source: GPU Buffer"
            GPU2[Contiguous GPU Buffer<br/>1GB]
            SOffset0[GPU + 0MB]
            SOffset1[GPU + 256MB]
            SOffset2[GPU + 512MB]
            SOffset3[GPU + 768MB]
        end

        subgraph "CUDA Operations"
            Stream2[CUDA Stream]
            RCopy0[cudaMemcpyAsync<br/>D2H, 256MB]
            RCopy1[cudaMemcpyAsync<br/>D2H, 256MB]
            RCopy2[cudaMemcpyAsync<br/>D2H, 256MB]
            RCopy3[cudaMemcpyAsync<br/>D2H, 232MB]
            Sync2[cudaStreamSynchronize]
        end

        subgraph "Destination: CPU Chunks"
            DC0[Chunk 0<br/>256MB]
            DC1[Chunk 1<br/>256MB]
            DC2[Chunk 2<br/>256MB]
            DC3[Chunk 3<br/>232MB]
        end

        SOffset0 --> RCopy0
        SOffset1 --> RCopy1
        SOffset2 --> RCopy2
        SOffset3 --> RCopy3

        RCopy0 --> DC0
        RCopy1 --> DC1
        RCopy2 --> DC2
        RCopy3 --> DC3

        RCopy0 --> Stream2
        RCopy1 --> Stream2
        RCopy2 --> Stream2
        RCopy3 --> Stream2
        Stream2 --> Sync2
    end
```

### P2P传输支持

P2P传输根据源和目标的内存布局采用不同策略：

```mermaid
graph TB
    subgraph "P2P Transfer Scenarios"

        subgraph "CPU to CPU (Supported)"
            direction LR
            RemoteCPU[Remote CPU<br/>Chunks]
            LocalCPU[Local CPU<br/>Chunks]

            RC0[Remote Chunk 0] --> LC0[Local Chunk 0]
            RC1[Remote Chunk 1] --> LC1[Local Chunk 1]
            RC2[Remote Chunk 2] --> LC2[Local Chunk 2]
            RC3[Remote Chunk 3] --> LC3[Local Chunk 3]

            RC0 -.->|P2P read_tensor| LC0
            RC1 -.->|P2P read_tensor| LC1
            RC2 -.->|P2P read_tensor| LC2
            RC3 -.->|P2P read_tensor| LC3
        end

        subgraph "GPU to GPU (Supported)"
            direction LR
            RemoteGPU[Remote GPU<br/>Buffer]
            LocalGPU[Local GPU<br/>Buffer]

            RemoteGPU -->|P2P read_tensor<br/>Full Buffer| LocalGPU
        end

        subgraph "CPU to GPU (Unsupported)"
            direction LR
            RemoteCPU2[Remote CPU<br/>Chunks]
            LocalGPU2[Local GPU<br/>Buffer]

            RemoteCPU2 -.->|Requires Offset Support| LocalGPU2

            Note1[Requires read_tensor support<br/>for remote offsets]
        end

        subgraph "GPU to CPU (Unsupported)"
            direction LR
            RemoteGPU2[Remote GPU<br/>Buffer]
            LocalCPU2[Local CPU<br/>Chunks]

            RemoteGPU2 -.->|Requires Offset Support| LocalCPU2

            Note2[Requires read_tensor support<br/>for remote offsets]
        end
    end
```

**P2P Transfer Limitations**:
- **CPU↔CPU**: Fully supported, 1:1 chunk-to-chunk mapping
- **GPU↔GPU**: Fully supported, single buffer to single buffer transfer
- **CPU↔GPU**: Fully supported, multiple chunks to single buffer transfer

### Memory Transfer Performance Optimization

```mermaid
graph TB
    subgraph "Performance Optimization Strategies"

        subgraph "Concurrency Optimization"
            MultiThread[Multi-threaded disk reading]
            AsyncCUDA[Asynchronous CUDA transfer]
            StreamOverlap[Stream overlapped execution]
        end

        subgraph "Memory Optimization"
            PinnedMem[Pinned memory reduces copying]
            MemoryPool[Memory pool reduces allocation overhead]
            ChunkAlignment[Chunk alignment optimizes bandwidth]
        end

        subgraph "Transfer Optimization"
            DirectP2P[Direct P2P transfer]
            IPCSharing[IPC zero-copy sharing]
            PipelineTransfer[Pipeline transfer]
        end

        MultiThread --> PinnedMem
        AsyncCUDA --> StreamOverlap
        MemoryPool --> ChunkAlignment
        DirectP2P --> IPCSharing
        PipelineTransfer --> StreamOverlap
    end
```

### Transfer Failure Handling

```mermaid
stateDiagram-v2
    [*] --> TransferStart

    TransferStart --> CheckStates: Verify source/destination state
    CheckStates --> SetLoading: State check passed
    CheckStates --> TransferFailed: State check failed

    SetLoading --> CapturePointers: Capture memory pointers
    CapturePointers --> LaunchAsync: Launch async task

    LaunchAsync --> Transferring: Execute transfer operation

    Transferring --> ValidateResult: Transfer completed
    Transferring --> TransferError: Transfer error

    ValidateResult --> SetLoaded: Validation succeeded
    ValidateResult --> TransferError: Validation failed

    TransferError --> SetFailed: Set failed state
    SetLoaded --> TransferComplete
    SetFailed --> TransferComplete
    TransferFailed --> TransferComplete

    TransferComplete --> [*]
```

## Core Interaction Flows

### Model Loading Flow

```mermaid
sequenceDiagram
    participant User
    participant CS as CheckpointStore
    participant M as Model
    participant MM as MemoryManager
    participant L as Loader
    participant PM as PinnedMemory
    participant BV as BatchVector

    User->>CS: load_model_from_disk(id)
    CS->>M: create(DiskModelSource)
    M->>MM: set_model_size(size)
    M->>MM: allocate_memory(CPU)

    MM->>MM: transition to ALLOCATED
    Note over MM: CPU chunked memory allocation completed

    M->>L: load_async(CPU)
    L->>MM: get_pinned_memory()
    MM->>PM: return chunks vector
    L->>MM: get_host_chunk_queue()
    MM->>BV: return batch tracker

    L->>MM: set_state(CPU, LOADING)

    Note over L: Multi-threaded parallel reading of disk partitions
    loop Each disk partition file
        L->>L: multi_thread_read(partition_i)
        L->>PM: write_chunk(chunk_id, data)
        L->>BV: enqueue(chunk_id, batch_info)
    end

    L->>MM: finalize_load_state(CPU, LOADED)
    MM->>M: notify completion
    M->>CS: return success
```

### P2P Loading Flow

Different P2P transfer scenarios adopt different handling approaches:

```mermaid
sequenceDiagram
    participant User
    participant CS as CheckpointStore
    participant M as Model
    participant MM as MemoryManager
    participant RL as P2PLoader
    participant CM as CommunicationManager

    User->>CS: load_model(p2p_config)
    CS->>M: create(P2PModelSource)

    Note over M: Check P2P transfer type
    alt CPU to CPU Transfer
        M->>MM: allocate_memory(CPU)
        MM->>MM: transition CPU to ALLOCATED
        M->>RL: load_async(CPU)

        RL->>MM: set_state(CPU, LOADING)

        Note over RL: Chunk-to-Chunk 1:1 mapping
        loop Each remote CPU chunk
            RL->>CM: read_tensor(remote_key_i, local_chunk_i, size)
            CM->>RL: transfer complete
        end

        RL->>MM: finalize_load_state(CPU, LOADED)

    else GPU to GPU Transfer
        M->>MM: allocate_memory(GPU)
        MM->>MM: transition GPU to ALLOCATED
        M->>RL: load_async(GPU)

        RL->>MM: set_state(GPU, LOADING)

        Note over RL: Single Buffer to Single Buffer transfer
        RL->>CM: read_tensor(remote_gpu_key, local_gpu_ptr, model_size)
        CM->>RL: transfer complete

        RL->>MM: finalize_load_state(GPU, LOADED)

    else CPU to GPU / GPU to CPU
        Note over RL: Currently unsupported, return error
        RL->>M: return UnimplementedError
    end

    M->>CS: return success/error
```

### IPC Memory Sharing Flow

GPU memory can be shared between processes through IPC handles:

```mermaid
sequenceDiagram
    participant P1 as Process1(Owner)
    participant P2 as Process2(User)
    participant MM1 as MemoryManager1
    participant MM2 as MemoryManager2
    participant CUDA as CUDA_Runtime

    Note over P1: Original process allocates and loads model
    P1->>MM1: allocate_memory(GPU)
    MM1->>CUDA: cudaMalloc(model_size)
    P1->>MM1: load_model_data()
    MM1->>MM1: state = LOADED

    Note over P1: Get IPC handle for sharing
    P1->>MM1: get_cuda_ipc_handle()
    MM1->>CUDA: cudaIpcGetMemHandle(gpu_ptr)
    CUDA->>MM1: return ipc_handle
    MM1->>P1: return ipc_handle

    Note over P1,P2: Transfer IPC handle through some method
    P1->>P2: share ipc_handle + size + device_id

    Note over P2: Process 2 can use the handle for direct access
    P2->>P2: Use ipc_handle for CUDA operations
```

## Design Principles

### 1. Asynchronous First
- All I/O operations are asynchronous
- Use `std::future` and `std::shared_future`
- Avoid blocking the main thread

### 2. State-Driven
- Clear state transition rules
- Condition variable notifications on state changes
- Thread-safe state queries

### 3. Resource Management
- RAII principle ensures resource release
- Smart pointers manage object lifecycles
- Memory pools reduce allocation overhead

### 4. Error Handling
- Use `absl::Status` for unified error reporting
- Automatic cleanup of failed states
- Detailed logging

### 5. Extensibility
- Plugin-style Loader design
- Configuration-driven behavior
- Modular component structure

## Performance Optimization

### 1. Concurrency Strategy
- Multi-threaded parallel disk reading
- CPU-GPU transfer pipeline
- Overlapped execution of async operations

### 2. Memory Optimization
- Pre-allocated memory pools
- CUDA pinned memory improves transfer speed
- Zero-copy IPC sharing

### 3. Caching Strategy
- LRU policy for model eviction
- Intelligent location selection
- Locality-aware data access

## Security Considerations

### 1. Memory Safety
- Smart pointers prevent memory leaks
- Boundary checks avoid out-of-bounds access
- CUDA error checking

### 2. Thread Safety
- Fine-grained lock design
- Lock-free data structures
- Condition variable synchronization

### 3. Resource Isolation
- Memory isolation between processes
- Exclusive access to GPU devices
- Secure management of network resources

## Extension Points

The system is designed with multiple extension points to support future requirements:

1. **New Loader Types**: Implement `IModelLoader` interface
2. **New Memory Types**: Extend `MemoryManager` support
3. **New Transfer Protocols**: Extend communication engine
4. **New Verification Methods**: Extend verification framework

## Related Guides

- **Device Registry**: Learn how GPUs are mapped to logical `DeviceKey`s in the [Device Registry guide](./device-registry.md).
- **Communicator Internals**: See communication engine details in `../communicator/README.md`.
- **CheckpointStore API**: High-level usage patterns are documented in `../checkpoint/README.md`.