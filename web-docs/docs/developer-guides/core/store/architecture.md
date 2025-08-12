---
title: Architecture Design
description: Layered architectural design with clear responsibility boundaries
sidebar_position: 2
---

# Architecture Design

## System Architecture

The Core Store module adopts a layered architectural design with clear responsibility boundaries from external interfaces to low-level memory management. The architecture has evolved to support multi-device binding, distributed virtual memory pools (DVMP), and unified type systems for model sources and targets.

## Overall Architecture Diagram

```mermaid
graph TB
    subgraph "External Interface Layer"
        CS[CheckpointStore]
        PO[PrepareOrchestrator]
    end

    subgraph "Core Components"
        MR[ModelRegistry]
        DM[DeviceManager]
        GSC[GlobalStoreClient]
        MC[MetricsCollector]
    end

    subgraph "Model Layer"
        M[Model]
        MCF[ModelConfig]
        IK[InstanceKey]
    end

    subgraph "Memory Management"
        MM[MemoryManager]
        TS[TransferService]
        CES[ChunkExportService]
        MS[MemoryState]
        ML[ModelLocation]
    end

    subgraph "Data Loading Layer"
        IL[IModelLoader]
        DL[DiskLoader]
        PL[P2PLoader]
        SS[SeekableSource]
        P[Pump]
    end

    subgraph "Memory Pools"
        PMP[PinnedMemoryPool]
        DVMP[DistributedVirtualMemoryPool]
        SPB[StreamingPinnedBuffer]
    end

    subgraph "GPU Memory"
        CM[CudaMemory]
        IPC[IPC Handle]
    end

    subgraph "Communication Layer"
        CMN[CommunicationManager]
        CE[CommunicateEngine]
        CRI[CommRegistrationInfo]
    end

    CS --> PO
    CS --> MR
    CS --> DM
    CS --> GSC
    CS --> MC
    PO --> GSC
    PO --> MR

    MR --> M
    M --> IK
    M --> MM
    M --> IL
    M --> MCF

    MM --> TS
    MM --> CES
    MM --> MS
    MM --> ML
    MM --> PMP
    MM --> DVMP
    MM --> SPB
    MM --> CM

    IL --> DL
    IL --> PL
    DL --> SS
    PL --> SS
    SS --> P

    CES --> CMN
    CMN --> CE
    CE --> CRI

    CM --> IPC
```

## Layer Details

### 1. External Interface Layer

**CheckpointStore** is the entry point of the entire system, providing:

- Model registration and management via ModelRegistry
- GPU device management via DeviceManager
- Global resource coordination through GlobalStoreClient
- High-level API encapsulation with prepare() method
- Metrics collection through MetricsCollector

**PrepareOrchestrator** handles the prepare() API workflow:

- Remote replica selection from Global Store
- P2P transport setup and coordination
- Disk fallback when P2P unavailable
- Replica registration after successful loading

```cpp
class CheckpointStore {
public:
    // Multi-device binding API
    absl::StatusOr<ModelHandle> prepare(
        std::string_view model_id,
        const DeviceKey& target_device,
        PrepareMode mode = PrepareMode::AUTO,
        const LoadingHints& hints = {});

    // Instance-based management
    int wait_instance_ready(const InstanceKey& key);
    int unload_instance(const InstanceKey& key);

    // DVMP chunk locking for H2D/P2P transfers
    absl::Status lock_chunks(const InstanceKey& instance_key,
                             absl::Span<const uint32_t> chunk_indices);
    absl::Status unlock_chunks(const InstanceKey& instance_key,
                               absl::Span<const uint32_t> chunk_indices,
                               bool copied_gpu);
};
```

### 2. Model Management Layer

**Model** class encapsulates the complete lifecycle of a single model instance bound to a specific device:

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
- Factory pattern with `Model::create()` for instance creation
- Each Model instance is uniquely identified by `InstanceKey` (model_id + device + replica)
- Asynchronous operation management via `std::shared_future`
- Supports GPU↔GPU direct P2P transfers via `copy_from()` method
- Integrated model verification capabilities

### 3. Data Loading Layer

Adopts strategy pattern design with pump-based streaming architecture:

```mermaid
classDiagram
    class IModelLoader {
        <<interface>>
        +initialize() Status
        +get_model_size() StatusOr~uint64_t~
        +open_source() StatusOr~SeekableSource~
    }

    class DiskLoader {
        -source_: DiskSource
        -partition_paths_: vector~path~
        -partition_sizes_: vector~size_t~
        +open_source() StatusOr~SeekableSource~
    }

    class P2PLoader {
        -source_: P2PSource
        +open_source() StatusOr~SeekableSource~
    }

    IModelLoader <|-- DiskLoader
    IModelLoader <|-- P2PLoader
```

**DiskLoader Workflow**:
1. Scan partition files (`tensor.data_0`, `tensor.data_1`, ...) from DiskSource path
2. Create `FilePartitionSource` that implements `SeekableSource` interface
3. Return source handle for pump-based streaming
4. Actual loading handled by `MemoryManager::load_async_from_source()` using Pump
5. Data flows: FilePartitionSource → Pump → MemorySink (DvmpRegionSink or GpuMemorySink)

**P2PLoader Workflow**:
1. Validate P2PSource configuration (IP, port, memory keys)
2. Create `RemoteKeySource` that wraps remote memory access via CommunicateEngine
3. Can be muxed with DiskLoader as fallback via `MuxSeekableSource`
4. Support various transfer scenarios (CPU↔CPU, GPU↔GPU, CPU↔GPU)
5. Optional checksum verification during transfer

### 4. Memory Management Layer

**MemoryManager** manages memory for a single model instance at both CPU and GPU locations, integrating with Distributed Virtual Memory Pool (DVMP) for pageable CPU memory:

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
- CPU ↔ GPU: Asynchronous cudaMemcpyAsync via dedicated CUDA stream
- DISK → CPU/GPU: Pump-based streaming with `load_async_from_source()`
- REMOTE → CPU/GPU: Direct P2P transfer via CommunicateEngine
- GPU ↔ GPU: Direct peer-to-peer copy via `copy_from_peer()`

### 5. Memory Implementation Layer

The memory implementation layer provides the low-level memory management and data transfer mechanisms:

```mermaid
graph TB
    subgraph "CPU Memory Management"
        PMP[PinnedMemoryPool]
        DVMP[DistributedVirtualMemoryPool]
        SPB[StreamingPinnedBuffer]

        PMP -->|Allocates chunks| SPB
        DVMP -->|Virtual pages| UMA[UMA Space]
    end

    subgraph "GPU Memory Management"
        CM[CudaMemory]
        CS[CUDA Stream]
        IPC[IPC Handle]

        CM -->|Manages| CS
        CM -->|Exports| IPC
    end

    subgraph "Data Transfer Components"
        SS[SeekableSource]
        Sink[MemorySink]
        P[Pump]
        BP[BufferPool]

        SS -->|Reads from| P
        P -->|Writes to| Sink
        P -->|Uses| BP
    end

    subgraph "Service Layer"
        TS[TransferService]
        CES[ChunkExportService]
        MMC[ModelMemoryCoordinator]

        TS -->|Orchestrates| P
        CES -->|Manages| CRI[CommRegistrationInfo]
        MMC -->|Coordinates| TS
    end
```

**Memory Pool Design**:
- **PinnedMemoryPool**: Pre-allocated CUDA host pinned memory chunks with 4KB alignment for optimal performance
- **DistributedVirtualMemoryPool (DVMP)**: System-wide virtual address space for pageable CPU memory with lazy physical page binding
- **StreamingPinnedBuffer**: Circular buffer pool for streaming data transfers using producer-consumer pattern
- **BufferPool**: Reusable buffer management for pump-based transfers
- Automatic alignment (4KB for memory pages, 512B for DIRECT_IO) and NUMA optimization

**Pump-Based Transfer Architecture**:
- **SeekableSource**: Abstract interface for data sources (disk files, remote memory)
- **MemorySink**: Abstract interface for data destinations (DVMP regions, GPU memory)
- **Pump**: Core transfer engine that streams data from source to sink
- **DirectWriteToken**: Enables zero-copy transfers when supported by source and sink

**GPU Memory Features**:
- Direct allocation via `CudaMemory` class with device-specific management
- Automatic CUDA context and non-blocking stream management
- Cross-process memory sharing via IPC handles (`get_cuda_ipc_handle()`)
- Device-bound memory management (each MemoryManager tied to specific device via InstanceKey)
- Support for peer-to-peer transfers between different GPU devices

## 内存传输机制详解

### CPU and GPU Memory Layout Differences

CPU and GPU use different memory layout strategies to optimize their respective access patterns:

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

**Design Principles**:
- **CPU Chunked Design**: Enables parallel I/O, reduces memory fragmentation, simplifies pool management
- **GPU Contiguous Design**: Maximizes GPU bandwidth utilization, simplifies kernel access patterns
- **DVMP Integration**: CPU chunks can now be backed by virtual memory pages for better scalability

### Memory State Management

Each memory location maintains independent state management with thread-safe transitions:

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

### Disk to CPU Loading Mechanism

When loading from disk to CPU, data is processed in chunks using a pump-based streaming approach:

```mermaid
sequenceDiagram
    participant DL as DiskLoader
    participant MM as MemoryManager
    participant PM as PinnedMemory
    participant T1 as Thread1
    participant T2 as Thread2
    participant TN as ThreadN

    Note over DL,TN: Multi-threaded parallel disk reading

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

    Note over T1,TN: All threads complete
    DL->>MM: finalize_load_state(CPU, LOADED)
```

### CPU to GPU Transfer Mechanism

CPU to GPU transfer reassembles chunked CPU memory into contiguous GPU memory:

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

### P2P Transfer Support

P2P transfer adopts different strategies based on source and destination memory layouts:

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

**P2P Transfer Capabilities**:
- **CPU↔CPU**: Fully supported, 1:1 chunk-to-chunk mapping
- **GPU↔GPU**: Fully supported, single buffer to single buffer transfer
- **CPU→GPU**: Fully supported, multiple chunks assembled to single buffer
- **GPU→CPU**: Fully supported, single buffer split to multiple chunks
- **Cross-device GPU↔GPU**: Supported via `copy_from_peer()` method

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

### New Unified Loading Flow with prepare() API

```mermaid
sequenceDiagram
    participant User
    participant CS as CheckpointStore
    participant PO as PrepareOrchestrator
    participant MR as ModelRegistry
    participant M as Model
    participant MM as MemoryManager
    participant L as Loader
    participant DVMP

    User->>CS: prepare(model_id, target_device)
    CS->>PO: orchestrate loading
    PO->>MR: get_or_create_model(instance_key)

    alt Model not exists
        MR->>M: Model::create(config)
        M->>L: create appropriate loader
        M->>MM: initialize with DVMP
        MM->>DVMP: reserve virtual address space
    end

    PO->>M: ensure_loaded_async(target_location)
    M->>MM: allocate_memory(location)
    M->>L: open_source()
    L-->>M: return SeekableSource
    M->>MM: load_async_from_source(source)

    MM->>MM: setup streaming buffers
    MM->>MM: pump data from source
    MM->>MM: finalize_load_state(LOADED)

    M-->>PO: return future
    PO->>CS: return ModelHandle
    CS->>User: ModelHandle{instance_key, ready_future}
```

### P2P Loading Flow with CommunicateEngine

P2P transfers leverage the CommunicateEngine for efficient remote memory access:

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
- Use `std::future` and `std::shared_future` for operation tracking
- Non-blocking CUDA streams for GPU operations
- Pump-based streaming for data transfers

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
- Plugin-style Loader design via `IModelLoader` interface
- Unified type system (`ModelSource`, `ModelTarget`, `DeviceKey`)
- Configuration-driven behavior via `LoadingHints`
- Modular component structure with clear service boundaries

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
- DVMP-based chunk-level eviction policies
- Intelligent location selection based on device capabilities
- Locality-aware data access with NUMA optimization
- Chunk locking mechanism to prevent eviction during transfers

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

1. **New Loader Types**: Implement `IModelLoader` interface and provide `SeekableSource`
2. **New Source Types**: Add variants to `ModelSource` (e.g., S3Source, AzureBlobSource)
3. **New Memory Types**: Extend `MemoryManager` and `ModelLocation` enum
4. **New Transfer Protocols**: Extend `CommunicateEngine` implementations
5. **New Verification Methods**: Extend `ModelVerificationInfo` framework
6. **Custom Device Types**: Extend `DeviceKey` and device registry

## Key Implementation Details

### Multi-Device Binding
- Each model instance is uniquely identified by `InstanceKey` (model_id + device + replica)
- Supports multiple replicas of the same model on different devices
- Device abstraction via `DeviceKey` for stable device references

### Distributed Virtual Memory Pool (DVMP)
- System-wide virtual address space management
- Chunk-based memory allocation with lazy physical page binding
- Supports chunk locking during H2D/P2P transfers
- Enables efficient memory sharing across processes

### Unified Type System
- `ModelSource`: Describes where data comes from (Disk, P2P, Buffer)
- `ModelTarget`: Describes where data goes (Location with device info)
- `LoadingHints`: Tuning parameters for loading operations
- `ModelHandle`: Returned from loading operations with instance info

### Service Architecture
- **TransferService**: Manages data transfers between locations
- **ChunkExportService**: Handles P2P memory registration/export
- **PrepareOrchestrator**: Coordinates the prepare() API workflow
- **MetricsCollector**: Tracks performance and resource usage

## Related Guides

- **Device Registry**: Learn how GPUs are mapped to logical `DeviceKey`s in the [Device Registry guide](./device-registry.md).
- **Communicator Internals**: See communication engine details in `../communicator/README.md`.
- **CheckpointStore API**: High-level usage patterns are documented in `../checkpoint/README.md`.
- **DeviceManager** — runtime GPU enumeration and streams ([Device Manager](./device-manager.md))