---
title: Architecture Design
description: Layered architectural design with clear responsibility boundaries
sidebar_position: 2
---

# Architecture Design

## System Architecture

The Core Store module adopts a layered architectural design with clear responsibility boundaries from external interfaces to low-level memory management. The architecture has evolved to support multi-device binding, distributed virtual memory pools (DVMP), and unified type systems for replica sources and targets.

> Notes for readers:
> - In code, the CPU location is named `MemoryLocation::PAGEABLE_CPU` (file: `core/common/memory/memory_location.h`). In this document, we refer to it as "CPU" for readability but always provide the code name on first mention.

## Overall Architecture Diagram

```mermaid
graph TB
    subgraph "External Interface Layer"
        CS[StoreEngine]
        PO[MaterializeOrchestrator]
    end

    subgraph "Core Components"
        MR[ReplicaRegistry]
        DM[DeviceManager]
        GSC[GlobalStoreClient]
        MC[MetricsCollector]
    end

    subgraph "Replica Layer"
        M[Replica]
        MCF[ReplicaConfig]
        IK[ReplicaKey]
    end

    subgraph "Memory Management"
        MM[MemoryManager]
        TS[TransferService]
        CES[ChunkExportService]
        MS[MemoryState]
        ML[MemoryLocation]
    end

    subgraph "Data Loading Layer"
        IL[IArtifactLoader]
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

**StoreEngine** is the entry point of the entire system, providing:

- Replica registration and management via ReplicaRegistry
- GPU device management via DeviceManager
- Global resource coordination through GlobalStoreClient
- High-level API encapsulation with materialize_replica() method
- Metrics collection through MetricsCollector

- Source: `core/store/store_engine.h`, `core/store/store_engine.cc`

**MaterializeOrchestrator** handles the materialize_replica() API workflow:

- Remote replica selection from Global Store
- P2P transport setup and coordination
- Disk fallback when P2P unavailable
- Replica registration after successful loading

- Source: `core/store/loading/materialize_orchestrator.{h,cc}`, `core/store/components/global_store_client.{h,cc}`

```cpp
class StoreEngine {
public:
    // Multi-device binding API
    absl::StatusOr<ReplicaHandle> materialize_replica(
        std::string_view artifact_id,
        const DeviceKey& target_device,
        MaterializeMode mode = MaterializeMode::AUTO,
        const MaterializeHints& hints = {});

    // Instance-based management
    int wait_replica_ready(const ReplicaKey& key);
    int unload_replica(const ReplicaKey& key);

    // DVMP chunk locking for H2D/P2P transfers
    absl::Status lock_chunks(const ReplicaKey& replica_key,
                             absl::Span<const uint32_t> chunk_indices);
    absl::Status unlock_chunks(const ReplicaKey& replica_key,
                               absl::Span<const uint32_t> chunk_indices,
                               bool copied_gpu);
};
```

- Definitions: `ReplicaKey`, `ReplicaHandle`, `MaterializeHints` in `core/store/loading/loading_spec.h`
- Device key: `DeviceKey` in `core/store/device_types.h`

### 2. Replica Management Layer

**Replica** class encapsulates the complete lifecycle of a single replica instance bound to a specific device:

- Source: `core/store/replica/replica.{h,cc}`

```mermaid
graph LR
    subgraph "Replica Internal Architecture"
        M[Replica] --> MM[MemoryManager]
        M --> IL[IArtifactLoader]
        M --> CF[CPU Future]
        M --> GF[GPU Future]

        MM --> PM[PinnedMemory]
        MM --> CM[CudaMemory]

        IL --> DL[DiskLoader]
        IL --> RL[P2PLoader]
    end
```

**Design Features**:
- Factory pattern with `Replica::create()` for instance creation
- Each Replica instance is uniquely identified by `ReplicaKey` (artifact_id + device + replica) — `core/store/loading/loading_spec.h`
- Asynchronous operation management via `std::shared_future` — `Replica::ensure_loaded_async()` in `core/store/replica/replica.{h,cc}`
- Supports device copies via `Replica::copy_from()` and `MemoryManager::copy_from_peer()` — `core/store/replica/replica.h`, `core/store/replica/memory_manager.h`
- Integrated replica verification — `core/common/artifact_verification.{h,cc}`, used by loaders and `Replica`

### 3. Data Loading Layer

Adopts strategy pattern design with pump-based streaming architecture:

- Sources: `core/store/loader/loader.h`, `core/store/loader/disk_loader.{h,cc}`, `core/store/loader/p2p_loader.{h,cc}`
- Streaming: `core/store/loader/source.h`, `core/store/loader/pump.{h,cc}`, `core/store/loader/buffer_pool.h`
- Remote: `core/store/loader/remote_key_source.{h,cc}`, `core/store/loader/mux_seekable_source.{h,cc}`

```mermaid
classDiagram
    class IArtifactLoader {
        <<interface>>
        +initialize() Status
        +get_artifact_size() StatusOr~uint64_t~
        +open_source() StatusOr~SeekableSource~
    }

    class DiskLoader {
        -source_: DiskSource
        +open_source() StatusOr~SeekableSource~
    }

    class P2PLoader {
        -source_: P2PSource
        +open_source() StatusOr~SeekableSource~
    }

    IArtifactLoader <|-- DiskLoader
    IArtifactLoader <|-- P2PLoader
```

**DiskLoader Workflow**:
1. Scan partition files (`tensor.data`, `tensor.data_<n>`) — `core/store/loader/disk_loader.cc`
2. Create `FilePartitionSource` implementing `SeekableSource` — `core/store/loader/file_partition_source.{h,cc}`
3. Return source handle for pump-based streaming — `DiskLoader::open_source()`
4. Actual loading handled by `MemoryManager::load_async_from_source()` using `TransferService` + `pump_ranges()`
5. Data flows: FilePartitionSource → Pump → MemorySink (`DVMPRegionSink` for CPU or `GPUMemorySink` for GPU)

**P2PLoader Workflow**:
1. Validate `P2PSource` configuration (IP, port, memory keys) — `core/store/loader/p2p_loader.{h,cc}`
2. Create `RemoteKeySource` that wraps remote memory via `CommunicateEngine` — `core/store/loader/remote_key_source.{h,cc}`
3. Optional disk fallback via `MuxSeekableSource` — `core/store/loader/mux_seekable_source.{h,cc}`
4. Uses the same `load_async_from_source()` path to target CPU (PAGEABLE_CPU) or GPU
5. Optional checksum or direct-write support depends on communicator — see `RemoteKeySource::supports_direct_write()`

### 4. Memory Management Layer

**MemoryManager** manages memory for a single replica instance at both CPU (PAGEABLE_CPU) and GPU locations, integrating with DVMP for pageable CPU memory:

- Source: `core/store/replica/memory_manager.{h,cc}`
- UMA (Unified Memory): `core/store/replica/replica_memory_coordinator.{h,cc}`
- Transfers: `core/store/replica/transfer_service.{h,cc}`, `core/store/replica/transfer_helpers.{h,cc}`
- States: `core/store/replica/memory_state.h`, Locations: `core/common/memory/memory_location.h`

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
- CPU (PAGEABLE_CPU) ↔ GPU: Asynchronous copy via dedicated CUDA stream — `MemoryManager::copy_data_async()`
- DISK → CPU/GPU: Pump-based streaming via `load_async_from_source()` and `TransferService::load_from_source()`
- REMOTE → CPU/GPU: Same pump-based streaming using `RemoteKeySource` and `DVMPRegionSink`/`GPUMemorySink`
- GPU ↔ GPU: Direct peer copy via `MemoryManager::copy_from_peer()`

### 5. Memory Implementation Layer

The memory implementation layer provides the low-level memory management and data transfer mechanisms:

- CPU Memory: `core/common/memory/pinned_memory_pool.{h,cc}`, `core/common/memory/streaming_pinned_buffer.{h,cc}`
- DVMP: `core/common/memory/distributed_virtual_memory_pool.{h,cc}`
- GPU Memory: `core/common/memory/cuda_memory.{h,cc}`
- Sinks/Sources: `core/store/loader/dvmp_region_sink.{h,cc}`, `core/store/loader/gpu_memory_sink.{h,cc}`
- Pump: `core/store/loader/pump.{h,cc}` (`pump_ranges`), `core/store/loader/buffer_pool.h`

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
        MMC[ReplicaMemoryCoordinator]

        TS -->|Orchestrates| P
        CES -->|Manages| CRI[CommRegistrationInfo]
        MMC -->|Coordinates| TS
    end
```

**GPU Memory Features**:
- CUDA allocation and stream management — `core/common/memory/cuda_memory.{h,cc}`
- Cross-process memory sharing via `MemoryManager::get_cuda_ipc_handle()` — `core/store/replica/memory_manager.h`
- Device-bound memory management (via `ReplicaKey`) — `core/store/loading/loading_spec.h`

## Memory Transfer Mechanism

### Disk to CPU Loading Mechanism

Updated to reflect `TransferService` + `pump_ranges` orchestration:

```mermaid
sequenceDiagram
    participant DL as DiskLoader
    participant MM as MemoryManager
    participant TS as TransferService
    participant SRC as FilePartitionSource
    participant SNK as DVMPRegionSink

    DL->>DL: initialize()
    DL->>DL: get_artifact_size()
    DL->>DL: open_source()
    DL-->>MM: return SeekableSource (SRC)

    MM->>MM: allocate_memory(PAGEABLE_CPU)
    MM->>MM: set_state(PAGEABLE_CPU, LOADING)
    MM->>TS: load_from_source(SRC, PAGEABLE_CPU, concurrency)
    TS->>SNK: build DVMPRegionSink via DVMP::open()
    TS->>TS: build_ranges(chunk_indices or full)
    TS->>TS: pump_ranges(SRC, SNK, buffer_pool, ranges)
    TS-->>MM: return OkStatus

    MM->>MM: finalize_load(PAGEABLE_CPU)
    MM->>MM: set_state(PAGEABLE_CPU, LOADED)
```

- Source: `MemoryManager::load_async_from_source()` and `TransferService::load_from_source()`

### CPU to GPU Transfer Mechanism

```mermaid
graph TB
    subgraph "CPU (PAGEABLE_CPU)"
        VA[DVMP Base Ptr]
    end

    subgraph "CUDA Operations"
        Stream[CUDA Stream]
        Copy["cudaMemcpyAsync (H2D) Streaming"]
        Sync[cudaStreamSynchronize]
    end

    subgraph "GPU Buffer"
        GPU[Contiguous GPU Buffer]
    end

    VA --> Copy
    Copy --> GPU
    Copy --> Stream
    Stream --> Sync
```

- Source: `TransferService::copy_cpu_to_gpu_streaming()`

### GPU to CPU Transfer Mechanism

```mermaid
graph TB
    subgraph "Reverse Transfer Flow"
        direction TB

        subgraph "GPU Buffer"
            GPU2[Contiguous GPU Buffer]
        end

        subgraph "CUDA Operations"
            Stream2[CUDA Stream]
            RCopy[cudaMemcpyAsync (D2H) Streaming]
            Sync2[cudaStreamSynchronize]
        end

        subgraph "CPU (PAGEABLE_CPU)"
            DVMPBase[DVMP Base Ptr]
        end

        GPU2 --> RCopy
        RCopy --> DVMPBase
        RCopy --> Stream2
        Stream2 --> Sync2
    end
```

- Source: `TransferService::copy_gpu_to_cpu_streaming()`

### P2P Transfer Support

P2P transfer strategies based on memory layouts, via `RemoteKeySource` + `pump_ranges` and appropriate sinks:

```mermaid
graph TB
    subgraph "P2P Transfer Scenarios"

        subgraph "CPU to CPU (Supported)"
            direction LR
            RemoteCPU[Remote CPU<br/>Chunks]
            LocalCPU[Local CPU<br/>DVMP]
            RemoteCPU -->|RemoteKeySource.read_at| LocalCPU
        end

        subgraph "GPU to GPU (Supported)"
            direction LR
            RemoteGPU[Remote GPU<br/>Buffer]
            LocalGPU[Local GPU<br/>Buffer]
            RemoteGPU -->|RemoteKeySource.read_at| LocalGPU
        end

        subgraph "CPU to GPU (Supported)"
            direction LR
            RemoteCPU2[Remote CPU<br/>Chunks]
            LocalGPU2[Local GPU<br/>Buffer]
            RemoteCPU2 -->|read_at → GPUMemorySink.write_at| LocalGPU2
        end

        subgraph "GPU to CPU (Supported)"
            direction LR
            RemoteGPU2[Remote GPU<br/>Buffer]
            LocalCPU2[Local CPU<br/>DVMP]
            RemoteGPU2 -->|read_at → DVMPRegionSink.write_at| LocalCPU2
        end
    end
```

- Sources: `core/store/loader/remote_key_source.{h,cc}`, `core/store/loader/gpu_memory_sink.{h,cc}`, `core/store/loader/dvmp_region_sink.{h,cc}`, `core/store/loader/pump.{h,cc}`

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

- Sources: `core/store/replica/memory_manager.{h,cc}` (`set_state`, `finalize_load`, error paths)

## Core Interaction Flows

### New Unified Loading Flow with materialize_replica() API

```mermaid
sequenceDiagram
    participant User
    participant CS as StoreEngine
    participant PO as MaterializeOrchestrator
    participant MR as ReplicaRegistry
    participant M as Replica
    participant MM as MemoryManager
    participant L as Loader
    participant DVMP

    User->>CS: materialize_replica(artifact_id, target_device)
    CS->>PO: orchestrate loading
    PO->>MR: get_or_create_replica(replica_key)

    alt Replica not exists
        MR->>M: Replica::create(config)
        M->>L: create appropriate loader
        M->>MM: initialize with DVMP
        MM->>DVMP: reserve virtual address space
    end

    PO->>M: ensure_loaded_async(target_location)
    M->>MM: allocate_memory(location)
    M->>L: open_source()
    L-->>M: return SeekableSource
    M->>MM: load_async_from_source(source)

    MM->>MM: setup streaming buffers (via TransferService)
    MM->>MM: pump data from source
    MM->>MM: finalize_load_state(LOADED)

    M-->>PO: return future
    PO->>CS: return ReplicaHandle
    CS->>User: ReplicaHandle{replica_key, ready_future}
```

- Sources: `core/store/store_engine.{h,cc}`, `core/store/loading/materialize_orchestrator.{h,cc}`, `core/store/replica/replica.{h,cc}`, `core/store/replica/memory_manager.{h,cc}`

### P2P Loading Flow with CommunicateEngine

P2P transfers leverage the `CommunicateEngine` for remote memory access with `RemoteKeySource`:

```mermaid
sequenceDiagram
    participant User
    participant CS as StoreEngine
    participant M as Replica
    participant MM as MemoryManager
    participant RL as P2PLoader
    participant CM as CommunicationManager

    User->>CS: load_model(p2p_config)
    CS->>M: create(P2PModelSource)

    Note over M: Select target (CPU/GPU) and open_source()
    M->>RL: open_source()
    RL-->>M: SeekableSource(RemoteKeySource)
    M->>MM: load_async_from_source(source, target)

    alt Target = CPU
        MM->>MM: allocate_memory(PAGEABLE_CPU)
    else Target = GPU
        MM->>MM: allocate_memory(GPU)
    end

    MM->>MM: set_state(target, LOADING)
    MM->>MM: pump via TransferService
    MM->>MM: finalize and set_state(target, LOADED)

    M->>CS: return success/error
```

- Sources: `core/store/loader/p2p_loader.{h,cc}`, `core/store/loader/remote_key_source.{h,cc}`, `core/store/replica/memory_manager.{h,cc}`

### IPC Memory Sharing Flow

GPU memory can be shared between processes through IPC handles:

```mermaid
sequenceDiagram
    participant P1 as Process1(Owner)
    participant P2 as Process2(User)
    participant MM1 as MemoryManager1
    participant MM2 as MemoryManager2
    participant CUDA as CUDA_Runtime

    P1->>MM1: allocate_memory(GPU)
    MM1->>CUDA: cudaMalloc(artifact_size)
    P1->>MM1: load_model_data()
    MM1->>MM1: state = LOADED

    P1->>MM1: get_cuda_ipc_handle()
    MM1->>CUDA: cudaIpcGetMemHandle(gpu_ptr)
    CUDA->>MM1: return ipc_handle
    MM1->>P1: return ipc_handle

    P1->>P2: share ipc_handle + size + device_id
    P2->>P2: Use ipc_handle for CUDA operations
```

- Source: `core/store/replica/memory_manager.h` (`get_cuda_ipc_handle()`)

## Performance Optimization

### 1. Concurrency Strategy
- Multi-threaded parallel disk reading (via `pump_ranges` concurrency)
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

1. **New Loader Types**: Implement `IArtifactLoader` interface and provide `SeekableSource`
2. **New Source Types**: Add variants to `ArtifactSource` (e.g., S3Source, AzureBlobSource)
3. **New Memory Types**: Extend `MemoryManager` and `MemoryLocation` enum
4. **New Transfer Protocols**: Extend `CommunicateEngine` implementations
5. **New Verification Methods**: Extend `ArtifactVerificationInfo` framework
6. **Custom Device Types**: Extend `DeviceKey` and device registry

## Key Implementation Details

### Multi-Device Binding
- Each replica instance is uniquely identified by `ReplicaKey` (artifact_id + device + replica) — `core/store/loading/loading_spec.h`
- Supports multiple replicas of the same replica on different devices
- Device abstraction via `DeviceKey` for stable device references — `core/store/device_types.h`

### Distributed Virtual Memory Pool (DVMP)
- System-wide virtual address space management — `core/common/memory/distributed_virtual_memory_pool.{h,cc}`
- Chunk-based memory allocation with lazy physical page binding
- Supports chunk locking during H2D/P2P transfers
- Enables efficient memory sharing across processes

- ### Unified Type System
- `ArtifactSource` / `ArtifactTarget` / `MaterializeHints` — `core/store/loading/loading_spec.h`
- `ReplicaHandle`: returned from loading operations with instance info — `core/store/loading/loading_spec.h`

### Service Architecture
- **TransferService**: Manages data transfers between locations — `core/store/replica/transfer_service.{h,cc}`
- **ChunkExportService**: Handles P2P memory registration/export — `core/store/replica/chunk_export_service.h`
- **MaterializeOrchestrator**: Coordinates the materialize_replica() API workflow — `core/store/loading/materialize_orchestrator.{h,cc}`
- **MetricsCollector**: Tracks performance and resource usage — `core/store/components/metrics_collector.{h,cc}`

## Related Guides

- **Device Registry**: Learn how GPUs are mapped to logical `DeviceKey`s in the [Device Registry guide](./device-registry.md).
- **Communicator Internals**: See communication engine details in `../communicator/README.md`.
- **StoreEngine API**: High-level usage patterns are documented in `../checkpoint/README.md`.
- **DeviceManager** — runtime GPU enumeration and streams ([Device Manager](./device-manager.md))