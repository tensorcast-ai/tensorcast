---
title: State Management
description: Core system reliability through precise memory lifecycle and state transitions
sidebar_position: 5
---

# State Management

## Overview

State management in the Core Store module is the core of system reliability, ensuring correct memory lifecycle management through precise state transitions. The system involves two main state dimensions: **Memory State (MemoryState)** and **Model Location (ModelLocation)**.

## Memory State (MemoryState)

### State Definition

```cpp
enum class MemoryState : uint8_t {
    UNINITIALIZED = 0,  // Initial state, no operations performed
    UNALLOCATED = 1,    // Initialized but memory not allocated
    ALLOCATED = 2,      // Memory allocated but no data
    LOADING = 3,        // Loading data in progress
    LOADED = 4,         // Data loading completed
    FAILED = 5          // Operation failed
};
```

### State Transition Diagram

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED: Object creation

    UNINITIALIZED --> UNALLOCATED: Initialize memory pool
    UNINITIALIZED --> FAILED: Initialization failed

    UNALLOCATED --> ALLOCATED: allocate_memory()
    UNALLOCATED --> FAILED: Allocation failed

    ALLOCATED --> LOADING: Start loading data
    ALLOCATED --> UNALLOCATED: release_memory()

    LOADING --> LOADED: Loading successful
    LOADING --> FAILED: Loading failed
    LOADING --> UNALLOCATED: Force release

    LOADED --> UNALLOCATED: release_memory()
    LOADED --> LOADING: Data transfer

    FAILED --> UNALLOCATED: Cleanup resources

    note right of LOADING: Can wait for completion via wait_for_state()
    note right of LOADED: Data can be safely accessed
    note right of FAILED: Needs cleanup before restart
```

### State Semantics

| State | Memory Allocated | Data Valid | Executable Operations |
|------|-------------|-------------|-----------|
| UNINITIALIZED | ❌ | ❌ | Initialize |
| UNALLOCATED | ❌ | ❌ | Allocate memory |
| ALLOCATED | ✅ | ❌ | Start loading, release memory |
| LOADING | ✅ | ⚠️ | Wait for completion, force release |
| LOADED | ✅ | ✅ | Access data, transfer data, release memory |
| FAILED | ⚠️ | ❌ | Cleanup resources |

## Model Location (ModelLocation)

### Location Definition

```cpp
enum class ModelLocation : uint8_t {
    NONE = 0,    // Invalid location
    DISK,        // Disk storage
    CPU,         // CPU memory
    GPU,         // GPU memory
    REMOTE       // Remote memory (RDMA)
};
```

### Location Relationship Diagram

```mermaid
graph TB
    subgraph "Data Sources"
        DISK[DISK<br/>Disk Storage]
        REMOTE[REMOTE<br/>RDMA Source]
    end

    subgraph "Local Memory"
        CPU[CPU<br/>Host Memory]
        GPU[GPU<br/>Device Memory]
    end

    DISK -->|DiskLoader| CPU
    REMOTE -->|P2PLoader| CPU
    REMOTE -->|P2PLoader| GPU
    CPU <-->|copy_data_async| GPU

    style DISK fill:#e1f5fe
    style REMOTE fill:#f3e5f5
    style CPU fill:#e8f5e8
    style GPU fill:#fff3e0
```

## State Manager

### MemoryManager State Management

Each MemoryManager instance independently manages states at both CPU and GPU locations:

```mermaid
graph TB
    subgraph "MemoryManager Instance"
        MM[MemoryManager]

        subgraph "CPU State Management"
            CS[cpu_state_]
            CC[cpu_cond_]
            PM[pinned_mem_]
            HQ[host_chunk_queue_]
        end

        subgraph "GPU State Management"
            GS[gpu_state_]
            GC[gpu_cond_]
            CM[cuda_mem_]
        end

        subgraph "Streaming"
            SB[streaming_buffer_]
        end

        MM --> CS
        MM --> GS
        CS --> CC
        GS --> GC
        CS --> PM
        CS --> HQ
        GS --> CM
        MM --> SB
    end
```

### State Transition Operations

#### 1. Memory Allocation

```mermaid
sequenceDiagram
    participant Client
    participant MM as MemoryManager
    participant Pool as MemoryPool

    Client->>MM: allocate_memory(CPU)
    MM->>MM: check current state
    alt state == UNALLOCATED
        MM->>Pool: allocate chunks
        Pool-->>MM: return buffers
        MM->>MM: set_state(ALLOCATED)
        MM-->>Client: success
    else state != UNALLOCATED
        MM-->>Client: error
    end
```

#### 2. Data Loading

```mermaid
sequenceDiagram
    participant Client
    participant MM as MemoryManager
    participant Loader

    Client->>MM: Trigger loading
    MM->>MM: set_state(LOADING)
    MM->>Loader: load_async()

    par Loading Process
        Loader->>Loader: Read data
        Loader->>MM: Write to buffer
    and State Waiting
        Client->>MM: wait_for_state(LOADED)
        MM->>MM: Wait for condition variable
    end

    Loader->>MM: finalize_load_state(OK)
    MM->>MM: set_state(LOADED)
    MM->>MM: Notify waiters
    MM-->>Client: Return success
```

#### 3. Memory Release

```mermaid
sequenceDiagram
    participant Client
    participant MM as MemoryManager
    participant Pool as MemoryPool

    Client->>MM: release_memory(safe=true)
    MM->>MM: check current state

    alt state == LOADING && safe_release
        MM-->>Client: error (unsafe)
    else state == LOADING && !safe_release
        MM->>MM: wait briefly
        MM->>MM: force transition to FAILED
        MM->>Pool: return buffers
        MM->>MM: set_state(UNALLOCATED)
        MM-->>Client: success with warning
    else state >= ALLOCATED
        MM->>Pool: return buffers
        MM->>MM: set_state(UNALLOCATED)
        MM-->>Client: success
    end
```

## Concurrency Control

### Lock Strategy

The system adopts fine-grained lock design to avoid deadlocks:

```mermaid
graph TB
    subgraph "MemoryManager 锁层次"
        MM[MemoryManager::mutex_]

        subgraph "Condition Variables"
            CC[cpu_cond_]
            GC[gpu_cond_]
        end

        subgraph "External Resource Locks"
            PL[Pool locks]
            CL[CUDA locks]
        end
    end

    MM --> CC
    MM --> GC
    MM -.-> PL
    MM -.-> CL

    note1[["锁顺序: MM::mutex_ -> 外部锁<br/>避免持有 MM 锁时调用可能阻塞的外部 API"]]
```

**Key Additions After Refactor**

| Member / Flag | Purpose |
|---------------|---------|
| `host_chunk_queue_` | Tracks per-chunk load completion on the CPU side, enabling streaming loaders to coordinate consumer progress. |
| `streaming_buffer_` | Optional `StreamingPinnedBuffer` pool used by high-throughput producer/consumer pipelines. Allocated via `allocate_buffer_pool()` and released with `release_buffer_pool()`. |
| `pinned_memory_timeout_` | Maximum duration to wait for pinned memory allocation from the pool before aborting with `ResourceExhausted`. |

> These additions do **not** change the state machine itself, but introduce auxiliary resources and configuration options that improve throughput and memory efficiency.

### Mandatory CPU Memory Release (RFC 0001)

As of RFC 0001 §4.3, CPU memory release after GPU copy is **mandatory**. When a model is successfully copied from CPU to GPU:

1. **DVMP Integration**: The system automatically marks chunks as `COPIED_GPU` via `unlock_chunks()`
2. **Memory Eviction**: Physical pages are reclaimed through `evict_tail_bytes()`  
3. **State Transition**: CPU memory state transitions to `UNALLOCATED` (virtual address space retained)

This ensures optimal memory utilization and prevents RSS bloat in multi-model scenarios.
