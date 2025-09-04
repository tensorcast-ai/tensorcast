---
title: Architecture
description: Layered architecture design for high-level tensor operations and I/O buffering
sidebar_position: 1
---

# Architecture

## Overview

The Checkpoint module follows a layered architecture design that separates concerns between high-level tensor operations, I/O buffering, and low-level file management.

## Module Relationships

```mermaid
graph TD
    A[Client Code] --> B[checkpoint.h API]
    B --> C[TensorWriter]
    B --> D[CUDA Memory Management]
    B --> E[Replica Verification]
    C --> F[AlignedBuffer]
    F --> G[File System]
    D --> H[CUDA Runtime]
    E --> I[Replica Verification System]

    B --> J[Progress Reporting]
    J --> K[progress_bar.h]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#fce4ec
```

## Component Architecture

### 1. High-Level API Layer (`checkpoint.h/cpp`)

The main API layer provides:
- **Tensor Save/Restore Operations**
- **CUDA Memory Management**
- **Replica Verification Integration**
- **Cross-platform compatibility**

```mermaid
classDiagram
    class CheckpointAPI {
        +save_tensors(names, data, path) unordered_map~string, uint64_t~
        +restore_tensors(...) unordered_map~string, torch::Tensor~
        +restore_tensors_from_disk(...) unordered_map~string, torch::Tensor~
        +allocate_cuda_memory(device_id, size) uint64_t
        +get_cuda_memory_handle(device_id, ptr) string
        +generate_verification_info_from_disk(path) ModelVerificationInfo
    }

    CheckpointAPI --> TensorWriter
    CheckpointAPI --> CUDAMemory
    CheckpointAPI --> ModelVerification
```

### 2. Tensor Writing Layer (`tensor_writer.h/cpp`)

Manages partitioned tensor writing:
- **Automatic Partitioning**: Splits data into 10GB chunks
- **Offset Management**: Tracks global offsets across partitions
- **Data Alignment**: Ensures 64-bit alignment for all records

```mermaid
classDiagram
    class TensorWriter {
        -size_t offset_
        -int partition_idx_
        -size_t partition_size_
        -string filename_
        -unique_ptr~AlignedBuffer~ buffer_
        +write_record(data, size) uint64_t
    }

    TensorWriter --> AlignedBuffer
```

### 3. I/O Buffer Layer (`aligned_buffer.h/cpp`)

Provides optimized file I/O:
- **4K Alignment**: Uses aligned_alloc for optimal performance
- **Buffer Management**: 1GB internal buffer with overflow handling
- **Direct Writes**: Uses pwrite for positioned file I/O

```mermaid
classDiagram
    class AlignedBuffer {
        -int fd_
        -size_t buf_size_
        -size_t buf_pos_
        -size_t file_offset_
        -void* buffer_
        +write_data(data, size) size_t
        +write_padding(padding_size) size_t
    }
```

## Data Flow Architecture

### Save Operation Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as checkpoint.h
    participant Writer as TensorWriter
    participant Buffer as AlignedBuffer
    participant FS as File System

    Client->>API: save_tensors(names, data, path)
    API->>Writer: create TensorWriter

    loop for each tensor
        API->>Writer: write_record(tensor_data, size)
        Writer->>Writer: check partition size
        alt need new partition
            Writer->>Buffer: create new AlignedBuffer
        end
        Writer->>Buffer: write_data(data, size)
        Buffer->>Buffer: check buffer space
        alt buffer full or direct write
            Buffer->>FS: pwrite(data)
        end
        Writer->>Writer: add 64-bit alignment padding
        Writer->>API: return offset
    end

    API->>Client: return tensor_offsets
```

### Restore Operation Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as checkpoint.h
    participant FS as File System
    participant CUDA as CUDA Runtime

    Client->>API: restore_tensors_from_disk(...)

    loop for each partition file
        API->>FS: read tensor.data_N
        FS->>API: partition data
    end

    loop for each tensor
        API->>API: calculate tensor offset and size
        API->>API: create torch::Tensor from data
        alt CUDA tensor
            API->>CUDA: copy to GPU memory
        end
    end

    API->>Client: return tensor map
```

## Memory Management Architecture

### CUDA Memory States

```mermaid
stateDiagram-v2
    [*] --> Allocated: allocate_cuda_memory()
    Allocated --> IPCShared: get_cuda_memory_handle()
    IPCShared --> RemoteAccess: get_cuda_memory_ptr()
    RemoteAccess --> Closed: close_cuda_memory_handle()
    Allocated --> Freed: cudaFree()
    Closed --> [*]
    Freed --> [*]
```

### Buffer States

```mermaid
stateDiagram-v2
    [*] --> Empty: AlignedBuffer()
    Empty --> Filling: write_data()
    Filling --> Filling: write_data() (space available)
    Filling --> Flushing: write_data() (buffer full)
    Flushing --> Empty: pwrite() completes
    Filling --> Padded: write_padding()
    Padded --> Flushing: buffer full
    Padded --> [*]: ~AlignedBuffer()
    Empty --> [*]: ~AlignedBuffer()
```

## Performance Considerations

### I/O Optimization Strategy

1. **4K Alignment**: All writes are aligned to filesystem block boundaries
2. **Large Buffers**: 1GB buffers reduce system call overhead
3. **Partitioning**: 10GB partitions balance memory usage and file management
4. **Direct Writes**: Uses pwrite system calls for efficient positioned writes

### Memory Layout

```mermaid
graph LR
    A[Tensor Data] --> B[64-bit Aligned]
    B --> C[1GB Buffer]
    C --> D[4K Aligned Write]
    D --> E[Disk Write]

    F[Partition 0<br/>10GB max] --> G[Partition 1<br/>10GB max]
    G --> H[Partition N<br/>remaining]
```

## Error Handling Strategy

The module uses a defensive programming approach:
- **Resource Management**: RAII pattern for automatic cleanup
- **Error Propagation**: Clear error messages with context
- **Validation**: Input validation at API boundaries
- **Recovery**: Graceful handling of partial failures

## Thread Safety

The current implementation is **not thread-safe**. For concurrent access:
- Use separate instances per thread
- External synchronization required for shared resources
- CUDA operations require proper device context management

## Streaming GPU Tensor Writing Layer (`streaming_tensor_writer.h/cpp`)

The streaming writer extends the core design with a **producer–consumer pipeline** that performs asynchronous GPU→Host copies and background disk flushing.

Key components:

1. **StreamingTensorWriter** – Orchestrates the streaming flow, maintains global offsets, and optionally spawns a dedicated disk-writer thread when `enable_async_write` is true.
2. **StreamingPinnedBuffer** – A circular queue backed by pinned (page-locked) memory to enable high-throughput `cudaMemcpyAsync` transfers. Producers acquire free chunks, fill them, and consumers flush them to disk.
3. **PinnedMemoryPool** – Re-usable pool of pinned buffers shared across multiple writers.

```mermaid
classDiagram
    class StreamingTensorWriter {
        +initialize()
        +write_tensor(data, size, is_gpu, stream)
        +finalize()
        -disk_writer_thread()
    }

    class StreamingPinnedBuffer {
        +get_free_chunk() int
        +mark_chunk_ready(slot, id, bytes)
        +get_ready_chunk() ReadyChunk
        +return_chunk(slot)
    }

    StreamingTensorWriter --> StreamingPinnedBuffer : uses buffers
    StreamingTensorWriter --> TensorWriter : writes to partitions
```

### Streaming Save Flow (GPU)

```mermaid
sequenceDiagram
    participant GPU as CUDA Device
    participant Writer as StreamingTensorWriter
    participant Buffer as StreamingPinnedBuffer
    participant Disk as TensorWriter/FS

    loop for each chunk
        GPU-->>Writer: enqueue cudaMemcpyAsync
        Writer->>Buffer: copy chunk (pinned)
        Buffer->>Writer: chunk ready
        alt async_write
            Writer->>Disk: (background thread) write_record()
        else sync_write
            Writer->>Disk: write_record()
        end
    end
```

Compared to the classic `TensorWriter`, this pipeline overlaps compute, PCIe transfers, and disk I/O, achieving significantly higher end-to-end throughput on multi-GB checkpoints.

## Python Binding Overview (`checkpoint_py.cc`)

`checkpoint_py.cc` exposes the full C++ API to Python via **pybind11**. The binding lives in `tensorcast.csrc` and provides:

* `save_tensors` / `restore_tensors` – Original synchronous APIs.
* `save_tensors_streaming` – Thin wrapper that forwards a `dict`-based config to `StreamingTensorWriter`.
* CUDA utilities – `allocate_cuda_memory`, `get_cuda_memory_handle`, `get_cuda_memory_ptr`, and `close_cuda_memory_handle` for zero-copy data exchange across processes.
* Replica verification helpers – `generate_artifact_verification_info` and `verify_artifact_data_from_gpu`.

Because the binding re-exports **pointers and CUDA IPC handles** as Python `bytes` objects, transferring large tensors between Python and C++ can be done without additional copies, aligning with the module's performance goals.