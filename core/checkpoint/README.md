---
title: Checkpoint Module Overview
description: Efficient tensor serialization and deserialization capabilities for TensorCast
sidebar_position: 1
---

# Checkpoint Module Documentation

## Overview

The Checkpoint module provides efficient tensor serialization and deserialization capabilities for the TensorCast. It's designed to handle large-scale replica checkpoints with optimized I/O performance and memory management.

## Key Features

- **Efficient I/O**: Uses 4K-aligned buffers for optimal disk performance
- **Partitioned Storage**: Automatically splits large models into manageable chunks (10GB per partition)
- **CUDA Integration**: Native support for GPU memory management and IPC handles
- **Verification Support**: Built-in replica integrity verification
- **Streaming GPU Tensor Saving**: Asynchronous GPU→Host→Disk pipeline using pinned memory (`StreamingTensorWriter`)
- **Python Bindings**: First-class PyTorch integration via `tensorcast.csrc.checkpoint_py` (zero-copy where possible)
- **Memory Optimization**: Handles shared tensor storages to avoid duplication

## Module Structure

```
core/checkpoint/
├── aligned_buffer.h/cpp     # 4K-aligned buffer for optimized file I/O
├── tensor_writer.h/cpp      # High-level tensor writing with partitioning
├── streaming_tensor_writer.h/cpp # GPU streaming writer built on pinned memory
├── checkpoint.h/cpp         # Main API for save/restore operations
└── progress_bar.h          # Progress reporting utilities
```

core/common/memory/
└── streaming_pinned_buffer.h # Circular pinned buffer for StreamingTensorWriter
tensorcast/csrc/
└── checkpoint_py.cc         # PyBind11 bindings exposing C++ APIs to Python

## Documentation Index

1. [Architecture Design](./docs/architecture.md) - Module relationships and design patterns
2. [Data Format Specification](./docs/data-format.md) - Checkpoint file format details

## Quick Start

### Saving Tensors
```cpp
// Prepare tensor data
std::vector<std::string> tensor_names = {"layer1.weight", "layer1.bias"};
std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> tensor_data;
// ... populate tensor_data with (data_ptr, size) pairs

// Save to disk
auto offsets = tensorcast::store::save_tensors(tensor_names, tensor_data, "/path/to/replica");
```

### Loading Tensors
```cpp
// Load from disk
auto tensors = tensorcast::store::restore_tensors_from_disk(
    meta_state_dict, "/path/to/replica", tensor_offsets);
```

### Streaming Save Example (GPU)
```cpp
#include "core/checkpoint/streaming_tensor_writer.h"

// Assume `gpu_tensor_ptr` points to data on CUDA device 0
const void* gpu_tensor_ptr = ...;
size_t num_bytes = ...;

// Configure writer (4 buffers × 256 MB each)
tensorcast::store::StreamingTensorWriter::Config cfg;
cfg.num_buffers = 4;
cfg.buffer_size_mb = 256;

// Instantiate and use the writer
tensorcast::store::StreamingTensorWriter writer("/path/to/replica/tensor.data_0", cfg, nullptr);
SC_CHECK_OK(writer.initialize());
SC_CHECK_OK(writer.write_tensor(gpu_tensor_ptr, num_bytes, /*is_gpu=*/true));
SC_CHECK_OK(writer.finalize());
```

Python users can call the same functionality via the binding:

```python
import tensorcast.store as scs

# tensor_names, tensor_data prepared as before
cfg = {
    "num_buffers": 4,
    "buffer_size_mb": 256,
    "enable_async_write": True,
}
offsets = scs.save_tensors_streaming(tensor_names, tensor_data, "/path/to/replica", cfg)
```

### Async Copy Manager (ACM) Usage

`StreamingTensorWriter` uses `AsyncCopyManager` to perform GPU→Host (D2H) copies into pinned slots without per-chunk synchronizations.
The ACM schedules a host callback that marks the slot ready for the disk writer thread, improving overlap and throughput.

## File Format Overview

The checkpoint system generates:
- `tensor.data_N`: Partitioned binary tensor data (N = 0, 1, 2, ...)
- `tensor_index.json`: Metadata with tensor offsets and sizes
- Verification information for integrity checking

See [Data Format Specification](./docs/data-format.md) for detailed format documentation.
