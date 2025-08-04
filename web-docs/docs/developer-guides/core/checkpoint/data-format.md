---
title: Data File Format Specification
description: Complete format specifications for partitioned binary checkpoint format
sidebar_position: 3
---

# Data File Format Specification

## Overview

The checkpoint system stores model data in a partitioned binary format optimized for large-scale models. This document provides complete format specifications to enable custom checkpoint loaders.

## File Structure

A checkpoint consists of multiple files in a directory:

```
model_checkpoint/
├── tensor.data_0      # First partition (up to 10GB)
├── tensor.data_1      # Second partition (up to 10GB)
├── tensor.data_N      # Additional partitions as needed
├── tensor_index.json  # Tensor metadata and offsets
└── verification.json  # Model integrity information (optional)
```

### Backward Compatibility

For single-file models (< 10GB), the system may use:
```
model_checkpoint/
├── tensor.data        # Single file (legacy format)
└── tensor_index.json  # Tensor metadata
```

## Binary Data Format (`tensor.data_*`)

### File-Level Structure

Each partition file contains raw tensor data with specific alignment requirements:

```
┌─────────────────────────────────────────────────────────────┐
│ Tensor Record 1                                             │
├─────────────────────────────────────────────────────────────┤
│ Alignment Padding (0-7 bytes)                               │
├─────────────────────────────────────────────────────────────┤
│ Tensor Record 2                                             │
├─────────────────────────────────────────────────────────────┤
│ ...                                                         │
└─────────────────────────────────────────────────────────────┘
```

### Tensor Record Format

Each tensor is stored as raw binary data:

```cpp
// Pseudo-structure for documentation
struct TensorRecord {
    uint8_t raw_data[tensor_size];  // Raw tensor bytes in native format
    uint8_t padding[0-7];           // 64-bit alignment padding
};
```

### Alignment Rules

1. **Tensor Record Alignment (8-byte)**:
   - Each tensor record starts at an 8-byte (64-bit) aligned offset
   - Padding is added after each tensor: `padding = (8 - (size % 8)) % 8`
   - This ensures efficient memory access and correct pointer alignment

2. **File I/O Alignment (4K)**:
   - Data is written using 4K-aligned buffers for performance
   - The AlignedBuffer flushes data in 4K chunks
   - Overall partition file size is a multiple of 4096 bytes
   - Individual tensor boundaries are NOT 4K-aligned

3. **Key Distinction**:
   - Tensor-level: 8-byte alignment for memory layout
   - File-level: 4K alignment for I/O performance

## Metadata Format (`tensor_index.json`)

The `tensor_index.json` file contains tensor metadata in the following format:

```json
{
  "tensor_name_1": [offset, size, shape, stride, dtype, storage_offset],
  "tensor_name_2": [offset, size, shape, stride, dtype, storage_offset],
  "layer.weight": [1024, 524288, [768, 768], [768, 1], "torch.float16", 0],
  "layer.bias": [525312, 2048, [2048], [1], "torch.float32", 0],
  "view.tensor": [1024, 524288, [384, 768], [768, 1], "torch.float16", 196608]
}
```

### Field Descriptions

- **Key**: Tensor name (string) - matches PyTorch `state_dict` keys
- **Value**: Array with **six** elements (v2 format) or **five** elements (v1 legacy):
  - `offset` (uint64): Global byte offset across all partitions
  - `size` (uint64): Storage size in bytes (may be larger than tensor size for views)
  - `shape` (list&lt;uint64&gt;): Tensor shape as reported by PyTorch
  - `stride` (list&lt;uint64&gt;): Stride information used by PyTorch to map indices to memory
  - `dtype` (string): PyTorch dtype (e.g., `"torch.float16"`, `"torch.float32"`)
  - `storage_offset` (uint64): **[v2+ only]** Offset in elements within the storage (for tensor views/slices)

### Global Offset Calculation

Offsets are cumulative across partitions:

```
Partition 0: tensor.data_0 [0 ... 10GB-1]
Partition 1: tensor.data_1 [10GB ... 20GB-1]
Partition N: tensor.data_N [N*10GB ... end]
```

Example:
```json
{
  "embedding.weight": [0, 1048576, [256, 4096], [4096, 1], "torch.float16", 0],          // In partition 0
  "layer1.weight": [10737418240, 2097152, [4096, 4096], [4096, 1], "torch.float16", 0],   // In partition 1 (10GB + offset)
  "layer2.weight": [21474836480, 4194304, [4096, 4096], [4096, 1], "torch.float16", 0]    // In partition 2 (20GB + offset)
}
```

## Tensor Data Layout

### Data Types

Tensors are stored in their native PyTorch format without type conversion:

| PyTorch Type | C++ Type | Size (bytes) | Format |
|--------------|----------|--------------|---------|
| torch.float32 | float | 4 | IEEE 754 single precision |
| torch.float16 | half | 2 | IEEE 754 half precision |
| torch.bfloat16 | bfloat16 | 2 | Brain floating point |
| torch.int64 | int64_t | 8 | Signed 64-bit integer |
| torch.int32 | int32_t | 4 | Signed 32-bit integer |
| torch.uint8 | uint8_t | 1 | Unsigned 8-bit integer |

### Memory Layout

Tensors are stored in row-major order (C-style) as contiguous memory blocks:

```cpp
// For a tensor with shape [2, 3, 4]
// Storage order: [0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], ...
for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
        for (int k = 0; k < dim2; k++) {
            // tensor[i][j][k] is stored sequentially
        }
    }
}
```

## Partition Management

### Partition Size Limits

- **Maximum Size**: 10GB (10,737,418,240 bytes) per partition
- **Rollover**: When current partition + tensor size > 10GB, create new partition
- **Aligned Writes**: Data is flushed using 4K-aligned buffers. No explicit file-level padding is added when rolling over to a new partition.

### Partition Naming Convention

```
tensor.data_0    # First partition (0-based indexing)
tensor.data_1    # Second partition
tensor.data_N    # Nth partition
```

For backward compatibility, single files use `tensor.data` (no suffix).

## Model Verification Format

### Verification File (`verification.json`)

Optional integrity verification data:

```json
{
  "model_size": 21474836480,
  "partition_count": 3,
  "hash_algorithm": "sha256",
  "full_model_hash": "abc123...",
  "partition_hashes": [
    "def456...",
    "ghi789...",
    "jkl012..."
  ],
  "checkpoints": [
    {
      "offset": 1073741824,
      "hash": "mno345..."
    }
  ]
}
```

## Storage Deduplication

PyTorch tensors can share underlying storage (e.g., views, slices). The checkpoint system handles this by:

1. **Write-once**: Each unique storage is written only once, using the largest size among all tensors sharing it
2. **Multiple references**: Multiple tensors in `tensor_index.json` may have the same `offset`
3. **Storage offset**: The 6th field (`storage_offset`) indicates where within the storage a tensor's data begins

Example:
```json
{
  "model.weight": [0, 4194304, [1024, 1024], [1024, 1], "torch.float32", 0],
  "model.weight_T": [0, 4194304, [1024, 1024], [1, 1024], "torch.float32", 0],
  "model.weight_slice": [0, 4194304, [512, 1024], [1024, 1], "torch.float32", 0]
}
```

All three tensors share the same storage at offset 0, but have different shapes/strides.

## Common Pitfalls

1. **Endianness**: Data is stored in native byte order
2. **Alignment**: Respect 8-byte alignment when calculating offsets
3. **File Handles**: Close files properly to avoid resource leaks
4. **Error Handling**: Check file operations for failures
5. **Large Files**: Use 64-bit offsets for files > 4GB
6. **Shared Storage**: Multiple tensors may reference the same storage offset
7. **Storage Offset**: When using v2 format, apply `storage_offset * element_size` to find actual data start
8. **Version Compatibility**: Always check tensor index array length to distinguish v1 (5 elements) from v2 (6 elements)