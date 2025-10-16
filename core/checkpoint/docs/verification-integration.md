---
title: Replica Integrity Verification
description: Lightweight replica integrity verification solution for storage and transfer
sidebar_position: 5
---

# Replica Integrity Verification

This document summarizes the lightweight replica integrity verification solution integrated in the replica storage, loading, and transfer system.

## Design Philosophy

- **Zero-Copy Design**: Directly verify loaded replica memory to avoid additional memory allocation
- **Chunked Streaming Computation**: Process large replica data chunk by chunk using a fixed buffer size
  - Protocol: segment hashing uses a fixed buffer size of 256KB (constexpr `SEGMENT_HASH_BUFFER_SIZE`)
  - This constant is used in both generation and verification paths to guarantee determinism
- **Memory-Friendly**: Use minimal temporary memory
- **Multi-Level Verification**: From fast key-point verification to complete hash verification
- **Millisecond Performance**: Overall verification time controlled at millisecond level, supporting ultra-large models

## Multi-Level Verification Strategy

### 1. Key Points Verification (KEY_POINTS)
- Check key positions at the beginning, middle, and end of the replica
- Almost zero overhead, suitable for real-time checking
- Time complexity: O(1)

### 2. Sparse Sampling (SPARSE_SAMPLING)
- Sample 16 points at fixed intervals for verification
- Balance performance and detection accuracy
- Time complexity: O(1)

### 3. Segment Verification (SEGMENT_HASHES)
- Divide replica into 8 segments, calculate independent checksum for each segment
- Fixed buffer policy: both generation and verification read data using 256KB chunks
- Default recommended level, balancing performance and accuracy
- Time complexity: O(n)

### 4. Full Verification (FULL_HASH)
- Generate overall data fingerprint using rolling hash
- Highest accuracy, suitable for critical scenarios
- Time complexity: O(n)

## Integration Points Overview

### 1. Replica Saving Phase (`tensorcast/api/_io_disk.py` + `checkpoint_py.cc`)

```python
# In save_dict function
verification_info = generate_artifact_verification_info(disk_path)
with open(verification_path, "w") as f:
    json.dump(verification_info, f, indent=2)
```

**Key Features:**
- Automatically generate verification information after saving
- Support multi-partition file format (`tensor.data_0`, `tensor.data_1`, ...)
- Save verification information to `verification.json`

### 2. Replica Loading Phase (`tensorcast/api/services/materialize.py` + `Store.get`)

```python
# In load_dict_non_blocking function
if enable_verification:
    verification_result = verify_artifact_data_from_gpu(
        device_id, cuda_memory_ptr, memory_size, expected_verification, 2
    )
```

**Key Features:**
- Automatically verify data integrity after loading
- Use segment-level verification by default (level=2)
- Throw exception when verification fails to prevent continued execution

### 3. P2P Transfer Phase (`p2p_loader.cc`)

```cpp
// After P2P transfer completion
if (config.enable_verification && config.verification_info.has_value()) {
    absl::Status verification_status = ModelVerifier::verify_key_points(
        local_ptrs, local_data_sizes, config.verification_info.value(), device_id);
}
```

**Key Features:**
- Immediate verification after transfer completion
- Use fast key-point verification to avoid transfer performance impact
- Return `DataLossError` when verification fails

### 4. Replica Class Interface (`replica.h` + `replica.cc`)

```cpp
// Generate verification information
absl::StatusOr<ModelVerificationInfo> generate_verification_info(MemoryLocation location);

// Verify replica data
absl::Status verify_artifact_data(MemoryLocation location,
                               const ModelVerificationInfo& expected_info,
                               VerificationLevel level = VerificationLevel::SEGMENT_HASHES);

// Fast key-point verification
absl::Status verify_key_points(MemoryLocation location,
                               const ModelVerificationInfo& expected_info);
```

**Key Features:**
- Unified verification interface
- Support CPU and GPU memory verification
- Thread-safe implementation

### 5. Configuration Support (`replica_config.h`)

```cpp
struct P2PModelSource {
    // ...
    bool enable_verification = true;
    std::optional<ModelVerificationInfo> verification_info;
};
```

**Key Features:**
- Integrate verification options in P2P source configuration
- Optional expected verification information passing

## Usage Examples

### Python-side Saving and Loading

```python
# Automatically generate verification information when saving
save_dict(state_dict, "/path/to/replica")

# Automatically verify when loading via the Store facade
from tensorcast import FallbackOptions, GetArtifactOptions, get

state_dict = get(
    fallback=FallbackOptions(disk_path="/path/to/replica", prefer_disk=True),
    options=GetArtifactOptions(enable_verification=True),
)
```

### C++-side Replica Operations

```cpp
// Generate verification information
auto verification_info = replica->generate_verification_info(MemoryLocation::GPU);

// Fast verification
auto status = replica->verify_key_points(MemoryLocation::GPU, verification_info.value());

// Full verification
auto status = replica->verify_artifact_data(MemoryLocation::GPU, verification_info.value(),
                                       VerificationLevel::FULL_HASH);
```

### P2P Transfer Testing

```cpp
// Server-side generates verification information
auto verification_info = replica->generate_verification_info(register_location);

// Client configuration includes verification information
P2PModelSource source;
source.enable_verification = true;
source.verification_info = verification_info;

// Automatic verification after transfer
```

## Performance Characteristics

- **Key Points Verification**: < 1ms (any artifact size)
- **Segment Verification**: ~10ms (67B parameter replica) using 256KB chunks
- **Full Verification**: ~100ms (67B parameter replica)
- **Memory Overhead**: ~256KB (fixed segment hashing buffer)

## Why a fixed chunk size?

- **Determinism across code paths**: The current segment hashing uses repeated xxhash64 with the previous result as the seed per chunk. This composition is not equivalent to the official streaming algorithm; the hash outcome becomes sensitive to chunk boundaries. A fixed chunk size guarantees the same boundaries are used during both generation and verification, avoiding false mismatches.
- **Protocol guarantee**: By fixing the buffer at 256KB via a single constexpr (`SEGMENT_HASH_BUFFER_SIZE`), all implementations (CPU/GPU, save/load, P2P) agree on the same segmentation behavior without per-site configuration drift.
- **Cross-device parity**: Ensures GPU-generated `verification.json` matches GPU/CPU verification paths even when execution environments differ.

## Optimization directions

- **Adopt streaming hashing**: Switch segment hashing to a true streaming xxhash64 implementation (or equivalent) so results are chunk-size invariant. This would allow dynamic buffers without affecting outcomes.
- **Embed metadata in `verification.json`**: Record `hash_algo`, `algo_version`, `byte_space_id`, and `chunk_size_bytes` for forward compatibility. Verifiers select the correct record for canonical or variant ByteSpaces and can respect legacy files while migrating algorithms.
- **Parallelize per-segment hashing**: Each of the 8 segments can be hashed concurrently; aggregate into the final record to reduce wall-clock time on large artifacts.
- **Hardware-accelerated paths**: Explore CUDA kernels or vendor intrinsics for hashing on GPU-resident data to reduce PCIe copies and improve throughput.
- **Robust fallback levels**: If segment-level verification fails due to legacy records, optionally auto-downgrade to `KEY_POINTS`/`SPARSE_SAMPLING` with clear telemetry, while surfacing a remediation hint.
- **Storage Overhead**: < 1KB (`verification.json`)

## Error Handling

- **Verification Failed**: Throw detailed error information, prevent subsequent operations
- **Verification Information Missing**: Warning log, continue execution
- **Verification System Exception**: Warning log, downgrade to basic functionality

## Extensibility

- **New Verification Algorithms**: Add through `VerificationLevel` enumeration
- **New Data Sources**: Implement `ModelVerifier` interface methods
- **Custom Strategies**: Adjust verification behavior through configuration parameters

This integration solution provides complete replica lifecycle verification coverage, providing reliable data integrity guarantees while maintaining extremely low overhead.
