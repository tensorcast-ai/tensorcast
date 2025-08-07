# RFC 0004 — Unified Memory Abstraction Refactor

Status: Proposed  
Date: 2025-01-09  
Owner: StepCast Core Team  
Depends on: 0001 (DVMP), 0002 (Unified Loader), 0003 (DVMP IO)  

## 0. Executive Summary

This RFC proposes a comprehensive refactoring of the DVMP and memory management architecture to achieve better separation of concerns, improved performance, and enhanced maintainability. The refactor introduces a hierarchical memory abstraction that cleanly separates virtual memory management, physical memory allocation, chunk state tracking, and device-specific operations while maintaining backward compatibility with existing APIs.

## 1. Motivation

After deep analysis of the current DVMP implementation and its integration with Model/MemoryManager abstractions, several architectural limitations have been identified:

### 1.1 Current Pain Points

1. **Tight Coupling**: DVMP conflates virtual memory management with chunk state tracking, IO operations, and distributed coordination
2. **Complex State Management**: Chunk states are scattered across DVMP, UnifiedModelMemory, and MemoryManager with unclear ownership
3. **Device Abstraction Leakage**: GPU-specific logic permeates through multiple layers instead of being properly encapsulated
4. **Inefficient Memory Patterns**: Fixed 256MiB chunks may be suboptimal for diverse model architectures (transformers vs CNNs)
5. **Limited Extensibility**: Adding new memory types (e.g., persistent memory, CXL) requires invasive changes
6. **Performance Bottlenecks**: 
   - Global mutex in DVMP for all operations
   - Synchronous chunk state updates blocking IO paths
   - No NUMA-aware allocation strategies
   - Missing zero-copy optimizations for UVA systems

### 1.2 Design Goals

1. **Clean Layering**: Separate virtual memory, physical allocation, state tracking, and device operations
2. **High Performance**: Lock-free data structures, NUMA awareness, and adaptive chunk sizing
3. **Extensibility**: Easy addition of new memory types and transport mechanisms
4. **Maintainability**: Clear ownership boundaries and minimal cross-layer dependencies
5. **Backward Compatibility**: Preserve existing Model API while improving internals

## 2. Proposed Architecture

### 2.1 Layered Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Model API (Unchanged)                     │
├─────────────────────────────────────────────────────────────┤
│                  Unified Memory Controller                   │
│  (Orchestration, Policy, Cross-Device Coordination)         │
├─────────────────┬─────────────────┬────────────────────────┤
│  Memory Backend │  State Tracker  │   Transport Layer      │
│  (Allocation)   │  (Chunk Meta)   │   (Load/Copy/P2P)     │
├─────────────────┴─────────────────┴────────────────────────┤
│                  Virtual Memory Layer                        │
│            (Address Space Management)                        │
├─────────────────────────────────────────────────────────────┤
│              Physical Memory Providers                       │
│        (DRAM, HBM, PMem, CXL, Remote)                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### 2.2.1 Virtual Memory Layer

```cpp
namespace stepcast::memory {

// Pure virtual address space management
class VirtualAddressManager {
public:
  struct Reservation {
    void* base;
    size_t size;
    int numa_node;
    std::string_view tag;
  };
  
  // Reserve contiguous VA range (no physical backing)
  absl::StatusOr<Reservation> reserve(size_t bytes, const ReserveOptions& opts);
  absl::Status release(const Reservation& reservation);
  
  // Query reservations
  std::vector<Reservation> get_reservations() const;
  absl::StatusOr<Reservation> find_by_tag(std::string_view tag) const;
};

} // namespace stepcast::memory
```

#### 2.2.2 Memory Backend Interface

```cpp
namespace stepcast::memory {

// Abstract interface for memory allocation backends
class IMemoryBackend {
public:
  virtual ~IMemoryBackend() = default;
  
  // Allocate physical memory within a VA reservation
  virtual absl::Status allocate(const Reservation& va, const AllocateOptions& opts) = 0;
  
  // Map file segments into VA (zero-copy when possible)
  virtual absl::Status map_file(const Reservation& va, const FileMapping& mapping) = 0;
  
  // Write data at offset
  virtual absl::Status write(const Reservation& va, uint64_t offset, 
                            const void* src, size_t bytes) = 0;
  
  // Memory attributes
  virtual MemoryAttributes get_attributes(const Reservation& va) const = 0;
  
  // Release physical backing (VA reservation remains)
  virtual absl::Status deallocate(const Reservation& va) = 0;
};

// Concrete implementations
class DRAMBackend : public IMemoryBackend { /* Anonymous mmap */ };
class HugePageBackend : public IMemoryBackend { /* 2MB/1GB pages */ };
class FileBackedBackend : public IMemoryBackend { /* mmap(MAP_PRIVATE) */ };
class PersistentMemoryBackend : public IMemoryBackend { /* DAX mmap */ };

} // namespace stepcast::memory
```

#### 2.2.3 Chunk State Tracker

```cpp
namespace stepcast::memory {

// Lock-free chunk metadata tracking
class ChunkStateTracker {
public:
  // Adaptive chunk sizing based on model characteristics
  struct ChunkLayout {
    size_t base_chunk_size;  // e.g., 64MB for small models, 512MB for large
    std::vector<size_t> chunk_boundaries;  // Support variable-sized chunks
  };
  
  // Initialize tracker with layout
  absl::Status initialize(std::string_view model_id, 
                         const ChunkLayout& layout);
  
  // Atomic state operations (lock-free)
  absl::Status update_state(std::string_view model_id, uint32_t chunk_idx, 
                           ChunkState new_state);
  ChunkState get_state(std::string_view model_id, uint32_t chunk_idx) const;
  
  // Bulk operations with memory ordering guarantees
  absl::Status update_states_bulk(std::string_view model_id,
                                 absl::Span<const uint32_t> indices,
                                 ChunkState new_state);
  
  // Zero-copy snapshot for schedulers
  absl::Span<const ChunkMeta> get_snapshot(std::string_view model_id) const;
  
  // Advanced queries
  std::vector<uint32_t> find_chunks_by_state(std::string_view model_id,
                                            ChunkState state) const;
  std::vector<uint32_t> find_chunks_for_eviction(std::string_view model_id,
                                                size_t target_bytes) const;
};

} // namespace stepcast::memory
```

#### 2.2.4 Transport Layer

```cpp
namespace stepcast::transport {

// Unified transport abstraction for all data movement
class ITransport {
public:
  virtual ~ITransport() = default;
  
  struct TransferRequest {
    MemoryLocation source;
    MemoryLocation destination;
    std::vector<ChunkRange> chunks;
    TransferPriority priority;
    CompletionCallback on_complete;
  };
  
  // Async transfer with automatic path selection
  virtual absl::StatusOr<TransferHandle> transfer_async(
      const TransferRequest& request) = 0;
  
  // Query capabilities
  virtual TransportCapabilities get_capabilities() const = 0;
};

// Concrete transports
class DirectMemoryTransport : public ITransport { /* CPU memcpy */ };
class CUDATransport : public ITransport { /* H2D, D2H, D2D */ };
class RDMATransport : public ITransport { /* RDMA verbs */ };
class NVLinkTransport : public ITransport { /* GPU P2P */ };

// Transport multiplexer with automatic path selection
class SmartTransport : public ITransport {
  // Automatically selects optimal transport based on:
  // - Source/destination characteristics
  // - Available bandwidth
  // - System topology (NUMA, PCIe, NVLink)
};

} // namespace stepcast::transport
```

#### 2.2.5 Unified Memory Controller

```cpp
namespace stepcast::memory {

// High-level orchestrator (replaces current MemoryManager)
class UnifiedMemoryController {
public:
  struct Config {
    std::shared_ptr<VirtualAddressManager> va_manager;
    std::shared_ptr<ChunkStateTracker> state_tracker;
    std::vector<std::shared_ptr<IMemoryBackend>> backends;
    std::shared_ptr<transport::SmartTransport> transport;
    AdaptivePolicy policy;  // For chunk sizing, eviction, etc.
  };
  
  explicit UnifiedMemoryController(Config config);
  
  // High-level operations
  absl::Status allocate_model(const ModelDescriptor& desc);
  absl::StatusOr<std::future<void>> load_model_async(
      const ModelDescriptor& desc,
      const LoadOptions& opts);
  
  // Memory operations
  absl::StatusOr<MemoryView> get_memory_view(const InstanceKey& key,
                                            MemoryLocation location);
  absl::Status copy_between_devices(const InstanceKey& key,
                                   DeviceID source, DeviceID target);
  
  // State management
  MemoryState get_state(const InstanceKey& key, MemoryLocation location) const;
  absl::Status mark_preemptible(const InstanceKey& key, float ratio);
  
  // Advanced features
  absl::Status enable_compression(const InstanceKey& key, CompressionType type);
  absl::Status enable_deduplication(const InstanceKey& key);
  absl::StatusOr<MemoryStats> get_stats(const InstanceKey& key) const;
};

} // namespace stepcast::memory
```

### 2.3 Key Improvements

#### 2.3.1 Lock-Free Chunk State Management

Replace current mutex-protected state with lock-free data structures:

```cpp
struct alignas(64) ChunkMeta {  // Cache-line aligned
  std::atomic<ChunkState> state{ChunkState::COLD};
  std::atomic<uint32_t> last_access_epoch{0};
  std::atomic<uint16_t> pin_count{0};
  std::atomic<uint16_t> access_count{0};  // For frequency-based eviction
};
```

#### 2.3.2 NUMA-Aware Memory Allocation

```cpp
struct AllocateOptions {
  PreferredNUMA numa_policy;
  bool use_huge_pages = true;
  bool populate_on_alloc = false;  // MAP_POPULATE
  MemoryAdvice advice = MemoryAdvice::SEQUENTIAL;  // madvise hints
};
```

#### 2.3.3 Adaptive Chunk Sizing

Instead of fixed 256MB chunks:

```cpp
class AdaptiveChunkPolicy {
  size_t compute_optimal_chunk_size(const ModelDescriptor& desc) {
    // Factors:
    // - Model size (larger models → larger chunks)
    // - Model type (transformer → larger, CNN → smaller)
    // - Available memory (low memory → smaller chunks)
    // - Network bandwidth (high BW → larger chunks)
    return calculated_size;
  }
};
```

#### 2.3.4 Zero-Copy Optimizations

```cpp
class ZeroCopyOptimizer {
  bool can_use_zero_copy(const MemoryLocation& src, 
                        const MemoryLocation& dst) {
    // Check for:
    // - Unified Virtual Addressing (UVA)
    // - Same NUMA node
    // - GPU peer access capability
    return eligible;
  }
};
```

#### 2.3.5 Compression and Deduplication

```cpp
class CompressionBackend : public IMemoryBackend {
  // Transparent compression for cold chunks
  // LZ4 for speed, ZSTD for ratio
};

class DeduplicationEngine {
  // Content-based chunk deduplication
  // Useful for models with repeated patterns
};
```

## 3. Migration Strategy

### 3.1 Phase 1: Foundation (4 weeks)

1. Implement VirtualAddressManager as a wrapper around current DVMP VA operations
2. Extract ChunkStateTracker from DVMP without changing external APIs
3. Create IMemoryBackend interface with DRAMBackend wrapping current implementation

### 3.2 Phase 2: Transport Layer (3 weeks)

1. Implement transport abstractions wrapping current loaders
2. Add SmartTransport with basic path selection
3. Integrate with pump architecture from RFC 0002

### 3.3 Phase 3: Controller Integration (4 weeks)

1. Implement UnifiedMemoryController wrapping current MemoryManager
2. Gradually move logic from MemoryManager to new components
3. Update Model to use UnifiedMemoryController

### 3.4 Phase 4: Advanced Features (6 weeks)

1. Lock-free chunk state implementation
2. NUMA-aware allocation
3. Adaptive chunk sizing
4. Compression/deduplication (optional)

### 3.5 Phase 5: Cleanup (2 weeks)

1. Remove old DVMP implementation
2. Deprecate old MemoryManager APIs
3. Update documentation and tests

## 4. Performance Optimizations

### 4.1 Memory Access Patterns

```cpp
// Optimize for sequential access during loading
class SequentialPrefetcher {
  void prefetch_ahead(const Reservation& va, size_t offset, size_t ahead_bytes) {
    // Use madvise(MADV_WILLNEED) or manual prefetch
  }
};
```

### 4.2 Parallel State Updates

```cpp
// Batch state updates to amortize atomic operations
class BatchedStateUpdater {
  void update_states_parallel(absl::Span<ChunkMeta> chunks,
                             ChunkState new_state) {
    // Use parallel_for with proper memory ordering
    std::for_each(std::execution::par_unseq, 
                  chunks.begin(), chunks.end(),
                  [new_state](ChunkMeta& chunk) {
                    chunk.state.store(new_state, std::memory_order_release);
                  });
  }
};
```

### 4.3 Smart Eviction

```cpp
class EvictionPolicy {
  // Multi-factor scoring: recency, frequency, transfer cost
  float compute_eviction_score(const ChunkMeta& chunk) {
    float recency_score = compute_recency(chunk.last_access_epoch);
    float frequency_score = compute_frequency(chunk.access_count);
    float location_score = is_gpu_backed(chunk) ? 0.1f : 1.0f;
    return recency_score * 0.5f + frequency_score * 0.3f + location_score * 0.2f;
  }
};
```

## 5. Compatibility

### 5.1 API Preservation

The Model class API remains unchanged:

```cpp
// These APIs continue to work as before
model->ensure_loaded_async(location, concurrency);
model->get_data_pointer(location);
model->enable_remote_memory_access(location, comm_engine);
```

### 5.2 Configuration Migration

```yaml
# Old configuration
dvmp:
  chunk_size: 256MiB
  
# New configuration (backward compatible)
memory:
  dvmp:
    chunk_size: 256MiB  # Honored if present
  controller:
    adaptive_chunks: true  # New default
    min_chunk_size: 64MiB
    max_chunk_size: 1GiB
```

## 6. Testing Strategy

### 6.1 Unit Tests

- Lock-free chunk state operations under contention
- NUMA allocation correctness
- Transport path selection
- Adaptive chunk sizing

### 6.2 Integration Tests

- End-to-end model loading with new architecture
- Cross-device transfers
- Memory pressure handling
- Fallback mechanisms

### 6.3 Performance Tests

- Benchmark against current implementation
- Measure lock contention reduction
- Validate zero-copy optimizations
- Test adaptive chunk sizing impact

## 7. Risk Mitigation

### 7.1 Performance Regression

- Maintain performance benchmarks throughout migration
- Feature flags for gradual rollout
- Ability to revert to old implementation

### 7.2 Compatibility Issues

- Extensive testing with existing workloads
- Backward compatibility shims where needed
- Clear migration documentation

### 7.3 Complexity Management

- Incremental refactoring approach
- Clear ownership boundaries
- Comprehensive documentation

## 8. Success Metrics

1. **Performance**
   - 50% reduction in lock contention for state updates
   - 20% improvement in model loading throughput
   - <1μs chunk state query latency (maintained)

2. **Maintainability**
   - 40% reduction in cross-component dependencies
   - Clear layer boundaries with defined interfaces
   - Improved testability with mockable components

3. **Extensibility**
   - New memory backend in <500 LoC
   - New transport in <300 LoC
   - Policy changes without core modifications

## 9. Future Work

1. **CXL Memory Support**: Add CXLBackend for memory expansion
2. **Persistent Memory**: Leverage PMem for model caching
3. **Heterogeneous Chunks**: Mix chunk sizes within a model
4. **Smart Placement**: ML-based placement optimization
5. **Global Optimization**: Cross-node memory balancing

## 10. Conclusion

This refactoring addresses fundamental architectural limitations while preserving the strengths of the current system. By introducing clear abstractions and separation of concerns, we achieve better performance, maintainability, and extensibility. The incremental migration strategy ensures system stability while delivering improvements progressively.

The proposed architecture positions StepCast for future growth in model sizes, hardware diversity, and deployment scenarios while maintaining the high performance required for production ML workloads.