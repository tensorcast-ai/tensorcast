# CheckpointStore Refactoring

This directory contains the modular components that make up the refactored CheckpointStore implementation.

## Architecture Overview

The refactored design separates concerns into focused components:

```
CheckpointStore (Main API)
    ├── DeviceManager       - GPU device discovery and management
    ├── ModelRegistry       - Thread-safe model storage and lifecycle
    ├── MetricsCollector    - Centralized metrics collection
    ├── CommunicationManager - P2P/RDMA communication handling
    └── PinnedMemoryPool    - Memory allocation (existing component)
```

## Components

### DeviceManager (`device_manager.h/cc`)
- Manages GPU devices and CUDA operations
- Handles device discovery and UUID mapping
- Manages CUDA streams per device
- Tracks GPU memory usage

### ModelRegistry (`model_registry.h/cc`)
- Thread-safe storage of loaded models
- Tracks model access times for LRU eviction
- Provides model queries by location and state
- Manages model lifecycle

### MetricsCollector (`metrics_collector.h/cc`)
- Centralizes all metric collection logic
- Updates memory pool, model, and GPU metrics
- Records operation latencies and counters
- Tracks P2P transfers and memory evictions

### CommunicationManager (`communication_manager.h/cc`)
- Wraps the communication engine for P2P transfers
- Handles memory registration for RDMA
- Manages remote transfer setup

