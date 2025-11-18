# StoreEngine Refactoring

This directory contains the modular components that make up the refactored StoreEngine implementation.

## Architecture Overview

The refactored design separates concerns into focused components:

```
StoreEngine (Main API)
    ├── DeviceManager       - GPU device discovery and management
    ├── ReplicaRegistry       - Thread-safe replica storage and lifecycle
    ├── MetricsCollector    - Centralized metrics collection
    ├── CommunicationManager - P2P/RDMA communication handling
    └── PinnedBufferPool    - Memory allocation (canonical)
```

## Components

### DeviceManager (`device_manager.h/cc`)
- Manages GPU devices and CUDA operations
- Handles device discovery and UUID mapping
- Manages CUDA streams per device
- Tracks GPU memory usage

### ReplicaRegistry (`replica_registry.h/cc`)
- Thread-safe storage of loaded replicas
- Tracks replica access times for LRU eviction
- Provides replica queries by location and state
- Manages replica lifecycle

### MetricsCollector (`metrics_collector.h/cc`)
- Centralizes all metric collection logic
- Updates memory pool, replica, and GPU metrics
- Records operation latencies and counters
- Tracks P2P transfers and memory evictions
- Provides registration observability consumed by `RegistrationFacade` (`tc_register_pending_gauge`,
  `tc_register_commit_seconds{result=...}`) so daemon dashboards can monitor in-flight registrations and commit latency.

### RegistrationFacade (`registration/registration_facade.h/cc`)
- Owns the RFC-0006 lifecycle (begin → ingest → keepalive → commit/abort) outside of `StoreEngine`.
- Maintains `PendingRegistrationContext` instances with TTL metadata, CUDA IPC handles, replicas, and view planners.
- Delegates GPU memory allocation/eviction through `DeviceManager`, replica creation via `ReplicaFactory`, and emits registration events.
- Global Store publication stays in `components::runtime::GlobalStorePublisher` so the facade never reaches the Global Store client directly.
- Emits structured metrics (`tc_register_pending_gauge`, `tc_register_commit_seconds{result=...}`) and wraps Begin/Commit in
  `SC_TRACE_INIT_GUARD` spans for distributed tracing.

### GlobalStorePublisher (`runtime/global_store_publisher.h/cc`)
- The only component allowed to call `IGlobalStoreClient`.
- Consumes ingestion/registration events and registers/deregisters replicas, variant residency, and key mappings.
- Exposes helper methods to StoreEngine when manual registration/unregistration is required.

### TelemetryService (`runtime/telemetry_service.h/cc`)
- Wraps `ReplicaService` read-only queries (`get_resident_devices`, UMA snapshots, device memory, etc.) behind a consistent snapshot API.
- Consumes ingestion events to update metrics counters and P2P telemetry without duplicating logic in StoreEngine or pipeline stages.
- Keeps daemon-facing status APIs entirely outside the StoreEngine facade.

### CommunicationManager (`communication_manager.h/cc`)
- Wraps the communication engine for P2P transfers
- Handles memory registration for RDMA
- Manages remote transfer setup
- `initialize()` primes the staged TCP defaults (16 MiB GPU slices, 4 MiB CPU slices, 4 buffers/flow) so tests and toy setups meet `Communicator` invariants without bespoke config.

### ComponentCatalog (`runtime/component_catalog.h/cc`)
- Aggregates the module-level services (`DeviceManager`, `ReplicaRegistry`, `MetricsCollector`, `PinnedBufferPool`, and optional `IGlobalStoreClient`).
- `start()` enforces `StoreEngineOptions` invariants, wires up the communication manager, initializes devices, and connects to Global Store when configured.
- `shutdown()` tears down registries and shared clients so tests can hot-reload `StoreEngine` instances without leaking state.
