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
- Provides registration observability consumed by `runtime::metadata::MetadataGateway`
  (`tc_register_pending_gauge`, `tc_register_commit_seconds{result=...}`) so daemon dashboards can monitor in-flight
  registrations and commit latency without querying StoreEngine directly.

### Metadata Runtime (`runtime/metadata/**`)
- `metadata::RegistrationBackend` (core/store/runtime/metadata/registration_backend.{h,cc}) owns the RFC-0006 lifecycle (begin → keepalive → commit/abort),
  coordinates UMA allocations via `DeviceManager`/`ReplicaRegistry`, emits registration events, and hands off Global Store publication through the MetadataGateway via the `RegistrationPublisher` interface.
- `metadata::MetadataGateway` (core/store/runtime/metadata/metadata_gateway.{h,cc}) is the only component wired to
  `IGlobalStoreClient`. It consumes ingestion completions directly from the pipeline, publishes registration results, refreshes TTLs,
  performs key-mapping CRUD, and re-emits compact RuntimeContext events for observers. StoreEngine, IngestionRuntime, and the registration backend
  now interact exclusively with this gateway instead of touching the Global Store client directly.

### TelemetryService (`runtime/telemetry_service.h/cc`)
- Wraps `ReplicaService` read-only queries (`get_resident_devices`, UMA snapshots, device memory, etc.) behind a consistent snapshot API.
- Consumes ingestion events to update metrics counters and P2P telemetry without duplicating logic in StoreEngine or pipeline stages.
- Keeps daemon-facing status APIs entirely outside the StoreEngine facade.

### CommunicationManager (`communication_manager.h/cc`)
- Wraps the communication engine for P2P transfers
- Handles memory registration for RDMA
- Manages remote transfer setup
- `initialize()` primes the staged TCP defaults (16 MiB GPU slices, 4 MiB CPU slices, 4 buffers/flow) so tests and toy setups meet `Communicator` invariants without bespoke config.

### RuntimeContext (`runtime/context/runtime_context.h/cc`)
- Aggregates the module-level services (`DeviceManager`, `ReplicaRegistry`, `MetricsCollector`, `PinnedBufferPool`, optional `IGlobalStoreClient`, `ViewHashComputer`, and `CommunicationManager`) and embeds the shared RuntimeContextEvents dispatcher.
- `start()` enforces `StoreEngineOptions` invariants, wires up the communication manager, initializes devices, and connects to Global Store when configured.
- `shutdown()` drains the event dispatcher, tears down registries and shared clients so tests can hot-reload `StoreEngine` instances without leaking state.
