---
title: Architecture Overview
description: High-level overview of the distributed artifact storage system architecture
sidebar_position: 1
---

# Architecture Overview

This document provides a high-level overview of the distributed artifact storage system's architecture. For detailed information about specific components, please refer to the dedicated guides.

## System Architecture

The system comprises a control plane (Global Store), a data plane (Store Daemons), and clients (User Process Workers) working together to provide efficient artifact storage and serving across a cluster:

```mermaid
graph TD
    subgraph "Control Plane"
        GS[Global Store<br/>Metadata & Coordination]
    end

    subgraph "Data Plane"
        SD1[Store Daemon 1<br/>Artifact Storage & Serving]
        SD2[Store Daemon 2<br/>Artifact Storage & Serving]
        SDN[Store Daemon N<br/>Artifact Storage & Serving]
    end

    subgraph "Clients"
        C1[Inference Worker 1]
        C2[Inference Worker 2]
        CN[Inference Worker N]
    end

    GS -.->|gRPC<br/>Metadata| SD1
    GS -.->|gRPC<br/>Metadata| SD2
    GS -.->|gRPC<br/>Metadata| SDN

    SD1 <-->|RDMA/TCP<br/>Artifact Transfer| SD2
    SD2 <-->|RDMA/TCP<br/>Artifact Transfer| SDN
    SD1 <-->|RDMA/TCP<br/>Artifact Transfer| SDN

    C1 -->|CUDA IPC| SD1
    C2 -->|CUDA IPC| SD2
    CN -->|CUDA IPC| SDN
```

## Core Components

### 1. Global Store
**Role**: Centralized metadata service and coordination layer

- **Responsibility**: Manages artifact registry, replica locations **and per-chunk directory metadata**, coordinates transfers
- **Technology**: gRPC service with DuckDB persistence
- **Key Feature**: Artifact data never flows through Global Store — only metadata (including chunk directory)
- **Chunk Directory**: Tracks residency/state for every chunk across the cluster
- **Documentation**: [Global Store Development Guide](./global-store.md)

### 2. Store Daemon
**Role**: Distributed artifact storage and serving engine

- **Responsibility**: Stores artifacts locally, serves them to clients, handles P2P transfers
- **Technology**: C++ service with gRPC interface (high-performance StoreEngine core)
- **Key Feature**: Zero-copy GPU memory sharing via CUDA IPC
- **Implementation Notes**: Thin gRPC layer over `StoreEngine`; manages sessions, PID references, and transport locks; exports CUDA IPC handles to clients
- **Documentation**: [Store Daemon Architecture](../../daemon/README.md)

> See also: [Store Daemon (C++) Internals](../../daemon/README.md) — thin gRPC layer over the StoreEngine with session/ref tracking, transport locks, lifecycle management, and background sweepers.

### 3. User Process Worker
**Role**: PyTorch client process accessing artifacts

- **Interface**: Uses `tensorcast.torch_util.get_artifact(key, ...)` to request artifacts via daemon `MaterializeByKey` (RFC‑0017)
- **Memory Access**: Maps CUDA IPC handles for zero‑copy GPU access; falls back to RAM/DISK as needed
- **Lifecycle**: Confirms, references, and unloads replicas via daemon RPCs

## Key Design Principles

### 1. Separation of Control and Data Planes
- **Control Plane** (Global Store): Handles metadata, coordination, and decision-making
- **Data Plane** (Store Daemons): Handles actual artifact storage and high-speed transfers
- **Benefit**: Scalability and performance - metadata operations don't interfere with data transfers

### 2. Zero-Copy Artifact Serving
- Artifacts are loaded once into GPU memory by Store Daemon
- Multiple client processes access the same GPU memory via CUDA IPC handles
- Eliminates redundant memory copies and reduces GPU memory usage

### 3. Peer-to-Peer Artifact Transfers
- Artifacts transfer directly between Store Daemons
- Uses RDMA for high-speed transfers when available, falls back to TCP
- Global Store coordinates transfers but doesn't handle artifact data

### 4. High Availability
- Global Store uses persistent database for recovery
- Store Daemons can reconnect and resynchronize state
- System designed for eventual consistency
- **Documentation**: [High Availability Design](./high-availability-design.md)

## Artifact Load Paths (Key‑based)

### P2P‑first Loading (Preferred)
```
Client                                 Store Daemon                        Global Store
   |                                         |                                   |
   |-- MaterializeByKey(key, device, uuid) ->|                                   |
   |                                          |-- ResolveKeyMapping(key) ------->|
   |                                          |<----------- artifact_id ---------|
   |                                          |-- Request Transport ------------>|
   |                                          |<------ Transport Grant ----------|
   |                                          |   (RDMA/TCP transfer from peer) |
   |                                          |-- Complete Transport ----------->|
   |                                          |<------ Confirmation -------------|
   |<- ALLOCATED + CUDA IPC handle -----------|                                   |
```

### Disk Fallback
```
Client                                 Store Daemon                        Global Store
   |                                         |                                   |
   |-- MaterializeByKey(key, device, uuid) ->|                                   |
   |                                          |-- ResolveKeyMapping(key) ------->|
   |                                          |<-- artifact_id (+ disk_path) ----|
   |                                          |   (Load from local disk)         |
   |                                          |-- Register Local Replica ------->|
   |                                          |<------ Replica ID ---------------|
   |<- ALLOCATED + CUDA IPC handle -----------|                                   |
```

## Load Balancing & Concurrency
- Prioritization: GPU > RAM > DISK, then by per‑replica load ratio
- Each replica tracks `max_concurrency` and `current_requests`; selection is atomic
- Daemon enforces transport locks; engine limits per‑GPU active transfers (1/session)
- **Further reading**: [P2P Transfer Strategies](./p2p-transfer-strategies.md)
