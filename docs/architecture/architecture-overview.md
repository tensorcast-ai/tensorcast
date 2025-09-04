---
title: Architecture Overview
description: High-level overview of the distributed artifact storage system architecture
sidebar_position: 1
---

# Architecture Overview

This document provides a high-level overview of the distributed artifact storage system's architecture. For detailed information about specific components, please refer to the dedicated guides.

## System Architecture

The distributed artifact storage system consists of two main components working together to provide efficient artifact storage and serving capabilities across a cluster:

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
- **Documentation**: [Store Daemon Architecture](../../daemon/README.md)

> See also: [Store Daemon (C++) Internals](../../daemon/README.md) — thin gRPC layer over the StoreEngine with session/ref tracking, transport locks, lifecycle management, and background sweepers.

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

## Interaction Patterns

### Artifact Loading Flow
1. Client requests artifact from local Store Daemon
2. Store Daemon checks local storage
3. If not available locally:
   - Queries Global Store for replica locations
   - Global Store selects optimal source using load balancing
   - Store Daemon transfers artifact via P2P from source
4. Store Daemon returns CUDA IPC handle to client
5. Client maps GPU memory and accesses artifact directly

### Load Balancing
The system implements sophisticated load balancing for artifact transfers:
- Prioritizes by memory type (GPU > RAM > DISK)
- Considers current load on each replica
- Ensures even distribution across the cluster
- **Documentation**: [P2P Transfer Strategies](./p2p-transfer-strategies.md)

## Deployment Modes

### 1. Standalone Mode
- Single Store Daemon without Global Store
- Suitable for single-node deployments
- Artifacts loaded from local disk only

### 2. Distributed Mode
- Multiple Store Daemons coordinated by Global Store
- Full P2P transfer capabilities
- Load balancing and high availability features

## Quick Start

### Starting Global Store
```bash
python -m tensorcast.global_store --db /path/to/models.db
```

### Starting Store Daemon
```bash
tensorcast start --config config.yaml
```

## Next Steps

- **Global Store Development**: See [Global Store Guide](./global-store.md)
- **Store Daemon Development**: See [Store Daemon Architecture](../../daemon/README.md)
- **High Availability**: See [HA Design](./high-availability-design.md)
- **P2P Transfers**: See [P2P Transfer Strategies](./p2p-transfer-strategies.md)
