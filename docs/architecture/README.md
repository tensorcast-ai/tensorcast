---
title: Architecture Documentation
description: Technical architecture documentation for the distributed artifact storage system
---

# Architecture Documentation

This section contains detailed technical documentation about the system architecture of the distributed artifact storage system.

## Documentation Structure

### [Architecture Overview](./architecture-overview.md)
High-level overview of the system architecture, core components, and design principles. Start here to understand how the system works.

### [API Design](./api/README.md)
Public SDK surface and internal API flows (registration, materialization, policy, persistence, region-backed).

### [Artifact Views and Retrieval](./artifact-views-and-retrieval.md)
Canonical view semantics, planning, and retrieval/materialization pipeline.

### [View Replicas and Assembly](./view-replicas-and-assembly.md)
Dense piece replicas, assembly lifecycle, and sealing semantics.

### Key-based Loading
Recommended client API is key-based: clients call the daemon’s `MaterializeByKey` and reconstruct tensors using canonical indices fetched from Global Store (`GetArtifactIndexById`). The daemon resolves the key, orchestrates P2P transfers, and performs disk fallback when needed. See:
- [P2P Transfer Strategies](./p2p-transfer-strategies.md)
- [Internals: Artifact Loading Workflow](../internals/model-loading.md)

### [Global Store Development Guide](../../tensorcast/global_store/README.md)
Detailed development guide for the Global Store component, including its layered architecture, API reference, and implementation details.

### [Store Daemon Architecture](../../daemon/README.md)
In-depth documentation of the Store Daemon's internal architecture, component interactions, and C++ core integration.

### VRAM Leased-In-Place
Lease existing GPU allocations to the daemon with `in_place=true` (no coalescing at commit). Same‑device consumers are rejected; cross‑device consumers are served via a D2D copy to a coalesced replica. P2P remains staged‑only. See the LIP section in [Store Daemon Architecture](../../daemon/README.md).


### [High Availability Design](./high-availability-design.md)
Comprehensive guide to the system's high availability features, failure recovery mechanisms, and state synchronization protocols.

### [P2P Transfer Strategies](./p2p-transfer-strategies.md)
Details about peer-to-peer artifact transfer strategies, load balancing algorithms, and performance optimizations.

## Quick Navigation

- **New to the system?** → Start with [Architecture Overview](./architecture-overview.md)
- **Working on SDK APIs?** → See [API Design](./api/README.md)
- **Working on Global Store?** → See [Global Store Development Guide](../../tensorcast/global_store/README.md)
- **Working on Store Daemon?** → See [Store Daemon Architecture](../../daemon/README.md)
- **Need HA features?** → Read [High Availability Design](./high-availability-design.md)
- **Optimizing transfers?** → Check [P2P Transfer Strategies](./p2p-transfer-strategies.md)
