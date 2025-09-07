---
title: Architecture Documentation
description: Technical architecture documentation for the distributed artifact storage system
---

# Architecture Documentation

This section contains detailed technical documentation about the system architecture of the distributed artifact storage system.

## Documentation Structure

### [Architecture Overview](./architecture-overview.md)
High-level overview of the system architecture, core components, and design principles. Start here to understand how the system works.

### Key-based Loading (RFC‑0015)
Recommended client API is key-based: clients call the daemon’s `MaterializeByKey` and reconstruct tensors using canonical indices fetched from Global Store (`GetArtifactIndexById`). The daemon resolves the key, orchestrates P2P transfers, and performs disk fallback when needed. See:
- [P2P Transfer Strategies](./p2p-transfer-strategies.md)
- [Internals: Artifact Loading Workflow](../internals/model-loading.md)

### [Global Store Development Guide](../../tensorcast/global_store/README.md)
Detailed development guide for the Global Store component, including its layered architecture, API reference, and implementation details.

### [Store Daemon Architecture](../../daemon/README.md)
In-depth documentation of the Store Daemon's internal architecture, component interactions, and C++ core integration.

### [High Availability Design](./high-availability-design.md)
Comprehensive guide to the system's high availability features, failure recovery mechanisms, and state synchronization protocols.

### [P2P Transfer Strategies](./p2p-transfer-strategies.md)
Details about peer-to-peer artifact transfer strategies, load balancing algorithms, and performance optimizations.

## Quick Navigation

- **New to the system?** → Start with [Architecture Overview](./architecture-overview.md)
- **Working on Global Store?** → See [Global Store Development Guide](../../tensorcast/global_store/README.md)
- **Working on Store Daemon?** → See [Store Daemon Architecture](../../daemon/README.md)
- **Need HA features?** → Read [High Availability Design](./high-availability-design.md)
- **Optimizing transfers?** → Check [P2P Transfer Strategies](./p2p-transfer-strategies.md)
