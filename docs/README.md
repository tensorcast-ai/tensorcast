---
title: Developer Guides
description: Comprehensive guides for developers working on or extending TensorCast
sidebar_position: 4
hide_title: true
---

# Developer Guides

This section contains comprehensive guides for developers working on or extending TensorCast.

## 🏗️ System Architecture

Understand how TensorCast works:

- **[Component Interactions](architecture/README.md)** - How components work together
- **[API Design](architecture/api/README.md)** - SDK surface and internal API flows
- **[Binding Unified Model](designs/0084-binding-unified-model-and-contract.md)** - Canonical binding semantics, lifecycle, mapped binding, and publishability
- **[Retrieval Policy Plane Cleanup](designs/0107-retrieval-policy-plane-cleanup.md)** - Separates retrieval policy from execution topology context and defines the normalized daemon request boundary
- **[Tensor-Aware Materialization Strategy Plane](designs/0108-tensor-aware-materialization-strategy-plane.md)** - Sole long-term strategy-plane owner for ordinary and source-bound execution planning, explicit lane allocation, and no-implicit-fallback runtime semantics
- **[Tensor-Aware Materialization Strategy Plane Plan](plans/0108-tensor-aware-materialization-strategy-plane.md)** - Active checklist for host-local local-executor convergence, ordinary-path cleanup, and shared strategy validation
- **[Composite Materialization and Vectored Direct-Write](designs/0115-composite-materialization-and-vectored-direct-write.md)** - Shared dataplane contract for composite source -> composite target execution plus routed vectored pull fast path below the strategy seam
- **[Composite Materialization and Vectored Direct-Write Plan](plans/0115-composite-materialization-and-vectored-direct-write.md)** - Implementation checklist for batched direct-write source contracts, communicator `read_plan(...)`, and RDMA vectored realization
- **[Batched Owner-File Collective Executor](designs/0109-batched-owner-file-collective-executor.md)** - Shared-FS same-host TP collective executor design for bounded owner batching and cross-rank source dedup
- **[Batched Owner-File Collective Rollout and Residual Policy Plan](plans/0109-01-batched-owner-file-collective-rollout-and-residual-policy.md)** - Active executor-specific rollout plan for mixed residual policy, shared-source evidence, and collective prototype cleanup
- **[Representation Semantic Core](designs/0110-artifact-representation-contract-and-transform-unification.md)** - Shared tensor-semantic transform core for mapped binding, mapped-target, builder lowering, and strategy convergence
- **[Source-to-Serving Builder and Publication](designs/0111-source-to-serving-builder-and-representation-publication.md)** - Builder modes, serving-artifact identity and manifests, `serving_build_digest`, and typed `representation_publish` handoff into `PublishedModelVersion`
- **[Binding-Native Serving Realization and Publication](designs/0112-binding-native-serving-realization-and-publication.md)** - Canonical owner for binding-native same-binding serving ingress, audited Step3p5 closure, and the shipped publication and closeout model
- **[Binding-Native Serving Mounted Rollout and Delete-Gate Cleanup Plan](plans/0112-01-binding-native-serving-mounted-rollout-and-delete-gate-cleanup.md)** - Active mounted rollout, operator-evidence, and delete-gate checklist for the binding-native serving path
- **[Assembly Attempt Seal Remediation Handoff Plan](plans/0105-01-assembly-attempt-cut-driven-seal-remediation-handoff.md)** - Ordered execution handoff for finishing cut-driven seal, contract-family enforcement, and validation closure
- **[Daemon-Served Directory and Target Resolution](designs/0106-daemon-served-directory-and-target-resolution.md)** - Stable worker or instance identity, bounded-staleness directory reads, and NodeAgentDirectory contract
- **[High Availability Design](architecture/high-availability-design.md)** - HA architecture deep-dive
- **[Artifact Views and Retrieval](architecture/artifact-views-and-retrieval.md)** - View semantics, planning, and materialization overview
- **[View Replicas and Assembly](architecture/view-replicas-and-assembly.md)** - Dense pieces, assembly, and sealing semantics

## 🧭 Coordination Strategy Series

Cross-cutting high-level reasoning for centralized vs decentralized coordination, hot/cold state placement, and workload-driven mode switching:

- **[Distributed Coordination Series](distributed-coordination-series/README.md)** - Index for the full strategy and trade-off document set


## 📘 User Guides

Integration-oriented guides for SDK consumers:

- **[SDK Startup User Guide](guides/sdk-startup-user-guide.md)** - Best practices for `tensorcast.init`, daemon lifecycle ownership, API/SDK startup modes, and TP/multi-process integration.

## 🔧 Component Development

Guides for developing specific components:

- **[Global Store Development](../tensorcast/global_store/README.md)** - Developing the Global Store service
- **[Store Daemon Development](../daemon/README.md)** - Store Daemon internals and development
- **[Adding New Metrics](internals/adding-metrics.md)** - How to create and expose metrics
- **[tensor_dict_into Dataflow](internals/tensor_dict_into_dataflow.md)** - Legacy vs region-backed into paths
- **[Disk Load Strategy](internals/disk-load-strategy.md)** - TP-aware disk loading decisions for local SSD, shared filesystems, and mapped target flows
- **[Preemptible Memory Internals](internals/preemptible-memory.md)** - UMA/VS preemption workflow and tuning
- **[Byte-Range Mapping and Execution](internals/byte-range-mapping-and-execution.md)** - Unified byte-range executor semantics
- **[Chaos Report Template](internals/chaos-report-template.md)** - Standard handoff template for multi-host chaos runs
- **[Chaos Gate Checklist](internals/chaos-gate-checklist.md)** - Phase gate review checklist and required artifacts

## ⚙️ Core Modules (C++)

Deep-dive into the high-performance C++ core:

### Checkpoint Module
- **[Overview](../core/checkpoint/README.md)** - Introduction to the checkpoint system
- **[Architecture](../core/checkpoint/docs/architecture.md)** - Detailed architecture design
- **[Data Format](../core/checkpoint/docs/data-format.md)** - Binary data format specification
- **[Verification Integration](../core/checkpoint/docs/verification-integration.md)** - Artifact integrity verification

### Store Module
- **[Overview](../core/store/README.md)** - Introduction to the core store system
- **[Architecture](../core/store/docs/architecture.md)** - Detailed architecture design
- **[State Management](../core/store/docs/state-management.md)** - Memory state management

## Development Workflow

### Build System
- **Primary**: Bazel with MODULE.bazel (Bzlmod) for C++ core
- **Secondary**: setuptools + uv for Python packaging

### Code Quality
```bash
# C++ formatting and linting
make format
make check-style
make tidy

# Python formatting and linting
ruff check .
ruff format .
mypy .
```

### Build targets
```bash
# Build core and Python extension together
BUILD_CORE=1 BUILD_EXTENSION=1 python setup.py develop

# libcheckpoint_ext.so used for tensorcast._C bindings (DEBUG + PRE_CXX_ABI)
bazel build //core:libcheckpoint_ext.so --compilation_mode=opt --config=linux

# Build tests
bazel build //tests/cpp:gpu_ce_test
bazel build //tests/cpp:replica_p2p_registration_test --compilation_mode=dbg
bazel build //tests/cpp:replica_p2p_transfer_test --compilation_mode=dbg
```

### Testing

- **[Testing Guide](development/testing.md)** - Python, C++, P2P, and multi-machine tests

## Development Topics by Area

| Area                | Key Documents                                                                                                                |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**    | [Component Interactions](architecture/README.md), [HA Design](architecture/high-availability-design.md)                      |
| **Python Services** | [Global Store Development](../tensorcast/global_store/README.md), [Store Daemon Development](../daemon/README.md)            |
| **C++ Core**        | [Checkpoint Architecture](../core/checkpoint/docs/architecture.md), [Store Architecture](../core/store/docs/architecture.md) |
| **Data Integrity**  | [Verification Integration](../core/checkpoint/docs/verification-integration.md)                                              |
