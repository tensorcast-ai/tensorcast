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
- **[Volatile Publication Subjects](designs/0103-volatile-publication-subjects-and-multi-replica-semantics.md)** - Publication-subject model for target-memory publish, modes, and multi-replica behavior
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
