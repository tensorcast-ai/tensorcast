---
title: Developer Guides
description: Comprehensive guides for developers working on or extending StepCast Store
sidebar_position: 4
hide_title: true
---

# Developer Guides

This section contains comprehensive guides for developers working on or extending StepCast Store.

## 🏗️ System Architecture

Understand how StepCast Store works:

- **[Component Interactions](architecture/README.md)** - How components work together
- **[High Availability Design](architecture/high-availability-design.md)** - HA architecture deep-dive

## 🔧 Component Development

Guides for developing specific components:

- **[Global Store Development](architecture/global-store.md)** - Developing the Global Store service
- **[Store Daemon Development](architecture/store-daemon.md)** - Store Daemon internals and development
- **[Adding New Metrics](internals/adding-metrics.md)** - How to create and expose metrics

## ⚙️ Core Modules (C++)

Deep-dive into the high-performance C++ core:

### Checkpoint Module
- **[Overview](core/checkpoint/README.md)** - Introduction to the checkpoint system
- **[Architecture](core/checkpoint/architecture.md)** - Detailed architecture design
- **[Data Format](core/checkpoint/data-format.md)** - Binary data format specification
- **[Verification Integration](core/checkpoint/verification-integration.md)** - Model integrity verification

### Store Module
- **[Overview](core/store/overview.md)** - Introduction to the core store system
- **[Architecture](core/store/architecture.md)** - Detailed architecture design
- **[Performance Guide](core/store/performance.md)** - Performance optimization guide
- **[State Management](core/store/state-management.md)** - Memory state management

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

# libscstore.so used for Python (DEBUG + PRE_CXX_ABI)
bazel build //core:libscstore.so --compilation_mode=opt --config=linux

# libscstore.so used for Server (DEBUG + CXX_ABI)
bazel build //core:libscstore.so --compilation_mode=dbg --config=linux

# Build tests
bazel build //tests/cpp:gpu_ce_test
bazel build //tests/cpp:model_p2p_registration_test --compilation_mode=dbg
bazel build //tests/cpp:model_p2p_transfer_test --compilation_mode=dbg
```

## Development Topics by Area

| Area | Key Documents |
|------|---------------|
| **Architecture** | [Component Interactions](architecture/README.md), [HA Design](architecture/high-availability-design.md) |
| **Python Services** | [Global Store Development](architecture/global-store.md), [Store Daemon Development](architecture/store-daemon.md) |
| **C++ Core** | [Checkpoint Architecture](core/checkpoint/architecture.md), [Store Architecture](core/store/architecture.md) |
| **Data Integrity** | [Verification Integration](core/checkpoint/verification-integration.md) |

