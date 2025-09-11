# AGENTS.md

This file provides guidance to AI when working with code in this repository.

## Project Overview

TensorCast is a high-performance distributed artifact storage and loading system for machine learning inference and training. It uses a distributed master-worker architecture; see Architecture Overview below for details.

## Development Environment Setup

### Build Commands

#### C++ Core (Bazel) and Python Extension
```bash
# Build all. BUILD_CORE means cxx files in core/ and ./daemon
#  BUILD_EXTENSION means cxx files in tensorcast/csrc
# Always should run this command when you modify any cxx files and
# you want to test the changes in the python code.
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext

# Build and run tests
# Tests are now colocated with their implementation
# Examples:
bazel test //core/store:store_engine_test
bazel test //core/communicator/engine:tcp_engine_test
bazel test //core/store/loader:disk_loader_streaming_buffer_test
```

#### Fake CUDA Backend (Development Without GPU)
The project supports a fake CUDA backend for development and testing without GPU hardware. C++ tests default to the fake backend so they run on CPU‑only machines.

```bash
# Build Python extension with fake CUDA
USE_FAKE_CUDA=1 BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext

# Run C++ tests with fake CUDA backend
# Most tests already run with fake CUDA by default. You can also select explicitly with a single define:
bazel test //core/store:store_engine_test --define=use_fake_cuda=true
bazel test //core/communicator/engine:gpu_ce_test --define=use_fake_cuda=true

# To run with real CUDA instead of the default fake backend, pass:
bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=false
```

**Fake CUDA Mode Features:**
- Enables development and testing without GPU hardware
- Simulates 4 GPUs for testing multi-device scenarios
- All CUDA operations return successful status
- Memory allocations tracked but use CPU memory
- Zero overhead when using real CUDA backend
- Complete API coverage for all CUDA operations used in codebase

### Code Quality and Linting

#### C++ Code Formatting
```bash
uv run clang-tidy ./core/xxx.cc
```

#### Python Code Quality
```bash
# Run Python linting (After you write code, run this command to check if your code is correct)
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright ./tensorcast
uv run mypy ./tensorcast
```

### Protocol Buffer Code Generation

TensorCast uses Buf for proto management and Python code generation, with Bazel for C++.

**CRITICAL**: After modifying any `.proto` file, run:
```bash
# Generate Python stubs via Buf and build C++ headers via Bazel
bash tools/build_proto_python.sh
```

Buf helper targets:
```bash
# Format and rewrite .proto files in-place
bazel run @rules_buf_toolchains//:buf -- format ./proto -w

# Lint protos in //proto
bazel test //proto/... --test_output=streamed
```

Note: Communicator config proto package is `tensorcast.communicator` and is consumed directly by C++.
The daemon loads communicator config from a YAML/JSON file (see `--comm_config_path`).

### Common Build Issues

1. **Protocol buffer changes not reflected**: Always run `bash tools/build_proto_python.sh` after modifying `.proto` files
2. **C++ changes not visible in Python**: Ensure both `BUILD_CORE=1` and `BUILD_EXTENSION=1` are set
3. **Clean build needed**: Run `bazel clean --expunge` and `rm -rf build/` for a complete clean build

## Architecture Overview

### Core Components
- **C++ Core** (`/core/`): Store Engine, Checkpoint, and Communicator. The Store Engine provides DVMP/UMA memory model, replica lifecycle, loaders (disk and P2P), and CUDA IPC export for clients.
- **Store Daemon (C++)** (`/daemon`): Thin gRPC service over `StoreEngine` that manages sessions, PID refs, and transport locks. Binary target `//daemon:tensorcast_daemon` (also shipped with the Python wheel).
- **Global Store (Python)** (`/tensorcast/global_store`): Central metadata and coordination service backed by DuckDB; gRPC API, Prometheus metrics, optional Web UI.
- **Protocol Buffers** (`/proto/`): gRPC surfaces for daemon and control plane.
- **User Process Worker**: Client process consuming artifacts via `tensorcast.api` (e.g., `load_dict_sync()`), mapping CUDA IPC handles for zero‑copy GPU access.

### Build Systems
- **Primary**: Bazel (Bzlmod) for C++ Core and Daemon
- **Secondary**: setuptools + `uv` for Python packaging/clients
- **Dependencies**: LibTorch 2.6/2.7, CUDA 12.6+, gRPC, Protocol Buffers

### Runtime Topology
```
                          ┌─────────────────────────────┐
                          │         Global Store        │
                          │  (Metadata & Coordination)  │
                          └──────────────┬──────────────┘
                                         │ gRPC (metadata only)
                ┌────────────────────────┴────────────────────────┐
                │                                               │
    ┌───────────▼───────────┐                       ┌───────────▼───────────┐
    │   Store Daemon #1     │  RDMA/TCP (P2P data)  │    Store Daemon #N    │
    │  (C++ over StoreEngine)│<-------------------->│  (C++ over StoreEngine)│
    │ - DVMP/UMA memory     │                       │ - DVMP/UMA memory     │
    │ - Disk & P2P loaders  │                       │ - Disk & P2P loaders  │
    │ - CUDA IPC export     │                       │ - CUDA IPC export     │
    └───────────┬───────────┘                       └───────────┬───────────┘
                │                                               │
                │ CUDA IPC (GPU mapping)                        │
    ┌───────────▼───────────┐                       ┌───────────▼───────────┐
    │ User Process Worker   │                       │ User Process Worker   │
    │  (PyTorch client)     │                       │  (PyTorch client)     │
    └───────────────────────┘                       └───────────────────────┘
```

## Documentation Structure and Project Hierarchy

This repository’s documentation has been reorganized so that each module owns a focused `README.md`, with cross-cutting guides under `docs/`. Use the map below to navigate, and when you update any module, also update its corresponding documentation.
Note: When writing documentation, you may use Mermaid diagrams to illustrate flows, state machines, hierarchies, and architecture where appropriate.

### Repository-level docs

- [Project Quickstart](./README.md) — Top-level guide for setup, builds, tests, and running key services.
- [Developer Guides Index](./docs/README.md) — Entry point to component development, architecture, and workflows.
- [Architecture Index](./docs/architecture/README.md) — Architecture docs index and quick navigation by concern.
- [Architecture Overview](./docs/architecture/architecture-overview.md) — High-level component overview and interactions.
- [High Availability Design](./docs/architecture/high-availability-design.md) — HA design, recovery, and state synchronization details.
- [P2P Transfer Strategies](./docs/architecture/p2p-transfer-strategies.md) — P2P transfer strategies, load balancing, and performance notes.
- [Model Loading Internals](./docs/internals/model-loading.md) — Internal model loading flow and integration points.
- [Save Dict Flow](./docs/internals/save_dict_flow.md) — End-to-end save_dict data path and artifacts produced.
- [Adding Metrics](./docs/internals/adding-metrics.md) — How to add and expose new metrics.

### Modules (C++ core and services)

- [Store Engine](./core/store/README.md) — Internals of the C++ Store Engine (API surface, data paths, memory model, P2P orchestration).
  - [Store Engine Architecture](./core/store/docs/architecture.md) — Detailed engine architecture and component responsibilities.
  - [State Management](./core/store/docs/state-management.md) — UMA/DVMP state model and lifecycle.
  - [Device Manager](./core/store/docs/device-manager.md) — Device discovery, UUID/ordinal mapping, and per-device state.
  - [Device Registry](./core/store/docs/device-registry.md) — Replica/device registry structures and indexing.

- [Checkpoint Module](./core/checkpoint/README.md) — Overview and streaming save/restore.
  - [Checkpoint Architecture](./core/checkpoint/docs/architecture.md) — Detailed design and relationships.
  - [Checkpoint Data Format](./core/checkpoint/docs/data-format.md) — Binary file format and index schema.
  - [Verification Integration](./core/checkpoint/docs/verification-integration.md) — Integrity verification and integration paths.

- [Communicator](./core/communicator/README.md) — TCP/MTCP/RDMA data-movement engine internals.

- [Store Daemon](./daemon/README.md) — C++ Store Daemon architecture, gRPC surface, lifecycles, and flows.

- [Global Store](./tensorcast/global_store/README.md) — Control plane internals (layered architecture, data model, services, flows).
  - [Global Store Web UI](./tensorcast/global_store/webui_frontend/README.md) — Web UI features, development, and integration.

### Tests and examples

- [tests/python/README.md](./tests/python/README.md) — Python test layout and commands for running suites.

### Doc sync rule (required for agents)

- When you change any module code, you must also update its docs:
  - Update the module’s README.md and any linked docs under docs/ that describe behavior you changed.
  - Keep links consistent across docs/architecture, docs/internals, and module sub-docs.
  - If you modify Protocol Buffers, also regenerate code as described in this file under “Protocol Buffer Code Generation”.
  - In PRs, include doc updates in the same change set so readers can rely on documentation being current.

- When authoring any design or plan document, follow the repository’s documentation system specification in [0001-docs-system-reorg-design](./docs/designs/0001-docs-system-reorg-design.md) for required structure, metadata/frontmatter, and cross-linking. Use the templates defined there and maintain the 1:1 design↔plan linkage.


## Coding Standards

### Package & Directory Structure

- Directory layout must mirror package/namespace hierarchy.
- C++: Derive namespace from path (`a/b/c` -> `a::b::c`); open matching nested namespaces. In the innermost namespace, prefer unqualified local names; fully qualify only when crossing namespaces or to disambiguate. Do not use `using namespace` at file scope.
  - `/core/`: drop leading `core` when deriving (`core/a/b/c` -> `tensorcast::a::b::c`).
  - `/daemon/`: do NOT drop `daemon`; derive as `tensorcast::daemon::<subdirs>` (e.g., `daemon/a/b` -> `tensorcast::daemon::a::b`).
  - Tests (`*_test.cc`): allow narrowly scoped `using`; prefer `using a::b::Symbol;` over broad imports.
  - Qualification elision:
    - Most sources sit under outer `tensorcast`; inside it, omit the `tensorcast::` prefix (e.g., `communicator::misc::GB`).
    - You can always omit the outer `tensorcast::` qualifier inside `core/` and `daemon/`; for example, use `common::` rather than `tensorcast::common`.
    - `core/store/**` is `tensorcast::store`; inside these files, also omit `store::` and refer directly to sub-namespaces (consistent with `core/store/store_engine.cc`): `loading::ReplicaHandle`, `replica::Replica`, `components::DeviceManager`.
    - Fully qualify with `tensorcast::...` only when leaving the implicit root or to resolve ambiguity. Never add `using namespace` at file scope.
  - Examples:
    - Good: `loading::ReplicaHandle`, `replica::Replica::create(...)`, `communicator::misc::GB`
    - Bad: `tensorcast::store::loading::ReplicaHandle`, `tensorcast::store::replica::Replica::create(...)`, `tensorcast::communicator::misc::GB`
    - Good (inside daemon): `grpc::ServiceImpl`
    - Bad (inside daemon): `tensorcast::daemon::grpc::ServiceImpl`

### C++ Guidelines (Simplified)

#### Includes vs. Forward Declarations
- Avoid forward declarations; include the correct headers for any types used in headers and translation units.
- Follow include-what-you-use: include the minimal header that directly provides the symbols you reference.

#### C++ Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes/Structs**: `PascalCase`
- **Constants/Macros**: `ALL_CAPS`
- **Files/Directories**: `snake_case`

### Bazel BUILD Rules
- **One logical unit per target** - Each class or related functions group gets its own `cc_library`
- **Default private visibility** - Only expose true public APIs
- **Consistent naming** - Use `_lib` suffix for libraries, `_test` for tests, `_binary` for binaries
- Always use `sc_cc_library` and `sc_header_only_library` instead of `cc_library` (includes absl/log, absl/status, absl/status:statusor)

### Build & Dependencies
- **Language**: C++20 standard (No compatibility shims)
- **Build System**: Bazel with Clang18
- **Common Deps**: absl, catch2, nlohmann_json

#### Code Style
- Prefer modern C++ features and language standards. No compatibility shims.
- **Functions**: Short (<20 lines), single purpose, verb-based names
- **Early Returns**: Use guard clauses to avoid nesting
- **Parameters**: Use structs for multiple params/returns
- **Immutability**: Prefer `const` and `constexpr`
- **RAII**: Always use for resource management
- **Logging**: Use `LOG` for logging, `VLOG` for verbose logging, `PLOG` for error messages with `errno`
- **Assertions**: Use `ABSL_CHECK` for assertions, `ABSL_CHECK_OK` for assertions that return statuses
- Statement should be inside braces

#### Best Practices
- **Error Handling**: Use `absl::Status` or `absl::StatusOr<T>`; `absl::ErrnoToStatus` to convert errno to status
- **Errno handling policy**: Avoid direct use of `errno`, `strerror`, and `perror`. Prefer `absl::ErrnoToStatus` for converting OS errors to statuses and `PLOG(...)` for logging errors that include `errno` safely.

```cpp
// Bad
int fd = open(path.c_str(), O_RDONLY);
if (fd < 0) {
  LOG(ERROR) << "open failed: " << strerror(errno);
  return absl::ErrnoToStatus(errno, "open failed");
}

// Good
int fd = open(path.c_str(), O_RDONLY);
if (fd < 0) {
  PLOG(ERROR) << "open failed";            // logs message with errno string
  return absl::ErrnoToStatus(errno, "open failed");
}
```
- **Testing**: Catch2 framework with Arrange-Act-Assert pattern
- **Concurrency**: Use absl thread annotations (`ABSL_GUARDED_BY`, etc.)
- **Documentation**: Doxygen style for public APIs
 - **Pointer annotations**: Prefer `std::unique_ptr` over `std::shared_ptr`. When using pointers, annotate intent with GSL: use `gsl::not_null<>` for non-null, non-owning pointers and `gsl::owner<>` for owning pointers. Constrain pointers at the type level wherever possible.

### Python Guidelines

#### Development Environment
- use `uv run xxx.py` to run python scripts instead of `python xxx.py`
- use `uv run pytest tests/python/xxxx` to run python tests
- use `bazel test //core/component:xxx_test` to run cxx tests (e.g., `bazel test //core/store:store_engine_test`)

#### Python Naming Conventions
- **Files/Directories**: `snake_case` (e.g., `tensorcast/global_store/manager.py`)
- **Variables/Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `ALL_CAPS`

#### Core Principles
- **Functional programming**: Prefer functional, declarative style over classes where possible
- **Type safety**:
  - Try your best to make the code type safe
  - Always use type hints for function signatures
- **Modern Python**: Use Python 3.10+ features freely

#### Type Safety Requirements
- Use type hints for ALL function parameters and return values
- Prefer `T | None` over `Optional[T]`, `dict[T]` over `Dict[T]`
- Avoid dynamic attribute access (`getattr`, `hasattr`, `setattr`)
- Avoid Optional types unless necessary
- Avoid `isinstance` checks unless truly necessary; rely on precise type hints and intentional polymorphism instead of runtime type checks
- Prefer Pydantic models over raw dictionaries for any use cases

#### Error Handling
- Use `logger.exception()` instead of `logger.error()` for exceptions with traceback
- Handle errors early with guard clauses and early returns
- Provide meaningful error messages
- Prefer `contextlib.suppress` over `try/except` for suppressing exceptions

#### Dependencies & Tools
- **Package Management**: `uv` (MUST)
- **Testing**: `pytest` in `tests/python/`; run with `uv run pytest tests/python/...`
- **Type Checking & Linting**: `mypy` and `ruff`
  - `uv run mypy ./tensorcast`
  - `uv run ruff check .` and `uv run ruff format .`
- **Data Modeling**: Prefer Pydantic models over raw dictionaries for validation

#### Protobuf Imports
- Import generated protobuf modules using the canonical package path `tensorcast.proto.<pkg>.v1`.
- Example: `from tensorcast.proto.config.v1 import client_config_pb2`

#### Best Practices
- Prefer decomposition and iteration over duplication
- Keep functions small and focused
- Use explicit, named imports for utilities
- Use `dataclasses` or Pydantic models for complex data structures
- Use absolute imports for in-repo modules: `from tensorcast.xxx.yyy import Z`; avoid relative imports like `from .xxx import Z`


## Development Principles

### Code Fixing and Testing
- When debugging or fixing tests—or investigating/fixing any issue—first identify the root cause and implement the fundamental solution. Make tests reflect the real, reasonable system behavior rather than bending the system to fit an original test, and avoid quick or convenient workarounds. Solve from a system-wide perspective, aiming for globally optimal changes, not local patches.
