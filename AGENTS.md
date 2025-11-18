# AGENTS.md

This file provides guidance to AI when working with code in this repository.

## Project Overview

TensorCast is a high-performance distributed artifact storage and loading system. It uses a distributed master-worker architecture; see Architecture Overview below for details.

## Command Execution

- When running any Python script or module, you MUST use `uv run <script>`.
- Never invoke `python`, `python3`, or `python -m` directly; this applies to ad-hoc scripts, tooling helpers, tests, `setup.py`, and all other Python entry points.
- You must use `uv run pytest tests/python/xxxx` to run python tests
- You must use `bazel test //core/component:xxx_test` to run cxx tests
- These policies keep virtualenv isolation consistent; violating them can break the build and introduce environment skew.

## Architecture Overview

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
    │ - VS/UMA memory       │                       │ - VS/UMA memory       │
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

### Core Components
- **C++ Core** (`/core/`): Store Engine, Checkpoint, and Communicator. The Store Engine uses a UnifiedMemoryAuthority (UMA) single-ledger memory model, replica lifecycle helpers, loaders (disk and P2P), and CUDA IPC export for clients.
- **Store Daemon (C++)** (`/daemon`): Thin gRPC service over `StoreEngine` that manages sessions, PID refs, and transport locks. Binary target `//daemon:tensorcast_daemon` (also shipped with the Python wheel).
- **Global Store (Python)** (`/tensorcast/global_store`): Central metadata and coordination service backed by DuckDB; exposes gRPC APIs and Prometheus metrics.
- **Protocol Buffers** (`/proto/`): gRPC surfaces for daemon and control plane.
- **User Process Worker**: Client process consuming artifacts via `tensorcast` APIs (e.g., `tensorcast.get()`), mapping CUDA IPC handles for zero‑copy GPU access.

### Build Systems
- **Primary**: Bazel (Bzlmod) for C++ Core and Daemon
- **Secondary**: setuptools + `uv` for Python packaging/clients
- **Dependencies**: LibTorch 2.6/2.7, CUDA 12.6+, gRPC, Protocol Buffers

## Documentation Structure and Project Hierarchy

This repository’s documentation has been organized so that each module owns a focused `README.md`, with cross-cutting guides under `docs/`. Use the map below to navigate, and when you update any module, also update its corresponding documentation.
Note: When writing documentation, you may use Mermaid diagrams to illustrate flows, state machines, hierarchies, and architecture where appropriate.

### Repository-level docs

- ./README.md — Top-level guide for setup, builds, tests, and running key services.
- ./docs/README.md — Entry point to component development, architecture, and workflows.
- ./docs/architecture/README.md — Architecture docs index and quick navigation by concern.
- ./docs/architecture/architecture-overview.md — High-level component overview and interactions.
- ./docs/architecture/high-availability-design.md — HA design, recovery, and state synchronization details.
- ./docs/architecture/p2p-transfer-strategies.md — P2P transfer strategies, load balancing, and performance notes.
- ./docs/internals/model-loading.md — Internal model loading flow and integration points.
- ./docs/internals/save_dict_flow.md — End-to-end save_dict data path and artifacts produced.
- ./docs/internals/adding-metrics.md — How to add and expose new metrics.

### Modules (C++ core and services, tests)

- ./core/store/README.md — Internals of the C++ Store Engine (API surface, data paths, memory model, P2P orchestration).
  - ./core/store/docs/architecture.md — Detailed engine architecture and component responsibilities.
  - ./core/store/docs/state-management.md — UMA/VS state model and lifecycle.
  - ./core/store/docs/device-manager.md — Device discovery, UUID/ordinal mapping, and per-device state.
  - ./core/store/docs/device-registry.md — Replica/device registry structures and indexing.
- ./core/checkpoint/README.md — Overview and streaming save/restore.
  - ./core/checkpoint/docs/architecture.md — Detailed design and relationships.
  - ./core/checkpoint/docs/data-format.md — Binary file format and index schema.
  - ./core/checkpoint/docs/verification-integration.md — Integrity verification and integration paths.
- ./core/communicator/README.md — TCP/MTCP/RDMA data-movement engine internals.
- ./daemon/README.md — C++ Store Daemon architecture, gRPC surface, lifecycles, and flows.
- ./tensorcast/global_store/README.md — Control plane internals (layered architecture, data model, services, flows).
- tests/python/README.md — Python test layout and commands for running suites.

### Doc sync rule (required for agents)

- When you change any module code, you must also update its docs:
  - Update the module’s README.md and any linked docs under docs/ that describe behavior you changed.
  - Keep links consistent across docs/architecture, docs/internals, and module sub-docs.
  - If you modify Protocol Buffers, also regenerate code as described in this file under “Protocol Buffer Code Generation”.
  - In PRs, include doc updates in the same change set so readers can rely on documentation being current.

- When authoring any design or plan document, follow the repository’s documentation system specification in ./docs/designs/0001-docs-system-design.md for required structure, metadata/frontmatter, and cross-linking. Use the templates defined there and maintain the 1:1 design↔plan linkage.
- `docs/designs/` and `docs/plans/` filenames begin with a zero-padded sequence number: `0001-<slug>.md`, `0002-<slug>.md`, etc.

## Platform Assumptions

- Supported OS: Linux only
- Baseline kernel: Linux 5.10.x (no need to support versions < 5.10)

## Development Environment Setup

### Build Commands

#### C++ Core (Bazel) and Python Extension
```bash
# Build all.
#  BUILD_CORE means cxx files in core/ and ./daemon
#  BUILD_EXTENSION means cxx files in tensorcast/csrc and the output is tensorcast._C
# Always should run this command when you modify any cxx files and
# you want to test the changes in the python code.
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext

# Build and run tests
# Tests are now colocated with their implementation
# Examples:
bazel test //core/store:store_engine_test
bazel test //core/communicator/engine:tcp_engine_test
bazel test //core/store/materialization/dataplane:disk_loader_streaming_buffer_test
```

#### Bazel Quiet / Reduced Output
Bazel does not support the `-q` parameter (do not use it). To reduce build/test log noise, use the following composable options:

```bash
# Build quietly: hide progress and loading info, only show warnings and errors
bazel build //daemon:tensorcast_daemon \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error

# Run tests with minimal output: only show failure details, reduce general logging
bazel test //daemon:session_lifecycle_test \
  --define=use_fake_cuda=true \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error
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
- Stream and event APIs now execute callbacks asynchronously on a lightweight worker so `AsyncCopyManager` and staged transfers behave like the real runtime
- All CUDA operations return successful status
- Memory allocations tracked but use CPU memory
- Zero overhead when using real CUDA backend
- Complete API coverage for all CUDA operations used in codebase

### Code Quality and Linting

#### C++ Code Formatting
```bash
# It's not necessary to add other arguments to the command, just run it directly
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

## Coding Standards

### Package & Directory Structure

- Mirror directories to namespaces/packages.
- C++: Derive namespaces from paths and open matching nested namespaces. Avoid file-scope `using namespace`.
  - `core/a/b/c` → `tensorcast::a::b::c` (drop `core`).
  - `daemon/a/b` → `tensorcast::daemon::a::b` (keep `daemon`).
  - Tests (`*_test.cc`): allow narrow `using` (e.g., `using a::b::Symbol;`).
- Qualification rules:
  - Inside `core/` and `daemon/`, omit the outer `tensorcast::` unless leaving the root or to disambiguate.
  - Inside `core/store/**`, treat it as `tensorcast::store` and also omit `store::`; refer directly to sub-namespaces (e.g., `loading::ReplicaHandle`).
- Examples:
  - Good: `loading::ReplicaHandle`, `replica::Replica::create(...)`, `communicator::misc::GB`, `grpc::ServiceImpl` (inside daemon).
  - Avoid: `tensorcast::store::loading::ReplicaHandle`, `tensorcast::store::replica::Replica::create(...)`, `tensorcast::communicator::misc::GB`, `tensorcast::daemon::grpc::ServiceImpl`.

### C++ Guidelines (Simplified)

#### Includes vs. Forward Declarations
- Avoid forward declarations; include the correct headers for any types used in headers and translation units.
- Follow include-what-you-use: include the minimal header that directly provides the symbols you reference.

#### UMA V3 canonical headers (required)
- Use canonical headers after UMA V3 final cutover. Legacy and alias headers are forbidden in new/modified code:
  - `core/common/memory/virtual_address_space.h` (canonical; DVMP removed)
  - `core/store/replica/unified_memory_authority.h` (canonical; alias under `core/store/uma/*` removed)
  - `core/store/replica/memory_export_registry.h` (canonical; replaces `chunk_export_service.h`)
- The helper script `tools/lint/check_uma_aliases.sh` enforces this policy (error-level) and fails on legacy includes.

#### C++ Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes/Structs**: `PascalCase`
- **Constants/Macros**: `ALL_CAPS`
- **Files/Directories**: `snake_case`

### Bazel BUILD Rules
- **One logical unit per target** - Each class or related functions group gets its own `cc_library`
- **Default private visibility** - Only expose true public APIs
- **Consistent naming** - Use `_lib` suffix for libraries, `_test` for tests, `_binary` for binaries
- Always use `sc_cc_library` and `sc_header_only_library` instead of `cc_library` (sc_cc_library has already included the common dependencies of absl/log, absl/status, absl/status:statusor)
- **Resolve missing headers via BUILD deps first** - When a header appears "missing", fix the Bazel BUILD dependencies by adding the precise library that exports the header to the target's `deps`. Do not rely on global include paths or accidental transitive includes; wire dependencies explicitly in BUILD files.

### Build & Dependencies
- **Language**: C++20 standard (No compatibility shims)
- **Build System**: Bazel with Clang18 (Dependencies are on MODULE.bazel)
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
- **Never ignore Status**: Do not discard `absl::Status`/`absl::StatusOr` results (avoid `(void)expr`). For invariants, assert with `ABSL_CHECK_OK(...)`. For recoverable or expected errors, handle/log/propagate them explicitly and continue safely when possible.
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

#### Logging (C++)
- `LOG(ERROR)`: Invariants/consistency risks or crashes (e.g., background task threw, checkpoint offset mismatch, fatal loop crashes).
- `LOG(WARNING)`: Recoverable but user‑visible failures (e.g., lease create/release failures, unload failures, GS sync/keepalive/remote‑access toggle failures).
- `LOG(INFO)`: One‑time milestones and service summaries (engine/comm init, server ready, successful registration summary).
- `VLOG(1)`: Flow‑level debug; avoid in hot loops.
- `VLOG(2+)`: Inner‑loop/high‑frequency traces only.
- `PLOG(level)`: Use when OS/syscalls fail and `errno` should be included.
- **Testing**: Catch2 framework with Arrange-Act-Assert pattern
- **Concurrency**: Use absl thread annotations (`ABSL_GUARDED_BY`, etc.)
- **Documentation**: Doxygen style for public APIs
 - **Pointer annotations**: Prefer `std::unique_ptr` over `std::shared_ptr`. When using pointers, annotate intent with GSL: use `gsl::not_null<>` for non-null, non-owning pointers and `gsl::owner<>` for owning pointers. Constrain pointers at the type level wherever possible.
 - **not_null dereference**: For `gsl::not_null<T*>` and `gsl::not_null<std::shared_ptr<T>>`, use `var->member` instead of `var.get()->member`; prefer `operator->` for readability.

### Python Guidelines

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
- Binding: Use absolute imports for all in-repo modules: `from tensorcast.xxx.yyy import Z`. NEVER use relative imports (`from .xxx import Z`, `from ..yyy import Z`).


### Commenting & Self-Documentation
- Use comments to capture what and why, not how. Focus on intent, invariants, and decision rationale; the code should express the implementation.
- Make code self-explanatory.


## Development Principles

### Code Fixing and Testing
- When debugging or fixing tests—or investigating/fixing any issue—first identify the root cause and implement the fundamental solution. Make tests reflect the real, reasonable system behavior rather than bending the system to fit an original test, and avoid quick or convenient workarounds. Never modify the system merely to make a test pass; find the root cause and design a fundamental fix, aiming for the minimal sufficient change. When a test must change, update it to assert the clearly documented, reasonable system behavior. Solve from a system-wide perspective, aiming for globally optimal changes, not local patches.
