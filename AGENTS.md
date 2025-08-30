# AGENTS.md

This file provides guidance to AI when working with code in this repository.

## Project Overview

TensorCast is a high-performance, distributed artifact storage and loading system for machine learning inference and training. It follows a **distributed master-worker architecture** with:

## Development Environment Setup

### Build Commands

#### C++ Core (Bazel) and Python Extension
```bash
# Build all. BUILD_CORE means cxx files in core/
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
The project supports a fake CUDA backend for development and testing without GPU hardware:

```bash
# Build with fake CUDA using environment variable
USE_FAKE_CUDA=1 BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext

# Run C++ tests with fake CUDA backend
# Example for specific component tests:
bazel test //core/store:store_engine_test --define use_fake_cuda=true
bazel test //core/communicator/engine:gpu_ce_test --define use_fake_cuda=true
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
uv run mypy ./tensorcast
```

### Protocol Buffer Code Generation

**CRITICAL**: When modifying any `.proto` files, you MUST regenerate the protocol buffer code:
```bash
bash tools/build_proto_python.sh
```
This updates generated Python code in `./tensorcast/proto/` directory.

### Common Build Issues

1. **Protocol buffer changes not reflected**: Always run `bash tools/build_proto_python.sh` after modifying `.proto` files
2. **C++ changes not visible in Python**: Ensure both `BUILD_CORE=1` and `BUILD_EXTENSION=1` are set
3. **Clean build needed**: Run `bazel clean --expunge` and `rm -rf build/` for a complete clean build


## Additional Development Resources

### Cursor Rules Directory
The `.cursor/rules/` directory contains detailed guidelines for specific aspects of development:
- `architecture.mdc`: Detailed system architecture and design patterns
- `build.mdc`: Build system reference and troubleshooting
- `code_review.mdc`: Comprehensive code review checklist
- `common.mdc`: Software design principles and best practices
- `cpp.mdc`: C++ specific coding standards
- `proto.mdc`: Protocol buffer guidelines
- `python.mdc`: Python specific coding standards
- `technical_design.mdc`: Technical design document templates

## Architecture Overview

### Core Components
- **C++ Core** (`/core/`): High-performance checkpoint, store, and communicator modules with CUDA/P2P support
- **Python Services** (`/tensorcast/`): gRPC-based global store and client libraries with PyTorch integration. The Store Daemon is now implemented in C++ (see `/daemon`), while the Global Store and Python clients remain in Python.
- **Protocol Buffers** (`/proto/`): Service definitions for distributed communication
- **User Process Worker**: Responsible for final artifact loading and utilization

### Key Directories
- `/core/checkpoint/`: Streaming tensor serialization with GPU-to-disk optimization
- `/core/store/`: Artifact lifecycle management and memory optimization. **The Store Engine used by ArtifactLoader/StoreDaemon is implemented in C++ at `core/store/store_engine.h` with Python bindings at `tensorcast/csrc/store_engine_py.cc`**.
- `/core/communicator/`: RDMA/TCP communication engines
- `/tensorcast/global_store/`: Centralized metadata registry with DuckDB backend
- `/daemon/`: C++ StoreDaemon service (Python layer provides CLI/manager only)
- `/tests/`: Comprehensive C++ and Python test suites

### Build Systems
- **Primary**: Bazel with MODULE.bazel (Bzlmod) for C++ core
- **Secondary**: setuptools + uv for Python packaging
- **Dependencies**: LibTorch 2.6.0/2.7.0, CUDA 12.6+, gRPC, Protocol Buffers

### Distributed System Architecture
```
                          ┌─────────────────┐
                          │  Global Store   │
                          │   (Master)      │
                          │                 │
                          │ - Artifact Registry│
                          │ - Load Balancer │
                          │ - P2P Coord.    │
                          └────────┬────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                      │
    ┌───────────▼───────────┐          ┌──────────────▼──────────┐
    │   Store Daemon #1     │          │    Store Daemon #N      │
    │     (Worker)          │   ...    │      (Worker)           │
    │                       │          │                         │
    │ - Local Artifact Storage │          │ - Local Artifact Storage   │
    │ - CPU/GPU Memory      │          │ - CPU/GPU Memory        │
    │ - P2P Server          │          │ - P2P Server            │
    └───────────┬───────────┘          └──────────────┬──────────┘
                │                                      │
    ┌───────────▼───────────┐          ┌──────────────▼──────────┐
    │  User Process Worker  │          │  User Process Worker    │
    │                       │          │                         │
    │ - Artifact Inference     │          │ - Artifact Inference       │
    │ - Training Workloads  │          │ - Training Workloads    │
    │ - PyTorch Integration │          │ - PyTorch Integration   │
    │ - Artifact Access via    │          │ - Artifact Access via      │
    │   torch_util.py       │          │   torch_util.py         │
    └───────────────────────┘          └─────────────────────────┘
          Node 1                              Node N
```

### System Layer Responsibilities

#### Global Store (Control Plane)
- Centralized artifact registry and metadata management
- Load balancing across available replicas
- P2P transport coordination between nodes
- Worker health monitoring and failover

#### Store Daemon (Storage Layer)
- Local artifact storage and memory management
- CPU/GPU memory pool allocation
- P2P server for peer-to-peer transfers (RDMA or TCP)
- Artifact loading from disk or remote nodes
- Registration of local replicas with Global Store

#### User Process Worker (Application Layer)
- **Primary Interface**: Uses `tensorcast.torch_util.py` for artifact operations
- **Artifact Loading**: Calls `load_dict()` to load models from Store Daemon
- **Artifact Usage**: Runs inference, training, and other ML workloads
- **PyTorch Integration**: Direct tensor operations and artifact manipulation
- **Memory Access**: Accesses models in Store Daemon's memory pools
- **Lifecycle Management**: Triggers artifact confirmation and cleanup via Store Daemon

### Artifact Loading Workflows

#### P2P-Enabled Loading (Preferred)
```
Store Daemon                        Global Store
     |                                   |
     |-------- GetArtifactInfoById ----->|
     |<------- Available Replicas -------|
     |                                   |
     |-- RequestReplicaTransport ------->|
     |<------ Transport Info & ID -------|
     |                                   |
     |   (Load via RDMA/TCP from peer)   |
     |                                   |
     |-- CompleteReplicaTransport ------>|
     |<------ Confirmation --------------|
     |                                   |
     |-- RegisterReplica --------------->|
     |<-------- Replica ID --------------|
```

#### Fallback to Disk Loading
```
Store Daemon                    Global Store
     |                               |
     |   (Load from local disk)      |
     |                               |
     |-- RegisterReplica ---------->|
     |<------ Replica ID ------------|
```

### Load Balancing & Concurrency
- Global Store prioritizes replicas by: **GPU > RAM > DISK**, then by load ratio
- Each replica tracks `max_concurrency` and `current_requests`
- Atomic operations ensure thread-safe request allocation
- Transport requests automatically select optimal available replica


## Documentation Guidelines

### Docusaurus Configuration Updates

**CRITICAL**: When adding, removing, or moving documentation files, you MUST update the Docusaurus configuration files:

1. **Update `web-docs/sidebars.ts`**: Add/remove/move entries in the appropriate sidebar section
2. **Verify paths**: Ensure all file paths in sidebars match actual file locations
3. **Test navigation**: Confirm that all links work after changes
4. **Consider structure**: Place user-focused content in user-guides, developer-focused content in developer-guides

Example sidebar update:
```typescript
// Adding new user guide
'user-guides/new-feature-guide',

// Moving from user-guides to developer-guides
'developer-guides/internals/technical-workflow',
```

### Documentation Organization

- **User Guides** (`web-docs/docs/user-guides/`): End-user focused, practical guides for using the system
- **Developer Guides** (`web-docs/docs/developer-guides/`): Technical implementation details, architecture, and development workflows
- **Reference** (`web-docs/docs/reference/`): API documentation, troubleshooting, and reference materials

### Technical Design and RFCs

Technical design documents and RFCs are now unified in the `rfcs/` directory using the same naming convention.

**Filename Format**: `NNNN-feature-name.md` (e.g., `0001-distributed-virtual-memory-pool.md`)

**Required Sections**:
- Problem Statement
- Solution Overview
- Technical Details (Implementation steps)
- Testing Strategy
- Rollout Plan
- Progress Tracking (using standardized table format)

## Coding Standards

### C++ Guidelines (Simplified)

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
- **Logging**: Use `LOG` for logging, `CHECK` for assertions, `VLOG` for verbose logging, `PLOG` for error messages with `errno`
- Statement should be inside braces

#### Best Practices
- **Error Handling**: Use `absl::Status` or `absl::StatusOr<T>`; `absl::ErrnoToStatus` to convert errno to status
- **Testing**: Catch2 framework with Arrange-Act-Assert pattern
- **Concurrency**: Use absl thread annotations (`ABSL_GUARDED_BY`, etc.)
- **Documentation**: Doxygen style for public APIs

### Python Guidelines

#### Development Environment
- use `uv run xxx.py` to run python scripts instead of `python xxx.py`
- use `uv run pytest tests/python/xxxx` to run python tests
- use `bazel test //core/component:xxx_test` to run cxx tests (e.g., `bazel test //core/store:store_engine_test`)

#### Core Principles
- **Functional programming**: Prefer functional, declarative style over classes where possible
- **Type safety**: Always use type hints for function signatures
- **Modern Python**: Use Python 3.10+ features freely

#### Type Safety Requirements
- Use type hints for ALL function parameters and return values
- Prefer `T | None` over `Optional[T]`, `dict[T]` over `Dict[T]`
- Avoid dynamic attribute access (`getattr`, `hasattr`, `setattr`)
- Avoid Optional types unless necessary
- Avoid `isinstance` checks unless truly necessary
- Use Pydantic models over raw dictionaries for input validation

#### Error Handling
- Use `logger.exception()` instead of `logger.error()` for exceptions with traceback
- Handle errors early with guard clauses and early returns
- Provide meaningful error messages
- Prefer `contextlib.suppress` over `try/except` for suppressing exceptions


## Development Principles

### Software Design Philosophy

#### Complexity Reduction
- **Strategic Programming**: Work for the team and future maintainers, not just yourself
- **Deep Modules**: Create modules with simple interfaces but powerful functionality
- **Minimize Optional Types**: Avoid Optional[T] and Result[T, E] unless absolutely necessary
- **Layer Architecture**: Each layer should only import from layers below it
- **Information Hiding**: Keep implementation details private, expose only what's necessary

#### Comment-First Development
1. Write the interface comment first
2. Write the function signature
3. Write the implementation
4. Comments should describe the "what" and "why", not the "how"

#### Error Handling Philosophy
- Define errors out of existence when possible
- Use exceptions for exceptional cases
- Make common errors impossible through API design
- Avoid partial failures - operations should be atomic

### Code Fixing and Testing
- When fixing tests, always first understand the actual functionality of the test and the code behind that functionality. Make the test match the functionality, rather than making the functionality match the original test (while also ensuring the functionality is reasonable)
# Repository Guidelines

## Project Structure & Module Organization
- core/: C++20 core (checkpoint, store engine, communicator). Bazel-first builds.
- tensorcast/: Python services and client libs (gRPC, PyTorch integration).
- daemon/: C++ StoreDaemon service (launched by Python CLI).
- proto/: Protocol Buffers used across components.
- tests/: C++ and Python tests; C++ tests live next to code, Python under tests/python/.
- rfcs/: Technical designs (e.g., unified stager, staged P2P).

## Build, Test, and Development Commands
- Build C++ core + Python extension:
  ```bash
  BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
  ```
- Develop without GPUs (fake CUDA):
  ```bash
  USE_FAKE_CUDA=1 BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
  bazel test //core/communicator:tcp_engine_test --define use_fake_cuda=true
  ```
- Run C++ tests (examples):
  ```bash
  bazel test //core/store:store_engine_test
  bazel test //core/store/loader:disk_loader_streaming_buffer_test
  ```
- Run Python tests and quality:
  ```bash
  uv run pytest tests/python
  uv run ruff check . && uv run ruff format .
  uv run mypy ./tensorcast
  ```
- Regenerate protobufs after .proto changes:
  ```bash
  bash tools/build_proto_python.sh
  ```

## Coding Style & Naming Conventions
- C++: C++20, Bazel with sc_cc_library targets; naming — snake_case (vars/funcs), PascalCase (types), ALL_CAPS (consts/macros). Prefer early returns, RAII, absl::Status/StatusOr; logging via LOG/CHECK/VLOG.
- Python: 3.10+, type hints required; prefer functional style; use Pydantic models for validation; avoid dynamic getattr/hasattr.
- Formatting: use clang-tidy for C++ and ruff for Python. Keep changes minimal and consistent with neighbors.

## Testing Guidelines
- Frameworks: Catch2 (C++) and pytest (Python).
- Placement: C++ tests beside sources; Python under tests/python/ with test_*.py names.
- Running: prefer Bazel for C++ (see above) and uv run pytest for Python. Use fake CUDA for GPU paths when no hardware is available.

## Commit & Pull Request Guidelines
- Commits: concise subject, imperative mood; explain why and what. Scope changes to a single concern.
- PRs: include description, linked issues, test plan (commands + results), and impact notes (e.g., proto changes require running tools/build_proto_python.sh). Update docs (web-docs/sidebars.ts) if moving/adding docs.
