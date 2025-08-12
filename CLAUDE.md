# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StepCast Store is a high-performance, distributed model storage and loading system for machine learning inference and training. It follows a **distributed master-worker architecture** with:
- **One Global Store**: Centralized coordinator and model registry
- **Many Store Daemons**: Distributed worker nodes across the cluster
- **C++ Core**: High-performance checkpoint, storage, and P2P communication layer
- **User Process Worker**: Loads and uses the model in the final application context

## Development Environment Setup

### Build Commands

#### C++ Core (Bazel) and Python Extension
```bash
# Build all. BUILD_CORE means cxx files in core/
#  BUILD_EXTENSION means cxx files in scstore/csrc
# Always should run this command when you modify any cxx files and
# you want to test the changes in the python code.
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext

# Build and run tests
# Tests are now colocated with their implementation
# Examples:
bazel test //core/store:checkpoint_store_test
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
bazel test //core/store:checkpoint_store_test --define use_fake_cuda=true
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
uv run mypy ./scstore
```

### Protocol Buffer Code Generation

**CRITICAL**: When modifying any `.proto` files, you MUST regenerate the protocol buffer code:
```bash
bash tools/build_proto_python.sh
```
This updates generated Python code in `./scstore/proto/` directory.

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
- **Python Services** (`/scstore/`): gRPC-based global store and daemon services with PyTorch integration. **All gRPC service and client implementations are in Python**.
- **Protocol Buffers** (`/proto/`): Service definitions for distributed communication
- **User Process Worker**: Responsible for final model loading and utilization

### Key Directories
- `/core/checkpoint/`: Streaming tensor serialization with GPU-to-disk optimization
- `/core/store/`: Model lifecycle management and memory optimization. **The checkpoint store used by ModelLoader/StoreDaemon is implemented in C++ at `core/store/checkpoint_store.h` with Python bindings at `scstore/csrc/checkpoint_store_py.cc`**.
- `/core/communicator/`: RDMA/TCP communication engines
- `/scstore/global_store/`: Centralized metadata registry with DuckDB backend
- `/scstore/store_daemon/`: Local model storage service
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
                          │ - Model Registry│
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
    │ - Local Model Storage │          │ - Local Model Storage   │
    │ - CPU/GPU Memory      │          │ - CPU/GPU Memory        │
    │ - P2P Server          │          │ - P2P Server            │
    └───────────┬───────────┘          └──────────────┬──────────┘
                │                                      │
    ┌───────────▼───────────┐          ┌──────────────▼──────────┐
    │  User Process Worker  │          │  User Process Worker    │
    │                       │          │                         │
    │ - Model Inference     │          │ - Model Inference       │
    │ - Training Workloads  │          │ - Training Workloads    │
    │ - PyTorch Integration │          │ - PyTorch Integration   │
    │ - Model Access via    │          │ - Model Access via      │
    │   torch_util.py       │          │   torch_util.py         │
    └───────────────────────┘          └─────────────────────────┘
          Node 1                              Node N
```

### System Layer Responsibilities

#### Global Store (Control Plane)
- Centralized model registry and metadata management
- Load balancing across available replicas
- P2P transport coordination between nodes
- Worker health monitoring and failover

#### Store Daemon (Storage Layer)
- Local model storage and memory management
- CPU/GPU memory pool allocation
- P2P server for peer-to-peer transfers (RDMA or TCP)
- Model loading from disk or remote nodes
- Registration of local replicas with Global Store

#### User Process Worker (Application Layer)
- **Primary Interface**: Uses `scstore.torch_util.py` for model operations
- **Model Loading**: Calls `load_dict()` to load models from Store Daemon
- **Model Usage**: Runs inference, training, and other ML workloads
- **PyTorch Integration**: Direct tensor operations and model manipulation
- **Memory Access**: Accesses models in Store Daemon's memory pools
- **Lifecycle Management**: Triggers model confirmation and cleanup via Store Daemon

### Model Loading Workflows

#### P2P-Enabled Loading (Preferred)
```
Store Daemon                        Global Store
     |                                   |
     |-------- GetModelInfo- ----------->|
     |<------- Available Replicas -------|
     |                                   |
     |-- RequestModelReplicaTransport -->|
     |<------ Transport Info & ID -------|
     |                                   |
     |   (Load via RDMA/TCP from peer)   |
     |                                   |
     |-- CompleteModelReplicaTransport ->|
     |<------ Confirmation --------------|
     |                                   |
     |-- RegisterModelReplica ---------->|
     |<-------- Replica ID --------------|
```

#### Fallback to Disk Loading
```
Store Daemon                    Global Store
     |                               |
     |   (Load from local disk)      |
     |                               |
     |-- RegisterModelReplica ------>|
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

### Bazel BUILD Rules
- **One logical unit per target** - Each class or related functions group gets its own `cc_library`
- **Default private visibility** - Only expose true public APIs
- **Explicit file lists** - Never use `glob()`, list all files explicitly
- **Consistent naming** - Use `_lib` suffix for libraries, `_test` for tests, `_binary` for binaries
- Always use `sc_cc_library` and `sc_header_only_library` instead of `cc_library` (includes absl/log, absl/status, absl/status:statusor)

### Build & Dependencies
- **Language**: C++20 standard (No compatibility shims)
- **Build System**: Bazel with Clang18
- **Common Deps**: absl, catch2, nlohmann_json

#### Naming Conventions
- **Variables/Functions**: `snake_case`
- **Classes/Structs**: `PascalCase`
- **Constants/Macros**: `ALL_CAPS`
- **Files/Directories**: `snake_case`

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
- **Error Handling**: Use `absl::Status` or `absl::StatusOr<T>`
- **Testing**: Catch2 framework with Arrange-Act-Assert pattern
- **Concurrency**: Use absl thread annotations (`ABSL_GUARDED_BY`, etc.)
- **Documentation**: Doxygen style for public APIs

### Python Guidelines

#### Development Environment
- use `uv run xxx.py` to run python scripts instead of `python xxx.py`
- use `uv run pytest tests/python/xxxx` to run python tests
- use `bazel test //core/component:xxx_test` to run cxx tests (e.g., `bazel test //core/store:checkpoint_store_test`)

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