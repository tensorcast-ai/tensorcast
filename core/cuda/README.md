# `core/cuda`

CUDA backend and runtime utilities for TensorCast.

## Backend API

`core/cuda/cuda_api.h` defines the stable `tensorcast::cuda::*` API backed by the real and fake implementations
(`cuda_backend_real.cc`, `cuda_backend_fake.cc`) selected at runtime.

## Runtime facade

- `CudaDeviceGuard`/`OptionalCudaDeviceGuard` in `core/cuda/device_guard.h` provide RAII device switching.
- `CudaStreamPool`, `CudaStream`, and `CudaStreamGuard` in `core/cuda/cuda_stream.h` manage pooled streams and
  per-thread current stream state.
- `CudaEvent` in `core/cuda/cuda_event.h` wraps lazy event creation and stream-aware record/block/query.
- `CudaRuntime::instance()` in `core/cuda/cuda_runtime.h` exposes the shared stream pool.
- `core/cuda/error_handling.h` provides `TC_CUDA_*` macros for consistent error handling.

## `dynamic_library`

`core/cuda/dynamic_library.h` provides a small RAII wrapper around `dlopen(3)`/`dlclose(3)` and symbol lookup via
`dlsym(3)`.

- `DynamicLibrary::open(...)` loads a shared object with `RTLD_NOW | RTLD_LOCAL` and returns `absl::StatusOr`.
- `DynamicLibrary::resolve_symbol(...)` returns the symbol address (`void*`) or a descriptive `absl::Status` on failure.
