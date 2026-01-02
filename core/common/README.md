# `core/common`

Shared C++ utilities used across TensorCast core and services.

## `dynamic_library`

`core/common/dynamic_library.h` provides a small RAII wrapper around `dlopen(3)`/`dlclose(3)` and symbol lookup via
`dlsym(3)`.

- `DynamicLibrary::open(...)` loads a shared object with `RTLD_NOW | RTLD_LOCAL` and returns `absl::StatusOr`.
- `DynamicLibrary::resolve_symbol(...)` returns the symbol address (`void*`) or a descriptive `absl::Status` on failure.

