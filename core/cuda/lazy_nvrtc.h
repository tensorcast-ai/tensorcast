// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <nvrtc.h>

#include <mutex>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/cuda/dynamic_library.h"

namespace tensorcast::cuda {

#define TENSORCAST_NVRTC_SYMBOLS(X) \
  X(nvrtcVersion)                   \
  X(nvrtcCreateProgram)             \
  X(nvrtcCompileProgram)            \
  X(nvrtcGetPTXSize)                \
  X(nvrtcGetPTX)                    \
  X(nvrtcGetProgramLogSize)         \
  X(nvrtcGetProgramLog)             \
  X(nvrtcDestroyProgram)            \
  X(nvrtcGetErrorString)

struct NvrtcApi {
#define TENSORCAST_DECLARE_NVRTC_SYMBOL(name) decltype(&::name) name = nullptr;
  TENSORCAST_NVRTC_SYMBOLS(TENSORCAST_DECLARE_NVRTC_SYMBOL)
#undef TENSORCAST_DECLARE_NVRTC_SYMBOL
};

class LazyNvrtc {
 public:
  static LazyNvrtc& get();

  absl::Status ensure_loaded();

#define TENSORCAST_DECLARE_NVRTC_ACCESSOR(name) absl::StatusOr<decltype(&::name)> name();
  TENSORCAST_NVRTC_SYMBOLS(TENSORCAST_DECLARE_NVRTC_ACCESSOR)
#undef TENSORCAST_DECLARE_NVRTC_ACCESSOR

 private:
  absl::Status open_library();

  template <typename Fn>
  absl::StatusOr<Fn> resolve_symbol(const char* symbol_name, Fn* slot) {
    absl::Status status = ensure_loaded();
    if (!status.ok()) {
      return status;
    }

    {
      absl::MutexLock lock(&mutex_);
      if (slot != nullptr && *slot != nullptr) {
        return *slot;
      }
      if (!library_.has_value()) {
        return absl::FailedPreconditionError("NVRTC library not loaded");
      }
      auto symbol_or = library_->resolve_symbol(symbol_name);
      if (!symbol_or.ok()) {
        return symbol_or.status();
      }
      *slot = reinterpret_cast<Fn>(symbol_or.value());
      return *slot;
    }
  }

  absl::Mutex mutex_;
  std::once_flag load_once_;
  absl::Status library_status_ ABSL_GUARDED_BY(mutex_) = absl::OkStatus();
  std::optional<DynamicLibrary> library_ ABSL_GUARDED_BY(mutex_);
  NvrtcApi api_ ABSL_GUARDED_BY(mutex_);
};

} // namespace tensorcast::cuda
