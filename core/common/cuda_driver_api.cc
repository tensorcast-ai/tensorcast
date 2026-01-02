// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/cuda_driver_api.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <mutex>

#include "absl/base/no_destructor.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::cuda {
namespace {

static_assert(CUDA_VERSION >= 12040, "TensorCast requires CUDA Toolkit 12.4+");

std::atomic<bool> g_driver_api_load_attempted{false};

absl::Status resolve_driver_symbol(const char* symbol_name, void** fn_ptr) {
  if (fn_ptr == nullptr) {
    return absl::InvalidArgumentError("driver function pointer is null");
  }

  cudaDriverEntryPointQueryResult driver_status = cudaDriverEntryPointSuccess;
  const cudaError_t status = cudaGetDriverEntryPoint(symbol_name, fn_ptr, cudaEnableDefault, &driver_status);
  if (status != cudaSuccess) {
    return absl::UnavailableError(
        absl::StrCat("cudaGetDriverEntryPoint(", symbol_name, ") failed: ", cudaGetErrorString(status)));
  }
  if (driver_status != cudaDriverEntryPointSuccess || *fn_ptr == nullptr) {
    return absl::UnavailableError(absl::StrCat("missing CUDA driver entrypoint: ", symbol_name));
  }
  return absl::OkStatus();
}

DriverApi& driver_api_singleton() {
  static absl::NoDestructor<DriverApi> api;
  return *api;
}

} // namespace

absl::Status DriverApi::ensure_loaded() {
  static std::once_flag once;
  static absl::Status load_status = absl::OkStatus();
  std::call_once(once, [&]() { load_status = driver_api_singleton().load(); });
  return load_status;
}

const DriverApi& DriverApi::get() {
  ABSL_CHECK_OK(ensure_loaded());
  return driver_api_singleton();
}

bool DriverApi::load_attempted_for_testing() {
  return g_driver_api_load_attempted.load(std::memory_order_relaxed);
}

absl::Status DriverApi::to_status(CUresult result, absl::string_view context) const {
  if (result == CUDA_SUCCESS) {
    return absl::OkStatus();
  }
  const char* name = "unknown";
  const char* desc = "unknown";
  if (cuGetErrorName != nullptr) {
    (void)cuGetErrorName(result, &name);
  }
  if (cuGetErrorString != nullptr) {
    (void)cuGetErrorString(result, &desc);
  }
  return absl::InternalError(absl::StrCat(context, " - ", name, ": ", desc));
}

absl::Status DriverApi::load() {
  g_driver_api_load_attempted.store(true, std::memory_order_relaxed);
#define TENSORCAST_RESOLVE_DRIVER_SYMBOL(name)                                           \
  do {                                                                                   \
    absl::Status status = resolve_driver_symbol(#name, reinterpret_cast<void**>(&name)); \
    if (!status.ok()) {                                                                  \
      return status;                                                                     \
    }                                                                                    \
  } while (0);

  TENSORCAST_DRIVER_API_SYMBOLS(TENSORCAST_RESOLVE_DRIVER_SYMBOL)

#undef TENSORCAST_RESOLVE_DRIVER_SYMBOL

  return to_status(cuInit(0), "cuInit");
}

} // namespace tensorcast::cuda
