// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cuda.h>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace tensorcast::cuda {

#define TENSORCAST_DRIVER_API_SYMBOLS(X) \
  X(cuInit)                              \
  X(cuDeviceGet)                         \
  X(cuDeviceGetAttribute)                \
  X(cuMemGetAddressRange)                \
  X(cuMemGetAllocationGranularity)       \
  X(cuMemAddressReserve)                 \
  X(cuMemCreate)                         \
  X(cuMemMap)                            \
  X(cuMemSetAccess)                      \
  X(cuMemUnmap)                          \
  X(cuMemRelease)                        \
  X(cuMemAddressFree)                    \
  X(cuMemGetHandleForAddressRange)       \
  X(cuModuleLoadData)                    \
  X(cuModuleGetFunction)                 \
  X(cuLaunchKernel)                      \
  X(cuModuleUnload)                      \
  X(cuGetErrorName)                      \
  X(cuGetErrorString)

class DriverApi {
 public:
  static absl::Status ensure_loaded();
  static const DriverApi& get();
  // For tests: indicates whether driver loading has been attempted in-process.
  static bool load_attempted_for_testing();

  absl::Status to_status(CUresult result, absl::string_view context) const;

#define TENSORCAST_DECLARE_DRIVER_SYMBOL(name) decltype(&::name) name = nullptr;
  TENSORCAST_DRIVER_API_SYMBOLS(TENSORCAST_DECLARE_DRIVER_SYMBOL)
#undef TENSORCAST_DECLARE_DRIVER_SYMBOL

 private:
  absl::Status load();
};

} // namespace tensorcast::cuda
