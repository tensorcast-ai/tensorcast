// Copyright (c) 2025-2026, TensorCast Team.

#include "core/cuda/cuda_runtime.h"

#include "absl/base/no_destructor.h"

namespace tensorcast::cuda {

CudaRuntime& CudaRuntime::instance() {
  static absl::NoDestructor<CudaRuntime> runtime;
  return *runtime;
}

} // namespace tensorcast::cuda
