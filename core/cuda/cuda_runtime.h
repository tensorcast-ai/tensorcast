// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include "core/cuda/cuda_stream.h"

namespace tensorcast::cuda {

class CudaRuntime {
 public:
  static CudaRuntime& instance();

  CudaRuntime() = default;
  CudaRuntime(const CudaRuntime&) = delete;
  CudaRuntime& operator=(const CudaRuntime&) = delete;
  CudaRuntime(CudaRuntime&&) = delete;
  CudaRuntime& operator=(CudaRuntime&&) = delete;

  CudaStreamPool& stream_pool() {
    return stream_pool_;
  }

 private:
  CudaStreamPool stream_pool_;
};

} // namespace tensorcast::cuda
