// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/device_guard.h"

namespace tensorcast::cuda {

class CudaStream {
 public:
  CudaStream() = default;

  CudaStream(cudaStream_t stream, int device_id) : stream_(stream), device_id_(device_id) {}

  cudaStream_t stream() const {
    return stream_;
  }

  int device_id() const {
    return device_id_;
  }

  bool is_valid() const {
    return device_id_ >= 0;
  }

  absl::Status synchronize() const {
    return stream_synchronize(stream_);
  }

 private:
  cudaStream_t stream_{nullptr};
  int device_id_{-1};
};

enum class CudaStreamPriority : int {
  kDefault = 0,
  kHigh = -1,
};

class CudaStreamPool {
 public:
  static constexpr int k_streams_per_pool = 32;

  absl::StatusOr<CudaStream> get_stream(int device_id, CudaStreamPriority priority);
  void release_all();

 private:
  struct DevicePool {
    absl::Mutex mu;
    bool initialized ABSL_GUARDED_BY(mu) = false;
    std::vector<cudaStream_t> default_streams ABSL_GUARDED_BY(mu);
    std::vector<cudaStream_t> high_streams ABSL_GUARDED_BY(mu);
    size_t default_index ABSL_GUARDED_BY(mu) = 0;
    size_t high_index ABSL_GUARDED_BY(mu) = 0;
  };

  absl::Status init_device_pool_(int device_id, DevicePool* pool) ABSL_EXCLUSIVE_LOCKS_REQUIRED(pool->mu);

  absl::Mutex pools_mu_;
  absl::flat_hash_map<int, std::unique_ptr<DevicePool>> pools_ ABSL_GUARDED_BY(pools_mu_);
};

CudaStream current_cuda_stream(int device_id);
void set_current_cuda_stream(const CudaStream& stream);

class CudaStreamGuard {
 public:
  explicit CudaStreamGuard(const CudaStream& stream);
  ~CudaStreamGuard();

  CudaStreamGuard(const CudaStreamGuard&) = delete;
  CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;
  CudaStreamGuard(CudaStreamGuard&&) = delete;
  CudaStreamGuard& operator=(CudaStreamGuard&&) = delete;

  CudaStream original_stream() const {
    return original_stream_;
  }

  CudaStream current_stream() const {
    return current_stream_;
  }

  absl::Status status() const {
    return device_guard_.status();
  }

 private:
  CudaDeviceGuard device_guard_;
  CudaStream original_stream_;
  CudaStream current_stream_;
};

} // namespace tensorcast::cuda
