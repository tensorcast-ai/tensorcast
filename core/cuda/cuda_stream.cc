// Copyright (c) 2025-2026, TensorCast Team.

#include "core/cuda/cuda_stream.h"

#include "absl/log/log.h"
#include "core/cuda/error_handling.h"

namespace tensorcast::cuda {
namespace {

thread_local absl::flat_hash_map<int, cudaStream_t> g_current_streams;

} // namespace

absl::Status CudaStreamPool::init_device_pool_(int device_id, DevicePool* pool) {
  if (pool->initialized) {
    return absl::OkStatus();
  }

  CudaDeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }

  pool->default_streams.reserve(k_streams_per_pool);
  pool->high_streams.reserve(k_streams_per_pool);
  for (int i = 0; i < k_streams_per_pool; ++i) {
    cudaStream_t stream = nullptr;
    absl::Status status = stream_create_with_flags(&stream, cudaStreamNonBlocking);
    if (!status.ok()) {
      return status;
    }
    pool->default_streams.push_back(stream);
  }

  int least_priority = 0;
  int greatest_priority = 0;
  if (!is_fake()) {
    absl::Status status = TC_CUDA_STATUS(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    if (!status.ok()) {
      return status;
    }
  }
  (void)least_priority;

  for (int i = 0; i < k_streams_per_pool; ++i) {
    cudaStream_t stream = nullptr;
    absl::Status status = stream_create_with_priority(&stream, cudaStreamNonBlocking, greatest_priority);
    if (!status.ok()) {
      return status;
    }
    pool->high_streams.push_back(stream);
  }

  pool->initialized = true;
  return absl::OkStatus();
}

absl::StatusOr<CudaStream> CudaStreamPool::get_stream(int device_id, CudaStreamPriority priority) {
  if (device_id < 0) {
    return absl::InvalidArgumentError("invalid device_id for CUDA stream pool");
  }

  DevicePool* pool = nullptr;
  {
    absl::MutexLock lock(&pools_mu_);
    auto it = pools_.find(device_id);
    if (it == pools_.end()) {
      auto pool_ptr = std::make_unique<DevicePool>();
      pool = pool_ptr.get();
      pools_.emplace(device_id, std::move(pool_ptr));
    } else {
      pool = it->second.get();
    }
  }

  absl::MutexLock lock(&pool->mu);
  if (!pool->initialized) {
    absl::Status init_status = init_device_pool_(device_id, pool);
    if (!init_status.ok()) {
      return init_status;
    }
  }

  if (priority == CudaStreamPriority::kHigh && !pool->high_streams.empty()) {
    cudaStream_t stream = pool->high_streams[pool->high_index % pool->high_streams.size()];
    pool->high_index++;
    return CudaStream(stream, device_id);
  }

  if (pool->default_streams.empty()) {
    return absl::InternalError("CUDA stream pool is empty");
  }

  cudaStream_t stream = pool->default_streams[pool->default_index % pool->default_streams.size()];
  pool->default_index++;
  return CudaStream(stream, device_id);
}

void CudaStreamPool::release_all() {
  absl::MutexLock lock(&pools_mu_);
  for (auto& kv : pools_) {
    const int device_id = kv.first;
    DevicePool* pool = kv.second.get();
    absl::MutexLock pool_lock(&pool->mu);
    if (!pool->initialized) {
      continue;
    }
    CudaDeviceGuard guard(device_id);
    if (!guard.status().ok()) {
      LOG(WARNING) << "Failed to set CUDA device during stream pool cleanup: " << guard.status();
      continue;
    }

    for (cudaStream_t stream : pool->default_streams) {
      if (stream != nullptr) {
        (void)stream_destroy(stream);
      }
    }
    for (cudaStream_t stream : pool->high_streams) {
      if (stream != nullptr) {
        (void)stream_destroy(stream);
      }
    }

    pool->default_streams.clear();
    pool->high_streams.clear();
    pool->default_index = 0;
    pool->high_index = 0;
    pool->initialized = false;
  }
}

CudaStream current_cuda_stream(int device_id) {
  auto it = g_current_streams.find(device_id);
  if (it == g_current_streams.end()) {
    return CudaStream(nullptr, device_id);
  }
  return CudaStream(it->second, device_id);
}

void set_current_cuda_stream(const CudaStream& stream) {
  if (stream.device_id() < 0) {
    return;
  }
  g_current_streams[stream.device_id()] = stream.stream();
}

CudaStreamGuard::CudaStreamGuard(const CudaStream& stream)
    : device_guard_(stream.device_id()),
      original_stream_(current_cuda_stream(stream.device_id())),
      current_stream_(stream) {
  if (device_guard_.status().ok()) {
    set_current_cuda_stream(stream);
  }
}

CudaStreamGuard::~CudaStreamGuard() {
  if (!device_guard_.status().ok()) {
    return;
  }
  set_current_cuda_stream(original_stream_);
}

} // namespace tensorcast::cuda
