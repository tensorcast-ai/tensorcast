// Copyright (c) 2025-2026, TensorCast Team.

// Lightweight RAII wrapper for CUDA IPC memory mappings used by the daemon.
// This avoids scattered open/close pairs and double-close bugs.
// Header-only to keep build simple for now.

#pragma once

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::daemon {

class CudaIpcMapping {
 public:
  CudaIpcMapping() = default;

  explicit CudaIpcMapping(void* p) : ptr_(p) {}

  CudaIpcMapping(const CudaIpcMapping&) = delete;
  CudaIpcMapping& operator=(const CudaIpcMapping&) = delete;

  CudaIpcMapping(CudaIpcMapping&& o) noexcept : ptr_(o.ptr_) {
    o.ptr_ = nullptr;
  }

  CudaIpcMapping& operator=(CudaIpcMapping&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      o.ptr_ = nullptr;
    }
    return *this;
  }

  ~CudaIpcMapping() {
    reset();
  }

  static absl::StatusOr<CudaIpcMapping> open(const std::vector<uint8_t>& handle_bytes, unsigned flags) {
    return open(absl::MakeConstSpan(handle_bytes.data(), handle_bytes.size()), flags);
  }

  static absl::StatusOr<CudaIpcMapping> open(absl::Span<const uint8_t> handle_bytes, unsigned flags) {
    if (handle_bytes.size() != sizeof(cudaIpcMemHandle_t)) {
      return absl::InvalidArgumentError("invalid CUDA IPC handle size");
    }
    cudaIpcMemHandle_t h{};
    std::memcpy(&h, handle_bytes.data(), sizeof(h));
    void* p = nullptr;
    if (auto st = cuda::open_ipc_mem_handle(&p, h, flags); !st.ok())
      return st;
    return CudaIpcMapping(p);
  }

  static absl::StatusOr<CudaIpcMapping> open(const std::string& handle_bytes, unsigned flags) {
    return open(
        absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(handle_bytes.data()), handle_bytes.size()), flags);
  }

  static absl::StatusOr<CudaIpcMapping> open_from_raw(const void* data, size_t nbytes, unsigned flags) {
    if (nbytes != sizeof(cudaIpcMemHandle_t)) {
      return absl::InvalidArgumentError("invalid CUDA IPC handle size");
    }
    cudaIpcMemHandle_t h{};
    std::memcpy(&h, data, sizeof(h));
    void* p = nullptr;
    if (auto st = cuda::open_ipc_mem_handle(&p, h, flags); !st.ok())
      return st;
    return CudaIpcMapping(p);
  }

  template <typename T, size_t N>
  static absl::StatusOr<CudaIpcMapping> open(const std::array<T, N>& arr, unsigned flags) {
    return open_from_raw(arr.data(), N * sizeof(T), flags);
  }

  void* get() const {
    return ptr_;
  }

  void reset() {
    if (ptr_ != nullptr) {
      absl::Status _st = cuda::close_ipc_mem_handle(ptr_);
      if (!_st.ok()) {
        LOG(WARNING) << "CudaIpcMapping: close_ipc_mem_handle failed: " << _st;
      }
      ptr_ = nullptr;
    }
  }

 private:
  void* ptr_{nullptr};
};

} // namespace tensorcast::daemon
