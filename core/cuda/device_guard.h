// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <optional>

#include "absl/log/absl_check.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::cuda {

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(int device_id) : target_device_(device_id) {
    status_ = ::tensorcast::cuda::get_device(&original_device_);
    if (!status_.ok()) {
      return;
    }
    initialized_ = true;
    if (original_device_ != target_device_) {
      status_ = ::tensorcast::cuda::set_device(target_device_);
    }
  }

  explicit CudaDeviceGuard(std::optional<int> device_id) {
    if (!device_id.has_value()) {
      return;
    }
    target_device_ = *device_id;
    status_ = ::tensorcast::cuda::get_device(&original_device_);
    if (!status_.ok()) {
      return;
    }
    initialized_ = true;
    if (original_device_ != target_device_) {
      status_ = ::tensorcast::cuda::set_device(target_device_);
    }
  }

  ~CudaDeviceGuard() {
    if (!initialized_) {
      return;
    }
    if (original_device_ != target_device_) {
      ABSL_CHECK_OK(::tensorcast::cuda::set_device(original_device_)); // Best effort
    }
  }

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard(CudaDeviceGuard&&) = delete;
  CudaDeviceGuard& operator=(CudaDeviceGuard&&) = delete;

  void set_device(int device_id) {
    target_device_ = device_id;
    if (!initialized_) {
      status_ = ::tensorcast::cuda::get_device(&original_device_);
      if (!status_.ok()) {
        return;
      }
      initialized_ = true;
    }
    status_ = ::tensorcast::cuda::set_device(target_device_);
  }

  void reset_device(int device_id) {
    set_device(device_id);
  }

  int original_device() const {
    return original_device_;
  }

  int current_device() const {
    return target_device_;
  }

  absl::Status status() const {
    return status_;
  }

 private:
  int original_device_ = -1;
  int target_device_ = -1;
  bool initialized_ = false;
  absl::Status status_ = absl::OkStatus();
};

class OptionalCudaDeviceGuard {
 public:
  OptionalCudaDeviceGuard() = default;

  explicit OptionalCudaDeviceGuard(std::optional<int> device_id) {
    if (device_id.has_value()) {
      guard_.emplace(*device_id);
    }
  }

  OptionalCudaDeviceGuard(const OptionalCudaDeviceGuard&) = delete;
  OptionalCudaDeviceGuard& operator=(const OptionalCudaDeviceGuard&) = delete;
  OptionalCudaDeviceGuard(OptionalCudaDeviceGuard&&) = delete;
  OptionalCudaDeviceGuard& operator=(OptionalCudaDeviceGuard&&) = delete;

  void set_device(int device_id) {
    if (!guard_.has_value()) {
      guard_.emplace(device_id);
      return;
    }
    guard_->set_device(device_id);
  }

  void reset_device(int device_id) {
    set_device(device_id);
  }

  std::optional<int> original_device() const {
    if (!guard_.has_value()) {
      return std::nullopt;
    }
    return guard_->original_device();
  }

  std::optional<int> current_device() const {
    if (!guard_.has_value()) {
      return std::nullopt;
    }
    return guard_->current_device();
  }

  void reset() {
    guard_.reset();
  }

  absl::Status status() const {
    if (!guard_.has_value()) {
      return absl::OkStatus();
    }
    return guard_->status();
  }

 private:
  std::optional<CudaDeviceGuard> guard_{};
};

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) : original_device_(-1), target_device_(device_id) {
    auto status = ::tensorcast::cuda::get_device(&original_device_);
    if (!status.ok()) {
      status_ = status;
      return;
    }
    if (original_device_ != target_device_) {
      status = ::tensorcast::cuda::set_device(target_device_);
      if (!status.ok()) {
        status_ = status;
      }
    }
  }

  ~DeviceGuard() {
    if (original_device_ != -1 && original_device_ != target_device_) {
      ABSL_CHECK_OK(::tensorcast::cuda::set_device(original_device_)); // Best effort
    }
  }

  absl::Status status() const {
    return status_;
  }

 private:
  int original_device_;
  int target_device_;
  absl::Status status_ = absl::OkStatus();
};

} // namespace tensorcast::cuda
