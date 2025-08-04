// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include "absl/log/absl_check.h"
#include "core/common/cuda_api.h"

namespace stepcast::common {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) : original_device_(-1), target_device_(device_id) {
    auto status = cuda::get_device(&original_device_);
    if (!status.ok()) {
      status_ = status;
      return;
    }
    if (original_device_ != target_device_) {
      status = cuda::set_device(target_device_);
      if (!status.ok()) {
        status_ = status;
      }
    }
  }
  ~DeviceGuard() {
    if (original_device_ != -1 && original_device_ != target_device_) {
      ABSL_CHECK_OK(cuda::set_device(original_device_)); // Best effort
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

} // namespace stepcast::common