// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstdint>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_stream.h"
#include "core/cuda/device_guard.h"

namespace tensorcast::cuda {

class CudaEvent {
 public:
  CudaEvent() noexcept = default;

  explicit CudaEvent(unsigned int flags) noexcept : flags_(flags) {}

  ~CudaEvent();

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept;
  CudaEvent& operator=(CudaEvent&& other) noexcept;

  bool is_created() const {
    return is_created_;
  }

  int device_id() const {
    return device_id_;
  }

  absl::Status record(const CudaStream& stream);
  absl::Status record_once(const CudaStream& stream);
  absl::Status block(const CudaStream& stream) const;
  absl::Status query(bool* ready) const;
  absl::Status synchronize() const;
  absl::StatusOr<float> elapsed_time(const CudaEvent& other) const;

 private:
  absl::Status create_event_(int device_id);
  void move_from_(CudaEvent&& other) noexcept;

  cudaEvent_t event_{nullptr};
  int device_id_ = -1;
  unsigned int flags_ = cudaEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
};

} // namespace tensorcast::cuda
