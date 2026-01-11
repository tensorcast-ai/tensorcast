// Copyright (c) 2025-2026, TensorCast Team.

#include "core/cuda/cuda_event.h"

namespace tensorcast::cuda {

CudaEvent::~CudaEvent() {
  if (!is_created_) {
    return;
  }
  CudaDeviceGuard guard(device_id_);
  if (!guard.status().ok()) {
    return;
  }
  (void)event_destroy(event_);
}

CudaEvent::CudaEvent(CudaEvent&& other) noexcept {
  move_from_(std::move(other));
}

CudaEvent& CudaEvent::operator=(CudaEvent&& other) noexcept {
  if (this != &other) {
    move_from_(std::move(other));
  }
  return *this;
}

absl::Status CudaEvent::create_event_(int device_id) {
  CudaDeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    return guard.status();
  }
  absl::Status status = event_create_with_flags(&event_, flags_);
  if (!status.ok()) {
    return status;
  }
  device_id_ = device_id;
  is_created_ = true;
  return absl::OkStatus();
}

absl::Status CudaEvent::record(const CudaStream& stream) {
  if (!is_created_) {
    absl::Status status = create_event_(stream.device_id());
    if (!status.ok()) {
      return status;
    }
  }
  if (device_id_ != stream.device_id()) {
    return absl::InvalidArgumentError("CUDA event device does not match recording stream device");
  }
  CudaDeviceGuard guard(device_id_);
  if (!guard.status().ok()) {
    return guard.status();
  }
  absl::Status status = event_record(event_, stream.stream());
  if (!status.ok()) {
    return status;
  }
  was_recorded_ = true;
  return absl::OkStatus();
}

absl::Status CudaEvent::record_once(const CudaStream& stream) {
  if (was_recorded_) {
    return absl::OkStatus();
  }
  return record(stream);
}

absl::Status CudaEvent::block(const CudaStream& stream) const {
  if (!is_created_) {
    return absl::OkStatus();
  }
  if (device_id_ != stream.device_id()) {
    return absl::InvalidArgumentError("CUDA event device does not match target stream device");
  }
  CudaDeviceGuard guard(stream.device_id());
  if (!guard.status().ok()) {
    return guard.status();
  }
  return stream_wait_event(stream.stream(), event_);
}

absl::Status CudaEvent::query(bool* ready) const {
  if (ready == nullptr) {
    return absl::InvalidArgumentError("ready pointer must not be null");
  }
  if (!is_created_) {
    *ready = true;
    return absl::OkStatus();
  }
  return event_query(event_, ready);
}

absl::Status CudaEvent::synchronize() const {
  if (!is_created_) {
    return absl::OkStatus();
  }
  return event_synchronize(event_);
}

absl::StatusOr<float> CudaEvent::elapsed_time(const CudaEvent& other) const {
  if (!is_created_ || !other.is_created_) {
    return absl::FailedPreconditionError("CUDA events must be created before measuring elapsed time");
  }
  if ((flags_ & cudaEventDisableTiming) != 0 || (other.flags_ & cudaEventDisableTiming) != 0) {
    return absl::FailedPreconditionError("CUDA events must enable timing to measure elapsed time");
  }

  bool ready = false;
  absl::Status status = query(&ready);
  if (!status.ok()) {
    return status;
  }
  if (!ready) {
    return absl::FailedPreconditionError("start CUDA event not completed");
  }
  status = other.query(&ready);
  if (!status.ok()) {
    return status;
  }
  if (!ready) {
    return absl::FailedPreconditionError("end CUDA event not completed");
  }

  float ms = 0.0f;
  status = event_elapsed_time(&ms, event_, other.event_);
  if (!status.ok()) {
    return status;
  }
  return ms;
}

void CudaEvent::move_from_(CudaEvent&& other) noexcept {
  if (this == &other) {
    return;
  }
  if (is_created_) {
    CudaDeviceGuard guard(device_id_);
    if (guard.status().ok()) {
      (void)event_destroy(event_);
    }
  }
  event_ = other.event_;
  device_id_ = other.device_id_;
  flags_ = other.flags_;
  is_created_ = other.is_created_;
  was_recorded_ = other.was_recorded_;

  other.event_ = nullptr;
  other.device_id_ = -1;
  other.is_created_ = false;
  other.was_recorded_ = false;
}

} // namespace tensorcast::cuda
