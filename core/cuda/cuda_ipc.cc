// Copyright (c) 2025-2026, TensorCast Team.

#include "core/cuda/cuda_ipc.h"

#include <algorithm>
#include <cstring>

#include "absl/log/log.h"
#include "core/cuda/device_guard.h"

namespace tensorcast::cuda {

bool IpcHandleBytes::is_valid() const {
  return std::any_of(bytes.begin(), bytes.end(), [](std::uint8_t b) { return b != 0; });
}

cudaIpcMemHandle_t IpcHandleBytes::to_native() const {
  cudaIpcMemHandle_t handle{};
  std::memcpy(&handle, bytes.data(), bytes.size());
  return handle;
}

IpcHandleBytes IpcHandleBytes::from_native(const cudaIpcMemHandle_t& handle) {
  IpcHandleBytes out;
  std::memcpy(out.bytes.data(), &handle, out.bytes.size());
  return out;
}

absl::Span<const std::uint8_t> IpcHandleBytes::as_bytes() const {
  return absl::Span<const std::uint8_t>(bytes.data(), bytes.size());
}

absl::string_view IpcHandleBytes::as_string_view() const {
  return absl::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

IpcMapping::IpcMapping(void* ptr) : ptr_(ptr) {}

IpcMapping::IpcMapping(void* ptr, int device_id) : ptr_(ptr), device_id_(device_id) {}

IpcMapping::IpcMapping(IpcMapping&& other) noexcept : ptr_(other.ptr_), device_id_(other.device_id_) {
  other.ptr_ = nullptr;
  other.device_id_ = -1;
}

IpcMapping& IpcMapping::operator=(IpcMapping&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = other.ptr_;
    device_id_ = other.device_id_;
    other.ptr_ = nullptr;
    other.device_id_ = -1;
  }
  return *this;
}

IpcMapping::~IpcMapping() {
  reset();
}

void* IpcMapping::get() const {
  return ptr_;
}

absl::Status IpcMapping::close() {
  if (ptr_ == nullptr) {
    return absl::OkStatus();
  }
  void* ptr = ptr_;
  const int device_id = device_id_;
  ptr_ = nullptr;
  device_id_ = -1;

  if (device_id >= 0) {
    CudaDeviceGuard guard(device_id);
    absl::Status close_status = cuda::close_ipc_mem_handle(ptr);
    if (!close_status.ok()) {
      return close_status;
    }
    return guard.status();
  }
  return cuda::close_ipc_mem_handle(ptr);
}

void IpcMapping::reset() {
  absl::Status status = close();
  if (!status.ok()) {
    LOG(WARNING) << "IpcMapping: close_ipc_mem_handle failed: " << status;
  }
}

absl::StatusOr<IpcMapping> IpcMapping::open(const IpcHandleBytes& handle, OpenOptions opts) {
  return open(handle.to_native(), opts);
}

absl::StatusOr<IpcMapping> IpcMapping::open(cudaIpcMemHandle_t handle, OpenOptions opts) {
  int current_device = -1;
  absl::Status status = cuda::get_device(&current_device);
  if (!status.ok()) {
    return status;
  }
  void* ptr = nullptr;
  status = cuda::open_ipc_mem_handle(&ptr, handle, opts.flags);
  if (!status.ok()) {
    return status;
  }
  if (ptr == nullptr) {
    return absl::InternalError("cudaIpcOpenMemHandle returned nullptr");
  }
  return IpcMapping(ptr, current_device);
}

absl::StatusOr<IpcMapping> IpcMapping::open(absl::Span<const std::uint8_t> bytes, OpenOptions opts) {
  if (bytes.size() != IpcHandleBytes::kHandleSize) {
    return absl::InvalidArgumentError("invalid CUDA IPC handle size");
  }
  cudaIpcMemHandle_t handle{};
  std::memcpy(&handle, bytes.data(), bytes.size());
  return open(handle, opts);
}

absl::StatusOr<IpcMapping> IpcMapping::open(absl::string_view bytes, OpenOptions opts) {
  return open(absl::Span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()), opts);
}

} // namespace tensorcast::cuda
