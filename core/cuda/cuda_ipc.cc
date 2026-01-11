// Copyright (c) 2025-2026, TensorCast Team.

#include "core/cuda/cuda_ipc.h"

#include <algorithm>
#include <cstring>

#include "absl/log/log.h"

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

IpcMapping::IpcMapping(IpcMapping&& other) noexcept : ptr_(other.ptr_) {
  other.ptr_ = nullptr;
}

IpcMapping& IpcMapping::operator=(IpcMapping&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  return *this;
}

IpcMapping::~IpcMapping() {
  reset();
}

void* IpcMapping::get() const {
  return ptr_;
}

void IpcMapping::reset() {
  if (ptr_ == nullptr) {
    return;
  }
  absl::Status status = cuda::close_ipc_mem_handle(ptr_);
  if (!status.ok()) {
    LOG(WARNING) << "IpcMapping: close_ipc_mem_handle failed: " << status;
  }
  ptr_ = nullptr;
}

absl::StatusOr<IpcMapping> IpcMapping::open(const IpcHandleBytes& handle, OpenOptions opts) {
  return open(handle.to_native(), opts);
}

absl::StatusOr<IpcMapping> IpcMapping::open(cudaIpcMemHandle_t handle, OpenOptions opts) {
  void* ptr = nullptr;
  absl::Status status = cuda::open_ipc_mem_handle(&ptr, handle, opts.flags);
  if (!status.ok()) {
    return status;
  }
  if (ptr == nullptr) {
    return absl::InternalError("cudaIpcOpenMemHandle returned nullptr");
  }
  return IpcMapping(ptr);
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
