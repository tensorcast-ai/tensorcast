// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/memory_source.h"

#include <algorithm>
#include <cstring>

#include "absl/status/status.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::store::loader {

CpuMemorySource::CpuMemorySource(gsl::not_null<const void*> base_ptr, uint64_t total_size)
    : base_ptr_(static_cast<const uint8_t*>(base_ptr.get())), total_size_(total_size) {}

absl::StatusOr<size_t> CpuMemorySource::read(void* dst, size_t max_bytes) {
  auto read_or = read_at(current_offset_, dst, max_bytes);
  if (!read_or.ok()) {
    return read_or;
  }
  current_offset_ += *read_or;
  return read_or;
}

absl::StatusOr<size_t> CpuMemorySource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (offset >= total_size_ || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
  std::memcpy(dst, base_ptr_ + offset, to_copy);
  return to_copy;
}

GpuMemorySource::GpuMemorySource(gsl::not_null<void*> device_ptr, int device_id, uint64_t total_size)
    : device_ptr_(device_ptr), device_id_(device_id), total_size_(total_size) {}

absl::StatusOr<size_t> GpuMemorySource::read(void* dst, size_t max_bytes) {
  auto read_or = read_at(current_offset_, dst, max_bytes);
  if (!read_or.ok()) {
    return read_or;
  }
  current_offset_ += *read_or;
  return read_or;
}

absl::StatusOr<size_t> GpuMemorySource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (offset >= total_size_ || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
  if (auto st = tensorcast::cuda::set_device(device_id_); !st.ok()) {
    return st;
  }
  auto st =
      tensorcast::cuda::memcpy(dst, static_cast<uint8_t*>(device_ptr_.get()) + offset, to_copy, cudaMemcpyDeviceToHost);
  if (!st.ok()) {
    return st;
  }
  auto sync = tensorcast::cuda::device_synchronize();
  if (!sync.ok()) {
    return sync;
  }
  return to_copy;
}

} // namespace tensorcast::store::loader
