// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/memory_source.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::store::loader {

namespace {

uint64_t elapsed_us(std::chrono::steady_clock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

} // namespace

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
  const auto call_start = std::chrono::steady_clock::now();
  if (offset >= total_size_ || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
  const auto memcpy_start = std::chrono::steady_clock::now();
  std::memcpy(dst, base_ptr_ + offset, to_copy);
  const uint64_t memcpy_us = elapsed_us(memcpy_start);
  const uint64_t total_us = elapsed_us(call_start);
  const double copied_mib = static_cast<double>(to_copy) / (1024.0 * 1024.0);
  const double total_sec = static_cast<double>(total_us) / 1e6;
  const double throughput_mib_s = total_sec > 0.0 ? (copied_mib / total_sec) : 0.0;
  VLOG(2) << "seekable.cpu_memory.read_at offset=" << offset << " bytes=" << to_copy << " memcpy_us=" << memcpy_us
          << " total_us=" << total_us << " throughput_mib_s=" << throughput_mib_s;
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
  const auto call_start = std::chrono::steady_clock::now();
  if (offset >= total_size_ || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, total_size_ - offset));
  const auto set_device_start = std::chrono::steady_clock::now();
  if (auto st = tensorcast::cuda::set_device(device_id_); !st.ok()) {
    return st;
  }
  const uint64_t set_device_us = elapsed_us(set_device_start);
  const auto memcpy_start = std::chrono::steady_clock::now();
  auto st =
      tensorcast::cuda::memcpy(dst, static_cast<uint8_t*>(device_ptr_.get()) + offset, to_copy, cudaMemcpyDeviceToHost);
  if (!st.ok()) {
    return st;
  }
  const uint64_t memcpy_us = elapsed_us(memcpy_start);
  const auto sync_start = std::chrono::steady_clock::now();
  auto sync = tensorcast::cuda::device_synchronize();
  if (!sync.ok()) {
    return sync;
  }
  const uint64_t sync_us = elapsed_us(sync_start);
  const uint64_t total_us = elapsed_us(call_start);
  const double copied_mib = static_cast<double>(to_copy) / (1024.0 * 1024.0);
  const double total_sec = static_cast<double>(total_us) / 1e6;
  const double throughput_mib_s = total_sec > 0.0 ? (copied_mib / total_sec) : 0.0;
  VLOG(2) << "seekable.gpu_memory.read_at offset=" << offset << " bytes=" << to_copy
          << " set_device_us=" << set_device_us << " cuda_memcpy_us=" << memcpy_us << " sync_us=" << sync_us
          << " total_us=" << total_us << " throughput_mib_s=" << throughput_mib_s;
  return to_copy;
}

} // namespace tensorcast::store::loader
