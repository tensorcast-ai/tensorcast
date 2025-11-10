// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/data_chunk.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <stdexcept>

#include "absl/log/log.h"
#include "core/local/loader/chunk_loader.h"

namespace tensorcast::local::data {

// only accept a prepared gpu mem
GPUDataChunk::GPUDataChunk(
    meta::Chunk* chunk,
    store::DeviceKey device_key,
    common::memory::GpuDeviceMemory* gpu_memory,
    off_t offset)
    : DataChunk(chunk, std::move(device_key)) {
  // check size is page-aligned
  int64_t page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    PLOG(ERROR) << "Failed to get system page size";
    throw std::runtime_error("Failed to get system page size");
  }
  if ((get_size() % static_cast<size_t>(page_size)) != 0) {
    LOG(ERROR) << "mmap request: size (" << get_size() << ") is not page-aligned (page size: " << page_size << ")";
    throw std::invalid_argument("mmap region size must be page-aligned");
  }

  if (gpu_memory != nullptr) {
    auto s = bind_gpu_memory(gpu_memory, offset);
    if (!s.ok()) {
      throw std::runtime_error("Failed to bind GPU memory");
    }
  }
}

absl::Status GPUDataChunk::bind_gpu_memory(common::memory::GpuDeviceMemory* gpu_memory, off_t offset) {
  if (gpu_memory_ != nullptr) {
    return absl::AlreadyExistsError("GPU memory already bound");
  }

  assert(gpu_memory != nullptr);
  assert(gpu_memory->get() != nullptr);
  if (gpu_memory == nullptr) {
    return absl::InvalidArgumentError("GPU memory is null");
  }
  if (gpu_memory->get() == nullptr) {
    return absl::InvalidArgumentError("GPU memory is not initialized");
  }
  gpu_memory_ = gpu_memory;
  offset_ = offset;
  base_addr_ = static_cast<std::byte*>(gpu_memory_->get()) + offset;
  loaded_ = true;
  return absl::OkStatus();
}

} // namespace tensorcast::local::data