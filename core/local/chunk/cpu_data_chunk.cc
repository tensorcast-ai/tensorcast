// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/data_chunk.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <stdexcept>

#include "absl/log/log.h"
#include "core/local/loader/chunk_loader.h"

namespace tensorcast::local::data {

CPUDataChunk::CPUDataChunk(meta::Chunk* chunk) : DataChunk(chunk) {
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

  // mmap anonymous memory
  void* mapped = ::mmap(nullptr, get_size(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapped == MAP_FAILED) {
    PLOG(ERROR) << "Failed to mmap anonymous memory for size " << get_size();
    base_addr = nullptr;
    throw std::runtime_error("Failed to mmap anonymous memory");
  }
  base_addr = mapped;

  // Note: DataChunk no longer performs file I/O. Loading is delegated to loaders.
}

absl::Status CPUDataChunk::load() {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);

  if (base_addr == nullptr || get_size() == 0) {
    return absl::FailedPreconditionError("DataChunk not mapped or size is zero");
  }

  // pin memory before trying loaders
  if (::mlock(base_addr, get_size()) != 0) {
    int err = errno;
    PLOG(ERROR) << "mlock failed for chunk";
    return absl::ErrnoToStatus(err, "mlock failed for DataChunk");
  }

  absl::Status last_status = absl::NotFoundError("no loaders registered");
  // Try high-priority loaders first
  for (auto* loader : high_priority_loaders_) {
    if (loader == nullptr)
      continue;
    absl::Status s = loader->load();
    if (s.ok()) {
      loaded = true;
      return absl::OkStatus();
    }
    last_status = s;
  }
  // Then low-priority loaders
  for (auto* loader : low_priority_loaders_) {
    if (loader == nullptr)
      continue;
    absl::Status s = loader->load();
    if (s.ok()) {
      loaded = true;
      return absl::OkStatus();
    }
    last_status = s;
  }

  // loading failed, unlock the memory
  ::munlock(base_addr, get_size());
  return last_status;
}

std::future<absl::Status> CPUDataChunk::load_async() {
  return std::async(std::launch::async, [this]() { return this->load(); });
}

// File I/O helpers removed. Loading is handled by registered loaders.

absl::Status CPUDataChunk::drop() {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);

  if (base_addr == nullptr || get_size() == 0) {
    return absl::OkStatus();
  }

  if (::munlock(base_addr, get_size()) != 0) {
    int err = errno;
    PLOG(ERROR) << "munlock failed for chunk";
    return absl::ErrnoToStatus(err, "munlock failed for DataChunk");
  }

  int rc = ::madvise(base_addr, get_size(), MADV_FREE);
  if (rc != 0 && errno == EINVAL) {
    rc = ::madvise(base_addr, get_size(), MADV_DONTNEED);
  }
  if (rc != 0) {
    int err = errno;
    PLOG(WARNING) << "madvise failed for chunk";
    return absl::ErrnoToStatus(err, "madvise failed for DataChunk drop");
  }

  return absl::OkStatus();
}

} // namespace tensorcast::local::data