// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/data_chunk.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <stdexcept>

#include "absl/log/log.h"
#include "core/local/loader/chunk_loader.h"

namespace tensorcast::local::chunk {

int DataChunk::lock_refcnt() const {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);
  return lock_state_.lock_refcnt;
}

bool DataChunk::is_locked() const {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);
  return lock_state_.locked;
}

void DataChunk::register_loader(loader::ChunkLoader* loader, LoaderPriority priority) {
  if (loader == nullptr) {
    return;
  }
  if (priority == LoaderPriority::High) {
    high_priority_loaders_.push_back(loader);
  } else {
    low_priority_loaders_.push_back(loader);
  }
}

DataChunk::DataChunk(std::shared_ptr<store::replica::Replica> replica_ptr, off_t replica_offset, size_t size)
    : size(size), replica(std::move(replica_ptr)), r_offset(replica_offset) {
  // check size is page-aligned
  int64_t page_size = ::sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    PLOG(ERROR) << "Failed to get system page size";
    throw std::runtime_error("Failed to get system page size");
  }
  if ((size % static_cast<size_t>(page_size)) != 0) {
    LOG(ERROR) << "mmap request: size (" << size << ") is not page-aligned (page size: " << page_size << ")";
    throw std::invalid_argument("mmap region size must be page-aligned");
  }

  // mmap anonymous memory
  void* mapped = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapped == MAP_FAILED) {
    PLOG(ERROR) << "Failed to mmap anonymous memory for size " << size;
    cpu_base = nullptr;
    throw std::runtime_error("Failed to mmap anonymous memory");
  }
  cpu_base = mapped;

  // Note: DataChunk no longer performs file I/O. Loading is delegated to loaders.
}

absl::Status DataChunk::load() {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);

  if (cpu_base == nullptr || size == 0) {
    return absl::FailedPreconditionError("DataChunk not mapped or size is zero");
  }

  // pin memory before trying loaders
  if (::mlock(cpu_base, size) != 0) {
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
      in_dram = true;
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
      in_dram = true;
      return absl::OkStatus();
    }
    last_status = s;
  }

  // loading failed, unlock the memory
  ::munlock(cpu_base, size);
  return last_status;
}

std::future<absl::Status> DataChunk::load_async() {
  return std::async(std::launch::async, [this]() { return this->load(); });
}

// File I/O helpers removed. Loading is handled by registered loaders.

absl::Status DataChunk::drop() {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);

  if (cpu_base == nullptr || size == 0) {
    return absl::OkStatus();
  }

  if (::munlock(cpu_base, size) != 0) {
    int err = errno;
    PLOG(ERROR) << "munlock failed for chunk";
    return absl::ErrnoToStatus(err, "munlock failed for DataChunk");
  }

  int rc = ::madvise(cpu_base, size, MADV_FREE);
  if (rc != 0 && errno == EINVAL) {
    rc = ::madvise(cpu_base, size, MADV_DONTNEED);
  }
  if (rc != 0) {
    int err = errno;
    PLOG(WARNING) << "madvise failed for chunk";
    return absl::ErrnoToStatus(err, "madvise failed for DataChunk drop");
  }

  return absl::OkStatus();
}

ChunkPinLease::~ChunkPinLease() {
  release();
}

ChunkPinLease::ChunkPinLease(Impl impl) : impl_(std::make_shared<Impl>(std::move(impl))) {
  for (const auto& chunk : impl_->chunks) {
    if (!chunk) {
      continue;
    }
    std::lock_guard<std::mutex> guard(chunk->lock_state_.mutex);
    chunk->lock_state_.lock_refcnt += 1;
    chunk->lock_state_.locked = true;
  }
}

ChunkPinLease::ChunkPinLease(ChunkPinLease&& other) noexcept {
  impl_ = std::move(other.impl_);
}

ChunkPinLease& ChunkPinLease::operator=(ChunkPinLease&& other) noexcept {
  if (this != &other) {
    release();
    impl_ = std::move(other.impl_);
  }
  return *this;
}

bool ChunkPinLease::is_expired() const {
  if (!impl_ || !impl_->expiry_time)
    return false;
  return std::chrono::steady_clock::now() > *impl_->expiry_time;
}

void ChunkPinLease::release() noexcept {
  if (!impl_)
    return;
  if (is_expired()) {
    if (!impl_->chunks.empty() && impl_->chunks[0] && impl_->chunks[0]->replica) {
      LOG(WARNING) << "ChunkPinLease expired before being destroyed for replica: "
                   << impl_->chunks[0]->replica->artifact_id();
    } else {
      LOG(WARNING) << "ChunkPinLease expired before being destroyed";
    }
  }
  for (const auto& chunk : impl_->chunks) {
    if (!chunk) {
      continue;
    }
    std::lock_guard<std::mutex> guard(chunk->lock_state_.mutex);
    if (chunk->lock_state_.lock_refcnt == 0) {
      continue;
    }
    chunk->lock_state_.lock_refcnt -= 1;
    if (chunk->lock_state_.lock_refcnt == 0) {
      chunk->lock_state_.locked = false;
    }
  }

  impl_.reset();
}

ChunkPinLease ChunkPinLease::pin_chunks(
    std::vector<std::shared_ptr<DataChunk>>&& chunks,
    std::optional<std::chrono::steady_clock::time_point> expiry_time) {
  Impl impl;
  impl.chunks = std::move(chunks);
  impl.expiry_time = expiry_time;
  return ChunkPinLease(std::move(impl));
}

} // namespace tensorcast::local::chunk
