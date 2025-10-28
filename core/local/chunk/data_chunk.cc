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

ChunkPinLease::~ChunkPinLease() {
  release();
}

ChunkPinLease::ChunkPinLease(Impl impl) : impl_(std::make_shared<Impl>(std::move(impl))) {
  for (const auto& chunk : impl_->data_chunks) {
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
    if (!impl_->data_chunks.empty() && impl_->data_chunks[0] && impl_->data_chunks[0]->chunk_->get_replica()) {
      LOG(WARNING) << "ChunkPinLease expired before being destroyed for replica: "
                   << impl_->data_chunks[0]->chunk_->get_replica()->artifact_id();
    } else {
      LOG(WARNING) << "ChunkPinLease expired before being destroyed";
    }
  }
  for (const auto& chunk : impl_->data_chunks) {
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
    std::vector<std::shared_ptr<DataChunk>>&& data_chunks,
    std::optional<std::chrono::steady_clock::time_point> expiry_time) {
  Impl impl;
  impl.data_chunks = std::move(data_chunks);
  impl.expiry_time = expiry_time;
  return ChunkPinLease(std::move(impl));
}

} // namespace tensorcast::local::chunk
