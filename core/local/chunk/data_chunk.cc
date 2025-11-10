// Copyright (c) 2025, TensorCast Team.

#include "core/local/chunk/data_chunk.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>

#include "absl/log/log.h"
#include "core/local/chunk/chunk.h"
#include "core/local/loader/chunk_loader.h"

// #include "core/local/artifact/artifact.h"

namespace tensorcast::local::data {

int DataChunk::lock_refcnt() const {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);
  return lock_state_.lock_refcnt;
}

bool DataChunk::is_locked() const {
  std::lock_guard<std::mutex> guard(lock_state_.mutex);
  return lock_state_.locked;
}

void DataChunk::register_loader(std::shared_ptr<ChunkLoader> loader, LoaderPriority priority) {
  if (!loader) {
    return;
  }
  if (priority == LoaderPriority::High) {
    high_priority_loaders_.push_back(std::move(loader));
  } else {
    low_priority_loaders_.push_back(std::move(loader));
  }
}

size_t DataChunk::get_size() const {
  return chunk_->get_size();
};

void* DataChunk::get_base_addr() const {
  return base_addr_;
}

bool DataChunk::is_loaded() const {
  return loaded_;
}

int DataChunk::get_preempt_level() const {
  return preempt_level_;
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
    //   if (!impl_->data_chunks.empty() && impl_->data_chunks[0]) {
    //     auto& chunk = impl_->data_chunks[0]->chunk_;
    //     // auto artifact = data_chunk->chunk_->get_artifact();
    //     if (chunk) {
    //       LOG(WARNING) << "ChunkPinLease expired before being destroyed for replica: "
    //                    << chunk->to_string();
    //     } else {
    //       LOG(WARNING)
    //           << "ChunkPinLease expired before being destroyed (artifact already released)";
    //     }
    //   } else {
    LOG(WARNING) << "ChunkPinLease expired before being destroyed";
    // }
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

ChunkPinLease::ChunkPinLease(
    const std::vector<DataChunk*>& data_chunks,
    std::optional<std::chrono::steady_clock::time_point> expiry_time)
    : ChunkPinLease(Impl{.data_chunks = data_chunks, .expiry_time = expiry_time}) {}

ChunkPinLease::ChunkPinLease(std::optional<std::chrono::steady_clock::time_point> expiry_time)
    : ChunkPinLease(Impl{.data_chunks = {}, .expiry_time = expiry_time}) {}

// {
//   // Impl impl;
//   impl_->data_chunks = std::move(data_chunks);
//   impl_->expiry_time = expiry_time;
//   // return ChunkPinLease(std::move(impl));
// }

absl::Status ChunkPinLease::pin(DataChunk* data_chunk) {
  if (!data_chunk) {
    return absl::InvalidArgumentError("DataChunk pointer is null");
  }
  if (std::find(impl_->data_chunks.begin(), impl_->data_chunks.end(), data_chunk) != impl_->data_chunks.end()) {
    return absl::AlreadyExistsError("DataChunk is already pinned in this lease");
  }
  std::lock_guard<std::mutex> guard(data_chunk->lock_state_.mutex);
  data_chunk->lock_state_.lock_refcnt += 1;
  data_chunk->lock_state_.locked = true;
  return absl::OkStatus();
}

} // namespace tensorcast::local::data
