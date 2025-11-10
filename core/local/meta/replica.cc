// Copyright (c) 2025, TensorCast Team.

#include "core/local/meta/replica.h"

#include <utility>

#include "core/local/chunk/chunk.h"
#include "core/local/meta/view.h"

namespace tensorcast::local::meta {

Replica::Iterator::Iterator(const Replica* replica)
    : replica_(replica), chunks_it_(replica->view_->chunks_map_.begin()) {}

Replica::Iterator::Iterator(const Replica* replica, ChunksIterator it) : replica_(replica), chunks_it_(it) {}

Replica::Iterator& Replica::Iterator::operator++() {
  if (chunks_it_ != replica_->view_->chunks_map_.end()) {
    ++chunks_it_;
  }
  return *this;
}

Replica::Iterator Replica::Iterator::operator++(int) {
  Iterator tmp(*this);
  ++(*this);
  return tmp;
}

data::DataChunk* Replica::Iterator::operator*() const {
  if (chunks_it_ == replica_->view_->chunks_map_.end()) {
    return nullptr;
  }
  auto* chunk = chunks_it_->second.get();
  if (chunk == nullptr) {
    return nullptr;
  }
  return chunk->get_data_chunk(replica_->device_key_);
}

bool Replica::Iterator::operator==(const Iterator& other) const {
  return replica_ == other.replica_ && chunks_it_ == other.chunks_it_;
}

bool Replica::Iterator::operator!=(const Iterator& other) const {
  return !(*this == other);
}

off_t Replica::Iterator::get_offset() const {
  if (chunks_it_ == replica_->view_->chunks_map_.end()) {
    return -1;
  }
  return chunks_it_->first;
}

Replica::Replica(std::string replica_id, View* view, const store::DeviceKey& device_key)
    : replica_id_(std::move(replica_id)), view_(view), device_key_(device_key) {}

const std::string& Replica::get_replica_id() const {
  return replica_id_;
}

View* Replica::get_view() const {
  return view_;
}

const store::DeviceKey& Replica::get_device_key() const {
  return device_key_;
}

Replica::Iterator Replica::begin() const {
  return Iterator(this);
}

Replica::Iterator Replica::end() const {
  return Iterator(this, view_->chunks_map_.end());
}

data::DataChunk* Replica::operator[](off_t offset) const {
  if (view_ == nullptr) {
    return nullptr;
  }
  auto chunk = view_->get_chunk_at(offset);
  if (chunk == nullptr) {
    return nullptr;
  }
  return chunk->get_data_chunk(device_key_);
}

data::DataChunk* Replica::find(off_t offset) const {
  return (*this)[offset];
}

bool Replica::empty() const {
  return view_ == nullptr || view_->chunks_map_.empty();
}

size_t Replica::size() const {
  if (view_ == nullptr) {
    return 0;
  }
  return view_->chunks_map_.size();
}

} // namespace tensorcast::local::meta
