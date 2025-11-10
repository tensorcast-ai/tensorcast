// Copyright (c) 2025, TensorCast Team.

#pragma once
#include <sys/types.h>
#include <iterator>
#include <map>
#include <memory>
#include <string>

#include "core/local/chunk/data_chunk.h"
#include "core/store/device_types.h"

namespace tensorcast::local::data {
class DataChunk;
class ChunkPinLease;
} // namespace tensorcast::local::data

namespace tensorcast::local::meta {
class Chunk;
class View;

// Replica 表示特定设备上的一个视图，提供迭代和按偏移访问能力。
class Replica {
 public:
  using ChunksIterator = std::map<off_t, std::shared_ptr<Chunk>>::iterator;

  class Iterator {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = data::DataChunk*;
    using difference_type = std::ptrdiff_t;
    using pointer = data::DataChunk**;
    using reference = data::DataChunk*&;

    explicit Iterator(const Replica* replica);
    Iterator(const Replica* replica, ChunksIterator it);

    Iterator(const Iterator& other) = default;
    Iterator& operator=(const Iterator& other) = default;

    Iterator& operator++();
    Iterator operator++(int);

    data::DataChunk* operator*() const;

    bool operator==(const Iterator& other) const;
    bool operator!=(const Iterator& other) const;

    off_t get_offset() const;

   private:
    const Replica* replica_;
    ChunksIterator chunks_it_;
  };

  Replica(std::string replica_id, View* view, const store::DeviceKey& device_key);

  const std::string& get_replica_id() const;
  View* get_view() const;
  const store::DeviceKey& get_device_key() const;

  Iterator begin() const;
  Iterator end() const;

  data::DataChunk* operator[](off_t offset) const;
  data::DataChunk* find(off_t offset) const;
  bool empty() const;
  size_t size() const;

 private:
  std::string replica_id_;
  View* view_;
  store::DeviceKey device_key_;
};

class ReplicaHandler {
 public:
  ReplicaHandler(std::unique_ptr<data::ChunkPinLease> lease, Replica replica)
      : lease_(std::move(lease)), replica_(std::move(replica)) {}

  Replica get_replica() const {
    return replica_;
  }

  void release() {
    lease_->release();
  }

 private:
  std::unique_ptr<data::ChunkPinLease> lease_;
  Replica replica_;
};

} // namespace tensorcast::local::meta
