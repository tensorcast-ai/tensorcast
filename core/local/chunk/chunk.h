// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/store/device_types.h"
#include "core/store/replica/replica.h"

namespace tensorcast::local::chunk {

class DataChunk;

/**
 * A chunk is a part of an artifact view. This class is the
 * metadata of a chunk. the data of a chunk can be replicated
 * across different devices. For each physical instance of a
 * chunk, there is a DataChunk object that manages the data.
 * DataChunks across all devices are gathered in dev_data_chunks_.
 */
class Chunk {
 public:
  explicit Chunk(size_t size, store::replica::Replica* replica_ptr, off_t r_offset)
      : replica_(replica_ptr), r_offset_(r_offset), size_(size) {}

  virtual ~Chunk() = default;

  void generate_data_chunks(const std::vector<store::DeviceKey>& device_keys);
  DataChunk* get_data_chunk(const store::DeviceKey& device_key) const;

  size_t get_size() const {
    return size_;
  };

  store::replica::Replica* get_replica() const {
    return replica_;
  };

  off_t get_r_offset() const {
    return r_offset_;
  };

 private:
  // TODO: this should be artifact
  store::replica::Replica* replica_;
  off_t r_offset_{0};
  size_t size_{0};
  // data chunks on different devices
  std::unordered_map<store::DeviceKey, std::unique_ptr<DataChunk>, store::DeviceKeyHash> dev_data_chunks_;
};
} // namespace tensorcast::local::chunk