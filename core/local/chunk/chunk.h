// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <sys/types.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/local/chunk/data_chunk.h"
#include "core/store/device_types.h"

// namespace tensorcast::local::data {
// class DataChunk;
// }  // namespace tensorcast::local::data

namespace tensorcast::local::meta {
class Artifact;
class View;

/**
 * A chunk is a part of an artifact view. This class is the
 * metadata of a chunk. the data of a chunk can be replicated
 * across different devices. For each physical instance of a
 * chunk, there is a DataChunk object that manages the data.
 * DataChunks across all devices are gathered in dev_data_chunks_.
 */
class Chunk {
 public:
  explicit Chunk(size_t size, meta::Artifact* artifact_ptr) : artifact_(std::move(artifact_ptr)), size_(size) {}

  virtual ~Chunk() = default;

  // generate data chunks on different devices
  void generate_data_chunks(const std::vector<store::DeviceKey>& device_keys);
  // get the data chunk on a specific device
  data::DataChunk* get_data_chunk(const store::DeviceKey& device_key) const;

  size_t get_size() const {
    return size_;
  };

  Artifact* get_artifact() const {
    return artifact_;
  };

  std::string to_string() const;

  off_t get_offset_in_view(const View* view) const;

  std::vector<View*> get_holder_views() const;

 private:
  Artifact* artifact_;
  size_t size_{0};
  // data chunks on different devices
  std::unordered_map<store::DeviceKey, std::unique_ptr<data::DataChunk>, store::DeviceKeyHash> dev_data_chunks_;

  // view id to offset
  std::unordered_map<std::string, off_t> offset_in_view_;
  friend class View;
};
} // namespace tensorcast::local::meta