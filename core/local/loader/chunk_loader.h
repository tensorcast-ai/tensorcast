// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <future>
#include "absl/status/status.h"

namespace tensorcast::local::data {
class DataChunk;

class ChunkLoader {
 public:
  explicit ChunkLoader(DataChunk* data_chunk) : data_chunk_(data_chunk) {}

  virtual ~ChunkLoader() = default;

  // Load data into the associated DataChunk. No parameters.
  virtual absl::Status load() = 0;

  // Async load version, returns a future for completion status.
  virtual std::future<absl::Status> load_async() = 0;

 protected:
  DataChunk* data_chunk_{nullptr};
};

} // namespace tensorcast::local::data
