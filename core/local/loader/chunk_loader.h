// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <future>
#include "absl/status/status.h"

namespace tensorcast::local::chunk {
class DataChunk;
}

namespace tensorcast::local::loader {

class ChunkLoader {
 public:
  explicit ChunkLoader(chunk::DataChunk* chunk) : chunk_(chunk) {}

  virtual ~ChunkLoader() = default;

  // Load data into the associated DataChunk. No parameters.
  virtual absl::Status load() = 0;

  // Async load version, returns a future for completion status.
  virtual std::future<absl::Status> load_async() = 0;

 protected:
  chunk::DataChunk* chunk_{nullptr};
};

} // namespace tensorcast::local::loader
