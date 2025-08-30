// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gsl/pointers"

namespace tensorcast::communicator {

class PartitionTensor;

// Unified interface for staging memory into host-pinned buffers suitable
// for network transport. Implementations may perform memcpy (CPU) or
// cudaMemcpyAsync (GPU) and must provide a way to release buffers.
class MemoryStager {
 public:
  virtual ~MemoryStager() = default;

  // Stage a view from the given tensor starting at offset for bytes.
  // Returns a host pointer valid until release_staged_buffer() is called.
  // Implementations must ensure bytes <= get_chunk_size().
  virtual absl::StatusOr<void*> stage(
      const std::shared_ptr<PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes) = 0;

  // Release a previously staged buffer. Must be called once per stage().
  virtual absl::Status release_staged_buffer(gsl::not_null<void*> host_ptr) = 0;

  // Size of a single staging chunk (bytes). Callers should not request
  // more than this in a single stage() call.
  virtual size_t get_chunk_size() const = 0;

  // Hint for how many buffers are available for pipelining per flow.
  virtual size_t get_num_buffers() const = 0;
};

} // namespace tensorcast::communicator

