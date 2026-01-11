// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/communicator/transport/partition_tensor.h"
#include "gsl/pointers"

namespace tensorcast::communicator::engine {

// Unified interface for staging memory into RDMA/MTCP-exposed buffers.
// Implementations may return host-pinned or device pointers; callers must
// treat the returned pointer as an exposed address for transport.
class MemoryStager {
 public:
  virtual ~MemoryStager() = default;

  enum class StageMode {
    kBlocking,
    kTry,
  };

  // Stage a view from the given tensor starting at offset for bytes.
  // Returns an exposed pointer valid until release_staged_buffer() is called.
  // Implementations must ensure bytes <= get_chunk_size().
  virtual absl::StatusOr<void*> stage(
      const std::shared_ptr<transport::PartitionTensor>& tensor,
      uint64_t offset,
      uint64_t bytes,
      StageMode mode = StageMode::kBlocking) = 0;

  // Release a previously staged buffer. Must be called once per stage().
  virtual absl::Status release_staged_buffer(gsl::not_null<void*> exposed_ptr) = 0;

  struct MrSlab {
    gsl::not_null<void*> base;
    size_t bytes = 0;
  };

  // Optional MR slab lookup for preregistered staging pools. When present,
  // callers may normalize MR registrations to the slab base.
  virtual std::optional<MrSlab> mr_slab_for_ptr(gsl::not_null<void*> /*exposed_ptr*/) const {
    return std::nullopt;
  }

  // Size of a single staging chunk (bytes). Callers should not request
  // more than this in a single stage() call.
  [[nodiscard]] virtual size_t get_chunk_size() const = 0;

  // Hint for how many buffers are available for pipelining per flow.
  [[nodiscard]] virtual size_t get_num_buffers() const = 0;
};

inline MemoryStager::MrSlab NormalizeMrRegion(
    const MemoryStager& stager,
    gsl::not_null<void*> exposed_ptr,
    size_t bytes) {
  if (auto slab = stager.mr_slab_for_ptr(exposed_ptr); slab.has_value()) {
    return *slab;
  }
  return MemoryStager::MrSlab{exposed_ptr, bytes};
}

} // namespace tensorcast::communicator::engine
