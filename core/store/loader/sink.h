// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/async_copy_manager.h"
#include "core/store/replica/types/direct_write_grant.h"

namespace tensorcast::store::loader {

class Sink {
 public:
  virtual ~Sink() = default;

  virtual absl::Status write(const void* src, size_t bytes) = 0;

  virtual absl::Status close() {
    return absl::OkStatus();
  }
};

// Sink that supports positioned writes into a destination space.
class PositionedSink {
 public:
  virtual ~PositionedSink() = default;

  // Write bytes starting at destination-global offset.
  virtual absl::Status write_at(uint64_t offset, const void* src, size_t bytes) = 0;

  // Optional close hook mirroring Sink API for positioned sinks that also
  // need lifecycle management in tests/pipelines. Default no-op.
  virtual absl::Status close() {
    return absl::OkStatus();
  }
};

// Optional capability: asynchronous positioned writes.
// If provided by a sink, callers (e.g., pump) can submit a transfer and
// delay buffer reclamation until the returned CopyHandle completes, enabling
// true overlap between I/O production and GPU DMA consumption without per-
// chunk synchronizations.
class AsyncPositionedSink {
 public:
  virtual ~AsyncPositionedSink() = default;

  // Asynchronously write bytes at destination-global offset. The returned
  // handle becomes ready when the sink-side transfer has completed (e.g.,
  // after an H2D cudaMemcpyAsync finishes). Implementations should schedule
  // copies via AsyncCopyManager to ensure unified tracing and stream usage.
  virtual absl::StatusOr<common::CopyHandle> write_at_async(uint64_t offset, const void* src, size_t bytes) = 0;
};

// Optional capability: destination can plan direct writes into its VA ranges.
class DirectWriteCapable {
 public:
  virtual ~DirectWriteCapable() = default;

  // Plan a direct write token for the given destination VA ranges.
  // The returned token authorizes writing into these ranges and carries
  // any required keepalive resources (e.g., VS pin leases).
  virtual absl::StatusOr<DirectWriteGrant> plan_direct_write(absl::Span<const VaRange> ranges) = 0;
};

} // namespace tensorcast::store::loader
