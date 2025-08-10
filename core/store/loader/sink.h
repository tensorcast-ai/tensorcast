// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/direct_write.h"

namespace stepcast::store::loader {

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

// Optional capability: destination can plan direct writes into its VA ranges.
class DirectWritableSink {
 public:
  virtual ~DirectWritableSink() = default;

  // Plan a direct write token for the given destination VA ranges.
  // The returned token authorizes writing into these ranges and carries
  // any required keepalive resources (e.g., DVMP pin leases).
  virtual absl::StatusOr<stepcast::store::DirectWriteToken> plan_direct_write(
      absl::Span<const stepcast::store::VaRange> ranges) = 0;
};

} // namespace stepcast::store::loader
