// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstddef>

#include "absl/status/status.h"

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
};

} // namespace stepcast::store::loader
