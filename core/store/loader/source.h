// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/status/statusor.h"

namespace stepcast::store::loader {

class Source {
 public:
  virtual ~Source() = default;

  virtual absl::StatusOr<size_t> read(void* dst, size_t max_bytes) = 0;
};

class SeekableSource : public Source {
 public:
  virtual absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) = 0;
};

} // namespace stepcast::store::loader