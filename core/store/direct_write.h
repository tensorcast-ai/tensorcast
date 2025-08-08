// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace stepcast::store {

// Byte range expressed in destination VA space (offset within model) and length.
struct VaRange {
  uint64_t offset;
  uint64_t length;
};

// Token describing permitted direct-write segments into destination address space.
// The producer must only write to the described (offset,length) ranges.
// keepalive is an opaque handle that keeps underlying pin leases/registrations alive.
struct DirectWriteToken {
  struct Segment {
    uint64_t va_offset; // VA offset within the model
    uint64_t local_addr; // Resolved local CPU address for the start of the segment
    uint64_t length; // Segment length in bytes
  };
  std::vector<Segment> segments;
  std::shared_ptr<void> keepalive; // opaque lifetime anchor
};

} // namespace stepcast::store
