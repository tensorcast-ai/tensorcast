// Copyright (c) 2025, TensorCast Team.

// UMA V3: DirectWriteGrant with windowed authorization semantics.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

// VaRange definition
namespace tensorcast::store {
struct VaRange {
  uint64_t offset;
  uint64_t length;
};
} // namespace tensorcast::store

namespace tensorcast::store {

// Grant describing permitted direct-write windows into destination VA space.
// Each window authorizes writing a contiguous [va_offset, va_offset+length)
// range and carries the resolved local_addr (CPU VA) for zero-copy transfers.
// keepalive holds any underlying leases/registrations needed to keep the
// mapping valid for the lifetime of the grant.
struct DirectWriteGrant {
  struct Window {
    uint64_t va_offset; // VA offset within the replica
    uint64_t local_addr; // Resolved local CPU address for the start of the window
    uint64_t length; // Window length in bytes
  };

  std::vector<Window> windows;
  std::shared_ptr<void> keepalive; // opaque lifetime anchor
};

} // namespace tensorcast::store
