// Copyright (c) 2025, StepCast Team. All rights reserved.

// ----------------------------------------------------------------------------
//  Shared device type definitions
//
//  This header defines the lightweight DeviceType enum that is shared across
//  multiple core modules (common, communicator, store).  The definition lives
//  in the top-level `stepcast` namespace so that nested namespaces (e.g.
//  `stepcast::store`) can refer to it without additional qualifiers.
// ----------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace stepcast {

// Logical device categories recognised by the runtime.  Keep this header free
// from heavy dependencies so that it can be included from low-level modules
// without pulling in the rest of the store / communicator implementation.
enum class DeviceType : uint8_t {
  // Note: The ordering of existing values (CPU, GPU, REMOTE) is preserved to
  // keep wire compatibility with any serialized representations.  New values
  // are appended afterwards so they do not disturb the previous numeric layout.

  CPU = 0,
  GPU = 1,
  REMOTE = 2,

  // Additional locations that were previously covered by `MemoryLocation`.
  DISK = 3,
  NONE = 4,
};

// Helper for logging / debugging.
inline const char* to_string(DeviceType t) {
  switch (t) {
    case DeviceType::NONE:
      return "NONE";
    case DeviceType::DISK:
      return "DISK";
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::GPU:
      return "GPU";
    case DeviceType::REMOTE:
      return "REMOTE";
    default:
      return "UNKNOWN";
  }
}

} // namespace stepcast