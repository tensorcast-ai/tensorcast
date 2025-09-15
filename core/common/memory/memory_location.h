// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace tensorcast::common::memory {

/**
 * @brief Specifies the target or current location of replica data.
 * Unlike the old bitmask, this represents a single distinct location.
 */
enum class MemoryLocation : uint8_t {
  NONE = 0,
  DISK, // Represents data stored persistently on disk.
  GPU, // Represents data loaded into GPU device memory.
  CPU, // Unified name for CPU memory
  REMOTE, // Represents data accessible via RDMA (source only for now).
};

// Add operator<< overload for MemoryLocation
inline std::ostream& operator<<(std::ostream& os, MemoryLocation loc) {
  switch (loc) {
    case MemoryLocation::NONE:
      os << "NONE";
      break;
    case MemoryLocation::DISK:
      os << "DISK";
      break;
    case MemoryLocation::GPU:
      os << "GPU";
      break;
    case MemoryLocation::CPU:
      os << "CPU";
      break;
    case MemoryLocation::REMOTE:
      os << "REMOTE";
      break;
    default:
      os << "Unknown(" << static_cast<int>(loc) << ")";
      break;
  }
  return os;
}

// Helper function for string conversion
inline std::string location_to_string(MemoryLocation loc) {
  switch (loc) {
    case MemoryLocation::NONE:
      return "NONE";
    case MemoryLocation::DISK:
      return "DISK";
    case MemoryLocation::GPU:
      return "GPU";
    case MemoryLocation::CPU:
      return "CPU";
    case MemoryLocation::REMOTE:
      return "REMOTE";
    default:
      return "Unknown(" + std::to_string(static_cast<int>(loc)) + ")";
  }
}

// Optional: Helper function if bitmask-like checks are still needed at a higher level,
// but generally prefer specific location checks.
// inline bool is_location_set(uint32_t current_mask, MemoryLocation single_loc) {
//    // ... implementation if needed ...
// }

} // namespace tensorcast::common::memory
