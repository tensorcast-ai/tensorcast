// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace stepcast::store {

/**
 * @brief Specifies the target or current location of model data.
 * Unlike the old bitmask, this represents a single distinct location.
 */
enum class ModelLocation : uint8_t {
  NONE = 0,
  DISK, // Represents data stored persistently on disk.
  GPU, // Represents data loaded into GPU device memory.
  REMOTE, // Represents data accessible via RDMA (source only for now).
  PAGEABLE_CPU // Represents data loaded into Pageable-Chunk CPU cache (UPC-Cache).
};

// Add operator<< overload for ModelLocation
inline std::ostream& operator<<(std::ostream& os, ModelLocation loc) {
  switch (loc) {
    case ModelLocation::NONE:
      os << "NONE";
      break;
    case ModelLocation::DISK:
      os << "DISK";
      break;
    case ModelLocation::GPU:
      os << "GPU";
      break;
    case ModelLocation::REMOTE:
      os << "REMOTE";
      break;
    case ModelLocation::PAGEABLE_CPU:
      os << "PAGEABLE_CPU";
      break;
    default:
      os << "Unknown(" << static_cast<int>(loc) << ")";
      break;
  }
  return os;
}

// Helper function for string conversion
inline std::string location_to_string(ModelLocation loc) {
  switch (loc) {
    case ModelLocation::NONE:
      return "NONE";
    case ModelLocation::DISK:
      return "DISK";
    case ModelLocation::GPU:
      return "GPU";
    case ModelLocation::REMOTE:
      return "REMOTE";
    case ModelLocation::PAGEABLE_CPU:
      return "PAGEABLE_CPU";
    default:
      return "Unknown(" + std::to_string(static_cast<int>(loc)) + ")";
  }
}

// Optional: Helper function if bitmask-like checks are still needed at a higher level,
// but generally prefer specific location checks.
// inline bool is_location_set(uint32_t current_mask, ModelLocation single_loc) {
//    // ... implementation if needed ...
// }

} // namespace stepcast::store