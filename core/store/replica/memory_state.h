// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace tensorcast::store {

/**
 * @brief Represents the detailed state of memory allocation and loading for a specific location (CPU or GPU).
 */
enum class MemoryState : uint8_t {
  UNINITIALIZED = 0, // Before any action is taken.
  UNALLOCATED = 1, // Initialized but no memory buffer allocated yet.
  ALLOCATED = 2, // Memory buffer exists but contains no valid data.
  LOADING = 3, // Data transfer is in progress.
  LOADED = 4, // Data transfer completed successfully.
  FAILED = 5 // An error occurred during allocation or loading.
};

inline std::ostream& operator<<(std::ostream& os, const MemoryState& state) {
  switch (state) {
    case MemoryState::UNINITIALIZED:
      os << "UNINITIALIZED";
      break;
    case MemoryState::UNALLOCATED:
      os << "UNALLOCATED";
      break;
    case MemoryState::ALLOCATED:
      os << "ALLOCATED";
      break;
    case MemoryState::LOADING:
      os << "LOADING";
      break;
    case MemoryState::LOADED:
      os << "LOADED";
      break;
    case MemoryState::FAILED:
      os << "FAILED";
      break;
    default:
      // Handle potential unknown values, though with a scoped enum this is less likely
      // unless there's a cast from an integer.
      os << "UNKNOWN_MemoryState(" << static_cast<int>(state) << ")";
      break;
  }
  return os;
}

// Helper function for string conversion
inline std::string state_to_string(const MemoryState& state) {
  switch (state) {
    case MemoryState::UNINITIALIZED:
      return "UNINITIALIZED";
    case MemoryState::UNALLOCATED:
      return "UNALLOCATED";
    case MemoryState::ALLOCATED:
      return "ALLOCATED";
    case MemoryState::LOADING:
      return "LOADING";
    case MemoryState::LOADED:
      return "LOADED";
    case MemoryState::FAILED:
      return "FAILED";
    default:
      return "UNKNOWN";
  }
}

} // namespace tensorcast::store