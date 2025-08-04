// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <atomic>
#include <cstdint>

namespace stepcast::store {

// clang-format off
/**
 * @brief Chunk life-cycle state used by DVMP and memory subsystems.
 *
 *  PREEMPTIBLE indicates that the underlying CPU pages have been
 *  marked with MADV_FREE / MADV_PAGEOUT and can be reclaimed by the
 *  kernel under memory pressure without prior write-back cost.
 */
enum class ChunkState : uint8_t {
    HOT,         ///< Resident, actively used or not yet transferred
    LOCKED_TX,   ///< Locked for H2D or P2P transfer
    COPIED_GPU,  ///< Fully copied to GPU memory
    COLD,        ///< Resident but eligible for eviction
    EVICTED,     ///< Pages evicted (MADV_PAGEOUT) – need reload on access
    PREEMPTIBLE  ///< Resident pages are preemptible via MADV_FREE/PAGEOUT
};
// clang-format on

// -------------------------------------------------------------------------
// Human-readable representation helper
// -------------------------------------------------------------------------

inline const char* chunk_state_to_string(ChunkState s) noexcept {
  switch (s) {
    case ChunkState::HOT:
      return "HOT";
    case ChunkState::LOCKED_TX:
      return "LOCKED_TX";
    case ChunkState::COPIED_GPU:
      return "COPIED_GPU";
    case ChunkState::COLD:
      return "COLD";
    case ChunkState::EVICTED:
      return "EVICTED";
    case ChunkState::PREEMPTIBLE:
      return "PREEMPTIBLE";
    default:
      return "UNKNOWN";
  }
}

struct ChunkMeta {
  std::atomic<ChunkState> state{ChunkState::COLD};
  std::atomic<uint32_t> last_touch_s{0}; ///< Last access heartbeat (seconds)
};

} // namespace stepcast::store