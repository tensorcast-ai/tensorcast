// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <cstdint>
#include "absl/status/status.h"

namespace tensorcast::store {

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

// -------------------------------------------------------------------------
// State transition utilities
// -------------------------------------------------------------------------

/**
 * @brief Compile-time check for whether a transition is allowed by policy.
 *
 * This function encodes the expected state machine to keep policy centralized
 * and cheap to query. It does not perform any atomic operation.
 */
constexpr bool is_valid_chunk_transition(ChunkState from, ChunkState to) noexcept {
  switch (from) {
    case ChunkState::HOT:
      return (
          to == ChunkState::LOCKED_TX || to == ChunkState::COLD || to == ChunkState::EVICTED ||
          to == ChunkState::PREEMPTIBLE || to == ChunkState::HOT);
    case ChunkState::LOCKED_TX:
      return (to == ChunkState::HOT || to == ChunkState::COPIED_GPU || to == ChunkState::LOCKED_TX);
    case ChunkState::COPIED_GPU:
      return (to == ChunkState::EVICTED || to == ChunkState::LOCKED_TX || to == ChunkState::COPIED_GPU);
    case ChunkState::COLD:
      return (
          to == ChunkState::LOCKED_TX || to == ChunkState::EVICTED || to == ChunkState::PREEMPTIBLE ||
          to == ChunkState::COLD);
    case ChunkState::EVICTED:
      return (to == ChunkState::HOT || to == ChunkState::EVICTED);
    case ChunkState::PREEMPTIBLE:
      return (to == ChunkState::LOCKED_TX || to == ChunkState::EVICTED || to == ChunkState::PREEMPTIBLE);
    default:
      return false;
  }
}

/**
 * @brief Try to transition a chunk's state with policy validation.
 *
 * Preconditions:
 * - `meta.state` may change concurrently; this function will re-check the
 *   policy against the observed current state each time the CAS fails.
 *
 * Behavior:
 * - If `to` equals the current state, returns OK without changing state.
 * - If the transition is disallowed by policy, returns InvalidArgument.
 * - If CAS repeatedly fails due to contention but the transition remains
 *   valid, it will retry until success or until the observed state makes the
 *   transition invalid, in which case Aborted is returned.
 */
inline absl::Status try_transition_chunk_state(ChunkMeta& meta, ChunkState to) noexcept {
  ChunkState current = meta.state.load(std::memory_order_acquire);

  if (to == current) {
    return absl::OkStatus();
  }

  if (!is_valid_chunk_transition(current, to)) {
    return absl::InvalidArgumentError("invalid chunk state transition");
  }

  while (true) {
    ChunkState expected = current;
    if (meta.state.compare_exchange_weak(expected, to, std::memory_order_acq_rel)) {
      return absl::OkStatus();
    }
    // CAS failed; expected now holds the observed value.
    current = expected;
    if (to == current) {
      return absl::OkStatus();
    }
    if (!is_valid_chunk_transition(current, to)) {
      return absl::AbortedError("transition became invalid due to concurrent state change");
    }
  }
}

} // namespace tensorcast::store