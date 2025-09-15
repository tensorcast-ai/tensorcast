// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstddef>

namespace tensorcast::common::consts {

// Canonical data granularity constants across the system.
// These are long-term defaults and may be adapted at runtime by controllers.

// Artifact layout chunk (VS/UMA granularity)
inline constexpr std::size_t kArtifactChunkDefault = 256ULL << 20; // 256 MiB

// Transfer slice/window (pinned buffer block size)
inline constexpr std::size_t kTxSliceDefault = 32ULL << 20; // 32 MiB

// Hash tree leaf size (protocol constant)
inline constexpr std::size_t kHashLeafBytes = 4ULL << 20; // 4 MiB

// Minimal I/O batch size recommendation (may vary per source)
inline constexpr std::size_t kIoBatchDefault = 8ULL << 20; // 8 MiB

// Alignment helpers
constexpr bool is_aligned(std::size_t value, std::size_t alignment) {
  return (value % alignment) == 0;
}

} // namespace tensorcast::common::consts
