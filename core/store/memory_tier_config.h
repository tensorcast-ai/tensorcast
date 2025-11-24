// Copyright (c) 2025, TensorCast Team.
//
// Stable/preemptible memory tier settings shared across daemon components.

#pragma once

#include <cstdint>

namespace tensorcast::store {

struct MemoryTierConfig {
  bool enable_preemptible_memory{false};
  uint64_t stable_bytes{0};
  uint64_t preemptible_limit_bytes{0};
  double preemptible_low_watermark_ratio{0.4};
};

} // namespace tensorcast::store
