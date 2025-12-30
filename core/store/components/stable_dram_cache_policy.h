// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace tensorcast::store::components {

enum class StableRetentionPolicy : std::uint8_t {
  kBestEffort = 0,
  kTtl = 1,
  kPinned = 2,
};

enum class StableOverflowPolicy : std::uint8_t {
  kEvict = 0,
  kSpill = 1,
  kReject = 2,
};

struct StableDramCachePolicy {
  StableRetentionPolicy retention_policy{StableRetentionPolicy::kBestEffort};
  std::optional<std::chrono::milliseconds> retention_ttl;
  StableOverflowPolicy overflow_policy{StableOverflowPolicy::kEvict};
  bool required{false};
  bool require_shared_disk_for_spill{false};
  bool require_remote_stable_for_spill{false};
};

} // namespace tensorcast::store::components
