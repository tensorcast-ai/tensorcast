// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>

namespace tensorcast::store {

enum class StableLocalBackingKind {
  kNone = 0,
  kHostSharedRegion = 1,
};

struct StableLocalBackingRef {
  StableLocalBackingKind kind = StableLocalBackingKind::kNone;
  std::string backing_id;
  uint64_t backing_base_addr = 0;
  uint64_t backing_bytes = 0;
  uint64_t slot_bytes = 0;
  int dev_type = 0;
  int dev_id = 0;

  bool operator==(const StableLocalBackingRef& other) const = default;
};

} // namespace tensorcast::store
