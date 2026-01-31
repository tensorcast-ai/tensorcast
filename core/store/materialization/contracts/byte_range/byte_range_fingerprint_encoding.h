// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>

namespace tensorcast::store::loader::byte_range_fingerprint_internal {

inline void append_u64(std::string* out, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<char>(value & 0xFF));
    value >>= 8;
  }
}

inline void append_u32(std::string* out, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<char>(value & 0xFF));
    value >>= 8;
  }
}

} // namespace tensorcast::store::loader::byte_range_fingerprint_internal
