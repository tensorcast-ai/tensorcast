// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/statusor.h"

namespace tensorcast::store::loader {

struct ByteRangeSegment {
  enum class Kind : uint8_t { kData = 0, kPad = 1 };

  Kind kind{Kind::kPad};
  uint64_t dst_offset{0};
  uint64_t length{0};
  uint64_t src_offset{0};
  uint32_t source_index{0};
};

struct ByteRangeMap {
  uint64_t total_bytes{0};
  uint32_t num_sources{0};
  std::vector<ByteRangeSegment> segments;
};

struct ByteRangeFingerprint {
  std::array<uint8_t, 16> bytes{};

  bool operator==(const ByteRangeFingerprint& other) const = default;
};

template <typename H>
H AbslHashValue(H h, const ByteRangeFingerprint& fp) {
  return H::combine_contiguous(std::move(h), fp.bytes.data(), fp.bytes.size());
}

[[nodiscard]] std::string byte_range_fingerprint_hex(const ByteRangeFingerprint& fp);

// Normalize a raw byte-range map into canonical form.
absl::StatusOr<ByteRangeMap> normalize_byte_range_map(ByteRangeMap map);

// Fingerprint a normalized map. The map must be normalized before calling.
absl::StatusOr<ByteRangeFingerprint> fingerprint_byte_range_map(const ByteRangeMap& map);

} // namespace tensorcast::store::loader
