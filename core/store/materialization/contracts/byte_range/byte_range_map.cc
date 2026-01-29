// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/contracts/byte_range/byte_range_map.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_hash.h"

namespace tensorcast::store::loader {

namespace {

constexpr std::string_view kMapFingerprintVersion = "byte_range_map_v1";

void append_u64(std::string* out, uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<char>(value & 0xFF));
    value >>= 8;
  }
}

void append_u32(std::string* out, uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<char>(value & 0xFF));
    value >>= 8;
  }
}

absl::Status validate_total_bytes(const ByteRangeMap& map) {
  if (map.total_bytes == 0) {
    if (!map.segments.empty()) {
      return absl::InvalidArgumentError("byte range map total_bytes is zero with non-empty segments");
    }
    return absl::OkStatus();
  }
  if (map.num_sources == 0) {
    return absl::InvalidArgumentError("byte range map num_sources must be > 0");
  }
  return absl::OkStatus();
}

} // namespace

std::string byte_range_fingerprint_hex(const ByteRangeFingerprint& fp) {
  static const char* kHex = "0123456789abcdef";
  std::string out;
  out.reserve(fp.bytes.size() * 2);
  for (uint8_t b : fp.bytes) {
    out.push_back(kHex[(b >> 4) & 0xF]);
    out.push_back(kHex[b & 0xF]);
  }
  return out;
}

absl::StatusOr<ByteRangeMap> normalize_byte_range_map(ByteRangeMap map) {
  if (auto st = validate_total_bytes(map); !st.ok()) {
    return st;
  }

  std::vector<ByteRangeSegment> filtered;
  filtered.reserve(map.segments.size());
  for (auto seg : map.segments) {
    if (seg.length == 0) {
      continue;
    }
    if (seg.kind == ByteRangeSegment::Kind::kPad) {
      seg.src_offset = 0;
      seg.source_index = 0;
    } else {
      if (map.num_sources == 0 || seg.source_index >= map.num_sources) {
        return absl::InvalidArgumentError("byte range map segment source_index out of range");
      }
    }
    filtered.push_back(seg);
  }

  if (map.total_bytes == 0) {
    map.segments.clear();
    return map;
  }

  std::sort(filtered.begin(), filtered.end(), [](const ByteRangeSegment& a, const ByteRangeSegment& b) {
    return a.dst_offset < b.dst_offset;
  });

  std::vector<ByteRangeSegment> merged;
  merged.reserve(filtered.size());
  for (const auto& seg : filtered) {
    if (seg.dst_offset >= map.total_bytes) {
      return absl::InvalidArgumentError("byte range map segment dst_offset out of bounds");
    }
    if (seg.length > map.total_bytes - seg.dst_offset) {
      return absl::InvalidArgumentError("byte range map segment length out of bounds");
    }
    if (merged.empty()) {
      merged.push_back(seg);
      continue;
    }
    auto& prev = merged.back();
    const uint64_t prev_end = prev.dst_offset + prev.length;
    if (seg.dst_offset < prev_end) {
      return absl::InvalidArgumentError("byte range map contains overlapping segments");
    }
    if (seg.dst_offset == prev_end) {
      const bool same_kind = seg.kind == prev.kind;
      const bool same_source = seg.kind == ByteRangeSegment::Kind::kPad ||
          (seg.source_index == prev.source_index && seg.src_offset == prev.src_offset + prev.length);
      if (same_kind && same_source) {
        prev.length += seg.length;
        continue;
      }
    }
    merged.push_back(seg);
  }

  uint64_t cursor = 0;
  for (const auto& seg : merged) {
    if (seg.dst_offset != cursor) {
      return absl::InvalidArgumentError("byte range map contains destination gaps");
    }
    cursor = seg.dst_offset + seg.length;
  }
  if (cursor != map.total_bytes) {
    return absl::InvalidArgumentError("byte range map does not cover total_bytes");
  }

  map.segments = std::move(merged);
  return map;
}

absl::StatusOr<ByteRangeFingerprint> fingerprint_byte_range_map(const ByteRangeMap& map) {
  if (auto st = validate_total_bytes(map); !st.ok()) {
    return st;
  }
  std::string payload;
  payload.reserve(map.segments.size() * 48 + 64);
  payload.append(kMapFingerprintVersion);
  payload.push_back('\0');
  append_u64(&payload, map.total_bytes);
  append_u32(&payload, map.num_sources);
  append_u32(&payload, static_cast<uint32_t>(map.segments.size()));

  for (const auto& seg : map.segments) {
    append_u32(&payload, static_cast<uint32_t>(seg.kind));
    append_u64(&payload, seg.dst_offset);
    append_u64(&payload, seg.length);
    append_u32(&payload, seg.source_index);
    append_u64(&payload, seg.src_offset);
  }

  std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size()));
  if (digest.size() < 16) {
    return absl::InternalError("sha256 digest too short for map fingerprint");
  }
  ByteRangeFingerprint fp;
  std::memcpy(fp.bytes.data(), digest.data(), fp.bytes.size());
  return fp;
}

} // namespace tensorcast::store::loader
