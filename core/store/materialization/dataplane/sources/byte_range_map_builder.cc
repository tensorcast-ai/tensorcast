// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::loader {

using nlohmann::json;

absl::StatusOr<ByteRangeMap> build_byte_range_map_from_canonical_index_json(
    std::string_view index_json,
    uint64_t total_size,
    uint64_t /*align_bytes*/) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }
  if (index_json.empty()) {
    return absl::InvalidArgumentError("index_json must not be empty");
  }
  json j;
  try {
    j = json::parse(index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", e.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }

  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  ranges.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    uint64_t off = arr[0].get<uint64_t>();
    uint64_t sz = arr[1].get<uint64_t>();
    ranges.emplace_back(off, sz);
  }
  std::sort(ranges.begin(), ranges.end(), [](auto& a, auto& b) { return a.first < b.first; });

  ByteRangeMap map;
  map.total_bytes = total_size;
  map.num_sources = 1;
  map.segments.reserve(ranges.size() * 2 + 1);

  uint64_t cur = 0;
  for (const auto& [off, sz] : ranges) {
    if (off > cur) {
      map.segments.push_back(
          ByteRangeSegment{
              .kind = ByteRangeSegment::Kind::kPad,
              .dst_offset = cur,
              .length = off - cur,
              .src_offset = 0,
              .source_index = 0,
          });
      cur = off;
    }
    if (sz > 0) {
      map.segments.push_back(
          ByteRangeSegment{
              .kind = ByteRangeSegment::Kind::kData,
              .dst_offset = off,
              .length = sz,
              .src_offset = off,
              .source_index = 0,
          });
      cur = off + sz;
    }
  }
  if (cur < total_size) {
    map.segments.push_back(
        ByteRangeSegment{
            .kind = ByteRangeSegment::Kind::kPad,
            .dst_offset = cur,
            .length = total_size - cur,
            .src_offset = 0,
            .source_index = 0,
        });
  }

  return normalize_byte_range_map(std::move(map));
}

} // namespace tensorcast::store::loader
