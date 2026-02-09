// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "nlohmann/json.hpp"

namespace tensorcast::store::loader {

using nlohmann::json;

struct IndexEntry {
  uint64_t offset{0};
  uint64_t size{0};
};

absl::StatusOr<std::unordered_map<std::string, IndexEntry>> parse_index_entries(std::string_view index_json) {
  if (index_json.empty()) {
    return absl::InvalidArgumentError("index_json must not be empty");
  }
  json j;
  try {
    j = json::parse(index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse index JSON: ", e.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("index JSON must be an object");
  }
  std::unordered_map<std::string, IndexEntry> entries;
  entries.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      return absl::InvalidArgumentError("index entry must be array [offset,size,...]");
    }
    IndexEntry entry;
    entry.offset = arr[0].get<uint64_t>();
    entry.size = arr[1].get<uint64_t>();
    entries.emplace(it.key(), entry);
  }
  return entries;
}

uint64_t compute_total_size_from_entries(const std::unordered_map<std::string, IndexEntry>& entries) {
  uint64_t total = 0;
  for (const auto& [_, entry] : entries) {
    total = std::max<uint64_t>(total, entry.offset + entry.size);
  }
  return total;
}

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

absl::StatusOr<ByteRangeMap> build_byte_range_map_from_canonical_and_source_index_json(
    std::string_view canonical_index_json,
    std::string_view source_index_json,
    uint64_t total_size,
    uint64_t /*align_bytes*/) {
  if (total_size == 0) {
    return absl::InvalidArgumentError("total_size must be > 0");
  }

  auto canonical_or = parse_index_entries(canonical_index_json);
  if (!canonical_or.ok()) {
    return canonical_or.status();
  }
  auto source_or = parse_index_entries(source_index_json);
  if (!source_or.ok()) {
    return source_or.status();
  }
  const auto& canonical = *canonical_or;
  const auto& source = *source_or;

  if (canonical.size() != source.size()) {
    return absl::InvalidArgumentError("canonical and source index tensor counts differ");
  }
  for (const auto& [name, entry] : canonical) {
    auto it = source.find(name);
    if (it == source.end()) {
      return absl::InvalidArgumentError("canonical/source index tensor name mismatch");
    }
    if (entry.size != it->second.size) {
      return absl::InvalidArgumentError("canonical/source index tensor size mismatch");
    }
  }

  const uint64_t source_total_size = compute_total_size_from_entries(source);
  if (source_total_size == 0) {
    return absl::InvalidArgumentError("source index total_size is zero");
  }

  struct NamedRange {
    std::string name;
    uint64_t offset{0};
    uint64_t size{0};
  };

  std::vector<NamedRange> ranges;
  ranges.reserve(canonical.size());
  for (const auto& [name, entry] : canonical) {
    ranges.push_back(NamedRange{name, entry.offset, entry.size});
  }
  std::sort(ranges.begin(), ranges.end(), [](const NamedRange& a, const NamedRange& b) { return a.offset < b.offset; });

  ByteRangeMap map;
  map.total_bytes = total_size;
  map.num_sources = 1;
  map.segments.reserve(ranges.size() * 2 + 1);

  uint64_t cur = 0;
  for (const auto& range : ranges) {
    if (range.offset > cur) {
      map.segments.push_back(
          ByteRangeSegment{
              .kind = ByteRangeSegment::Kind::kPad,
              .dst_offset = cur,
              .length = range.offset - cur,
              .src_offset = 0,
              .source_index = 0,
          });
      cur = range.offset;
    }
    if (range.size > 0) {
      const auto& src_entry = source.at(range.name);
      if (src_entry.offset > source_total_size || src_entry.size > source_total_size - src_entry.offset) {
        return absl::InvalidArgumentError("source index offsets out of bounds");
      }
      map.segments.push_back(
          ByteRangeSegment{
              .kind = ByteRangeSegment::Kind::kData,
              .dst_offset = range.offset,
              .length = range.size,
              .src_offset = src_entry.offset,
              .source_index = 0,
          });
      cur = range.offset + range.size;
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

absl::StatusOr<ByteRangeMap> compose_byte_range_maps(const ByteRangeMap& outer, const ByteRangeMap& inner) {
  if (outer.total_bytes == 0) {
    return ByteRangeMap{};
  }
  if (inner.total_bytes == 0) {
    return absl::InvalidArgumentError("inner map total_bytes is zero");
  }
  auto outer_or = normalize_byte_range_map(outer);
  if (!outer_or.ok()) {
    return outer_or.status();
  }
  auto inner_or = normalize_byte_range_map(inner);
  if (!inner_or.ok()) {
    return inner_or.status();
  }
  const auto& outer_map = *outer_or;
  const auto& inner_map = *inner_or;

  ByteRangeMap composed;
  composed.total_bytes = outer_map.total_bytes;
  composed.num_sources = inner_map.num_sources;
  composed.segments.reserve(outer_map.segments.size() + inner_map.segments.size());
  const auto& inner_segments = inner_map.segments;

  std::vector<uint64_t> inner_starts;
  inner_starts.reserve(inner_segments.size());
  for (const auto& seg : inner_segments) {
    inner_starts.push_back(seg.dst_offset);
  }

  auto find_inner_index = [&](uint64_t canonical_offset) -> absl::StatusOr<size_t> {
    if (inner_segments.empty()) {
      return absl::InvalidArgumentError("inner map has no segments");
    }
    if (canonical_offset >= inner_map.total_bytes) {
      return absl::InvalidArgumentError("outer map references canonical offsets beyond inner map total_bytes");
    }
    auto it = std::upper_bound(inner_starts.begin(), inner_starts.end(), canonical_offset);
    size_t idx = 0;
    if (it == inner_starts.begin()) {
      idx = 0;
    } else {
      idx = static_cast<size_t>(std::distance(inner_starts.begin(), std::prev(it)));
    }
    if (idx >= inner_segments.size()) {
      return absl::InvalidArgumentError("inner map segment lookup out of range");
    }
    const auto& seg = inner_segments[idx];
    if (canonical_offset < seg.dst_offset || canonical_offset >= seg.dst_offset + seg.length) {
      return absl::InvalidArgumentError("inner map does not cover canonical range for composition");
    }
    return idx;
  };

  for (const auto& outer_seg : outer_map.segments) {
    if (outer_seg.length == 0) {
      continue;
    }
    if (outer_seg.kind == ByteRangeSegment::Kind::kPad) {
      composed.segments.push_back(
          ByteRangeSegment{
              .kind = ByteRangeSegment::Kind::kPad,
              .dst_offset = outer_seg.dst_offset,
              .length = outer_seg.length,
              .src_offset = 0,
              .source_index = 0,
          });
      continue;
    }

    const uint64_t canonical_begin = outer_seg.src_offset;
    const uint64_t canonical_end = canonical_begin + outer_seg.length;
    if (canonical_end > inner_map.total_bytes) {
      return absl::InvalidArgumentError("outer map references canonical offsets beyond inner map total_bytes");
    }

    uint64_t remaining = outer_seg.length;
    uint64_t dst_cursor = outer_seg.dst_offset;
    uint64_t canonical_cursor = canonical_begin;

    while (remaining > 0) {
      auto idx_or = find_inner_index(canonical_cursor);
      if (!idx_or.ok()) {
        return idx_or.status();
      }
      const auto& inner_seg = inner_segments[*idx_or];
      const uint64_t inner_offset = canonical_cursor - inner_seg.dst_offset;
      const uint64_t available = inner_seg.length - inner_offset;
      const uint64_t take = std::min<uint64_t>(remaining, available);

      if (inner_seg.kind == ByteRangeSegment::Kind::kPad) {
        composed.segments.push_back(
            ByteRangeSegment{
                .kind = ByteRangeSegment::Kind::kPad,
                .dst_offset = dst_cursor,
                .length = take,
                .src_offset = 0,
                .source_index = 0,
            });
      } else {
        composed.segments.push_back(
            ByteRangeSegment{
                .kind = ByteRangeSegment::Kind::kData,
                .dst_offset = dst_cursor,
                .length = take,
                .src_offset = inner_seg.src_offset + inner_offset,
                .source_index = inner_seg.source_index,
            });
      }

      remaining -= take;
      dst_cursor += take;
      canonical_cursor += take;
    }
  }

  return normalize_byte_range_map(std::move(composed));
}

} // namespace tensorcast::store::loader
