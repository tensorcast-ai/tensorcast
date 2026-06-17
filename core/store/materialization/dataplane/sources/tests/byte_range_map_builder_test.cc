// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"

#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

namespace tensorcast::store::loader {

namespace {

std::string make_index_json(const std::vector<std::tuple<std::string, uint64_t, uint64_t>>& entries) {
  nlohmann::json out = nlohmann::json::object();
  for (const auto& [name, offset, size] : entries) {
    out[name] = nlohmann::json::array();
    out[name].push_back(offset);
    out[name].push_back(size);
    out[name].push_back(nlohmann::json::array({1}));
    out[name].push_back(nlohmann::json::array({1}));
    out[name].push_back("torch.uint8");
    out[name].push_back(0);
  }
  return out.dump();
}

} // namespace

TEST_CASE("build_byte_range_map_from_canonical_and_source_index_json maps offsets", "[byte_range_map]") {
  const std::string canonical = make_index_json({{"a", 0, 8}, {"b", 16, 8}});
  const std::string source = make_index_json({{"a", 16, 8}, {"b", 0, 8}});

  auto map_or = build_byte_range_map_from_canonical_and_source_index_json(canonical, source, /*total_size=*/24);
  REQUIRE(map_or.ok());
  const auto& map = *map_or;
  REQUIRE(map.segments.size() == 3);
  CHECK(map.segments[0].kind == ByteRangeSegment::Kind::kData);
  CHECK(map.segments[0].dst_offset == 0);
  CHECK(map.segments[0].src_offset == 16);
  CHECK(map.segments[1].kind == ByteRangeSegment::Kind::kPad);
  CHECK(map.segments[1].dst_offset == 8);
  CHECK(map.segments[1].length == 8);
  CHECK(map.segments[2].kind == ByteRangeSegment::Kind::kData);
  CHECK(map.segments[2].dst_offset == 16);
  CHECK(map.segments[2].src_offset == 0);
}

TEST_CASE("build_byte_range_map_from_canonical_index_json coalesces shared storage ranges", "[byte_range_map]") {
  const std::string canonical = make_index_json({{"a", 0, 64}, {"b", 0, 64}, {"c", 32, 32}});

  auto map_or = build_byte_range_map_from_canonical_index_json(canonical, /*total_size=*/64);
  REQUIRE(map_or.ok());
  const auto& map = *map_or;
  REQUIRE(map.segments.size() == 1);
  CHECK(map.segments[0].kind == ByteRangeSegment::Kind::kData);
  CHECK(map.segments[0].dst_offset == 0);
  CHECK(map.segments[0].src_offset == 0);
  CHECK(map.segments[0].length == 64);
}

TEST_CASE(
    "build_byte_range_map_from_canonical_and_source_index_json coalesces shared storage ranges",
    "[byte_range_map]") {
  const std::string canonical = make_index_json({{"a", 0, 64}, {"b", 0, 64}, {"c", 32, 32}});
  const std::string source = make_index_json({{"a", 128, 64}, {"b", 128, 64}, {"c", 160, 32}});

  auto map_or = build_byte_range_map_from_canonical_and_source_index_json(canonical, source, /*total_size=*/64);
  REQUIRE(map_or.ok());
  const auto& map = *map_or;
  REQUIRE(map.segments.size() == 1);
  CHECK(map.segments[0].kind == ByteRangeSegment::Kind::kData);
  CHECK(map.segments[0].dst_offset == 0);
  CHECK(map.segments[0].src_offset == 128);
  CHECK(map.segments[0].length == 64);
}

TEST_CASE("compose_byte_range_maps builds view->source map", "[byte_range_map]") {
  ByteRangeMap view_map;
  view_map.total_bytes = 8;
  view_map.num_sources = 1;
  view_map.segments.push_back(
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 16,
          .source_index = 0,
      });

  const std::string canonical = make_index_json({{"a", 0, 8}, {"b", 16, 8}});
  const std::string source = make_index_json({{"a", 16, 8}, {"b", 0, 8}});
  auto map_or = build_byte_range_map_from_canonical_and_source_index_json(canonical, source, /*total_size=*/24);
  REQUIRE(map_or.ok());

  auto composed_or = compose_byte_range_maps(view_map, *map_or);
  REQUIRE(composed_or.ok());
  const auto& composed = *composed_or;
  REQUIRE(composed.segments.size() == 1);
  CHECK(composed.segments[0].kind == ByteRangeSegment::Kind::kData);
  CHECK(composed.segments[0].dst_offset == 0);
  CHECK(composed.segments[0].src_offset == 0);
  CHECK(composed.segments[0].length == 8);
}

TEST_CASE("compose_byte_range_maps supports packed reorder views", "[byte_range_map]") {
  // Canonical: a @0, b @16. Source swaps a/b offsets.
  const std::string canonical = make_index_json({{"a", 0, 8}, {"b", 16, 8}});
  const std::string source = make_index_json({{"a", 16, 8}, {"b", 0, 8}});
  auto canonical_to_source_or =
      build_byte_range_map_from_canonical_and_source_index_json(canonical, source, /*total_size=*/24);
  REQUIRE(canonical_to_source_or.ok());

  // Packed reorder view: [b, a] in destination order.
  ByteRangeMap view_map;
  view_map.total_bytes = 16;
  view_map.num_sources = 1;
  view_map.segments = {
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 8,
          .src_offset = 16,
          .source_index = 0,
      },
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 8,
          .length = 8,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  auto composed_or = compose_byte_range_maps(view_map, *canonical_to_source_or);
  REQUIRE(composed_or.ok());
  const auto& composed = *composed_or;
  REQUIRE(composed.total_bytes == 16);
  REQUIRE(composed.segments.size() == 2);
  CHECK(composed.segments[0].kind == ByteRangeSegment::Kind::kData);
  CHECK(composed.segments[0].dst_offset == 0);
  CHECK(composed.segments[0].src_offset == 0);
  CHECK(composed.segments[0].length == 8);
  CHECK(composed.segments[1].kind == ByteRangeSegment::Kind::kData);
  CHECK(composed.segments[1].dst_offset == 8);
  CHECK(composed.segments[1].src_offset == 16);
  CHECK(composed.segments[1].length == 8);
}

} // namespace tensorcast::store::loader
