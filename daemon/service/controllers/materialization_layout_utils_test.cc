// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_layout_utils.h"

#include <catch2/catch_test_macros.hpp>

#include <future>
#include <memory>
#include <string>
#include <vector>

namespace {

using tensorcast::daemon::materialization_layout::dtype_element_size;
using tensorcast::daemon::materialization_layout::parse_canonical_index_shared;
using tensorcast::daemon::materialization_layout::parse_canonical_index_shared_with_identity;

std::string make_index_json(int tensor_count, int offset_delta) {
  std::string json = "{";
  for (int i = 0; i < tensor_count; ++i) {
    if (i > 0) {
      json += ",";
    }
    const int offset = i * offset_delta;
    json += "\"tensor_" + std::to_string(i) + "\":[" + std::to_string(offset) + "," + std::to_string(offset_delta) +
        ",[" + std::to_string(offset_delta) + "],[1],\"torch.uint8\",0]";
  }
  json += "}";
  return json;
}

TEST_CASE("dtype_element_size supports float8 families", "[daemon][materialization_layout]") {
  CHECK(dtype_element_size("torch.float8_e4m3fn").value() == 1);
  CHECK(dtype_element_size("torch.float8_e5m2").value() == 1);
  CHECK(dtype_element_size("torch.float8_e4m3fnuz").value() == 1);
  CHECK(dtype_element_size("torch.float8_e5m2fnuz").value() == 1);
  CHECK(dtype_element_size("torch.float8_e8m0fnu").value() == 1);
}

TEST_CASE("parse_canonical_index_shared coalesces concurrent same-key parses", "[daemon][materialization_layout]") {
  const std::string index_json = make_index_json(/*tensor_count=*/128, /*offset_delta=*/16);

  std::vector<std::future<
      absl::StatusOr<std::shared_ptr<const tensorcast::daemon::materialization_layout::CanonicalIndexTable>>>>
      futures;
  futures.reserve(16);
  for (int i = 0; i < 16; ++i) {
    futures.push_back(std::async(std::launch::async, [&]() { return parse_canonical_index_shared(index_json); }));
  }

  std::shared_ptr<const tensorcast::daemon::materialization_layout::CanonicalIndexTable> first;
  for (auto& future : futures) {
    auto table_or = future.get();
    REQUIRE(table_or.ok());
    REQUIRE((*table_or)->entries.size() == 128);
    REQUIRE((*table_or)->logical_total_size == 128 * 16);
    if (first == nullptr) {
      first = *table_or;
    } else {
      CHECK(*table_or == first);
    }
  }

  auto cached_or = parse_canonical_index_shared(index_json);
  REQUIRE(cached_or.ok());
  CHECK(*cached_or == first);
}

TEST_CASE(
    "parse_canonical_index_shared_with_identity reuses validated identity keys",
    "[daemon][materialization_layout]") {
  const std::string index_json = make_index_json(/*tensor_count=*/16, /*offset_delta=*/32);
  auto first_or = parse_canonical_index_shared_with_identity(index_json, "artifact-a:canonical-index-a");
  REQUIRE(first_or.ok());
  auto second_or = parse_canonical_index_shared_with_identity(index_json, "artifact-a:canonical-index-a");
  REQUIRE(second_or.ok());
  CHECK(*second_or == *first_or);
}

TEST_CASE(
    "parse_canonical_index_shared_with_identity fails closed on identity mismatch",
    "[daemon][materialization_layout]") {
  const std::string first_json = make_index_json(/*tensor_count=*/2, /*offset_delta=*/16);
  const std::string second_json = make_index_json(/*tensor_count=*/3, /*offset_delta=*/16);
  auto first_or = parse_canonical_index_shared_with_identity(first_json, "artifact-b:canonical-index-b");
  REQUIRE(first_or.ok());
  auto second_or = parse_canonical_index_shared_with_identity(second_json, "artifact-b:canonical-index-b");
  REQUIRE_FALSE(second_or.ok());
}

} // namespace
