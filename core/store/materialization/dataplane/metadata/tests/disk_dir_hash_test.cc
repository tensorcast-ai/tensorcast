// Copyright (c) 2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"

namespace fs = std::filesystem;
using tensorcast::store::loader::compute_data_multihash_from_cpu_memory;
using tensorcast::store::loader::compute_data_multihash_from_disk_dir;
using tensorcast::testing::create_dummy_file;

TEST_CASE("disk dir hash orders multipart data numerically", "[disk][hash][ordering]") {
  const fs::path base = fs::temp_directory_path() / "tensorcast_disk_dir_hash_ordering";
  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(base, ec);
  REQUIRE(!ec);

  constexpr size_t part_size = 7;
  constexpr int part_count = 12;
  std::vector<char> expected;
  expected.reserve(part_size * part_count);

  for (int i = 0; i < part_count; ++i) {
    const auto path = base / ("tensor.data_" + std::to_string(i));
    const char start = static_cast<char>('A' + i);
    REQUIRE(create_dummy_file(path, part_size, start));
    for (size_t j = 0; j < part_size; ++j) {
      expected.push_back(static_cast<char>(start + (j % 26)));
    }
  }

  // Write minimal tensor_index.json to satisfy compute_data_multihash_from_disk_dir
  nlohmann::json idx = nlohmann::json::object();
  nlohmann::json entry = nlohmann::json::array();
  entry.push_back(0);
  entry.push_back(static_cast<uint64_t>(expected.size()));
  entry.push_back(nlohmann::json::array());
  entry.push_back(nlohmann::json::array());
  entry.push_back("torch.uint8");
  entry.push_back(0);
  idx["__dummy__"] = std::move(entry);
  {
    std::ofstream out(base / "tensor_index.json");
    REQUIRE(out.is_open());
    out << idx.dump();
  }

  auto expected_hash_or = compute_data_multihash_from_cpu_memory(expected.data(), expected.size());
  REQUIRE(expected_hash_or.ok());
  auto disk_hash_or = compute_data_multihash_from_disk_dir(base.string());
  REQUIRE(disk_hash_or.ok());
  REQUIRE(*disk_hash_or == *expected_hash_or);

  fs::remove_all(base, ec);
}
