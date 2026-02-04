// Copyright (c) 2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <string>
#include <vector>

#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/loaders/disk_loader.h"

namespace fs = std::filesystem;
using tensorcast::store::DiskLoader;
using tensorcast::store::loading::DiskSource;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir;

TEST_CASE("DiskLoader orders multipart data numerically", "[loader][disk][ordering]") {
  const fs::path base = fs::temp_directory_path() / "tensorcast_disk_loader_ordering";
  std::error_code ec;
  fs::remove_all(base, ec);
  fs::create_directories(base, ec);
  REQUIRE(!ec);

  constexpr size_t part_size = 8;
  constexpr int part_count = 13; // triggers lexicographic trap (10 > 2)
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

  auto desc_status = write_rfc0007_descriptor_for_standard_artifact_dir(base);
  REQUIRE(desc_status.ok());

  DiskSource src;
  src.path = base;
  DiskLoader loader(src);
  REQUIRE(loader.initialize().ok());

  auto source_or = loader.open_source();
  REQUIRE(source_or.ok());
  auto source = std::move(*source_or);
  const uint64_t total_bytes = source->total_bytes();
  REQUIRE(total_bytes == expected.size());

  std::vector<char> buffer(total_bytes);
  auto read_or = source->read_at(0, buffer.data(), buffer.size());
  REQUIRE(read_or.ok());
  REQUIRE(read_or.value() == buffer.size());
  REQUIRE(buffer == expected);

  fs::remove_all(base, ec);
}
