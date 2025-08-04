// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "tests/cpp/common.h"

#include <filesystem>

#include "absl/status/status.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using namespace stepcast::tests;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("DiskModel creation errors", "[model][disk][error]") {
  const std::string model_id = "error_model";
  fs::path base = fs::temp_directory_path() / "error_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base);

  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 256 * 1024;
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Non-existent subdirectory") {
    ModelConfig cfg;
    cfg.model_identifier = model_id;

    // Use new DiskSource
    DiskSource disk_src;
    disk_src.path = base / "no_such_dir";
    cfg.source = disk_src;

    cfg.pinned_memory_pool = pool;
    cfg.local_device_id = 0;

    auto mstatus = Model::create(cfg);
    REQUIRE(!mstatus.ok());
    REQUIRE_THAT(mstatus.status().ToString(), ContainsSubstring("Model directory not found"));
  }

  SECTION("Mismatched expected size") {
    const std::string subdir = "mismatch_dir";
    fs::create_directories(base / subdir);
    fs::path file0 = base / subdir / "tensor.data_0";
    // Create a file larger than the 1 KiB minimum to pass DiskLoader's size validation.
    REQUIRE(create_dummy_file(file0, 2048, 'Z'));

    ModelConfig cfg;
    cfg.model_identifier = model_id;

    // Use new DiskSource with empty directory
    DiskSource disk_src;
    disk_src.path = base / subdir;
    cfg.source = disk_src;

    cfg.pinned_memory_pool = pool;
    cfg.local_device_id = 0;
    cfg.expected_model_size = 1024; // wrong expected size

    auto mstatus = Model::create(cfg);
    REQUIRE(!mstatus.ok());
    REQUIRE_THAT(mstatus.status().ToString(), ContainsSubstring("Model size mismatch"));
  }

  // Teardown
  pool.reset();
  fs::remove_all(base);
}