// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>

#include "absl/status/status.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir;

TEST_CASE("DiskArtifact creation errors", "[replica][disk][error]") {
  const std::string artifact_id = "error_artifact";
  fs::path base = fs::temp_directory_path() / "error_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base);

  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 256 * 1024;
  auto pool = std::make_shared<PinnedBufferPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Non-existent subdirectory") {
    // Use new DiskSource
    DiskSource disk_src;
    disk_src.path = base / "no_such_dir";

    // Use aggregate initialization for ReplicaConfig
    // Set max_buffer_bytes to match the available pool size
    ReplicaConfig cfg{
        .source = disk_src,
        .artifact_identifier = artifact_id,
        .device_type = ::tensorcast::DeviceType::CPU,
        .local_device_id = 0,
        .pinned_buffer_pool = pool,
        .max_buffer_bytes = pool_total};

    auto mstatus = Replica::create(cfg);
    REQUIRE(!mstatus.ok());
    REQUIRE_THAT(mstatus.status().ToString(), ContainsSubstring("Replica directory not found"));
  }

  SECTION("Mismatched expected size") {
    const std::string subdir = "mismatch_dir";
    fs::create_directories(base / subdir);
    fs::path file0 = base / subdir / "tensor.data_0";
    // Create a file larger than the 1 KiB minimum to pass DiskLoader's size validation.
    REQUIRE(create_dummy_file(file0, 2048, 'Z'));

    // RFC-0007: standard partition directories must include descriptor + canonical index
    REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / subdir).ok());

    // Use new DiskSource with empty directory
    DiskSource disk_src;
    disk_src.path = base / subdir;

    // Use aggregate initialization for ReplicaConfig
    ReplicaConfig cfg{
        .source = disk_src,
        .artifact_identifier = artifact_id,
        .device_type = ::tensorcast::DeviceType::CPU,
        .local_device_id = 0,
        .pinned_buffer_pool = pool,
        .expected_artifact_size = 1024 // wrong expected size
    };

    auto mstatus = Replica::create(cfg);
    REQUIRE(!mstatus.ok());
    REQUIRE_THAT(mstatus.status().ToString(), ContainsSubstring("Artifact size mismatch"));
  }

  // Teardown
  pool.reset();
  fs::remove_all(base);
}
