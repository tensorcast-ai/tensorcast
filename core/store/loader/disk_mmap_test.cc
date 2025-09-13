// Copyright (c) 2025, TensorCast Team.

// Copyright (c) 2025, TensorCast Team.
// New test to verify DiskLoader zero-copy mmap path when partition sizes are page-aligned.

#include <catch2/catch_test_macros.hpp>
#include "core/testing/common.h"

#include <unistd.h>
#include <filesystem>

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::common::memory::VirtualAddressSpace;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::read_file_content;
using tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir;

TEST_CASE("DiskArtifact page-aligned load to CPU via mmap", "[replica][disk][cpu][mmap]") {
  // System page size
  const int64_t page_sz_long = static_cast<int64_t>(::sysconf(_SC_PAGESIZE));
  REQUIRE(page_sz_long > 0);
  const auto page_sz = static_cast<size_t>(page_sz_long);

  // Partition sizes (page aligned and > page size)
  const size_t size0 = page_sz * 2;
  const size_t size1 = page_sz * 3;
  const size_t total_size = size0 + size1;

  const std::string artifact_id = "aligned_cpu_artifact";
  const std::string artifact_dir_name = "aligned_cpu_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";

  fs::path base = fs::temp_directory_path() / "aligned_cpu_test";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / artifact_dir_name);

  fs::path path0 = base / artifact_dir_name / p0;
  fs::path path1 = base / artifact_dir_name / p1;
  REQUIRE(create_dummy_file(path0, size0, 'C'));
  REQUIRE(create_dummy_file(path1, size1, 'D'));

  // RFC-0007 metadata for standard partitions
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir_name).ok());

  // Combined expected data
  auto data0 = read_file_content(path0);
  auto data1 = read_file_content(path1);
  std::vector<char> combined;
  combined.reserve(total_size);
  combined.insert(combined.end(), data0.begin(), data0.end());
  combined.insert(combined.end(), data1.begin(), data1.end());

  // Pinned pool setup
  const size_t pool_total = 1024 * 1024 * 8; // 8 MiB
  const size_t pool_chunk = 1024;
  auto pool = std::make_shared<PinnedBufferPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  // Create VS
  auto virtual_addr_space = std::make_shared<VirtualAddressSpace>();

  // Create DiskSource
  DiskSource disk_src;
  disk_src.path = base / artifact_dir_name;

  // Use aggregate initialization for ReplicaConfig
  // Set max_buffer_bytes to match the available pool size
  ReplicaConfig cfg{
      .source = disk_src,
      .artifact_identifier = artifact_id,
      .device_type = ::tensorcast::DeviceType::CPU,
      .local_device_id = 0,
      .pinned_buffer_pool = pool,
      .virtual_addr_space = virtual_addr_space,
      .expected_artifact_size = total_size,
      .max_buffer_bytes = pool_total};

  auto mstatus = Replica::create(cfg);
  REQUIRE(mstatus.ok());
  auto replica = std::move(*mstatus);

  SECTION("Load via mmap and verify content") {
    auto fut = replica->ensure_loaded_async(MemoryLocation::CPU);
    REQUIRE(fut.valid());
    auto status = fut.get();
    REQUIRE(status.ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);

    auto ptrs = replica->get_data_pointer(MemoryLocation::CPU);
    REQUIRE_FALSE(ptrs.empty());

    // Lightweight verification: inspect first byte of the mapped region.
    REQUIRE(static_cast<char*>(ptrs.front())[0] == 'C');
  }

  // Teardown
  replica.reset();
  pool.reset();
  fs::remove_all(base);
}
