// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <cstring>
#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using namespace tensorcast::store;
using namespace tensorcast::tests;

TEST_CASE("DiskArtifact get size and load to CPU", "[replica][disk][cpu]") {
  // Setup dummy replica with two partitions
  const std::string artifact_id = "basic_cpu_artifact";
  const std::string artifact_dir_name = "basic_cpu_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  // Use page-aligned sizes to trigger DVMP mmap path
  const size_t page_size = 4096; // Common page size
  const size_t size0 = page_size * 2; // 8192 bytes
  const size_t size1 = page_size * 3; // 12288 bytes
  const size_t total_size = size0 + size1;

  fs::path base = fs::temp_directory_path() / "basic_cpu_test";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / artifact_dir_name);

  fs::path path0 = base / artifact_dir_name / p0;
  fs::path path1 = base / artifact_dir_name / p1;
  REQUIRE(create_dummy_file(path0, size0, 'A'));
  REQUIRE(create_dummy_file(path1, size1, 'B'));

  // RFC-0007 metadata for standard partitions
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir_name).ok());

  // Read combined original data
  auto data0 = read_file_content(path0);
  auto data1 = read_file_content(path1);
  std::vector<char> combined;
  combined.reserve(total_size);
  combined.insert(combined.end(), data0.begin(), data0.end());
  combined.insert(combined.end(), data1.begin(), data1.end());
  REQUIRE(combined.size() == total_size);

  // Pinned pool setup
  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 1024;
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  // Create DVMP
  auto dvmp = std::make_shared<::tensorcast::memory::DistributedVirtualMemoryPool>();
  // Use new DiskSource
  DiskSource disk_src;
  disk_src.path = base / artifact_dir_name;

  // Use aggregate initialization for ReplicaConfig
  // Set max_buffer_bytes to match the available pool size
  ReplicaConfig cfg{
      .source = disk_src,
      .artifact_identifier = artifact_id,
      .device_type = ::tensorcast::DeviceType::CPU,
      .local_device_id = 0,
      .pinned_memory_pool = pool,
      .dvmp = dvmp,
      .expected_artifact_size = total_size,
      .max_buffer_bytes = pool_total};

  auto mstatus = Replica::create(cfg);
  REQUIRE(mstatus.ok());
  auto replica = std::move(*mstatus);

  SECTION("Get artifact size") {
    auto size_status = replica->get_artifact_size();
    REQUIRE(size_status.ok());
    REQUIRE(*size_status == total_size);
  }

  SECTION("Load to CPU and verify content") {
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    auto fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(fut.valid());
    auto wait_status = replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15));
    REQUIRE(wait_status.ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    // With DVMP, get_data_pointer returns a single pointer to the contiguous memory block
    auto ptrs = replica->get_data_pointer(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(ptrs.size() == 1);
    REQUIRE(ptrs[0] != nullptr);

    // The entire replica is now in a single contiguous memory block
    // Note: With DVMP and mmap-based loading, we need to be careful about accessing memory
    // that might be lazily mapped. Let's verify the content safely.
    char* data_ptr = static_cast<char*>(ptrs[0]);

    // Create a buffer to read into to avoid potential page faults
    std::vector<char> loaded(total_size);

    // Copy data from DVMP memory to our buffer
    // This ensures we trigger any page faults in a controlled manner
    std::memcpy(loaded.data(), data_ptr, total_size);

    REQUIRE(loaded == combined);
  }

  SECTION("Release CPU memory") {
    replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU).wait();
    replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).IgnoreError();
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::LOADED);
    auto status = replica->release_memory(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(status.ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    auto ptrs_after = replica->get_data_pointer(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(ptrs_after.empty());
  }

  // Teardown
  replica.reset();
  pool.reset();
  fs::remove_all(base);
}