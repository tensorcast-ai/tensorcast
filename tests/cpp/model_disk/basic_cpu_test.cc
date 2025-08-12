// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "tests/cpp/common.h"

#include <cstring>
#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using namespace stepcast::tests;

TEST_CASE("DiskModel get size and load to CPU", "[model][disk][cpu]") {
  // Setup dummy model with two partitions
  const std::string model_id = "basic_cpu_model";
  const std::string model_subdir = "basic_cpu_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  // Use page-aligned sizes to trigger DVMP mmap path
  const size_t page_size = 4096; // Common page size
  const size_t size0 = page_size * 2; // 8192 bytes
  const size_t size1 = page_size * 3; // 12288 bytes
  const size_t total_size = size0 + size1;

  fs::path base = fs::temp_directory_path() / "basic_cpu_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base / model_subdir);

  fs::path path0 = base / model_subdir / p0;
  fs::path path1 = base / model_subdir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'A'));
  REQUIRE(create_dummy_file(path1, size1, 'B'));

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
  auto dvmp = std::make_shared<::stepcast::memory::DistributedVirtualMemoryPool>();
  // Use new DiskSource
  DiskSource disk_src;
  disk_src.path = base / model_subdir;

  // Use aggregate initialization for ModelConfig
  // Set max_buffer_bytes to match the available pool size
  ModelConfig cfg{
      .source = disk_src,
      .model_identifier = model_id,
      .device_type = ::stepcast::DeviceType::CPU,
      .local_device_id = 0,
      .pinned_memory_pool = pool,
      .dvmp = dvmp,
      .max_buffer_bytes = pool_total};

  auto mstatus = Model::create(cfg);
  REQUIRE(mstatus.ok());
  auto model = std::move(*mstatus);

  SECTION("Get model size") {
    auto size_status = model->get_model_size();
    REQUIRE(size_status.ok());
    REQUIRE(*size_status == total_size);
  }

  SECTION("Load to CPU and verify content") {
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    auto fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(fut.valid());
    auto wait_status = model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15));
    REQUIRE(wait_status.ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    // With DVMP, get_data_pointer returns a single pointer to the contiguous memory block
    auto ptrs = model->get_data_pointer(ModelLocation::PAGEABLE_CPU);
    REQUIRE(ptrs.size() == 1);
    REQUIRE(ptrs[0] != nullptr);

    // The entire model is now in a single contiguous memory block
    // Note: With DVMP and mmap-based loading, we need to be careful about accessing memory
    // that might be lazily mapped. Let's verify the content safely.
    char* model_data = static_cast<char*>(ptrs[0]);

    // Create a buffer to read into to avoid potential page faults
    std::vector<char> loaded(total_size);

    // Copy data from DVMP memory to our buffer
    // This ensures we trigger any page faults in a controlled manner
    std::memcpy(loaded.data(), model_data, total_size);

    REQUIRE(loaded == combined);
  }

  SECTION("Release CPU memory") {
    model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU).wait();
    model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).IgnoreError();
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);
    auto status = model->release_memory(ModelLocation::PAGEABLE_CPU);
    REQUIRE(status.ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    auto ptrs_after = model->get_data_pointer(ModelLocation::PAGEABLE_CPU);
    REQUIRE(ptrs_after.empty());
  }

  // Teardown
  model.reset();
  pool.reset();
  fs::remove_all(base);
}