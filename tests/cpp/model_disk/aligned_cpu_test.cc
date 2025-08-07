// Copyright (c) 2025, StepCast Team. All rights reserved.

// Copyright (c) 2025, StepCast Team.
// New test to verify DiskLoader zero-copy mmap path when partition sizes are page-aligned.

#include <catch2/catch_test_macros.hpp>
#include "tests/cpp/common.h"

#include <unistd.h>
#include <filesystem>

#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using namespace stepcast::tests;
using namespace stepcast::memory;

TEST_CASE("DiskModel page-aligned load to CPU via mmap", "[model][disk][cpu][mmap]") {
  // System page size
  const long page_sz_long = ::sysconf(_SC_PAGESIZE);
  REQUIRE(page_sz_long > 0);
  const auto page_sz = static_cast<size_t>(page_sz_long);

  // Partition sizes (page aligned and > page size)
  const size_t size0 = page_sz * 2;
  const size_t size1 = page_sz * 3;
  const size_t total_size = size0 + size1;

  const std::string model_id = "aligned_cpu_model";
  const std::string model_subdir = "aligned_cpu_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";

  fs::path base = fs::temp_directory_path() / "aligned_cpu_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base / model_subdir);

  fs::path path0 = base / model_subdir / p0;
  fs::path path1 = base / model_subdir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'C'));
  REQUIRE(create_dummy_file(path1, size1, 'D'));

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
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  // Create DVMP
  auto dvmp = std::make_shared<DistributedMemoryPool>();

  // Create DiskSource
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

  SECTION("Load via mmap and verify content") {
    auto fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(fut.valid());
    auto status = fut.get();
    REQUIRE(status.ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    auto ptrs = model->get_data_pointer(ModelLocation::PAGEABLE_CPU);
    REQUIRE_FALSE(ptrs.empty());

    // Lightweight verification: inspect first byte of the mapped region.
    REQUIRE(static_cast<char*>(ptrs.front())[0] == 'C');
  }

  // Teardown
  model.reset();
  pool.reset();
  fs::remove_all(base);
}