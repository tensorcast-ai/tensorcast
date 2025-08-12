// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "tests/cpp/common.h"

#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using namespace stepcast::tests;

TEST_CASE("DiskModel load to CPU then GPU and verify content", "[model][disk][copy]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping CPU->GPU copy test.");
  }

  const std::string model_id = "cpu_to_gpu_model";
  const std::string model_subdir = "cpu_to_gpu_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  const size_t size0 = 1024 * 4;
  const size_t size1 = 1024 * 2;
  const size_t total_size = size0 + size1;

  fs::path base = fs::temp_directory_path() / "cpu_to_gpu_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base / model_subdir);

  fs::path path0 = base / model_subdir / p0;
  fs::path path1 = base / model_subdir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'X'));
  REQUIRE(create_dummy_file(path1, size1, 'Y'));

  // Combined original data
  auto data0 = read_file_content(path0);
  auto data1 = read_file_content(path1);
  std::vector<char> combined;
  combined.reserve(total_size);
  combined.insert(combined.end(), data0.begin(), data0.end());
  combined.insert(combined.end(), data1.begin(), data1.end());
  REQUIRE(combined.size() == total_size);

  // Setup pinned pool
  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 512;
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

  // Load to CPU
  REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
  auto cpu_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
  REQUIRE(cpu_fut.valid());
  REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
  REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

  // Copy to GPU
  absl::Status set_dev = stepcast::cuda::set_device(cfg.local_device_id);
  REQUIRE(set_dev.ok());
  auto gpu_fut = model->ensure_loaded_async(ModelLocation::GPU);
  REQUIRE(gpu_fut.valid());
  REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
  REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);

  auto gpu_ptrs = model->get_data_pointer(ModelLocation::GPU);
  REQUIRE(gpu_ptrs.size() == 1);
  void* gpu_ptr = gpu_ptrs[0];
  REQUIRE(gpu_ptr != nullptr);

  std::vector<char> host_buf(total_size);
  absl::Status copy_status = stepcast::cuda::memcpy(host_buf.data(), gpu_ptr, total_size, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());

  REQUIRE(host_buf == combined);

  // Teardown
  model.reset();
  pool.reset();
  fs::remove_all(base);
}