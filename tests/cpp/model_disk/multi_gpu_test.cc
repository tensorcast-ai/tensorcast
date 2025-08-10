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

TEST_CASE("Multi-GPU Disk Load and Verification", "[model][disk][multi_gpu]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available or no CUDA devices found.");
  }
  int device_count = 0;
  absl::Status cuda_status = stepcast::cuda::get_device_count(&device_count);
  REQUIRE(cuda_status.ok());
  if (device_count < 2) {
    SKIP("Less than 2 CUDA devices found. Skipping multi-GPU test.");
  }

  const std::string model_id = "multi_gpu_model";
  const std::string model_subdir = "multi_gpu_model_files";
  const std::string f0 = "tensor.data_0";
  const std::string f1 = "tensor.data_1";
  const size_t s0 = 1024 * 5;
  const size_t s1 = 1024 * 3;
  const size_t total = s0 + s1;

  fs::path base = fs::temp_directory_path() / "multi_gpu_test_dir";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / model_subdir);
  fs::path p0 = base / model_subdir / f0;
  fs::path p1 = base / model_subdir / f1;
  REQUIRE(create_dummy_file(p0, s0, 'M'));
  REQUIRE(create_dummy_file(p1, s1, 'N'));

  // Prepare original combined data
  std::vector<char> d0 = read_file_content(p0);
  std::vector<char> d1 = read_file_content(p1);
  std::vector<char> orig;
  orig.reserve(total);
  orig.insert(orig.end(), d0.begin(), d0.end());
  orig.insert(orig.end(), d1.begin(), d1.end());
  REQUIRE(orig.size() == total);

  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 1024;

  for (int dev = 0; dev < std::min(device_count, 2); ++dev) {
    CAPTURE(dev);
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
        .model_identifier = model_id + std::to_string(dev),
        .device_type = ::stepcast::DeviceType::CPU,
        .local_device_id = dev,
        .pinned_memory_pool = pool,
        .dvmp = dvmp,
        .max_buffer_bytes = pool_total};

    auto mstat = Model::create(cfg);
    REQUIRE(mstat.ok());
    auto model = std::move(*mstat);

    // Load to CPU then to GPU on device 'dev'
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    auto cfut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(cfut.valid());
    REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    absl::Status set_dev_status = stepcast::cuda::set_device(dev);
    REQUIRE(set_dev_status.ok());
    auto gfut = model->ensure_loaded_async(ModelLocation::GPU);
    REQUIRE(gfut.valid());
    REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);

    // Copy back from GPU and verify
    auto gptrs = model->get_data_pointer(ModelLocation::GPU);
    REQUIRE(gptrs.size() == 1);
    void* gpu_ptr = gptrs[0];
    REQUIRE(gpu_ptr != nullptr);

    std::vector<char> hostbuf(total);
    absl::Status copy_status = stepcast::cuda::memcpy(hostbuf.data(), gpu_ptr, total, cudaMemcpyDeviceToHost);
    REQUIRE(copy_status.ok());
    REQUIRE(hostbuf == orig);

    // Cleanup model before next device
    model.reset();
    pool.reset();
  }

  // Teardown
  fs::remove_all(base);
}