// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "tests/cpp/common.h"

#include <filesystem>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/model/model.h"
#include "core/store/model/model_config.h"

namespace fs = std::filesystem;
using namespace stepcast::store;
using namespace stepcast::tests;

TEST_CASE("GPU auto-release mandatory after CPU to GPU copy", "[model][gpu][release]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping GPU auto-release test.");
  }

  const std::string model_id = "gpu_auto_release_model";
  const std::string model_subdir = "gpu_auto_release_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  const size_t size0 = 64 * 1024 * 1024; // 64 MB
  const size_t size1 = 128 * 1024 * 1024; // 128 MB
  const size_t total_size = size0 + size1;

  fs::path base = fs::temp_directory_path() / "gpu_auto_release_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base / model_subdir);

  fs::path path0 = base / model_subdir / p0;
  fs::path path1 = base / model_subdir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'A'));
  REQUIRE(create_dummy_file(path1, size1, 'B'));

  // Setup pinned pool
  const size_t pool_total = 512 * 1024 * 1024; // 512MB
  const size_t pool_chunk = 1024 * 1024; // 1MB chunks
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Load to CPU then GPU with auto-release enabled") {
    ModelConfig cfg;
    cfg.model_identifier = model_id;

    DiskSource disk_src;
    disk_src.path = base / model_subdir;
    cfg.source = disk_src;

    cfg.pinned_memory_pool = pool;
    cfg.local_device_id = 0;
    // Auto-release is now mandatory per RFC 0001

    auto mstatus = Model::create(cfg);
    REQUIRE(mstatus.ok());
    auto model = std::move(*mstatus);

    // Load to CPU first
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    auto cpu_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(cpu_fut.valid());
    REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    // Verify CPU data is accessible before GPU copy
    auto cpu_ptrs = model->get_data_pointer(ModelLocation::PAGEABLE_CPU);
    REQUIRE(!cpu_ptrs.empty());
    REQUIRE(cpu_ptrs[0] != nullptr);

    // Copy to GPU
    absl::Status set_dev = stepcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());
    auto gpu_fut = model->ensure_loaded_async(ModelLocation::GPU);
    REQUIRE(gpu_fut.valid());
    REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
    // Ensure the asynchronous copy task has finished so that any post-copy
    // auto-release logic is completed before we inspect CPU memory state.
    REQUIRE(gpu_fut.get().ok());
    REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);

    // After GPU copy, CPU memory should be marked UNALLOCATED (physical pages released but virtual space retained)
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);

    // CPU pointers should now be empty
    auto cpu_ptrs_after = model->get_data_pointer(ModelLocation::PAGEABLE_CPU);
    REQUIRE(cpu_ptrs_after.empty());

    // But GPU data should still be accessible
    auto gpu_ptrs = model->get_data_pointer(ModelLocation::GPU);
    REQUIRE(gpu_ptrs.size() == 1);
    void* gpu_ptr = gpu_ptrs[0];
    REQUIRE(gpu_ptr != nullptr);

    // Perform a lightweight host copy to ensure the GPU pointer is readable.
    std::vector<char> host_buf(16); // only copy first 16 bytes for sanity
    absl::Status copy_status =
        stepcast::cuda::memcpy(host_buf.data(), gpu_ptr, host_buf.size(), cudaMemcpyDeviceToHost);
    REQUIRE(copy_status.ok());
    // Expect that at least the first byte matches the expected pattern.
    REQUIRE(host_buf[0] == 'A');

    LOG(INFO) << "Successfully verified mandatory CPU memory release after GPU copy";
  }

  // Section for disabled auto-release removed - auto-release is now mandatory per RFC 0001

  // Teardown
  fs::remove_all(base);
}

TEST_CASE("Multi-GPU model loading with mandatory CPU release", "[model][gpu][multi]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping multi-GPU test.");
  }

  int device_count = 0;
  auto count_status = stepcast::cuda::get_device_count(&device_count);
  if (!count_status.ok() || device_count < 2) {
    SKIP("Multi-GPU test requires at least 2 GPUs.");
  }

  const std::string model_id = "multi_gpu_release_model";
  const std::string model_subdir = "multi_gpu_release_files";
  const size_t total_size = 256 * 1024 * 1024; // 256 MB

  fs::path base = fs::temp_directory_path() / "multi_gpu_release_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base / model_subdir);

  fs::path model_file = base / model_subdir / "tensor.data";
  REQUIRE(create_dummy_file(model_file, total_size, 'M'));

  // Setup pinned pool
  const size_t pool_total = 1024 * 1024 * 1024; // 1GB
  const size_t pool_chunk = 4 * 1024 * 1024; // 4MB chunks
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Sequential GPU loading with CPU auto-release") {
    // First GPU load
    {
      ModelConfig cfg;
      cfg.model_identifier = model_id;

      DiskSource disk_src;
      disk_src.path = base / model_subdir;
      cfg.source = disk_src;

      cfg.pinned_memory_pool = pool;
      cfg.local_device_id = 0;
      // Auto-release is now mandatory per RFC 0001

      auto mstatus = Model::create(cfg);
      REQUIRE(mstatus.ok());
      auto model = std::move(*mstatus);

      // Load to CPU then GPU
      auto cpu_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
      REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

      absl::Status set_dev = stepcast::cuda::set_device(0);
      REQUIRE(set_dev.ok());

      auto gpu_fut = model->ensure_loaded_async(ModelLocation::GPU);
      REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
      // Wait for async copy task to finish and mandatory CPU release to take effect.
      REQUIRE(gpu_fut.get().ok());

      // Verify CPU was released
      REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
      REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);
    }

    // Second GPU load (different device)
    {
      ModelConfig cfg;
      cfg.model_identifier = model_id + "_gpu1";

      DiskSource disk_src;
      disk_src.path = base / model_subdir;
      cfg.source = disk_src;

      cfg.pinned_memory_pool = pool;
      cfg.local_device_id = 1;
      // Auto-release is now mandatory per RFC 0001

      auto mstatus = Model::create(cfg);
      REQUIRE(mstatus.ok());
      auto model = std::move(*mstatus);

      // Load to CPU then GPU
      auto cpu_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
      REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

      absl::Status set_dev = stepcast::cuda::set_device(1);
      REQUIRE(set_dev.ok());

      auto gpu_fut = model->ensure_loaded_async(ModelLocation::GPU);
      REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
      // Wait for async copy task to finish and mandatory CPU release to take effect.
      REQUIRE(gpu_fut.get().ok());

      // Verify CPU was released
      REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
      REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);
    }

    LOG(INFO) << "Multi-GPU test passed: CPU memory released for both GPU loads (mandatory per RFC 0001)";
  }

  // Cleanup
  fs::remove_all(base);
}

TEST_CASE("GPU auto-release with very small models (boundary condition)", "[model][gpu][release][boundary]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping small model test.");
  }

  const std::string model_subdir = "small_model_files";
  fs::path base = fs::temp_directory_path() / "small_model_test";
  if (fs::exists(base))
    fs::remove_all(base);
  fs::create_directories(base / model_subdir);

  // Setup pinned pool
  const size_t pool_total = 128 * 1024 * 1024; // 128MB
  const size_t pool_chunk = 1024 * 1024; // 1MB chunks
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Model smaller than one chunk") {
    const std::string model_id = "tiny_model";
    const size_t model_size = 512 * 1024; // 512KB - smaller than 1MB chunk

    fs::path model_file = base / model_subdir / "tiny.data";
    REQUIRE(create_dummy_file(model_file, model_size, 'T'));

    ModelConfig cfg;
    cfg.model_identifier = model_id;
    
    DiskSource disk_src;
    disk_src.path = base / model_subdir;
    cfg.source = disk_src;
    
    cfg.pinned_memory_pool = pool;
    cfg.local_device_id = 0;

    auto mstatus = Model::create(cfg);
    REQUIRE(mstatus.ok());
    auto model = std::move(*mstatus);

    // Load to CPU
    auto cpu_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    // Copy to GPU
    absl::Status set_dev = stepcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());
    auto gpu_fut = model->ensure_loaded_async(ModelLocation::GPU);
    REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(gpu_fut.get().ok());

    // Verify CPU was released even for tiny model
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
    REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);

    LOG(INFO) << "Successfully verified CPU release for model smaller than chunk size";
  }

  SECTION("Model exactly one chunk size") {
    const std::string model_id = "one_chunk_model";
    const size_t model_size = 1024 * 1024; // Exactly 1MB

    fs::path model_file = base / model_subdir / "one_chunk.data";
    REQUIRE(create_dummy_file(model_file, model_size, 'C'));

    ModelConfig cfg;
    cfg.model_identifier = model_id;
    
    DiskSource disk_src;
    disk_src.path = base / model_subdir;
    cfg.source = disk_src;
    
    cfg.pinned_memory_pool = pool;
    cfg.local_device_id = 0;

    auto mstatus = Model::create(cfg);
    REQUIRE(mstatus.ok());
    auto model = std::move(*mstatus);

    // Load to CPU then GPU
    auto cpu_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

    absl::Status set_dev = stepcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());
    
    auto gpu_fut = model->ensure_loaded_async(ModelLocation::GPU);
    REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(gpu_fut.get().ok());

    // Verify CPU was released
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
    REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);

    LOG(INFO) << "Successfully verified CPU release for model exactly one chunk size";
  }

  SECTION("Model just over one chunk") {
    const std::string model_id = "slightly_over_chunk_model";
    const size_t model_size = 1024 * 1024 + 1024; // 1MB + 1KB

    fs::path model_file = base / model_subdir / "over_chunk.data";
    REQUIRE(create_dummy_file(model_file, model_size, 'O'));

    ModelConfig cfg;
    cfg.model_identifier = model_id;
    
    DiskSource disk_src;
    disk_src.path = base / model_subdir;
    cfg.source = disk_src;
    
    cfg.pinned_memory_pool = pool;
    cfg.local_device_id = 0;

    auto mstatus = Model::create(cfg);
    REQUIRE(mstatus.ok());
    auto model = std::move(*mstatus);

    // Load to CPU then GPU
    auto cpu_fut = model->ensure_loaded_async(ModelLocation::PAGEABLE_CPU);
    REQUIRE(model->wait_until_loaded(ModelLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

    absl::Status set_dev = stepcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());
    
    auto gpu_fut = model->ensure_loaded_async(ModelLocation::GPU);
    REQUIRE(model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(gpu_fut.get().ok());

    // Verify CPU was released
    REQUIRE(model->get_memory_state(ModelLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
    REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);

    LOG(INFO) << "Successfully verified CPU release for model just over one chunk";
  }

  // Cleanup
  fs::remove_all(base);
}