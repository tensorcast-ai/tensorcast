// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "tests/cpp/common.h"

#include <filesystem>
#include <memory>
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

TEST_CASE("Streaming Disk Load to GPU", "[model][disk][streaming]") {
  // Setup dummy model with two partitions
  const std::string model_id = "streaming_disk_model";
  const std::string model_subdir = "streaming_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  const size_t size0 = 1024 * 4;
  const size_t size1 = 1024 * 3;
  const size_t total_size = size0 + size1;

  // Skip if CUDA is not available on this system
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping streaming GPU test.");
  }

  fs::path base = fs::temp_directory_path() / "streaming_disk_test";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / model_subdir);

  fs::path path0 = base / model_subdir / p0;
  fs::path path1 = base / model_subdir / p1;
  REQUIRE(create_dummy_file(path0, size0, 'S'));
  REQUIRE(create_dummy_file(path1, size1, 'T'));

  // Combined original data
  std::vector<char> data0 = read_file_content(path0);
  std::vector<char> data1 = read_file_content(path1);
  std::vector<char> combined;
  combined.reserve(total_size);
  combined.insert(combined.end(), data0.begin(), data0.end());
  combined.insert(combined.end(), data1.begin(), data1.end());
  REQUIRE(combined.size() == total_size);

  // Pinned pool for streaming
  const size_t pool_total = 1024 * 1024;
  const size_t pool_chunk = 4096; // Use 4 KiB-aligned chunk size per alignment requirements
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  // Create DVMP
  auto dvmp = std::make_shared<::stepcast::memory::DistributedVirtualMemoryPool>();

  // Create and initialize a streaming buffer matching the pool's chunk size
  auto spb = std::make_shared<StreamingPinnedBuffer>(/*num_chunks=*/16, pool_chunk, pool);
  REQUIRE(spb != nullptr);
  REQUIRE(spb->initialize().ok());

  // Use new DiskSource
  DiskSource disk_src;
  disk_src.path = base / model_subdir;

  // Use aggregate initialization for ModelConfig
  ModelConfig cfg{
      .source = disk_src,
      .model_identifier = model_id,
      .device_type = ::stepcast::DeviceType::GPU,
      .local_device_id = 0,
      .pinned_memory_pool = pool,
      .dvmp = dvmp,
      .streaming_buffer = spb,
      .max_buffer_bytes = 1024 * 2 // 2 KB buffer to force streaming
  };

  auto mstatus = Model::create(cfg);
  REQUIRE(mstatus.ok());
  auto model = std::move(*mstatus);

  // Perform streaming load directly to GPU
  REQUIRE(model->get_memory_state(ModelLocation::GPU) <= MemoryState::UNALLOCATED);
  auto fut = model->ensure_loaded_async(ModelLocation::GPU);
  REQUIRE(fut.valid());
  model->wait_until_loaded(ModelLocation::GPU, absl::Seconds(30)).IgnoreError();
  REQUIRE(model->get_memory_state(ModelLocation::GPU) == MemoryState::LOADED);

  // Get GPU data pointer and copy back
  auto gpu_ptrs = model->get_data_pointer(ModelLocation::GPU);
  REQUIRE(gpu_ptrs.size() == 1);
  void* gpu_ptr = gpu_ptrs[0];
  REQUIRE(gpu_ptr != nullptr);

  std::vector<char> host_buf(total_size);
  absl::Status copy_status = stepcast::cuda::memcpy(host_buf.data(), gpu_ptr, total_size, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());

  // Verify data matches
  REQUIRE(host_buf.size() == total_size);
  REQUIRE(host_buf == combined);

  // Cleanup
  model.reset();
  pool.reset();
  fs::remove_all(base);
}