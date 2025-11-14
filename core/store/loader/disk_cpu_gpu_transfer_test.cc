// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using tensorcast::testing::create_dummy_file;
using tensorcast::testing::is_cuda_available;
using tensorcast::testing::read_file_content;
using tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir;

TEST_CASE("DiskArtifact load to CPU then GPU and verify content", "[replica][disk][copy]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping CPU->GPU copy test.");
  }

  const std::string artifact_id = "cpu_to_gpu_artifact";
  const std::string artifact_dir_name = "cpu_to_gpu_model_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  const size_t size0 = 1024 * 4;
  const size_t size1 = 1024 * 2;
  const size_t total_size = size0 + size1;

  fs::path base = fs::temp_directory_path() / "cpu_to_gpu_test";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / artifact_dir_name);

  fs::path path0 = base / artifact_dir_name / p0;
  fs::path path1 = base / artifact_dir_name / p1;
  REQUIRE(create_dummy_file(path0, size0, 'X'));
  REQUIRE(create_dummy_file(path1, size1, 'Y'));

  // RFC-0007 metadata for standard partitions
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir_name).ok());

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
  auto pool = std::make_shared<PinnedBufferPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

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
      .pinned_buffer_pool = pool,
      .expected_artifact_size = total_size,
      .max_buffer_bytes = pool_total};

  auto mstatus = Replica::create(cfg);
  REQUIRE(mstatus.ok());
  auto replica = std::move(*mstatus);

  // Load to CPU
  REQUIRE(replica->get_memory_state(MemoryLocation::CPU) <= MemoryState::UNALLOCATED);
  auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::CPU);
  REQUIRE(cpu_fut.valid());
  REQUIRE(replica->wait_until_loaded(MemoryLocation::CPU, absl::Seconds(15)).ok());
  REQUIRE(replica->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);

  // Copy to GPU
  absl::Status set_dev = tensorcast::cuda::set_device(cfg.local_device_id);
  REQUIRE(set_dev.ok());
  auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
  REQUIRE(gpu_fut.valid());
  REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
  REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

  auto gpu_ptrs = replica->get_data_pointer(MemoryLocation::GPU);
  REQUIRE(gpu_ptrs.size() == 1);
  void* gpu_ptr = gpu_ptrs[0];
  REQUIRE(gpu_ptr != nullptr);

  std::vector<char> host_buf(total_size);
  absl::Status copy_status = tensorcast::cuda::memcpy(host_buf.data(), gpu_ptr, total_size, cudaMemcpyDeviceToHost);
  REQUIRE(copy_status.ok());

  REQUIRE(host_buf == combined);

  // Teardown
  replica.reset();
  pool.reset();
  fs::remove_all(base);
}
