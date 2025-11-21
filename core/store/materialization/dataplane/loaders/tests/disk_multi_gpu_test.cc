// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/materialization/contracts/loading_spec.h"
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

TEST_CASE("Multi-GPU Disk Load and Verification", "[replica][disk][multi_gpu]") {
  if (!is_cuda_available()) {
    INFO("CUDA not available or no CUDA devices found. Marking as success.");
    SUCCEED();
    return;
  }
  int device_count = 0;
  absl::Status cuda_status = tensorcast::cuda::get_device_count(&device_count);
  REQUIRE(cuda_status.ok());
  if (device_count < 2) {
    INFO("Less than 2 CUDA devices found. Marking as success.");
    SUCCEED();
    return;
  }

  const std::string artifact_id = "multi_gpu_artifact";
  const std::string artifact_dir = "multi_gpu_artifact_files";
  const std::string f0 = "tensor.data_0";
  const std::string f1 = "tensor.data_1";
  const size_t s0 = 1024 * 5;
  const size_t s1 = 1024 * 3;
  const size_t total = s0 + s1;

  fs::path base = fs::temp_directory_path() / "multi_gpu_test_dir";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / artifact_dir);
  fs::path p0 = base / artifact_dir / f0;
  fs::path p1 = base / artifact_dir / f1;
  REQUIRE(create_dummy_file(p0, s0, 'M'));
  REQUIRE(create_dummy_file(p1, s1, 'N'));
  // RFC-0007 metadata for standard partitions
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir).ok());

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
    auto pool = std::make_shared<PinnedBufferPool>(pool_total, pool_chunk);
    REQUIRE(pool != nullptr);

    // Use new DiskSource
    DiskSource disk_src;
    disk_src.path = base / artifact_dir;

    // Use aggregate initialization for ReplicaConfig
    // Set max_buffer_bytes to match the available pool size
    ReplicaConfig cfg{
        .source = disk_src,
        .artifact_identifier = artifact_id,
        .device_type = ::tensorcast::DeviceType::CPU,
        .local_device_id = dev,
        .pinned_buffer_pool = pool,
        .expected_artifact_size = total,
        .max_buffer_bytes = pool_total};

    auto mstat = Replica::create(cfg);
    REQUIRE(mstat.ok());
    auto replica = std::move(*mstat);

    // Load to CPU then to GPU on device 'dev'
    REQUIRE(replica->get_memory_state(MemoryLocation::CPU) <= MemoryState::UNALLOCATED);
    auto cfut = replica->ensure_loaded_async(MemoryLocation::CPU);
    REQUIRE(cfut.valid());
    REQUIRE(replica->wait_until_loaded(MemoryLocation::CPU, absl::Seconds(15)).ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);

    absl::Status set_dev_status = tensorcast::cuda::set_device(dev);
    REQUIRE(set_dev_status.ok());
    auto gfut = replica->ensure_loaded_async(MemoryLocation::GPU);
    REQUIRE(gfut.valid());
    REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

    // Copy back from GPU and verify
    auto gptrs = replica->get_data_pointer(MemoryLocation::GPU);
    REQUIRE(gptrs.size() == 1);
    void* gpu_ptr = gptrs[0];
    REQUIRE(gpu_ptr != nullptr);

    std::vector<char> hostbuf(total);
    absl::Status copy_status = tensorcast::cuda::memcpy(hostbuf.data(), gpu_ptr, total, cudaMemcpyDeviceToHost);
    REQUIRE(copy_status.ok());
    REQUIRE(hostbuf == orig);

    // Cleanup replica before next device
    replica.reset();
    pool.reset();
  }

  // Teardown
  fs::remove_all(base);
}
