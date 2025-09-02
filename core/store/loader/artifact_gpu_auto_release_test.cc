// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/common/cuda_api.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "core/common/memory/pinned_memory_pool.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::DistributedVirtualMemoryPool;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedMemoryPool;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::MemoryState;
using tensorcast::store::replica::Replica;
using tensorcast::store::replica::ReplicaConfig;
using namespace tensorcast::testing;

TEST_CASE("GPU auto-release mandatory after CPU to GPU copy", "[replica][gpu][release]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping GPU auto-release test.");
  }

  const std::string artifact_id = "gpu_auto_release_artifact";
  const std::string artifact_dir_name = "gpu_auto_release_files";
  const std::string p0 = "tensor.data_0";
  const std::string p1 = "tensor.data_1";
  const size_t size0 = 64 * 1024 * 1024; // 64 MB
  const size_t size1 = 128 * 1024 * 1024; // 128 MB

  fs::path base = fs::temp_directory_path() / "gpu_auto_release_test";
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

  // Setup pinned pool
  const size_t pool_total = 512 * 1024 * 1024; // 512MB
  const size_t pool_chunk = 1024 * 1024; // 1MB chunks
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Load to CPU then GPU with auto-release enabled") {
    // Create DVMP
    auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();

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
        .expected_artifact_size = size0 + size1,
        .max_buffer_bytes = pool_total};

    auto mstatus = Replica::create(cfg);
    REQUIRE(mstatus.ok());
    auto replica = std::move(*mstatus);

    // Load to CPU first
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) <= MemoryState::UNALLOCATED);
    auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(cpu_fut.valid());
    REQUIRE(replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    // Verify CPU data is accessible before GPU copy
    auto cpu_ptrs = replica->get_data_pointer(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(!cpu_ptrs.empty());
    REQUIRE(cpu_ptrs[0] != nullptr);

    // Copy to GPU
    absl::Status set_dev = tensorcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());
    auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
    REQUIRE(gpu_fut.valid());
    REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
    // Ensure the asynchronous copy task has finished so that any post-copy
    // auto-release logic is completed before we inspect CPU memory state.
    REQUIRE(gpu_fut.get().ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

    // After GPU copy, CPU memory should be marked UNALLOCATED (physical pages released but virtual space retained)
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);

    // CPU pointers should now be empty
    auto cpu_ptrs_after = replica->get_data_pointer(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(cpu_ptrs_after.empty());

    // But GPU data should still be accessible
    auto gpu_ptrs = replica->get_data_pointer(MemoryLocation::GPU);
    REQUIRE(gpu_ptrs.size() == 1);
    void* gpu_ptr = gpu_ptrs[0];
    REQUIRE(gpu_ptr != nullptr);

    // Perform a lightweight host copy to ensure the GPU pointer is readable.
    std::vector<char> host_buf(16); // only copy first 16 bytes for sanity
    absl::Status copy_status =
        tensorcast::cuda::memcpy(host_buf.data(), gpu_ptr, host_buf.size(), cudaMemcpyDeviceToHost);
    REQUIRE(copy_status.ok());
    // Expect that at least the first byte matches the expected pattern.
    REQUIRE(host_buf[0] == 'A');

    LOG(INFO) << "Successfully verified mandatory CPU memory release after GPU copy";
  }

  // Section for disabled auto-release removed - auto-release is now mandatory per RFC 0001

  // Teardown
  fs::remove_all(base);
}

TEST_CASE("Multi-GPU replica loading with mandatory CPU release", "[replica][gpu][multi]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping multi-GPU test.");
  }

  int device_count = 0;
  auto count_status = tensorcast::cuda::get_device_count(&device_count);
  if (!count_status.ok() || device_count < 2) {
    SKIP("Multi-GPU test requires at least 2 GPUs.");
  }

  const std::string artifact_id = "multi_gpu_release_artifact";
  const std::string artifact_dir_name = "multi_gpu_release_files";
  const size_t total_size = 256 * 1024 * 1024; // 256 MB

  fs::path base = fs::temp_directory_path() / "multi_gpu_release_test";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / artifact_dir_name);

  fs::path data_file_path = base / artifact_dir_name / "tensor.data";
  REQUIRE(create_dummy_file(data_file_path, total_size, 'M'));
  // RFC-0007: standard partition directories must include descriptor + index
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir_name).ok());

  // Setup pinned pool
  const size_t pool_total = 1024 * 1024 * 1024; // 1GB
  const size_t pool_chunk = 4 * 1024 * 1024; // 4MB chunks
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Sequential GPU loading with CPU auto-release") {
    // First GPU load
    {
      // Create DVMP
      auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();

      // Use new DiskSource
      DiskSource disk_src;
      disk_src.path = base / artifact_dir_name;

      // Use aggregate initialization for ReplicaConfig
      const size_t total_size = 256 * 1024 * 1024; // must match file above
      ReplicaConfig cfg{
          .source = disk_src,
          .artifact_identifier = artifact_id,
          .device_type = ::tensorcast::DeviceType::CPU,
          .local_device_id = 0,
          .pinned_memory_pool = pool,
          .dvmp = dvmp,
          .expected_artifact_size = total_size};

      auto mstatus = Replica::create(cfg);
      REQUIRE(mstatus.ok());
      auto replica = std::move(*mstatus);

      // Load to CPU then GPU
      auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
      REQUIRE(replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

      absl::Status set_dev = tensorcast::cuda::set_device(0);
      REQUIRE(set_dev.ok());

      auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
      REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
      // Wait for async copy task to finish and mandatory CPU release to take effect.
      REQUIRE(gpu_fut.get().ok());

      // Verify CPU was released
      REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
      REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);
    }

    // Second GPU load (different device)
    {
      // Create DVMP
      auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();

      // Use new DiskSource
      DiskSource disk_src;
      disk_src.path = base / artifact_dir_name;

      // Use aggregate initialization for ReplicaConfig
      const size_t total_size2 = 256 * 1024 * 1024;
      ReplicaConfig cfg{
          .source = disk_src,
          .artifact_identifier = artifact_id + "_gpu1",
          .device_type = ::tensorcast::DeviceType::CPU,
          .local_device_id = 1,
          .pinned_memory_pool = pool,
          .dvmp = dvmp,
          .expected_artifact_size = total_size2};

      auto mstatus = Replica::create(cfg);
      REQUIRE(mstatus.ok());
      auto replica = std::move(*mstatus);

      // Load to CPU then GPU
      auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
      REQUIRE(replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

      absl::Status set_dev = tensorcast::cuda::set_device(1);
      REQUIRE(set_dev.ok());

      auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
      REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
      // Wait for async copy task to finish and mandatory CPU release to take effect.
      REQUIRE(gpu_fut.get().ok());

      // Verify CPU was released
      REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
      REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);
    }

    LOG(INFO) << "Multi-GPU test passed: CPU memory released for both GPU loads (mandatory per RFC 0001)";
  }

  // Cleanup
  fs::remove_all(base);
}

TEST_CASE("GPU auto-release with very small artifacts (boundary condition)", "[replica][gpu][release][boundary]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping small replica test.");
  }

  const std::string artifact_dir_name = "small_artifact_files";
  fs::path base = fs::temp_directory_path() / "small_artifact_test";
  if (fs::exists(base)) {
    fs::remove_all(base);
  }
  fs::create_directories(base / artifact_dir_name);

  // Setup pinned pool
  const size_t pool_total = 128 * 1024 * 1024; // 128MB
  const size_t pool_chunk = 1024 * 1024; // 1MB chunks
  auto pool = std::make_shared<PinnedMemoryPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);

  SECTION("Replica smaller than one chunk") {
    const std::string artifact_id = "tiny_artifact";
    const size_t artifact_size = 512 * 1024; // 512KB - smaller than 1MB chunk

    fs::path data_file_path = base / artifact_dir_name / "tensor.data_0";
    REQUIRE(create_dummy_file(data_file_path, artifact_size, 'T'));
    REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir_name).ok());

    // Create DVMP
    auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();
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
        .expected_artifact_size = artifact_size,
        .max_buffer_bytes = pool_total};

    auto mstatus = Replica::create(cfg);
    REQUIRE(mstatus.ok());
    auto replica = std::move(*mstatus);

    // Load to CPU
    auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::LOADED);

    // Copy to GPU
    absl::Status set_dev = tensorcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());
    auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
    REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(gpu_fut.get().ok());

    // Verify CPU was released even for tiny replica
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

    LOG(INFO) << "Successfully verified CPU release for replica smaller than chunk size";
  }

  SECTION("Replica exactly one chunk size") {
    const std::string artifact_id = "one_chunk_artifact";
    const size_t artifact_size = 1024 * 1024; // Exactly 1MB

    fs::path data_file_path2 = base / artifact_dir_name / "tensor.data_0";
    REQUIRE(create_dummy_file(data_file_path2, artifact_size, 'C'));
    REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir_name).ok());

    // Create DVMP
    auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();

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
        .expected_artifact_size = artifact_size,
        .max_buffer_bytes = pool_total};

    auto mstatus = Replica::create(cfg);
    REQUIRE(mstatus.ok());
    auto replica = std::move(*mstatus);

    // Load to CPU then GPU
    auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

    absl::Status set_dev = tensorcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());

    auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
    REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(gpu_fut.get().ok());

    // Verify CPU was released
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

    LOG(INFO) << "Successfully verified CPU release for replica exactly one chunk size";
  }

  SECTION("Replica just over one chunk") {
    const std::string artifact_id = "slightly_over_chunk_artifact";
    const size_t artifact_size = 1024 * 1024 + 1024; // 1MB + 1KB

    fs::path data_file_path3 = base / artifact_dir_name / "tensor.data_0";
    REQUIRE(create_dummy_file(data_file_path3, artifact_size, 'O'));
    REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base / artifact_dir_name).ok());

    // Create DVMP
    auto dvmp = std::make_shared<DistributedVirtualMemoryPool>();
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
        .expected_artifact_size = artifact_size,
        .max_buffer_bytes = pool_total};

    auto mstatus = Replica::create(cfg);
    REQUIRE(mstatus.ok());
    auto replica = std::move(*mstatus);

    // Load to CPU then GPU
    auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::PAGEABLE_CPU);
    REQUIRE(replica->wait_until_loaded(MemoryLocation::PAGEABLE_CPU, absl::Seconds(15)).ok());

    absl::Status set_dev = tensorcast::cuda::set_device(cfg.local_device_id);
    REQUIRE(set_dev.ok());

    auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
    REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
    REQUIRE(gpu_fut.get().ok());

    // Verify CPU was released
    REQUIRE(replica->get_memory_state(MemoryLocation::PAGEABLE_CPU) == MemoryState::UNALLOCATED);
    REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

    LOG(INFO) << "Successfully verified CPU release for replica just over one chunk";
  }

  // Cleanup
  fs::remove_all(base);
}
