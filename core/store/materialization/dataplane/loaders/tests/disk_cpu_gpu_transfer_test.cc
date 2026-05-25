// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/testing/common.h"

#include <filesystem>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/cuda/cuda_api.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/memory_tier_config.h"
#include "core/store/replica/chunk_state.h"
#include "core/store/replica/replica.h"
#include "core/store/replica/replica_config.h"

namespace fs = std::filesystem;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::common::memory::PinnedBufferPool;
using tensorcast::store::MemoryTierConfig;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::replica::ChunkState;
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

  tensorcast::testing::TestTempDir base("cpu_to_gpu_test");
  fs::create_directories(base.path() / artifact_dir_name);

  fs::path path0 = base.path() / artifact_dir_name / p0;
  fs::path path1 = base.path() / artifact_dir_name / p1;
  REQUIRE(create_dummy_file(path0, size0, 'X'));
  REQUIRE(create_dummy_file(path1, size1, 'Y'));

  // RFC-0007 metadata for standard partitions
  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base.path() / artifact_dir_name).ok());

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
  auto async_runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  REQUIRE(async_runtime != nullptr);

  // Use new DiskSource
  DiskSource disk_src;
  disk_src.path = base.path() / artifact_dir_name;

  // Use aggregate initialization for ReplicaConfig
  // Set max_buffer_bytes to match the available pool size
  ReplicaConfig cfg{
      .source = disk_src,
      .artifact_identifier = artifact_id,
      .device_type = ::tensorcast::DeviceType::GPU,
      .local_device_id = 0,
      .pinned_buffer_pool = pool,
      .async_runtime = async_runtime,
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
}

TEST_CASE(
    "Preemptible-disabled replicas keep CPU residency through GPU load",
    "[replica][disk][copy][preemptible_off]") {
  if (!is_cuda_available()) {
    SKIP("CUDA not available. Skipping preemptible-off residency test.");
  }

  const std::string artifact_id = "preemptible_off_artifact";
  const std::string artifact_dir_name = "preemptible_off_model_files";
  const std::string shard_name = "tensor.data_0";
  const size_t chunk_bytes = 1 * 1024 * 1024; // 1 MiB UMA chunking for the test fixture
  const size_t payload_size = chunk_bytes * 2; // two full chunks

  tensorcast::testing::TestTempDir base("preemptible_off_test");
  fs::create_directories(base.path() / artifact_dir_name);

  fs::path shard_path = base.path() / artifact_dir_name / shard_name;
  REQUIRE(create_dummy_file(shard_path, payload_size, 'Z'));

  REQUIRE(write_rfc0007_descriptor_for_standard_artifact_dir(base.path() / artifact_dir_name).ok());

  const size_t pool_total = 4 * chunk_bytes;
  const size_t pool_chunk = chunk_bytes / 4;
  auto pool = std::make_shared<PinnedBufferPool>(pool_total, pool_chunk);
  REQUIRE(pool != nullptr);
  auto async_runtime = std::make_shared<tensorcast::common::AsyncRuntime>();
  REQUIRE(async_runtime != nullptr);

  MemoryTierConfig mem_tier_cfg;
  mem_tier_cfg.enable_preemptible_memory = false;
  mem_tier_cfg.stable_bytes = pool_total;
  mem_tier_cfg.preemptible_limit_bytes = 0;
  mem_tier_cfg.preemptible_low_watermark_ratio = 0.2;

  DiskSource disk_src;
  disk_src.path = base.path() / artifact_dir_name;

  ReplicaConfig cfg{
      .source = disk_src,
      .artifact_identifier = artifact_id,
      .device_type = ::tensorcast::DeviceType::GPU,
      .local_device_id = 0,
      .pinned_buffer_pool = pool,
      .async_runtime = async_runtime,
      .artifact_chunk_bytes = chunk_bytes,
      .expected_artifact_size = payload_size,
      .max_buffer_bytes = pool_total,
      .memory_tier_config = mem_tier_cfg};

  auto replica_or = Replica::create(cfg);
  REQUIRE(replica_or.ok());
  auto replica = std::move(*replica_or);

  // Initial CPU load should not mark any chunk as PREEMPTIBLE when the tier is disabled.
  auto cpu_fut = replica->ensure_loaded_async(MemoryLocation::CPU);
  REQUIRE(cpu_fut.valid());
  REQUIRE(replica->wait_until_loaded(MemoryLocation::CPU, absl::Seconds(15)).ok());
  REQUIRE(replica->get_memory_state(MemoryLocation::CPU) == MemoryState::LOADED);

  auto& mem_manager = replica->get_memory_manager();
  auto uma = mem_manager.memory_authority();
  const auto replica_key = mem_manager.replica_key();
  auto cpu_snapshot = uma->snapshot_cpu_chunks(replica_key);
  REQUIRE_FALSE(cpu_snapshot.empty());
  for (const auto& rec : cpu_snapshot) {
    REQUIRE(rec.cpu != ChunkState::PREEMPTIBLE);
  }

  // Load to GPU and ensure post-GPU policy keeps CPU residency and stable states.
  REQUIRE(tensorcast::cuda::set_device(cfg.local_device_id).ok());
  auto gpu_fut = replica->ensure_loaded_async(MemoryLocation::GPU);
  REQUIRE(gpu_fut.valid());
  REQUIRE(replica->wait_until_loaded(MemoryLocation::GPU, absl::Seconds(30)).ok());
  REQUIRE(replica->get_memory_state(MemoryLocation::GPU) == MemoryState::LOADED);

  REQUIRE(mem_manager.get_state(MemoryLocation::CPU) == MemoryState::LOADED);
  auto cpu_ptrs = mem_manager.get_pointer(MemoryLocation::CPU);
  REQUIRE(cpu_ptrs.size() == 1);
  REQUIRE(cpu_ptrs[0] != nullptr);

  auto snapshot_after_gpu = uma->snapshot_cpu_chunks(replica_key);
  REQUIRE(snapshot_after_gpu.size() == cpu_snapshot.size());
  for (const auto& rec : snapshot_after_gpu) {
    REQUIRE(rec.cpu != ChunkState::PREEMPTIBLE);
    REQUIRE(rec.cpu != ChunkState::EVICTED);
  }
}
