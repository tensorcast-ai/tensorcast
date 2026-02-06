// Copyright (c) 2025-2026, TensorCast Team.

// StoreEngine multi-GPU tests (B-series)
// Test multi-GPU loading, cross-GPU operations, and device-specific eviction.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "core/common/device_types.h"
#include "core/cuda/cuda_api.h"
#include "core/testing/concurrency_utils.h"

using namespace tensorcast::testing;
using tensorcast::DeviceType;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::StoreEngine;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaHandle;
using tensorcast::store::replica::MemoryState;

// B1: Load same replica to multiple GPUs
TEST_CASE("B1: Same replica on multiple GPUs", "[store_engine][multi_gpu][b1]") {
  skip_if_insufficient_gpus(2, "B1");

  const std::string artifact_id = "cgid:multi_gpu_model_b1";
  const size_t artifact_size = 50 * 1024 * 1024; // 50MB

  TempArtifactFixture fixture("multi_gpu_b1");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

  auto store = make_test_store(fixture.root(), 512); // 512MB pool

  // Get actual GPU count
  int gpu_count = 0;
  {
    auto st = tensorcast::cuda::get_device_count(&gpu_count);
    ABSL_CHECK(st.ok()) << "Failed to get GPU count: " << st.message();
  }
  gpu_count = std::min(gpu_count, 4); // Test up to 4 GPUs
  REQUIRE(gpu_count >= 1);

  // Load replica to each GPU
  std::vector<ReplicaHandle> handles;
  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    MaterializeHints hints;
    hints.artifact_id = artifact_id;
    auto handle_or =
        store->materialize_replica(make_gpu_key(gpu), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
    REQUIRE(handle_or.ok());
    handles.push_back(std::move(handle_or).value());
  }

  // Wait for all loads to complete
  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    auto status = handles[gpu].wait_ready(std::chrono::milliseconds(30000));
    REQUIRE(status.ok());
    REQUIRE(handles[gpu].gpu_base_ptr != nullptr);
  }

  // Verify replica is loaded on all GPUs
  auto loaded_devices = store->get_resident_devices(artifact_id);
  REQUIRE(loaded_devices.size() == gpu_count);

  // Verify each GPU has the replica
  std::unordered_map<int, bool> gpu_has_replica;
  for (const auto& device : loaded_devices) {
    REQUIRE(device.type == DeviceType::GPU);
    gpu_has_replica[device.ordinal] = true;
  }

  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    REQUIRE(gpu_has_replica[gpu]);

    // Verify instance on each GPU
    auto replica_key = make_replica_key(artifact_id, gpu);
    auto state = store->get_replica_state(replica_key, DeviceType::GPU);
    REQUIRE(state == MemoryState::LOADED);

    // Strengthen: verify GPU memory content matches file pattern for first few KB
    // Read disk sample
    auto file_path = fixture.root() / artifact_id / "tensor.data_0";
    auto host_data = tensorcast::testing::read_file_content(file_path);
    REQUIRE_FALSE(host_data.empty());

    // Get GPU pointer and compare prefix
    auto ptr_or = store->get_replica_gpu_ptr(replica_key);
    REQUIRE(ptr_or.ok());
    auto gpu_ptr_u64 = ptr_or.value();
    REQUIRE(gpu_ptr_u64 != 0);

    // Copy back a small prefix (e.g., 4KB) for content validation
    size_t verify_bytes = std::min<size_t>(4096, host_data.size());
    std::vector<char> gpu_prefix(verify_bytes);
    auto memcpy_st = tensorcast::cuda::memcpy(
        gpu_prefix.data(), reinterpret_cast<void*>(gpu_ptr_u64), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_st.ok());
    REQUIRE(std::memcmp(gpu_prefix.data(), host_data.data(), verify_bytes) == 0);

    // Also verify the last bytes match to catch truncated tails
    size_t tail_offset = host_data.size() - verify_bytes;
    std::vector<char> gpu_suffix(verify_bytes);
    auto memcpy_tail = tensorcast::cuda::memcpy(
        gpu_suffix.data(), reinterpret_cast<void*>(gpu_ptr_u64 + tail_offset), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_tail.ok());
    REQUIRE(std::memcmp(gpu_suffix.data(), host_data.data() + tail_offset, verify_bytes) == 0);

    // Replace single-byte check with full content verification
    std::vector<char> gpu_full(host_data.size());
    auto memcpy_full = tensorcast::cuda::memcpy(
        gpu_full.data(), reinterpret_cast<void*>(gpu_ptr_u64), host_data.size(), cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_full.ok());
    REQUIRE(std::memcmp(gpu_full.data(), host_data.data(), host_data.size()) == 0);
  }

  // Verify memory usage
  store->update_memory_pool_metrics();
  auto available_memory = store->get_available_memory();
  // In some builds unified memory accounting may differ; ensure we do not exceed pool size
  REQUIRE(available_memory <= store->get_mem_pool_size());
}

// B2: COPY_ONLY requires artifact_id
TEST_CASE("B2: COPY_ONLY requires artifact_id", "[store_engine][multi_gpu][b2]") {
  skip_if_insufficient_gpus(2, "B2");

  const std::string artifact_id = "cgid:gpu_copy_require_id_b2";
  const size_t artifact_size = 16 * 1024 * 1024; // 16MB

  TempArtifactFixture fixture("multi_gpu_b2");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

  auto store = make_test_store(fixture.root());

  // Load to GPU 0 first via LOAD_ONLY (using disk source)
  {
    MaterializeHints hints;
    hints.artifact_id = artifact_id;
    auto handle0_or =
        store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
    REQUIRE(handle0_or.ok());
    auto handle0 = std::move(handle0_or).value();
    REQUIRE(handle0.wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  // Now try COPY_ONLY to GPU 1 WITHOUT setting artifact_id → should fail with InvalidArgument
  {
    MaterializeHints hints; // intentionally leave artifact_id empty
    auto handle1_or = store->materialize_replica(make_gpu_key(1), StoreEngine::MaterializeMode::COPY_ONLY, hints);
    REQUIRE_FALSE(handle1_or.ok());
    REQUIRE(handle1_or.status().code() == absl::StatusCode::kInvalidArgument);
    REQUIRE(handle1_or.status().message().find("artifact_id") != std::string::npos);
  }
}

// B3: GPU-to-GPU copy
TEST_CASE("B3: GPU-to-GPU copy", "[store_engine][multi_gpu][b3]") {
  skip_if_insufficient_gpus(2, "B3");

  const std::string artifact_id = "cgid:gpu_copy_model_b3";
  const size_t artifact_size = 40 * 1024 * 1024; // 40MB

  TempArtifactFixture fixture("multi_gpu_b3");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

  auto store = make_test_store(fixture.root());

  // First load to GPU 0
  {
    MaterializeHints hints;
    hints.artifact_id = artifact_id;
    auto handle0_or =
        store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
    REQUIRE(handle0_or.ok());
    auto handle0 = std::move(handle0_or).value();
    REQUIRE(handle0.wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  // Verify loaded on GPU 0
  auto loaded_devices = store->get_resident_devices(artifact_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].ordinal == 0);

  // Now copy to GPU 1 using COPY_ONLY (GPU-to-GPU transfer enforced).
  auto copy_start = std::chrono::high_resolution_clock::now();
  {
    MaterializeHints hints;
    hints.artifact_id = artifact_id; // Needed by COPY_ONLY to locate source instance
    auto handle1_or = store->materialize_replica(make_gpu_key(1), StoreEngine::MaterializeMode::COPY_ONLY, hints);
    REQUIRE(handle1_or.ok());
    auto handle1 = std::move(handle1_or).value();
    REQUIRE(handle1.wait_ready(std::chrono::milliseconds(30000)).ok());
  }
  auto copy_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - copy_start);

  // Verify now loaded on both GPUs
  loaded_devices = store->get_resident_devices(artifact_id);
  REQUIRE(loaded_devices.size() == 2);

  // Both instances should be ready
  for (int gpu = 0; gpu < 2; ++gpu) {
    auto replica_key = make_replica_key(artifact_id, gpu);
    auto state = store->get_replica_state(replica_key, DeviceType::GPU);
    REQUIRE(state == MemoryState::LOADED);

    // Get GPU pointer for each instance
    auto ptr_or = store->get_replica_gpu_ptr(replica_key);
    REQUIRE(ptr_or.ok());
    REQUIRE(ptr_or.value() != 0);

    // Validate content prefix
    auto file_path = fixture.root() / artifact_id / "tensor.data_0";
    auto host_data = tensorcast::testing::read_file_content(file_path);
    REQUIRE_FALSE(host_data.empty());
    size_t verify_bytes = std::min<size_t>(4096, host_data.size());
    std::vector<char> gpu_prefix(verify_bytes);
    auto memcpy_st = tensorcast::cuda::memcpy(
        gpu_prefix.data(), reinterpret_cast<void*>(ptr_or.value()), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_st.ok());
    REQUIRE(std::memcmp(gpu_prefix.data(), host_data.data(), verify_bytes) == 0);

    // Validate content suffix
    size_t tail_offset = host_data.size() - verify_bytes;
    std::vector<char> gpu_suffix(verify_bytes);
    auto memcpy_tail = tensorcast::cuda::memcpy(
        gpu_suffix.data(), reinterpret_cast<void*>(ptr_or.value() + tail_offset), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_tail.ok());
    REQUIRE(std::memcmp(gpu_suffix.data(), host_data.data() + tail_offset, verify_bytes) == 0);

    // Replace single-byte check with full content verification
    std::vector<char> gpu_full(host_data.size());
    auto memcpy_full = tensorcast::cuda::memcpy(
        gpu_full.data(), reinterpret_cast<void*>(ptr_or.value()), host_data.size(), cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_full.ok());
    REQUIRE(std::memcmp(gpu_full.data(), host_data.data(), host_data.size()) == 0);
  }

  INFO("GPU-to-GPU copy time (COPY_ONLY): " << copy_time.count() << "ms");
}

// B4: Multi-GPU load balancing
TEST_CASE("B4: Multi-GPU load balancing", "[store_engine][multi_gpu][b4]") {
  skip_if_insufficient_gpus(2, "B4");

  const int num_artifacts = 8;
  const size_t artifact_size = 25 * 1024 * 1024; // 25MB each

  TempArtifactFixture fixture("multi_gpu_b4");

  // Create artifacts
  std::vector<std::string> artifact_ids;
  for (int i = 0; i < num_artifacts; ++i) {
    auto artifact_id = generate_artifact_id("balance_model_b4", i);
    auto canonical_id = std::string("cgid:") + artifact_id;
    artifact_ids.push_back(canonical_id);
    fixture.create_model(canonical_id, artifact_size);
  }

  auto store = make_test_store(fixture.root(), 512); // 512MB pool

  // Get GPU count
  int gpu_count = 0;
  {
    auto _st2 = tensorcast::cuda::get_device_count(&gpu_count);
    (void)_st2;
  }
  gpu_count = std::min(gpu_count, 4);
  REQUIRE(gpu_count > 0);

  // Load replicas with round-robin distribution
  for (size_t i = 0; i < artifact_ids.size(); ++i) {
    int target_gpu = i % gpu_count;
    MaterializeHints hints;
    hints.artifact_id = artifact_ids[i];
    DiskSource disk_source{.path = fixture.root() / artifact_ids[i], .expected_size = std::nullopt};
    auto handle_or = store->materialize_replica(
        make_gpu_key(target_gpu), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
    if (handle_or.ok()) {
      REQUIRE(handle_or.value().wait_ready(std::chrono::milliseconds(30000)).ok());
    }
  }

  // Check distribution across GPUs
  std::vector<int> replicas_per_gpu(gpu_count, 0);
  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    auto replicas = store->list_device_replicas(make_gpu_key(gpu));
    replicas_per_gpu[gpu] = replicas.size();
  }

  // Verify relatively even distribution
  int min_models = *std::min_element(replicas_per_gpu.begin(), replicas_per_gpu.end());
  int max_models = *std::max_element(replicas_per_gpu.begin(), replicas_per_gpu.end());

  // Distribution should be reasonably balanced
  REQUIRE(max_models - min_models <= 2);

  // All GPUs should have at least one replica (if enough models)
  if (num_artifacts >= gpu_count) {
    for (int gpu = 0; gpu < gpu_count; ++gpu) {
      REQUIRE(replicas_per_gpu[gpu] > 0);
    }
  }
}

// B5: Device-specific operations
TEST_CASE("B5: Device-specific operations", "[store_engine][multi_gpu][b5]") {
  skip_if_insufficient_gpus(2, "B5");

  const std::string artifact_id = "cgid:device_ops_model_b5";
  const size_t artifact_size = 30 * 1024 * 1024; // 30MB

  TempArtifactFixture fixture("multi_gpu_b5");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

  auto store = make_test_store(fixture.root());

  // Load to both GPU 0 and GPU 1
  MaterializeHints hints;
  hints.artifact_id = artifact_id;
  auto handle0 =
      store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
  REQUIRE(handle0.ok());
  REQUIRE(handle0.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  auto handle1 =
      store->materialize_replica(make_gpu_key(1), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);

  LOG(INFO) << "handle0: " << handle0.status().message();
  LOG(INFO) << "handle1: " << handle1.status().message();
  REQUIRE(handle1.ok());

  REQUIRE(handle1.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  // Verify full content on both GPUs to ensure data correctness before unload
  {
    auto file_path = fixture.root() / artifact_id / "tensor.data_0";
    auto host_data = tensorcast::testing::read_file_content(file_path);
    REQUIRE_FALSE(host_data.empty());

    for (int gpu = 0; gpu < 2; ++gpu) {
      auto replica_key = make_replica_key(artifact_id, gpu);
      auto ptr_or = store->get_replica_gpu_ptr(replica_key);
      REQUIRE(ptr_or.ok());

      std::vector<char> gpu_full(host_data.size());
      auto memcpy_full = tensorcast::cuda::memcpy(
          gpu_full.data(), reinterpret_cast<void*>(ptr_or.value()), host_data.size(), cudaMemcpyDeviceToHost);
      REQUIRE(memcpy_full.ok());
      REQUIRE(std::memcmp(gpu_full.data(), host_data.data(), host_data.size()) == 0);
    }
  }

  // Test device-specific unload
  auto instance0 = make_replica_key(artifact_id, 0);
  auto instance1 = make_replica_key(artifact_id, 1);

  // Unload from GPU 0 only
  REQUIRE(store->unload_replica(instance0) == 0);

  // Verify GPU 0 no longer has the replica
  auto gpu0_replicas = store->list_device_replicas(make_gpu_key(0));
  REQUIRE(gpu0_replicas.empty());

  // Verify GPU 1 still has the replica
  auto gpu1_replicas = store->list_device_replicas(make_gpu_key(1));
  REQUIRE(gpu1_replicas.size() == 1);
  REQUIRE(gpu1_replicas[0].artifact_id == artifact_id);

  // Verify get_resident_devices reflects the change
  auto loaded_devices = store->get_resident_devices(artifact_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].ordinal == 1);

  // Test remote access registration per device
  auto reg_info = store->enable_remote_replica_access(instance1, MemoryLocation::GPU);
  REQUIRE(reg_info.ok());
  // Try to enable on already unloaded instance
  auto reg_fail = store->enable_remote_replica_access(instance0, MemoryLocation::GPU);
  REQUIRE(!reg_fail.ok());
  // Disable remote access
  auto disable_status = store->disable_remote_replica_access(instance1, MemoryLocation::GPU);
  REQUIRE(disable_status.ok());
}
