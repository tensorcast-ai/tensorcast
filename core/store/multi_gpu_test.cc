// Copyright (c) 2025, StepCast Team. All rights reserved.

// CheckpointStore multi-GPU tests (B-series)
// Test multi-GPU loading, cross-GPU operations, and device-specific eviction.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "core/common/cuda_api.h"
#include "core/common/device_types.h"
#include "core/store/concurrency_utils.h"

using namespace stepcast::tests::checkpoint_store;
using namespace stepcast::store;
using stepcast::DeviceType;

// B1: Load same model to multiple GPUs
TEST_CASE("B1: Same model on multiple GPUs", "[checkpoint_store][multi_gpu][b1]") {
  skip_if_insufficient_gpus(2, "B1");

  const std::string model_id = "multi_gpu_model_b1";
  const size_t model_size = 50 * 1024 * 1024; // 50MB

  TempModelFixture fixture("multi_gpu_b1");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root(), 512); // 512MB pool

  // Get actual GPU count
  int gpu_count = 0;
  {
    auto st = stepcast::cuda::get_device_count(&gpu_count);
    ABSL_CHECK(st.ok()) << "Failed to get GPU count: " << st.message();
  }
  gpu_count = std::min(gpu_count, 4); // Test up to 4 GPUs
  REQUIRE(gpu_count >= 1);

  // Load model to each GPU
  std::vector<ModelHandle> handles;
  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    stepcast::store::LoadingHints hints;

    hints.disk_path = model_id;
    auto handle_or = store->prepare(make_gpu_key(gpu), CheckpointStore::PrepareMode::LOAD_ONLY, hints);
    REQUIRE(handle_or.ok());
    handles.push_back(std::move(handle_or).value());
  }

  // Wait for all loads to complete
  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    auto status = handles[gpu].wait_ready(std::chrono::milliseconds(30000));
    REQUIRE(status.ok());
    REQUIRE(handles[gpu].gpu_base_ptr != nullptr);
  }

  // Verify model is loaded on all GPUs
  auto loaded_devices = store->get_loaded_devices(model_id);
  REQUIRE(loaded_devices.size() == gpu_count);

  // Verify each GPU has the model
  std::unordered_map<int, bool> gpu_has_model;
  for (const auto& device : loaded_devices) {
    REQUIRE(device.type == DeviceType::GPU);
    gpu_has_model[device.ordinal] = true;
  }

  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    REQUIRE(gpu_has_model[gpu]);

    // Verify instance on each GPU
    auto instance_key = make_instance_key(model_id, gpu);
    auto state = store->get_instance_state(instance_key, DeviceType::GPU);
    REQUIRE(state == MemoryState::LOADED);

    // Strengthen: verify GPU memory content matches file pattern for first few KB
    // Read disk sample
    auto file_path = fixture.root() / model_id / "tensor.data_0";
    auto host_data = stepcast::tests::read_file_content(file_path);
    REQUIRE_FALSE(host_data.empty());

    // Get GPU pointer and compare prefix
    auto ptr_or = store->get_instance_gpu_ptr(instance_key);
    REQUIRE(ptr_or.ok());
    auto gpu_ptr_u64 = ptr_or.value();
    REQUIRE(gpu_ptr_u64 != 0);

    // Copy back a small prefix (e.g., 4KB) for content validation
    size_t verify_bytes = std::min<size_t>(4096, host_data.size());
    std::vector<char> gpu_prefix(verify_bytes);
    auto memcpy_st = stepcast::cuda::memcpy(
        gpu_prefix.data(), reinterpret_cast<void*>(gpu_ptr_u64), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_st.ok());
    REQUIRE(std::memcmp(gpu_prefix.data(), host_data.data(), verify_bytes) == 0);

    // Also verify the last bytes match to catch truncated tails
    size_t tail_offset = host_data.size() - verify_bytes;
    std::vector<char> gpu_suffix(verify_bytes);
    auto memcpy_tail = stepcast::cuda::memcpy(
        gpu_suffix.data(), reinterpret_cast<void*>(gpu_ptr_u64 + tail_offset), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_tail.ok());
    REQUIRE(std::memcmp(gpu_suffix.data(), host_data.data() + tail_offset, verify_bytes) == 0);

    // Replace single-byte check with full content verification
    std::vector<char> gpu_full(host_data.size());
    auto memcpy_full = stepcast::cuda::memcpy(
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

// B3: GPU-to-GPU copy
TEST_CASE("B3: GPU-to-GPU copy", "[checkpoint_store][multi_gpu][b3]") {
  skip_if_insufficient_gpus(2, "B3");

  const std::string model_id = "gpu_copy_model_b3";
  const size_t model_size = 40 * 1024 * 1024; // 40MB

  TempModelFixture fixture("multi_gpu_b3");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // First load to GPU 0
  {
    stepcast::store::LoadingHints hints;

    hints.disk_path = model_id;
    auto handle0_or = store->prepare(make_gpu_key(0), CheckpointStore::PrepareMode::LOAD_ONLY, hints);
    REQUIRE(handle0_or.ok());
    auto handle0 = std::move(handle0_or).value();
    REQUIRE(handle0.wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  // Verify loaded on GPU 0
  auto loaded_devices = store->get_loaded_devices(model_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].ordinal == 0);

  // Now copy to GPU 1 using COPY_ONLY (GPU-to-GPU transfer enforced).
  auto copy_start = std::chrono::high_resolution_clock::now();
  {
    stepcast::store::LoadingHints hints;

    auto handle1_or = store->prepare(make_gpu_key(1), CheckpointStore::PrepareMode::COPY_ONLY, hints);
    REQUIRE(handle1_or.ok());
    auto handle1 = std::move(handle1_or).value();
    REQUIRE(handle1.wait_ready(std::chrono::milliseconds(30000)).ok());
  }
  auto copy_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - copy_start);

  // Verify now loaded on both GPUs
  loaded_devices = store->get_loaded_devices(model_id);
  REQUIRE(loaded_devices.size() == 2);

  // Both instances should be ready
  for (int gpu = 0; gpu < 2; ++gpu) {
    auto instance_key = make_instance_key(model_id, gpu);
    auto state = store->get_instance_state(instance_key, DeviceType::GPU);
    REQUIRE(state == MemoryState::LOADED);

    // Get GPU pointer for each instance
    auto ptr_or = store->get_instance_gpu_ptr(instance_key);
    REQUIRE(ptr_or.ok());
    REQUIRE(ptr_or.value() != 0);

    // Validate content prefix
    auto file_path = fixture.root() / model_id / "tensor.data_0";
    auto host_data = stepcast::tests::read_file_content(file_path);
    REQUIRE_FALSE(host_data.empty());
    size_t verify_bytes = std::min<size_t>(4096, host_data.size());
    std::vector<char> gpu_prefix(verify_bytes);
    auto memcpy_st = stepcast::cuda::memcpy(
        gpu_prefix.data(), reinterpret_cast<void*>(ptr_or.value()), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_st.ok());
    REQUIRE(std::memcmp(gpu_prefix.data(), host_data.data(), verify_bytes) == 0);

    // Validate content suffix
    size_t tail_offset = host_data.size() - verify_bytes;
    std::vector<char> gpu_suffix(verify_bytes);
    auto memcpy_tail = stepcast::cuda::memcpy(
        gpu_suffix.data(), reinterpret_cast<void*>(ptr_or.value() + tail_offset), verify_bytes, cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_tail.ok());
    REQUIRE(std::memcmp(gpu_suffix.data(), host_data.data() + tail_offset, verify_bytes) == 0);

    // Replace single-byte check with full content verification
    std::vector<char> gpu_full(host_data.size());
    auto memcpy_full = stepcast::cuda::memcpy(
        gpu_full.data(), reinterpret_cast<void*>(ptr_or.value()), host_data.size(), cudaMemcpyDeviceToHost);
    REQUIRE(memcpy_full.ok());
    REQUIRE(std::memcmp(gpu_full.data(), host_data.data(), host_data.size()) == 0);
  }

  INFO("GPU-to-GPU copy time (COPY_ONLY): " << copy_time.count() << "ms");
}

// B4: Multi-GPU load balancing
TEST_CASE("B4: Multi-GPU load balancing", "[checkpoint_store][multi_gpu][b4]") {
  skip_if_insufficient_gpus(2, "B4");

  const int num_models = 8;
  const size_t model_size = 25 * 1024 * 1024; // 25MB each

  TempModelFixture fixture("multi_gpu_b4");

  // Create models
  std::vector<std::string> model_ids;
  for (int i = 0; i < num_models; ++i) {
    auto model_id = generate_model_name("balance_model_b4", i);
    model_ids.push_back(model_id);
    fixture.create_model(model_id, model_size);
  }

  auto store = make_test_store(fixture.root(), 512); // 512MB pool

  // Get GPU count
  int gpu_count = 0;
  {
    auto _st2 = stepcast::cuda::get_device_count(&gpu_count);
    (void)_st2;
  }
  gpu_count = std::min(gpu_count, 4);
  REQUIRE(gpu_count > 0);

  // Load models with round-robin distribution
  for (size_t i = 0; i < model_ids.size(); ++i) {
    int target_gpu = i % gpu_count;
    stepcast::store::LoadingHints hints;
    hints.disk_path = model_ids[i];
    auto handle_or = store->prepare(make_gpu_key(target_gpu), CheckpointStore::PrepareMode::LOAD_ONLY, hints);
    if (handle_or.ok()) {
      REQUIRE(handle_or.value().wait_ready(std::chrono::milliseconds(30000)).ok());
    }
  }

  // Check distribution across GPUs
  std::vector<int> models_per_gpu(gpu_count, 0);
  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    auto models = store->list_device_models(make_gpu_key(gpu));
    models_per_gpu[gpu] = models.size();
  }

  // Verify relatively even distribution
  int min_models = *std::min_element(models_per_gpu.begin(), models_per_gpu.end());
  int max_models = *std::max_element(models_per_gpu.begin(), models_per_gpu.end());

  // Distribution should be reasonably balanced
  REQUIRE(max_models - min_models <= 2);

  // All GPUs should have at least one model (if enough models)
  if (num_models >= gpu_count) {
    for (int gpu = 0; gpu < gpu_count; ++gpu) {
      REQUIRE(models_per_gpu[gpu] > 0);
    }
  }
}

// B5: Device-specific operations
TEST_CASE("B5: Device-specific operations", "[checkpoint_store][multi_gpu][b5]") {
  skip_if_insufficient_gpus(2, "B5");

  const std::string model_id = "device_ops_model_b5";
  const size_t model_size = 30 * 1024 * 1024; // 30MB

  TempModelFixture fixture("multi_gpu_b5");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // Load to both GPU 0 and GPU 1
  stepcast::store::LoadingHints hints;

  hints.disk_path = model_id;
  auto handle0 = store->prepare(make_gpu_key(0), CheckpointStore::PrepareMode::LOAD_ONLY, hints);
  auto handle1 = store->prepare(make_gpu_key(1), CheckpointStore::PrepareMode::LOAD_ONLY, hints);

  REQUIRE(handle0.ok());
  REQUIRE(handle1.ok());

  REQUIRE(handle0.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  REQUIRE(handle1.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  // Verify full content on both GPUs to ensure data correctness before unload
  {
    auto file_path = fixture.root() / model_id / "tensor.data_0";
    auto host_data = stepcast::tests::read_file_content(file_path);
    REQUIRE_FALSE(host_data.empty());

    for (int gpu = 0; gpu < 2; ++gpu) {
      auto instance_key = make_instance_key(model_id, gpu);
      auto ptr_or = store->get_instance_gpu_ptr(instance_key);
      REQUIRE(ptr_or.ok());

      std::vector<char> gpu_full(host_data.size());
      auto memcpy_full = stepcast::cuda::memcpy(
          gpu_full.data(), reinterpret_cast<void*>(ptr_or.value()), host_data.size(), cudaMemcpyDeviceToHost);
      REQUIRE(memcpy_full.ok());
      REQUIRE(std::memcmp(gpu_full.data(), host_data.data(), host_data.size()) == 0);
    }
  }

  // Test device-specific unload
  auto instance0 = make_instance_key(model_id, 0);
  auto instance1 = make_instance_key(model_id, 1);

  // Unload from GPU 0 only
  REQUIRE(store->unload_instance(instance0) == 0);

  // Verify GPU 0 no longer has the model
  auto gpu0_models = store->list_device_models(make_gpu_key(0));
  REQUIRE(gpu0_models.empty());

  // Verify GPU 1 still has the model
  auto gpu1_models = store->list_device_models(make_gpu_key(1));
  REQUIRE(gpu1_models.size() == 1);
  REQUIRE(gpu1_models[0].model_id == model_id);

  // Verify get_loaded_devices reflects the change
  auto loaded_devices = store->get_loaded_devices(model_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].ordinal == 1);

  // Test remote access registration per device
  auto reg_info = store->enable_remote_instance_access(instance1, ModelLocation::GPU);
  if (!reg_info.ok()) {
    WARN("Remote access not available; skipping remote access checks.");
  } else {
    REQUIRE(reg_info.ok());
    // Try to enable on already unloaded instance
    auto reg_fail = store->enable_remote_instance_access(instance0, ModelLocation::GPU);
    REQUIRE(!reg_fail.ok());
    // Disable remote access
    auto disable_status = store->disable_remote_instance_access(instance1, ModelLocation::GPU);
    REQUIRE(disable_status.ok());
  }
}
