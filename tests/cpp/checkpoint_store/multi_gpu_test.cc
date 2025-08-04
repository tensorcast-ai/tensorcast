// Copyright (c) 2025, StepCast Team. All rights reserved.

// CheckpointStore multi-GPU tests (B-series)
// Test multi-GPU loading, cross-GPU operations, and device-specific eviction.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "common/device_types.h"
#include "concurrency_utils.h"

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
#ifdef __CUDACC__
  cudaGetDeviceCount(&gpu_count);
#endif
  gpu_count = std::min(gpu_count, 4); // Test up to 4 GPUs

  // Load model to each GPU
  std::vector<ModelHandle> handles;
  for (int gpu = 0; gpu < gpu_count; ++gpu) {
    auto handle_or = store->prepare(model_id, make_gpu_key(gpu));
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
  }

  // Verify memory usage
  store->update_memory_pool_metrics();
  auto available_memory = store->get_available_memory();
  auto expected_usage = model_size * gpu_count;
  REQUIRE(available_memory <= store->get_mem_pool_size() - expected_usage);
}

// B2: Cross-GPU eviction
TEST_CASE("B2: Cross-GPU eviction", "[checkpoint_store][multi_gpu][b2]") {
  skip_if_insufficient_gpus(2, "B2");

  const size_t pool_size = 200 * 1024 * 1024; // 200MB pool
  const size_t model_size = 60 * 1024 * 1024; // 60MB per model

  TempModelFixture fixture("multi_gpu_b2");

  // Create 4 models
  std::vector<std::string> model_ids;
  for (int i = 0; i < 4; ++i) {
    auto model_id = generate_model_name("evict_model_b2", i);
    model_ids.push_back(model_id);
    fixture.create_model(model_id, model_size);
  }

  auto store = make_test_store(fixture.root(), pool_size / (1024 * 1024));

  // Load models to GPU 0 and GPU 1 alternately
  for (size_t i = 0; i < model_ids.size(); ++i) {
    int target_gpu = i % 2;
    auto handle_or = store->prepare(model_ids[i], make_gpu_key(target_gpu));
    REQUIRE(handle_or.ok());
    auto status = handle_or.value().wait_ready(std::chrono::milliseconds(30000));

    if (i < 3) {
      // First 3 models should load successfully
      REQUIRE(status.ok());
    } else {
      // 4th model should trigger eviction
      // With 200MB pool and 60MB models, only 3 can fit
      if (status.ok()) {
        // If it loaded, verify something was evicted
        int total_loaded = 0;
        for (const auto& model_id : model_ids) {
          total_loaded += store->get_loaded_devices(model_id).size();
        }
        REQUIRE(total_loaded <= 3);
      }
    }
  }

  // Verify models are distributed across GPUs
  auto gpu0_models = store->list_device_models(make_gpu_key(0));
  auto gpu1_models = store->list_device_models(make_gpu_key(1));

  // Both GPUs should have models
  REQUIRE(gpu0_models.size() > 0);
  REQUIRE(gpu1_models.size() > 0);

  // Total models should not exceed what can fit in memory
  REQUIRE(gpu0_models.size() + gpu1_models.size() <= 3);
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
  auto handle0_or = store->prepare(model_id, make_gpu_key(0), CheckpointStore::PrepareMode::LOAD_ONLY);
  REQUIRE(handle0_or.ok());
  auto handle0 = std::move(handle0_or).value();
  REQUIRE(handle0.wait_ready(std::chrono::milliseconds(30000)).ok());

  // Verify loaded on GPU 0
  auto loaded_devices = store->get_loaded_devices(model_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].ordinal == 0);

  // Now copy to GPU 1 (should use GPU-to-GPU transfer)
  auto handle1_or = store->prepare(model_id, make_gpu_key(1), CheckpointStore::PrepareMode::COPY_ONLY);
  REQUIRE(handle1_or.ok());
  auto handle1 = std::move(handle1_or).value();

  auto copy_start = std::chrono::high_resolution_clock::now();
  REQUIRE(handle1.wait_ready(std::chrono::milliseconds(30000)).ok());
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
  }

  INFO("GPU-to-GPU copy time: " << copy_time.count() << "ms");
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
#ifdef __CUDACC__
  cudaGetDeviceCount(&gpu_count);
#endif
  gpu_count = std::min(gpu_count, 4);

  // Load models with round-robin distribution
  for (size_t i = 0; i < model_ids.size(); ++i) {
    int target_gpu = i % gpu_count;
    auto handle_or = store->prepare(model_ids[i], make_gpu_key(target_gpu));
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
  auto handle0 = store->prepare(model_id, make_gpu_key(0));
  auto handle1 = store->prepare(model_id, make_gpu_key(1));

  REQUIRE(handle0.ok());
  REQUIRE(handle1.ok());

  REQUIRE(handle0.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  REQUIRE(handle1.value().wait_ready(std::chrono::milliseconds(30000)).ok());

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
  REQUIRE(reg_info.ok());

  // Try to enable on already unloaded instance
  auto reg_fail = store->enable_remote_instance_access(instance0, ModelLocation::GPU);
  REQUIRE(!reg_fail.ok());

  // Disable remote access
  auto disable_status = store->disable_remote_instance_access(instance1, ModelLocation::GPU);
  REQUIRE(disable_status.ok());
}

// B6: Multi-GPU memory pressure handling
TEST_CASE("B6: Multi-GPU memory pressure", "[checkpoint_store][multi_gpu][b6]") {
  skip_if_insufficient_gpus(2, "B6");

  const size_t pool_size = 150 * 1024 * 1024; // 150MB pool (tight)
  const size_t large_model_size = 80 * 1024 * 1024; // 80MB
  const size_t small_model_size = 20 * 1024 * 1024; // 20MB

  TempModelFixture fixture("multi_gpu_b6");

  // Create models of different sizes
  std::string large_model = "large_model_b6";
  std::string small_model1 = "small_model1_b6";
  std::string small_model2 = "small_model2_b6";
  std::string small_model3 = "small_model3_b6";

  fixture.create_model(large_model, large_model_size);
  fixture.create_model(small_model1, small_model_size);
  fixture.create_model(small_model2, small_model_size);
  fixture.create_model(small_model3, small_model_size);

  auto store = make_test_store(fixture.root(), pool_size / (1024 * 1024));

  // Load small models first
  auto h1 = store->prepare(small_model1, make_gpu_key(0));
  auto h2 = store->prepare(small_model2, make_gpu_key(1));
  auto h3 = store->prepare(small_model3, make_gpu_key(0));

  REQUIRE(h1.ok());
  REQUIRE(h2.ok());
  REQUIRE(h3.ok());

  REQUIRE(h1.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  REQUIRE(h2.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  REQUIRE(h3.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  // Now try to load large model - should trigger eviction
  auto large_handle = store->prepare(large_model, make_gpu_key(1));

  if (large_handle.ok()) {
    auto status = large_handle.value().wait_ready(std::chrono::milliseconds(30000));

    // Check what got evicted
    int total_models = 0;
    total_models += store->get_loaded_devices(large_model).size();
    total_models += store->get_loaded_devices(small_model1).size();
    total_models += store->get_loaded_devices(small_model2).size();
    total_models += store->get_loaded_devices(small_model3).size();

    // With 150MB pool: can fit either (80MB + 2*20MB) or 3*20MB
    REQUIRE(total_models <= 3);

    // If large model loaded, at least one small model was evicted
    if (!store->get_loaded_devices(large_model).empty()) {
      REQUIRE(total_models < 4);
    }
  }

  // Verify memory consistency
  store->update_memory_pool_metrics();
  auto available = store->get_available_memory();
  REQUIRE(available <= pool_size);
}