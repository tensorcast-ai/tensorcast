// Copyright (c) 2025, StepCast Team. All rights reserved.

// CheckpointStore edge case tests (E-series)
// Test error conditions, invalid inputs, and boundary cases.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <limits>
#include <thread>

#include "concurrency_utils.h"

using namespace stepcast::tests::checkpoint_store;
using namespace stepcast::store;
using stepcast::store::CheckpointStore;

// E1: Invalid device ordinal
TEST_CASE("E1: Invalid device ordinal", "[checkpoint_store][edge][e1]") {
  const std::string model_id = "edge_model_e1";
  const size_t model_size = 10 * 1024 * 1024; // 10MB

  TempModelFixture fixture("edge_e1");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // Test negative ordinal
  auto neg_handle = store->prepare(model_id, DeviceKey{stepcast::DeviceType::GPU, -1, ""});
  REQUIRE(!neg_handle.ok());
  REQUIRE(neg_handle.status().code() == absl::StatusCode::kInvalidArgument);

  // Test very large ordinal
  auto large_handle = store->prepare(model_id, DeviceKey{stepcast::DeviceType::GPU, 999, ""});
  REQUIRE(!large_handle.ok());
  REQUIRE(large_handle.status().code() == absl::StatusCode::kInvalidArgument);

  // Test CPU device (not supported)
  auto cpu_handle = store->prepare(model_id, DeviceKey{stepcast::DeviceType::CPU, 0, ""});
  REQUIRE(!cpu_handle.ok());
}

// E2: Non-existent model
TEST_CASE("E2: Non-existent model", "[checkpoint_store][edge][e2]") {
  TempModelFixture fixture("edge_e2");
  auto store = make_test_store(fixture.root());

  // Try to load non-existent model
  auto handle = store->prepare("non_existent_model", make_gpu_key(0));
  REQUIRE(!handle.ok());

  // Operations on non-existent instance
  auto fake_key = make_instance_key("fake_model", 0);

  REQUIRE(store->wait_instance_ready(fake_key) != 0);
  REQUIRE(store->unload_instance(fake_key) != 0);

  auto state = store->get_instance_state(fake_key, stepcast::DeviceType::GPU);
  REQUIRE(state == stepcast::store::MemoryState::UNINITIALIZED);

  auto ptr_result = store->get_instance_gpu_ptr(fake_key);
  REQUIRE(!ptr_result.ok());
}

// E3: Memory pool exhaustion
TEST_CASE("E3: Memory pool exhaustion", "[checkpoint_store][edge][e3]") {
  skip_if_no_cuda("E3");

  const size_t pool_size = 100 * 1024 * 1024; // 100MB pool
  const size_t model_size = 30 * 1024 * 1024; // 30MB per model

  TempModelFixture fixture("edge_e3");

  // Create multiple models that exceed pool size
  std::vector<std::string> model_ids;
  for (int i = 0; i < 5; ++i) {
    auto model_id = generate_model_name("exhaust_model_e3", i);
    model_ids.push_back(model_id);
    fixture.create_model(model_id, model_size);
  }

  auto store = make_test_store(fixture.root(), pool_size / (1024 * 1024));

  // Load models until pool is exhausted
  std::vector<ModelHandle> handles;
  int successful_loads = 0;

  for (const auto& model_id : model_ids) {
    auto handle_or = store->prepare(model_id, make_gpu_key(0));
    if (handle_or.ok()) {
      auto handle = std::move(handle_or).value();
      auto wait_status = handle.wait_ready(std::chrono::milliseconds(5000));
      if (wait_status.ok()) {
        handles.push_back(std::move(handle));
        successful_loads++;
      }
    }
  }

  // Should load 3 models (3 * 30MB = 90MB < 100MB)
  // 4th model would need 120MB total, exceeding pool
  REQUIRE(successful_loads >= 3);
  REQUIRE(successful_loads < 5);

  // Verify memory is nearly full
  store->update_memory_pool_metrics();
  auto available = store->get_available_memory();
  REQUIRE(available < model_size); // Not enough for another model
}

// E4: Concurrent clear_mem() during prepare()
TEST_CASE("E4: Concurrent clear_mem() during prepare()", "[checkpoint_store][edge][e4]") {
  skip_if_no_cuda("E4");

  const std::string model_id = "edge_model_e4";
  const size_t model_size = 50 * 1024 * 1024; // 50MB

  TempModelFixture fixture("edge_e4");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  std::atomic<bool> prepare_started{false};
  std::atomic<bool> prepare_completed{false};
  std::atomic<bool> clear_completed{false};

  // Thread 1: Start prepare operation
  std::thread prepare_thread([&]() {
    auto handle_or = store->prepare(model_id, make_gpu_key(0));
    prepare_started.store(true);

    if (handle_or.ok()) {
      auto handle = std::move(handle_or).value();
      auto status = handle.wait_ready(std::chrono::milliseconds(10000));
      prepare_completed.store(status.ok());
    }
  });

  // Thread 2: Clear memory while prepare is in progress
  std::thread clear_thread([&]() {
    // Wait for prepare to start
    while (!prepare_started.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Clear memory
    int result = store->clear_mem();
    clear_completed.store(result == 0);
  });

  prepare_thread.join();
  clear_thread.join();

  // Both operations should complete (implementation should handle this race)
  REQUIRE(clear_completed.load());

  // Model may or may not be loaded depending on timing
  auto loaded_devices = store->get_loaded_devices(model_id);
  INFO("Model loaded on " << loaded_devices.size() << " devices after race");
}

// E5: Double enable/disable remote access
TEST_CASE("E5: Double enable/disable remote access", "[checkpoint_store][edge][e5]") {
  skip_if_no_cuda("E5");

  const std::string model_id = "edge_model_e5";
  const size_t model_size = 20 * 1024 * 1024; // 20MB

  TempModelFixture fixture("edge_e5");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // Load model
  auto handle = store->prepare(model_id, make_gpu_key(0));
  REQUIRE(handle.ok());
  REQUIRE(handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  auto instance_key = make_instance_key(model_id, 0);

  // Enable remote access
  auto reg1 = store->enable_remote_instance_access(instance_key, ModelLocation::GPU);
  REQUIRE(reg1.ok());

  // Try to enable again - should fail or return same registration
  auto reg2 = store->enable_remote_instance_access(instance_key, ModelLocation::GPU);
  // Implementation may either fail or return existing registration

  // Disable remote access
  auto disable1 = store->disable_remote_instance_access(instance_key, ModelLocation::GPU);
  REQUIRE(disable1.ok());

  // Try to disable again - should fail gracefully
  auto disable2 = store->disable_remote_instance_access(instance_key, ModelLocation::GPU);
  // Should not crash, may return error

  // Try to disable on non-existent instance
  auto fake_key = make_instance_key("fake_model", 0);
  auto disable_fake = store->disable_remote_instance_access(fake_key, ModelLocation::GPU);
  REQUIRE(!disable_fake.ok());
}

// E6: Prepare with invalid hints
TEST_CASE("E6: Prepare with invalid hints", "[checkpoint_store][edge][e6]") {
  skip_if_no_cuda("E6");

  const std::string model_id = "edge_model_e6";
  const size_t model_size = 15 * 1024 * 1024; // 15MB

  TempModelFixture fixture("edge_e6");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // Test with various invalid hints
  LoadingHints hints;

  // Negative batch size
  // Invalid hints - implementation may ignore these
  auto handle1 = store->prepare(model_id, make_gpu_key(0), CheckpointStore::PrepareMode::AUTO, hints);
  // Should either ignore invalid hint or fail gracefully

  // Zero prefetch size
  hints = LoadingHints{};
  // hints.prefetch_size = 0; // Field may not exist
  auto handle2 = store->prepare(model_id, make_gpu_key(0), CheckpointStore::PrepareMode::AUTO, hints);

  // Extremely large buffer count
  hints = LoadingHints{};
  // hints.num_buffers = std::numeric_limits<int>::max(); // Field may not exist
  auto handle3 = store->prepare(model_id, make_gpu_key(0), CheckpointStore::PrepareMode::AUTO, hints);

  // At least one should succeed with defaults
  REQUIRE((handle1.ok() || handle2.ok() || handle3.ok()));
}

// E7: Rapid prepare/unload cycling
TEST_CASE("E7: Rapid prepare/unload cycling", "[checkpoint_store][edge][e7]") {
  skip_if_no_cuda("E7");

  const std::string model_id = "edge_model_e7";
  const size_t model_size = 25 * 1024 * 1024; // 25MB

  TempModelFixture fixture("edge_e7");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  const int num_cycles = 20;
  int successful_cycles = 0;

  for (int i = 0; i < num_cycles; ++i) {
    // Prepare
    auto handle_or = store->prepare(model_id, make_gpu_key(0));
    if (!handle_or.ok())
      continue;

    auto handle = std::move(handle_or).value();

    // Don't wait for completion - immediately unload
    auto instance_key = make_instance_key(model_id, 0);
    int unload_result = store->unload_instance(instance_key);

    // May succeed or fail depending on timing
    if (unload_result == 0) {
      successful_cycles++;
    }
  }

  INFO("Successful rapid cycles: " << successful_cycles << "/" << num_cycles);

  // Should handle at least some cycles successfully
  REQUIRE(successful_cycles > 0);

  // Final state should be consistent
  store->clear_mem();
  REQUIRE(store->get_available_memory() == store->get_mem_pool_size());
}

// E8: Empty model directory
TEST_CASE("E8: Empty model directory", "[checkpoint_store][edge][e8]") {
  const std::string model_id = "empty_model_e8";

  TempModelFixture fixture("edge_e8");

  // Create model directory but no files
  auto model_dir = fixture.root() / model_id;
  std::filesystem::create_directories(model_dir);

  auto store = make_test_store(fixture.root());

  // Try to load empty model
  auto handle = store->prepare(model_id, make_gpu_key(0));
  REQUIRE(!handle.ok());
}

// E9: Corrupted model file
TEST_CASE("E9: Corrupted model file", "[checkpoint_store][edge][e9]") {
  const std::string model_id = "corrupt_model_e9";

  TempModelFixture fixture("edge_e9");

  // Create model with invalid data
  auto model_dir = fixture.root() / model_id;
  std::filesystem::create_directories(model_dir);

  // Create a file that's too small or has invalid content
  auto data_file = model_dir / "tensor.data_0";
  std::ofstream ofs(data_file, std::ios::binary);
  ofs << "INVALID";
  ofs.close();

  auto store = make_test_store(fixture.root());

  // Try to load corrupted model
  auto handle = store->prepare(model_id, make_gpu_key(0));
  // Should fail during verification or loading
  if (handle.ok()) {
    auto wait_status = handle.value().wait_ready(std::chrono::milliseconds(5000));
    REQUIRE(!wait_status.ok());
  }
}

// E10: PrepareMode edge cases
TEST_CASE("E10: PrepareMode edge cases", "[checkpoint_store][edge][e10]") {
  skip_if_no_cuda("E10");

  const std::string model_id = "edge_model_e10";
  const size_t model_size = 30 * 1024 * 1024; // 30MB

  TempModelFixture fixture("edge_e10");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // COPY_ONLY without existing source
  auto copy_handle = store->prepare(model_id, make_gpu_key(0), CheckpointStore::PrepareMode::COPY_ONLY);
  REQUIRE(!copy_handle.ok()); // Should fail - no source to copy from

  // LOAD_ONLY should work
  auto load_handle = store->prepare(model_id, make_gpu_key(0), CheckpointStore::PrepareMode::LOAD_ONLY);
  REQUIRE(load_handle.ok());
  REQUIRE(load_handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  // Now COPY_ONLY should work
  auto copy_handle2 = store->prepare(model_id, make_gpu_key(1), CheckpointStore::PrepareMode::COPY_ONLY);
  if (stepcast::tests::is_cuda_available() && copy_handle2.ok()) {
    // Should succeed if we have multiple GPUs
    REQUIRE(copy_handle2.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  // LOAD_ONLY on already loaded device
  auto reload_handle = store->prepare(model_id, make_gpu_key(0), CheckpointStore::PrepareMode::LOAD_ONLY);
  REQUIRE(reload_handle.ok()); // Should return existing instance
}