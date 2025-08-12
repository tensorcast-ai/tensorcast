// Copyright (c) 2025, StepCast Team. All rights reserved.

// CheckpointStore concurrency tests (A-series)
// Test concurrent prepare() and unload_instance() operations.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <unordered_set>

#include "core/common/logging_init.h"
#include "core/store/concurrency_utils.h"

using namespace stepcast::tests::checkpoint_store;
using namespace stepcast::store;

namespace {
struct LoggingInitializer {
  LoggingInitializer() {
    stepcast::store::ensure_logging_initialized();
  }
};

const LoggingInitializer kLoggingInitializer; // NOLINT(cert-err58-cpp)
} // namespace

// A1: 32 threads calling prepare() on the same model
TEST_CASE("A1: Concurrent prepare() same model", "[checkpoint_store][concurrency][a1]") {
  skip_if_no_cuda("A1");

  const int num_threads = 32;
  const std::string model_id = "concurrent_model_a1";
  const size_t model_size = 10 * 1024 * 1024; // 10MB

  TempModelFixture fixture("concurrency_a1");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root(), 1024); // 1GB pool for 32 threads x 10MB
  ThreadBarrier barrier(num_threads);
  ConcurrentLoadTracker tracker;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto start_time = std::chrono::high_resolution_clock::now();

  // Launch threads
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, thread_id = i]() {
      barrier.arrive_and_wait(); // Synchronize start

      auto load_start = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Thread " << thread_id << " preparing model " << model_id;
      auto handle_or = store->prepare(model_id, make_gpu_key(0));

      LoadResult result;
      result.model_id = model_id;
      result.device = make_gpu_key(0);

      if (handle_or.ok()) {
        auto handle = std::move(handle_or).value();
        auto wait_status = handle.wait_ready(std::chrono::milliseconds(30000));
        result.success = wait_status.ok();
        if (!wait_status.ok()) {
          result.error_message = wait_status.ToString();
        }
      } else {
        result.success = false;
        result.error_message = handle_or.status().ToString();
      }

      auto load_end = std::chrono::high_resolution_clock::now();
      result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
      LOG(INFO) << "Thread " << thread_id << " prepared model " << model_id << " in " << result.load_time.count()
                << "ms";

      tracker.record_load(result);
    });
  }

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  auto total_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);

  // Verify results
  REQUIRE(tracker.successful_loads() == num_threads);
  REQUIRE(tracker.failed_loads() == 0);

  // Verify single allocation - model should be loaded exactly once
  auto loaded_devices = store->get_loaded_devices(model_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].type == stepcast::DeviceType::GPU);
  REQUIRE(loaded_devices[0].ordinal == 0);

  // Verify memory pool consistency
  INFO("Total test time: " << total_time.count() << "ms");
}

// A2: Multiple threads calling prepare() on different models
TEST_CASE("A2: Concurrent prepare() different models", "[checkpoint_store][concurrency][a2]") {
  skip_if_no_cuda("A2");

  const int num_threads = 16;
  const int num_models = 8;

  TempModelFixture fixture("concurrency_a2");

  // Create multiple models
  std::vector<std::string> model_ids;
  for (int i = 0; i < num_models; ++i) {
    auto model_id = generate_model_name("model_a2", i);
    model_ids.push_back(model_id);
    fixture.create_model(model_id, random_model_size(5, 20));
  }

  auto store = make_test_store(fixture.root(), 1024); // 1GB pool
  ConcurrentLoadTracker tracker;
  ThreadBarrier barrier(num_threads);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Launch threads
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, thread_id = i]() {
      barrier.arrive_and_wait();

      // Each thread loads a random model
      const auto& model_id = model_ids[thread_id % num_models];

      auto load_start = std::chrono::high_resolution_clock::now();
      auto handle_or = store->prepare(model_id, make_gpu_key(0));

      LoadResult result;
      result.model_id = model_id;
      result.device = make_gpu_key(0);

      if (handle_or.ok()) {
        auto handle = std::move(handle_or).value();
        auto wait_status = handle.wait_ready(std::chrono::milliseconds(30000));
        result.success = wait_status.ok();
        if (!wait_status.ok()) {
          result.error_message = wait_status.ToString();
        }
      } else {
        result.success = false;
        result.error_message = handle_or.status().ToString();
      }

      auto load_end = std::chrono::high_resolution_clock::now();
      result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);

      tracker.record_load(result);
    });
  }

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  // Verify results
  REQUIRE(tracker.successful_loads() == num_threads);

  // Verify each model is loaded at most once per device
  std::unordered_set<std::string> loaded_models;
  for (const auto& model_id : model_ids) {
    auto devices = store->get_loaded_devices(model_id);
    if (!devices.empty()) {
      REQUIRE(devices.size() == 1);
      loaded_models.insert(model_id);
    }
  }

  // At least some models should be loaded
  REQUIRE(!loaded_models.empty());
  REQUIRE(loaded_models.size() <= num_models);
}

// A3: Concurrent prepare() and unload_instance()
TEST_CASE("A3: Concurrent prepare() and unload_instance()", "[checkpoint_store][concurrency][a3]") {
  skip_if_no_cuda("A3");

  const int num_loaders = 8;
  const int num_unloaders = 8;
  const std::string model_id = "concurrent_model_a3";
  const size_t model_size = 20 * 1024 * 1024; // 20MB

  TempModelFixture fixture("concurrency_a3");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // First, load the model
  auto initial_handle = store->prepare(model_id, make_gpu_key(0));
  REQUIRE(initial_handle.ok());
  REQUIRE(initial_handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  ThreadBarrier barrier(num_loaders + num_unloaders);
  std::atomic<int> successful_loads{0};
  std::atomic<int> successful_unloads{0};
  std::atomic<bool> stop_flag{false};

  std::vector<std::thread> threads;

  // Loader threads
  threads.reserve(num_loaders);
  for (int i = 0; i < num_loaders; ++i) {
    threads.emplace_back([&]() {
      barrier.arrive_and_wait();

      while (!stop_flag.load()) {
        auto handle_or = store->prepare(model_id, make_gpu_key(0));
        if (handle_or.ok()) {
          auto handle = std::move(handle_or).value();
          if (handle.wait_ready(std::chrono::milliseconds(1000)).ok()) {
            successful_loads.fetch_add(1);
          }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }

  // Unloader threads
  for (int i = 0; i < num_unloaders; ++i) {
    threads.emplace_back([&]() {
      barrier.arrive_and_wait();

      while (!stop_flag.load()) {
        auto key = make_instance_key(model_id, 0);
        int result = store->unload_instance(key);
        if (result == 0) {
          successful_unloads.fetch_add(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
  }

  // Let threads run for a while
  std::this_thread::sleep_for(std::chrono::seconds(3));
  stop_flag.store(true);

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  // Verify operations completed
  REQUIRE(successful_loads.load() > 0);
  REQUIRE(successful_unloads.load() > 0);
}

// A4: Concurrent unload of the same instance
TEST_CASE("A4: Concurrent unload_instance() same model", "[checkpoint_store][concurrency][a4]") {
  skip_if_no_cuda("A4");

  const int num_threads = 16;
  const std::string model_id = "concurrent_model_a4";
  const size_t model_size = 15 * 1024 * 1024; // 15MB

  TempModelFixture fixture("concurrency_a4");
  fixture.create_model(model_id, model_size);

  auto store = make_test_store(fixture.root());

  // Load the model
  auto handle = store->prepare(model_id, make_gpu_key(0));
  REQUIRE(handle.ok());
  REQUIRE(handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());

  ThreadBarrier barrier(num_threads);
  std::atomic<int> successful_unloads{0};
  std::atomic<int> failed_unloads{0};

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Launch threads to unload the same instance
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      barrier.arrive_and_wait();

      auto key = make_instance_key(model_id, 0);
      int result = store->unload_instance(key);

      if (result == 0) {
        successful_unloads.fetch_add(1);
      } else {
        failed_unloads.fetch_add(1);
      }
    });
  }

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  // Exactly one thread should succeed in unloading
  REQUIRE(successful_unloads.load() == 1);
  REQUIRE(failed_unloads.load() == num_threads - 1);

  // Model should no longer be loaded
  auto loaded_devices = store->get_loaded_devices(model_id);
  REQUIRE(loaded_devices.empty());
}

// A5: Concurrent clear_mem() operations
TEST_CASE("A5: Concurrent clear_mem()", "[checkpoint_store][concurrency][a5]") {
  skip_if_no_cuda("A5");

  const int num_threads = 8;
  const int num_models = 5;

  TempModelFixture fixture("concurrency_a5");

  // Create and load multiple models
  auto store = make_test_store(fixture.root(), 512); // 512MB pool

  for (int i = 0; i < num_models; ++i) {
    auto model_id = generate_model_name("model_a5", i);
    fixture.create_model(model_id, 10 * 1024 * 1024); // 10MB each

    auto handle = store->prepare(model_id, make_gpu_key(0));
    REQUIRE(handle.ok());
    REQUIRE(handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  // Verify models are loaded
  REQUIRE(store->list_device_models(make_gpu_key(0)).size() == num_models);

  ThreadBarrier barrier(num_threads);
  std::atomic<int> successful_clears{0};

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  // Launch threads to clear memory
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      barrier.arrive_and_wait();

      int result = store->clear_mem();
      if (result == 0) {
        successful_clears.fetch_add(1);
      }
    });
  }

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  // All threads should succeed (clear_mem is idempotent)
  REQUIRE(successful_clears.load() == num_threads);

  // No models should be loaded
  REQUIRE(store->list_device_models(make_gpu_key(0)).empty());
}