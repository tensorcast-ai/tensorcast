// Copyright (c) 2025-2026, TensorCast Team.

// StoreEngine concurrency tests (A-series)
// Test concurrent materialize_replica() and unload_replica() operations.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <thread>
#include <unordered_set>

#include "core/common/logging_init.h"
#include "core/testing/concurrency_utils.h"

using namespace tensorcast::testing;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::DiskSource;
using tensorcast::store::loading::MaterializeHints;

namespace {
struct LoggingInitializer {
  LoggingInitializer() {
    tensorcast::common::ensure_logging_initialized();
  }
};

const LoggingInitializer kLoggingInitializer; // NOLINT(cert-err58-cpp)
} // namespace

// A0: Single-thread smoke test for materialize + unload flow
TEST_CASE("A0: Smoke materialize/unload flow", "[store_engine][concurrency][a0]") {
  skip_if_no_cuda("A0");

  const std::string artifact_id = "cgid:concurrent_model_a0";
  const size_t artifact_size = 8 * 1024 * 1024; // 8MB

  TempArtifactFixture fixture("concurrency_a0");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

  auto store = make_test_store(fixture.root());

  MaterializeHints hints;
  hints.artifact_id = artifact_id;
  auto handle_or =
      store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
  REQUIRE(handle_or.ok());

  auto handle = std::move(handle_or).value();
  REQUIRE(handle.wait_ready(std::chrono::milliseconds(30000)).ok());

  auto loaded_devices = store->get_resident_devices(artifact_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].type == tensorcast::DeviceType::GPU);
  REQUIRE(loaded_devices[0].ordinal == 0);

  auto replica_key = make_replica_key(artifact_id, 0);
  REQUIRE(store->unload_replica(replica_key) == 0);
  REQUIRE(store->get_resident_devices(artifact_id).empty());
}

// A1: 32 threads calling materialize_replica() on the same replica
TEST_CASE("A1: Concurrent materialize_replica() same replica", "[store_engine][concurrency][a1]") {
  skip_if_no_cuda("A1");

  const int num_threads = 32;
  const std::string artifact_id = "cgid:concurrent_model_a1";
  const size_t artifact_size = 10 * 1024 * 1024; // 10MB

  TempArtifactFixture fixture("concurrency_a1");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

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
      LOG(INFO) << "Thread " << thread_id << " preparing replica " << artifact_id;
      MaterializeHints hints;
      hints.artifact_id = artifact_id;
      auto handle_or =
          store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);

      LoadResult result;
      result.artifact_id = artifact_id;
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
      LOG(INFO) << "Thread " << thread_id << " materialize_replicad replica " << artifact_id << " in "
                << result.load_time.count() << "ms";

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

  // Verify single allocation - replica should be loaded exactly once
  auto loaded_devices = store->get_resident_devices(artifact_id);
  REQUIRE(loaded_devices.size() == 1);
  REQUIRE(loaded_devices[0].type == tensorcast::DeviceType::GPU);
  REQUIRE(loaded_devices[0].ordinal == 0);

  // Verify memory pool consistency
  INFO("Total test time: " << total_time.count() << "ms");
}

// A2: Multiple threads calling materialize_replica() on different artifacts
TEST_CASE("A2: Concurrent materialize_replica() different artifacts", "[store_engine][concurrency][a2]") {
  skip_if_no_cuda("A2");

  const int num_threads = 16;
  const int num_models = 8; // number of distinct artifacts

  TempArtifactFixture fixture("concurrency_a2");

  // Create multiple artifacts
  std::vector<std::string> artifact_ids;
  for (int i = 0; i < num_models; ++i) {
    auto artifact_id = generate_artifact_id("artifact", i);
    auto canonical_id = std::string("cgid:") + artifact_id;
    artifact_ids.push_back(canonical_id);
    fixture.create_model(canonical_id, random_model_size(5, 20));
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

      // Each thread loads a random replica
      const auto& artifact_id = artifact_ids[thread_id % num_models];

      auto load_start = std::chrono::high_resolution_clock::now();
      MaterializeHints hints;
      hints.artifact_id = artifact_id;
      DiskSource disk_source{.path = fixture.root() / artifact_id, .expected_size = std::nullopt};
      auto handle_or =
          store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);

      LoadResult result;
      result.artifact_id = artifact_id;
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

  // Verify each artifact has at most one replica per device
  std::unordered_set<std::string> loaded_artifacts;
  for (const auto& artifact_id : artifact_ids) {
    auto devices = store->get_resident_devices(artifact_id);
    if (!devices.empty()) {
      REQUIRE(devices.size() == 1);
      loaded_artifacts.insert(artifact_id);
    }
  }

  // At least some artifacts should be loaded
  REQUIRE(!loaded_artifacts.empty());
  REQUIRE(loaded_artifacts.size() <= num_models);
}

// A3: Concurrent materialize_replica() and unload_replica()
TEST_CASE("A3: Concurrent materialize_replica() and unload_replica()", "[store_engine][concurrency][a3]") {
  skip_if_no_cuda("A3");

  const int num_loaders = 8;
  const int num_unloaders = 8;
  const std::string artifact_id = "cgid:concurrent_model_a3";
  const size_t artifact_size = 20 * 1024 * 1024; // 20MB

  TempArtifactFixture fixture("concurrency_a3");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

  auto store = make_test_store(fixture.root());

  // First, load the replica
  {
    MaterializeHints hints;
    hints.artifact_id = artifact_id;
    auto initial_handle =
        store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
    REQUIRE(initial_handle.ok());
    REQUIRE(initial_handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  }

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
        MaterializeHints hints;
        hints.artifact_id = artifact_id;
        auto handle_or =
            store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
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
        auto key = make_replica_key(artifact_id, 0);
        int result = store->unload_replica(key);
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
TEST_CASE("A4: Concurrent unload_replica() same replica", "[store_engine][concurrency][a4]") {
  skip_if_no_cuda("A4");

  const int num_threads = 16;
  const std::string artifact_id = "cgid:concurrent_model_a4";
  const size_t artifact_size = 15 * 1024 * 1024; // 15MB

  TempArtifactFixture fixture("concurrency_a4");
  auto artifact_dir = fixture.create_model(artifact_id, artifact_size);
  DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};

  auto store = make_test_store(fixture.root());

  // Load the replica
  MaterializeHints hints;
  hints.artifact_id = artifact_id;
  auto handle =
      store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
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

      auto key = make_replica_key(artifact_id, 0);
      int result = store->unload_replica(key);

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

  // unload_replica is idempotent for existing replicas; all callers should succeed.
  REQUIRE(successful_unloads.load() == num_threads);
  REQUIRE(failed_unloads.load() == 0);

  // Artifact should no longer be loaded
  auto loaded_devices = store->get_resident_devices(artifact_id);
  REQUIRE(loaded_devices.empty());
}

// A5: Concurrent clear_mem() operations
TEST_CASE("A5: Concurrent clear_mem()", "[store_engine][concurrency][a5]") {
  skip_if_no_cuda("A5");

  const int num_threads = 8;
  const int num_models = 5; // number of artifacts

  TempArtifactFixture fixture("concurrency_a5");

  // Create and load multiple artifacts
  auto store = make_test_store(fixture.root(), 512); // 512MB pool

  for (int i = 0; i < num_models; ++i) {
    auto artifact_id = generate_artifact_id("artifact", i);
    auto canonical_id = std::string("cgid:") + artifact_id;
    auto artifact_dir = fixture.create_model(canonical_id, 10 * 1024 * 1024); // 10MB each

    MaterializeHints hints;
    hints.artifact_id = canonical_id;
    DiskSource disk_source{.path = artifact_dir, .expected_size = std::nullopt};
    auto handle =
        store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints, disk_source);
    REQUIRE(handle.ok());
    REQUIRE(handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  // Verify artifacts are loaded
  REQUIRE(store->list_device_replicas(make_gpu_key(0)).size() == num_models);

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

  // No artifacts should be loaded
  REQUIRE(store->list_device_replicas(make_gpu_key(0)).empty());
}
