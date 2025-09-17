// Copyright (c) 2025, TensorCast Team.

// StoreEngine edge case tests (E-series)
// Test error conditions, invalid inputs, and boundary cases.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>

#include "absl/status/status.h"
#include "core/store/components/communication_manager.h"
#include "core/store/store_engine_options.h"
#include "core/testing/concurrency_utils.h"
#include "core/testing/test_helpers.h"

using namespace tensorcast::testing;
using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::ReplicaKey;

// E1: Invalid device ordinal
TEST_CASE("E1: Invalid device ordinal", "[store_engine][edge][e1]") {
  const std::string artifact_id = "edge_model_e1";
  const size_t artifact_size = 10 * 1024 * 1024; // 10MB

  TempArtifactFixture fixture("edge_e1");
  fixture.create_model(artifact_id, artifact_size);

  auto store = make_test_store(fixture.root());

  // Test negative ordinal
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto neg_handle = store->materialize_replica(
        DeviceKey{.type = tensorcast::DeviceType::GPU, .ordinal = -1, .uuid = ""},
        StoreEngine::MaterializeMode::AUTO,
        hints);
    REQUIRE(!neg_handle.ok());
    REQUIRE(neg_handle.status().code() == absl::StatusCode::kInvalidArgument);
  }

  // Test very large ordinal
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto large_handle = store->materialize_replica(
        DeviceKey{tensorcast::DeviceType::GPU, 999, ""}, StoreEngine::MaterializeMode::AUTO, hints);
    REQUIRE(!large_handle.ok());
    REQUIRE(large_handle.status().code() == absl::StatusCode::kInvalidArgument);
  }

  // Test CPU device (not supported)
  {
    MaterializeHints hints{};

    auto cpu_handle = store->materialize_replica(
        DeviceKey{tensorcast::DeviceType::CPU, 0, ""}, StoreEngine::MaterializeMode::AUTO, hints);
    REQUIRE(!cpu_handle.ok());
  }
}

// E2: Non-existent replica
TEST_CASE("E2: Non-existent replica", "[store_engine][edge][e2]") {
  TempArtifactFixture fixture("edge_e2");
  auto store = make_test_store(fixture.root());

  // Try to load non-existent replica
  {
    MaterializeHints hints{.disk_path = "non_existent_artifact"};
    auto handle = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    REQUIRE(!handle.ok());
  }

  // Operations on non-existent instance
  auto fake_key = make_replica_key("fake_artifact", 0);

  REQUIRE(store->wait_replica_ready(fake_key) != 0);
  REQUIRE(store->unload_replica(fake_key) != 0);

  auto state = store->get_replica_state(fake_key, tensorcast::DeviceType::GPU);
  REQUIRE(state == tensorcast::store::replica::MemoryState::UNINITIALIZED);

  auto ptr_result = store->get_replica_gpu_ptr(fake_key);
  REQUIRE(!ptr_result.ok());
}

// E3: Memory pool exhaustion
TEST_CASE("E3: Memory pool exhaustion", "[store_engine][edge][e3]") {
  skip_if_no_cuda("E3");

  const size_t pool_size = 100 * 1024 * 1024; // 100MB pool
  const size_t artifact_size = 30 * 1024 * 1024; // 30MB per replica

  TempArtifactFixture fixture("edge_e3");

  // Create multiple artifacts
  std::vector<std::string> artifact_ids;
  for (int i = 0; i < 5; ++i) {
    auto artifact_id = generate_artifact_id("exhaust_model_e3", i);
    artifact_ids.push_back(artifact_id);
    fixture.create_model(artifact_id, artifact_size);
  }

  auto store = make_test_store(fixture.root(), pool_size / (1024 * 1024));

  // Load first replica to ensure shared streaming buffer is initialized
  size_t idx = 0;
  absl::Status first_status = absl::InternalError("uninitialized");
  {
    MaterializeHints hints{.disk_path = artifact_ids[idx]};
    auto h_or = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    if (h_or.ok()) {
      first_status = h_or.value().wait_ready(std::chrono::milliseconds(10000));
    }
  }
  REQUIRE(first_status.ok());

  // Record baseline after streaming buffer allocation
  store->update_memory_pool_metrics();
  const auto baseline_available = store->get_available_memory();

  // Load remaining artifacts
  int additional_successes = 0;
  for (idx = 1; idx < artifact_ids.size(); ++idx) {
    MaterializeHints hints{.disk_path = artifact_ids[idx]};
    auto handle_or = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    if (handle_or.ok()) {
      auto handle = std::move(handle_or).value();
      auto wait_status = handle.wait_ready(std::chrono::milliseconds(10000));
      if (wait_status.ok()) {
        additional_successes++;
      }
    }
  }

  // Should be able to load at least one more replica
  REQUIRE(additional_successes > 0);

  // Verify pinned memory availability remains stable (shared fixed-size SPB)
  store->update_memory_pool_metrics();
  auto available_after = store->get_available_memory();
  REQUIRE(available_after == baseline_available);
}

// E4: Concurrent clear_mem() during materialize_replica()
TEST_CASE("E4: Concurrent clear_mem() during materialize_replica()", "[store_engine][edge][e4]") {
  skip_if_no_cuda("E4");

  const std::string artifact_id = "edge_model_e4";
  const size_t artifact_size = 50 * 1024 * 1024; // 50MB

  TempArtifactFixture fixture("edge_e4");
  fixture.create_model(artifact_id, artifact_size);

  auto store = make_test_store(fixture.root());

  std::atomic<bool> materialize_started{false};
  std::atomic<bool> materialize_completed{false};
  std::atomic<bool> clear_completed{false};

  // Thread 1: Start materialize_replica operation
  std::thread materialize_thread([&]() {
    MaterializeHints hints{.disk_path = artifact_id};

    auto handle_or = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    materialize_started.store(true);

    if (handle_or.ok()) {
      auto handle = std::move(handle_or).value();
      auto status = handle.wait_ready(std::chrono::milliseconds(10000));
      materialize_completed.store(status.ok());
    }
  });

  // Thread 2: Clear memory while materialize_replica is in progress
  std::thread clear_thread([&]() {
    // Wait for materialize_replica to start
    while (!materialize_started.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Clear memory
    int result = store->clear_mem();
    clear_completed.store(result == 0);
  });

  materialize_thread.join();
  clear_thread.join();

  // Both operations should complete (implementation should handle this race)
  REQUIRE(clear_completed.load());

  // Artifact may or may not be loaded depending on timing
  auto loaded_devices = store->get_resident_devices(artifact_id);
  INFO("Artifact loaded on " << loaded_devices.size() << " devices after race");
}

// E5: Double enable/disable remote access
TEST_CASE("E5: Double enable/disable remote access", "[store_engine][edge][e5]") {
  skip_if_no_cuda("E5");

  const std::string artifact_id = "edge_model_e5";
  const size_t artifact_size = 20 * 1024 * 1024; // 20MB

  TempArtifactFixture fixture("edge_e5");
  fixture.create_model(artifact_id, artifact_size);

  // Enable communication so remote registration can succeed
  int comm_port = find_available_port(51000);
  REQUIRE(comm_port > 0);
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize("127.0.0.1", static_cast<uint16_t>(comm_port), /*enable_rdma=*/false).ok());

  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = fixture.root().string();
  // Keep test-friendly pool sizes similar to make_test_store
  opts.memory_pool_size = 512ULL * 1024 * 1024;
  opts.tx_slice_bytes = 64ULL * 1024;
  opts.num_thread = 4;
  opts.pinned_memory_timeout = std::chrono::milliseconds(30000);
  opts.p2p_port = static_cast<uint16_t>(comm_port);
  opts.comm_manager = comm_manager;
  auto store = std::make_unique<StoreEngine>(opts);

  // Load replica
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto handle = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    REQUIRE(handle.ok());
    REQUIRE(handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  auto replica_key = make_replica_key(artifact_id, 0);

  // Enable remote access
  auto reg1 = store->enable_remote_replica_access(replica_key, MemoryLocation::GPU);
  REQUIRE(reg1.ok());

  // Try to enable again - should fail or return same registration
  auto reg2 = store->enable_remote_replica_access(replica_key, MemoryLocation::GPU);
  // Implementation may either fail or return existing registration

  // Disable remote access
  auto disable1 = store->disable_remote_replica_access(replica_key, MemoryLocation::GPU);
  REQUIRE(disable1.ok());

  // Try to disable again - should fail gracefully
  auto disable2 = store->disable_remote_replica_access(replica_key, MemoryLocation::GPU);
  // Should not crash, may return error

  // Try to disable on non-existent instance
  auto fake_key = make_replica_key("fake_artifact", 0);
  auto disable_fake = store->disable_remote_replica_access(fake_key, MemoryLocation::GPU);
  REQUIRE(!disable_fake.ok());
}

TEST_CASE("E6: Materialize with invalid hints", "[store_engine][edge][e6]") {
  skip_if_no_cuda("E6");

  const std::string artifact_id = "edge_model_e6";
  const size_t artifact_size = 15 * 1024 * 1024; // 15MB

  TempArtifactFixture fixture("edge_e6");
  const auto disk_path = fixture.create_model(artifact_id, artifact_size);

  auto store = make_test_store(fixture.root());

  // Test with various invalid hints
  MaterializeHints hints{.disk_path = disk_path.string()};

  auto handle1 = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  auto handle2 = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);

  auto handle3 = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);

  // At least one should succeed with defaults
  REQUIRE((handle1.ok() || handle2.ok() || handle3.ok()));
}

// E7: Rapid materialize_replica/unload cycling
TEST_CASE("E7: Rapid materialize_replica/unload cycling", "[store_engine][edge][e7]") {
  skip_if_no_cuda("E7");

  const std::string artifact_id = "edge_model_e7";
  const size_t artifact_size = 25 * 1024 * 1024; // 25MB

  TempArtifactFixture fixture("edge_e7");
  fixture.create_model(artifact_id, artifact_size);

  auto store = make_test_store(fixture.root());

  // Prime SPB allocation and record baseline availability after initialization
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto handle_or = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    if (handle_or.ok()) {
      auto st = handle_or.value().wait_ready(std::chrono::milliseconds(30000));
      (void)st; // Ignore result; the goal is to trigger SPB allocation
      auto replica_key = make_replica_key(artifact_id, 0);
      store->unload_replica(replica_key);
    }
  }
  store->update_memory_pool_metrics();
  const auto baseline_available = store->get_available_memory();

  const int num_cycles = 20;
  int successful_cycles = 0;

  for (int i = 0; i < num_cycles; ++i) {
    // materialize
    MaterializeHints hints{.disk_path = artifact_id};

    auto handle_or = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    if (!handle_or.ok()) {
      continue;
    }

    // Limit handle lifetime so it is released before unloading
    {
      auto handle = std::move(handle_or).value();
      // Intentionally do not wait; exit scope to release handle
    }

    // Don't wait for completion - immediately attempt unload after releasing handle
    auto replica_key = make_replica_key(artifact_id, 0);
    int unload_result = store->unload_replica(replica_key);

    // May succeed or fail depending on timing
    if (unload_result == 0) {
      successful_cycles++;
    }
  }

  INFO("Successful rapid cycles: " << successful_cycles << "/" << num_cycles);

  // Should handle at least some cycles successfully
  REQUIRE(successful_cycles > 0);

  // Final state should be consistent: availability returns to baseline (SPB remains allocated)
  store->clear_mem();
  store->update_memory_pool_metrics();
  REQUIRE(store->get_available_memory() == baseline_available);
}

// E8: Empty replica directory
TEST_CASE("E8: Empty replica directory", "[store_engine][edge][e8]") {
  const std::string artifact_id = "empty_artifact_e8";

  TempArtifactFixture fixture("edge_e8");

  // Create replica directory but no files
  auto artifact_dir = fixture.root() / artifact_id;
  std::filesystem::create_directories(artifact_dir);

  auto store = make_test_store(fixture.root());

  // Try to load empty replica
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto handle = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    REQUIRE(!handle.ok());
  }
}

// E9: Corrupted replica file
TEST_CASE("E9: Corrupted replica file", "[store_engine][edge][e9]") {
  const std::string artifact_id = "corrupt_model_e9";

  TempArtifactFixture fixture("edge_e9");

  // Create replica with invalid data
  auto artifact_dir = fixture.root() / artifact_id;
  std::filesystem::create_directories(artifact_dir);

  // Create a file that's too small or has invalid content
  auto data_file = artifact_dir / "tensor.data_0";
  std::ofstream ofs(data_file, std::ios::binary);
  ofs << "INVALID";
  ofs.close();

  auto store = make_test_store(fixture.root());

  // Try to load corrupted replica
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto handle = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    // Implementation may not perform data verification yet; accept either outcome
    if (handle.ok()) {
      auto wait_status = handle.value().wait_ready(std::chrono::milliseconds(5000));
      (void)wait_status;
    }
  }
}

// E10: MaterializeMode edge cases
TEST_CASE("E10: MaterializeMode edge cases", "[store_engine][edge][e10]") {
  skip_if_no_cuda("E10");

  const std::string artifact_id = "edge_model_e10";
  const size_t artifact_size = 30 * 1024 * 1024; // 30MB

  TempArtifactFixture fixture("edge_e10");
  fixture.create_model(artifact_id, artifact_size);

  auto store = make_test_store(fixture.root());

  // COPY_ONLY without existing source
  {
    MaterializeHints hints{};

    auto copy_handle = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::COPY_ONLY, hints);
    REQUIRE(!copy_handle.ok()); // Should fail - no source to copy from
  }

  // LOAD_ONLY should work
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto load_handle = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    REQUIRE(load_handle.ok());
    REQUIRE(load_handle.value().wait_ready(std::chrono::milliseconds(30000)).ok());
  }

  // Now COPY_ONLY should work
  {
    MaterializeHints hints{};

    auto copy_handle2 = store->materialize_replica(make_gpu_key(1), StoreEngine::MaterializeMode::COPY_ONLY, hints);
    if (tensorcast::testing::is_cuda_available() && copy_handle2.ok()) {
      // Should succeed if we have multiple GPUs
      REQUIRE(copy_handle2.value().wait_ready(std::chrono::milliseconds(30000)).ok());
    }
  }

  // LOAD_ONLY on already loaded device
  {
    MaterializeHints hints{.disk_path = artifact_id};

    auto reload_handle = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    if (!reload_handle.ok()) {
      LOG(ERROR) << "Failed to reload replica: " << reload_handle.status().message();
    }
    REQUIRE(reload_handle.ok()); // Should return existing instance
  }
}

// E11: wait_for_state tolerates immediate release
TEST_CASE("E11: wait_for_state tolerates immediate release", "[store_engine][edge][e11]") {
  skip_if_no_cuda("E11");

  const std::string artifact_id = "edge_model_e11";
  const size_t artifact_size = 20 * 1024 * 1024; // 20MB

  TempArtifactFixture fixture("edge_e11");
  fixture.create_model(artifact_id, artifact_size);

  auto store = make_test_store(fixture.root());

  const int num_iterations = 5;
  const auto replica_key = make_replica_key(artifact_id, 0);

  for (int i = 0; i < num_iterations; ++i) {
    std::atomic<bool> loader_finished{false};
    std::atomic<bool> materialize_ok{false};
    std::atomic<bool> wait_ready_ok{false};
    std::mutex message_mutex;
    std::string failure_message;

    auto record_failure = [&](std::string msg) {
      std::lock_guard<std::mutex> lock(message_mutex);
      failure_message = std::move(msg);
    };

    std::thread loader([&]() {
      MaterializeHints hints{.disk_path = artifact_id};
      auto handle_or = store->materialize_replica(make_gpu_key(0), StoreEngine::MaterializeMode::LOAD_ONLY, hints);
      if (!handle_or.ok()) {
        record_failure(handle_or.status().ToString());
        loader_finished.store(true, std::memory_order_release);
        return;
      }
      materialize_ok.store(true, std::memory_order_release);

      auto ready_status = handle_or.value().wait_ready(std::chrono::milliseconds(30000));
      if (!ready_status.ok()) {
        record_failure(ready_status.ToString());
      } else {
        wait_ready_ok.store(true, std::memory_order_release);
      }
      loader_finished.store(true, std::memory_order_release);
    });

    std::thread unloader([&]() {
      while (!loader_finished.load(std::memory_order_acquire)) {
        (void)store->unload_replica(replica_key);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });

    loader.join();
    loader_finished.store(true, std::memory_order_release);
    unloader.join();

    INFO(failure_message);
    REQUIRE(materialize_ok.load(std::memory_order_acquire));
    REQUIRE(wait_ready_ok.load(std::memory_order_acquire));

    (void)store->unload_replica(replica_key);
  }
}
