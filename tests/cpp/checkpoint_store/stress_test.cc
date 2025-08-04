// Copyright (c) 2025, StepCast Team. All rights reserved.

// CheckpointStore stress tests (C-series)
// Long-running stress tests with randomized operations.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include "concurrency_utils.h"

using namespace stepcast::tests::checkpoint_store;
using namespace stepcast::store;
using stepcast::DeviceType;

// Stress test configuration
struct StressConfig {
  int num_models = 30;
  int num_workers = 8;
  std::chrono::seconds duration{5};
  size_t min_model_size_mb = 5;
  size_t max_model_size_mb = 50;
  size_t pool_size_mb = 2048; // 2GB
  int max_gpu_ordinal = 3;
};

// Operation statistics
struct StressStats {
  std::atomic<uint64_t> prepare_attempts{0};
  std::atomic<uint64_t> prepare_success{0};
  std::atomic<uint64_t> unload_attempts{0};
  std::atomic<uint64_t> unload_success{0};
  std::atomic<uint64_t> clear_attempts{0};
  std::atomic<uint64_t> clear_success{0};
  std::atomic<uint64_t> query_attempts{0};
  std::atomic<uint64_t> query_success{0};
  std::atomic<uint64_t> evictions{0};

  void print_summary() const {
    INFO("Stress Test Statistics:");
    INFO(
        "  Prepare: " << prepare_success << "/" << prepare_attempts << " ("
                      << (prepare_attempts > 0 ? 100.0 * prepare_success / prepare_attempts : 0.0) << "%)");
    INFO(
        "  Unload: " << unload_success << "/" << unload_attempts << " ("
                     << (unload_attempts > 0 ? 100.0 * unload_success / unload_attempts : 0.0) << "%)");
    INFO("  Clear: " << clear_success << "/" << clear_attempts);
    INFO("  Query: " << query_success << "/" << query_attempts);
    INFO("  Evictions: " << evictions);
  }
};

// Worker that performs random operations
class StressWorker {
 public:
  StressWorker(
      int id,
      CheckpointStore* store,
      const StressConfig& config,
      const std::vector<std::string>& model_ids,
      StressStats* stats)
      : store_(store), config_(config), model_ids_(model_ids), stats_(stats), rng_(std::random_device{}() ^ id) {}

  void run(std::atomic<bool>& stop_flag) {
    while (!stop_flag.load(std::memory_order_relaxed)) {
      perform_random_operation();
      std::this_thread::sleep_for(std::chrono::milliseconds(uniform_int(1, 10)));
    }
  }

 private:
  void perform_random_operation() {
    enum Operation { PREPARE, UNLOAD, CLEAR, QUERY };

    // Weighted operation selection
    std::discrete_distribution<> op_dist({40, 30, 5, 25}); // prepare: 40%, unload: 30%, clear: 5%, query: 25%
    Operation op = static_cast<Operation>(op_dist(rng_));

    switch (op) {
      case PREPARE:
        perform_prepare();
        break;
      case UNLOAD:
        perform_unload();
        break;
      case CLEAR:
        perform_clear();
        break;
      case QUERY:
        perform_query();
        break;
    }
  }

  void perform_prepare() {
    stats_->prepare_attempts.fetch_add(1);

    const auto& model_id = model_ids_[uniform_int(0, model_ids_.size() - 1)];
    int gpu_ordinal = uniform_int(0, config_.max_gpu_ordinal);

    auto handle_or = store_->prepare(model_id, make_gpu_key(gpu_ordinal));
    if (handle_or.ok()) {
      auto handle = std::move(handle_or).value();
      auto wait_status = handle.wait_ready(std::chrono::milliseconds(5000));
      if (wait_status.ok()) {
        stats_->prepare_success.fetch_add(1);
      }
    }
  }

  void perform_unload() {
    stats_->unload_attempts.fetch_add(1);

    const auto& model_id = model_ids_[uniform_int(0, model_ids_.size() - 1)];
    int gpu_ordinal = uniform_int(0, config_.max_gpu_ordinal);

    auto instance_key = make_instance_key(model_id, gpu_ordinal);
    int result = store_->unload_instance(instance_key);
    if (result == 0) {
      stats_->unload_success.fetch_add(1);
    }
  }

  void perform_clear() {
    stats_->clear_attempts.fetch_add(1);

    int result = store_->clear_mem();
    if (result == 0) {
      stats_->clear_success.fetch_add(1);
    }
  }

  void perform_query() {
    stats_->query_attempts.fetch_add(1);

    // Random query type
    int query_type = uniform_int(0, 3);

    try {
      switch (query_type) {
        case 0: {
          // Query loaded devices for a model
          const auto& model_id = model_ids_[uniform_int(0, model_ids_.size() - 1)];
          auto devices = store_->get_loaded_devices(model_id);
          stats_->query_success.fetch_add(1);
          break;
        }
        case 1: {
          // List models on a device
          int gpu_ordinal = uniform_int(0, config_.max_gpu_ordinal);
          auto models = store_->list_device_models(make_gpu_key(gpu_ordinal));
          stats_->query_success.fetch_add(1);
          break;
        }
        case 2: {
          // Get all models info
          auto models_info = store_->get_all_models_info();
          stats_->query_success.fetch_add(1);
          break;
        }
        case 3: {
          // Check available memory
          store_->update_memory_pool_metrics();
          stats_->query_success.fetch_add(1);
          break;
        }
      }
    } catch (...) {
      // Query failed
    }
  }

  int uniform_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng_);
  }

  CheckpointStore* store_;
  const StressConfig& config_;
  const std::vector<std::string>& model_ids_;
  StressStats* stats_;
  std::mt19937 rng_;
};

// C1: Basic stress test with mixed operations
TEST_CASE("C1: Basic stress test", "[checkpoint_store][stress][c1]") {
  skip_if_no_cuda("C1");

  StressConfig config;
  config.num_models = 20;
  config.num_workers = 4;
  config.duration = std::chrono::seconds(3);
  config.pool_size_mb = 1024; // 1GB

  TempModelFixture fixture("stress_c1");

  // Create models with random sizes
  std::vector<std::string> model_ids;
  for (int i = 0; i < config.num_models; ++i) {
    auto model_id = generate_model_name("stress_model_c1", i);
    model_ids.push_back(model_id);
    fixture.create_model(model_id, random_model_size(config.min_model_size_mb, config.max_model_size_mb));
  }

  auto store = make_test_store(fixture.root(), config.pool_size_mb);
  StressStats stats;

  // Start workers
  std::vector<std::thread> workers;
  std::atomic<bool> stop_flag{false};

  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < config.num_workers; ++i) {
    workers.emplace_back([&, worker_id = i]() {
      StressWorker worker(worker_id, store.get(), config, model_ids, &stats);
      worker.run(stop_flag);
    });
  }

  // Let it run
  std::this_thread::sleep_for(config.duration);
  stop_flag.store(true);

  // Join workers
  for (auto& t : workers) {
    t.join();
  }

  auto elapsed = std::chrono::steady_clock::now() - start_time;

  // Print statistics
  stats.print_summary();
  INFO("Test duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << "ms");

  // Verify store is in consistent state
  store->update_memory_pool_metrics();
  auto available = store->get_available_memory();
  auto pool_size = store->get_mem_pool_size();

  REQUIRE(available <= pool_size);

  // Should have completed many operations
  REQUIRE(stats.prepare_attempts > 0);
  REQUIRE(stats.prepare_success > 0);
  REQUIRE(stats.query_attempts > 0);
  REQUIRE(stats.query_success > 0);
}

// C2: Heavy concurrent load stress test
TEST_CASE("C2: Heavy concurrent load", "[checkpoint_store][stress][c2]") {
  skip_if_no_cuda("C2");

  StressConfig config;
  config.num_models = 30;
  config.num_workers = 8;
  config.duration = std::chrono::seconds(5);
  config.pool_size_mb = 2048; // 2GB

  TempModelFixture fixture("stress_c2");

  // Create models
  std::vector<std::string> model_ids;
  for (int i = 0; i < config.num_models; ++i) {
    auto model_id = generate_model_name("heavy_model_c2", i);
    model_ids.push_back(model_id);
    fixture.create_model(model_id, random_model_size(10, 100)); // 10-100MB
  }

  auto store = make_test_store(fixture.root(), config.pool_size_mb, 128, 8); // More IO threads
  StressStats stats;

  std::vector<std::thread> workers;
  std::atomic<bool> stop_flag{false};

  // Track registry consistency
  std::atomic<int> consistency_checks{0};
  std::atomic<int> consistency_failures{0};

  // Add a consistency checker thread
  workers.emplace_back([&]() {
    while (!stop_flag.load()) {
      consistency_checks.fetch_add(1);

      try {
        // Check that registry is consistent
        auto all_models = store->get_all_models_info();
        size_t total_size = 0;

        for (const auto& info : all_models) {
          total_size += info.size_bytes;

          // Verify instance exists
          DeviceKey device_key{DeviceType::GPU, info.gpu_device_id, info.gpu_device_uuid};
          InstanceKey instance_key{info.model_id, device_key, 0};

          auto state = store->get_instance_state(instance_key, DeviceType::GPU);
          if (state != MemoryState::LOADED && state != MemoryState::LOADING) {
            consistency_failures.fetch_add(1);
          }
        }

        // Verify memory accounting
        store->update_memory_pool_metrics();
        auto available = store->get_available_memory();
        auto pool_size = store->get_mem_pool_size();

        if (available + total_size > pool_size) {
          consistency_failures.fetch_add(1);
        }
      } catch (...) {
        consistency_failures.fetch_add(1);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  // Start worker threads
  for (int i = 0; i < config.num_workers; ++i) {
    workers.emplace_back([&, worker_id = i]() {
      StressWorker worker(worker_id, store.get(), config, model_ids, &stats);
      worker.run(stop_flag);
    });
  }

  // Run test
  std::this_thread::sleep_for(config.duration);
  stop_flag.store(true);

  // Join all threads
  for (auto& t : workers) {
    t.join();
  }

  // Print results
  stats.print_summary();
  INFO("Consistency checks: " << consistency_checks << " (failures: " << consistency_failures << ")");

  // Verify final state
  REQUIRE(consistency_failures == 0);

  // Clear all memory
  REQUIRE(store->clear_mem() == 0);
  REQUIRE(store->get_available_memory() == store->get_mem_pool_size());
}

// C3: Memory pressure stress test
TEST_CASE("C3: Memory pressure stress", "[checkpoint_store][stress][c3]") {
  skip_if_no_cuda("C3");

  StressConfig config;
  config.num_models = 15;
  config.num_workers = 6;
  config.duration = std::chrono::seconds(4);
  config.pool_size_mb = 512; // Small pool to force evictions
  config.min_model_size_mb = 50;
  config.max_model_size_mb = 150; // Large models relative to pool

  TempModelFixture fixture("stress_c3");

  // Create large models
  std::vector<std::string> model_ids;
  size_t total_model_size = 0;

  for (int i = 0; i < config.num_models; ++i) {
    auto model_id = generate_model_name("pressure_model_c3", i);
    model_ids.push_back(model_id);
    size_t size = random_model_size(config.min_model_size_mb, config.max_model_size_mb);
    fixture.create_model(model_id, size);
    total_model_size += size;
  }

  INFO("Total model size: " << total_model_size / (1024 * 1024) << "MB, Pool size: " << config.pool_size_mb << "MB");

  auto store = make_test_store(fixture.root(), config.pool_size_mb);
  StressStats stats;

  // Track evictions
  std::atomic<uint64_t> last_available{store->get_mem_pool_size()};

  std::vector<std::thread> workers;
  std::atomic<bool> stop_flag{false};

  // Eviction monitor thread
  workers.emplace_back([&]() {
    while (!stop_flag.load()) {
      store->update_memory_pool_metrics();
      uint64_t current_available = store->get_available_memory();

      // Detect when memory increases (likely due to eviction)
      if (current_available > last_available.load() + 10 * 1024 * 1024) { // 10MB threshold
        stats.evictions.fetch_add(1);
      }

      last_available.store(current_available);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  // Start worker threads
  for (int i = 0; i < config.num_workers; ++i) {
    workers.emplace_back([&, worker_id = i]() {
      StressWorker worker(worker_id, store.get(), config, model_ids, &stats);
      worker.run(stop_flag);
    });
  }

  // Run test
  std::this_thread::sleep_for(config.duration);
  stop_flag.store(true);

  // Join threads
  for (auto& t : workers) {
    t.join();
  }

  stats.print_summary();

  // Should have triggered evictions
  REQUIRE(stats.evictions > 0);

  // Verify memory is within bounds
  store->update_memory_pool_metrics();
  REQUIRE(store->get_available_memory() <= store->get_mem_pool_size());
}

// C4: Multi-GPU stress test
TEST_CASE("C4: Multi-GPU stress", "[checkpoint_store][stress][c4][multi_gpu]") {
  skip_if_insufficient_gpus(2, "C4");

  StressConfig config;
  config.num_models = 25;
  config.num_workers = 6;
  config.duration = std::chrono::seconds(4);
  config.pool_size_mb = 1536; // 1.5GB

  // Get actual GPU count
  int gpu_count = 0;
#ifdef __CUDACC__
  cudaGetDeviceCount(&gpu_count);
#endif
  config.max_gpu_ordinal = std::min(gpu_count - 1, 3);

  TempModelFixture fixture("stress_c4");

  // Create models
  std::vector<std::string> model_ids;
  for (int i = 0; i < config.num_models; ++i) {
    auto model_id = generate_model_name("multi_gpu_model_c4", i);
    model_ids.push_back(model_id);
    fixture.create_model(model_id, random_model_size(20, 80));
  }

  auto store = make_test_store(fixture.root(), config.pool_size_mb);
  StressStats stats;

  // Track per-GPU statistics
  std::vector<std::atomic<int>> models_per_gpu(gpu_count);

  std::vector<std::thread> workers;
  std::atomic<bool> stop_flag{false};

  // GPU balance monitor
  workers.emplace_back([&]() {
    while (!stop_flag.load()) {
      for (int gpu = 0; gpu <= config.max_gpu_ordinal; ++gpu) {
        auto models = store->list_device_models(make_gpu_key(gpu));
        models_per_gpu[gpu].store(models.size());
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  });

  // Start worker threads
  for (int i = 0; i < config.num_workers; ++i) {
    workers.emplace_back([&, worker_id = i]() {
      StressWorker worker(worker_id, store.get(), config, model_ids, &stats);
      worker.run(stop_flag);
    });
  }

  // Run test
  std::this_thread::sleep_for(config.duration);
  stop_flag.store(true);

  // Join threads
  for (auto& t : workers) {
    t.join();
  }

  stats.print_summary();

  // Print per-GPU stats
  for (int gpu = 0; gpu <= config.max_gpu_ordinal; ++gpu) {
    INFO("GPU " << gpu << " final model count: " << models_per_gpu[gpu].load());
  }

  // All GPUs should have been used
  int gpus_used = 0;
  for (int gpu = 0; gpu <= config.max_gpu_ordinal; ++gpu) {
    if (models_per_gpu[gpu].load() > 0) {
      gpus_used++;
    }
  }

  // With enough operations, all GPUs should have been used at some point
  REQUIRE(gpus_used >= std::min(2, gpu_count));
}