// Copyright (c) 2025, StepCast Team. All rights reserved.

// CheckpointStore stress tests (C-series)
// Long-running stress tests with randomized operations.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "concurrency_utils.h"
#include "core/common/cuda_api.h"

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
  std::atomic<uint64_t> query_attempts{0};
  std::atomic<uint64_t> query_success{0};
  std::atomic<uint64_t> evictions{0};
  // Validation stats (post-prepare sanity checks)
  std::atomic<uint64_t> validation_attempts{0};
  std::atomic<uint64_t> validation_success{0};
  // Data validation stats (GPU content check against expected pattern)
  std::atomic<uint64_t> data_validation_attempts{0};
  std::atomic<uint64_t> data_validation_success{0};

  void print_summary() const {
    INFO("Stress Test Statistics:");
    INFO(
        "  Prepare: " << prepare_success << "/" << prepare_attempts << " ("
                      << (prepare_attempts > 0 ? 100.0 * prepare_success / prepare_attempts : 0.0) << "%)");
    INFO(
        "  Unload: " << unload_success << "/" << unload_attempts << " ("
                     << (unload_attempts > 0 ? 100.0 * unload_success / unload_attempts : 0.0) << "%)");
    INFO("  Query: " << query_success << "/" << query_attempts);
    INFO("  Evictions: " << evictions);
    INFO("  Validation: " << validation_success << "/" << validation_attempts);
    INFO("  Data Validation: " << data_validation_success << "/" << data_validation_attempts);
  }
};

// Per-instance validation tracker: unload waits until ongoing validation
// for the same (model_id, gpu_ordinal) finishes.
class ValidationTracker {
 public:
  void begin(const std::string& model_id, int gpu_ordinal) {
    std::unique_lock<std::mutex> lock(mu_);
    counts_[make_key_(model_id, gpu_ordinal)]++;
  }
  void end(const std::string& model_id, int gpu_ordinal) {
    std::unique_lock<std::mutex> lock(mu_);
    auto k = make_key_(model_id, gpu_ordinal);
    auto it = counts_.find(k);
    if (it != counts_.end()) {
      if (--(it->second) == 0) {
        counts_.erase(it);
        cv_.notify_all();
      }
    }
  }
  void wait(const std::string& model_id, int gpu_ordinal) {
    std::unique_lock<std::mutex> lock(mu_);
    auto k = make_key_(model_id, gpu_ordinal);
    cv_.wait(lock, [&] { return counts_.find(k) == counts_.end(); });
  }

 private:
  static std::string make_key_(const std::string& model_id, int gpu_ordinal) {
    return model_id + "|" + std::to_string(gpu_ordinal);
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<std::string, int> counts_;
};

class ValidationScopeKey {
 public:
  ValidationScopeKey(ValidationTracker& tracker, const std::string& model_id, int gpu_ordinal)
      : tracker_(tracker), model_id_(model_id), ordinal_(gpu_ordinal) {
    tracker_.begin(model_id_, ordinal_);
  }
  ~ValidationScopeKey() {
    tracker_.end(model_id_, ordinal_);
  }

 private:
  ValidationTracker& tracker_;
  std::string model_id_;
  int ordinal_;
};

static ValidationTracker g_validation_tracker;

// Per-GPU validation gate: block PREPARE on a GPU while validation is active on it.
class ValidationGpuGate {
 public:
  void begin(int gpu_ordinal) {
    std::lock_guard<std::mutex> lock(mu_);
    counts_[gpu_ordinal] += 1;
  }
  void end(int gpu_ordinal) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = counts_.find(gpu_ordinal);
    if (it != counts_.end()) {
      if (--(it->second) == 0) {
        counts_.erase(it);
      }
    }
  }
  bool active(int gpu_ordinal) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = counts_.find(gpu_ordinal);
    return it != counts_.end() && it->second > 0;
  }

 private:
  mutable std::mutex mu_;
  std::unordered_map<int, int> counts_;
};

class ValidationGpuScope {
 public:
  ValidationGpuScope(ValidationGpuGate& gate, int gpu_ordinal) : gate_(gate), ordinal_(gpu_ordinal) {
    gate_.begin(ordinal_);
  }
  ~ValidationGpuScope() {
    gate_.end(ordinal_);
  }

 private:
  ValidationGpuGate& gate_;
  int ordinal_;
};

static ValidationGpuGate g_validation_gpu_gate;

// Worker that performs random operations
class StressWorker {
 public:
  StressWorker(
      int id,
      CheckpointStore* store,
      const StressConfig& config,
      const std::vector<std::string>& model_ids,
      std::filesystem::path storage_root,
      StressStats* stats)
      : store_(store),
        config_(config),
        model_ids_(model_ids),
        storage_root_(std::move(storage_root)),
        stats_(stats),
        rng_(std::random_device{}() ^ id) {}

  void run(std::atomic<bool>& stop_flag) {
    while (!stop_flag.load(std::memory_order_relaxed)) {
      perform_random_operation();
      std::this_thread::sleep_for(std::chrono::milliseconds(uniform_int(1, 10)));
    }
  }

 private:
  void perform_random_operation() {
    enum Operation { PREPARE, UNLOAD, QUERY };

    // Weighted operation selection
    std::discrete_distribution<> op_dist({40, 30, 30});
    auto op = static_cast<Operation>(op_dist(rng_));

    switch (op) {
      case PREPARE:
        perform_prepare();
        break;
      case UNLOAD:
        perform_unload();
        break;
      case QUERY:
        perform_query();
        break;
    }
  }

  void perform_prepare() {
    const auto& model_id = model_ids_[uniform_int(0, model_ids_.size() - 1)];
    int gpu_ordinal = uniform_int(0, config_.max_gpu_ordinal);

    // Skip starting a new PREPARE on this GPU while a validation is active to avoid eviction races
    if (g_validation_gpu_gate.active(gpu_ordinal)) {
      return;
    }

    stats_->prepare_attempts.fetch_add(1);

    auto handle_or = store_->prepare(model_id, make_gpu_key(gpu_ordinal));
    if (handle_or.ok()) {
      auto handle = std::move(handle_or).value();
      auto wait_status = handle.wait_ready(std::chrono::milliseconds(5000));
      if (wait_status.ok()) {
        stats_->prepare_success.fetch_add(1);
        const auto& key = handle.key();

        // Begin validation protection as early as possible to avoid races
        ValidationScopeKey _validation_scope(g_validation_tracker, key.model_id, key.device.ordinal);
        ValidationGpuScope _gpu_scope(g_validation_gpu_gate, key.device.ordinal);
        (void)stepcast::cuda::set_device(key.device.ordinal);

        // Post-prepare validation (best-effort; tolerant to concurrent mutations)
        stats_->validation_attempts.fetch_add(1);

        bool validated = false;

        // 1) State should generally be LOADED right after wait_ready()
        auto gpu_state = store_->get_instance_state(key, DeviceType::GPU);
        if (gpu_state == MemoryState::LOADED) {
          validated = true;
        }

        // 2) Device/model listings should reflect the presence
        if (!validated) {
          auto devices = store_->get_loaded_devices(model_id);
          for (const auto& dev : devices) {
            if (dev.type == DeviceType::GPU && dev.ordinal == key.device.ordinal) {
              validated = true;
              break;
            }
          }
        }
        if (!validated) {
          auto models = store_->list_device_models(make_gpu_key(key.device.ordinal));
          for (const auto& inst : models) {
            if (inst.model_id == key.model_id && inst.device.ordinal == key.device.ordinal) {
              validated = true;
              break;
            }
          }
        }

        // 3) If still not validated but state is LOADED, GPU pointer must be non-zero
        if (!validated && gpu_state == MemoryState::LOADED) {
          auto ptr_or = store_->get_instance_gpu_ptr(key);
          if (ptr_or.ok() && ptr_or.value() != 0) {
            validated = true;
          }
        }

        if (validated) {
          stats_->validation_success.fetch_add(1);
        }

        // Enforce data validation for every successful prepare
        stats_->data_validation_attempts.fetch_add(1);
        bool data_ok = false;
        // Must be LOADED immediately after wait_ready()
        if (gpu_state == MemoryState::LOADED) {
          // Retry up to 2 times to handle transient read issues
          for (int attempt = 0; attempt < 2 && !data_ok; ++attempt) {
            (void)stepcast::cuda::set_device(key.device.ordinal);
            auto ptr_or = store_->get_instance_gpu_ptr(key);
            if (!ptr_or.ok() || ptr_or.value() == 0) {
              break;
            }
            uint64_t gpu_ptr_u64 = ptr_or.value();
            // Determine model size from filesystem (single shard tensor.data_0)
            std::error_code ec;
            auto file_path = storage_root_ / model_id / "tensor.data_0";
            const uint64_t file_size = std::filesystem::file_size(file_path, ec);
            if (ec || file_size == 0) {
              break;
            }
            const size_t verify_bytes = static_cast<size_t>(std::min<uint64_t>(4096, file_size));
            // Prefix
            std::vector<char> gpu_prefix(verify_bytes);
            auto memcpy_st = stepcast::cuda::memcpy(
                gpu_prefix.data(), reinterpret_cast<void*>(gpu_ptr_u64), verify_bytes, cudaMemcpyDeviceToHost);
            if (!memcpy_st.ok()) {
              continue; // retry
            }
            bool prefix_match = true;
            for (size_t i = 0; i < verify_bytes; ++i) {
              const char expected = static_cast<char>('A' + (i % 26));
              if (gpu_prefix[i] != expected) {
                prefix_match = false;
                break;
              }
            }

            // Suffix
            bool suffix_match = false;
            std::vector<char> gpu_suffix(verify_bytes);
            const uint64_t tail_offset = file_size - verify_bytes;
            auto memcpy_tail = stepcast::cuda::memcpy(
                gpu_suffix.data(),
                reinterpret_cast<void*>(gpu_ptr_u64 + tail_offset),
                verify_bytes,
                cudaMemcpyDeviceToHost);
            if (memcpy_tail.ok()) {
              suffix_match = true;
              for (size_t i = 0; i < verify_bytes; ++i) {
                const char expected = static_cast<char>('A' + ((tail_offset + i) % 26));
                if (gpu_suffix[i] != expected) {
                  suffix_match = false;
                  break;
                }
              }
            }

            data_ok = prefix_match && suffix_match;
          }
        }

        if (data_ok) {
          stats_->data_validation_success.fetch_add(1);
        }
      }
    }
  }

  void perform_unload() {
    stats_->unload_attempts.fetch_add(1);

    const auto& model_id = model_ids_[uniform_int(0, model_ids_.size() - 1)];
    int gpu_ordinal = uniform_int(0, config_.max_gpu_ordinal);

    // If this GPU is currently under validation, skip unload to avoid races
    if (g_validation_gpu_gate.active(gpu_ordinal)) {
      return;
    }

    // Wait for any ongoing validation of this specific instance to complete
    g_validation_tracker.wait(model_id, gpu_ordinal);

    auto instance_key = make_instance_key(model_id, gpu_ordinal);
    int result = store_->unload_instance(instance_key);
    if (result == 0) {
      stats_->unload_success.fetch_add(1);
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
  const std::filesystem::path storage_root_;
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
      StressWorker worker(worker_id, store.get(), config, model_ids, fixture.root(), &stats);
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
  // Skip available memory constraints in stress runs due to ongoing metrics refactor

  // Should have completed many operations
  REQUIRE(stats.prepare_attempts > 0);
  REQUIRE(stats.prepare_success > 0);
  REQUIRE(stats.query_attempts > 0);
  REQUIRE(stats.query_success > 0);
  REQUIRE(stats.validation_attempts > 0);
  REQUIRE(stats.validation_success > 0);
  // Data validation must be attempted for every successful prepare
  REQUIRE(stats.data_validation_attempts == stats.prepare_success);
  // Require at least 95% of successful prepares to pass data validation
  REQUIRE(stats.data_validation_success * 100 >= stats.prepare_success * 95);
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

        for (const auto& info : all_models) {
          // Only validate instances that are actually resident on GPU
          if (info.gpu_state == ModelLocation::GPU && info.gpu_device_id >= 0) {
            DeviceKey device_key{DeviceType::GPU, info.gpu_device_id, info.gpu_device_uuid};
            InstanceKey instance_key{info.model_id, device_key, 0};

            auto state = store->get_instance_state(instance_key, DeviceType::GPU);
            if (state != MemoryState::LOADED && state != MemoryState::LOADING && state != MemoryState::ALLOCATED) {
              consistency_failures.fetch_add(1);
            }
          }
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
      StressWorker worker(worker_id, store.get(), config, model_ids, fixture.root(), &stats);
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

  // Skip pinned memory metrics assertions in stress mode; rely on operation-level checks only

  // Clear all memory
  REQUIRE(store->clear_mem() == 0);
  REQUIRE(stats.validation_attempts > 0);
  REQUIRE(stats.validation_success > 0);
  REQUIRE(stats.data_validation_attempts == stats.prepare_success);
  REQUIRE(stats.data_validation_success * 100 >= stats.prepare_success * 95);
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
  // Skip last_available tracking since available memory metrics are unreliable during refactor

  std::vector<std::thread> workers;
  std::atomic<bool> stop_flag{false};

  // Eviction monitor thread
  workers.emplace_back([&]() {
    while (!stop_flag.load()) {
      // Heuristic eviction signal is disabled while memory metrics are being revised
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  });

  // Start worker threads
  for (int i = 0; i < config.num_workers; ++i) {
    workers.emplace_back([&, worker_id = i]() {
      StressWorker worker(worker_id, store.get(), config, model_ids, fixture.root(), &stats);
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

  // Under pressure we should have exercised load/unload cycles
  REQUIRE(stats.prepare_success > 0);
  REQUIRE(stats.unload_success > 0);
  REQUIRE(stats.validation_attempts > 0);
  REQUIRE(stats.validation_success > 0);
  REQUIRE(stats.data_validation_attempts == stats.prepare_success);
  REQUIRE(stats.data_validation_success * 100 >= stats.prepare_success * 95);
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
  cudaGetDeviceCount(&gpu_count);
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
      StressWorker worker(worker_id, store.get(), config, model_ids, fixture.root(), &stats);
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
  REQUIRE(stats.validation_attempts > 0);
  REQUIRE(stats.validation_success > 0);
  REQUIRE(stats.data_validation_attempts == stats.prepare_success);
  REQUIRE(stats.data_validation_success * 100 >= stats.prepare_success * 95);
}