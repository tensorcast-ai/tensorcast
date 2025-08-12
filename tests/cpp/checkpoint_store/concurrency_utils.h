// Copyright (c) 2025, StepCast Team. All rights reserved.

// Shared utilities for CheckpointStore concurrency tests.
#pragma once

#include <latch>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/common/cuda_api.h"
#include "core/store/checkpoint_store.h"
#include "tests/cpp/common.h"

namespace stepcast::tests::checkpoint_store {

// Test setup utilities
inline void skip_if_no_cuda(const std::string& test_name) {
  if (!is_cuda_available()) {
    WARN("CUDA not available - skipping " + test_name);
    SKIP();
  }
}

inline void skip_if_insufficient_gpus(int required_gpus, const std::string& test_name) {
  if (!is_cuda_available()) {
    WARN("CUDA not available - skipping " + test_name);
    SKIP();
  }
  int device_count = 0;
  {
    auto st = stepcast::cuda::get_device_count(&device_count);
    ABSL_CHECK(st.ok()) << "Failed to get GPU count: " << st.message();
  }
  if (device_count < required_gpus) {
    WARN(
        "Insufficient GPUs (" + std::to_string(device_count) + " < " + std::to_string(required_gpus) + ") - skipping " +
        test_name);
    SKIP();
  }
}

// Random utilities
inline size_t random_model_size(size_t min_mb = 1, size_t max_mb = 100) {
  static thread_local std::mt19937 gen(
      std::random_device{}() ^ std::hash<std::thread::id>{}(std::this_thread::get_id()));
  std::uniform_int_distribution<size_t> dist(min_mb, max_mb);
  return dist(gen) * 1024 * 1024; // Convert MB to bytes
}

inline int random_device_ordinal(int max_device = 3) {
  static thread_local std::mt19937 gen(
      std::random_device{}() ^ std::hash<std::thread::id>{}(std::this_thread::get_id()));
  std::uniform_int_distribution<int> dist(0, max_device);
  return dist(gen);
}

inline std::string generate_model_name(const std::string& prefix, int index) {
  return prefix + "_" + std::to_string(index);
}

// Thread synchronization helpers
class ThreadBarrier {
 public:
  explicit ThreadBarrier(std::ptrdiff_t count) : latch_(count) {}

  void arrive_and_wait() {
    latch_.arrive_and_wait();
  }

 private:
  std::latch latch_;
};

// Verification helpers
struct LoadResult {
  bool success;
  std::string model_id;
  store::DeviceKey device;
  std::chrono::milliseconds load_time;
  std::string error_message;
};

class ConcurrentLoadTracker {
 public:
  void record_load(const LoadResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    results_.push_back(result);
  }

  std::vector<LoadResult> get_results() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return results_;
  }

  size_t successful_loads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::count_if(results_.begin(), results_.end(), [](const LoadResult& r) { return r.success; });
  }

  size_t failed_loads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::count_if(results_.begin(), results_.end(), [](const LoadResult& r) { return !r.success; });
  }

 private:
  mutable std::mutex mutex_;
  std::vector<LoadResult> results_;
};

// Test fixture for creating temporary model files
class TempModelFixture {
 public:
  explicit TempModelFixture(const std::string& test_name)
      : root_path_(std::filesystem::temp_directory_path() / ("checkpoint_store_" + test_name)) {
    std::filesystem::create_directories(root_path_);
  }

  ~TempModelFixture() {
    std::error_code ec;
    std::filesystem::remove_all(root_path_, ec);
  }

  std::filesystem::path create_model(const std::string& model_id, size_t size_bytes) {
    auto model_dir = root_path_ / model_id;
    std::filesystem::create_directories(model_dir);
    auto data_file = model_dir / "tensor.data_0";
    if (!create_dummy_file(data_file, size_bytes)) {
      throw std::runtime_error("Failed to create dummy model file");
    }
    return model_dir;
  }

  std::filesystem::path root() const {
    return root_path_;
  }

 private:
  std::filesystem::path root_path_;
};

// Helper to create a CheckpointStore with sensible test defaults
inline std::unique_ptr<store::CheckpointStore> make_test_store(
    const std::filesystem::path& storage_root,
    size_t pool_size_mb = 512,
    size_t chunk_size_kb = 64,
    int io_threads = 4) {
  store::CheckpointStoreOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = pool_size_mb * 1024 * 1024;
  opts.chunk_size = chunk_size_kb * 1024;
  opts.num_thread = io_threads;
  opts.pinned_memory_timeout = std::chrono::milliseconds(30000);
  // P2P is configured via p2p_port (already has default)
  return std::make_unique<store::CheckpointStore>(opts);
}

// Device key helpers
inline store::DeviceKey make_gpu_key(int ordinal, const std::string& uuid = "") {
  return store::DeviceKey{DeviceType::GPU, ordinal, uuid};
}

inline store::InstanceKey make_instance_key(const std::string& model_id, int gpu_ordinal) {
  return store::InstanceKey{model_id, make_gpu_key(gpu_ordinal), 0};
}

} // namespace stepcast::tests::checkpoint_store