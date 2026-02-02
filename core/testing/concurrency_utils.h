// Copyright (c) 2025-2026, TensorCast Team.

// Shared utilities for StoreEngine concurrency tests.
#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <latch>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"
#include "core/cuda/cuda_api.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/store_engine.h"
#include "core/testing/common.h"

namespace tensorcast::testing {

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
    auto st = tensorcast::cuda::get_device_count(&device_count);
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

inline std::string generate_artifact_id(const std::string& prefix, int index) {
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
  std::string artifact_id;
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

// Test fixture for creating temporary replica files
class TempArtifactFixture {
 public:
  explicit TempArtifactFixture(const std::string& test_name) : root_path_(make_unique_root(test_name)) {
    std::filesystem::create_directories(root_path_);
  }

  ~TempArtifactFixture() {
    std::error_code ec;
    std::filesystem::remove_all(root_path_, ec);
  }

  std::filesystem::path create_model(const std::string& artifact_id, size_t size_bytes) {
    auto artifact_dir = root_path_ / artifact_id;
    std::filesystem::create_directories(artifact_dir);
    auto data_file = artifact_dir / "tensor.data_0";
    if (!create_dummy_file(data_file, size_bytes)) {
      throw std::runtime_error("Failed to create dummy replica file");
    }
    // Ensure RFC-0007 metadata so DiskLoader can initialize
    auto st = write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir);
    if (!st.ok()) {
      throw std::runtime_error(std::string("Failed to write descriptor/index: ") + std::string(st.message()));
    }
    return artifact_dir;
  }

  std::filesystem::path root() const {
    return root_path_;
  }

 private:
  static std::filesystem::path make_unique_root(const std::string& test_name) {
    const auto base = std::filesystem::temp_directory_path();
    // Isolate fixture directories per test run to avoid cross-test interference when running in parallel.
    static std::atomic<uint64_t> fixture_counter{0};
    const auto now_ticks = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto thread_hash = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    std::ostringstream oss;
    oss << std::hex << now_ticks << "_" << fixture_counter.fetch_add(1) << "_" << thread_hash;
    return base / ("store_engine_" + test_name + "_" + oss.str());
  }

  std::filesystem::path root_path_;
};

// Helper to create a StoreEngine with sensible test defaults
inline std::unique_ptr<store::StoreEngine> make_test_store(
    const std::filesystem::path& storage_root,
    size_t pool_size_mb = 512,
    size_t chunk_size_kb = 64,
    int io_threads = 4) {
  store::StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = pool_size_mb * 1024 * 1024;
  opts.tx_slice_bytes = chunk_size_kb * 1024;
  opts.num_thread = io_threads;
  opts.pinned_memory_timeout = std::chrono::milliseconds(30000);
  // P2P is configured via p2p_port (already has default)
  auto store = std::make_unique<store::StoreEngine>(opts);
  auto comm_mgr = store->get_shared_comm_manager();
  if (!comm_mgr->is_enabled()) {
    absl::Status init_status = comm_mgr->initialize("127.0.0.1", /*listen_port=*/0, /*enable_rdma=*/false);
    ABSL_CHECK(init_status.ok()) << "Failed to initialize CommunicationManager for test store: " << init_status;
  }
  return store;
}

// Device key helpers
inline store::DeviceKey make_gpu_key(int ordinal, const std::string& uuid = "") {
  if (!uuid.empty()) {
    return store::DeviceKey{DeviceType::GPU, ordinal, uuid};
  }
  return store::DeviceRegistry::instance().gpu_key(ordinal);
}

inline store::loading::ReplicaKey make_replica_key(const std::string& artifact_id, int gpu_ordinal) {
  return store::loading::ReplicaKey{
      .artifact_id = artifact_id, .view_id = std::nullopt, .device = make_gpu_key(gpu_ordinal), .replica = 0};
}

} // namespace tensorcast::testing
