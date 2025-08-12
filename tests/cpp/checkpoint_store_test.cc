// Copyright (c) 2025, StepCast Team. All rights reserved.

// Rewritten tests for CheckpointStore using the new multi-device `prepare()` API.

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/store/checkpoint_store.h"
#include "core/store/checkpoint_store_options.h"
#include "tests/cpp/common.h"

namespace fs = std::filesystem;
using stepcast::DeviceType;
using stepcast::store::CheckpointStore;
using stepcast::store::CheckpointStoreOptions;
using stepcast::store::DeviceKey;
using stepcast::store::InstanceKey;

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────
static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{DeviceType::GPU, ordinal, /*uuid=*/""};
}
static CheckpointStore make_store(
    const fs::path& storage_root,
    size_t pool_size_bytes = 32ULL * 1024 * 1024,
    size_t chunk_size_bytes = 64ULL * 1024,
    int io_threads = 2) {
  CheckpointStoreOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = pool_size_bytes;
  opts.chunk_size = chunk_size_bytes;
  opts.num_thread = io_threads;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return CheckpointStore(opts);
}
static absl::Status wait_ready(stepcast::store::ModelHandle& handle, absl::Duration timeout = absl::Seconds(60)) {
  return handle.wait_ready(std::chrono::milliseconds(absl::ToInt64Milliseconds(timeout)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 1: Basic CPU → GPU workflow using prepare()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("CheckpointStore prepare() GPU workflow", "[checkpoint_store][prepare][cpu][gpu]") {
  const std::string model_id = "dummy_model";
  const size_t model_size = 1 * 1024 * 1024; // 1 MiB

  // Create temporary directory and dummy model file
  fs::path temp_root = fs::temp_directory_path() / "checkpoint_store_prepare_test";
  fs::create_directories(temp_root);
  fs::path model_dir = temp_root / model_id;
  fs::create_directories(model_dir);
  REQUIRE(stepcast::tests::create_dummy_file(model_dir / "tensor.data_0", model_size));

  CheckpointStore store = make_store(temp_root);

  // Load to GPU (Now only GPU is supported)
  REQUIRE(stepcast::tests::is_cuda_available());
  auto gpu_handle_or = store.prepare(model_id, make_gpu_key(0));
  REQUIRE(gpu_handle_or.ok());
  auto gpu_handle = std::move(gpu_handle_or).value();
  REQUIRE(wait_ready(gpu_handle).ok());
  REQUIRE(gpu_handle.gpu_base_ptr != nullptr);

  DeviceKey gpu0{DeviceType::GPU, 0, ""};
  InstanceKey key{model_id, gpu0, 0};
  REQUIRE(store.wait_instance_ready(key) == 0);

  REQUIRE(store.unload_instance(key) == 0);
  REQUIRE(store.clear_mem() == 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 2: Query helpers after prepare()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("CheckpointStore helper queries after prepare()", "[checkpoint_store][prepare][status]") {
  if (!stepcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping status tests with prepare().");
    return;
  }

  const std::string model_id = "status_model";
  const size_t model_size = 2 * 1024 * 1024; // 2 MiB

  fs::path temp_root = fs::temp_directory_path() / "checkpoint_store_prepare_status_test";
  fs::create_directories(temp_root);
  fs::path model_dir = temp_root / model_id;
  fs::create_directories(model_dir);
  REQUIRE(stepcast::tests::create_dummy_file(model_dir / "tensor.data_0", model_size));

  CheckpointStore store = make_store(temp_root);

  // Load to GPU.
  {
    auto gpu_handle_or = store.prepare(model_id, make_gpu_key(0));
    REQUIRE(gpu_handle_or.ok());
    auto gpu_handle = std::move(gpu_handle_or).value();
    REQUIRE(wait_ready(gpu_handle).ok());
  }

  // Verify get_loaded_devices()
  auto devices = store.get_loaded_devices(model_id);
  REQUIRE(devices.size() >= 1);

  // Verify list_device_models() on GPU 0.
  auto gpu_models = store.list_device_models(make_gpu_key(0));
  REQUIRE(!gpu_models.empty());

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}