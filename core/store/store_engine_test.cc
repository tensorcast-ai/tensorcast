// Copyright (c) 2025, StepCast Team. All rights reserved.

// Rewritten tests for StoreEngine using the new multi-device `prepare()` API.

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"

namespace fs = std::filesystem;
using stepcast::DeviceType;
using stepcast::store::DeviceKey;
using stepcast::store::InstanceKey;
using stepcast::store::StoreEngine;
using stepcast::store::StoreEngineOptions;

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────
static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{DeviceType::GPU, ordinal, /*uuid=*/""};
}
static StoreEngine make_store(
    const fs::path& storage_root,
    size_t pool_size_bytes = 32ULL * 1024 * 1024,
    size_t chunk_size_bytes = 64ULL * 1024,
    int io_threads = 2) {
  StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = pool_size_bytes;
  opts.chunk_size = chunk_size_bytes;
  opts.num_thread = io_threads;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return StoreEngine(opts);
}
static absl::Status wait_ready(stepcast::store::ModelHandle& handle, absl::Duration timeout = absl::Seconds(60)) {
  return handle.wait_ready(std::chrono::milliseconds(absl::ToInt64Milliseconds(timeout)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 1: Basic CPU → GPU workflow using prepare()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("StoreEngine prepare() GPU workflow", "[store_engine][prepare][cpu][gpu]") {
  const std::string model_id = "dummy_model";
  const size_t model_size = 1 * 1024 * 1024; // 1 MiB

  // Create temporary directory and dummy model file
  fs::path temp_root = fs::temp_directory_path() / "store_engine_prepare_test";
  fs::create_directories(temp_root);
  fs::path model_dir = temp_root / model_id;
  fs::create_directories(model_dir);
  REQUIRE(stepcast::tests::create_dummy_file(model_dir / "tensor.data_0", model_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(stepcast::tests::write_rfc0007_descriptor_for_standard_model_dir(model_dir).ok());

  StoreEngine store = make_store(temp_root);

  // Load to GPU (Now only GPU is supported)
  REQUIRE(stepcast::tests::is_cuda_available());
  stepcast::store::LoadingHints hints;
  hints.disk_path = model_id;
  auto gpu_handle_or = store.prepare(make_gpu_key(0), stepcast::store::StoreEngine::PrepareMode::LOAD_ONLY, hints);
  REQUIRE(gpu_handle_or.ok());
  auto gpu_handle = std::move(gpu_handle_or).value();
  REQUIRE(wait_ready(gpu_handle).ok());
  REQUIRE(gpu_handle.gpu_base_ptr != nullptr);

  DeviceKey gpu0{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  InstanceKey key{.model_id = model_id, .device = gpu0, .replica = 0};
  REQUIRE(store.wait_instance_ready(key) == 0);

  REQUIRE(store.unload_instance(key) == 0);
  REQUIRE(store.clear_mem() == 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 2: Query helpers after prepare()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("StoreEngine helper queries after prepare()", "[store_engine][prepare][status]") {
  if (!stepcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping status tests with prepare().");
    return;
  }

  const std::string model_id = "status_model";
  const size_t model_size = 2 * 1024 * 1024; // 2 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_prepare_status_test";
  fs::create_directories(temp_root);
  fs::path model_dir = temp_root / model_id;
  fs::create_directories(model_dir);
  REQUIRE(stepcast::tests::create_dummy_file(model_dir / "tensor.data_0", model_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(stepcast::tests::write_rfc0007_descriptor_for_standard_model_dir(model_dir).ok());

  StoreEngine store = make_store(temp_root);

  // Load to GPU.
  {
    stepcast::store::LoadingHints hints2;
    hints2.disk_path = model_id;
    auto gpu_handle_or = store.prepare(make_gpu_key(0), stepcast::store::StoreEngine::PrepareMode::LOAD_ONLY, hints2);
    REQUIRE(gpu_handle_or.ok());
    auto gpu_handle = std::move(gpu_handle_or).value();
    REQUIRE(wait_ready(gpu_handle).ok());
  }

  // Verify get_loaded_devices()
  auto devices = store.get_loaded_devices(model_id);
  REQUIRE(!devices.empty());

  // Verify list_device_models() on GPU 0.
  auto gpu_models = store.list_device_models(make_gpu_key(0));
  REQUIRE(!gpu_models.empty());

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}