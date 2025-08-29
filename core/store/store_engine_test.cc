// Copyright (c) 2025, TensorCast Team.

// Rewritten tests for StoreEngine using the new multi-device `materialize_replica()` API.

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
using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::ReplicaKey;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;

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
static absl::Status wait_ready(tensorcast::store::ReplicaHandle& handle, absl::Duration timeout = absl::Seconds(60)) {
  return handle.wait_ready(std::chrono::milliseconds(absl::ToInt64Milliseconds(timeout)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 1: Basic CPU → GPU workflow using materialize_replica()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("StoreEngine materialize_replica() GPU workflow", "[store_engine][materialize_replica][cpu][gpu]") {
  const std::string artifact_id = "dummy_artifact";
  const size_t artifact_size = 1 * 1024 * 1024; // 1 MiB

  // Create temporary directory and dummy replica file
  fs::path temp_root = fs::temp_directory_path() / "store_engine_prepare_test";
  fs::create_directories(temp_root);
  fs::path artifact_dir = temp_root / artifact_id;
  fs::create_directories(artifact_dir);
  REQUIRE(tensorcast::tests::create_dummy_file(artifact_dir / "tensor.data_0", artifact_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  StoreEngine store = make_store(temp_root);

  // Load to GPU (Now only GPU is supported)
  REQUIRE(tensorcast::tests::is_cuda_available());
  tensorcast::store::MaterializeHints hints;
  hints.disk_path = artifact_id;
  auto gpu_handle_or =
      store.materialize_replica(make_gpu_key(0), tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  REQUIRE(gpu_handle_or.ok());
  auto gpu_handle = std::move(gpu_handle_or).value();
  REQUIRE(wait_ready(gpu_handle).ok());
  REQUIRE(gpu_handle.gpu_base_ptr != nullptr);

  DeviceKey gpu0{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  ReplicaKey key{.artifact_id = artifact_id, .device = gpu0, .replica = 0};
  REQUIRE(store.wait_replica_ready(key) == 0);

  REQUIRE(store.unload_replica(key) == 0);
  REQUIRE(store.clear_mem() == 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 2: Query helpers after materialize_replica()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("StoreEngine helper queries after materialize_replica()", "[store_engine][materialize_replica][status]") {
  if (!tensorcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping status tests with materialize_replica().");
    return;
  }

  const std::string artifact_id = "status_artifact";
  const size_t artifact_size = 2 * 1024 * 1024; // 2 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_prepare_status_test";
  fs::create_directories(temp_root);
  fs::path artifact_dir2 = temp_root / artifact_id;
  fs::create_directories(artifact_dir2);
  REQUIRE(tensorcast::tests::create_dummy_file(artifact_dir2 / "tensor.data_0", artifact_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(tensorcast::tests::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir2).ok());

  StoreEngine store = make_store(temp_root);

  // Load to GPU.
  {
    tensorcast::store::MaterializeHints hints2;
    hints2.disk_path = artifact_id;
    auto gpu_handle_or =
        store.materialize_replica(make_gpu_key(0), tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints2);
    REQUIRE(gpu_handle_or.ok());
    auto gpu_handle = std::move(gpu_handle_or).value();
    REQUIRE(wait_ready(gpu_handle).ok());
  }

  // Verify get_resident_devices()
  auto devices = store.get_resident_devices(artifact_id);
  REQUIRE(!devices.empty());

  // Verify list_device_replicas() on GPU 0.
  auto gpu_models = store.list_device_replicas(make_gpu_key(0));
  REQUIRE(!gpu_models.empty());

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}