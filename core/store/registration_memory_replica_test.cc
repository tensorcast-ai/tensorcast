// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "core/store/checkpoint_store.h"
#include "core/store/checkpoint_store_options.h"
#include "core/store/model/memory_state.h"
#include "core/testing/common.h"

namespace fs = std::filesystem;
using stepcast::DeviceType;
using stepcast::store::CheckpointStore;
using stepcast::store::CheckpointStoreOptions;
using stepcast::store::DeviceKey;
using stepcast::store::InstanceKey;
using stepcast::store::MemoryState;

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

static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{DeviceType::GPU, ordinal, /*uuid=*/""};
}

TEST_CASE("Memory TensorDict registration: begin/commit lifecycle", "[checkpoint_store][memory-registration]") {
  if (!stepcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  const std::string model_id = "mem_reg_model";
  const uint64_t size_bytes = 1ULL * 1024 * 1024; // 1 MiB

  fs::path temp_root = fs::temp_directory_path() / "checkpoint_store_mem_reg_test";
  fs::create_directories(temp_root);

  // Create a minimal on-disk model directory so Model::create(DiskSource) initializes
  fs::path model_dir = temp_root / model_id;
  fs::create_directories(model_dir);
  REQUIRE(stepcast::tests::create_dummy_file(model_dir / "tensor.data_0", static_cast<size_t>(size_bytes)));

  CheckpointStore store = make_store(temp_root);

  CheckpointStore::TensorDictRegistration reg;
  reg.model_id = model_id;
  reg.tensor_index_key = "0123456789abcdef"; // any non-empty hex string
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;
  reg.enable_p2p = false; // no comm manager in this test

  auto begin_or = store.begin_register_tensor_dict(reg);
  REQUIRE(begin_or.ok());
  const auto& begin = begin_or.value();
  REQUIRE_FALSE(begin.registration_id.empty());
  REQUIRE(begin.device_id == 0);
  REQUIRE(begin.size_bytes == size_bytes);
  REQUIRE(begin.cuda_ipc_handle_bytes.size() == sizeof(cudaIpcMemHandle_t));

  // Commit should finalize and keep memory owned by the daemon (store)
  auto commit_or = store.commit_registered_tensor_dict(begin.registration_id);
  REQUIRE(commit_or.ok());
  auto commit = commit_or.value();
  REQUIRE(commit.registration_id == begin.registration_id);
  REQUIRE(commit.model_id == model_id);
  REQUIRE(commit.device_id == 0);
  REQUIRE(commit.size_bytes == size_bytes);

  // Validate the instance exists and has a GPU pointer
  InstanceKey key{.model_id = model_id, .device = make_gpu_key(0), .replica = 0};
  auto gpu_ptr_or = store.get_instance_gpu_ptr(key);
  REQUIRE(gpu_ptr_or.ok());
  REQUIRE(gpu_ptr_or.value() != 0);

  // Cleanup
  REQUIRE(store.unload_instance(key) == 0);
  REQUIRE(store.clear_mem() == 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory TensorDict registration: abort releases allocation", "[checkpoint_store][memory-registration]") {
  if (!stepcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  const std::string model_id = "mem_reg_abort";
  const uint64_t size_bytes = 2ULL * 1024 * 1024; // 2 MiB

  fs::path temp_root = fs::temp_directory_path() / "checkpoint_store_mem_abort_test";
  fs::create_directories(temp_root);

  // Create a minimal on-disk model directory so Model::create(DiskSource) initializes
  fs::path model_dir = temp_root / model_id;
  fs::create_directories(model_dir);
  REQUIRE(stepcast::tests::create_dummy_file(model_dir / "tensor.data_0", static_cast<size_t>(size_bytes)));

  CheckpointStore store = make_store(temp_root);

  CheckpointStore::TensorDictRegistration reg;
  reg.model_id = model_id;
  reg.tensor_index_key = "deadbeef";
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;

  auto begin_or = store.begin_register_tensor_dict(reg);
  REQUIRE(begin_or.ok());
  const auto& begin = begin_or.value();

  // Abort pending registration
  auto st = store.abort_registered_tensor_dict(begin.registration_id);
  REQUIRE(st.ok());

  // Commit after abort should fail with NotFound
  auto commit_or = store.commit_registered_tensor_dict(begin.registration_id);
  REQUIRE_FALSE(commit_or.ok());
  REQUIRE(commit_or.status().code() == absl::StatusCode::kNotFound);

  // Instance should either be absent or have no GPU memory
  InstanceKey key{.model_id = model_id, .device = make_gpu_key(0), .replica = 0};
  auto state = store.get_instance_state(key, DeviceType::GPU);
  const bool ok_state = (state == MemoryState::UNINITIALIZED) || (state == MemoryState::UNALLOCATED);
  REQUIRE(ok_state);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory TensorDict registration: TTL expiry prevents commit", "[checkpoint_store][memory-registration]") {
  if (!stepcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  const std::string model_id = "mem_reg_ttl";
  const uint64_t size_bytes = 1ULL * 1024 * 1024; // 1 MiB

  fs::path temp_root = fs::temp_directory_path() / "checkpoint_store_mem_ttl_test";
  fs::create_directories(temp_root);

  // Create a minimal on-disk model directory so Model::create(DiskSource) initializes
  fs::path model_dir = temp_root / model_id;
  fs::create_directories(model_dir);
  REQUIRE(stepcast::tests::create_dummy_file(model_dir / "tensor.data_0", static_cast<size_t>(size_bytes)));

  CheckpointStore store = make_store(temp_root);

  CheckpointStore::TensorDictRegistration reg;
  reg.model_id = model_id;
  reg.tensor_index_key = "0123";
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;
  reg.ttl_ms = 5; // expire quickly

  auto begin_or = store.begin_register_tensor_dict(reg);
  REQUIRE(begin_or.ok());
  const auto& begin = begin_or.value();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  auto commit_or = store.commit_registered_tensor_dict(begin.registration_id);
  REQUIRE_FALSE(commit_or.ok());
  REQUIRE(commit_or.status().code() == absl::StatusCode::kDeadlineExceeded);

  // After TTL cleanup, instance should not hold GPU memory
  InstanceKey key{.model_id = model_id, .device = make_gpu_key(0), .replica = 0};
  auto state = store.get_instance_state(key, DeviceType::GPU);
  const bool ttl_state_ok = (state == MemoryState::UNINITIALIZED) || (state == MemoryState::UNALLOCATED);
  REQUIRE(ttl_state_ok);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory TensorDict registration: invalid arguments rejected", "[checkpoint_store][memory-registration]") {
  if (!stepcast::tests::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  fs::path temp_root = fs::temp_directory_path() / "checkpoint_store_mem_invalid_test";
  fs::create_directories(temp_root);
  CheckpointStore store = make_store(temp_root);

  // total_size_bytes == 0
  {
    CheckpointStore::TensorDictRegistration reg;
    reg.model_id = "m1";
    reg.tensor_index_key = "a";
    reg.device_id = 0;
    reg.total_size_bytes = 0;
    auto res = store.begin_register_tensor_dict(reg);
    REQUIRE_FALSE(res.ok());
    REQUIRE(res.status().code() == absl::StatusCode::kInvalidArgument);
  }

  // missing tensor_index_key
  {
    CheckpointStore::TensorDictRegistration reg;
    reg.model_id = "m2";
    reg.tensor_index_key = "";
    reg.device_id = 0;
    reg.total_size_bytes = 1024;
    auto res = store.begin_register_tensor_dict(reg);
    REQUIRE_FALSE(res.ok());
    REQUIRE(res.status().code() == absl::StatusCode::kInvalidArgument);
  }

  // negative device_id
  {
    CheckpointStore::TensorDictRegistration reg;
    reg.model_id = "m3";
    reg.tensor_index_key = "a";
    reg.device_id = -1;
    reg.total_size_bytes = 1024;
    auto res = store.begin_register_tensor_dict(reg);
    REQUIRE_FALSE(res.ok());
    REQUIRE(res.status().code() == absl::StatusCode::kInvalidArgument);
  }

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}
