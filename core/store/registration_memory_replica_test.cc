// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "core/store/replica/memory_state.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"

namespace fs = std::filesystem;
using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::replica::MemoryState;

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

static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{DeviceType::GPU, ordinal, /*uuid=*/""};
}

TEST_CASE("Memory Artifact registration: begin/commit lifecycle", "[store_engine][memory-registration]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  const std::string artifact_id = "mem_reg_artifact";
  const uint64_t size_bytes = 1ULL * 1024 * 1024; // 1 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_mem_reg_test";
  fs::create_directories(temp_root);

  // Create a minimal on-disk replica directory so Replica::create(DiskSource) initializes
  fs::path artifact_dir = temp_root / artifact_id;
  fs::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", static_cast<size_t>(size_bytes)));

  StoreEngine store = make_store(temp_root);

  StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  // RFC-0007: tensor_index_key must be a 32-byte sha256 hex string (64 hex chars)
  reg.tensor_index_key = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;
  reg.enable_p2p = false; // no comm manager in this test

  auto begin_or = store.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  const auto& begin = begin_or.value();
  REQUIRE_FALSE(begin.registration_id.empty());
  REQUIRE(begin.device_id == 0);
  REQUIRE(begin.size_bytes == size_bytes);
  REQUIRE(begin.cuda_ipc_handle_bytes.size() == sizeof(cudaIpcMemHandle_t));

  // Commit should finalize and keep memory owned by the daemon (store)
  auto commit_or = store.commit_registered_artifact(begin.registration_id);
  REQUIRE(commit_or.ok());
  auto commit = commit_or.value();
  REQUIRE(commit.registration_id == begin.registration_id);
  // RFC-0007: Commit returns content-addressed artifact_id (mi2:<index_mh>:<data_mh>)
  REQUIRE(commit.artifact_id.rfind("mi2:", 0) == 0);
  REQUIRE_FALSE(commit.index_multihash.empty());
  REQUIRE_FALSE(commit.data_multihash.empty());
  REQUIRE(commit.artifact_id == (std::string("mi2:") + commit.index_multihash + ":" + commit.data_multihash));
  REQUIRE(commit.device_id == 0);
  REQUIRE(commit.size_bytes == size_bytes);

  // Validate the instance exists and has a GPU pointer
  ReplicaKey key{.artifact_id = artifact_id, .device = make_gpu_key(0), .replica = 0};
  auto gpu_ptr_or = store.get_replica_gpu_ptr(key);
  REQUIRE(gpu_ptr_or.ok());
  REQUIRE(gpu_ptr_or.value() != 0);

  // Cleanup
  REQUIRE(store.unload_replica(key) == 0);
  REQUIRE(store.clear_mem() == 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory Artifact registration: abort releases allocation", "[store_engine][memory-registration]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  const std::string artifact_id = "mem_reg_abort";
  const uint64_t size_bytes = 2ULL * 1024 * 1024; // 2 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_mem_abort_test";
  fs::create_directories(temp_root);

  // Create a minimal on-disk replica directory so Replica::create(DiskSource) initializes
  fs::path artifact_dir2 = temp_root / artifact_id;
  fs::create_directories(artifact_dir2);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir2 / "tensor.data_0", static_cast<size_t>(size_bytes)));

  StoreEngine store = make_store(temp_root);

  StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  // Use a valid 64-hex digest placeholder
  reg.tensor_index_key = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;

  auto begin_or = store.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  const auto& begin = begin_or.value();

  // Abort pending registration
  auto st = store.abort_registered_artifact(begin.registration_id);
  REQUIRE(st.ok());

  // Commit after abort should fail with NotFound
  auto commit_or = store.commit_registered_artifact(begin.registration_id);
  REQUIRE_FALSE(commit_or.ok());
  REQUIRE(commit_or.status().code() == absl::StatusCode::kNotFound);

  // Instance should either be absent or have no GPU memory
  ReplicaKey key{.artifact_id = artifact_id, .device = make_gpu_key(0), .replica = 0};
  auto state = store.get_replica_state(key, DeviceType::GPU);
  const bool ok_state = (state == MemoryState::UNINITIALIZED) || (state == MemoryState::UNALLOCATED);
  REQUIRE(ok_state);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory Artifact registration: TTL expiry prevents commit", "[store_engine][memory-registration]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  const std::string artifact_id = "mem_reg_ttl";
  const uint64_t size_bytes = 1ULL * 1024 * 1024; // 1 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_mem_ttl_test";
  fs::create_directories(temp_root);

  // Create a minimal on-disk replica directory so Replica::create(DiskSource) initializes
  fs::path artifact_dir3 = temp_root / artifact_id;
  fs::create_directories(artifact_dir3);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir3 / "tensor.data_0", static_cast<size_t>(size_bytes)));

  StoreEngine store = make_store(temp_root);

  StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  // Use a valid 64-hex digest placeholder
  reg.tensor_index_key = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;
  reg.ttl_ms = 5; // expire quickly

  auto begin_or = store.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  const auto& begin = begin_or.value();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  auto commit_or = store.commit_registered_artifact(begin.registration_id);
  REQUIRE_FALSE(commit_or.ok());
  REQUIRE(commit_or.status().code() == absl::StatusCode::kDeadlineExceeded);

  // After TTL cleanup, instance should not hold GPU memory
  ReplicaKey key{.artifact_id = artifact_id, .device = make_gpu_key(0), .replica = 0};
  auto state = store.get_replica_state(key, DeviceType::GPU);
  const bool ttl_state_ok = (state == MemoryState::UNINITIALIZED) || (state == MemoryState::UNALLOCATED);
  REQUIRE(ttl_state_ok);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory Artifact registration: invalid arguments rejected", "[store_engine][memory-registration]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  fs::path temp_root = fs::temp_directory_path() / "store_engine_mem_invalid_test";
  fs::create_directories(temp_root);
  StoreEngine store = make_store(temp_root);

  // total_size_bytes == 0
  {
    StoreEngine::ArtifactRegistration reg;
    reg.artifact_id = "m1";
    reg.tensor_index_key = "a";
    reg.device_id = 0;
    reg.total_size_bytes = 0;
    auto res = store.begin_register_artifact(reg);
    REQUIRE_FALSE(res.ok());
    REQUIRE(res.status().code() == absl::StatusCode::kInvalidArgument);
  }

  // missing tensor_index_key
  {
    StoreEngine::ArtifactRegistration reg;
    reg.artifact_id = "m2";
    reg.tensor_index_key = "";
    reg.device_id = 0;
    reg.total_size_bytes = 1024;
    auto res = store.begin_register_artifact(reg);
    REQUIRE_FALSE(res.ok());
    REQUIRE(res.status().code() == absl::StatusCode::kInvalidArgument);
  }

  // negative device_id
  {
    StoreEngine::ArtifactRegistration reg;
    reg.artifact_id = "m3";
    reg.tensor_index_key = "a";
    reg.device_id = -1;
    reg.total_size_bytes = 1024;
    auto res = store.begin_register_artifact(reg);
    REQUIRE_FALSE(res.ok());
    REQUIRE(res.status().code() == absl::StatusCode::kInvalidArgument);
  }

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory Artifact registration: double commit returns NotFound", "[store_engine][memory-registration]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  const std::string artifact_id = "mem_reg_double_commit";
  const uint64_t size_bytes = 1ULL * 1024 * 1024; // 1 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_mem_double_commit_test";
  fs::create_directories(temp_root);

  // Create a minimal on-disk replica directory so Replica::create(DiskSource) initializes
  fs::path artifact_dir4 = temp_root / artifact_id;
  fs::create_directories(artifact_dir4);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir4 / "tensor.data_0", static_cast<size_t>(size_bytes)));

  StoreEngine store = make_store(temp_root);

  StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  // Use a valid 64-hex digest placeholder
  reg.tensor_index_key = "0011001100110011001100110011001100110011001100110011001100110011";
  reg.device_id = 0;
  reg.total_size_bytes = size_bytes;

  auto begin_or = store.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  const auto& begin = begin_or.value();

  auto commit1 = store.commit_registered_artifact(begin.registration_id);
  REQUIRE(commit1.ok());

  auto commit2 = store.commit_registered_artifact(begin.registration_id);
  REQUIRE_FALSE(commit2.ok());
  REQUIRE(commit2.status().code() == absl::StatusCode::kNotFound);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("Memory Artifact registration: commit unknown id returns NotFound", "[store_engine][memory-registration]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping memory registration tests.");
    return;
  }

  fs::path temp_root = fs::temp_directory_path() / "store_engine_mem_unknown_commit_test";
  fs::create_directories(temp_root);
  StoreEngine store = make_store(temp_root);

  auto commit_or = store.commit_registered_artifact("non-existent-id");
  REQUIRE_FALSE(commit_or.ok());
  REQUIRE(commit_or.status().code() == absl::StatusCode::kNotFound);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}
