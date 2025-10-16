// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/store/components/global_store_client.h"
#include "core/store/loader/view_planner.h"
#include "core/store/replica/memory_state.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "nlohmann/json.hpp"
#include "tensorcast/global_store/v1/global_store.pb.h"

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
  opts.tx_slice_bytes = chunk_size_bytes;
  opts.artifact_chunk_bytes = chunk_size_bytes;
  opts.num_thread = io_threads;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return StoreEngine(opts);
}

static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{DeviceType::GPU, ordinal, /*uuid=*/""};
}

class RecordingViewGlobalStoreClient final : public tensorcast::store::components::IGlobalStoreClient {
 public:
  bool connected{true};
  std::vector<tensorcast::store::components::VariantViewUpdate> view_updates;

  absl::Status initialize() override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_worker(
      std::string_view,
      std::string_view,
      uint32_t,
      uint32_t,
      uint64_t,
      uint64_t,
      bool,
      std::string_view) override {
    return absl::UnimplementedError("register_worker not supported in test stub");
  }

  absl::Status send_heartbeat(std::string_view, uint64_t, bool) override {
    return absl::UnimplementedError("send_heartbeat not supported in test stub");
  }

  absl::StatusOr<tensorcast::global_store::v1::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view,
      uint64_t,
      bool,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      int64_t,
      tensorcast::global_store::v1::ConnectionStatus) override {
    return absl::UnimplementedError("send_heartbeat_enhanced not supported in test stub");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not supported in test stub");
  }

  absl::StatusOr<std::string> register_replica(
      std::string_view,
      std::string_view,
      const tensorcast::store::DeviceKey&,
      tensorcast::common::memory::MemoryLocation,
      uint64_t,
      uint32_t) override {
    return std::string("replica-0");
  }

  absl::Status record_variant_residency(std::string_view, std::string_view, uint64_t, std::optional<std::string_view>)
      override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_memory_replica(
      std::string_view,
      std::string_view,
      const tensorcast::store::DeviceKey&,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      const std::vector<uint64_t>&,
      const std::optional<std::string>&,
      std::string_view,
      std::string_view,
      uint32_t,
      const std::optional<std::string>&) override {
    return std::string("memory-replica-0");
  }

  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::UnimplementedError("unregister_replica not supported in test stub");
  }

  absl::Status update_artifact_view_state(const tensorcast::store::components::VariantViewUpdate& update) override {
    view_updates.push_back(update);
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_replica_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_replica_transport not supported in test stub");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_view_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_view_transport not supported in test stub");
  }

  absl::Status complete_replica_transport(std::string_view) override {
    return absl::UnimplementedError("complete_replica_transport not supported in test stub");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view) override {
    return absl::UnimplementedError("get_artifact_replicas not supported in test stub");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ChunkLocationInfo>> query_chunk_locations(
      std::string_view,
      const std::vector<uint32_t>&) override {
    return absl::UnimplementedError("query_chunk_locations not supported in test stub");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> synchronize_worker_state(
      const tensorcast::global_store::v1::WorkerLocalState&,
      bool,
      std::vector<tensorcast::global_store::v1::StateChange>*) override {
    return absl::UnimplementedError("synchronize_worker_state not supported in test stub");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> request_full_state_sync(
      std::string_view,
      uint64_t,
      std::vector<tensorcast::common::v1::ReplicaInfo>*) override {
    return absl::UnimplementedError("request_full_state_sync not supported in test stub");
  }

  bool is_connected() const override {
    return connected;
  }

  absl::Status batch_update_chunk_states(
      std::string_view,
      std::string_view,
      const std::vector<tensorcast::store::components::ChunkStateUpdate>&) override {
    return absl::UnimplementedError("batch_update_chunk_states not supported in test stub");
  }

  absl::StatusOr<tensorcast::store::components::KeyMapping> resolve_key_mapping(std::string_view) override {
    return absl::UnimplementedError("resolve_key_mapping not supported in test stub");
  }

  absl::Status upsert_key_mapping(std::string_view, std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not supported in test stub");
  }

  absl::Status revoke_key_mapping(std::string_view) override {
    return absl::UnimplementedError("revoke_key_mapping not supported in test stub");
  }

  void update_local_endpoint(std::string, std::string, uint32_t, uint32_t) override {}

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view) override {
    return absl::UnimplementedError("get_artifact_index_by_id not supported in test stub");
  }
};

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
  if (!commit_or.ok()) {
    FAIL("commit failed: " + commit_or.status().ToString());
  }
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
  ReplicaKey key{.artifact_id = artifact_id, .view_id = std::nullopt, .device = make_gpu_key(0), .replica = 0};
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
  ReplicaKey key{.artifact_id = artifact_id, .view_id = std::nullopt, .device = make_gpu_key(0), .replica = 0};
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
  ReplicaKey key{.artifact_id = artifact_id, .view_id = std::nullopt, .device = make_gpu_key(0), .replica = 0};
  auto state = store.get_replica_state(key, DeviceType::GPU);
  const bool ttl_state_ok = (state == MemoryState::UNINITIALIZED) || (state == MemoryState::UNALLOCATED);
  REQUIRE(ttl_state_ok);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE(
    "Memory Artifact view registration publishes leaf digests",
    "[store_engine][memory-registration][view][global-store]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping memory view registration test.");
    return;
  }

  const std::string artifact_id = "mem_reg_view";
  constexpr size_t kElements = 1024;
  const uint64_t canonical_size = static_cast<uint64_t>(kElements * sizeof(float));

  fs::path temp_root = fs::temp_directory_path() / "store_engine_mem_view_test";
  fs::create_directories(temp_root);

  StoreEngine store = make_store(
      temp_root,
      /*pool_size_bytes=*/32ULL * 1024 * 1024,
      /*chunk_size_bytes=*/canonical_size);
  auto gs_stub = std::make_shared<RecordingViewGlobalStoreClient>();
  store.set_global_store_client_for_testing(gs_stub);

  nlohmann::json index_entry = nlohmann::json::array();
  index_entry.push_back(0);
  index_entry.push_back(canonical_size);
  index_entry.push_back(nlohmann::json::array({static_cast<int64_t>(kElements)}));
  index_entry.push_back(nlohmann::json::array({1}));
  index_entry.push_back("torch.float32");
  index_entry.push_back(0);

  nlohmann::json index_json = nlohmann::json::object();
  index_json["weights"] = index_entry;
  const std::string index_data = index_json.dump();

  StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  reg.tensor_index_key = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
  reg.tensor_index_data = index_data;
  reg.encoding = "json";
  reg.schema_version = "v3";
  reg.device_id = 0;
  reg.total_size_bytes = canonical_size;
  reg.enable_p2p = false;

  StoreEngine::ViewRegistration view_reg;
  view_reg.view_id = "view-full";
  tensorcast::store::loader::ViewSpec spec;
  tensorcast::store::loader::TensorViewOps ops;
  ops.ops.push_back(
      tensorcast::store::loader::ViewOp::Narrow(
          tensorcast::store::loader::NarrowOp{.dim = 0, .start = 0, .length = kElements}));
  spec.tensors.emplace("weights", ops);
  view_reg.spec = spec;
  view_reg.placement = StoreEngine::ViewPlacement::kServer;
  view_reg.canonical_size_bytes = canonical_size;
  view_reg.allow_partial = false;
  reg.view = view_reg;

  auto begin_or = store.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  const auto begin = begin_or.value();
  REQUIRE(begin.size_bytes == canonical_size);

  std::array<float, kElements> view_payload{};
  for (size_t i = 0; i < kElements; ++i) {
    view_payload[i] = static_cast<float>(i + 1);
  }
  auto ingest_status = store.ingest_view_registration_chunk(
      begin.registration_id,
      /*view_offset=*/0,
      absl::Span<const std::byte>(reinterpret_cast<const std::byte*>(view_payload.data()), canonical_size));
  REQUIRE(ingest_status.ok());

  auto commit_or = store.commit_registered_artifact(begin.registration_id);
  if (!commit_or.ok()) {
    FAIL("view commit failed: " + commit_or.status().ToString());
  }
  const auto commit = commit_or.value();
  REQUIRE(commit.view_id.has_value());
  CHECK(commit.view_id.value() == "view-full");
  CHECK(commit.view_data_multihash.has_value());

  REQUIRE(gs_stub->view_updates.size() == 1);
  const auto& update = gs_stub->view_updates.front();
  CHECK(update.view_id == "view-full");
  CHECK(update.view_data_hash.has_value());
  CHECK(update.canonical_size_bytes == canonical_size);
  CHECK(update.canonical_bytes_covered == canonical_size);
  size_t variant_count = 0;
  size_t canonical_count = 0;
  for (const auto& leaf : update.leaf_writes) {
    if (leaf.space_kind() == tensorcast::global_store::v1::BYTE_SPACE_KIND_VARIANT) {
      ++variant_count;
      CHECK(leaf.space_id() == "view-full");
      CHECK(leaf.leaf_idx() == 0);
      CHECK(leaf.digest().size() == 32);
    } else if (leaf.space_kind() == tensorcast::global_store::v1::BYTE_SPACE_KIND_CANONICAL) {
      ++canonical_count;
      CHECK(leaf.leaf_idx() == 0);
      CHECK(leaf.digest().size() == 32);
    }
  }
  CHECK(variant_count == 1);
  CHECK(canonical_count == 1);

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
