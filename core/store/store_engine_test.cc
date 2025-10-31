// Copyright (c) 2025, TensorCast Team.

// Rewritten tests for StoreEngine using the new multi-device `materialize_replica()` API.

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/artifact_identity.h"
#include "core/store/loader/disk_dir_hash.h"
#include "core/store/loader/source_hash.h"
#include "core/store/loader/view_planner.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::loading::ReplicaKeyHash;

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────
static DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{.type = DeviceType::GPU, .ordinal = ordinal, /*uuid=*/.uuid = ""};
}

static StoreEngine make_store(
    const fs::path& storage_root,
    size_t pool_size_bytes = 32ULL * 1024 * 1024,
    size_t chunk_size_bytes = 64ULL * 1024,
    int io_threads = 2) {
  StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.memory_pool_size = pool_size_bytes;
  opts.tx_slice_bytes = chunk_size_bytes;
  opts.num_thread = io_threads;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return StoreEngine(opts);
}

static absl::StatusOr<std::string> write_descriptor_with_index(
    const fs::path& artifact_dir,
    const nlohmann::json& index_json) {
  if (!fs::exists(artifact_dir) || !fs::is_directory(artifact_dir)) {
    return absl::NotFoundError("artifact_dir does not exist");
  }
  const auto index_path = artifact_dir / "tensor_index.json";
  {
    std::ofstream idx(index_path);
    if (!idx.is_open()) {
      return absl::InternalError("failed to write tensor_index.json");
    }
    idx << index_json.dump();
  }

  auto index_mh_or = tensorcast::common::compute_index_multihash(
      std::optional<std::string>(index_json.dump()), /*descriptor_json=*/"");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  auto data_mh_or = tensorcast::store::loader::compute_data_multihash_from_disk_dir(artifact_dir.string());
  if (!data_mh_or.ok()) {
    return data_mh_or.status();
  }

  uint64_t total_size = 0;
  for (auto it = index_json.begin(); it != index_json.end(); ++it) {
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    const uint64_t offset = arr[0].get<uint64_t>();
    const uint64_t size_bytes = arr[1].get<uint64_t>();
    total_size = std::max(total_size, offset + size_bytes);
  }

  nlohmann::json descriptor;
  descriptor["artifact_id"] = absl::StrCat("mi2:", *index_mh_or, ":", *data_mh_or);
  descriptor["index_multihash"] = *index_mh_or;
  descriptor["data_multihash"] = *data_mh_or;
  descriptor["schema_version"] = "v3";
  descriptor["encoding"] = "json";
  descriptor["total_size"] = total_size;
  nlohmann::json hp;
  hp["chunk_size"] = 4 * 1024 * 1024;
  hp["fanout"] = 2;
  hp["algorithm"] = "sha2-256";
  descriptor["hash_params"] = hp;

  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  {
    std::ofstream of(descriptor_path);
    if (!of.is_open()) {
      return absl::InternalError("failed to write artifact_descriptor.json");
    }
    of << descriptor.dump(2);
  }
  return descriptor["artifact_id"].get<std::string>();
}

static nlohmann::json make_tensor_entry(
    uint64_t offset,
    uint64_t size,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    const std::string& dtype,
    uint64_t storage_offset = 0) {
  nlohmann::json entry = nlohmann::json::array();
  entry.push_back(offset);
  entry.push_back(size);
  nlohmann::json shape_json = nlohmann::json::array();
  for (int64_t dim : shape) {
    shape_json.push_back(dim);
  }
  entry.push_back(shape_json);
  nlohmann::json stride_json = nlohmann::json::array();
  for (int64_t s : stride) {
    stride_json.push_back(s);
  }
  entry.push_back(stride_json);
  entry.push_back(dtype);
  entry.push_back(storage_offset);
  return entry;
}

TEST_CASE("ReplicaKey distinguishes variant byte spaces", "[store_engine][replica_key]") {
  ReplicaKey canonical{
      .artifact_id = "artifact-A",
      .view_id = std::nullopt,
      .device = {.type = DeviceType::GPU, .ordinal = 0, .uuid = ""},
      .replica = 0};
  ReplicaKey variant{
      .artifact_id = "artifact-A",
      .view_id = std::string("view-123"),
      .device = {.type = DeviceType::GPU, .ordinal = 0, .uuid = ""},
      .replica = 0};

  REQUIRE(canonical != variant);
  ReplicaKeyHash hasher;
  REQUIRE(hasher(canonical) != hasher(variant));
}

TEST_CASE("StoreEngine commit reports MI2 identity", "[store_engine][registration]") {
  auto storage = fs::temp_directory_path() / "store-engine-mi2";
  fs::create_directories(storage);
  StoreEngine engine = make_store(storage);

  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = "temp-reg-mi2";
  reg.device_id = 0;
  reg.total_size_bytes = 256 * 1024;
  reg.tensor_index_data = std::string("{}");

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());

  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  const auto& result = commit_or.value();
  REQUIRE(result.id_kind == tensorcast::common::ArtifactIdKind::kMi2);
  REQUIRE(result.artifact_id.rfind("mi2:", 0) == 0);
  REQUIRE(!result.index_multihash.empty());
}

TEST_CASE("StoreEngine commit honours CGID", "[store_engine][registration]") {
  auto storage = fs::temp_directory_path() / "store-engine-cgid";
  fs::create_directories(storage);
  StoreEngine engine = make_store(storage);

  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = "temp-reg-cgid";
  reg.device_id = 0;
  reg.total_size_bytes = 128 * 1024;
  reg.tensor_index_data = std::string("{}");
  reg.client_artifact_id = std::string("cgid:engine-test-1");

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());

  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  const auto& result = commit_or.value();
  REQUIRE(result.id_kind == tensorcast::common::ArtifactIdKind::kCgid);
  REQUIRE(result.artifact_id == "cgid:engine-test-1");
  REQUIRE(result.index_multihash.empty());
  REQUIRE(result.data_multihash.empty());
}

static absl::Status wait_ready(
    tensorcast::store::loading::ReplicaHandle& handle,
    absl::Duration timeout = absl::Seconds(60)) {
  return handle.wait_ready(std::chrono::milliseconds(absl::ToInt64Milliseconds(timeout)));
}

class RecordingGlobalStoreClient final : public tensorcast::store::components::IGlobalStoreClient {
 public:
  bool connected{true};
  std::vector<std::string> view_requests;
  std::vector<std::string> replica_requests;
  std::vector<std::string> registered_replicas;
  std::vector<std::tuple<std::string, std::string, uint64_t>> recorded_variants;
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
      std::string_view artifact_id,
      std::string_view,
      const tensorcast::store::DeviceKey&,
      tensorcast::common::memory::MemoryLocation,
      uint64_t,
      uint32_t) override {
    registered_replicas.emplace_back(artifact_id);
    return std::string("replica-0");
  }

  absl::Status record_variant_residency(
      std::string_view canonical_artifact_id,
      std::string_view view_id,
      uint64_t view_size_bytes,
      std::optional<std::string_view>) override {
    recorded_variants.emplace_back(std::string(canonical_artifact_id), std::string(view_id), view_size_bytes);
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
    return absl::UnimplementedError("register_memory_replica not supported in test stub");
  }

  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::UnimplementedError("unregister_replica not supported in test stub");
  }

  absl::Status unregister_replica_by_worker(
      std::string_view,
      std::string_view,
      std::optional<tensorcast::common::memory::MemoryLocation>,
      std::optional<uint32_t>) override {
    return absl::UnimplementedError("unregister_replica_by_worker not supported in test stub");
  }

  absl::Status update_artifact_view_state(const tensorcast::store::components::VariantViewUpdate& update) override {
    view_requests.emplace_back(update.view_id);
    view_updates.push_back(update);
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_replica_transport(
      std::string_view artifact_id,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey&,
      uint32_t) override {
    replica_requests.emplace_back(artifact_id);
    return absl::UnavailableError("canonical replica unavailable");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_view_transport(
      std::string_view,
      std::string_view view_id,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey&,
      uint32_t) override {
    view_requests.emplace_back(view_id);
    return absl::NotFoundError("variant not registered");
  }

  absl::Status complete_replica_transport(std::string_view) override {
    return absl::OkStatus();
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

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view) override {
    return absl::UnimplementedError("get_artifact_index_by_id not supported in test stub");
  }

  absl::Status upsert_key_mapping(std::string_view, std::string_view, std::string_view, absl::Duration) override {
    return absl::UnimplementedError("upsert_key_mapping not supported in test stub");
  }

  absl::Status revoke_key_mapping(std::string_view) override {
    return absl::UnimplementedError("revoke_key_mapping not supported in test stub");
  }

  void update_local_endpoint(std::string, std::string, uint32_t, uint32_t) override {}
};

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
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", artifact_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  StoreEngine store = make_store(temp_root);

  // Load to GPU (Now only GPU is supported)
  REQUIRE(tensorcast::testing::is_cuda_available());
  tensorcast::store::loading::MaterializeHints hints;
  hints.disk_path = artifact_id;
  auto gpu_handle_or =
      store.materialize_replica(make_gpu_key(0), tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  REQUIRE(gpu_handle_or.ok());
  auto gpu_handle = std::move(gpu_handle_or).value();
  REQUIRE(wait_ready(gpu_handle).ok());
  REQUIRE(gpu_handle.gpu_base_ptr != nullptr);

  DeviceKey gpu0{.type = DeviceType::GPU, .ordinal = 0, .uuid = ""};
  ReplicaKey key{.artifact_id = artifact_id, .view_id = std::nullopt, .device = gpu0, .replica = 0};
  REQUIRE(store.wait_replica_ready(key) == 0);

  REQUIRE(store.unload_replica(key) == 0);
  REQUIRE(store.clear_mem() == 0);

  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE("StoreEngine materializes variant slice with distinct residency", "[store_engine][variant][view]") {
  const std::string artifact_name = "variant_artifact";
  fs::path temp_root = fs::temp_directory_path() / "store_engine_variant_view";
  fs::create_directories(temp_root);
  fs::path artifact_dir = temp_root / artifact_name;
  fs::create_directories(artifact_dir);

  const size_t element_count = 8;
  std::vector<float> payload(element_count);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<float>(i + 1);
  }
  {
    std::ofstream data_file(artifact_dir / "tensor.data_0", std::ios::binary);
    REQUIRE(data_file.is_open());
    data_file.write(reinterpret_cast<const char*>(payload.data()), payload.size() * sizeof(float));
  }

  nlohmann::json index = nlohmann::json::object();
  index["weights"] = make_tensor_entry(
      /*offset=*/0,
      /*size=*/payload.size() * sizeof(float),
      /*shape=*/{static_cast<int64_t>(element_count)},
      /*stride=*/{1},
      /*dtype=*/"torch.float32");

  auto artifact_id_or = write_descriptor_with_index(artifact_dir, index);
  REQUIRE(artifact_id_or.ok());
  const std::string canonical_artifact_id = *artifact_id_or;

  StoreEngine store = make_store(temp_root);
  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};

  tensorcast::store::loading::MaterializeHints canonical_hints;
  canonical_hints.disk_path = artifact_name;
  auto canonical_handle_or =
      store.materialize_replica(cpu_device, StoreEngine::MaterializeMode::LOAD_ONLY, canonical_hints);
  INFO("canonical load status: " << canonical_handle_or.status());
  REQUIRE(canonical_handle_or.ok());
  auto canonical_handle = std::move(canonical_handle_or).value();
  REQUIRE(wait_ready(canonical_handle).ok());

  // Canonical load should produce verification metadata for the canonical ByteSpace.
  const fs::path canonical_verification_path = artifact_dir / "verification.json";
  REQUIRE(fs::exists(canonical_verification_path));

  tensorcast::store::loader::ViewSpec spec;
  tensorcast::store::loader::TensorViewOps ops;
  ops.ops.push_back(
      tensorcast::store::loader::ViewOp::Narrow(
          tensorcast::store::loader::NarrowOp{.dim = 0, .start = 2, .length = 4}));
  spec.tensors.emplace("weights", ops);

  tensorcast::store::loading::VariantIdentity variant_identity;
  variant_identity.canonical_artifact_id = artifact_name;
  variant_identity.view_id = std::string("view-weights-narrow");
  variant_identity.view_spec = spec;
  variant_identity.canonical_index_json = index.dump();

  tensorcast::store::loading::MaterializeHints variant_hints;
  variant_hints.disk_path = artifact_name;
  variant_hints.variant = variant_identity;

  auto variant_handle_or =
      store.materialize_replica(cpu_device, StoreEngine::MaterializeMode::LOAD_ONLY, variant_hints);
  INFO("variant load status: " << variant_handle_or.status());
  REQUIRE(variant_handle_or.ok());
  auto variant_handle = std::move(variant_handle_or).value();
  REQUIRE(wait_ready(variant_handle).ok());

  REQUIRE(variant_handle.replica_key.view_id.has_value());
  REQUIRE(variant_handle.replica_key.view_id.value() == "view-weights-narrow");

  const auto replicas_on_cpu = store.list_device_replicas(cpu_device);
  REQUIRE(replicas_on_cpu.size() == 2);

  tensorcast::store::loading::ReplicaKey canonical_key{
      .artifact_id = artifact_name, .view_id = std::nullopt, .device = cpu_device, .replica = 0};
  tensorcast::store::loading::ReplicaKey variant_key{
      .artifact_id = artifact_name,
      .view_id = std::optional<std::string>("view-weights-narrow"),
      .device = cpu_device,
      .replica = 0};

  auto canonical_size_or = store.get_replica_size(canonical_key);
  INFO("canonical size status: " << canonical_size_or.status());
  REQUIRE(canonical_size_or.ok());
  CHECK(*canonical_size_or == payload.size() * sizeof(float));

  auto variant_size_or = store.get_replica_size(variant_key);
  INFO("variant size status: " << variant_size_or.status());
  REQUIRE(variant_size_or.ok());
  CHECK(*variant_size_or == 4 * sizeof(float));

  const fs::path variant_verification_path = artifact_dir / "verification.view_view-weights-narrow.json";
  REQUIRE(fs::exists(canonical_verification_path));
  REQUIRE(fs::exists(variant_verification_path));

  {
    std::ifstream vf(variant_verification_path);
    REQUIRE(vf.is_open());
    nlohmann::json j;
    vf >> j;
    REQUIRE(j.contains("byte_space_id"));
    CHECK(j["byte_space_id"].get<std::string>() == "view-weights-narrow");
    REQUIRE(j.contains("artifact_size"));
    CHECK(j["artifact_size"].get<uint64_t>() == 4 * sizeof(float));
  }

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec_variant;
  fs::remove_all(temp_root, ec_variant);
}

TEST_CASE(
    "StoreEngine AUTO variant falls back to disk when Global Store lacks view",
    "[store_engine][variant][auto][fallback]") {
  const std::string artifact_name = "auto_variant_artifact";
  fs::path temp_root = fs::temp_directory_path() / "store_engine_variant_auto";
  fs::create_directories(temp_root);
  fs::path artifact_dir = temp_root / artifact_name;
  fs::create_directories(artifact_dir);

  // Build canonical tensor payload
  std::vector<float> payload = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<uint8_t> raw_bytes(payload.size() * sizeof(float));
  std::memcpy(raw_bytes.data(), payload.data(), raw_bytes.size());
  {
    std::ofstream data_file(artifact_dir / "tensor.data_0", std::ios::binary);
    REQUIRE(data_file.is_open());
    data_file.write(reinterpret_cast<const char*>(raw_bytes.data()), static_cast<std::streamsize>(raw_bytes.size()));
  }

  nlohmann::json index = nlohmann::json::object();
  index["weights"] = make_tensor_entry(
      /*offset=*/0,
      /*size=*/raw_bytes.size(),
      /*shape=*/{8},
      /*stride=*/{1},
      /*dtype=*/"torch.float32");
  auto descriptor_or = write_descriptor_with_index(artifact_dir, index);
  REQUIRE(descriptor_or.ok());

  StoreEngine store = make_store(temp_root);
  auto gs_stub = std::make_shared<RecordingGlobalStoreClient>();
  store.set_global_store_client_for_testing(gs_stub);

  tensorcast::store::loader::ViewSpec spec;
  tensorcast::store::loader::TensorViewOps ops;
  ops.ops.push_back(
      tensorcast::store::loader::ViewOp::Narrow(
          tensorcast::store::loader::NarrowOp{.dim = 0, .start = 2, .length = 4}));
  spec.tensors.emplace("weights", ops);

  tensorcast::store::loading::VariantIdentity variant_identity;
  variant_identity.canonical_artifact_id = artifact_name;
  variant_identity.view_id = std::string("view-weights-narrow");
  variant_identity.view_spec = spec;
  variant_identity.canonical_index_json = index.dump();

  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact_name;
  hints.disk_path = artifact_name;
  hints.variant = variant_identity;

  tensorcast::store::DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
  auto handle_or = store.materialize_replica(cpu_device, StoreEngine::MaterializeMode::AUTO, hints);
  INFO("auto variant load status: " << handle_or.status());
  REQUIRE(handle_or.ok());
  auto handle = std::move(handle_or).value();
  REQUIRE(wait_ready(handle).ok());
  REQUIRE(handle.replica_key.view_id.has_value());
  CHECK(handle.replica_key.view_id.value() == "view-weights-narrow");

  REQUIRE(gs_stub->view_requests.size() == 1);
  CHECK(gs_stub->view_requests[0] == "view-weights-narrow");
  REQUIRE(gs_stub->replica_requests.size() == 1);
  CHECK(gs_stub->replica_requests[0] == artifact_name);
  REQUIRE(gs_stub->registered_replicas.size() == 1);
  CHECK(gs_stub->registered_replicas[0] == artifact_name);
  REQUIRE(gs_stub->recorded_variants.size() == 1);
  const auto& recorded = gs_stub->recorded_variants[0];
  CHECK(std::get<0>(recorded) == artifact_name);
  CHECK(std::get<1>(recorded) == "view-weights-narrow");
  CHECK(std::get<2>(recorded) == 4 * sizeof(float));

  tensorcast::store::loading::ReplicaKey variant_key{
      .artifact_id = artifact_name,
      .view_id = std::optional<std::string>("view-weights-narrow"),
      .device = cpu_device,
      .replica = 0};
  auto variant_size_or = store.get_replica_size(variant_key);
  REQUIRE(variant_size_or.ok());
  CHECK(*variant_size_or == 4 * sizeof(float));

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec_cleanup;
  fs::remove_all(temp_root, ec_cleanup);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 2: Query helpers after materialize_replica()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("StoreEngine helper queries after materialize_replica()", "[store_engine][materialize_replica][status]") {
  if (!tensorcast::testing::is_cuda_available()) {
    WARN("CUDA not available – skipping status tests with materialize_replica().");
    return;
  }

  const std::string artifact_id = "status_artifact";
  const size_t artifact_size = 2 * 1024 * 1024; // 2 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_prepare_status_test";
  fs::create_directories(temp_root);
  fs::path artifact_dir2 = temp_root / artifact_id;
  fs::create_directories(artifact_dir2);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir2 / "tensor.data_0", artifact_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir2).ok());

  StoreEngine store = make_store(temp_root);

  // Load to GPU.
  {
    tensorcast::store::loading::MaterializeHints hints2;
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
