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
#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "core/store/memory_tier_config.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/common.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::MemoryTierConfig;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::loading::ReplicaKeyHash;
namespace components = tensorcast::store::components;

using tensorcast::store::testing::MakeRecordingGlobalStoreClient;
using tensorcast::store::testing::RecordingGlobalStoreClient;

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────
static absl::Status wait_ready(
    tensorcast::store::loading::ReplicaHandle& handle,
    absl::Duration timeout = absl::Seconds(60));

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
    const nlohmann::json& index_json,
    uint64_t chunk_size_bytes = 4ULL * 1024 * 1024) {
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
  hp["chunk_size"] = chunk_size_bytes;
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

static absl::StatusOr<std::string> read_descriptor_artifact_id(const fs::path& artifact_dir) {
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  std::error_code ec;
  if (!fs::exists(descriptor_path, ec)) {
    return absl::NotFoundError("artifact_descriptor.json not found");
  }
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), "failed to stat artifact_descriptor.json");
  }
  try {
    nlohmann::json j;
    std::ifstream in(descriptor_path);
    if (!in.is_open()) {
      return absl::InternalError("failed to open artifact_descriptor.json");
    }
    in >> j;
    if (!j.contains("artifact_id") || !j["artifact_id"].is_string()) {
      return absl::FailedPreconditionError("artifact_descriptor.json missing artifact_id");
    }
    return j["artifact_id"].get<std::string>();
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("failed to parse artifact_descriptor.json: ", ex.what()));
  }
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
  auto stub_client = MakeRecordingGlobalStoreClient();
  engine.set_global_store_client_for_testing(stub_client);

  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = "temp-reg-mi2";
  reg.device_id = 0;
  reg.total_size_bytes = 256 * 1024;
  reg.tensor_index_data = std::string("{}");

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());

  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  INFO("commit status (mi2): " << commit_or.status());
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
  auto stub_client = MakeRecordingGlobalStoreClient();
  engine.set_global_store_client_for_testing(stub_client);

  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = "temp-reg-cgid";
  reg.device_id = 0;
  reg.total_size_bytes = 128 * 1024;
  reg.tensor_index_data = std::string("{}");
  reg.client_artifact_id = std::string("cgid:engine-test-1");

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());

  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  INFO("commit status (cgid): " << commit_or.status());
  REQUIRE(commit_or.ok());
  const auto& result = commit_or.value();
  REQUIRE(result.id_kind == tensorcast::common::ArtifactIdKind::kCgid);
  REQUIRE(result.artifact_id == "cgid:engine-test-1");
  REQUIRE(result.index_multihash.empty());
  REQUIRE(result.data_multihash.empty());
}

TEST_CASE("StoreEngine supports CGID load/list/lease without hashes", "[store_engine][cgid][materialize]") {
  const std::string artifact_dir_name = "cgid_artifact";
  fs::path temp_root = fs::temp_directory_path() / "store_engine_cgid_materialize";
  fs::create_directories(temp_root);
  fs::path artifact_dir = temp_root / artifact_dir_name;
  fs::create_directories(artifact_dir);
  const uint64_t payload_size = 256 * 1024; // 256 KiB single chunk under default UMA granularity
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", payload_size));

  const std::string cgid = "cgid:test-cgid-materialize";

  StoreEngine store = make_store(temp_root);
  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};

  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = cgid;
  hints.disk_path = artifact_dir.string();

  auto handle_or = store.materialize_replica(cpu_device, StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  INFO("cgid load status: " << handle_or.status());
  REQUIRE(handle_or.ok());
  auto handle = std::move(*handle_or);
  REQUIRE(wait_ready(handle).ok());
  CHECK(handle.replica_key.artifact_id == cgid);

  auto replicas = store.list_device_replicas(cpu_device);
  REQUIRE(replicas.size() == 1);
  CHECK(replicas[0].artifact_id == cgid);

  auto devices = store.get_resident_devices(cgid);
  REQUIRE(!devices.empty());

  auto size_or = store.get_replica_size(handle.replica_key);
  REQUIRE(size_or.ok());
  CHECK(*size_or == payload_size);

  components::MemoryTierLeaseDescriptor lease_req;
  lease_req.artifact_id = cgid;
  lease_req.kind = components::MemoryTierLeaseKind::kStable;
  lease_req.chunk_start = 0;
  lease_req.chunk_count = 1;
  auto lease_or = store.acquire_memory_tier_lease(lease_req);
  INFO("cgid lease status: " << lease_or.status());
  REQUIRE(lease_or.ok());
  auto lease = *lease_or;
  REQUIRE_FALSE(lease.chunk_ids.empty());
  REQUIRE(lease.bytes > 0);

  auto release_or = store.release_memory_tier_lease(lease);
  INFO("cgid release status: " << release_or.status());
  REQUIRE(release_or.ok());

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

static absl::Status wait_ready(tensorcast::store::loading::ReplicaHandle& handle, absl::Duration timeout) {
  return handle.wait_ready(std::chrono::milliseconds(absl::ToInt64Milliseconds(timeout)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Test case 1: Basic CPU → GPU workflow using materialize_replica()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("StoreEngine materialize_replica() GPU workflow", "[store_engine][materialize_replica][cpu][gpu]") {
  const std::string artifact_dir_name = "dummy_artifact";
  const size_t artifact_size = 1 * 1024 * 1024; // 1 MiB

  // Create temporary directory and dummy replica file
  fs::path temp_root = fs::temp_directory_path() / "store_engine_prepare_test";
  fs::create_directories(temp_root);
  fs::path artifact_dir = temp_root / artifact_dir_name;
  fs::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", artifact_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());
  auto artifact_id_or = read_descriptor_artifact_id(artifact_dir);
  REQUIRE(artifact_id_or.ok());
  const std::string artifact_id = *artifact_id_or;

  StoreEngine store = make_store(temp_root);

  // Load to GPU (Now only GPU is supported)
  REQUIRE(tensorcast::testing::is_cuda_available());
  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  hints.disk_path = artifact_dir.string();
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
  canonical_hints.artifact_id = canonical_artifact_id;
  canonical_hints.disk_path = artifact_dir.string();
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
  variant_identity.canonical_artifact_id = canonical_artifact_id;
  variant_identity.view_id = std::string("view-weights-narrow");
  variant_identity.view_spec = spec;
  variant_identity.canonical_index_json = index.dump();

  tensorcast::store::loading::MaterializeHints variant_hints;
  variant_hints.artifact_id = canonical_artifact_id;
  variant_hints.disk_path = artifact_dir.string();
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
      .artifact_id = canonical_artifact_id, .view_id = std::nullopt, .device = cpu_device, .replica = 0};
  tensorcast::store::loading::ReplicaKey variant_key{
      .artifact_id = canonical_artifact_id,
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
  const std::string canonical_artifact_id = *descriptor_or;

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
  variant_identity.canonical_artifact_id = canonical_artifact_id;
  variant_identity.view_id = std::string("view-weights-narrow");
  variant_identity.view_spec = spec;
  variant_identity.canonical_index_json = index.dump();

  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = canonical_artifact_id;
  hints.disk_path = artifact_dir.string();
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
  CHECK(gs_stub->replica_requests[0] == canonical_artifact_id);
  REQUIRE(gs_stub->registered_replicas.size() == 1);
  CHECK(gs_stub->registered_replicas[0] == canonical_artifact_id);
  REQUIRE(gs_stub->recorded_variants.size() == 1);
  const auto& recorded = gs_stub->recorded_variants[0];
  CHECK(std::get<0>(recorded) == canonical_artifact_id);
  CHECK(std::get<1>(recorded) == "view-weights-narrow");
  CHECK(std::get<2>(recorded) == 4 * sizeof(float));

  tensorcast::store::loading::ReplicaKey variant_key{
      .artifact_id = canonical_artifact_id,
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

  const std::string artifact_dir_name = "status_artifact";
  const size_t artifact_size = 2 * 1024 * 1024; // 2 MiB

  fs::path temp_root = fs::temp_directory_path() / "store_engine_prepare_status_test";
  fs::create_directories(temp_root);
  fs::path artifact_dir2 = temp_root / artifact_dir_name;
  fs::create_directories(artifact_dir2);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir2 / "tensor.data_0", artifact_size));

  // RFC-0007: standard partitions require descriptor and canonical index
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir2).ok());
  auto artifact_id_or = read_descriptor_artifact_id(artifact_dir2);
  REQUIRE(artifact_id_or.ok());
  const std::string artifact_id = *artifact_id_or;

  StoreEngine store = make_store(temp_root);

  // Load to GPU.
  {
    tensorcast::store::loading::MaterializeHints hints2;
    hints2.artifact_id = artifact_id;
    hints2.disk_path = artifact_dir2.string();
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

TEST_CASE("StoreEngine reconciles stable memory tier leases locally", "[store_engine][memory_tier][stable_lease]") {
  const std::string artifact_name = "memory_tier_artifact";
  fs::path temp_root = fs::temp_directory_path() / "store_engine_memory_tier";
  fs::create_directories(temp_root);
  fs::path artifact_dir = temp_root / artifact_name;
  fs::create_directories(artifact_dir);

  const size_t payload_size = 2 * 1024 * 1024; // 2 MiB
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", payload_size));

  nlohmann::json index = nlohmann::json::object();
  index["weights"] = make_tensor_entry(
      /*offset=*/0,
      /*size=*/payload_size,
      /*shape=*/{static_cast<int64_t>(payload_size / sizeof(float))},
      /*stride=*/{1},
      /*dtype=*/"torch.float32");
  auto artifact_id_or = write_descriptor_with_index(artifact_dir, index);
  REQUIRE(artifact_id_or.ok());
  const std::string artifact_id = *artifact_id_or;

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 128ULL * 1024 * 1024; // ensure pinned pool can stage streaming buffers
  opts.tx_slice_bytes = 4ULL * 1024 * 1024;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  MemoryTierConfig tiers;
  tiers.enable_preemptible_memory = false;
  tiers.stable_bytes = 16ULL * 1024 * 1024;
  tiers.preemptible_limit_bytes = 0;
  tiers.preemptible_low_watermark_ratio = 0.4;
  opts.memory_tier_config = tiers;

  StoreEngine store(opts);
  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};

  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  hints.disk_path = artifact_dir.string();
  auto cpu_handle_or = store.materialize_replica(cpu_device, StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  REQUIRE(cpu_handle_or.ok());
  auto cpu_handle = std::move(cpu_handle_or).value();
  REQUIRE(wait_ready(cpu_handle).ok());

  components::MemoryTierLeaseDescriptor lease_req;
  lease_req.artifact_id = artifact_id;
  lease_req.kind = components::MemoryTierLeaseKind::kStable;
  lease_req.chunk_start = 0;
  lease_req.chunk_count = 1;

  auto acquired_or = store.acquire_memory_tier_lease(lease_req);
  REQUIRE(acquired_or.ok());
  auto acquired = *acquired_or;
  REQUIRE(acquired.bytes > 0);
  REQUIRE_FALSE(acquired.chunk_ids.empty());

  auto snap_after_acquire = store.get_memory_tier_snapshot();
  REQUIRE(snap_after_acquire.has_value());
  REQUIRE(snap_after_acquire->stable_used_bytes >= acquired.bytes);

  auto released_or = store.release_memory_tier_lease(acquired);
  REQUIRE(released_or.ok());
  auto snap_after_release = store.get_memory_tier_snapshot();
  REQUIRE(snap_after_release.has_value());
  REQUIRE(snap_after_release->stable_used_bytes <= snap_after_acquire->stable_used_bytes);

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}

TEST_CASE(
    "StoreEngine memory tier leases normalize ranges and reuse held leases",
    "[store_engine][memory_tier][leases]") {
  const uint64_t chunk_bytes = 1ULL * 1024 * 1024; // 1 MiB UMA granularity for predictable chunks
  const std::string artifact_name = "memory_tier_normalize_artifact";
  fs::path temp_root = fs::temp_directory_path() / "store_engine_memory_tier_normalize";
  fs::create_directories(temp_root);
  fs::path artifact_dir = temp_root / artifact_name;
  fs::create_directories(artifact_dir);

  const uint64_t part_a = chunk_bytes;
  const uint64_t part_b = chunk_bytes / 2;
  const uint64_t total_size = part_a + part_b;

  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", part_a));
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_1", part_b));

  nlohmann::json index = nlohmann::json::object();
  index["a"] = make_tensor_entry(0, part_a, {static_cast<int64_t>(part_a)}, {1}, "torch.uint8");
  index["b"] = make_tensor_entry(part_a, part_b, {static_cast<int64_t>(part_b)}, {1}, "torch.uint8");

  auto artifact_id_or = write_descriptor_with_index(artifact_dir, index, chunk_bytes);
  REQUIRE(artifact_id_or.ok());
  const std::string artifact_id = *artifact_id_or;

  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 64ULL * 1024 * 1024; // avoid pinned pool starvation during streaming loads
  opts.tx_slice_bytes = chunk_bytes;
  opts.artifact_chunk_bytes = chunk_bytes;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  MemoryTierConfig tiers;
  tiers.enable_preemptible_memory = true;
  tiers.stable_bytes = 16ULL * chunk_bytes;
  tiers.preemptible_limit_bytes = 4ULL * chunk_bytes;
  tiers.preemptible_low_watermark_ratio = 0.3;
  opts.memory_tier_config = tiers;

  StoreEngine store(opts);
  DeviceKey cpu_device{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};

  tensorcast::store::loading::MaterializeHints hints;
  hints.artifact_id = artifact_id;
  hints.disk_path = artifact_dir.string();
  auto cpu_handle_or = store.materialize_replica(cpu_device, StoreEngine::MaterializeMode::LOAD_ONLY, hints);
  REQUIRE(cpu_handle_or.ok());
  auto cpu_handle = std::move(cpu_handle_or).value();
  REQUIRE(wait_ready(cpu_handle).ok());
  std::string lease_artifact_id = cpu_handle.key().artifact_id;
  auto resident_descriptor = store.get_resident_devices(artifact_id);
  INFO("resident (descriptor id) count: " << resident_descriptor.size());
  auto runtime_cpu_replicas = store.list_device_replicas(cpu_device);
  INFO("runtime cpu replicas: " << runtime_cpu_replicas.size());
  if (!runtime_cpu_replicas.empty()) {
    INFO("runtime cpu replica id: " << runtime_cpu_replicas.front().artifact_id);
    lease_artifact_id = runtime_cpu_replicas.front().artifact_id;
  }
  INFO("lease artifact id: " << lease_artifact_id);

  const uint32_t total_chunks = static_cast<uint32_t>((total_size + chunk_bytes - 1) / chunk_bytes);

  auto expected_bytes = [&](const std::vector<uint32_t>& ids) {
    uint64_t total = 0;
    for (uint32_t idx : ids) {
      const uint64_t start = static_cast<uint64_t>(idx) * chunk_bytes;
      const uint64_t end = std::min<uint64_t>(total_size, start + chunk_bytes);
      if (end > start) {
        total += end - start;
      }
    }
    return total;
  };

  auto make_base_descriptor = [&]() {
    components::MemoryTierLeaseDescriptor lease_req;
    lease_req.artifact_id = lease_artifact_id;
    lease_req.kind = components::MemoryTierLeaseKind::kStable;
    lease_req.chunk_start = 0;
    lease_req.chunk_count = 0;
    return lease_req;
  };

  SECTION("expands ranges and clamps to available chunks") {
    auto lease_req = make_base_descriptor();
    lease_req.chunk_start = 1;
    lease_req.chunk_count = 5;

    auto lease_or = store.acquire_memory_tier_lease(lease_req);
    INFO("expand status: " << lease_or.status());
    REQUIRE(lease_or.ok());
    auto lease = *lease_or;
    std::vector<uint32_t> expected_ids{1};
    REQUIRE(lease.chunk_ids == expected_ids);
    REQUIRE(lease.bytes == expected_bytes(expected_ids));
    REQUIRE(lease.ledger_version > 0);
  }

  SECTION("rejects out-of-range inputs") {
    auto lease_req = make_base_descriptor();
    lease_req.chunk_start = total_chunks;
    lease_req.chunk_count = 1;

    auto lease_or = store.acquire_memory_tier_lease(lease_req);
    INFO("out-of-range status: " << lease_or.status());
    REQUIRE_FALSE(lease_or.ok());
    REQUIRE(absl::IsOutOfRange(lease_or.status()));

    lease_req = make_base_descriptor();
    lease_req.chunk_ids = {total_chunks + 1};
    auto lease_ids_or = store.acquire_memory_tier_lease(lease_req);
    INFO("out-of-range ids status: " << lease_ids_or.status());
    REQUIRE_FALSE(lease_ids_or.ok());
    REQUIRE(absl::IsOutOfRange(lease_ids_or.status()));
  }

  SECTION("re-acquiring held leases is idempotent") {
    auto lease_req = make_base_descriptor();
    lease_req.chunk_count = 2;

    auto first_or = store.acquire_memory_tier_lease(lease_req);
    INFO("first acquire status: " << first_or.status());
    REQUIRE(first_or.ok());
    auto first = *first_or;
    REQUIRE(first.ledger_version > 0);

    auto second_or = store.acquire_memory_tier_lease(lease_req);
    INFO("second acquire status: " << second_or.status());
    REQUIRE(second_or.ok());
    auto second = *second_or;

    REQUIRE(second.chunk_ids == first.chunk_ids);
    REQUIRE(second.bytes == first.bytes);
    REQUIRE(second.ledger_version == first.ledger_version);
  }

  SECTION("release gracefully handles missing leases") {
    auto lease_req = make_base_descriptor();
    lease_req.chunk_count = 1;

    auto cpu_replicas = store.list_device_replicas(cpu_device);
    if (cpu_replicas.empty()) {
      tensorcast::store::loading::MaterializeHints reload_hints;
      reload_hints.artifact_id = artifact_id;
      reload_hints.disk_path = artifact_dir.string();
      auto reload_or = store.materialize_replica(cpu_device, StoreEngine::MaterializeMode::LOAD_ONLY, reload_hints);
      REQUIRE(reload_or.ok());
      auto reload_handle = std::move(*reload_or);
      REQUIRE(wait_ready(reload_handle).ok());
      cpu_replicas = store.list_device_replicas(cpu_device);
    }
    INFO("cpu replicas present: " << cpu_replicas.size());
    if (!cpu_replicas.empty()) {
      INFO("cpu replica key id: " << cpu_replicas.front().artifact_id);
      lease_artifact_id = cpu_replicas.front().artifact_id;
    }

    auto acquired_or = store.acquire_memory_tier_lease(lease_req);
    INFO("acquire status: " << acquired_or.status());
    REQUIRE(acquired_or.ok());
    auto acquired = *acquired_or;

    auto released_or = store.release_memory_tier_lease(acquired);
    INFO("first release status: " << released_or.status());
    REQUIRE(released_or.ok());

    auto released_again_or = store.release_memory_tier_lease(acquired);
    INFO("second release status: " << released_again_or.status());
    REQUIRE(released_again_or.ok());
    auto released_again = *released_again_or;
    REQUIRE(released_again.bytes == expected_bytes(acquired.chunk_ids));
    REQUIRE(released_again.ledger_version >= released_or->ledger_version);
  }

  REQUIRE(store.clear_mem() == 0);
  std::error_code ec;
  fs::remove_all(temp_root, ec);
}
