// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/dataplane/metadata/disk_dir_hash.h"
#include "core/store/store_engine.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/common.h"
#include "grpcpp/server_context.h"
#include "nlohmann/json.hpp"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_disk_index_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 0; // Let the OS pick an available port for test isolation.
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear(); // explicit offline mode
  return opts;
}

std::string read_artifact_id(const std::filesystem::path& artifact_dir) {
  std::ifstream descriptor_in(artifact_dir / "artifact_descriptor.json");
  nlohmann::json descriptor_json;
  descriptor_in >> descriptor_json;
  return descriptor_json.value("artifact_id", "");
}

bool write_file(const std::filesystem::path& path, std::string_view payload) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  return out.good();
}

absl::Status write_custom_artifact_dir(
    const std::filesystem::path& artifact_dir,
    std::string_view data_payload,
    const nlohmann::json& index_json) {
  const auto data_path = artifact_dir / "tensor.data_0";
  if (!write_file(data_path, data_payload)) {
    return absl::InternalError("failed to write tensor.data_0");
  }
  const std::string index_payload = index_json.dump();
  if (!write_file(artifact_dir / "tensor_index.json", index_payload)) {
    return absl::InternalError("failed to write tensor_index.json");
  }

  auto index_mh_or = tensorcast::common::compute_index_multihash(std::optional<std::string>(index_payload), "");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  auto data_mh_or = tensorcast::store::loader::compute_data_multihash_from_disk_dir(artifact_dir.string());
  if (!data_mh_or.ok()) {
    return data_mh_or.status();
  }
  nlohmann::json descriptor = {
      {"artifact_id", std::string("mi2:") + *index_mh_or + ":" + *data_mh_or},
      {"index_multihash", *index_mh_or},
      {"data_multihash", *data_mh_or},
      {"schema_version", "v3"},
      {"encoding", "json"},
      {"total_size", static_cast<uint64_t>(data_payload.size())},
  };
  if (!write_file(artifact_dir / "artifact_descriptor.json", descriptor.dump(2))) {
    return absl::InternalError("failed to write artifact_descriptor.json");
  }
  return absl::OkStatus();
}

void register_disk_location(
    tensorcast::store::testing::RecordingGlobalStoreClient& client,
    std::string_view artifact_id,
    const std::filesystem::path& relative_path) {
  tensorcast::store::components::ArtifactDiskLocation loc;
  loc.artifact_id = std::string(artifact_id);
  loc.cluster_id = client.cluster_id;
  loc.relative_path = relative_path.string();
  loc.kind = tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED;
  client.disk_locations.push_back(std::move(loc));
}

} // namespace

TEST_CASE(
    "MaterializeReplicaResponse carries canonical index for disk loads without Global Store",
    "[daemon][materialize][disk]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  gs_client->connected = false;

  // Prepare a minimal disk artifact with descriptor + tensor_index.json.
  const auto storage_root = test_tmpdir();
  const auto artifact_rel = std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());
  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  auto status = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE(status.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);
  REQUIRE_FALSE(resp.view_index_json().empty());

  // Ensure the bytes are valid canonical index JSON (no Global Store needed).
  const auto parsed = nlohmann::json::parse(resp.view_index_json());
  REQUIRE(parsed.contains("__dummy__"));
}

TEST_CASE(
    "MaterializeReplica view planning uses disk index when Global Store is offline",
    "[daemon][materialize][disk][view]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  gs_client->connected = false;

  const auto storage_root = test_tmpdir();
  const auto artifact_rel = std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_view";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  // Load the descriptor artifact_id so the request matches verification metadata.
  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  // Set an (identity) view spec to exercise canonical planning without requiring Global Store.
  req.mutable_selection()->mutable_view_spec();

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  const auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);
}

TEST_CASE(
    "MaterializeReplica rejects invalid serving manifest from disk",
    "[daemon][materialize][disk][serving-manifest]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  gs_client->connected = false;

  const auto storage_root = test_tmpdir();
  const auto artifact_rel =
      std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_serving_invalid";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);

  const std::string manifest_payload =
      R"({"schema_version":1,"artifact_kind":"serving","framework_name":"torch","adapter_version":"adapter-v1","serving_abi_version":"abi-v1","representation_contract_hash":"bafkrepresentation","serving_build_digest":"bafkbuilddigest","tensor_schema_hash":"wrong-hash","canonical_tensor_count":1,"serving_manifest_ref":"tensor:__tensorcast_meta__.manifest_json","builder_mode":"pure_transform","build_pipeline_version":"pipeline-v1"})";
  const std::string data_payload = std::string("ABCD") + manifest_payload;
  nlohmann::json index_json = nlohmann::json::object();
  index_json["weights"] =
      nlohmann::json::array({0, 4, nlohmann::json::array({4}), nlohmann::json::array({1}), "torch.uint8", 0});
  index_json["__tensorcast_meta__.manifest_json"] = nlohmann::json::array(
      {4,
       static_cast<uint64_t>(manifest_payload.size()),
       nlohmann::json::array({static_cast<uint64_t>(manifest_payload.size())}),
       nlohmann::json::array({1}),
       "torch.uint8",
       0});
  REQUIRE(write_custom_artifact_dir(artifact_dir, data_payload, index_json).ok());

  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  req.set_wait_for_completion(false);
  req.set_lease_mode(tensorcast::daemon::v2::LeaseMode::LEASE_MODE_NO_LEASE);
  req.set_replica_uuid("serving-preflight-invalid");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  const auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(st.error_message().find("tensor_schema_hash") != std::string::npos);
}

TEST_CASE(
    "MaterializeReplica strict serving policy rejects artifacts without manifest",
    "[daemon][materialize][disk][serving-manifest][strict]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  gs_client->connected = false;

  const auto storage_root = test_tmpdir();
  const auto artifact_rel =
      std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_strict_missing_manifest";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);

  const std::string data_payload = "ABCD";
  nlohmann::json index_json = nlohmann::json::object();
  index_json["weights"] =
      nlohmann::json::array({0, 4, nlohmann::json::array({4}), nlohmann::json::array({1}), "torch.uint8", 0});
  REQUIRE(write_custom_artifact_dir(artifact_dir, data_payload, index_json).ok());

  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  req.set_wait_for_completion(false);
  req.set_lease_mode(tensorcast::daemon::v2::LeaseMode::LEASE_MODE_NO_LEASE);
  req.set_replica_uuid("serving-preflight-strict-missing");
  req.mutable_serving_artifact_policy()->set_require_manifest(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  const auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE_FALSE(st.ok());
  const auto error_code = st.error_code();
  REQUIRE((error_code == grpc::StatusCode::DATA_LOSS || error_code == grpc::StatusCode::INTERNAL));
  REQUIRE(st.error_message().find("missing manifest tensor") != std::string::npos);
}
