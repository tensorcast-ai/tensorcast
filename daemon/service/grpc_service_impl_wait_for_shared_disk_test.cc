// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "core/testing/common.h"
#include "grpcpp/server_context.h"
#include "nlohmann/json.hpp"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_wait_for_shared_disk_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 0; // Let the OS pick an available port for test isolation.
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear(); // explicit offline mode; tests wire a stub client directly
  return opts;
}

std::string read_artifact_id(const std::filesystem::path& artifact_dir) {
  std::ifstream descriptor_in(artifact_dir / "artifact_descriptor.json");
  nlohmann::json descriptor_json;
  descriptor_in >> descriptor_json;
  return descriptor_json.value("artifact_id", "");
}

class WaitForSharedDiskGlobalStoreClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::string cluster_id{"cluster-test"};
  std::unordered_map<std::string, std::string> key_mappings;
  std::vector<tensorcast::store::components::ArtifactDiskLocation> disk_locations;

  std::atomic<int> list_calls{0};
  int ready_after_calls{0};

  absl::StatusOr<std::string> get_cluster_id() override {
    if (cluster_id.empty()) {
      return absl::NotFoundError("cluster_id unavailable");
    }
    return cluster_id;
  }

  absl::StatusOr<tensorcast::store::components::KeyMapping> resolve_key_mapping(std::string_view key) override {
    auto it = key_mappings.find(std::string(key));
    if (it == key_mappings.end()) {
      return absl::NotFoundError("key not found");
    }
    tensorcast::store::components::KeyMapping mapping;
    mapping.artifact_id = it->second;
    return mapping;
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ArtifactDiskLocation>> list_artifact_disk_locations(
      std::string_view artifact_id,
      bool include_deleted = false) override {
    (void)include_deleted;
    const int call_count = list_calls.fetch_add(1) + 1;
    if (call_count <= ready_after_calls) {
      return absl::NotFoundError("disk_locations_not_ready");
    }
    std::vector<tensorcast::store::components::ArtifactDiskLocation> out;
    out.reserve(disk_locations.size());
    for (const auto& entry : disk_locations) {
      if (entry.artifact_id == artifact_id) {
        out.push_back(entry);
      }
    }
    if (out.empty()) {
      return absl::NotFoundError("disk_locations_not_found");
    }
    return out;
  }
};

void register_disk_location(
    WaitForSharedDiskGlobalStoreClient& client,
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
    "MaterializeReplica waits for managed shared-disk and retries disk-only",
    "[daemon][materialize][disk][wait]") {
  auto gs_client = std::make_shared<WaitForSharedDiskGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->ready_after_calls = 2;

  const auto storage_root = test_tmpdir() / "storage_ready";
  std::filesystem::create_directories(storage_root);

  // Prepare a disk artifact that already exists, but only becomes discoverable via disk_locations after a short delay.
  const auto artifact_rel =
      std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_wait_ready";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());
  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  req.set_wait_for_shared_disk_ms(200);
  req.mutable_source_policy()->set_allow_disk(true);
  req.mutable_source_policy()->set_allow_p2p(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  auto status = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE(status.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);
  REQUIRE_FALSE(resp.disk_path().empty());
  REQUIRE(gs_client->list_calls.load() >= 3);
}

TEST_CASE(
    "MaterializeReplica fails when managed shared-disk becomes ready but disk validation fails",
    "[daemon][materialize][disk][wait]") {
  auto gs_client = std::make_shared<WaitForSharedDiskGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->ready_after_calls = 2;

  const auto storage_root = test_tmpdir() / "storage_bad";
  std::filesystem::create_directories(storage_root);

  const auto artifact_rel = std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_wait_bad";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());
  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  std::filesystem::remove(artifact_dir / "artifact_descriptor.json");
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  req.set_wait_for_shared_disk_ms(200);
  req.mutable_source_policy()->set_allow_disk(true);
  req.mutable_source_policy()->set_allow_p2p(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  auto status = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(gs_client->list_calls.load() >= 3);
}

TEST_CASE(
    "MaterializeByKey waits for managed shared-disk and retries disk-only",
    "[daemon][materialize][by-key][disk][wait]") {
  auto gs_client = std::make_shared<WaitForSharedDiskGlobalStoreClient>();
  gs_client->connected = true;
  gs_client->ready_after_calls = 2;

  const auto storage_root = test_tmpdir() / "storage_ready_by_key";
  std::filesystem::create_directories(storage_root);

  const auto artifact_rel =
      std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_wait_ready_by_key";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());
  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  gs_client->key_mappings.emplace("key_wait_ready", artifact_id);
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeByKeyRequest req;
  req.set_key("key_wait_ready");
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  req.set_device_id(0);
  req.set_wait_for_shared_disk_ms(200);
  req.mutable_source_policy()->set_allow_disk(true);
  req.mutable_source_policy()->set_allow_p2p(true);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeByKeyResponse resp;
  const auto status = svc.MaterializeByKey(&ctx, &req, &resp);
  REQUIRE(status.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);
  REQUIRE_FALSE(resp.used_disk_path().empty());
  REQUIRE(gs_client->list_calls.load() >= 3);
}
