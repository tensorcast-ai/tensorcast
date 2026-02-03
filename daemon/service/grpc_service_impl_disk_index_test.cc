// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
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

} // namespace

TEST_CASE(
    "MaterializeReplicaResponse carries canonical index for disk loads without Global Store",
    "[daemon][materialize][disk]") {
  // Prepare a minimal disk artifact with descriptor + tensor_index.json.
  const auto artifact_dir = test_tmpdir() / "artifact";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.set_disk_path(artifact_dir.string());
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);

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
  const auto artifact_dir = test_tmpdir() / "artifact_view";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  // Load the descriptor artifact_id so the request matches verification metadata.
  std::ifstream descriptor_in(artifact_dir / "artifact_descriptor.json");
  REQUIRE(descriptor_in.is_open());
  nlohmann::json descriptor_json;
  descriptor_in >> descriptor_json;
  const std::string artifact_id = descriptor_json.value("artifact_id", "");
  REQUIRE_FALSE(artifact_id.empty());

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.set_artifact_id(artifact_id);
  req.set_disk_path(artifact_dir.string());
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  // Set an (identity) view spec to exercise canonical planning without requiring Global Store.
  req.mutable_view();

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  const auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);
}
