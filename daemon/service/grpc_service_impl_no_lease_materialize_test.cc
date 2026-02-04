// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <unistd.h>

#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
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
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_no_lease_materialize_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47022;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

std::string read_artifact_id(const std::filesystem::path& artifact_dir) {
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  std::ifstream in(descriptor_path);
  nlohmann::json j;
  in >> j;
  return j.at("artifact_id").get<std::string>();
}

} // namespace

TEST_CASE("lease_mode=NO_LEASE omits mem_handle and skips PID guards", "[daemon][materialize][no_lease]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto storage_root = test_tmpdir();
  const auto artifact_rel = std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());
  const std::string artifact_id = read_artifact_id(artifact_dir);
  tensorcast::store::components::ArtifactDiskLocation loc;
  loc.artifact_id = artifact_id;
  loc.cluster_id = gs_client->cluster_id;
  loc.relative_path = artifact_rel.string();
  loc.kind = tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED;
  gs_client->disk_locations.push_back(std::move(loc));

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

  const int pid = static_cast<int>(::getpid());

  // NO_LEASE forbids wait_for_completion (no handle export).
  {
    tensorcast::daemon::v2::MaterializeReplicaRequest req;
    req.set_artifact_id(artifact_id);
    req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
    req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
    req.set_wait_for_completion(true);
    req.set_pid(pid);
    req.set_replica_uuid("op-no-lease");
    req.set_lease_mode(tensorcast::daemon::v2::LeaseMode::LEASE_MODE_NO_LEASE);

    grpc::ServerContext ctx;
    tensorcast::daemon::v2::MaterializeReplicaResponse resp;
    const auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  }

  // NO_LEASE with wait_for_completion=false returns a ticket but no mem_handle.
  {
    tensorcast::daemon::v2::MaterializeReplicaRequest req;
    req.set_artifact_id(artifact_id);
    req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
    req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
    req.set_wait_for_completion(false);
    req.set_pid(pid);
    req.set_replica_uuid("op-no-lease");
    req.set_lease_mode(tensorcast::daemon::v2::LeaseMode::LEASE_MODE_NO_LEASE);

    grpc::ServerContext ctx;
    tensorcast::daemon::v2::MaterializeReplicaResponse resp;
    const auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE_FALSE(resp.has_mem_handle());
    REQUIRE(resp.has_ticket());
    REQUIRE(resp.ticket().replica_uuid() == "op-no-lease");
  }

  REQUIRE_FALSE(harness->kernel().lifecycle_manager().has_pid_guard_for_test(pid));
}
