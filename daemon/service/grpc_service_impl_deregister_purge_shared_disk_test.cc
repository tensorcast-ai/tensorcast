// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include <unistd.h>

#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/common.h"
#include "daemon/state/types.h"
#include "grpcpp/server_context.h"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_deregister_purge_shared_disk_test";
}

tensorcast::store::StoreEngineOptions make_opts(const std::filesystem::path& storage_root) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (storage_root / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47023;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

tensorcast::daemon::LipLeaseEntry make_test_lease(std::string artifact_id, int device_id, int owner_pid) {
  tensorcast::daemon::LipLeaseEntry entry;
  entry.registration_id = "reg";
  entry.artifact_id = std::move(artifact_id);
  entry.view_id.clear();
  entry.client_artifact_id = entry.artifact_id;
  entry.device_id = device_id;
  entry.owner_pid = owner_pid;
  entry.ttl_ms = 60'000;
  entry.expiry = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  entry.epoch = 1;
  entry.total_size = 0;
  return entry;
}

} // namespace

TEST_CASE("DeregisterArtifact purges managed shared disk by default", "[daemon][deregister][disk]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto storage_root = test_tmpdir() / "purge_default";
  std::filesystem::remove_all(storage_root);
  std::filesystem::create_directories(storage_root);

  const std::string artifact_id = "mi2:indexhash:datahash";
  const auto artifact_rel =
      std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "mi2_indexhash_datahash";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", 64));
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor_index.json", 16));
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "artifact_descriptor.json", 16));

  tensorcast::store::components::ArtifactDiskLocation loc;
  loc.artifact_id = artifact_id;
  loc.cluster_id = gs_client->cluster_id;
  loc.relative_path = artifact_rel.string();
  loc.kind = tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED;
  gs_client->disk_locations.push_back(std::move(loc));

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts(storage_root));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());

  const int owner_pid = static_cast<int>(::getpid());
  const int device_id = 0;
  tensorcast::daemon::LipLeaseEntry lease = make_test_lease(artifact_id, device_id, owner_pid);
  const std::string registration_id = "reg-1";
  lease.registration_id = registration_id;
  const tensorcast::daemon::ArtifactDeviceKey key{
      .artifact_id = artifact_id,
      .view_id = "",
      .device_id = device_id,
  };
  harness->kernel().lip_manager().put_lease(registration_id, key, std::move(lease));

  tensorcast::daemon::v2::DeregisterArtifactRequest req;
  req.set_artifact_id(artifact_id);
  req.set_wait_for_drain(false);
  req.set_owner_pid(owner_pid);
  req.set_device_id(device_id);
  grpc::ServerContext ctx;
  tensorcast::daemon::v2::DeregisterArtifactResponse resp;
  const auto st = harness->service().DeregisterArtifact(&ctx, &req, &resp);
  INFO("grpc status=" << st.error_code() << " msg=" << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.removed());
  REQUIRE(resp.drained());

  REQUIRE_FALSE(std::filesystem::exists(artifact_dir));

  bool tombstoned = false;
  for (const auto& entry : gs_client->disk_locations) {
    if (entry.artifact_id == artifact_id && entry.relative_path == artifact_rel.string()) {
      tombstoned = entry.is_deleted;
      break;
    }
  }
  REQUIRE(tombstoned);
}

TEST_CASE(
    "DeregisterArtifact keeps managed shared disk when keep_shared_disk_copy=true",
    "[daemon][deregister][disk]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto storage_root = test_tmpdir() / "purge_keep";
  std::filesystem::remove_all(storage_root);
  std::filesystem::create_directories(storage_root);

  const std::string artifact_id = "mi2:indexhash:datahash2";
  const auto artifact_rel =
      std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "mi2_indexhash_datahash2";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", 64));

  tensorcast::store::components::ArtifactDiskLocation loc;
  loc.artifact_id = artifact_id;
  loc.cluster_id = gs_client->cluster_id;
  loc.relative_path = artifact_rel.string();
  loc.kind = tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED;
  gs_client->disk_locations.push_back(std::move(loc));

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts(storage_root));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());

  const int owner_pid = static_cast<int>(::getpid());
  const int device_id = 0;
  tensorcast::daemon::LipLeaseEntry lease = make_test_lease(artifact_id, device_id, owner_pid);
  const std::string registration_id = "reg-2";
  lease.registration_id = registration_id;
  const tensorcast::daemon::ArtifactDeviceKey key{
      .artifact_id = artifact_id,
      .view_id = "",
      .device_id = device_id,
  };
  harness->kernel().lip_manager().put_lease(registration_id, key, std::move(lease));

  tensorcast::daemon::v2::DeregisterArtifactRequest req;
  req.set_artifact_id(artifact_id);
  req.set_wait_for_drain(false);
  req.set_owner_pid(owner_pid);
  req.set_device_id(device_id);
  req.set_keep_shared_disk_copy(true);
  grpc::ServerContext ctx;
  tensorcast::daemon::v2::DeregisterArtifactResponse resp;
  const auto st = harness->service().DeregisterArtifact(&ctx, &req, &resp);
  INFO("grpc status=" << st.error_code() << " msg=" << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.removed());

  REQUIRE(std::filesystem::exists(artifact_dir));

  bool deleted = false;
  for (const auto& entry : gs_client->disk_locations) {
    if (entry.artifact_id == artifact_id && entry.relative_path == artifact_rel.string()) {
      deleted = entry.is_deleted;
      break;
    }
  }
  REQUIRE_FALSE(deleted);
}

TEST_CASE(
    "DeregisterArtifact without active lease still retires local disk fallback and purges managed disk",
    "[daemon][deregister][disk]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto storage_root = test_tmpdir() / "purge_no_active_lease";
  std::filesystem::remove_all(storage_root);
  std::filesystem::create_directories(storage_root);

  const std::string artifact_id = "mi2:indexhash:datahash3";
  const auto artifact_rel =
      std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "mi2_indexhash_datahash3";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data_0", 64));

  tensorcast::store::components::ArtifactDiskLocation loc;
  loc.artifact_id = artifact_id;
  loc.cluster_id = gs_client->cluster_id;
  loc.relative_path = artifact_rel.string();
  loc.kind = tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED;
  gs_client->disk_locations.push_back(std::move(loc));

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts(storage_root));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());

  tensorcast::daemon::v2::DeregisterArtifactRequest req;
  req.set_artifact_id(artifact_id);
  req.set_wait_for_drain(false);
  req.set_device_id(0);
  grpc::ServerContext ctx;
  tensorcast::daemon::v2::DeregisterArtifactResponse resp;
  const auto st = harness->service().DeregisterArtifact(&ctx, &req, &resp);
  INFO("grpc status=" << st.error_code() << " msg=" << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.removed());
  REQUIRE(resp.drained());
  REQUIRE(resp.message().find("no active lease found") != std::string::npos);

  REQUIRE_FALSE(std::filesystem::exists(artifact_dir));

  bool tombstoned = false;
  for (const auto& entry : gs_client->disk_locations) {
    if (entry.artifact_id == artifact_id && entry.relative_path == artifact_rel.string()) {
      tombstoned = entry.is_deleted;
      break;
    }
  }
  REQUIRE(tombstoned);
}

TEST_CASE(
    "DeregisterArtifact retires loaded local import replica and prevents rematerialization",
    "[daemon][deregister][disk][local_import]") {
  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto storage_root = test_tmpdir() / "retire_loaded_local_import";
  std::filesystem::remove_all(storage_root);
  std::filesystem::create_directories(storage_root);

  const auto artifact_dir = storage_root / "imports" / "v00001";
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts(storage_root));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());

  tensorcast::daemon::v2::ImportArtifactFromPathRequest resolve_req;
  resolve_req.set_path(artifact_dir.string());
  resolve_req.set_verify_checksums(true);
  grpc::ServerContext resolve_ctx;
  tensorcast::daemon::v2::ImportArtifactFromPathResponse resolve_resp;
  auto resolve_st = harness->service().ImportArtifactFromPath(&resolve_ctx, &resolve_req, &resolve_resp);
  INFO("resolve status=" << resolve_st.error_code() << " msg=" << resolve_st.error_message());
  REQUIRE(resolve_st.ok());
  REQUIRE(resolve_resp.artifact_id().starts_with("mi2:"));
  REQUIRE(harness->kernel().source_registry().lookup_binding(resolve_resp.artifact_id()).has_value());

  tensorcast::daemon::v2::MaterializeReplicaRequest materialize_req;
  materialize_req.mutable_selection()->set_artifact_id(resolve_resp.artifact_id());
  materialize_req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  materialize_req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  grpc::ServerContext materialize_ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse materialize_resp;
  auto materialize_st = harness->service().MaterializeReplica(&materialize_ctx, &materialize_req, &materialize_resp);
  INFO("materialize status=" << materialize_st.error_code() << " msg=" << materialize_st.error_message());
  REQUIRE(materialize_st.ok());
  REQUIRE(materialize_resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);

  tensorcast::daemon::v2::DeregisterArtifactRequest deregister_req;
  deregister_req.set_artifact_id(resolve_resp.artifact_id());
  deregister_req.set_wait_for_drain(false);
  grpc::ServerContext deregister_ctx;
  tensorcast::daemon::v2::DeregisterArtifactResponse deregister_resp;
  auto deregister_st = harness->service().DeregisterArtifact(&deregister_ctx, &deregister_req, &deregister_resp);
  INFO("deregister status=" << deregister_st.error_code() << " msg=" << deregister_st.error_message());
  REQUIRE(deregister_st.ok());
  REQUIRE(deregister_resp.removed());
  REQUIRE_FALSE(harness->kernel().source_registry().lookup_binding(resolve_resp.artifact_id()).has_value());

  grpc::ServerContext materialize_again_ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse materialize_again_resp;
  auto materialize_again_st =
      harness->service().MaterializeReplica(&materialize_again_ctx, &materialize_req, &materialize_again_resp);
  INFO(
      "materialize_again status=" << materialize_again_st.error_code()
                                  << " msg=" << materialize_again_st.error_message());
  REQUIRE_FALSE(materialize_again_st.ok());
  const bool expected_status = materialize_again_st.error_code() == grpc::StatusCode::NOT_FOUND ||
      materialize_again_st.error_code() == grpc::StatusCode::FAILED_PRECONDITION ||
      materialize_again_st.error_code() == grpc::StatusCode::UNAVAILABLE;
  REQUIRE(expected_status);
}
