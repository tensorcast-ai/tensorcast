// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/state/types.h"
#include "grpcpp/server_context.h"
#include "tensorcast/common/v1/common.pb.h"

namespace {

using tensorcast::daemon::ArtifactDeviceKey;
using tensorcast::daemon::LipLeaseEntry;

constexpr int kDeviceId = 0;

std::filesystem::path make_storage_root() {
  auto root = std::filesystem::temp_directory_path() / "tensorcast_retire_slot_test";
  std::filesystem::create_directories(root);
  return root;
}

tensorcast::store::StoreEngineOptions make_engine_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = make_storage_root();
  opts.p2p_port = 47012;
  opts.memory_pool_size = 64ull << 20;
  opts.tx_slice_bytes = 1ull << 20;
  opts.num_thread = 2;
  return opts;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
    const std::shared_ptr<tensorcast::store::testing::RecordingGlobalStoreClient>& gs) {
  tensorcast::daemon::DaemonOptions options;
  options.storage_path = make_storage_root();
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options, nullptr, gs);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

LipLeaseEntry make_lease(std::string registration_id, std::string artifact_id) {
  LipLeaseEntry entry;
  entry.registration_id = std::move(registration_id);
  entry.artifact_id = std::move(artifact_id);
  entry.device_id = kDeviceId;
  entry.owner_pid = getpid();
  entry.ttl_ms = 60000;
  entry.expiry = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  return entry;
}

} // namespace

TEST_CASE("RetirePublishedReplica marks before drain", "[daemon][retire]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto& lip_manager = harness->kernel().lip_manager();
  LipLeaseEntry lease = make_lease("lease-1", "artifact-1");
  ArtifactDeviceKey key{.artifact_id = lease.artifact_id, .view_id = "", .device_id = lease.device_id};
  lip_manager.put_lease(lease.registration_id, key, lease);
  lip_manager.attach_replica_id(lease.registration_id, "replica-1");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::RetirePublishedReplicaRequest req;
  tensorcast::daemon::v2::RetirePublishedReplicaResponse resp;
  req.set_artifact_id(lease.artifact_id);
  req.set_lease_id(lease.registration_id);
  req.set_device_id(lease.device_id);
  req.set_owner_pid(lease.owner_pid);
  req.set_wait_for_drain(true);
  req.mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);

  auto st = harness->service().RetirePublishedReplica(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.removed());
  REQUIRE(resp.drained());
  REQUIRE(gs->call_sequence == std::vector<std::string>{"mark:replica-1", "drain:replica-1"});
}

TEST_CASE("RetirePublishedReplica deadline exceeded leaves lease", "[daemon][retire]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  gs->drain_success = false;
  gs->drain_current_requests = 2;
  auto harness = make_harness(engine, gs);

  auto& lip_manager = harness->kernel().lip_manager();
  LipLeaseEntry lease = make_lease("lease-2", "artifact-2");
  ArtifactDeviceKey key{.artifact_id = lease.artifact_id, .view_id = "", .device_id = lease.device_id};
  lip_manager.put_lease(lease.registration_id, key, lease);
  lip_manager.attach_replica_id(lease.registration_id, "replica-2");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::RetirePublishedReplicaRequest req;
  tensorcast::daemon::v2::RetirePublishedReplicaResponse resp;
  req.set_artifact_id(lease.artifact_id);
  req.set_lease_id(lease.registration_id);
  req.set_device_id(lease.device_id);
  req.set_owner_pid(lease.owner_pid);
  req.set_wait_for_drain(true);
  req.set_drain_timeout_ms(5);
  req.mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);

  auto st = harness->service().RetirePublishedReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED);
  REQUIRE(gs->call_sequence == std::vector<std::string>{"mark:replica-2", "drain:replica-2"});
  auto still_active = lip_manager.find_active_by_key(key);
  REQUIRE(still_active.has_value());
}
