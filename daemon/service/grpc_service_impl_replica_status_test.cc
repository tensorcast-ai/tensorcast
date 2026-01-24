// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "core/common/ready_signal.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "grpcpp/server_context.h"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_replica_status_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47020;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

} // namespace

TEST_CASE("Query/Wait/ReleaseReplica operate on replica_uuid sessions", "[daemon][grpc][replica_status]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(daemon_opts.storage_path);

  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());

  auto& svc = harness->service();

  const std::string replica_uuid = "op-1";
  tensorcast::store::loading::ReplicaKey key;
  key.artifact_id = "mi2:test:replica_status";
  key.device = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  key.replica = 0;

  auto ready = std::make_shared<tensorcast::common::ReadySignal<absl::Status>>();
  REQUIRE(harness->kernel().sessions_service().put_with_verification(replica_uuid, key, ready).ok());

  // Query: running
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::QueryReplicaStatusRequest req;
    req.mutable_ticket()->set_replica_uuid(replica_uuid);
    tensorcast::daemon::v2::QueryReplicaStatusResponse resp;
    const auto st = svc.QueryReplicaStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.status().state() == tensorcast::daemon::v2::REPLICA_OPERATION_STATE_RUNNING);
    REQUIRE_FALSE(resp.replica_key_hash().empty());
  }

  // Wait: timeout budget -> degraded
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::WaitReplicaStatusRequest req;
    req.mutable_ticket()->set_replica_uuid(replica_uuid);
    req.set_timeout_ms(1);
    tensorcast::daemon::v2::WaitReplicaStatusResponse resp;
    const auto st = svc.WaitReplicaStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.status().state() == tensorcast::daemon::v2::REPLICA_OPERATION_STATE_DEGRADED);
    REQUIRE(resp.status().has_error());
    REQUIRE(resp.status().error().status_code() == "DEADLINE_EXCEEDED");
  }

  ready->set_value(absl::OkStatus());

  // Wait: now succeeds
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::WaitReplicaStatusRequest req;
    req.mutable_ticket()->set_replica_uuid(replica_uuid);
    req.set_timeout_ms(1000);
    tensorcast::daemon::v2::WaitReplicaStatusResponse resp;
    const auto st = svc.WaitReplicaStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.status().state() == tensorcast::daemon::v2::REPLICA_OPERATION_STATE_SUCCESS);
  }

  // Release: removes the operation record only.
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::ReleaseReplicaRequest req;
    req.mutable_ticket()->set_replica_uuid(replica_uuid);
    tensorcast::daemon::v2::ReleaseReplicaResponse resp;
    const auto st = svc.ReleaseReplica(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.released());
  }

  // Query: not found
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::QueryReplicaStatusRequest req;
    req.mutable_ticket()->set_replica_uuid(replica_uuid);
    tensorcast::daemon::v2::QueryReplicaStatusResponse resp;
    const auto st = svc.QueryReplicaStatus(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::NOT_FOUND);
  }
}

TEST_CASE("ReleaseReplica is operation-scoped", "[daemon][grpc][replica_status]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(daemon_opts.storage_path);

  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());

  auto& svc = harness->service();

  tensorcast::store::loading::ReplicaKey key;
  key.artifact_id = "mi2:test:replica_status";
  key.device = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  key.replica = 0;

  const std::string op1_uuid = "op-1";
  const std::string op2_uuid = "op-2";
  auto ready1 = std::make_shared<tensorcast::common::ReadySignal<absl::Status>>();
  auto ready2 = std::make_shared<tensorcast::common::ReadySignal<absl::Status>>();
  REQUIRE(harness->kernel().sessions_service().put_with_verification(op1_uuid, key, ready1).ok());
  REQUIRE(harness->kernel().sessions_service().put_with_verification(op2_uuid, key, ready2).ok());

  ready1->set_value(absl::OkStatus());
  ready2->set_value(absl::OkStatus());

  // Cancel/release one operation; the other should remain queryable and complete.
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::ReleaseReplicaRequest req;
    req.mutable_ticket()->set_replica_uuid(op1_uuid);
    tensorcast::daemon::v2::ReleaseReplicaResponse resp;
    const auto st = svc.ReleaseReplica(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.released());
  }

  // op1: not found
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::QueryReplicaStatusRequest req;
    req.mutable_ticket()->set_replica_uuid(op1_uuid);
    tensorcast::daemon::v2::QueryReplicaStatusResponse resp;
    const auto st = svc.QueryReplicaStatus(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::NOT_FOUND);
  }

  // op2: still present and successful
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::QueryReplicaStatusRequest req;
    req.mutable_ticket()->set_replica_uuid(op2_uuid);
    tensorcast::daemon::v2::QueryReplicaStatusResponse resp;
    const auto st = svc.QueryReplicaStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.status().state() == tensorcast::daemon::v2::REPLICA_OPERATION_STATE_SUCCESS);
  }
}
