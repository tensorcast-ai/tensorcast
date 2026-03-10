// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

static tensorcast::store::StoreEngineOptions make_opts_basic() {
  tensorcast::store::StoreEngineOptions opts;
  // Small pool for test environments
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

static tensorcast::daemon::DaemonOptions make_daemon_options() {
  tensorcast::daemon::DaemonOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path();
  return opts;
}

static std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
    const tensorcast::daemon::DaemonOptions& options) {
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

TEST_CASE("DISK unload is idempotent success", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  tensorcast::daemon::v2::UnloadReplicaRequest req;
  req.set_disk_path("/tmp/does-not-matter");
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_DISK);
  tensorcast::daemon::v2::UnloadReplicaResponse resp;
  grpc::ServerContext ctx;
  auto st = svc.UnloadReplica(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.code() == 0);
}

TEST_CASE("MaterializeReplica rejects while shutting down", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  harness->kernel().begin_shutdown();
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  grpc::ServerContext ctx;
  auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::UNAVAILABLE);
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_FAILED);
}

TEST_CASE("MaterializeReplica validates selection inputs", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  // Missing selection.artifact_id -> INVALID_ARGUMENT
  {
    tensorcast::daemon::v2::MaterializeReplicaRequest req;
    req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
    tensorcast::daemon::v2::MaterializeReplicaResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  }

  // selection.artifact_id present is accepted (may still fail deeper in the stack, but not on input validation)
  {
    tensorcast::daemon::v2::MaterializeReplicaRequest req;
    req.mutable_selection()->set_artifact_id("mi2:abc:def");
    req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
    tensorcast::daemon::v2::MaterializeReplicaResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    REQUIRE(st.error_code() != grpc::StatusCode::INVALID_ARGUMENT);
  }
}

TEST_CASE("WaitReplicaVerification unknown returns UNKNOWN", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  auto harness = make_harness(engine, make_daemon_options());
  auto& svc = harness->service();

  tensorcast::daemon::v2::WaitReplicaVerificationRequest req;
  req.set_artifact_id("nonexistent");
  req.set_replica_uuid("deadbeef-dead-beef-dead-beefdeadbeef");
  req.set_timeout_ms(10);
  tensorcast::daemon::v2::WaitReplicaVerificationResponse resp;
  grpc::ServerContext ctx;
  auto st = svc.WaitReplicaVerification(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::VerificationStatus::VERIFICATION_STATUS_UNSPECIFIED);
}
