// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

static tensorcast::store::StoreEngineOptions make_opts_basic() {
  tensorcast::store::StoreEngineOptions opts;
  // Small pool for test environments
  opts.memory_pool_size = 64ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

TEST_CASE("DISK unload is idempotent success", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  StoreDaemonServiceImpl svc(engine);

  tensorcast::daemon::v1::UnloadReplicaRequest req;
  req.set_disk_path("/tmp/does-not-matter");
  req.set_target_device_type(tensorcast::daemon::v1::DeviceType::DEVICE_TYPE_DISK);
  tensorcast::daemon::v1::UnloadReplicaResponse resp;
  grpc::ServerContext ctx;
  auto st = svc.UnloadReplica(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.code() == 0);
}

TEST_CASE("MaterializeReplica rejects while shutting down", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  StoreDaemonServiceImpl svc(engine);
  svc.begin_shutdown();

  tensorcast::daemon::v1::MaterializeReplicaRequest req;
  req.set_disk_path("/tmp/anything");
  req.set_target_device_type(tensorcast::daemon::v1::DeviceType::DEVICE_TYPE_GPU);
  tensorcast::daemon::v1::MaterializeReplicaResponse resp;
  grpc::ServerContext ctx;
  auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::UNAVAILABLE);
  REQUIRE(resp.status() == tensorcast::daemon::v1::MATERIALIZE_REPLICA_STATUS_FAILED);
}

TEST_CASE("MaterializeReplica validates one-of inputs", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  StoreDaemonServiceImpl svc(engine);

  // Both missing -> INVALID_ARGUMENT
  {
    tensorcast::daemon::v1::MaterializeReplicaRequest req;
    req.set_target_device_type(tensorcast::daemon::v1::DeviceType::DEVICE_TYPE_GPU);
    tensorcast::daemon::v1::MaterializeReplicaResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  }

  // Both present -> INVALID_ARGUMENT
  {
    tensorcast::daemon::v1::MaterializeReplicaRequest req;
    req.set_disk_path("/tmp/x");
    req.set_artifact_id("mi2:abc:def");
    req.set_target_device_type(tensorcast::daemon::v1::DeviceType::DEVICE_TYPE_GPU);
    tensorcast::daemon::v1::MaterializeReplicaResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  }
}

TEST_CASE("WaitReplicaVerification unknown returns UNKNOWN", "[daemon][parity]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_basic());
  StoreDaemonServiceImpl svc(engine);

  tensorcast::daemon::v1::WaitReplicaVerificationRequest req;
  req.set_artifact_id("nonexistent");
  req.set_replica_uuid("deadbeef-dead-beef-dead-beefdeadbeef");
  req.set_timeout_ms(10);
  tensorcast::daemon::v1::WaitReplicaVerificationResponse resp;
  grpc::ServerContext ctx;
  auto st = svc.WaitReplicaVerification(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v1::VerificationStatus::VERIFICATION_STATUS_UNSPECIFIED);
}
