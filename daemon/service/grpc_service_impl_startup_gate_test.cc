// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/app/startup_coordinator.h"
#include "daemon/testing/daemon_service_harness.h"

#include <memory>

#include <catch2/catch_test_macros.hpp>

#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace {

tensorcast::store::StoreEngineOptions opts_small() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

} // namespace

TEST_CASE("startup gate blocks data-plane RPCs but allows status RPCs", "[daemon][startup_gate]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(opts_small());
  tensorcast::daemon::DaemonOptions daemon_opts;
  auto startup = std::make_shared<tensorcast::daemon::StartupCoordinator>();
  startup->begin_startup("daemon startup still in progress");
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, nullptr, nullptr, startup);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  {
    tensorcast::daemon::v2::GetServerConfigRequest req;
    tensorcast::daemon::v2::GetServerConfigResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.GetServerConfig(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.startup_phase() == tensorcast::daemon::v2::DAEMON_STARTUP_PHASE_LISTENING);
    REQUIRE(resp.source_bound_contract_version() == 3);
  }

  {
    tensorcast::daemon::v2::GetDetailedStatusRequest req;
    tensorcast::daemon::v2::GetDetailedStatusResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.GetDetailedStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
  }

  {
    tensorcast::daemon::v2::MaterializeReplicaRequest req;
    tensorcast::daemon::v2::MaterializeReplicaResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.error_code() == grpc::StatusCode::UNAVAILABLE);
  }
}
