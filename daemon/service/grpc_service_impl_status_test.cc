// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

static tensorcast::store::StoreEngineOptions opts_small() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

TEST_CASE("Status RPCs reflect worker registration", "[daemon][status]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(opts_small());
  tensorcast::daemon::DaemonOptions daemon_opts;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();
  harness->kernel().worker_identity_store().set_registered("worker-1");

  {
    tensorcast::daemon::v2::GetWorkerStatusRequest req;
    tensorcast::daemon::v2::GetWorkerStatusResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.GetWorkerStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.is_registered());
    REQUIRE(resp.worker_id() == "worker-1");
    REQUIRE(resp.mem_pool_total_size() == engine->get_mem_pool_size());
  }

  {
    tensorcast::daemon::v2::GetDetailedStatusRequest req;
    tensorcast::daemon::v2::GetDetailedStatusResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.GetDetailedStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.is_registered());
    REQUIRE(resp.worker_id() == "worker-1");
    // With no replicas and default settings, comm enabled is false
    REQUIRE_FALSE(resp.communication_info().enabled());
  }
}
