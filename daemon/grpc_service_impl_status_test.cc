// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "proto/store_daemon.grpc.pb.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

static tensorcast::store::StoreEngineOptions opts_small() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

TEST_CASE("Status RPCs reflect worker registration", "[daemon][status]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(opts_small());
  StoreDaemonServiceImpl svc(engine);

  svc.set_worker_registered("worker-1");

  {
    ::store_daemon::GetWorkerStatusRequest req;
    ::store_daemon::GetWorkerStatusResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.GetWorkerStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.is_registered());
    REQUIRE(resp.worker_id() == "worker-1");
    REQUIRE(resp.mem_pool_total_size() == engine->get_mem_pool_size());
  }

  {
    ::store_daemon::GetDetailedStatusRequest req;
    ::store_daemon::GetDetailedStatusResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.GetDetailedStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.is_registered());
    REQUIRE(resp.worker_id() == "worker-1");
    // With no replicas and default compat, comm enabled is false
    REQUIRE_FALSE(resp.communication_info().enabled());
  }
}
