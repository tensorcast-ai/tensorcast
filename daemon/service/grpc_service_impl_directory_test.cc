// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace {

using tensorcast::daemon::v2::ListDirectoryInstancesRequest;
using tensorcast::daemon::v2::ListDirectoryInstancesResponse;
using tensorcast::daemon::v2::ListDirectoryWorkersRequest;
using tensorcast::daemon::v2::ListDirectoryWorkersResponse;
using tensorcast::daemon::v2::ResolveInstanceExecutionRequest;
using tensorcast::daemon::v2::ResolveInstanceExecutionResponse;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::ActiveInstanceInfo;
using tensorcast::store::components::ActiveWorkerInfo;
using tensorcast::store::components::RpcOptions;
using tensorcast::store::testing::GlobalStoreClientStub;

StoreEngineOptions opts_small() {
  StoreEngineOptions opts;
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

class DirectoryClient final : public GlobalStoreClientStub {
 public:
  absl::StatusOr<std::vector<ActiveWorkerInfo>> list_active_workers(bool, uint64_t, const RpcOptions&) override {
    return workers;
  }

  absl::StatusOr<std::vector<ActiveInstanceInfo>> list_active_instances(bool, uint64_t, const RpcOptions&) override {
    return instances;
  }

  std::vector<ActiveWorkerInfo> workers;
  std::vector<ActiveInstanceInfo> instances;
};

TEST_CASE("Directory RPCs expose local-only singleton authority", "[daemon][directory]") {
  auto engine = std::make_shared<StoreEngine>(opts_small());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.daemon_id = "daemon-local";
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  harness->kernel().worker_identity_store().set_registered("worker-local");
  auto& svc = harness->service();

  {
    ListDirectoryWorkersRequest req;
    ListDirectoryWorkersResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.ListDirectoryWorkers(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.authority_mode() == "LOCAL_ONLY");
    REQUIRE(resp.freshness_state() == "current");
    REQUIRE(resp.workers_size() == 1);
    REQUIRE(resp.workers(0).daemon_id() == "daemon-local");
    REQUIRE(resp.workers(0).worker_id() == "worker-local");
  }

  {
    ListDirectoryInstancesRequest req;
    ListDirectoryInstancesResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.ListDirectoryInstances(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.authority_mode() == "LOCAL_ONLY");
    REQUIRE(resp.instances_size() == 0);
  }

  {
    ResolveInstanceExecutionRequest req;
    ResolveInstanceExecutionResponse resp;
    grpc::ServerContext ctx;
    req.set_instance_id("inst-local");
    auto st = svc.ResolveInstanceExecution(&ctx, &req, &resp);
    REQUIRE_FALSE(st.ok());
    REQUIRE(st.error_code() == grpc::StatusCode::NOT_FOUND);
  }
}

TEST_CASE("Directory RPCs expose Global-Store-backed routes", "[daemon][directory]") {
  auto engine = std::make_shared<StoreEngine>(opts_small());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.daemon_id = "daemon-front";

  auto client = std::make_shared<DirectoryClient>();
  client->connected = true;
  client->workers = {
      ActiveWorkerInfo{
          .worker_id = "worker-a",
          .node_id = "node-a",
          .node_address = "10.0.0.1",
          .grpc_port = 50051,
          .p2p_port = 50052,
          .accepting_new_requests = true,
          .daemon_id = "daemon-a",
          .capability_flags = 7,
      },
  };
  client->instances = {
      ActiveInstanceInfo{
          .instance_id = "inst-a",
          .daemon_id = "daemon-a",
          .worker_id = "worker-a",
          .engine = "test",
          .signals_endpoint = "ipc://signals",
          .execution_endpoint = "10.0.0.1:7001",
          .execution_host_kind = "node_agent_grpc",
          .capability_flags = 9,
      },
  };

  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, nullptr, client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  {
    ListDirectoryWorkersRequest req;
    ListDirectoryWorkersResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.ListDirectoryWorkers(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.authority_mode() == "GLOBAL_STORE_BACKED");
    REQUIRE(resp.freshness_state() == "current");
    REQUIRE(resp.workers_size() == 1);
    REQUIRE(resp.workers(0).daemon_id() == "daemon-a");
    REQUIRE(resp.workers(0).daemon_address() == "10.0.0.1:50051");
    REQUIRE(resp.workers(0).capability_flags() == 7);
  }

  {
    ListDirectoryInstancesRequest req;
    ListDirectoryInstancesResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.ListDirectoryInstances(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.authority_mode() == "GLOBAL_STORE_BACKED");
    REQUIRE(resp.instances_size() == 1);
    REQUIRE(resp.instances(0).instance_id() == "inst-a");
    REQUIRE(resp.instances(0).execution_endpoint() == "10.0.0.1:7001");
    REQUIRE(resp.instances(0).execution_host_kind() == "node_agent_grpc");
  }

  {
    ResolveInstanceExecutionRequest req;
    ResolveInstanceExecutionResponse resp;
    grpc::ServerContext ctx;
    req.set_instance_id("inst-a");
    auto st = svc.ResolveInstanceExecution(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.authority_mode() == "GLOBAL_STORE_BACKED");
    REQUIRE(resp.route().instance_id() == "inst-a");
    REQUIRE(resp.route().daemon_id() == "daemon-a");
    REQUIRE(resp.route().execution_endpoint() == "10.0.0.1:7001");
    REQUIRE(resp.route().engine() == "test");
  }
}

} // namespace
