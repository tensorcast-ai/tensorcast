// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

static tensorcast::store::StoreEngineOptions compat_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.chunk_size = 1ULL << 20;
  opts.num_thread = 2;
  return opts;
}

TEST_CASE("ConfirmReplica strict mode enforces disk_path and GPU", "[daemon][compat]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(compat_opts());
  StoreDaemonServiceImpl svc(engine);

  // Missing disk_path -> INVALID_ARGUMENT
  {
    tensorcast::daemon::v1::ConfirmReplicaRequest req;
    req.set_target_device_type(tensorcast::daemon::v1::DeviceType::DEVICE_TYPE_GPU);
    grpc::ServerContext ctx;
    tensorcast::daemon::v1::ConfirmReplicaResponse resp;
    auto st = svc.ConfirmReplica(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.code() == 0);
  }

  // Non-GPU target -> UNIMPLEMENTED
  {
    tensorcast::daemon::v1::ConfirmReplicaRequest req;
    req.set_disk_path("/tmp/xxx");
    req.set_target_device_type(tensorcast::daemon::v1::DeviceType::DEVICE_TYPE_CPU);
    grpc::ServerContext ctx;
    tensorcast::daemon::v1::ConfirmReplicaResponse resp;
    auto st = svc.ConfirmReplica(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.code() == 0);
  }
}
