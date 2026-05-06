// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "grpcpp/server_context.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"
#include "tensorcast/global_store/v1/global_store.pb.h"

namespace {

namespace common = tensorcast::common::v1;
namespace daemon = tensorcast::daemon;
namespace daemon_v2 = tensorcast::daemon::v2;
namespace global_store = tensorcast::global_store::v1;
namespace store = tensorcast::store;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env != nullptr && *env != '\0') {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_broadcast_session_test";
}

store::StoreEngineOptions make_opts() {
  store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 0;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

class BroadcastSessionClient final : public store::testing::GlobalStoreClientStub {
 public:
  std::vector<global_store::CreateBroadcastSessionRequest> requests;
  absl::Status rpc_status = absl::OkStatus();
  global_store::Status response_status = global_store::STATUS_OK;
  std::string response_session_id{"session-from-global"};

  absl::StatusOr<global_store::CreateBroadcastSessionResponse> create_broadcast_session(
      const global_store::CreateBroadcastSessionRequest& request,
      const store::components::RpcOptions&) override {
    requests.push_back(request);
    if (!rpc_status.ok()) {
      return rpc_status;
    }
    global_store::CreateBroadcastSessionResponse response;
    response.set_status(response_status);
    response.mutable_session()->set_session_id(response_session_id);
    return response;
  }
};

std::unique_ptr<daemon::DaemonServiceHarness> make_harness(std::shared_ptr<BroadcastSessionClient> client) {
  auto engine = std::make_shared<store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(client);

  daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir() / "daemon";
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or = daemon::DaemonServiceHarness::create(engine, daemon_opts, nullptr, client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

} // namespace

TEST_CASE("CreateBroadcastSession forwards daemon request to Global Store", "[daemon][broadcast]") {
  auto client = std::make_shared<BroadcastSessionClient>();
  auto harness = make_harness(client);

  daemon_v2::CreateBroadcastSessionRequest request;
  request.set_session_id("session-a");
  request.set_artifact_id("artifact-a");
  request.set_requested_view_id("view-a");
  request.set_epoch(7);
  request.set_fanout(2);
  request.add_target_worker_ids("worker-a");
  request.add_target_worker_ids("worker-b");
  request.add_target_daemon_ids("daemon-a");
  request.set_root_replica_id("replica-root");
  request.set_strict_parent(true);
  request.set_max_attempts(3);

  daemon_v2::CreateBroadcastSessionResponse response;
  grpc::ServerContext context;
  const auto status = harness->service().CreateBroadcastSession(&context, &request, &response);

  REQUIRE(status.ok());
  REQUIRE(response.status() == daemon_v2::BROADCAST_SESSION_STATUS_OK);
  REQUIRE(response.session_id() == "session-from-global");
  REQUIRE(client->requests.size() == 1);

  const auto& global_request = client->requests.front();
  REQUIRE(global_request.session_id() == "session-a");
  REQUIRE(global_request.artifact_id() == "artifact-a");
  REQUIRE(global_request.requested_byte_space().kind() == common::BYTE_SPACE_KIND_VIEW);
  REQUIRE(global_request.requested_byte_space().id() == "view-a");
  REQUIRE(global_request.epoch() == 7);
  REQUIRE(global_request.fanout() == 2);
  REQUIRE(global_request.root_replica_id() == "replica-root");
  REQUIRE(global_request.strict_parent());
  REQUIRE(global_request.max_attempts() == 3);
  REQUIRE(global_request.targets_size() == 3);
  REQUIRE(global_request.targets(0).worker_id() == "worker-a");
  REQUIRE(global_request.targets(1).worker_id() == "worker-b");
  REQUIRE(global_request.targets(2).daemon_id() == "daemon-a");
}

TEST_CASE("CreateBroadcastSession reports daemon status when Global Store is unavailable", "[daemon][broadcast]") {
  auto client = std::make_shared<BroadcastSessionClient>();
  client->connected = false;
  auto harness = make_harness(client);

  daemon_v2::CreateBroadcastSessionRequest request;
  request.set_session_id("session-a");
  request.set_artifact_id("artifact-a");
  request.set_fanout(2);
  request.set_max_attempts(3);
  request.add_target_daemon_ids("daemon-a");

  daemon_v2::CreateBroadcastSessionResponse response;
  grpc::ServerContext context;
  const auto status = harness->service().CreateBroadcastSession(&context, &request, &response);

  REQUIRE(status.ok());
  REQUIRE(response.status() == daemon_v2::BROADCAST_SESSION_STATUS_ERROR);
  REQUIRE(client->requests.empty());
}
