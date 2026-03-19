// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/instance_execution_directory_cache.h"

#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/store/testing/global_store_client_stub.h"

namespace {

using tensorcast::daemon::InstanceExecutionDirectoryCache;
using tensorcast::store::components::ActiveInstanceInfo;
using tensorcast::store::components::RpcOptions;
using tensorcast::store::testing::GlobalStoreClientStub;

class DirectoryClient final : public GlobalStoreClientStub {
 public:
  absl::StatusOr<std::vector<ActiveInstanceInfo>> list_active_instances(bool, uint64_t, const RpcOptions&) override {
    return instances;
  }

  std::vector<ActiveInstanceInfo> instances;
};

TEST_CASE(
    "InstanceExecutionDirectoryCache drops stale entries after refresh miss",
    "[daemon][instance_execution_directory_cache]") {
  auto client = std::make_shared<DirectoryClient>();
  client->connected = true;

  InstanceExecutionDirectoryCache cache(client);
  const absl::Time now = absl::UnixEpoch() + absl::Seconds(1);
  const absl::Duration staleness_budget = absl::Seconds(1);

  client->instances = {
      ActiveInstanceInfo{
          .instance_id = "inst-a",
          .daemon_id = "daemon-a",
          .execution_endpoint = "10.0.0.1:7001",
          .execution_host_kind = "node_agent_grpc",
      },
  };

  auto initial_route_or = cache.resolve_instance_execution("inst-a", now, staleness_budget);
  REQUIRE(initial_route_or.ok());
  REQUIRE(initial_route_or->execution_endpoint == "10.0.0.1:7001");

  client->instances.clear();

  auto stale_route_or = cache.resolve_instance_execution("inst-a", now + absl::Seconds(2), staleness_budget);
  REQUIRE_FALSE(stale_route_or.ok());
  REQUIRE(absl::IsNotFound(stale_route_or.status()));
}

TEST_CASE(
    "InstanceExecutionDirectoryCache refreshes changed execution endpoints",
    "[daemon][instance_execution_directory_cache]") {
  auto client = std::make_shared<DirectoryClient>();
  client->connected = true;

  InstanceExecutionDirectoryCache cache(client);
  const absl::Time now = absl::UnixEpoch() + absl::Seconds(1);
  const absl::Duration staleness_budget = absl::Seconds(1);

  client->instances = {
      ActiveInstanceInfo{
          .instance_id = "inst-a",
          .daemon_id = "daemon-a",
          .execution_endpoint = "10.0.0.1:7001",
          .execution_host_kind = "node_agent_grpc",
      },
  };

  auto initial_route_or = cache.resolve_instance_execution("inst-a", now, staleness_budget);
  REQUIRE(initial_route_or.ok());
  REQUIRE(initial_route_or->execution_endpoint == "10.0.0.1:7001");

  client->instances = {
      ActiveInstanceInfo{
          .instance_id = "inst-a",
          .daemon_id = "daemon-a",
          .execution_endpoint = "10.0.0.2:7002",
      },
  };

  auto refreshed_route_or = cache.resolve_instance_execution("inst-a", now + absl::Seconds(2), staleness_budget);
  REQUIRE(refreshed_route_or.ok());
  REQUIRE(refreshed_route_or->execution_endpoint == "10.0.0.2:7002");
  REQUIRE(refreshed_route_or->execution_host_kind == "node_agent_grpc");
}

} // namespace
