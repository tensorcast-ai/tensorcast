// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/worker_directory_cache.h"

#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "core/communicator/misc/utils.h"
#include "core/store/testing/global_store_client_stub.h"

namespace {

using tensorcast::daemon::WorkerDirectoryCache;
using tensorcast::store::components::ActiveWorkerInfo;
using tensorcast::store::components::RpcOptions;
using tensorcast::store::testing::GlobalStoreClientStub;

class DirectoryClient final : public GlobalStoreClientStub {
 public:
  absl::StatusOr<std::vector<ActiveWorkerInfo>> list_active_workers(bool, uint64_t, const RpcOptions&) override {
    return workers;
  }

  std::vector<ActiveWorkerInfo> workers;
};

TEST_CASE("WorkerDirectoryCache drops stale daemon entries after refresh miss", "[daemon][worker_directory_cache]") {
  auto client = std::make_shared<DirectoryClient>();
  client->connected = true;

  WorkerDirectoryCache cache(client);
  const absl::Time now = absl::UnixEpoch() + absl::Seconds(1);
  const absl::Duration staleness_budget = absl::Seconds(1);

  client->workers = {
      ActiveWorkerInfo{
          .daemon_id = "daemon-a",
          .node_address = "10.0.0.1",
          .grpc_port = 50051,
      },
  };

  auto initial_address_or = cache.resolve_daemon_address("daemon-a", now, staleness_budget);
  REQUIRE(initial_address_or.ok());
  REQUIRE(*initial_address_or == "10.0.0.1:50051");

  client->workers.clear();

  auto stale_address_or = cache.resolve_daemon_address("daemon-a", now + absl::Seconds(2), staleness_budget);
  REQUIRE_FALSE(stale_address_or.ok());
  REQUIRE(absl::IsNotFound(stale_address_or.status()));
}

TEST_CASE("WorkerDirectoryCache refreshes changed daemon endpoints", "[daemon][worker_directory_cache]") {
  auto client = std::make_shared<DirectoryClient>();
  client->connected = true;

  WorkerDirectoryCache cache(client);
  const absl::Time now = absl::UnixEpoch() + absl::Seconds(1);
  const absl::Duration staleness_budget = absl::Seconds(1);

  client->workers = {
      ActiveWorkerInfo{
          .daemon_id = "daemon-a",
          .node_address = "10.0.0.1",
          .grpc_port = 50051,
      },
  };

  auto initial_address_or = cache.resolve_daemon_address("daemon-a", now, staleness_budget);
  REQUIRE(initial_address_or.ok());
  REQUIRE(*initial_address_or == "10.0.0.1:50051");

  client->workers = {
      ActiveWorkerInfo{
          .daemon_id = "daemon-a",
          .node_address = "10.0.0.2",
          .grpc_port = 50052,
      },
  };

  auto refreshed_address_or = cache.resolve_daemon_address("daemon-a", now + absl::Seconds(2), staleness_budget);
  REQUIRE(refreshed_address_or.ok());
  REQUIRE(*refreshed_address_or == "10.0.0.2:50052");
}

TEST_CASE(
    "WorkerDirectoryCache canonicalizes same-host daemon endpoints to loopback",
    "[daemon][worker_directory_cache]") {
  const std::string local_default_ip = tensorcast::communicator::misc::get_default_ip();
  if (local_default_ip.empty()) {
    SUCCEED("no default IP available in test environment");
    return;
  }

  auto client = std::make_shared<DirectoryClient>();
  client->connected = true;

  WorkerDirectoryCache cache(client);
  const absl::Time now = absl::UnixEpoch() + absl::Seconds(1);
  const absl::Duration staleness_budget = absl::Seconds(1);

  client->workers = {
      ActiveWorkerInfo{
          .daemon_id = "daemon-a",
          .node_address = local_default_ip,
          .grpc_port = 50051,
      },
  };

  auto address_or = cache.resolve_daemon_address("daemon-a", now, staleness_budget);
  REQUIRE(address_or.ok());
  REQUIRE(*address_or == "127.0.0.1:50051");

  auto entry_or = cache.resolve_daemon_entry("daemon-a", now, staleness_budget);
  REQUIRE(entry_or.ok());
  REQUIRE(entry_or->address == "127.0.0.1:50051");
  REQUIRE(entry_or->node_address == local_default_ip);
}

TEST_CASE(
    "WorkerDirectoryCache resolves local daemon entry without refreshing global store",
    "[daemon][worker_directory_cache]") {
  auto client = std::make_shared<DirectoryClient>();
  client->connected = false;

  WorkerDirectoryCache cache(client);
  cache.update_local_entry(
      WorkerDirectoryCache::Entry{
          .daemon_id = "daemon-local",
          .worker_id = "worker-local",
          .node_id = "node-local",
          .node_address = "10.0.0.9",
          .grpc_port = 50051,
          .p2p_port = 60061,
          .address = "10.0.0.9:50051",
          .capability_flags = 7,
      });

  const absl::Time now = absl::UnixEpoch() + absl::Seconds(10);
  const absl::Duration staleness_budget = absl::Seconds(1);

  REQUIRE(cache.is_fresh("daemon-local", now, staleness_budget));
  REQUIRE(cache.warm_for_daemons({"daemon-local"}, now, staleness_budget).ok());

  auto entry_or = cache.resolve_daemon_entry("daemon-local", now, staleness_budget);
  REQUIRE(entry_or.ok());
  REQUIRE(entry_or->daemon_id == "daemon-local");
  REQUIRE(entry_or->worker_id == "worker-local");
  REQUIRE(entry_or->node_id == "node-local");
  REQUIRE(entry_or->grpc_port == 50051);
  REQUIRE(entry_or->p2p_port == 60061);
}

} // namespace
