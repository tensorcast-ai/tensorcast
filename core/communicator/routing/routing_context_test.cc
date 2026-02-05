// Copyright (c) 2026, TensorCast Team.

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/common/async_runtime.h"
#include "core/communicator/routing/adapter.h"
#include "core/communicator/routing/connection.h"
#include "core/communicator/routing/routing_context.h"
#include "core/communicator/topology/topology.h"

namespace {

using tensorcast::communicator::routing::Connection;
using tensorcast::communicator::routing::ConnectionAdapter;
using tensorcast::communicator::routing::ConnectionKey;
using tensorcast::communicator::routing::ConnectionType;
using tensorcast::communicator::routing::EndpointBinding;
using tensorcast::communicator::routing::HealthState;
using tensorcast::communicator::routing::LinkState;
using tensorcast::communicator::routing::ReadRequest;
using tensorcast::communicator::routing::RoutingContext;
using tensorcast::communicator::topology::Endpoint;
using tensorcast::communicator::topology::EndpointKind;
using tensorcast::communicator::topology::EndpointType;
using tensorcast::communicator::topology::Link;
using tensorcast::communicator::topology::LinkType;
using tensorcast::communicator::topology::Pool;
using tensorcast::communicator::topology::PoolType;
using tensorcast::communicator::topology::Topology;

class FakeAdapter final : public ConnectionAdapter {
 public:
  explicit FakeAdapter(absl::Status status)
      : status_(std::move(status)) {}

  tensorcast::communicator::routing::ConnectionProtocol protocol() const override {
    return tensorcast::communicator::routing::ConnectionProtocol::kAuto;
  }

  bool is_available() const override {
    return true;
  }

  tensorcast::communicator::transport::future_read_result_t read_tensor(
      const ReadRequest&,
      const EndpointBinding&,
      const EndpointBinding&) override {
    std::promise<tensorcast::communicator::transport::read_result_t> promise;
    auto future = promise.get_future();
    tensorcast::communicator::transport::read_result_t result;
    result.status = status_;
    promise.set_value(std::move(result));
    return future;
  }

  absl::Status close(const EndpointBinding&) override {
    return absl::OkStatus();
  }

 private:
  absl::Status status_;
};

Topology build_minimal_topology() {
  std::vector<Pool> pools;
  pools.push_back(Pool{"cpu0", "cpu0", PoolType::kCpu});
  pools.push_back(Pool{"gpu0", "gpu0", PoolType::kGpu});

  std::vector<Endpoint> endpoints;
  Endpoint nic0;
  nic0.id = "nic0";
  nic0.name = "nic0";
  nic0.kind = EndpointKind::kClient;
  nic0.type = EndpointType::kNic;
  nic0.pool_ids = {"cpu0", "gpu0"};
  endpoints.push_back(nic0);

  Endpoint nic1;
  nic1.id = "nic1";
  nic1.name = "nic1";
  nic1.kind = EndpointKind::kClient;
  nic1.type = EndpointType::kNic;
  nic1.pool_ids = {"cpu0", "gpu0"};
  endpoints.push_back(nic1);

  std::vector<Link> links;
  Link link;
  link.id = "nic0_to_nic1";
  link.name = link.id;
  link.type = LinkType::kP2P;
  link.src_endpoint_id = "nic0";
  link.dst_endpoint_id = "nic1";
  links.push_back(link);

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = true});
  REQUIRE(topology_or.ok());
  return std::move(topology_or).value();
}

} // namespace

TEST_CASE("Connection records success and failure", "[communicator][routing]") {
  Link link;
  link.id = "link0";
  link.src_endpoint_id = "ep0";
  link.dst_endpoint_id = "ep1";

  auto link_state = std::make_shared<LinkState>(link.id);
  EndpointBinding local{"ep0", "node0", "127.0.0.1", 1001};
  EndpointBinding remote{"ep1", "node0", "127.0.0.1", 1002};
  auto runtime = std::make_shared<tensorcast::common::AsyncRuntime>();

  auto ok_adapter = std::make_shared<FakeAdapter>(absl::OkStatus());
  auto ok_connection = std::make_shared<Connection>(
      ConnectionKey{"ep0", "ep1"},
      ConnectionType::kP2P,
      nullptr,
      &link,
      local,
      remote,
      ok_adapter,
      link_state,
      runtime);

  ReadRequest request;
  request.tensor_key = "tensor";
  ok_connection->read_tensor(request).get();
  auto ok_stats = ok_connection->snapshot();
  CHECK(ok_stats.success_count == 1);
  CHECK(ok_connection->health() == HealthState::kHealthy);

  auto fail_adapter = std::make_shared<FakeAdapter>(absl::InternalError("fail"));
  auto fail_connection = std::make_shared<Connection>(
      ConnectionKey{"ep0", "ep1"},
      ConnectionType::kP2P,
      nullptr,
      &link,
      local,
      remote,
      fail_adapter,
      link_state,
      runtime);
  fail_connection->read_tensor(request).get();
  auto fail_stats = fail_connection->snapshot();
  CHECK(fail_stats.failure_count == 1);
  CHECK(fail_connection->health() == HealthState::kUnhealthy);
}

TEST_CASE("RoutingContext caches communicators and builds direct channels", "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(
      RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_minimal_topology()).ok());

  std::vector<EndpointBinding> bindings;
  bindings.push_back(EndpointBinding{"nic0", "node0", "127.0.0.1", 1234});
  bindings.push_back(EndpointBinding{"nic1", "node0", "127.0.0.1", 1235});
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  auto comm1_or = context->get_communicator("nic0", "nic1");
  REQUIRE(comm1_or.ok());
  auto comm2_or = context->get_communicator("nic0", "nic1");
  REQUIRE(comm2_or.ok());
  CHECK(comm1_or.value() == comm2_or.value());

  auto channel_or = comm1_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  CHECK(channel_or.value()->hop_count() == 1);
  CHECK(channel_or.value()->src_endpoint_id() == "nic0");
  CHECK(channel_or.value()->dst_endpoint_id() == "nic1");
}
