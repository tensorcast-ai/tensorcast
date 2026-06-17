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
using tensorcast::communicator::routing::ConnectionProtocol;
using tensorcast::communicator::routing::ConnectionType;
using tensorcast::communicator::routing::EndpointBinding;
using tensorcast::communicator::routing::HealthState;
using tensorcast::communicator::routing::LinkState;
using tensorcast::communicator::routing::LocalRegion;
using tensorcast::communicator::routing::ReadPlan;
using tensorcast::communicator::routing::ReadPlanSlice;
using tensorcast::communicator::routing::ReadRequest;
using tensorcast::communicator::routing::ReadRouteContext;
using tensorcast::communicator::routing::RoutingContext;
using tensorcast::communicator::routing::SourceSlice;
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
  explicit FakeAdapter(absl::Status status) : status_(std::move(status)) {}

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

  tensorcast::communicator::transport::future_read_result_t read_plan(
      const ReadPlan&,
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

class ProtocolAdapterStub final : public ConnectionAdapter {
 public:
  ProtocolAdapterStub(ConnectionProtocol protocol, bool available) : protocol_(protocol), available_(available) {}

  ConnectionProtocol protocol() const override {
    return protocol_;
  }

  bool is_available() const override {
    return available_;
  }

  tensorcast::communicator::transport::future_read_result_t read_tensor(
      const ReadRequest&,
      const EndpointBinding&,
      const EndpointBinding&) override {
    std::promise<tensorcast::communicator::transport::read_result_t> promise;
    auto future = promise.get_future();
    tensorcast::communicator::transport::read_result_t result;
    result.status = absl::UnimplementedError("protocol adapter stub has no datapath");
    promise.set_value(std::move(result));
    return future;
  }

  tensorcast::communicator::transport::future_read_result_t read_plan(
      const ReadPlan& plan,
      const EndpointBinding&,
      const EndpointBinding&) override {
    read_plan_calls_ += 1;
    last_plan_ = plan;
    std::promise<tensorcast::communicator::transport::read_result_t> promise;
    auto future = promise.get_future();
    tensorcast::communicator::transport::read_result_t result;
    result.status = absl::OkStatus();
    promise.set_value(std::move(result));
    return future;
  }

  absl::Status close(const EndpointBinding&) override {
    return absl::OkStatus();
  }

  int read_plan_calls() const {
    return read_plan_calls_;
  }

  const ReadPlan& last_plan() const {
    return last_plan_;
  }

 private:
  ConnectionProtocol protocol_;
  bool available_ = false;
  int read_plan_calls_ = 0;
  ReadPlan last_plan_;
};

ReadPlan make_single_slice_read_plan(
    const std::string& local_endpoint_id,
    const std::string& remote_endpoint_id,
    ConnectionProtocol protocol) {
  ReadPlan plan;
  plan.local_regions = {
      LocalRegion{
          .addr = 0x1000,
          .bytes = 256,
      },
  };
  plan.source_slices = {
      SourceSlice{
          .authority_id = "authority-a",
          .route =
              ReadRouteContext{
                  .local_endpoint_id = local_endpoint_id,
                  .remote_endpoint_id = remote_endpoint_id,
                  .protocol = protocol,
                  .rail_id = -1,
              },
          .tensor_key = "tensor-a",
          .remote_offset = 64,
          .bytes = 128,
      },
  };
  plan.slices = {
      ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .local_region_offset = 32,
          .bytes = 128,
      },
  };
  return plan;
}

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

Topology build_local_fabric_topology() {
  std::vector<Pool> pools;
  pools.push_back(Pool{"cpu0", "cpu0", PoolType::kCpu});
  pools.push_back(Pool{"gpu0", "gpu0", PoolType::kGpu});
  pools.push_back(Pool{"gpu1", "gpu1", PoolType::kGpu});

  std::vector<Endpoint> endpoints;
  Endpoint nvlink0;
  nvlink0.id = "nvlink0";
  nvlink0.name = "nvlink0";
  nvlink0.kind = EndpointKind::kClient;
  nvlink0.type = EndpointType::kNvlink;
  nvlink0.pool_ids = {"gpu0"};
  endpoints.push_back(nvlink0);

  Endpoint nvlink1;
  nvlink1.id = "nvlink1";
  nvlink1.name = "nvlink1";
  nvlink1.kind = EndpointKind::kClient;
  nvlink1.type = EndpointType::kNvlink;
  nvlink1.pool_ids = {"gpu1"};
  endpoints.push_back(nvlink1);

  Endpoint pcie0;
  pcie0.id = "pcie0";
  pcie0.name = "pcie0";
  pcie0.kind = EndpointKind::kClient;
  pcie0.type = EndpointType::kPcie;
  pcie0.pool_ids = {"cpu0", "gpu0"};
  endpoints.push_back(pcie0);

  Endpoint pcie1;
  pcie1.id = "pcie1";
  pcie1.name = "pcie1";
  pcie1.kind = EndpointKind::kClient;
  pcie1.type = EndpointType::kPcie;
  pcie1.pool_ids = {"cpu0", "gpu1"};
  endpoints.push_back(pcie1);

  std::vector<Link> links;
  Link nvlink_link;
  nvlink_link.id = "nvlink0_to_nvlink1";
  nvlink_link.name = nvlink_link.id;
  nvlink_link.type = LinkType::kP2P;
  nvlink_link.src_endpoint_id = "nvlink0";
  nvlink_link.dst_endpoint_id = "nvlink1";
  links.push_back(nvlink_link);

  Link pcie_link;
  pcie_link.id = "pcie0_to_pcie1";
  pcie_link.name = pcie_link.id;
  pcie_link.type = LinkType::kP2P;
  pcie_link.src_endpoint_id = "pcie0";
  pcie_link.dst_endpoint_id = "pcie1";
  links.push_back(pcie_link);

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = false});
  REQUIRE(topology_or.ok());
  return std::move(topology_or).value();
}

Topology build_rail_switch_topology() {
  std::vector<Pool> pools;
  pools.push_back(Pool{"cpu0", "cpu0", PoolType::kCpu});
  pools.push_back(Pool{"gpu0", "gpu0", PoolType::kGpu});
  pools.push_back(Pool{"gpu1", "gpu1", PoolType::kGpu});

  std::vector<Endpoint> endpoints;
  Endpoint nic0;
  nic0.id = "nic_0";
  nic0.name = "nic_0";
  nic0.kind = EndpointKind::kClient;
  nic0.type = EndpointType::kNic;
  nic0.pool_ids = {"cpu0", "gpu0"};
  endpoints.push_back(nic0);

  Endpoint nic1;
  nic1.id = "nic_1";
  nic1.name = "nic_1";
  nic1.kind = EndpointKind::kClient;
  nic1.type = EndpointType::kNic;
  nic1.pool_ids = {"cpu0", "gpu1"};
  endpoints.push_back(nic1);

  Endpoint rail0;
  rail0.id = "netsw_rail_0";
  rail0.name = rail0.id;
  rail0.kind = EndpointKind::kSwitch;
  rail0.type = EndpointType::kNic;
  endpoints.push_back(rail0);

  Endpoint rail1;
  rail1.id = "netsw_rail_1";
  rail1.name = rail1.id;
  rail1.kind = EndpointKind::kSwitch;
  rail1.type = EndpointType::kNic;
  endpoints.push_back(rail1);

  std::vector<Link> links;
  Link nic0_to_rail0;
  nic0_to_rail0.id = "nic_0_to_netsw_rail_0";
  nic0_to_rail0.name = nic0_to_rail0.id;
  nic0_to_rail0.type = LinkType::kSwitch;
  nic0_to_rail0.src_endpoint_id = "nic_0";
  nic0_to_rail0.dst_endpoint_id = "netsw_rail_0";
  links.push_back(nic0_to_rail0);

  Link nic1_to_rail1;
  nic1_to_rail1.id = "nic_1_to_netsw_rail_1";
  nic1_to_rail1.name = nic1_to_rail1.id;
  nic1_to_rail1.type = LinkType::kSwitch;
  nic1_to_rail1.src_endpoint_id = "nic_1";
  nic1_to_rail1.dst_endpoint_id = "netsw_rail_1";
  links.push_back(nic1_to_rail1);

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = false});
  REQUIRE(topology_or.ok());
  return std::move(topology_or).value();
}

Topology build_eight_rail_switch_topology() {
  std::vector<Pool> pools;
  pools.push_back(Pool{"cpu0", "cpu0", PoolType::kCpu});
  pools.push_back(Pool{"cpu1", "cpu1", PoolType::kCpu});
  for (int gpu_id = 0; gpu_id < 8; ++gpu_id) {
    pools.push_back(
        Pool{
            "gpu" + std::to_string(gpu_id),
            "gpu" + std::to_string(gpu_id),
            PoolType::kGpu,
        });
  }

  std::vector<Endpoint> endpoints;
  for (int rail_id = 0; rail_id < 8; ++rail_id) {
    Endpoint nic;
    nic.id = "nic_" + std::to_string(rail_id);
    nic.name = nic.id;
    nic.kind = EndpointKind::kClient;
    nic.type = EndpointType::kNic;
    nic.pool_ids = {
        rail_id < 4 ? "cpu0" : "cpu1",
        "gpu" + std::to_string(rail_id),
    };
    endpoints.push_back(std::move(nic));

    Endpoint rail_switch;
    rail_switch.id = "netsw_rail_" + std::to_string(rail_id);
    rail_switch.name = rail_switch.id;
    rail_switch.kind = EndpointKind::kSwitch;
    rail_switch.type = EndpointType::kNic;
    endpoints.push_back(std::move(rail_switch));
  }

  std::vector<Link> links;
  for (int rail_id = 0; rail_id < 8; ++rail_id) {
    Link link;
    link.id = "nic_" + std::to_string(rail_id) + "_to_netsw_rail_" + std::to_string(rail_id);
    link.name = link.id;
    link.type = LinkType::kSwitch;
    link.src_endpoint_id = "nic_" + std::to_string(rail_id);
    link.dst_endpoint_id = "netsw_rail_" + std::to_string(rail_id);
    links.push_back(std::move(link));
  }

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = false});
  REQUIRE(topology_or.ok());
  return std::move(topology_or).value();
}

Topology build_two_node_transfer_and_fanout_topology() {
  std::vector<Pool> pools;
  pools.push_back(Pool{"cpu0", "cpu0", PoolType::kCpu});
  pools.push_back(Pool{"cpu1", "cpu1", PoolType::kCpu});
  for (int gpu_id = 0; gpu_id < 8; ++gpu_id) {
    pools.push_back(
        Pool{
            "gpu" + std::to_string(gpu_id),
            "gpu" + std::to_string(gpu_id),
            PoolType::kGpu,
        });
  }

  std::vector<Endpoint> endpoints;
  for (int rail_id = 0; rail_id < 8; ++rail_id) {
    Endpoint nic;
    nic.id = "nic_" + std::to_string(rail_id);
    nic.name = nic.id;
    nic.kind = EndpointKind::kClient;
    nic.type = EndpointType::kNic;
    nic.pool_ids = {
        rail_id < 4 ? "cpu0" : "cpu1",
        "gpu" + std::to_string(rail_id),
    };
    endpoints.push_back(std::move(nic));

    Endpoint rail_switch;
    rail_switch.id = "netsw_rail_" + std::to_string(rail_id);
    rail_switch.name = rail_switch.id;
    rail_switch.kind = EndpointKind::kSwitch;
    rail_switch.type = EndpointType::kNic;
    endpoints.push_back(std::move(rail_switch));
  }

  for (int gpu_id = 0; gpu_id < 8; ++gpu_id) {
    Endpoint node_gpu;
    node_gpu.id = "node1/dev/gpu/" + std::to_string(gpu_id);
    node_gpu.name = node_gpu.id;
    node_gpu.kind = EndpointKind::kClient;
    node_gpu.type = EndpointType::kPcie;
    node_gpu.pool_ids = {
        "cpu0",
        "gpu" + std::to_string(gpu_id),
    };
    endpoints.push_back(std::move(node_gpu));
  }

  Endpoint node_cpu;
  node_cpu.id = "node1/dev/cpu/0";
  node_cpu.name = node_cpu.id;
  node_cpu.kind = EndpointKind::kClient;
  node_cpu.type = EndpointType::kPcie;
  node_cpu.pool_ids = {"cpu0", "gpu0"};
  endpoints.push_back(std::move(node_cpu));

  std::vector<Link> links;
  for (int rail_id = 0; rail_id < 8; ++rail_id) {
    Link link;
    link.id = "nic_" + std::to_string(rail_id) + "_to_netsw_rail_" + std::to_string(rail_id);
    link.name = link.id;
    link.type = LinkType::kSwitch;
    link.src_endpoint_id = "nic_" + std::to_string(rail_id);
    link.dst_endpoint_id = "netsw_rail_" + std::to_string(rail_id);
    links.push_back(std::move(link));
  }

  for (int gpu_id = 1; gpu_id < 8; ++gpu_id) {
    Link fanout_link;
    fanout_link.id = "node1_gpu0_to_gpu" + std::to_string(gpu_id);
    fanout_link.name = fanout_link.id;
    fanout_link.type = LinkType::kP2P;
    fanout_link.src_endpoint_id = "node1/dev/gpu/0";
    fanout_link.dst_endpoint_id = "node1/dev/gpu/" + std::to_string(gpu_id);
    links.push_back(std::move(fanout_link));
  }

  Link fanout_mem_link;
  fanout_mem_link.id = "node1_gpu0_to_cpu0";
  fanout_mem_link.name = fanout_mem_link.id;
  fanout_mem_link.type = LinkType::kP2P;
  fanout_mem_link.src_endpoint_id = "node1/dev/gpu/0";
  fanout_mem_link.dst_endpoint_id = "node1/dev/cpu/0";
  links.push_back(std::move(fanout_mem_link));

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = false});
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
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
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
  REQUIRE(channel_or.value()->hops().size() == 1);
  CHECK(channel_or.value()->hops().front()->type() == ConnectionType::kP2P);
  CHECK(channel_or.value()->hops().front()->remote_binding().endpoint_id == "nic1");
  REQUIRE(channel_or.value()->hops().front()->link() != nullptr);
  CHECK(channel_or.value()->hops().front()->link()->id == "nic0_to_nic1");
}

TEST_CASE("RoutingContext rejects topology and binding mutation", "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_minimal_topology()).ok());
  auto second_topology = context->set_topology(build_minimal_topology());
  CHECK(second_topology.code() == absl::StatusCode::kFailedPrecondition);

  std::vector<EndpointBinding> bindings;
  bindings.push_back(EndpointBinding{"nic0", "node0", "127.0.0.1", 1234});
  bindings.push_back(EndpointBinding{"nic1", "node0", "127.0.0.1", 1235});
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  std::vector<EndpointBinding> second_bindings;
  second_bindings.push_back(EndpointBinding{"nic0", "node0", "127.0.0.1", 2001});
  second_bindings.push_back(EndpointBinding{"nic1", "node0", "127.0.0.1", 2002});
  auto bindings_status = context->set_endpoint_bindings(std::move(second_bindings));
  CHECK(bindings_status.code() == absl::StatusCode::kFailedPrecondition);

  EndpointBinding update_binding{"nic0", "node0", "127.0.0.1", 3000};
  auto update_status = context->update_endpoint_binding(std::move(update_binding));
  CHECK(update_status.code() == absl::StatusCode::kFailedPrecondition);
}

TEST_CASE(
    "RoutingContext prefers NVLINK protocol for same-node NVLINK endpoints when adapter is available",
    "[communicator][routing]") {
  auto nvlink_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kNvlink, true);
  auto pcie_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kPcie, true);

  auto context = std::make_shared<RoutingContext>(
      RoutingContext::Options{},
      /*engine=*/nullptr,
      std::move(nvlink_adapter),
      std::move(pcie_adapter));
  REQUIRE(context->set_topology(build_local_fabric_topology()).ok());
  REQUIRE(context
              ->set_endpoint_bindings({
                  EndpointBinding{.endpoint_id = "nvlink0", .node_id = "node0"},
                  EndpointBinding{.endpoint_id = "nvlink1", .node_id = "node0"},
              })
              .ok());

  auto communicator_or = context->get_communicator("nvlink0", "nvlink1");
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);
  CHECK(channel_or.value()->hops().front()->protocol() == ConnectionProtocol::kNvlink);
}

TEST_CASE("RoutingContext communicator forwards read_plan to the selected adapter", "[communicator][routing]") {
  auto nvlink_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kNvlink, true);
  auto pcie_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kPcie, true);

  auto context = std::make_shared<RoutingContext>(
      RoutingContext::Options{},
      /*engine=*/nullptr,
      nvlink_adapter,
      pcie_adapter);
  REQUIRE(context->set_topology(build_local_fabric_topology()).ok());
  REQUIRE(context
              ->set_endpoint_bindings({
                  EndpointBinding{.endpoint_id = "nvlink0", .node_id = "node0"},
                  EndpointBinding{.endpoint_id = "nvlink1", .node_id = "node0"},
              })
              .ok());

  auto communicator_or = context->get_communicator("nvlink0", "nvlink1");
  REQUIRE(communicator_or.ok());

  const ReadPlan plan = make_single_slice_read_plan("nvlink0", "nvlink1", ConnectionProtocol::kNvlink);
  auto result = communicator_or.value()->read_plan(plan).get();
  REQUIRE(result.status.ok());
  CHECK(nvlink_adapter->read_plan_calls() == 1);
  CHECK(nvlink_adapter->last_plan().source_slices.size() == 1);
  CHECK(nvlink_adapter->last_plan().source_slices.front().tensor_key == "tensor-a");
}

TEST_CASE("RoutingContext communicator rejects read_plan route mismatch before issue", "[communicator][routing]") {
  auto nvlink_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kNvlink, true);
  auto pcie_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kPcie, true);

  auto context = std::make_shared<RoutingContext>(
      RoutingContext::Options{},
      /*engine=*/nullptr,
      nvlink_adapter,
      pcie_adapter);
  REQUIRE(context->set_topology(build_local_fabric_topology()).ok());
  REQUIRE(context
              ->set_endpoint_bindings({
                  EndpointBinding{.endpoint_id = "nvlink0", .node_id = "node0"},
                  EndpointBinding{.endpoint_id = "nvlink1", .node_id = "node0"},
              })
              .ok());

  auto communicator_or = context->get_communicator("nvlink0", "nvlink1");
  REQUIRE(communicator_or.ok());

  ReadPlan plan = make_single_slice_read_plan("nvlink0", "pcie1", ConnectionProtocol::kNvlink);
  auto result = communicator_or.value()->read_plan(plan).get();
  REQUIRE_FALSE(result.status.ok());
  CHECK(result.status.code() == absl::StatusCode::kInvalidArgument);
  CHECK(nvlink_adapter->read_plan_calls() == 0);
}

TEST_CASE("RoutingContext falls back to AUTO protocol when NVLINK adapter is unavailable", "[communicator][routing]") {
  auto nvlink_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kNvlink, false);
  auto pcie_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kPcie, true);

  auto context = std::make_shared<RoutingContext>(
      RoutingContext::Options{},
      /*engine=*/nullptr,
      std::move(nvlink_adapter),
      std::move(pcie_adapter));
  REQUIRE(context->set_topology(build_local_fabric_topology()).ok());
  REQUIRE(context
              ->set_endpoint_bindings({
                  EndpointBinding{.endpoint_id = "nvlink0", .node_id = "node0"},
                  EndpointBinding{.endpoint_id = "nvlink1", .node_id = "node0"},
              })
              .ok());

  auto communicator_or = context->get_communicator("nvlink0", "nvlink1");
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);
  CHECK(channel_or.value()->hops().front()->protocol() == ConnectionProtocol::kAuto);
}

TEST_CASE(
    "RoutingContext prefers PCIE protocol for same-node PCIE endpoints when adapter is available",
    "[communicator][routing]") {
  auto nvlink_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kNvlink, false);
  auto pcie_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kPcie, true);

  auto context = std::make_shared<RoutingContext>(
      RoutingContext::Options{},
      /*engine=*/nullptr,
      std::move(nvlink_adapter),
      std::move(pcie_adapter));
  REQUIRE(context->set_topology(build_local_fabric_topology()).ok());
  REQUIRE(context
              ->set_endpoint_bindings({
                  EndpointBinding{.endpoint_id = "pcie0", .node_id = "node0"},
                  EndpointBinding{.endpoint_id = "pcie1", .node_id = "node0"},
              })
              .ok());

  auto communicator_or = context->get_communicator("pcie0", "pcie1");
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);
  CHECK(channel_or.value()->hops().front()->protocol() == ConnectionProtocol::kPcie);
}

TEST_CASE("RoutingContext falls back to AUTO protocol when PCIE adapter is unavailable", "[communicator][routing]") {
  auto options = RoutingContext::Options{};
  options.prefer_nvlink = true;
  options.prefer_pcie = true;
  auto nvlink_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kNvlink, false);
  auto pcie_adapter = std::make_shared<ProtocolAdapterStub>(ConnectionProtocol::kPcie, false);

  auto context = std::make_shared<RoutingContext>(
      options,
      /*engine=*/nullptr,
      std::move(nvlink_adapter),
      std::move(pcie_adapter));
  REQUIRE(context->set_topology(build_local_fabric_topology()).ok());
  REQUIRE(context
              ->set_endpoint_bindings({
                  EndpointBinding{.endpoint_id = "pcie0", .node_id = "node0"},
                  EndpointBinding{.endpoint_id = "pcie1", .node_id = "node0"},
              })
              .ok());

  auto communicator_or = context->get_communicator("pcie0", "pcie1");
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);
  CHECK(channel_or.value()->hops().front()->protocol() == ConnectionProtocol::kAuto);
}

TEST_CASE("RoutingContext rail-matched fallback selects remote NIC by source rail", "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_rail_switch_topology()).ok());

  std::vector<EndpointBinding> bindings;
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_a/dev/gpu/0",
          .node_id = "node_a",
          .rail_id = 0,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_b/dev/gpu/0",
          .node_id = "node_b",
          .rail_id = 1,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_remote_0",
          .node_id = "node_b",
          .ip = "10.0.0.10",
          .port = 4010,
          .rail_id = 0,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_remote_1",
          .node_id = "node_b",
          .ip = "10.0.0.11",
          .port = 4011,
          .rail_id = 1,
      });
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  auto comm_or = context->get_communicator("node_a/dev/gpu/0", "node_b/dev/gpu/0");
  REQUIRE(comm_or.ok());
  auto channel_or = comm_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);

  const std::shared_ptr<Connection>& connection = channel_or.value()->hops().front();
  CHECK(connection->remote_binding().endpoint_id == "nic_remote_0");
  CHECK(connection->remote_binding().ip == "10.0.0.10");
  CHECK(connection->remote_binding().port == 4010);
  CHECK(connection->type() == ConnectionType::kSwitch);
  REQUIRE(connection->link() != nullptr);
  CHECK(connection->link()->id == "nic_0_to_netsw_rail_0");
}

TEST_CASE(
    "RoutingContext rail-matched fallback uses destination rail when preferred rail missing",
    "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_rail_switch_topology()).ok());

  std::vector<EndpointBinding> bindings;
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_a/dev/gpu/0",
          .node_id = "node_a",
          .rail_id = 0,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_b/dev/gpu/1",
          .node_id = "node_b",
          .rail_id = 1,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_remote_1",
          .node_id = "node_b",
          .ip = "10.0.0.21",
          .port = 4021,
          .rail_id = 1,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_remote_2",
          .node_id = "node_b",
          .ip = "10.0.0.22",
          .port = 4022,
          .rail_id = 2,
      });
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  auto comm_or = context->get_communicator("node_a/dev/gpu/0", "node_b/dev/gpu/1");
  REQUIRE(comm_or.ok());
  auto channel_or = comm_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);

  const std::shared_ptr<Connection>& connection = channel_or.value()->hops().front();
  CHECK(connection->remote_binding().endpoint_id == "nic_remote_1");
  CHECK(connection->remote_binding().rail_id == 1);
}

TEST_CASE(
    "RoutingContext rail-matched fallback maps GPU-GPU traffic across eight rails by affinity",
    "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_eight_rail_switch_topology()).ok());

  std::vector<EndpointBinding> bindings;
  for (int idx = 0; idx < 8; ++idx) {
    bindings.push_back(
        EndpointBinding{
            .endpoint_id = "node_a/dev/gpu/" + std::to_string(idx),
            .node_id = "node_a",
            .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
            .dev_id = idx,
        });
    bindings.push_back(
        EndpointBinding{
            .endpoint_id = "node_b/dev/gpu/" + std::to_string(idx),
            .node_id = "node_b",
            .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
            .dev_id = idx,
        });
    bindings.push_back(
        EndpointBinding{
            .endpoint_id = "nic_" + std::to_string(idx),
            .node_id = "node_b",
            .ip = "10.0.1." + std::to_string(10 + idx),
            .port = static_cast<uint16_t>(4100 + idx),
        });
  }
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  for (int idx = 0; idx < 8; ++idx) {
    auto comm_or =
        context->get_communicator("node_a/dev/gpu/" + std::to_string(idx), "node_b/dev/gpu/" + std::to_string(idx));
    REQUIRE(comm_or.ok());

    auto channel_or = comm_or.value()->primary_channel();
    REQUIRE(channel_or.ok());
    REQUIRE(channel_or.value()->hops().size() == 1);

    const std::shared_ptr<Connection>& connection = channel_or.value()->hops().front();
    CHECK(connection->type() == ConnectionType::kSwitch);
    CHECK(connection->remote_binding().endpoint_id == "nic_" + std::to_string(idx));
    REQUIRE(connection->link() != nullptr);
    CHECK(connection->link()->id == "nic_" + std::to_string(idx) + "_to_netsw_rail_" + std::to_string(idx));
  }
}

TEST_CASE(
    "RoutingContext rail-matched fallback picks CPU-affine remote NIC for GPU-to-CPU traffic when preferred rail is absent",
    "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_eight_rail_switch_topology()).ok());

  std::vector<EndpointBinding> bindings;
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_a/dev/gpu/3",
          .node_id = "node_a",
          .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
          .dev_id = 3,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_b/dev/cpu/1",
          .node_id = "node_b",
          .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 1,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_1",
          .node_id = "node_b",
          .ip = "10.0.2.11",
          .port = 4201,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_5",
          .node_id = "node_b",
          .ip = "10.0.2.15",
          .port = 4205,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_7",
          .node_id = "node_b",
          .ip = "10.0.2.17",
          .port = 4207,
      });
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  auto comm_or = context->get_communicator("node_a/dev/gpu/3", "node_b/dev/cpu/1");
  REQUIRE(comm_or.ok());
  auto channel_or = comm_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);

  const std::shared_ptr<Connection>& connection = channel_or.value()->hops().front();
  CHECK(connection->type() == ConnectionType::kSwitch);
  CHECK(connection->remote_binding().endpoint_id == "nic_5");
  REQUIRE(connection->link() != nullptr);
  CHECK(connection->link()->id == "nic_3_to_netsw_rail_3");
}

TEST_CASE(
    "RoutingContext rail-matched fallback keeps CPU affinity for CPU-to-CPU traffic on both nodes",
    "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_eight_rail_switch_topology()).ok());

  std::vector<EndpointBinding> bindings;
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_a/dev/cpu/1",
          .node_id = "node_a",
          .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 1,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node_b/dev/cpu/0",
          .node_id = "node_b",
          .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_0",
          .node_id = "node_b",
          .ip = "10.0.3.10",
          .port = 4300,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_1",
          .node_id = "node_b",
          .ip = "10.0.3.11",
          .port = 4301,
      });
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "nic_5",
          .node_id = "node_b",
          .ip = "10.0.3.15",
          .port = 4305,
      });
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  auto comm_or = context->get_communicator("node_a/dev/cpu/1", "node_b/dev/cpu/0");
  REQUIRE(comm_or.ok());
  auto channel_or = comm_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hops().size() == 1);

  const std::shared_ptr<Connection>& connection = channel_or.value()->hops().front();
  CHECK(connection->type() == ConnectionType::kSwitch);
  CHECK(connection->remote_binding().endpoint_id == "nic_0");
  REQUIRE(connection->link() != nullptr);
  CHECK(connection->link()->id == "nic_4_to_netsw_rail_4");
}

TEST_CASE(
    "RoutingContext supports cross-node GPU bootstrap then same-node GPU and memory fanout",
    "[communicator][routing]") {
  auto context = std::make_shared<RoutingContext>(RoutingContext::Options{}, /*engine=*/nullptr);
  REQUIRE(context->set_topology(build_two_node_transfer_and_fanout_topology()).ok());

  std::vector<EndpointBinding> bindings;
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node0/dev/gpu/0",
          .node_id = "node0",
          .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
          .dev_id = 0,
      });
  for (int gpu_id = 0; gpu_id < 8; ++gpu_id) {
    bindings.push_back(
        EndpointBinding{
            .endpoint_id = "node1/dev/gpu/" + std::to_string(gpu_id),
            .node_id = "node1",
            .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU,
            .dev_id = gpu_id,
        });
  }
  bindings.push_back(
      EndpointBinding{
          .endpoint_id = "node1/dev/cpu/0",
          .node_id = "node1",
          .dev_type = tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      });
  for (int rail_id = 0; rail_id < 8; ++rail_id) {
    bindings.push_back(
        EndpointBinding{
            .endpoint_id = "nic_" + std::to_string(rail_id),
            .node_id = "node1",
            .ip = "10.2.0." + std::to_string(20 + rail_id),
            .port = static_cast<uint16_t>(4500 + rail_id),
        });
  }
  REQUIRE(context->set_endpoint_bindings(std::move(bindings)).ok());

  auto cross_node_comm_or = context->get_communicator("node0/dev/gpu/0", "node1/dev/gpu/0");
  REQUIRE(cross_node_comm_or.ok());
  auto cross_node_channel_or = cross_node_comm_or.value()->primary_channel();
  REQUIRE(cross_node_channel_or.ok());
  REQUIRE(cross_node_channel_or.value()->hops().size() == 1);
  const std::shared_ptr<Connection>& cross_node_connection = cross_node_channel_or.value()->hops().front();
  CHECK(cross_node_connection->type() == ConnectionType::kSwitch);
  CHECK(cross_node_connection->remote_binding().endpoint_id == "nic_0");
  REQUIRE(cross_node_connection->link() != nullptr);
  CHECK(cross_node_connection->link()->id == "nic_0_to_netsw_rail_0");

  for (int gpu_id = 1; gpu_id < 8; ++gpu_id) {
    auto fanout_comm_or = context->get_communicator("node1/dev/gpu/0", "node1/dev/gpu/" + std::to_string(gpu_id));
    REQUIRE(fanout_comm_or.ok());
    auto fanout_channel_or = fanout_comm_or.value()->primary_channel();
    REQUIRE(fanout_channel_or.ok());
    REQUIRE(fanout_channel_or.value()->hops().size() == 1);

    const std::shared_ptr<Connection>& fanout_connection = fanout_channel_or.value()->hops().front();
    CHECK(fanout_connection->type() == ConnectionType::kP2P);
    CHECK(fanout_connection->remote_binding().endpoint_id == "node1/dev/gpu/" + std::to_string(gpu_id));
    REQUIRE(fanout_connection->link() != nullptr);
    CHECK(fanout_connection->link()->id == "node1_gpu0_to_gpu" + std::to_string(gpu_id));
  }

  auto fanout_mem_comm_or = context->get_communicator("node1/dev/gpu/0", "node1/dev/cpu/0");
  REQUIRE(fanout_mem_comm_or.ok());
  auto fanout_mem_channel_or = fanout_mem_comm_or.value()->primary_channel();
  REQUIRE(fanout_mem_channel_or.ok());
  REQUIRE(fanout_mem_channel_or.value()->hops().size() == 1);
  const std::shared_ptr<Connection>& fanout_mem_connection = fanout_mem_channel_or.value()->hops().front();
  CHECK(fanout_mem_connection->type() == ConnectionType::kP2P);
  CHECK(fanout_mem_connection->remote_binding().endpoint_id == "node1/dev/cpu/0");
  REQUIRE(fanout_mem_connection->link() != nullptr);
  CHECK(fanout_mem_connection->link()->id == "node1_gpu0_to_cpu0");
}
