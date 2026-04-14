// Copyright (c) 2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include "core/store/components/communication_manager.h"

namespace {

using tensorcast::store::components::CommunicationManager;

tensorcast::communicator::v1::CommunicatorConfig make_simple_numa_config(bool enable = true, int tcp_conn_count = 2) {
  tensorcast::communicator::v1::CommunicatorConfig config;
  config.set_enable_rdma(false);
  config.mutable_stager()->set_buffers_per_flow(2);
  config.mutable_transport()->set_tcp_conn_count(tcp_conn_count);
  auto* simple_numa = config.mutable_simple_numa();
  simple_numa->set_enable(enable);
  auto* node = simple_numa->add_nodes();
  node->set_id(0);
  node->add_nics("mlx5_0");
  node->add_gpus(0);
  return config;
}

tensorcast::store::P2PSource make_cpu_source(uint16_t port) {
  tensorcast::store::P2PSource source;
  source.ip = "10.0.0.2";
  source.port = port;
  source.local_endpoint_id = "node-local/dev/cpu/0";
  source.remote_endpoint_id = "node-remote/dev/cpu/0";
  source.location = tensorcast::store::Location{
      .type = tensorcast::common::memory::MemoryLocation::CPU,
      .device_id = 0,
  };
  return source;
}

} // namespace

TEST_CASE("CommunicationManager bootstraps routed topology from communicator config", "[store][communication][routing]") {
  tensorcast::communicator::v1::CommunicatorConfig config = make_simple_numa_config();
  CommunicationManager manager;
  REQUIRE(manager.initialize_with_config("127.0.0.1", /*listen_port=*/0, config).ok());
  REQUIRE(manager.bootstrap_routing_context().ok());
  REQUIRE(manager.routing_context() != nullptr);

  tensorcast::store::P2PSource source = make_cpu_source(/*port=*/4010);
  manager.remember_p2p_source(source);

  auto communicator_or = manager.routing_context()->get_communicator("node-local/dev/cpu/0", "node-remote/dev/cpu/0");
  REQUIRE(communicator_or.ok());

  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  REQUIRE(channel_or.value()->hop_count() == 1);
  REQUIRE(channel_or.value()->hops().size() == 1);
  CHECK(channel_or.value()->hops().front()->remote_binding().endpoint_id == "node-remote/dev/cpu/0");
  CHECK(channel_or.value()->hops().front()->remote_binding().ip == "10.0.0.2");
  CHECK(channel_or.value()->hops().front()->remote_binding().port == 4010);
}

TEST_CASE("CommunicationManager accumulates bindings before topology bootstrap", "[store][communication][routing]") {
  CommunicationManager manager;
  REQUIRE(manager.initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());

  manager.remember_p2p_source(make_cpu_source(/*port=*/4020));
  CHECK(manager.routing_context() == nullptr);

  REQUIRE(manager.bootstrap_routing_context().ok());
  auto routing_context = manager.routing_context();
  REQUIRE(routing_context != nullptr);

  auto communicator_or = routing_context->get_communicator("node-local/dev/cpu/0", "node-remote/dev/cpu/0");
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  CHECK(channel_or.value()->hops().front()->remote_binding().port == 4020);
}

TEST_CASE("CommunicationManager upserts live routing bindings without replacing context", "[store][communication][routing]") {
  CommunicationManager manager;
  REQUIRE(manager.initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());
  REQUIRE(manager.bootstrap_routing_context().ok());

  tensorcast::store::P2PSource source = make_cpu_source(/*port=*/4030);
  manager.remember_p2p_source(source);
  auto routing_context = manager.routing_context();
  REQUIRE(routing_context != nullptr);

  auto communicator_or = routing_context->get_communicator(source.local_endpoint_id, source.remote_endpoint_id);
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  CHECK(channel_or.value()->hops().front()->remote_binding().port == 4030);

  source.port = 4040;
  manager.remember_p2p_source(source);
  CHECK(manager.routing_context() == routing_context);

  auto updated_channel_or = communicator_or.value()->primary_channel();
  REQUIRE(updated_channel_or.ok());
  CHECK(updated_channel_or.value()->hops().front()->remote_binding().port == 4040);
}

TEST_CASE("CommunicationManager clears routing context when topology bootstrap is disabled", "[store][communication][routing]") {
  CommunicationManager manager;
  REQUIRE(manager.initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config(/*enable=*/false)).ok());
  manager.set_routing_context(std::make_shared<tensorcast::communicator::routing::RoutingContext>(
      tensorcast::communicator::routing::RoutingContext::Options{}, manager.get_shared_engine()));
  REQUIRE(manager.routing_context() != nullptr);

  REQUIRE(manager.bootstrap_routing_context().ok());
  CHECK(manager.routing_context() == nullptr);
}

TEST_CASE("CommunicationManager normalizes low tcp_conn_count for staging pool sizing", "[store][communication][routing]") {
  CommunicationManager manager;
  REQUIRE(
      manager.initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config(/*enable=*/true, /*tcp_conn_count=*/1))
          .ok());
  CHECK(manager.is_enabled());
}
