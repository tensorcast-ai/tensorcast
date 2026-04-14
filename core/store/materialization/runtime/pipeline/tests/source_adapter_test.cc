// Copyright (c) 2026, TensorCast Team.

#include "catch2/catch_test_macros.hpp"
#include "core/communicator/routing/routing_context.h"
#include "core/store/components/communication_manager.h"
#include "core/store/materialization/runtime/pipeline/source_adapter.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/store_engine_options.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

using tensorcast::common::memory::MemoryLocation;
using tensorcast::communicator::routing::RoutingContext;
using tensorcast::store::P2PSource;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::materialization::runtime::pipeline::IngestionContext;
using tensorcast::store::materialization::runtime::pipeline::P2PSourceAdapter;
using tensorcast::store::materialization::runtime::pipeline::SourceType;
using tensorcast::store::runtime::RuntimeContext;

namespace {

StoreEngineOptions make_context_options() {
  StoreEngineOptions options;
  options.storage_path = "";
  options.memory_pool_size = 16ULL * 1024 * 1024;
  options.tx_slice_bytes = 512 * 1024;
  options.artifact_chunk_bytes = options.tx_slice_bytes * 2;
  options.num_thread = 1;
  options.pinned_memory_timeout = std::chrono::milliseconds(0);
  return options;
}

StoreEngineOptions::MaterializationStrategyConfig make_observe_only_strategy() {
  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_topology_guided_transfer = true;
  strategy.topology_guided_mode = StoreEngineOptions::MaterializationStrategyConfig::TopologyGuidedMode::kObserveOnly;
  return strategy;
}

tensorcast::communicator::v1::CommunicatorConfig make_simple_numa_config() {
  tensorcast::communicator::v1::CommunicatorConfig config;
  config.set_enable_rdma(false);
  auto* simple_numa = config.mutable_simple_numa();
  simple_numa->set_enable(true);
  auto* node = simple_numa->add_nodes();
  node->set_id(0);
  node->add_nics("mlx5_0");
  node->add_gpus(0);
  return config;
}

} // namespace

TEST_CASE("P2PSourceAdapter preserves caller routing context in observe-only mode", "[pipeline][p2p][routing]") {
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());

  StoreEngineOptions options = make_context_options();
  options.comm_manager = comm_manager;
  options.materialization_strategy = make_observe_only_strategy();
  RuntimeContext runtime_context(options);
  REQUIRE(runtime_context.start().ok());

  IngestionContext context{};
  context.source_type = SourceType::kP2P;
  context.options = &options;
  context.runtime_context = &runtime_context;

  auto caller_routing_context =
      std::make_shared<RoutingContext>(RoutingContext::Options{}, comm_manager->get_shared_engine());

  P2PSource source;
  source.ip = "10.0.0.2";
  source.port = 61000;
  source.local_endpoint_id = "node-local/dev/cpu/0";
  source.remote_endpoint_id = "node-remote/dev/cpu/0";
  source.location = {
      .type = MemoryLocation::CPU,
      .device_id = 0,
  };
  source.routing_context = caller_routing_context;

  REQUIRE(P2PSourceAdapter::prepare(source, context).ok());
  CHECK(context.p2p.source.routing_context == caller_routing_context);
  CHECK(context.p2p.source.comm_engine == comm_manager->get_shared_engine());

  runtime_context.shutdown();
}

TEST_CASE("P2PSourceAdapter injects runtime routing context in guided mode", "[pipeline][p2p][routing]") {
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());
  REQUIRE(comm_manager->bootstrap_routing_context().ok());
  REQUIRE(comm_manager->routing_context() != nullptr);

  StoreEngineOptions options = make_context_options();
  options.comm_manager = comm_manager;
  options.materialization_strategy.enable_topology_guided_transfer = true;
  options.materialization_strategy.topology_guided_mode =
      StoreEngineOptions::MaterializationStrategyConfig::TopologyGuidedMode::kPreferGuided;
  RuntimeContext runtime_context(options);
  REQUIRE(runtime_context.start().ok());

  IngestionContext context{};
  context.source_type = SourceType::kP2P;
  context.options = &options;
  context.runtime_context = &runtime_context;

  P2PSource source;
  source.ip = "10.0.0.3";
  source.port = 61001;
  source.local_endpoint_id = "node-local/dev/cpu/0";
  source.remote_endpoint_id = "node-remote/dev/cpu/0";
  source.location = {
      .type = MemoryLocation::CPU,
      .device_id = 0,
  };

  REQUIRE(P2PSourceAdapter::prepare(source, context).ok());
  CHECK(context.p2p.source.routing_context == comm_manager->routing_context());
  CHECK(context.p2p.source.comm_engine == comm_manager->get_shared_engine());

  runtime_context.shutdown();
}
