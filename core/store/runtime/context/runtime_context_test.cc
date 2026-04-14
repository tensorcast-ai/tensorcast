// Copyright (c) 2025, TensorCast Team.

#include <atomic>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "catch2/catch_test_macros.hpp"
#include "core/communicator/routing/routing_context.h"
#include "core/store/components/communication_manager.h"
#include "core/store/memory_tier_budget.h"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/store_engine_options.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

using tensorcast::store::MemoryTierBudget;
using tensorcast::store::MemoryTierConfig;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::CommunicationManager;
using tensorcast::store::runtime::RegistrationEvent;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::RuntimeEvent;
using tensorcast::store::runtime::RuntimeEventType;

namespace {

StoreEngineOptions MakeContextOptions() {
  StoreEngineOptions opts;
  opts.storage_path = "";
  opts.memory_pool_size = 16ull * 1024 * 1024;
  opts.tx_slice_bytes = 512 * 1024;
  opts.artifact_chunk_bytes = opts.tx_slice_bytes * 2;
  opts.num_thread = 1;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return opts;
}

StoreEngineOptions::MaterializationStrategyConfig make_topology_guided_strategy(
    StoreEngineOptions::MaterializationStrategyConfig::TopologyGuidedMode mode) {
  StoreEngineOptions::MaterializationStrategyConfig strategy;
  strategy.enable_topology_guided_transfer = true;
  strategy.topology_guided_mode = mode;
  return strategy;
}

tensorcast::communicator::v1::CommunicatorConfig make_simple_numa_config() {
  tensorcast::communicator::v1::CommunicatorConfig comm_cfg;
  comm_cfg.set_enable_rdma(false);
  comm_cfg.mutable_stager()->set_buffers_per_flow(2);
  comm_cfg.mutable_transport()->set_tcp_conn_count(1);
  auto* simple_numa = comm_cfg.mutable_simple_numa();
  simple_numa->set_enable(true);
  auto* node = simple_numa->add_nodes();
  node->set_id(0);
  node->add_nics("mlx5_0");
  node->add_gpus(0);
  return comm_cfg;
}

} // namespace

TEST_CASE("RuntimeContext keeps topology-guided routing disabled by default", "[runtime][context][routing]") {
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());

  StoreEngineOptions opts = MakeContextOptions();
  opts.comm_manager = comm_manager;
  RuntimeContext context(opts);
  context.set_worker_identity(
      {.worker_id = "worker-disabled",
       .node_id = "node-disabled",
       .node_address = "127.0.0.1",
       .grpc_port = 50051,
       .p2p_port = comm_manager->listen_port()});

  REQUIRE(context.start().ok());
  CHECK(comm_manager->routing_context() == nullptr);

  context.shutdown();
}

TEST_CASE("RuntimeContext bootstraps routing context with typed communicator config", "[runtime][context][routing]") {
  StoreEngineOptions opts = MakeContextOptions();
  opts.communicator_config = make_simple_numa_config();
  opts.materialization_strategy = make_topology_guided_strategy(
      StoreEngineOptions::MaterializationStrategyConfig::TopologyGuidedMode::kPreferGuided);
  RuntimeContext context(opts);

  REQUIRE(context.start().ok());
  auto comm_manager = context.communication_manager();
  REQUIRE(comm_manager != nullptr);
  REQUIRE(comm_manager->is_enabled());
  CHECK(comm_manager->routing_context() == nullptr);

  context.set_worker_identity(
      {.worker_id = "worker-built-in",
       .node_id = "node-built-in",
       .node_address = "127.0.0.1",
       .grpc_port = 50051,
       .p2p_port = comm_manager->listen_port()});

  auto routing_context = comm_manager->routing_context();
  REQUIRE(routing_context != nullptr);

  const std::string local_endpoint = "node-built-in/dev/cpu/0";
  const std::string remote_endpoint = "node-remote/dev/cpu/0";
  comm_manager->remember_p2p_source(
      tensorcast::store::P2PSource{
          .local_endpoint_id = "",
          .remote_endpoint_id = remote_endpoint,
          .ip = "10.0.0.3",
          .port = 64010,
          .location =
              {
                  .type = tensorcast::common::memory::MemoryLocation::CPU,
                  .device_id = 0,
              },
      });

  auto communicator_or = routing_context->get_communicator(local_endpoint, remote_endpoint);
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  CHECK(channel_or.value()->src_endpoint_id() == local_endpoint);
  CHECK(channel_or.value()->dst_endpoint_id() == remote_endpoint);
  CHECK(channel_or.value()->hop_count() == 1);

  context.shutdown();
}

TEST_CASE("RuntimeContext bootstraps routing context for product communicator", "[runtime][context][routing]") {
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());

  StoreEngineOptions opts = MakeContextOptions();
  opts.comm_manager = comm_manager;
  opts.materialization_strategy = make_topology_guided_strategy(
      StoreEngineOptions::MaterializationStrategyConfig::TopologyGuidedMode::kPreferGuided);
  RuntimeContext context(opts);
  context.set_worker_identity(
      {.worker_id = "worker-local",
       .node_id = "node-local",
       .node_address = "127.0.0.1",
       .grpc_port = 50051,
       .p2p_port = comm_manager->listen_port()});

  REQUIRE(context.start().ok());
  auto routing_context = comm_manager->routing_context();
  REQUIRE(routing_context != nullptr);

  const std::string local_endpoint = "node-local/dev/cpu/0";
  const std::string remote_endpoint = "node-remote/dev/cpu/0";
  comm_manager->remember_p2p_source(
      tensorcast::store::P2PSource{
          .local_endpoint_id = "",
          .remote_endpoint_id = remote_endpoint,
          .ip = "10.0.0.2",
          .port = 64000,
          .location =
              {
                  .type = tensorcast::common::memory::MemoryLocation::CPU,
                  .device_id = 0,
              },
      });
  routing_context = comm_manager->routing_context();
  REQUIRE(routing_context != nullptr);

  auto communicator_or = routing_context->get_communicator(local_endpoint, remote_endpoint);
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  CHECK(channel_or.value()->src_endpoint_id() == local_endpoint);
  CHECK(channel_or.value()->dst_endpoint_id() == remote_endpoint);
  CHECK(channel_or.value()->hop_count() == 1);

  context.shutdown();
}

TEST_CASE(
    "RuntimeContext refreshes routing context when worker identity is set after start",
    "[runtime][context][routing]") {
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());

  StoreEngineOptions opts = MakeContextOptions();
  opts.comm_manager = comm_manager;
  opts.materialization_strategy = make_topology_guided_strategy(
      StoreEngineOptions::MaterializationStrategyConfig::TopologyGuidedMode::kPreferGuided);
  RuntimeContext context(opts);

  REQUIRE(context.start().ok());
  CHECK(comm_manager->routing_context() == nullptr);

  context.set_worker_identity(
      {.worker_id = "worker-late",
       .node_id = "node-late",
       .node_address = "127.0.0.1",
       .grpc_port = 50051,
       .p2p_port = comm_manager->listen_port()});

  auto routing_context = comm_manager->routing_context();
  REQUIRE(routing_context != nullptr);

  const std::string local_endpoint = "node-late/dev/cpu/0";
  const std::string remote_endpoint = "node-remote/dev/cpu/0";
  comm_manager->remember_p2p_source(
      tensorcast::store::P2PSource{
          .local_endpoint_id = "",
          .remote_endpoint_id = remote_endpoint,
          .ip = "10.0.0.2",
          .port = 64001,
          .location =
              {
                  .type = tensorcast::common::memory::MemoryLocation::CPU,
                  .device_id = 0,
              },
      });

  auto communicator_or = routing_context->get_communicator(local_endpoint, remote_endpoint);
  REQUIRE(communicator_or.ok());
  auto channel_or = communicator_or.value()->primary_channel();
  REQUIRE(channel_or.ok());
  CHECK(channel_or.value()->src_endpoint_id() == local_endpoint);
  CHECK(channel_or.value()->dst_endpoint_id() == remote_endpoint);
  CHECK(channel_or.value()->hop_count() == 1);

  context.shutdown();
}

TEST_CASE(
    "RuntimeContext keeps observe-only topology-guided rollout on baseline execution",
    "[runtime][context][routing]") {
  auto comm_manager = std::make_shared<CommunicationManager>();
  REQUIRE(comm_manager->initialize_with_config("127.0.0.1", /*listen_port=*/0, make_simple_numa_config()).ok());

  StoreEngineOptions opts = MakeContextOptions();
  opts.comm_manager = comm_manager;
  opts.materialization_strategy = make_topology_guided_strategy(
      StoreEngineOptions::MaterializationStrategyConfig::TopologyGuidedMode::kObserveOnly);
  RuntimeContext context(opts);
  context.set_worker_identity(
      {.worker_id = "worker-observe",
       .node_id = "node-observe",
       .node_address = "127.0.0.1",
       .grpc_port = 50051,
       .p2p_port = comm_manager->listen_port()});

  REQUIRE(context.start().ok());
  CHECK(comm_manager->routing_context() != nullptr);

  context.shutdown();
}

TEST_CASE("MemoryTierBudget fails fast on invalid capacities", "[runtime][memory_tier][budget]") {
  MemoryTierConfig cfg;
  cfg.enable_preemptible_memory = false;
  cfg.preemptible_limit_bytes = 0;
  cfg.preemptible_low_watermark_ratio = 0.5;

  cfg.stable_bytes = 0;
  auto budget_zero = MemoryTierBudget::from_config(cfg, /*host_dram_bytes=*/1024, /*pinned_pool_bytes=*/0);
  REQUIRE_FALSE(budget_zero.ok());
  CHECK(budget_zero.status().code() == absl::StatusCode::kInvalidArgument);

  cfg.stable_bytes = 900;
  auto budget_pinned_over = MemoryTierBudget::from_config(cfg, /*host_dram_bytes=*/512, /*pinned_pool_bytes=*/600);
  REQUIRE_FALSE(budget_pinned_over.ok());
  CHECK(budget_pinned_over.status().code() == absl::StatusCode::kInvalidArgument);

  cfg.stable_bytes = 512;
  auto budget_uma_over = MemoryTierBudget::from_config(cfg, /*host_dram_bytes=*/256, /*pinned_pool_bytes=*/32);
  REQUIRE_FALSE(budget_uma_over.ok());
  CHECK(budget_uma_over.status().code() == absl::StatusCode::kInvalidArgument);

  cfg.stable_bytes = 128;
  auto budget_ok = MemoryTierBudget::from_config(cfg, /*host_dram_bytes=*/512, /*pinned_pool_bytes=*/64);
  REQUIRE(budget_ok.ok());
  CHECK(budget_ok->snapshot().stable_total_bytes == cfg.stable_bytes);
  CHECK(budget_ok->snapshot().preemptible_total_bytes == 0);
}

TEST_CASE("RuntimeContext validates memory tier thresholds before startup", "[runtime][context][memory_tier]") {
  StoreEngineOptions opts = MakeContextOptions();
  MemoryTierConfig cfg;
  cfg.enable_preemptible_memory = true;
  cfg.preemptible_limit_bytes = 1024;
  cfg.stable_bytes = 1024;
  cfg.preemptible_low_watermark_ratio = 1.2;
  opts.memory_tier_config = cfg;

  RuntimeContext bad_watermark(opts);
  auto status = bad_watermark.start();
  REQUIRE_FALSE(status.ok());
  CHECK(status.code() == absl::StatusCode::kInvalidArgument);

  cfg.preemptible_low_watermark_ratio = 0.5;
  cfg.stable_bytes = std::numeric_limits<uint64_t>::max() / 4;
  opts.memory_tier_config = cfg;
  RuntimeContext bad_capacity(opts);
  status = bad_capacity.start();
  REQUIRE_FALSE(status.ok());
  CHECK(status.code() == absl::StatusCode::kInvalidArgument);

  cfg.stable_bytes = 1024;
  opts.memory_tier_config = cfg;
  RuntimeContext ok_context(opts);
  status = ok_context.start();
  REQUIRE(status.ok());
  ok_context.shutdown();
}

TEST_CASE("RuntimeContext publishes events and drains", "[runtime][context]") {
  RuntimeContext context(MakeContextOptions());
  REQUIRE(context.start().ok());

  std::vector<RuntimeEventType> observed;
  absl::Mutex mu;
  auto subscription = context.subscribe_to_events([&](const RuntimeEvent& event) {
    absl::MutexLock lock(&mu);
    observed.push_back(event.type);
  });
  REQUIRE(subscription != nullptr);

  auto publisher = context.event_publisher();
  REQUIRE(static_cast<bool>(publisher));

  RuntimeEvent registration_event;
  registration_event.type = RuntimeEventType::kRegistrationCommitted;
  RegistrationEvent payload;
  payload.registration_id = "test-reg";
  payload.committed = true;
  registration_event.payload = payload;
  publisher.publish(registration_event);

  RuntimeEvent abort_event;
  abort_event.type = RuntimeEventType::kRegistrationAborted;
  RegistrationEvent abort_payload;
  abort_payload.registration_id = "test-reg";
  abort_event.payload = abort_payload;
  publisher.publish(abort_event);

  context.drain_events();
  {
    absl::MutexLock lock(&mu);
    REQUIRE(observed.size() == 2);
    CHECK(observed.front() == RuntimeEventType::kRegistrationCommitted);
    CHECK(observed.back() == RuntimeEventType::kRegistrationAborted);
  }
  context.shutdown();
}

TEST_CASE("RuntimeContext handles concurrent publishers", "[runtime][context][concurrency]") {
  constexpr int kThreads = 4;
  constexpr int kEventsPerThread = 200;

  RuntimeContext context(MakeContextOptions());
  REQUIRE(context.start().ok());

  std::vector<std::string> registrations;
  absl::Mutex mu;
  auto subscription = context.subscribe_to_events([&](const RuntimeEvent& event) {
    if (event.type != RuntimeEventType::kRegistrationCommitted) {
      return;
    }
    const auto* reg = std::get_if<RegistrationEvent>(&event.payload);
    if (reg == nullptr) {
      return;
    }
    absl::MutexLock lock(&mu);
    registrations.push_back(reg->registration_id);
  });
  REQUIRE(subscription != nullptr);

  auto publisher = context.event_publisher();
  REQUIRE(static_cast<bool>(publisher));

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([t, publisher]() mutable {
      for (int i = 0; i < kEventsPerThread; ++i) {
        RuntimeEvent event;
        event.type = RuntimeEventType::kRegistrationCommitted;
        RegistrationEvent payload;
        payload.registration_id = "thread-" + std::to_string(t) + "-" + std::to_string(i);
        payload.committed = true;
        event.payload = payload;
        publisher.publish(event);
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }
  context.drain_events();

  {
    absl::MutexLock lock(&mu);
    REQUIRE(registrations.size() == static_cast<size_t>(kThreads * kEventsPerThread));
  }
  context.shutdown();
}
