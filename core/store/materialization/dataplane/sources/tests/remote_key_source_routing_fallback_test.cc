// Copyright (c) 2026, TensorCast Team.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/routing_context.h"
#include "core/communicator/topology/topology.h"
#include "core/store/materialization/dataplane/sources/remote_key_source.h"
#include "core/testing/test_helpers.h"
#include "gsl/pointers"

namespace {

using tensorcast::communicator::engine::Communicator;
using tensorcast::communicator::routing::EndpointBinding;
using tensorcast::communicator::routing::RoutingContext;
using tensorcast::communicator::topology::Endpoint;
using tensorcast::communicator::topology::EndpointKind;
using tensorcast::communicator::topology::EndpointType;
using tensorcast::communicator::topology::Link;
using tensorcast::communicator::topology::LinkType;
using tensorcast::communicator::topology::Pool;
using tensorcast::communicator::topology::PoolType;
using tensorcast::communicator::topology::Topology;
using tensorcast::store::loader::RemoteKeySource;
using tensorcast::testing::create_test_pattern;
using tensorcast::testing::find_available_port;
using tensorcast::testing::make_tcp_communicator_config;
using tensorcast::testing::make_test_pinned_staging_pools;

Topology build_direct_topology(const std::string& local_endpoint, const std::string& remote_endpoint) {
  std::vector<Pool> pools;
  pools.push_back(Pool{"pool_local_cpu", "pool_local_cpu", PoolType::kCpu});
  pools.push_back(Pool{"pool_local_gpu", "pool_local_gpu", PoolType::kGpu});
  pools.push_back(Pool{"pool_remote_cpu", "pool_remote_cpu", PoolType::kCpu});
  pools.push_back(Pool{"pool_remote_gpu", "pool_remote_gpu", PoolType::kGpu});

  std::vector<Endpoint> endpoints;
  Endpoint src;
  src.id = local_endpoint;
  src.name = local_endpoint;
  src.kind = EndpointKind::kClient;
  src.type = EndpointType::kNic;
  src.pool_ids = {"pool_local_cpu", "pool_local_gpu"};
  endpoints.push_back(src);

  Endpoint dst;
  dst.id = remote_endpoint;
  dst.name = remote_endpoint;
  dst.kind = EndpointKind::kClient;
  dst.type = EndpointType::kNic;
  dst.pool_ids = {"pool_remote_cpu", "pool_remote_gpu"};
  endpoints.push_back(dst);

  std::vector<Link> links;
  Link link;
  link.id = "link_" + local_endpoint + "_to_" + remote_endpoint;
  link.name = link.id;
  link.type = LinkType::kP2P;
  link.src_endpoint_id = local_endpoint;
  link.dst_endpoint_id = remote_endpoint;
  links.push_back(link);

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = true});
  REQUIRE(topology_or.ok());
  return std::move(topology_or).value();
}

std::shared_ptr<Communicator> make_engine_on_port(int port) {
  auto cfg = make_tcp_communicator_config(/*enable_rdma=*/false, /*buffers_per_flow=*/2);
  auto pools = make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(4ULL << 20),
      /*cpu_slice_bytes=*/(2ULL << 20),
      /*enable_rdma=*/false);
  auto engine = std::make_shared<Communicator>(cfg, std::move(pools));
  REQUIRE(engine->init("127.0.0.1", static_cast<uint16_t>(port)).ok());
  return engine;
}

void register_cpu_tensor(const std::shared_ptr<Communicator>& engine, const std::string& key, std::vector<uint8_t>& src) {
  Communicator::RegisterTensorOptions reg_opts;
  reg_opts.register_mr = false;
  reg_opts.needs_staging = false;
  reg_opts.async = false;
  REQUIRE(engine
              ->register_tensor_ex(
                  key,
                  reinterpret_cast<uint64_t>(src.data()),
                  src.size(),
                  tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU,
                  /*dev_id=*/0,
                  reg_opts)
              .ok());
}

} // namespace

TEST_CASE("RemoteKeySource uses routed communicator when routing metadata is present", "[store][p2p][routing]") {
  constexpr std::size_t kArtifactBytes = 1024 * 1024;
  const std::string tensor_key = "remote_key_source_routed_read_tensor";
  const std::string local_endpoint = "node_local/dev/cpu/0";
  const std::string remote_endpoint = "node_remote/dev/cpu/0";

  const int src_port = find_available_port(52000);
  REQUIRE(src_port > 0);
  const int dst_port = find_available_port(src_port + 1);
  REQUIRE(dst_port > 0);

  auto src_engine = make_engine_on_port(src_port);
  auto dst_engine = make_engine_on_port(dst_port);

  std::vector<uint8_t> payload = create_test_pattern(kArtifactBytes, /*seed=*/7);
  register_cpu_tensor(src_engine, tensor_key, payload);

  auto routing_context = std::make_shared<RoutingContext>(RoutingContext::Options{}, dst_engine);
  REQUIRE(routing_context->set_topology(build_direct_topology(local_endpoint, remote_endpoint)).ok());
  std::vector<EndpointBinding> bindings{
      EndpointBinding{
          .endpoint_id = local_endpoint,
          .node_id = "node_local",
          .ip = "127.0.0.1",
          .port = static_cast<uint16_t>(dst_port),
      },
      EndpointBinding{
          .endpoint_id = remote_endpoint,
          .node_id = "node_remote",
          .ip = "127.0.0.1",
          .port = static_cast<uint16_t>(src_port),
      },
  };
  REQUIRE(routing_context->set_endpoint_bindings(std::move(bindings)).ok());

  // Intentionally provide an invalid direct address. If the read succeeds, it
  // must have gone through routed endpoint bindings.
  RemoteKeySource::Options opts{
      .comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{dst_engine},
      .memory_keys = {tensor_key},
      .buffer_sizes = {kArtifactBytes},
      .ip = "127.0.0.1",
      .port = 1,
      .local_endpoint_id = local_endpoint,
      .remote_endpoint_id = remote_endpoint,
      .routing_context = routing_context,
      .total_size = kArtifactBytes,
  };
  RemoteKeySource source(std::move(opts));

  std::vector<uint8_t> received(kArtifactBytes, 0);
  auto read_or = source.read(received.data(), received.size());
  REQUIRE(read_or.ok());
  REQUIRE(*read_or == received.size());
  REQUIRE(received == payload);
}

TEST_CASE("RemoteKeySource falls back to direct reads when routed lookup fails", "[store][p2p][routing]") {
  constexpr std::size_t kArtifactBytes = 2 * 1024 * 1024;
  constexpr std::size_t kReadChunkBytes = 256 * 1024;
  const std::string tensor_key = "remote_key_source_routing_fallback_tensor";

  const int src_port = find_available_port(53000);
  REQUIRE(src_port > 0);
  const int dst_port = find_available_port(src_port + 1);
  REQUIRE(dst_port > 0);

  auto src_engine = make_engine_on_port(src_port);
  auto dst_engine = make_engine_on_port(dst_port);

  std::vector<uint8_t> payload = create_test_pattern(kArtifactBytes, /*seed=*/42);
  register_cpu_tensor(src_engine, tensor_key, payload);

  // Routing metadata is present, but topology/bindings are intentionally absent,
  // forcing routed lookup failure and direct ip/port fallback.
  auto routing_context = std::make_shared<RoutingContext>(RoutingContext::Options{}, dst_engine);
  RemoteKeySource::Options opts{
      .comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{dst_engine},
      .memory_keys = {tensor_key},
      .buffer_sizes = {kArtifactBytes},
      .ip = "127.0.0.1",
      .port = static_cast<uint16_t>(src_port),
      .local_endpoint_id = "node_local/dev/cpu/0",
      .remote_endpoint_id = "node_remote/dev/cpu/0",
      .routing_context = routing_context,
      .total_size = kArtifactBytes,
  };
  RemoteKeySource source(std::move(opts));

  std::vector<uint8_t> received(kArtifactBytes, 0);
  std::size_t offset = 0;
  while (offset < received.size()) {
    const std::size_t ask = std::min(kReadChunkBytes, received.size() - offset);
    auto read_or = source.read(received.data() + offset, ask);
    REQUIRE(read_or.ok());
    REQUIRE(*read_or == ask);
    offset += *read_or;
  }

  REQUIRE(received == payload);
}
