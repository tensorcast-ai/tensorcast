// Copyright (c) 2026, TensorCast Team.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/routing_context.h"
#include "core/communicator/topology/topology.h"
#include "core/store/materialization/dataplane/sources/remote_key_source.h"
#include "core/store/replica/types/direct_write_grant.h"
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
using tensorcast::store::DirectWriteGrant;
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

  std::vector<Link> links;
  Link nvlink_link;
  nvlink_link.id = "nvlink0_to_nvlink1";
  nvlink_link.name = nvlink_link.id;
  nvlink_link.type = LinkType::kP2P;
  nvlink_link.src_endpoint_id = "nvlink0";
  nvlink_link.dst_endpoint_id = "nvlink1";
  links.push_back(nvlink_link);

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      {.require_endpoint_links = true, .require_connected = false});
  REQUIRE(topology_or.ok());
  return std::move(topology_or).value();
}

std::shared_ptr<Communicator> make_engine_on_port(int port, bool enable_rdma = false) {
  auto cfg = make_tcp_communicator_config(enable_rdma, /*buffers_per_flow=*/2);
  auto pools = make_test_pinned_staging_pools(
      cfg.stager().buffers_per_flow(),
      cfg.transport().tcp_conn_count(),
      /*gpu_slice_bytes=*/(4ULL << 20),
      /*cpu_slice_bytes=*/(2ULL << 20),
      enable_rdma);
  auto engine = std::make_shared<Communicator>(cfg, std::move(pools));
  REQUIRE(engine->init("127.0.0.1", static_cast<uint16_t>(port)).ok());
  return engine;
}

void register_cpu_tensor(
    const std::shared_ptr<Communicator>& engine,
    const std::string& key,
    std::vector<uint8_t>& src) {
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

DirectWriteGrant make_direct_write_grant(std::vector<uint8_t>& buffer) {
  DirectWriteGrant grant;
  grant.windows.push_back(
      DirectWriteGrant::Window{
          .va_offset = 0,
          .local_addr = reinterpret_cast<uint64_t>(buffer.data()),
          .length = buffer.size(),
      });
  return grant;
}

int require_available_port_or_skip(int base_port) {
  const int port = find_available_port(base_port);
  if (port <= 0) {
    SKIP("TCP port probing is unavailable in the current test sandbox");
  }
  return port;
}

} // namespace

TEST_CASE("RemoteKeySource uses routed communicator when routing metadata is present", "[store][p2p][routing]") {
  constexpr std::size_t kArtifactBytes = 1024 * 1024;
  const std::string tensor_key = "remote_key_source_routed_read_tensor";
  const std::string local_endpoint = "node_local/dev/cpu/0";
  const std::string remote_endpoint = "node_remote/dev/cpu/0";

  const int src_port = require_available_port_or_skip(52000);
  const int dst_port = require_available_port_or_skip(src_port + 1);

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

  const int src_port = require_available_port_or_skip(53000);
  const int dst_port = require_available_port_or_skip(src_port + 1);

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

TEST_CASE("RemoteKeySource keeps strict direct fallback after routed failure", "[store][p2p][routing]") {
  constexpr std::size_t kArtifactBytes = 512 * 1024;
  const std::string tensor_key = "remote_key_source_strict_fallback_tensor";
  const std::string local_endpoint = "node_local/dev/cpu/0";
  const std::string remote_endpoint = "node_remote/dev/cpu/0";

  const int direct_src_port = require_available_port_or_skip(54000);
  const int routed_src_port = require_available_port_or_skip(direct_src_port + 1);
  const int dst_port = require_available_port_or_skip(routed_src_port + 1);

  auto direct_src_engine = make_engine_on_port(direct_src_port);
  auto routed_src_engine = make_engine_on_port(routed_src_port);
  auto dst_engine = make_engine_on_port(dst_port);

  std::vector<uint8_t> direct_payload = create_test_pattern(kArtifactBytes, /*seed=*/11);
  std::vector<uint8_t> routed_payload = create_test_pattern(kArtifactBytes, /*seed=*/73);
  REQUIRE(direct_payload != routed_payload);
  register_cpu_tensor(direct_src_engine, tensor_key, direct_payload);
  register_cpu_tensor(routed_src_engine, tensor_key, routed_payload);

  auto routing_context = std::make_shared<RoutingContext>(RoutingContext::Options{}, dst_engine);
  RemoteKeySource::Options opts{
      .comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{dst_engine},
      .memory_keys = {tensor_key},
      .buffer_sizes = {kArtifactBytes},
      .ip = "127.0.0.1",
      .port = static_cast<uint16_t>(direct_src_port),
      .local_endpoint_id = local_endpoint,
      .remote_endpoint_id = remote_endpoint,
      .routing_context = routing_context,
      .total_size = kArtifactBytes,
  };
  RemoteKeySource source(std::move(opts));

  std::vector<uint8_t> first_read(kArtifactBytes, 0);
  auto first_or = source.read_at(/*offset=*/0, first_read.data(), first_read.size());
  REQUIRE(first_or.ok());
  REQUIRE(*first_or == first_read.size());
  REQUIRE(first_read == direct_payload);

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
          .port = static_cast<uint16_t>(routed_src_port),
      },
  };
  REQUIRE(routing_context->set_endpoint_bindings(std::move(bindings)).ok());

  std::vector<uint8_t> second_read(kArtifactBytes, 0);
  auto second_or = source.read_at(/*offset=*/0, second_read.data(), second_read.size());
  REQUIRE(second_or.ok());
  REQUIRE(*second_or == second_read.size());
  REQUIRE(second_read == direct_payload);
  CHECK(second_read != routed_payload);
}

TEST_CASE(
    "RemoteKeySource direct write reports capability miss when RDMA is disabled",
    "[store][p2p][routing][direct]") {
  constexpr std::size_t kArtifactBytes = 4096;
  const int dst_port = require_available_port_or_skip(55000);

  auto dst_engine = make_engine_on_port(dst_port, /*enable_rdma=*/false);
  RemoteKeySource source(
      RemoteKeySource::Options{
          .comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{dst_engine},
          .memory_keys = {"unused"},
          .buffer_sizes = {kArtifactBytes},
          .ip = "127.0.0.1",
          .port = 1,
          .total_size = kArtifactBytes,
      });

  std::vector<uint8_t> sink_buffer(kArtifactBytes, 0);
  auto grant = make_direct_write_grant(sink_buffer);
  const std::vector<tensorcast::store::loader::DirectWriteOp> ops = {
      tensorcast::store::loader::DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 512},
  };

  auto wrote_or = source.readv_into_at(ops, grant);
  REQUIRE_FALSE(wrote_or.ok());
  REQUIRE(wrote_or.status().code() == absl::StatusCode::kUnimplemented);
  REQUIRE(sink_buffer == std::vector<uint8_t>(sink_buffer.size(), 0));
}

TEST_CASE("RemoteKeySource direct write fails route freeze before issue", "[store][p2p][routing][direct]") {
  constexpr std::size_t kArtifactBytes = 4096;
  const int dst_port = require_available_port_or_skip(56000);

  auto dst_engine = make_engine_on_port(dst_port, /*enable_rdma=*/true);
  auto routing_context = std::make_shared<RoutingContext>(RoutingContext::Options{}, dst_engine);
  RemoteKeySource source(
      RemoteKeySource::Options{
          .comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{dst_engine},
          .memory_keys = {"unused"},
          .buffer_sizes = {kArtifactBytes},
          .ip = "127.0.0.1",
          .port = 1,
          .local_endpoint_id = "node_local/dev/cpu/0",
          .remote_endpoint_id = "node_remote/dev/cpu/0",
          .routing_context = routing_context,
          .total_size = kArtifactBytes,
      });

  std::vector<uint8_t> sink_buffer(kArtifactBytes, 0);
  auto grant = make_direct_write_grant(sink_buffer);

  auto wrote_or = source.read_into_at(/*src_offset=*/0, /*dest_va_offset=*/0, /*bytes=*/512, grant);
  REQUIRE_FALSE(wrote_or.ok());
  REQUIRE(wrote_or.status().code() == absl::StatusCode::kFailedPrecondition);
  REQUIRE(std::string(wrote_or.status().message()).find("freeze failed") != std::string::npos);
  REQUIRE(sink_buffer == std::vector<uint8_t>(sink_buffer.size(), 0));
}

TEST_CASE(
    "RemoteKeySource batched direct write lowers a logical ReadPlan and falls back before issue",
    "[store][p2p][routing][direct]") {
  constexpr std::size_t kChunkBytes = 32;
  const int port = require_available_port_or_skip(56500);

  auto engine = make_engine_on_port(port, /*enable_rdma=*/false);
  std::vector<uint8_t> payload0 = create_test_pattern(kChunkBytes, /*seed=*/5);
  std::vector<uint8_t> payload1 = create_test_pattern(kChunkBytes, /*seed=*/17);
  register_cpu_tensor(engine, "tensor-part-0", payload0);
  register_cpu_tensor(engine, "tensor-part-1", payload1);

  auto routing_context = std::make_shared<RoutingContext>(RoutingContext::Options{}, engine);
  REQUIRE(routing_context->set_topology(build_local_fabric_topology()).ok());
  REQUIRE(routing_context
              ->set_endpoint_bindings({
                  EndpointBinding{.endpoint_id = "nvlink0", .node_id = "node0"},
                  EndpointBinding{.endpoint_id = "nvlink1", .node_id = "node0"},
              })
              .ok());

  RemoteKeySource source(
      RemoteKeySource::Options{
          .comm_engine = gsl::not_null<std::shared_ptr<Communicator>>{engine},
          .memory_keys = {"tensor-part-0", "tensor-part-1"},
          .buffer_sizes = {kChunkBytes, kChunkBytes},
          .local_endpoint_id = "nvlink0",
          .remote_endpoint_id = "nvlink1",
          .routing_context = routing_context,
          .total_size = 2 * kChunkBytes,
          .artifact_id = "artifact-plan-fallback",
          .authority_id = "authority-plan-fallback",
      });

  REQUIRE(source.supports_batched_direct_write_at());

  std::vector<uint8_t> sink_buffer(64, 0);
  DirectWriteGrant grant;
  grant.windows.push_back(
      DirectWriteGrant::Window{
          .va_offset = 0,
          .local_addr = reinterpret_cast<uint64_t>(sink_buffer.data()),
          .length = 16,
      });
  grant.windows.push_back(
      DirectWriteGrant::Window{
          .va_offset = 16,
          .local_addr = reinterpret_cast<uint64_t>(sink_buffer.data() + 16),
          .length = sink_buffer.size() - 16,
      });

  const std::vector<tensorcast::store::loader::DirectWriteOp> ops = {
      tensorcast::store::loader::DirectWriteOp{.src_offset = 8, .dest_va_offset = 8, .bytes = 24},
      tensorcast::store::loader::DirectWriteOp{.src_offset = 24, .dest_va_offset = 32, .bytes = 24},
  };

  auto wrote_or = source.readv_into_at(ops, grant);
  REQUIRE(wrote_or.ok());
  REQUIRE(*wrote_or == 48);

  std::vector<uint8_t> logical_source;
  logical_source.insert(logical_source.end(), payload0.begin(), payload0.end());
  logical_source.insert(logical_source.end(), payload1.begin(), payload1.end());

  std::vector<uint8_t> expected(sink_buffer.size(), 0);
  std::copy(logical_source.begin() + 8, logical_source.begin() + 32, expected.begin() + 8);
  std::copy(logical_source.begin() + 24, logical_source.begin() + 48, expected.begin() + 32);
  CHECK(sink_buffer == expected);
}
