// Copyright (c) 2026, TensorCast Team.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/communicator/topology/topology.h"

namespace {

using tensorcast::communicator::topology::Endpoint;
using tensorcast::communicator::topology::EndpointKind;
using tensorcast::communicator::topology::EndpointType;
using tensorcast::communicator::topology::Link;
using tensorcast::communicator::topology::LinkType;
using tensorcast::communicator::topology::Pool;
using tensorcast::communicator::topology::PoolType;
using tensorcast::communicator::topology::Topology;
using tensorcast::communicator::topology::ValidationOptions;

constexpr std::string_view kDotEnv = "TENSORCAST_TOPOLOGY_DOT_STDOUT";
constexpr std::string_view kTestOutputsEnv = "TEST_UNDECLARED_OUTPUTS_DIR";

std::string env_value(std::string_view name) {
  std::string key(name);
  const char* value = std::getenv(key.c_str());
  if (value == nullptr) {
    return {};
  }
  return std::string(value);
}

bool env_enabled(std::string_view name) {
  std::string value = env_value(name);
  if (value.empty()) {
    return false;
  }
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

std::string dot_output_path(std::string_view label) {
  std::string output_dir = env_value(kTestOutputsEnv);
  if (output_dir.empty()) {
    return {};
  }
  if (output_dir.back() != '/') {
    output_dir.push_back('/');
  }
  return output_dir + "topology_" + std::string(label) + ".dot";
}

void maybe_emit_dot(std::string_view label, std::string_view dot) {
  const bool to_stdout = env_enabled(kDotEnv);
  const std::string output_path = dot_output_path(label);
  if (!to_stdout && output_path.empty()) {
    return;
  }
  if (to_stdout) {
    std::cout << "\n=== " << label << " ===\n";
    std::cout << dot << '\n';
  }
  if (!output_path.empty()) {
    std::ofstream output(output_path);
    if (!output) {
      std::cerr << "Failed to write topology DOT to " << output_path << '\n';
      return;
    }
    output << dot;
  }
}

std::string make_id(std::string_view prefix, int index) {
  return std::string(prefix) + std::to_string(index);
}

} // namespace

TEST_CASE("Topology rail-optimized 8-GPU/8-NIC/2-CPU NVLINK layout", "[communicator][topology]") {
  std::vector<Pool> pools;
  for (int cpu = 0; cpu < 2; ++cpu) {
    pools.push_back(Pool{make_id("cpu", cpu), make_id("cpu", cpu), PoolType::kCpu});
  }
  for (int gpu = 0; gpu < 8; ++gpu) {
    pools.push_back(Pool{make_id("gpu", gpu), make_id("gpu", gpu), PoolType::kGpu});
  }

  std::vector<Endpoint> endpoints;
  for (int nic = 0; nic < 8; ++nic) {
    Endpoint endpoint;
    endpoint.id = make_id("nic", nic);
    endpoint.name = endpoint.id;
    endpoint.kind = EndpointKind::kClient;
    endpoint.type = EndpointType::kNic;
    endpoint.pool_ids = {make_id("cpu", nic / 4), make_id("gpu", nic)};
    endpoint.bandwidth_gbps = 200.0;
    endpoints.push_back(std::move(endpoint));
  }
  for (int gpu = 0; gpu < 8; ++gpu) {
    Endpoint endpoint;
    endpoint.id = make_id("nvlink", gpu);
    endpoint.name = endpoint.id;
    endpoint.kind = EndpointKind::kClient;
    endpoint.type = EndpointType::kNvlink;
    endpoint.pool_ids = {make_id("gpu", gpu)};
    endpoint.bandwidth_gbps = 900.0;
    endpoints.push_back(std::move(endpoint));
  }
  Endpoint nvswitch;
  nvswitch.id = "nvsw0";
  nvswitch.name = "nvsw0";
  nvswitch.kind = EndpointKind::kSwitch;
  nvswitch.type = EndpointType::kNvlink;
  endpoints.push_back(std::move(nvswitch));
  Endpoint netswitch;
  netswitch.id = "netsw0";
  netswitch.name = "netsw0";
  netswitch.kind = EndpointKind::kSwitch;
  netswitch.type = EndpointType::kNic;
  endpoints.push_back(std::move(netswitch));

  std::vector<Link> links;
  for (int gpu = 0; gpu < 8; ++gpu) {
    Link nic_sw;
    nic_sw.id = "nic" + std::to_string(gpu) + "_to_netsw0";
    nic_sw.name = nic_sw.id;
    nic_sw.type = LinkType::kSwitch;
    nic_sw.src_endpoint_id = make_id("nic", gpu);
    nic_sw.dst_endpoint_id = "netsw0";
    nic_sw.bandwidth_gbps = 200.0;
    nic_sw.latency_us = 2.0;
    links.push_back(std::move(nic_sw));

    Link nvlink_sw;
    nvlink_sw.id = "nvlink" + std::to_string(gpu) + "_to_nvsw0";
    nvlink_sw.name = nvlink_sw.id;
    nvlink_sw.type = LinkType::kSwitch;
    nvlink_sw.src_endpoint_id = make_id("nvlink", gpu);
    nvlink_sw.dst_endpoint_id = "nvsw0";
    nvlink_sw.bandwidth_gbps = 900.0;
    nvlink_sw.latency_us = 1.0;
    links.push_back(std::move(nvlink_sw));
  }

  ValidationOptions options;
  options.require_endpoint_links = true;
  options.require_connected = false; // NVLINK and network fabrics are modeled as separate components.

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      options);
  INFO(topology_or.status());
  REQUIRE(topology_or.ok());
  const Topology& topology = topology_or.value();

  CHECK(topology.pools().size() == 10);
  CHECK(topology.endpoints().size() == 18);
  CHECK(topology.links().size() == 16);
  CHECK(topology.find_endpoint("nvsw0") != nullptr);
  CHECK(topology.find_endpoint("netsw0") != nullptr);
  const std::string dot = topology.to_dot();
  CHECK_FALSE(dot.empty());
  maybe_emit_dot("rail_optimized_nvlink", dot);
}

TEST_CASE("Topology no-rail 4-GPU/1-NIC/1-CPU layout", "[communicator][topology]") {
  std::vector<Pool> pools;
  pools.push_back(Pool{"cpu0", "cpu0", PoolType::kCpu});
  for (int gpu = 0; gpu < 4; ++gpu) {
    pools.push_back(Pool{make_id("gpu", gpu), make_id("gpu", gpu), PoolType::kGpu});
  }

  std::vector<Endpoint> endpoints;
  Endpoint nic;
  nic.id = "nic0";
  nic.name = "nic0";
  nic.kind = EndpointKind::kClient;
  nic.type = EndpointType::kNic;
  nic.pool_ids = {"cpu0", "gpu0", "gpu1", "gpu2", "gpu3"};
  nic.bandwidth_gbps = 200.0;
  endpoints.push_back(std::move(nic));
  Endpoint netswitch;
  netswitch.id = "netsw0";
  netswitch.name = "netsw0";
  netswitch.kind = EndpointKind::kSwitch;
  netswitch.type = EndpointType::kNic;
  endpoints.push_back(std::move(netswitch));

  std::vector<Link> links;
  Link nic_sw;
  nic_sw.id = "nic0_to_netsw0";
  nic_sw.name = nic_sw.id;
  nic_sw.type = LinkType::kSwitch;
  nic_sw.src_endpoint_id = "nic0";
  nic_sw.dst_endpoint_id = "netsw0";
  nic_sw.bandwidth_gbps = 200.0;
  nic_sw.latency_us = 2.0;
  links.push_back(std::move(nic_sw));

  ValidationOptions options;
  options.require_endpoint_links = true;
  options.require_connected = true;

  auto topology_or = Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      options);
  INFO(topology_or.status());
  REQUIRE(topology_or.ok());
  const Topology& topology = topology_or.value();

  CHECK(topology.pools().size() == 5);
  CHECK(topology.endpoints().size() == 2);
  CHECK(topology.links().size() == 1);
  CHECK(topology.find_endpoint("netsw0") != nullptr);
  const std::string dot = topology.to_dot();
  CHECK_FALSE(dot.empty());
  maybe_emit_dot("no_rail_single_nic", dot);
}
