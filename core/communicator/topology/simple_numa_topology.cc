// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/topology/simple_numa_topology.h"

#include <format>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"

namespace tensorcast::communicator::topology {
namespace {

std::string cpu_pool_id(int node_id) {
  return std::format("cpu{}", node_id);
}

std::string gpu_pool_id(int gpu_id) {
  return std::format("gpu{}", gpu_id);
}

} // namespace

absl::StatusOr<Topology> BuildSwitchTopologyFromSimpleNuma(
    const tensorcast::communicator::v1::CommunicatorConfig& config,
    const SimpleNumaTopologyOptions& options) {
  const auto& simple = config.simple_numa();
  if (!simple.enable()) {
    return absl::InvalidArgumentError("simple_numa is disabled");
  }
  if (simple.nodes().empty()) {
    return absl::InvalidArgumentError("simple_numa.nodes is empty");
  }
  if (options.switch_id.empty()) {
    return absl::InvalidArgumentError("switch_id must be non-empty");
  }

  std::vector<Pool> pools;
  pools.reserve(static_cast<size_t>(simple.nodes().size()));

  absl::flat_hash_set<int> seen_nodes;
  absl::flat_hash_set<int> seen_gpus;
  for (const auto& node : simple.nodes()) {
    if (!seen_nodes.insert(node.id()).second) {
      return absl::InvalidArgumentError(std::format("duplicate simple_numa node id: {}", node.id()));
    }
    pools.push_back(Pool{cpu_pool_id(node.id()), cpu_pool_id(node.id()), PoolType::kCpu});
    for (int gpu_id : node.gpus()) {
      if (seen_gpus.insert(gpu_id).second) {
        pools.push_back(Pool{gpu_pool_id(gpu_id), gpu_pool_id(gpu_id), PoolType::kGpu});
      }
    }
  }

  std::vector<Endpoint> endpoints;
  std::vector<Link> links;
  absl::flat_hash_set<std::string> seen_nics;
  size_t nic_count = 0;

  for (const auto& node : simple.nodes()) {
    if (node.gpus().empty()) {
      return absl::InvalidArgumentError(
          std::format("simple_numa node {} has no gpus", node.id()));
    }

    absl::flat_hash_set<std::string> pool_ids_seen;
    std::vector<std::string> pool_ids;
    pool_ids.reserve(static_cast<size_t>(1 + node.gpus().size()));

    const std::string cpu_id = cpu_pool_id(node.id());
    pool_ids_seen.insert(cpu_id);
    pool_ids.push_back(cpu_id);

    for (int gpu_id : node.gpus()) {
      const std::string gpu_id_str = gpu_pool_id(gpu_id);
      if (pool_ids_seen.insert(gpu_id_str).second) {
        pool_ids.push_back(gpu_id_str);
      }
    }

    for (const auto& nic : node.nics()) {
      if (nic.empty()) {
        return absl::InvalidArgumentError(
            std::format("simple_numa node {} has empty nic name", node.id()));
      }
      const std::string endpoint_id = std::format("nic_{}", nic);
      if (!seen_nics.insert(endpoint_id).second) {
        return absl::InvalidArgumentError(std::format("duplicate nic name: {}", nic));
      }

      Endpoint endpoint;
      endpoint.id = endpoint_id;
      endpoint.name = nic;
      endpoint.kind = EndpointKind::kClient;
      endpoint.type = EndpointType::kNic;
      endpoint.pool_ids = pool_ids;
      endpoints.push_back(std::move(endpoint));

      Link link;
      link.id = std::format("{}_to_{}", endpoint_id, options.switch_id);
      link.name = link.id;
      link.type = LinkType::kSwitch;
      link.src_endpoint_id = endpoint_id;
      link.dst_endpoint_id = options.switch_id;
      links.push_back(std::move(link));

      nic_count += 1;
    }
  }

  if (nic_count == 0) {
    return absl::InvalidArgumentError("simple_numa has no nics");
  }
  if (seen_nics.contains(options.switch_id)) {
    return absl::InvalidArgumentError(
        std::format("switch_id collides with nic endpoint id: {}", options.switch_id));
  }

  Endpoint switch_endpoint;
  switch_endpoint.id = options.switch_id;
  switch_endpoint.name = options.switch_name.empty() ? options.switch_id : options.switch_name;
  switch_endpoint.kind = EndpointKind::kSwitch;
  switch_endpoint.type = EndpointType::kNic;
  endpoints.push_back(std::move(switch_endpoint));

  ValidationOptions validation;
  validation.require_endpoint_links = true;
  validation.require_connected = options.require_connected;

  return Topology::Build(
      std::move(pools),
      std::move(endpoints),
      std::move(links),
      validation);
}

} // namespace tensorcast::communicator::topology
