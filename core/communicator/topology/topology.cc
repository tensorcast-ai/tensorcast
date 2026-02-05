// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/topology/topology.h"

#include <format>
#include <queue>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"

namespace tensorcast::communicator::topology {
namespace {

std::string join_ids(const std::vector<std::string>& ids) {
  std::string output;
  bool first = true;
  for (const auto& id : ids) {
    if (!first) {
      output += ",";
    }
    output += id;
    first = false;
  }
  return output;
}

std::string escape_dot(std::string_view value) {
  std::string output;
  output.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output.push_back(ch);
        break;
    }
  }
  return output;
}

std::string join_label_lines(const std::vector<std::string>& lines) {
  std::string output;
  bool first = true;
  for (const auto& line : lines) {
    if (!first) {
      output += "\\n";
    }
    output += line;
    first = false;
  }
  return output;
}

} // namespace

std::string_view to_string(PoolType type) {
  switch (type) {
    case PoolType::kCpu:
      return "CPU";
    case PoolType::kGpu:
      return "GPU";
    case PoolType::kUnspecified:
    default:
      return "UNSPECIFIED";
  }
}

std::string_view to_string(EndpointKind kind) {
  switch (kind) {
    case EndpointKind::kClient:
      return "CLIENT";
    case EndpointKind::kSwitch:
      return "SWITCH";
    default:
      return "UNKNOWN";
  }
}

std::string_view to_string(EndpointType type) {
  switch (type) {
    case EndpointType::kNic:
      return "NIC";
    case EndpointType::kPcie:
      return "PCIE";
    case EndpointType::kNvlink:
      return "NVLINK";
    case EndpointType::kUnspecified:
    default:
      return "UNSPECIFIED";
  }
}

std::string_view to_string(LinkType type) {
  switch (type) {
    case LinkType::kForward:
      return "FORWARD";
    case LinkType::kP2P:
      return "P2P";
    case LinkType::kSwitch:
      return "SW";
    default:
      return "UNKNOWN";
  }
}

const Pool* Topology::find_pool(std::string_view id) const {
  auto it = pools_.find(std::string{id});
  if (it == pools_.end()) {
    return nullptr;
  }
  return &it->second;
}

const Endpoint* Topology::find_endpoint(std::string_view id) const {
  auto it = endpoints_.find(std::string{id});
  if (it == endpoints_.end()) {
    return nullptr;
  }
  return &it->second;
}

const Link* Topology::find_link(std::string_view id) const {
  auto it = links_.find(std::string{id});
  if (it == links_.end()) {
    return nullptr;
  }
  return &it->second;
}

absl::StatusOr<Topology> Topology::Build(
    std::vector<Pool> pools,
    std::vector<Endpoint> endpoints,
    std::vector<Link> links,
    ValidationOptions options) {
  Topology topology;

  for (auto& pool : pools) {
    if (pool.id.empty()) {
      return absl::InvalidArgumentError("pool id must be non-empty");
    }
    const std::string pool_id = pool.id;
    auto [_, inserted] = topology.pools_.emplace(pool_id, std::move(pool));
    if (!inserted) {
      return absl::InvalidArgumentError(std::format("duplicate pool id: {}", pool_id));
    }
  }

  for (auto& endpoint : endpoints) {
    if (endpoint.id.empty()) {
      return absl::InvalidArgumentError("endpoint id must be non-empty");
    }
    const std::string endpoint_id = endpoint.id;
    auto [_, inserted] = topology.endpoints_.emplace(endpoint_id, std::move(endpoint));
    if (!inserted) {
      return absl::InvalidArgumentError(std::format("duplicate endpoint id: {}", endpoint_id));
    }
  }

  for (auto& link : links) {
    if (link.id.empty()) {
      return absl::InvalidArgumentError("link id must be non-empty");
    }
    const std::string link_id = link.id;
    auto [_, inserted] = topology.links_.emplace(link_id, std::move(link));
    if (!inserted) {
      return absl::InvalidArgumentError(std::format("duplicate link id: {}", link_id));
    }
  }

  absl::Status status = topology.validate(options);
  if (!status.ok()) {
    return status;
  }

  return topology;
}

absl::Status Topology::validate(ValidationOptions options) const {
  if (endpoints_.empty()) {
    return absl::InvalidArgumentError("topology requires at least one endpoint");
  }

  for (const auto& [pool_id, pool] : pools_) {
    if (pool_id.empty()) {
      return absl::InvalidArgumentError("pool id must be non-empty");
    }
    if (pool.type == PoolType::kUnspecified) {
      return absl::InvalidArgumentError(std::format("pool type unspecified: {}", pool_id));
    }
  }

  for (const auto& [endpoint_id, endpoint] : endpoints_) {
    if (endpoint_id.empty()) {
      return absl::InvalidArgumentError("endpoint id must be non-empty");
    }
    if (endpoint.type == EndpointType::kUnspecified) {
      return absl::InvalidArgumentError(std::format("endpoint type unspecified: {}", endpoint_id));
    }
    if (endpoint.bandwidth_gbps < 0.0) {
      return absl::InvalidArgumentError(std::format("endpoint bandwidth must be >= 0: {}", endpoint_id));
    }

    if (endpoint.kind == EndpointKind::kSwitch) {
      if (!endpoint.pool_ids.empty()) {
        return absl::InvalidArgumentError(
            std::format("switch endpoint must not bind to pool: {}", endpoint_id));
      }
      continue;
    }

    bool require_cpu_pool = false;
    bool require_gpu_pool = false;
    bool forbid_cpu_pool = false;
    bool forbid_gpu_pool = false;
    int cpu_pool_count = 0;
    int gpu_pool_count = 0;
    std::string first_cpu_pool;
    std::string first_gpu_pool;

    absl::flat_hash_set<std::string> seen_pool_ids;
    seen_pool_ids.reserve(endpoint.pool_ids.size());

    for (const auto& pool_id : endpoint.pool_ids) {
      if (pool_id.empty()) {
        return absl::InvalidArgumentError(
            std::format("endpoint pool id must be non-empty: {}", endpoint_id));
      }
      if (!seen_pool_ids.insert(pool_id).second) {
        return absl::InvalidArgumentError(
            std::format("endpoint has duplicate pool id: {} -> {}", endpoint_id, pool_id));
      }

      auto pool_it = pools_.find(pool_id);
      if (pool_it == pools_.end()) {
        return absl::InvalidArgumentError(
            std::format("endpoint references unknown pool: {} -> {}", endpoint_id, pool_id));
      }

      switch (pool_it->second.type) {
        case PoolType::kCpu:
          cpu_pool_count += 1;
          if (first_cpu_pool.empty()) {
            first_cpu_pool = pool_id;
          }
          break;
        case PoolType::kGpu:
          gpu_pool_count += 1;
          if (first_gpu_pool.empty()) {
            first_gpu_pool = pool_id;
          }
          break;
        case PoolType::kUnspecified:
        default:
          return absl::InvalidArgumentError(
              std::format("endpoint pool type unspecified: {} -> {}", endpoint_id, pool_id));
      }
    }

    switch (endpoint.type) {
      case EndpointType::kNic:
      case EndpointType::kPcie:
        require_cpu_pool = true;
        require_gpu_pool = true;
        break;
      case EndpointType::kNvlink:
        require_gpu_pool = true;
        forbid_cpu_pool = true;
        break;
      case EndpointType::kUnspecified:
      default:
        return absl::InvalidArgumentError(
            std::format("client endpoint type unsupported: {}", endpoint_id));
    }

    if (require_cpu_pool && cpu_pool_count == 0) {
      return absl::InvalidArgumentError(
          std::format("client endpoint requires at least one CPU pool: {}", endpoint_id));
    }
    if (forbid_cpu_pool && cpu_pool_count > 0) {
      return absl::InvalidArgumentError(
          std::format("client endpoint must not bind CPU pools: {} -> {}", endpoint_id, first_cpu_pool));
    }
    if (require_gpu_pool && gpu_pool_count == 0) {
      return absl::InvalidArgumentError(
          std::format("client endpoint requires at least one GPU pool: {}", endpoint_id));
    }
    if (forbid_gpu_pool && gpu_pool_count > 0) {
      return absl::InvalidArgumentError(
          std::format("client endpoint must not bind GPU pools: {} -> {}", endpoint_id, first_gpu_pool));
    }
  }

  absl::flat_hash_map<std::string, int> link_degree;
  if (options.require_endpoint_links) {
    link_degree.reserve(endpoints_.size());
    for (const auto& [endpoint_id, endpoint] : endpoints_) {
      link_degree.emplace(endpoint_id, 0);
    }
  }

  for (const auto& [link_id, link] : links_) {
    if (link_id.empty()) {
      return absl::InvalidArgumentError("link id must be non-empty");
    }
    if (link.src_endpoint_id.empty() || link.dst_endpoint_id.empty()) {
      return absl::InvalidArgumentError(std::format("link endpoints must be non-empty: {}", link_id));
    }
    if (link.src_endpoint_id == link.dst_endpoint_id) {
      return absl::InvalidArgumentError(std::format("link endpoints must differ: {}", link_id));
    }
    if (link.bandwidth_gbps < 0.0) {
      return absl::InvalidArgumentError(std::format("link bandwidth must be >= 0: {}", link_id));
    }
    if (link.latency_us < 0.0) {
      return absl::InvalidArgumentError(std::format("link latency must be >= 0: {}", link_id));
    }

    auto src_it = endpoints_.find(link.src_endpoint_id);
    if (src_it == endpoints_.end()) {
      return absl::InvalidArgumentError(std::format("link source endpoint not found: {}", link_id));
    }
    auto dst_it = endpoints_.find(link.dst_endpoint_id);
    if (dst_it == endpoints_.end()) {
      return absl::InvalidArgumentError(std::format("link destination endpoint not found: {}", link_id));
    }

    const Endpoint& src = src_it->second;
    const Endpoint& dst = dst_it->second;
    const bool src_switch = src.kind == EndpointKind::kSwitch;
    const bool dst_switch = dst.kind == EndpointKind::kSwitch;

    if (link.type == LinkType::kSwitch) {
      if (!(src_switch || dst_switch)) {
        return absl::InvalidArgumentError(
            std::format("SW link must include switch endpoint: {}", link_id));
      }
    } else {
      if (src_switch || dst_switch) {
        return absl::InvalidArgumentError(
            std::format("non-SW link cannot include switch endpoint: {}", link_id));
      }
    }

    if (src.type != dst.type) {
      return absl::InvalidArgumentError(
          std::format(
              "link endpoint types must match: {} ({}:{} vs {}:{})",
              link_id,
              link.src_endpoint_id,
              to_string(src.type),
              link.dst_endpoint_id,
              to_string(dst.type)));
    }

    if (options.require_endpoint_links) {
      link_degree[link.src_endpoint_id] += 1;
      link_degree[link.dst_endpoint_id] += 1;
    }
  }

  if (options.require_endpoint_links) {
    for (const auto& [endpoint_id, degree] : link_degree) {
      if (degree == 0) {
        return absl::InvalidArgumentError(
            std::format("endpoint must have at least one link: {}", endpoint_id));
      }
    }
  }

  if (options.require_connected) {
    absl::flat_hash_map<std::string, std::vector<std::string>> adjacency;
    adjacency.reserve(endpoints_.size());
    for (const auto& [endpoint_id, endpoint] : endpoints_) {
      adjacency.emplace(endpoint_id, std::vector<std::string>{});
    }
    for (const auto& [link_id, link] : links_) {
      adjacency[link.src_endpoint_id].push_back(link.dst_endpoint_id);
      adjacency[link.dst_endpoint_id].push_back(link.src_endpoint_id);
    }

    absl::flat_hash_set<std::string> visited;
    visited.reserve(endpoints_.size());

    std::queue<std::string> pending;
    pending.push(endpoints_.begin()->first);
    visited.insert(endpoints_.begin()->first);

    while (!pending.empty()) {
      std::string current = std::move(pending.front());
      pending.pop();
      const auto& neighbors = adjacency[current];
      for (const auto& neighbor : neighbors) {
        if (visited.insert(neighbor).second) {
          pending.push(neighbor);
        }
      }
    }

    if (visited.size() != endpoints_.size()) {
      return absl::InvalidArgumentError("topology graph is disconnected");
    }
  }

  return absl::OkStatus();
}

std::string Topology::to_dot() const {
  std::string output;
  output += "digraph Topology {\n";

  for (const auto& [endpoint_id, endpoint] : endpoints_) {
    std::vector<std::string> label_lines;
    label_lines.push_back(
        escape_dot(endpoint.name.empty() ? endpoint_id : endpoint.name));
    label_lines.push_back(
        escape_dot(std::format("{} {}", to_string(endpoint.kind), to_string(endpoint.type))));
    std::vector<std::string> cpu_pools;
    std::vector<std::string> gpu_pools;
    std::vector<std::string> other_pools;
    cpu_pools.reserve(endpoint.pool_ids.size());
    gpu_pools.reserve(endpoint.pool_ids.size());
    other_pools.reserve(endpoint.pool_ids.size());

    for (const auto& pool_id : endpoint.pool_ids) {
      auto pool_it = pools_.find(pool_id);
      if (pool_it == pools_.end()) {
        other_pools.push_back(pool_id);
        continue;
      }
      switch (pool_it->second.type) {
        case PoolType::kCpu:
          cpu_pools.push_back(pool_id);
          break;
        case PoolType::kGpu:
          gpu_pools.push_back(pool_id);
          break;
        case PoolType::kUnspecified:
        default:
          other_pools.push_back(pool_id);
          break;
      }
    }

    if (!cpu_pools.empty()) {
      label_lines.push_back(
          escape_dot(std::format("cpu_pools={}", join_ids(cpu_pools))));
    }
    if (!gpu_pools.empty()) {
      label_lines.push_back(
          escape_dot(std::format("gpu_pools={}", join_ids(gpu_pools))));
    }
    if (!other_pools.empty()) {
      label_lines.push_back(
          escape_dot(std::format("pools={}", join_ids(other_pools))));
    }
    if (endpoint.bandwidth_gbps > 0.0) {
      label_lines.push_back(
          escape_dot(std::format("bw={}Gbps", endpoint.bandwidth_gbps)));
    }
    const std::string label = join_label_lines(label_lines);
    output += std::format(
        "  \"{}\" [label=\"{}\"];\n",
        escape_dot(endpoint_id),
        label);
  }

  for (const auto& [link_id, link] : links_) {
    std::vector<std::string> label_lines;
    label_lines.push_back(escape_dot(link.name.empty() ? link_id : link.name));
    label_lines.push_back(escape_dot(std::format("{}", to_string(link.type))));
    if (link.bandwidth_gbps > 0.0) {
      label_lines.push_back(
          escape_dot(std::format("bw={}Gbps", link.bandwidth_gbps)));
    }
    if (link.latency_us > 0.0) {
      label_lines.push_back(
          escape_dot(std::format("lat={}us", link.latency_us)));
    }
    const std::string label = join_label_lines(label_lines);
    output += std::format(
        "  \"{}\" -> \"{}\" [label=\"{}\"];\n",
        escape_dot(link.src_endpoint_id),
        escape_dot(link.dst_endpoint_id),
        label);
  }

  output += "}\n";
  return output;
}

} // namespace tensorcast::communicator::topology
