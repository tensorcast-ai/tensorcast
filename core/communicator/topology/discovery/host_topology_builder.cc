// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/topology/discovery/host_topology_builder.h"

#include <algorithm>
#include <format>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/communicator/topology/discovery/lldp_source.h"
#include "core/communicator/topology/discovery/nvlink_source.h"

namespace tc = tensorcast::communicator::v1;

namespace tensorcast::communicator::topology::discovery {
namespace {

struct BaselineBuildResult {
  std::vector<Pool> pools;
  std::vector<Endpoint> endpoints;
  absl::flat_hash_map<int, std::string> gpu_pool_by_index;
};

std::string cpu_pool_id(int node_id) {
  return std::format("cpu{}", node_id);
}

std::string gpu_pool_id(int gpu_id) {
  return std::format("gpu{}", gpu_id);
}

std::string nic_endpoint_id(const std::string& nic_name) {
  return std::format("nic_{}", nic_name);
}

bool source_is_enabled(tc::NvlinkDiscoveryConfig::Source source) {
  return source == tc::NvlinkDiscoveryConfig::SOURCE_SNAPSHOT_FILE ||
         source == tc::NvlinkDiscoveryConfig::SOURCE_RUNTIME_PROBE;
}

std::string nvlink_source_to_string(tc::NvlinkDiscoveryConfig::Source source) {
  switch (source) {
    case tc::NvlinkDiscoveryConfig::SOURCE_DISABLED:
      return "disabled";
    case tc::NvlinkDiscoveryConfig::SOURCE_SNAPSHOT_FILE:
      return "snapshot_file";
    case tc::NvlinkDiscoveryConfig::SOURCE_RUNTIME_PROBE:
      return "runtime_probe";
    case tc::NvlinkDiscoveryConfig::SOURCE_UNSPECIFIED:
      return "unspecified";
    default:
      return "unknown";
  }
}

absl::StatusOr<BaselineBuildResult> build_baseline_from_simple_numa(
    const tc::CommunicatorConfig& config) {
  const auto& simple = config.simple_numa();
  if (!simple.enable()) {
    return absl::InvalidArgumentError("simple_numa is disabled");
  }
  if (simple.nodes().empty()) {
    return absl::InvalidArgumentError("simple_numa.nodes is empty");
  }

  BaselineBuildResult baseline;
  absl::flat_hash_set<int> seen_nodes;
  absl::flat_hash_set<int> seen_gpus;
  absl::flat_hash_set<std::string> seen_nics;

  for (const auto& node : simple.nodes()) {
    if (!seen_nodes.insert(node.id()).second) {
      return absl::InvalidArgumentError(std::format("duplicate simple_numa node id: {}", node.id()));
    }
    if (node.nics().empty()) {
      return absl::InvalidArgumentError(std::format("simple_numa node {} has no nics", node.id()));
    }
    if (node.gpus().empty()) {
      return absl::InvalidArgumentError(std::format("simple_numa node {} has no gpus", node.id()));
    }

    baseline.pools.push_back(Pool{cpu_pool_id(node.id()), cpu_pool_id(node.id()), PoolType::kCpu});

    absl::flat_hash_set<std::string> endpoint_pool_ids_seen;
    std::vector<std::string> endpoint_pool_ids;
    endpoint_pool_ids.reserve(static_cast<size_t>(1 + node.gpus().size()));
    const std::string cpu_id = cpu_pool_id(node.id());
    endpoint_pool_ids_seen.insert(cpu_id);
    endpoint_pool_ids.push_back(cpu_id);

    for (int gpu_id : node.gpus()) {
      if (!seen_gpus.insert(gpu_id).second) {
        return absl::InvalidArgumentError(
            std::format("duplicate simple_numa gpu id: {}", gpu_id));
      }
      const std::string gpu_id_str = gpu_pool_id(gpu_id);
      baseline.pools.push_back(Pool{gpu_id_str, gpu_id_str, PoolType::kGpu});
      baseline.gpu_pool_by_index.emplace(gpu_id, gpu_id_str);
      if (endpoint_pool_ids_seen.insert(gpu_id_str).second) {
        endpoint_pool_ids.push_back(gpu_id_str);
      }
    }

    for (const auto& nic_name : node.nics()) {
      if (nic_name.empty()) {
        return absl::InvalidArgumentError(
            std::format("simple_numa node {} has empty nic name", node.id()));
      }
      const std::string endpoint_id = nic_endpoint_id(nic_name);
      if (!seen_nics.insert(endpoint_id).second) {
        return absl::InvalidArgumentError(std::format("duplicate nic name: {}", nic_name));
      }
      Endpoint endpoint;
      endpoint.id = endpoint_id;
      endpoint.name = nic_name;
      endpoint.kind = EndpointKind::kClient;
      endpoint.type = EndpointType::kNic;
      endpoint.pool_ids = endpoint_pool_ids;
      baseline.endpoints.push_back(std::move(endpoint));
    }
  }

  return baseline;
}

std::string default_lldp_file_path(const tc::CommunicatorConfig& config) {
  const std::string configured = config.topology_discovery().lldp().file_path();
  if (!configured.empty()) {
    return configured;
  }
  return "/host-config/lldp-info.txt";
}

std::string lldp_source_description(const tc::CommunicatorConfig& config) {
  return std::format("file:{}", default_lldp_file_path(config));
}

tc::NvlinkDiscoveryConfig::Source normalized_nvlink_source(const tc::CommunicatorConfig& config) {
  const auto source = config.topology_discovery().nvlink().source();
  if (source == tc::NvlinkDiscoveryConfig::SOURCE_UNSPECIFIED) {
    return tc::NvlinkDiscoveryConfig::SOURCE_DISABLED;
  }
  return source;
}

void add_nvlink_topology(
    std::vector<Endpoint>* endpoints,
    std::vector<Link>* links,
    const NvlinkSnapshot& snapshot,
    const absl::flat_hash_map<int, std::string>& gpu_pool_by_index) {
  absl::flat_hash_map<std::string, int> gpu_index_by_uuid;
  gpu_index_by_uuid.reserve(snapshot.gpus.size());
  for (const auto& gpu : snapshot.gpus) {
    gpu_index_by_uuid[gpu.gpu_uuid] = gpu.gpu_index;
  }

  absl::flat_hash_set<std::string> edge_gpu_uuids;
  edge_gpu_uuids.reserve(snapshot.edges.size() * 2);
  for (const auto& edge : snapshot.edges) {
    const auto src_gpu_it = gpu_index_by_uuid.find(edge.src_gpu_uuid);
    const auto dst_gpu_it = gpu_index_by_uuid.find(edge.dst_gpu_uuid);
    if (src_gpu_it == gpu_index_by_uuid.end() || dst_gpu_it == gpu_index_by_uuid.end()) {
      continue;
    }
    if (!gpu_pool_by_index.contains(src_gpu_it->second) ||
        !gpu_pool_by_index.contains(dst_gpu_it->second)) {
      continue;
    }
    edge_gpu_uuids.insert(edge.src_gpu_uuid);
    edge_gpu_uuids.insert(edge.dst_gpu_uuid);
  }

  if (edge_gpu_uuids.empty()) {
    return;
  }

  absl::flat_hash_map<std::string, std::string> endpoint_id_by_gpu_uuid;
  endpoint_id_by_gpu_uuid.reserve(edge_gpu_uuids.size());

  absl::flat_hash_set<std::string> existing_endpoint_ids;
  existing_endpoint_ids.reserve(endpoints->size());
  for (const auto& endpoint : *endpoints) {
    existing_endpoint_ids.insert(endpoint.id);
  }

  for (const auto& gpu : snapshot.gpus) {
    if (!edge_gpu_uuids.contains(gpu.gpu_uuid)) {
      continue;
    }
    auto pool_it = gpu_pool_by_index.find(gpu.gpu_index);
    if (pool_it == gpu_pool_by_index.end()) {
      VLOG(1) << "Skipping NVLINK GPU not in simple_numa baseline: index=" << gpu.gpu_index;
      continue;
    }

    const std::string endpoint_id = std::format("nvlink_{}", gpu.gpu_uuid);
    endpoint_id_by_gpu_uuid[gpu.gpu_uuid] = endpoint_id;
    if (!existing_endpoint_ids.insert(endpoint_id).second) {
      continue;
    }

    Endpoint endpoint;
    endpoint.id = endpoint_id;
    endpoint.name = endpoint_id;
    endpoint.kind = EndpointKind::kClient;
    endpoint.type = EndpointType::kNvlink;
    endpoint.pool_ids = {pool_it->second};
    endpoints->push_back(std::move(endpoint));
  }

  absl::flat_hash_set<std::string> existing_link_ids;
  existing_link_ids.reserve(links->size());
  for (const auto& link : *links) {
    existing_link_ids.insert(link.id);
  }

  for (const auto& edge : snapshot.edges) {
    auto src_it = endpoint_id_by_gpu_uuid.find(edge.src_gpu_uuid);
    auto dst_it = endpoint_id_by_gpu_uuid.find(edge.dst_gpu_uuid);
    if (src_it == endpoint_id_by_gpu_uuid.end() || dst_it == endpoint_id_by_gpu_uuid.end()) {
      VLOG(1) << "Skipping NVLINK edge due to missing GPU endpoints: "
              << edge.src_gpu_uuid << " <-> " << edge.dst_gpu_uuid;
      continue;
    }

    const std::string src = src_it->second;
    const std::string dst = dst_it->second;
    if (src == dst) {
      continue;
    }

    const std::string forward_id = std::format("{}_to_{}", src, dst);
    const std::string reverse_id = std::format("{}_to_{}", dst, src);

    if (existing_link_ids.insert(forward_id).second) {
      Link forward;
      forward.id = forward_id;
      forward.name = forward_id;
      forward.type = LinkType::kP2P;
      forward.src_endpoint_id = src;
      forward.dst_endpoint_id = dst;
      forward.bandwidth_gbps = edge.bandwidth_hint_gbps;
      links->push_back(std::move(forward));
    }
    if (existing_link_ids.insert(reverse_id).second) {
      Link reverse;
      reverse.id = reverse_id;
      reverse.name = reverse_id;
      reverse.type = LinkType::kP2P;
      reverse.src_endpoint_id = dst;
      reverse.dst_endpoint_id = src;
      reverse.bandwidth_gbps = edge.bandwidth_hint_gbps;
      links->push_back(std::move(reverse));
    }
  }
}

} // namespace

absl::StatusOr<HostTopologyBuildResult> build_topology_from_discovery_with_observability(
    const tc::CommunicatorConfig& config,
    const HostTopologyBuilderOptions& options) {
  auto baseline_or = build_baseline_from_simple_numa(config);
  if (!baseline_or.ok()) {
    return baseline_or.status();
  }
  BaselineBuildResult baseline = std::move(baseline_or).value();

  TopologyDiscoveryObservability observability;
  std::vector<Endpoint> endpoints = std::move(baseline.endpoints);
  std::vector<Link> links;

  const bool discovery_enabled = config.topology_discovery().enable();
  observability.discovery_enabled = discovery_enabled;
  observability.nic_endpoint_count = static_cast<int>(endpoints.size());
  observability.lldp_source = discovery_enabled ? lldp_source_description(config) : "disabled";
  const bool emit_rail_switch_endpoints =
      discovery_enabled
          ? config.topology_discovery().merge_policy().emit_rail_switch_endpoints()
          : false;

  absl::flat_hash_map<std::string, LldpNicRecord> lldp_records;
  if (discovery_enabled) {
    LldpParseOptions parse_options;
    parse_options.strict = options.strict_parsing || config.topology_discovery().lldp().required();
    auto lldp_or = load_lldp_records_by_nic(default_lldp_file_path(config), parse_options);
    if (!lldp_or.ok()) {
      if (config.topology_discovery().lldp().required()) {
        return absl::Status(
            lldp_or.status().code(),
            absl::StrCat("failed to load required LLDP records: ", lldp_or.status().message()));
      }
      observability.lldp_degraded = true;
      observability.lldp_degrade_reason = lldp_or.status().ToString();
      LOG(WARNING) << "LLDP discovery unavailable, degrading to unknown rail: " << lldp_or.status();
    } else {
      lldp_records = std::move(lldp_or).value();
      observability.lldp_record_count = static_cast<int>(lldp_records.size());
    }
  } else {
    observability.lldp_source = "disabled";
  }

  absl::flat_hash_set<std::string> switch_endpoint_ids;
  absl::flat_hash_map<std::string, std::string> switch_id_by_nic_endpoint_id;

  if (emit_rail_switch_endpoints) {
    for (const auto& endpoint : endpoints) {
      if (endpoint.type != EndpointType::kNic || endpoint.kind != EndpointKind::kClient) {
        continue;
      }
      auto lldp_it = lldp_records.find(endpoint.name);
      if (lldp_it == lldp_records.end()) {
        switch_id_by_nic_endpoint_id[endpoint.id] = options.unknown_rail_switch_id;
        switch_endpoint_ids.insert(options.unknown_rail_switch_id);
        continue;
      }
      const std::string switch_id = std::format("netsw_rail_{}", lldp_it->second.rail_id);
      switch_id_by_nic_endpoint_id[endpoint.id] = switch_id;
      switch_endpoint_ids.insert(switch_id);
    }
  } else {
    switch_endpoint_ids.insert(options.default_switch_id);
    for (const auto& endpoint : endpoints) {
      if (endpoint.type != EndpointType::kNic || endpoint.kind != EndpointKind::kClient) {
        continue;
      }
      switch_id_by_nic_endpoint_id[endpoint.id] = options.default_switch_id;
    }
  }

  observability.rail_switch_endpoint_count = static_cast<int>(switch_endpoint_ids.size());
  std::vector<std::string> sorted_switch_ids(switch_endpoint_ids.begin(), switch_endpoint_ids.end());
  std::sort(sorted_switch_ids.begin(), sorted_switch_ids.end());
  for (const std::string& switch_id : sorted_switch_ids) {
    Endpoint switch_endpoint;
    switch_endpoint.id = switch_id;
    switch_endpoint.name = switch_id;
    switch_endpoint.kind = EndpointKind::kSwitch;
    switch_endpoint.type = EndpointType::kNic;
    endpoints.push_back(std::move(switch_endpoint));
  }

  for (const auto& endpoint : endpoints) {
    if (endpoint.type != EndpointType::kNic || endpoint.kind != EndpointKind::kClient) {
      continue;
    }
    auto switch_it = switch_id_by_nic_endpoint_id.find(endpoint.id);
    if (switch_it == switch_id_by_nic_endpoint_id.end()) {
      continue;
    }
    Link link;
    link.id = std::format("{}_to_{}", endpoint.id, switch_it->second);
    link.name = link.id;
    link.type = LinkType::kSwitch;
    link.src_endpoint_id = endpoint.id;
    link.dst_endpoint_id = switch_it->second;
    links.push_back(std::move(link));
  }

  if (discovery_enabled) {
    const auto nvlink_source = normalized_nvlink_source(config);
    observability.nvlink_source = nvlink_source_to_string(nvlink_source);
    if (source_is_enabled(nvlink_source)) {
      const bool nvlink_required = config.topology_discovery().nvlink().required();
      if (nvlink_source == tc::NvlinkDiscoveryConfig::SOURCE_RUNTIME_PROBE) {
        NvlinkRuntimeProbeOptions probe_options = options.nvlink_runtime_probe;
        probe_options.strict = options.strict_parsing || nvlink_required;
        auto snapshot_or = load_nvlink_runtime_probe(probe_options);
        if (!snapshot_or.ok()) {
          if (nvlink_required) {
            return absl::Status(
                snapshot_or.status().code(),
                absl::StrCat(
                    "failed to run required NVLINK runtime probe: ",
                    snapshot_or.status().message()));
          }
          observability.nvlink_degraded = true;
          observability.nvlink_degrade_reason = snapshot_or.status().ToString();
          LOG(WARNING) << "NVLINK runtime probe unavailable, skipping NVLINK links: "
                       << snapshot_or.status();
        } else {
          const NvlinkSnapshot& snapshot = snapshot_or.value();
          observability.nvlink_gpu_count = static_cast<int>(snapshot.gpus.size());
          observability.nvlink_edge_count = static_cast<int>(snapshot.edges.size());
          add_nvlink_topology(&endpoints, &links, snapshot, baseline.gpu_pool_by_index);
        }
      } else {
        const std::string snapshot_file = config.topology_discovery().nvlink().snapshot_file_path();
        if (snapshot_file.empty()) {
          if (nvlink_required) {
            return absl::InvalidArgumentError(
                "nvlink.snapshot_file_path must be set when nvlink.source=SOURCE_SNAPSHOT_FILE");
          }
          observability.nvlink_degraded = true;
          observability.nvlink_degrade_reason = "snapshot source enabled but snapshot_file_path is empty";
          LOG(WARNING) << "NVLINK snapshot source selected but snapshot_file_path is empty; skipping";
        } else {
          NvlinkSnapshotOptions nvlink_options;
          nvlink_options.strict = options.strict_parsing || nvlink_required;
          auto snapshot_or = load_nvlink_snapshot(snapshot_file, nvlink_options);
          if (!snapshot_or.ok()) {
            if (nvlink_required) {
              return absl::Status(
                  snapshot_or.status().code(),
                  absl::StrCat("failed to load required NVLINK snapshot: ", snapshot_or.status().message()));
            }
            observability.nvlink_degraded = true;
            observability.nvlink_degrade_reason = snapshot_or.status().ToString();
            LOG(WARNING) << "NVLINK snapshot unavailable, skipping NVLINK links: " << snapshot_or.status();
          } else {
            observability.nvlink_gpu_count = static_cast<int>(snapshot_or->gpus.size());
            observability.nvlink_edge_count = static_cast<int>(snapshot_or->edges.size());
            add_nvlink_topology(&endpoints, &links, snapshot_or.value(), baseline.gpu_pool_by_index);
          }
        }
      }
    }
  } else {
    observability.nvlink_source = "disabled";
  }

  ValidationOptions validation;
  validation.require_endpoint_links = true;
  validation.require_connected =
      discovery_enabled
          ? config.topology_discovery().merge_policy().require_connected()
          : options.require_connected_when_discovery_disabled;

  auto topology_or = Topology::Build(
      std::move(baseline.pools),
      std::move(endpoints),
      std::move(links),
      validation);
  if (!topology_or.ok()) {
    return topology_or.status();
  }
  return HostTopologyBuildResult{
      .topology = std::move(topology_or).value(),
      .observability = std::move(observability),
  };
}

absl::StatusOr<Topology> build_topology_from_discovery(
    const tc::CommunicatorConfig& config,
    const HostTopologyBuilderOptions& options) {
  auto result_or = build_topology_from_discovery_with_observability(config, options);
  if (!result_or.ok()) {
    return result_or.status();
  }
  return std::move(result_or).value().topology;
}

} // namespace tensorcast::communicator::topology::discovery
