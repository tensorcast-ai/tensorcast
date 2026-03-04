// Copyright (c) 2026, TensorCast Team.

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "core/communicator/config_io.h"
#include "core/communicator/topology/discovery/host_topology_builder.h"
#include "core/communicator/topology/simple_numa_topology.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: simple_numa_topology_tool <communicator_or_daemon_yaml>\n";
    return 1;
  }

  const std::string path = argv[1];
  auto cfg_or = tensorcast::communicator::LoadCommunicatorConfigFromFile(path);
  if (!cfg_or.ok()) {
    std::cerr << "Failed to load communicator config: " << cfg_or.status().ToString() << "\n";
    return 1;
  }

  tensorcast::communicator::topology::Topology topology;
  if (cfg_or->topology_discovery().enable()) {
    auto build_or = tensorcast::communicator::topology::discovery::build_topology_from_discovery_with_observability(
        cfg_or.value());
    if (!build_or.ok()) {
      std::cerr << "Failed to build topology: " << build_or.status().ToString() << "\n";
      return 1;
    }
    const auto& observability = build_or->observability;
    std::cerr << "topology_discovery: enabled=" << std::boolalpha << observability.discovery_enabled
              << " lldp_source=" << observability.lldp_source << " lldp_records=" << observability.lldp_record_count
              << " lldp_degraded=" << observability.lldp_degraded;
    if (!observability.lldp_degrade_reason.empty()) {
      std::cerr << " lldp_degrade_reason=\"" << observability.lldp_degrade_reason << "\"";
    }
    std::cerr << " nvlink_source=" << observability.nvlink_source << " nvlink_gpus=" << observability.nvlink_gpu_count
              << " nvlink_edges=" << observability.nvlink_edge_count
              << " nvlink_degraded=" << observability.nvlink_degraded;
    if (!observability.nvlink_degrade_reason.empty()) {
      std::cerr << " nvlink_degrade_reason=\"" << observability.nvlink_degrade_reason << "\"";
    }
    std::cerr << " affinity_nic_candidates=" << observability.affinity_nic_candidate_count
              << " affinity_nic_scored=" << observability.affinity_nic_scored_count
              << " affinity_nic_narrowed=" << observability.affinity_nic_narrowed_count
              << " affinity_degraded=" << observability.affinity_degraded;
    if (!observability.affinity_degrade_reason.empty()) {
      std::cerr << " affinity_degrade_reason=\"" << observability.affinity_degrade_reason << "\"";
    }
    std::cerr << " nic_endpoints=" << observability.nic_endpoint_count
              << " rail_switch_endpoints=" << observability.rail_switch_endpoint_count << "\n";
    topology = std::move(build_or).value().topology;
  } else {
    auto topology_or = tensorcast::communicator::topology::BuildSwitchTopologyFromSimpleNuma(cfg_or.value());
    if (!topology_or.ok()) {
      std::cerr << "Failed to build topology: " << topology_or.status().ToString() << "\n";
      return 1;
    }
    topology = std::move(topology_or).value();
  }

  std::cout << topology.to_dot();
  return 0;
}
