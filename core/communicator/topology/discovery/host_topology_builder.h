// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_HOST_TOPOLOGY_BUILDER_H_
#define CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_HOST_TOPOLOGY_BUILDER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "core/communicator/topology/discovery/nvlink_source.h"
#include "core/communicator/topology/topology.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

namespace tensorcast::communicator::topology::discovery {

struct HostTopologyBuilderOptions {
  std::string default_switch_id = "netsw0";
  std::string unknown_rail_switch_id = "netsw_unknown";
  bool require_connected_when_discovery_disabled = true;
  bool strict_parsing = false;
  NvlinkRuntimeProbeOptions nvlink_runtime_probe;
  // Test-only and deterministic override hooks for PCI affinity inference.
  // Keyed by simple_numa GPU index.
  absl::flat_hash_map<int, std::string> gpu_pci_path_overrides;
  // Keyed by NIC endpoint name (e.g. mlx5_0).
  absl::flat_hash_map<std::string, std::string> nic_pci_path_overrides;
};

struct TopologyDiscoveryObservability {
  bool discovery_enabled = false;
  std::string lldp_source = "disabled";
  int lldp_record_count = 0;
  bool lldp_degraded = false;
  std::string lldp_degrade_reason;
  std::string nvlink_source = "disabled";
  int nvlink_gpu_count = 0;
  int nvlink_edge_count = 0;
  bool nvlink_degraded = false;
  std::string nvlink_degrade_reason;
  int nic_endpoint_count = 0;
  int rail_switch_endpoint_count = 0;
  int affinity_nic_candidate_count = 0;
  int affinity_nic_scored_count = 0;
  int affinity_nic_narrowed_count = 0;
  bool affinity_degraded = false;
  std::string affinity_degrade_reason;
};

struct HostTopologyBuildResult {
  Topology topology;
  TopologyDiscoveryObservability observability;
};

absl::StatusOr<HostTopologyBuildResult> build_topology_from_discovery_with_observability(
    const tensorcast::communicator::v1::CommunicatorConfig& config,
    const HostTopologyBuilderOptions& options = {});

absl::StatusOr<Topology> build_topology_from_discovery(
    const tensorcast::communicator::v1::CommunicatorConfig& config,
    const HostTopologyBuilderOptions& options = {});

} // namespace tensorcast::communicator::topology::discovery

#endif // CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_HOST_TOPOLOGY_BUILDER_H_
