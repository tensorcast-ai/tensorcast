// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TOPOLOGY_SIMPLE_NUMA_TOPOLOGY_H_
#define CORE_COMMUNICATOR_TOPOLOGY_SIMPLE_NUMA_TOPOLOGY_H_

#include <string>

#include "absl/status/statusor.h"
#include "core/communicator/topology/topology.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

namespace tensorcast::communicator::topology {

struct SimpleNumaTopologyOptions {
  std::string switch_id = "netsw0";
  std::string switch_name = "netsw0";
  bool require_connected = true;
};

absl::StatusOr<Topology> BuildSwitchTopologyFromSimpleNuma(
    const tensorcast::communicator::v1::CommunicatorConfig& config,
    const SimpleNumaTopologyOptions& options = {});

} // namespace tensorcast::communicator::topology

#endif // CORE_COMMUNICATOR_TOPOLOGY_SIMPLE_NUMA_TOPOLOGY_H_
