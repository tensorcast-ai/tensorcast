// Copyright (c) 2026, TensorCast Team.

#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "core/communicator/config_io.h"
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

  auto topology_or = tensorcast::communicator::topology::BuildSwitchTopologyFromSimpleNuma(cfg_or.value());
  if (!topology_or.ok()) {
    std::cerr << "Failed to build topology: " << topology_or.status().ToString() << "\n";
    return 1;
  }

  std::cout << topology_or->to_dot();
  return 0;
}
