// Copyright (c) 2026, TensorCast Team.

#include <fstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/communicator/topology/discovery/host_topology_builder.h"
#include "core/communicator/topology/discovery/lldp_source.h"
#include "core/communicator/topology/discovery/nvlink_source.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

namespace {

using tensorcast::communicator::topology::Topology;
using tensorcast::communicator::topology::discovery::LldpParseOptions;
using tensorcast::communicator::topology::discovery::build_topology_from_discovery;
using tensorcast::communicator::topology::discovery::build_topology_from_discovery_with_observability;
using tensorcast::communicator::topology::discovery::load_lldp_records_by_nic;
using tensorcast::communicator::topology::discovery::load_nvlink_snapshot;
using tensorcast::communicator::topology::discovery::parse_nvlink_runtime_probe_outputs;
using tensorcast::communicator::topology::discovery::NvlinkSnapshotOptions;
using tensorcast::communicator::v1::CommunicatorConfig;
using tensorcast::communicator::v1::NvlinkDiscoveryConfig;

std::string write_temp_file(const std::string& content, const std::string& suffix) {
  static int sequence = 0;
  sequence += 1;
  const std::string path = "topology_discovery_test_" + std::to_string(sequence) + "_" + suffix;
  std::ofstream output(path);
  output << content;
  output.close();
  return path;
}

CommunicatorConfig make_simple_numa_config() {
  CommunicatorConfig config;
  auto* simple = config.mutable_simple_numa();
  simple->set_enable(true);

  auto* node0 = simple->add_nodes();
  node0->set_id(0);
  node0->add_nics("mlx5_0");
  node0->add_nics("mlx5_1");
  node0->add_gpus(0);
  node0->add_gpus(1);

  auto* node1 = simple->add_nodes();
  node1->set_id(1);
  node1->add_nics("mlx5_2");
  node1->add_gpus(2);

  return config;
}

std::vector<std::string> endpoint_pool_ids(
    const Topology& topology,
    const std::string& endpoint_id) {
  const auto* endpoint = topology.find_endpoint(endpoint_id);
  if (endpoint == nullptr) {
    return {};
  }
  return endpoint->pool_ids;
}

} // namespace

TEST_CASE("LLDP parser loads nic->rail records", "[communicator][topology][discovery]") {
  const std::string file = write_temp_file(
      "# comment\n"
      "brainpf0=0000:0e:00.0,mlx5_0,1\n"
      "bad line\n"
      "brainpf1=0000:35:00.0,mlx5_1,2\n",
      "lldp.txt");

  auto records_or = load_lldp_records_by_nic(file, LldpParseOptions{.strict = false});
  REQUIRE(records_or.ok());
  const auto& records = records_or.value();
  REQUIRE(records.size() == 2);
  REQUIRE(records.contains("mlx5_0"));
  REQUIRE(records.contains("mlx5_1"));
  CHECK(records.at("mlx5_0").rail_id == 1);
  CHECK(records.at("mlx5_1").rail_id == 2);
}

TEST_CASE("LLDP parser fails in strict mode on malformed rows", "[communicator][topology][discovery]") {
  const std::string file = write_temp_file(
      "brainpf0=0000:0e:00.0,mlx5_0,1\n"
      "bad line\n",
      "lldp_strict.txt");

  auto records_or = load_lldp_records_by_nic(file, LldpParseOptions{.strict = true});
  CHECK_FALSE(records_or.ok());
}

TEST_CASE("NVLINK snapshot parser normalizes duplicate edges", "[communicator][topology][discovery]") {
  const std::string file = write_temp_file(
      "gpu,GPU-0,0\n"
      "gpu,GPU-1,1\n"
      "edge,GPU-0,GPU-1,1,300\n"
      "edge,GPU-1,GPU-0,2,250\n"
      "edge,GPU-1,GPU-1,1,100\n",
      "nvlink.txt");

  auto snapshot_or = load_nvlink_snapshot(file, NvlinkSnapshotOptions{.strict = false});
  REQUIRE(snapshot_or.ok());
  const auto& snapshot = snapshot_or.value();
  REQUIRE(snapshot.gpus.size() == 2);
  REQUIRE(snapshot.edges.size() == 1);
  CHECK(snapshot.edges[0].src_gpu_uuid == "GPU-0");
  CHECK(snapshot.edges[0].dst_gpu_uuid == "GPU-1");
  CHECK(snapshot.edges[0].link_count == 3);
  CHECK(snapshot.edges[0].bandwidth_hint_gbps == 300.0);
}

TEST_CASE("NVLINK runtime parser extracts GPU edges from topology matrix", "[communicator][topology][discovery]") {
  const std::string gpu_query_output =
      "0, GPU-0\n"
      "1, GPU-1\n"
      "2, GPU-2\n";

  const std::string topology_matrix_output =
      "        GPU0  GPU1  GPU2\n"
      "GPU0     X    NV2   SYS\n"
      "GPU1    NV2    X    NV1\n"
      "GPU2    SYS   NV1    X\n"
      "Legend:\n";

  auto snapshot_or = parse_nvlink_runtime_probe_outputs(
      gpu_query_output,
      topology_matrix_output,
      NvlinkSnapshotOptions{.strict = true});
  REQUIRE(snapshot_or.ok());
  const auto& snapshot = snapshot_or.value();
  REQUIRE(snapshot.gpus.size() == 3);
  REQUIRE(snapshot.edges.size() == 2);

  CHECK(snapshot.edges[0].src_gpu_uuid == "GPU-0");
  CHECK(snapshot.edges[0].dst_gpu_uuid == "GPU-1");
  CHECK(snapshot.edges[0].link_count == 2);
  CHECK(snapshot.edges[1].src_gpu_uuid == "GPU-1");
  CHECK(snapshot.edges[1].dst_gpu_uuid == "GPU-2");
  CHECK(snapshot.edges[1].link_count == 1);
}

TEST_CASE(
    "NVLINK runtime parser strips ANSI topology header sequences",
    "[communicator][topology][discovery]") {
  const std::string gpu_query_output =
      "0, GPU-0\n"
      "1, GPU-1\n"
      "2, GPU-2\n"
      "3, GPU-3\n";

  const std::string topology_matrix_output =
      "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tNIC0\tCPU Affinity\x1b[0m\n"
      "GPU0\tX\tNV8\tNV8\tNV8\tPIX\t0-43\n"
      "GPU1\tNV8\tX\tNV8\tNV8\tSYS\t0-43\n"
      "GPU2\tNV8\tNV8\tX\tNV8\tSYS\t0-43\n"
      "GPU3\tNV8\tNV8\tNV8\tX\tSYS\t0-43\n"
      "NIC0\tPIX\tSYS\tSYS\tSYS\tX\n"
      "\n"
      "Legend:\n";

  auto snapshot_or = parse_nvlink_runtime_probe_outputs(
      gpu_query_output,
      topology_matrix_output,
      NvlinkSnapshotOptions{.strict = true});
  REQUIRE(snapshot_or.ok());
  const auto& snapshot = snapshot_or.value();
  REQUIRE(snapshot.gpus.size() == 4);
  REQUIRE(snapshot.edges.size() == 6);

  CHECK(snapshot.edges[0].src_gpu_uuid == "GPU-0");
  CHECK(snapshot.edges[0].dst_gpu_uuid == "GPU-1");
  CHECK(snapshot.edges[0].link_count == 8);
  CHECK(snapshot.edges[5].src_gpu_uuid == "GPU-2");
  CHECK(snapshot.edges[5].dst_gpu_uuid == "GPU-3");
  CHECK(snapshot.edges[5].link_count == 8);
}

TEST_CASE("Host topology builder merges rail switches and NVLINK edges", "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(true);
  discovery->mutable_merge_policy()->set_require_connected(false);

  const std::string lldp_file = write_temp_file(
      "brainpf0=0000:0e:00.0,mlx5_0,1\n"
      "brainpf1=0000:35:00.0,mlx5_1,2\n",
      "lldp_builder.txt");
  discovery->mutable_lldp()->set_file_path(lldp_file);
  discovery->mutable_lldp()->set_required(true);

  const std::string nvlink_file = write_temp_file(
      "gpu,GPU-0,0\n"
      "gpu,GPU-1,1\n"
      "edge,GPU-0,GPU-1,1,450\n",
      "nvlink_builder.txt");
  discovery->mutable_nvlink()->set_source(NvlinkDiscoveryConfig::SOURCE_SNAPSHOT_FILE);
  discovery->mutable_nvlink()->set_snapshot_file_path(nvlink_file);

  auto topology_or = build_topology_from_discovery(config);
  INFO(topology_or.status());
  REQUIRE(topology_or.ok());
  const Topology& topology = topology_or.value();

  CHECK(topology.find_endpoint("netsw_rail_1") != nullptr);
  CHECK(topology.find_endpoint("netsw_rail_2") != nullptr);
  CHECK(topology.find_endpoint("netsw_unknown") != nullptr);
  CHECK(topology.find_endpoint("nvlink_GPU-0") != nullptr);
  CHECK(topology.find_endpoint("nvlink_GPU-1") != nullptr);
  CHECK(topology.find_link("nvlink_GPU-0_to_nvlink_GPU-1") != nullptr);
  CHECK(topology.find_link("nvlink_GPU-1_to_nvlink_GPU-0") != nullptr);
}

TEST_CASE("Host topology builder degrades when LLDP is missing", "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(true);
  discovery->mutable_lldp()->set_file_path("/tmp/does-not-exist-lldp.txt");
  discovery->mutable_lldp()->set_required(false);

  auto topology_or = build_topology_from_discovery(config);
  INFO(topology_or.status());
  REQUIRE(topology_or.ok());
  CHECK(topology_or->find_endpoint("netsw_unknown") != nullptr);
}

TEST_CASE("Host topology builder fails when required LLDP file is missing", "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(true);
  discovery->mutable_lldp()->set_file_path("/tmp/does-not-exist-lldp-required.txt");
  discovery->mutable_lldp()->set_required(true);

  auto topology_or = build_topology_from_discovery(config);
  CHECK_FALSE(topology_or.ok());
}

TEST_CASE("Host topology builder reports discovery observability on degrade", "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(true);
  discovery->mutable_lldp()->set_file_path("/tmp/does-not-exist-lldp-observability.txt");
  discovery->mutable_lldp()->set_required(false);
  discovery->mutable_nvlink()->set_source(NvlinkDiscoveryConfig::SOURCE_SNAPSHOT_FILE);
  discovery->mutable_nvlink()->set_snapshot_file_path("/tmp/does-not-exist-nvlink-observability.txt");
  discovery->mutable_nvlink()->set_required(false);

  auto build_or = build_topology_from_discovery_with_observability(config);
  INFO(build_or.status());
  REQUIRE(build_or.ok());
  const auto& observability = build_or->observability;

  CHECK(observability.discovery_enabled == true);
  CHECK(observability.lldp_source == "file:/tmp/does-not-exist-lldp-observability.txt");
  CHECK(observability.lldp_record_count == 0);
  CHECK(observability.lldp_degraded == true);
  CHECK_FALSE(observability.lldp_degrade_reason.empty());
  CHECK(observability.nvlink_source == "snapshot_file");
  CHECK(observability.nvlink_gpu_count == 0);
  CHECK(observability.nvlink_edge_count == 0);
  CHECK(observability.nvlink_degraded == true);
  CHECK_FALSE(observability.nvlink_degrade_reason.empty());
  CHECK(observability.nic_endpoint_count == 3);
  CHECK(observability.rail_switch_endpoint_count == 1);
}

TEST_CASE("Host topology builder supports NVLINK runtime probe source", "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(true);
  discovery->mutable_merge_policy()->set_require_connected(false);

  const std::string lldp_file = write_temp_file(
      "brainpf0=0000:0e:00.0,mlx5_0,1\n"
      "brainpf1=0000:35:00.0,mlx5_1,2\n"
      "brainpf2=0000:36:00.0,mlx5_2,3\n",
      "lldp_runtime_builder.txt");
  discovery->mutable_lldp()->set_file_path(lldp_file);
  discovery->mutable_lldp()->set_required(true);

  discovery->mutable_nvlink()->set_source(NvlinkDiscoveryConfig::SOURCE_RUNTIME_PROBE);
  discovery->mutable_nvlink()->set_required(true);

  tensorcast::communicator::topology::discovery::HostTopologyBuilderOptions options;
  options.nvlink_runtime_probe.gpu_query_output_override =
      "0, GPU-0\n"
      "1, GPU-1\n"
      "2, GPU-2\n";
  options.nvlink_runtime_probe.topology_matrix_output_override =
      "        GPU0  GPU1  GPU2\n"
      "GPU0     X    NV2   SYS\n"
      "GPU1    NV2    X    NV1\n"
      "GPU2    SYS   NV1    X\n"
      "Legend:\n";

  auto build_or = build_topology_from_discovery_with_observability(config, options);
  INFO(build_or.status());
  REQUIRE(build_or.ok());
  const auto& observability = build_or->observability;

  CHECK(observability.nvlink_source == "runtime_probe");
  CHECK(observability.nvlink_degraded == false);
  CHECK(observability.nvlink_gpu_count == 3);
  CHECK(observability.nvlink_edge_count == 2);

  const Topology& topology = build_or->topology;
  CHECK(topology.find_endpoint("nvlink_GPU-0") != nullptr);
  CHECK(topology.find_endpoint("nvlink_GPU-1") != nullptr);
  CHECK(topology.find_endpoint("nvlink_GPU-2") != nullptr);
  CHECK(topology.find_link("nvlink_GPU-0_to_nvlink_GPU-1") != nullptr);
  CHECK(topology.find_link("nvlink_GPU-1_to_nvlink_GPU-0") != nullptr);
  CHECK(topology.find_link("nvlink_GPU-1_to_nvlink_GPU-2") != nullptr);
  CHECK(topology.find_link("nvlink_GPU-2_to_nvlink_GPU-1") != nullptr);
}

TEST_CASE(
    "Host topology builder infers NIC GPU affinity from PCI paths",
    "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(false);
  discovery->mutable_merge_policy()->set_require_connected(false);
  discovery->mutable_nvlink()->set_source(NvlinkDiscoveryConfig::SOURCE_DISABLED);

  const std::string lldp_file = write_temp_file(
      "pf0=0000:41:00.1,mlx5_0,1\n"
      "pf1=0000:42:00.1,mlx5_1,1\n"
      "pf2=0000:81:00.1,mlx5_2,2\n",
      "lldp_affinity.txt");
  discovery->mutable_lldp()->set_file_path(lldp_file);
  discovery->mutable_lldp()->set_required(true);

  tensorcast::communicator::topology::discovery::HostTopologyBuilderOptions options;
  options.gpu_pci_path_overrides = {
      {0, "/sys/devices/pci0000:00/0000:00:01.0/0000:41:00.0"},
      {1, "/sys/devices/pci0000:00/0000:00:02.0/0000:42:00.0"},
      {2, "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0"},
  };
  options.nic_pci_path_overrides = {
      {"mlx5_0", "/sys/devices/pci0000:00/0000:00:01.0/0000:41:00.1"},
      {"mlx5_1", "/sys/devices/pci0000:00/0000:00:02.0/0000:42:00.1"},
      {"mlx5_2", "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.1"},
  };

  auto build_or = build_topology_from_discovery_with_observability(config, options);
  INFO(build_or.status());
  REQUIRE(build_or.ok());

  const Topology& topology = build_or->topology;
  CHECK(endpoint_pool_ids(topology, "nic_mlx5_0") == std::vector<std::string>{"cpu0", "gpu0"});
  CHECK(endpoint_pool_ids(topology, "nic_mlx5_1") == std::vector<std::string>{"cpu0", "gpu1"});
  CHECK(endpoint_pool_ids(topology, "nic_mlx5_2") == std::vector<std::string>{"cpu1", "gpu2"});

  const auto& observability = build_or->observability;
  CHECK(observability.affinity_nic_candidate_count == 3);
  CHECK(observability.affinity_nic_scored_count == 3);
  CHECK(observability.affinity_nic_narrowed_count == 2);
  CHECK(observability.affinity_degraded == false);
}

TEST_CASE(
    "Host topology builder keeps baseline pools when affinity inference degrades",
    "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(false);
  discovery->mutable_merge_policy()->set_require_connected(false);
  discovery->mutable_nvlink()->set_source(NvlinkDiscoveryConfig::SOURCE_DISABLED);

  const std::string lldp_file = write_temp_file(
      "pf0=0000:41:00.1,mlx5_0,1\n"
      "pf1=0000:42:00.1,mlx5_1,1\n"
      "pf2=0000:81:00.1,mlx5_2,2\n",
      "lldp_affinity_degrade.txt");
  discovery->mutable_lldp()->set_file_path(lldp_file);
  discovery->mutable_lldp()->set_required(true);

  tensorcast::communicator::topology::discovery::HostTopologyBuilderOptions options;
  options.gpu_pci_path_overrides = {
      {0, "/sys/devices/pci0000:00/0000:00:01.0/0000:41:00.0"},
      {1, "/sys/devices/pci0000:00/0000:00:02.0/0000:42:00.0"},
      {2, "/sys/devices/pci0000:80/0000:80:01.0/0000:81:00.0"},
  };
  options.nic_pci_path_overrides = {
      {"mlx5_0", ""},
      {"mlx5_1", ""},
      {"mlx5_2", ""},
  };

  auto build_or = build_topology_from_discovery_with_observability(config, options);
  INFO(build_or.status());
  REQUIRE(build_or.ok());

  const Topology& topology = build_or->topology;
  CHECK(endpoint_pool_ids(topology, "nic_mlx5_0") == std::vector<std::string>{"cpu0", "gpu0", "gpu1"});
  CHECK(endpoint_pool_ids(topology, "nic_mlx5_1") == std::vector<std::string>{"cpu0", "gpu0", "gpu1"});
  CHECK(endpoint_pool_ids(topology, "nic_mlx5_2") == std::vector<std::string>{"cpu1", "gpu2"});

  const auto& observability = build_or->observability;
  CHECK(observability.affinity_nic_candidate_count == 3);
  CHECK(observability.affinity_nic_scored_count == 0);
  CHECK(observability.affinity_nic_narrowed_count == 0);
  CHECK(observability.affinity_degraded == true);
  CHECK_FALSE(observability.affinity_degrade_reason.empty());
}

TEST_CASE("Host topology builder ignores NVLINK GPUs without discovered edges", "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(false);
  discovery->mutable_merge_policy()->set_require_connected(false);
  discovery->mutable_nvlink()->set_source(NvlinkDiscoveryConfig::SOURCE_RUNTIME_PROBE);
  discovery->mutable_nvlink()->set_required(true);

  tensorcast::communicator::topology::discovery::HostTopologyBuilderOptions options;
  options.nvlink_runtime_probe.gpu_query_output_override =
      "0, GPU-0\n"
      "1, GPU-1\n";
  options.nvlink_runtime_probe.topology_matrix_output_override =
      "        GPU0  GPU1\n"
      "GPU0     X    SYS\n"
      "GPU1    SYS    X\n"
      "Legend:\n";

  auto build_or = build_topology_from_discovery_with_observability(config, options);
  INFO(build_or.status());
  REQUIRE(build_or.ok());
  const auto& observability = build_or->observability;
  CHECK(observability.nvlink_source == "runtime_probe");
  CHECK(observability.nvlink_degraded == false);
  CHECK(observability.nvlink_gpu_count == 2);
  CHECK(observability.nvlink_edge_count == 0);

  const Topology& topology = build_or->topology;
  CHECK(topology.find_endpoint("nvlink_GPU-0") == nullptr);
  CHECK(topology.find_endpoint("nvlink_GPU-1") == nullptr);
}

TEST_CASE("Host topology builder fails on required NVLINK runtime probe error", "[communicator][topology][discovery]") {
  CommunicatorConfig config = make_simple_numa_config();
  auto* discovery = config.mutable_topology_discovery();
  discovery->set_enable(true);
  discovery->mutable_merge_policy()->set_emit_rail_switch_endpoints(false);
  discovery->mutable_merge_policy()->set_require_connected(false);

  discovery->mutable_nvlink()->set_source(NvlinkDiscoveryConfig::SOURCE_RUNTIME_PROBE);
  discovery->mutable_nvlink()->set_required(true);

  tensorcast::communicator::topology::discovery::HostTopologyBuilderOptions options;
  options.nvlink_runtime_probe.gpu_query_command = "/bin/false";
  options.nvlink_runtime_probe.topology_matrix_command = "/bin/false";

  auto build_or = build_topology_from_discovery_with_observability(config, options);
  CHECK_FALSE(build_or.ok());
  CHECK(build_or.status().message().find("failed to run required NVLINK runtime probe") != std::string::npos);
}
