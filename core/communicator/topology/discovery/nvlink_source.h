// Copyright (c) 2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_NVLINK_SOURCE_H_
#define CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_NVLINK_SOURCE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace tensorcast::communicator::topology::discovery {

struct NvlinkGpuRecord {
  std::string gpu_uuid;
  int gpu_index = -1;
};

struct NvlinkEdge {
  std::string src_gpu_uuid;
  std::string dst_gpu_uuid;
  int link_count = 1;
  double bandwidth_hint_gbps = 0.0;
};

struct NvlinkSnapshot {
  std::vector<NvlinkGpuRecord> gpus;
  std::vector<NvlinkEdge> edges;
};

struct NvlinkSnapshotOptions {
  // strict=true: malformed rows fail-fast.
  // strict=false: malformed rows are skipped with warnings.
  bool strict = false;
};

struct NvlinkRuntimeProbeOptions {
  // strict=true: malformed probe output fails fast.
  // strict=false: malformed rows are skipped with warnings.
  bool strict = false;
  // Runtime commands used when overrides are empty.
  std::string gpu_query_command = "nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits";
  std::string topology_matrix_command = "nvidia-smi topo -m";
  // Test-only deterministic overrides. When non-empty, command execution is
  // skipped and these strings are parsed directly.
  std::string gpu_query_output_override;
  std::string topology_matrix_output_override;
};

// Snapshot format:
//   gpu,<gpu_uuid>,<gpu_index>
//   edge,<src_gpu_uuid>,<dst_gpu_uuid>,<link_count>,<bandwidth_hint_gbps>
absl::StatusOr<NvlinkSnapshot> load_nvlink_snapshot(const std::string& file_path, NvlinkSnapshotOptions options = {});

// Parse outputs from:
//   nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits
//   nvidia-smi topo -m
absl::StatusOr<NvlinkSnapshot> parse_nvlink_runtime_probe_outputs(
    const std::string& gpu_query_output,
    const std::string& topology_matrix_output,
    NvlinkSnapshotOptions options = {});

// Discover NVLINK topology from runtime environment.
absl::StatusOr<NvlinkSnapshot> load_nvlink_runtime_probe(NvlinkRuntimeProbeOptions options = {});

} // namespace tensorcast::communicator::topology::discovery

#endif // CORE_COMMUNICATOR_TOPOLOGY_DISCOVERY_NVLINK_SOURCE_H_
