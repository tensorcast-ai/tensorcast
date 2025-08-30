// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/engine/communicator_config.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "core/communicator/misc/envs.h"

namespace tensorcast::communicator {

static bool legacy_gate_enabled() {
  const char* v = std::getenv("TENSORCAST_ALLOW_LEGACY_ENV");
  if (!v) return false;
  // Accept 1/true/TRUE/yes/YES
  std::string s(v);
  absl::StripAsciiWhitespace(&s);
  for (auto& c : s) c = static_cast<char>(::tolower(c));
  return (s == "1" || s == "true" || s == "yes");
}

static void warn_map(const char* env, const char* field, const std::string& val) {
  LOG(WARNING) << "[deprecated-env] " << env << " -> " << field << " = '" << val << "'";
}
static void warn_map(const char* env, const char* field, int64_t val) {
  LOG(WARNING) << "[deprecated-env] " << env << " -> " << field << " = " << val;
}

CommunicatorConfig CommunicatorConfig::FromEnvDeprecated(bool enable_rdma, bool log_warnings) {
  CommunicatorConfig cfg;
  cfg.enable_rdma = enable_rdma;
  cfg.legacy_env_mode = true;

  if (!legacy_gate_enabled()) {
    LOG(ERROR) << "Deprecated env loader called but TENSORCAST_ALLOW_LEGACY_ENV is not set. "
               << "Ignorning envs and using typed defaults.";
    return cfg;
  }

  // Map: GPU_TCP_STAGER_CHUNK_SIZE_MB -> stager.stage_chunk_mb_gpu
  int gpu_chunk_mb = get_env("TENSORCAST_COMM_GPU_TCP_STAGER_CHUNK_SIZE_MB", static_cast<int>(cfg.stager.stage_chunk_mb_gpu));
  cfg.stager.stage_chunk_mb_gpu = static_cast<uint32_t>(gpu_chunk_mb);
  if (log_warnings) warn_map("TENSORCAST_COMM_GPU_TCP_STAGER_CHUNK_SIZE_MB", "stager.stage_chunk_mb_gpu", gpu_chunk_mb);

  // Map: GPU_TCP_STAGER_NUM_BUFFERS -> stager.buffers_per_flow
  int num_buf = get_env("TENSORCAST_COMM_GPU_TCP_STAGER_NUM_BUFFERS", cfg.stager.buffers_per_flow);
  cfg.stager.buffers_per_flow = num_buf;
  if (log_warnings) warn_map("TENSORCAST_COMM_GPU_TCP_STAGER_NUM_BUFFERS", "stager.buffers_per_flow", num_buf);

  // Map: GPU_TCP_RECV_NUM_BUFFERS -> stager.buffers_per_flow (unified)
  int recv_buf = get_env("TENSORCAST_COMM_GPU_TCP_RECV_NUM_BUFFERS", cfg.stager.buffers_per_flow);
  if (recv_buf > cfg.stager.buffers_per_flow) {
    cfg.stager.buffers_per_flow = recv_buf;
  }
  if (log_warnings) warn_map("TENSORCAST_COMM_GPU_TCP_RECV_NUM_BUFFERS", "stager.buffers_per_flow (unified)", recv_buf);

  // Map: RDMA_ACK_TTL_MS -> rdma.ack_ttl_ms
  int ack_ttl = get_env("TENSORCAST_COMM_RDMA_ACK_TTL_MS", static_cast<int>(cfg.rdma.ack_ttl_ms));
  cfg.rdma.ack_ttl_ms = static_cast<uint32_t>(ack_ttl);
  if (log_warnings) warn_map("TENSORCAST_COMM_RDMA_ACK_TTL_MS", "rdma.ack_ttl_ms", ack_ttl);

  // Map: STAGER_NUMA_ENABLE -> simple_numa.enable
  int numa_enable = get_env("TENSORCAST_COMM_STAGER_NUMA_ENABLE", 0);
  cfg.simple_numa.enable = (numa_enable != 0);
  if (log_warnings) warn_map("TENSORCAST_COMM_STAGER_NUMA_ENABLE", "simple_numa.enable", numa_enable);

  // Map: STAGER_NUMA_GPU_MAP -> simple_numa.nodes[].gpus
  // Format: "0:0,1;1:2,3"
  std::string gpu_map = get_env("TENSORCAST_COMM_STAGER_NUMA_GPU_MAP", "");
  std::string nic_map = get_env("TENSORCAST_COMM_STAGER_NUMA_NIC_MAP", "");
  if (log_warnings) warn_map("TENSORCAST_COMM_STAGER_NUMA_GPU_MAP", "simple_numa.nodes[].gpus (parsed)", gpu_map);
  if (log_warnings) warn_map("TENSORCAST_COMM_STAGER_NUMA_NIC_MAP", "simple_numa.nodes[].nics (parsed)", nic_map);

  std::unordered_map<int, std::vector<int>> node_gpus;
  std::unordered_map<int, std::vector<std::string>> node_nics;
  auto parse_gpu_map = [](const std::string& s) {
    std::unordered_map<int, std::vector<int>> out;
    for (absl::string_view part : absl::StrSplit(s, ';', absl::SkipEmpty())) {
      absl::string_view trimmed = absl::StripAsciiWhitespace(part);
      size_t pos = trimmed.find(':');
      if (pos == absl::string_view::npos) continue;
      absl::string_view node_sv = absl::StripAsciiWhitespace(trimmed.substr(0, pos));
      absl::string_view list_sv = absl::StripAsciiWhitespace(trimmed.substr(pos + 1));
      int node_id = 0;
      if (!absl::SimpleAtoi(node_sv, &node_id)) continue;
      std::vector<int> ids;
      for (absl::string_view idsv : absl::StrSplit(list_sv, ',', absl::SkipEmpty())) {
        int id = -1;
        if (absl::SimpleAtoi(idsv, &id)) ids.push_back(id);
      }
      if (!ids.empty()) out[node_id] = std::move(ids);
    }
    return out;
  };
  auto parse_nic_map = [](const std::string& s) {
    std::unordered_map<int, std::vector<std::string>> out;
    for (absl::string_view part : absl::StrSplit(s, ';', absl::SkipEmpty())) {
      absl::string_view trimmed = absl::StripAsciiWhitespace(part);
      size_t pos = trimmed.find(':');
      if (pos == absl::string_view::npos) continue;
      absl::string_view node_sv = absl::StripAsciiWhitespace(trimmed.substr(0, pos));
      absl::string_view list_sv = absl::StripAsciiWhitespace(trimmed.substr(pos + 1));
      int node_id = 0;
      if (!absl::SimpleAtoi(node_sv, &node_id)) continue;
      std::vector<std::string> names;
      for (absl::string_view nsv : absl::StrSplit(list_sv, ',', absl::SkipEmpty())) {
        std::string sname(nsv);
        absl::StripAsciiWhitespace(&sname);
        if (!sname.empty()) names.emplace_back(std::move(sname));
      }
      if (!names.empty()) out[node_id] = std::move(names);
    }
    return out;
  };

  node_gpus = parse_gpu_map(gpu_map);
  node_nics = parse_nic_map(nic_map);

  // Merge nodes and build config
  std::vector<int> nodes;
  nodes.reserve(node_gpus.size() + node_nics.size());
  for (auto& kv : node_gpus) nodes.push_back(kv.first);
  for (auto& kv : node_nics) nodes.push_back(kv.first);
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

  cfg.simple_numa.nodes.clear();
  for (int node_id : nodes) {
    SimpleNumaNode node;
    node.id = node_id;
    if (node_gpus.count(node_id)) node.gpus = node_gpus[node_id];
    if (node_nics.count(node_id)) node.nics = node_nics[node_id];
    cfg.simple_numa.nodes.push_back(std::move(node));
  }

  // DEFAULT_DEV → put into a default NUMA node's NICs if provided.
  std::string default_dev = get_env("TENSORCAST_COMM_DEFAULT_DEV", "");
  if (!default_dev.empty()) {
    if (cfg.simple_numa.nodes.empty()) {
      SimpleNumaNode node;
      node.id = 0;
      node.is_default = true;
      node.nics.push_back(default_dev);
      cfg.simple_numa.nodes.push_back(std::move(node));
    } else {
      // Mark first node as default and append NIC if not present
      cfg.simple_numa.nodes.front().is_default = true;
      auto& nics = cfg.simple_numa.nodes.front().nics;
      if (std::find(nics.begin(), nics.end(), default_dev) == nics.end()) {
        nics.push_back(default_dev);
      }
    }
    if (log_warnings) warn_map("TENSORCAST_COMM_DEFAULT_DEV", "simple_numa.nodes[0].nics[+]", default_dev);
  }

  return cfg;
}

} // namespace tensorcast::communicator

