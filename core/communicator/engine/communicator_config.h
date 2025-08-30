// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tensorcast::communicator {

struct StagerConfig {
  bool stage_cpu_for_rdma = true;
  uint32_t direct_mr_max_bytes = 4 * 1024 * 1024; // 4 MiB
  int max_inflight_direct_mr = 32;
  uint32_t stage_chunk_mb_cpu = 4;
  uint32_t stage_chunk_mb_gpu = 16;
  int buffers_per_flow = 4;
};

struct RdmaConfig {
  int outstanding_wr = 64;
  uint32_t ack_ttl_ms = 30000;
  // RDMA QP tuning (was env-based)
  int traffic_class = 186; // GRH traffic_class (TOS)
  int qp_timeout = 20;     // QP timeout
  int qp_retry = 7;        // QP retry count
};

struct PoolConfig {
  bool preregister_mr = true;
  uint64_t pool_size_bytes = 8ull * 1024 * 1024 * 1024; // 8 GiB
  uint64_t chunk_bytes = 64ull * 1024 * 1024;            // 64 MiB
};

struct TransportConfig {
  int tcp_conn_count = 8;
  // TCP/IP tuning (was env-based)
  int tcp_tos = 0;              // IP_TOS value; 0 to leave unchanged
  int connect_timeout_sec = 10; // connect/send timeout (seconds)
};

struct AffinityConfig {
  bool enable = false; // reserved
};

struct SimpleNumaNode {
  int id = 0;
  // NIC device names for this NUMA node
  std::vector<std::string> nics = {};
  // GPU IDs local to this NUMA node
  std::vector<int> gpus = {};
  // Whether this node is the default fallback (reserved)
  bool is_default = false;
};

struct SimpleNumaConfig {
  bool enable = false;
  std::vector<SimpleNumaNode> nodes = {};
};

struct CommunicatorConfig {
  bool enable_rdma = false;
  StagerConfig stager = {};
  RdmaConfig rdma = {};
  PoolConfig pool = {};
  TransportConfig transport = {};
  AffinityConfig affinity = {};
  SimpleNumaConfig simple_numa = {};
};

} // namespace tensorcast::communicator
