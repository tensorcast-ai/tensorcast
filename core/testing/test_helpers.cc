// Copyright (c) 2025-2026, TensorCast Team.

#include "core/testing/test_helpers.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "absl/log/log.h"

namespace tensorcast::testing {

std::vector<uint8_t> create_test_pattern(std::size_t size, uint8_t seed) {
  std::vector<uint8_t> data(size);
  for (std::size_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>((i + seed) % 256);
  }
  return data;
}

bool verify_pattern(const void* data, std::size_t size, uint8_t seed) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    if (bytes[i] != static_cast<uint8_t>((i + seed) % 256)) {
      LOG(ERROR) << "Mismatch at offset " << i << ": expected " << static_cast<int>((i + seed) % 256) << ", got "
                 << static_cast<int>(bytes[i]);
      return false;
    }
  }
  return true;
}

namespace {

constexpr int kDefaultPortMin = 32768;
constexpr int kDefaultPortMax = 61000;

struct EphemeralRange {
  int min_port;
  int max_port;
};

EphemeralRange load_ephemeral_range() {
  std::ifstream range_file("/proc/sys/net/ipv4/ip_local_port_range");
  if (!range_file.is_open()) {
    return {kDefaultPortMin, kDefaultPortMax};
  }

  int min_port = kDefaultPortMin;
  int max_port = kDefaultPortMax;
  range_file >> min_port >> max_port;
  if (min_port <= 0 || max_port <= min_port) {
    return {kDefaultPortMin, kDefaultPortMax};
  }
  return {min_port, max_port};
}

int try_bind_port(int port) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    PLOG(ERROR) << "Failed to create socket while probing port";
    return -1;
  }

  int optval = 1;
  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) != 0) {
    PLOG(WARNING) << "Failed to set SO_REUSEADDR while probing port";
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  addr.sin_port = htons(port);

  if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
    if (port == 0) {
      socklen_t len = sizeof(addr);
      if (getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
        port = ntohs(addr.sin_port);
      }
    }
    close(sock);
    return port;
  }

  close(sock);
  return -1;
}

std::vector<int> build_candidate_ports(int base_port, int max_attempts) {
  EphemeralRange range = load_ephemeral_range();
  const int range_span = range.max_port - range.min_port + 1;
  if (range_span <= 0) {
    return {};
  }

  thread_local std::mt19937 rng([]() {
    std::random_device rd;
    return std::mt19937(rd());
  }());
  std::uniform_int_distribution<int> dist(range.min_port, range.max_port);

  auto clamp_to_range = [&](int port) {
    if (port < range.min_port) {
      return range.min_port;
    }
    if (port > range.max_port) {
      return range.max_port;
    }
    return port;
  };

  std::unordered_set<int> seen;
  seen.reserve(static_cast<size_t>(max_attempts) * 2);
  std::vector<int> candidates;
  candidates.reserve(max_attempts);

  auto try_add = [&](int port) {
    if (port < range.min_port || port > range.max_port) {
      return false;
    }
    if (seen.insert(port).second) {
      candidates.push_back(port);
      return true;
    }
    return false;
  };

  if (base_port <= 0) {
    for (int attempt = 0; attempt < max_attempts && static_cast<int>(candidates.size()) < max_attempts; ++attempt) {
      try_add(dist(rng));
    }
  } else {
    int clamped_base = clamp_to_range(base_port);
    // Apply a jitter so repeated callers with the same base do not concentrate on a single port.
    int jitter = std::uniform_int_distribution<int>(-512, 512)(rng);
    int jittered = clamp_to_range(clamped_base + jitter);
    try_add(jittered);

    // Expand symmetrically around the jittered base.
    for (int delta = 1; static_cast<int>(candidates.size()) < max_attempts; ++delta) {
      bool added = false;
      if (try_add(jittered + delta)) {
        added = true;
      }
      if (static_cast<int>(candidates.size()) >= max_attempts) {
        break;
      }
      if (try_add(jittered - delta)) {
        added = true;
      }
      if (!added && (jittered + delta > range.max_port && jittered - delta < range.min_port)) {
        break;
      }
    }
  }

  // If we still do not have enough candidates, mix in random ports to broaden the search window.
  int safety_budget = range_span * 2;
  while (static_cast<int>(candidates.size()) < max_attempts && safety_budget-- > 0) {
    try_add(dist(rng));
  }

  std::shuffle(candidates.begin(), candidates.end(), rng);

  return candidates;
}

} // namespace

int find_available_port(int base_port, int max_attempts) {
  max_attempts = std::max(1, max_attempts);
  auto candidates = build_candidate_ports(base_port, max_attempts);

  for (int port : candidates) {
    int bound = try_bind_port(port);
    if (bound > 0) {
      LOG(INFO) << "Found available port: " << bound;
      return bound;
    }
  }

  // Fallback: let kernel choose any ephemeral port.
  int auto_port = try_bind_port(0);
  if (auto_port > 0) {
    LOG(INFO) << "Found available ephemeral port via kernel assignment: " << auto_port;
    return auto_port;
  }

  LOG(ERROR) << "Failed to find available port after " << max_attempts << " attempts";
  return -1;
}

void configure_tcp_stager_defaults(tensorcast::communicator::v1::CommunicatorConfig* cfg, uint32_t buffers_per_flow) {
  if (cfg == nullptr) {
    return;
  }
  auto* st = cfg->mutable_stager();
  st->set_stage_cpu_for_rdma(true);
  st->set_buffers_per_flow(buffers_per_flow);
  cfg->mutable_transport()->set_so_reuseport(false);
}

tensorcast::communicator::v1::CommunicatorConfig make_tcp_communicator_config(
    bool enable_rdma,
    uint32_t buffers_per_flow) {
  tensorcast::communicator::v1::CommunicatorConfig cfg;
  cfg.set_enable_rdma(enable_rdma);
  configure_tcp_stager_defaults(&cfg, buffers_per_flow);
  // Keep tests small and deterministic.
  cfg.mutable_transport()->set_tcp_conn_count(2);
  // Keep RDMA test defaults explicit in helper-generated configs so standalone
  // test binaries do not run with protobuf zero-values (e.g. outstanding_wr=0).
  auto* rdma = cfg.mutable_rdma();
  rdma->set_outstanding_wr(64);
  rdma->set_ack_ttl_ms(30000);
  rdma->set_qp_count(1);
  return cfg;
}

tensorcast::communicator::engine::Communicator::PinnedStagingPools make_test_pinned_staging_pools(
    uint32_t buffers_per_flow,
    int tcp_conn_count,
    size_t gpu_slice_bytes,
    size_t cpu_slice_bytes,
    bool enable_rdma) {
  const size_t num_buffers = static_cast<size_t>(std::max<uint32_t>(1, buffers_per_flow));
  const size_t conn_count = static_cast<size_t>(std::max(2, tcp_conn_count));
  const size_t required_gpu_slices = num_buffers + (num_buffers * conn_count);
  const size_t required_cpu_slices = num_buffers;
  auto gpu_pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(
      required_gpu_slices * gpu_slice_bytes, gpu_slice_bytes);
  auto cpu_pool = std::make_shared<tensorcast::common::memory::PinnedBufferPool>(
      required_cpu_slices * cpu_slice_bytes, cpu_slice_bytes);
  return tensorcast::communicator::engine::Communicator::PinnedStagingPools{
      .gpu_pool = std::move(gpu_pool),
      .cpu_pool = std::move(cpu_pool),
      .preregister_gpu = enable_rdma,
      .preregister_cpu = enable_rdma,
  };
}

} // namespace tensorcast::testing
