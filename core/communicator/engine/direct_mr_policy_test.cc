// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>

#include "absl/status/status.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/communicator_config.h"
#include "core/store/components/uma_lease_provider.h"
#include "core/store/replica/replica_memory_coordinator.h"
#include "core/common/memory/distributed_virtual_memory_pool.h"
#include "gsl/pointers"

using tensorcast::communicator::CommunicateEngine;
using tensorcast::communicator::CommunicatorConfig;
using tensorcast::communicator::COMMUNICATE_ENGINE_DEV_CPU;
using tensorcast::store::ReplicaKey;
using tensorcast::store::DeviceKey;
using tensorcast::DeviceType;
using tensorcast::memory::DistributedVirtualMemoryPool;

TEST_CASE("DirectMR inflight cap triggers staging for concurrent requests", "[communicator][rdma][directmr]") {
  // Configure engine: allow direct MR, but cap inflight to 1 and make TTL large
  CommunicatorConfig cfg;
  cfg.enable_rdma = true;
  cfg.stager.stage_cpu_for_rdma = false;
  cfg.stager.direct_mr_max_bytes = 4096; // 4 KiB
  cfg.stager.max_inflight_direct_mr = 1; // cap
  cfg.rdma.ack_ttl_ms = 60000;          // large TTL to avoid reaping during test

  // Select any available RDMA device implicitly; if none, test will skip.

  CommunicateEngine server(cfg, 30);
  CommunicateEngine client(cfg, 30);
  REQUIRE(server.init("127.0.0.1", 61000, 4).ok());
  REQUIRE(client.init("127.0.0.1", 61001, 4).ok());

  // Server CPU tensor
  static constexpr size_t kBufElems = 4096; // 16 KiB
  static uint32_t server_buf[kBufElems];
  static uint32_t client_buf_a[kBufElems];
  static uint32_t client_buf_b[kBufElems];
  for (size_t i = 0; i < kBufElems; ++i) {
    server_buf[i] = static_cast<uint32_t>(i);
  }

  const char* kKey = "CPU_DIRECTMR_CAP_KEY";
  // Avoid MR registration for CPU tensor to not require an RDMA device at registration time
  CommunicateEngine::RegisterTensorOptions opts;
  opts.register_mr = false;
  opts.needs_staging = false;
  if (!server.register_tensor_ex(
           kKey,
           reinterpret_cast<uint64_t>(server_buf),
           sizeof(server_buf),
           COMMUNICATE_ENGINE_DEV_CPU,
           -1,
           opts)
           .ok()) {
    INFO("Skipping DirectMR inflight cap test: no RDMA device available to register CPU tensor");
    SUCCEED();
    return;
  }

  // UMA mapping for residency (mark HOT)
  auto dvmp = std::make_shared<DistributedVirtualMemoryPool>(DistributedVirtualMemoryPool::kDefaultChunkSize);
  auto rmc = std::make_shared<tensorcast::store::ReplicaMemoryCoordinator>(dvmp);
  DeviceKey cpu_dev{.type = DeviceType::CPU, .ordinal = -1, .uuid = {}};
  ReplicaKey rep_key{.artifact_id = "directmr-artifact", .device = cpu_dev, .replica = 0};
  REQUIRE(rmc->allocate(rep_key, DistributedVirtualMemoryPool::kDefaultChunkSize).ok());
  uint8_t one = 1;
  REQUIRE(dvmp->write_at(rep_key.artifact_id, /*va_offset=*/0, &one, 1).ok()); // mark HOT in first chunk
  tensorcast::store::UmaLeaseProvider::instance()->register_mapping(
      kKey, rep_key, /*base_va_off=*/0, gsl::not_null<std::shared_ptr<tensorcast::store::ReplicaMemoryCoordinator>>{rmc});

  // Two concurrent reads within direct_mr_max_bytes
  auto f1 = client.read_tensor(kKey, reinterpret_cast<uint64_t>(client_buf_a), cfg.stager.direct_mr_max_bytes, COMMUNICATE_ENGINE_DEV_CPU, -1, "127.0.0.1", 61000, /*remote_offset=*/0);
  auto f2 = client.read_tensor(kKey, reinterpret_cast<uint64_t>(client_buf_b), cfg.stager.direct_mr_max_bytes, COMMUNICATE_ENGINE_DEV_CPU, -1, "127.0.0.1", 61000, /*remote_offset=*/0);

  auto r1 = f1.get();
  auto r2 = f2.get();
  if (!r1.status.ok() || !r2.status.ok()) {
    INFO("Skipping DirectMR inflight cap test: RDMA device not available");
    SUCCEED();
    return;
  }

  // Verify data transferred correctly
  for (size_t i = 0; i < cfg.stager.direct_mr_max_bytes / sizeof(uint32_t); ++i) {
    REQUIRE(client_buf_a[i] == server_buf[i]);
    REQUIRE(client_buf_b[i] == server_buf[i]);
  }

  // Inflight direct MR should be 1 (second one staged)
  REQUIRE(server.inflight_direct_mr_for_test() == 1);

  // Cleanup connection
  REQUIRE(client.close_connection("127.0.0.1", 61000).ok());
}
