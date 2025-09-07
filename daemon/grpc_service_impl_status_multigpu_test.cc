// Copyright (c) 2025, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include "absl/strings/str_format.h"
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v1/store_daemon.grpc.pb.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

static tensorcast::store::StoreEngineOptions opts_small_pool() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 64ULL * 1024 * 1024; // 64 MiB
  opts.chunk_size = 1ULL << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

// Helper to create a memory-only GPU replica via Begin/CommitRegisterArtifact
static void create_gpu_memory_replica(
    std::shared_ptr<tensorcast::store::StoreEngine> engine,
    const std::string& logical_artifact_id,
    int device_id,
    uint64_t size_bytes) {
  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = logical_artifact_id;
  reg.device_id = device_id;
  reg.total_size_bytes = size_bytes;
  // Provide a dummy canonical tensor index key (sha256 hex-like) to satisfy API
  reg.tensor_index_key = std::string(64, 'a');
  reg.encoding = "json";
  reg.schema_version = "v2";
  reg.enable_p2p = false; // avoid registering comm keys for this test

  auto beg = engine->begin_register_artifact(reg);
  REQUIRE(beg.ok());
  // No further modifications; commit should compute mi2 id and register in-engine
  auto com = engine->commit_registered_artifact(beg->registration_id);
  REQUIRE(com.ok());
  (void)com; // suppress unused warning in opt builds
}

TEST_CASE("GetDetailedStatus aggregates multi-GPU replicas", "[daemon][status][multigpu]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(opts_small_pool());
  StoreDaemonServiceImpl svc(engine);
  svc.set_worker_registered("worker-xyz");

  // Simulate two GPU replicas on device 0 and 1
  create_gpu_memory_replica(engine, "logical-A", /*device_id=*/0, /*size_bytes=*/4ULL * 1024 * 1024);
  create_gpu_memory_replica(engine, "logical-B", /*device_id=*/1, /*size_bytes=*/8ULL * 1024 * 1024);

  tensorcast::daemon::v1::GetDetailedStatusRequest req;
  tensorcast::daemon::v1::GetDetailedStatusResponse resp;
  grpc::ServerContext ctx;
  auto st = svc.GetDetailedStatus(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.is_registered());
  REQUIRE(resp.worker_id() == "worker-xyz");

  // Should have at least two GPU devices populated (IDs 0 and 1)
  REQUIRE(resp.gpu_devices_size() >= 2);
  bool seen_dev0 = false;
  bool seen_dev1 = false;
  int total_loaded = 0;
  for (const auto& gpu : resp.gpu_devices()) {
    if (gpu.device_id() == 0)
      seen_dev0 = true;
    if (gpu.device_id() == 1)
      seen_dev1 = true;
    // Memory totals should be non-zero under Fake CUDA and consistent
    // with used = total - free (allow equality when pool usage is negligible)
    REQUIRE(gpu.total_memory_bytes() >= gpu.used_memory_bytes());
    REQUIRE(gpu.total_memory_bytes() >= gpu.free_memory_bytes());
    // At least one replica listed for the devices we populated
    total_loaded += gpu.loaded_replicas_size();
  }
  REQUIRE(seen_dev0);
  REQUIRE(seen_dev1);
  REQUIRE(total_loaded >= 2);

  // Top-level totals should count both GPU replicas
  REQUIRE(resp.total_replicas_loaded() >= 2);
  REQUIRE(resp.total_artifact_size_bytes() >= 12 * 1024 * 1024);
}
