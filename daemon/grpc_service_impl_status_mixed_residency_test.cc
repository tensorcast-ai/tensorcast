// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <string>

#include "core/store/store_engine.h"
#include "core/testing/common.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

using tensorcast::daemon::StoreDaemonServiceImpl;
namespace fs = std::filesystem;

static tensorcast::store::StoreEngineOptions make_opts_small() {
  tensorcast::store::StoreEngineOptions opts;
  opts.memory_pool_size = 64ULL * 1024 * 1024; // 64 MiB
  opts.tx_slice_bytes = 1ULL << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

TEST_CASE("GetDetailedStatus aggregates mixed CPU/GPU residency", "[daemon][status][mixed]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_small());
  StoreDaemonServiceImpl svc(engine);
  svc.set_worker_registered("worker-mixed");

  // Prepare a minimal on-disk artifact directory
  fs::path dir = fs::temp_directory_path() / fs::path("tc_mixed_residency_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  // Create a small data file and RFC-0007 descriptors for DiskLoader
  REQUIRE(tensorcast::testing::create_dummy_file(dir / "tensor.data", 1ULL * 1024 * 1024));
  auto st_desc = tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(dir);
  REQUIRE(st_desc.ok());

  // Load CPU replica from disk
  {
    tensorcast::store::loading::MaterializeHints hints;
    hints.disk_path = dir.string();
    auto h = engine->materialize_replica(
        tensorcast::store::DeviceKey{tensorcast::DeviceType::CPU, -1, ""},
        tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY,
        hints);
    REQUIRE(h.ok());
    REQUIRE(std::move(h->subscribe_ready()).get().ok());
  }

  // Load GPU replica (device 0) from the same disk artifact
  {
    tensorcast::store::loading::MaterializeHints hints;
    hints.disk_path = dir.string();
    auto h = engine->materialize_replica(
        tensorcast::store::DeviceKey{tensorcast::DeviceType::GPU, 0, ""},
        tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY,
        hints);
    REQUIRE(h.ok());
    REQUIRE(std::move(h->subscribe_ready()).get().ok());
  }

  // Query detailed status
  {
    tensorcast::daemon::v2::GetDetailedStatusRequest req;
    tensorcast::daemon::v2::GetDetailedStatusResponse resp;
    grpc::ServerContext ctx;
    auto st = svc.GetDetailedStatus(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.is_registered());
    // Should have at least one GPU entry with a loaded replica
    bool have_gpu0 = false;
    bool gpu_has_rep = false;
    for (const auto& gpu : resp.gpu_devices()) {
      if (gpu.device_id() == 0)
        have_gpu0 = true;
      if (gpu.loaded_replicas_size() > 0)
        gpu_has_rep = true;
      // Totals should be consistent
      REQUIRE(gpu.total_memory_bytes() >= gpu.used_memory_bytes());
      REQUIRE(gpu.total_memory_bytes() >= gpu.free_memory_bytes());
    }
    REQUIRE(have_gpu0);
    REQUIRE(gpu_has_rep);
    // Should list at least one CPU replica
    REQUIRE(resp.cpu_replicas_size() >= 1);
    // Totals must account for both CPU+GPU entries
    REQUIRE(resp.total_replicas_loaded() >= 2);
    REQUIRE(resp.total_artifact_size_bytes() >= 2 * 1 * 1024 * 1024);
  }

  // Query loaded replicas list and ensure both CPU(-1) and GPU(0) are present
  {
    tensorcast::daemon::v2::GetLoadedReplicasV2Request req;
    tensorcast::daemon::v2::GetLoadedReplicasV2Response resp;
    grpc::ServerContext ctx;
    auto st = svc.GetLoadedReplicasV2(&ctx, &req, &resp);
    REQUIRE(st.ok());
    bool seen_cpu = false, seen_gpu0 = false;
    for (const auto& r : resp.replicas()) {
      if (r.device_id() == -1)
        seen_cpu = true;
      if (r.device_id() == 0)
        seen_gpu0 = true;
    }
    REQUIRE(seen_cpu);
    REQUIRE(seen_gpu0);
  }
}

TEST_CASE("GetLoadedReplicas filters by artifact_id and device_id", "[daemon][status][filters]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts_small());
  StoreDaemonServiceImpl svc(engine);
  svc.set_worker_registered("worker-filters");

  // Create another minimal artifact on disk
  fs::path dir = fs::temp_directory_path() / fs::path("tc_filters_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  REQUIRE(tensorcast::testing::create_dummy_file(dir / "tensor.data", 512ULL * 1024));
  auto st_desc = tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(dir);
  REQUIRE(st_desc.ok());

  // Load both CPU and GPU replicas from the same artifact
  for (int i = 0; i < 2; ++i) {
    tensorcast::store::loading::MaterializeHints hints;
    hints.disk_path = dir.string();
    auto dev = (i == 0) ? tensorcast::store::DeviceKey{tensorcast::DeviceType::CPU, -1, ""}
                        : tensorcast::store::DeviceKey{tensorcast::DeviceType::GPU, 0, ""};
    auto h = engine->materialize_replica(dev, tensorcast::store::StoreEngine::MaterializeMode::LOAD_ONLY, hints);
    REQUIRE(h.ok());
    REQUIRE(std::move(h->subscribe_ready()).get().ok());
  }

  // Filter by artifact_id substring (directory basename)
  const std::string filter = dir.filename().string();
  {
    tensorcast::daemon::v2::GetLoadedReplicasV2Request req;
    tensorcast::daemon::v2::GetLoadedReplicasV2Response resp;
    req.set_artifact_id_filter(filter);
    grpc::ServerContext ctx;
    auto st = svc.GetLoadedReplicasV2(&ctx, &req, &resp);
    REQUIRE(st.ok());
    // Expect two entries (CPU -1, GPU 0)
    REQUIRE(resp.replicas_size() >= 2);
    bool seen_cpu = false, seen_gpu0 = false;
    for (const auto& r : resp.replicas()) {
      if (r.device_id() == -1)
        seen_cpu = true;
      if (r.device_id() == 0)
        seen_gpu0 = true;
    }
    REQUIRE(seen_cpu);
    REQUIRE(seen_gpu0);
  }

  // Filter by device_id only (GPU 0)
  {
    tensorcast::daemon::v2::GetLoadedReplicasV2Request req;
    tensorcast::daemon::v2::GetLoadedReplicasV2Response resp;
    req.set_device_id_filter(0);
    grpc::ServerContext ctx;
    auto st = svc.GetLoadedReplicasV2(&ctx, &req, &resp);
    REQUIRE(st.ok());
    // Only GPU entries should remain
    REQUIRE(resp.replicas_size() >= 1);
    for (const auto& r : resp.replicas()) {
      REQUIRE(r.device_id() == 0);
    }
  }

  // Combined filter (artifact_id + device_id = CPU should return only CPU replica)
  {
    tensorcast::daemon::v2::GetLoadedReplicasV2Request req;
    tensorcast::daemon::v2::GetLoadedReplicasV2Response resp;
    req.set_artifact_id_filter(filter);
    req.set_device_id_filter(-1);
    grpc::ServerContext ctx;
    auto st = svc.GetLoadedReplicasV2(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.replicas_size() >= 1);
    for (const auto& r : resp.replicas()) {
      REQUIRE(r.device_id() == -1);
    }
  }
}
