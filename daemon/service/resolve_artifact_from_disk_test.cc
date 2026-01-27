// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/store_engine.h"
#include "core/testing/common.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/background_scheduler.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/verification_tracker.h"
#include "grpcpp/server_context.h"
#include "nlohmann/json.hpp"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

using tensorcast::daemon::MaterializationController;
using tensorcast::daemon::v2::ResolveArtifactFromDiskRequest;
using tensorcast::daemon::v2::ResolveArtifactFromDiskResponse;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_resolve_disk_test";
}

std::filesystem::path ensure_dir(std::filesystem::path path) {
  std::filesystem::create_directories(path);
  return path;
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47015;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

uint64_t generation_from_bytes(const std::string& canonical_index_json) {
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(
          reinterpret_cast<const uint8_t*>(canonical_index_json.data()), canonical_index_json.size()));
  uint64_t value = 0;
  const size_t limit = std::min<size_t>(8, digest.size());
  for (size_t i = 0; i < limit; ++i) {
    value = (value << 8) | static_cast<uint64_t>(digest[i]);
  }
  return value;
}

struct ResolveFixture {
  std::shared_ptr<tensorcast::store::StoreEngine> engine;
  tensorcast::daemon::RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions;
  tensorcast::daemon::LipManager lip_mgr;
  tensorcast::daemon::LipBridge lip_bridge;
  tensorcast::daemon::ReplicaSessionManager session_mgr;
  tensorcast::daemon::VerificationTracker verif_tracker;
  tensorcast::daemon::BackgroundScheduler scheduler;
  tensorcast::daemon::SessionsService sessions_svc;
  tensorcast::daemon::DeviceResolver devices;
  tensorcast::daemon::ShutdownSignal shutdown_signal;
  MaterializationController controller;

  explicit ResolveFixture(std::filesystem::path storage_root = test_tmpdir())
      : engine(std::make_shared<tensorcast::store::StoreEngine>(make_opts())),
        regions(tensorcast::daemon::IpcRegionRegistry::Options{}),
        lip_mgr(engine, &regions),
        lip_bridge(lip_mgr),
        session_mgr(std::chrono::seconds(60)),
        verif_tracker(),
        scheduler(),
        sessions_svc(session_mgr, verif_tracker, &scheduler, /*lifecycle=*/nullptr, absl::Seconds(60)),
        devices(tensorcast::store::DeviceRegistry::instance()),
        controller(MaterializationController(
            MaterializationController::Dep{
                .engine = *engine,
                .refs = refs,
                .sessions = sessions_svc,
                .lip = lip_bridge,
                .devices = devices,
                .regions = regions,
                .shutdown_signal = shutdown_signal,
                .global_store_client = nullptr,
                .lifecycle = nullptr,
                .storage_path = ensure_dir(std::move(storage_root)),
            })) {}
};

} // namespace

TEST_CASE(
    "ResolveArtifactFromDisk returns canonical index bytes and generation for disk artifacts",
    "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_ok";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  ResolveFixture fix;
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  req.set_verify_checksums(true);
  ResolveArtifactFromDiskResponse resp;
  auto status = fix.controller.resolve_artifact_from_disk(rctx, req, resp);

  REQUIRE(status.ok());
  REQUIRE_FALSE(resp.artifact_id().empty());
  REQUIRE(resp.generation() == generation_from_bytes(resp.canonical_index_bytes()));
  REQUIRE_FALSE(resp.canonical_index_bytes().empty());
}

TEST_CASE("ResolveArtifactFromDisk enforces shared root and missing paths", "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_denied";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  ResolveFixture denied_fix(artifact_dir.parent_path() / "unrelated_root");
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext denied_rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  ResolveArtifactFromDiskResponse resp;
  auto status = denied_fix.controller.resolve_artifact_from_disk(denied_rctx, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);

  ResolveFixture ok_fix(artifact_dir.parent_path());
  tensorcast::daemon::RpcContext ok_rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskResponse missing_resp;
  auto missing_status = ok_fix.controller.resolve_artifact_from_disk(ok_rctx, req, missing_resp);
  REQUIRE_FALSE(missing_status.ok());
  REQUIRE(missing_status.error_code() == grpc::StatusCode::NOT_FOUND);
}

TEST_CASE("ResolveArtifactFromDisk fails checksum validation when descriptor mismatches", "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_checksum";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 32));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  // Corrupt index_multihash to trigger checksum failure.
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  nlohmann::json descriptor = nlohmann::json::parse(std::ifstream(descriptor_path));
  descriptor["index_multihash"] = "mh:corrupt";
  {
    std::ofstream out(descriptor_path, std::ios::trunc);
    REQUIRE(out.is_open());
    out << descriptor.dump(2);
  }

  ResolveFixture fix;
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  req.set_verify_checksums(true);
  ResolveArtifactFromDiskResponse resp;
  auto status = fix.controller.resolve_artifact_from_disk(rctx, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
}
