// Copyright (c) 2025, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <cstdlib>
#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/store/store_engine.h"
#include "daemon/background_scheduler.h"
#include "daemon/device_resolver.h"
#include "daemon/ipc_region_registry.h"
#include "daemon/lip_bridge.h"
#include "daemon/lip_manager.h"
#include "daemon/ref_tracker.h"
#include "daemon/replica_session_manager.h"
#include "daemon/rpc_context.h"
#include "daemon/sessions_service.h"
#include "daemon/verification_tracker.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

using tensorcast::daemon::MaterializationController;
using tensorcast::daemon::v2::MaterializeIntoTargetRequest;
using tensorcast::daemon::v2::MaterializeIntoTargetResponse;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_materialize_into_target_test";
}

std::filesystem::path ensure_dir(std::filesystem::path path) {
  std::filesystem::create_directories(path);
  return path;
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47016;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

struct ValidationFixture {
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
  std::atomic<bool> shutting_down{false};
  std::filesystem::path storage_root;
  MaterializationController controller;

  ValidationFixture()
      : engine(std::make_shared<tensorcast::store::StoreEngine>(make_opts())),
        regions(tensorcast::daemon::IpcRegionRegistry::Options{}),
        lip_mgr(engine, &regions),
        lip_bridge(lip_mgr),
        session_mgr(std::chrono::seconds(60)),
        verif_tracker(),
        scheduler(),
        sessions_svc(session_mgr, verif_tracker, &scheduler, /*lifecycle=*/nullptr, absl::Seconds(60)),
        devices(tensorcast::store::DeviceRegistry::instance()),
        storage_root(ensure_dir(test_tmpdir())),
        controller(MaterializationController(
            MaterializationController::Dep{
                .engine = *engine,
                .refs = refs,
                .sessions = sessions_svc,
                .lip = lip_bridge,
                .devices = devices,
                .regions = regions,
                .is_shutting_down = shutting_down,
                .lifecycle = nullptr,
                .storage_path = storage_root,
            })) {}
};

grpc::Status run_request(
    MaterializationController& controller,
    const MaterializeIntoTargetRequest& req,
    MaterializeIntoTargetResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"MaterializeIntoTargetTest", ctx, /*allow_high_card_attrs=*/true};
  return controller.materialize_into_target(rctx, req, resp);
}

} // namespace

TEST_CASE("MaterializeIntoTarget rejects missing artifact_id", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("MaterializeIntoTarget rejects empty disk_fallback path", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.mutable_disk_fallback();
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("MaterializeIntoTarget rejects subset requests", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.mutable_target_layout();
  req.add_tensor_names("foo");
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("MaterializeIntoTarget rejects view/view_id", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.mutable_target_layout();
  req.set_view_id("view:dummy");
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("MaterializeIntoTarget requires device_uuid", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.mutable_target_layout();
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}
