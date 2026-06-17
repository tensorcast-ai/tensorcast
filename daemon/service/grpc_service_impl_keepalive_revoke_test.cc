// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <unistd.h>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/state/types.h"
#include "grpcpp/server_context.h"

namespace {

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = std::filesystem::temp_directory_path() / "tensorcast_daemon_cpp_test";
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47004;
  opts.memory_pool_size = 64ull << 20;
  opts.tx_slice_bytes = 1ull << 20;
  opts.num_thread = 2;
  return opts;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
    std::shared_ptr<tensorcast::store::testing::RecordingGlobalStoreClient> gs = nullptr) {
  tensorcast::daemon::DaemonOptions options;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options, nullptr, std::move(gs));
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

tensorcast::daemon::LipLeaseEntry make_lip_lease(std::string registration_id, std::string artifact_id) {
  tensorcast::daemon::LipLeaseEntry entry;
  entry.registration_id = std::move(registration_id);
  entry.artifact_id = std::move(artifact_id);
  entry.device_id = 0;
  entry.owner_pid = getpid();
  entry.ttl_ms = 60000;
  entry.expiry = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  return entry;
}

} // namespace

TEST_CASE("KeepAlive/Revoke lifecycle no-ops", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  // Begin coalesced registration
  grpc::ServerContext ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactRequest breq;
  breq.set_device_id(0);
  breq.set_total_size(1024 * 1024);
  breq.set_owner_pid(getpid());
  auto* idx = breq.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v3");
  idx->set_encoding("json");
  tensorcast::daemon::v2::BeginRegisterArtifactResponse bresp;
  auto st = service.BeginRegisterArtifact(&ctx, &breq, &bresp);
  REQUIRE(st.ok());

  // KeepAlive should return OK
  tensorcast::daemon::v2::KeepAliveRegisterArtifactRequest kreq;
  tensorcast::daemon::v2::KeepAliveRegisterArtifactResponse kresp;
  kreq.set_registration_id(bresp.registration_id());
  kreq.set_ttl_ms(2000);
  kreq.set_epoch(1);
  kreq.set_owner_pid(getpid());
  st = service.KeepAliveRegisterArtifact(&ctx, &kreq, &kresp);
  REQUIRE(st.ok());

  // Revoke should return OK
  tensorcast::daemon::v2::RevokeRegisteredArtifactRequest rreq;
  tensorcast::daemon::v2::RevokeRegisteredArtifactResponse rresp;
  rreq.set_registration_id(bresp.registration_id());
  rreq.set_reason("test");
  st = service.RevokeRegisteredArtifact(&ctx, &rreq, &rresp);
  REQUIRE(st.ok());
}

TEST_CASE("RevokeRegisteredArtifact is a no-op for unknown registration", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::RevokeRegisteredArtifactRequest rreq;
  tensorcast::daemon::v2::RevokeRegisteredArtifactResponse rresp;
  rreq.set_registration_id("missing-registration");
  rreq.set_reason("test");
  auto st = service.RevokeRegisteredArtifact(&ctx, &rreq, &rresp);
  REQUIRE(st.ok());
}

TEST_CASE("RevokeRegisteredArtifact unregisters routable Global Store replica", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);
  auto& lip_manager = harness->kernel().lip_manager();
  auto& service = harness->service();

  auto lease = make_lip_lease("registration-revoke", "artifact-revoke");
  tensorcast::daemon::ArtifactDeviceKey key{
      .artifact_id = lease.artifact_id,
      .view_id = "",
      .device_id = lease.device_id,
  };
  lip_manager.put_lease(lease.registration_id, key, lease);
  lip_manager.attach_replica_id(lease.registration_id, "replica-revoke");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::RevokeRegisteredArtifactRequest req;
  tensorcast::daemon::v2::RevokeRegisteredArtifactResponse resp;
  req.set_registration_id(lease.registration_id);
  req.set_reason("source_visibility_revoke");
  const auto st = service.RevokeRegisteredArtifact(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE(
      gs->unregistered_replicas ==
      std::vector<std::pair<std::string, std::string>>{
          {"artifact-revoke", "replica-revoke"},
      });
  REQUIRE_FALSE(lip_manager.find_active_by_key(key).has_value());
}

TEST_CASE("KeepAliveRegisterArtifact rejects unknown registration", "[daemon][registration]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  auto harness = make_harness(engine);
  auto& service = harness->service();

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::KeepAliveRegisterArtifactRequest kreq;
  tensorcast::daemon::v2::KeepAliveRegisterArtifactResponse kresp;
  kreq.set_registration_id("missing-registration");
  kreq.set_ttl_ms(2000);
  kreq.set_epoch(1);
  kreq.set_owner_pid(getpid());
  auto st = service.KeepAliveRegisterArtifact(&ctx, &kreq, &kresp);
  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::NOT_FOUND);
}
