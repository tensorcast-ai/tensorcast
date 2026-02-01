// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>

#include "core/common/capability_token.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "grpcpp/server_context.h"
#include "tensorcast/common/v1/capability_token.pb.h"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_placement_lease_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47021;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

} // namespace

TEST_CASE("Placement lease RPCs require capability tokens", "[daemon][grpc][placement_lease]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(daemon_opts.storage_path);

  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  const std::string artifact_id = "mi2:test:pin";
  const int device_id = 0;

  tensorcast::store::loading::ReplicaKey key;
  key.artifact_id = artifact_id;
  key.device = tensorcast::store::DeviceRegistry::instance().gpu_key(device_id);
  key.replica = 0;

  std::string token;
  uint64_t lease_id = 0;

  // Create.
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::CreatePlacementLeaseRequest req;
    req.set_artifact_id(artifact_id);
    req.set_device_id(device_id);
    req.set_ttl_ms(1000);
    tensorcast::daemon::v2::CreatePlacementLeaseResponse resp;
    const auto st = svc.CreatePlacementLease(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.lease_id() != 0);
    REQUIRE_FALSE(resp.lease_token().empty());
    REQUIRE(resp.has_expires_at());
    lease_id = resp.lease_id();
    token = resp.lease_token();
  }

  REQUIRE(harness->kernel().lifecycle_manager().placement_pin_count_for(key) == 1);

  // Renew with invalid token.
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::RenewPlacementLeaseRequest req;
    req.set_lease_token("bad-token");
    req.set_ttl_ms(1000);
    tensorcast::daemon::v2::RenewPlacementLeaseResponse resp;
    const auto st = svc.RenewPlacementLease(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::NOT_FOUND);
  }

  // Renew.
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::RenewPlacementLeaseRequest req;
    req.set_lease_token(token);
    req.set_ttl_ms(2000);
    tensorcast::daemon::v2::RenewPlacementLeaseResponse resp;
    const auto st = svc.RenewPlacementLease(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.lease_id() == lease_id);
    REQUIRE(resp.has_expires_at());
  }

  // Release.
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::ReleasePlacementLeaseRequest req;
    req.set_lease_token(token);
    tensorcast::daemon::v2::ReleasePlacementLeaseResponse resp;
    const auto st = svc.ReleasePlacementLease(&ctx, &req, &resp);
    REQUIRE(st.ok());
    REQUIRE(resp.released());
  }

  REQUIRE(harness->kernel().lifecycle_manager().placement_pin_count_for(key) == 0);

  // Renew after release fails.
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::RenewPlacementLeaseRequest req;
    req.set_lease_token(token);
    req.set_ttl_ms(1000);
    tensorcast::daemon::v2::RenewPlacementLeaseResponse resp;
    const auto st = svc.RenewPlacementLease(&ctx, &req, &resp);
    REQUIRE(st.error_code() == grpc::StatusCode::NOT_FOUND);
  }
}

TEST_CASE("Placement lease tokens use capability envelope when configured", "[daemon][grpc][placement_lease]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  daemon_opts.daemon_id = "daemon-test";
  daemon_opts.capability_tokens.active.version = 1;
  daemon_opts.capability_tokens.active.secret = "secret_v1";
  std::filesystem::create_directories(daemon_opts.storage_path);

  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  const std::string artifact_id = "mi2:test:pin_v2";
  const int device_id = 0;

  std::string token;
  uint64_t lease_id = 0;

  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::CreatePlacementLeaseRequest req;
    req.set_artifact_id(artifact_id);
    req.set_device_id(device_id);
    req.set_ttl_ms(1000);
    tensorcast::daemon::v2::CreatePlacementLeaseResponse resp;
    const auto st = svc.CreatePlacementLease(&ctx, &req, &resp);
    REQUIRE(st.ok());
    lease_id = resp.lease_id();
    token = resp.lease_token();
    REQUIRE_FALSE(token.empty());
  }

  tensorcast::common::CapabilityTokenManager mgr(
      tensorcast::common::CapabilityTokenConfig{
          .active = tensorcast::common::CapabilityTokenKey{.version = 1, .secret = "secret_v1"},
          .previous = {},
      });
  REQUIRE(mgr.configured());

  auto env_or = mgr.verify(
      token,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE,
      "daemon-test",
      absl::Now(),
      /*require_not_expired=*/true);
  REQUIRE(env_or.ok());
  tensorcast::common::v1::PlacementLeaseScope scope;
  REQUIRE(scope.ParseFromString(env_or->scope()));
  REQUIRE(scope.lease_id() == lease_id);

  std::string token2;
  {
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::RenewPlacementLeaseRequest req;
    req.set_lease_token(token);
    req.set_ttl_ms(2000);
    tensorcast::daemon::v2::RenewPlacementLeaseResponse resp;
    const auto st = svc.RenewPlacementLease(&ctx, &req, &resp);
    REQUIRE(st.ok());
    token2 = resp.lease_token();
    REQUIRE_FALSE(token2.empty());
  }

  auto env_or2 = mgr.verify(
      token2,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_PLACEMENT_LEASE,
      "daemon-test",
      absl::Now(),
      /*require_not_expired=*/true);
  REQUIRE(env_or2.ok());
}
