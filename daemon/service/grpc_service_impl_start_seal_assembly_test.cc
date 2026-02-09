// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "grpcpp/server_context.h"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_start_seal_assembly_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 0;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

class StartSealLeaseReleaseClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::string expected_lease_token{"lease-token-test"};
  std::atomic<int> running_update_calls{0};
  std::atomic<int> release_calls{0};
  std::atomic<bool> released_expected_token{false};

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest& req) override {
    tensorcast::operation::v1::AcquireOperationLeaseResponse resp;
    resp.set_acquired(true);
    auto* lease = resp.mutable_lease();
    lease->set_operation_id(req.operation_id());
    lease->set_lease_token(expected_lease_token);
    lease->set_owner_id("test-owner");
    lease->set_lease_generation(1);
    return resp;
  }

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest&) override {
    return absl::NotFoundError("operation not found");
  }

  absl::Status update_operation(const tensorcast::operation::v1::UpdateOperationRequest& req) override {
    if (req.status().state() == tensorcast::operation::v1::OperationState::OPERATION_STATE_RUNNING) {
      running_update_calls.fetch_add(1, std::memory_order_relaxed);
      return absl::InternalError("injected running update failure");
    }
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::operation::v1::ReleaseOperationLeaseResponse> release_operation_lease(
      const tensorcast::operation::v1::ReleaseOperationLeaseRequest& req) override {
    release_calls.fetch_add(1, std::memory_order_relaxed);
    if (req.lease_token() == expected_lease_token) {
      released_expected_token.store(true, std::memory_order_relaxed);
    }
    tensorcast::operation::v1::ReleaseOperationLeaseResponse resp;
    resp.set_released(true);
    return resp;
  }
};

} // namespace

TEST_CASE("StartSealAssembly releases operation lease when RUNNING update fails", "[daemon][seal][lease]") {
  auto gs_client = std::make_shared<StartSealLeaseReleaseClient>();
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::StartSealAssemblyRequest req;
  req.set_assembly_id("assembly:test");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::StartSealAssemblyResponse resp;
  const auto st = svc.StartSealAssembly(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE_FALSE(resp.operation().operation_id().empty());

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline && gs_client->release_calls.load(std::memory_order_relaxed) == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  REQUIRE(gs_client->running_update_calls.load(std::memory_order_relaxed) > 0);
  REQUIRE(gs_client->release_calls.load(std::memory_order_relaxed) > 0);
  REQUIRE(gs_client->released_expected_token.load(std::memory_order_relaxed));
}
