// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "grpcpp/server_context.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace {

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env != nullptr && *env != '\0') {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_operation_rpc_test";
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

class OperationClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::vector<tensorcast::operation::v1::OperationState> states;
  std::atomic<int> get_calls{0};
  absl::Status get_error = absl::OkStatus();
  std::function<void(tensorcast::operation::v1::OperationRef*)> fill_ref;

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest& req) override {
    get_calls.fetch_add(1, std::memory_order_relaxed);
    if (!get_error.ok()) {
      return get_error;
    }
    tensorcast::operation::v1::GetOperationResponse resp;
    auto* ref = resp.mutable_ref();
    ref->set_operation_id(req.operation_id());
    if (fill_ref != nullptr) {
      fill_ref(ref);
    }
    auto* status = resp.mutable_status();
    const size_t idx = static_cast<size_t>(std::max(0, get_calls.load(std::memory_order_relaxed) - 1));
    const size_t state_idx = states.empty() ? 0 : std::min(idx, states.size() - 1);
    status->set_state(
        states.empty() ? tensorcast::operation::v1::OperationState::OPERATION_STATE_UNSPECIFIED : states[state_idx]);
    return resp;
  }
};

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(std::shared_ptr<OperationClient> client) {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

} // namespace

TEST_CASE("GetOperation surfaces backend errors", "[daemon][operation]") {
  auto client = std::make_shared<OperationClient>();
  client->get_error = absl::UnavailableError("backend unavailable");
  auto harness = make_harness(client);

  tensorcast::operation::v1::GetOperationRequest req;
  req.set_operation_id("op-1");
  tensorcast::operation::v1::GetOperationResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().GetOperation(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::UNAVAILABLE);
}

TEST_CASE("GetOperation bypasses child-owner admission for unrelated operation kinds", "[daemon][operation]") {
  auto client = std::make_shared<OperationClient>();
  client->states = {
      tensorcast::operation::v1::OperationState::OPERATION_STATE_RUNNING,
  };
  client->fill_ref = [](tensorcast::operation::v1::OperationRef* ref) {
    ref->set_kind("assembly_attempt");
    ref->set_authority_scope_kind("assembly_attempt");
    ref->set_authority_scope_id("attempt-1");
    ref->set_attachment_kind("assembly_attempt");
    ref->set_recovery_class("coordinator_process");
  };
  auto harness = make_harness(client);

  tensorcast::operation::v1::GetOperationRequest req;
  req.set_operation_id("op-assembly");
  tensorcast::operation::v1::GetOperationResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().GetOperation(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.ref().kind() == "assembly_attempt");
}

TEST_CASE("GetOperation routes publish observation through shared admission dispatcher", "[daemon][operation]") {
  auto client = std::make_shared<OperationClient>();
  client->states = {
      tensorcast::operation::v1::OperationState::OPERATION_STATE_RUNNING,
  };
  client->fill_ref = [](tensorcast::operation::v1::OperationRef* ref) {
    ref->set_kind("publish_target_replica");
    ref->set_authority_scope_kind("wrong_scope");
    ref->set_authority_scope_id("wf-1");
    ref->set_attachment_kind("target_publication");
    ref->set_recovery_class("ephemeral_process_local");
  };
  auto harness = make_harness(client);

  tensorcast::operation::v1::GetOperationRequest req;
  req.set_operation_id("op-publish");
  tensorcast::operation::v1::GetOperationResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().GetOperation(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(std::string(st.error_message()).find("authority_scope_kind mismatch") != std::string::npos);
}

TEST_CASE("WaitOperation returns once operation becomes terminal", "[daemon][operation]") {
  auto client = std::make_shared<OperationClient>();
  client->states = {
      tensorcast::operation::v1::OperationState::OPERATION_STATE_RUNNING,
      tensorcast::operation::v1::OperationState::OPERATION_STATE_RUNNING,
      tensorcast::operation::v1::OperationState::OPERATION_STATE_SUCCESS,
  };
  auto harness = make_harness(client);

  tensorcast::daemon::v2::WaitOperationRequest req;
  req.set_operation_id("op-2");
  req.set_timeout_ms(1000);
  tensorcast::daemon::v2::WaitOperationResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().WaitOperation(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.operation().status().state() == tensorcast::operation::v1::OperationState::OPERATION_STATE_SUCCESS);
  REQUIRE(client->get_calls.load(std::memory_order_relaxed) >= 3);
}

TEST_CASE("WaitOperation returns latest state on timeout", "[daemon][operation]") {
  auto client = std::make_shared<OperationClient>();
  client->states = {tensorcast::operation::v1::OperationState::OPERATION_STATE_RUNNING};
  auto harness = make_harness(client);

  tensorcast::daemon::v2::WaitOperationRequest req;
  req.set_operation_id("op-3");
  req.set_timeout_ms(1);
  tensorcast::daemon::v2::WaitOperationResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().WaitOperation(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.operation().status().state() == tensorcast::operation::v1::OperationState::OPERATION_STATE_RUNNING);
  REQUIRE(client->get_calls.load(std::memory_order_relaxed) >= 1);
}
