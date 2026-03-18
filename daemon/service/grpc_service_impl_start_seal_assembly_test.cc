// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "google/protobuf/util/time_util.h"
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

class StartAssemblyAttemptClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  tensorcast::operation::v1::UpdateOperationRequest last_update;
  tensorcast::store::components::AssemblyAttemptRecordInfo last_attempt;
  std::atomic<int> acquire_calls{0};
  std::atomic<int> update_calls{0};
  std::atomic<int> attempt_upserts{0};

  absl::StatusOr<tensorcast::layout::v1::LayoutSpecRecord> get_layout_spec(std::string_view layout_id) override {
    tensorcast::layout::v1::LayoutSpecRecord record;
    record.set_layout_id(std::string(layout_id));
    record.mutable_layout()->set_layout_schema_version(1);
    record.mutable_layout()->set_index_multihash("bafkindex");
    record.mutable_layout()->add_expected_view_ids("view-a");
    record.mutable_layout()->add_expected_view_ids("view-b");
    return record;
  }

  absl::StatusOr<tensorcast::global_store::v1::AssemblyLayoutBinding> update_assembly_layout_binding(
      std::string_view assembly_id,
      std::string_view layout_id,
      uint64_t expected_binding_version) override {
    if (expected_binding_version != 0) {
      return absl::FailedPreconditionError("unexpected binding version");
    }
    tensorcast::global_store::v1::AssemblyLayoutBinding binding;
    binding.set_assembly_id(std::string(assembly_id));
    binding.set_layout_id(std::string(layout_id));
    binding.set_binding_version(1);
    return binding;
  }

  absl::StatusOr<tensorcast::store::components::AssemblyAttemptRecordInfo> upsert_assembly_attempt(
      const tensorcast::store::components::AssemblyAttemptRecordInfo& attempt) override {
    last_attempt = attempt;
    attempt_upserts.fetch_add(1, std::memory_order_relaxed);
    return attempt;
  }

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest& req) override {
    acquire_calls.fetch_add(1, std::memory_order_relaxed);
    tensorcast::operation::v1::AcquireOperationLeaseResponse resp;
    resp.set_acquired(true);
    auto* lease = resp.mutable_lease();
    lease->set_operation_id(req.operation_id());
    lease->set_lease_token("attempt-lease-token");
    lease->set_owner_id(req.owner_id());
    lease->set_lease_generation(1);
    return resp;
  }

  absl::Status update_operation(const tensorcast::operation::v1::UpdateOperationRequest& req) override {
    update_calls.fetch_add(1, std::memory_order_relaxed);
    last_update = req;
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::operation::v1::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const tensorcast::operation::v1::KeepaliveOperationLeaseRequest&) override {
    tensorcast::operation::v1::KeepaliveOperationLeaseResponse resp;
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

TEST_CASE("StartAssemblyAttempt persists attempt record and continuation metadata", "[daemon][assembly][attempt]") {
  auto gs_client = std::make_shared<StartAssemblyAttemptClient>();
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

  tensorcast::daemon::v2::StartAssemblyAttemptRequest req;
  req.set_layout_id("layout-1");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::StartAssemblyAttemptResponse resp;
  const auto st = svc.StartAssemblyAttempt(&ctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(gs_client->attempt_upserts.load(std::memory_order_relaxed) == 1);
  REQUIRE(gs_client->acquire_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(gs_client->update_calls.load(std::memory_order_relaxed) == 1);

  const auto& attempt = resp.attempt();
  REQUIRE_FALSE(attempt.attempt_id().empty());
  REQUIRE_FALSE(attempt.workspace_assembly_id().empty());
  REQUIRE(attempt.layout_id() == "layout-1");
  REQUIRE_FALSE(attempt.attempt_intent_digest().empty());
  REQUIRE(attempt.coordinator_generation() == 1);
  REQUIRE(attempt.has_coordinator_operation());
  REQUIRE(attempt.coordinator_operation().kind() == "assembly_attempt");
  REQUIRE(attempt.coordinator_operation().authority_scope_id() == attempt.attempt_id());
  REQUIRE(attempt.coordinator_operation().target_artifact_id() == attempt.workspace_assembly_id());

  tensorcast::daemon::v2::AssemblyAttemptRecord record;
  REQUIRE(record.ParseFromString(gs_client->last_attempt.attempt_record_proto));
  REQUIRE(record.attempt_id() == attempt.attempt_id());
  REQUIRE(record.workspace_assembly_id() == attempt.workspace_assembly_id());
  REQUIRE(record.intent().layout_id() == "layout-1");
  REQUIRE(record.intent().requirements().inline_requirements_size() == 2);
  REQUIRE(
      record.intent().closeout_contract().kind() == tensorcast::daemon::v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY);

  REQUIRE(gs_client->last_update.status().state() == tensorcast::operation::v1::OPERATION_STATE_PENDING);
  REQUIRE(gs_client->last_update.has_snapshot());
  tensorcast::operation::v1::OperationContinuationMetadata metadata;
  REQUIRE(gs_client->last_update.snapshot().UnpackTo(&metadata));
  REQUIRE(metadata.ref().kind() == "assembly_attempt");
  REQUIRE(metadata.ref().authority_scope_id() == attempt.attempt_id());
  REQUIRE(metadata.ref().target_artifact_id() == attempt.workspace_assembly_id());
}

TEST_CASE("StartAssemblyAttempt rejects non dependency-ready closeout kinds", "[daemon][assembly][attempt]") {
  auto gs_client = std::make_shared<StartAssemblyAttemptClient>();
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

  tensorcast::daemon::v2::StartAssemblyAttemptRequest req;
  req.set_layout_id("layout-1");
  req.mutable_closeout_contract()->set_kind(tensorcast::daemon::v2::ASSEMBLY_CLOSEOUT_KIND_REPRESENTATION_PUBLISH);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::StartAssemblyAttemptResponse resp;
  const auto st = svc.StartAssemblyAttempt(&ctx, &req, &resp);

  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::UNIMPLEMENTED);
  REQUIRE(st.error_message().find("source_publish_only") != std::string::npos);
}

TEST_CASE("StartAssemblyAttempt rejects serving fields for source_publish_only", "[daemon][assembly][attempt]") {
  auto gs_client = std::make_shared<StartAssemblyAttemptClient>();
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

  tensorcast::daemon::v2::StartAssemblyAttemptRequest req;
  req.set_layout_id("layout-1");
  auto* closeout = req.mutable_closeout_contract();
  closeout->set_kind(tensorcast::daemon::v2::ASSEMBLY_CLOSEOUT_KIND_SOURCE_PUBLISH_ONLY);
  closeout->set_source_version_key("models/demo/source/v1");
  closeout->set_serving_version_key("models/demo/serving/v1");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::StartAssemblyAttemptResponse resp;
  const auto st = svc.StartAssemblyAttempt(&ctx, &req, &resp);

  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(st.error_message().find("serving_version_key") != std::string::npos);
}
