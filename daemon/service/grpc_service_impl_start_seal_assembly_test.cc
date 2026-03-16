// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <thread>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/service/controllers/assembly_coordination_utils.h"
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

class StartAssemblyAttemptClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  tensorcast::operation::v1::UpdateOperationRequest last_update;
  std::atomic<int> acquire_calls{0};
  std::atomic<int> update_calls{0};

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

class StartSealRequiresSnapshotClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::atomic<int> acquire_calls{0};

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest& req) override {
    tensorcast::operation::v1::GetOperationResponse resp;
    (void)req;
    resp.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_PENDING);
    return resp;
  }

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest& req) override {
    acquire_calls.fetch_add(1, std::memory_order_relaxed);
    tensorcast::operation::v1::AcquireOperationLeaseResponse resp;
    resp.set_acquired(true);
    auto* lease = resp.mutable_lease();
    lease->set_operation_id(req.operation_id());
    lease->set_lease_token("unexpected-lease");
    lease->set_owner_id(req.owner_id());
    lease->set_lease_generation(1);
    return resp;
  }
};

class StartSealStaleGenerationClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::atomic<int> acquire_calls{0};

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest& req) override {
    tensorcast::operation::v1::GetOperationResponse resp;
    resp.mutable_ref()->set_operation_id(req.operation_id());
    resp.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_PENDING);
    resp.set_lease_generation(2);
    tensorcast::daemon::v2::SealAssemblySnapshot snapshot;
    snapshot.set_assembly_id("assembly:test");
    snapshot.set_layout_id("layout-1");
    snapshot.set_contribution_contract_hash("contract-hash");
    auto* entry = snapshot.mutable_contribution_contract()->add_required_contributions();
    entry->set_view_id("view-a");
    entry->set_contribution_kind(tensorcast::daemon::v2::BINDING_CONTRIBUTION_KIND_PIECE_PARTIAL);
    entry->set_coverage_semantics("phase1_layout_expected_view");
    snapshot.mutable_contribution_contract()->set_layout_id("layout-1");
    snapshot.mutable_contribution_contract()->set_require_live_contributions(true);
    resp.mutable_snapshot()->PackFrom(snapshot);
    return resp;
  }

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest& req) override {
    acquire_calls.fetch_add(1, std::memory_order_relaxed);
    tensorcast::operation::v1::AcquireOperationLeaseResponse resp;
    resp.set_acquired(true);
    auto* lease = resp.mutable_lease();
    lease->set_operation_id(req.operation_id());
    lease->set_lease_token("unexpected-lease");
    lease->set_owner_id(req.owner_id());
    lease->set_lease_generation(3);
    return resp;
  }
};

class CanonicalFullContributionLivenessClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  tensorcast::operation::v1::GetOperationResponse current_operation;
  tensorcast::store::components::AssemblyContributionInfo contribution_row;
  std::atomic<int> update_calls{0};

  CanonicalFullContributionLivenessClient() {
    current_operation.mutable_ref()->set_operation_id("pending-op");
    current_operation.mutable_ref()->set_kind("seal_assembly");
    current_operation.mutable_ref()->set_target_artifact_id("assembly:canonical");
    current_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_PENDING);
    current_operation.mutable_status()->set_message("assembly attempt open");
    current_operation.set_lease_generation(1);
    current_operation.set_lease_owner("test-owner");
    *current_operation.mutable_lease_expires_at() =
        google::protobuf::util::TimeUtil::TimeTToTimestamp(std::time(nullptr) + 60);

    tensorcast::daemon::v2::SealAssemblySnapshot snapshot;
    snapshot.set_assembly_id("assembly:canonical");
    snapshot.set_layout_id("layout-canonical");
    snapshot.set_contribution_contract_hash("contract-hash-canonical");
    auto* entry = snapshot.mutable_contribution_contract()->add_required_contributions();
    entry->set_view_id(std::string(tensorcast::daemon::assembly_coordination::kCanonicalFullContributionViewId));
    entry->set_contribution_kind(tensorcast::daemon::v2::BINDING_CONTRIBUTION_KIND_CANONICAL_FULL);
    entry->set_coverage_semantics("phase1_canonical_full");
    snapshot.mutable_contribution_contract()->set_layout_id("layout-canonical");
    snapshot.mutable_contribution_contract()->set_require_live_contributions(true);
    current_operation.mutable_snapshot()->PackFrom(snapshot);

    contribution_row.assembly_id = "assembly:canonical";
    contribution_row.view_id = std::string(tensorcast::daemon::assembly_coordination::kCanonicalFullContributionViewId);
    contribution_row.binding_id = "binding-1";
    contribution_row.binding_value_id = "value-1";
    contribution_row.coverage_plan_hash = "cph-1";
    contribution_row.contributor_daemon_id = "daemon-stale";
    contribution_row.coordinator_operation_id = "pending-op";
    contribution_row.coordinator_generation = 1;
    contribution_row.lease_id = "lease-1";
    contribution_row.lease_generation = 1;
    contribution_row.lease_expires_at = absl::Now() + absl::Seconds(30);
    contribution_row.state = "accepted";
  }

  absl::StatusOr<tensorcast::operation::v1::GetOperationResponse> get_operation(
      const tensorcast::operation::v1::GetOperationRequest& req) override {
    auto response = current_operation;
    response.mutable_ref()->set_operation_id(req.operation_id());
    return response;
  }

  absl::StatusOr<tensorcast::operation::v1::AcquireOperationLeaseResponse> acquire_operation_lease(
      const tensorcast::operation::v1::AcquireOperationLeaseRequest& req) override {
    tensorcast::operation::v1::AcquireOperationLeaseResponse resp;
    resp.set_acquired(true);
    auto* lease = resp.mutable_lease();
    lease->set_operation_id(req.operation_id());
    lease->set_lease_token("lease-token-canonical");
    lease->set_owner_id(req.owner_id());
    lease->set_lease_generation(1);
    return resp;
  }

  absl::StatusOr<tensorcast::operation::v1::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const tensorcast::operation::v1::KeepaliveOperationLeaseRequest&) override {
    tensorcast::operation::v1::KeepaliveOperationLeaseResponse resp;
    return resp;
  }

  absl::StatusOr<tensorcast::operation::v1::ReleaseOperationLeaseResponse> release_operation_lease(
      const tensorcast::operation::v1::ReleaseOperationLeaseRequest&) override {
    tensorcast::operation::v1::ReleaseOperationLeaseResponse resp;
    resp.set_released(true);
    return resp;
  }

  absl::StatusOr<std::vector<tensorcast::store::components::AssemblyContributionInfo>> list_assembly_contributions(
      std::optional<std::string_view> assembly_id,
      std::optional<std::string_view>,
      std::optional<std::string_view>,
      std::optional<std::string_view>,
      const std::vector<std::string>& states) override {
    if (assembly_id.has_value() && *assembly_id != contribution_row.assembly_id) {
      return std::vector<tensorcast::store::components::AssemblyContributionInfo>{};
    }
    if (std::find(states.begin(), states.end(), contribution_row.state) == states.end()) {
      return std::vector<tensorcast::store::components::AssemblyContributionInfo>{};
    }
    return std::vector<tensorcast::store::components::AssemblyContributionInfo>{contribution_row};
  }

  absl::StatusOr<std::vector<std::string>> list_active_worker_identities(bool) override {
    return std::vector<std::string>{};
  }

  absl::Status update_operation(const tensorcast::operation::v1::UpdateOperationRequest& req) override {
    update_calls.fetch_add(1, std::memory_order_relaxed);
    current_operation.mutable_status()->CopyFrom(req.status());
    if (req.has_snapshot()) {
      current_operation.mutable_snapshot()->CopyFrom(req.snapshot());
    }
    return absl::OkStatus();
  }
};

} // namespace

TEST_CASE("Contribution contract hash changes when per-view semantics change", "[daemon][assembly][contract]") {
  google::protobuf::RepeatedPtrField<std::string> expected_view_ids;
  *expected_view_ids.Add() = "view-a";
  *expected_view_ids.Add() = "view-b";

  auto baseline = tensorcast::daemon::assembly_coordination::build_phase1_contribution_contract(
      "layout-1",
      expected_view_ids,
      /*require_live_contributions=*/true);
  auto modified = baseline;
  modified.mutable_required_contributions(0)->set_coverage_semantics("phase1_view_semantics_v2");

  REQUIRE(
      tensorcast::daemon::assembly_coordination::compute_contribution_contract_hash(baseline) !=
      tensorcast::daemon::assembly_coordination::compute_contribution_contract_hash(modified));
}

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

TEST_CASE("StartAssemblyAttempt snapshots a live contribution contract", "[daemon][assembly][attempt]") {
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
  REQUIRE(gs_client->acquire_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(gs_client->update_calls.load(std::memory_order_relaxed) == 1);

  const auto& attempt = resp.attempt();
  REQUIRE_FALSE(attempt.assembly_id().empty());
  REQUIRE(attempt.layout_id() == "layout-1");
  REQUIRE(attempt.coordinator_generation() == 1);
  REQUIRE(attempt.expected_view_ids_size() == 2);
  REQUIRE(attempt.has_contribution_contract());
  REQUIRE(attempt.contribution_contract().require_live_contributions());

  REQUIRE(gs_client->last_update.status().state() == tensorcast::operation::v1::OPERATION_STATE_PENDING);
  REQUIRE(gs_client->last_update.has_snapshot());
  tensorcast::daemon::v2::SealAssemblySnapshot snapshot;
  REQUIRE(gs_client->last_update.snapshot().UnpackTo(&snapshot));
  REQUIRE(snapshot.contribution_contract_hash() == attempt.contribution_contract_hash());
  REQUIRE(snapshot.has_contribution_contract());
  REQUIRE(snapshot.contribution_contract().require_live_contributions());
  REQUIRE(snapshot.contribution_contract().required_contributions_size() == 2);
}

TEST_CASE(
    "StartSealAssembly requires an existing attempt snapshot when layout_id is provided",
    "[daemon][assembly][attempt]") {
  auto gs_client = std::make_shared<StartSealRequiresSnapshotClient>();
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
  req.set_layout_id("layout-1");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::StartSealAssemblyResponse resp;
  const auto st = svc.StartSealAssembly(&ctx, &req, &resp);

  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(st.error_message() == "assembly attempt snapshot is unavailable");
  REQUIRE(gs_client->acquire_calls.load(std::memory_order_relaxed) == 0);
}

TEST_CASE("StartSealAssembly rejects stale attempt coordinator generations", "[daemon][assembly][attempt]") {
  auto gs_client = std::make_shared<StartSealStaleGenerationClient>();
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
  req.set_layout_id("layout-1");
  req.set_expected_coordinator_generation(1);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::StartSealAssemblyResponse resp;
  const auto st = svc.StartSealAssembly(&ctx, &req, &resp);

  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(st.error_message() == "assembly attempt coordinator generation changed");
  REQUIRE(gs_client->acquire_calls.load(std::memory_order_relaxed) == 0);
}

TEST_CASE("StartSealAssembly requires a live canonical_full contributor", "[daemon][assembly][attempt]") {
  auto gs_client = std::make_shared<CanonicalFullContributionLivenessClient>();
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

  tensorcast::daemon::v2::StartSealAssemblyRequest start_req;
  start_req.set_assembly_id("assembly:canonical");
  start_req.set_layout_id("layout-canonical");
  start_req.set_expected_coordinator_generation(1);

  grpc::ServerContext start_ctx;
  tensorcast::daemon::v2::StartSealAssemblyResponse start_resp;
  const auto start_status = svc.StartSealAssembly(&start_ctx, &start_req, &start_resp);
  REQUIRE(start_status.ok());
  REQUIRE_FALSE(start_resp.operation().operation_id().empty());

  tensorcast::daemon::v2::WaitOperationRequest wait_req;
  wait_req.set_operation_id(start_resp.operation().operation_id());
  wait_req.set_timeout_ms(5000);
  grpc::ServerContext wait_ctx;
  tensorcast::daemon::v2::WaitOperationResponse wait_resp;
  const auto wait_status = svc.WaitOperation(&wait_ctx, &wait_req, &wait_resp);
  REQUIRE(wait_status.ok());
  REQUIRE(wait_resp.operation().status().state() == tensorcast::operation::v1::OPERATION_STATE_FAILED);
  REQUIRE(
      wait_resp.operation().status().error().message() ==
      "required canonical_full contribution missing from live contributor set");
}
