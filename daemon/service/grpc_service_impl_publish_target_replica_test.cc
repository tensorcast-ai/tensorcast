// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <unistd.h>
#include "absl/time/time.h"
#include "core/common/capability_token.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/state/target_write_registry.h"
#include "grpcpp/server_context.h"
#include "tensorcast/common/v1/capability_token.pb.h"
#include "tensorcast/common/v1/common.pb.h"

namespace {

constexpr int kDeviceId = 0;

std::filesystem::path make_storage_root() {
  auto root = std::filesystem::temp_directory_path() / "tensorcast_publish_target_test";
  std::filesystem::create_directories(root);
  return root;
}

tensorcast::store::StoreEngineOptions make_engine_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = make_storage_root();
  opts.p2p_port = 47013;
  opts.memory_pool_size = 64ull << 20;
  opts.tx_slice_bytes = 1ull << 20;
  opts.num_thread = 2;
  return opts;
}

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    const std::shared_ptr<tensorcast::store::StoreEngine>& engine,
    const std::shared_ptr<tensorcast::store::testing::RecordingGlobalStoreClient>& gs) {
  tensorcast::daemon::DaemonOptions options;
  options.storage_path = make_storage_root();
  options.daemon_id = "daemon-test";
  options.capability_tokens.active.version = 1;
  options.capability_tokens.active.secret = "secret";
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options, nullptr, gs);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

tensorcast::common::v1::TargetWriteScope make_scope(
    std::string write_id,
    std::string artifact_id,
    std::string device_uuid,
    int owner_pid,
    bool publishable) {
  tensorcast::common::v1::TargetWriteScope scope;
  scope.set_write_id(std::move(write_id));
  scope.set_device_uuid(std::move(device_uuid));
  scope.set_owner_pid(owner_pid);
  scope.set_target_layout_hash("layout-hash");
  auto* space = scope.mutable_byte_space();
  space->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  space->set_id("");
  auto* selection = scope.mutable_selection();
  selection->set_artifact_id(std::move(artifact_id));
  selection->set_view_id("");
  selection->set_logical_layout_hash("logical-hash");
  selection->set_selection_hash("selection-hash");
  if (!publishable) {
    selection->add_tensor_names("alpha");
  }
  return scope;
}

tensorcast::common::v1::TargetWriteScope make_view_subset_scope(
    std::string write_id,
    std::string artifact_id,
    std::string device_uuid,
    int owner_pid) {
  auto scope = make_scope(
      std::move(write_id),
      std::move(artifact_id),
      std::move(device_uuid),
      owner_pid,
      /*publishable=*/false);
  scope.mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_VIEW);
  scope.mutable_byte_space()->set_id("mapped:v1:rank0");
  scope.mutable_selection()->set_view_id("mapped:v1:rank0");
  scope.mutable_selection()->set_view_subset_hash("subset-hash");
  return scope;
}

std::string mint_token(
    const tensorcast::common::CapabilityTokenManager& manager,
    std::string_view issuer,
    const tensorcast::common::v1::TargetWriteScope& scope) {
  auto scope_bytes_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_bytes_or.ok());
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Minutes(5)));
  auto token_or =
      manager.mint(issuer, tensorcast::common::v1::CAPABILITY_AUDIENCE_TARGET_WRITE, *scope_bytes_or, expires_at_ms);
  REQUIRE(token_or.ok());
  return *token_or;
}

tensorcast::daemon::TargetWriteRegistry::Record make_record_from_scope(
    const tensorcast::common::v1::TargetWriteScope& scope) {
  tensorcast::daemon::TargetWriteRegistry::Record record;
  record.write_id = scope.write_id();
  record.layout_key = "layout-hash";
  record.target_layout_hash = "layout-hash";
  record.selection.CopyFrom(scope.selection());
  record.byte_space.CopyFrom(scope.byte_space());
  record.canonical_index_json = "{}";
  record.index_key_hex = "deadbeef";
  record.device_uuid = scope.device_uuid();
  record.owner_pid = scope.owner_pid();
  record.expires_at = absl::Now() + absl::Minutes(5);
  return record;
}

} // namespace

TEST_CASE("PublishTargetReplica rejects owner mismatch", "[daemon][publish]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  const auto scope = make_scope("write-1", "artifact-1", "gpu-0", owner_pid, true);
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  req.set_target_write_token(token);
  req.mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  req.set_owner_pid(owner_pid + 1);

  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::PERMISSION_DENIED);
}

TEST_CASE("PublishTargetReplica rejects packed selections", "[daemon][publish]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  const auto scope = make_scope("write-2", "artifact-2", "gpu-0", owner_pid, false);
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_record_from_scope(scope);
  harness->materialization_controller().insert_target_write_for_testing(std::move(record));

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  req.set_target_write_token(token);
  req.mutable_byte_space()->set_kind(tensorcast::common::v1::BYTE_SPACE_KIND_CANONICAL);
  req.set_owner_pid(owner_pid);

  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_CASE("PublishTargetReplica allows packed selection for view byte-space", "[daemon][publish]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  const auto scope = make_view_subset_scope("write-3", "artifact-3", "gpu-0", owner_pid);
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_record_from_scope(scope);
  harness->materialization_controller().insert_target_write_for_testing(std::move(record));

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  req.set_target_write_token(token);
  req.mutable_byte_space()->CopyFrom(scope.byte_space());
  req.set_owner_pid(owner_pid);

  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(st.error_message() == "target_write_token has empty segments");
  REQUIRE(resp.lease_id().empty());
  REQUIRE(resp.replica_id().empty());
  REQUIRE(gs->registered_replicas.empty());
}
