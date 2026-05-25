// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <unistd.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/common/capability_token.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/state/routed_authority_protocol.h"
#include "daemon/state/routed_authority_wire.h"
#include "daemon/state/target_publication_registry.h"
#include "daemon/state/types.h"
#include "daemon/testing/cuda_ipc_spawn_helper.h"
#include "grpcpp/server_context.h"
#include "tensorcast/common/v1/capability_token.pb.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace {

constexpr int kDeviceId = 0;

using tensorcast::daemon::ArtifactDeviceKey;
using tensorcast::daemon::LipLeaseEntry;

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
    const std::shared_ptr<tensorcast::store::testing::RecordingGlobalStoreClient>& gs,
    bool progressive_replication = false) {
  tensorcast::daemon::DaemonOptions options;
  options.storage_path = make_storage_root();
  options.daemon_id = "daemon-test";
  options.capability_tokens.active.version = 1;
  options.capability_tokens.active.secret = "secret";
  options.progressive_replication.enabled = progressive_replication;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, options, nullptr, gs);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

tensorcast::common::v1::BindingCurrentValuePublicationScope make_scope(
    std::string publication_id,
    std::string artifact_id,
    std::string device_uuid,
    int owner_pid,
    bool publishable) {
  tensorcast::common::v1::BindingCurrentValuePublicationScope scope;
  scope.set_publication_id(std::move(publication_id));
  scope.set_device_uuid(std::move(device_uuid));
  scope.set_owner_pid(owner_pid);
  scope.set_target_layout_hash("layout-hash");
  scope.set_binding_id(absl::StrCat("binding-", publication_id));
  scope.set_binding_layout_id("binding-layout");
  scope.set_binding_value_id(absl::StrCat("binding-value-", publication_id));
  scope.set_seal_generation(1);
  scope.set_daemon_id("daemon-test");
  scope.set_daemon_session_id("session-test");
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

tensorcast::common::v1::BindingCurrentValuePublicationScope make_view_subset_scope(
    std::string publication_id,
    std::string artifact_id,
    std::string device_uuid,
    int owner_pid) {
  auto scope = make_scope(
      std::move(publication_id),
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
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope) {
  auto scope_bytes_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_bytes_or.ok());
  const uint64_t expires_at_ms = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Minutes(5)));
  auto token_or = manager.mint(
      issuer,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_BINDING_CURRENT_VALUE_PUBLICATION,
      *scope_bytes_or,
      expires_at_ms);
  REQUIRE(token_or.ok());
  return *token_or;
}

tensorcast::daemon::TargetPublicationRegistry::Record make_record_from_scope(
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope) {
  tensorcast::daemon::TargetPublicationRegistry::Record record;
  record.publication_id = tensorcast::daemon::PublicationInstanceId{.value = scope.publication_id()};
  record.publication_subject_key = tensorcast::daemon::build_publication_subject_key(
      scope.selection(), scope.byte_space(), "layout-hash", scope.device_uuid());
  record.target_layout_hash = "layout-hash";
  record.selection.CopyFrom(scope.selection());
  record.byte_space.CopyFrom(scope.byte_space());
  record.canonical_index_json = "{}";
  record.index_key_hex = "deadbeef";
  record.device_uuid = scope.device_uuid();
  record.owner_pid = scope.owner_pid();
  record.daemon_id = scope.daemon_id();
  record.daemon_session_id = scope.daemon_session_id();
  record.binding_id = scope.binding_id();
  record.binding_layout_id = scope.binding_layout_id();
  record.binding_value_id = scope.binding_value_id();
  record.seal_generation = scope.seal_generation();
  record.expires_at = absl::Now() + absl::Minutes(5);
  return record;
}

tensorcast::daemon::TargetPublicationRegistry::Record make_publishable_record_from_scope(
    const tensorcast::common::v1::BindingCurrentValuePublicationScope& scope,
    std::string handle_bytes = "fake-cuda-ipc-handle") {
  auto record = make_record_from_scope(scope);
  tensorcast::daemon::RegisterStorageMeta storage;
  storage.storage_id = "storage-0";
  storage.device_id = kDeviceId;
  storage.handle_bytes = std::move(handle_bytes);
  storage.storage_length = 16;
  record.storages.push_back(storage);
  record.segments.push_back(
      tensorcast::daemon::LeaseSegMeta{
          .storage_id = storage.storage_id,
          .storage_offset = 0,
          .artifact_offset = 0,
          .length = 16,
      });
  return record;
}

LipLeaseEntry make_active_lip_lease(std::string registration_id, std::string artifact_id, int owner_pid) {
  LipLeaseEntry entry;
  entry.registration_id = std::move(registration_id);
  entry.artifact_id = std::move(artifact_id);
  entry.device_id = kDeviceId;
  entry.owner_pid = owner_pid;
  entry.ttl_ms = 60000;
  entry.expiry = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  return entry;
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
  req.set_binding_current_value_publication_token(token);
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
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  req.set_binding_current_value_publication_token(token);
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
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  req.set_binding_current_value_publication_token(token);
  req.mutable_byte_space()->CopyFrom(scope.byte_space());
  req.set_owner_pid(owner_pid);

  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(st.error_message() == "binding_current_value_publication_token has empty segments");
  REQUIRE(resp.lease_id().empty());
  REQUIRE(resp.replica_id().empty());
  REQUIRE(gs->registered_replicas.empty());
}

TEST_CASE("PublishTargetReplica reports terminal progressive coverage when enabled", "[daemon][publish][progressive]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs, /*progressive_replication=*/true);
  harness->kernel().worker_identity_store().set_registered("worker-test", "node-a");

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = 16, .fill_byte = 7},
  };
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(*helper_path_or, kDeviceId, buffers);
  INFO("cuda_ipc_helper spawn status: " << child_or.status());
  REQUIRE(child_or.ok());
  auto child = std::move(*child_or);
  REQUIRE(child.handle_bytes().size() == 1);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = child.pid();
  const auto device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(kDeviceId);
  REQUIRE(!device_key.uuid.empty());
  const auto scope = make_scope("write-progressive", "mi2:indexabc:data", device_key.uuid, owner_pid, true);
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_publishable_record_from_scope(scope, child.handle_bytes().front());
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  req.set_binding_current_value_publication_token(token);
  req.mutable_byte_space()->CopyFrom(scope.byte_space());
  req.set_owner_pid(owner_pid);

  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  INFO(st.error_message());
  REQUIRE(st.ok());
  REQUIRE(!resp.replica_id().empty());
  REQUIRE(gs->progressive_coverage_reports.size() == 1);
  const auto& report = gs->progressive_coverage_reports.front();
  CHECK(report.replica_id() == resp.replica_id());
  CHECK(report.daemon_id() == "daemon-test");
  CHECK(report.worker_id() == "worker-test");
  CHECK(report.source_domain() == "node-a");
  CHECK(report.source_export_generation() == 1);
  CHECK(report.coverage_kind() == tensorcast::global_store::v1::PROGRESSIVE_COVERAGE_KIND_BYTE_PREFIX);
  CHECK(report.state() == tensorcast::global_store::v1::PROGRESSIVE_COVERAGE_STATE_VERIFIED);
  CHECK(report.export_state() == tensorcast::global_store::v1::PROGRESSIVE_EXPORT_STATE_COMPLETE_EXPORTABLE);
  CHECK(report.verified_bytes() == 16);
  CHECK(report.total_bytes() == 16);
  CHECK(report.identity().artifact_id() == scope.selection().artifact_id());
  CHECK(report.identity().selection_hash() == scope.selection().selection_hash());
  CHECK(report.identity().logical_layout_hash() == scope.selection().logical_layout_hash());
  CHECK(report.identity().coverage_order_hash().size() == 32);

  gs->allow_replica_transport = true;
  const tensorcast::store::DeviceKey target_device{
      .type = tensorcast::DeviceType::GPU, .ordinal = kDeviceId, .uuid = device_key.uuid};
  auto before_cleanup = gs->request_replica_transport(
      scope.selection().artifact_id(),
      "consumer-node",
      "127.0.0.1",
      12345,
      target_device,
      /*wait_timeout_ms=*/1,
      std::nullopt,
      "consumer-worker",
      "request-before-owner-cleanup");
  REQUIRE(before_cleanup.ok());

  harness->kernel().lifecycle_manager().handle_pid_exit(owner_pid);

  auto context_after_cleanup =
      harness->materialization_controller().inspect_target_publication_context_for_testing(req, absl::Now());
  REQUIRE_FALSE(context_after_cleanup.ok());
  CHECK(context_after_cleanup.status().code() == absl::StatusCode::kNotFound);
  auto capability_after_cleanup = harness->kernel().lifecycle_kernel().inspect_capability(inserted_or->capability_id);
  REQUIRE_FALSE(capability_after_cleanup.ok());
  CHECK(capability_after_cleanup.status().code() == absl::StatusCode::kNotFound);
  REQUIRE(gs->marked_unavailable == std::vector<std::string>{resp.replica_id()});
  REQUIRE(
      gs->unregistered_replicas ==
      std::vector<std::pair<std::string, std::string>>{
          {scope.selection().artifact_id(), resp.replica_id()},
      });

  auto after_cleanup = gs->request_replica_transport(
      scope.selection().artifact_id(),
      "consumer-node",
      "127.0.0.1",
      12345,
      target_device,
      /*wait_timeout_ms=*/1,
      std::nullopt,
      "consumer-worker",
      "request-after-owner-cleanup");
  REQUIRE(!after_cleanup.ok());
  REQUIRE(absl::IsNotFound(after_cleanup.status()));
}

TEST_CASE(
    "Target publication remembers workflow semantic binding and recovery metadata",
    "[daemon][publish][workflow]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  const int owner_pid = getpid();
  auto scope = make_scope("write-4", "artifact-4", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-workflow");

  auto record = make_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  const std::string expected_publication_subject_key = record.publication_subject_key.value;

  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());
  CHECK(inserted_or->workflow_recovery_class == tensorcast::daemon::WorkflowRecoveryClass::kEphemeralProcessLocal);
  CHECK(inserted_or->subject_generation == 1);
  REQUIRE(inserted_or->workflow_binding_projection.has_value());
  CHECK(
      inserted_or->workflow_binding_projection->resolved_workflow_ref.owner_kind ==
      tensorcast::daemon::WorkflowOwnerKind::kPublication);
  CHECK(inserted_or->workflow_binding_projection->resolved_workflow_ref.workflow_id == scope.publication_id());
  CHECK(
      inserted_or->workflow_binding_projection->resolved_workflow_ref.currentness_key ==
      std::optional<std::string>(expected_publication_subject_key));
  CHECK(
      inserted_or->workflow_binding_projection->resolved_workflow_ref.operation_id ==
      std::optional<std::string>(scope.operation_id()));

  REQUIRE(inserted_or->replay_outcome_projection.has_value());
  CHECK(
      inserted_or->replay_outcome_projection->projection_kind ==
      tensorcast::daemon::WorkflowOutcomeProjectionKind::kExistingCapability);
  CHECK(inserted_or->replay_outcome_projection->owner_workflow_id == scope.publication_id());
  REQUIRE(inserted_or->replay_outcome_projection->attachment_ref != nullptr);
  CHECK(
      inserted_or->replay_outcome_projection->attachment_ref->authority_ref.authority_kind ==
      tensorcast::daemon::AuthorityKind::kWorkflowOwner);
  CHECK(inserted_or->replay_outcome_projection->attachment_ref->authority_ref.authority_id == scope.publication_id());
  CHECK(inserted_or->replay_outcome_projection->attachment_ref->attachment_kind == "target_publication");
  CHECK(inserted_or->replay_outcome_projection->attachment_ref->attachment_id == scope.publication_id());
  CHECK(inserted_or->replay_outcome_projection->attachment_ref->attachment_id != scope.operation_id());
  CHECK(
      inserted_or->replay_outcome_projection->existing_capability_id ==
      std::optional<std::string>(inserted_or->capability_id));

  auto capability_or = harness->kernel().lifecycle_kernel().inspect_capability(inserted_or->capability_id);
  REQUIRE(capability_or.ok());
  REQUIRE(capability_or->workflow_companion.has_value());
  CHECK(*capability_or->workflow_companion == inserted_or->workflow_binding_projection->resolved_workflow_ref);
}

TEST_CASE(
    "Target publication front-door context preserves raw token evidence and local observations",
    "[daemon][publish][front_door]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto scope = make_scope("write-5", "artifact-5", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-front-door");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  req.set_binding_current_value_publication_token(token);
  req.mutable_byte_space()->CopyFrom(scope.byte_space());
  req.set_owner_pid(owner_pid);
  req.set_operation_id(scope.operation_id());

  auto context_or =
      harness->materialization_controller().inspect_target_publication_context_for_testing(req, absl::Now());
  REQUIRE(context_or.ok());
  CHECK(context_or->record.publication_id.value == scope.publication_id());
  CHECK(context_or->scope.publication_id() == scope.publication_id());
  CHECK(context_or->normalized_byte_space.kind() == scope.byte_space().kind());
  CHECK(context_or->normalized_byte_space.id() == scope.byte_space().id());

  const auto& front_door_context = context_or->front_door_context;
  CHECK(
      front_door_context.parsed_credential.front_door_kind ==
      tensorcast::daemon::LifecycleFrontDoorKind::kTargetPublicationToken);
  CHECK(
      front_door_context.parsed_credential.carriage_kind ==
      tensorcast::daemon::CredentialCarriageKind::kSelfDescribing);
  CHECK(front_door_context.parsed_credential.binding_mode == tensorcast::daemon::LifecycleBindingMode::kAddressDerived);
  CHECK(front_door_context.parsed_credential.address.route_principal.principal_id == "daemon-test");
  CHECK(front_door_context.parsed_credential.address.family == tensorcast::daemon::LifecycleCapabilityFamily::kPublish);
  CHECK(
      front_door_context.parsed_credential.address.binding_space ==
      tensorcast::daemon::LifecycleBindingSpace::kPublication);
  CHECK(
      front_door_context.parsed_credential.address.binding_key_kind ==
      tensorcast::daemon::BindingKeyKind::kPublicationId);
  CHECK(front_door_context.parsed_credential.address.binding_key == scope.publication_id());
  CHECK(front_door_context.parsed_credential.address.epochs.subject_generation == inserted_or->subject_generation);
  CHECK(front_door_context.parsed_credential.constraint_claims.artifact_id == scope.selection().artifact_id());
  CHECK(front_door_context.parsed_credential.constraint_claims.operation_id == scope.operation_id());

  REQUIRE(front_door_context.forwardable_evidence.has_value());
  CHECK(
      front_door_context.forwardable_evidence->evidence_kind ==
      tensorcast::daemon::CredentialEvidenceKind::kRawCredential);
  CHECK(front_door_context.forwardable_evidence->raw_credential_bytes == std::optional<std::string>(token));

  REQUIRE(front_door_context.local_observations.observations.size() == 1);
  CHECK(
      front_door_context.local_observations.observations.front().observation_kind ==
      "target_publication_owner_pid_assertion");
  CHECK(front_door_context.local_observations.observations.front().observation_payload == std::to_string(owner_pid));
}

TEST_CASE(
    "PublishTargetReplica stale current remains workflow-shaped instead of lifecycle not-found",
    "[daemon][publish][workflow]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();

  auto stale_scope = make_scope("write-6-stale", "artifact-6", "gpu-0", owner_pid, true);
  stale_scope.set_operation_id("op-publish-stale");
  const std::string stale_token = mint_token(*tokens, "daemon-test", stale_scope);

  auto stale_record = make_record_from_scope(stale_scope);
  stale_record.request_operation_id = stale_scope.operation_id();
  auto inserted_stale_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(stale_record));
  REQUIRE(inserted_stale_or.ok());

  auto current_scope = stale_scope;
  current_scope.set_publication_id("write-6-current");
  current_scope.set_operation_id("op-publish-current");
  auto current_record = make_record_from_scope(current_scope);
  current_record.request_operation_id = current_scope.operation_id();
  auto inserted_current_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(current_record));
  REQUIRE(inserted_current_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  req.set_binding_current_value_publication_token(stale_token);
  req.mutable_byte_space()->CopyFrom(stale_scope.byte_space());
  req.set_owner_pid(owner_pid);
  req.set_operation_id(stale_scope.operation_id());

  auto context_or =
      harness->materialization_controller().inspect_target_publication_context_for_testing(req, absl::Now());
  REQUIRE(context_or.ok());
  CHECK(context_or->record.publication_id.value == stale_scope.publication_id());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  CHECK(st.error_message() == "binding_current_value_publication_token is stale for target");
  CHECK(resp.lease_id().empty());
  CHECK(resp.replica_id().empty());
  CHECK(gs->registered_replicas.empty());
}

TEST_CASE(
    "PublishTargetReplica keeps same-artifact different target subjects independent",
    "[daemon][publish][multi_replica]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();

  auto primary_scope = make_scope("write-5b-primary", "artifact-5b", "gpu-0", owner_pid, true);
  primary_scope.set_operation_id("op-publish-primary");
  const std::string primary_token = mint_token(*tokens, "daemon-test", primary_scope);
  auto primary_record = make_publishable_record_from_scope(primary_scope);
  primary_record.request_operation_id = primary_scope.operation_id();
  auto inserted_primary_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(primary_record));
  REQUIRE(inserted_primary_or.ok());

  auto sibling_scope = primary_scope;
  sibling_scope.set_publication_id("write-5b-sibling");
  sibling_scope.set_device_uuid("gpu-1");
  sibling_scope.set_operation_id("op-publish-sibling");
  auto sibling_record = make_record_from_scope(sibling_scope);
  sibling_record.request_operation_id = sibling_scope.operation_id();
  auto inserted_sibling_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(sibling_record));
  REQUIRE(inserted_sibling_or.ok());

  CHECK(inserted_primary_or->publication_subject_key != inserted_sibling_or->publication_subject_key);
  CHECK(inserted_primary_or->subject_generation == inserted_sibling_or->subject_generation);

  tensorcast::daemon::v2::PublishTargetReplicaRequest publish_req;
  publish_req.set_binding_current_value_publication_token(primary_token);
  publish_req.mutable_byte_space()->CopyFrom(primary_scope.byte_space());
  publish_req.set_owner_pid(owner_pid);
  publish_req.set_operation_id(primary_scope.operation_id());

  auto routed_request_or =
      harness->materialization_controller().build_target_publication_workflow_routed_request_for_testing(
          publish_req, absl::Now());
  REQUIRE(routed_request_or.ok());

  tensorcast::daemon::v2::RouteAuthorityStageRequest route_req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, route_req.mutable_routed_request());

  grpc::ServerContext route_ctx;
  tensorcast::daemon::v2::RouteAuthorityStageResponse route_resp;
  REQUIRE(harness->service().RouteAuthorityStage(&route_ctx, &route_req, &route_resp).ok());
  REQUIRE(route_resp.status() == tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
  REQUIRE(route_resp.has_owner_stage_reply());
  CHECK(
      route_resp.owner_stage_reply().reply_kind() ==
      tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_CONTINUE_WITH_AUTHORITY);
}

TEST_CASE(
    "PublishTargetReplica stale current releases lifecycle use before capability teardown",
    "[daemon][publish][workflow][release]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();

  auto stale_scope = make_scope("write-6b-stale", "artifact-6b", "gpu-0", owner_pid, true);
  stale_scope.set_operation_id("op-publish-stale-release");
  const std::string stale_token = mint_token(*tokens, "daemon-test", stale_scope);

  auto stale_record = make_record_from_scope(stale_scope);
  stale_record.request_operation_id = stale_scope.operation_id();
  auto inserted_stale_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(stale_record));
  REQUIRE(inserted_stale_or.ok());
  const std::string stale_capability_id = inserted_stale_or->capability_id;

  auto current_scope = stale_scope;
  current_scope.set_publication_id("write-6b-current");
  current_scope.set_operation_id("op-publish-current-release");
  auto current_record = make_record_from_scope(current_scope);
  current_record.request_operation_id = current_scope.operation_id();
  auto inserted_current_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(current_record));
  REQUIRE(inserted_current_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  req.set_binding_current_value_publication_token(stale_token);
  req.mutable_byte_space()->CopyFrom(stale_scope.byte_space());
  req.set_owner_pid(owner_pid);
  req.set_operation_id(stale_scope.operation_id());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  CHECK(st.error_message() == "binding_current_value_publication_token is stale for target");

  REQUIRE(harness->kernel().lifecycle_kernel().release_capability(stale_capability_id).ok());
  auto inspect_or = harness->kernel().lifecycle_kernel().inspect_capability(stale_capability_id);
  REQUIRE_FALSE(inspect_or.ok());
  CHECK(inspect_or.status().code() == absl::StatusCode::kNotFound);
}

TEST_CASE(
    "PublishTargetReplica duplicate target releases lifecycle use before capability teardown",
    "[daemon][publish][workflow][release]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();

  auto scope = make_scope("write-6c", "artifact-6c", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-duplicate-release");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());
  const std::string capability_id = inserted_or->capability_id;

  ArtifactDeviceKey key{.artifact_id = scope.selection().artifact_id(), .view_id = "", .device_id = 0};
  auto active_lease = make_active_lip_lease("other-registration", scope.selection().artifact_id(), owner_pid);
  harness->kernel().lip_manager().put_lease(active_lease.registration_id, key, active_lease);
  harness->kernel().lip_manager().attach_replica_id(active_lease.registration_id, "replica-existing");

  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  req.set_binding_current_value_publication_token(token);
  req.mutable_byte_space()->CopyFrom(scope.byte_space());
  req.set_owner_pid(owner_pid);
  req.set_operation_id(scope.operation_id());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishTargetReplicaResponse resp;
  auto st = harness->service().PublishTargetReplica(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::ALREADY_EXISTS);
  CHECK(st.error_message() == "another lease already exists for target");

  REQUIRE(harness->kernel().lifecycle_kernel().release_capability(capability_id).ok());
  auto inspect_or = harness->kernel().lifecycle_kernel().inspect_capability(capability_id);
  REQUIRE_FALSE(inspect_or.ok());
  CHECK(inspect_or.status().code() == absl::StatusCode::kNotFound);
}

TEST_CASE(
    "RouteAuthorityStage returns continuation for target_publication workflow gate",
    "[daemon][publish][route][workflow]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto scope = make_scope("write-7", "artifact-7", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-route");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_publishable_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest publish_req;
  publish_req.set_binding_current_value_publication_token(token);
  publish_req.mutable_byte_space()->CopyFrom(scope.byte_space());
  publish_req.set_owner_pid(owner_pid);
  publish_req.set_operation_id(scope.operation_id());

  auto routed_request_or =
      harness->materialization_controller().build_target_publication_workflow_routed_request_for_testing(
          publish_req, absl::Now());
  REQUIRE(routed_request_or.ok());
  CHECK(routed_request_or->authority_ref.authority_kind == tensorcast::daemon::AuthorityKind::kWorkflowOwner);
  CHECK(routed_request_or->authority_ref.authority_id == "daemon-test");
  CHECK(routed_request_or->path_family == "gate_continue_then_adopt");
  CHECK(routed_request_or->stage_ref == "workflow_gate");
  REQUIRE(routed_request_or->forwardable_evidence.has_value());
  CHECK(
      routed_request_or->forwardable_evidence->evidence_kind ==
      tensorcast::daemon::CredentialEvidenceKind::kRawCredential);
  CHECK(routed_request_or->forwardable_evidence->raw_credential_bytes == std::optional<std::string>(token));

  tensorcast::daemon::v2::RouteAuthorityStageRequest route_req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, route_req.mutable_routed_request());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::RouteAuthorityStageResponse route_resp;
  REQUIRE(harness->service().RouteAuthorityStage(&ctx, &route_req, &route_resp).ok());
  REQUIRE(route_resp.status() == tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
  REQUIRE(route_resp.has_owner_stage_reply());
  const auto& reply = route_resp.owner_stage_reply();
  CHECK(reply.answered_by().authority_kind() == tensorcast::daemon::v2::ROUTED_AUTHORITY_KIND_WORKFLOW_OWNER);
  CHECK(reply.answered_by().authority_id() == "daemon-test");
  CHECK(reply.path_family() == "gate_continue_then_adopt");
  CHECK(reply.stage_ref() == "workflow_gate");
  CHECK(reply.reply_kind() == tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_CONTINUE_WITH_AUTHORITY);
  REQUIRE(reply.has_continuation());
  CHECK(
      reply.continuation().next_authority_ref().authority_kind() ==
      tensorcast::daemon::v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  CHECK(reply.continuation().next_authority_ref().authority_id() == "daemon-test");
  CHECK(reply.continuation().edge_ref() == "workflow_to_issuer");
  REQUIRE(reply.continuation().forwarded_claims_size() == 1);
  const auto& claim = reply.continuation().forwarded_claims(0);
  CHECK(claim.claim_kind() == "publish_workflow_gate");
  CHECK(claim.provenance() == tensorcast::daemon::v2::ROUTED_FORWARDED_CLAIM_PROVENANCE_AUTHORITY_AUTHENTICATED);
  CHECK(claim.claim_payload() == "admit");
  CHECK(claim.minted_by_authority_ref().authority_id() == "daemon-test");
  CHECK(claim.audience_authority_ref().authority_id() == "daemon-test");
  CHECK(claim.bound_root_request_id() == route_req.routed_request().request_metadata().root_request_id());
  CHECK(claim.bound_path_family() == "gate_continue_then_adopt");
  CHECK(claim.bound_edge() == "workflow_to_issuer");
  REQUIRE(reply.continuation().has_forwarded_claims_envelope());
  CHECK(reply.continuation().forwarded_claims_envelope().bound_edge() == "workflow_to_issuer");
}

TEST_CASE(
    "RouteAuthorityStage completes target_publication issuer validation after workflow continuation",
    "[daemon][publish][route][workflow]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto scope = make_scope("write-8", "artifact-8", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-route-issuer");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_publishable_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest publish_req;
  publish_req.set_binding_current_value_publication_token(token);
  publish_req.mutable_byte_space()->CopyFrom(scope.byte_space());
  publish_req.set_owner_pid(owner_pid);
  publish_req.set_operation_id(scope.operation_id());

  auto routed_request_or =
      harness->materialization_controller().build_target_publication_workflow_routed_request_for_testing(
          publish_req, absl::Now());
  REQUIRE(routed_request_or.ok());

  tensorcast::daemon::v2::RouteAuthorityStageRequest workflow_req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, workflow_req.mutable_routed_request());
  grpc::ServerContext workflow_ctx;
  tensorcast::daemon::v2::RouteAuthorityStageResponse workflow_resp;
  REQUIRE(harness->service().RouteAuthorityStage(&workflow_ctx, &workflow_req, &workflow_resp).ok());
  REQUIRE(workflow_resp.status() == tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
  REQUIRE(workflow_resp.has_owner_stage_reply());
  REQUIRE(workflow_resp.owner_stage_reply().has_continuation());

  auto workflow_reply_shell_or =
      tensorcast::daemon::routed_authority_wire::owner_stage_reply_shell_from_proto(workflow_resp.owner_stage_reply());
  REQUIRE(workflow_reply_shell_or.ok());
  auto issuer_routed_request_or =
      harness->materialization_controller().build_target_publication_workflow_continuation_request_for_testing(
          *routed_request_or, workflow_reply_shell_or->reply);
  REQUIRE(issuer_routed_request_or.ok());

  tensorcast::daemon::v2::RouteAuthorityStageRequest issuer_req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *issuer_routed_request_or, issuer_req.mutable_routed_request());

  grpc::ServerContext issuer_ctx;
  tensorcast::daemon::v2::RouteAuthorityStageResponse issuer_resp;
  REQUIRE(harness->service().RouteAuthorityStage(&issuer_ctx, &issuer_req, &issuer_resp).ok());
  INFO(issuer_resp.message());
  REQUIRE(issuer_resp.status() == tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
  REQUIRE(issuer_resp.has_owner_stage_reply());
  const auto& reply = issuer_resp.owner_stage_reply();
  CHECK(reply.answered_by().authority_kind() == tensorcast::daemon::v2::ROUTED_AUTHORITY_KIND_ISSUER_DAEMON);
  CHECK(reply.answered_by().authority_id() == "daemon-test");
  CHECK(reply.path_family() == "gate_continue_then_adopt");
  CHECK(reply.stage_ref() == "issuer_validate");
  CHECK(reply.reply_kind() == tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_TERMINAL);
  REQUIRE(reply.has_terminal_projection());
  CHECK(
      reply.terminal_projection().projection_kind() ==
      tensorcast::daemon::v2::ROUTED_TERMINAL_PROJECTION_KIND_SEMANTIC_SUCCESS);
  CHECK(reply.terminal_projection().status_code() == "ok");
  CHECK(reply.terminal_projection().family_payload() == "publish_workflow_gate_admitted");
}

TEST_CASE(
    "RouteAuthorityStage returns terminal reject for stale target_publication workflow gate",
    "[daemon][publish][route][workflow]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();

  auto stale_scope = make_scope("write-9-stale", "artifact-9", "gpu-0", owner_pid, true);
  stale_scope.set_operation_id("op-publish-route-stale");
  const std::string stale_token = mint_token(*tokens, "daemon-test", stale_scope);
  auto stale_record = make_publishable_record_from_scope(stale_scope);
  stale_record.request_operation_id = stale_scope.operation_id();
  auto inserted_stale_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(stale_record));
  REQUIRE(inserted_stale_or.ok());

  auto current_scope = stale_scope;
  current_scope.set_publication_id("write-9-current");
  current_scope.set_operation_id("op-publish-route-current");
  auto current_record = make_record_from_scope(current_scope);
  current_record.request_operation_id = current_scope.operation_id();
  auto inserted_current_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(current_record));
  REQUIRE(inserted_current_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest publish_req;
  publish_req.set_binding_current_value_publication_token(stale_token);
  publish_req.mutable_byte_space()->CopyFrom(stale_scope.byte_space());
  publish_req.set_owner_pid(owner_pid);
  publish_req.set_operation_id(stale_scope.operation_id());

  auto routed_request_or =
      harness->materialization_controller().build_target_publication_workflow_routed_request_for_testing(
          publish_req, absl::Now());
  REQUIRE(routed_request_or.ok());

  tensorcast::daemon::v2::RouteAuthorityStageRequest route_req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, route_req.mutable_routed_request());
  grpc::ServerContext ctx;
  tensorcast::daemon::v2::RouteAuthorityStageResponse route_resp;
  REQUIRE(harness->service().RouteAuthorityStage(&ctx, &route_req, &route_resp).ok());
  REQUIRE(route_resp.status() == tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
  REQUIRE(route_resp.has_owner_stage_reply());
  CHECK(route_resp.owner_stage_reply().reply_kind() == tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_TERMINAL);
  REQUIRE(route_resp.owner_stage_reply().has_terminal_projection());
  CHECK(route_resp.owner_stage_reply().terminal_projection().status_code() == "failed_precondition");
  CHECK(route_resp.owner_stage_reply().terminal_projection().family_payload() == "publish_stale_current");
}

TEST_CASE(
    "StartPublishTargetReplica projects publish continuation into OperationRef and WaitOperation",
    "[daemon][publish][operation]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto scope = make_scope("write-10", "artifact-10", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-operation");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_publishable_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  req.set_binding_current_value_publication_token(token);
  req.mutable_byte_space()->CopyFrom(scope.byte_space());
  req.set_owner_pid(owner_pid);
  req.set_operation_id(scope.operation_id());

  grpc::ServerContext start_ctx;
  tensorcast::daemon::v2::StartPublishTargetReplicaResponse start_resp;
  REQUIRE(harness->service().StartPublishTargetReplica(&start_ctx, &req, &start_resp).ok());
  REQUIRE(start_resp.has_operation());
  absl::SleepFor(absl::Milliseconds(20));
  CHECK(start_resp.operation().operation_id() == absl::StrCat("publish-target:", scope.publication_id()));
  CHECK(start_resp.operation().kind() == "publish_target_replica");
  CHECK(start_resp.operation().target_artifact_id() == scope.selection().artifact_id());
  CHECK(start_resp.operation().authority_scope_kind() == "workflow_owner");
  CHECK(start_resp.operation().authority_scope_id() == scope.publication_id());
  CHECK(start_resp.operation().attachment_kind() == "target_publication");
  CHECK(start_resp.operation().recovery_class() == "ephemeral_process_local");

  auto& stored_operation = gs->operations[start_resp.operation().operation_id()];
  stored_operation.mutable_ref()->CopyFrom(start_resp.operation());
  stored_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
  stored_operation.mutable_status()->set_message("publish complete");
  tensorcast::daemon::v2::PublishTargetReplicaResponse stored_result;
  stored_result.set_lease_id(scope.publication_id());
  stored_result.set_replica_id("memory_replica");
  stored_operation.mutable_status()->mutable_result()->PackFrom(stored_result);

  tensorcast::daemon::v2::WaitOperationRequest wait_req;
  wait_req.set_operation_id(start_resp.operation().operation_id());
  wait_req.mutable_ref()->CopyFrom(start_resp.operation());
  wait_req.set_timeout_ms(5000);

  grpc::ServerContext wait_ctx;
  tensorcast::daemon::v2::WaitOperationResponse wait_resp;
  REQUIRE(harness->service().WaitOperation(&wait_ctx, &wait_req, &wait_resp).ok());
  CHECK(wait_resp.operation().status().state() == tensorcast::operation::v1::OPERATION_STATE_SUCCESS);

  tensorcast::daemon::v2::PublishTargetReplicaResponse result;
  REQUIRE(wait_resp.operation().status().has_result());
  REQUIRE(wait_resp.operation().status().result().UnpackTo(&result));
  CHECK(result.lease_id() == scope.publication_id());
  CHECK(result.replica_id() == "memory_replica");
}

TEST_CASE(
    "RouteAuthorityStage returns attach_existing while publish operation is still running",
    "[daemon][publish][route][operation]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  gs->register_memory_replica_delay = absl::Milliseconds(200);
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto scope = make_scope("write-11", "artifact-11", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-attach");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest publish_req;
  publish_req.set_binding_current_value_publication_token(token);
  publish_req.mutable_byte_space()->CopyFrom(scope.byte_space());
  publish_req.set_owner_pid(owner_pid);
  publish_req.set_operation_id(scope.operation_id());

  grpc::ServerContext start_ctx;
  tensorcast::daemon::v2::StartPublishTargetReplicaResponse start_resp;
  REQUIRE(harness->service().StartPublishTargetReplica(&start_ctx, &publish_req, &start_resp).ok());
  REQUIRE(start_resp.has_operation());
  absl::SleepFor(absl::Milliseconds(20));

  auto& stored_operation = gs->operations[start_resp.operation().operation_id()];
  stored_operation.mutable_ref()->CopyFrom(start_resp.operation());
  stored_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
  stored_operation.mutable_status()->set_message("publish running");

  auto routed_request_or =
      harness->materialization_controller().build_target_publication_workflow_routed_request_for_testing(
          publish_req, absl::Now());
  REQUIRE(routed_request_or.ok());

  tensorcast::daemon::v2::RouteAuthorityStageRequest route_req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, route_req.mutable_routed_request());

  grpc::ServerContext route_ctx;
  tensorcast::daemon::v2::RouteAuthorityStageResponse route_resp;
  REQUIRE(harness->service().RouteAuthorityStage(&route_ctx, &route_req, &route_resp).ok());
  REQUIRE(route_resp.status() == tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
  REQUIRE(route_resp.has_owner_stage_reply());
  CHECK(
      route_resp.owner_stage_reply().reply_kind() ==
      tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_ATTACH_EXISTING);
  REQUIRE(route_resp.owner_stage_reply().has_attachment_ref());
  CHECK(route_resp.owner_stage_reply().attachment_ref().attachment_kind() == "target_publication");
  CHECK(route_resp.owner_stage_reply().attachment_ref().attachment_id() == scope.publication_id());

  tensorcast::daemon::v2::PublishTargetReplicaResponse stored_result;
  stored_result.set_lease_id(scope.publication_id());
  stored_result.set_replica_id("memory_replica");
  stored_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
  stored_operation.mutable_status()->mutable_result()->PackFrom(stored_result);

  tensorcast::daemon::v2::WaitOperationRequest wait_req;
  wait_req.set_operation_id(start_resp.operation().operation_id());
  wait_req.mutable_ref()->CopyFrom(start_resp.operation());
  wait_req.set_timeout_ms(5000);
  grpc::ServerContext wait_ctx;
  tensorcast::daemon::v2::WaitOperationResponse wait_resp;
  REQUIRE(harness->service().WaitOperation(&wait_ctx, &wait_req, &wait_resp).ok());
  CHECK(wait_resp.operation().status().state() == tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
}

TEST_CASE(
    "RouteAuthorityStage returns terminal replay for same current subject and generation",
    "[daemon][publish][route][operation]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto scope = make_scope("write-11b", "artifact-11b", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-terminal");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest publish_req;
  publish_req.set_binding_current_value_publication_token(token);
  publish_req.mutable_byte_space()->CopyFrom(scope.byte_space());
  publish_req.set_owner_pid(owner_pid);
  publish_req.set_operation_id(scope.operation_id());

  grpc::ServerContext start_ctx;
  tensorcast::daemon::v2::StartPublishTargetReplicaResponse start_resp;
  REQUIRE(harness->service().StartPublishTargetReplica(&start_ctx, &publish_req, &start_resp).ok());
  REQUIRE(start_resp.has_operation());
  absl::SleepFor(absl::Milliseconds(20));

  auto& stored_operation = gs->operations[start_resp.operation().operation_id()];
  stored_operation.mutable_ref()->CopyFrom(start_resp.operation());
  stored_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
  stored_operation.mutable_status()->set_message("publish complete");
  tensorcast::daemon::v2::PublishTargetReplicaResponse stored_result;
  stored_result.set_lease_id(scope.publication_id());
  stored_result.set_replica_id("memory_replica");
  stored_operation.mutable_status()->mutable_result()->PackFrom(stored_result);

  auto routed_request_or =
      harness->materialization_controller().build_target_publication_workflow_routed_request_for_testing(
          publish_req, absl::Now());
  REQUIRE(routed_request_or.ok());

  tensorcast::daemon::v2::RouteAuthorityStageRequest route_req;
  tensorcast::daemon::routed_authority_wire::populate_proto_routed_authority_request(
      *routed_request_or, route_req.mutable_routed_request());

  grpc::ServerContext route_ctx;
  tensorcast::daemon::v2::RouteAuthorityStageResponse route_resp;
  REQUIRE(harness->service().RouteAuthorityStage(&route_ctx, &route_req, &route_resp).ok());
  REQUIRE(route_resp.status() == tensorcast::daemon::v2::BATCH_ITEM_STATUS_OK);
  REQUIRE(route_resp.has_owner_stage_reply());
  CHECK(route_resp.owner_stage_reply().reply_kind() == tensorcast::daemon::v2::ROUTED_OWNER_STAGE_REPLY_KIND_TERMINAL);
  REQUIRE(route_resp.owner_stage_reply().has_terminal_projection());
  CHECK(route_resp.owner_stage_reply().terminal_projection().status_code() == "ok");
  CHECK(route_resp.owner_stage_reply().terminal_projection().family_payload() == "publish_replay_terminal");
}

TEST_CASE("GetOperation fails closed when publish owner state is lost", "[daemon][publish][operation][owner_loss]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  gs->register_memory_replica_delay = absl::Milliseconds(200);
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto scope = make_scope("write-12", "artifact-12", "gpu-0", owner_pid, true);
  scope.set_operation_id("op-publish-owner-loss");
  const std::string token = mint_token(*tokens, "daemon-test", scope);

  auto record = make_record_from_scope(scope);
  record.request_operation_id = scope.operation_id();
  auto inserted_or = harness->materialization_controller().insert_target_publication_for_testing(std::move(record));
  REQUIRE(inserted_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  req.set_binding_current_value_publication_token(token);
  req.mutable_byte_space()->CopyFrom(scope.byte_space());
  req.set_owner_pid(owner_pid);
  req.set_operation_id(scope.operation_id());

  grpc::ServerContext start_ctx;
  tensorcast::daemon::v2::StartPublishTargetReplicaResponse start_resp;
  REQUIRE(harness->service().StartPublishTargetReplica(&start_ctx, &req, &start_resp).ok());
  REQUIRE(start_resp.has_operation());

  auto& stored_operation = gs->operations[start_resp.operation().operation_id()];
  stored_operation.mutable_ref()->CopyFrom(start_resp.operation());
  stored_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);

  harness->kernel().lifecycle_manager().release_lease(inserted_or->lease_id);
  absl::SleepFor(absl::Milliseconds(20));

  tensorcast::operation::v1::GetOperationRequest get_req;
  get_req.set_operation_id(start_resp.operation().operation_id());
  get_req.mutable_ref()->CopyFrom(start_resp.operation());
  tensorcast::operation::v1::GetOperationResponse get_resp;
  grpc::ServerContext get_ctx;
  const auto status = harness->service().GetOperation(&get_ctx, &get_req, &get_resp);
  CHECK(status.error_code() == grpc::StatusCode::UNAVAILABLE);
}

TEST_CASE(
    "Publish continuation fails closed when same stable subject is rebound to a new owner generation",
    "[daemon][publish][operation][owner_loss]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();

  auto original_scope = make_scope("write-12b-original", "artifact-12b", "gpu-0", owner_pid, true);
  original_scope.set_operation_id("op-publish-original");
  const std::string original_token = mint_token(*tokens, "daemon-test", original_scope);
  auto original_record = make_record_from_scope(original_scope);
  original_record.request_operation_id = original_scope.operation_id();
  auto inserted_original_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(original_record));
  REQUIRE(inserted_original_or.ok());
  CHECK(inserted_original_or->subject_generation == 1);

  tensorcast::daemon::v2::PublishTargetReplicaRequest original_req;
  original_req.set_binding_current_value_publication_token(original_token);
  original_req.mutable_byte_space()->CopyFrom(original_scope.byte_space());
  original_req.set_owner_pid(owner_pid);
  original_req.set_operation_id(original_scope.operation_id());

  grpc::ServerContext start_ctx;
  tensorcast::daemon::v2::StartPublishTargetReplicaResponse start_resp;
  REQUIRE(harness->service().StartPublishTargetReplica(&start_ctx, &original_req, &start_resp).ok());
  REQUIRE(start_resp.has_operation());

  auto& stored_operation = gs->operations[start_resp.operation().operation_id()];
  stored_operation.mutable_ref()->CopyFrom(start_resp.operation());
  stored_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);

  auto replacement_scope = original_scope;
  replacement_scope.set_publication_id("write-12b-replacement");
  replacement_scope.set_owner_pid(owner_pid + 1);
  replacement_scope.set_operation_id("op-publish-replacement");
  auto replacement_record = make_record_from_scope(replacement_scope);
  replacement_record.request_operation_id = replacement_scope.operation_id();
  auto inserted_replacement_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(replacement_record));
  REQUIRE(inserted_replacement_or.ok());
  CHECK(inserted_replacement_or->publication_subject_key == inserted_original_or->publication_subject_key);
  CHECK(inserted_replacement_or->subject_generation == inserted_original_or->subject_generation + 1);

  grpc::ServerContext publish_ctx;
  tensorcast::daemon::v2::PublishTargetReplicaResponse publish_resp;
  const auto publish_status = harness->service().PublishTargetReplica(&publish_ctx, &original_req, &publish_resp);
  CHECK(publish_status.error_code() == grpc::StatusCode::UNAVAILABLE);
  CHECK(publish_status.error_message() == "publish workflow owner lost");

  tensorcast::operation::v1::GetOperationRequest get_req;
  get_req.set_operation_id(start_resp.operation().operation_id());
  get_req.mutable_ref()->CopyFrom(start_resp.operation());
  tensorcast::operation::v1::GetOperationResponse get_resp;
  grpc::ServerContext get_ctx;
  const auto get_status = harness->service().GetOperation(&get_ctx, &get_req, &get_resp);
  CHECK(get_status.error_code() == grpc::StatusCode::UNAVAILABLE);
}

TEST_CASE(
    "GetOperation fails closed on stale or mismatched publish continuation metadata",
    "[daemon][publish][operation][fencing]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  auto gs = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  auto harness = make_harness(engine, gs);

  auto* tokens = harness->kernel().capability_tokens();
  REQUIRE(tokens != nullptr);
  const int owner_pid = getpid();
  auto stale_scope = make_scope("write-13-stale", "artifact-13", "gpu-0", owner_pid, true);
  stale_scope.set_operation_id("op-publish-stale-attach");
  const std::string token = mint_token(*tokens, "daemon-test", stale_scope);

  auto stale_record = make_record_from_scope(stale_scope);
  stale_record.request_operation_id = stale_scope.operation_id();
  auto inserted_stale_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(stale_record));
  REQUIRE(inserted_stale_or.ok());

  tensorcast::daemon::v2::PublishTargetReplicaRequest req;
  req.set_binding_current_value_publication_token(token);
  req.mutable_byte_space()->CopyFrom(stale_scope.byte_space());
  req.set_owner_pid(owner_pid);
  req.set_operation_id(stale_scope.operation_id());

  grpc::ServerContext start_ctx;
  tensorcast::daemon::v2::StartPublishTargetReplicaResponse start_resp;
  REQUIRE(harness->service().StartPublishTargetReplica(&start_ctx, &req, &start_resp).ok());
  REQUIRE(start_resp.has_operation());

  auto& stored_operation = gs->operations[start_resp.operation().operation_id()];
  stored_operation.mutable_ref()->CopyFrom(start_resp.operation());
  stored_operation.mutable_status()->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);

  auto current_scope = stale_scope;
  current_scope.set_publication_id("write-13-current");
  current_scope.set_operation_id("op-publish-current-attach");
  auto current_record = make_record_from_scope(current_scope);
  current_record.request_operation_id = current_scope.operation_id();
  auto inserted_current_or =
      harness->materialization_controller().insert_target_publication_for_testing(std::move(current_record));
  REQUIRE(inserted_current_or.ok());

  tensorcast::operation::v1::GetOperationRequest stale_get_req;
  stale_get_req.set_operation_id(start_resp.operation().operation_id());
  stale_get_req.mutable_ref()->CopyFrom(start_resp.operation());
  tensorcast::operation::v1::GetOperationResponse stale_get_resp;
  grpc::ServerContext stale_get_ctx;
  const auto stale_status = harness->service().GetOperation(&stale_get_ctx, &stale_get_req, &stale_get_resp);
  CHECK(stale_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);

  auto& stored_operation_ref = gs->operations[start_resp.operation().operation_id()];
  stored_operation_ref.mutable_ref()->set_attachment_kind("wrong_attachment");

  tensorcast::operation::v1::GetOperationRequest wrong_get_req;
  wrong_get_req.set_operation_id(start_resp.operation().operation_id());
  wrong_get_req.mutable_ref()->CopyFrom(start_resp.operation());
  tensorcast::operation::v1::GetOperationResponse wrong_get_resp;
  grpc::ServerContext wrong_get_ctx;
  const auto wrong_status = harness->service().GetOperation(&wrong_get_ctx, &wrong_get_req, &wrong_get_resp);
  CHECK(wrong_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
}
