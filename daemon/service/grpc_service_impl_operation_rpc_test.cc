// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
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

std::string ensure_fake_gpu_uuid() {
  auto device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  if (device_key.uuid.empty()) {
    tensorcast::store::DeviceRegistry::instance().register_gpu(0, "gpu-0");
    device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  }
  return device_key.uuid.empty() ? std::string("gpu-0") : device_key.uuid;
}

std::string make_target_index_json() {
  return R"({"alpha":[0,4,[1],[1],"torch.float32",0]})";
}

bool write_file(const std::filesystem::path& path, std::string_view payload) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  return out.good();
}

tensorcast::daemon::ArtifactSourceRegistry::FingerprintMap to_registry_fingerprints(
    const tensorcast::daemon::materialization_disk_resolve::SourceFingerprintMap& fingerprints) {
  tensorcast::daemon::ArtifactSourceRegistry::FingerprintMap out;
  for (const auto& [relative_path, fingerprint] : fingerprints) {
    out.insert_or_assign(
        relative_path,
        tensorcast::daemon::ArtifactSourceRegistry::SourceFileFingerprint{
            .inode = fingerprint.inode,
            .size = fingerprint.size,
            .mtime_ns = fingerprint.mtime_ns,
        });
  }
  return out;
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

std::unique_ptr<tensorcast::daemon::DaemonServiceHarness> make_harness(
    std::shared_ptr<OperationClient> client,
    bool serving_prefetch_enabled = false) {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  engine->set_global_store_client_for_testing(client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = test_tmpdir();
  daemon_opts.serving_prefetch.enabled = serving_prefetch_enabled;
  std::filesystem::create_directories(daemon_opts.storage_path);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  return harness;
}

void seed_mounted_source_artifact(tensorcast::daemon::DaemonServiceHarness& harness, std::string_view artifact_id) {
  const auto artifact_dir = test_tmpdir() / "prefetch_serving_source";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor.data", "ABCD"));
  REQUIRE(write_file(artifact_dir / "tensor_index.json", make_target_index_json()));
  auto metadata_or = tensorcast::daemon::materialization_disk_resolve::build_mounted_source_metadata(artifact_dir);
  REQUIRE(metadata_or.ok());
  harness.kernel().source_registry().upsert_binding(
      std::string(artifact_id),
      tensorcast::daemon::ArtifactSourceRegistry::Entry{
          .source_kind = tensorcast::daemon::ArtifactSourceRegistry::SourceKind::kMountedSourceArtifact,
          .canonical_source_path = artifact_dir.string(),
          .canonical_index_json = metadata_or->index_info.canonical_index_json,
          .source_index_json = metadata_or->index_info.source_index_json,
          .source_disk_path = artifact_dir.string(),
          .tensor_aware_metadata = true,
          .validate_before_read = true,
          .file_fingerprints = to_registry_fingerprints(metadata_or->file_fingerprints),
      });
}

tensorcast::daemon::v2::PrefetchServingBindingRequest make_serving_prefetch_request(bool include_layout) {
  tensorcast::daemon::v2::PrefetchServingBindingRequest req;
  auto* target = req.mutable_serving_binding_target();
  target->set_runtime("vllm");
  target->set_device("cuda:0");
  target->set_device_uuid(ensure_fake_gpu_uuid());
  target->set_model_config_digest("model-config");
  target->set_serving_build_digest("serving-build");
  target->mutable_topology()->set_schema_topology_digest("topology");
  auto* member = target->mutable_member();
  member->set_member_id("member-0");
  member->set_member_index(0);
  member->set_member_count(1);
  member->set_group_id("group-1");
  if (include_layout) {
    tensorcast::daemon::v2::TargetLayout target_layout;
    target_layout.set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_TENSOR_TABLE);
    target_layout.set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
    target_layout.set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);
    target_layout.set_logical_layout_hash("logical-layout-hash");
    auto* storage = target_layout.add_storages();
    storage->set_storage_id("storage-0");
    storage->set_device_id(0);
    storage->set_storage_length(4);
    auto* offset = target_layout.add_offsets();
    offset->set_name("alpha");
    offset->set_storage_id("storage-0");
    offset->set_storage_offset(0);
    offset->set_logical_length(4);

    auto* layout = target->mutable_resolved_layout();
    layout->set_binding_layout_id("layout-1");
    layout->mutable_topology()->CopyFrom(target->topology());
    layout->mutable_member()->CopyFrom(*member);
    layout->set_target_layout(target_layout.SerializeAsString());
    layout->set_target_index_bytes(make_target_index_json());
    layout->set_target_layout_hash("layout-hash");
    layout->set_tensor_schema_hash("tensor-schema");
    layout->set_spec_digest("spec-digest");
    layout->set_source_schema_hash("source-schema");
    layout->mutable_source_reuse()->set_mode(
        tensorcast::operation::v1::SERVING_BINDING_SOURCE_REUSE_MODE_CHECKPOINT_TO_SERVING);
  }
  return req;
}

tensorcast::daemon::v2::PrefetchServingBindingRequest make_serving_prefetch_set_request() {
  auto member_0_req = make_serving_prefetch_request(true);
  auto member_1_req = make_serving_prefetch_request(true);
  auto* member_1 = member_1_req.mutable_serving_binding_target()->mutable_member();
  member_1->set_member_id("member-1");
  member_1->set_member_index(1);
  member_1->set_member_count(2);
  member_1_req.mutable_serving_binding_target()->set_device("cuda:1");
  member_1_req.mutable_serving_binding_target()->set_device_uuid("GPU-1");
  auto* layout_member_1 = member_1_req.mutable_serving_binding_target()->mutable_resolved_layout();
  layout_member_1->mutable_member()->CopyFrom(*member_1);
  layout_member_1->set_target_layout_hash("layout-hash-member-1");

  auto* member_0 = member_0_req.mutable_serving_binding_target()->mutable_member();
  member_0->set_member_count(2);
  member_0_req.mutable_serving_binding_target()->mutable_resolved_layout()->mutable_member()->CopyFrom(*member_0);
  member_0_req.mutable_serving_binding_target()->mutable_resolved_layout()->set_target_layout_hash(
      "layout-hash-member-0");

  tensorcast::daemon::v2::PrefetchServingBindingRequest req;
  auto* target_set = req.mutable_serving_binding_set_target();
  target_set->set_runtime("vllm");
  target_set->mutable_topology()->CopyFrom(member_0_req.serving_binding_target().topology());
  target_set->set_group_id("group-1");
  target_set->add_members()->CopyFrom(member_0_req.serving_binding_target());
  target_set->add_members()->CopyFrom(member_1_req.serving_binding_target());
  return req;
}

std::shared_ptr<tensorcast::daemon::BindingRegistry::Record> make_retained_serving_record() {
  auto record = std::make_shared<tensorcast::daemon::BindingRegistry::Record>();
  record->binding_id = "binding-acquire";
  record->binding_layout_id = "layout-1";
  record->device_uuid = "GPU-0";
  record->state = tensorcast::daemon::v2::BINDING_STATE_READY_LOCAL;
  record->control_lifetime = tensorcast::daemon::BindingRegistry::ControlLifetime::kDaemonRetained;
  record->retained_ref = true;
  record->current_binding_value_id = "value-1";
  record->seal_generation = 7;
  record->target_index_json = R"({"tensors":[]})";
  record->target_layout_hash = "layout-hash";
  record->tensor_schema_hash = "tensor-schema";
  record->daemon_id = "daemon-1";
  record->daemon_session_id = "session-1";
  record->serving_build_digest = "serving-build";
  record->reservation_capability_id = "capability-1";
  record->reservation_expires_at = absl::Now() + absl::Hours(1);
  record->local_serving_ref = "binding-local:binding-acquire:value-1";
  record->serving_member.set_member_id("member-0");
  record->serving_member.set_member_index(0);
  record->serving_member.set_member_count(1);
  record->serving_member.set_group_id("group-1");
  return record;
}

tensorcast::daemon::v2::AcquireBindingValueRequest make_acquire_request() {
  tensorcast::daemon::v2::AcquireBindingValueRequest req;
  auto* ref = req.mutable_binding_value_ref();
  ref->set_binding_id("binding-acquire");
  ref->set_binding_layout_id("layout-1");
  ref->set_binding_value_id("value-1");
  ref->set_seal_generation(7);
  req.set_expected_daemon_id("daemon-1");
  req.set_expected_daemon_session_id("session-1");
  req.set_expected_device_uuid("GPU-0");
  auto* member = req.mutable_expected_member();
  member->set_member_id("member-0");
  member->set_member_index(0);
  member->set_member_count(1);
  member->set_group_id("group-1");
  req.set_expected_target_layout_hash("layout-hash");
  req.set_expected_tensor_schema_hash("tensor-schema");
  req.set_expected_serving_build_digest("serving-build");
  req.set_local_serving_ref("binding-local:binding-acquire:value-1");
  req.set_caller_pid(4321);
  auto* capability = req.mutable_reservation_capability();
  capability->set_capability_id("capability-1");
  capability->mutable_binding_value_ref()->CopyFrom(*ref);
  capability->set_daemon_id("daemon-1");
  capability->set_daemon_session_id("session-1");
  capability->set_device_uuid("GPU-0");
  capability->mutable_member()->CopyFrom(*member);
  capability->set_reservation_bytes(4096);
  capability->set_scope_digest("scope");
  capability->set_expires_at_ms(static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() + absl::Hours(1))));
  return req;
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

TEST_CASE("PrefetchServingBinding is gated by daemon feature flag", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, false);

  auto req = make_serving_prefetch_request(true);
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_CASE("AcquireBindingValue is gated by daemon feature flag", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, false);

  tensorcast::daemon::v2::AcquireBindingValueRequest req;
  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_CASE(
    "AcquireBindingValue mints fresh lease and releases attachment on lease release",
    "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  auto record = make_retained_serving_record();
  REQUIRE(harness->kernel().binding_registry().insert(record).ok());

  auto req = make_acquire_request();
  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE_FALSE(resp.lease_token().empty());
  REQUIRE(resp.mem_handle().lease_token() == resp.lease_token());
  REQUIRE(resp.target_index_bytes() == R"({"tensors":[]})");
  REQUIRE(resp.reservation_bytes() == 4096);
  REQUIRE(resp.current_value().binding_value_id() == "value-1");
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 1);
  }

  REQUIRE(harness->kernel().handle_leases() != nullptr);
  REQUIRE(harness->kernel().handle_leases()->release(resp.lease_token()).ok());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
  }
}

TEST_CASE("AcquireBindingValue rejects stale daemon session before minting lease", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  auto record = make_retained_serving_record();
  REQUIRE(harness->kernel().binding_registry().insert(record).ok());

  auto req = make_acquire_request();
  req.set_expected_daemon_session_id("session-after-restart");
  req.mutable_reservation_capability()->set_daemon_session_id("session-after-restart");
  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);

  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(std::string(st.error_message()).find("daemon_session_id mismatch") != std::string::npos);
  REQUIRE(resp.lease_token().empty());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
  }
}

TEST_CASE("PrefetchServingBinding rejects unresolved layout before executor", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);

  auto req = make_serving_prefetch_request(false);
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(std::string(st.error_message()).find("resolved_layout") != std::string::npos);
}

TEST_CASE("PrefetchServingBinding rejects non-TargetLayout bytes before executor", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);

  auto req = make_serving_prefetch_request(true);
  req.mutable_serving_binding_target()->mutable_resolved_layout()->set_target_layout("not-a-target-layout");
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(std::string(st.error_message()).find("TargetLayout") != std::string::npos);
}

TEST_CASE(
    "PrefetchServingBinding materializes local-ready binding and acquire reads it",
    "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  const std::string artifact_id = "msa1:test-session~policy~partitioned~serving";
  seed_mounted_source_artifact(*harness, artifact_id);

  auto req = make_serving_prefetch_request(true);
  req.mutable_source_selection()->set_artifact_id(artifact_id);
  req.set_operation_id("prefetch-serving-op");
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);

  INFO("PrefetchServingBinding status: " << st.error_code() << " " << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.operation_ref().operation_id() == "prefetch-serving-op");
  REQUIRE(resp.status().state() == tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
  tensorcast::operation::v1::PrefetchServingBindingResult result;
  REQUIRE(resp.status().result().UnpackTo(&result));
  REQUIRE(result.readiness() == tensorcast::operation::v1::SERVING_BINDING_READINESS_LOCAL_READY);
  REQUIRE(result.verification_state() == "local_only");
  REQUIRE(result.reservation_bytes() == 4);
  REQUIRE_FALSE(result.local_serving_ref().empty());

  tensorcast::daemon::v2::AcquireBindingValueRequest acquire_req;
  acquire_req.mutable_binding_value_ref()->CopyFrom(result.binding_value_ref());
  acquire_req.mutable_reservation_capability()->CopyFrom(result.reservation_capability());
  acquire_req.set_expected_daemon_id(result.daemon_id());
  acquire_req.set_expected_daemon_session_id(result.daemon_session_id());
  acquire_req.set_expected_device_uuid(result.device_uuid());
  acquire_req.mutable_expected_member()->CopyFrom(result.member());
  acquire_req.set_expected_target_layout_hash(req.serving_binding_target().resolved_layout().target_layout_hash());
  acquire_req.set_expected_tensor_schema_hash(req.serving_binding_target().resolved_layout().tensor_schema_hash());
  acquire_req.set_expected_serving_build_digest(req.serving_binding_target().serving_build_digest());
  acquire_req.set_local_serving_ref(result.local_serving_ref());
  acquire_req.set_caller_pid(4321);
  tensorcast::daemon::v2::AcquireBindingValueResponse acquire_resp;
  grpc::ServerContext acquire_ctx;
  const auto acquire_status = harness->service().AcquireBindingValue(&acquire_ctx, &acquire_req, &acquire_resp);

  REQUIRE(acquire_status.ok());
  REQUIRE_FALSE(acquire_resp.lease_token().empty());
  REQUIRE(acquire_resp.reservation_bytes() == 4);
  REQUIRE(acquire_resp.current_value().binding_value_id() == result.binding_value_ref().binding_value_id());
  REQUIRE(harness->kernel().handle_leases()->release(acquire_resp.lease_token()).ok());
}

TEST_CASE("PrefetchServingBinding set reports per-member failures", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);

  auto req = make_serving_prefetch_set_request();
  req.set_operation_id("prefetch-set-op");
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE(resp.operation_ref().operation_id() == "prefetch-set-op");
  REQUIRE(resp.status().state() == tensorcast::operation::v1::OPERATION_STATE_FAILED);
  REQUIRE(resp.status().has_result());
  tensorcast::operation::v1::PrefetchServingBindingSetResult result;
  REQUIRE(resp.status().result().UnpackTo(&result));
  REQUIRE(result.member_failures_size() == 2);
  REQUIRE(result.members_size() == 0);
  REQUIRE_FALSE(result.partial());
  REQUIRE(result.member_failures(0).phase() == "member_materialization");
}

TEST_CASE("PrefetchServingBinding validates layout then requires source selection", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);

  auto req = make_serving_prefetch_request(true);
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);
  REQUIRE(st.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(std::string(st.error_message()).find("source_selection") != std::string::npos);
}
