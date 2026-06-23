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
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <unistd.h>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "grpcpp/server_context.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace {

namespace global_store = tensorcast::global_store::v1;
namespace operation = tensorcast::operation::v1;

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
  std::vector<operation::OperationState> states;
  std::atomic<int> get_calls{0};
  std::atomic<int> acquire_operation_calls{0};
  std::atomic<int> update_operation_calls{0};
  std::atomic<int> release_operation_calls{0};
  std::atomic<int> begin_group_calls{0};
  std::atomic<int> register_group_version_set_calls{0};
  std::atomic<int> prepared_group_calls{0};
  std::atomic<int> wait_group_calls{0};
  absl::Status get_error = absl::OkStatus();
  absl::Status begin_group_error = absl::OkStatus();
  absl::Status prepared_group_error = absl::OkStatus();
  absl::Status wait_group_error = absl::OkStatus();
  std::string begin_transaction_id{"txn-1"};
  std::string begin_version_set_id{"version-set-1"};
  global_store::Status prepared_group_status{global_store::STATUS_OK};
  global_store::Status wait_group_status{global_store::STATUS_OK};
  global_store::GroupRealizationState wait_group_state{global_store::GROUP_REALIZATION_STATE_PUBLISHED};
  global_store::BeginOrJoinGroupRealizationRequest last_begin_group_request;
  global_store::RegisterGroupVersionSetRequest last_register_group_version_set_request;
  global_store::ReportGroupRealizationPreparedRequest last_prepared_group_request;
  global_store::WaitGroupRealizationPublishedRequest last_wait_group_request;
  std::function<void(operation::OperationRef*)> fill_ref;
  std::function<absl::Status(const global_store::ReportGroupRealizationPreparedRequest&)> report_prepared_hook;

  absl::StatusOr<operation::AcquireOperationLeaseResponse> acquire_operation_lease(
      const operation::AcquireOperationLeaseRequest& req) override {
    acquire_operation_calls.fetch_add(1, std::memory_order_relaxed);
    operation::AcquireOperationLeaseResponse resp;
    auto* lease = resp.mutable_lease();
    lease->set_operation_id(req.operation_id());
    lease->set_lease_token("lease:" + req.operation_id());
    lease->set_owner_id(req.owner_id());
    {
      absl::MutexLock lock(&operation_mu_);
      auto& record = operations_[req.operation_id()];
      record.mutable_ref()->set_operation_id(req.operation_id());
      record.mutable_ref()->set_kind(req.kind());
      record.mutable_ref()->set_target_artifact_id(req.target_artifact_id());
      record.set_lease_generation(record.lease_generation() + 1);
      record.set_lease_owner(req.owner_id());
      lease->set_lease_generation(record.lease_generation());
      active_lease_tokens_[lease->lease_token()] = req.operation_id();
    }
    resp.set_acquired(true);
    return resp;
  }

  absl::Status update_operation(const operation::UpdateOperationRequest& req) override {
    update_operation_calls.fetch_add(1, std::memory_order_relaxed);
    absl::MutexLock lock(&operation_mu_);
    auto& record = operations_[req.operation_id()];
    if (record.ref().operation_id().empty()) {
      record.mutable_ref()->set_operation_id(req.operation_id());
    }
    record.mutable_status()->CopyFrom(req.status());
    record.set_lease_generation(req.lease_generation());
    if (req.has_snapshot()) {
      record.mutable_snapshot()->CopyFrom(req.snapshot());
      operation::OperationContinuationMetadata metadata;
      if (req.snapshot().UnpackTo(&metadata)) {
        record.mutable_ref()->CopyFrom(metadata.ref());
      }
    }
    return absl::OkStatus();
  }

  absl::StatusOr<operation::KeepaliveOperationLeaseResponse> keepalive_operation_lease(
      const operation::KeepaliveOperationLeaseRequest& req) override {
    operation::KeepaliveOperationLeaseResponse resp;
    absl::MutexLock lock(&operation_mu_);
    auto it = active_lease_tokens_.find(req.lease_token());
    if (it == active_lease_tokens_.end()) {
      return absl::NotFoundError("unknown lease token");
    }
    auto record_it = operations_.find(it->second);
    if (record_it == operations_.end()) {
      return absl::NotFoundError("unknown operation");
    }
    auto* lease = resp.mutable_lease();
    lease->set_operation_id(it->second);
    lease->set_lease_token(req.lease_token());
    lease->set_lease_generation(record_it->second.lease_generation());
    lease->set_owner_id(record_it->second.lease_owner());
    return resp;
  }

  absl::StatusOr<operation::ReleaseOperationLeaseResponse> release_operation_lease(
      const operation::ReleaseOperationLeaseRequest& req) override {
    release_operation_calls.fetch_add(1, std::memory_order_relaxed);
    operation::ReleaseOperationLeaseResponse resp;
    absl::MutexLock lock(&operation_mu_);
    resp.set_released(active_lease_tokens_.erase(req.lease_token()) > 0);
    return resp;
  }

  absl::StatusOr<operation::GetOperationResponse> get_operation(const operation::GetOperationRequest& req) override {
    get_calls.fetch_add(1, std::memory_order_relaxed);
    if (!get_error.ok()) {
      return get_error;
    }
    {
      absl::MutexLock lock(&operation_mu_);
      auto it = operations_.find(req.operation_id());
      if (it != operations_.end()) {
        return it->second;
      }
    }
    operation::GetOperationResponse resp;
    auto* ref = resp.mutable_ref();
    ref->set_operation_id(req.operation_id());
    if (fill_ref != nullptr) {
      fill_ref(ref);
    }
    auto* status = resp.mutable_status();
    const size_t idx = static_cast<size_t>(std::max(0, get_calls.load(std::memory_order_relaxed) - 1));
    const size_t state_idx = states.empty() ? 0 : std::min(idx, states.size() - 1);
    status->set_state(states.empty() ? operation::OperationState::OPERATION_STATE_UNSPECIFIED : states[state_idx]);
    return resp;
  }

  absl::StatusOr<global_store::RegisterGroupVersionSetResponse> register_group_version_set(
      const global_store::RegisterGroupVersionSetRequest& req,
      const tensorcast::store::components::RpcOptions&) override {
    register_group_version_set_calls.fetch_add(1, std::memory_order_relaxed);
    {
      absl::MutexLock lock(&operation_mu_);
      last_register_group_version_set_request.CopyFrom(req);
    }
    global_store::RegisterGroupVersionSetResponse resp;
    resp.set_status(global_store::STATUS_OK);
    resp.mutable_version_set()->set_version_set_id("gvs-registered");
    resp.mutable_version_set()->set_manifest_hash("manifest-hash");
    resp.mutable_version_set()->set_manifest_generation(5);
    resp.set_realization_kind(req.realization_kind());
    for (const auto& part : req.parts()) {
      resp.add_parts()->CopyFrom(part);
    }
    return resp;
  }

  absl::StatusOr<global_store::BeginOrJoinGroupRealizationResponse> begin_or_join_group_realization(
      const global_store::BeginOrJoinGroupRealizationRequest& req,
      const tensorcast::store::components::RpcOptions&) override {
    begin_group_calls.fetch_add(1, std::memory_order_relaxed);
    {
      absl::MutexLock lock(&operation_mu_);
      last_begin_group_request.CopyFrom(req);
    }
    if (!begin_group_error.ok()) {
      return begin_group_error;
    }
    global_store::BeginOrJoinGroupRealizationResponse resp;
    resp.set_status(global_store::STATUS_OK);
    resp.set_transaction_id(begin_transaction_id);
    resp.mutable_version_set()->set_version_set_id(begin_version_set_id);
    resp.set_realization_kind(global_store::GROUP_REALIZATION_KIND_SAME_SELECTION);
    auto* part = resp.mutable_part();
    part->set_part_id(req.context().part_id());
    if (req.version().has_explicit_selection()) {
      part->mutable_selection()->CopyFrom(req.version().explicit_selection());
    }
    part->set_selection_hash("selection-hash");
    resp.set_state(global_store::GROUP_REALIZATION_STATE_PREPARING);
    resp.set_transaction_fingerprint("transaction-fingerprint");
    return resp;
  }

  absl::StatusOr<global_store::ReportGroupRealizationPreparedResponse> report_group_realization_prepared(
      const global_store::ReportGroupRealizationPreparedRequest& req,
      const tensorcast::store::components::RpcOptions&) override {
    prepared_group_calls.fetch_add(1, std::memory_order_relaxed);
    if (report_prepared_hook != nullptr) {
      const absl::Status hook_status = report_prepared_hook(req);
      if (!hook_status.ok()) {
        return hook_status;
      }
    }
    {
      absl::MutexLock lock(&operation_mu_);
      last_prepared_group_request.CopyFrom(req);
    }
    if (!prepared_group_error.ok()) {
      return prepared_group_error;
    }
    global_store::ReportGroupRealizationPreparedResponse resp;
    resp.set_status(prepared_group_status);
    resp.set_state(global_store::GROUP_REALIZATION_STATE_PUBLISHED);
    resp.set_member_state(global_store::GROUP_REALIZATION_MEMBER_STATE_PREPARED);
    resp.set_member_fingerprint("member-fingerprint");
    return resp;
  }

  absl::StatusOr<global_store::WaitGroupRealizationPublishedResponse> wait_group_realization_published(
      const global_store::WaitGroupRealizationPublishedRequest& req,
      const tensorcast::store::components::RpcOptions&) override {
    wait_group_calls.fetch_add(1, std::memory_order_relaxed);
    {
      absl::MutexLock lock(&operation_mu_);
      last_wait_group_request.CopyFrom(req);
    }
    if (!wait_group_error.ok()) {
      return wait_group_error;
    }
    global_store::WaitGroupRealizationPublishedResponse resp;
    resp.set_status(wait_group_status);
    resp.set_state(wait_group_state);
    return resp;
  }

 private:
  absl::Mutex operation_mu_;
  std::unordered_map<std::string, operation::GetOperationResponse> operations_;
  std::unordered_map<std::string, std::string> active_lease_tokens_;
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

operation::PrefetchServingBindingResult wait_prefetch_serving_binding_result(
    tensorcast::daemon::DaemonServiceHarness& harness,
    std::string_view operation_id) {
  tensorcast::daemon::v2::WaitOperationRequest wait_req;
  wait_req.set_operation_id(std::string(operation_id));
  wait_req.set_timeout_ms(5000);
  tensorcast::daemon::v2::WaitOperationResponse wait_resp;
  grpc::ServerContext wait_ctx;
  const auto wait_status = harness.service().WaitOperation(&wait_ctx, &wait_req, &wait_resp);
  INFO("WaitOperation status: " << wait_status.error_code() << " " << wait_status.error_message());
  REQUIRE(wait_status.ok());
  REQUIRE(wait_resp.operation().status().state() == operation::OPERATION_STATE_SUCCESS);
  operation::PrefetchServingBindingResult result;
  REQUIRE(wait_resp.operation().status().result().UnpackTo(&result));
  return result;
}

operation::GetOperationResponse wait_prefetch_serving_binding_terminal(
    tensorcast::daemon::DaemonServiceHarness& harness,
    std::string_view operation_id) {
  tensorcast::daemon::v2::WaitOperationRequest wait_req;
  wait_req.set_operation_id(std::string(operation_id));
  wait_req.set_timeout_ms(5000);
  tensorcast::daemon::v2::WaitOperationResponse wait_resp;
  grpc::ServerContext wait_ctx;
  const auto wait_status = harness.service().WaitOperation(&wait_ctx, &wait_req, &wait_resp);
  INFO("WaitOperation status: " << wait_status.error_code() << " " << wait_status.error_message());
  REQUIRE(wait_status.ok());
  return wait_resp.operation();
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
  auto* offset = record->target_layout.add_offsets();
  offset->set_storage_offset(0);
  offset->set_logical_length(4096);
  record->target_layout_hash = "layout-hash";
  record->tensor_schema_hash = "tensor-schema";
  record->daemon_id = "daemon-1";
  record->daemon_session_id = "session-1";
  record->serving_build_digest = "serving-build";
  auto& payload = record->payloads.emplace_back();
  payload.set_name("alpha");
  payload.add_shape(4);
  payload.add_stride(1);
  payload.set_buffer_offset(0);
  payload.set_byte_length(4);
  payload.set_storage_offset(0);
  payload.set_dtype("torch.uint8");
  payload.set_device_uuid("GPU-0");
  record->reservation_capability_id = "capability-1";
  record->reservation_expires_at = absl::Now() + absl::Hours(1);
  record->local_serving_ref = "binding-local:binding-acquire:value-1";
  record->serving_member.set_member_id("member-0");
  record->serving_member.set_member_index(0);
  record->serving_member.set_member_count(1);
  record->serving_member.set_group_id("group-1");
  return record;
}

tensorcast::daemon::BindingRegistry::StagedBindingValue make_staged_value_for_acquire() {
  tensorcast::daemon::BindingRegistry::StagedBindingValue staged;
  staged.transaction_id = "txn-1";
  staged.version_set_id = "version-set-1";
  staged.part_id = "part-0";
  staged.binding_value_id = "staged-value-1";
  staged.staging_token = "staging-token-1";
  staged.staging_epoch = 11;
  staged.materialization_attempt_id = "attempt-1";
  staged.selection.set_artifact_id("artifact-staged");
  staged.artifact_id = "artifact-staged";
  staged.target_index_json = R"({"tensors":["staged"]})";
  staged.target_layout_hash = "staged-layout-hash";
  staged.tensor_schema_hash = "staged-schema";
  staged.logical_total_size = 4096;
  staged.expected_previous_seal_generation = 7;
  staged.source_replica_id = "source-replica-1";
  staged.source_export_generation = 13;
  staged.child_transport_request_id = "child-transport-1";
  staged.verification_state = tensorcast::daemon::v2::BINDING_VALUE_VERIFICATION_STATE_LOCAL_ONLY;
  staged.expires_at = absl::Now() + absl::Hours(1);
  return staged;
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
  req.set_caller_pid(static_cast<int32_t>(::getpid()));
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

tensorcast::daemon::v2::AcquireBindingValueRequest make_group_acquire_request() {
  auto req = make_acquire_request();
  req.mutable_binding_value_ref()->set_binding_value_id("staged-value-1");
  req.set_expected_target_layout_hash("staged-layout-hash");
  req.set_expected_tensor_schema_hash("staged-schema");
  req.mutable_reservation_capability()->mutable_binding_value_ref()->CopyFrom(req.binding_value_ref());
  auto* acquire = req.mutable_group_realization_acquire();
  acquire->set_transaction_id("txn-1");
  acquire->set_version_set_id("version-set-1");
  acquire->set_part_id("part-0");
  acquire->set_staging_token("staging-token-1");
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

TEST_CASE("RegisterGroupVersionSet proxies through daemon Global Store client", "[daemon][operation][group]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, false);

  tensorcast::daemon::v2::RegisterGroupVersionSetRequest req;
  req.set_realization_kind(tensorcast::daemon::v2::GROUP_REALIZATION_KIND_PER_PART_SELECTION);
  req.set_namespace_("wp-run");
  req.set_key("model:m:v1");
  auto* part0 = req.add_parts();
  part0->set_part_id("rx0:r0");
  part0->mutable_selection()->set_artifact_id("mi2:rank0");
  auto* part1 = req.add_parts();
  part1->set_part_id("rx0:r1");
  part1->mutable_selection()->set_artifact_id("mi2:rank1");

  tensorcast::daemon::v2::RegisterGroupVersionSetResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().RegisterGroupVersionSet(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE(client->register_group_version_set_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(
      client->last_register_group_version_set_request.realization_kind() ==
      global_store::GROUP_REALIZATION_KIND_PER_PART_SELECTION);
  REQUIRE(client->last_register_group_version_set_request.namespace_() == "wp-run");
  REQUIRE(client->last_register_group_version_set_request.key() == "model:m:v1");
  REQUIRE(client->last_register_group_version_set_request.parts_size() == 2);
  REQUIRE(resp.version_set().version_set_id() == "gvs-registered");
  REQUIRE(resp.realization_kind() == tensorcast::daemon::v2::GROUP_REALIZATION_KIND_PER_PART_SELECTION);
  REQUIRE(resp.parts_size() == 2);
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
  REQUIRE(resp.payloads_size() == 1);
  REQUIRE(resp.payloads(0).name() == "alpha");
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

TEST_CASE("AcquireBindingValue can borrow retained binding by local_serving_ref", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  auto record = make_retained_serving_record();
  REQUIRE(harness->kernel().binding_registry().insert(record).ok());

  tensorcast::daemon::v2::AcquireBindingValueRequest req;
  req.set_local_serving_ref("binding-local:binding-acquire:value-1");
  req.set_expected_device_uuid("GPU-0");
  auto* expected_member = req.mutable_expected_member();
  expected_member->set_member_id("member-0");
  expected_member->set_member_index(0);
  expected_member->set_member_count(1);
  expected_member->set_group_id("group-1");
  req.set_expected_tensor_schema_hash("tensor-schema");
  req.set_expected_serving_build_digest("serving-build");
  req.set_caller_pid(static_cast<int32_t>(::getpid()));

  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE_FALSE(resp.lease_token().empty());
  REQUIRE(resp.mem_handle().lease_token() == resp.lease_token());
  REQUIRE(resp.target_index_bytes() == R"({"tensors":[]})");
  REQUIRE(resp.payloads_size() == 1);
  REQUIRE(resp.payloads(0).name() == "alpha");
  REQUIRE(resp.reservation_bytes() == 4096);
  REQUIRE(resp.current_value().binding_id() == "binding-acquire");
  REQUIRE(resp.current_value().binding_layout_id() == "layout-1");
  REQUIRE(resp.current_value().binding_value_id() == "value-1");
  REQUIRE(resp.current_value().local_serving_ref() == "binding-local:binding-acquire:value-1");
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 1);
  }

  tensorcast::daemon::v2::ReleasePlacementLeaseRequest release_req;
  release_req.set_lease_token(resp.lease_token());
  tensorcast::daemon::v2::ReleasePlacementLeaseResponse release_resp;
  grpc::ServerContext release_ctx;
  const auto release_status = harness->service().ReleasePlacementLease(&release_ctx, &release_req, &release_resp);
  REQUIRE(release_status.ok());
  REQUIRE(release_resp.released());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
  }
}

TEST_CASE(
    "AcquireBindingValue can borrow caller-owned local binding by local_serving_ref",
    "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  auto record = make_retained_serving_record();
  {
    absl::MutexLock lock(&record->mu);
    record->control_lifetime = tensorcast::daemon::BindingRegistry::ControlLifetime::kPidBound;
    record->retained_ref = false;
    record->owner_pid = static_cast<int32_t>(::getpid());
    record->serving_member.Clear();
  }
  REQUIRE(harness->kernel().binding_registry().insert(record).ok());

  tensorcast::daemon::v2::AcquireBindingValueRequest req;
  req.set_local_serving_ref("binding-local:binding-acquire:value-1");
  req.set_expected_device_uuid("GPU-0");
  auto* expected_member = req.mutable_expected_member();
  expected_member->set_member_id("member-0");
  expected_member->set_member_index(0);
  expected_member->set_member_count(1);
  expected_member->set_group_id("group-1");
  req.set_expected_tensor_schema_hash("serving-schema");
  req.set_expected_serving_build_digest("published-serving-build");
  req.set_caller_pid(static_cast<int32_t>(::getpid()));

  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE_FALSE(resp.lease_token().empty());
  REQUIRE(resp.mem_handle().lease_token() == resp.lease_token());
  REQUIRE(resp.target_index_bytes() == R"({"tensors":[]})");
  REQUIRE(resp.reservation_bytes() == 4096);
  REQUIRE(resp.current_value().binding_id() == "binding-acquire");
  REQUIRE(resp.current_value().binding_value_id() == "value-1");
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
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

TEST_CASE("AcquireBindingValue returns staged value only after group publish", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  auto record = make_retained_serving_record();
  REQUIRE(harness->kernel().binding_registry().insert(record).ok());
  REQUIRE(harness->kernel()
              .binding_registry()
              .insert_staged_value(record->binding_id, make_staged_value_for_acquire(), absl::Now())
              .ok());

  auto req = make_group_acquire_request();
  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE(client->wait_group_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(client->last_wait_group_request.transaction_id() == "txn-1");
  REQUIRE(resp.acquired_staged_value());
  REQUIRE_FALSE(resp.has_current_value());
  REQUIRE(resp.acquired_value().binding_value_id() == "staged-value-1");
  REQUIRE(resp.acquired_value().seal_generation() == 7);
  REQUIRE(resp.acquired_value().source_artifact_id() == "artifact-staged");
  REQUIRE(resp.target_index_bytes() == R"({"tensors":["staged"]})");
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->current_binding_value_id == "value-1");
    REQUIRE(record->active_attachment_refs == 1);
  }

  REQUIRE(harness->kernel().handle_leases() != nullptr);
  REQUIRE(harness->kernel().handle_leases()->release(resp.lease_token()).ok());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->active_attachment_refs == 0);
  }
}

TEST_CASE("AcquireBindingValue rejects staged value before group publish", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  client->wait_group_state = global_store::GROUP_REALIZATION_STATE_PREPARING;
  auto harness = make_harness(client, true);
  auto record = make_retained_serving_record();
  REQUIRE(harness->kernel().binding_registry().insert(record).ok());
  REQUIRE(harness->kernel()
              .binding_registry()
              .insert_staged_value(record->binding_id, make_staged_value_for_acquire(), absl::Now())
              .ok());

  auto req = make_group_acquire_request();
  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);

  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(client->wait_group_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(resp.lease_token().empty());
  REQUIRE(harness->kernel().binding_registry().get_staged_value("binding-acquire", "staged-value-1").ok());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->current_binding_value_id == "value-1");
    REQUIRE(record->active_attachment_refs == 0);
  }
}

TEST_CASE(
    "AcquireBindingValue removes staged value after group terminal non-published state",
    "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  client->wait_group_state = global_store::GROUP_REALIZATION_STATE_ABORTED;
  auto harness = make_harness(client, true);
  auto record = make_retained_serving_record();
  REQUIRE(harness->kernel().binding_registry().insert(record).ok());
  REQUIRE(harness->kernel()
              .binding_registry()
              .insert_staged_value(record->binding_id, make_staged_value_for_acquire(), absl::Now())
              .ok());

  auto req = make_group_acquire_request();
  tensorcast::daemon::v2::AcquireBindingValueResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().AcquireBindingValue(&ctx, &req, &resp);

  REQUIRE(st.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(client->wait_group_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE_FALSE(harness->kernel().binding_registry().get_staged_value("binding-acquire", "staged-value-1").ok());
  REQUIRE(resp.lease_token().empty());
  {
    absl::MutexLock lock(&record->mu);
    REQUIRE(record->current_binding_value_id == "value-1");
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
  req.mutable_retention_policy()->set_idle_ttl_after_last_release_ms(1000);
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);

  INFO("PrefetchServingBinding status: " << st.error_code() << " " << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.operation_ref().operation_id() == "prefetch-serving-op");
  REQUIRE(resp.status().state() == operation::OPERATION_STATE_RUNNING);
  const auto result = wait_prefetch_serving_binding_result(*harness, resp.operation_ref().operation_id());
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
  acquire_req.set_caller_pid(static_cast<int32_t>(::getpid()));
  tensorcast::daemon::v2::AcquireBindingValueResponse acquire_resp;
  grpc::ServerContext acquire_ctx;
  const auto acquire_status = harness->service().AcquireBindingValue(&acquire_ctx, &acquire_req, &acquire_resp);

  REQUIRE(acquire_status.ok());
  REQUIRE_FALSE(acquire_resp.lease_token().empty());
  REQUIRE(acquire_resp.reservation_bytes() == 4);
  REQUIRE(acquire_resp.current_value().binding_value_id() == result.binding_value_ref().binding_value_id());
  REQUIRE(harness->kernel().handle_leases()->release(acquire_resp.lease_token()).ok());
  auto record_or = harness->kernel().binding_registry().get(result.binding_value_ref().binding_id());
  REQUIRE(record_or.ok());
  {
    absl::MutexLock lock(&(*record_or)->mu);
    REQUIRE((*record_or)->active_attachment_refs == 0);
    REQUIRE((*record_or)->idle_ttl_after_last_release == absl::Milliseconds(1000));
    REQUIRE((*record_or)->idle_deadline != absl::InfiniteFuture());
  }
  REQUIRE(harness->kernel().binding_registry().sweep_retention(absl::Now() + absl::Seconds(2)) == 1);
  REQUIRE_FALSE(harness->kernel().binding_registry().get(result.binding_value_ref().binding_id()).ok());
}

TEST_CASE("PrefetchServingBinding reuses live completed operation", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  const std::string artifact_id = "msa1:test-session~policy~partitioned~serving-reuse";
  seed_mounted_source_artifact(*harness, artifact_id);

  auto req = make_serving_prefetch_request(true);
  req.mutable_source_selection()->set_artifact_id(artifact_id);
  req.set_operation_id("prefetch-serving-reuse-op");
  req.mutable_retention_policy()->set_idle_ttl_after_last_release_ms(30000);
  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);

  INFO("PrefetchServingBinding status: " << st.error_code() << " " << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.operation_ref().operation_id() == "prefetch-serving-reuse-op");
  REQUIRE(resp.status().state() == operation::OPERATION_STATE_RUNNING);
  const auto result = wait_prefetch_serving_binding_result(*harness, resp.operation_ref().operation_id());
  REQUIRE_FALSE(result.local_serving_ref().empty());

  const auto binding_count_after_first = harness->kernel().binding_registry().size();
  const int acquire_calls_after_first = client->acquire_operation_calls.load(std::memory_order_relaxed);
  const int update_calls_after_first = client->update_operation_calls.load(std::memory_order_relaxed);

  tensorcast::daemon::v2::PrefetchServingBindingResponse second_resp;
  grpc::ServerContext second_ctx;
  const auto second_st = harness->service().PrefetchServingBinding(&second_ctx, &req, &second_resp);

  INFO("Second PrefetchServingBinding status: " << second_st.error_code() << " " << second_st.error_message());
  REQUIRE(second_st.ok());
  REQUIRE(second_resp.operation_ref().operation_id() == "prefetch-serving-reuse-op");
  REQUIRE(second_resp.status().state() == operation::OPERATION_STATE_SUCCESS);
  operation::PrefetchServingBindingResult second_result;
  REQUIRE(second_resp.status().result().UnpackTo(&second_result));
  REQUIRE(second_result.local_serving_ref() == result.local_serving_ref());
  REQUIRE(second_result.binding_value_ref().binding_id() == result.binding_value_ref().binding_id());
  REQUIRE(harness->kernel().binding_registry().size() == binding_count_after_first);
  REQUIRE(client->acquire_operation_calls.load(std::memory_order_relaxed) == acquire_calls_after_first);
  REQUIRE(client->update_operation_calls.load(std::memory_order_relaxed) == update_calls_after_first);
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

TEST_CASE("PrefetchServingBinding set fans out members before collecting", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  client->report_prepared_hook = [client](const global_store::ReportGroupRealizationPreparedRequest&) -> absl::Status {
    const absl::Time deadline = absl::Now() + absl::Seconds(2);
    while (client->prepared_group_calls.load(std::memory_order_relaxed) < 2 && absl::Now() < deadline) {
      absl::SleepFor(absl::Milliseconds(10));
    }
    if (client->prepared_group_calls.load(std::memory_order_relaxed) < 2) {
      return absl::DeadlineExceededError("serving binding set members were not materialized concurrently");
    }
    return absl::OkStatus();
  };
  auto harness = make_harness(client, true);
  const std::string artifact_id = "msa1:test-session~policy~partitioned~serving-set-fanout";
  seed_mounted_source_artifact(*harness, artifact_id);

  auto req = make_serving_prefetch_set_request();
  req.mutable_source_selection()->set_artifact_id(artifact_id);
  req.set_operation_id("prefetch-set-fanout-op");
  auto* target_set = req.mutable_serving_binding_set_target();
  const std::string device_uuid = ensure_fake_gpu_uuid();
  for (auto& member : *target_set->mutable_members()) {
    member.set_device("cuda:0");
    member.set_device_uuid(device_uuid);
  }
  auto* group_realization = req.mutable_group_realization();
  group_realization->set_enabled(true);
  group_realization->set_require_staged_publish(true);
  group_realization->mutable_version()->mutable_explicit_selection()->CopyFrom(req.source_selection());
  auto* group = group_realization->mutable_group();
  group->set_group_kind("serving_prefetch");
  group->set_group_id("group-1");
  group->set_epoch(1);
  group->set_total_parts(2);
  group->add_required_part_ids("member-0");
  group->add_required_part_ids("member-1");

  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);

  INFO("PrefetchServingBinding set fanout status: " << st.error_code() << " " << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.operation_ref().operation_id() == "prefetch-set-fanout-op");
  REQUIRE(resp.status().state() == operation::OPERATION_STATE_SUCCESS);
  operation::PrefetchServingBindingSetResult result;
  REQUIRE(resp.status().result().UnpackTo(&result));
  REQUIRE_FALSE(result.partial());
  REQUIRE(result.member_failures_size() == 0);
  REQUIRE(result.members_size() == 2);
  REQUIRE(result.members(0).member().member_id() == "member-0");
  REQUIRE(result.members(1).member().member_id() == "member-1");
  REQUIRE(client->prepared_group_calls.load(std::memory_order_relaxed) == 2);
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

TEST_CASE("PrefetchServingBinding creates staged value for group realization", "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  auto harness = make_harness(client, true);
  const std::string artifact_id = "msa1:test-session~policy~partitioned~serving-group";
  seed_mounted_source_artifact(*harness, artifact_id);

  auto req = make_serving_prefetch_request(true);
  req.mutable_source_selection()->set_artifact_id(artifact_id);
  req.set_operation_id("prefetch-serving-group-op");
  auto* group_realization = req.mutable_group_realization();
  group_realization->set_enabled(true);
  group_realization->set_require_staged_publish(true);
  group_realization->mutable_version()->mutable_explicit_selection()->CopyFrom(req.source_selection());
  auto* group = group_realization->mutable_group();
  group->set_group_kind("serving_prefetch");
  group->set_group_id("group-1");
  group->set_epoch(1);
  group->set_total_parts(1);
  group->set_part_id("member-0");
  group->add_required_part_ids("member-0");

  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);

  INFO("PrefetchServingBinding status: " << st.error_code() << " " << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(resp.operation_ref().operation_id() == "prefetch-serving-group-op");
  REQUIRE(resp.status().state() == operation::OPERATION_STATE_RUNNING);
  const auto result = wait_prefetch_serving_binding_result(*harness, resp.operation_ref().operation_id());
  REQUIRE(result.staged_value());
  REQUIRE(result.group_realization_transaction_id() == client->begin_transaction_id);
  REQUIRE(result.group_realization_version_set_id() == client->begin_version_set_id);
  REQUIRE(result.group_realization_part_id() == "member-0");
  REQUIRE_FALSE(result.group_realization_staging_token().empty());
  REQUIRE(result.group_realization_wait_for_publish());
  REQUIRE(result.group_realization_wait_timeout_ms() == 30000);
  REQUIRE_FALSE(result.binding_value_ref().binding_value_id().empty());
  REQUIRE(result.binding_value_ref().seal_generation() == 0);
  REQUIRE(client->begin_group_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(client->prepared_group_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(client->last_begin_group_request.context().part_id() == "member-0");
  REQUIRE(client->last_prepared_group_request.transaction_id() == client->begin_transaction_id);
  REQUIRE(client->last_prepared_group_request.part_id() == "member-0");
  REQUIRE(
      client->last_prepared_group_request.staged_value().binding_value_id() ==
      result.binding_value_ref().binding_value_id());
  REQUIRE(client->last_prepared_group_request.expected_previous_seal_generation() == 0);

  auto record_or = harness->kernel().binding_registry().get(result.binding_value_ref().binding_id());
  REQUIRE(record_or.ok());
  {
    absl::MutexLock lock(&(*record_or)->mu);
    REQUIRE((*record_or)->current_binding_value_id.empty());
    REQUIRE((*record_or)->retained_ref);
    REQUIRE((*record_or)->staged_values_by_id.contains(result.binding_value_ref().binding_value_id()));
  }

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
  acquire_req.set_caller_pid(static_cast<int32_t>(::getpid()));
  auto* acquire = acquire_req.mutable_group_realization_acquire();
  acquire->set_transaction_id(result.group_realization_transaction_id());
  acquire->set_version_set_id(result.group_realization_version_set_id());
  acquire->set_part_id(result.group_realization_part_id());
  acquire->set_staging_token(result.group_realization_staging_token());
  tensorcast::daemon::v2::AcquireBindingValueResponse acquire_resp;
  grpc::ServerContext acquire_ctx;
  const auto acquire_status = harness->service().AcquireBindingValue(&acquire_ctx, &acquire_req, &acquire_resp);

  REQUIRE(acquire_status.ok());
  REQUIRE(acquire_resp.acquired_staged_value());
  REQUIRE_FALSE(acquire_resp.has_current_value());
  REQUIRE(acquire_resp.acquired_value().binding_value_id() == result.binding_value_ref().binding_value_id());
  REQUIRE(harness->kernel().handle_leases()->release(acquire_resp.lease_token()).ok());
}

TEST_CASE(
    "PrefetchServingBinding cleans staged value when prepared report is cancelled",
    "[daemon][operation][serving]") {
  auto client = std::make_shared<OperationClient>();
  client->prepared_group_error = absl::CancelledError("prepared report cancelled");
  auto harness = make_harness(client, true);
  const std::string artifact_id = "msa1:test-session~policy~partitioned~serving-group-cancelled";
  seed_mounted_source_artifact(*harness, artifact_id);

  auto req = make_serving_prefetch_request(true);
  req.mutable_source_selection()->set_artifact_id(artifact_id);
  req.set_operation_id("prefetch-serving-group-cancelled-op");
  auto* group_realization = req.mutable_group_realization();
  group_realization->set_enabled(true);
  group_realization->set_require_staged_publish(true);
  group_realization->mutable_version()->mutable_explicit_selection()->CopyFrom(req.source_selection());
  auto* group = group_realization->mutable_group();
  group->set_group_kind("serving_prefetch");
  group->set_group_id("group-1");
  group->set_epoch(1);
  group->set_total_parts(1);
  group->set_part_id("member-0");
  group->add_required_part_ids("member-0");

  tensorcast::daemon::v2::PrefetchServingBindingResponse resp;
  grpc::ServerContext ctx;
  const auto st = harness->service().PrefetchServingBinding(&ctx, &req, &resp);

  REQUIRE(st.ok());
  REQUIRE(resp.operation_ref().operation_id() == "prefetch-serving-group-cancelled-op");
  REQUIRE(resp.status().state() == operation::OPERATION_STATE_RUNNING);
  const auto terminal = wait_prefetch_serving_binding_terminal(*harness, resp.operation_ref().operation_id());
  REQUIRE(terminal.status().state() == operation::OPERATION_STATE_FAILED);
  REQUIRE(terminal.status().has_error());
  REQUIRE(terminal.status().error().status_code() == "CANCELLED");
  REQUIRE(client->begin_group_calls.load(std::memory_order_relaxed) == 1);
  REQUIRE(client->prepared_group_calls.load(std::memory_order_relaxed) == 1);
  const auto& staged_ref = client->last_prepared_group_request.staged_value();
  REQUIRE_FALSE(staged_ref.binding_id().empty());
  REQUIRE_FALSE(staged_ref.binding_value_id().empty());
  REQUIRE_FALSE(harness->kernel().binding_registry().get(staged_ref.binding_id()).ok());
  REQUIRE_FALSE(harness->kernel()
                    .binding_registry()
                    .get_staged_value(staged_ref.binding_id(), staged_ref.binding_value_id())
                    .ok());
  REQUIRE(harness->kernel().handle_leases() != nullptr);
  REQUIRE(harness->kernel().handle_leases()->size() == 0);
}
