// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/owned_binding_service.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/service/rpc_context.h"
#include "grpcpp/server_context.h"

namespace {

namespace v2 = tensorcast::daemon::v2;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_owned_binding_service_test";
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

v2::TargetLayout make_target_layout() {
  v2::TargetLayout layout;
  layout.set_layout_kind(v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout.set_index_kind(v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout.set_tensor_spec_kind(v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout.add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);

  auto* offset = layout.add_offsets();
  offset->set_name("alpha");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);
  return layout;
}

class BindingContributionGuardClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::vector<tensorcast::store::components::AssemblySlotOccupancyInfo> accepted_rows;
  std::vector<std::string> active_identities;

  absl::StatusOr<std::vector<std::string>> list_active_worker_identities(bool) override {
    return active_identities;
  }

  absl::StatusOr<std::vector<tensorcast::store::components::AssemblySlotOccupancyInfo>> list_assembly_slot_occupancies(
      std::optional<std::string_view>,
      std::optional<std::string_view>,
      std::optional<std::string_view> binding_id,
      std::optional<std::string_view> binding_value_id,
      const std::vector<std::string>& states) override {
    std::vector<tensorcast::store::components::AssemblySlotOccupancyInfo> out;
    for (const auto& row : accepted_rows) {
      if (binding_id.has_value() && row.binding_id != *binding_id) {
        continue;
      }
      if (binding_value_id.has_value() && row.binding_value_id != *binding_value_id) {
        continue;
      }
      if (!states.empty()) {
        bool matched = false;
        for (const auto& state : states) {
          if (row.state == state) {
            matched = true;
            break;
          }
        }
        if (!matched) {
          continue;
        }
      }
      out.push_back(row);
    }
    return out;
  }
};

struct Fixture {
  std::shared_ptr<tensorcast::store::StoreEngine> engine;
  std::shared_ptr<BindingContributionGuardClient> global_store_client;
  tensorcast::common::AsyncRuntime async_runtime;
  tensorcast::daemon::DeviceResolver devices;
  tensorcast::daemon::ArtifactSourceRegistry disk_imports;
  tensorcast::daemon::BindingRegistry binding_registry;
  tensorcast::daemon::ShutdownSignal shutdown_signal;
  tensorcast::daemon::WorkerIdentityStore identity;
  std::filesystem::path storage_root;
  tensorcast::daemon::OwnedBindingService service;

  Fixture()
      : engine(std::make_shared<tensorcast::store::StoreEngine>(make_opts())),
        global_store_client(std::make_shared<BindingContributionGuardClient>()),
        devices(tensorcast::store::DeviceRegistry::instance()),
        storage_root(test_tmpdir()),
        service(
            tensorcast::daemon::OwnedBindingService::Dep{
                .engine = *engine,
                .devices = devices,
                .disk_imports = disk_imports,
                .bindings = binding_registry,
                .shutdown_signal = shutdown_signal,
                .async_runtime = async_runtime,
                .identity = identity,
                .global_store_client = global_store_client,
                .storage_path = storage_root}) {
    std::filesystem::create_directories(storage_root);
    engine->set_global_store_client_for_testing(global_store_client);
    ensure_fake_gpu_uuid();
  }

  std::shared_ptr<tensorcast::daemon::BindingRegistry::Record> insert_ready_artifact_record() {
    auto record = std::make_shared<tensorcast::daemon::BindingRegistry::Record>();
    record->binding_id = "binding-1";
    record->binding_layout_id = "layout-1";
    record->owner_pid = 123;
    record->device_id = 0;
    record->device_uuid = ensure_fake_gpu_uuid();
    record->ownership = v2::BINDING_OWNERSHIP_DAEMON;
    record->state = v2::BINDING_STATE_READY_ARTIFACT;
    record->target_layout = make_target_layout();
    record->target_index_json = make_target_index_json();
    record->current_artifact_id = "artifact-old";
    record->current_binding_value_id = "value-1";
    record->seal_generation = 1;
    record->current_selection.set_artifact_id("artifact-old");
    record->source_selection.set_artifact_id("artifact-old");
    REQUIRE(binding_registry.insert(record).ok());
    return record;
  }
};

grpc::Status run_create_owned_binding(
    tensorcast::daemon::OwnedBindingService& service,
    const v2::CreateOwnedBindingRequest& req,
    v2::CreateOwnedBindingResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"CreateOwnedBindingTest", ctx, /*allow_high_card_attrs=*/true};
  return service.create_owned_binding(rctx, req, resp);
}

grpc::Status run_begin_binding_update(
    tensorcast::daemon::OwnedBindingService& service,
    const v2::BeginBindingUpdateRequest& req,
    v2::BeginBindingUpdateResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"BeginBindingUpdateTest", ctx, /*allow_high_card_attrs=*/true};
  return service.begin_binding_update(rctx, req, resp);
}

grpc::Status run_commit_binding_artifact(
    tensorcast::daemon::OwnedBindingService& service,
    const v2::CommitBindingArtifactRequest& req,
    v2::CommitBindingArtifactResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"CommitBindingArtifactTest", ctx, /*allow_high_card_attrs=*/true};
  return service.commit_binding_artifact(rctx, req, resp);
}

grpc::Status run_refill_owned_binding(
    tensorcast::daemon::OwnedBindingService& service,
    const v2::RefillOwnedBindingRequest& req,
    v2::RefillOwnedBindingResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"RefillOwnedBindingTest", ctx, /*allow_high_card_attrs=*/true};
  return service.refill_owned_binding(rctx, req, resp);
}

void seed_live_contribution(Fixture& fix) {
  tensorcast::store::components::AssemblySlotOccupancyInfo row;
  row.attempt_id = "attempt-1";
  row.slot_id = "view-a";
  row.structural_view_id = std::string("view-a");
  row.binding_id = "binding-1";
  row.binding_value_id = "value-1";
  row.contributor_daemon_id = "daemon-1";
  row.state = "accepted";
  row.lease_expires_at = absl::Now() + absl::Seconds(30);
  fix.global_store_client->accepted_rows = {row};
  fix.global_store_client->active_identities = {"daemon-1"};
}

void seed_expired_contribution(Fixture& fix) {
  tensorcast::store::components::AssemblySlotOccupancyInfo row;
  row.attempt_id = "attempt-1";
  row.slot_id = "view-a";
  row.structural_view_id = std::string("view-a");
  row.binding_id = "binding-1";
  row.binding_value_id = "value-1";
  row.contributor_daemon_id = "daemon-1";
  row.state = "accepted";
  row.lease_expires_at = absl::Now() - absl::Seconds(30);
  fix.global_store_client->accepted_rows = {row};
  fix.global_store_client->active_identities = {"daemon-1"};
}

} // namespace

TEST_CASE("CreateOwnedBinding requires binding_layout_id", "[daemon][binding]") {
  Fixture fix;

  v2::CreateOwnedBindingRequest req;
  req.mutable_source_selection()->set_artifact_id("artifact-1");
  *req.mutable_target_layout() = make_target_layout();
  req.set_target_index_bytes(make_target_index_json());
  req.set_device_uuid(ensure_fake_gpu_uuid());
  req.set_pid(123);

  v2::CreateOwnedBindingResponse resp;
  const auto status = run_create_owned_binding(fix.service, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "binding_layout_id is required");
}

TEST_CASE("BeginBindingUpdate rejects live assembly contributions", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();
  seed_live_contribution(fix);

  v2::BeginBindingUpdateRequest req;
  req.set_binding_id(record->binding_id);
  v2::BeginBindingUpdateResponse resp;

  const auto status = run_begin_binding_update(fix.service, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("live assembly contributions") != std::string::npos);
}

TEST_CASE("CommitBindingArtifact rejects live assembly contributions", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();
  seed_live_contribution(fix);

  v2::CommitBindingArtifactRequest req;
  req.set_binding_id(record->binding_id);
  req.mutable_selection()->set_artifact_id("artifact-new");
  v2::CommitBindingArtifactResponse resp;

  const auto status = run_commit_binding_artifact(fix.service, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("live assembly contributions") != std::string::npos);
  absl::MutexLock lock(&record->mu);
  REQUIRE(record->current_artifact_id == "artifact-old");
  REQUIRE(record->current_binding_value_id == "value-1");
}

TEST_CASE("BeginBindingUpdate ignores expired assembly contributions", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();
  seed_expired_contribution(fix);

  v2::BeginBindingUpdateRequest req;
  req.set_binding_id(record->binding_id);
  v2::BeginBindingUpdateResponse resp;

  const auto status = run_begin_binding_update(fix.service, req, resp);

  REQUIRE(status.ok());
  REQUIRE_FALSE(resp.update_epoch().empty());
}

TEST_CASE("RefillOwnedBinding rejects live assembly contributions before overwrite", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();
  seed_live_contribution(fix);

  v2::RefillOwnedBindingRequest req;
  req.set_binding_id(record->binding_id);
  req.set_artifact_id("artifact-new");
  v2::RefillOwnedBindingResponse resp;

  const auto status = run_refill_owned_binding(fix.service, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("live assembly contributions") != std::string::npos);
  absl::MutexLock lock(&record->mu);
  REQUIRE(record->current_artifact_id == "artifact-old");
  REQUIRE(record->current_binding_value_id == "value-1");
}

TEST_CASE(
    "RefillOwnedBinding ignores operation_id collective metadata once first-class topology is present",
    "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();

  auto build_request = [&](bool with_operation_metadata) {
    v2::RefillOwnedBindingRequest req;
    req.set_binding_id(record->binding_id);
    req.set_artifact_id("artifact-new");
    if (with_operation_metadata) {
      req.set_operation_id("binding-op#tcg:clid=compat-group;clws=8;clrk=1");
    } else {
      req.set_operation_id("binding-op");
    }
    auto* topology = req.mutable_execution_topology();
    topology->mutable_collective_load_group()->set_group_id("first-class-group");
    topology->mutable_collective_load_group()->set_world_size(8);
    topology->mutable_collective_load_group()->set_rank(1);
    req.set_collective_policy(v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE);
    return req;
  };

  v2::RefillOwnedBindingResponse resp_without_metadata;
  const auto status_without_metadata =
      run_refill_owned_binding(fix.service, build_request(/*with_operation_metadata=*/false), resp_without_metadata);

  v2::RefillOwnedBindingResponse resp_with_metadata;
  const auto status_with_metadata =
      run_refill_owned_binding(fix.service, build_request(/*with_operation_metadata=*/true), resp_with_metadata);

  REQUIRE(status_with_metadata.error_code() == status_without_metadata.error_code());
  REQUIRE(status_with_metadata.error_message() == status_without_metadata.error_message());
  REQUIRE(
      status_with_metadata.error_message().find("conflicts with operation_id collective metadata") ==
      std::string::npos);
}
