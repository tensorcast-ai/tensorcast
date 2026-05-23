// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/owned_binding_service.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/service/rpc_context.h"
#include "grpcpp/server_context.h"

namespace {

namespace v2 = tensorcast::daemon::v2;
using ByteRangeMap = tensorcast::store::loader::ByteRangeMap;
using ByteRangeSegment = tensorcast::store::loader::ByteRangeSegment;
using SourceBoundLoweringArtifacts = tensorcast::store::runtime::ingestion::strategy::SourceBoundLoweringArtifacts;

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

std::string make_source_index_with_extra_tensor_json() {
  return R"({"alpha":[0,4,[1],[1],"torch.float32",0],"beta":[4,4,[1],[1],"torch.float32",0]})";
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

ByteRangeMap make_data_map(uint64_t total_bytes) {
  ByteRangeMap map;
  map.total_bytes = total_bytes;
  map.num_sources = total_bytes > 0 ? 1 : 0;
  if (total_bytes > 0) {
    map.segments.push_back(
        ByteRangeSegment{
            .kind = ByteRangeSegment::Kind::kData,
            .dst_offset = 0,
            .length = total_bytes,
            .src_offset = 0,
            .source_index = 0,
        });
  }
  return map;
}

class BindingContributionGuardClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  std::vector<tensorcast::store::components::AssemblySlotOccupancyInfo> accepted_rows;
  std::vector<std::string> active_identities;
  std::string cluster_id{"cluster-test"};
  std::vector<tensorcast::store::components::ArtifactDiskLocation> disk_locations;

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

  absl::StatusOr<std::string> get_cluster_id() override {
    if (cluster_id.empty()) {
      return absl::NotFoundError("cluster_id unavailable");
    }
    return cluster_id;
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ArtifactDiskLocation>> list_artifact_disk_locations(
      std::string_view artifact_id,
      bool include_deleted = false) override {
    std::vector<tensorcast::store::components::ArtifactDiskLocation> out;
    for (const auto& entry : disk_locations) {
      if (entry.artifact_id != artifact_id) {
        continue;
      }
      if (entry.is_deleted && !include_deleted) {
        continue;
      }
      out.push_back(entry);
    }
    if (out.empty()) {
      return absl::NotFoundError("disk_locations_not_found");
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

  std::shared_ptr<tensorcast::daemon::BindingRegistry::Record> insert_mutable_record(
      std::string_view update_epoch = "epoch-1",
      std::string_view source_artifact_id = "msa1:test-session~policy~partitioned~source") {
    auto record = std::make_shared<tensorcast::daemon::BindingRegistry::Record>();
    record->binding_id = "binding-1";
    record->binding_layout_id = "layout-1";
    record->owner_pid = 123;
    record->device_id = 0;
    record->device_uuid = ensure_fake_gpu_uuid();
    record->ownership = v2::BINDING_OWNERSHIP_DAEMON;
    record->state = v2::BINDING_STATE_MUTABLE;
    record->target_layout = make_target_layout();
    record->target_index_json = make_target_index_json();
    record->source_selection.set_artifact_id(std::string(source_artifact_id));
    record->allocation = std::make_unique<tensorcast::common::memory::GpuDeviceMemory>();
    REQUIRE(record->allocation->allocate(4, record->device_id).ok());
    record->active_update_epoch = std::string(update_epoch);
    record->current_binding_value_id = "value-before-freeze";
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

grpc::Status run_freeze_binding_current_value(
    tensorcast::daemon::OwnedBindingService& service,
    const v2::FreezeBindingCurrentValueRequest& req,
    v2::FreezeBindingCurrentValueResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"FreezeBindingCurrentValueTest", ctx, /*allow_high_card_attrs=*/true};
  return service.freeze_binding_current_value(rctx, req, resp);
}

grpc::Status run_start_promote_binding_current_value(
    tensorcast::daemon::OwnedBindingService& service,
    const v2::StartPromoteBindingCurrentValueRequest& req,
    v2::StartPromoteBindingCurrentValueResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"StartPromoteBindingCurrentValueTest", ctx, /*allow_high_card_attrs=*/true};
  return service.start_promote_binding_current_value(rctx, req, resp);
}

grpc::Status run_get_binding_promotion_status(
    tensorcast::daemon::OwnedBindingService& service,
    const v2::GetBindingPromotionStatusRequest& req,
    v2::GetBindingPromotionStatusResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"GetBindingPromotionStatusTest", ctx, /*allow_high_card_attrs=*/true};
  return service.get_binding_promotion_status(rctx, req, resp);
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

void register_managed_disk_location(Fixture& fix, std::string_view artifact_id, const std::filesystem::path& path) {
  tensorcast::store::components::ArtifactDiskLocation loc;
  loc.artifact_id = std::string(artifact_id);
  loc.cluster_id = fix.global_store_client->cluster_id;
  loc.relative_path = path.lexically_relative(fix.storage_root).string();
  loc.kind = tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED;
  loc.created_at = absl::Now();
  loc.updated_at = absl::Now();
  fix.global_store_client->disk_locations.push_back(std::move(loc));
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

TEST_CASE("FreezeBindingCurrentValue marks mutable binding local-ready pending", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_mutable_record();

  v2::FreezeBindingCurrentValueRequest req;
  req.set_binding_id(record->binding_id);
  req.set_update_epoch("epoch-1");
  req.set_source_artifact_ref("msa1:test-session~policy~partitioned~override");
  v2::FreezeBindingCurrentValueResponse resp;

  const auto status = run_freeze_binding_current_value(fix.service, req, resp);

  INFO("FreezeBindingCurrentValue status: " << status.error_code() << " " << status.error_message());
  REQUIRE(status.ok());
  REQUIRE(resp.state() == v2::BINDING_STATE_READY_LOCAL);
  REQUIRE(resp.has_current_value());
  const auto& value = resp.current_value();
  REQUIRE(value.binding_id() == record->binding_id);
  REQUIRE(value.binding_layout_id() == record->binding_layout_id);
  REQUIRE_FALSE(value.binding_value_id().empty());
  REQUIRE_FALSE(value.is_artifact_backed());
  REQUIRE(value.verification_state() == v2::BINDING_VALUE_VERIFICATION_STATE_PENDING);
  REQUIRE(value.verification_job_id().empty());
  REQUIRE(value.source_artifact_ref() == "msa1:test-session~policy~partitioned~override");
  REQUIRE(
      value.local_serving_ref() == absl::StrCat("binding-local:", record->binding_id, ":", value.binding_value_id()));
  REQUIRE_FALSE(value.has_serving_artifact_id());

  absl::MutexLock lock(&record->mu);
  REQUIRE(record->state == v2::BINDING_STATE_READY_LOCAL);
  REQUIRE(record->active_update_epoch.empty());
  REQUIRE(record->current_artifact_id.empty());
  REQUIRE_FALSE(record->sealed_commit_result.has_value());
  REQUIRE(record->verification_state == v2::BINDING_VALUE_VERIFICATION_STATE_PENDING);
  REQUIRE(record->serving_artifact_id.empty());
}

TEST_CASE("StartPromoteBindingCurrentValue exposes async failure status", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_mutable_record();

  v2::FreezeBindingCurrentValueRequest freeze_req;
  freeze_req.set_binding_id(record->binding_id);
  freeze_req.set_update_epoch("epoch-1");
  v2::FreezeBindingCurrentValueResponse freeze_resp;
  const auto freeze_status = run_freeze_binding_current_value(fix.service, freeze_req, freeze_resp);
  INFO("FreezeBindingCurrentValue status: " << freeze_status.error_code() << " " << freeze_status.error_message());
  REQUIRE(freeze_status.ok());
  const std::string binding_value_id = freeze_resp.current_value().binding_value_id();

  v2::StartPromoteBindingCurrentValueRequest start_req;
  start_req.set_binding_id(record->binding_id);
  start_req.set_binding_value_id(binding_value_id);
  v2::StartPromoteBindingCurrentValueResponse start_resp;
  const auto start_status = run_start_promote_binding_current_value(fix.service, start_req, start_resp);

  REQUIRE(start_status.ok());
  REQUIRE(start_resp.has_status());
  REQUIRE_FALSE(start_resp.status().verification_job_id().empty());
  REQUIRE(start_resp.status().binding_id() == record->binding_id);
  REQUIRE(start_resp.status().binding_value_id() == binding_value_id);

  const auto drain_status = fix.async_runtime.drain(absl::Now() + absl::Seconds(2));
  REQUIRE(drain_status.ok());

  v2::GetBindingPromotionStatusRequest get_req;
  get_req.set_verification_job_id(start_resp.status().verification_job_id());
  v2::GetBindingPromotionStatusResponse get_resp;
  const auto get_status = run_get_binding_promotion_status(fix.service, get_req, get_resp);

  REQUIRE(get_status.ok());
  REQUIRE(get_resp.has_status());
  REQUIRE(get_resp.status().state() == v2::BINDING_PROMOTION_JOB_STATE_FAILED);
  REQUIRE(get_resp.status().failure_reason().find("LipManager is unavailable") != std::string::npos);

  absl::MutexLock lock(&record->mu);
  REQUIRE(record->verification_state == v2::BINDING_VALUE_VERIFICATION_STATE_FAILED);
  REQUIRE(record->verification_job_id == start_resp.status().verification_job_id());
  REQUIRE(record->verification_failure_reason.find("LipManager is unavailable") != std::string::npos);
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

TEST_CASE("RefillOwnedBinding uses first-class collective topology", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();

  auto build_request = [&] {
    v2::RefillOwnedBindingRequest req;
    req.set_binding_id(record->binding_id);
    req.set_artifact_id("artifact-new");
    req.set_operation_id("binding-op");
    auto* topology = req.mutable_execution_topology();
    topology->mutable_collective_load_group()->set_group_id("first-class-group");
    topology->mutable_collective_load_group()->set_world_size(8);
    topology->mutable_collective_load_group()->set_rank(1);
    req.set_collective_policy(v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE);
    return req;
  };

  v2::RefillOwnedBindingResponse resp;
  const auto status = run_refill_owned_binding(fix.service, build_request(), resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_message().find("operation_id collective metadata") == std::string::npos);
}

TEST_CASE("RefillOwnedBinding strict preflight rejection preserves ready artifact state", "[daemon][binding]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();

  const auto plan_summary = tensorcast::store::runtime::ingestion::strategy::SourceBoundExecutionPlanSummary{
      .planned_collective_candidate_bytes = 4,
      .planned_collective_admitted_bytes = 4,
      .planned_local_typed_bytes = 0,
      .planned_non_admitted_typed_bytes = 0,
      .planned_generic_residual_bytes = 0,
      .planner_reject_reason_buckets = {{"source_locality_host_local", 4}},
      .collective_lane_eligible = false,
      .strict_pure_collective_eligible = false,
  };

  const auto status = tensorcast::daemon::evaluate_strict_collective_preflight_for_testing(
      /*rctx=*/nullptr, &plan_summary, v2::CollectivePolicy::COLLECTIVE_POLICY_REQUIRE_COLLECTIVE);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  CHECK(status.error_message().find("pure-collective eligible") != std::string::npos);
  CHECK(status.error_message().find("planned_local_typed_bytes=0") != std::string::npos);
  CHECK(status.error_message().find("source_locality_host_local=4") != std::string::npos);

  absl::MutexLock lock(&record->mu);
  REQUIRE(record->state == v2::BINDING_STATE_READY_ARTIFACT);
  REQUIRE(record->current_artifact_id == "artifact-old");
  REQUIRE(record->current_binding_value_id == "value-1");
}

TEST_CASE(
    "RefillOwnedBinding directly refills mapped binding from selected serving artifact view",
    "[daemon][binding][mapped]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();

  const auto artifact_dir = fix.storage_root / "binding_selected_serving_artifact";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor.data", "ABCDEFGH"));
  REQUIRE(write_file(artifact_dir / "tensor_index.json", make_source_index_with_extra_tensor_json()));

  auto imported_or = tensorcast::daemon::materialization_disk_resolve::import_artifact_from_path(
      artifact_dir.string(),
      fix.storage_root,
      /*verify_checksums=*/false);
  REQUIRE(imported_or.ok());
  register_managed_disk_location(fix, imported_or->artifact_id, imported_or->normalized_path);

  {
    absl::MutexLock lock(&record->mu);
    record->allocation = std::make_unique<tensorcast::common::memory::GpuDeviceMemory>();
    REQUIRE(record->allocation->allocate(4, record->device_id).ok());
    record->mapped = true;
    record->copy_plan.set_version(1);
    record->dst_tensors.clear();
    auto& spec = record->dst_tensors.emplace_back();
    spec.set_name("alpha");
    spec.add_shape(1);
    spec.add_stride(1);
    spec.set_dtype("torch.float32");
    spec.set_storage_offset(0);
    spec.set_logical_length(4);
  }

  v2::RefillOwnedBindingRequest req;
  req.set_binding_id(record->binding_id);
  req.set_artifact_id(imported_or->artifact_id);
  req.mutable_source_selection()->set_artifact_id(imported_or->artifact_id);
  req.mutable_source_selection()->add_tensor_names("alpha");
  req.mutable_source_policy()->set_preference(v2::SOURCE_PREFERENCE_PREFER_DISK);
  req.mutable_source_policy()->set_allow_p2p(false);
  req.mutable_source_policy()->set_allow_disk(true);
  v2::RefillOwnedBindingResponse resp;

  const auto status = run_refill_owned_binding(fix.service, req, resp);

  INFO("RefillOwnedBinding status: " << status.error_code() << " " << status.error_message());
  REQUIRE(status.ok());
  REQUIRE(resp.artifact_id() == imported_or->artifact_id);
  REQUIRE(resp.state() == v2::BINDING_STATE_READY_ARTIFACT);
  REQUIRE(resp.source() == v2::MATERIALIZATION_SOURCE_DISK);
  REQUIRE(resp.resolved_selection().artifact_id() == imported_or->artifact_id);
  REQUIRE(resp.resolved_selection().tensor_names_size() == 1);
  REQUIRE(resp.resolved_selection().tensor_names(0) == "alpha");

  absl::MutexLock lock(&record->mu);
  REQUIRE(record->current_artifact_id == imported_or->artifact_id);
  REQUIRE(record->state == v2::BINDING_STATE_READY_ARTIFACT);
}

TEST_CASE(
    "RefillOwnedBinding rejects byte-only mounted sources for tensor-aware realization",
    "[daemon][binding][byte_only]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();

  const auto artifact_id = "msa1:test-session~policy~partitioned~deadbeef";
  const auto artifact_dir = test_tmpdir() / "binding_byte_only_source";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor.data", "ABCD"));

  auto metadata_or = tensorcast::daemon::materialization_disk_resolve::build_mounted_source_metadata(artifact_dir);
  REQUIRE(metadata_or.ok());
  fix.disk_imports.upsert_binding(
      artifact_id,
      tensorcast::daemon::ArtifactSourceRegistry::Entry{
          .source_kind = tensorcast::daemon::ArtifactSourceRegistry::SourceKind::kMountedSourceArtifact,
          .canonical_source_path = artifact_dir.string(),
          .canonical_index_json = metadata_or->index_info.canonical_index_json,
          .source_index_json = metadata_or->index_info.source_index_json,
          .source_disk_path = artifact_dir.string(),
          .tensor_aware_metadata = false,
          .validate_before_read = true,
          .file_fingerprints = to_registry_fingerprints(metadata_or->file_fingerprints),
      });

  v2::RefillOwnedBindingRequest req;
  req.set_binding_id(record->binding_id);
  req.mutable_source_policy()->set_preference(v2::SOURCE_PREFERENCE_PREFER_DISK);
  auto* public_disk_source = req.mutable_public_disk_source();
  public_disk_source->set_artifact_id(artifact_id);
  public_disk_source->set_path(artifact_dir.string());
  public_disk_source->set_canonical_index_bytes(metadata_or->index_info.canonical_index_json);
  public_disk_source->set_metadata_capability(v2::DISK_METADATA_CAPABILITY_BYTE_ONLY);
  public_disk_source->set_exact_size_bytes(metadata_or->exact_size_bytes);

  auto* realization_plan = req.mutable_realization_plan();
  realization_plan->set_version(1);
  auto* entry = realization_plan->add_entries();
  entry->set_op_kind(v2::BINDING_REALIZATION_OP_KIND_COPY);
  entry->set_source_name("payload");
  entry->set_dst_name("alpha");
  auto* source_range = entry->add_source_ranges();
  source_range->set_dim(0);
  source_range->set_start(0);
  source_range->set_end(4);
  auto* dst_range = entry->add_dst_ranges();
  dst_range->set_dim(0);
  dst_range->set_start(0);
  dst_range->set_end(4);

  v2::RefillOwnedBindingResponse resp;
  const auto status = run_refill_owned_binding(fix.service, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("tensor-aware mounted-source metadata") != std::string::npos);
}

TEST_CASE(
    "RefillOwnedBinding rejects mismatched public disk source canonical index hints",
    "[daemon][binding][public_disk_source][hints]") {
  Fixture fix;
  const auto record = fix.insert_ready_artifact_record();

  const auto artifact_id = "msa1:test-session~policy~partitioned~trusted";
  const auto artifact_dir = test_tmpdir() / "binding_public_disk_source_hint_mismatch";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor.data", "ABCD"));
  REQUIRE(write_file(artifact_dir / "tensor_index.json", make_target_index_json()));

  auto metadata_or = tensorcast::daemon::materialization_disk_resolve::build_mounted_source_metadata(artifact_dir);
  REQUIRE(metadata_or.ok());
  REQUIRE(
      metadata_or->metadata_capability ==
      tensorcast::daemon::materialization_disk_resolve::MountedSourceMetadataCapability::kTensorAware);
  fix.disk_imports.upsert_binding(
      artifact_id,
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

  v2::RefillOwnedBindingRequest req;
  req.set_binding_id(record->binding_id);
  req.mutable_source_policy()->set_preference(v2::SOURCE_PREFERENCE_PREFER_DISK);
  auto* public_disk_source = req.mutable_public_disk_source();
  public_disk_source->set_artifact_id(artifact_id);
  public_disk_source->set_path(artifact_dir.string());
  public_disk_source->set_canonical_index_bytes("bogus-index");

  auto* realization_plan = req.mutable_realization_plan();
  realization_plan->set_version(1);
  auto* entry = realization_plan->add_entries();
  entry->set_op_kind(v2::BINDING_REALIZATION_OP_KIND_COPY);
  entry->set_source_name("alpha");
  entry->set_dst_name("alpha");
  auto* source_range = entry->add_source_ranges();
  source_range->set_dim(0);
  source_range->set_start(0);
  source_range->set_end(1);
  auto* dst_range = entry->add_dst_ranges();
  dst_range->set_dim(0);
  dst_range->set_start(0);
  dst_range->set_end(1);

  v2::RefillOwnedBindingResponse resp;
  const auto status = run_refill_owned_binding(fix.service, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("canonical_index_bytes") != std::string::npos);
}

TEST_CASE("SourceBoundPlanSummary keeps collective lane eligibility separate from pure blockers", "[daemon][binding]") {
  using WorkItem = tensorcast::store::materialization::contracts::RepresentationWorkItem;
  using WorkItemKind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind;
  using WorkPartitionKind = tensorcast::store::materialization::contracts::WorkPartitionKind;

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  tensorcast::store::materialization::contracts::RepresentationWorkPlan work_plan;

  WorkItem collective_item;
  collective_item.kind = WorkItemKind::kTensorCopy;
  collective_item.partition_kind = WorkPartitionKind::kReplicated;
  collective_item.committed_bytes = 8;
  work_plan.items.push_back(collective_item);

  WorkItem residual_item;
  residual_item.kind = WorkItemKind::kResidualByteRange;
  residual_item.committed_bytes = 4;
  work_plan.items.push_back(residual_item);

  WorkItem fill_item;
  fill_item.kind = WorkItemKind::kPadFill;
  fill_item.committed_bytes = 2;
  work_plan.items.push_back(fill_item);

  resolved_plan.representation_work_plan = work_plan;

  tensorcast::store::StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  strategy_config.enable_owner_file_collective = true;
  strategy_config.allow_mixed_execution = true;
  strategy_config.owner_file_collective_allow_mixed_residual = true;
  strategy_config.owner_file_collective_min_dedup_saving_bytes = 0;

  tensorcast::store::loading::ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      tensorcast::store::loading::CollectiveLoadGroupHint{.group_id = "group-a", .world_size = 4, .rank = 1};
  execution_topology.source_locality = tensorcast::store::loading::SourceLocalityHint::kSharedSource;

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(12);

  const auto summary = tensorcast::daemon::summarize_source_bound_plan_for_testing(
      resolved_plan,
      lowering_artifacts,
      strategy_config,
      execution_topology,
      v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST,
      /*disk_source_available=*/true);

  CHECK(summary.collective_lane_eligible);
  CHECK_FALSE(summary.strict_pure_collective_eligible);
  CHECK(summary.execution_plan_kind == "collective_first_mixed");
  CHECK(summary.planned_collective_candidate_bytes == 8);
  CHECK(summary.planned_collective_admitted_bytes == 8);
  CHECK(summary.planned_generic_residual_bytes == 4);
  CHECK(summary.planned_local_typed_bytes == 2);
  CHECK(summary.planner_reject_reason_buckets.empty());
}

TEST_CASE(
    "SourceBoundPlanSummary keeps admitted collective bytes when typed work falls back to generic",
    "[daemon][binding]") {
  using WorkItem = tensorcast::store::materialization::contracts::RepresentationWorkItem;
  using WorkItemKind = tensorcast::store::materialization::contracts::RepresentationWorkItemKind;
  using WorkPartitionKind = tensorcast::store::materialization::contracts::WorkPartitionKind;

  tensorcast::store::runtime::ingestion::strategy::ResolvedMaterializationPlan resolved_plan;
  tensorcast::store::materialization::contracts::RepresentationWorkPlan work_plan;

  WorkItem collective_item;
  collective_item.kind = WorkItemKind::kTensorCopy;
  collective_item.partition_kind = WorkPartitionKind::kReplicated;
  collective_item.committed_bytes = 8;
  work_plan.items.push_back(collective_item);

  WorkItem concat_item;
  concat_item.kind = WorkItemKind::kConcatAssemble;
  concat_item.committed_bytes = 4;
  work_plan.items.push_back(concat_item);

  resolved_plan.representation_work_plan = work_plan;

  tensorcast::store::StoreEngineOptions::MaterializationStrategyConfig strategy_config;
  strategy_config.enable_owner_file_collective = true;
  strategy_config.allow_mixed_execution = true;
  strategy_config.owner_file_collective_min_dedup_saving_bytes = 0;

  tensorcast::store::loading::ExecutionTopologyContext execution_topology;
  execution_topology.collective_load_group =
      tensorcast::store::loading::CollectiveLoadGroupHint{.group_id = "group-a", .world_size = 4, .rank = 1};
  execution_topology.source_locality = tensorcast::store::loading::SourceLocalityHint::kSharedSource;

  SourceBoundLoweringArtifacts lowering_artifacts;
  lowering_artifacts.collective_data_map = make_data_map(8);
  lowering_artifacts.executor_generic_data_map = make_data_map(12);

  const auto summary = tensorcast::daemon::summarize_source_bound_plan_for_testing(
      resolved_plan,
      lowering_artifacts,
      strategy_config,
      execution_topology,
      v2::CollectivePolicy::COLLECTIVE_POLICY_COLLECTIVE_FIRST,
      /*disk_source_available=*/true);

  CHECK(summary.collective_lane_eligible);
  CHECK_FALSE(summary.strict_pure_collective_eligible);
  CHECK(summary.execution_plan_kind == "collective_first_mixed");
  CHECK(summary.planned_collective_candidate_bytes == 8);
  CHECK(summary.planned_collective_admitted_bytes == 8);
  CHECK(summary.planned_non_admitted_typed_bytes == 4);
  REQUIRE(summary.planner_reject_reason_buckets.contains("typed_work_without_source_overlap"));
  CHECK(summary.planner_reject_reason_buckets.at("typed_work_without_source_overlap") == 4);
  REQUIRE(summary.planner_reject_reason_buckets.contains("typed_work_not_collective_admitted"));
  CHECK(summary.planner_reject_reason_buckets.at("typed_work_not_collective_admitted") == 4);
  CHECK_FALSE(summary.planner_reject_reason_buckets.contains("local_typed_work_present"));
  CHECK_FALSE(summary.planner_reject_reason_buckets.contains("true_generic_residual_present"));
}
