// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <unistd.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/view/view_identity.h"
#include "core/store/store_engine.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/service/controllers/materialization_disk_resolve_utils.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/background_scheduler.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/registration_manager.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/verification_tracker.h"
#include "daemon/testing/cuda_ipc_spawn_helper.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

using tensorcast::daemon::MaterializationController;
using tensorcast::daemon::v2::MaterializeIntoTargetRequest;
using tensorcast::daemon::v2::MaterializeIntoTargetResponse;

std::string make_canonical_index_json() {
  return R"({"a":[0,4,[4],[1],"torch.uint8",0],"b":[4,4,[4],[1],"torch.uint8",0]})";
}

std::string make_multi_region_index_json() {
  return R"({"a":[0,8,[8],[1],"torch.uint8",0],"b":[8,8,[8],[1],"torch.uint8",0]})";
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

absl::StatusOr<std::string> compute_view_id(
    const tensorcast::common::v1::ViewSpec& view_spec,
    std::string_view canonical_index_json) {
  tensorcast::store::loader::ViewSpec parsed_spec;
  for (const auto& [tensor_name, ops_proto] : view_spec.tensors()) {
    tensorcast::store::loader::TensorViewOps ops;
    for (const auto& op_proto : ops_proto.ops()) {
      switch (op_proto.kind_case()) {
        case tensorcast::common::v1::Op::kNarrow: {
          const auto& narrow = op_proto.narrow();
          ops.ops.push_back(
              tensorcast::store::loader::ViewOp::Narrow(
                  tensorcast::store::loader::NarrowOp{
                      .dim = static_cast<int32_t>(narrow.dim()),
                      .start = narrow.start(),
                      .length = narrow.length(),
                  }));
          break;
        }
        case tensorcast::common::v1::Op::kTranspose: {
          const auto& transpose = op_proto.transpose();
          ops.ops.push_back(
              tensorcast::store::loader::ViewOp::Transpose(
                  tensorcast::store::loader::TransposeOp{
                      .dim0 = static_cast<int32_t>(transpose.dim0()),
                      .dim1 = static_cast<int32_t>(transpose.dim1()),
                  }));
          break;
        }
        case tensorcast::common::v1::Op::KIND_NOT_SET:
          return absl::InvalidArgumentError("view op kind not set");
      }
    }
    parsed_spec.tensors.emplace(tensor_name, std::move(ops));
  }
  return tensorcast::store::loader::compute_view_id_from_spec(parsed_spec, canonical_index_json);
}

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_materialize_into_target_test";
}

std::filesystem::path ensure_dir(std::filesystem::path path) {
  std::filesystem::create_directories(path);
  return path;
}

void register_disk_location(
    tensorcast::store::testing::RecordingGlobalStoreClient& client,
    std::string_view artifact_id,
    const std::filesystem::path& relative_path) {
  tensorcast::store::components::ArtifactDiskLocation loc;
  loc.artifact_id = std::string(artifact_id);
  loc.cluster_id = client.cluster_id;
  loc.relative_path = relative_path.string();
  loc.kind = tensorcast::global_store::v1::DISK_LOCATION_KIND_MANAGED;
  client.disk_locations.push_back(std::move(loc));
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 0; // Let the OS pick an available port for test isolation.
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

struct ValidationFixture {
  std::shared_ptr<tensorcast::store::StoreEngine> engine;
  std::shared_ptr<tensorcast::store::testing::RecordingGlobalStoreClient> global_store_client;
  tensorcast::daemon::RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions;
  tensorcast::daemon::LipManager lip_mgr;
  tensorcast::daemon::LipBridge lip_bridge;
  tensorcast::daemon::ReplicaSessionManager session_mgr;
  tensorcast::daemon::VerificationTracker verif_tracker;
  tensorcast::daemon::BackgroundScheduler scheduler;
  tensorcast::daemon::SessionLifecycleManager lifecycle_mgr;
  tensorcast::daemon::LifecycleKernel lifecycle_kernel;
  tensorcast::daemon::SessionsService sessions_svc;
  tensorcast::daemon::RegistrationManager registration_mgr;
  tensorcast::daemon::DeviceResolver devices;
  tensorcast::daemon::ExternalTargetAccessService external_target_access_service;
  tensorcast::daemon::ArtifactSourceRegistry disk_imports;
  tensorcast::daemon::BindingRegistry binding_registry;
  tensorcast::daemon::ShutdownSignal shutdown_signal;
  tensorcast::common::AsyncRuntime async_runtime;
  tensorcast::daemon::WorkerIdentityStore identity;
  std::filesystem::path storage_root;
  MaterializationController controller;

  explicit ValidationFixture(bool external_target_verification_enabled = false)
      : engine(std::make_shared<tensorcast::store::StoreEngine>(make_opts())),
        global_store_client(std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>()),
        regions(tensorcast::daemon::IpcRegionRegistry::Options{}),
        lip_mgr(engine, &regions),
        lip_bridge(lip_mgr),
        session_mgr(std::chrono::seconds(60)),
        verif_tracker(),
        scheduler(),
        lifecycle_mgr(session_mgr, refs, lip_mgr, *engine),
        lifecycle_kernel("daemon-validation-test"),
        sessions_svc(session_mgr, verif_tracker, &scheduler, &lifecycle_mgr, absl::Seconds(60)),
        devices(tensorcast::store::DeviceRegistry::instance()),
        external_target_access_service(
            tensorcast::daemon::ExternalTargetAccessService::Dep{.devices = devices, .regions = regions}),
        binding_registry(),
        storage_root(ensure_dir(test_tmpdir())),
        controller(MaterializationController(
            MaterializationController::Dep{
                .engine = *engine,
                .refs = refs,
                .sessions = sessions_svc,
                .lip = lip_bridge,
                .lip_manager = lip_mgr,
                .registration_manager = registration_mgr,
                .devices = devices,
                .regions = regions,
                .disk_imports = disk_imports,
                .binding_registry = binding_registry,
                .shutdown_signal = shutdown_signal,
                .async_runtime = async_runtime,
                .identity = identity,
                .external_target_access_service = external_target_access_service,
                .global_store_client = global_store_client,
                .lifecycle = &lifecycle_mgr,
                .lifecycle_kernel = &lifecycle_kernel,
                .external_target_verification_enabled = external_target_verification_enabled,
                .storage_path = storage_root,
            })) {
    global_store_client->canonical_index_json = make_canonical_index_json();
    engine->set_global_store_client_for_testing(global_store_client);
  }
};

struct RegisteredGpuRegion {
  tensorcast::daemon::testing::CudaIpcChild child;
  std::string region_id;
  int owner_pid{0};
  std::string device_uuid;
};

absl::StatusOr<RegisteredGpuRegion> register_single_gpu_region(ValidationFixture& fix, uint64_t size_bytes) {
  auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  if (!helper_path_or.ok()) {
    return helper_path_or.status();
  }
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = size_bytes, .fill_byte = -1},
  };
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(*helper_path_or, 0, buffers);
  if (!child_or.ok()) {
    return child_or.status();
  }
  auto child = std::move(*child_or);
  if (child.handle_bytes().empty()) {
    return absl::FailedPreconditionError("cuda ipc child did not expose handle bytes");
  }

  const int owner_pid = child.pid();
  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.device_id = 0;
  params.owner_pid = owner_pid;
  params.size_bytes = size_bytes;
  params.ttl_ms = 10'000;
  params.handle_bytes = child.handle_bytes().front();
  auto region_or = fix.regions.register_region(params);
  if (!region_or.ok()) {
    return region_or.status();
  }

  auto device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  if (device_key.uuid.empty()) {
    tensorcast::store::DeviceRegistry::instance().register_gpu(0, "fake-uuid-0");
    device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  }

  return RegisteredGpuRegion{
      .child = std::move(child),
      .region_id = region_or->region_id,
      .owner_pid = owner_pid,
      .device_uuid = device_key.uuid,
  };
}

grpc::Status run_request(
    MaterializationController& controller,
    const MaterializeIntoTargetRequest& req,
    MaterializeIntoTargetResponse& resp) {
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"MaterializeIntoTargetTest", ctx, /*allow_high_card_attrs=*/true};
  return controller.materialize_into_target(rctx, req, resp);
}

} // namespace

TEST_CASE("MaterializeIntoTarget rejects missing artifact_id", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("MaterializeIntoTarget accepts subset selection", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.mutable_selection()->add_tensor_names("a");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id("region-0");
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  const bool expected_status = status.error_code() == grpc::StatusCode::NOT_FOUND ||
      status.error_code() == grpc::StatusCode::FAILED_PRECONDITION;
  REQUIRE(expected_status);
}

TEST_CASE(
    "MaterializeIntoTarget falls back to disk index when Global Store disconnected",
    "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  fix.global_store_client->connected = false;

  const auto storage_root = fix.storage_root;
  const auto artifact_rel =
      std::filesystem::path("clusters") / fix.global_store_client->cluster_id / "objects" / "artifact_disk_only";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor_index.json", make_canonical_index_json()));
  register_disk_location(*fix.global_store_client, "mi2:dummy:dummy", artifact_rel);

  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);
  req.mutable_selection()->add_tensor_names("a");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id("region-0");
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  const bool expected_status = status.error_code() == grpc::StatusCode::NOT_FOUND ||
      status.error_code() == grpc::StatusCode::FAILED_PRECONDITION;
  REQUIRE(expected_status);
}

TEST_CASE(
    "MaterializeIntoTarget falls back to local import disk when Global Store connected",
    "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  fix.global_store_client->connected = true;

  const auto artifact_dir = test_tmpdir() / "artifact_target_local_import_fallback";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor_index.json", make_canonical_index_json()));

  fix.disk_imports.upsert_binding(
      "mi2:dummy:dummy",
      tensorcast::daemon::ArtifactSourceRegistry::Entry{
          .source_kind = tensorcast::daemon::ArtifactSourceRegistry::SourceKind::kLocalImport,
          .canonical_source_path = artifact_dir.string(),
      });

  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);
  req.mutable_selection()->add_tensor_names("a");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id("region-0");
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  const bool expected_status = status.error_code() == grpc::StatusCode::NOT_FOUND ||
      status.error_code() == grpc::StatusCode::FAILED_PRECONDITION;
  REQUIRE(expected_status);
}

TEST_CASE(
    "MaterializeIntoTarget rejects tensor-aware selection from byte-only mounted sources",
    "[daemon][materialize][into_target][byte_only]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 8);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);

  const auto artifact_id = "msa1:test-session~policy~partitioned~deadbeef";
  const auto artifact_dir = test_tmpdir() / "artifact_target_byte_only_selection";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor.data", "ABCDEFGH"));

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

  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.mutable_selection()->add_tensor_names("payload");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("payload");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(8);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  INFO("status=" << status.error_code() << " message=" << status.error_message());
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("tensor-aware mounted-source metadata") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoTarget rejects stale msa1 after mounted source mutation",
    "[daemon][materialize][into_target][msa1][mutation]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 8);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);

  const auto artifact_id = "msa1:test-session~policy~partitioned~deadbeef";
  const auto artifact_dir = test_tmpdir() / "artifact_target_mutated_msa1";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(write_file(artifact_dir / "tensor.data", "ABCDEFGH"));

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

  REQUIRE(std::filesystem::remove(artifact_dir / "tensor.data"));

  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("payload");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(8);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
  REQUIRE(status.error_message().find("SOURCE_MUTATED") != std::string::npos);
}

TEST_CASE("MaterializeIntoTarget accepts ordered full selection", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.mutable_selection()->add_tensor_names("b");
  req.mutable_selection()->add_tensor_names("a");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(12);
  storage->set_vram_region_id("region-0");
  storage->set_mapping_base_offset(0);

  auto* offset_b = layout->add_offsets();
  offset_b->set_name("b");
  offset_b->set_storage_id("storage-0");
  offset_b->set_storage_offset(0);
  offset_b->set_logical_length(4);

  auto* offset_a = layout->add_offsets();
  offset_a->set_name("a");
  offset_a->set_storage_id("storage-0");
  offset_a->set_storage_offset(8);
  offset_a->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::NOT_FOUND);
}

TEST_CASE("MaterializeIntoTarget accepts view spec subset", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.mutable_selection()->add_tensor_names("a");

  tensorcast::common::v1::ViewSpec view_spec;
  auto* ops = (*view_spec.mutable_tensors())["a"].add_ops();
  ops->mutable_narrow()->set_dim(0);
  ops->mutable_narrow()->set_start(0);
  ops->mutable_narrow()->set_length(2);
  req.mutable_selection()->mutable_view_spec()->CopyFrom(view_spec);

  auto view_id_or = compute_view_id(view_spec, make_canonical_index_json());
  REQUIRE(view_id_or.ok());

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);
  layout->set_view_id(*view_id_or);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(2);
  storage->set_vram_region_id("region-0");
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(2);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::NOT_FOUND);
}

TEST_CASE("MaterializeIntoTarget requires device_uuid", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.mutable_target_layout();
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE(
    "MaterializeIntoTarget rejects tensor_names mismatch with target_layout",
    "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 4);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);
  req.mutable_selection()->add_tensor_names("a");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("b");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "selection.tensor_names do not match target_layout entries");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget rejects unknown tensor in target_layout", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 4);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("unknown_tensor");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "target_layout includes unknown tensor name");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget canonical index rejects subset selection", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 4);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "index_kind CANONICAL cannot be used with view/subset selection");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget rejects offset referencing unknown storage_id", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 4);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-missing");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "target_layout references unknown storage_id");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget rejects offset mismatch", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 4);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(1);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().starts_with("target_layout storage_offset mismatch"));
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget view index requires view or ordered selection", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 8);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset_a = layout->add_offsets();
  offset_a->set_name("a");
  offset_a->set_storage_id("storage-0");
  offset_a->set_storage_offset(0);
  offset_a->set_logical_length(4);

  auto* offset_b = layout->add_offsets();
  offset_b->set_name("b");
  offset_b->set_storage_id("storage-0");
  offset_b->set_storage_offset(4);
  offset_b->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "index_kind VIEW requires view or selection order");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget canonical layout forbids target view_id", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 8);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);
  layout->set_view_id("unexpected-view-id");

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset_a = layout->add_offsets();
  offset_a->set_name("a");
  offset_a->set_storage_id("storage-0");
  offset_a->set_storage_offset(0);
  offset_a->set_logical_length(4);

  auto* offset_b = layout->add_offsets();
  offset_b->set_name("b");
  offset_b->set_storage_id("storage-0");
  offset_b->set_storage_offset(4);
  offset_b->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "target_layout.view_id not allowed for canonical layout");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE(
    "MaterializeIntoTarget subset-only view layout requires empty view_id",
    "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 4);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);
  req.mutable_selection()->add_tensor_names("a");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);
  layout->set_view_id("unexpected-view-id");

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "target_layout.view_id must be empty for subset-only layouts");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget rejects alias dtype mismatch", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 8);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* alias_a = layout->add_aliases();
  alias_a->set_name("a");
  alias_a->set_storage_id("storage-0");
  alias_a->set_storage_offset(0);
  alias_a->set_logical_length(4);
  alias_a->set_dtype("torch.float16");
  alias_a->add_shape(4);
  alias_a->add_stride(1);

  auto* alias_b = layout->add_aliases();
  alias_b->set_name("b");
  alias_b->set_storage_id("storage-0");
  alias_b->set_storage_offset(4);
  alias_b->set_logical_length(4);
  alias_b->set_dtype("torch.uint8");
  alias_b->add_shape(4);
  alias_b->add_stride(1);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "target_layout alias dtype mismatch");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget rejects alias stride mismatch", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 8);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_ALIAS_UNSPECIFIED);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* alias_a = layout->add_aliases();
  alias_a->set_name("a");
  alias_a->set_storage_id("storage-0");
  alias_a->set_storage_offset(0);
  alias_a->set_logical_length(4);
  alias_a->set_dtype("torch.uint8");
  alias_a->add_shape(4);
  alias_a->add_stride(2);

  auto* alias_b = layout->add_aliases();
  alias_b->set_name("b");
  alias_b->set_storage_id("storage-0");
  alias_b->set_storage_offset(4);
  alias_b->set_logical_length(4);
  alias_b->set_dtype("torch.uint8");
  alias_b->add_shape(4);
  alias_b->add_stride(1);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "target_layout alias stride mismatch");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget rejects mismatched view_subset_hash", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 4);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);
  req.mutable_selection()->add_tensor_names("a");
  req.mutable_selection()->set_view_subset_hash("invalid-subset-hash");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(4);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "view_subset_hash does not match tensor_names");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget rejects subset hash for full selection", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  auto registered_or = register_single_gpu_region(fix, 8);
  REQUIRE(registered_or.ok());
  auto registered = std::move(*registered_or);
  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid(registered.device_uuid);
  req.set_pid(registered.owner_pid);
  req.mutable_selection()->set_view_subset_hash("unexpected-full-hash");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(registered.region_id);
  storage->set_mapping_base_offset(0);

  auto* offset_a = layout->add_offsets();
  offset_a->set_name("a");
  offset_a->set_storage_id("storage-0");
  offset_a->set_storage_offset(0);
  offset_a->set_logical_length(4);

  auto* offset_b = layout->add_offsets();
  offset_b->set_name("b");
  offset_b->set_storage_id("storage-0");
  offset_b->set_storage_offset(4);
  offset_b->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message() == "view_subset_hash must be empty for full selection");
  REQUIRE(fix.regions.unregister_region(registered.region_id, registered.owner_pid, /*force=*/true).ok());
  REQUIRE(registered.child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget writes into multiple regions", "[daemon][materialize][into_target]") {
  ValidationFixture fix;

  const auto storage_root = fix.storage_root;
  const auto artifact_rel =
      std::filesystem::path("clusters") / fix.global_store_client->cluster_id / "objects" / "artifact_multi_region";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);

  const std::string payload = std::string(8, 'A') + std::string(8, 'B');
  REQUIRE(write_file(artifact_dir / "tensor.data_0", payload));

  const std::string canonical_index_json = make_multi_region_index_json();
  REQUIRE(write_file(artifact_dir / "tensor_index.json", canonical_index_json));
  fix.global_store_client->canonical_index_json = canonical_index_json;

  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  REQUIRE(device_count > 0);
  tensorcast::cuda::DeviceGuard guard(0);
  REQUIRE(guard.status().ok());

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = 8, .fill_byte = -1},
      {.size_bytes = 8, .fill_byte = -1},
  };
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(*helper_path_or, 0, buffers);
  INFO("cuda_ipc_helper spawn status: " << child_or.status());
  REQUIRE(child_or.ok());
  auto child = std::move(*child_or);
  REQUIRE(child.handle_bytes().size() == 2);
  const std::string& handle_bytes_first = child.handle_bytes()[0];
  const std::string& handle_bytes_second = child.handle_bytes()[1];

  const int owner_pid = child.pid();
  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.device_id = 0;
  params.owner_pid = owner_pid;
  params.size_bytes = 8;
  params.ttl_ms = 10'000;
  params.handle_bytes = handle_bytes_first;
  auto region0_or = fix.regions.register_region(params);
  REQUIRE(region0_or.ok());
  params.handle_bytes = handle_bytes_second;
  auto region1_or = fix.regions.register_region(params);
  REQUIRE(region1_or.ok());

  auto device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  if (device_key.uuid.empty()) {
    tensorcast::store::DeviceRegistry::instance().register_gpu(0, "fake-uuid-0");
    device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  }

  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_multi_region");
  req.set_device_uuid(device_key.uuid);
  req.set_pid(owner_pid);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);
  register_disk_location(*fix.global_store_client, "artifact_multi_region", artifact_rel);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(8);
  storage0->set_vram_region_id(region0_or->region_id);
  storage0->set_mapping_base_offset(0);

  auto* storage1 = layout->add_storages();
  storage1->set_storage_id("storage-1");
  storage1->set_device_id(0);
  storage1->set_storage_length(8);
  storage1->set_vram_region_id(region1_or->region_id);
  storage1->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(8);

  auto* offset1 = layout->add_offsets();
  offset1->set_name("b");
  offset1->set_storage_id("storage-1");
  offset1->set_storage_offset(0);
  offset1->set_logical_length(8);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE(status.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);

  std::array<char, 8> out0{};
  std::array<char, 8> out1{};
  auto mapping0_or = tensorcast::cuda::IpcMapping::open(
      handle_bytes_first, tensorcast::cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
  REQUIRE(mapping0_or.ok());
  auto mapping1_or = tensorcast::cuda::IpcMapping::open(
      handle_bytes_second, tensorcast::cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
  REQUIRE(mapping1_or.ok());
  auto mapping0 = std::move(*mapping0_or);
  auto mapping1 = std::move(*mapping1_or);
  REQUIRE(tensorcast::cuda::memcpy(out0.data(), mapping0.get(), out0.size(), cudaMemcpyDeviceToHost).ok());
  REQUIRE(tensorcast::cuda::memcpy(out1.data(), mapping1.get(), out1.size(), cudaMemcpyDeviceToHost).ok());

  REQUIRE(std::string(out0.begin(), out0.end()) == std::string(8, 'A'));
  REQUIRE(std::string(out1.begin(), out1.end()) == std::string(8, 'B'));

  REQUIRE(fix.regions.unregister_region(region0_or->region_id, owner_pid, /*force=*/true).ok());
  REQUIRE(fix.regions.unregister_region(region1_or->region_id, owner_pid, /*force=*/true).ok());
  REQUIRE(child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoTarget poisons region on verification failure", "[daemon][materialize][into_target]") {
  ValidationFixture fix(/*external_target_verification_enabled=*/true);

  const auto storage_root = fix.storage_root;
  const auto artifact_rel =
      std::filesystem::path("clusters") / fix.global_store_client->cluster_id / "objects" / "artifact_verify_fail";
  const auto artifact_dir = storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);

  const std::string payload = "ABCDEFGH";
  REQUIRE(write_file(artifact_dir / "tensor.data_0", payload));

  const std::string canonical_index_json = R"({"a":[0,8,[8],[1],"torch.uint8",0]})";
  REQUIRE(write_file(artifact_dir / "tensor_index.json", canonical_index_json));
  fix.global_store_client->canonical_index_json = canonical_index_json;

  auto index_mh_or = tensorcast::common::compute_index_multihash(std::optional<std::string>(canonical_index_json), "");
  REQUIRE(index_mh_or.ok());
  const std::string artifact_id = std::string("mi2:") + *index_mh_or + ":bad";
  const std::string descriptor_json = std::string("{\"artifact_id\":\"") + artifact_id + "\",\"index_multihash\":\"" +
      *index_mh_or + "\",\"data_multihash\":\"bad\",\"schema_version\":\"v3\",\"encoding\":\"json\",\"total_size\":8}";
  REQUIRE(write_file(artifact_dir / "artifact_descriptor.json", descriptor_json));
  register_disk_location(*fix.global_store_client, artifact_id, artifact_rel);

  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  REQUIRE(device_count > 0);
  tensorcast::cuda::DeviceGuard guard(0);
  REQUIRE(guard.status().ok());

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = 8, .fill_byte = -1},
  };
  auto child_or = tensorcast::daemon::testing::CudaIpcChild::Spawn(*helper_path_or, 0, buffers);
  INFO("cuda_ipc_helper spawn status: " << child_or.status());
  REQUIRE(child_or.ok());
  auto child = std::move(*child_or);
  REQUIRE(child.handle_bytes().size() == 1);
  const std::string& handle_bytes = child.handle_bytes().front();

  const int owner_pid = child.pid();
  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.device_id = 0;
  params.owner_pid = owner_pid;
  params.size_bytes = 8;
  params.ttl_ms = 10'000;
  params.handle_bytes = handle_bytes;
  auto region_or = fix.regions.register_region(params);
  REQUIRE(region_or.ok());

  auto device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  if (device_key.uuid.empty()) {
    tensorcast::store::DeviceRegistry::instance().register_gpu(0, "fake-uuid-0");
    device_key = tensorcast::store::DeviceRegistry::instance().gpu_key(0);
  }

  MaterializeIntoTargetRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_device_uuid(device_key.uuid);
  req.set_pid(owner_pid);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id(region_or->region_id);
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(8);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  if (status.ok()) {
    REQUIRE_FALSE(fix.regions.is_poisoned(region_or->region_id));
  } else {
    REQUIRE(status.error_code() == grpc::StatusCode::DATA_LOSS);
    REQUIRE(fix.regions.is_poisoned(region_or->region_id));
  }

  REQUIRE(fix.regions.unregister_region(region_or->region_id, owner_pid, /*force=*/true).ok());
  REQUIRE(child.Shutdown().ok());
}
