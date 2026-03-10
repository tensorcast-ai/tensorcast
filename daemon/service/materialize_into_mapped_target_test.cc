// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <unistd.h>
#include <array>
#include <chrono>
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

#include "absl/time/time.h"
#include "core/common/capability_token.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/device_registry.h"
#include "core/store/store_engine.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/service/rpc_context.h"
#include "daemon/state/background_scheduler.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_bridge.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/sessions_service.h"
#include "daemon/state/shutdown_signal.h"
#include "daemon/state/verification_tracker.h"
#include "daemon/testing/cuda_ipc_spawn_helper.h"
#include "grpcpp/server_context.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

using tensorcast::daemon::MaterializationController;
using tensorcast::daemon::RpcContext;
using tensorcast::daemon::v2::CopyPlan;
using tensorcast::daemon::v2::CopyPlanEntry;
using tensorcast::daemon::v2::MaterializeIntoMappedTargetRequest;
using tensorcast::daemon::v2::MaterializeIntoTargetResponse;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_materialize_into_mapped_target_test";
}

std::filesystem::path ensure_dir(std::filesystem::path path) {
  if (!path.empty()) {
    std::filesystem::create_directories(path);
  }
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

bool write_file(const std::filesystem::path& path, std::string_view payload) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }
  out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  return out.good();
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

struct MappedFixture {
  std::shared_ptr<tensorcast::store::StoreEngine> engine;
  std::shared_ptr<tensorcast::store::testing::RecordingGlobalStoreClient> global_store_client;
  tensorcast::daemon::RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions;
  tensorcast::daemon::LipManager lip_mgr;
  tensorcast::daemon::LipBridge lip_bridge;
  tensorcast::daemon::ReplicaSessionManager session_mgr;
  tensorcast::daemon::VerificationTracker verif_tracker;
  tensorcast::daemon::BackgroundScheduler scheduler;
  tensorcast::daemon::SessionsService sessions_svc;
  tensorcast::daemon::DeviceResolver devices;
  tensorcast::daemon::ArtifactSourceRegistry disk_imports;
  tensorcast::daemon::ShutdownSignal shutdown_signal;
  tensorcast::common::AsyncRuntime async_runtime;
  tensorcast::daemon::WorkerIdentityStore identity;
  tensorcast::common::CapabilityTokenManager capability_tokens;
  std::filesystem::path storage_root;
  MaterializationController controller;

  MappedFixture()
      : engine(std::make_shared<tensorcast::store::StoreEngine>(make_opts())),
        global_store_client(std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>()),
        regions(tensorcast::daemon::IpcRegionRegistry::Options{}),
        lip_mgr(engine, &regions),
        lip_bridge(lip_mgr),
        session_mgr(std::chrono::seconds(60)),
        verif_tracker(),
        scheduler(),
        sessions_svc(session_mgr, verif_tracker, &scheduler, /*lifecycle=*/nullptr, absl::Seconds(60)),
        devices(tensorcast::store::DeviceRegistry::instance()),
        capability_tokens(
            tensorcast::common::CapabilityTokenConfig{
                .active = tensorcast::common::CapabilityTokenKey{.version = 1, .secret = "mapped-test-secret"}}),
        storage_root(ensure_dir(test_tmpdir())),
        controller(MaterializationController(
            MaterializationController::Dep{
                .engine = *engine,
                .refs = refs,
                .sessions = sessions_svc,
                .lip = lip_bridge,
                .lip_manager = lip_mgr,
                .devices = devices,
                .regions = regions,
                .disk_imports = disk_imports,
                .shutdown_signal = shutdown_signal,
                .async_runtime = async_runtime,
                .identity = identity,
                .global_store_client = global_store_client,
                .lifecycle = nullptr,
                .capability_tokens = &capability_tokens,
                .external_target_verification_enabled = false,
                .storage_path = storage_root,
            })) {
    engine->set_global_store_client_for_testing(global_store_client);
    identity.set_daemon_id("daemon-mapped-test");
  }
};

grpc::Status run_request(
    MaterializationController& controller,
    const MaterializeIntoMappedTargetRequest& req,
    MaterializeIntoTargetResponse& resp) {
  grpc::ServerContext ctx;
  RpcContext rctx{"MaterializeIntoMappedTargetTest", ctx, /*allow_high_card_attrs=*/true};
  return controller.materialize_into_mapped_target(rctx, req, resp);
}

CopyPlan make_split_plan(std::string_view src, std::string_view dst_a, std::string_view dst_b) {
  CopyPlan plan;
  plan.set_version(1);

  auto* entry0 = plan.add_entries();
  entry0->set_ckpt_name(std::string(src));
  entry0->set_dst_name(std::string(dst_a));
  entry0->mutable_ckpt_range()->set_dim(0);
  entry0->mutable_ckpt_range()->set_start(0);
  entry0->mutable_ckpt_range()->set_end(4);
  entry0->mutable_dst_range()->set_dim(0);
  entry0->mutable_dst_range()->set_start(0);
  entry0->mutable_dst_range()->set_end(4);

  auto* entry1 = plan.add_entries();
  entry1->set_ckpt_name(std::string(src));
  entry1->set_dst_name(std::string(dst_b));
  entry1->mutable_ckpt_range()->set_dim(0);
  entry1->mutable_ckpt_range()->set_start(4);
  entry1->mutable_ckpt_range()->set_end(8);
  entry1->mutable_dst_range()->set_dim(0);
  entry1->mutable_dst_range()->set_start(0);
  entry1->mutable_dst_range()->set_end(4);

  return plan;
}

} // namespace

TEST_CASE("MaterializeIntoMappedTarget maps slices into target regions", "[daemon][materialize][mapped_target]") {
  MappedFixture fix;

  const auto artifact_rel =
      std::filesystem::path("clusters") / fix.global_store_client->cluster_id / "objects" / "artifact_mapped";
  const auto artifact_dir = fix.storage_root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const std::string payload = "ABCDEFGH";
  REQUIRE(write_file(artifact_dir / "tensor.data_0", payload));

  const std::string canonical_index_json = R"({"src":[0,8,[8],[1],"torch.uint8",0]})";
  REQUIRE(write_file(artifact_dir / "tensor_index.json", canonical_index_json));
  fix.global_store_client->canonical_index_json = canonical_index_json;
  register_disk_location(*fix.global_store_client, "artifact_mapped", artifact_rel);

  int device_count = 0;
  REQUIRE(tensorcast::cuda::get_device_count(&device_count).ok());
  REQUIRE(device_count > 0);
  tensorcast::cuda::DeviceGuard guard(0);
  REQUIRE(guard.status().ok());

  const auto helper_path_or = tensorcast::daemon::testing::resolve_cuda_ipc_helper_path();
  REQUIRE(helper_path_or.ok());
  std::vector<tensorcast::daemon::testing::CudaIpcBufferSpec> buffers = {
      {.size_bytes = 4, .fill_byte = -1},
      {.size_bytes = 4, .fill_byte = -1},
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
  params.size_bytes = 4;
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

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped");
  req.set_device_uuid(device_key.uuid);
  req.set_pid(owner_pid);
  req.set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id(region0_or->region_id);
  storage0->set_mapping_base_offset(0);

  auto* storage1 = layout->add_storages();
  storage1->set_storage_id("storage-1");
  storage1->set_device_id(0);
  storage1->set_storage_length(4);
  storage1->set_vram_region_id(region1_or->region_id);
  storage1->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);

  auto* offset1 = layout->add_offsets();
  offset1->set_name("b");
  offset1->set_storage_id("storage-1");
  offset1->set_storage_offset(0);
  offset1->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(4);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);

  auto* spec1 = req.add_dst_tensors();
  spec1->set_name("b");
  spec1->add_shape(4);
  spec1->add_stride(1);
  spec1->set_dtype("torch.uint8");
  spec1->set_storage_offset(0);
  spec1->set_logical_length(4);

  req.mutable_copy_plan()->CopyFrom(make_split_plan("src", "a", "b"));

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE(status.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);
  REQUIRE(!resp.target_write_token().empty());

  std::array<char, 4> out0{};
  std::array<char, 4> out1{};
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

  REQUIRE(std::string(out0.begin(), out0.end()) == "ABCD");
  REQUIRE(std::string(out1.begin(), out1.end()) == "EFGH");

  REQUIRE(fix.regions.unregister_region(region0_or->region_id, owner_pid, /*force=*/true).ok());
  REQUIRE(fix.regions.unregister_region(region1_or->region_id, owner_pid, /*force=*/true).ok());
  REQUIRE(child.Shutdown().ok());
}

TEST_CASE("MaterializeIntoMappedTarget rejects dst coverage gaps", "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,8,[8],[1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_gap");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* storage1 = layout->add_storages();
  storage1->set_storage_id("storage-1");
  storage1->set_device_id(0);
  storage1->set_storage_length(4);
  storage1->set_vram_region_id("region-1");
  storage1->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);

  auto* offset1 = layout->add_offsets();
  offset1->set_name("b");
  offset1->set_storage_id("storage-1");
  offset1->set_storage_offset(0);
  offset1->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(4);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);

  auto* spec1 = req.add_dst_tensors();
  spec1->set_name("b");
  spec1->add_shape(4);
  spec1->add_stride(1);
  spec1->set_dtype("torch.uint8");
  spec1->set_storage_offset(0);
  spec1->set_logical_length(4);

  CopyPlan plan;
  plan.set_version(1);
  auto* ok_entry = plan.add_entries();
  ok_entry->set_ckpt_name("src");
  ok_entry->set_dst_name("b");
  ok_entry->mutable_ckpt_range()->set_dim(0);
  ok_entry->mutable_ckpt_range()->set_start(4);
  ok_entry->mutable_ckpt_range()->set_end(8);
  ok_entry->mutable_dst_range()->set_dim(0);
  ok_entry->mutable_dst_range()->set_start(0);
  ok_entry->mutable_dst_range()->set_end(4);

  auto* gap_entry = plan.add_entries();
  gap_entry->set_ckpt_name("src");
  gap_entry->set_dst_name("a");
  gap_entry->mutable_ckpt_range()->set_dim(0);
  gap_entry->mutable_ckpt_range()->set_start(0);
  gap_entry->mutable_ckpt_range()->set_end(3);
  gap_entry->mutable_dst_range()->set_dim(0);
  gap_entry->mutable_dst_range()->set_start(0);
  gap_entry->mutable_dst_range()->set_end(3);

  req.mutable_copy_plan()->CopyFrom(plan);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects overlapping dst ranges",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,4,[4],[1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_overlap");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(4);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);

  CopyPlan plan;
  plan.set_version(1);

  auto* entry0 = plan.add_entries();
  entry0->set_ckpt_name("src");
  entry0->set_dst_name("a");
  entry0->mutable_ckpt_range()->set_dim(0);
  entry0->mutable_ckpt_range()->set_start(0);
  entry0->mutable_ckpt_range()->set_end(3);
  entry0->mutable_dst_range()->set_dim(0);
  entry0->mutable_dst_range()->set_start(0);
  entry0->mutable_dst_range()->set_end(3);

  auto* entry1 = plan.add_entries();
  entry1->set_ckpt_name("src");
  entry1->set_dst_name("a");
  entry1->mutable_ckpt_range()->set_dim(0);
  entry1->mutable_ckpt_range()->set_start(2);
  entry1->mutable_ckpt_range()->set_end(4);
  entry1->mutable_dst_range()->set_dim(0);
  entry1->mutable_dst_range()->set_start(2);
  entry1->mutable_dst_range()->set_end(4);

  req.mutable_copy_plan()->CopyFrom(plan);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("overlapping dst ranges") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects mixed slice dimensions for the same dst tensor",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,4,[2,2],[2,1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_mixed_dim");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(2);
  spec0->add_shape(2);
  spec0->add_stride(2);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);

  CopyPlan plan;
  plan.set_version(1);

  auto* entry0 = plan.add_entries();
  entry0->set_ckpt_name("src");
  entry0->set_dst_name("a");
  entry0->mutable_ckpt_range()->set_dim(0);
  entry0->mutable_ckpt_range()->set_start(0);
  entry0->mutable_ckpt_range()->set_end(1);
  entry0->mutable_dst_range()->set_dim(0);
  entry0->mutable_dst_range()->set_start(0);
  entry0->mutable_dst_range()->set_end(1);

  auto* entry1 = plan.add_entries();
  entry1->set_ckpt_name("src");
  entry1->set_dst_name("a");
  entry1->mutable_ckpt_range()->set_dim(1);
  entry1->mutable_ckpt_range()->set_start(0);
  entry1->mutable_ckpt_range()->set_end(1);
  entry1->mutable_dst_range()->set_dim(1);
  entry1->mutable_dst_range()->set_start(0);
  entry1->mutable_dst_range()->set_end(1);

  req.mutable_copy_plan()->CopyFrom(plan);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("mixes slice dims") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects unsupported copy-plan version",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_bad_version");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);
  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(1);
  storage->set_vram_region_id("region-0");
  storage->set_mapping_base_offset(0);

  auto* offset = layout->add_offsets();
  offset->set_name("a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(1);

  auto* spec = req.add_dst_tensors();
  spec->set_name("a");
  spec->add_shape(1);
  spec->add_stride(1);
  spec->set_dtype("torch.uint8");
  spec->set_storage_offset(0);
  spec->set_logical_length(1);

  req.mutable_copy_plan()->set_version(2);
  auto* entry = req.mutable_copy_plan()->add_entries();
  entry->set_ckpt_name("src");
  entry->set_dst_name("a");

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("unsupported copy_plan version") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects non-contiguous dst tensor specs",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,4,[2,2],[2,1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_non_contig");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(2);
  spec0->add_shape(2);
  spec0->add_stride(1);
  spec0->add_stride(2); // non-contiguous
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);

  auto* entry = req.mutable_copy_plan()->add_entries();
  req.mutable_copy_plan()->set_version(1);
  entry->set_ckpt_name("src");
  entry->set_dst_name("a");
  entry->mutable_ckpt_range()->set_dim(0);
  entry->mutable_ckpt_range()->set_start(0);
  entry->mutable_ckpt_range()->set_end(2);
  entry->mutable_dst_range()->set_dim(0);
  entry->mutable_dst_range()->set_start(0);
  entry->mutable_dst_range()->set_end(2);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("must be contiguous") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects dst storage offset mismatch",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,4,[4],[1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_offset_mismatch");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(5);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(1); // bytes
  offset0->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(4);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0); // elements => mismatch with offset=1 byte
  spec0->set_logical_length(4);

  auto* entry = req.mutable_copy_plan()->add_entries();
  req.mutable_copy_plan()->set_version(1);
  entry->set_ckpt_name("src");
  entry->set_dst_name("a");
  entry->mutable_ckpt_range()->set_dim(0);
  entry->mutable_ckpt_range()->set_start(0);
  entry->mutable_ckpt_range()->set_end(4);
  entry->mutable_dst_range()->set_dim(0);
  entry->mutable_dst_range()->set_start(0);
  entry->mutable_dst_range()->set_end(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("storage_offset mismatch") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects copy-plan missing dst coverage",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,8,[8],[1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_missing_dst_coverage");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);
  auto* storage1 = layout->add_storages();
  storage1->set_storage_id("storage-1");
  storage1->set_device_id(0);
  storage1->set_storage_length(4);
  storage1->set_vram_region_id("region-1");
  storage1->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);
  auto* offset1 = layout->add_offsets();
  offset1->set_name("b");
  offset1->set_storage_id("storage-1");
  offset1->set_storage_offset(0);
  offset1->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(4);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);
  auto* spec1 = req.add_dst_tensors();
  spec1->set_name("b");
  spec1->add_shape(4);
  spec1->add_stride(1);
  spec1->set_dtype("torch.uint8");
  spec1->set_storage_offset(0);
  spec1->set_logical_length(4);

  CopyPlan plan;
  plan.set_version(1);
  auto* entry = plan.add_entries();
  entry->set_ckpt_name("src");
  entry->set_dst_name("a"); // only covers a, misses b
  entry->mutable_ckpt_range()->set_dim(0);
  entry->mutable_ckpt_range()->set_start(0);
  entry->mutable_ckpt_range()->set_end(4);
  entry->mutable_dst_range()->set_dim(0);
  entry->mutable_dst_range()->set_start(0);
  entry->mutable_dst_range()->set_end(4);
  req.mutable_copy_plan()->CopyFrom(plan);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("copy_plan must cover every dst tensor") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects transpose view ops",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,4,[2,2],[2,1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_view_transpose");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(2);
  spec0->add_shape(2);
  spec0->add_stride(2);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);

  CopyPlan plan;
  plan.set_version(1);
  auto* entry = plan.add_entries();
  entry->set_ckpt_name("src");
  entry->set_dst_name("a");
  entry->mutable_ckpt_range()->set_dim(0);
  entry->mutable_ckpt_range()->set_start(0);
  entry->mutable_ckpt_range()->set_end(2);
  entry->mutable_dst_range()->set_dim(0);
  entry->mutable_dst_range()->set_start(0);
  entry->mutable_dst_range()->set_end(2);
  req.mutable_copy_plan()->CopyFrom(plan);

  auto* view = req.mutable_selection()->mutable_view_spec();
  auto& ops = (*view->mutable_tensors())["src"];
  auto* transpose = ops.add_ops()->mutable_transpose();
  transpose->set_dim0(0);
  transpose->set_dim1(1);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("transpose views") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget rejects multiple narrow ops per tensor",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,8,[8],[1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_multiple_narrow");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(8);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(8);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(8);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(8);

  CopyPlan plan;
  plan.set_version(1);
  auto* entry = plan.add_entries();
  entry->set_ckpt_name("src");
  entry->set_dst_name("a");
  entry->mutable_ckpt_range()->set_dim(0);
  entry->mutable_ckpt_range()->set_start(0);
  entry->mutable_ckpt_range()->set_end(8);
  entry->mutable_dst_range()->set_dim(0);
  entry->mutable_dst_range()->set_start(0);
  entry->mutable_dst_range()->set_end(8);
  req.mutable_copy_plan()->CopyFrom(plan);

  auto* view = req.mutable_selection()->mutable_view_spec();
  auto& ops = (*view->mutable_tensors())["src"];
  auto* narrow0 = ops.add_ops()->mutable_narrow();
  narrow0->set_dim(0);
  narrow0->set_start(0);
  narrow0->set_length(4);
  auto* narrow1 = ops.add_ops()->mutable_narrow();
  narrow1->set_dim(0);
  narrow1->set_start(4);
  narrow1->set_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  REQUIRE(status.error_message().find("one narrow per tensor") != std::string::npos);
}

TEST_CASE(
    "MaterializeIntoMappedTarget accepts opaque view_id without metadata lookup hard failure",
    "[daemon][materialize][mapped_target][validation]") {
  MappedFixture fix;

  const std::string canonical_index_json = R"({"src":[0,4,[4],[1],"torch.uint8",0]})";
  fix.global_store_client->canonical_index_json = canonical_index_json;

  MaterializeIntoMappedTargetRequest req;
  req.mutable_selection()->set_artifact_id("artifact_mapped_unknown_view");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.mutable_selection()->set_view_id("missing-view-id");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage0 = layout->add_storages();
  storage0->set_storage_id("storage-0");
  storage0->set_device_id(0);
  storage0->set_storage_length(4);
  storage0->set_vram_region_id("region-0");
  storage0->set_mapping_base_offset(0);

  auto* offset0 = layout->add_offsets();
  offset0->set_name("a");
  offset0->set_storage_id("storage-0");
  offset0->set_storage_offset(0);
  offset0->set_logical_length(4);

  auto* spec0 = req.add_dst_tensors();
  spec0->set_name("a");
  spec0->add_shape(4);
  spec0->add_stride(1);
  spec0->set_dtype("torch.uint8");
  spec0->set_storage_offset(0);
  spec0->set_logical_length(4);

  CopyPlan plan;
  plan.set_version(1);
  auto* entry = plan.add_entries();
  entry->set_ckpt_name("src");
  entry->set_dst_name("a");
  entry->mutable_ckpt_range()->set_dim(0);
  entry->mutable_ckpt_range()->set_start(0);
  entry->mutable_ckpt_range()->set_end(4);
  entry->mutable_dst_range()->set_dim(0);
  entry->mutable_dst_range()->set_start(0);
  entry->mutable_dst_range()->set_end(4);
  req.mutable_copy_plan()->CopyFrom(plan);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() != grpc::StatusCode::UNIMPLEMENTED);
}
