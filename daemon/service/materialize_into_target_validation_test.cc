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

absl::StatusOr<std::string> compute_view_id(
    const tensorcast::common::v1::ViewSpec& view_spec,
    std::string_view canonical_index_json) {
  auto index_mh_or =
      tensorcast::common::compute_index_multihash(std::optional<std::string>(std::string(canonical_index_json)), "");
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  const std::string spec_bytes = view_spec.SerializeAsString();
  std::vector<uint8_t> buffer;
  buffer.reserve(spec_bytes.size() + index_mh_or->size());
  buffer.insert(buffer.end(), spec_bytes.begin(), spec_bytes.end());
  buffer.insert(buffer.end(), index_mh_or->begin(), index_mh_or->end());
  const std::vector<uint8_t> digest =
      tensorcast::common::sha256_digest_bytes(absl::Span<const uint8_t>(buffer.data(), buffer.size()));
  return tensorcast::common::multibase_multihash_sha256(digest);
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

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47016;
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
  tensorcast::daemon::SessionsService sessions_svc;
  tensorcast::daemon::DeviceResolver devices;
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
        sessions_svc(session_mgr, verif_tracker, &scheduler, /*lifecycle=*/nullptr, absl::Seconds(60)),
        devices(tensorcast::store::DeviceRegistry::instance()),
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
                .shutdown_signal = shutdown_signal,
                .async_runtime = async_runtime,
                .identity = identity,
                .global_store_client = global_store_client,
                .lifecycle = nullptr,
                .external_target_verification_enabled = external_target_verification_enabled,
                .storage_path = storage_root,
            })) {
    global_store_client->canonical_index_json = make_canonical_index_json();
    engine->set_global_store_client_for_testing(global_store_client);
  }
};

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

TEST_CASE("MaterializeIntoTarget rejects empty disk_fallback path", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.mutable_disk_fallback();
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("MaterializeIntoTarget accepts subset selection", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.add_tensor_names("a");

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
  REQUIRE(status.error_code() == grpc::StatusCode::NOT_FOUND);
}

TEST_CASE("MaterializeIntoTarget accepts ordered full selection", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.add_tensor_names("b");
  req.add_tensor_names("a");

  auto* layout = req.mutable_target_layout();
  layout->set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout->set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_VIEW);
  layout->set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout->add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(0);
  storage->set_storage_length(8);
  storage->set_vram_region_id("region-0");
  storage->set_mapping_base_offset(0);

  auto* offset_b = layout->add_offsets();
  offset_b->set_name("b");
  offset_b->set_storage_id("storage-0");
  offset_b->set_storage_offset(4);
  offset_b->set_logical_length(4);

  auto* offset_a = layout->add_offsets();
  offset_a->set_name("a");
  offset_a->set_storage_id("storage-0");
  offset_a->set_storage_offset(0);
  offset_a->set_logical_length(4);

  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::NOT_FOUND);
}

TEST_CASE("MaterializeIntoTarget accepts view spec subset", "[daemon][materialize][into_target]") {
  ValidationFixture fix;
  MaterializeIntoTargetRequest req;
  req.set_artifact_id("mi2:dummy:dummy");
  req.set_device_uuid("gpu-0");
  req.set_pid(123);
  req.add_tensor_names("a");

  tensorcast::common::v1::ViewSpec view_spec;
  auto* ops = (*view_spec.mutable_tensors())["a"].add_ops();
  ops->mutable_narrow()->set_dim(0);
  ops->mutable_narrow()->set_start(0);
  ops->mutable_narrow()->set_length(2);
  req.mutable_view()->CopyFrom(view_spec);

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
  req.set_artifact_id("mi2:dummy:dummy");
  req.mutable_target_layout();
  MaterializeIntoTargetResponse resp;
  auto status = run_request(fix.controller, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_CASE("MaterializeIntoTarget writes into multiple regions", "[daemon][materialize][into_target]") {
  ValidationFixture fix;

  const auto storage_root = fix.storage_root;
  const auto artifact_dir = storage_root / "artifact_multi_region";
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
  req.set_artifact_id("artifact_multi_region");
  req.set_device_uuid(device_key.uuid);
  req.set_pid(owner_pid);
  req.set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);
  req.mutable_disk_fallback()->set_disk_path(artifact_dir.string());

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
  const auto artifact_dir = storage_root / "artifact_verify_fail";
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
  req.set_artifact_id(artifact_id);
  req.set_device_uuid(device_key.uuid);
  req.set_pid(owner_pid);
  req.set_preference(tensorcast::daemon::v2::SOURCE_PREFERENCE_PREFER_DISK);
  req.mutable_disk_fallback()->set_disk_path(artifact_dir.string());

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
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::DATA_LOSS);
  REQUIRE(fix.regions.is_poisoned(region_or->region_id));

  REQUIRE(fix.regions.unregister_region(region_or->region_id, owner_pid, /*force=*/true).ok());
  REQUIRE(child.Shutdown().ok());
}
