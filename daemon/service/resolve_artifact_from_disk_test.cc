// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_controller.h"

#include <algorithm>

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/store_engine.h"
#include "core/testing/common.h"
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
#include "grpcpp/server_context.h"
#include "nlohmann/json.hpp"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

using tensorcast::daemon::MaterializationController;
using tensorcast::daemon::v2::MaterializeReplicaRequest;
using tensorcast::daemon::v2::MaterializeReplicaResponse;
using tensorcast::daemon::v2::ResolveArtifactFromDiskRequest;
using tensorcast::daemon::v2::ResolveArtifactFromDiskResponse;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_resolve_disk_test";
}

std::filesystem::path ensure_dir(std::filesystem::path path) {
  if (!path.empty()) {
    std::filesystem::create_directories(path);
  }
  return path;
}

void create_safetensors_file(const std::filesystem::path& path, const std::string& tensor_name, uint64_t size_bytes) {
  const std::string header_json =
      nlohmann::json({{tensor_name, {{"dtype", "U8"}, {"shape", {size_bytes}}, {"data_offsets", {0, size_bytes}}}}})
          .dump();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  REQUIRE(out.is_open());
  const uint64_t header_size = header_json.size();
  for (int i = 0; i < 8; ++i) {
    const unsigned char byte = static_cast<unsigned char>((header_size >> (8 * i)) & 0xFF);
    out.put(static_cast<char>(byte));
  }
  out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  std::vector<char> payload(static_cast<size_t>(size_bytes), '\0');
  if (!payload.empty()) {
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
  }
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

uint64_t generation_from_bytes(const std::string& canonical_index_json) {
  const auto digest = tensorcast::common::sha256_digest_bytes(
      absl::Span<const uint8_t>(
          reinterpret_cast<const uint8_t*>(canonical_index_json.data()), canonical_index_json.size()));
  uint64_t value = 0;
  const size_t limit = std::min<size_t>(8, digest.size());
  for (size_t i = 0; i < limit; ++i) {
    value = (value << 8) | static_cast<uint64_t>(digest[i]);
  }
  return value;
}

struct ResolveFixture {
  std::shared_ptr<tensorcast::store::StoreEngine> engine;
  tensorcast::daemon::RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry regions;
  tensorcast::daemon::LipManager lip_mgr;
  tensorcast::daemon::LipBridge lip_bridge;
  tensorcast::daemon::ReplicaSessionManager session_mgr;
  tensorcast::daemon::VerificationTracker verif_tracker;
  tensorcast::daemon::BackgroundScheduler scheduler;
  tensorcast::daemon::SessionsService sessions_svc;
  tensorcast::daemon::DeviceResolver devices;
  tensorcast::daemon::LocalDiskImportCatalog disk_imports;
  tensorcast::daemon::ShutdownSignal shutdown_signal;
  tensorcast::common::AsyncRuntime async_runtime;
  tensorcast::daemon::WorkerIdentityStore identity;
  MaterializationController controller;

  explicit ResolveFixture(std::filesystem::path storage_root = test_tmpdir())
      : engine(std::make_shared<tensorcast::store::StoreEngine>(make_opts())),
        regions(tensorcast::daemon::IpcRegionRegistry::Options{}),
        lip_mgr(engine, &regions),
        lip_bridge(lip_mgr),
        session_mgr(std::chrono::seconds(60)),
        verif_tracker(),
        scheduler(),
        sessions_svc(session_mgr, verif_tracker, &scheduler, /*lifecycle=*/nullptr, absl::Seconds(60)),
        devices(tensorcast::store::DeviceRegistry::instance()),
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
                .global_store_client = nullptr,
                .lifecycle = nullptr,
                .storage_path = ensure_dir(std::move(storage_root)),
            })) {}
};

} // namespace

TEST_CASE(
    "ResolveArtifactFromDisk returns canonical index bytes and generation for disk artifacts",
    "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_ok";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  ResolveFixture fix;
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  req.set_verify_checksums(true);
  ResolveArtifactFromDiskResponse resp;
  auto status = fix.controller.resolve_artifact_from_disk(rctx, req, resp);

  REQUIRE(status.ok());
  REQUIRE_FALSE(resp.artifact_id().empty());
  REQUIRE(resp.artifact_id().starts_with("mi2:"));
  REQUIRE(resp.generation() == generation_from_bytes(resp.canonical_index_bytes()));
  REQUIRE_FALSE(resp.canonical_index_bytes().empty());
  const auto imported = fix.disk_imports.lookup_import(resp.artifact_id());
  REQUIRE(imported.has_value());
  REQUIRE(imported->normalized_disk_path == resp.disk_path());
}

TEST_CASE("ResolveArtifactFromDisk accepts absolute imports and rejects missing paths", "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_denied";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  ResolveFixture denied_fix(artifact_dir.parent_path() / "unrelated_root");
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext denied_rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  ResolveArtifactFromDiskResponse resp;
  auto status = denied_fix.controller.resolve_artifact_from_disk(denied_rctx, req, resp);
  REQUIRE(status.ok());
  REQUIRE(resp.artifact_id().starts_with("mi2:"));

  ResolveFixture ok_fix(artifact_dir.parent_path() / "other_root");
  tensorcast::daemon::RpcContext ok_rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  req.set_disk_path((artifact_dir.parent_path() / "missing_artifact").string());
  ResolveArtifactFromDiskResponse missing_resp;
  auto missing_status = ok_fix.controller.resolve_artifact_from_disk(ok_rctx, req, missing_resp);
  REQUIRE_FALSE(missing_status.ok());
  REQUIRE(missing_status.error_code() == grpc::StatusCode::NOT_FOUND);
}

TEST_CASE("ResolveArtifactFromDisk allows absolute paths when storage root is empty", "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_root_empty";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  ResolveFixture fix(std::filesystem::path{});
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  ResolveArtifactFromDiskResponse resp;
  auto status = fix.controller.resolve_artifact_from_disk(rctx, req, resp);

  REQUIRE(status.ok());
  REQUIRE_FALSE(resp.artifact_id().empty());
  REQUIRE_FALSE(resp.canonical_index_bytes().empty());
}

TEST_CASE(
    "ResolveArtifactFromDisk resolves safetensors directories without tensor_index.json",
    "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_safetensors";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  create_safetensors_file(artifact_dir / "part0.safetensors", "weights", /*size_bytes=*/32);

  ResolveFixture fix;
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  req.set_verify_checksums(true);
  ResolveArtifactFromDiskResponse resp;
  auto status = fix.controller.resolve_artifact_from_disk(rctx, req, resp);

  REQUIRE(status.ok());
  REQUIRE(resp.artifact_id().starts_with("mi2:"));
  REQUIRE_FALSE(resp.canonical_index_bytes().empty());
  nlohmann::json parsed = nlohmann::json::parse(resp.canonical_index_bytes());
  REQUIRE(parsed.contains("weights"));
}

TEST_CASE("ResolveArtifactFromDisk fails checksum validation when descriptor mismatches", "[daemon][disk][resolve]") {
  const auto artifact_dir = test_tmpdir() / "artifact_checksum";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 32));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  // Corrupt index_multihash to trigger checksum failure.
  const auto descriptor_path = artifact_dir / "artifact_descriptor.json";
  nlohmann::json descriptor = nlohmann::json::parse(std::ifstream(descriptor_path));
  descriptor["index_multihash"] = "mh:corrupt";
  {
    std::ofstream out(descriptor_path, std::ios::trunc);
    REQUIRE(out.is_open());
    out << descriptor.dump(2);
  }

  ResolveFixture fix;
  grpc::ServerContext ctx;
  tensorcast::daemon::RpcContext rctx{"ResolveArtifactFromDiskTest", ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest req;
  req.set_disk_path(artifact_dir.string());
  req.set_verify_checksums(true);
  ResolveArtifactFromDiskResponse resp;
  auto status = fix.controller.resolve_artifact_from_disk(rctx, req, resp);

  REQUIRE_FALSE(status.ok());
  REQUIRE(status.error_code() == grpc::StatusCode::FAILED_PRECONDITION);
}

TEST_CASE("Standalone disk import materializes without Global Store", "[daemon][disk][resolve][standalone]") {
  const auto artifact_dir = test_tmpdir() / "artifact_standalone";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  ResolveFixture fix;
  grpc::ServerContext resolve_ctx;
  tensorcast::daemon::RpcContext resolve_rctx{
      "ResolveArtifactFromDiskTest", resolve_ctx, /*allow_high_card_attrs=*/true};
  ResolveArtifactFromDiskRequest resolve_req;
  resolve_req.set_disk_path(artifact_dir.string());
  resolve_req.set_verify_checksums(true);
  ResolveArtifactFromDiskResponse resolve_resp;
  auto resolve_status = fix.controller.resolve_artifact_from_disk(resolve_rctx, resolve_req, resolve_resp);
  REQUIRE(resolve_status.ok());
  REQUIRE(resolve_resp.artifact_id().starts_with("mi2:"));

  grpc::ServerContext materialize_ctx;
  tensorcast::daemon::RpcContext materialize_rctx{
      "MaterializeReplicaStandaloneTest", materialize_ctx, /*allow_high_card_attrs=*/true};
  MaterializeReplicaRequest materialize_req;
  materialize_req.set_artifact_id(resolve_resp.artifact_id());
  materialize_req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_GPU);
  materialize_req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);

  MaterializeReplicaResponse materialize_resp;
  auto materialize_status = fix.controller.materialize_replica(materialize_rctx, materialize_req, materialize_resp);
  REQUIRE(materialize_status.ok());
  REQUIRE(materialize_resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(materialize_resp.source() == tensorcast::daemon::v2::MATERIALIZATION_SOURCE_DISK);
}
