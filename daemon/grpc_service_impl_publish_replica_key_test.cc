// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/store/store_engine.h"
#include "grpcpp/server_context.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

namespace {

std::filesystem::path test_root() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env) / "tensorcast_daemon_publish_replica_key_test";
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_publish_replica_key_test";
}

tensorcast::store::StoreEngineOptions make_engine_opts(const std::filesystem::path& storage_root) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  std::filesystem::create_directories(storage_root);
  opts.p2p_port = 0; // let the OS pick an ephemeral port during tests
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear(); // explicit offline mode (test supplies a stub client)
  return opts;
}

class KeyMappingGlobalStoreClient final : public tensorcast::store::components::IGlobalStoreClient {
 public:
  absl::Status initialize() override {
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> register_worker(
      std::string_view,
      std::string_view,
      uint32_t,
      uint32_t,
      uint64_t,
      uint64_t,
      bool,
      std::string_view) override {
    return absl::UnimplementedError("register_worker not needed for key-mapping tests");
  }

  absl::Status send_heartbeat(std::string_view, uint64_t, bool) override {
    return absl::UnimplementedError("send_heartbeat not needed for key-mapping tests");
  }

  absl::StatusOr<tensorcast::global_store::v1::WorkerHeartbeatResponse> send_heartbeat_enhanced(
      std::string_view,
      uint64_t,
      bool,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      int64_t,
      tensorcast::global_store::v1::ConnectionStatus) override {
    return absl::UnimplementedError("send_heartbeat_enhanced not needed for key-mapping tests");
  }

  absl::Status unregister_worker(std::string_view, bool) override {
    return absl::UnimplementedError("unregister_worker not needed for key-mapping tests");
  }

  absl::StatusOr<std::string> register_replica(
      std::string_view,
      std::string_view,
      const tensorcast::store::DeviceKey&,
      tensorcast::common::memory::MemoryLocation,
      uint64_t,
      uint32_t) override {
    return absl::UnimplementedError("register_replica not needed for key-mapping tests");
  }

  absl::Status record_variant_residency(std::string_view, std::string_view, uint64_t, std::optional<std::string_view>)
      override {
    return absl::UnimplementedError("record_variant_residency not needed for key-mapping tests");
  }

  absl::StatusOr<std::string> register_memory_replica(
      std::string_view,
      std::string_view,
      const tensorcast::store::DeviceKey&,
      uint64_t,
      std::string_view,
      const std::vector<std::string>&,
      const std::vector<uint64_t>&,
      const std::optional<std::string>&,
      std::string_view,
      std::string_view,
      uint32_t,
      const std::optional<std::string>&) override {
    return absl::UnimplementedError("register_memory_replica not needed for key-mapping tests");
  }

  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::UnimplementedError("unregister_replica not needed for key-mapping tests");
  }

  absl::Status unregister_replica_by_worker(
      std::string_view,
      std::string_view,
      std::optional<tensorcast::common::memory::MemoryLocation>,
      std::optional<uint32_t>) override {
    return absl::UnimplementedError("unregister_replica_by_worker not needed for key-mapping tests");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_replica_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_replica_transport not needed for key-mapping tests");
  }

  absl::StatusOr<tensorcast::store::components::TransportSession> request_view_transport(
      std::string_view,
      std::string_view,
      std::string_view,
      std::string_view,
      uint32_t,
      const tensorcast::store::DeviceKey&,
      uint32_t) override {
    return absl::UnimplementedError("request_view_transport not needed for key-mapping tests");
  }

  absl::Status complete_replica_transport(std::string_view) override {
    return absl::UnimplementedError("complete_replica_transport not needed for key-mapping tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::RemoteReplicaInfo>> get_artifact_replicas(
      std::string_view) override {
    return absl::UnimplementedError("get_artifact_replicas not needed for key-mapping tests");
  }

  absl::StatusOr<std::vector<tensorcast::store::components::ChunkLocationInfo>> query_chunk_locations(
      std::string_view,
      const std::vector<uint32_t>&) override {
    return absl::UnimplementedError("query_chunk_locations not needed for key-mapping tests");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> synchronize_worker_state(
      const tensorcast::global_store::v1::WorkerLocalState&,
      bool,
      std::vector<tensorcast::global_store::v1::StateChange>*) override {
    return absl::UnimplementedError("synchronize_worker_state not needed for key-mapping tests");
  }

  absl::StatusOr<std::pair<uint64_t, std::string>> request_full_state_sync(
      std::string_view,
      uint64_t,
      std::vector<tensorcast::common::v1::ReplicaInfo>*) override {
    return absl::UnimplementedError("request_full_state_sync not needed for key-mapping tests");
  }

  bool is_connected() const override {
    return true;
  }

  absl::Status batch_update_chunk_states(
      std::string_view,
      std::string_view,
      const std::vector<tensorcast::store::components::ChunkStateUpdate>&) override {
    return absl::UnimplementedError("batch_update_chunk_states not needed for key-mapping tests");
  }

  absl::StatusOr<tensorcast::store::components::KeyMapping> resolve_key_mapping(std::string_view key) override {
    auto it = mappings_.find(std::string(key));
    if (it == mappings_.end()) {
      return absl::NotFoundError("key not found");
    }
    return it->second;
  }

  absl::StatusOr<std::string> get_artifact_index_by_id(std::string_view) override {
    return absl::UnimplementedError("get_artifact_index_by_id not needed for key-mapping tests");
  }

  absl::Status upsert_key_mapping(
      std::string_view key,
      std::string_view artifact_id,
      std::string_view disk_path,
      absl::Duration) override {
    tensorcast::store::components::KeyMapping mapping;
    mapping.artifact_id = std::string(artifact_id);
    mapping.disk_path = std::string(disk_path);
    last_disk_path = mapping.disk_path;
    mappings_[std::string(key)] = std::move(mapping);
    return absl::OkStatus();
  }

  absl::Status revoke_key_mapping(std::string_view key) override {
    mappings_.erase(std::string(key));
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::store::components::PlacementPlanResult> plan_placement(
      std::string_view,
      tensorcast::global_store::v1::PlacementPolicy,
      const std::vector<tensorcast::store::components::PlacementShardSpec>&,
      std::string_view) override {
    return absl::UnimplementedError("plan_placement not needed for key-mapping tests");
  }

  absl::Status report_persistence_status(const tensorcast::store::components::PersistenceReport&) override {
    return absl::UnimplementedError("report_persistence_status not needed for key-mapping tests");
  }

  void update_local_endpoint(std::string, std::string, uint32_t, uint32_t) override {}

  absl::Status update_artifact_view_state(const tensorcast::store::components::VariantViewUpdate&) override {
    return absl::UnimplementedError("update_artifact_view_state not needed for key-mapping tests");
  }

  std::string last_disk_path;

 private:
  std::unordered_map<std::string, tensorcast::store::components::KeyMapping> mappings_;
};

} // namespace

TEST_CASE("PublishReplicaKey canonicalizes storage_path before validating disk_path", "[daemon][key-mapping][disk]") {
  const auto root = test_root();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_real = root / "storage_real";
  const auto storage_link = root / "storage_link";
  const auto storage_link_via_dotdot = root / "subdir" / ".." / "storage_link";
  std::filesystem::create_directories(storage_real);
  std::filesystem::create_directories(root / "subdir");

  std::error_code ec;
  std::filesystem::remove(storage_link, ec);
  ec.clear();
  std::filesystem::create_directory_symlink(storage_real, storage_link, ec);
  REQUIRE_FALSE(ec);

  const auto artifact_dir = storage_real / "artifact";
  std::filesystem::create_directories(artifact_dir);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_real));
  auto gs_client = std::make_shared<KeyMappingGlobalStoreClient>();
  engine->set_global_store_client_for_testing(gs_client);

  StoreDaemonServiceImpl::Options svc_opts;
  svc_opts.storage_path = storage_link_via_dotdot;
  StoreDaemonServiceImpl svc(engine, svc_opts);

  tensorcast::daemon::v2::PublishReplicaKeyRequest req;
  req.set_key("key-1");
  req.mutable_artifact_descriptor()->set_artifact_id("mi2:test");
  req.set_disk_path(artifact_dir.string());

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishReplicaKeyResponse resp;
  auto status = svc.PublishReplicaKey(&ctx, &req, &resp);
  REQUIRE(status.ok());
  REQUIRE(resp.ok());

  std::error_code canonical_ec;
  const auto expected = std::filesystem::weakly_canonical(artifact_dir, canonical_ec);
  REQUIRE_FALSE(canonical_ec);
  REQUIRE(gs_client->last_disk_path == expected.string());
}
