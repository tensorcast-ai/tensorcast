// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "grpcpp/server_context.h"

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

class KeyMappingGlobalStoreClient final : public tensorcast::store::testing::GlobalStoreClientStub {
 public:
  bool is_connected() const override {
    return true;
  }

  absl::StatusOr<tensorcast::store::components::KeyMapping> resolve_key_mapping(std::string_view key) override {
    auto it = mappings_.find(std::string(key));
    if (it == mappings_.end()) {
      return absl::NotFoundError("key not found");
    }
    return it->second;
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

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_link_via_dotdot;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

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
