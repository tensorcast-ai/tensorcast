// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "core/testing/common.h"
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
  void set_mapping(std::string key, std::string artifact_id, uint64_t generation, uint32_t ttl_seconds) {
    tensorcast::store::components::KeyMapping mapping;
    mapping.artifact_id = std::move(artifact_id);
    mapping.generation = generation;
    mapping.cache_ttl_seconds = ttl_seconds;
    mappings_[std::move(key)] = std::move(mapping);
  }

  bool is_connected() const override {
    return connected;
  }

  absl::StatusOr<tensorcast::store::components::KeyMapping> resolve_key_mapping(std::string_view key) override {
    if (!connected) {
      return absl::UnavailableError("disconnected");
    }
    ++resolve_calls;
    auto it = mappings_.find(std::string(key));
    if (it == mappings_.end()) {
      return absl::NotFoundError("key not found");
    }
    return it->second;
  }

  absl::StatusOr<tensorcast::store::components::KeyMapping> resolve_key_mapping_with_options(
      std::string_view key,
      const tensorcast::store::components::RpcOptions& rpc_options) override {
    last_resolve_rpc_options = rpc_options;
    return resolve_key_mapping(key);
  }

  absl::Status upsert_key_mapping(std::string_view key, std::string_view artifact_id, absl::Duration) override {
    ++upsert_calls;
    if (upsert_ready_failures_remaining > 0) {
      --upsert_ready_failures_remaining;
      return absl::FailedPreconditionError(
          absl::StrCat("artifact/index not ready for key mapping upsert: artifact_id=", artifact_id));
    }
    tensorcast::store::components::KeyMapping mapping;
    mapping.artifact_id = std::string(artifact_id);
    mappings_[std::string(key)] = std::move(mapping);
    last_key = std::string(key);
    last_artifact_id = std::string(artifact_id);
    return absl::OkStatus();
  }

  absl::Status upsert_artifact_metadata(
      const tensorcast::common::v1::ArtifactDescriptor& descriptor,
      std::string_view canonical_index_data) override {
    ++artifact_metadata_upsert_calls;
    last_metadata_artifact_id = descriptor.artifact_id();
    last_metadata_index.assign(canonical_index_data);
    return absl::OkStatus();
  }

  absl::Status revoke_key_mapping(std::string_view key) override {
    mappings_.erase(std::string(key));
    return absl::OkStatus();
  }

  bool connected{true};
  int resolve_calls{0};
  int upsert_calls{0};
  int upsert_ready_failures_remaining{0};
  int artifact_metadata_upsert_calls{0};
  std::string last_key;
  std::string last_artifact_id;
  std::string last_metadata_artifact_id;
  std::string last_metadata_index;
  std::optional<tensorcast::store::components::RpcOptions> last_resolve_rpc_options;

 private:
  std::unordered_map<std::string, tensorcast::store::components::KeyMapping> mappings_;
};

} // namespace

TEST_CASE("PublishReplicaKey upserts key mapping", "[daemon][key-mapping]") {
  const auto root = test_root();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_root = root / "storage_root";
  std::filesystem::create_directories(storage_root);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));
  auto gs_client = std::make_shared<KeyMappingGlobalStoreClient>();
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::PublishReplicaKeyRequest req;
  req.set_key("key-1");
  req.mutable_artifact_descriptor()->set_artifact_id("mi2:test");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishReplicaKeyResponse resp;
  auto status = svc.PublishReplicaKey(&ctx, &req, &resp);
  REQUIRE(status.ok());
  REQUIRE(resp.ok());
  REQUIRE(gs_client->last_key == "key-1");
  REQUIRE(gs_client->last_artifact_id == "mi2:test");
}

TEST_CASE("PublishReplicaKey retries transient artifact readiness race", "[daemon][key-mapping]") {
  const auto root = test_root();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_root = root / "storage_root_retry";
  std::filesystem::create_directories(storage_root);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));
  auto gs_client = std::make_shared<KeyMappingGlobalStoreClient>();
  gs_client->upsert_ready_failures_remaining = 2;
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::PublishReplicaKeyRequest req;
  req.set_key("key-retry");
  req.mutable_artifact_descriptor()->set_artifact_id("mi2:test-retry");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::PublishReplicaKeyResponse resp;
  auto status = svc.PublishReplicaKey(&ctx, &req, &resp);
  REQUIRE(status.ok());
  REQUIRE(resp.ok());
  REQUIRE(gs_client->upsert_calls == 3);
  REQUIRE(gs_client->last_key == "key-retry");
  REQUIRE(gs_client->last_artifact_id == "mi2:test-retry");
}

TEST_CASE("PublishReplicaKey repairs local import metadata after late Global Store connect", "[daemon][key-mapping]") {
  const auto root = test_root();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_root = root / "storage_root_local_import_repair";
  const auto artifact_dir = storage_root / "artifact";
  std::filesystem::create_directories(storage_root);
  std::filesystem::create_directories(artifact_dir);
  REQUIRE(tensorcast::testing::create_dummy_file(artifact_dir / "tensor.data", 64));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));
  auto gs_client = std::make_shared<KeyMappingGlobalStoreClient>();
  gs_client->connected = false;
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::ImportArtifactFromPathRequest import_req;
  import_req.set_path(artifact_dir.string());
  import_req.set_verify_checksums(true);
  grpc::ServerContext import_ctx;
  tensorcast::daemon::v2::ImportArtifactFromPathResponse import_resp;
  auto import_status = svc.ImportArtifactFromPath(&import_ctx, &import_req, &import_resp);
  REQUIRE(import_status.ok());
  REQUIRE(import_resp.artifact_id().starts_with("mi2:"));
  REQUIRE(gs_client->artifact_metadata_upsert_calls == 0);

  gs_client->connected = true;
  gs_client->upsert_ready_failures_remaining = 1;

  tensorcast::daemon::v2::PublishReplicaKeyRequest publish_req;
  publish_req.set_key("key-local-import-repair");
  publish_req.mutable_artifact_descriptor()->set_artifact_id(import_resp.artifact_id());
  grpc::ServerContext publish_ctx;
  tensorcast::daemon::v2::PublishReplicaKeyResponse publish_resp;
  auto publish_status = svc.PublishReplicaKey(&publish_ctx, &publish_req, &publish_resp);

  REQUIRE(publish_status.ok());
  REQUIRE(publish_resp.ok());
  REQUIRE(gs_client->artifact_metadata_upsert_calls == 1);
  REQUIRE(gs_client->last_metadata_artifact_id == import_resp.artifact_id());
  REQUIRE(gs_client->last_metadata_index == import_resp.canonical_index_bytes());
  REQUIRE(gs_client->upsert_calls == 2);
  REQUIRE(gs_client->last_key == "key-local-import-repair");
  REQUIRE(gs_client->last_artifact_id == import_resp.artifact_id());
}

TEST_CASE("ResolveKeyMapping serves daemon-local cache when Global Store is unavailable", "[daemon][key-mapping]") {
  const auto root = test_root();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_root = root / "storage_root";
  std::filesystem::create_directories(storage_root);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));
  auto gs_client = std::make_shared<KeyMappingGlobalStoreClient>();
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::PublishReplicaKeyRequest publish_req;
  publish_req.set_key("key-cache-1");
  publish_req.mutable_artifact_descriptor()->set_artifact_id("mi2:test-cache");
  grpc::ServerContext publish_ctx;
  tensorcast::daemon::v2::PublishReplicaKeyResponse publish_resp;
  auto publish_status = svc.PublishReplicaKey(&publish_ctx, &publish_req, &publish_resp);
  REQUIRE(publish_status.ok());
  REQUIRE(publish_resp.ok());

  tensorcast::daemon::v2::ResolveKeyMappingRequest resolve_req;
  resolve_req.set_key("key-cache-1");
  grpc::ServerContext resolve_ctx;
  tensorcast::daemon::v2::ResolveKeyMappingResponse resolve_resp;
  auto resolve_status = svc.ResolveKeyMapping(&resolve_ctx, &resolve_req, &resolve_resp);
  REQUIRE(resolve_status.ok());
  REQUIRE(resolve_resp.artifact_id() == "mi2:test-cache");
  const int resolve_calls_before_disconnect = gs_client->resolve_calls;

  gs_client->connected = false;
  grpc::ServerContext resolve_cached_ctx;
  tensorcast::daemon::v2::ResolveKeyMappingResponse resolve_cached_resp;
  auto resolve_cached_status = svc.ResolveKeyMapping(&resolve_cached_ctx, &resolve_req, &resolve_cached_resp);
  REQUIRE(resolve_cached_status.ok());
  REQUIRE(resolve_cached_resp.artifact_id() == "mi2:test-cache");
  REQUIRE(gs_client->resolve_calls == resolve_calls_before_disconnect);
}

TEST_CASE("ResolveKeyMapping cache keeps newer generation on stale local mutation", "[daemon][key-mapping]") {
  const auto root = test_root();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_root = root / "storage_root";
  std::filesystem::create_directories(storage_root);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));
  auto gs_client = std::make_shared<KeyMappingGlobalStoreClient>();
  gs_client->set_mapping("key-gen-guard", "mi2:new", /*generation=*/10, /*ttl_seconds=*/30);
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::ResolveKeyMappingRequest resolve_req;
  resolve_req.set_key("key-gen-guard");
  grpc::ServerContext resolve_ctx;
  tensorcast::daemon::v2::ResolveKeyMappingResponse resolve_resp;
  auto resolve_status = svc.ResolveKeyMapping(&resolve_ctx, &resolve_req, &resolve_resp);
  REQUIRE(resolve_status.ok());
  REQUIRE(resolve_resp.artifact_id() == "mi2:new");
  REQUIRE(resolve_resp.generation() == 10);

  tensorcast::daemon::v2::PublishReplicaKeyRequest publish_req;
  publish_req.set_key("key-gen-guard");
  publish_req.mutable_artifact_descriptor()->set_artifact_id("mi2:stale");
  grpc::ServerContext publish_ctx;
  tensorcast::daemon::v2::PublishReplicaKeyResponse publish_resp;
  auto publish_status = svc.PublishReplicaKey(&publish_ctx, &publish_req, &publish_resp);
  REQUIRE(publish_status.ok());
  REQUIRE(publish_resp.ok());

  gs_client->connected = false;
  grpc::ServerContext resolve_cached_ctx;
  tensorcast::daemon::v2::ResolveKeyMappingResponse resolve_cached_resp;
  auto resolve_cached_status = svc.ResolveKeyMapping(&resolve_cached_ctx, &resolve_req, &resolve_cached_resp);
  REQUIRE(resolve_cached_status.ok());
  REQUIRE(resolve_cached_resp.artifact_id() == "mi2:new");
  REQUIRE(resolve_cached_resp.generation() == 10);
}

TEST_CASE("ResolveKeyMapping forwards bounded upstream rpc options", "[daemon][key-mapping]") {
  const auto root = test_root();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  const auto storage_root = root / "storage_root";
  std::filesystem::create_directories(storage_root);

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(storage_root));
  auto gs_client = std::make_shared<KeyMappingGlobalStoreClient>();
  gs_client->set_mapping("key-budget", "mi2:budget", /*generation=*/1, /*ttl_seconds=*/0);
  engine->set_global_store_client_for_testing(gs_client);

  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = storage_root;
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::ResolveKeyMappingRequest resolve_req;
  resolve_req.set_key("key-budget");
  grpc::ServerContext resolve_ctx;
  tensorcast::daemon::v2::ResolveKeyMappingResponse resolve_resp;
  auto resolve_status = svc.ResolveKeyMapping(&resolve_ctx, &resolve_req, &resolve_resp);
  REQUIRE(resolve_status.ok());
  REQUIRE(resolve_resp.artifact_id() == "mi2:budget");
  REQUIRE(gs_client->last_resolve_rpc_options.has_value());
  REQUIRE(gs_client->last_resolve_rpc_options->timeout.has_value());
  REQUIRE(*gs_client->last_resolve_rpc_options->timeout <= absl::Seconds(5));
  REQUIRE(*gs_client->last_resolve_rpc_options->timeout > absl::ZeroDuration());
  REQUIRE(gs_client->last_resolve_rpc_options->max_retries.has_value());
  REQUIRE(*gs_client->last_resolve_rpc_options->max_retries == 0);
  REQUIRE(static_cast<bool>(gs_client->last_resolve_rpc_options->cancel_check));
}
