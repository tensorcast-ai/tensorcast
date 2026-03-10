// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

#include <unistd.h>

#include <sys/socket.h>
#include <sys/un.h>

#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/common.h"
#include "grpcpp/server_context.h"
#include "nlohmann/json.hpp"

namespace {

constexpr uint64_t kChunkBytes = 1ULL << 20;

bool write_all(int fd, const void* buf, size_t n) {
  const auto* p = static_cast<const uint8_t*>(buf);
  size_t off = 0;
  while (off < n) {
    const ssize_t rc = ::write(fd, p + off, n - off);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    off += static_cast<size_t>(rc);
  }
  return true;
}

bool read_all(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t off = 0;
  while (off < n) {
    const ssize_t rc = ::read(fd, p + off, n - off);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (rc == 0) {
      return false;
    }
    off += static_cast<size_t>(rc);
  }
  return true;
}

uint8_t local_handle_release_handle(const std::string& socket_path, const std::string& lease_token) {
  const int sock = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(sock >= 0);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  REQUIRE(socket_path.size() < sizeof(addr.sun_path));
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  REQUIRE(::connect(sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0);

  const uint8_t opcode = 2;
  const uint32_t token_len = static_cast<uint32_t>(lease_token.size());
  REQUIRE(write_all(sock, &opcode, sizeof(opcode)));
  REQUIRE(write_all(sock, &token_len, sizeof(token_len)));
  REQUIRE(write_all(sock, lease_token.data(), lease_token.size()));

  uint8_t resp = 0xFF;
  REQUIRE(read_all(sock, &resp, sizeof(resp)));
  ::close(sock);
  return resp;
}

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_cpu_memfd_stable_budget_test";
}

std::filesystem::path make_socket_dir() {
  const auto dir = std::filesystem::temp_directory_path() / ("tensorcast_local_handle_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
  return dir;
}

tensorcast::store::StoreEngineOptions make_engine_opts(const std::filesystem::path& root, uint64_t stable_bytes) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (root / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47008;
  opts.memory_pool_size = 64ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  opts.artifact_chunk_bytes = static_cast<size_t>(kChunkBytes);
  opts.cpu_shared_memory_enabled = true;
  tensorcast::store::MemoryTierConfig tiers;
  tiers.enable_preemptible_memory = false;
  tiers.stable_bytes = stable_bytes;
  opts.memory_tier_config = tiers;
  return opts;
}

std::string read_artifact_id(const std::filesystem::path& artifact_dir) {
  std::ifstream descriptor_in(artifact_dir / "artifact_descriptor.json");
  nlohmann::json descriptor_json;
  descriptor_in >> descriptor_json;
  return descriptor_json.value("artifact_id", "");
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

} // namespace

TEST_CASE("CPU memfd materialization is gated by stable_bytes", "[daemon][cpu_memfd][stable_budget]") {
  const auto root = test_tmpdir();
  std::filesystem::create_directories(root);

  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto artifact_rel = std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact";
  const auto artifact_dir = root / artifact_rel;
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 2 * kChunkBytes, 'A'));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());
  const std::string artifact_id = read_artifact_id(artifact_dir);
  REQUIRE_FALSE(artifact_id.empty());
  register_disk_location(*gs_client, artifact_id, artifact_rel);

  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle.sock").string();

  // stable_bytes smaller than the required export coverage (2 chunks).
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root, /*stable_bytes=*/kChunkBytes));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = root;
  daemon_opts.cpu_shared_memory_enabled = true;
  daemon_opts.local_handle_socket_path = socket_path;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  req.set_wait_for_completion(true);
  req.set_pid(getpid());
  req.set_replica_uuid("cpu_memfd_stable_budget_replica");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED);
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_FAILED);
}

TEST_CASE("CPU memfd stable_bytes recovers after lease release", "[daemon][cpu_memfd][stable_budget]") {
  const auto root = test_tmpdir();
  std::filesystem::create_directories(root);

  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto artifact_rel_a = std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_a";
  const auto artifact_rel_b = std::filesystem::path("clusters") / gs_client->cluster_id / "objects" / "artifact_b";
  const auto artifact_dir_a = root / artifact_rel_a;
  const auto artifact_dir_b = root / artifact_rel_b;
  std::filesystem::remove_all(artifact_dir_a);
  std::filesystem::remove_all(artifact_dir_b);
  std::filesystem::create_directories(artifact_dir_a);
  std::filesystem::create_directories(artifact_dir_b);

  const auto data_path_a = artifact_dir_a / "tensor.data_0";
  const auto data_path_b = artifact_dir_b / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path_a, 2 * kChunkBytes, 'A'));
  REQUIRE(tensorcast::testing::create_dummy_file(data_path_b, 2 * kChunkBytes, 'B'));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir_a).ok());
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir_b).ok());
  const std::string artifact_id_a = read_artifact_id(artifact_dir_a);
  const std::string artifact_id_b = read_artifact_id(artifact_dir_b);
  REQUIRE_FALSE(artifact_id_a.empty());
  REQUIRE_FALSE(artifact_id_b.empty());
  register_disk_location(*gs_client, artifact_id_a, artifact_rel_a);
  register_disk_location(*gs_client, artifact_id_b, artifact_rel_b);

  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle.sock").string();

  // stable_bytes only admits one export of 2 chunks at a time.
  auto engine =
      std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root, /*stable_bytes=*/2 * kChunkBytes));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = root;
  daemon_opts.cpu_shared_memory_enabled = true;
  daemon_opts.local_handle_socket_path = socket_path;
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  auto materialize_cpu = [&](const std::string& artifact_id, const std::string& replica_uuid) {
    tensorcast::daemon::v2::MaterializeReplicaRequest req;
    req.mutable_selection()->set_artifact_id(artifact_id);
    req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
    req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
    req.set_wait_for_completion(true);
    req.set_pid(getpid());
    req.set_replica_uuid(replica_uuid);
    grpc::ServerContext ctx;
    tensorcast::daemon::v2::MaterializeReplicaResponse resp;
    auto st = svc.MaterializeReplica(&ctx, &req, &resp);
    return std::pair<grpc::Status, tensorcast::daemon::v2::MaterializeReplicaResponse>{st, resp};
  };

  auto [st_a, resp_a] = materialize_cpu(artifact_id_a, "cpu_memfd_budget_a");
  REQUIRE(st_a.ok());
  REQUIRE(resp_a.mem_handle().has_cpu_memfd());
  REQUIRE_FALSE(resp_a.mem_handle().lease_token().empty());
  const std::string token_a = resp_a.mem_handle().lease_token();

  auto [st_b, resp_b] = materialize_cpu(artifact_id_b, "cpu_memfd_budget_b");
  REQUIRE_FALSE(st_b.ok());
  REQUIRE(st_b.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED);

  REQUIRE(local_handle_release_handle(socket_path, token_a) == 0);

  auto [st_b2, resp_b2] = materialize_cpu(artifact_id_b, "cpu_memfd_budget_b2");
  REQUIRE(st_b2.ok());
  REQUIRE(resp_b2.mem_handle().has_cpu_memfd());
  REQUIRE_FALSE(resp_b2.mem_handle().lease_token().empty());
  const std::string token_b = resp_b2.mem_handle().lease_token();

  REQUIRE(local_handle_release_handle(socket_path, token_b) == 0);
}
