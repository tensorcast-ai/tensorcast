// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "core/testing/common.h"
#include "grpcpp/server_context.h"
#include "nlohmann/json.hpp"

namespace {

constexpr uint64_t kChunkBytes = 1ULL << 20;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_cpu_memfd_fd_pressure_test";
}

std::filesystem::path make_socket_dir() {
  const auto dir = std::filesystem::temp_directory_path() / ("tensorcast_local_handle_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
  return dir;
}

tensorcast::store::StoreEngineOptions make_engine_opts(const std::filesystem::path& root) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (root / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47011;
  opts.memory_pool_size = 64ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  opts.artifact_chunk_bytes = static_cast<size_t>(kChunkBytes);
  opts.cpu_shared_memory_enabled = true;
  tensorcast::store::MemoryTierConfig tiers;
  tiers.enable_preemptible_memory = false;
  tiers.stable_bytes = 64 * kChunkBytes;
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

uint64_t count_open_fds() {
  DIR* d = ::opendir("/proc/self/fd");
  REQUIRE(d != nullptr);
  uint64_t count = 0;
  for (;;) {
    errno = 0;
    const dirent* ent = ::readdir(d);
    if (ent == nullptr) {
      REQUIRE(errno == 0);
      break;
    }
    if (ent->d_name[0] == '.') {
      if (ent->d_name[1] == '\0') {
        continue;
      }
      if (ent->d_name[1] == '.' && ent->d_name[2] == '\0') {
        continue;
      }
    }
    count++;
  }
  ::closedir(d);
  return count;
}

struct ScopedRlimitNoFile {
  rlimit old{};
  bool installed{false};

  explicit ScopedRlimitNoFile(rlim_t soft_limit) {
    if (::getrlimit(RLIMIT_NOFILE, &old) != 0) {
      return;
    }
    rlimit rl = old;
    rl.rlim_cur = std::min(soft_limit, old.rlim_max);
    if (::setrlimit(RLIMIT_NOFILE, &rl) != 0) {
      return;
    }
    installed = true;
  }

  ~ScopedRlimitNoFile() {
    if (installed) {
      (void)::setrlimit(RLIMIT_NOFILE, &old);
    }
  }
};

struct ScopedFdList {
  std::vector<int> fds;

  ~ScopedFdList() {
    for (int fd : fds) {
      if (fd >= 0) {
        ::close(fd);
      }
    }
  }
};

} // namespace

TEST_CASE("CPU memfd fails with clear error near RLIMIT_NOFILE", "[daemon][cpu_memfd][fd_pressure]") {
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

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root));
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

  ScopedRlimitNoFile limit_guard(/*soft_limit=*/256);
  REQUIRE(limit_guard.installed);

  rlimit rl{};
  REQUIRE(::getrlimit(RLIMIT_NOFILE, &rl) == 0);
  const uint64_t limit = static_cast<uint64_t>(rl.rlim_cur);
  REQUIRE(limit > 0);

  const uint64_t target_headroom = 24;
  const uint64_t desired_open = (limit > target_headroom) ? (limit - target_headroom) : 0;

  ScopedFdList fds;
  while (count_open_fds() < desired_open) {
    const int fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
    REQUIRE(fd >= 0);
    fds.fds.push_back(fd);
  }

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  req.set_wait_for_completion(true);
  req.set_pid(getpid());
  req.set_replica_uuid("cpu_memfd_fd_pressure_replica");

  grpc::ServerContext ctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  const auto st = svc.MaterializeReplica(&ctx, &req, &resp);
  REQUIRE_FALSE(st.ok());
  REQUIRE(st.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED);
  REQUIRE(st.error_message().find("RLIMIT_NOFILE") != std::string::npos);
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_FAILED);
}
