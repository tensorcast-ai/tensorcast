// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>

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
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_cpu_memfd_e2e_test";
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
  opts.p2p_port = 47007;
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

struct LocalHandleFdResponse {
  uint8_t code{0xFF};
  int fd{-1};
};

LocalHandleFdResponse local_handle_get_cpu_memfd_fd(const std::string& socket_path, const std::string& lease_token) {
  LocalHandleFdResponse out;

  const int sock = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(sock >= 0);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  REQUIRE(socket_path.size() < sizeof(addr.sun_path));
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  REQUIRE(::connect(sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0);

  const uint8_t opcode = 1;
  const uint32_t token_len = static_cast<uint32_t>(lease_token.size());
  REQUIRE(write_all(sock, &opcode, sizeof(opcode)));
  REQUIRE(write_all(sock, &token_len, sizeof(token_len)));
  REQUIRE(write_all(sock, lease_token.data(), lease_token.size()));

  char resp_byte = 0;
  iovec iov{.iov_base = &resp_byte, .iov_len = 1};
  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  std::memset(cmsg_buf, 0, sizeof(cmsg_buf));
  msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  int flags = 0;
#ifdef MSG_CMSG_CLOEXEC
  flags |= MSG_CMSG_CLOEXEC;
#endif
  const ssize_t rc = ::recvmsg(sock, &msg, flags);
  REQUIRE(rc == 1);
  out.code = static_cast<uint8_t>(resp_byte);

  if (out.code == 0) {
    for (cmsghdr* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
      if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        continue;
      }
      std::memcpy(&out.fd, CMSG_DATA(cmsg), sizeof(int));
      break;
    }
  }

  ::close(sock);
  return out;
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

} // namespace

TEST_CASE("CPU memfd end-to-end materialization and LocalHandle FD exchange", "[daemon][cpu_memfd][local_handle]") {
  const auto root = test_tmpdir();
  std::filesystem::create_directories(root);

  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();

  // Prepare a minimal disk artifact with deterministic content.
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

  auto engine =
      std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root, /*stable_bytes=*/64 * kChunkBytes));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = root;
  daemon_opts.cpu_shared_memory_enabled = true;
  daemon_opts.local_handle_socket_path = socket_path;
  daemon_opts.handle_lease_ttl = std::chrono::milliseconds(5000);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  const std::string replica_uuid = "cpu_memfd_e2e_replica";
  const std::string replica_uuid2 = "cpu_memfd_e2e_replica_2";

  tensorcast::daemon::v2::MaterializeReplicaRequest req;
  req.mutable_selection()->set_artifact_id(artifact_id);
  req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  req.mutable_source_policy()->set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
  req.set_wait_for_completion(true);
  req.set_pid(getpid());
  req.set_replica_uuid(replica_uuid);

  grpc::ServerContext mctx;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp;
  auto st = svc.MaterializeReplica(&mctx, &req, &resp);
  REQUIRE(st.ok());
  REQUIRE(resp.status() == tensorcast::daemon::v2::MATERIALIZE_REPLICA_STATUS_ALLOCATED);
  REQUIRE(resp.mem_handle().has_cpu_memfd());
  REQUIRE_FALSE(resp.mem_handle().lease_token().empty());
  REQUIRE_FALSE(resp.mem_handle().has_cuda_ipc_handle());

  // Confirm readiness (SDK invariant: import after ConfirmReplica).
  tensorcast::daemon::v2::ConfirmReplicaRequest creq;
  creq.set_disk_path(artifact_dir.string());
  creq.set_replica_uuid(replica_uuid);
  creq.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
  grpc::ServerContext cctx;
  tensorcast::daemon::v2::ConfirmReplicaResponse cresp;
  st = svc.ConfirmReplica(&cctx, &creq, &cresp);
  REQUIRE(st.ok());

  tensorcast::daemon::v2::MaterializeReplicaRequest req2 = req;
  req2.set_replica_uuid(replica_uuid2);
  grpc::ServerContext mctx2;
  tensorcast::daemon::v2::MaterializeReplicaResponse resp2;
  st = svc.MaterializeReplica(&mctx2, &req2, &resp2);
  REQUIRE(st.ok());
  REQUIRE(resp2.mem_handle().has_cpu_memfd());
  REQUIRE_FALSE(resp2.mem_handle().lease_token().empty());

  tensorcast::daemon::v2::ConfirmReplicaRequest creq2 = creq;
  creq2.set_replica_uuid(replica_uuid2);
  grpc::ServerContext cctx2;
  tensorcast::daemon::v2::ConfirmReplicaResponse cresp2;
  st = svc.ConfirmReplica(&cctx2, &creq2, &cresp2);
  REQUIRE(st.ok());

  const std::string lease_token = resp.mem_handle().lease_token();
  const std::string lease_token2 = resp2.mem_handle().lease_token();
  const uint64_t size_bytes = resp.mem_handle().cpu_memfd().size_bytes();
  const uint64_t offset_bytes = resp.mem_handle().cpu_memfd().offset_bytes();
  REQUIRE(size_bytes > 0);
  REQUIRE(lease_token != lease_token2);

  // Exchange lease token for memfd FD via LocalHandle.
  auto fd_resp = local_handle_get_cpu_memfd_fd(socket_path, lease_token);
  REQUIRE(fd_resp.code == 0);
  REQUIRE(fd_resp.fd >= 0);

  const int fd_flags = ::fcntl(fd_resp.fd, F_GETFD);
  REQUIRE(fd_flags >= 0);
  REQUIRE((fd_flags & FD_CLOEXEC) != 0);

  const int seals = ::fcntl(fd_resp.fd, F_GET_SEALS);
  REQUIRE(seals >= 0);
  REQUIRE((seals & F_SEAL_GROW) != 0);
  REQUIRE((seals & F_SEAL_SHRINK) != 0);

  void* mapped = ::mmap(
      nullptr,
      static_cast<size_t>(size_bytes),
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE,
      fd_resp.fd,
      static_cast<off_t>(offset_bytes));
  REQUIRE(mapped != MAP_FAILED);

  const auto expected = tensorcast::testing::read_file_content(data_path);
  REQUIRE(expected.size() <= static_cast<size_t>(size_bytes));
  REQUIRE(std::memcmp(mapped, expected.data(), expected.size()) == 0);

  // Validate MAP_PRIVATE copy-on-write semantics (mutate a byte and re-map).
  if (!expected.empty()) {
    const uint8_t original = static_cast<const uint8_t*>(mapped)[0];
    static_cast<uint8_t*>(mapped)[0] = static_cast<uint8_t>(original ^ 0xFF);
    REQUIRE(static_cast<const uint8_t*>(mapped)[0] != original);
    REQUIRE(::munmap(mapped, static_cast<size_t>(size_bytes)) == 0);
    mapped = ::mmap(
        nullptr,
        static_cast<size_t>(size_bytes),
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE,
        fd_resp.fd,
        static_cast<off_t>(offset_bytes));
    REQUIRE(mapped != MAP_FAILED);
    REQUIRE(static_cast<const uint8_t*>(mapped)[0] == original);
  }

  REQUIRE(::munmap(mapped, static_cast<size_t>(size_bytes)) == 0);
  ::close(fd_resp.fd);

  auto fd_resp2 = local_handle_get_cpu_memfd_fd(socket_path, lease_token2);
  REQUIRE(fd_resp2.code == 0);
  REQUIRE(fd_resp2.fd >= 0);
  ::close(fd_resp2.fd);

  // Release lease (idempotent behavior: OK then NOT_FOUND).
  REQUIRE(local_handle_release_handle(socket_path, lease_token) == 0);
  REQUIRE(local_handle_release_handle(socket_path, lease_token) == 1);

  // After release, token is no longer usable.
  auto fd_after = local_handle_get_cpu_memfd_fd(socket_path, lease_token);
  REQUIRE(fd_after.code != 0);
  REQUIRE(fd_after.fd < 0);

  // Releasing one lease must not invalidate other active leases.
  auto fd_still = local_handle_get_cpu_memfd_fd(socket_path, lease_token2);
  REQUIRE(fd_still.code == 0);
  REQUIRE(fd_still.fd >= 0);
  ::close(fd_still.fd);

  REQUIRE(local_handle_release_handle(socket_path, lease_token2) == 0);
  REQUIRE(local_handle_release_handle(socket_path, lease_token2) == 1);

  auto fd_after2 = local_handle_get_cpu_memfd_fd(socket_path, lease_token2);
  REQUIRE(fd_after2.code != 0);
  REQUIRE(fd_after2.fd < 0);
}

TEST_CASE(
    "Stable DRAM registration advertises CPU memfd publish handle and accepts range-only feed",
    "[daemon][cpu_memfd][registration]") {
  const auto root = test_tmpdir();
  std::filesystem::create_directories(root);

  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle_registration.sock").string();

  auto engine =
      std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root, /*stable_bytes=*/64 * kChunkBytes));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = root;
  daemon_opts.cpu_shared_memory_enabled = true;
  daemon_opts.local_handle_socket_path = socket_path;
  daemon_opts.handle_lease_ttl = std::chrono::milliseconds(5000);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest begin_req;
  begin_req.set_device_id(0);
  begin_req.set_total_size(16);
  begin_req.set_owner_pid(getpid());
  begin_req.mutable_stable_dram()->set_stage_on_gpu(false);
  begin_req.mutable_stable_dram()->set_release_gpu_on_commit(false);
  auto* idx = begin_req.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  grpc::ServerContext begin_ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse begin_resp;
  auto st = svc.BeginRegisterArtifact(&begin_ctx, &begin_req, &begin_resp);
  REQUIRE(st.ok());
  REQUIRE(begin_resp.has_stable_dram());
  REQUIRE(begin_resp.stable_dram().publish_cpu_memfd().size_bytes() >= 16);
  REQUIRE_FALSE(begin_resp.stable_dram().publish_cpu_memfd_lease_token().empty());
  REQUIRE(begin_resp.stable_dram().staging_cuda_ipc_handle().empty());

  const std::string lease_token = begin_resp.stable_dram().publish_cpu_memfd_lease_token();
  auto fd_resp = local_handle_get_cpu_memfd_fd(socket_path, lease_token);
  REQUIRE(fd_resp.code == 0);
  REQUIRE(fd_resp.fd >= 0);

  const uint64_t size_bytes = begin_resp.stable_dram().publish_cpu_memfd().size_bytes();
  const uint64_t offset_bytes = begin_resp.stable_dram().publish_cpu_memfd().offset_bytes();
  void* mapped = ::mmap(
      nullptr,
      static_cast<size_t>(size_bytes),
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      fd_resp.fd,
      static_cast<off_t>(offset_bytes));
  REQUIRE(mapped != MAP_FAILED);

  std::array<uint8_t, 16> payload{};
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>(i + 1);
  }
  std::memcpy(mapped, payload.data(), payload.size());
  REQUIRE(::munmap(mapped, static_cast<size_t>(size_bytes)) == 0);
  ::close(fd_resp.fd);

  tensorcast::daemon::v2::FeedRegisterArtifactStreamRequest feed_req;
  feed_req.set_registration_id(begin_resp.registration_id());
  auto* progress = feed_req.mutable_stable_dram_write_progress();
  auto* range = progress->add_ranges();
  range->set_canonical_offset(0);
  range->set_length(payload.size());
  progress->set_upload_complete(true);
  st = svc.feed_register_artifact_stream_vector({feed_req});
  REQUIRE(st.ok());

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest commit_req;
  commit_req.set_registration_id(begin_resp.registration_id());
  grpc::ServerContext commit_ctx;
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse commit_resp;
  st = svc.CommitRegisteredArtifact(&commit_ctx, &commit_req, &commit_resp);
  INFO("commit error_code=" << static_cast<int>(st.error_code()) << " message=" << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(commit_resp.has_artifact_descriptor());

  auto fd_after_commit = local_handle_get_cpu_memfd_fd(socket_path, lease_token);
  REQUIRE(fd_after_commit.code != 0);
  REQUIRE(fd_after_commit.fd < 0);
}

TEST_CASE(
    "Stable DRAM pinned commit uses one stable admission when budget equals payload",
    "[daemon][cpu_memfd][registration][stable_budget]") {
  const auto root = test_tmpdir();
  std::filesystem::create_directories(root);

  auto gs_client = std::make_shared<tensorcast::store::testing::RecordingGlobalStoreClient>();
  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle_registration_exact_budget.sock").string();

  constexpr uint64_t kPayloadBytes = 16;
  auto engine =
      std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root, /*stable_bytes=*/kPayloadBytes));
  engine->set_global_store_client_for_testing(gs_client);
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = root;
  daemon_opts.cpu_shared_memory_enabled = true;
  daemon_opts.local_handle_socket_path = socket_path;
  daemon_opts.handle_lease_ttl = std::chrono::milliseconds(5000);
  auto harness_or =
      tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts, /*async_runtime=*/nullptr, gs_client);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::BeginRegisterArtifactRequest begin_req;
  begin_req.set_device_id(0);
  begin_req.set_total_size(kPayloadBytes);
  begin_req.set_owner_pid(getpid());
  begin_req.mutable_stable_dram()->set_stage_on_gpu(false);
  begin_req.mutable_stable_dram()->set_release_gpu_on_commit(false);
  begin_req.mutable_policy()->set_profile(tensorcast::daemon::v2::POLICY_PROFILE_PINNED);
  auto* idx = begin_req.mutable_tensor_index_data();
  idx->set_data("{}");
  idx->set_schema_version("v3");
  idx->set_encoding("json");

  grpc::ServerContext begin_ctx;
  tensorcast::daemon::v2::BeginRegisterArtifactResponse begin_resp;
  auto st = svc.BeginRegisterArtifact(&begin_ctx, &begin_req, &begin_resp);
  REQUIRE(st.ok());
  REQUIRE(begin_resp.has_stable_dram());
  REQUIRE(begin_resp.stable_dram().publish_cpu_memfd().size_bytes() >= kPayloadBytes);
  REQUIRE_FALSE(begin_resp.stable_dram().publish_cpu_memfd_lease_token().empty());

  const std::string lease_token = begin_resp.stable_dram().publish_cpu_memfd_lease_token();
  auto fd_resp = local_handle_get_cpu_memfd_fd(socket_path, lease_token);
  REQUIRE(fd_resp.code == 0);
  REQUIRE(fd_resp.fd >= 0);

  const uint64_t size_bytes = begin_resp.stable_dram().publish_cpu_memfd().size_bytes();
  const uint64_t offset_bytes = begin_resp.stable_dram().publish_cpu_memfd().offset_bytes();
  void* mapped = ::mmap(
      nullptr,
      static_cast<size_t>(size_bytes),
      PROT_READ | PROT_WRITE,
      MAP_SHARED,
      fd_resp.fd,
      static_cast<off_t>(offset_bytes));
  REQUIRE(mapped != MAP_FAILED);

  std::array<uint8_t, kPayloadBytes> payload{};
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>(i + 1);
  }
  std::memcpy(mapped, payload.data(), payload.size());
  REQUIRE(::munmap(mapped, static_cast<size_t>(size_bytes)) == 0);
  ::close(fd_resp.fd);

  tensorcast::daemon::v2::FeedRegisterArtifactStreamRequest feed_req;
  feed_req.set_registration_id(begin_resp.registration_id());
  auto* progress = feed_req.mutable_stable_dram_write_progress();
  auto* range = progress->add_ranges();
  range->set_canonical_offset(0);
  range->set_length(payload.size());
  progress->set_upload_complete(true);
  st = svc.feed_register_artifact_stream_vector({feed_req});
  REQUIRE(st.ok());

  tensorcast::daemon::v2::CommitRegisteredArtifactRequest commit_req;
  commit_req.set_registration_id(begin_resp.registration_id());
  grpc::ServerContext commit_ctx;
  tensorcast::daemon::v2::CommitRegisteredArtifactResponse commit_resp;
  st = svc.CommitRegisteredArtifact(&commit_ctx, &commit_req, &commit_resp);
  INFO("commit error_code=" << static_cast<int>(st.error_code()) << " message=" << st.error_message());
  REQUIRE(st.ok());
  REQUIRE(commit_resp.has_artifact_descriptor());
  REQUIRE(commit_resp.has_local_stable_tier());
  REQUIRE(commit_resp.local_stable_tier().status() == tensorcast::daemon::v2::LOCAL_STABLE_TIER_STATUS_READY);

  auto budget_snapshot = engine->get_memory_tier_snapshot();
  REQUIRE(budget_snapshot.has_value());
  REQUIRE(budget_snapshot->stable_used_bytes == kPayloadBytes);

  auto fd_after_commit = local_handle_get_cpu_memfd_fd(socket_path, lease_token);
  REQUIRE(fd_after_commit.code != 0);
  REQUIRE(fd_after_commit.fd < 0);
}
