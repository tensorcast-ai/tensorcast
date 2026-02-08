// Copyright (c) 2026, TensorCast Team.

#include "daemon/testing/daemon_service_harness.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>

#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/testing/common.h"
#include "grpcpp/server_context.h"

namespace {

constexpr uint64_t kChunkBytes = 1ULL << 20;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_local_handle_ttl_test";
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
  opts.p2p_port = 47009;
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

} // namespace

TEST_CASE("Handle lease TTL expiry invalidates LocalHandle tokens", "[daemon][local_handle][ttl]") {
  const auto root = test_tmpdir();
  std::filesystem::create_directories(root);

  const auto artifact_dir = root / "artifact";
  std::filesystem::remove_all(artifact_dir);
  std::filesystem::create_directories(artifact_dir);
  const auto data_path = artifact_dir / "tensor.data_0";
  REQUIRE(tensorcast::testing::create_dummy_file(data_path, 2 * kChunkBytes, 'A'));
  REQUIRE(tensorcast::testing::write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir).ok());

  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle.sock").string();

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root));
  tensorcast::daemon::DaemonOptions daemon_opts;
  daemon_opts.storage_path = root;
  daemon_opts.cpu_shared_memory_enabled = true;
  daemon_opts.local_handle_socket_path = socket_path;
  daemon_opts.handle_lease_ttl = std::chrono::milliseconds(100);
  auto harness_or = tensorcast::daemon::DaemonServiceHarness::create(engine, daemon_opts);
  REQUIRE(harness_or.ok());
  auto harness = std::move(*harness_or);
  REQUIRE(harness->start().ok());
  auto& svc = harness->service();

  tensorcast::daemon::v2::ResolveArtifactFromDiskRequest resolve_req;
  resolve_req.set_disk_path(artifact_dir.string());
  grpc::ServerContext resolve_ctx;
  tensorcast::daemon::v2::ResolveArtifactFromDiskResponse resolve_resp;
  REQUIRE(svc.ResolveArtifactFromDisk(&resolve_ctx, &resolve_req, &resolve_resp).ok());
  REQUIRE_FALSE(resolve_resp.artifact_id().empty());
  const std::string artifact_id = resolve_resp.artifact_id();

  auto materialize = [&](const std::string& replica_uuid) -> std::string {
    tensorcast::daemon::v2::MaterializeReplicaRequest req;
    req.set_artifact_id(artifact_id);
    req.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
    req.set_preference(tensorcast::daemon::v2::SourcePreference::SOURCE_PREFERENCE_PREFER_DISK);
    req.set_wait_for_completion(true);
    req.set_pid(getpid());
    req.set_replica_uuid(replica_uuid);

    grpc::ServerContext mctx;
    tensorcast::daemon::v2::MaterializeReplicaResponse resp;
    const auto st = svc.MaterializeReplica(&mctx, &req, &resp);
    CAPTURE(st.error_code());
    CAPTURE(st.error_message());
    REQUIRE(st.ok());
    REQUIRE(resp.mem_handle().has_cpu_memfd());
    REQUIRE_FALSE(resp.mem_handle().lease_token().empty());

    tensorcast::daemon::v2::ConfirmReplicaRequest creq;
    creq.set_disk_path(artifact_dir.string());
    creq.set_replica_uuid(replica_uuid);
    creq.set_target_device_type(tensorcast::daemon::v2::DeviceType::DEVICE_TYPE_CPU);
    grpc::ServerContext cctx;
    tensorcast::daemon::v2::ConfirmReplicaResponse cresp;
    REQUIRE(svc.ConfirmReplica(&cctx, &creq, &cresp).ok());

    return resp.mem_handle().lease_token();
  };

  const std::string token1 = materialize("ttl_replica_1");
  {
    auto resp1 = local_handle_get_cpu_memfd_fd(socket_path, token1);
    REQUIRE(resp1.code == 0);
    if (resp1.fd >= 0) {
      ::close(resp1.fd);
    }
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  harness->kernel().sweep_session_lifecycle_for_test();

  // Token should be expired/swept.
  REQUIRE(local_handle_get_cpu_memfd_fd(socket_path, token1).code != 0);

  // Fresh materialization issues a new token that succeeds.
  const std::string token2 = materialize("ttl_replica_2");
  REQUIRE(token2 != token1);
  {
    auto resp2 = local_handle_get_cpu_memfd_fd(socket_path, token2);
    REQUIRE(resp2.code == 0);
    if (resp2.fd >= 0) {
      ::close(resp2.fd);
    }
  }
}
