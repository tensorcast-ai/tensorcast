// Copyright (c) 2026, TensorCast Team.

#include "daemon/grpc_service_impl.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>

#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"

using tensorcast::daemon::StoreDaemonServiceImpl;

namespace {

constexpr uint64_t kChunkBytes = 1ULL << 20;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_daemon_local_handle_unknown_token_test";
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
  opts.p2p_port = 47010;
  opts.memory_pool_size = 64ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  opts.artifact_chunk_bytes = static_cast<size_t>(kChunkBytes);
  opts.cpu_shared_memory_enabled = true;
  tensorcast::store::MemoryTierConfig tiers;
  tiers.enable_preemptible_memory = false;
  tiers.stable_bytes = kChunkBytes;
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

TEST_CASE("LocalHandle rejects unknown lease_token", "[daemon][local_handle][unknown_token]") {
  const auto root = test_tmpdir();
  std::filesystem::create_directories(root);
  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle.sock").string();

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts(root));
  StoreDaemonServiceImpl::Options svc_opts;
  svc_opts.storage_path = root;
  svc_opts.local_handle_socket_path = socket_path;
  svc_opts.cpu_shared_memory_enabled = true;
  StoreDaemonServiceImpl svc(engine, svc_opts);

  // Unknown/random token should be rejected (NOT_FOUND).
  const std::string token(32, 'x');
  const auto resp = local_handle_get_cpu_memfd_fd(socket_path, token);
  REQUIRE(resp.code == 1);
  if (resp.fd >= 0) {
    ::close(resp.fd);
  }
  REQUIRE(resp.fd < 0);
}
