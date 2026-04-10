// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/local_handle_server.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cerrno>

namespace {

constexpr uint64_t kRegionBytes = 2ULL << 20;

std::filesystem::path make_socket_dir() {
  const auto dir = std::filesystem::temp_directory_path() / ("tensorcast_local_handle_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  std::filesystem::permissions(dir, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace);
  return dir;
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

LocalHandleFdResponse local_handle_get_region_memfd_fd(
    const std::string& socket_path,
    const std::string& attach_token) {
  LocalHandleFdResponse out;

  const int sock = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(sock >= 0);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  REQUIRE(socket_path.size() < sizeof(addr.sun_path));
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  REQUIRE(::connect(sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0);

  const uint8_t opcode = 3;
  const uint32_t token_len = static_cast<uint32_t>(attach_token.size());
  REQUIRE(write_all(sock, &opcode, sizeof(opcode)));
  REQUIRE(write_all(sock, &token_len, sizeof(token_len)));
  REQUIRE(write_all(sock, attach_token.data(), attach_token.size()));

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

uint8_t local_handle_release_token(const std::string& socket_path, const std::string& token) {
  const int sock = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  REQUIRE(sock >= 0);

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  REQUIRE(socket_path.size() < sizeof(addr.sun_path));
  std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
  REQUIRE(::connect(sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0);

  const uint8_t opcode = 2;
  const uint32_t token_len = static_cast<uint32_t>(token.size());
  REQUIRE(write_all(sock, &opcode, sizeof(opcode)));
  REQUIRE(write_all(sock, &token_len, sizeof(token_len)));
  REQUIRE(write_all(sock, token.data(), token.size()));

  uint8_t resp = 0xFF;
  REQUIRE(read_all(sock, &resp, sizeof(resp)));
  ::close(sock);
  return resp;
}

} // namespace

TEST_CASE("LocalHandle serves daemon-managed HOST_SHARED region attachments", "[daemon][local_handle][host_shared]") {
  tensorcast::daemon::IpcRegionRegistry registry(
      tensorcast::daemon::IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });
  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle.sock").string();

  tensorcast::daemon::LocalHandleServer server(
      tensorcast::daemon::LocalHandleServer::Options{
          .socket_path = socket_path,
          .cpu_shared_memory_enabled = true,
      },
      registry,
      nullptr);
  REQUIRE(server.start().ok());

  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = getpid();
  params.size_bytes = kRegionBytes;
  params.ttl_ms = 1000;
  params.daemon_managed = true;
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = registry.register_region(params);
  REQUIRE(desc_or.ok());
  REQUIRE_FALSE(desc_or->attach_token.empty());

  auto fd_resp = local_handle_get_region_memfd_fd(socket_path, desc_or->attach_token);
  REQUIRE(fd_resp.code == 0);
  REQUIRE(fd_resp.fd >= 0);

  void* mapped = ::mmap(nullptr, kRegionBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_resp.fd, 0);
  REQUIRE(mapped != MAP_FAILED);
  std::memset(mapped, 0x5A, 4096);
  REQUIRE(::munmap(mapped, kRegionBytes) == 0);
  REQUIRE(::close(fd_resp.fd) == 0);

  auto unregister_busy_or = registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE_FALSE(unregister_busy_or.ok());

  REQUIRE(local_handle_release_token(socket_path, desc_or->attach_token) == 0);
  REQUIRE(local_handle_release_token(socket_path, desc_or->attach_token) != 0);

  auto unregister_or = registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE(unregister_or.ok());
  REQUIRE(*unregister_or);

  auto fd_after = local_handle_get_region_memfd_fd(socket_path, desc_or->attach_token);
  REQUIRE(fd_after.code == 1);
  REQUIRE(fd_after.fd < 0);

  server.stop();
}

TEST_CASE(
    "LocalHandle duplicate HOST_SHARED release does not consume local mapping holds",
    "[daemon][local_handle][host_shared][refcount]") {
  tensorcast::daemon::IpcRegionRegistry registry(
      tensorcast::daemon::IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });
  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle_refcount.sock").string();

  tensorcast::daemon::LocalHandleServer server(
      tensorcast::daemon::LocalHandleServer::Options{
          .socket_path = socket_path,
          .cpu_shared_memory_enabled = true,
      },
      registry,
      nullptr);
  REQUIRE(server.start().ok());

  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = getpid();
  params.size_bytes = kRegionBytes;
  params.ttl_ms = 1000;
  params.daemon_managed = true;
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = registry.register_region(params);
  REQUIRE(desc_or.ok());
  REQUIRE_FALSE(desc_or->attach_token.empty());

  auto fd_resp = local_handle_get_region_memfd_fd(socket_path, desc_or->attach_token);
  REQUIRE(fd_resp.code == 0);
  REQUIRE(fd_resp.fd >= 0);
  REQUIRE(::close(fd_resp.fd) == 0);

  auto mapping_or = registry.acquire_host_shared_local_mapping(desc_or->region_id, params.owner_pid);
  REQUIRE(mapping_or.ok());

  REQUIRE(local_handle_release_token(socket_path, desc_or->attach_token) == 0);
  REQUIRE(local_handle_release_token(socket_path, desc_or->attach_token) != 0);

  auto unregister_busy_or = registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE_FALSE(unregister_busy_or.ok());

  REQUIRE(registry.release(desc_or->region_id).ok());

  auto unregister_or = registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE(unregister_or.ok());
  REQUIRE(*unregister_or);

  server.stop();
}

TEST_CASE(
    "Daemon-managed HOST_SHARED region is cleaned up on owner exit",
    "[daemon][local_handle][host_shared][pid_exit]") {
  tensorcast::daemon::IpcRegionRegistry registry(
      tensorcast::daemon::IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });
  const auto socket_dir = make_socket_dir();
  const std::string socket_path = (socket_dir / "local_handle_pid_exit.sock").string();

  tensorcast::daemon::LocalHandleServer server(
      tensorcast::daemon::LocalHandleServer::Options{
          .socket_path = socket_path,
          .cpu_shared_memory_enabled = true,
      },
      registry,
      nullptr);
  REQUIRE(server.start().ok());

  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = getpid();
  params.size_bytes = kRegionBytes;
  params.ttl_ms = 0;
  params.daemon_managed = true;
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = registry.register_region(params);
  REQUIRE(desc_or.ok());

  auto fd_resp = local_handle_get_region_memfd_fd(socket_path, desc_or->attach_token);
  REQUIRE(fd_resp.code == 0);
  REQUIRE(fd_resp.fd >= 0);
  REQUIRE(::close(fd_resp.fd) == 0);
  REQUIRE(local_handle_release_token(socket_path, desc_or->attach_token) == 0);

  auto removed = registry.handle_pid_exit(params.owner_pid);
  REQUIRE(removed.size() == 1);
  REQUIRE(removed[0].region_id == desc_or->region_id);

  auto fd_after = local_handle_get_region_memfd_fd(socket_path, desc_or->attach_token);
  REQUIRE(fd_after.code == 1);
  REQUIRE(fd_after.fd < 0);

  server.stop();
}
