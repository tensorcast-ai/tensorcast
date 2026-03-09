// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/local_handle_server.h"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"

namespace tensorcast::daemon {
namespace {

constexpr size_t kMaxTokenBytes = 1024;
constexpr std::chrono::milliseconds kConnIoTimeout{200};

absl::Status send_fd_with_status(int fd, int send_fd, uint8_t status_byte) {
  char buf[1];
  buf[0] = static_cast<char>(status_byte);
  struct iovec iov{.iov_base = buf, .iov_len = sizeof(buf)};

  char cmsg_buf[CMSG_SPACE(sizeof(int))];
  std::memset(cmsg_buf, 0, sizeof(cmsg_buf));
  struct msghdr msg{};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf;
  msg.msg_controllen = sizeof(cmsg_buf);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  std::memcpy(CMSG_DATA(cmsg), &send_fd, sizeof(int));
  msg.msg_controllen = sizeof(cmsg_buf);

  ssize_t rc = ::sendmsg(fd, &msg, 0);
  if (rc < 0) {
    return absl::ErrnoToStatus(errno, "sendmsg failed");
  }
  if (static_cast<size_t>(rc) != sizeof(buf)) {
    return absl::InternalError("sendmsg short write");
  }
  return absl::OkStatus();
}

absl::Status set_socket_timeouts(int fd, std::chrono::milliseconds timeout) {
  if (timeout.count() <= 0) {
    return absl::OkStatus();
  }
  struct timeval tv{};
  tv.tv_sec = static_cast<time_t>(timeout.count() / 1000);
  tv.tv_usec = static_cast<suseconds_t>((timeout.count() % 1000) * 1000);
  if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
    return absl::ErrnoToStatus(errno, "setsockopt(SO_RCVTIMEO) failed");
  }
  if (::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) < 0) {
    return absl::ErrnoToStatus(errno, "setsockopt(SO_SNDTIMEO) failed");
  }
  return absl::OkStatus();
}

} // namespace

LocalHandleServer::LocalHandleServer(Options opts, HandleLeaseRegistry& leases)
    : opts_(std::move(opts)), leases_(&leases) {}

LocalHandleServer::~LocalHandleServer() {
  stop();
}

absl::Status LocalHandleServer::start() {
  if (opts_.socket_path.empty()) {
    return absl::InvalidArgumentError("socket_path is required");
  }
  bool expected = false;
  if (!running_.compare_exchange_strong(expected, true)) {
    return absl::OkStatus();
  }
  auto st = bind_and_listen_();
  if (!st.ok()) {
    running_.store(false);
    return st;
  }
  th_ = std::thread([this]() { this->run_loop_(); });
  return absl::OkStatus();
}

void LocalHandleServer::stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  if (listen_fd_ >= 0) {
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (th_.joinable()) {
    th_.join();
  }
  // Best-effort cleanup of socket path.
  std::error_code ec;
  (void)std::filesystem::remove(opts_.socket_path, ec);
}

absl::Status LocalHandleServer::bind_and_listen_() {
  auto st = validate_socket_parent_dir_(opts_.socket_path);
  if (!st.ok()) {
    return st;
  }
  st = remove_stale_socket_if_safe_(opts_.socket_path);
  if (!st.ok()) {
    return st;
  }

  const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, "socket(AF_UNIX) failed");
  }

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (opts_.socket_path.size() >= sizeof(addr.sun_path)) {
    ::close(fd);
    return absl::InvalidArgumentError("socket path too long for sockaddr_un");
  }
  std::strncpy(addr.sun_path, opts_.socket_path.c_str(), sizeof(addr.sun_path) - 1);

  if (::bind(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0) {
    const int err = errno;
    ::close(fd);
    return absl::ErrnoToStatus(err, absl::StrCat("bind failed for ", opts_.socket_path));
  }
  if (::chmod(opts_.socket_path.c_str(), 0600) < 0) {
    const int err = errno;
    ::close(fd);
    return absl::ErrnoToStatus(err, absl::StrCat("chmod(0600) failed for ", opts_.socket_path));
  }
  if (::listen(fd, 64) < 0) {
    const int err = errno;
    ::close(fd);
    return absl::ErrnoToStatus(err, "listen failed");
  }

  // Non-blocking accept loop (poll).
  int flags = ::fcntl(fd, F_GETFL);
  if (flags >= 0) {
    (void)::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }

  listen_fd_ = fd;
  return absl::OkStatus();
}

void LocalHandleServer::run_loop_() {
  while (running_.load()) {
    if (listen_fd_ < 0) {
      break;
    }
    pollfd pfd{.fd = listen_fd_, .events = POLLIN, .revents = 0};
    int rc = ::poll(&pfd, 1, 100);
    if (rc <= 0) {
      continue;
    }
    if ((pfd.revents & POLLIN) == 0) {
      continue;
    }
    for (;;) {
      int conn_fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
      if (conn_fd < 0) {
        if (errno == EINTR) {
          continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          break;
        }
        break;
      }
      auto st = handle_conn_(conn_fd);
      if (!st.ok()) {
        VLOG(2) << "LocalHandleServer: connection handling failed: " << st;
      }
      ::close(conn_fd);
    }
  }
}

absl::Status LocalHandleServer::handle_conn_(int conn_fd) {
  // Enforce peer credentials (uid==daemon uid).
  struct ucred cred{};
  socklen_t len = sizeof(cred);
  if (::getsockopt(conn_fd, SOL_SOCKET, SO_PEERCRED, &cred, &len) < 0) {
    return absl::ErrnoToStatus(errno, "getsockopt(SO_PEERCRED) failed");
  }
  const uid_t uid = ::geteuid();
  if (cred.uid != uid) {
    const uint8_t code = static_cast<uint8_t>(RespCode::kPermissionDenied);
    (void)write_exact_(conn_fd, &code, sizeof(code));
    return absl::PermissionDeniedError("peer uid mismatch");
  }

  (void)set_socket_timeouts(conn_fd, kConnIoTimeout);

  uint8_t opcode = 0;
  auto st = read_exact_(conn_fd, &opcode, sizeof(opcode));
  if (!st.ok()) {
    return st;
  }
  uint32_t token_len = 0;
  st = read_exact_(conn_fd, &token_len, sizeof(token_len));
  if (!st.ok()) {
    const uint8_t code = static_cast<uint8_t>(map_status_(st));
    (void)write_exact_(conn_fd, &code, sizeof(code));
    return st;
  }
  if (token_len == 0 || token_len > kMaxTokenBytes) {
    const auto bad = absl::InvalidArgumentError("invalid lease_token length");
    const uint8_t code = static_cast<uint8_t>(map_status_(bad));
    (void)write_exact_(conn_fd, &code, sizeof(code));
    return bad;
  }
  std::string token;
  token.resize(token_len);
  st = read_exact_(conn_fd, token.data(), token.size());
  if (!st.ok()) {
    const uint8_t code = static_cast<uint8_t>(map_status_(st));
    (void)write_exact_(conn_fd, &code, sizeof(code));
    return st;
  }

  switch (static_cast<OpCode>(opcode)) {
    case OpCode::kGetCpuMemfdFd: {
      if (!opts_.cpu_shared_memory_enabled) {
        const uint8_t code = static_cast<uint8_t>(RespCode::kFailedPrecondition);
        (void)write_exact_(conn_fd, &code, sizeof(code));
        return absl::FailedPreconditionError("cpu shared memory is disabled");
      }
      auto parsed_or =
          leases_->build_parsed_credential(token, LifecycleFrontDoorKind::kLocalCpuMemfdExport, absl::Now());
      if (!parsed_or.ok()) {
        const uint8_t code = static_cast<uint8_t>(map_status_(parsed_or.status()));
        (void)write_exact_(conn_fd, &code, sizeof(code));
        return parsed_or.status();
      }
      auto* lifecycle_kernel = leases_->lifecycle_kernel();
      if (lifecycle_kernel == nullptr) {
        const auto st = absl::FailedPreconditionError("lifecycle kernel is unavailable");
        const uint8_t code = static_cast<uint8_t>(map_status_(st));
        (void)write_exact_(conn_fd, &code, sizeof(code));
        return st;
      }
      auto admitted_or = lifecycle_kernel->admit_redemption(*parsed_or);
      if (!admitted_or.ok()) {
        const uint8_t code = static_cast<uint8_t>(map_status_(admitted_or.status()));
        (void)write_exact_(conn_fd, &code, sizeof(code));
        return admitted_or.status();
      }
      auto desc_or = leases_->get_cpu_memfd_descriptor(token);
      if (!desc_or.ok()) {
        auto release_status = lifecycle_kernel->release_use_guard(admitted_or->use_guard);
        LOG_IF(WARNING, !release_status.ok())
            << "LocalHandleServer: failed to release export use guard: " << release_status;
        const uint8_t code = static_cast<uint8_t>(map_status_(desc_or.status()));
        (void)write_exact_(conn_fd, &code, sizeof(code));
        return desc_or.status();
      }
      const auto desc = *desc_or;
      if (desc.fd < 0) {
        auto release_status = lifecycle_kernel->release_use_guard(admitted_or->use_guard);
        LOG_IF(WARNING, !release_status.ok())
            << "LocalHandleServer: failed to release export use guard: " << release_status;
        const uint8_t code = static_cast<uint8_t>(RespCode::kInternal);
        (void)write_exact_(conn_fd, &code, sizeof(code));
        return absl::InternalError("memfd is invalid");
      }
      auto send_status = send_fd_with_status(conn_fd, desc.fd, static_cast<uint8_t>(RespCode::kOk));
      auto release_status = lifecycle_kernel->release_use_guard(admitted_or->use_guard);
      LOG_IF(WARNING, !release_status.ok())
          << "LocalHandleServer: failed to release export use guard: " << release_status;
      return send_status;
    }
    case OpCode::kReleaseHandle: {
      auto rel = leases_->release(token);
      const uint8_t code = static_cast<uint8_t>(map_status_(rel));
      (void)write_exact_(conn_fd, &code, sizeof(code));
      return rel;
    }
    default: {
      const uint8_t code = static_cast<uint8_t>(RespCode::kInternal);
      (void)write_exact_(conn_fd, &code, sizeof(code));
      return absl::InvalidArgumentError("unknown opcode");
    }
  }
}

absl::Status LocalHandleServer::validate_socket_parent_dir_(const std::string& path) {
  std::filesystem::path p(path);
  const std::filesystem::path parent = p.parent_path();
  if (parent.empty()) {
    return absl::InvalidArgumentError("socket path must include a parent directory");
  }
  struct stat st{};
  if (::stat(parent.c_str(), &st) < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("stat failed for ", parent.string()));
  }
  if (!S_ISDIR(st.st_mode)) {
    return absl::InvalidArgumentError(absl::StrCat("socket parent is not a directory: ", parent.string()));
  }
  const uid_t uid = ::geteuid();
  if (st.st_uid != uid) {
    return absl::PermissionDeniedError(absl::StrCat("socket parent dir not owned by daemon uid: ", parent.string()));
  }
  // Fail closed on world-writable directories without sticky bit.
  if ((st.st_mode & S_IWOTH) != 0 && (st.st_mode & S_ISVTX) == 0) {
    return absl::PermissionDeniedError(
        absl::StrCat("socket parent dir is world-writable without sticky bit: ", parent.string()));
  }
  return absl::OkStatus();
}

absl::Status LocalHandleServer::remove_stale_socket_if_safe_(const std::string& path) {
  struct stat st{};
  if (::lstat(path.c_str(), &st) < 0) {
    if (errno == ENOENT) {
      return absl::OkStatus();
    }
    return absl::ErrnoToStatus(errno, absl::StrCat("lstat failed for ", path));
  }
  if (!S_ISSOCK(st.st_mode)) {
    return absl::FailedPreconditionError(absl::StrCat("existing path is not a socket: ", path));
  }
  const uid_t uid = ::geteuid();
  if (st.st_uid != uid) {
    return absl::PermissionDeniedError(absl::StrCat("existing socket not owned by daemon uid: ", path));
  }
  if (::unlink(path.c_str()) < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("unlink failed for ", path));
  }
  return absl::OkStatus();
}

absl::Status LocalHandleServer::read_exact_(int fd, void* buf, size_t n) {
  auto* p = static_cast<uint8_t*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t rc = ::read(fd, p + off, n - off);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return absl::DeadlineExceededError("read timed out");
      }
      return absl::ErrnoToStatus(errno, "read failed");
    }
    if (rc == 0) {
      return absl::UnavailableError("peer closed connection");
    }
    off += static_cast<size_t>(rc);
  }
  return absl::OkStatus();
}

absl::Status LocalHandleServer::write_exact_(int fd, const void* buf, size_t n) {
  const auto* p = static_cast<const uint8_t*>(buf);
  size_t off = 0;
  while (off < n) {
    ssize_t rc = ::write(fd, p + off, n - off);
    if (rc < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return absl::DeadlineExceededError("write timed out");
      }
      return absl::ErrnoToStatus(errno, "write failed");
    }
    off += static_cast<size_t>(rc);
  }
  return absl::OkStatus();
}

LocalHandleServer::RespCode LocalHandleServer::map_status_(const absl::Status& st) {
  if (st.ok()) {
    return RespCode::kOk;
  }
  if (absl::IsNotFound(st)) {
    return RespCode::kNotFound;
  }
  if (absl::IsPermissionDenied(st)) {
    return RespCode::kPermissionDenied;
  }
  if (absl::IsFailedPrecondition(st)) {
    return RespCode::kFailedPrecondition;
  }
  return RespCode::kInternal;
}

} // namespace tensorcast::daemon
