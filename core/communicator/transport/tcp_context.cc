// Copyright (c) 2025, TensorCast Team.

extern "C" {
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <strings.h>
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
}

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"

#include "core/communicator/misc/epoll_wrap.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/tcp_context.h"

namespace tensorcast::communicator {

constexpr static int kTcpContextBatchSize = 16;

TcpContext::TcpContext() : stop_(false) {
  listen_epoll_fd_ = wrap_epoll_create(1);
  recv_epoll_fd_ = wrap_epoll_create(1);
  recv_thread_ = std::thread([this]() { this->recv_event_loop(); });
}

TcpContext::~TcpContext() {
  stop_.store(true);
  if (listen_fd_ != 0) {
    ::close(listen_epoll_fd_);
    ::close(listen_fd_);
    listen_fd_ = 0;
  }

  if (recv_epoll_fd_ != 0) {
    ::close(recv_epoll_fd_);
    recv_epoll_fd_ = 0;
  }

  if (listen_thread_.joinable()) {
    listen_thread_.join();
  }

  if (recv_thread_.joinable()) {
    recv_thread_.join();
  }
}

result_t TcpContext::open(const std::string& ip, uint16_t port, on_accept_func_t func) {
  ASSERT(listen_fd_ == 0, "failed to open due to valid listen fd");
  on_accept_ = std::move(func);
  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ == -1) {
    return SYS_ERROR;
  }

  // Enable address/port reuse to avoid EADDRINUSE due to TIME_WAIT and reduce bind conflicts
  int one = 1;
  if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0) {
    PLOG(WARNING) << "failed to set SO_REUSEADDR";
  }

  if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one)) != 0) {
    PLOG(WARNING) << "failed to set SO_REUSEPORT";
  }

  CLEAR(local_addr_);
  local_addr_.sin_family = AF_INET;
  local_addr_.sin_addr.s_addr = inet_addr(ip.c_str());
  local_addr_.sin_port = htons(port);

  int ret = ::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&local_addr_), sizeof(local_addr_));
  if (ret < 0) {
    return SYS_ERROR;
  }

  ret = ::listen(listen_fd_, 1024);
  if (ret < 0) {
    return SYS_ERROR;
  }

  CLEAR(local_addr_);
  socklen_t len = sizeof(local_addr_);
  getsockname(listen_fd_, (struct sockaddr*)&local_addr_, &len);

  listen_thread_ = std::thread([this]() { this->listen_event_loop(); });

  epoll_event ev{};
  bzero(&ev, sizeof(epoll_event));
  ev.events = EPOLLIN | EPOLLRDHUP | EPOLLERR | EPOLLHUP;
  ev.data.ptr = nullptr;

  ret = wrap_epoll_ctl(listen_epoll_fd_, EPOLL_CTL_ADD, listen_fd_, &ev);
  if (ret != 0) {
    LOG(WARNING) << "failed to add epoll: ret=" << ret << " error" << strerror(errno);
    return ret;
  }

  return SUCCESS;
}

absl::StatusOr<tcp_transport_t> TcpContext::connect(const std::string& ip, uint16_t port) {
  struct sockaddr_in remote_addr = {};
  remote_addr.sin_family = AF_INET;
  remote_addr.sin_port = htons(port);
  remote_addr.sin_addr.s_addr = inet_addr(ip.c_str());
  int addr_len = sizeof(remote_addr);
  auto sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd == -1) {
    return absl::InvalidArgumentError(
        absl::StrFormat("failed to connect to %s:%d, return=%d, error=%s", ip, port, sock_fd, strerror(errno)));
  }

  struct timeval timeo;
  timeo.tv_sec = connect_timeout_sec_;
  timeo.tv_usec = 0;

  int ret = setsockopt(sock_fd, SOL_SOCKET, SO_SNDTIMEO, &timeo, sizeof(timeo));
  CHECK_EQ(ret, 0) << "Failed to send connect timeout: error=" << strerror(errno);

  ret = ::connect(sock_fd, reinterpret_cast<struct sockaddr*>(&remote_addr), (socklen_t)addr_len);
  if (ret < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("failed to connect to %s:%d, return=%d, error=%s", ip, port, ret, strerror(errno)));
  }

  return std::make_shared<TcpTransport>(this, sock_fd, remote_addr);
}

std::string TcpContext::get_local_ip() const {
  if (local_addr_.sin_addr.s_addr == INADDR_ANY) {
    return get_default_ip();
  }
  return {inet_ntoa(local_addr_.sin_addr)};
}

result_t TcpContext::register_transport(TcpTransport* t) {
  epoll_event ev{};
  bzero(&ev, sizeof(epoll_event));
  ev.events = EPOLLIN | EPOLLRDHUP | EPOLLERR | EPOLLHUP;
  ev.data.ptr = t;

  auto ret = wrap_epoll_ctl(recv_epoll_fd_, EPOLL_CTL_ADD, t->get_fd(), &ev);
  if (ret != 0) {
    LOG(WARNING) << "failed to add epoll: ret=" << ret << " " << strerror(errno);
    return ret;
  }

  return SUCCESS;
}

result_t TcpContext::unregister_transport(TcpTransport* t) {
  int ret = wrap_epoll_ctl(recv_epoll_fd_, EPOLL_CTL_DEL, t->get_fd(), nullptr);
  if (ret != 0) {
    LOG(WARNING) << "failed to delete listen epoll: ret=" << ret << " " << strerror(errno);
    return ret;
  }

  return SUCCESS;
}

void TcpContext::listen_event_loop() {
  int num_fd = 0;
  epoll_event events[kTcpContextBatchSize];
  CLEAR_PTR(events, kTcpContextBatchSize * sizeof(epoll_event));
  while (!stop_.load()) {
    num_fd = wrap_epoll_wait(listen_epoll_fd_, events, kTcpContextBatchSize, 1000);
    for (int i = 0; i < num_fd; i++) {
      auto ev = &events[i];
      switch (ev->events) {
        case EPOLLIN:
          do_accept();
          continue;
        case EPOLLERR:
          LOG(ERROR) << "[listen_event_loop] " << "EPOLLERR err " << strerror(errno);
          continue;
        case EPOLLHUP:
          LOG(ERROR) << "[listen_event_loop] " << "EPOLLHUP err " << strerror(errno);
          continue;
        default:
          LOG(ERROR) << "[listen_event_loop] " << "default err " << strerror(errno) << " " << ev->events;
          continue;
      }
    }
  }
}

void TcpContext::do_accept() {
  int sock_fd = 0;
  struct sockaddr_in remote_addr = {};
  int addr_len = sizeof(remote_addr);
  CLEAR(remote_addr);

  if (listen_fd_ == 0) {
    LOG(ERROR) << "[do_accept] " << "failed to do accept due to invalid listen_fd_";
    return;
  }

  sock_fd =
      accept(listen_fd_, reinterpret_cast<struct sockaddr*>(&remote_addr), reinterpret_cast<socklen_t*>(&addr_len));

  // accept failed
  if (sock_fd <= 0) {
    LOG(ERROR) << "[do_accept] " << "failed to do accept " << strerror(errno);
    return;
  }

  on_accept_(std::make_shared<TcpTransport>(this, sock_fd, remote_addr));
}

void TcpContext::recv_event_loop() {
  int num_fd = 0;
  epoll_event events[kTcpContextBatchSize];
  CLEAR_PTR(events, kTcpContextBatchSize * sizeof(epoll_event));
  while (!stop_.load()) {
    num_fd = wrap_epoll_wait(recv_epoll_fd_, events, kTcpContextBatchSize, 10);
    for (int i = 0; i < num_fd; i++) {
      auto ev = &events[i];
      if (ev->data.ptr != nullptr) {
        auto t = reinterpret_cast<TcpTransport*>(ev->data.ptr);
        t->process_event(ev->events);
      } else {
        LOG(WARNING) << "failed to process event due to nil user data";
      }
    }
  }
}

} // namespace tensorcast::communicator
