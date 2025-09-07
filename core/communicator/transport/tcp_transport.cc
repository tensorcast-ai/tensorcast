
// Copyright (c) 2025, TensorCast Team.

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <mutex>

#include "absl/log/log.h"

#include "core/communicator/misc/epoll_wrap.h" // IWYU pragma: keep
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/base_transport.h"
#include "core/communicator/transport/tcp_context.h"
#include "core/communicator/transport/tcp_transport.h"

namespace tensorcast::communicator::transport {

// Helper function to decode epoll event flags into a readable string
static std::string decode_epoll_events(uint32_t events) {
  std::stringstream ss;
  ss << "0x" << std::hex << std::setw(4) << std::setfill('0') << events << " [";

  bool first = true;
  auto add_flag = [&](const char* name) {
    if (!first)
      ss << " | ";
    ss << name;
    first = false;
  };

  if (events & EPOLLIN)
    add_flag("EPOLLIN");
  if (events & EPOLLPRI)
    add_flag("EPOLLPRI");
  if (events & EPOLLOUT)
    add_flag("EPOLLOUT");
  if (events & EPOLLERR)
    add_flag("EPOLLERR");
  if (events & EPOLLHUP)
    add_flag("EPOLLHUP");
  if (events & EPOLLRDHUP)
    add_flag("EPOLLRDHUP");
  if (events & EPOLLONESHOT)
    add_flag("EPOLLONESHOT");
  if (events & EPOLLET)
    add_flag("EPOLLET");

  ss << "]";
  return ss.str();
}

TcpTransport::TcpTransport(TcpContext* context, int fd, struct sockaddr_in remote_addr)
    : context_(context), fd_(fd), local_addr_(), remote_addr_(remote_addr), recv_func_(nullptr) {
  misc::ASSERT(fd_ != 0, "failed to init tcp transport");
  misc::CLEAR(local_addr_);
  socklen_t len = sizeof(local_addr_);
  if (getsockname(fd_, reinterpret_cast<struct sockaddr*>(&local_addr_), &len) < 0) {
    PLOG(WARNING) << "[TcpTransport::TcpTransport] getsockname failed for fd=" << fd_;
    misc::CLEAR(local_addr_);
  }

  CHECK_WARN(context_->register_transport(this), "failed to register transport");
}

TcpTransport::~TcpTransport() {
  if (context_ != nullptr) {
    close();
  }
}

misc::result_t TcpTransport::close() {
  if (context_ == nullptr) {
    return misc::SUCCESS;
  }
  LOG(INFO) << "[TcpTransport::close] Closing transport " << get_remote_url();
  context_->unregister_transport(this);
  recv_func_ = nullptr;
  close_func_ = nullptr;
  if (fd_ != 0) {
    ::close(fd_);
    fd_ = 0;
  }
  context_ = nullptr;
  return misc::SUCCESS;
}

misc::result_t TcpTransport::send(const transport_message_t& msg) {
  std::unique_lock<std::mutex> lock(mu_);
  COMM_CHECK(send_bytes(fd_, msg->get_header<uint8_t>(), msg->get_header_size()));
  COMM_CHECK(send_bytes(fd_, msg->get_payload<uint8_t>(), msg->get_payload_size()));
  return misc::SUCCESS;
}

misc::result_t TcpTransport::recv(uint8_t* buf, uint32_t buf_size) {
  std::unique_lock<std::mutex> lock(mu_);
  COMM_CHECK(recv_bytes(fd_, buf, buf_size));
  return misc::SUCCESS;
}

void TcpTransport::set_recv_func(on_recv_func_t recv_func) {
  recv_func_ = std::move(recv_func);
}

void TcpTransport::set_close_func(on_close_func_t recv_func) {
  close_func_ = std::move(recv_func);
}

misc::result_t TcpTransport::process_event(uint32_t event) {
  // Log the full event for debugging
  LOG(INFO) << "[TcpTransport::process_event] Processing event for " << get_remote_url() << " - "
            << decode_epoll_events(event);

  // Handle combined events by checking individual flags
  if (event & EPOLLERR) {
    PLOG(ERROR) << "[TcpTransport::process_event] EPOLLERR detected for "
                << "peer=" << get_remote_url() << ", local=" << get_local_url() << " - socket error occurred";
    return do_close();
  }

  if (event & EPOLLHUP) {
    LOG(ERROR) << "[TcpTransport::process_event] EPOLLHUP detected for " << get_remote_url() << " - socket hung up";
    return do_close();
  }

  if (event & EPOLLRDHUP) {
    LOG(ERROR) << "[TcpTransport::process_event] EPOLLRDHUP detected for " << get_remote_url()
               << " - remote peer closed connection";
    return do_close();
  }

  if (event & EPOLLIN) {
    // Only process EPOLLIN if no error conditions are present
    if (!(event & (EPOLLERR | EPOLLHUP | EPOLLRDHUP))) {
      return do_recv();
    }
    // If EPOLLIN is combined with error events, log it but close
    LOG(WARNING) << "[TcpTransport::process_event] EPOLLIN with error conditions for " << get_remote_url()
                 << " - closing connection due to error flags";
    return do_close();
  }

  if (event & EPOLLOUT) {
    VLOG(1) << "[TcpTransport::process_event] EPOLLOUT for " << get_remote_url() << " - no pending writes; ignored";
    return misc::IN_PROGRESS;
  }

  if (event & EPOLLPRI) {
    VLOG(1) << "[TcpTransport::process_event] EPOLLPRI (OOB data) for " << get_remote_url() << " - ignored";
    return misc::IN_PROGRESS;
  }

  // Unexpected event
  LOG(WARNING) << "[TcpTransport::process_event] Unexpected event for " << get_remote_url() << " - "
               << decode_epoll_events(event) << "; ignoring";
  return misc::IN_PROGRESS;
}

int TcpTransport::get_fd() const {
  return fd_;
}

std::string TcpTransport::get_remote_url() const {
  char addr_buf[INET_ADDRSTRLEN] = {0};
  const char* res = inet_ntop(AF_INET, &remote_addr_.sin_addr, addr_buf, sizeof(addr_buf));
  std::stringstream url;
  url << (res ? addr_buf : "0.0.0.0") << ":" << ntohs(remote_addr_.sin_port);
  return url.str();
}

std::string TcpTransport::get_local_url() const {
  char addr_buf[INET_ADDRSTRLEN] = {0};
  const char* res = inet_ntop(AF_INET, &local_addr_.sin_addr, addr_buf, sizeof(addr_buf));
  std::stringstream url;
  url << (res ? addr_buf : "0.0.0.0") << ":" << ntohs(local_addr_.sin_port);
  return url.str();
}

misc::result_t TcpTransport::do_close() {
  if (close_func_ == nullptr) {
    return misc::IN_PROGRESS;
  }
  COMM_CHECK(close_func_(shared_from_this()));
  return misc::SUCCESS;
}

misc::result_t TcpTransport::do_recv() {
  if (recv_func_ == nullptr) {
    return misc::IN_PROGRESS;
  }

  COMM_CHECK(recv_func_(shared_from_this()));
  return misc::SUCCESS;
}

} // namespace tensorcast::communicator::transport
