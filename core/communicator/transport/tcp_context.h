
// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TRANSPORT_TCP_CONTEXT_H_
#define CORE_COMMUNICATOR_TRANSPORT_TCP_CONTEXT_H_

#include <memory>
#include <string>
#include <thread>

#include "absl/status/statusor.h"

#include "core/communicator/misc/common.h"
#include "core/communicator/transport/tcp_transport.h"

namespace tensorcast::communicator {

typedef std::function<result_t(tcp_transport_t)> on_accept_func_t;

class TcpContext {
 public:
  TcpContext();
  ~TcpContext();

  result_t open(const std::string& ip, uint16_t port, on_accept_func_t func);
  absl::StatusOr<tcp_transport_t> connect(const std::string& ip, uint16_t port);

  std::string get_local_ip() const;

  // Typed configuration injection
  void set_connect_timeout(int seconds) {
    connect_timeout_sec_ = seconds;
  }

 protected:
  result_t register_transport(TcpTransport* t);
  result_t unregister_transport(TcpTransport* t);

 private:
  void listen_event_loop();
  void recv_event_loop();
  void do_accept();

 private:
  int listen_epoll_fd_;
  int recv_epoll_fd_;
  int listen_fd_{};
  std::thread listen_thread_;
  std::thread recv_thread_;
  mutable std::mutex mu_;
  std::atomic_bool stop_;
  std::condition_variable cv_;
  struct sockaddr_in local_addr_{};
  on_accept_func_t on_accept_;
  int connect_timeout_sec_ = 10;

  friend class TcpTransport;
};
typedef std::shared_ptr<TcpContext> tcp_context_t;

} // namespace tensorcast::communicator

#endif // CORE_COMMUNICATOR_TRANSPORT_TCP_CONTEXT_H_
