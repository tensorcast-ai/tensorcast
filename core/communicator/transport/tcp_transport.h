// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TRANSPORT_TCP_TRANSPORT_H_
#define CORE_COMMUNICATOR_TRANSPORT_TCP_TRANSPORT_H_

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

#include "core/communicator/misc/common.h"
#include "core/communicator/misc/map.h"
#include "core/communicator/transport/rdma_transport.h"
#include "core/communicator/transport/transport_message.h"

namespace tensorcast::communicator::transport {

class TcpContext;
class TcpTransport;

typedef std::shared_ptr<TcpTransport> tcp_transport_t;

typedef std::function<misc::result_t(tcp_transport_t)> on_recv_func_t;
typedef std::function<misc::result_t(tcp_transport_t)> on_close_func_t;

class TcpTransport : public std::enable_shared_from_this<TcpTransport> {
 public:
  TcpTransport(TcpContext* context, int fd, struct sockaddr_in remote_addr);
  ~TcpTransport();

  misc::result_t send(const transport_message_t& msg);
  misc::result_t recv(uint8_t* buf, uint32_t buf_size);

  template <class T>
  misc::result_t recv(T* data) {
    return recv(reinterpret_cast<uint8_t*>(data), sizeof(T));
  }

  void set_recv_func(on_recv_func_t recv_func);
  void set_close_func(on_close_func_t recv_func);

  misc::result_t process_event(uint32_t events);

  int get_fd() const;

  std::string get_remote_url() const;
  std::string get_local_url() const;

  misc::result_t close();

 private:
  misc::result_t do_recv();
  misc::result_t do_close();

 private:
  TcpContext* context_;
  int fd_;
  struct sockaddr_in local_addr_;
  struct sockaddr_in remote_addr_;
  on_recv_func_t recv_func_;
  on_close_func_t close_func_;
  mutable std::mutex mu_;
};

} // namespace tensorcast::communicator::transport

#endif // CORE_COMMUNICATOR_TRANSPORT_TCP_TRANSPORT_H_
