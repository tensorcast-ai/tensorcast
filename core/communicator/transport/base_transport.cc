// Copyright (c) 2025, TensorCast Team.

extern "C" {
#include <sys/socket.h>
}

#include "core/communicator/transport/base_transport.h"

namespace tensorcast::communicator {

result_t send_bytes(int sock_fd, uint8_t* buf, int size) {
  ssize_t remain_bytes = size;
  ssize_t offset = 0;
  ssize_t bytes;
  while (remain_bytes > 0) {
    bytes = ::send(sock_fd, buf + offset, remain_bytes, 0);
    if (bytes <= 0) {
      return SYS_ERROR;
    }

    if (bytes < remain_bytes) {
      remain_bytes -= bytes;
      offset += bytes;
    } else {
      remain_bytes = 0;
    }
  }
  return SUCCESS;
}

result_t recv_bytes(int sock_fd, uint8_t* buf, int size) {
  ssize_t remain_bytes = size;
  ssize_t offset = 0;
  ssize_t bytes;
  while (remain_bytes > 0) {
    bytes = ::recv(sock_fd, buf + offset, remain_bytes, 0);
    if (bytes <= 0) {
      return SYS_ERROR;
    }
    if (bytes < remain_bytes) {
      remain_bytes -= bytes;
      offset += bytes;
    } else {
      remain_bytes = 0;
    }
  }
  return SUCCESS;
}

} // namespace tensorcast::communicator
