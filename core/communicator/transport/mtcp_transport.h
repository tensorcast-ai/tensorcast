// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_TRANSPORT_MTCP_TRANSPORT_H_
#define CORE_COMMUNICATOR_TRANSPORT_MTCP_TRANSPORT_H_

extern "C" {
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <strings.h>
#include <sys/fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
}

#include <memory>
#include <mutex>
#include <string>

#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/communicator/misc/common.h"
#include "core/communicator/misc/metric.h"
#include "core/communicator/transport/request.h"

namespace stepcast::communicator {

// Forward declaration
class GpuTcpStager;

constexpr int kMaxTcpConns = 32;
constexpr int kMaxFd = 32;

using chunk_result_t = struct MTcpTransportChunkResult {
  result_t status = SUCCESS;
  uint64_t cost = 0;
} __attribute__((aligned(16)));

using future_chunk_result_t = std::future<chunk_result_t>;

class MTcpTransportChunk {
 public:
  MTcpTransportChunk(uint8_t* addr, uint64_t len);
  ~MTcpTransportChunk() = default;

  future_chunk_result_t get_future() {
    return result_.get_future();
  }

  template <class T>
  T* addr() {
    return reinterpret_cast<T*>(addr_);
  }

  [[nodiscard]] uint64_t len() const {
    return len_;
  }

  void set_result(result_t status) {
    result_.set_value(
        chunk_result_t{
            .status = status,
            .cost = timer_.record(),
        });
  }

 private:
  uint8_t* addr_;
  uint64_t len_;
  std::promise<chunk_result_t> result_;
  Timer timer_;
};
using chunk_t = std::shared_ptr<MTcpTransportChunk>;

class MTcpTransportTask {
 public:
  explicit MTcpTransportTask(int sock_fd);
  ~MTcpTransportTask();

  void start();
  void stop();

  void push_send(const chunk_t& chunk);
  void push_recv(const chunk_t& chunk);

 private:
  void send_loop();
  void recv_loop();
  result_t do_recv_bytes(uint8_t* buf, int size) const;
  result_t do_send_bytes(uint8_t* buf, int size) const;

  int sock_fd_;
  std::thread recv_thread_;
  std::thread send_thread_;
  std::atomic_bool stop_;
  Queue<chunk_t> send_queue_;
  Queue<chunk_t> recv_queue_;
};
using task_t = std::shared_ptr<MTcpTransportTask>;

class MTcpTransport : public std::enable_shared_from_this<MTcpTransport> {
 public:
  explicit MTcpTransport(int conn_count);
  ~MTcpTransport();

  result_t listen(const std::string& ip, uint16_t* port);
  result_t connect(const std::string& dst_ip, uint16_t port, int retry);

  result_t send(const write_request_t& request);
  result_t recv(const read_request_t& request);

  void set_conn_count(int conn_count);
  void set_gpu_tcp_stager(std::shared_ptr<GpuTcpStager> stager);
  void set_memory_pool(std::shared_ptr<stepcast::store::PinnedMemoryPool> pool);

 private:
  void server_loop();
  void client_loop();

  void send_loop();
  void recv_loop();

  static result_t init_socket_fd(int sock_fd);

  int listen_fd_ = 0;
  int retry_count_;
  int sock_fds_[kMaxFd] = {0};
  int conn_count_ = 1;
  int max_retry_ = 10;
  struct sockaddr_in server_addr_;
  struct sockaddr_in client_addrs_[kMaxTcpConns];

  Queue<write_request_t> send_queue_;
  Queue<read_request_t> recv_queue_;

  std::thread recv_thread_;
  std::thread send_thread_;

  std::atomic_bool stop_;
  std::atomic_bool closed_;
  std::atomic_bool ready_;

  task_t tasks_[kMaxTcpConns];

  // GPU->CPU staging for TCP transport
  std::shared_ptr<GpuTcpStager> gpu_tcp_stager_;

  // GPU receive buffer management
  std::shared_ptr<stepcast::store::PinnedMemoryPool> memory_pool_;
  std::unique_ptr<stepcast::store::StreamingPinnedBuffer> gpu_recv_buffer_;

  // Track outstanding async tasks for proper cleanup
  mutable std::mutex async_tasks_mutex_;
  std::vector<std::shared_future<chunk_result_t>> outstanding_async_tasks_;
};
using mtcp_transport_t = std::shared_ptr<MTcpTransport>;

} // namespace stepcast::communicator

#endif // CORE_COMMUNICATOR_TRANSPORT_MTCP_TRANSPORT_H_
