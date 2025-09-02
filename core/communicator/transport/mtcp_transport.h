// Copyright (c) 2025, TensorCast Team.

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
#include "core/communicator/base/constants.h"
#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/misc/common.h"
#include "core/communicator/misc/metric.h"
#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::transport {

using chunk_result_t = struct MTcpTransportChunkResult {
  misc::result_t status = misc::SUCCESS;
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

  void set_result(misc::result_t status) {
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
  misc::Timer timer_;
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
  misc::result_t do_recv_bytes(uint8_t* buf, int size) const;
  misc::result_t do_send_bytes(uint8_t* buf, int size) const;

  int sock_fd_;
  std::thread recv_thread_;
  std::thread send_thread_;
  std::atomic_bool stop_;
  misc::Queue<chunk_t> send_queue_;
  misc::Queue<chunk_t> recv_queue_;
};
using task_t = std::shared_ptr<MTcpTransportTask>;

class MTcpTransport : public std::enable_shared_from_this<MTcpTransport> {
 public:
  explicit MTcpTransport(int conn_count);
  ~MTcpTransport();

  misc::result_t listen(const std::string& ip, uint16_t* port);
  misc::result_t connect(const std::string& dst_ip, uint16_t port, int retry);

  misc::result_t send(const write_request_t& request);
  misc::result_t recv(const read_request_t& request);

  void set_conn_count(int conn_count);
  void set_memory_pool(std::shared_ptr<common::memory::PinnedMemoryPool> pool);
  void set_memory_stager(std::shared_ptr<engine::MemoryStager> stager) {
    memory_stager_ = std::move(stager);
  }
  void set_gpu_memory_stager(std::shared_ptr<engine::MemoryStager> stager) {
    gpu_memory_stager_ = std::move(stager);
  }

 private:
  void server_loop();
  void client_loop();

  void send_loop();
  void recv_loop();
  misc::result_t init_socket_fd(int sock_fd);

  int listen_fd_ = 0;
  int retry_count_;
  int sock_fds_[base::kMaxFd] = {0};
  int conn_count_ = 1;
  int max_retry_ = 10;
  struct sockaddr_in server_addr_;
  struct sockaddr_in client_addrs_[base::kMaxTcpConns];

  misc::Queue<write_request_t> send_queue_;
  misc::Queue<read_request_t> recv_queue_;

  std::thread recv_thread_;
  std::thread send_thread_;

  std::atomic_bool stop_;
  std::atomic_bool closed_;
  std::atomic_bool ready_;

  task_t tasks_[base::kMaxTcpConns];

  // Unified GPU MemoryStager (required)
  std::shared_ptr<engine::MemoryStager> gpu_memory_stager_;

  // GPU receive buffer management
  std::shared_ptr<common::memory::PinnedMemoryPool> memory_pool_;
  std::unique_ptr<common::memory::StreamingPinnedBuffer> gpu_recv_buffer_;

  // Unified memory stager for CPU staging in TCP path (required)
  std::shared_ptr<engine::MemoryStager> memory_stager_;

  // Track outstanding async tasks for proper cleanup
  mutable std::mutex async_tasks_mutex_;
  std::vector<std::shared_future<chunk_result_t>> outstanding_async_tasks_;

  // Socket tuning (typed-config): IP_TOS value; 0 to leave unchanged
  int tcp_tos_ = 0;

 public:
  void set_tcp_tos(int tos) {
    tcp_tos_ = tos;
  }
};
using mtcp_transport_t = std::shared_ptr<MTcpTransport>;

} // namespace tensorcast::communicator::transport

#endif // CORE_COMMUNICATOR_TRANSPORT_MTCP_TRANSPORT_H_
