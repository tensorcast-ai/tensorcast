// Copyright (c) 2025-2026, TensorCast Team.

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

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/communicator/base/constants.h"
#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/engine/staging_flow_controller.h"
#include "core/communicator/misc/common.h"
#include "core/communicator/misc/metric.h"
#include "core/communicator/transport/request.h"
#include "gsl/pointers"

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
  MTcpTransport(
      int conn_count,
      gsl::not_null<std::shared_ptr<engine::MemoryStager>> memory_stager,
      gsl::not_null<std::shared_ptr<engine::MemoryStager>> gpu_memory_stager,
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool,
      int buffers_per_flow_limit);
  ~MTcpTransport();

  misc::result_t listen(const std::string& ip, uint16_t* port);
  misc::result_t connect(const std::string& dst_ip, uint16_t port, int retry);

  misc::result_t send(const write_request_t& request);
  misc::result_t recv(const read_request_t& request);

  struct StageSendSegment {
    void* data = nullptr;
    size_t bytes = 0;
    engine::StageLease::Metadata metadata;
    std::function<void(misc::result_t)> on_complete;
  };

  struct StageSendWindow {
    std::string request_key;
    uint32_t window_seq = 0;
    bool final_window = false;
    uint64_t total_bytes = 0;
    uint64_t stage_unit_bytes = 0;
    std::vector<StageSendSegment> segments;
    std::shared_ptr<std::atomic<int>> pending_segments;
    std::function<void()> on_window_complete;
  };

  void enqueue_stage_window(StageSendWindow window);
  void release_receive_resources();
  absl::Status ensure_gpu_buffer_ready();

  void set_conn_count(int conn_count);

 private:
  void server_loop();
  void client_loop();

  void send_loop();
  void staged_send_loop();
  void process_stage_window(const std::shared_ptr<StageSendWindow>& window);
  void track_async_task(std::shared_future<chunk_result_t> future, std::function<void(const chunk_result_t&)> on_ready);
  void fail_pending_async_tasks(misc::result_t status);
  [[nodiscard]] bool has_outstanding_async_tasks() const;
  void prune_async_tasks();
  void recv_loop();
  misc::result_t init_socket_fd(int sock_fd);

  void start_staged_thread();

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
  std::thread staged_thread_;

  std::atomic_bool stop_;
  std::atomic_bool closed_;
  std::atomic_bool ready_;

  task_t tasks_[base::kMaxTcpConns];

  // Unified GPU MemoryStager (required)
  gsl::not_null<std::shared_ptr<engine::MemoryStager>> gpu_memory_stager_;

  // GPU receive buffer management
  gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool_;
  gsl::not_null<std::shared_ptr<common::memory::StreamingPinnedBuffer>> gpu_recv_buffer_;

  std::atomic<bool> gpu_buffer_ready_{false};
  mutable std::mutex gpu_buffer_mutex_;
  const std::chrono::milliseconds gpu_init_retry_timeout_;

  int buffers_per_flow_limit_;

  // Unified memory stager for CPU staging in TCP path (required)
  gsl::not_null<std::shared_ptr<engine::MemoryStager>> memory_stager_;

  // Track outstanding async tasks for proper cleanup
  struct TrackedAsyncTask {
    std::shared_future<chunk_result_t> future;
    std::function<void(const chunk_result_t&)> on_ready;
  };

  mutable std::mutex async_tasks_mutex_;
  std::vector<TrackedAsyncTask> outstanding_async_tasks_;

  misc::Queue<std::shared_ptr<StageSendWindow>> staged_queue_;
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
