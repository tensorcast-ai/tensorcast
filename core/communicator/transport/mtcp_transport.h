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
#include "absl/status/statusor.h"
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

  void set_on_complete(std::function<void(misc::result_t)> cb) {
    on_complete_ = std::move(cb);
  }

  void set_result(misc::result_t status) {
    result_.set_value(
        chunk_result_t{
            .status = status,
            .cost = timer_.record(),
        });
    // Fire the completion hook (if any) on the thread that finished the recv/send
    // (the per-lane reader thread for recv). The recv path uses this to hand the
    // staging slot to the H2D consumer without an extra waiter thread.
    if (on_complete_) {
      on_complete_(status);
    }
  }

 private:
  uint8_t* addr_;
  uint64_t len_;
  std::promise<chunk_result_t> result_;
  std::function<void(misc::result_t)> on_complete_;
  misc::Timer timer_;
};

using chunk_t = std::shared_ptr<MTcpTransportChunk>;

// Tracks completion of all sub-chunks belonging to a single MTCP read request.
//
// The recv path is split into a network *producer* (recv_loop + per-lane reader
// threads) and an H2D *consumer* thread; sub-chunks of one request complete on
// different threads and possibly out of order. This object is the single place
// that accounts for a request: the producer issues sub-chunks (`add_chunk`) and
// then `seal`s the request; the consumer (or the lane thread, for CPU tensors)
// reports each sub-chunk via `on_chunk_done`. The thread whose action brings the
// outstanding count to zero finalizes the request exactly once.
//
// Concurrency: a single atomic counter (seeded with a producer "issuing" token
// that `seal` removes) avoids the classic check-then-act race between the last
// completing chunk and `seal`. No locks on the hot path.
class MTcpRecvRequestState {
 public:
  explicit MTcpRecvRequestState(read_request_t msg) : msg_(std::move(msg)), outstanding_(1) {}

  // Producer: called once per issued sub-chunk, before handing it to a lane.
  void add_chunk() {
    outstanding_.fetch_add(1, std::memory_order_acq_rel);
  }

  // Producer: mark the request failed independently of a chunk completion
  // (e.g. invalid lane / slot-acquire failure during issue).
  void mark_failed() {
    failed_.store(true, std::memory_order_release);
  }

  // Producer: called after all sub-chunks have been issued (or issuing aborted).
  // Removes the issuing token; may finalize if all chunks already completed.
  void seal() {
    complete_one(misc::SUCCESS, /*bytes=*/0, /*is_token=*/true);
  }

  // Consumer / lane thread: called exactly once per issued sub-chunk.
  void on_chunk_done(misc::result_t status, uint64_t bytes) {
    complete_one(status, bytes, /*is_token=*/false);
  }

  [[nodiscard]] bool finalized() const {
    return finalized_.load(std::memory_order_acquire);
  }

  // Valid once finalized(): true if any sub-chunk failed.
  [[nodiscard]] bool result_failed() const {
    return failed_.load(std::memory_order_acquire);
  }

  [[nodiscard]] const read_request_t& msg() const {
    return msg_;
  }

 private:
  void complete_one(misc::result_t status, uint64_t bytes, bool is_token) {
    if (!is_token) {
      if (status != misc::SUCCESS) {
        failed_.store(true, std::memory_order_release);
      } else if (bytes > 0 && msg_) {
        msg_->mark_completion_and_is_done(bytes);
      }
    }
    // The decrement that returns 1 (i.e. transitions the count to 0) is the sole
    // finalizer. RMWs on the same atomic form a total order, so this thread sees
    // every prior failed_ store.
    if (outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      finalized_.store(true, std::memory_order_release);
      if (msg_) {
        msg_->set_result(
            failed_.load(std::memory_order_acquire) ? absl::InternalError("failed to read chunk") : absl::OkStatus());
      }
    }
  }

  read_request_t msg_;
  std::atomic<int64_t> outstanding_;
  std::atomic_bool failed_{false};
  std::atomic_bool finalized_{false};
};

using recv_request_state_t = std::shared_ptr<MTcpRecvRequestState>;

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

  // Receive-side H2D consumer: drains the ready queue produced by the network
  // recv path and performs the staged H2D copy on a single dedicated thread,
  // decoupling slot lifetime from the producer so a slow H2D never blocks socket
  // draining. Started lazily on the first GPU receive.
  void start_h2d_consumer_if_needed();
  void h2d_consumer_loop();
  // Fired from the per-lane reader thread when a recv sub-chunk completes.
  void on_recv_chunk_complete(int slot_id, misc::result_t status);
  // Blocking-but-shutdown-aware acquisition of a free staging slot for the recv
  // producer. Returns Cancelled when the transport is stopping.
  absl::StatusOr<int> acquire_recv_slot();

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

  // H2D consumer thread for the receive path. Pulls ready staging slots, copies
  // to GPU, returns the slot, and reports completion to the owning request.
  std::thread h2d_consumer_thread_;
  std::atomic<bool> h2d_consumer_started_{false};
  std::mutex h2d_consumer_start_mutex_;

  // Per-slot side table linking a staging slot to its in-flight recv job. Sized
  // to the staging buffer's slot count; only the slot's current owner writes it.
  struct RecvSlotJob {
    uint8_t* gpu_ptr = nullptr;
    uint64_t bytes = 0;
    int device_id = 0;
    size_t chunk_index = 0;
    recv_request_state_t request;
    misc::result_t recv_status = misc::SUCCESS;
  };

  std::vector<RecvSlotJob> slot_jobs_;
  mutable std::mutex slot_jobs_mutex_;

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
