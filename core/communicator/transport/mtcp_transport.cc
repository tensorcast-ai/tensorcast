
// Copyright (c) 2025-2026, TensorCast Team.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>

#include <future>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

#include "core/common/async_copy_manager.h"
#include "core/common/memory/streaming_chunk_guard.h"
#include "core/communicator/base/constants.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/mtcp_transport.h"
#include "core/cuda/device_guard.h"
#include "gsl/pointers"

#include <chrono>

namespace tensorcast::communicator::transport {

namespace {

gsl::not_null<std::shared_ptr<common::memory::StreamingPinnedBuffer>> make_streaming_buffer(
    const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>& pool,
    int conn_count,
    int buffers_per_flow_limit) {
  const int buffer_budget = buffers_per_flow_limit > 0 ? buffers_per_flow_limit : conn_count;
  const size_t num_buffers = static_cast<size_t>(std::max(1, buffer_budget));
  auto streaming_buffer =
      std::make_shared<common::memory::StreamingPinnedBuffer>(num_buffers, pool->slice_bytes(), pool.get());
  return gsl::not_null<std::shared_ptr<common::memory::StreamingPinnedBuffer>>{std::move(streaming_buffer)};
}

int compute_gpu_lane_for_subchunk_internal(
    uint64_t chunk_base_offset,
    uint64_t sub_offset_in_chunk,
    uint64_t stage_unit,
    int lanes_to_use) {
  if (lanes_to_use <= 0) {
    return 0;
  }
  if (stage_unit == 0) {
    // Fallback to a single lane when staging information is unavailable.
    return 0;
  }

  const uint64_t segment_offset = chunk_base_offset + sub_offset_in_chunk;
  return static_cast<int>(((segment_offset / stage_unit) % static_cast<uint64_t>(lanes_to_use)));
}

int compute_active_lanes_internal(uint64_t total_bytes, size_t stage_unit, int conn_count, int buffers_per_flow_limit) {
  if (total_bytes == 0) {
    return 1;
  }
  const int total_conn = std::max(1, conn_count);
  int lanes_to_use = total_conn;
  if (buffers_per_flow_limit > 0) {
    lanes_to_use = std::min(lanes_to_use, buffers_per_flow_limit);
  }

  if (stage_unit > 0 && total_bytes > 0) {
    const uint64_t ceil_div = (total_bytes + stage_unit - 1) / stage_unit;
    const int required_lanes =
        static_cast<int>(std::min<uint64_t>(ceil_div, static_cast<uint64_t>(std::numeric_limits<int>::max())));
    lanes_to_use = std::min(lanes_to_use, std::max(1, required_lanes));
  }

  lanes_to_use = std::max(1, lanes_to_use);
  lanes_to_use = std::min(lanes_to_use, total_conn);
  return lanes_to_use;
}

} // namespace

namespace testing {

int compute_gpu_lane_for_subchunk(
    uint64_t chunk_base_offset,
    uint64_t sub_offset_in_chunk,
    uint64_t stage_unit,
    int lanes_to_use) {
  return compute_gpu_lane_for_subchunk_internal(chunk_base_offset, sub_offset_in_chunk, stage_unit, lanes_to_use);
}

int compute_active_lanes(uint64_t total_bytes, size_t stage_unit, int conn_count, int buffers_per_flow_limit) {
  return compute_active_lanes_internal(total_bytes, stage_unit, conn_count, buffers_per_flow_limit);
}

} // namespace testing

MTcpTransportChunk::MTcpTransportChunk(uint8_t* addr, uint64_t len) : addr_(addr), len_(len), timer_(true) {}

MTcpTransportTask::MTcpTransportTask(int sock_fd) : sock_fd_(sock_fd), stop_(false) {}

MTcpTransportTask::~MTcpTransportTask() {
  if (!stop_.load()) {
    stop();
  }
}

void MTcpTransportTask::start() {
  recv_thread_ = std::thread([this] { this->recv_loop(); });
  send_thread_ = std::thread([this] { this->send_loop(); });
}

void MTcpTransportTask::stop() {
  stop_.store(true);
  send_queue_.stop();
  recv_queue_.stop();
  if (recv_thread_.joinable()) {
    recv_thread_.join();
  }
  if (send_thread_.joinable()) {
    send_thread_.join();
  }

  while (!send_queue_.empty()) {
    auto data = send_queue_.pop();
    if (data != nullptr) {
      data->set_result(misc::TRANSPORT_FAILED);
    }
  }

  while (!recv_queue_.empty()) {
    auto data = recv_queue_.pop();
    if (data != nullptr) {
      data->set_result(misc::TRANSPORT_FAILED);
    }
  }

  if (sock_fd_ != 0) {
    ::close(sock_fd_);
    sock_fd_ = 0;
  }
}

void MTcpTransportTask::push_send(const chunk_t& chunk) {
  send_queue_.push(chunk);
}

void MTcpTransportTask::push_recv(const chunk_t& chunk) {
  VLOG(2) << "[MTcpTransportTask::push_recv] Pushing chunk, sock_fd=" << sock_fd_ << " len=" << chunk->len();
  recv_queue_.push(chunk);
}

void MTcpTransportTask::send_loop() {
  while (!stop_.load()) {
    if (stop_.load() || sock_fd_ == 0) {
      break;
    }
    auto data = send_queue_.pop(true, -1);
    if (stop_.load()) {
      break;
    }
    if (data == nullptr) {
      continue;
    }
    data->set_result(do_send_bytes(data->addr<uint8_t>(), data->len()));
  }
}

void MTcpTransportTask::recv_loop() {
  while (!stop_.load()) {
    if (stop_.load() || sock_fd_ == 0) {
      break;
    }
    auto data = recv_queue_.pop(true, -1);
    if (stop_.load()) {
      break;
    }
    if (data == nullptr) {
      continue;
    }
    VLOG(2) << "[MTcpTransportTask::recv_loop] Processing chunk, sock_fd=" << sock_fd_ << " len=" << data->len();
    auto result = do_recv_bytes(data->addr<uint8_t>(), data->len());
    VLOG(2) << "[MTcpTransportTask::recv_loop] Chunk complete, sock_fd=" << sock_fd_ << " result=" << result;
    data->set_result(result);
  }
}

misc::result_t MTcpTransportTask::do_recv_bytes(uint8_t* buf, int size) const {
  ssize_t remain_bytes = size;
  ssize_t offset = 0;
  ssize_t bytes;
  while (remain_bytes > 0) {
    bytes = ::recv(sock_fd_, buf + offset, remain_bytes, 0);
    if (bytes < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        std::this_thread::yield();
        continue;
      }
      PLOG(WARNING) << "MTcpTransportTask recv error, remain_bytes=" << remain_bytes;
      return misc::SYS_ERROR;
    }
    if (bytes == 0) {
      LOG(WARNING) << "MTcpTransportTask recv peer closed connection, remain_bytes=" << remain_bytes;
      return misc::REMOTE_ERROR;
    }
    if (bytes < remain_bytes) {
      remain_bytes -= bytes;
      offset += bytes;
    } else {
      remain_bytes = 0;
    }
  }
  return misc::SUCCESS;
}

misc::result_t MTcpTransportTask::do_send_bytes(uint8_t* buf, int size) const {
  ssize_t remain_bytes = size;
  ssize_t offset = 0;
  ssize_t bytes;
  while (remain_bytes > 0) {
    bytes = ::send(sock_fd_, buf + offset, remain_bytes, 0);
    if (bytes < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        std::this_thread::yield();
        continue;
      }
      PLOG(WARNING) << "MTcpTransportTask send error, remain_bytes=" << remain_bytes;
      return misc::SYS_ERROR;
    }
    if (bytes == 0) {
      std::this_thread::yield();
      continue;
    }
    if (bytes < remain_bytes) {
      remain_bytes -= bytes;
      offset += bytes;
    } else {
      remain_bytes = 0;
    }
  }
  return misc::SUCCESS;
}

MTcpTransport::MTcpTransport(
    int conn_count,
    gsl::not_null<std::shared_ptr<engine::MemoryStager>> memory_stager,
    gsl::not_null<std::shared_ptr<engine::MemoryStager>> gpu_memory_stager,
    gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>> memory_pool,
    int buffers_per_flow_limit)
    : listen_fd_(0),
      retry_count_(0),
      conn_count_(conn_count),
      stop_(false),
      closed_(false),
      ready_(false),
      gpu_memory_stager_(std::move(gpu_memory_stager)),
      memory_pool_(std::move(memory_pool)),
      gpu_recv_buffer_(make_streaming_buffer(memory_pool_, conn_count, buffers_per_flow_limit)),
      gpu_init_retry_timeout_(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::minutes(1))),
      buffers_per_flow_limit_(buffers_per_flow_limit),
      memory_stager_(std::move(memory_stager)) {
  ABSL_CHECK(conn_count > 1) << "illegal conn count"; // mtcp only process
  ABSL_CHECK(conn_count <= base::kMaxTcpConns) << "illegal conn count";
  bzero(sock_fds_, base::kMaxTcpConns * sizeof(int));
  bzero(&server_addr_, sizeof(struct sockaddr_in));
  bzero(client_addrs_, sizeof(struct sockaddr_in) * base::kMaxTcpConns);

  // One side-table entry per staging slot; only the slot's current owner writes it.
  slot_jobs_.resize(gpu_recv_buffer_->capacity());

  const auto init_wait_start = std::chrono::steady_clock::now();
  int init_retry_count = 0;

  // Block until pinned buffers become available; log a warning once per minute while waiting.
  while (true) {
    auto init_status = gpu_recv_buffer_->initialize(gpu_init_retry_timeout_);
    if (init_status.ok()) {
      gpu_buffer_ready_.store(true, std::memory_order_release);
      if (init_retry_count > 0) {
        const auto total_wait = std::chrono::steady_clock::now() - init_wait_start;
        const auto total_wait_seconds = std::chrono::duration_cast<std::chrono::seconds>(total_wait).count();
        LOG(INFO) << "[MTcpTransport] GPU receive buffer initialization succeeded after " << total_wait_seconds
                  << "s of waiting";
      }
      break;
    }

    if (init_status.code() == absl::StatusCode::kDeadlineExceeded) {
      ++init_retry_count;
      const auto elapsed = std::chrono::steady_clock::now() - init_wait_start;
      const auto elapsed_minutes =
          std::max(1, static_cast<int>(std::chrono::duration_cast<std::chrono::minutes>(elapsed).count()));
      LOG(WARNING) << "[MTcpTransport] Waiting for GPU receive buffer initialization; " << elapsed_minutes
                   << " minute(s) elapsed. Status: " << init_status;
      continue;
    }

    LOG(FATAL) << "Failed to initialize GPU receive buffer: " << init_status;
  }
}

MTcpTransport::~MTcpTransport() {
  LOG(INFO) << "[MTcpTransport] Destructor called, conn_count=" << conn_count_;

  stop_.store(true);
  send_queue_.stop();
  recv_queue_.stop();

  // Clear queues first to reject any pending operations
  send_queue_.clear();
  while (!recv_queue_.empty()) {
    auto data = recv_queue_.pop();
    if (data != nullptr) {
      LOG(INFO) << "[MTcpTransport] Rejecting pending read request during destruction";
      data->set_result(absl::InternalError("cannot read tensor due to closed MTcpTransport"));
    }
  }

  // Wait for threads to finish before destroying tasks
  LOG(INFO) << "[MTcpTransport] Waiting for send/recv threads to finish";
  if (send_thread_.joinable()) {
    send_thread_.join();
  }
  if (recv_thread_.joinable()) {
    recv_thread_.join();
  }

  staged_queue_.stop();
  if (staged_thread_.joinable()) {
    staged_thread_.join();
  }
  while (true) {
    auto pending = staged_queue_.pop(false);
    if (!pending) {
      break;
    }
    for (auto& seg : pending->segments) {
      if (seg.on_complete) {
        seg.on_complete(misc::TRANSPORT_FAILED);
      }
    }
  }

  if (listen_fd_ != 0) {
    ::close(listen_fd_);
    listen_fd_ = 0;
  }

  // Now safe to destroy tasks
  LOG(INFO) << "[MTcpTransport] Stopping " << conn_count_ << " tasks";
  for (int i = 0; i < conn_count_; i++) {
    if (tasks_[i] != nullptr) {
      tasks_[i]->stop();
    }
    tasks_[i].reset();
  }

  // Producer (recv_loop) and all lane readers are now stopped, so every staging
  // slot that was producer-owned has fired on_complete -> mark_chunk_ready. No
  // more chunks will ever become ready. Wake the H2D consumer so it drains the
  // remaining ready slots (skipping the copy under stop_) and exits, returning
  // every slot before release_receive_resources() waits on them.
  if (h2d_consumer_started_.load(std::memory_order_acquire)) {
    gpu_recv_buffer_->signal_production_complete();
    if (h2d_consumer_thread_.joinable()) {
      h2d_consumer_thread_.join();
    }
  }

  fail_pending_async_tasks(misc::TRANSPORT_FAILED);
  prune_async_tasks();
  release_receive_resources();

  LOG(INFO) << "[MTcpTransport] Destructor complete";
}

int MTcpTransport::listen(const std::string& ip, uint16_t* port) {
  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ == -1) {
    return misc::SYS_ERROR;
  }
  misc::CLEAR(server_addr_);

  server_addr_.sin_family = AF_INET;
  server_addr_.sin_addr.s_addr = inet_addr(ip.c_str());
  server_addr_.sin_port = htons(*port);

  int ret = ::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&server_addr_), sizeof(server_addr_));
  if (ret < 0) {
    LOG(WARNING) << "failed to bind address " << ip << ":" << port;
    return misc::SYS_ERROR;
  }

  // Use a reasonable backlog to reduce risk of connection drops under concurrency
  ret = ::listen(listen_fd_, conn_count_);
  if (ret < 0) {
    LOG(WARNING) << "failed to listen address" << ip << ":" << port;
    return misc::SYS_ERROR;
  }

  socklen_t len = sizeof(server_addr_);
  getsockname(listen_fd_, (struct sockaddr*)&server_addr_, &len);
  recv_thread_ = std::thread([this] { this->server_loop(); });
  send_thread_ = std::thread([this] { this->send_loop(); });
  start_staged_thread();

  *port = ntohs(server_addr_.sin_port);
  return misc::SUCCESS;
}

int MTcpTransport::connect(const std::string& ip, uint16_t port, int retry) {
  max_retry_ = retry;
  misc::CLEAR(server_addr_);
  server_addr_.sin_family = AF_INET;
  server_addr_.sin_addr.s_addr = inet_addr(ip.c_str());
  server_addr_.sin_port = htons(port);

  recv_thread_ = std::thread([this] { this->client_loop(); });
  send_thread_ = std::thread([this] { this->send_loop(); });
  start_staged_thread();
  return misc::SUCCESS;
}

misc::result_t MTcpTransport::send(const write_request_t& msg) {
  if (closed_.load()) {
    return misc::SYS_ERROR;
  }
  send_queue_.push(msg, true, -1);
  return misc::SUCCESS;
}

misc::result_t MTcpTransport::recv(const read_request_t& msg) {
  if (closed_.load()) {
    LOG(ERROR) << "[MTcpTransport::recv] Transport is closed, cannot process request";
    return misc::SYS_ERROR;
  }

  VLOG(1) << "[MTcpTransport::recv] Queueing read request: key=" << msg->get_key()
          << " bytes=" << msg->get_local_tensor()->get_bytes() << " stage_hint=" << msg->mtcp_stage_unit_hint_bytes();
  recv_queue_.push(msg, true, -1);
  return misc::SUCCESS;
}

void MTcpTransport::enqueue_stage_window(StageSendWindow window) {
  auto notify_segment = [](StageSendWindow& w, StageSendSegment& seg, misc::result_t status) {
    if (seg.on_complete) {
      seg.on_complete(status);
    }
    if (w.pending_segments) {
      const int previous = w.pending_segments->fetch_sub(1, std::memory_order_acq_rel);
      if (previous == 1 && w.on_window_complete) {
        w.on_window_complete();
      }
    }
  };

  if (window.pending_segments) {
    window.pending_segments->store(static_cast<int>(window.segments.size()), std::memory_order_relaxed);
  }

  if (stop_.load() || closed_.load()) {
    if (window.segments.empty()) {
      if (window.on_window_complete) {
        window.on_window_complete();
      }
      prune_async_tasks();
      return;
    }
    for (auto& seg : window.segments) {
      notify_segment(window, seg, misc::TRANSPORT_FAILED);
    }
    prune_async_tasks();
    return;
  }

  if (window.segments.empty()) {
    if (window.on_window_complete) {
      window.on_window_complete();
    }
    prune_async_tasks();
    return;
  }

  auto ptr = std::make_shared<StageSendWindow>(std::move(window));
  if (staged_queue_.push(ptr, true, -1) != misc::SUCCESS) {
    for (auto& seg : ptr->segments) {
      notify_segment(*ptr, seg, misc::TRANSPORT_FAILED);
    }
    if (ptr->segments.empty() && ptr->on_window_complete) {
      ptr->on_window_complete();
    }
    prune_async_tasks();
    return;
  }
  prune_async_tasks();
}

void MTcpTransport::start_staged_thread() {
  if (!staged_thread_.joinable()) {
    staged_thread_ = std::thread([this] { this->staged_send_loop(); });
  }
}

void MTcpTransport::set_conn_count(int conn_count) {
  ABSL_CHECK(!ready_.load()) << "failed to set connection count";
  conn_count_ = conn_count;
}

void MTcpTransport::release_receive_resources() {
  if (!gpu_buffer_ready_.load(std::memory_order_acquire)) {
    return;
  }
  std::unique_lock<std::mutex> lock(gpu_buffer_mutex_);
  if (!gpu_buffer_ready_.load(std::memory_order_acquire)) {
    return;
  }
  absl::Status release_status = gpu_recv_buffer_->release();
  if (!release_status.ok()) {
    LOG(WARNING) << "[MTcpTransport] Failed to release GPU receive buffer: " << release_status;
    return;
  }
  gpu_buffer_ready_.store(false, std::memory_order_release);
}

absl::Status MTcpTransport::ensure_gpu_buffer_ready() {
  if (gpu_buffer_ready_.load(std::memory_order_acquire)) {
    return absl::OkStatus();
  }
  std::unique_lock<std::mutex> lock(gpu_buffer_mutex_);
  if (gpu_buffer_ready_.load(std::memory_order_acquire)) {
    return absl::OkStatus();
  }
  auto init_status = gpu_recv_buffer_->initialize(gpu_init_retry_timeout_);
  if (!init_status.ok()) {
    return init_status;
  }
  gpu_buffer_ready_.store(true, std::memory_order_release);
  return absl::OkStatus();
}

void MTcpTransport::server_loop() {
  while (!stop_.load()) {
    for (int i = 0; i < conn_count_; i++) {
      if (tasks_[i] != nullptr) {
        tasks_[i]->stop();
        tasks_[i].reset();
      }
    }
    bzero(sock_fds_, sizeof(int) * conn_count_);
    int addr_len = sizeof(client_addrs_[0]);
    struct pollfd fds[1] = {};
    fds[0].fd = listen_fd_;
    fds[0].events = POLLIN;
    for (int i = 0; i < conn_count_; i++) {
      while (!stop_.load()) {
        int ret = poll(fds, 1, 1000);
        if (ret == -1) {
          break;
        }
        if (ret == 0) {
          continue;
        }

        if (fds[0].revents & POLLIN) {
          sock_fds_[i] = accept(
              listen_fd_,
              reinterpret_cast<struct sockaddr*>(&client_addrs_[i]),
              reinterpret_cast<socklen_t*>(&addr_len));
        }
        break;
      }

      if (stop_.load()) {
        return;
      }

      if (sock_fds_[i] <= 0) {
        // Accept failed (e.g., due to transient EINTR/EAGAIN). Retry this slot instead of aborting
        PLOG(WARNING) << "tcp transport accept failed — retrying";
        // Step back one index so that the for-loop retries the same connection index
        i--;
        // Briefly yield to avoid tight loop under persistent failures
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      this->init_socket_fd(sock_fds_[i]);
      tasks_[i] = std::make_unique<MTcpTransportTask>(sock_fds_[i]);
      tasks_[i]->start();
    }
    LOG(INFO) << "[MTcpTransport::server_loop] MTCP ready with _conn_count=" << conn_count_;
    recv_loop();
  }
}

void MTcpTransport::client_loop() {
  for (int i = 0; i < conn_count_; i++) {
    // Reset retry counter for each individual socket to avoid earlier failures
    // causing subsequent sockets to abort immediately. This was the root cause
    // of missing socket connections (only N-1 sockets ready) observed in
    // concurrent tests.
    retry_count_ = 0;
    int addr_len = sizeof(server_addr_);
    sock_fds_[i] = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fds_[i] == -1) {
      closed_.store(true);
      ready_.store(true);
      return;
    }

    int ret = 0;
    while (!stop_.load() && retry_count_ < max_retry_) {
      ret =
          ::connect(sock_fds_[i], reinterpret_cast<struct sockaddr*>(&server_addr_), static_cast<socklen_t>(addr_len));
      if (ret == 0) {
        break;
      }
      if (ret == -1) {
        retry_count_++;
        sleep(1);
      }
    }

    if (ret == -1) {
      PLOG(ERROR) << "[MTcpTransport::client_loop] socket index=" << i << " connect failed after " << retry_count_
                  << " retries";
      closed_.store(true);
      ready_.store(true);
      return;
    }

    init_socket_fd(sock_fds_[i]);
    tasks_[i] = std::make_unique<MTcpTransportTask>(sock_fds_[i]);
    tasks_[i]->start();
  }
  VLOG(1) << "[MTcpTransport::client_loop] MTCP ready with conn_count=" << conn_count_;
  recv_loop();
}

void MTcpTransport::send_loop() {
  while (!stop_.load()) {
    size_t spin_cnt = 0;
    while (!ready_.load() && !stop_.load()) {
      if (++spin_cnt % 10000000 == 0) {
        VLOG(1) << "[MTcpTransport::send_loop] waiting for ready_ (conn_count=" << conn_count_ << ")";
      }
      std::this_thread::yield();
    }

    while (!stop_.load()) {
      if (stop_.load() || closed_.load()) {
        break;
      }

      // Process multiple write requests in batch to improve concurrency
      std::vector<write_request_t> batch;
      const int max_batch_size = 4; // Process up to 4 requests concurrently

      // Collect a batch of requests
      for (int i = 0; i < max_batch_size; ++i) {
        auto msg = send_queue_.pop(false); // Non-blocking pop
        if (msg == nullptr) {
          break;
        }
        batch.push_back(msg);
      }

      // If no requests available, do a blocking wait
      if (batch.empty()) {
        auto msg = send_queue_.pop(true, -1);
        if (msg == nullptr || stop_.load() || closed_.load()) {
          break;
        }
        batch.push_back(msg);
      }

      // Process all requests in the batch
      std::vector<std::future<void>> batch_futures;

      for (const auto& msg : batch) {
        // Debug trace for send_loop – beginning processing of a WriteRequest
        VLOG(2) << "[MTcpTransport::send_loop] Start processing WriteRequest key=" << msg->tensor_key_
                << " bytes=" << msg->bytes_ << " offset=" << msg->offset_ << " conn_count=" << conn_count_;

        uint64_t global_offset = msg->offset_;
        auto tensor = msg->local_tensor_;
        auto bytes = msg->bytes_;
        auto chunk_size = (bytes + conn_count_ - 1) / conn_count_;
        auto idx = 0;
        std::vector<future_chunk_result_t> results;

        // Check if tensor needs GPU->CPU staging
        if (tensor->needs_staging()) {
          // Handle GPU tensor with staging - process in batches to avoid buffer exhaustion
          VLOG(1) << "Staging GPU tensor of " << bytes << " bytes in " << conn_count_ << " chunks";

          // Process chunks in batches to avoid exhausting staging buffers
          // Use a conservative batch size - leave one buffer free for pipelining
          const size_t stager_buffers = gpu_memory_stager_->get_num_buffers();
          const int available_buffers = static_cast<int>(stager_buffers);
          const int batch_size = std::max(1, available_buffers - 1);

          bool staging_failed = false;
          for (uint64_t offset = 0; offset < bytes && !staging_failed;) {
            // Process up to batch_size chunks in this iteration
            std::vector<future_chunk_result_t> batch_results;
            int batch_idx = 0;

            while (batch_idx < batch_size && offset < bytes && idx < conn_count_ && !staging_failed) {
              auto real_chunk_size = chunk_size;
              if (offset + real_chunk_size > bytes) {
                real_chunk_size = bytes - offset;
              }

              // If the computed real_chunk_size is larger than what a single staging buffer can handle,
              // we transparently split it into multiple "gpu sub-chunks" to fully accommodate the data.
              // This prevents an unnecessary failure and allows the sender to stream very large chunks
              // through a limited-size staging buffer.

              uint64_t remain_gpu_bytes = real_chunk_size;
              uint64_t gpu_offset_in_real_chunk = 0;

              while (remain_gpu_bytes > 0) {
                const size_t stager_chunk_size = gpu_memory_stager_->get_chunk_size();
                uint64_t gpu_chunk_size = std::min<uint64_t>(remain_gpu_bytes, stager_chunk_size);

                // Stage synchronously for the sub-chunk using unified stager if available
                absl::StatusOr<void*> staged_result = gpu_memory_stager_->stage(
                    tensor, global_offset + offset + gpu_offset_in_real_chunk, gpu_chunk_size);

                if (!staged_result.ok()) {
                  LOG(ERROR) << "Failed to stage GPU data: " << staged_result.status();
                  // Fail all remaining work for this and subsequent chunks
                  for (; idx < conn_count_; idx++) {
                    auto fail_chunk = std::make_shared<MTcpTransportChunk>(nullptr, 0);
                    fail_chunk->set_result(misc::TRANSPORT_FAILED);
                    results.push_back(fail_chunk->get_future());
                  }
                  staging_failed = true;
                  break;
                }

                auto* staged_ptr = static_cast<uint8_t*>(*staged_result);
                auto chunk = std::make_shared<MTcpTransportChunk>(staged_ptr, gpu_chunk_size);
                tasks_[idx]->push_send(chunk);

                // Asynchronously release the staged buffer after send completes
                auto chunk_future = chunk->get_future();
                // Keep a copy of the stager for the async releaser
                auto stager_mem = gpu_memory_stager_.get();
                auto release_future = std::async(
                    std::launch::async,
                    [stager_mem,
                     staged_ptr,
                     chunk_future = std::move(chunk_future),
                     idx,
                     gpu_offset_in_real_chunk]() mutable -> chunk_result_t {
                      VLOG(2) << "[MTcpTransport::send_loop] Release task #" << idx
                              << " waiting for send to complete (sub-chunk offset=" << gpu_offset_in_real_chunk << ")";
                      auto result = chunk_future.get();
                      absl::Status release_status = stager_mem->release_staged_buffer(gsl::not_null<void*>{staged_ptr});
                      if (!release_status.ok()) {
                        LOG(ERROR) << "Failed to release staged buffer: " << release_status;
                      }
                      return result;
                    });

                batch_results.push_back(std::move(release_future));

                remain_gpu_bytes -= gpu_chunk_size;
                gpu_offset_in_real_chunk += gpu_chunk_size;
              }

              if (staging_failed) {
                break; // Break out of while(remain_gpu_bytes) loop and outer loops
              }

              // After processing the (possibly split) real chunk, advance the outer loop state
              offset += real_chunk_size; // move to next real chunk
              idx++;
              batch_idx++;
            }

            // If we have more chunks to process, wait for this batch to complete
            // This ensures buffers are released before we try to stage more
            if (offset < bytes && idx < conn_count_ && !staging_failed) {
              for (auto& r : batch_results) {
                r.wait(); // Wait for send to complete and buffer to be released
              }
            }

            // Add batch results to overall results
            for (auto& r : batch_results) {
              results.push_back(std::move(r));
            }
          }
        } else {
          // CPU tensor path (MemoryStager required)
          VLOG(1) << "Staging CPU tensor of " << bytes << " bytes in " << conn_count_ << " chunks";

          const auto cpu_stager = memory_stager_.get();
          const int available_buffers = static_cast<int>(cpu_stager->get_num_buffers());
          const int batch_size = std::max(1, available_buffers - 1);

          bool staging_failed = false;
          for (uint64_t offset = 0; offset < bytes && !staging_failed;) {
            std::vector<future_chunk_result_t> batch_results;
            int batch_idx = 0;

            while (batch_idx < batch_size && offset < bytes && idx < conn_count_ && !staging_failed) {
              auto real_chunk_size = chunk_size;
              if (offset + real_chunk_size > bytes) {
                real_chunk_size = bytes - offset;
              }

              uint64_t remain_bytes = real_chunk_size;
              uint64_t sub_off = 0;

              while (remain_bytes > 0) {
                uint64_t stager_chunk = std::min<uint64_t>(remain_bytes, cpu_stager->get_chunk_size());
                auto staged_or = cpu_stager->stage(tensor, global_offset + offset + sub_off, stager_chunk);
                if (!staged_or.ok()) {
                  LOG(ERROR) << "Failed to stage CPU data: " << staged_or.status();
                  // Fail remaining connections
                  for (; idx < conn_count_; idx++) {
                    auto fail_chunk = std::make_shared<MTcpTransportChunk>(nullptr, 0);
                    fail_chunk->set_result(misc::TRANSPORT_FAILED);
                    results.push_back(fail_chunk->get_future());
                  }
                  staging_failed = true;
                  break;
                }
                auto* staged_ptr = static_cast<uint8_t*>(*staged_or);
                auto chunk = std::make_shared<MTcpTransportChunk>(staged_ptr, stager_chunk);
                tasks_[idx]->push_send(chunk);

                auto chunk_future = chunk->get_future();
                auto release_future = std::async(
                    std::launch::async,
                    [stager = cpu_stager, staged_ptr, chunk_future = std::move(chunk_future), idx, sub_off]() mutable {
                      VLOG(2) << "[MTcpTransport::send_loop] CPU staged release task #" << idx
                              << " waiting for send to complete (sub-off=" << sub_off << ")";
                      auto result = chunk_future.get();
                      auto st = stager->release_staged_buffer(gsl::not_null<void*>{staged_ptr});
                      if (!st.ok()) {
                        LOG(ERROR) << "Failed to release CPU staged buffer: " << st;
                      }
                      return result;
                    });
                batch_results.push_back(std::move(release_future));

                remain_bytes -= stager_chunk;
                sub_off += stager_chunk;
              }

              if (staging_failed) {
                break;
              }

              offset += real_chunk_size;
              idx++;
              batch_idx++;
            }

            if (offset < bytes && idx < conn_count_ && !staging_failed) {
              for (auto& r : batch_results) {
                r.wait();
              }
            }

            for (auto& r : batch_results) {
              results.push_back(std::move(r));
            }
          }
        }

        // Store futures for this request to wait later
        batch_futures.push_back(std::async(std::launch::async, [msg, results = std::move(results)]() mutable {
          // Wait for all chunks of this request
          for (size_t i = 0; i < results.size(); ++i) {
            auto result = results[i].get();
            if (result.status != misc::SUCCESS) {
              LOG(ERROR) << "[MTcpTransport::send_loop] Chunk " << i << " failed for " << msg->tensor_key_;
            }
          }
          VLOG(2) << "[MTcpTransport::send_loop] Finished WriteRequest key=" << msg->tensor_key_;
        }));
      }

      // Wait for all requests in the batch to complete
      for (auto& future : batch_futures) {
        future.wait();
      }
      prune_async_tasks();
    }
  }
}

void MTcpTransport::staged_send_loop() {
  constexpr int kIdlePopTimeoutMs = 100;
  constexpr int kPendingTaskPopTimeoutMs = 1;
  while (!stop_.load()) {
    if (!ready_.load()) {
      if (stop_.load()) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }

    prune_async_tasks();
    const int pop_timeout_ms = has_outstanding_async_tasks() ? kPendingTaskPopTimeoutMs : kIdlePopTimeoutMs;
    auto window_ptr = staged_queue_.pop(true, pop_timeout_ms);
    if (stop_.load()) {
      break;
    }
    if (!window_ptr) {
      prune_async_tasks();
      continue;
    }
    process_stage_window(window_ptr);
    prune_async_tasks();
  }

  while (true) {
    auto window_ptr = staged_queue_.pop(false);
    if (!window_ptr) {
      break;
    }
    for (auto& seg : window_ptr->segments) {
      if (seg.on_complete) {
        seg.on_complete(misc::TRANSPORT_FAILED);
      }
    }
  }
  prune_async_tasks();
}

void MTcpTransport::process_stage_window(const std::shared_ptr<StageSendWindow>& window) {
  if (!window) {
    return;
  }

  auto mark_segment = [&](StageSendSegment& seg, misc::result_t status) {
    if (seg.on_complete) {
      seg.on_complete(status);
    }
    if (window->pending_segments) {
      const int previous = window->pending_segments->fetch_sub(1, std::memory_order_acq_rel);
      if (previous == 1 && window->on_window_complete) {
        window->on_window_complete();
      }
    }
  };

  if (conn_count_ <= 0) {
    if (window->segments.empty()) {
      if (window->on_window_complete) {
        window->on_window_complete();
      }
      return;
    }
    for (auto& seg : window->segments) {
      mark_segment(seg, misc::TRANSPORT_FAILED);
    }
    return;
  }

  size_t stage_unit = window->stage_unit_bytes > 0 ? static_cast<size_t>(window->stage_unit_bytes)
                                                   : gpu_memory_stager_->get_chunk_size();
  if (stage_unit == 0) {
    stage_unit = memory_pool_->slice_bytes();
  }
  // Enforce strict non-zero stage unit; configuration must guarantee >0.
  if (stage_unit == 0) {
    LOG(FATAL) << "MTcpTransport::process_stage_window: stage_unit must be > 0 (check stager and pool config)";
  }
  const int lanes_to_use =
      compute_active_lanes_internal(window->total_bytes, stage_unit, conn_count_, buffers_per_flow_limit_);
  ABSL_CHECK_GE(lanes_to_use, 1) << "lanes_to_use must be >= 1";
  const int total_conn = std::max(1, conn_count_);

  if (window->segments.empty()) {
    if (window->on_window_complete) {
      window->on_window_complete();
    }
    return;
  }

  for (auto& seg : window->segments) {
    if (stop_.load() || closed_.load()) {
      mark_segment(seg, misc::TRANSPORT_FAILED);
      continue;
    }
    if (seg.data == nullptr || seg.bytes == 0) {
      mark_segment(seg, misc::FAILED);
      continue;
    }

    int task_index = compute_gpu_lane_for_subchunk_internal(seg.metadata.offset, 0, stage_unit, lanes_to_use);
    if (task_index >= total_conn) {
      task_index %= total_conn;
    }

    VLOG(1) << "[MTCP send] enqueue window=" << window->window_seq << " segment=" << seg.metadata.segment_idx
            << " offset=" << seg.metadata.offset << " bytes=" << seg.bytes << " lane=" << task_index
            << " lanes_to_use=" << lanes_to_use << " conn_count=" << conn_count_ << " stage_unit_bytes=" << stage_unit;

    if (task_index >= conn_count_ || tasks_[task_index] == nullptr) {
      LOG(WARNING) << "[MTcpTransport::process_stage_window] task unavailable idx=" << task_index
                   << " lanes_to_use=" << lanes_to_use << " conn_count=" << conn_count_;
      task_index = 0;
      if (task_index >= conn_count_ || tasks_[task_index] == nullptr) {
        mark_segment(seg, misc::TRANSPORT_FAILED);
        continue;
      }
    }

    auto chunk = std::make_shared<MTcpTransportChunk>(reinterpret_cast<uint8_t*>(seg.data), seg.bytes);
    tasks_[task_index]->push_send(chunk);
    auto chunk_future = chunk->get_future().share();
    auto callback = seg.on_complete;
    auto metadata = seg.metadata;
    auto request_key = window->request_key;
    auto window_ptr = window;
    track_async_task(
        std::move(chunk_future),
        [callback = std::move(callback), metadata, request_key, window_ptr](const chunk_result_t& result) mutable {
          VLOG(1) << "[MTCP send] complete window=" << window_ptr->window_seq << " segment=" << metadata.segment_idx
                  << " offset=" << metadata.offset << " status=" << (result.status == misc::SUCCESS ? "ok" : "err");
          if (callback) {
            callback(result.status);
          }
          if (window_ptr->pending_segments) {
            const int previous = window_ptr->pending_segments->fetch_sub(1, std::memory_order_acq_rel);
            if (previous == 1 && window_ptr->on_window_complete) {
              window_ptr->on_window_complete();
            }
          }
          if (result.status != misc::SUCCESS) {
            LOG(WARNING) << "[staging_credit] request=" << request_key
                         << " transport=mtcp window=" << metadata.window_seq << " segment=" << metadata.segment_idx
                         << " bytes=" << metadata.bytes << " status=error";
          } else {
            VLOG(2) << "[staging_credit] request=" << request_key << " transport=mtcp window=" << metadata.window_seq
                    << " segment=" << metadata.segment_idx << " bytes=" << metadata.bytes << " status=ok";
          }
        });
  }
}

void MTcpTransport::track_async_task(
    std::shared_future<chunk_result_t> future,
    std::function<void(const chunk_result_t&)> on_ready) {
  std::lock_guard<std::mutex> lk(async_tasks_mutex_);
  outstanding_async_tasks_.push_back(
      TrackedAsyncTask{
          .future = std::move(future),
          .on_ready = std::move(on_ready),
      });
}

void MTcpTransport::fail_pending_async_tasks(misc::result_t status) {
  std::vector<std::function<void(const chunk_result_t&)>> callbacks;
  {
    std::lock_guard<std::mutex> lk(async_tasks_mutex_);
    callbacks.reserve(outstanding_async_tasks_.size());
    for (auto& task : outstanding_async_tasks_) {
      if (task.on_ready) {
        callbacks.push_back(std::move(task.on_ready));
      }
    }
    outstanding_async_tasks_.clear();
  }

  const chunk_result_t forced_result{
      .status = status,
      .cost = 0,
  };
  for (auto& callback : callbacks) {
    callback(forced_result);
  }
}

bool MTcpTransport::has_outstanding_async_tasks() const {
  std::lock_guard<std::mutex> lk(async_tasks_mutex_);
  return !outstanding_async_tasks_.empty();
}

void MTcpTransport::prune_async_tasks() {
  std::vector<std::pair<std::function<void(const chunk_result_t&)>, chunk_result_t>> completed;
  {
    std::lock_guard<std::mutex> lk(async_tasks_mutex_);
    auto it = outstanding_async_tasks_.begin();
    while (it != outstanding_async_tasks_.end()) {
      if (!it->future.valid()) {
        it = outstanding_async_tasks_.erase(it);
        continue;
      }
      if (it->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        auto result = it->future.get();
        if (it->on_ready) {
          completed.emplace_back(std::move(it->on_ready), result);
        }
        it = outstanding_async_tasks_.erase(it);
        continue;
      }
      ++it;
    }
  }

  for (auto& [callback, result] : completed) {
    if (callback) {
      callback(result);
    }
  }
}

void MTcpTransport::recv_loop() {
  ready_.store(true);
  closed_.store(false);
  while (!stop_.load()) {
    if (stop_.load() || closed_.load()) {
      break;
    }

    auto msg = recv_queue_.pop(true, -1);
    if (msg == nullptr || stop_.load() || closed_.load()) {
      break;
    }

    VLOG(2) << "[MTcpTransport::recv_loop] Start processing ReadRequest key=" << msg->get_key()
            << " bytes=" << msg->get_local_tensor()->get_bytes() << " conn_count=" << conn_count_;

    auto tensor = msg->get_local_tensor();
    auto bytes = tensor->get_bytes();
    const uint64_t request_base_offset = msg->remote_offset_;

    uint64_t stage_unit = msg->mtcp_stage_unit_hint_bytes();
    if (stage_unit == 0) {
      stage_unit = gpu_memory_stager_->get_chunk_size();
    }
    if (stage_unit == 0) {
      stage_unit = memory_pool_->slice_bytes();
    }
    ABSL_CHECK_GT(stage_unit, 0) << "MTcpTransport::recv_loop: stage_unit must be > 0";
    const int lanes_to_use =
        compute_active_lanes_internal(bytes, static_cast<size_t>(stage_unit), conn_count_, buffers_per_flow_limit_);

    VLOG(1) << "[MTcpTransport::recv_loop] key=" << msg->get_key() << " bytes=" << bytes
            << " conn_count=" << conn_count_ << " lanes_to_use=" << lanes_to_use
            << " buffers_per_flow_limit=" << buffers_per_flow_limit_ << " stage_unit_bytes=" << stage_unit
            << " request_base_offset=" << request_base_offset;

    const uint64_t network_segment_bytes = stage_unit;

    // Single accounting object for this request. The producer (this loop) issues
    // sub-chunks and seals; each sub-chunk reports completion later from the lane
    // reader thread (CPU) or the H2D consumer thread (GPU). The thread whose
    // action drives the outstanding count to zero finalizes the request.
    auto state = std::make_shared<MTcpRecvRequestState>(msg);
    bool issue_failed = false;

    if (tensor->get_mem_type() == base::COMMUNICATE_ENGINE_DEV_GPU) {
      // GPU path: this loop is a pure network producer. It acquires a staging
      // slot, hands a recv chunk to a lane, and moves on. It never waits on the
      // H2D copy. When the lane finishes the recv, on_recv_chunk_complete fires
      // and publishes the slot to the dedicated H2D consumer thread. This
      // decouples socket draining from H2D so a slow copy can never stall the
      // lanes (and therefore can never apply TCP backpressure to the source).
      start_h2d_consumer_if_needed();
      if (!h2d_consumer_started_.load(std::memory_order_acquire)) {
        LOG(ERROR) << "[MTcpTransport::recv_loop] H2D consumer/GPU buffer not ready for " << msg->get_key();
        state->mark_failed();
        issue_failed = true;
      }

      size_t pool_chunk_size = issue_failed ? 0 : memory_pool_->slice_bytes();
      if (!issue_failed) {
        ABSL_CHECK_GT(pool_chunk_size, 0) << "MTcpTransport::recv_loop: pool_chunk_size must be > 0";
      }

      for (uint64_t segment_offset = 0; segment_offset < bytes && !issue_failed;
           segment_offset += network_segment_bytes) {
        const uint64_t segment_size = std::min<uint64_t>(network_segment_bytes, bytes - segment_offset);
        int lane = compute_gpu_lane_for_subchunk_internal(
            request_base_offset + segment_offset, /*sub_offset_in_chunk=*/0, stage_unit, lanes_to_use);
        if (lanes_to_use > 0 && lane >= lanes_to_use) {
          lane %= lanes_to_use;
        }
        if (lane >= conn_count_ || tasks_[lane] == nullptr) {
          LOG(ERROR) << "[MTcpTransport::recv_loop] Invalid lane index " << lane << " (lanes_to_use=" << lanes_to_use
                     << ") for " << msg->get_key();
          state->mark_failed();
          issue_failed = true;
          break;
        }

        // A single network segment can still be larger than one pinned slot.
        uint64_t remain_in_segment = segment_size;
        uint64_t sub_offset_in_segment = 0;
        while (remain_in_segment > 0 && !issue_failed) {
          uint64_t sub_chunk_size = std::min<uint64_t>(remain_in_segment, pool_chunk_size);

          auto slot_or = acquire_recv_slot();
          if (!slot_or.ok()) {
            LOG(ERROR) << "[MTcpTransport::recv_loop] Failed to acquire staging slot for " << msg->get_key() << ": "
                       << slot_or.status();
            state->mark_failed();
            issue_failed = true;
            break;
          }
          const int slot_id = *slot_or;
          char* staged_ptr = gpu_recv_buffer_->get_chunk_ptr(slot_id);

          const uint64_t gpu_global_offset = segment_offset + sub_offset_in_segment;
          auto* gpu_ptr = tensor->get_addr<uint8_t>() + gpu_global_offset;
          const size_t chunk_index = pool_chunk_size > 0 ? static_cast<size_t>(gpu_global_offset / pool_chunk_size) : 0;
          int device_id = std::max(tensor->get_device_id(), 0);

          {
            std::lock_guard<std::mutex> lk(slot_jobs_mutex_);
            slot_jobs_[slot_id] = RecvSlotJob{
                .gpu_ptr = gpu_ptr,
                .bytes = sub_chunk_size,
                .device_id = device_id,
                .chunk_index = chunk_index,
                .request = state,
                .recv_status = misc::SUCCESS,
            };
          }

          state->add_chunk();
          auto chunk = std::make_shared<MTcpTransportChunk>(reinterpret_cast<uint8_t*>(staged_ptr), sub_chunk_size);
          chunk->set_on_complete([this, slot_id](misc::result_t st) { on_recv_chunk_complete(slot_id, st); });
          tasks_[lane]->push_recv(chunk);

          VLOG(2) << "[MTcpTransport::recv_loop] Issued sub-chunk slot=" << slot_id << " lane=" << lane
                  << " gpu_off=" << gpu_global_offset << " bytes=" << sub_chunk_size;

          remain_in_segment -= sub_chunk_size;
          sub_offset_in_segment += sub_chunk_size;
        }
      }
    } else {
      // CPU path: recv directly into the destination tensor; completion is
      // reported straight from the lane reader thread (no H2D, no consumer).
      for (uint64_t segment_offset = 0; segment_offset < bytes; segment_offset += network_segment_bytes) {
        const uint64_t segment_size = std::min<uint64_t>(network_segment_bytes, bytes - segment_offset);
        int lane = compute_gpu_lane_for_subchunk_internal(
            request_base_offset + segment_offset, /*sub_offset_in_chunk=*/0, stage_unit, lanes_to_use);
        if (lanes_to_use > 0 && lane >= lanes_to_use) {
          lane %= lanes_to_use;
        }
        if (lane >= conn_count_ || tasks_[lane] == nullptr) {
          LOG(ERROR) << "[MTcpTransport::recv_loop] Invalid CPU lane index " << lane
                     << " (lanes_to_use=" << lanes_to_use << ") for " << msg->get_key();
          state->mark_failed();
          break;
        }
        const uint64_t seg_bytes = segment_size;
        auto state_for_cb = state;
        auto chunk = std::make_shared<MTcpTransportChunk>(segment_offset + tensor->get_addr<uint8_t>(), segment_size);
        chunk->set_on_complete(
            [state_for_cb, seg_bytes](misc::result_t st) { state_for_cb->on_chunk_done(st, seg_bytes); });
        state->add_chunk();
        tasks_[lane]->push_recv(chunk);
      }
    }

    // Remove the producer issuing token. If every sub-chunk already completed,
    // this finalizes the request (set_result) on this thread; otherwise the last
    // completing sub-chunk finalizes it asynchronously. Either way the producer
    // does not block here, so it is free to pop the next request immediately.
    state->seal();
  }
  closed_.store(true);
  ready_.store(false);
}

void MTcpTransport::start_h2d_consumer_if_needed() {
  if (h2d_consumer_started_.load(std::memory_order_acquire)) {
    return;
  }
  std::lock_guard<std::mutex> lk(h2d_consumer_start_mutex_);
  if (h2d_consumer_started_.load(std::memory_order_acquire)) {
    return;
  }
  if (!gpu_buffer_ready_.load(std::memory_order_acquire)) {
    absl::Status ensure_status = ensure_gpu_buffer_ready();
    if (!ensure_status.ok()) {
      LOG(ERROR) << "[MTcpTransport::start_h2d_consumer_if_needed] failed to prepare GPU receive buffer: "
                 << ensure_status;
      return;
    }
  }
  h2d_consumer_thread_ = std::thread([this] { this->h2d_consumer_loop(); });
  h2d_consumer_started_.store(true, std::memory_order_release);
}

absl::StatusOr<int> MTcpTransport::acquire_recv_slot() {
  // The H2D consumer frees slots asynchronously, so block by polling rather than
  // sleeping inside get_free_chunk(): this lets the producer wake promptly on
  // shutdown without depending on a consumer return to break a blocking wait.
  while (!stop_.load(std::memory_order_acquire)) {
    auto slot_or = gpu_recv_buffer_->try_get_free_chunk();
    if (slot_or.ok()) {
      return *slot_or;
    }
    if (!absl::IsUnavailable(slot_or.status())) {
      return slot_or.status();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  return absl::CancelledError("MTcpTransport stopping");
}

void MTcpTransport::on_recv_chunk_complete(int slot_id, misc::result_t status) {
  // Fired from the per-lane reader thread when a recv sub-chunk finishes (or
  // from the teardown drain). We are the producer-owned slot's owner here.
  recv_request_state_t request;
  uint64_t bytes = 0;
  size_t chunk_index = 0;
  {
    std::lock_guard<std::mutex> lk(slot_jobs_mutex_);
    slot_jobs_[slot_id].recv_status = status;
    request = slot_jobs_[slot_id].request;
    bytes = slot_jobs_[slot_id].bytes;
    chunk_index = slot_jobs_[slot_id].chunk_index;
  }

  if (status != misc::SUCCESS) {
    // Recv failed: the staging buffer holds garbage, so don't enqueue it for
    // H2D. Return the slot and finalize this sub-chunk as failed directly.
    absl::Status ab = gpu_recv_buffer_->abort_producer_slot(slot_id);
    if (!ab.ok()) {
      LOG(WARNING) << "[MTcpTransport::on_recv_chunk_complete] abort_producer_slot failed slot=" << slot_id << ": "
                   << ab;
    }
    if (request) {
      request->on_chunk_done(status, 0);
    }
    return;
  }

  // Hand the filled slot to the H2D consumer thread. The consumer reports the
  // sub-chunk completion (request->on_chunk_done) after the copy.
  absl::Status mr = gpu_recv_buffer_->mark_chunk_ready(slot_id, chunk_index, bytes);
  if (!mr.ok()) {
    LOG(ERROR) << "[MTcpTransport::on_recv_chunk_complete] mark_chunk_ready failed slot=" << slot_id << ": " << mr;
    absl::Status ab = gpu_recv_buffer_->abort_producer_slot(slot_id);
    if (!ab.ok()) {
      LOG(WARNING) << "[MTcpTransport::on_recv_chunk_complete] abort_producer_slot failed slot=" << slot_id << ": "
                   << ab;
    }
    if (request) {
      request->on_chunk_done(misc::TRANSPORT_FAILED, 0);
    }
  }
}

void MTcpTransport::h2d_consumer_loop() {
  // Single dedicated consumer for the receive path: drains ready staging slots,
  // performs the H2D copy, returns the slot, and reports completion to the owning
  // request. Exits when production is complete (teardown) and the queue drains.
  while (true) {
    auto ready_or = gpu_recv_buffer_->get_ready_chunk();
    if (!ready_or.ok()) {
      // OutOfRange => production complete and fully drained => exit.
      VLOG(1) << "[MTcpTransport::h2d_consumer_loop] exiting: " << ready_or.status();
      break;
    }
    const int slot_id = ready_or->slot_id;

    RecvSlotJob job;
    {
      std::lock_guard<std::mutex> lk(slot_jobs_mutex_);
      job = slot_jobs_[slot_id];
    }

    misc::result_t status = job.recv_status;
    if (status == misc::SUCCESS && !stop_.load(std::memory_order_acquire)) {
      cuda::DeviceGuard guard(job.device_id);
      if (!guard.status().ok()) {
        LOG(ERROR) << "[MTcpTransport::h2d_consumer_loop] failed to set CUDA device " << job.device_id << ": "
                   << guard.status();
        status = misc::TRANSPORT_FAILED;
      } else {
        char* staged_ptr = gpu_recv_buffer_->get_chunk_ptr(slot_id);
        tensorcast::common::HostRegion h{.base = staged_ptr, .length = static_cast<size_t>(job.bytes), .pinned = true};
        tensorcast::common::DeviceRegion d{
            .device_id = job.device_id, .dev_ptr = job.gpu_ptr, .length = static_cast<size_t>(job.bytes)};
        tensorcast::common::CopyOptions opts{.tracing_stage = "H2D/Copy"};
        auto hdl_or = tensorcast::common::AsyncCopyManager::instance().submit_h2d(h, d, opts);
        if (!hdl_or.ok()) {
          LOG(ERROR) << "[MTcpTransport::h2d_consumer_loop] submit_h2d failed slot=" << slot_id << ": "
                     << hdl_or.status();
          status = misc::TRANSPORT_FAILED;
        } else {
          auto wait_st = hdl_or->wait();
          if (!wait_st.ok()) {
            LOG(ERROR) << "[MTcpTransport::h2d_consumer_loop] H2D copy failed slot=" << slot_id << ": " << wait_st;
            status = misc::TRANSPORT_FAILED;
          }
        }
      }
    } else if (stop_.load(std::memory_order_acquire)) {
      // Shutting down: skip the copy and finalize as failed so the request
      // unblocks instead of waiting on a copy we will never perform.
      status = misc::TRANSPORT_FAILED;
    }

    // Return the slot to the free pool regardless of copy outcome so the
    // producer can reuse it.
    absl::Status rc = gpu_recv_buffer_->return_chunk(slot_id);
    if (!rc.ok()) {
      LOG(WARNING) << "[MTcpTransport::h2d_consumer_loop] return_chunk failed slot=" << slot_id << ": " << rc;
    }

    if (job.request) {
      job.request->on_chunk_done(status, status == misc::SUCCESS ? job.bytes : 0);
    }
  }
}

misc::result_t MTcpTransport::init_socket_fd(int sock_fd) {
  int opt = 1;
  int ret;
  ret = setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&opt), sizeof(int));
  if (ret != 0) {
    PLOG(WARNING) << "failed to setup tcp no-delay";
  }

#ifdef TCP_QUICKACK
  opt = 1;
  ret = setsockopt(sock_fd, IPPROTO_TCP, TCP_QUICKACK, reinterpret_cast<char*>(&opt), sizeof(int));
  if (ret != 0) {
    PLOG(WARNING) << "failed to setup tcp quickack";
  }
#endif

  // Size socket buffers from staging chunk characteristics to reduce sender backpressure.
  const size_t stage_chunk = std::max<size_t>(gpu_memory_stager_->get_chunk_size(), memory_pool_->slice_bytes());
  const size_t desired_bytes_unclamped = stage_chunk * static_cast<size_t>(std::max(2, buffers_per_flow_limit_));
  const size_t desired_bytes = std::clamp<size_t>(desired_bytes_unclamped, 1ULL * 1024 * 1024, 64ULL * 1024 * 1024);
  const int desired_sock_buffer =
      static_cast<int>(std::min<size_t>(desired_bytes, static_cast<size_t>(std::numeric_limits<int>::max())));

  ret = setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &desired_sock_buffer, sizeof(desired_sock_buffer));
  if (ret != 0) {
    PLOG(WARNING) << "failed to setup socket send buffer";
  }
  ret = setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &desired_sock_buffer, sizeof(desired_sock_buffer));
  if (ret != 0) {
    PLOG(WARNING) << "failed to setup socket recv buffer";
  }

  int actual_send_buffer = 0;
  int actual_recv_buffer = 0;
  socklen_t option_len = sizeof(int);
  ret = getsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &actual_send_buffer, &option_len);
  if (ret != 0) {
    PLOG(WARNING) << "failed to read socket send buffer";
  }
  option_len = sizeof(int);
  ret = getsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &actual_recv_buffer, &option_len);
  if (ret != 0) {
    PLOG(WARNING) << "failed to read socket recv buffer";
  }
  LOG(INFO) << "[mtcp_sockbuf] fd=" << sock_fd << " requested=" << desired_sock_buffer
            << " actual_send=" << actual_send_buffer << " actual_recv=" << actual_recv_buffer
            << " stage_chunk=" << stage_chunk << " buffers_per_flow_limit=" << buffers_per_flow_limit_;

  opt = 1;
  ret = setsockopt(sock_fd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(int));
  if (ret != 0) {
    PLOG(WARNING) << "failed to setup tcp keepalive";
  }

  struct linger l = {.l_onoff = 0, .l_linger = 0};
  ret = setsockopt(sock_fd, SOL_SOCKET, SO_LINGER, &l, sizeof(l));
  if (ret != 0) {
    PLOG(WARNING) << "failed to setup tcp linger";
  }

  if (tcp_tos_ > 0) {
    int tos = tcp_tos_;
    LOG(INFO) << "set tcp tos as " << tcp_tos_;
    socklen_t optlen = sizeof(tos);
    ret = setsockopt(sock_fd, IPPROTO_IP, IP_TOS, &tos, optlen);
    if (ret != 0) {
      PLOG(WARNING) << "failed to setup tcp tos";
    }
  }

  struct timeval tv{};
  tv.tv_sec = 10;
  tv.tv_usec = 0;
  setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

  opt = 1;
  ret = setsockopt(sock_fd, SOL_SOCKET, SO_ZEROCOPY, &opt, sizeof(opt));
  if (ret != 0) {
    PLOG(WARNING) << "failed to set zero copy";
  }

  return misc::SUCCESS;
}

} // namespace tensorcast::communicator::transport
