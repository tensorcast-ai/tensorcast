
// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <arpa/inet.h>
#include <error_handling.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"

#include "core/common/cuda_api.h"
#include "core/common/device_guard.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/gpu_tcp_stager.h"
#include "core/communicator/misc/envs.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/mtcp_transport.h"

#include <chrono>

namespace stepcast::communicator {

ENV_PARAM(TCP_TOS, 0);

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
      data->set_result(TRANSPORT_FAILED);
    }
  }

  while (!recv_queue_.empty()) {
    auto data = recv_queue_.pop();
    if (data != nullptr) {
      data->set_result(TRANSPORT_FAILED);
    }
  }

  if (sock_fd_ != 0) {
    ::close(sock_fd_);
    sock_fd_ = 0;
  }
}

void MTcpTransportTask::push_send(const chunk_t& data) {
  send_queue_.push(data);
}

void MTcpTransportTask::push_recv(const chunk_t& data) {
  VLOG(2) << "[MTcpTransportTask::push_recv] Pushing chunk, sock_fd=" << sock_fd_ << " len=" << data->len();
  recv_queue_.push(data);
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

result_t MTcpTransportTask::do_recv_bytes(uint8_t* buf, int size) const {
  ssize_t remain_bytes = size;
  ssize_t offset = 0;
  ssize_t bytes;
  while (remain_bytes > 0) {
    bytes = ::recv(sock_fd_, buf + offset, remain_bytes, 0);
    if (bytes <= 0) {
      LOG(WARNING) << "MTcpTransportTask recv error " << strerror(errno) << " remain_bytes=" << remain_bytes;
      return SYS_ERROR;
    } else if (bytes < remain_bytes) {
      remain_bytes -= bytes;
      offset += bytes;
    } else {
      remain_bytes = 0;
    }
  }
  return SUCCESS;
}

result_t MTcpTransportTask::do_send_bytes(uint8_t* buf, int size) const {
  ssize_t remain_bytes = size;
  ssize_t offset = 0;
  ssize_t bytes;
  while (remain_bytes > 0) {
    bytes = ::send(sock_fd_, buf + offset, remain_bytes, 0);
    if (bytes <= 0) {
      LOG(WARNING) << "MTcpTransportTask send error " << strerror(errno) << " remain_bytes=" << remain_bytes;
      return SYS_ERROR;
    } else if (bytes < remain_bytes) {
      remain_bytes -= bytes;
      offset += bytes;
    } else {
      remain_bytes = 0;
    }
  }
  return SUCCESS;
}

MTcpTransport::MTcpTransport(int conn_count)
    : listen_fd_(0),
      retry_count_(0),
      conn_count_(conn_count),
      send_queue_(),
      recv_queue_(),
      stop_(false),
      closed_(false),
      ready_(false) {
  ASSERT(conn_count > 1, "illegal conn count"); // mtcp only process
  ASSERT(conn_count <= kMaxTcpConns, "illegal conn count"); // mtcp only support 32 at max
  bzero(sock_fds_, kMaxTcpConns * sizeof(int));
  bzero(&server_addr_, sizeof(struct sockaddr_in));
  bzero(client_addrs_, sizeof(struct sockaddr_in) * kMaxTcpConns);
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

  LOG(INFO) << "[MTcpTransport] Destructor complete";
}

int MTcpTransport::listen(const std::string& ip, uint16_t* port) {
  listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ == -1) {
    return SYS_ERROR;
  }
  CLEAR(server_addr_);

  server_addr_.sin_family = AF_INET;
  server_addr_.sin_addr.s_addr = inet_addr(ip.c_str());
  server_addr_.sin_port = htons(*port);

  int ret = ::bind(listen_fd_, reinterpret_cast<struct sockaddr*>(&server_addr_), sizeof(server_addr_));
  if (ret < 0) {
    LOG(WARNING) << "failed to bind address " << ip << ":" << port;
    return SYS_ERROR;
  }

  // Use a reasonable backlog to reduce risk of connection drops under concurrency
  ret = ::listen(listen_fd_, conn_count_);
  if (ret < 0) {
    LOG(WARNING) << "failed to listen address" << ip << ":" << port;
    return SYS_ERROR;
  }

  socklen_t len = sizeof(server_addr_);
  getsockname(listen_fd_, (struct sockaddr*)&server_addr_, &len);
  recv_thread_ = std::thread([this] { this->server_loop(); });
  send_thread_ = std::thread([this] { this->send_loop(); });

  *port = ntohs(server_addr_.sin_port);
  return SUCCESS;
}

int MTcpTransport::connect(const std::string& ip, uint16_t port, int retry) {
  max_retry_ = retry;
  CLEAR(server_addr_);
  server_addr_.sin_family = AF_INET;
  server_addr_.sin_addr.s_addr = inet_addr(ip.c_str());
  server_addr_.sin_port = htons(port);

  recv_thread_ = std::thread([this] { this->client_loop(); });
  send_thread_ = std::thread([this] { this->send_loop(); });
  return SUCCESS;
}

result_t MTcpTransport::send(const write_request_t& msg) {
  if (closed_.load()) {
    return SYS_ERROR;
  }
  send_queue_.push(msg, true, -1);
  return SUCCESS;
}

result_t MTcpTransport::recv(const read_request_t& msg) {
  if (closed_.load()) {
    LOG(ERROR) << "[MTcpTransport::recv] Transport is closed, cannot process request";
    return SYS_ERROR;
  }

  VLOG(1) << "[MTcpTransport::recv] Queueing read request: key=" << msg->get_key()
          << " bytes=" << msg->get_local_tensor()->get_bytes();
  recv_queue_.push(msg, true, -1);
  return SUCCESS;
}

void MTcpTransport::set_conn_count(int conn_count) {
  ASSERT(!ready_.load(), "failed to set connection count");
  conn_count_ = conn_count;
}

void MTcpTransport::set_gpu_tcp_stager(std::shared_ptr<GpuTcpStager> stager) {
  gpu_tcp_stager_ = stager;
}

void MTcpTransport::set_memory_pool(std::shared_ptr<store::PinnedMemoryPool> pool) {
  memory_pool_ = pool;
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
        } else if (ret == 0) {
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
        int saved_errno = errno;
        LOG(WARNING) << "tcp transport accept failed (" << strerror(saved_errno) << ") – retrying";
        // Step back one index so that the for-loop retries the same connection index
        i--;
        // Briefly yield to avoid tight loop under persistent failures
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      init_socket_fd(sock_fds_[i]);
      tasks_[i] = std::make_unique<MTcpTransportTask>(sock_fds_[i]);
      tasks_[i]->start();
    }
    VLOG(1) << "[MTcpTransport::server_loop] MTCP ready with _conn_count=" << conn_count_;
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
      } else if (ret == -1) {
        retry_count_++;
        sleep(1);
      }
    }

    if (ret == -1) {
      LOG(ERROR) << "[MTcpTransport::client_loop] socket index=" << i << " connect failed after " << retry_count_
                 << " retries: " << strerror(errno);
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
        if (tensor->needs_staging() && gpu_tcp_stager_) {
          // Handle GPU tensor with staging - process in batches to avoid buffer exhaustion
          VLOG(1) << "Staging GPU tensor of " << bytes << " bytes in " << conn_count_ << " chunks";

          // Process chunks in batches to avoid exhausting staging buffers
          // Use a conservative batch size - leave one buffer free for pipelining
          const int available_buffers = static_cast<int>(gpu_tcp_stager_->get_num_buffers());
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
                uint64_t gpu_chunk_size = std::min<uint64_t>(remain_gpu_bytes, gpu_tcp_stager_->get_chunk_size());

                // Stage synchronously for the sub-chunk
                auto staged_result =
                    gpu_tcp_stager_->stage(tensor, global_offset + offset + gpu_offset_in_real_chunk, gpu_chunk_size);

                if (!staged_result.ok()) {
                  LOG(ERROR) << "Failed to stage GPU data: " << staged_result.status();
                  // Fail all remaining work for this and subsequent chunks
                  for (; idx < conn_count_; idx++) {
                    auto fail_chunk = std::make_shared<MTcpTransportChunk>(nullptr, 0);
                    fail_chunk->set_result(TRANSPORT_FAILED);
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
                auto stager = gpu_tcp_stager_; // Keep shared_ptr copy
                auto release_future = std::async(
                    std::launch::async,
                    [stager,
                     staged_ptr,
                     chunk_future = std::move(chunk_future),
                     idx,
                     gpu_offset_in_real_chunk]() mutable -> chunk_result_t {
                      VLOG(2) << "[MTcpTransport::send_loop] Release task #" << idx
                              << " waiting for send to complete (sub-chunk offset=" << gpu_offset_in_real_chunk << ")";
                      auto result = chunk_future.get();
                      auto release_status = stager->release_staged_buffer(staged_ptr);
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
          // Regular CPU tensor path
          for (uint64_t offset = 0; offset < bytes; offset += chunk_size) {
            auto real_chunk_size = chunk_size;
            if (offset + real_chunk_size > bytes) {
              real_chunk_size = bytes - offset;
            }
            auto chunk = std::make_shared<MTcpTransportChunk>(
                global_offset + offset + tensor->get_addr<uint8_t>(), real_chunk_size);
            tasks_[idx]->push_send(chunk);
            results.push_back(chunk->get_future());
            idx++;
          }
        }

        // Store futures for this request to wait later
        batch_futures.push_back(std::async(std::launch::async, [msg, results = std::move(results)]() mutable {
          // Wait for all chunks of this request
          for (size_t i = 0; i < results.size(); ++i) {
            auto result = results[i].get();
            if (result.status != SUCCESS) {
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
    }
  }
}

void MTcpTransport::recv_loop() {
  ready_.store(true);
  closed_.store(false);
  struct pollfd fds[1] = {};
  fds[0].fd = sock_fds_[0];
  fds[0].events = POLLIN;
  while (!stop_.load()) {
    if (stop_.load() || closed_.load()) {
      break;
    }

    auto msg = recv_queue_.pop(true, -1);
    if (msg == nullptr || stop_.load() || closed_.load()) {
      break;
    }

    // Debug trace for recv_loop – starting a ReadRequest
    VLOG(2) << "[MTcpTransport::recv_loop] Start processing ReadRequest key=" << msg->get_key()
            << " bytes=" << msg->get_local_tensor()->get_bytes() << " conn_count=" << conn_count_;

    auto tensor = msg->get_local_tensor();
    auto bytes = tensor->get_bytes();
    auto chunk_size = (bytes + conn_count_ - 1) / conn_count_;
    auto idx = 0;
    std::vector<future_chunk_result_t> results;

    // Check if tensor is on GPU and needs staging
    if (tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_GPU) {
      // Handle GPU tensor with staging - recv to CPU first, then copy to GPU
      VLOG(1) << "Receiving to GPU tensor of " << bytes << " bytes in " << conn_count_ << " chunks with staging";
      LOG(INFO) << "[MTcpTransport::recv_loop] GPU recv starting for " << msg->get_key() << " bytes=" << bytes
                << " chunks=" << conn_count_ << " tensor_ptr=" << tensor.get();

      // Enforce required configuration
      CHECK(memory_pool_ != nullptr) << "MTcpTransport requires memory_pool_ to be configured for GPU receive";

      // Determine pool chunk size once for this tensor receive
      size_t pool_chunk_size = memory_pool_->chunk_size();

      if (!gpu_recv_buffer_) {
        // Lazily construct StreamingPinnedBuffer using the configured memory pool.
        // The StreamingPinnedBuffer expects the per-chunk size to be exactly the same as the
        // chunk size used by the underlying PinnedMemoryPool. Otherwise the pool may deliver
        // fewer (but larger) buffers than requested which triggers the "Allocated chunk count
        // mismatch" check inside StreamingPinnedBuffer::initialize(). To avoid this we make
        // sure we request a buffer size that is **at least** the pool's chunk size.

        // If the computed network chunk_size exceeds the size of a single pinned
        // buffer provided by the pool, we will transparently split the network
        // chunk into multiple *sub-chunks* of at most pool_chunk_size bytes. This
        // allows large tensors to be streamed through a limited-size buffer
        // without aborting.
        if (chunk_size > pool_chunk_size) {
          VLOG(1) << "chunk_size (" << chunk_size << ") larger than pool_chunk_size (" << pool_chunk_size
                  << ") – enabling sub-chunked receive";
        }

        // Limit total pinned memory usage by capping the number of simultaneously
        // allocated buffers. We only need to pipeline a handful of chunks to fully
        // overlap network I/O and GPU copies, so four buffers is usually enough.
        size_t num_buffers = std::min(static_cast<size_t>(conn_count_), size_t(4));

        gpu_recv_buffer_ = std::make_unique<store::StreamingPinnedBuffer>(num_buffers, pool_chunk_size, memory_pool_);

        auto init_status = gpu_recv_buffer_->initialize();
        CHECK(init_status.ok()) << "Failed to initialize GPU receive buffer: " << init_status;
      }

      // Use StreamingPinnedBuffer for efficient memory management
      VLOG(1) << "Using StreamingPinnedBuffer for GPU receive";

      // Process chunks with streaming buffer (supports sub-chunking)
      bool processing_failed = false;
      for (uint64_t offset = 0; offset < bytes && !processing_failed;) {
        uint64_t real_chunk_size = chunk_size;
        if (offset + real_chunk_size > bytes) {
          real_chunk_size = bytes - offset;
        }

        // The current "network chunk" may itself be larger than a single pinned buffer.
        // We therefore stream it as a sequence of sub-chunks each no larger than
        // pool_chunk_size bytes.

        uint64_t remain_in_chunk = real_chunk_size;
        uint64_t sub_offset_in_chunk = 0;

        while (remain_in_chunk > 0 && !processing_failed) {
          uint64_t sub_chunk_size = std::min<uint64_t>(remain_in_chunk, pool_chunk_size);

          // Acquire a free staging buffer
          auto slot_result = gpu_recv_buffer_->get_free_chunk();
          if (!slot_result.ok()) {
            LOG(ERROR) << "Failed to get free chunk from GPU receive buffer: " << slot_result.status();
            processing_failed = true;
            break;
          }

          int slot_id = *slot_result;
          char* staged_ptr = gpu_recv_buffer_->get_chunk_ptr(slot_id);

          LOG(INFO) << "[MTcpTransport::recv_loop] Got buffer slot #" << slot_id << " for sub-chunk ("
                    << sub_offset_in_chunk << "/" << real_chunk_size << ") of chunk #" << idx;

          // Create chunk for receiving into staged buffer
          auto chunk = std::make_shared<MTcpTransportChunk>(reinterpret_cast<uint8_t*>(staged_ptr), sub_chunk_size);
          tasks_[idx]->push_recv(chunk);

          // Create a future that will copy to GPU and return buffer after recv completes
          auto chunk_future = chunk->get_future();
          auto gpu_global_offset = offset + sub_offset_in_chunk;
          auto* gpu_ptr = tensor->get_addr<uint8_t>() + gpu_global_offset;
          auto* recv_buffer = gpu_recv_buffer_.get(); // capture raw pointer

          auto copy_future = std::async(
              std::launch::async,
              [staged_ptr,
               gpu_ptr,
               sub_chunk_size,
               chunk_future = std::move(chunk_future),
               tensor,
               idx,
               slot_id,
               recv_buffer]() mutable -> chunk_result_t {
                LOG(INFO) << "[MTcpTransport::recv_loop] Async task #" << idx
                          << " waiting for sub-chunk recv to complete";

                auto result = chunk_future.get();
                if (result.status != SUCCESS) {
                  LOG(ERROR) << "[MTcpTransport::recv_loop] Async task #" << idx
                             << " sub-chunk recv failed, returning buffer";
                  CHECK_OK(recv_buffer->return_chunk(slot_id));
                  return result;
                }

                // Copy from staged buffer to GPU
                int device_id = tensor->get_device_id();
                if (device_id < 0) {
                  device_id = 0;
                }

                stepcast::common::DeviceGuard guard(device_id);
                if (!guard.status().ok()) {
                  LOG(ERROR) << "Failed to set CUDA device: " << guard.status();
                  CHECK_OK(recv_buffer->return_chunk(slot_id));
                  return {TRANSPORT_FAILED, result.cost};
                }

                auto copy_status = cuda::memcpy(gpu_ptr, staged_ptr, sub_chunk_size, cudaMemcpyHostToDevice);
                if (!copy_status.ok()) {
                  LOG(ERROR) << "Failed to copy to GPU: " << copy_status.message();
                  CHECK_OK(recv_buffer->return_chunk(slot_id));
                  return {TRANSPORT_FAILED, result.cost};
                }

                CHECK_OK(recv_buffer->return_chunk(slot_id));
                return result;
              });

          results.push_back(std::move(copy_future));

          remain_in_chunk -= sub_chunk_size;
          sub_offset_in_chunk += sub_chunk_size;
        }

        if (processing_failed) {
          // Fail all remaining high-level chunks (connections) if an error occurred
          for (; idx < conn_count_; idx++) {
            auto fail_chunk = std::make_shared<MTcpTransportChunk>(nullptr, 0);
            fail_chunk->set_result(TRANSPORT_FAILED);
            results.push_back(fail_chunk->get_future());
          }
          break;
        }

        offset += real_chunk_size;
        idx++;
      }
    } else {
      // Regular CPU tensor path
      for (uint64_t offset = 0; offset < bytes; offset += chunk_size) {
        auto real_chunk_size = chunk_size;
        if (offset + real_chunk_size > bytes) {
          real_chunk_size = bytes - offset;
        }
        auto chunk = std::make_shared<MTcpTransportChunk>(offset + tensor->get_addr<uint8_t>(), real_chunk_size);
        tasks_[idx]->push_recv(chunk);
        results.push_back(chunk->get_future());
        idx++;
      }
    }

    result_t result = SUCCESS;
    LOG(INFO) << "[MTcpTransport::recv_loop] Waiting for " << results.size() << " async operations for "
              << msg->get_key();
    for (size_t i = 0; i < results.size(); ++i) {
      auto chunk_result = results[i].get();
      if (chunk_result.status != SUCCESS) {
        LOG(ERROR) << "[MTcpTransport::recv_loop] Chunk " << i << " failed for " << msg->get_key();
        result = FAILED;
      }
    }

    if (result == SUCCESS) {
      msg->set_result(absl::OkStatus());
    } else {
      msg->set_result(absl::InternalError("failed to read chunk"));
    }

    // Debug trace for recv_loop – finished a ReadRequest with result
    VLOG(2) << "[MTcpTransport::recv_loop] Finished ReadRequest key=" << msg->get_key()
            << " status=" << (result == SUCCESS ? "SUCCESS" : "FAILED");
  }
  closed_.store(true);
  ready_.store(false);
}

result_t MTcpTransport::init_socket_fd(int sock_fd) {
  int opt = 1;
  int ret;
  ret = setsockopt(sock_fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char*>(&opt), sizeof(int));
  if (ret != 0) {
    LOG(WARNING) << "failed to setup tcp no-delay: err=" << errno << " " << strerror(errno);
  }
  opt = 1;
  ret = setsockopt(sock_fd, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(int));
  if (ret != 0) {
    LOG(WARNING) << "failed to setup tcp keepalive: err=" << errno << " " << strerror(errno);
  }

  struct linger l = {0, 0};
  ret = setsockopt(sock_fd, SOL_SOCKET, SO_LINGER, &l, sizeof(l));
  if (ret != 0) {
    LOG(WARNING) << "failed to setup tcp linger: err=" << errno << " " << strerror(errno);
  }

  if (TCP_TOS > 0) {
    int tos = TCP_TOS;
    LOG(INFO) << "set tcp tos as " << TCP_TOS;
    socklen_t optlen = sizeof(tos);
    ret = setsockopt(sock_fd, IPPROTO_IP, IP_TOS, &tos, optlen);
    if (ret != 0) {
      LOG(WARNING) << "failed to setup tcp tos: err=" << errno << " " << strerror(errno);
    }
  }

  struct timeval tv{};
  tv.tv_sec = 10;
  tv.tv_usec = 0;
  setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

  opt = 1;
  ret = setsockopt(sock_fd, SOL_SOCKET, SO_ZEROCOPY, &opt, sizeof(opt));
  if (ret != 0) {
    LOG(WARNING) << "failed to set zero copy" << errno << " " << strerror(errno);
  }

  return SUCCESS;
}

} // namespace stepcast::communicator
