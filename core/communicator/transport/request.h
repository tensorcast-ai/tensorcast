// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_REQUEST_H_
#define CORE_COMMUNICATOR_ENGINE_REQUEST_H_

#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

#include "core/communicator/misc/metric.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::transport {

using read_result_t = struct ReadResult {
  absl::Status status = absl::OkStatus();
  uint64_t request_cost = 0;
  uint64_t read_cost = 0;
  uint64_t rdma_queue_cost = 0;
  uint64_t rdma_regmr_cost = 0;
  std::string tensor_key;
} __attribute__((aligned(128)));

using future_read_result_t = std::future<read_result_t>;

static inline std::string get_request_key(std::string key, uint64_t offset) {
  std::stringstream url;
  url << key << ":" << offset;
  return url.str();
}

class ReadRequest {
 public:
  ReadRequest(std::string tensor_key, std::string dst_ip, uint16_t dst_port, tensor_t local, uint64_t remote_offset);
  ~ReadRequest() = default;

  [[nodiscard]] tensor_t get_local_tensor() const;
  [[nodiscard]] remote_tensor_t get_remote_tensor() const;
  void set_remote_tensor(remote_tensor_t tensor);
  future_read_result_t get_future();
  void set_result(absl::Status result);
  bool is_result_set();

  std::string get_dst_url();

  std::string get_key();

  void record_request_response();
  void record_rdma_regmr();
  void record_rdma_queue_done();
  void record_read_done();

  void add_expected_completions(int n) {
    expected_completions_.fetch_add(n);
  }

  // Returns true if all segments have completed
  bool mark_completion_and_is_done() {
    int done = completed_.fetch_add(1) + 1;

    std::vector<PendingAckWindow> ready;
    std::function<void(uint32_t, const std::vector<uint64_t>&, bool)> sender;
    {
      absl::MutexLock lk(&ack_mu_);
      if (!segment_window_queue_.empty()) {
        uint32_t window_seq = segment_window_queue_.front();
        segment_window_queue_.pop_front();
        if (!pending_ack_windows_.empty()) {
          auto it = pending_ack_windows_.begin();
          // Windows are enqueued in order; find matching window_seq
          for (; it != pending_ack_windows_.end(); ++it) {
            if (it->window_seq == window_seq) {
              break;
            }
          }
          if (it != pending_ack_windows_.end()) {
            it->remaining -= 1;
            if (it->remaining == 0) {
              ready.emplace_back(std::move(*it));
              pending_ack_windows_.erase(it);
            }
          }
        }
      }
      sender = ack_sender_;
    }
    for (auto& window : ready) {
      if (sender) {
        sender(window.window_seq, window.offsets, window.final_window);
      }
    }
    return done >= expected_completions_.load();
  }

  void set_ack_sender(std::function<void(uint32_t, const std::vector<uint64_t>&, bool)> fn) {
    absl::MutexLock lk(&ack_mu_);
    ack_sender_ = std::move(fn);
  }

  void enqueue_window_ack(uint32_t window_seq, std::vector<uint64_t> offsets, bool final_window) {
    absl::MutexLock lk(&ack_mu_);
    PendingAckWindow window;
    window.window_seq = window_seq;
    window.final_window = final_window;
    window.remaining = static_cast<int>(offsets.size());
    window.offsets = std::move(offsets);
    pending_ack_windows_.push_back(std::move(window));
    for (int i = 0; i < pending_ack_windows_.back().remaining; ++i) {
      segment_window_queue_.push_back(window_seq);
    }
  }

  static std::future<read_result_t> get_read_result_future(std::string error_message);

  tensor_t local_tensor_;
  remote_tensor_t remote_tensor_;
  std::string tensor_key_;
  std::string dst_ip_;
  uint16_t dst_port_;
  std::promise<read_result_t> result_;
  std::atomic_bool result_set_;

  misc::Timer timer_;
  read_result_t status_;
  uint64_t remote_offset_;

  // Number of expected RDMA READ completions for this request
  std::atomic<int> expected_completions_{0};
  std::atomic<int> completed_{0};

  struct PendingAckWindow {
    uint32_t window_seq = 0;
    bool final_window = false;
    std::vector<uint64_t> offsets;
    int remaining = 0;
  };

  absl::Mutex ack_mu_;
  std::deque<PendingAckWindow> pending_ack_windows_ ABSL_GUARDED_BY(ack_mu_);
  std::deque<uint32_t> segment_window_queue_ ABSL_GUARDED_BY(ack_mu_);
  std::function<void(uint32_t, const std::vector<uint64_t>&, bool)> ack_sender_ ABSL_GUARDED_BY(ack_mu_);
};

using read_request_t = std::shared_ptr<ReadRequest>;

class WriteRequest {
 public:
  WriteRequest(tensor_t local_tensor, std::string tensor_key, uint64_t offset, uint64_t bytes);
  ~WriteRequest() = default;

  std::string get_key();

  tensor_t local_tensor_;
  std::string tensor_key_;
  uint64_t offset_;
  uint64_t bytes_;
};

using write_request_t = std::shared_ptr<WriteRequest>;

} // namespace tensorcast::communicator::transport

#endif // CORE_COMMUNICATOR_ENGINE_REQUEST_H_
