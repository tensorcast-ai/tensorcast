// Copyright (c) 2025-2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_REQUEST_H_
#define CORE_COMMUNICATOR_ENGINE_REQUEST_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

#include "core/communicator/misc/metric.h"
#include "core/communicator/routing/types.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::transport {

using read_result_t = struct ReadResult {
  absl::Status status = absl::OkStatus();
  uint64_t request_cost = 0;
  uint64_t read_cost = 0;
  uint64_t rdma_queue_cost = 0;
  uint64_t rdma_regmr_cost = 0;
  uint64_t request_first_response_us = 0;
  uint64_t rdma_first_post_us = 0;
  uint64_t rdma_first_completion_us = 0;
  uint64_t rdma_last_completion_us = 0;
  uint64_t rdma_post_to_last_completion_us = 0;
  uint64_t rdma_response_windows = 0;
  uint64_t rdma_response_segments = 0;
  uint64_t rdma_wr_posted = 0;
  uint64_t rdma_wc_completed = 0;
  uint64_t rdma_ack_windows = 0;
  uint64_t rdma_ack_segments = 0;
  uint64_t rdma_handshake_queue_wait_us = 0;
  bool transport_is_rdma = false;
  bool rdma_staged_response = false;
  bool rdma_zero_copy_response = false;
  int16_t local_rail_id = -1;
  int16_t remote_rail_id = -1;
  std::string local_nic;
  std::string remote_nic;
  std::string tensor_key;
} __attribute__((aligned(128)));

using future_read_result_t = std::future<read_result_t>;

static inline std::string get_request_key(std::string key, uint64_t offset) {
  std::stringstream url;
  url << key << ":" << offset;
  return url.str();
}

static inline std::string get_request_instance_key(std::string key, uint64_t offset, uint64_t request_id) {
  std::stringstream url;
  url << key << ":" << offset << "#" << request_id;
  return url.str();
}

static inline std::string get_read_plan_request_key(uint64_t request_id) {
  std::stringstream url;
  url << "read_plan#" << request_id;
  return url.str();
}

struct PreparedSourcePlacement {
  uint32_t local_region_index = 0;
  uint64_t local_region_offset = 0;
  uint64_t source_slice_offset = 0;
  uint64_t bytes = 0;
};

struct PreparedLocalRegion {
  routing::LocalRegion logical_region;
  int16_t rail_id = -1;
  std::string nic_name;
  tensor_t tensor;
};

struct PreparedReadPlan {
  routing::ReadPlan logical_plan;
  std::string remote_endpoint_id;
  routing::ConnectionProtocol protocol = routing::ConnectionProtocol::kAuto;
  int16_t rail_id = -1;
  std::string local_nic;
  uint64_t total_bytes = 0;
  std::vector<PreparedLocalRegion> local_regions;
  std::vector<std::vector<PreparedSourcePlacement>> placements_by_source_slice;
};

class ReadRequest {
 public:
  enum class Kind {
    kTensor = 0,
    kReadPlan = 1,
  };

  enum class AckKind {
    kOffsets = 0,
    kSegmentCount = 1,
  };

  struct PendingAckWindow {
    uint32_t window_seq = 0;
    bool final_window = false;
    std::vector<uint64_t> offsets;
    uint32_t num_segments = 0;
    AckKind ack_kind = AckKind::kOffsets;
    int remaining = 0;
  };

  static bool rdma_profile_enabled_for_process();
  static void set_rdma_profile_enabled_for_process(bool enabled);

  ReadRequest(
      std::string tensor_key,
      std::string dst_ip,
      uint16_t dst_port,
      tensor_t local,
      uint64_t remote_offset,
      uint64_t request_id,
      int rail_id = -1);
  ReadRequest(
      std::string display_key,
      std::string dst_ip,
      uint16_t dst_port,
      std::shared_ptr<PreparedReadPlan> prepared_plan,
      uint64_t request_id,
      int rail_id = -1);
  ~ReadRequest() = default;

  [[nodiscard]] tensor_t get_local_tensor() const;
  [[nodiscard]] remote_tensor_t get_remote_tensor() const;
  void set_remote_tensor(remote_tensor_t tensor);
  [[nodiscard]] std::shared_ptr<PreparedReadPlan> get_prepared_read_plan() const;
  future_read_result_t get_future();
  void set_result(absl::Status result);
  void set_on_result(std::function<void()> callback);
  bool is_result_set();

  [[nodiscard]] Kind kind() const {
    return kind_;
  }

  [[nodiscard]] bool is_read_plan() const {
    return kind_ == Kind::kReadPlan;
  }

  [[nodiscard]] uint64_t request_id() const {
    return request_id_;
  }

  std::string get_dst_url();

  std::string get_key();

  int16_t get_rail_id() const {
    return rail_id_;
  }

  [[nodiscard]] uint64_t total_bytes() const {
    return total_bytes_;
  }

  [[nodiscard]] bool rdma_profile_enabled() const {
    return rdma_profile_enabled_;
  }

  void record_request_response();
  void record_rdma_regmr();
  void record_rdma_queue_done();
  void record_read_done();
  void note_rdma_response_window(uint32_t segment_count);
  void note_rdma_posted_wr(uint32_t wr_count);
  void note_rdma_completion();
  void note_rdma_ack_window(uint32_t segment_count);
  void note_rdma_handshake_queue_wait_us(uint64_t wait_us);

  void add_expected_completions(int n) {
    expected_completions_.fetch_add(n);
  }

  void note_rdma_window(int n, bool final_window) {
    if (n > 0) {
      expected_completions_.fetch_add(n);
    }
    if (final_window) {
      rdma_final_window_received_.store(true, std::memory_order_release);
    }
  }

  void set_mtcp_stage_unit_hint_bytes(uint64_t bytes) {
    mtcp_stage_unit_hint_bytes_.store(bytes, std::memory_order_relaxed);
  }

  [[nodiscard]] uint64_t mtcp_stage_unit_hint_bytes() const {
    return mtcp_stage_unit_hint_bytes_.load(std::memory_order_relaxed);
  }

  void enqueue_completion_bytes(uint32_t bytes) {
    absl::MutexLock lk(&ack_mu_);
    completion_bytes_queue_.push_back(bytes);
  }

  void set_progress_callbacks(
      std::function<void(uint64_t, uint64_t)> on_progress,
      std::function<void(const absl::Status&)> on_complete = {}) {
    absl::MutexLock lk(&progress_mu_);
    progress_callback_ = std::move(on_progress);
    completion_callback_ = std::move(on_complete);
  }

  // Returns true if all segments have completed
  bool mark_completion_and_is_done(uint64_t completion_bytes = 0) {
    int done = completed_.fetch_add(1) + 1;
    uint64_t bytes_to_report = completion_bytes;

    std::vector<PendingAckWindow> ready;
    std::function<void(const PendingAckWindow&)> sender;
    {
      absl::MutexLock lk(&ack_mu_);
      if (bytes_to_report == 0 && !completion_bytes_queue_.empty()) {
        bytes_to_report = completion_bytes_queue_.front();
        completion_bytes_queue_.pop_front();
      }
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
    if (bytes_to_report > 0) {
      notify_bytes_progress(bytes_to_report);
    }
    for (auto& window : ready) {
      if (sender) {
        sender(window);
      }
    }
    return rdma_final_window_received_.load(std::memory_order_acquire) &&
        done >= expected_completions_.load(std::memory_order_acquire);
  }

  void set_ack_sender(std::function<void(const PendingAckWindow&)> fn) {
    absl::MutexLock lk(&ack_mu_);
    ack_sender_ = std::move(fn);
  }

  void enqueue_window_ack(
      uint32_t window_seq,
      std::vector<uint64_t> offsets,
      bool final_window,
      uint32_t completion_count = 0) {
    absl::MutexLock lk(&ack_mu_);
    PendingAckWindow window;
    window.window_seq = window_seq;
    window.final_window = final_window;
    window.num_segments = static_cast<uint32_t>(offsets.size());
    window.ack_kind = AckKind::kOffsets;
    window.remaining = static_cast<int>(completion_count == 0 ? offsets.size() : completion_count);
    window.offsets = std::move(offsets);
    pending_ack_windows_.push_back(std::move(window));
    for (int i = 0; i < pending_ack_windows_.back().remaining; ++i) {
      segment_window_queue_.push_back(window_seq);
    }
  }

  void enqueue_plan_window_ack(
      uint32_t window_seq,
      uint32_t num_segments,
      uint32_t completion_count,
      bool final_window) {
    absl::MutexLock lk(&ack_mu_);
    PendingAckWindow window;
    window.window_seq = window_seq;
    window.final_window = final_window;
    window.num_segments = num_segments;
    window.ack_kind = AckKind::kSegmentCount;
    window.remaining = static_cast<int>(completion_count);
    pending_ack_windows_.push_back(std::move(window));
    for (int i = 0; i < pending_ack_windows_.back().remaining; ++i) {
      segment_window_queue_.push_back(window_seq);
    }
  }

  static std::future<read_result_t> get_read_result_future(std::string error_message);

  tensor_t local_tensor_;
  std::shared_ptr<PreparedReadPlan> prepared_plan_;
  remote_tensor_t remote_tensor_;
  std::string tensor_key_;
  std::string request_key_;
  std::string dst_ip_;
  uint16_t dst_port_;
  std::promise<read_result_t> result_;
  std::atomic_bool result_set_;
  absl::Mutex result_mu_;
  std::function<void()> on_result_ ABSL_GUARDED_BY(result_mu_);

  misc::Timer timer_;
  read_result_t status_;
  const std::chrono::steady_clock::time_point created_at_;
  std::atomic<uint64_t> request_first_response_us_{0};
  std::atomic<uint64_t> rdma_first_post_us_{0};
  std::atomic<uint64_t> rdma_first_completion_us_{0};
  std::atomic<uint64_t> rdma_last_completion_us_{0};
  std::atomic<uint64_t> rdma_response_windows_{0};
  std::atomic<uint64_t> rdma_response_segments_{0};
  std::atomic<uint64_t> rdma_wr_posted_{0};
  std::atomic<uint64_t> rdma_wc_completed_{0};
  std::atomic<uint64_t> rdma_ack_windows_{0};
  std::atomic<uint64_t> rdma_ack_segments_{0};
  std::atomic<uint64_t> rdma_handshake_queue_wait_us_{0};
  uint64_t remote_offset_;
  uint64_t request_id_;
  int16_t rail_id_;
  uint64_t total_bytes_ = 0;
  Kind kind_ = Kind::kTensor;
  bool rdma_profile_enabled_ = false;
  std::atomic<uint64_t> mtcp_stage_unit_hint_bytes_{0};

  // Number of expected RDMA READ completions for this request
  std::atomic<int> expected_completions_{0};
  std::atomic<int> completed_{0};
  std::atomic_bool rdma_final_window_received_{false};

  absl::Mutex ack_mu_;
  std::deque<PendingAckWindow> pending_ack_windows_ ABSL_GUARDED_BY(ack_mu_);
  std::deque<uint32_t> segment_window_queue_ ABSL_GUARDED_BY(ack_mu_);
  std::deque<uint32_t> completion_bytes_queue_ ABSL_GUARDED_BY(ack_mu_);
  std::function<void(const PendingAckWindow&)> ack_sender_ ABSL_GUARDED_BY(ack_mu_);

  void notify_bytes_progress(uint64_t bytes_delta);
  void notify_completion(const absl::Status& status);

  std::atomic<uint64_t> completed_bytes_{0};
  absl::Mutex progress_mu_;
  std::function<void(uint64_t, uint64_t)> progress_callback_ ABSL_GUARDED_BY(progress_mu_);
  std::function<void(const absl::Status&)> completion_callback_ ABSL_GUARDED_BY(progress_mu_);

  [[nodiscard]] uint64_t elapsed_since_create_us() const;
  void finalize_rdma_profile_status();
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
