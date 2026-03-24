
// Copyright (c) 2025-2026, TensorCast Team.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::transport {

namespace {

bool is_truthy_env_value(const char* value) {
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0
      || std::strcmp(value, "True") == 0 || std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0;
}

std::atomic<bool>& rdma_profile_enabled_flag() {
  static std::atomic<bool> enabled(is_truthy_env_value(std::getenv("TENSORCAST_COMM_RDMA_READ_PROFILE")));
  return enabled;
}

} // namespace

bool ReadRequest::rdma_profile_enabled_for_process() {
  return rdma_profile_enabled_flag().load(std::memory_order_relaxed);
}

void ReadRequest::set_rdma_profile_enabled_for_process(bool enabled) {
  rdma_profile_enabled_flag().store(enabled, std::memory_order_relaxed);
}

ReadRequest::ReadRequest(
    std::string tensor_key,
    std::string dst_ip,
    uint16_t dst_port,
    tensor_t local,
    uint64_t remote_offset,
    uint64_t request_id,
    int rail_id)
    : local_tensor_(std::move(local)),
      tensor_key_(std::move(tensor_key)),
      dst_ip_(std::move(dst_ip)),
      dst_port_(dst_port),
      result_set_(false),
      timer_(true),
      created_at_(std::chrono::steady_clock::now()),
      remote_offset_(remote_offset),
      request_id_(request_id),
      rail_id_(rail_id),
      rdma_profile_enabled_(rdma_profile_enabled_for_process()) {
  status_.tensor_key = tensor_key_;
  status_.local_rail_id = rail_id_;
  if (local_tensor_ != nullptr) {
    auto dev = local_tensor_->get_dev();
    if (dev != nullptr) {
      status_.local_nic = dev->get_name();
    }
  }
}

tensor_t ReadRequest::get_local_tensor() const {
  return local_tensor_;
}

remote_tensor_t ReadRequest::get_remote_tensor() const {
  return remote_tensor_;
}

void ReadRequest::set_remote_tensor(remote_tensor_t tensor) {
  remote_tensor_ = std::move(tensor);
}

future_read_result_t ReadRequest::get_future() {
  return result_.get_future();
}

void ReadRequest::set_result(absl::Status status) {
  bool expected = false;
  if (!result_set_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    return;
  }
  finalize_rdma_profile_status();
  status_.status = status;
  result_.set_value(status_);
  std::function<void()> callback;
  {
    absl::MutexLock lock(&result_mu_);
    callback = std::move(on_result_);
  }
  if (callback) {
    callback();
  }
  notify_completion(status);
}

void ReadRequest::set_on_result(std::function<void()> callback) {
  bool invoke_now = false;
  {
    absl::MutexLock lock(&result_mu_);
    if (result_set_.load(std::memory_order_acquire)) {
      invoke_now = true;
    } else {
      on_result_ = std::move(callback);
    }
  }
  if (invoke_now && callback) {
    callback();
  }
}

bool ReadRequest::is_result_set() {
  return result_set_.load();
}

std::string ReadRequest::get_dst_url() {
  std::stringstream url;
  url << dst_ip_ << ":" << dst_port_;
  return url.str();
}

std::string ReadRequest::get_key() {
  return get_request_instance_key(tensor_key_, remote_offset_, request_id_);
}

void ReadRequest::record_request_response() {
  status_.request_cost = timer_.record();
  if (!rdma_profile_enabled_) {
    return;
  }
  const uint64_t elapsed_us = elapsed_since_create_us();
  uint64_t unset = 0;
  request_first_response_us_.compare_exchange_strong(unset, elapsed_us, std::memory_order_acq_rel);
}

void ReadRequest::record_read_done() {
  status_.read_cost = timer_.record();
}

void ReadRequest::record_rdma_queue_done() {
  status_.rdma_queue_cost = timer_.record();
}

void ReadRequest::record_rdma_regmr() {
  status_.rdma_regmr_cost = timer_.record();
}

void ReadRequest::note_rdma_response_window(uint32_t segment_count) {
  if (!rdma_profile_enabled_) {
    return;
  }
  rdma_response_windows_.fetch_add(1, std::memory_order_relaxed);
  rdma_response_segments_.fetch_add(segment_count, std::memory_order_relaxed);
}

void ReadRequest::note_rdma_posted_wr(uint32_t wr_count) {
  if (wr_count == 0) {
    return;
  }
  if (!rdma_profile_enabled_) {
    return;
  }
  rdma_wr_posted_.fetch_add(static_cast<uint64_t>(wr_count), std::memory_order_relaxed);
  const uint64_t elapsed_us = elapsed_since_create_us();
  uint64_t unset = 0;
  rdma_first_post_us_.compare_exchange_strong(unset, elapsed_us, std::memory_order_acq_rel);
}

void ReadRequest::note_rdma_completion() {
  if (!rdma_profile_enabled_) {
    return;
  }
  rdma_wc_completed_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t elapsed_us = elapsed_since_create_us();
  uint64_t unset = 0;
  rdma_first_completion_us_.compare_exchange_strong(unset, elapsed_us, std::memory_order_acq_rel);
  rdma_last_completion_us_.store(elapsed_us, std::memory_order_relaxed);
}

void ReadRequest::note_rdma_ack_window(uint32_t segment_count) {
  if (!rdma_profile_enabled_) {
    return;
  }
  rdma_ack_windows_.fetch_add(1, std::memory_order_relaxed);
  rdma_ack_segments_.fetch_add(segment_count, std::memory_order_relaxed);
}

void ReadRequest::note_rdma_handshake_queue_wait_us(uint64_t wait_us) {
  if (wait_us == 0) {
    return;
  }
  if (!rdma_profile_enabled_) {
    return;
  }
  rdma_handshake_queue_wait_us_.fetch_add(wait_us, std::memory_order_relaxed);
}

std::future<read_result_t> ReadRequest::get_read_result_future(std::string error_message) {
  std::promise<read_result_t> promise;
  auto f = promise.get_future();
  promise.set_value(
      ReadResult{
          .status = absl::InternalError(error_message),
      });
  return f;
}

void ReadRequest::notify_bytes_progress(uint64_t bytes_delta) {
  if (bytes_delta == 0) {
    return;
  }
  const uint64_t done = completed_bytes_.fetch_add(bytes_delta, std::memory_order_relaxed) + bytes_delta;
  std::function<void(uint64_t, uint64_t)> callback;
  {
    absl::MutexLock lk(&progress_mu_);
    callback = progress_callback_;
  }
  if (callback) {
    callback(done, local_tensor_ ? local_tensor_->get_bytes() : 0);
  }
}

void ReadRequest::notify_completion(const absl::Status& status) {
  std::function<void(const absl::Status&)> callback;
  {
    absl::MutexLock lk(&progress_mu_);
    callback = completion_callback_;
  }
  if (callback) {
    callback(status);
  }
}

uint64_t ReadRequest::elapsed_since_create_us() const {
  const auto now = std::chrono::steady_clock::now();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now - created_at_).count());
}

void ReadRequest::finalize_rdma_profile_status() {
  if (!rdma_profile_enabled_) {
    return;
  }
  status_.request_first_response_us = request_first_response_us_.load(std::memory_order_relaxed);
  status_.rdma_first_post_us = rdma_first_post_us_.load(std::memory_order_relaxed);
  status_.rdma_first_completion_us = rdma_first_completion_us_.load(std::memory_order_relaxed);
  status_.rdma_last_completion_us = rdma_last_completion_us_.load(std::memory_order_relaxed);
  status_.rdma_response_windows = rdma_response_windows_.load(std::memory_order_relaxed);
  status_.rdma_response_segments = rdma_response_segments_.load(std::memory_order_relaxed);
  status_.rdma_wr_posted = rdma_wr_posted_.load(std::memory_order_relaxed);
  status_.rdma_wc_completed = rdma_wc_completed_.load(std::memory_order_relaxed);
  status_.rdma_ack_windows = rdma_ack_windows_.load(std::memory_order_relaxed);
  status_.rdma_ack_segments = rdma_ack_segments_.load(std::memory_order_relaxed);
  status_.rdma_handshake_queue_wait_us = rdma_handshake_queue_wait_us_.load(std::memory_order_relaxed);
  if (status_.rdma_last_completion_us >= status_.rdma_first_post_us && status_.rdma_first_post_us > 0) {
    status_.rdma_post_to_last_completion_us = status_.rdma_last_completion_us - status_.rdma_first_post_us;
  } else {
    status_.rdma_post_to_last_completion_us = 0;
  }
}

WriteRequest::WriteRequest(tensor_t local_tensor, std::string tensor_key, uint64_t offset, uint64_t bytes)
    : local_tensor_(local_tensor), tensor_key_(tensor_key), offset_(offset), bytes_(bytes) {}

std::string WriteRequest::get_key() {
  return get_request_key(tensor_key_, offset_);
}

} // namespace tensorcast::communicator::transport
