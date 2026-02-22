
// Copyright (c) 2025-2026, TensorCast Team.

#include <string>
#include <utility>

#include "absl/status/status.h"

#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::transport {

ReadRequest::ReadRequest(
    std::string tensor_key,
    std::string dst_ip,
    uint16_t dst_port,
    tensor_t local,
    uint64_t remote_offset,
    int rail_id)
    : local_tensor_(std::move(local)),
      tensor_key_(std::move(tensor_key)),
      dst_ip_(std::move(dst_ip)),
      dst_port_(dst_port),
      result_set_(false),
      timer_(true),
      remote_offset_(remote_offset),
      rail_id_(rail_id) {
  status_.tensor_key = tensor_key_;
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
  status_.status = status;
  result_.set_value(status_);
  result_set_.store(true);
  notify_completion(status);
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
  return get_request_key(tensor_key_, remote_offset_);
}

void ReadRequest::record_request_response() {
  status_.request_cost = timer_.record();
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

WriteRequest::WriteRequest(tensor_t local_tensor, std::string tensor_key, uint64_t offset, uint64_t bytes)
    : local_tensor_(local_tensor), tensor_key_(tensor_key), offset_(offset), bytes_(bytes) {}

std::string WriteRequest::get_key() {
  return get_request_key(tensor_key_, offset_);
}

} // namespace tensorcast::communicator::transport
