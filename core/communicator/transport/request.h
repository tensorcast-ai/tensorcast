// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_REQUEST_H_
#define CORE_COMMUNICATOR_ENGINE_REQUEST_H_

#include <string>

#include "absl/status/status.h"

#include "core/communicator/misc/metric.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator {

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

  static std::future<read_result_t> get_read_result_future(std::string error_message);

  tensor_t local_tensor_;
  remote_tensor_t remote_tensor_;
  std::string tensor_key_;
  std::string dst_ip_;
  uint16_t dst_port_;
  std::promise<read_result_t> result_;
  std::atomic_bool result_set_;

  Timer timer_;
  read_result_t status_;
  uint64_t remote_offset_;
};

typedef std::shared_ptr<ReadRequest> read_request_t;

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

} // namespace tensorcast::communicator

#endif // CORE_COMMUNICATOR_ENGINE_REQUEST_H_
