// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef COMMUNICATOR_TRANSPORT_PARTITION_TENSOR_H_
#define COMMUNICATOR_TRANSPORT_PARTITION_TENSOR_H_

#include <memory>
#include <string>

#include "core/communicator/transport/net_dev.h"

namespace stepcast::communicator {

class NetDev;
typedef std::shared_ptr<NetDev> net_dev_t;

class PartitionTensor {
 public:
  PartitionTensor(std::string tensor_key, uint64_t addr, uint64_t bytes, int mem_type, net_dev_t dev = nullptr);
  ~PartitionTensor();

  void register_mr();
  void set_read_ready();
  void wait_read_ready();

  uint64_t get_regmr_cost() const;

  std::string get_key();
  uint64_t get_bytes() const;
  struct ibv_mr* get_mr();
  net_dev_t get_dev();
  uint64_t get_uint64_addr() const;
  int get_mem_type() const;

  // Check if this tensor needs GPU->CPU staging for TCP transport
  [[nodiscard]] bool needs_staging() const {
    return needs_staging_;
  }
  void set_needs_staging(bool value) {
    needs_staging_ = value;
  }

  template <class T>
  T* get_addr() {
    return reinterpret_cast<T*>(addr_);
  }

  // Get device ID (for GPU tensors)
  [[nodiscard]] int get_device_id() const {
    return device_id_;
  }
  void set_device_id(int id) {
    device_id_ = id;
  }

 private:
  std::string tensor_key_;

  uint64_t addr_;
  uint64_t bytes_;
  std::atomic_bool registered_;
  std::atomic_bool ready_;
  struct ibv_mr* mr_;
  net_dev_t dev_;
  uint64_t regmr_cost_;
  int mem_type_;
  bool needs_staging_ = false; // Whether GPU->CPU staging is needed for TCP transport
  int device_id_ = -1; // GPU device ID (-1 for CPU)
};
typedef std::shared_ptr<PartitionTensor> tensor_t;

class RemotePartitionTensor {
 public:
  RemotePartitionTensor(std::string tensor_key, std::string net_dev, uint64_t addr, uint64_t bytes, uint32_t rkey);
  ~RemotePartitionTensor() = default;
  std::string get_key() const;
  uint64_t get_bytes() const;
  uint32_t get_rkey() const;
  std::string get_net_dev() const;
  uint64_t get_uint64_addr() const;

 private:
  std::string tensor_key_;
  std::string net_dev_;

  uint64_t addr_;
  uint64_t bytes_;
  uint32_t rkey_;
};
typedef std::shared_ptr<RemotePartitionTensor> remote_tensor_t;

} // namespace stepcast::communicator

#endif // COMMUNICATOR_TRANSPORT_PARTITION_TENSOR_H_
