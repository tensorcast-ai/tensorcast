// Copyright (c) 2025, TensorCast Team.

#ifndef COMMUNICATOR_TRANSPORT_PARTITION_TENSOR_H_
#define COMMUNICATOR_TRANSPORT_PARTITION_TENSOR_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "core/communicator/transport/net_dev.h"

namespace tensorcast::communicator::transport {

class NetDev;
typedef std::shared_ptr<NetDev> net_dev_t;

class PartitionTensor {
 public:
  PartitionTensor(std::string tensor_key, uint64_t addr, uint64_t bytes, int mem_type, net_dev_t dev = nullptr);
  ~PartitionTensor();

  void add_dev_and_register(const net_dev_t& dev);
  void add_dev(const net_dev_t& dev);
  void add_dev_list(const std::vector<net_dev_t>& devs);

  void register_mr(const NetDev* dev);
  void set_read_ready();
  void set_read_unready();
  void wait_read_ready();

  bool is_registered(const net_dev_t& dev);
  uint64_t get_regmr_cost(const net_dev_t& dev);
  uint64_t get_regmr_cost() const;

  std::string get_key();
  uint64_t get_bytes() const;
  struct ibv_mr* get_mr(const net_dev_t& dev);
  struct ibv_mr* get_mr_by_rail(int16_t rail_id);
  std::vector<net_dev_t> get_devs();

  net_dev_t get_dev();

  net_dev_t get_dev_by_rail(int rail_id);
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
  absl::flat_hash_map<std::string, std::shared_ptr<std::atomic_bool>> registered_;
  std::atomic_bool ready_;
  absl::flat_hash_map<std::string, struct ibv_mr*> mrs_;
  std::vector<net_dev_t> devs_;
  absl::flat_hash_map<std::string, uint64_t> regmr_costs_;
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

} // namespace tensorcast::communicator::transport

#endif // COMMUNICATOR_TRANSPORT_PARTITION_TENSOR_H_
