// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TRANSPORT_NET_DEV_H_
#define CORE_COMMUNICATOR_TRANSPORT_NET_DEV_H_

#include <memory>
#include <string>

#include <fstream>
#include <regex>
#include <sstream>

#include "core/communicator/misc/common.h"
#include "core/communicator/misc/ibv_wrap.h"
#include "core/communicator/misc/queue.h"

namespace tensorcast::communicator::transport {

class PartitionTensor;
typedef std::shared_ptr<PartitionTensor> tensor_t;

class NetDev {
 public:
  NetDev(struct ibv_context* context, int dev_id, struct ibv_device* dev, int port_id, struct ibv_port_attr port);
  ~NetDev();

  /**
   * Get best gid index
   * @return gid index
   */
  int get_best_gid_index();
  misc::result_t get_best_gid(ibv_gid* gid, int* gid_idx);

  int get_port() const;
  int get_link() const;
  int16_t get_rail_id() const;
  uint8_t get_numa_id() const;
  misc::result_t read_numa_id();

  int get_dev_id() {
    return dev_id_;
  }

  ibv_pd* get_pd() const;
  ibv_cq* get_cq() const;
  misc::result_t reg_async(const tensor_t& tensor);
  misc::result_t reg_mr(struct ibv_mr** ret, void* addr, size_t length, int access) const;
  misc::result_t create_qp(struct ibv_qp** ret, struct ibv_qp_init_attr* qp_init_attr) const;

  std::string get_name();
  char* get_pci_path();

 private:
  misc::result_t read_pci_path();
  void read_rail_id();
  void register_loop();

 protected:
  std::string dev_name_;
  ibv_context* context_ = nullptr;

  int dev_id_;
  uint64_t guid_;
  uint8_t port_;
  uint8_t link_;
  char* pci_path_;
  int real_port_;
  int max_qp_num_;
  int gid_tbl_len_;
  uint8_t numa_id_;

  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;

  int16_t rail_id_;
  std::atomic_bool stop_;
  std::thread register_thread_;
  misc::Queue<tensor_t> register_queue_;

  int gid_idx_ = -1;
  ibv_gid gid_;
};

typedef std::shared_ptr<NetDev> net_dev_t;

} // namespace tensorcast::communicator::transport

#endif // #define CORE_COMMUNICATOR_TRANSPORT_NET_DEV_H_
