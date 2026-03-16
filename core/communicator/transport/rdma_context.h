// Copyright (c) 2025-2026, TensorCast Team.

#ifndef COMMUNICATOR_TRANSPORT_RDMA_CONTEXT_H_
#define COMMUNICATOR_TRANSPORT_RDMA_CONTEXT_H_

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/communicator/base/constants.h"
#include "core/communicator/transport/net_dev.h"

namespace tensorcast::communicator::transport {

class RdmaThread;
using rdma_thread_t = std::shared_ptr<RdmaThread>;

class RdmaTransport;
using rdma_transport_t = std::shared_ptr<RdmaTransport>;

class RdmaContext {
 public:
  RdmaContext();
  ~RdmaContext();

  net_dev_t get_best_dev(int dev_type, int dev_id, int rail_id, const std::string& key);

  net_dev_t get_dev(const std::string& name);

  /**
   * Get the best net dev for the gpu
   * @param gpu_id the gpu index
   * @return the net dev supporting gdr
   */
  net_dev_t get_best_dev(int gpu_id);

  /**
   * Get the best dev name
   * @param gpu_id the gpu index
   * @return the net dev name supporting gdr
   */
  std::string get_best_dev_name(int gpu_id);

  net_dev_t get_dev_by_rail(int rail_id);

  rdma_transport_t create_transport(const std::string& dev_name);

  // Expose list of RDMA devices for warmup/registration.
  const std::vector<net_dev_t>& list_devs() const {
    return devs_;
  }

  // Typed tuning for RDMA QPs
  void set_qp_params(int traffic_class, int qp_timeout, int qp_retry) {
    traffic_class_ = traffic_class;
    qp_timeout_ = qp_timeout;
    qp_retry_ = qp_retry;
  }

  void set_outstanding_wr(int outstanding_wr) {
    outstanding_wr_ = outstanding_wr;
  }

  // Multi-QP configuration
  void set_multi_qp_config(int qp_count, bool bonding_balance) {
    qp_count_ = qp_count;
    bonding_balance_ = bonding_balance;
  }

  int traffic_class() const {
    return traffic_class_;
  }

  int qp_timeout() const {
    return qp_timeout_;
  }

  int qp_retry() const {
    return qp_retry_;
  }

  int qp_count() const {
    return qp_count_;
  }

  int outstanding_wr() const {
    return outstanding_wr_;
  }

  bool bonding_balance() const {
    return bonding_balance_;
  }

 private:
  misc::result_t ibv_init();

  std::vector<net_dev_t> devs_;
  std::unordered_map<int, net_dev_t> rail_devs_;
  std::vector<rdma_thread_t> io_threads_;
  std::array<net_dev_t, 16> dev_vector_;
  int traffic_class_ = 186;
  int qp_timeout_ = 20;
  int qp_retry_ = 7;
  int qp_count_ = 1;
  int outstanding_wr_ = 64;
  bool bonding_balance_ = false;
};

using rdma_context_t = std::shared_ptr<RdmaContext>;

class RdmaTransport;

class RdmaThread {
 public:
  explicit RdmaThread(net_dev_t dev);
  ~RdmaThread();

  void stop();
  void notify_send();
  void notify_poll();
  void notify_recv();
  misc::result_t register_transport(RdmaTransport*);
  misc::result_t unregister_transport(RdmaTransport*);

 private:
  void send_loop();
  void poll_loop();
  void recv_loop();

  void add_send_transport(RdmaTransport* t);
  void add_poll_transport(RdmaTransport* t);
  void del_send_transport(RdmaTransport* t);
  void del_poll_transport(RdmaTransport* t);
  void add_recv_transport(RdmaTransport* t);
  void del_recv_transport(RdmaTransport* t);

  std::array<RdmaTransport*, 1024> send_transports_{};
  std::array<RdmaTransport*, 1024> poll_transports_{};
  std::array<RdmaTransport*, 1024> recv_transports_{};

  std::thread send_thread_;
  std::thread poll_thread_;
  std::thread recv_thread_;

  mutable std::mutex send_mu_;
  mutable std::mutex poll_mu_;
  mutable std::mutex recv_mu_;

  std::condition_variable send_cv_;
  std::condition_variable poll_cv_;
  std::condition_variable recv_cv_;

  std::atomic_bool stop_;
  net_dev_t net_dev_;
};

} // namespace tensorcast::communicator::transport

#endif // COMMUNICATOR_TRANSPORT_RDMA_CONTEXT_H_
