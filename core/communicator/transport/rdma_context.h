// Copyright (c) 2025, TensorCast Team.

#ifndef COMMUNICATOR_TRANSPORT_RDMA_CONTEXT_H_
#define COMMUNICATOR_TRANSPORT_RDMA_CONTEXT_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/communicator/transport/net_dev.h"
#include "core/communicator/transport/rdma_transport.h"

namespace tensorcast::communicator {

class RdmaThread;
typedef std::shared_ptr<RdmaThread> rdma_thread_t;

class RdmaTransport;
typedef std::shared_ptr<RdmaTransport> rdma_transport_t;

class RdmaContext {
 public:
  RdmaContext();
  ~RdmaContext();

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

  rdma_transport_t create_transport(const std::string& dev_name);

  // Expose list of RDMA devices for warmup/registration.
  const std::vector<net_dev_t>& list_devs() const { return devs_; }

  // Typed tuning for RDMA QPs
  void set_qp_params(int traffic_class, int qp_timeout, int qp_retry) {
    traffic_class_ = traffic_class;
    qp_timeout_ = qp_timeout;
    qp_retry_ = qp_retry;
  }
  int traffic_class() const { return traffic_class_; }
  int qp_timeout() const { return qp_timeout_; }
  int qp_retry() const { return qp_retry_; }

 private:
  result_t ibv_init();

 private:
  std::vector<net_dev_t> devs_;
  std::vector<rdma_thread_t> io_threads_;
  std::array<net_dev_t, 16> dev_vector_;
  int traffic_class_ = 186;
  int qp_timeout_ = 20;
  int qp_retry_ = 7;
};
typedef std::shared_ptr<RdmaContext> rdma_context_t;

class RdmaTransport;
class RdmaThread {
 public:
  explicit RdmaThread(net_dev_t dev);
  ~RdmaThread();

  void stop();
  void notify_send();
  void notify_poll();
  void notify_recv();
  result_t register_transport(RdmaTransport*);
  result_t unregister_transport(RdmaTransport*);

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

 private:
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

  std::atomic_bool stop_{};
  net_dev_t net_dev_;
};

} // namespace tensorcast::communicator

#endif // COMMUNICATOR_TRANSPORT_RDMA_CONTEXT_H_
