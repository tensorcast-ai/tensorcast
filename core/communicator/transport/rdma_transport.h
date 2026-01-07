// Copyright (c) 2025-2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TRANSPORT_RDMA_TRANSPORT_H_
#define CORE_COMMUNICATOR_TRANSPORT_RDMA_TRANSPORT_H_

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "core/communicator/misc/mlx5_warp.h"
#include "core/communicator/transport/net_dev.h"
#include "core/communicator/transport/request.h"

namespace tensorcast::communicator::transport {

struct RdmaTransportInfo {
  uint32_t mtu = IBV_MTU_4096;
  uint32_t ib_port = 0;
  uint32_t link_layer = 0;
  uint32_t qpn = 0; // First QP number (backward compatibility)
  uint32_t lid = 0;
  uint32_t psn = 0;
  uint8_t gid[16] = {};
  // Multi-QP support: qp_count indicates number of QPs, qpns[0..qp_count-1] contain QPNs
  uint32_t qp_count = 1; // Number of QPs (1-16)
  uint32_t qpns[16] = {}; // QPN array for all QPs
};

class RdmaContext; // fwd decl to avoid circular include
using rdma_context_t = std::shared_ptr<RdmaContext>;

class RdmaThread; // fwd decl to avoid circular include
using rdma_thread_t = std::shared_ptr<RdmaThread>;

class RdmaTransport {
 public:
  RdmaTransport(RdmaContext* context, net_dev_t dev, rdma_thread_t th);
  ~RdmaTransport();
  misc::result_t read(read_request_t request);

  struct RdmaReadSeg {
    uint64_t local_addr;
    uint32_t length;
    uint64_t remote_addr;
    uint32_t rkey;
    uint32_t window_seq = 0;
    uint32_t segment_idx = 0;
  };

  misc::result_t read_multi(read_request_t request, const std::vector<RdmaReadSeg>& segs);
  misc::result_t connect(RdmaTransportInfo* info);
  misc::result_t get_local_info(RdmaTransportInfo* info);

  // Optional: set an ACK callback invoked when a READ completes.
  // The callback receives the completed request and can send control messages.
  void set_ack_callback(std::function<void(const read_request_t&)> cb) {
    ack_cb_ = std::move(cb);
  }

  bool ready() {
    return ready_.load();
  }

 protected:
  misc::result_t do_init_qp();
  misc::result_t do_modify_qp_rtr();
  misc::result_t do_modify_qp_rts();

  bool ready_to_send();
  bool ready_to_recv();

  misc::result_t do_post_send();
  misc::result_t do_process_wc(struct ibv_wc* wc);
  misc::result_t do_post_recv();

 public:
  int transport_index() const {
    return transport_index_;
  }

  void set_transport_index(int index) {
    transport_index_ = index;
  }

 private:
  RdmaContext* context_;
  net_dev_t dev_;
  rdma_thread_t io_thread_;
  std::vector<struct ibv_qp*> qps_;
  int qp_count_;
  std::atomic<int> next_qp_index_;
  union ibv_gid local_gid_;
  int gid_idx_;
  RdmaTransportInfo peer_info_;
  std::atomic_bool ready_;
  std::atomic_int inflight_send_;
  misc::Queue<read_request_t> read_queue_;
  misc::Queue<read_request_t> inflight_queue_;

  std::function<void(const read_request_t&)> ack_cb_;

  misc::result_t do_modify_qp_lag_port(struct ibv_qp* qp, int lag);

  friend class RdmaThread;
  int transport_index_{};
};

using rdma_transport_t = std::shared_ptr<RdmaTransport>;

} // namespace tensorcast::communicator::transport

#endif // CORE_COMMUNICATOR_TRANSPORT_RDMA_TRANSPORT_H_
