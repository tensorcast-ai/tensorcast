// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TRANSPORT_RDMA_TRANSPORT_H_
#define CORE_COMMUNICATOR_TRANSPORT_RDMA_TRANSPORT_H_

#include <memory>

#include "core/communicator/transport/net_dev.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/communicator/transport/request.h"

namespace tensorcast::communicator {

struct RdmaTransportInfo {
  uint32_t mtu = IBV_MTU_4096;
  uint32_t ib_port = 0;
  uint32_t link_layer = 0;
  uint32_t qpn = 0;
  uint32_t lid = 0;
  uint32_t psn = 0;
  uint8_t gid[16] = {};
};

class RdmaContext;
typedef std::shared_ptr<RdmaContext> rdma_context_t;

class RdmaThread;
typedef std::shared_ptr<RdmaThread> rdma_thread_t;

class RdmaTransport {
 public:
  RdmaTransport(RdmaContext* context, net_dev_t dev, rdma_thread_t th);
  ~RdmaTransport();
  result_t read(read_request_t request);
  struct RdmaReadSeg {
    uint64_t local_addr;
    uint32_t length;
    uint64_t remote_addr;
    uint32_t rkey;
  };
  result_t read_multi(read_request_t request, const std::vector<RdmaReadSeg>& segs);
  result_t connect(RdmaTransportInfo* info);
  result_t get_local_info(RdmaTransportInfo* info);
  // Optional: set an ACK callback invoked when a READ completes.
  // The callback receives the completed request and can send control messages.
  void set_ack_callback(std::function<void(const read_request_t&)> cb) { ack_cb_ = std::move(cb); }

  bool ready() {
    return ready_.load();
  }

 protected:
  result_t do_init_qp();
  result_t do_modify_qp_rtr();
  result_t do_modify_qp_rts();

  bool ready_to_send();
  bool ready_to_recv();

  result_t do_post_send();
  result_t do_process_wc(struct ibv_wc* wc);
  result_t do_post_recv();

 public:
  int transport_idx_;

 private:
  RdmaContext* context_;
  net_dev_t dev_;
  rdma_thread_t io_thread_;
  struct ibv_qp* qp_{};
  union ibv_gid local_gid_;
  int gid_idx_;
  RdmaTransportInfo peer_info_;
  std::atomic_bool ready_;
  std::atomic_int inflight_send_;
  Queue<read_request_t> read_queue_;
  Queue<read_request_t> inflight_queue_;

  std::function<void(const read_request_t&)> ack_cb_;

  friend class RdmaThread;
};
typedef std::shared_ptr<RdmaTransport> rdma_transport_t;

} // namespace tensorcast::communicator

#endif // CORE_COMMUNICATOR_TRANSPORT_RDMA_TRANSPORT_H_
