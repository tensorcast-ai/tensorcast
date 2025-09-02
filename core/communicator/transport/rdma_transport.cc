// Copyright (c) 2025, TensorCast Team.

#include <utility>

#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/communicator/transport/rdma_transport.h"

namespace tensorcast::communicator::transport {

RdmaTransport::RdmaTransport(RdmaContext* context, net_dev_t dev, rdma_thread_t th)
    : context_(context),
      dev_(std::move(dev)),
      io_thread_(std::move(th)),
      local_gid_({}),
      gid_idx_(-1),
      ready_(false),
      inflight_send_(0) {
  io_thread_->register_transport(this);
  CHECK_WARN(do_init_qp(), "failed to init qp");
}

RdmaTransport::~RdmaTransport() {
  while (!read_queue_.empty()) {
    auto req = read_queue_.pop();
    req->set_result(absl::InternalError("failed to read due to closed transport"));
  }

  io_thread_->unregister_transport(this);
  if (qp_ != nullptr) {
    CHECK_WARN(misc::wrap_ibv_destroy_qp(qp_), "failed to destroy qp");
    qp_ = nullptr;
  }
}

misc::result_t RdmaTransport::read(read_request_t request) {
  COMM_CHECK(read_queue_.push(request));
  if (io_thread_ != nullptr) {
    io_thread_->notify_send();
  }
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::connect(RdmaTransportInfo* info) {
  memcpy(&peer_info_, info, sizeof(RdmaTransportInfo));
  COMM_CHECK(do_modify_qp_rtr());
  COMM_CHECK(do_modify_qp_rts());
  ready_.store(true);
  if (io_thread_ != nullptr) {
    io_thread_->notify_send();
    io_thread_->notify_recv();
  }
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::get_local_info(RdmaTransportInfo* info) {
  info->link_layer = dev_->get_link();
  info->ib_port = dev_->get_port();
  info->mtu = IBV_MTU_4096;
  info->psn = 0;
  info->qpn = qp_->qp_num;
  info->lid = 0;
  memcpy(info->gid, local_gid_.raw, 16);
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_init_qp() {
  struct ibv_qp_init_attr init_attr{};
  misc::CLEAR(init_attr);
  init_attr.send_cq = dev_->get_cq();
  init_attr.recv_cq = dev_->get_cq();
  init_attr.qp_type = IBV_QPT_RC;
  init_attr.cap.max_send_wr = 64;
  init_attr.cap.max_recv_wr = 16;
  init_attr.cap.max_send_sge = 1;
  init_attr.cap.max_recv_sge = 1;
  init_attr.cap.max_inline_data = 256;

  COMM_CHECK(dev_->create_qp(&qp_, &init_attr));

  struct ibv_qp_attr qp_attr{};
  misc::CLEAR(qp_attr);
  qp_attr.qp_state = IBV_QPS_INIT;
  qp_attr.pkey_index = 0;
  qp_attr.port_num = dev_->get_port();
  qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
  COMM_CHECK(
      misc::wrap_ibv_modify_qp(qp_, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));

  COMM_CHECK(dev_->get_best_gid(&local_gid_, &gid_idx_));

  LOG(INFO) << "init qp done:"
            << " dev=" << dev_->get_name() << ", qpn=" << qp_->qp_num << ", type=" << qp_->qp_type
            << ", gid-idx=" << gid_idx_ << ", gid=" << misc::gid2str(local_gid_.raw);
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_modify_qp_rtr() {
  struct ibv_qp_attr qp_attr{};
  misc::CLEAR(qp_attr);
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = IBV_MTU_4096;
  qp_attr.dest_qp_num = peer_info_.qpn;
  qp_attr.rq_psn = peer_info_.psn;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 12;
  qp_attr.ah_attr.is_global = 1;

  memcpy(qp_attr.ah_attr.grh.dgid.raw, peer_info_.gid, 16);
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.sgid_index = gid_idx_;
  qp_attr.ah_attr.grh.hop_limit = 255;
  qp_attr.ah_attr.grh.traffic_class = context_->traffic_class();
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = peer_info_.ib_port;
  COMM_CHECK(
      misc::wrap_ibv_modify_qp(
          qp_,
          &qp_attr,
          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
              IBV_QP_MIN_RNR_TIMER));
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_modify_qp_rts() {
  struct ibv_qp_attr qp_attr{};
  misc::CLEAR(qp_attr);
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.timeout = context_->qp_timeout();
  qp_attr.retry_cnt = context_->qp_retry();
  qp_attr.rnr_retry = 7;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = 1;
  COMM_CHECK(
      misc::wrap_ibv_modify_qp(
          qp_,
          &qp_attr,
          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
              IBV_QP_MAX_QP_RD_ATOMIC));
  return misc::SUCCESS;
}

bool RdmaTransport::ready_to_send() {
  if (!ready_.load()) {
    return false;
  }

  return inflight_send_.load() <= 4 && !read_queue_.empty();
}

bool RdmaTransport::ready_to_recv() {
  if (!ready_.load()) {
    return false;
  }
  return false;
}

misc::result_t RdmaTransport::do_post_send() {
  auto req = read_queue_.pop(true);
  if (req == nullptr) {
    return misc::FAILED;
  }

  req->record_rdma_queue_done();
  req->set_expected_completions(1);

  struct ibv_send_wr read_wr{};
  struct ibv_send_wr* read_bad_wr;
  struct ibv_sge read_sge{};

  auto local_tensor = req->get_local_tensor();
  auto remote_tensor = req->get_remote_tensor();

  read_wr.wr_id = transport_index_;
  read_wr.opcode = IBV_WR_RDMA_READ;
  read_wr.send_flags = IBV_SEND_SIGNALED;

  read_wr.wr.rdma.remote_addr = remote_tensor->get_uint64_addr();
  read_wr.wr.rdma.rkey = remote_tensor->get_rkey();
  read_wr.next = nullptr;
  read_wr.num_sge = 1;

  auto mr = local_tensor->get_mr();

  req->record_rdma_regmr();

  read_wr.sg_list = &read_sge;
  read_sge.addr = local_tensor->get_uint64_addr();
  read_sge.length = local_tensor->get_bytes();
  read_sge.lkey = mr->lkey;

  inflight_queue_.push(req);
  auto res = misc::wrap_ibv_post_send(qp_, &read_wr, &read_bad_wr);
  if (res) {
    req->set_result(
        absl::InternalError(absl::StrFormat("rdma post_send failed: return=%d, error=%s", res, strerror(errno))));
    return misc::FAILED;
  }
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::read_multi(read_request_t request, const std::vector<RdmaReadSeg>& segs) {
  if (segs.empty()) {
    return misc::FAILED;
  }

  // Ensure QP is ready
  if (!ready_.load()) {
    CHECK_WARN(do_modify_qp_rtr(), "failed to modify qp rtr");
    CHECK_WARN(do_modify_qp_rts(), "failed to modify qp rts");
    ready_.store(true);
  }

  // Prepare batch of WRs
  std::vector<ibv_send_wr> wrs(segs.size());
  std::vector<ibv_sge> sges(segs.size());
  struct ibv_send_wr* bad_wr = nullptr;

  auto mr = request->get_local_tensor()->get_mr();
  request->record_rdma_queue_done();
  request->set_expected_completions(static_cast<int>(segs.size()));

  for (size_t i = 0; i < segs.size(); ++i) {
    auto& wr = wrs[i];
    auto& sge = sges[i];
    misc::CLEAR(wr);
    misc::CLEAR(sge);
    wr.wr_id = transport_index_;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.num_sge = 1;
    wr.sg_list = &sge;
    wr.wr.rdma.remote_addr = segs[i].remote_addr;
    wr.wr.rdma.rkey = segs[i].rkey;
    sge.addr = segs[i].local_addr;
    sge.length = segs[i].length;
    sge.lkey = mr->lkey;
    wr.next = (i + 1 < segs.size()) ? &wrs[i + 1] : nullptr;
  }

  // Track N inflight completions for this request
  for (size_t i = 0; i < segs.size(); ++i) {
    inflight_queue_.push(request);
  }

  auto res = misc::wrap_ibv_post_send(qp_, wrs.data(), &bad_wr);
  if (res) {
    request->set_result(
        absl::InternalError(
            absl::StrFormat("rdma post_send (multi) failed: return=%d, error=%s", res, strerror(errno))));
    return misc::FAILED;
  }
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_process_wc(struct ibv_wc* wc) {
  if (wc->opcode == IBV_WC_RDMA_READ) {
    auto req = inflight_queue_.pop(true);
    if (req == nullptr) {
      misc::ASSERT(false, "abnormal queue state");
    }
    req->record_read_done();
    if (wc->status == IBV_WC_SUCCESS) {
      if (req->mark_completion_and_is_done()) {
        req->set_result(absl::OkStatus());
        // Invoke per-request ACK once, after aggregate completion
        req->invoke_ack_action_once();
      }
    } else {
      LOG(WARNING) << "process err wc: status=" << wc->status;
      req->set_result(
          absl::InternalError(absl::StrFormat("failed to process work completion: wc_status=%d", wc->status)));
    }
  }
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_post_recv() {
  return misc::SUCCESS;
}

} // namespace tensorcast::communicator::transport
