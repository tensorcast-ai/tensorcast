// Copyright (c) 2025-2026, TensorCast Team.

#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/communicator/transport/rdma_transport.h"

namespace tensorcast::communicator::transport {

RdmaTransport::RdmaTransport(RdmaContext* context, net_dev_t dev, rdma_thread_t th)
    : context_(context),
      dev_(std::move(dev)),
      io_thread_(std::move(th)),
      qp_count_(context->qp_count()),
      next_qp_index_(0),
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

  // Destroy all QPs
  for (auto qp : qps_) {
    if (qp != nullptr) {
      CHECK_WARN(misc::wrap_ibv_destroy_qp(qp), "failed to destroy qp");
    }
  }
  qps_.clear();
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
  misc::result_t res = do_modify_qp_rtr();
  if (res != misc::SUCCESS) {
    ready_.store(false);
    std::memset(&peer_info_, 0, sizeof(peer_info_));
    return res;
  }
  res = do_modify_qp_rts();
  if (res != misc::SUCCESS) {
    ready_.store(false);
    std::memset(&peer_info_, 0, sizeof(peer_info_));
    return res;
  }

  // Apply bonding balance if enabled in config
  if (context_->bonding_balance() && qp_count_ > 1) {
    LOG(INFO) << "Applying LAG port balancing for " << qp_count_ << " QPs";
    for (size_t i = 0; i < qps_.size(); ++i) {
      int lag = (i % 2) + 1;
      do_modify_qp_lag_port(qps_[i], lag);
    }
  }

  ready_.store(true);
  if (io_thread_ != nullptr) {
    io_thread_->notify_send();
    io_thread_->notify_recv();
  }
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::get_local_info(RdmaTransportInfo* info) {
  if (qps_.empty()) {
    return misc::FAILED;
  }
  info->link_layer = dev_->get_link();
  info->ib_port = dev_->get_port();
  info->mtu = IBV_MTU_4096;
  info->psn = 0;
  info->qpn = qps_[0]->qp_num; // First QP (backward compatibility)
  info->lid = 0;
  std::memcpy(info->gid, local_gid_.raw, 16);

  // Fill QPN array for multi-QP support
  info->qp_count = static_cast<uint32_t>(qps_.size());
  for (size_t i = 0; i < qps_.size() && i < 16; ++i) {
    info->qpns[i] = qps_[i]->qp_num;
  }

  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_init_qp() {
  // Validate QP count from config
  if (qp_count_ <= 0) {
    LOG(WARNING) << "Invalid qp_count: " << qp_count_ << ", using default 1";
    qp_count_ = 1;
  } else if (qp_count_ > 16) {
    LOG(WARNING) << "qp_count too large: " << qp_count_ << ", limiting to 16";
    qp_count_ = 16;
  }

  LOG(INFO) << "Initializing " << qp_count_ << " QPs for device " << dev_->get_name()
            << " (bonding_balance=" << (context_->bonding_balance() ? "true" : "false") << ")";

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

  // Create all QPs
  for (int i = 0; i < qp_count_; i++) {
    struct ibv_qp* qp = nullptr;
    COMM_CHECK(dev_->create_qp(&qp, &init_attr));
    qps_.push_back(qp);

    // Initialize QP state
    struct ibv_qp_attr qp_attr{};
    misc::CLEAR(qp_attr);
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = dev_->get_port();
    qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
    COMM_CHECK(
        misc::wrap_ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  }

  COMM_CHECK(dev_->get_best_gid(&local_gid_, &gid_idx_));

  LOG(INFO) << "init " << qp_count_ << " qps done:"
            << " dev=" << dev_->get_name() << ", first_qpn=" << qps_[0]->qp_num << ", gid-idx=" << gid_idx_
            << ", gid=" << misc::gid2str(local_gid_.raw);
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

  std::memcpy(qp_attr.ah_attr.grh.dgid.raw, peer_info_.gid, 16);
  qp_attr.ah_attr.grh.flow_label = 0;
  qp_attr.ah_attr.grh.sgid_index = gid_idx_;
  qp_attr.ah_attr.grh.hop_limit = 255;
  qp_attr.ah_attr.grh.traffic_class = context_->traffic_class();
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  // `port_num` must refer to the local device port that owns this QP. Using the
  // peer's advertised port can map to a non-existent local port (e.g. mlx5_0 has
  // only port 1 while the peer runs on mlx5_9:2), which causes `ibv_modify_qp`
  // to fail with ENODEV during RTR. Always use the port associated with the
  // local NetDev to avoid that mismatch.
  qp_attr.ah_attr.port_num = dev_->get_port();

  // Determine number of remote QPNs available
  // Backward compatibility: if qp_count is 0, assume single QP mode (old protocol)
  uint32_t remote_qp_count = peer_info_.qp_count;
  bool use_multi_qp = (remote_qp_count > 1);

  // Modify each QP to RTR state with its corresponding remote QPN
  for (size_t i = 0; i < qps_.size(); ++i) {
    // For multi-QP: use corresponding remote QPN from array
    if (use_multi_qp && i < remote_qp_count && peer_info_.qpns[i] != 0) {
      qp_attr.dest_qp_num = peer_info_.qpns[i];
    } else if (remote_qp_count == 1 && peer_info_.qpns[0] != 0) {
      // Single QP mode with new protocol: use qpns[0]
      qp_attr.dest_qp_num = peer_info_.qpns[0];
    } else {
      // Backward compatibility: use qpn field (old protocol)
      qp_attr.dest_qp_num = peer_info_.qpn;
    }

    COMM_CHECK(
        misc::wrap_ibv_modify_qp(
            qps_[i],
            &qp_attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                IBV_QP_MIN_RNR_TIMER));

    LOG(INFO) << "QP[" << i << "] (local_qpn=" << qps_[i]->qp_num
              << ") configured to connect to remote_qpn=" << qp_attr.dest_qp_num;
  }
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

  // Modify all QPs to RTS state
  for (auto qp : qps_) {
    COMM_CHECK(
        misc::wrap_ibv_modify_qp(
            qp,
            &qp_attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                IBV_QP_MAX_QP_RD_ATOMIC));
  }
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
  req->add_expected_completions(1);

  struct ibv_send_wr read_wr{};
  struct ibv_send_wr* read_bad_wr;
  struct ibv_sge read_sge{};

  auto local_tensor = req->get_local_tensor();
  auto remote_tensor = req->get_remote_tensor();

  // Select QP using round robin first to encode in wr_id
  int qp_index = next_qp_index_.fetch_add(1) % qp_count_;
  struct ibv_qp* selected_qp = qps_[qp_index];

  // Encode both transport_index and qp_index in wr_id for completion routing
  read_wr.wr_id = (static_cast<uint64_t>(transport_index_) << 16) | static_cast<uint64_t>(qp_index);
  read_wr.opcode = IBV_WR_RDMA_READ;
  read_wr.send_flags = IBV_SEND_SIGNALED;

  read_wr.wr.rdma.remote_addr = remote_tensor->get_uint64_addr();
  read_wr.wr.rdma.rkey = remote_tensor->get_rkey();
  read_wr.next = nullptr;
  read_wr.num_sge = 1;

  auto* mr = local_tensor->get_mr_by_rail(req->get_rail_id());

  req->record_rdma_regmr();

  read_wr.sg_list = &read_sge;
  read_sge.addr = local_tensor->get_uint64_addr();
  read_sge.length = local_tensor->get_bytes();
  req->enqueue_completion_bytes(read_sge.length);
  read_sge.lkey = mr->lkey;

  auto res = misc::wrap_ibv_post_send(selected_qp, &read_wr, &read_bad_wr);
  if (res) {
    req->set_result(absl::ErrnoToStatus(errno, absl::StrCat("rdma post_send failed: return=", res)));
    return misc::FAILED;
  }

  // Push to per-QP queue for lock-free completion matching
  per_qp_inflight_queues_[qp_index].push(req);
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::read_multi(read_request_t request, const std::vector<RdmaReadSeg>& segs) {
  if (segs.empty()) {
    return misc::FAILED;
  }

  LOG(INFO) << "[rdma_transport] read_multi request=" << request->get_key() << " segs=" << segs.size()
            << " ready=" << ready_.load() << " rail_id=" << request->get_rail_id()
            << " dev=" << (dev_ ? dev_->get_name() : "<null>");

  // Ensure QP is ready
  if (!ready_.load()) {
    LOG(ERROR) << "[rdma_transport] read_multi invoked before handshake ready: request=" << request->get_key();
    return misc::INVALID_ARGUMENT;
  }

  // Prepare batch of WRs
  std::vector<ibv_send_wr> wrs(segs.size());
  std::vector<ibv_sge> sges(segs.size());
  struct ibv_send_wr* bad_wr = nullptr;

  auto* mr = request->get_local_tensor()->get_mr_by_rail(request->get_rail_id());
  request->record_rdma_queue_done();

  // Select QP using round robin first to encode in wr_id
  int qp_index = next_qp_index_.fetch_add(1) % qp_count_;
  struct ibv_qp* selected_qp = qps_[qp_index];

  // Encode both transport_index and qp_index in wr_id
  uint64_t encoded_wr_id = (static_cast<uint64_t>(transport_index_) << 16) | static_cast<uint64_t>(qp_index);

  for (size_t i = 0; i < segs.size(); ++i) {
    auto& wr = wrs[i];
    auto& sge = sges[i];
    misc::CLEAR(wr);
    misc::CLEAR(sge);
    wr.wr_id = encoded_wr_id;
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

  auto res = misc::wrap_ibv_post_send(selected_qp, wrs.data(), &bad_wr);
  if (res) {
    size_t posted_count = (bad_wr != nullptr) ? static_cast<size_t>(bad_wr - wrs.data()) : 0;
    if (posted_count > 0) {
      for (size_t i = 0; i < posted_count; ++i) {
        request->enqueue_completion_bytes(segs[i].length);
      }
      for (size_t i = 0; i < posted_count; ++i) {
        per_qp_inflight_queues_[qp_index].push(request);
      }
      LOG(WARNING) << "[rdma_transport] partial post: " << posted_count << "/" << segs.size()
                   << " WRs posted before failure";
    }
    request->set_result(absl::ErrnoToStatus(errno, absl::StrCat("rdma post_send (multi) failed: return=", res)));
    LOG(WARNING) << "[rdma_transport] ibv_post_send failed for request=" << request->get_key() << " res=" << res
                 << " errno=" << errno;
    return misc::FAILED;
  }

  // Track N inflight completions for this request in per-QP queue
  for (size_t i = 0; i < segs.size(); ++i) {
    request->enqueue_completion_bytes(segs[i].length);
    per_qp_inflight_queues_[qp_index].push(request);
  }

  LOG(INFO) << "[rdma_transport] Posted " << segs.size() << " RDMA READ WRs on QP " << qp_index
            << " for request=" << request->get_key();
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_process_wc(struct ibv_wc* wc) {
  if (wc->opcode == IBV_WC_RDMA_READ) {
    // Extract qp_index from lower 16 bits of wr_id
    int qp_index = static_cast<int>(wc->wr_id & 0xFFFF);
    if (qp_index < 0 || qp_index >= kMaxQpCount || qp_index >= qp_count_) {
      LOG(FATAL) << "invalid qp_index=" << qp_index << " from wr_id=" << wc->wr_id;
    }
    auto req = per_qp_inflight_queues_[qp_index].pop(true);
    if (req == nullptr) {
      LOG(FATAL) << "abnormal queue state for qp_index=" << qp_index;
    }
    req->record_read_done();
    if (wc->status == IBV_WC_SUCCESS) {
      if (req->mark_completion_and_is_done()) {
        req->set_result(absl::OkStatus());
        LOG(INFO) << "[rdma_transport] RDMA READ completion success for request=" << req->get_key();
      }
    } else {
      LOG(WARNING) << "process err wc: status=" << wc->status;
      req->set_result(absl::InternalError(absl::StrCat("failed to process work completion: wc_status=", wc->status)));
      LOG(WARNING) << "[rdma_transport] Work completion error for request=" << req->get_key()
                   << " status=" << wc->status;
    }
  }
  return misc::SUCCESS;
}

misc::result_t RdmaTransport::do_modify_qp_lag_port(struct ibv_qp* qp, int lag = 1) {
  int ret = misc::wrap_mlx5dv_modify_qp_lag_port(qp, lag);
  if (ret != 1) {
    LOG(WARNING) << "Failed to mlx5dv_modify_qp_lag_port qp [" << qp->qp_num << "] to port: " << lag
                 << ", qp type: " << qp->qp_type;
    return misc::FAILED;
  } else {
    uint8_t set_port = 0xff, act_port = 0xff;
    misc::wrap_mlx5dv_query_qp_lag_port(qp, &set_port, &act_port);
    LOG(INFO) << "QP LAG Port: QP: " << qp->qp_num << ", Modify Port: " << lag
              << ", Set to Port: " << static_cast<int>(set_port) << ", Active Port: " << static_cast<int>(act_port);
    return misc::SUCCESS;
  }
}

misc::result_t RdmaTransport::do_post_recv() {
  return misc::SUCCESS;
}

} // namespace tensorcast::communicator::transport
