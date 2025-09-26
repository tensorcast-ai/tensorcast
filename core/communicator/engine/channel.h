// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_CHANNEL_H_
#define CORE_COMMUNICATOR_ENGINE_CHANNEL_H_

#include <deque>
#include <memory>
#include <string>

#include "core/communicator/engine/staging_flow_controller.h"
#include "core/communicator/transport/mtcp_transport.h"
#include "core/communicator/transport/rdma_transport.h"
#include "core/communicator/transport/tcp_transport.h"

namespace tensorcast::communicator::engine {

struct RdmaReadSession;

class Channel {
 public:
  explicit Channel(
      communicator::transport::tcp_transport_t control,
      int type,
      int buffers_per_flow,
      uint32_t max_window_segments);
  ~Channel();

  transport::tcp_transport_t get_control();
  transport::mtcp_transport_t get_mtcp();
  transport::rdma_transport_t get_rdma(const std::string& local_dev_name, const std::string& remote_dev_name);

  void set_channel_type(int type);

  void set_transport(
      const std::string& local_dev_name,
      const std::string& remote_dev_name,
      transport::rdma_transport_t t);
  void set_transport(communicator::transport::mtcp_transport_t t);
  void del_transport(const std::string& local_dev_name, const std::string& remote_dev_name);

  misc::result_t close();

  void record_expire(uint64_t now);

  bool is_expired(uint64_t now) const;

  struct FlowState {
    FlowState(int credit, uint32_t max_window_segments)
        : ledger(credit),
          max_window_segments(max_window_segments == 0 ? static_cast<uint32_t>(credit) : max_window_segments) {}

    FlowCreditLedger ledger;
    StageLeaseRegistry registry;
    const uint32_t max_window_segments;
    std::deque<std::shared_ptr<RdmaReadSession>> rdma_pending_reads;
    bool rdma_refill_in_progress = false;
  };

  std::shared_ptr<FlowState> flow_state() const {
    return flow_state_;
  }

 private:
  int type_;
  transport::tcp_transport_t control_;
  misc::Map<std::string, transport::rdma_transport_t> rdma_;
  transport::mtcp_transport_t mtcp_;
  uint64_t expired_time_;
  std::shared_ptr<FlowState> flow_state_;
};

using channel_t = std::shared_ptr<Channel>;

} // namespace tensorcast::communicator::engine

#endif // COMMUNICATOR_ENGINE_CHANNEL_H_
