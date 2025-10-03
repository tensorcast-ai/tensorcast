// Copyright (c) 2025, TensorCast Team.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/communicator/engine/staging_flow_controller.h"
#include "core/communicator/transport/mtcp_transport.h"
#include "core/communicator/transport/rdma_transport.h"
#include "core/communicator/transport/tcp_transport.h"

namespace tensorcast::communicator::engine {

struct RdmaReadSession;

class Channel {
 public:
  enum class HandshakeState : std::int8_t {
    kIdle,
    kConnectRequested,
    kReady,
    kFailed,
  };

  struct PendingRdmaRead {
    transport::read_request_t request;
    std::vector<transport::RdmaTransport::RdmaReadSeg> segments;
    absl::Time enqueued_at = absl::InfinitePast();
    uint64_t generation = 0;
  };

  struct RdmaEndpoint {
    mutable absl::Mutex mu;
    transport::rdma_transport_t transport ABSL_GUARDED_BY(mu);
    HandshakeState state ABSL_GUARDED_BY(mu) = HandshakeState::kIdle;
    uint64_t generation ABSL_GUARDED_BY(mu) = 0;
    std::deque<PendingRdmaRead> pending_reads ABSL_GUARDED_BY(mu);
    absl::Time next_retry_at ABSL_GUARDED_BY(mu) = absl::InfinitePast();
    int failure_count ABSL_GUARDED_BY(mu) = 0;
    bool retry_scheduled ABSL_GUARDED_BY(mu) = false;
  };

  explicit Channel(
      communicator::transport::tcp_transport_t control,
      int type,
      int buffers_per_flow,
      uint32_t max_window_segments);
  ~Channel();

  transport::tcp_transport_t get_control();
  transport::mtcp_transport_t get_mtcp();
  transport::rdma_transport_t get_rdma(const std::string& local_dev_name, const std::string& remote_dev_name);
  std::shared_ptr<RdmaEndpoint> get_rdma_endpoint(
      const std::string& local_dev_name,
      const std::string& remote_dev_name);
  std::shared_ptr<RdmaEndpoint> ensure_rdma_endpoint(
      const std::string& local_dev_name,
      const std::string& remote_dev_name);

  bool has_gpu_slot() const {
    return gpu_slot_handle_ != nullptr;
  }

  void set_gpu_slot_handle(std::shared_ptr<void> handle);

  void set_channel_type(int type);

  void set_transport(
      const std::string& local_dev_name,
      const std::string& remote_dev_name,
      transport::rdma_transport_t t,
      HandshakeState initial_state = HandshakeState::kReady);
  void set_transport(communicator::transport::mtcp_transport_t t);
  void mtcp_request_started();
  void mtcp_request_finished();
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
  std::shared_ptr<RdmaEndpoint> find_rdma_endpoint_locked(const std::string& key) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(rdma_mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<RdmaEndpoint>> rdma_ ABSL_GUARDED_BY(rdma_mu_);
  mutable absl::Mutex rdma_mu_;
  transport::mtcp_transport_t mtcp_;
  std::atomic<int> mtcp_active_requests_{0};
  uint64_t expired_time_;
  std::shared_ptr<FlowState> flow_state_;
  std::shared_ptr<void> gpu_slot_handle_;
};

using channel_t = std::shared_ptr<Channel>;

} // namespace tensorcast::communicator::engine
