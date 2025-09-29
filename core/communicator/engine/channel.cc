// Copyright (c) 2025, TensorCast Team.

#include <sstream>
#include <utility>

#include "absl/log/log.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/misc/utils.h"

namespace tensorcast::communicator::engine {

Channel::Channel(
    communicator::transport::tcp_transport_t control,
    int type,
    int buffers_per_flow,
    uint32_t max_window_segments)
    : type_(type),
      control_(std::move(control)),
      mtcp_(nullptr),
      expired_time_(0),
      flow_state_(std::make_shared<FlowState>(buffers_per_flow, max_window_segments)) {}

Channel::~Channel() {
  close();
}

communicator::transport::tcp_transport_t Channel::get_control() {
  return control_;
}

communicator::transport::mtcp_transport_t Channel::get_mtcp() {
  return mtcp_;
}

communicator::transport::rdma_transport_t Channel::get_rdma(
    const std::string& local_dev_name,
    const std::string& remote_dev_name) {
  std::stringstream key;
  key << local_dev_name << ":" << remote_dev_name;
  auto endpoint = get_rdma_endpoint(local_dev_name, remote_dev_name);
  if (endpoint == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(&endpoint->mu);
  return endpoint->transport;
}

void Channel::set_channel_type(int type) {
  type_ = type;
}

void Channel::set_transport(
    const std::string& local_dev_name,
    const std::string& remote_dev_name,
    transport::rdma_transport_t t,
    HandshakeState initial_state) {
  misc::ASSERT(type_ == base::CHANNEL_RDMA, "cannot set rdma transport for tcp channel");
  std::stringstream key;
  key << local_dev_name << ":" << remote_dev_name;
  auto endpoint = ensure_rdma_endpoint(local_dev_name, remote_dev_name);
  absl::MutexLock endpoint_lock(&endpoint->mu);
  endpoint->transport = std::move(t);
  endpoint->state = initial_state;
  endpoint->generation += 1;
  endpoint->failure_count = 0;
  endpoint->next_retry_at = absl::InfinitePast();
  endpoint->retry_scheduled = false;
  if (initial_state == HandshakeState::kReady) {
    for (auto& pending : endpoint->pending_reads) {
      pending.generation = endpoint->generation;
    }
  }
}

void Channel::set_transport(communicator::transport::mtcp_transport_t t) {
  misc::ASSERT(type_ == base::CHANNEL_MTCP, "cannot set rdma transport for tcp channel");
  misc::ASSERT(mtcp_ == nullptr, "cannot set rdma transport for tcp channel");
  mtcp_ = std::move(t);
}

void Channel::del_transport(const std::string& local_dev_name, const std::string& remote_dev_name) {
  misc::ASSERT(type_ == base::CHANNEL_RDMA, "cannot set rdma transport for tcp channel");
  std::stringstream key;
  key << local_dev_name << ":" << remote_dev_name;
  absl::MutexLock lock(&rdma_mu_);
  rdma_.erase(key.str());
}

void Channel::mtcp_request_started() {
  mtcp_active_requests_.fetch_add(1, std::memory_order_relaxed);
}

void Channel::mtcp_request_finished() {
  const int previous = mtcp_active_requests_.fetch_sub(1, std::memory_order_acq_rel);
  if (previous <= 0) {
    LOG(WARNING) << "[Channel] mtcp_request_finished underflow for MTCP channel";
    mtcp_active_requests_.store(0, std::memory_order_relaxed);
    return;
  }
  if (previous == 1) {
    VLOG(1) << "[Channel] Last MTCP request completed; releasing receive buffers";
    if (mtcp_ != nullptr) {
      mtcp_->release_receive_resources();
    }
  }
}

misc::result_t Channel::close() {
  if (control_ != nullptr) {
    control_->close();
    control_.reset();
  }
  mtcp_active_requests_.store(0, std::memory_order_relaxed);
  if (mtcp_ != nullptr) {
    mtcp_->release_receive_resources();
    mtcp_.reset();
  }
  {
    absl::MutexLock lock(&rdma_mu_);
    rdma_.clear();
  }
  return misc::SUCCESS;
}

void Channel::record_expire(uint64_t now) {
  expired_time_ = misc::get_us() / 1000000 + now;
}

bool Channel::is_expired(uint64_t now) const {
  if (expired_time_ == 0) {
    return false;
  }
  return now > expired_time_;
}

std::shared_ptr<Channel::RdmaEndpoint> Channel::find_rdma_endpoint_locked(const std::string& key) const {
  auto it = rdma_.find(key);
  if (it == rdma_.end()) {
    return nullptr;
  }
  return it->second;
}

std::shared_ptr<Channel::RdmaEndpoint> Channel::get_rdma_endpoint(
    const std::string& local_dev_name,
    const std::string& remote_dev_name) {
  std::stringstream key;
  key << local_dev_name << ":" << remote_dev_name;
  absl::MutexLock lock(&rdma_mu_);
  return find_rdma_endpoint_locked(key.str());
}

std::shared_ptr<Channel::RdmaEndpoint> Channel::ensure_rdma_endpoint(
    const std::string& local_dev_name,
    const std::string& remote_dev_name) {
  std::stringstream key;
  key << local_dev_name << ":" << remote_dev_name;
  absl::MutexLock lock(&rdma_mu_);
  auto endpoint = find_rdma_endpoint_locked(key.str());
  if (endpoint == nullptr) {
    endpoint = std::make_shared<RdmaEndpoint>();
    rdma_.insert({key.str(), endpoint});
  }
  return endpoint;
}

} // namespace tensorcast::communicator::engine
