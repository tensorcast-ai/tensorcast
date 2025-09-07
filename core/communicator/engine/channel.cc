// Copyright (c) 2025, TensorCast Team.

#include <sstream>
#include <utility>

#include "core/communicator/engine/channel.h"
#include "core/communicator/misc/utils.h"

namespace tensorcast::communicator::engine {

Channel::Channel(communicator::transport::tcp_transport_t control, int type)
    : type_(type), control_(std::move(control)), mtcp_(nullptr), expired_time_(0) {}

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
  return rdma_.get(key.str());
}

void Channel::set_channel_type(int type) {
  type_ = type;
}

void Channel::set_transport(
    const std::string& local_dev_name,
    const std::string& remote_dev_name,
    transport::rdma_transport_t t) {
  misc::ASSERT(type_ == base::CHANNEL_RDMA, "cannot set rdma transport for tcp channel");
  std::stringstream key;
  key << local_dev_name << ":" << remote_dev_name;
  rdma_.put(key.str(), std::move(t));
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
  rdma_.del(key.str());
}

misc::result_t Channel::close() {
  if (control_ != nullptr) {
    control_->close();
    control_.reset();
  }
  if (mtcp_ != nullptr) {
    mtcp_.reset();
  }
  rdma_.clear();
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

} // namespace tensorcast::communicator::engine
