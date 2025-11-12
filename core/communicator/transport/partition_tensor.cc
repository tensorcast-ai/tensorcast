
// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/transport/partition_tensor.h"
#include "core/communicator/misc/metric.h"

#include <thread>
#include <utility>

namespace tensorcast::communicator::transport {

PartitionTensor::PartitionTensor(std::string tensor_key, uint64_t addr, uint64_t bytes, int mem_type, net_dev_t dev)
    : tensor_key_(std::move(tensor_key)),
      addr_(addr),
      bytes_(bytes),
      registered_(false),
      ready_(false),
      mr_(nullptr),
      dev_(std::move(dev)),
      mem_type_(mem_type) {}

PartitionTensor::~PartitionTensor() {
  if (registered_.load()) {
    if (mr_ != nullptr) {
      CHECK_WARN(misc::wrap_ibv_dereg_mr(mr_), "failed to dereg mr");
    }
    registered_.store(false);
    ready_.store(false);
  }
}

void PartitionTensor::set_read_ready() {
  ready_.store(true);
}

void PartitionTensor::wait_read_ready() {
  while (!ready_.load()) {
    std::this_thread::yield();
  }
}

void PartitionTensor::wait_mr_ready() {
  while (!registered_.load()) {
    std::this_thread::yield();
  }
}

std::string PartitionTensor::get_key() {
  return tensor_key_;
}

struct ibv_mr* PartitionTensor::get_mr() {
  while (!registered_.load()) {
    std::this_thread::yield();
  }
  return mr_;
}

net_dev_t PartitionTensor::get_dev() {
  return dev_;
}

uint64_t PartitionTensor::get_bytes() const {
  return bytes_;
}

uint64_t PartitionTensor::get_uint64_addr() const {
  return addr_;
}

uint64_t PartitionTensor::get_regmr_cost() const {
  return regmr_cost_;
}

int PartitionTensor::get_mem_type() const {
  return mem_type_;
}

void PartitionTensor::register_mr() {
  misc::Timer timer(true);
  int flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;

  auto addr = get_addr<void>();
  auto nb_bytes = get_bytes();
  misc::result_t res = dev_->reg_mr(&mr_, addr, nb_bytes, flags);
  if (res != misc::SUCCESS) {
    LOG(WARNING) << __FILE__ << ":" << __LINE__ << " " << res << " failed to register mr";
    mr_ = nullptr;
  } else {
    regmr_cost_ = timer.record();
  }
  registered_.store(true);
}

RemotePartitionTensor::RemotePartitionTensor(
    std::string tensor_key,
    std::string net_dev,
    uint64_t addr,
    uint64_t bytes,
    uint32_t rkey)
    : tensor_key_(std::move(tensor_key)), net_dev_(std::move(net_dev)), addr_(addr), bytes_(bytes), rkey_(rkey) {}

std::string RemotePartitionTensor::get_key() const {
  return tensor_key_;
}

uint64_t RemotePartitionTensor::get_bytes() const {
  return bytes_;
}

uint32_t RemotePartitionTensor::get_rkey() const {
  return rkey_;
}

std::string RemotePartitionTensor::get_net_dev() const {
  return net_dev_;
}

uint64_t RemotePartitionTensor::get_uint64_addr() const {
  return addr_;
}

} // namespace tensorcast::communicator::transport
