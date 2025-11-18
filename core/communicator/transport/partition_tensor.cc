
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
      // registered_(false),
      ready_(false),
      // mr_(nullptr),
      // dev_(std::move(dev)),
      mem_type_(mem_type) {
  if (dev != nullptr) {
    add_dev(dev);
  }
}

PartitionTensor::~PartitionTensor() {

  for (auto dev : devs_) {
    if (registered_[dev->get_name()]->load()) {
      if(mrs_[dev->get_name()] != nullptr) {
        CHECK_WARN(misc::wrap_ibv_dereg_mr(mrs_[dev->get_name()]), "failed to dereg mr");
      }
      registered_[dev->get_name()]->store(false);
      ready_.store(false);
    }
  }
}

void PartitionTensor::add_dev_and_register(const net_dev_t& dev) {
  add_dev(dev);
  dev->reg_async(std::shared_ptr<PartitionTensor>(this));
}

void PartitionTensor::add_dev(const net_dev_t& dev) {
  devs_.push_back(std::move(dev));
  registered_[dev->get_name()] = std::make_shared<std::atomic_bool>(false);
  mrs_[dev->get_name()] = nullptr;
}

void PartitionTensor::add_dev_list(const std::vector<net_dev_t>& devs) {
  for (const auto& dev : devs) {
    add_dev(dev);

  }
}

void PartitionTensor::set_read_ready() {
  ready_.store(true);
}

void PartitionTensor::set_read_unready() {
  ready_.store(false);
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

net_dev_t PartitionTensor::get_dev() {
  if (devs_.empty()) {
    return nullptr;
  }
  return devs_.front();
}

struct ibv_mr* PartitionTensor::get_mr(const net_dev_t& dev) {
  while (!registered_[dev->get_name()]->load()) {
    std::this_thread::yield();
  }
  return mrs_[dev->get_name()];
}

struct ibv_mr* PartitionTensor::get_mr_by_rail(int16_t rail_id) {
  for (const auto& dev : devs_) {
    if (dev->get_rail_id() == rail_id) {
      return get_mr(dev);
    }
  }
  LOG(WARNING) << "No MR found for rail_id: " << rail_id << "dev";
  return nullptr;
}

net_dev_t PartitionTensor::get_dev_by_rail(int rail_id) {
  for (const auto& dev : devs_) {
    if (dev->get_rail_id() == rail_id) {
      return dev;
    }
  }
  return nullptr;
}

std::vector<net_dev_t> PartitionTensor::get_devs() {
  return devs_;
}

uint64_t PartitionTensor::get_bytes() const {
  return bytes_;
}

uint64_t PartitionTensor::get_uint64_addr() const {
  return addr_;
}

uint64_t PartitionTensor::get_regmr_cost(const net_dev_t& dev) {
  return regmr_costs_[dev->get_name()];
}

uint64_t PartitionTensor::get_regmr_cost() const {
  return regmr_costs_.empty() ? 0 : regmr_costs_.begin()->second;
}

int PartitionTensor::get_mem_type() const {
  return mem_type_;
}

bool PartitionTensor::is_registered(const net_dev_t& dev) {
  return registered_[dev->get_name()]->load();
}

void PartitionTensor::register_mr(const NetDev* dev) {
  misc::Timer timer(true);
  int flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
  auto addr = get_addr<void>();
  auto nb_bytes = get_bytes();
  misc::result_t res = dev->reg_mr(&mrs_[dev->get_name()], addr, nb_bytes, flags);
  
  if (res != misc::SUCCESS) {
    LOG(WARNING) << __FILE__ << ":" << __LINE__ << " " << res << " failed to register mr";
    mrs_[dev->get_name()] = nullptr;
  } else {
    regmr_costs_[dev->get_name()] = timer.record();
  }
  registered_[dev->get_name()]->store(true);
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
