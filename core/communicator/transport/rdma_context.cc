// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "core/common/cuda_api.h"
#include "core/communicator/misc/envs.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/net_dev.h"
#include "core/communicator/transport/rdma_context.h"

namespace stepcast::communicator {

RdmaContext::RdmaContext() {
  ASSERT(ibv_init() == SUCCESS, "failed to init");
  for (int i = 0; i < 16; i++) {
    dev_vector_[i] = nullptr;
  }
}

RdmaContext::~RdmaContext() {
  devs_.clear();
}

result_t RdmaContext::ibv_init() {
  if (wrap_ibv_symbols() != SUCCESS) {
    LOG(WARNING) << "failed to init ibv symbols";
    return SYS_ERROR;
  }

  wrap_ibv_fork_init();

  struct ibv_device** devices;
  int num_dev = 0;
  if (SUCCESS != wrap_ibv_get_device_list(&devices, &num_dev)) {
    return SYS_ERROR;
  }

  for (int d = 0; d < num_dev; d++) {
    struct ibv_context* context = nullptr;
    if (SUCCESS != wrap_ibv_open_device(&context, devices[d]) || context == nullptr) {
      LOG(WARNING) << "unable to open IB device" << devices[d]->name;
      continue;
    }

    struct ibv_device_attr attr = {};
    if (SUCCESS != wrap_ibv_query_device(context, &attr)) {
      LOG(WARNING) << "unable to query device " << devices[d]->name;
      if (SUCCESS != wrap_ibv_close_device(context)) {
        return INTERNAL_ERROR;
      }
      continue;
    }

    net_dev_t dev = nullptr;
    for (int port = 1; port <= attr.phys_port_cnt; port++) {
      struct ibv_port_attr port_attr{};
      bzero(&port_attr, sizeof(struct ibv_port_attr));
      if (SUCCESS != wrap_ibv_query_port(context, port, &port_attr)) {
        LOG(WARNING) << "unable to query port attr " << devices[d]->name;
        continue;
      }
      if (port_attr.state != IBV_PORT_ACTIVE) {
        continue;
      }
      if (port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND && port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
        continue;
      }

      dev = std::make_shared<NetDev>(context, d, devices[d], port, port_attr);

      if (dev->get_best_gid_index() > 0) {
        devs_.push_back(dev);
      }
    }
  }

  if ((SUCCESS != wrap_ibv_free_device_list(devices))) {
    return INTERNAL_ERROR;
  }

  for (auto dev : devs_) {
    io_threads_.push_back(std::make_shared<RdmaThread>(dev));
  }

  return SUCCESS;
}

net_dev_t RdmaContext::get_dev(const std::string& name) {
  for (auto& dev : devs_) {
    if (dev->get_name() == name) {
      return dev;
    }
  }
  return nullptr;
}

net_dev_t RdmaContext::get_best_dev(int gpu_id) {
  if (dev_vector_[gpu_id] != nullptr) {
    return dev_vector_[gpu_id];
  }

  char pci_path[512] = {0};
  {
    cudaDeviceProp device_prop;
    auto status = stepcast::cuda::get_device_properties(gpu_id, &device_prop);
    if (!status.ok()) {
      LOG(WARNING) << "Unable to get CUDA device properties for device " << gpu_id << ": " << status;
      return nullptr;
    }

    snprintf(
        pci_path,
        sizeof(pci_path),
        "/sys/class/pci_bus/0000:%02x/device/0000:%02x:%02x.0/device",
        device_prop.pciBusID,
        device_prop.pciBusID,
        device_prop.pciDeviceID);
  }
  char* path = realpath(pci_path, nullptr);

  if (path == nullptr) {
    return nullptr;
  }
  int max_prefix_len = 0;
  int max_prefix_idx = -1;
  int max_gid_idx = -1;

  for (uint32_t i = 0; i < devs_.size(); i++) {
    int prefix_len = 0;
    auto& dev = devs_[i];
    if (dev->get_best_gid_index() < 0) {
      continue;
    }

    auto net_pci_path = devs_[i]->get_pci_path();
    if (net_pci_path == nullptr) {
      continue;
    }

    while (path[prefix_len] != '\0' && net_pci_path[prefix_len] != '\0' &&
           net_pci_path[prefix_len] == path[prefix_len] && prefix_len < 512) {
      prefix_len++;
    }

    if (prefix_len >= max_prefix_len) {
      max_prefix_idx = static_cast<int>(i);
      max_prefix_len = prefix_len;
    }

    if (dev->get_best_gid_index() > max_gid_idx) {
      max_gid_idx = dev->get_best_gid_index();
    }
  }

  if (max_prefix_idx < 0) {
    return nullptr;
  }

  std::vector<int> dev_idx_list;

  for (uint32_t i = 0; i < devs_.size(); i++) {
    int prefix_len = 0;
    auto& dev = devs_[i];
    if (dev->get_best_gid_index() < max_gid_idx) {
      continue;
    }

    auto net_pci_path = devs_[i]->get_pci_path();
    if (net_pci_path == nullptr) {
      continue;
    }

    while (path[prefix_len] != '\0' && net_pci_path[prefix_len] != '\0' &&
           net_pci_path[prefix_len] == path[prefix_len] && prefix_len < 512) {
      prefix_len++;
    }

    if (prefix_len == max_prefix_len) {
      dev_idx_list.push_back(static_cast<int>(i));
    }
  }

  max_prefix_idx = dev_idx_list[gpu_id % dev_idx_list.size()];
  dev_vector_[gpu_id] = devs_[max_prefix_idx];
  return dev_vector_[gpu_id];
}

std::string RdmaContext::get_best_dev_name(int gpu_id) {
  auto dev = get_best_dev(gpu_id);
  if (dev != nullptr) {
    return dev->get_name();
  }
  return "";
}

rdma_transport_t RdmaContext::create_transport(const std::string& dev_name) {
  net_dev_t dev = nullptr;
  rdma_thread_t th = nullptr;
  for (uint32_t i = 0; i < devs_.size(); i++) {
    if (devs_[i]->get_name() == dev_name) {
      dev = devs_[i];
      th = io_threads_[i];
    }
  }
  if (dev == nullptr || th == nullptr) {
    LOG(WARNING) << "failed to create transport due to cannot find dev_name: " << dev_name;
    return nullptr;
  }

  auto transport = std::make_shared<RdmaTransport>(this, dev, th);
  return transport;
}

RdmaThread::RdmaThread(net_dev_t dev) : net_dev_(std::move(dev)) {
  LOG(INFO) << "thread init with dev: " << net_dev_->get_name();
  for (int i = 0; i < 1024; i++) {
    poll_transports_[i] = nullptr;
    recv_transports_[i] = nullptr;
    send_transports_[i] = nullptr;
  }
  send_thread_ = std::thread([this]() { this->send_loop(); });
  poll_thread_ = std::thread([this]() { this->poll_loop(); });
  recv_thread_ = std::thread([this]() { this->recv_loop(); });
}

RdmaThread::~RdmaThread() {
  stop_.store(true);
  if (send_thread_.joinable()) {
    send_thread_.join();
  }
  if (poll_thread_.joinable()) {
    poll_thread_.join();
  }
  if (recv_thread_.joinable()) {
    recv_thread_.join();
  }
}

void RdmaThread::stop() {
  stop_.store(true);
  send_cv_.notify_all();
  poll_cv_.notify_all();
  recv_cv_.notify_all();
}

void RdmaThread::notify_send() {
  send_cv_.notify_all();
}

void RdmaThread::notify_poll() {
  poll_cv_.notify_all();
}

void RdmaThread::notify_recv() {
  recv_cv_.notify_all();
}

result_t RdmaThread::register_transport(RdmaTransport* t) {
  add_poll_transport(t);
  add_send_transport(t);
  add_recv_transport(t);
  return SUCCESS;
}

result_t RdmaThread::unregister_transport(RdmaTransport* t) {
  add_poll_transport(t);
  add_send_transport(t);
  add_recv_transport(t);
  return SUCCESS;
}

void RdmaThread::add_send_transport(RdmaTransport* t) {
  send_transports_[t->transport_idx_] = t;
}

void RdmaThread::add_poll_transport(RdmaTransport* t) {
  for (int i = 0; i < 1024; i++) {
    if (poll_transports_[i] == nullptr) {
      t->transport_idx_ = i;
      poll_transports_[i] = t;
      return;
    }
  }
  ASSERT(false, "failed to allocate a valid poll status");
}

void RdmaThread::del_send_transport(RdmaTransport* t) {
  send_transports_[t->transport_idx_] = nullptr;
}

void RdmaThread::del_poll_transport(RdmaTransport* t) {
  poll_transports_[t->transport_idx_] = nullptr;
}

void RdmaThread::add_recv_transport(RdmaTransport* t) {
  recv_transports_[t->transport_idx_] = t;
}

void RdmaThread::del_recv_transport(RdmaTransport* t) {
  recv_transports_[t->transport_idx_] = nullptr;
}

void RdmaThread::send_loop() {
  std::unique_lock<std::mutex> lock(send_mu_);
  while (!stop_.load()) {
    send_cv_.wait_for(lock, std::chrono::microseconds(1), [this] {
      if (stop_.load()) {
        return true;
      }
      for (auto& t : send_transports_) {
        if (t == nullptr) {
          continue;
        }
        if (!t->ready()) {
          continue;
        }
        if (t->ready_to_send()) {
          return true;
        }
      }
      return false;
    });

    for (auto& t : send_transports_) {
      if (t != nullptr && t->ready_to_send()) {
        t->do_post_send();
      }
    }
  }
}

void RdmaThread::poll_loop() {
  std::unique_lock<std::mutex> lock(poll_mu_);
  int wr_done = 0;
  struct ibv_wc wcs[4];

  while (!stop_.load()) {
    poll_cv_.wait_for(lock, std::chrono::microseconds(1), [this] {
      if (stop_.load()) {
        return true;
      }
      for (auto& t : poll_transports_) {
        if (t == nullptr) {
          continue;
        }
        if (!t->ready()) {
          continue;
        }
      }
      return false;
    });

    if (net_dev_ == nullptr) {
      LOG(ERROR) << "failed to start poll loop due to nil dev";
      break;
    }

    if (net_dev_->get_cq() == nullptr) {
      LOG(ERROR) << "failed to start poll loop due to nil cq";
      break;
    }

    lock.unlock();
    wrap_ibv_poll_cq(net_dev_->get_cq(), 4, wcs, &wr_done);
    lock.lock();
    if (wr_done == 0) {
      continue;
    }
    for (int w = 0; w < wr_done; w++) {
      struct ibv_wc* wc = wcs + w;
      if (wc->status != IBV_WC_SUCCESS) {
        LOG(WARNING) << "failed to poll cq: wr_id=" << wc->wr_id << " status=" << wc->status;
        continue;
      }

      if (wc->wr_id < 1024 && poll_transports_[wc->wr_id] != nullptr) {
        poll_transports_[wc->wr_id]->do_process_wc(wc);
      } else {
        LOG(WARNING) << "failed to process wc: id=" << wc->wr_id;
      }
    }
  }
}

void RdmaThread::recv_loop() {
  std::unique_lock<std::mutex> lock(recv_mu_);
  while (!stop_.load()) {
    recv_cv_.wait_for(lock, std::chrono::microseconds(10), [this] {
      if (stop_.load()) {
        return true;
      }
      for (auto& t : recv_transports_) {
        if (t == nullptr) {
          continue;
        }
        if (!t->ready()) {
          continue;
        }
        if (t->ready_to_recv()) {
          return true;
        }
      }
      return false;
    });

    if (stop_.load()) {
      break;
    }

    for (auto& t : recv_transports_) {
      if (t != nullptr && t->ready_to_recv()) {
        t->do_post_recv();
      }
    }
  }
}

} // namespace stepcast::communicator
