// Copyright (c) 2025-2026, TensorCast Team.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/strings/str_split.h"
#include "core/communicator/transport/net_dev.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/communicator/transport/rdma_transport.h"
#include "core/cuda/cuda_api.h"

namespace tensorcast::communicator::transport {

using base::COMMUNICATE_ENGINE_DEV_CPU;
using base::COMMUNICATE_ENGINE_DEV_GPU;

namespace {

struct PortScanCandidate {
  int dev_id = -1;
  ibv_device* device = nullptr;
  std::string device_name;
  int port = -1;
  ibv_port_attr port_attr{};
  int best_gid_index = -1;
};

bool is_supported_link_layer(uint8_t link_layer) {
  return link_layer == IBV_LINK_LAYER_INFINIBAND || link_layer == IBV_LINK_LAYER_ETHERNET;
}

bool is_port_usable(const ibv_port_attr& port_attr) {
  if (port_attr.state != IBV_PORT_ACTIVE) {
    return false;
  }
  return is_supported_link_layer(port_attr.link_layer);
}

int query_best_gid_index(struct ibv_context* context, int port, int gid_tbl_len) {
  int best_gid_index = -1;
  for (int gid_index = 1; gid_index < gid_tbl_len; gid_index++) {
    ibv_gid gid{};
    std::memset(&gid, 0, sizeof(gid));
    if (misc::SUCCESS != misc::wrap_ibv_query_gid(context, port, gid_index, &gid)) {
      continue;
    }
    if (gid.raw[10] == 0xFF && gid.raw[11] == 0xFF) {
      best_gid_index = gid_index;
    }
  }
  return best_gid_index;
}

bool should_replace_primary_for_rail(const net_dev_t& candidate, const net_dev_t& existing) {
  if (existing == nullptr) {
    return true;
  }
  if (candidate->get_best_gid_index() != existing->get_best_gid_index()) {
    return candidate->get_best_gid_index() > existing->get_best_gid_index();
  }
  if (candidate->get_port() != existing->get_port()) {
    return candidate->get_port() < existing->get_port();
  }
  if (candidate->get_name() != existing->get_name()) {
    return candidate->get_name() < existing->get_name();
  }
  return candidate->get_dev_id() < existing->get_dev_id();
}

} // namespace

RdmaContext::RdmaContext() {
  ABSL_CHECK(ibv_init() == misc::SUCCESS) << "failed to init";
  for (int i = 0; i < 16; i++) {
    dev_vector_[i] = nullptr;
  }
}

RdmaContext::~RdmaContext() {
  devs_.clear();
}

misc::result_t RdmaContext::ibv_init() {
  devs_.clear();
  rail_devs_.clear();
  io_threads_.clear();

  const char* mlx_dev_name_env = std::getenv("TENSORCAST_IB_HCA");
  std::unordered_set<std::string> mlx_dev_names = {};
  if (mlx_dev_name_env != nullptr) {
    // remove "=" in mlx_dev_names
    std::string mlx_dev_name = std::regex_replace(mlx_dev_name_env, std::regex("="), "");
    LOG(INFO) << "using IB HCA Config: " << mlx_dev_name;
    std::vector<std::string> mlx_dev_names_vec = absl::StrSplit(mlx_dev_name, ',');
    for (const auto& dev_name : mlx_dev_names_vec) {
      mlx_dev_names.insert(dev_name);
    }
  }

  if (misc::wrap_ibv_symbols() != misc::SUCCESS) {
    LOG(WARNING) << "failed to init ibv symbols";
    return misc::SYS_ERROR;
  }

  if (misc::wrap_mlx5dv_symbols() != 1) {
    LOG(WARNING) << "failed to init mlx5dv symbols";
    // return misc::SYS_ERROR;
  }

  misc::wrap_ibv_fork_init();

  struct ibv_device** devices = nullptr;
  int num_dev = 0;
  if (misc::SUCCESS != misc::wrap_ibv_get_device_list(&devices, &num_dev)) {
    return misc::SYS_ERROR;
  }
  if (devices == nullptr || num_dev <= 0) {
    LOG(INFO) << "No RDMA devices detected during ibverbs initialization";
    return misc::SUCCESS;
  }

  std::vector<PortScanCandidate> scan_candidates;
  scan_candidates.reserve(static_cast<size_t>(num_dev) * 2);
  for (int d = 0; d < num_dev; d++) {
    std::string device_name = devices[d]->name;
    if (!mlx_dev_names.empty() && !mlx_dev_names.contains(device_name)) {
      continue;
    }

    struct ibv_context* context = nullptr;
    if (misc::SUCCESS != misc::wrap_ibv_open_device(&context, devices[d]) || context == nullptr) {
      LOG(WARNING) << "unable to open IB device" << devices[d]->name;
      continue;
    }

    struct ibv_device_attr attr = {};
    if (misc::SUCCESS != misc::wrap_ibv_query_device(context, &attr)) {
      LOG(WARNING) << "unable to query device " << devices[d]->name;
      if (misc::SUCCESS != misc::wrap_ibv_close_device(context)) {
        return misc::INTERNAL_ERROR;
      }
      continue;
    }

    for (int port = 1; std::cmp_less_equal(port, attr.phys_port_cnt); port++) {
      struct ibv_port_attr port_attr{};
      std::memset(&port_attr, 0, sizeof(struct ibv_port_attr));
      if (misc::SUCCESS != misc::wrap_ibv_query_port(context, port, &port_attr)) {
        LOG(WARNING) << "unable to query port attr " << devices[d]->name;
        continue;
      }
      if (!is_port_usable(port_attr)) {
        continue;
      }

      const int best_gid_index = query_best_gid_index(context, port, port_attr.gid_tbl_len);
      if (best_gid_index <= 0) {
        continue;
      }

      scan_candidates.push_back(
          PortScanCandidate{
              .dev_id = d,
              .device = devices[d],
              .device_name = device_name,
              .port = port,
              .port_attr = port_attr,
              .best_gid_index = best_gid_index,
          });
    }

    if (misc::SUCCESS != misc::wrap_ibv_close_device(context)) {
      return misc::INTERNAL_ERROR;
    }
  }

  std::sort(
      scan_candidates.begin(), scan_candidates.end(), [](const PortScanCandidate& lhs, const PortScanCandidate& rhs) {
        if (lhs.device_name != rhs.device_name) {
          return lhs.device_name < rhs.device_name;
        }
        if (lhs.port != rhs.port) {
          return lhs.port < rhs.port;
        }
        if (lhs.best_gid_index != rhs.best_gid_index) {
          return lhs.best_gid_index > rhs.best_gid_index;
        }
        return lhs.dev_id < rhs.dev_id;
      });

  for (const PortScanCandidate& candidate : scan_candidates) {
    struct ibv_context* context = nullptr;
    if (misc::SUCCESS != misc::wrap_ibv_open_device(&context, candidate.device) || context == nullptr) {
      LOG(WARNING) << "unable to reopen IB device for candidate " << candidate.device_name
                   << " port=" << candidate.port;
      continue;
    }

    struct ibv_port_attr port_attr{};
    std::memset(&port_attr, 0, sizeof(port_attr));
    if (misc::SUCCESS != misc::wrap_ibv_query_port(context, candidate.port, &port_attr)) {
      LOG(WARNING) << "unable to re-query port attr " << candidate.device_name << " port=" << candidate.port;
      if (misc::SUCCESS != misc::wrap_ibv_close_device(context)) {
        return misc::INTERNAL_ERROR;
      }
      continue;
    }
    if (!is_port_usable(port_attr)) {
      LOG(INFO) << "skip unstable RDMA port after recheck: dev=" << candidate.device_name << " port=" << candidate.port
                << " state=" << port_attr.state << " link_layer=" << static_cast<int>(port_attr.link_layer);
      if (misc::SUCCESS != misc::wrap_ibv_close_device(context)) {
        return misc::INTERNAL_ERROR;
      }
      continue;
    }

    auto dev = std::make_shared<NetDev>(context, candidate.dev_id, candidate.device, candidate.port, port_attr);
    if (dev->get_best_gid_index() <= 0) {
      LOG(INFO) << "skip RDMA candidate due to missing usable GID after recheck: dev=" << candidate.device_name
                << " port=" << candidate.port;
      if (misc::SUCCESS != misc::wrap_ibv_close_device(context)) {
        return misc::INTERNAL_ERROR;
      }
      continue;
    }
    LOG(INFO) << "RDMA candidate accepted: dev=" << dev->get_name() << " port=" << dev->get_port()
              << " rail_id=" << dev->get_rail_id() << " gid_idx=" << dev->get_best_gid_index();
    devs_.push_back(dev);
  }

  if (devices != nullptr && (misc::SUCCESS != misc::wrap_ibv_free_device_list(devices))) {
    return misc::INTERNAL_ERROR;
  }

  std::sort(devs_.begin(), devs_.end(), [](const net_dev_t& lhs, const net_dev_t& rhs) {
    if (lhs->get_rail_id() != rhs->get_rail_id()) {
      return lhs->get_rail_id() < rhs->get_rail_id();
    }
    if (lhs->get_name() != rhs->get_name()) {
      return lhs->get_name() < rhs->get_name();
    }
    if (lhs->get_port() != rhs->get_port()) {
      return lhs->get_port() < rhs->get_port();
    }
    return lhs->get_dev_id() < rhs->get_dev_id();
  });

  for (const auto& dev : devs_) {
    auto it = rail_devs_.find(dev->get_rail_id());
    if (it == rail_devs_.end() || should_replace_primary_for_rail(dev, it->second)) {
      rail_devs_[dev->get_rail_id()] = dev;
    }
  }

  for (const auto& dev : devs_) {
    const auto rail_it = rail_devs_.find(dev->get_rail_id());
    if (rail_it == rail_devs_.end()) {
      continue;
    }
    const bool is_primary = rail_it->second == dev;
    LOG(INFO) << "RDMA rail device registered: dev=" << dev->get_name() << " port=" << dev->get_port()
              << " rail_id=" << dev->get_rail_id() << " role=" << (is_primary ? "primary" : "backup");
  }

  for (auto dev : devs_) {
    io_threads_.push_back(std::make_shared<RdmaThread>(dev));
  }

  return misc::SUCCESS;
}

net_dev_t RdmaContext::get_dev(const std::string& name) {
  for (auto& dev : devs_) {
    if (dev->get_name() == name) {
      return dev;
    }
  }
  return nullptr;
}

net_dev_t RdmaContext::get_dev_by_rail(int rail_id) {
  if (rail_devs_.find(rail_id) == rail_devs_.end()) {
    return nullptr;
  }
  return rail_devs_[rail_id];
}

net_dev_t RdmaContext::get_best_dev(int dev_type, int dev_id, int rail_id, const std::string& key) {
  if (devs_.empty()) {
    LOG(WARNING) << "No RDMA net devices available when selecting best device for dev_type=" << dev_type
                 << " dev_id=" << dev_id << " rail_id=" << rail_id;
    return nullptr;
  }
  if (dev_type == COMMUNICATE_ENGINE_DEV_CPU) {
    if (rail_id == -1) {
      if (devs_.empty()) {
        VLOG(1) << "No active RDMA device available for CPU key=" << key;
        return nullptr;
      }
      size_t index = std::hash<std::string>{}(key) % devs_.size();
      LOG(INFO) << "get_best_dev: dev_type=" << dev_type << " dev_id=" << dev_id << " rail input=" << rail_id
                << " key=" << key << " index=" << index << " dev=" << devs_[index]->get_name()
                << " dev rail_id=" << devs_[index]->get_rail_id();
      return devs_[index];
    }
    auto rail_it = rail_devs_.find(rail_id);
    if (rail_it == rail_devs_.end()) {
      VLOG(1) << "No RDMA device mapped for rail_id=" << rail_id << " key=" << key;
      return nullptr;
    }
    return rail_it->second;
  }
  return get_best_dev(dev_id);
}

net_dev_t RdmaContext::get_best_dev(int gpu_id) {
  if (gpu_id < 0 || gpu_id >= static_cast<int>(std::size(dev_vector_))) {
    LOG(WARNING) << "gpu_id is out of supported range for cached affinity map: gpu_id=" << gpu_id
                 << " max_supported=" << static_cast<int>(std::size(dev_vector_)) - 1;
    return nullptr;
  }
  if (dev_vector_[gpu_id] != nullptr) {
    return dev_vector_[gpu_id];
  }
  if (devs_.empty()) {
    VLOG(1) << "No active RDMA devices discovered";
    return nullptr;
  }

  char pci_path[512] = {0};
  {
    cudaDeviceProp device_prop;
    auto status = cuda::get_device_properties(gpu_id, &device_prop);
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
  std::unique_ptr<char, decltype(&std::free)> path(realpath(pci_path, nullptr), &std::free);
  if (path == nullptr) {
    return nullptr;
  }
  int max_prefix_len = 0;
  int max_prefix_idx = -1;
  int max_gid_idx = -1;
  std::vector<int> max_prefix_candidates;

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

    while (path.get()[prefix_len] != '\0' && net_pci_path[prefix_len] != '\0' &&
           net_pci_path[prefix_len] == path.get()[prefix_len] && prefix_len < 512) {
      prefix_len++;
    }

    if (prefix_len > max_prefix_len) {
      max_prefix_idx = static_cast<int>(i);
      max_prefix_len = prefix_len;
      max_prefix_candidates.clear();
      max_prefix_candidates.push_back(static_cast<int>(i));
    } else if (prefix_len == max_prefix_len) {
      max_prefix_idx = static_cast<int>(i);
      max_prefix_candidates.push_back(static_cast<int>(i));
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

    while (path.get()[prefix_len] != '\0' && net_pci_path[prefix_len] != '\0' &&
           net_pci_path[prefix_len] == path.get()[prefix_len] && prefix_len < 512) {
      prefix_len++;
    }

    if (prefix_len == max_prefix_len) {
      dev_idx_list.push_back(static_cast<int>(i));
    }
  }

  if (dev_idx_list.empty()) {
    if (max_prefix_candidates.empty()) {
      LOG(WARNING) << "No RDMA candidate remains for gpu_id=" << gpu_id
                   << " after affinity scan; max_prefix_candidates is empty";
      return nullptr;
    }
    const int fallback_idx = max_prefix_candidates[static_cast<size_t>(gpu_id) % max_prefix_candidates.size()];
    LOG(WARNING) << "No RDMA candidate satisfies max GID + max PCI-prefix jointly for gpu_id=" << gpu_id
                 << "; fallback to max-prefix candidate dev=" << devs_[fallback_idx]->get_name();
    dev_vector_[gpu_id] = devs_[fallback_idx];
    return dev_vector_[gpu_id];
  }

  max_prefix_idx = dev_idx_list[static_cast<size_t>(gpu_id) % dev_idx_list.size()];
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

misc::result_t RdmaThread::register_transport(RdmaTransport* t) {
  add_poll_transport(t);
  add_send_transport(t);
  add_recv_transport(t);
  return misc::SUCCESS;
}

misc::result_t RdmaThread::unregister_transport(RdmaTransport* t) {
  add_poll_transport(t);
  add_send_transport(t);
  add_recv_transport(t);
  return misc::SUCCESS;
}

void RdmaThread::add_send_transport(RdmaTransport* t) {
  send_transports_[t->transport_index_] = t;
}

void RdmaThread::add_poll_transport(RdmaTransport* t) {
  for (int i = 0; i < 1024; i++) {
    if (poll_transports_[i] == nullptr) {
      t->transport_index_ = i;
      poll_transports_[i] = t;
      return;
    }
  }
  ABSL_CHECK(false) << "failed to allocate a valid poll status";
}

void RdmaThread::del_send_transport(RdmaTransport* t) {
  send_transports_[t->transport_index_] = nullptr;
}

void RdmaThread::del_poll_transport(RdmaTransport* t) {
  poll_transports_[t->transport_index_] = nullptr;
}

void RdmaThread::add_recv_transport(RdmaTransport* t) {
  recv_transports_[t->transport_index_] = t;
}

void RdmaThread::del_recv_transport(RdmaTransport* t) {
  recv_transports_[t->transport_index_] = nullptr;
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
    misc::wrap_ibv_poll_cq(net_dev_->get_cq(), 4, wcs, &wr_done);
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

      // Extract transport_index from upper 16 bits of wr_id
      // wr_id encoding: (transport_index << 16) | qp_index
      // poll_transports_ has 1024 slots and transport_index is assigned 0-1023, so a 10-bit mask (0x3FF) is sufficient.
      int transport_index = static_cast<int>((wc->wr_id >> 16) & 0x3FF);
      if (transport_index < 1024 && poll_transports_[transport_index] != nullptr) {
        poll_transports_[transport_index]->do_process_wc(wc);
      } else {
        LOG(WARNING) << "failed to process wc: wr_id=" << wc->wr_id << " transport_index=" << transport_index;
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

} // namespace tensorcast::communicator::transport
