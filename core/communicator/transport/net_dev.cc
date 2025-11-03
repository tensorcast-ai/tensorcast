// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/transport/net_dev.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "core/communicator/misc/ibv_wrap.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator::transport {

NetDev::NetDev(struct ibv_context* context, int dev_id, struct ibv_device* dev, int port_id, struct ibv_port_attr port)
    : dev_name_(dev->name),
      context_(context),
      dev_id_(dev_id),
      guid_(),
      port_(port_id),
      link_(port.link_layer),
      pci_path_(nullptr),
      gid_tbl_len_(port.gid_tbl_len),
      numa_id_(0),
      pd_(nullptr),
      cq_(nullptr),
      rail_id_(0),
      stop_(false) {
  CHECK_WARN(misc::wrap_ibv_alloc_pd(&pd_, context_), "failed to allocate PD");
  CHECK_WARN(misc::wrap_ibv_create_cq(&cq_, context_, 128, nullptr, nullptr, 0), "failed to allocate CQ");
  CHECK_WARN(read_pci_path(), "failed to get pci path");
  read_numa_id();
  get_best_gid_index();
  read_rail_id();
  register_thread_ = std::thread(&NetDev::register_loop, this);
}

NetDev::~NetDev() {
  stop_.store(true);
  register_queue_.stop();
  if (register_thread_.joinable()) {
    register_thread_.join();
  }

  if (cq_) {
    misc::wrap_ibv_destroy_cq(cq_);
    cq_ = nullptr;
  }
  if (pd_) {
    misc::wrap_ibv_dealloc_pd(pd_);
    pd_ = nullptr;
  }
  if (pci_path_ != nullptr) {
    free(pci_path_);
    pci_path_ = nullptr;
  }

  if (context_ != nullptr) {
    misc::wrap_ibv_close_device(context_);
    context_ = nullptr;
  }
}

int NetDev::get_port() const {
  return port_;
}

int NetDev::get_link() const {
  return link_;
}

uint8_t NetDev::get_numa_id() const {
  return numa_id_;
}

int16_t NetDev::get_rail_id() const {
  return rail_id_;
}

ibv_pd* NetDev::get_pd() const {
  return pd_;
}

ibv_cq* NetDev::get_cq() const {
  return cq_;
}

std::string NetDev::get_name() {
  return dev_name_;
}

char* NetDev::get_pci_path() {
  return pci_path_;
}

misc::result_t NetDev::read_pci_path() {
  char device_path[1024];
  snprintf(device_path, sizeof(device_path), "/sys/class/infiniband/%s/device", dev_name_.c_str());
  char* p = realpath(device_path, nullptr);
  pci_path_ = p;
  if (p == nullptr) {
    LOG(WARNING) << "could not find real path of " << dev_name_ << " (" << device_path << ")";
    return misc::FAILED;
  }
  return misc::SUCCESS;
}

misc::result_t NetDev::read_numa_id() {
  char numa_id_path[1024];
  snprintf(numa_id_path, sizeof(numa_id_path), "/sys/class/infiniband/%s/device/numa_node", dev_name_.c_str());
  std::ifstream numa_id_file(numa_id_path);
  std::string line;
  while (std::getline(numa_id_file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    numa_id_ = uint8_t(std::stoi(line));
    break;
  }
  return misc::SUCCESS;
}

void NetDev::read_rail_id() {
  const char* lldp_file_name = std::getenv("TENSORCAST_LLDP_FILE_NAME");
  if (lldp_file_name != nullptr) {
    std::ifstream lldp_file(lldp_file_name);
    std::string line;
    // brainpf_bond0=0000:19:00.0,mlx5_bond100,1
    while (std::getline(lldp_file, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      // use = to split, get the left part of eth_name and the right part of value_part
      size_t eq_pos = line.find('=');
      if (eq_pos == std::string::npos) {
        continue;
      }

      std::string eth_name = line.substr(0, eq_pos), value_part = line.substr(eq_pos + 1);

      // use , to split the right part
      std::istringstream iss(value_part);
      std::string pci_path, mlx5_name, rail_id_str;

      if (!std::getline(iss, pci_path, ',') || !std::getline(iss, mlx5_name, ',') ||
          !std::getline(iss, rail_id_str, ','))
        continue;

      // check if mlx5_name matches the current device name
      if (mlx5_name == dev_name_) {
        try {
          int rail_id = std::stoi(rail_id_str);
          rail_id_ = uint16_t(rail_id);
        } catch (const std::exception& e) {
          LOG(WARNING) << "Failed to parse rail_id: " << rail_id_str;
        }
        break; // exit the loop
      }
    }
  } else {
    // base: by mlx5_name
    if (std::regex_match(this->dev_name_, std::regex("mlx5_(\\d+)"))) {
      this->rail_id_ = uint16_t(std::stoi(this->dev_name_.substr(this->dev_name_.find_last_of('_') + 1)));
    }
  }
}

int NetDev::get_best_gid_index() {
  if (gid_idx_ > 0) {
    return gid_idx_;
  }
  int gid_idx = -1;
  ibv_gid gid = {};
  for (int i = 1; i < gid_tbl_len_; i++) {
    std::memset(&gid, 0, sizeof(gid));
    misc::wrap_ibv_query_gid(context_, port_, i, &gid);
    if ((gid.raw[10] == 0xFF) && (gid.raw[11] == 0xFF)) {
      gid_idx = i;
      gid_idx_ = i;
      memcpy(&gid_, &gid, sizeof(gid));
    }
  }
  return gid_idx;
}

misc::result_t NetDev::get_best_gid(ibv_gid* gid, int* gid_idx) {
  ABSL_CHECK(gid != nullptr) << "gid cannot be nullptr";
  ABSL_CHECK(gid_idx != nullptr) << "gid index cannot be nullptr";
  if (gid_idx_ > 0) {
    *gid_idx = gid_idx_;
    memcpy(gid, &gid_, sizeof(gid_));
    return misc::SUCCESS;
  }

  ibv_gid tmp_gid = {};

  for (int i = 1; i < gid_tbl_len_; i++) {
    std::memset(&tmp_gid, 0, sizeof(tmp_gid));
    misc::wrap_ibv_query_gid(context_, port_, i, &tmp_gid);
    if ((tmp_gid.raw[10] == 0xFF) && (tmp_gid.raw[11] == 0xFF)) {
      *gid_idx = i;
      gid_idx_ = i;
      memcpy(&gid_, &tmp_gid, sizeof(tmp_gid));
      memcpy(gid, &tmp_gid, sizeof(tmp_gid));
    }
  }
  if (*gid_idx == -1) {
    return misc::FAILED;
  }
  return misc::SUCCESS;
}

misc::result_t NetDev::reg_async(const tensor_t& tensor) {
  register_queue_.push(tensor);
  return misc::SUCCESS;
}

misc::result_t NetDev::reg_mr(struct ibv_mr** ret, void* addr, size_t length, int access) const {
  return misc::wrap_ibv_reg_mr(ret, pd_, addr, length, access);
}

misc::result_t NetDev::create_qp(struct ibv_qp** ret, struct ibv_qp_init_attr* qp_init_attr) const {
  return misc::wrap_ibv_create_qp(ret, pd_, qp_init_attr);
}

void NetDev::register_loop() {
  uint64_t counter = 0;
  while (!stop_.load()) {
    auto t = register_queue_.pop(true);
    if (stop_.load()) {
      break;
    }
    if (t == nullptr) {
      counter++;
      if (counter % 3000 == 0) {
        std::this_thread::yield();
      }
      continue;
    }

    auto start = misc::get_us();
    t->register_mr(std::shared_ptr<NetDev>(this));
    LOG(INFO) << "register done: dev=" << dev_name_ << ", cost=" << misc::get_us() - start;
  }
}

} // namespace tensorcast::communicator::transport
