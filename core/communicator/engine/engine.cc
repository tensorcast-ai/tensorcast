
// Copyright (c) 2025, TensorCast Team.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/ascii.h"

#include "core/common/system_capabilities.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/dram_stager.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/gpu_net_stager.h"
#include "core/communicator/engine/protocol.h"
#include "core/communicator/engine/uma_residency_provider.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/rdma_context.h"

namespace tensorcast::communicator {

// Engine is fully typed-config driven; no environment-variable reads here.

// No legacy constructors; typed CommunicatorConfig is required.

CommunicateEngine::CommunicateEngine(const CommunicatorConfig& cfg, uint32_t channel_expire_sec)
    : stop_(false),
      inited_(false),
      server_context_(new TcpContext()),
      client_context_(new TcpContext()),
      enable_rdma_(cfg.enable_rdma),
      mtcp_conn_count_(cfg.transport.tcp_conn_count),
      ack_ttl_ms_(cfg.rdma.ack_ttl_ms),
      config_(cfg),
      channel_expire_(channel_expire_sec) {
  common::SystemCapabilities::instance().record_rdma_available(enable_rdma_);
  request_thread_ = std::thread([this]() { this->do_read_request_loop(); });
  gc_thread_ = std::thread([this]() { this->do_channel_gc_loop(); });

  // Default UMA-backed residency provider
  residency_provider_ = std::make_shared<UmaResidencyProvider>();

  // Staging resources sized from config
  const size_t gpu_chunk_size =
      (config_.stager.stage_chunk_mb_gpu > 0 ? config_.stager.stage_chunk_mb_gpu : 16) * 1024ull * 1024ull;
  const size_t cpu_chunk_size =
      (config_.stager.stage_chunk_mb_cpu > 0 ? config_.stager.stage_chunk_mb_cpu : 4) * 1024ull * 1024ull;
  const size_t num_buffers = (config_.stager.buffers_per_flow > 0 ? config_.stager.buffers_per_flow : 4);
  const size_t recv_num_buffers = num_buffers; // unify receiver buffering under stager policy
  const size_t total_pool_size = config_.pool.pool_size_bytes > 0
                                     ? config_.pool.pool_size_bytes
                                     : gpu_chunk_size * (num_buffers + recv_num_buffers);

  // GPU staging pool and stager
  gpu_memory_pool_ = std::make_shared<store::PinnedMemoryPool>(total_pool_size, gpu_chunk_size);
  gpu_tcp_stager_ = std::make_shared<GpuTcpStager>(gpu_chunk_size, num_buffers, gpu_memory_pool_);
  gpu_memory_stager_ = std::make_shared<GpuNetStager>(gsl::not_null<std::shared_ptr<GpuTcpStager>>{gpu_tcp_stager_});

  // CPU staging pool honors CPU chunk size; size conservatively for one flow
  if (cpu_chunk_size != gpu_chunk_size) {
    const size_t cpu_pool_size = cpu_chunk_size * num_buffers; // minimal to honor buffers_per_flow
    cpu_memory_pool_ = std::make_shared<store::PinnedMemoryPool>(cpu_pool_size, cpu_chunk_size);
  }
  auto dram_pool = cpu_memory_pool_ ? cpu_memory_pool_ : gpu_memory_pool_;
  memory_stager_ = std::make_shared<DRAMStager>(gsl::not_null<std::shared_ptr<store::PinnedMemoryPool>>{dram_pool}, /*num_buffers_hint=*/num_buffers);
  if (auto ds = std::dynamic_pointer_cast<DRAMStager>(memory_stager_)) {
    ds->set_lease_provider(DRAMStager::make_noop_lease_provider());
  }

  if (enable_rdma_) {
    rdma_context_ = std::make_shared<RdmaContext>();
    mr_cache_ = std::make_unique<MrCache>();

    if (config_.simple_numa.enable) {
      for (const auto& node : config_.simple_numa.nodes) {
        auto pool = std::make_shared<store::PinnedMemoryPool>(total_pool_size, gpu_chunk_size);
        numa_pools_.push_back(pool);
        auto cpu_stager = std::make_shared<DRAMStager>(gsl::not_null<std::shared_ptr<store::PinnedMemoryPool>>{pool}, /*num_buffers_hint=*/num_buffers);
        if (auto ds = std::dynamic_pointer_cast<DRAMStager>(cpu_stager)) {
          ds->set_lease_provider(DRAMStager::make_noop_lease_provider());
        }
        auto gpu_stager = std::make_shared<GpuTcpStager>(gpu_chunk_size, num_buffers, pool);
        auto gpu_mem_stager = std::make_shared<GpuNetStager>(gsl::not_null<std::shared_ptr<GpuTcpStager>>{gpu_stager});
        // Map GPU ids
        for (int gid : node.gpus) {
          gpu_stagers_[gid] = gpu_stager;
          gpu_mem_stagers_[gid] = gpu_mem_stager;
        }
        // Map NIC names
        for (const auto& nic : node.nics) {
          nic_cpu_stagers_[nic] = cpu_stager;
        }
      }
    }

    // Preregister MRs for all pools
    int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
    std::vector<std::shared_ptr<store::PinnedMemoryPool>> pools;
    pools.push_back(gpu_memory_pool_);
    if (cpu_memory_pool_ && cpu_memory_pool_.get() != gpu_memory_pool_.get()) {
      pools.push_back(cpu_memory_pool_);
    }
    for (auto& p : numa_pools_) pools.push_back(p);
    for (const auto& dev : rdma_context_->list_devs()) {
      for (auto& pool : pools) {
        auto buffers = pool->list_buffers();
        for (auto ptr : buffers) {
          auto* mr = mr_cache_->get_or_register(dev->get_pd(), ptr.get(), pool->chunk_size(), access);
          if (mr == nullptr) {
            LOG(WARNING) << "Failed to preregister MR for buffer " << static_cast<void*>(ptr.get()) << " on PD";
          }
        }
      }
    }
  }
}

CommunicateEngine::~CommunicateEngine() {
  store_.clear();
  stop_.store(true);
  request_queue_.stop();
  if (request_thread_.joinable()) {
    request_thread_.join();
  }
  if (gc_thread_.joinable()) {
    gc_thread_.join();
  }

  for (auto& channel : channels_.pairs()) {
    channel.second->close();
  }

  pending_requests_.clear();
}

void CommunicateEngine::set_dram_lease_provider(std::shared_ptr<DRAMStager::LeaseProvider> provider) {
  if (!memory_stager_) return;
  if (auto ds = std::dynamic_pointer_cast<DRAMStager>(memory_stager_)) {
    ds->set_lease_provider(std::move(provider));
  }
  // Also propagate to NUMA CPU stagers if present
  for (auto& kv : nic_cpu_stagers_) {
    if (auto ds2 = std::dynamic_pointer_cast<DRAMStager>(kv.second)) {
      ds2->set_lease_provider(provider);
    }
  }
}

absl::Status CommunicateEngine::init(const std::string& ip, uint16_t port, int conn_count) {
  inited_.store(true);
  if (server_context_->open(ip, port, [this](tcp_transport_t t) { return this->on_new_client(t); }) != SUCCESS) {
    return absl::InternalError("failed to open server " + ip + ":" + std::to_string(port));
  }
  mtcp_conn_count_ = conn_count;
  return absl::OkStatus();
}

future_read_result_t CommunicateEngine::read_tensor(
    const std::string& key,
    uint64_t addr,
    uint64_t bytes,
    int dev_type,
    int dev_id,
    const std::string& dst_ip,
    uint16_t dst_port,
    uint64_t remote_offset) {
  if (!inited_.load()) {
    LOG(ERROR) << "failed to read a tensor with a un-inited engine";
    return ReadRequest::get_read_result_future("failed to read tensor through un-initiated engine");
  }
  net_dev_t net_dev = nullptr;
  if (enable_rdma_) {
    net_dev = get_net_dev(dev_type, dev_id);
    if (net_dev == nullptr) {
      return ReadRequest::get_read_result_future("failed to get net dev for the rdma connection");
    }
  } else if (COMMUNICATE_ENGINE_DEV_GPU == dev_type && !gpu_tcp_stager_) {
    return ReadRequest::get_read_result_future("failed to read GPU tensor with tcp: GPU stager not initialized");
  }

  LOG(INFO) << "read tensor:"
            << " dst=" << dst_ip << ":" << dst_port << ", key=" << key << " ,offset=" << remote_offset
            << ", net_dev=" << (net_dev == nullptr ? "none" : net_dev->get_name());

  auto local_tensor = store_.get_tensor(key);
  if (local_tensor == nullptr) {
    local_tensor = std::make_shared<PartitionTensor>(key, addr, bytes, dev_type, net_dev);
    if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
      local_tensor->set_device_id(dev_id);
    }
    if (enable_rdma_) {
      net_dev->reg_async(local_tensor);
    }
  }

  auto req = std::make_shared<ReadRequest>(key, dst_ip, dst_port, local_tensor, remote_offset);
  LOG(INFO) << "[read_tensor] Creating request: key=" << key << " dst=" << dst_ip << ":" << dst_port
            << " req_key=" << req->get_key();
  request_queue_.push(req);
  LOG(INFO) << "[read_tensor] Request pushed to queue successfully for key=" << key;
  return req->get_future();
}

absl::Status CommunicateEngine::register_tensor(
    const std::string& tensor_key,
    uint64_t addr,
    uint64_t bytes,
    int dev_type,
    int dev_id,
    bool async) {
  RegisterTensorOptions opts;
  opts.register_mr = true;
  opts.needs_staging = (!enable_rdma_ && dev_type == COMMUNICATE_ENGINE_DEV_GPU);
  opts.async = async;
  return register_tensor_ex(tensor_key, addr, bytes, dev_type, dev_id, opts);
}

absl::Status CommunicateEngine::register_tensor_ex(
    const std::string& tensor_key,
    uint64_t addr,
    uint64_t bytes,
    int dev_type,
    int dev_id,
    const RegisterTensorOptions& opts) {
  // Check for zero-size tensor
  if (bytes == 0) {
    return absl::InvalidArgumentError("Cannot register zero-size tensor");
  }

  net_dev_t net_dev = nullptr;
  if (enable_rdma_) {
    net_dev = get_net_dev(dev_type, dev_id);
    if (net_dev == nullptr) {
      return absl::InternalError("failed to get net dev");
    }
  }

  // Note: In TCP mode, GPU tensors are now supported with staging
  if (COMMUNICATE_ENGINE_DEV_GPU == dev_type) {
    if (dev_id < 0 || dev_id >= 16) {
      return absl::InternalError("failed to register tensor on a invalid gpu");
    }
  }

  VLOG(1) << "register tensor:"
          << " key=" << tensor_key << ", addr=" << addr << ", bytes=" << bytes << ", gpu=" << dev_id
          << ", net_dev=" << (net_dev == nullptr ? "none" : net_dev->get_name());

  auto tensor = std::make_shared<PartitionTensor>(tensor_key, addr, bytes, dev_type, net_dev);
  tensor->set_read_ready();

  // Set device ID for GPU tensors
  if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
    tensor->set_device_id(dev_id);
  }

  // Mark tensors that need staging if requested by policy
  if (opts.needs_staging) {
    tensor->set_needs_staging(true);
  }

  if (enable_rdma_ && opts.register_mr) {
    net_dev->reg_async(tensor);
    if (!opts.async) {
      if (tensor->get_mr() == nullptr) {
        return absl::InternalError("failed to register mr");
      }
    }
  }

  store_.register_tensor(tensor);
  return absl::OkStatus();
}

absl::Status CommunicateEngine::unregister_tensor(const std::string& tensor_key) {
  if (store_.get_tensor(tensor_key) == nullptr) {
    return absl::InternalError("failed to unregister a non-existed tensor");
  }
  store_.unregister_tensor(tensor_key);
  return absl::OkStatus();
}

result_t CommunicateEngine::on_new_client(const tcp_transport_t& t) {
  LOG(INFO) << "[on_new_client] New client connection from " << t->get_remote_url() << " fd=" << t->get_fd();
  auto channel = std::make_shared<Channel>(t, enable_rdma_ ? CHANNEL_RDMA : CHANNEL_MTCP);
  channels_.put(t->get_remote_url(), channel);
  t->set_recv_func([this](const tcp_transport_t& t) -> result_t {
    auto channel = this->channels_.get(t->get_remote_url());
    if (channel == nullptr) {
      LOG(WARNING) << "failed to process recv message due to nil channel: " << t->get_remote_url();
      return INTERNAL_ERROR;
    }
    ProtoHeader header = {};
    COMM_CHECK(t->recv<ProtoHeader>(&header));
    auto msg = std::make_shared<EngineMessage>(&header);
    COMM_CHECK(t->recv(msg->get_payload<uint8_t>(), msg->get_payload_size()));
    return this->on_receive_request(channel, t, msg);
  });
  t->set_close_func([this](const tcp_transport_t& t) {
    LOG(INFO) << "[on_new_client] Client connection closed: " << t->get_remote_url();
    auto channel = channels_.get(t->get_remote_url());
    if (channel) {
      channel->close();
    }
    channels_.del(t->get_remote_url());
    return SUCCESS;
  });
  return SUCCESS;
}

absl::StatusOr<channel_t> CommunicateEngine::do_create_channel(const std::string& ip, uint16_t port) {
  absl::MutexLock lock(&create_channel_mu_);

  // Fast-path: if another thread has already created the channel, reuse it
  std::stringstream url_ss;
  url_ss << ip << ":" << port;
  const std::string url_key = url_ss.str();

  LOG(INFO) << "[do_create_channel] Attempting to create channel for " << url_key;

  if (channels_.exist(url_key)) {
    LOG(INFO) << "[do_create_channel] Channel already exists for " << url_key << ", reusing";
    return channels_.get(url_key);
  }

  LOG(INFO) << "create a channel: dst=" << ip << ":" << port;
  auto t = client_context_->connect(ip, port);
  if (!t.ok()) {
    LOG(WARNING) << "failed to connect peer " << ip << ":" << port;
    return absl::InternalError(t.status().message());
  }

  auto transport = *t;
  auto channel = std::make_shared<Channel>(transport, enable_rdma_ ? CHANNEL_RDMA : CHANNEL_MTCP);

  VLOG(1) << "[CommunicateEngine] Control channel connected: local=" << server_context_->get_local_ip() << ":" << port
          << " remote=" << ip << ":" << port << " fd=" << transport->get_fd();

  transport->set_recv_func([this](const tcp_transport_t& t) {
    ProtoHeader header = {};
    auto channel = this->channels_.get(t->get_remote_url());
    COMM_CHECK(t->recv<ProtoHeader>(&header));
    auto msg = std::make_shared<EngineMessage>(&header);
    COMM_CHECK(t->recv(msg->get_payload<uint8_t>(), msg->get_payload_size()));
    return this->on_receive_response(channel, t, msg);
  });
  transport->set_close_func([this, transport_ptr = transport.get()](const tcp_transport_t& t) {
    const std::string url_key = t->get_remote_url();
    LOG(INFO) << "[do_create_channel] TCP connection closed for " << url_key << ", transport ptr: " << t.get() << " vs "
              << transport_ptr;
    auto channel = channels_.get(url_key);
    if (channel && channel->get_control().get() == t.get()) {
      // Only remove the channel if this is the actual control connection
      LOG(INFO) << "[do_create_channel] This is the control connection, removing channel";
      channel->close();
      channels_.del(url_key);
    } else {
      LOG(INFO) << "[do_create_channel] This is not the control connection, keeping channel";
    }
    return SUCCESS;
  });
  if (channel_expire_ > 0) {
    channel->record_expire(channel_expire_);
  }
  // Only insert if still absent to avoid clobbering an existing active channel
  if (!channels_.exist(url_key)) {
    channels_.put(url_key, channel);
    LOG(INFO) << "[do_create_channel] Channel created and stored for " << url_key;
  } else {
    // Another thread beat us – use that channel, close the one we just created
    LOG(INFO) << "[do_create_channel] Another thread already created channel for " << url_key
              << ", closing duplicate transport (fd=" << transport->get_fd() << ")";
    channel_t existing = channels_.get(url_key);
    // Close just the transport, not the channel
    transport->close();
    return existing;
  }

  VLOG(1) << "[CommunicateEngine] Channel stored: " << transport->get_remote_url();
  return channel;
}

void CommunicateEngine::do_read_request_loop() {
  while (!stop_.load()) {
    auto req = request_queue_.pop(true);
    if (stop_.load()) {
      break;
    }
    if (req == nullptr) {
      continue;
    }

    auto channel = channels_.get(req->get_dst_url());
    if (channel == nullptr) {
      VLOG(1) << "[do_read_request_loop] No existing channel for " << req->get_dst_url() << ", creating new channel";
      auto status = do_create_channel(req->dst_ip_, req->dst_port_);
      if (!status.ok()) {
        LOG(WARNING) << "failed to create channel " << req->dst_ip_ << ":" << req->dst_port_;
        req->set_result(absl::InternalError(status.status().message()));
        continue;
      }
      channel = *status;
    } else {
      VLOG(1) << "[do_read_request_loop] Using existing channel for " << req->get_dst_url();
    }

    auto transport = channel->get_control();
    if (transport == nullptr) {
      req->set_result(absl::InternalError("failed to get transport control"));
      LOG(WARNING) << "failed to get control transport " << req->dst_ip_ << ":" << req->dst_port_;
      continue;
    }

    auto msg = EngineMessage::make_message<ProtoReadRequest>(ENGINE_OP_READ_REQUEST);
    auto* request = msg->get_payload<ProtoReadRequest>();
    STRNCPY(request->tensor_key, req->tensor_key_, kMaxTensorNameLen);

    request->transport_type = enable_rdma_ ? ENGINE_TRANSPORT_RDMA : ENGINE_TRANSPORT_MTCP;
    request->offset = req->remote_offset_;
    request->bytes = req->get_local_tensor()->get_bytes();

    VLOG(1) << "[do_read_request_loop] Sending READ_REQUEST: key=" << req->tensor_key_ << " to " << req->get_dst_url()
            << " transport_type=" << (request->transport_type == ENGINE_TRANSPORT_MTCP ? "MTCP" : "RDMA");

    // Put into pending BEFORE send to prevent response racing ahead of insertion
    const std::string req_key = req->get_key();
    pending_requests_.put(req_key, req);

    if (transport->send(msg) == SUCCESS) {
      LOG(INFO) << "[do_read_request_loop] READ_REQUEST sent successfully, pending: " << req_key;
    } else {
      // Rollback pending on failure
      pending_requests_.del(req_key);
      LOG(ERROR) << "[do_read_request_loop] Failed to send READ_REQUEST for key=" << req->tensor_key_ << " to "
                 << req->get_dst_url();
      req->set_result(absl::InternalError("failed to send request"));
    }

    if (channel_expire_ > 0) {
      channel->record_expire(channel_expire_);
    }
  }
}

result_t CommunicateEngine::on_receive_request(
    const channel_t& channel,
    const tcp_transport_t& t,
    const engine_message_t& msg) {
  static std::atomic<int> server_requests_received(0);

  switch (msg->get_op()) {
    case ENGINE_OP_RDMA_CONNECT_REQUEST: {
      auto* req = msg->get_payload<ProtoRdmaConnectRequest>();
      auto local_dev_name = std::string(req->dst_dev_name);
      auto peer_dev_name = std::string(req->src_dev_name);
      LOG(INFO) << "recv rdma connect from " << t->get_remote_url() << ": net_dev=" << local_dev_name;

      CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";
      auto transport = rdma_context_->create_transport(local_dev_name);

      if (transport->connect(&req->qp_info) == SUCCESS) {
        channel->set_transport(local_dev_name, peer_dev_name, transport);
        auto rsp = EngineMessage::make_message<ProtoRdmaConnectResponse>(ENGINE_OP_RDMA_CONNECT_RESPONSE);
        auto* payload = rsp->get_payload<ProtoRdmaConnectResponse>();
        COMM_CHECK(transport->get_local_info(&payload->qp_info));

        memcpy(payload->src_dev_name, req->src_dev_name, kMaxDevName);
        memcpy(payload->dst_dev_name, req->dst_dev_name, kMaxDevName);
        COMM_CHECK(t->send(rsp));
      } else {
        LOG(WARNING) << "failed to rdma connect from " << t->get_remote_url() << ": net_dev=" << local_dev_name;
        auto rsp = EngineMessage::make_message<ProtoRdmaConnectFailed>(ENGINE_OP_RDMA_CONNECT_RESPONSE);
        auto* payload = rsp->get_payload<ProtoRdmaConnectFailed>();
        memcpy(payload->src_dev_name, req->src_dev_name, kMaxDevName);
        memcpy(payload->dst_dev_name, req->dst_dev_name, kMaxDevName);
        COMM_CHECK(t->send(rsp));
      }
      break;
    }
    case ENGINE_OP_MTCP_CONNECT_REQUEST: {
      auto* req = msg->get_payload<ProtoMtcpConnectRequest>();
      LOG(INFO) << "recv mtcp connect from " << t->get_remote_url();

      auto transport = channel->get_mtcp();
      transport->set_conn_count(std::min(mtcp_conn_count_, req->conn_count));

      std::string ip = server_context_->get_local_ip();
      uint16_t port = 0;
      if (transport->listen(ip, &port) == SUCCESS) {
        auto rsp = EngineMessage::make_message<ProtoMtcpConnectResponse>(ENGINE_OP_MTCP_CONNECT_RESPONSE);
        auto* payload = rsp->get_payload<ProtoMtcpConnectResponse>();
        payload->conn_count = std::min(mtcp_conn_count_, req->conn_count);
        payload->port = port;
        auto ip_addr = inet_addr(ip.c_str());
        payload->ip = ntohl(ip_addr);
        COMM_CHECK(t->send(rsp));
      } else {
        LOG(WARNING) << "failed to create mtcp transport: source=" << t->get_remote_url();
        auto rsp = EngineMessage::make_message<ProtoMtcpConnectFailed>(ENGINE_OP_MTCP_CONNECT_FAILED);
        auto* payload = rsp->get_payload<ProtoMtcpConnectFailed>();
        payload->ip = inet_addr(ip.c_str());
        COMM_CHECK(t->send(rsp));
      }
      break;
    }
    case ENGINE_OP_READ_REQUEST: {
      auto* req = msg->get_payload<ProtoReadRequest>();
      auto tensor_key = std::string(req->tensor_key);

      int request_num = ++server_requests_received;
      LOG(INFO) << "[on_receive_request] Server received READ_REQUEST #" << request_num << " from "
                << t->get_remote_url() << ": key=" << tensor_key
                << ", transport=" << (req->transport_type == ENGINE_TRANSPORT_MTCP ? "mtcp" : "rdma")
                << ", offset=" << req->offset << ", size=" << req->bytes;

      LOG(INFO) << "read request from " << t->get_remote_url() << ": key=" << tensor_key
                << ", transport=" << (req->transport_type == ENGINE_TRANSPORT_MTCP ? "mtcp" : "rdma")
                << ", offset=" << req->offset << ", size=" << req->bytes;

      auto tensor = store_.get_tensor(tensor_key);
      if (tensor == nullptr) {
        auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
        auto* payload = rsp->get_payload<ProtoReadFailed>();
        memcpy(payload->tensor_key, req->tensor_key, 512);
        payload->offset = req->offset;
        payload->reason = TENSORCAST_READ_FAILED_NO_TENSOR;
        COMM_CHECK(t->send(rsp));
      } else if (req->offset + req->bytes > tensor->get_bytes()) {
        auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
        auto* payload = rsp->get_payload<ProtoReadFailed>();
        memcpy(payload->tensor_key, req->tensor_key, 512);
        payload->offset = req->offset;
        payload->reason = TENSORCAST_READ_FAILED_OVERFLOW;
        COMM_CHECK(t->send(rsp));
      } else {
        // Build response depending on transport type
        if (enable_rdma_ && req->transport_type == ENGINE_TRANSPORT_RDMA) {
          auto dev = tensor->get_dev();
          tensor->wait_read_ready();

          // Determine if staging is needed and compute segmenting
          bool use_direct_cpu_mr = false;
          bool do_stage = false;
          if (tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU) {
            if (config_.stager.stage_cpu_for_rdma) {
              do_stage = true;
            } else {
              const bool under_size = req->bytes <= config_.stager.direct_mr_max_bytes;
              const bool under_inflight = inflight_direct_mr_.load() < config_.stager.max_inflight_direct_mr;
              const bool hot = residency_provider_ && residency_provider_->is_hot(tensor_key, req->offset, req->bytes);
              use_direct_cpu_mr = under_size && under_inflight && hot;
              do_stage = !use_direct_cpu_mr;
            }
          } else {
            // GPU is always staged (TCP or RDMA)
            do_stage = true;
          }
          size_t max_seg_bytes = static_cast<size_t>(req->bytes);
          std::shared_ptr<MemoryStager> cpu_stager_sptr;
          std::shared_ptr<MemoryStager> gpu_stager_sptr;
          if (do_stage) {
            if (tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU) {
              cpu_stager_sptr = get_cpu_stager_for_nic(dev->get_name());
              if (!cpu_stager_sptr) cpu_stager_sptr = memory_stager_;
              max_seg_bytes = cpu_stager_sptr ? cpu_stager_sptr->get_chunk_size() : static_cast<size_t>(req->bytes);
            } else {
              gpu_stager_sptr = get_gpu_mem_stager_for_id(tensor->get_device_id());
              if (!gpu_stager_sptr) gpu_stager_sptr = gpu_memory_stager_;
              max_seg_bytes = gpu_stager_sptr ? gpu_stager_sptr->get_chunk_size() : static_cast<size_t>(req->bytes);
            }
            if (max_seg_bytes == 0) max_seg_bytes = static_cast<size_t>(req->bytes);
          }

          const uint64_t total = req->bytes;
          const uint64_t start_off = req->offset;
          // Avoid per-segment mixed modes: if using direct MR for CPU, use a single segment
          uint32_t num_segments = do_stage ? static_cast<uint32_t>((total + max_seg_bytes - 1) / max_seg_bytes) : 1u;

          // Allocate EX message with all segments
          auto rsp = std::make_shared<EngineMessage>(
              ENGINE_OP_READ_RESPONSE_EX,
              static_cast<uint32_t>(sizeof(ProtoReadResponseExHeader) + num_segments * sizeof(ProtoReadResponseExSeg)));
          auto* hdr = rsp->get_payload<ProtoReadResponseExHeader>();
          memcpy(hdr->tensor_key, req->tensor_key, kMaxTensorNameLen);
          hdr->transport_type = ENGINE_TRANSPORT_RDMA;
          hdr->staged = 0;
          STRNCPY(hdr->nic_name, dev->get_name(), kMaxDevName);
          hdr->num_segments = num_segments;

          for (uint32_t i = 0; i < num_segments; ++i) {
            uint64_t off = start_off + static_cast<uint64_t>(i) * max_seg_bytes;
            uint64_t remain = total - (off - start_off);
            uint32_t seg_bytes = static_cast<uint32_t>(std::min<uint64_t>(remain, max_seg_bytes));
            auto* seg_pl = reinterpret_cast<ProtoReadResponseExSeg*>(
                reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader) + i * sizeof(ProtoReadResponseExSeg));
            seg_pl->offset = off;
            seg_pl->bytes = seg_bytes;
            if (do_stage) {
              void* host_ptr = nullptr;
              struct ibv_mr* staged_mr = nullptr;
              int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
              MemoryStager* used_stager = nullptr;
              if (tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU) {
                CHECK(cpu_stager_sptr != nullptr) << "CPU staging requested but MemoryStager is null";
                used_stager = cpu_stager_sptr.get();
                auto staged = cpu_stager_sptr->stage(tensor, off, seg_bytes);
                if (!staged.ok()) {
                  auto rspf = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
                  auto* pf = rspf->get_payload<ProtoReadFailed>();
                  memcpy(pf->tensor_key, req->tensor_key, kMaxTensorNameLen);
                  pf->offset = off;
                  pf->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
                  COMM_CHECK(t->send(rspf));
                  break;
                }
                host_ptr = *staged;
              } else {
                CHECK(gpu_stager_sptr != nullptr) << "GPU staging requested but MemoryStager is null";
                used_stager = gpu_stager_sptr.get();
                auto staged = gpu_stager_sptr->stage(tensor, off, seg_bytes);
                if (!staged.ok()) {
                  auto rspf = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
                  auto* pf = rspf->get_payload<ProtoReadFailed>();
                  memcpy(pf->tensor_key, req->tensor_key, kMaxTensorNameLen);
                  pf->offset = off;
                  pf->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
                  COMM_CHECK(t->send(rspf));
                  break;
                }
                host_ptr = *staged;
              }

              if (mr_cache_) {
                staged_mr = mr_cache_->get_or_register(dev->get_pd(), gsl::not_null<void*>{host_ptr}, seg_bytes, access);
                if (staged_mr == nullptr) {
                  auto rspf = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
                  auto* pf = rspf->get_payload<ProtoReadFailed>();
                  memcpy(pf->tensor_key, req->tensor_key, kMaxTensorNameLen);
                  pf->offset = off;
                  pf->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
                  COMM_CHECK(t->send(rspf));
                  break;
                }
              } else {
                if (dev->reg_mr(&staged_mr, host_ptr, seg_bytes, access) != SUCCESS) {
                  auto rspf = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
                  auto* pf = rspf->get_payload<ProtoReadFailed>();
                  memcpy(pf->tensor_key, req->tensor_key, kMaxTensorNameLen);
                  pf->offset = off;
                  pf->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
                  COMM_CHECK(t->send(rspf));
                  break;
                }
              }
              seg_pl->addr = reinterpret_cast<uint64_t>(host_ptr);
              seg_pl->rkey = staged_mr->rkey;
              hdr->staged = 1;

              StagedRdmaSegment seg;
              seg.ptr = host_ptr;
              seg.bytes = seg_bytes;
              seg.mr = staged_mr;
              seg.kind = (tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU) ?
                             StagedRdmaSegment::Kind::CPU :
                             StagedRdmaSegment::Kind::GPU;
              seg.ts_us = get_us();
              seg.deregister_mr = (mr_cache_ == nullptr);
              seg.stager_ptr = used_stager;
              const std::string seg_req_key = get_request_key(tensor_key, off);
              {
                absl::MutexLock lk(&staged_mu_);
                staged_segments_.put(seg_req_key, std::move(seg));
              }
            } else {
              const uint64_t addr_u64 = tensor->get_uint64_addr() + off;
              struct ibv_mr* use_mr = tensor->get_mr();
              int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
              bool ephemeral = false;
              if (use_mr == nullptr && use_direct_cpu_mr) {
                // Register an ephemeral MR for this small CPU slab
                if (dev->reg_mr(&use_mr, reinterpret_cast<void*>(addr_u64), seg_bytes, access) != SUCCESS) {
                  auto rspf = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
                  auto* pf = rspf->get_payload<ProtoReadFailed>();
                  memcpy(pf->tensor_key, req->tensor_key, kMaxTensorNameLen);
                  pf->offset = off;
                  pf->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
                  COMM_CHECK(t->send(rspf));
                  break;
                }
                ephemeral = true;
                inflight_direct_mr_.fetch_add(1);
              }
              if (use_mr == nullptr) {
                auto rspf = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
                auto* pf = rspf->get_payload<ProtoReadFailed>();
                memcpy(pf->tensor_key, req->tensor_key, kMaxTensorNameLen);
                pf->offset = off;
                pf->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
                COMM_CHECK(t->send(rspf));
                break;
              }
              seg_pl->addr = addr_u64;
              seg_pl->rkey = use_mr->rkey;
              if (ephemeral) {
                // Track for TTL-based deregistration (no ACK for direct MR)
                StagedRdmaSegment seg;
                seg.ptr = nullptr;
                seg.bytes = seg_bytes;
                seg.mr = use_mr;
                seg.kind = StagedRdmaSegment::Kind::CPU;
                seg.ts_us = get_us();
                seg.deregister_mr = true;
                seg.stager_ptr = nullptr;
                const std::string seg_req_key = get_request_key(tensor_key, off);
                {
                  absl::MutexLock lk(&staged_mu_);
                  staged_segments_.put(seg_req_key, std::move(seg));
                }
              }
            }
          }
          COMM_CHECK(t->send(rsp));
        } else {
          // MTCP path: stream data and reply with legacy READ_RESPONSE for compatibility
          auto rsp = EngineMessage::make_message<ProtoReadResponse>(ENGINE_OP_READ_RESPONSE);
          auto* payload = rsp->get_payload<ProtoReadResponse>();
          memcpy(payload->tensor_key, req->tensor_key, kMaxTensorNameLen);
          payload->offset = req->offset;
          auto write_request = std::make_shared<WriteRequest>(tensor, payload->tensor_key, req->offset, req->bytes);
          auto transport = channel->get_mtcp();
          if (transport == nullptr) {
            transport = std::make_shared<MTcpTransport>(mtcp_conn_count_);
            channel->set_transport(transport);
          }
          if (gpu_tcp_stager_) transport->set_gpu_tcp_stager(gpu_tcp_stager_);
          if (gpu_memory_stager_) transport->set_gpu_memory_stager(gpu_memory_stager_);
          if (gpu_memory_pool_) transport->set_memory_pool(gpu_memory_pool_);
          if (memory_stager_) transport->set_memory_stager(memory_stager_);
          transport->send(write_request);
          payload->transport_type = ENGINE_TRANSPORT_MTCP;
          payload->bytes = req->bytes;
          payload->addr = 0;
          payload->rkey = 0;
          payload->staged = 0;
          STRCPY(payload->nic_name, "");
          COMM_CHECK(t->send(rsp));
        }
      }
      break;
    }
    case ENGINE_OP_RDMA_READ_DONE: {
      auto* ack = msg->get_payload<ProtoRdmaReadDone>();
      const std::string tensor_key = reinterpret_cast<char*>(ack->tensor_key);
      const std::string req_key = get_request_key(tensor_key, ack->offset);
      StagedRdmaSegment seg;
      {
        absl::MutexLock lk(&staged_mu_);
        seg = staged_segments_.get(req_key);
        if (seg.ptr != nullptr || seg.mr != nullptr) {
          staged_segments_.del(req_key);
        } else {
          // not found
          seg.ptr = nullptr;
        }
      }
      if (seg.ptr != nullptr || seg.mr != nullptr) {
        // Deregister MR and release buffer
        if (seg.mr && seg.deregister_mr) {
          CHECK_WARN(wrap_ibv_dereg_mr(seg.mr), "failed to dereg staged mr");
          if (seg.ptr == nullptr) {
            // Ephemeral direct-MR case: decrement inflight counter
            inflight_direct_mr_.fetch_sub(1);
          }
        }
        if (seg.ptr) {
          // Use the stager that produced the buffer if available
          if (seg.stager_ptr != nullptr) {
            auto st = seg.stager_ptr->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
            if (!st.ok()) {
              LOG(WARNING) << "Failed to release staged buffer on ACK: " << st;
            }
          } else if (memory_stager_) {
            auto st = memory_stager_->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
            if (!st.ok()) {
              LOG(WARNING) << "Failed to release staged buffer on ACK (default CPU stager): " << st;
            }
          } else if (gpu_memory_stager_) {
            auto st = gpu_memory_stager_->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
            if (!st.ok()) {
              LOG(WARNING) << "Failed to release staged buffer on ACK (default GPU stager): " << st;
            }
          }
        }
      } else {
        LOG(WARNING) << "RDMA_READ_DONE for unknown key: " << req_key;
      }
      break;
    }
    case ENGINE_OP_RDMA_READ_DONE_EX: {
      auto* hdr = msg->get_payload<ProtoRdmaReadDoneExHeader>();
      const std::string tensor_key = reinterpret_cast<char*>(hdr->tensor_key);
      for (uint32_t i = 0; i < hdr->num_segments; ++i) {
        auto* s = reinterpret_cast<ProtoRdmaReadDoneExSeg*>(
            reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoRdmaReadDoneExHeader) + i * sizeof(ProtoRdmaReadDoneExSeg));
        const std::string req_key = get_request_key(tensor_key, s->offset);
        StagedRdmaSegment seg;
        {
          absl::MutexLock lk(&staged_mu_);
          seg = staged_segments_.get(req_key);
          if (seg.ptr != nullptr || seg.mr != nullptr) {
            staged_segments_.del(req_key);
          } else {
            seg.ptr = nullptr;
          }
        }
        if (seg.ptr != nullptr || seg.mr != nullptr) {
        if (seg.mr && seg.deregister_mr) {
          CHECK_WARN(wrap_ibv_dereg_mr(seg.mr), "failed to dereg staged mr");
          if (seg.ptr == nullptr) {
            inflight_direct_mr_.fetch_sub(1);
          }
        }
        if (seg.ptr) {
          if (seg.stager_ptr != nullptr) {
            auto st = seg.stager_ptr->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
            if (!st.ok()) {
                LOG(WARNING) << "Failed to release staged buffer on ACK_EX: " << st;
              }
            } else if (memory_stager_) {
              auto st = memory_stager_->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
              if (!st.ok()) {
                LOG(WARNING) << "Failed to release staged buffer on ACK_EX (default CPU stager): " << st;
              }
            } else if (gpu_memory_stager_) {
              auto st = gpu_memory_stager_->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
              if (!st.ok()) {
                LOG(WARNING) << "Failed to release staged buffer on ACK_EX (default GPU stager): " << st;
              }
            }
          }
        } else {
          LOG(WARNING) << "RDMA_READ_DONE_EX for unknown key: " << req_key;
        }
      }
      break;
    }
    default:
      LOG(WARNING) << "failed to process request: " << msg->get_op();
      return FAILED;
  }
  return SUCCESS;
}

result_t CommunicateEngine::on_receive_response(
    const channel_t& channel,
    const tcp_transport_t& t,
    const engine_message_t& msg) {
  LOG(INFO) << "[on_receive_response] Received response op=" << msg->get_op() << " from " << t->get_remote_url();

  switch (msg->get_op()) {
    case ENGINE_OP_RDMA_CONNECT_RESPONSE: {
      LOG(INFO) << "get rdma response from " << t->get_remote_url();

      auto* req = msg->get_payload<ProtoRdmaConnectResponse>();
      std::string local_dev_name = reinterpret_cast<char*>(req->src_dev_name);
      std::string peer_dev_name = reinterpret_cast<char*>(req->dst_dev_name);
      auto transport = channel->get_rdma(local_dev_name, peer_dev_name);
      if (transport == nullptr) {
        LOG(ERROR) << "failed to find transport from " << t->get_remote_url() << ", local-dev=" << local_dev_name
                   << ", peer-dev=" << peer_dev_name;
      }
      COMM_CHECK(transport->connect(&req->qp_info));
      break;
    }
    case ENGINE_OP_MTCP_CONNECT_RESPONSE: {
      LOG(INFO) << "get mtcp response from " << t->get_remote_url();

      auto* rsp = msg->get_payload<ProtoMtcpConnectResponse>();
      struct in_addr sin_addr = {};
      sin_addr.s_addr = htonl(rsp->ip);
      auto transport = channel->get_mtcp();
      auto* ip = inet_ntoa(sin_addr);
      LOG(INFO) << "[on_receive_response] MTCP_CONNECT_RESPONSE: connecting to " << ip << ":" << rsp->port
                << " with conn_count=" << rsp->conn_count;
      COMM_CHECK(transport->connect(ip, rsp->port, mtcp_conn_count_));
      LOG(INFO) << "mtcp connect done " << ip << ":" << rsp->port << " " << mtcp_conn_count_;
      break;
    }
    case ENGINE_OP_READ_RESPONSE: {
      auto* rsp = msg->get_payload<ProtoReadResponse>();
      std::string tensor_key = reinterpret_cast<char*>(rsp->tensor_key);
      std::string peer_dev_name = reinterpret_cast<char*>(rsp->nic_name);

      LOG(INFO) << "[on_receive_response] READ_RESPONSE: key=" << tensor_key << " offset=" << rsp->offset
                << " bytes=" << rsp->bytes
                << " transport=" << (rsp->transport_type == ENGINE_TRANSPORT_MTCP ? "MTCP" : "RDMA");

      remote_tensor_t remote_tensor =
          std::make_shared<RemotePartitionTensor>(tensor_key, peer_dev_name, rsp->addr, rsp->bytes, rsp->rkey);

      auto req_key = get_request_key(tensor_key, rsp->offset);
      auto read_request = pending_requests_.get(req_key);
      if (read_request == nullptr) {
        LOG(ERROR) << "[on_receive_response] READ_RESPONSE: pending request not found for " << req_key;
        break;
      }
      LOG(INFO) << "[on_receive_response] READ_RESPONSE: found pending request for " << req_key;

      read_request->record_request_response();
      read_request->set_remote_tensor(remote_tensor);

      if (enable_rdma_ && rsp->transport_type == ENGINE_TRANSPORT_RDMA) {
        CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";

        auto tensor = read_request->get_local_tensor();
        auto transport = channel->get_rdma(tensor->get_dev()->get_name(), peer_dev_name);
        if (transport == nullptr) {
          transport = rdma_context_->create_transport(tensor->get_dev()->get_name());
          auto req = EngineMessage::make_message<ProtoRdmaConnectRequest>(ENGINE_OP_RDMA_CONNECT_REQUEST);
          auto* payload = req->get_payload<ProtoRdmaConnectRequest>();
          COMM_CHECK(transport->get_local_info(&payload->qp_info));
          STRCPY(payload->src_dev_name, tensor->get_dev()->get_name());
          STRCPY(payload->dst_dev_name, peer_dev_name);
          COMM_CHECK(t->send(req));
          channel->set_transport(tensor->get_dev()->get_name(), peer_dev_name, transport);
        }
        // Set per-request ACK action if server indicated staged response
        if (rsp->staged) {
          const std::string staged_key = tensor_key;
          const uint64_t staged_off = rsp->offset;
          auto ctrl = channel->get_control();
          read_request->set_ack_action([ctrl, staged_key, staged_off]() {
            auto ack = EngineMessage::make_message<ProtoRdmaReadDone>(ENGINE_OP_RDMA_READ_DONE);
            auto* pl = ack->get_payload<ProtoRdmaReadDone>();
            STRNCPY(pl->tensor_key, staged_key.c_str(), kMaxTensorNameLen);
            pl->offset = staged_off;
            COMM_CHECK(ctrl->send(ack));
          });
        }
        CHECK_WARN(transport->read(read_request), "failed to read");
      } else {
        auto transport = channel->get_mtcp();
        if (transport == nullptr) {
          transport = std::make_shared<MTcpTransport>(kMTcpConnCount);
          LOG(INFO) << "[on_receive_response] Sending MTCP_CONNECT_REQUEST for " << tensor_key;
          auto req = EngineMessage::make_message<ProtoMtcpConnectRequest>(ENGINE_OP_MTCP_CONNECT_REQUEST);
          auto* payload = req->get_payload<ProtoMtcpConnectRequest>();
          payload->conn_count = kMTcpConnCount;
          COMM_CHECK(t->send(req));

          channel->set_transport(transport);
        }

        if (gpu_tcp_stager_) {
          transport->set_gpu_tcp_stager(gpu_tcp_stager_);
        }
        if (gpu_memory_pool_) {
          transport->set_memory_pool(gpu_memory_pool_);
        }
        if (memory_stager_) {
          transport->set_memory_stager(memory_stager_);
        }

        LOG(INFO) << "[on_receive_response] Starting MTCP recv for " << tensor_key;
        CHECK_WARN(transport->recv(read_request), "failed to read");
      }

      pending_requests_.del(req_key);
      LOG(INFO) << "[on_receive_response] Removed pending request for " << req_key;
      break;
    }
    case ENGINE_OP_READ_RESPONSE_EX: {
      auto* hdr = msg->get_payload<ProtoReadResponseExHeader>();
      std::string tensor_key = reinterpret_cast<char*>(hdr->tensor_key);
      std::string peer_dev_name = reinterpret_cast<char*>(hdr->nic_name);

      LOG(INFO) << "[on_receive_response] READ_RESPONSE_EX: key=" << tensor_key << " segs=" << hdr->num_segments
                << " transport=" << (hdr->transport_type == ENGINE_TRANSPORT_MTCP ? "MTCP" : "RDMA");

      auto* seg0 = reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
      auto req_key = get_request_key(tensor_key, seg0->offset);
      auto read_request = pending_requests_.get(req_key);
      if (read_request == nullptr) {
        LOG(ERROR) << "[on_receive_response] READ_RESPONSE_EX: pending request not found for " << req_key;
        break;
      }
      read_request->record_request_response();

      if (enable_rdma_ && hdr->transport_type == ENGINE_TRANSPORT_RDMA) {
        CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";

        auto tensor = read_request->get_local_tensor();
        auto transport = channel->get_rdma(tensor->get_dev()->get_name(), peer_dev_name);
        if (transport == nullptr) {
          transport = rdma_context_->create_transport(tensor->get_dev()->get_name());
          auto req = EngineMessage::make_message<ProtoRdmaConnectRequest>(ENGINE_OP_RDMA_CONNECT_REQUEST);
          auto* payload = req->get_payload<ProtoRdmaConnectRequest>();
          COMM_CHECK(transport->get_local_info(&payload->qp_info));
          STRNCPY(payload->src_dev_name, tensor->get_dev()->get_name(), kMaxDevName);
          STRNCPY(payload->dst_dev_name, peer_dev_name.c_str(), kMaxDevName);
          COMM_CHECK(t->send(req));
          channel->set_transport(tensor->get_dev()->get_name(), peer_dev_name, transport);
        }
        // Build RDMA segment list and offsets for batched ACK
        std::vector<RdmaTransport::RdmaReadSeg> rdma_segs;
        rdma_segs.reserve(hdr->num_segments);
        std::vector<uint64_t> ack_offsets;
        ack_offsets.reserve(hdr->num_segments);
        uint64_t base_off = read_request->remote_offset_;
        for (uint32_t i = 0; i < hdr->num_segments; ++i) {
          auto* s = reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader) + i * sizeof(ProtoReadResponseExSeg));
          RdmaTransport::RdmaReadSeg seg{};
          seg.remote_addr = s->addr;
          seg.rkey = s->rkey;
          seg.length = s->bytes;
          seg.local_addr = tensor->get_uint64_addr() + (s->offset - base_off);
          rdma_segs.emplace_back(seg);
          ack_offsets.emplace_back(s->offset);
        }
        // Set per-request ACK action to send batched ACK_EX for all segments
        if (hdr->staged) {
          auto ctrl = channel->get_control();
          const std::string staged_key = tensor_key;
          auto offsets = std::make_shared<std::vector<uint64_t>>(std::move(ack_offsets));
          read_request->set_ack_action([ctrl, staged_key, offsets]() {
            auto ack = std::make_shared<EngineMessage>(
                ENGINE_OP_RDMA_READ_DONE_EX,
                static_cast<uint32_t>(sizeof(ProtoRdmaReadDoneExHeader) + offsets->size() * sizeof(ProtoRdmaReadDoneExSeg)));
            auto* h = ack->get_payload<ProtoRdmaReadDoneExHeader>();
            STRNCPY(h->tensor_key, staged_key.c_str(), kMaxTensorNameLen);
            h->num_segments = static_cast<uint32_t>(offsets->size());
            for (size_t i = 0; i < offsets->size(); ++i) {
              auto* s = reinterpret_cast<ProtoRdmaReadDoneExSeg*>(
                  reinterpret_cast<uint8_t*>(h) + sizeof(ProtoRdmaReadDoneExHeader) + i * sizeof(ProtoRdmaReadDoneExSeg));
              s->offset = (*offsets)[i];
            }
            COMM_CHECK(ctrl->send(ack));
          });
        }
        CHECK_WARN(transport->read_multi(read_request, rdma_segs), "failed to read (multi)");
      } else {
        LOG(ERROR) << "[on_receive_response] READ_RESPONSE_EX without RDMA not supported";
        read_request->set_result(absl::InternalError("READ_RESPONSE_EX without RDMA not supported"));
      }
      // Remove pending entry now; completion is tracked via the request future and ACK action
      pending_requests_.del(req_key);
      break;
    }
    case ENGINE_OP_RDMA_CONNECT_FAILED: {
      auto* req = msg->get_payload<ProtoRdmaConnectFailed>();
      std::string peer_dev_name = reinterpret_cast<char*>(req->dst_dev_name);
      std::string local_dev_name = reinterpret_cast<char*>(req->src_dev_name);
      LOG(ERROR) << "[on_receive_response] RDMA_CONNECT_FAILED: local=" << local_dev_name << " peer=" << peer_dev_name;
      channel->del_transport(local_dev_name, peer_dev_name);
      break;
    }
    case ENGINE_OP_READ_FAILED: {
      auto* rsp = msg->get_payload<ProtoReadFailed>();
      auto tensor_key = std::string(reinterpret_cast<char*>(rsp->tensor_key));
      auto req_key = get_request_key(tensor_key, rsp->offset);

      LOG(ERROR) << "[on_receive_response] READ_FAILED: key=" << tensor_key << " offset=" << rsp->offset
                 << " reason=" << rsp->reason;

      auto read_request = pending_requests_.get(req_key);
      if (read_request == nullptr) {
        LOG(WARNING) << "failed to get read response: key=" << tensor_key;
        break;
      }
      pending_requests_.del(req_key);
      read_request->set_result(absl::InternalError("failed to read from peer"));
      break;
    }
    default:
      LOG(WARNING) << "failed to process response: " << msg->get_op();
  }

  return SUCCESS;
}

net_dev_t CommunicateEngine::get_net_dev(int dev_type, int dev_id) {
  net_dev_t net_dev(nullptr);
  if (enable_rdma_) {
    CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";
    if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
      net_dev = rdma_context_->get_best_dev(dev_id);
    }
    if (net_dev == nullptr) {
      LOG(WARNING) << "failed to get net dev for gpu=" << dev_id << "; provide explicit device mapping in config";
      return nullptr;
    }
  }
  return net_dev;
}

absl::Status CommunicateEngine::close_connection(const std::string& dst_ip, uint16_t dst_port) {
  std::stringstream url;
  url << dst_ip << ":" << dst_port;
  if (channels_.exist(url.str())) {
    auto channel = channels_.get(url.str());
    channels_.del(url.str());
    if (channel != nullptr) {
      channel->close();
    }
  } else {
    return absl::InternalError("could not find the connection");
  }
  return absl::OkStatus();
}

void CommunicateEngine::do_channel_gc_loop() {
  while (!stop_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    auto pairs = channels_.pairs();
    auto now = get_us() / 1000000;
    for (auto& p : pairs) {
      if (p.second->is_expired(now)) {
        LOG(INFO) << "channel gc " << p.first;
        channels_.del(p.first);
        p.second->close();
      }
    }

    pairs.clear();

    // Reap expired staged RDMA segments if ACK missing
    const uint64_t ttl_ms = ack_ttl_ms_ ? ack_ttl_ms_ : 30000;
    if (ttl_ms > 0) {
      auto staged_pairs = staged_segments_.pairs();
      for (auto& kv : staged_pairs) {
        const auto& key = kv.first;
        auto seg = kv.second;
        if (seg.ts_us > 0) {
          uint64_t age_ms = (get_us() - seg.ts_us) / 1000;
          if (age_ms > ttl_ms) {
            LOG(WARNING) << "Reaping staged RDMA segment due to missing ACK: " << key;
            {
              absl::MutexLock lk(&staged_mu_);
              // Ensure it's still present
              auto cur = staged_segments_.get(key);
              if (cur.ptr == seg.ptr && cur.mr == seg.mr) {
                staged_segments_.del(key);
              } else {
                continue;
              }
            }
            if (seg.mr && seg.deregister_mr) {
              CHECK_WARN(wrap_ibv_dereg_mr(seg.mr), "failed to dereg staged mr (reap)");
              if (seg.ptr == nullptr) {
                inflight_direct_mr_.fetch_sub(1);
              }
            }
            if (seg.ptr) {
              if (seg.stager_ptr != nullptr) {
                auto st = seg.stager_ptr->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
                if (!st.ok()) {
                  LOG(WARNING) << "Failed to release staged buffer (reap): " << st;
                }
              } else if (memory_stager_) {
                auto st = memory_stager_->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
                if (!st.ok()) {
                  LOG(WARNING) << "Failed to release staged buffer (reap, default CPU): " << st;
                }
              } else if (gpu_memory_stager_) {
                auto st = gpu_memory_stager_->release_staged_buffer(gsl::not_null<void*>{seg.ptr});
                if (!st.ok()) {
                  LOG(WARNING) << "Failed to release staged buffer (reap, default GPU): " << st;
                }
              }
            }
          }
        }
      }
    }
  }
}

std::shared_ptr<MemoryStager> CommunicateEngine::get_cpu_stager_for_nic(const std::string& nic_name) const {
  auto it = nic_cpu_stagers_.find(nic_name);
  if (it != nic_cpu_stagers_.end()) return it->second;
  return nullptr;
}

std::shared_ptr<GpuTcpStager> CommunicateEngine::get_gpu_stager_for_id(int gpu_id) const {
  auto it = gpu_stagers_.find(gpu_id);
  if (it != gpu_stagers_.end()) return it->second;
  return nullptr;
}

std::shared_ptr<MemoryStager> CommunicateEngine::get_gpu_mem_stager_for_id(int gpu_id) const {
  auto it = gpu_mem_stagers_.find(gpu_id);
  if (it != gpu_mem_stagers_.end()) return it->second;
  return nullptr;
}

} // namespace tensorcast::communicator
