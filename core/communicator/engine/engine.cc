
// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"

#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/protocol.h"
#include "core/communicator/misc/envs.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/rdma_context.h"

namespace stepcast::communicator {

ENV_PARAM_STR(DEFAULT_DEV, "");
ENV_PARAM(GPU_TCP_STAGER_CHUNK_SIZE_MB, 64); // Default 64MB chunks
ENV_PARAM(GPU_TCP_STAGER_NUM_BUFFERS, 2); // Default 2 buffers for double buffering
ENV_PARAM(GPU_TCP_RECV_NUM_BUFFERS, 4); // Default 4 buffers for GPU receive operations

CommunicateEngine::CommunicateEngine(bool enable_rdma, uint32_t channel_expire_sec)
    : stop_(false),
      inited_(false),
      server_context_(new TcpContext()),
      client_context_(new TcpContext()),
      enable_rdma_(enable_rdma),
      mtcp_conn_count_(kMTcpConnCount),
      channel_expire_(channel_expire_sec) {
  request_thread_ = std::thread([this]() { this->do_read_request_loop(); });
  gc_thread_ = std::thread([this]() { this->do_channel_gc_loop(); });

  if (enable_rdma_) {
    rdma_context_ = std::make_shared<RdmaContext>();
  } else {
    const size_t chunk_size = GPU_TCP_STAGER_CHUNK_SIZE_MB * 1024 * 1024;
    const size_t num_buffers = GPU_TCP_STAGER_NUM_BUFFERS;

    // Create a shared memory pool for both staging and receiving
    // The pool should be large enough for both send and receive operations
    const size_t recv_num_buffers = GPU_TCP_RECV_NUM_BUFFERS;
    const size_t total_pool_size = chunk_size * (num_buffers + recv_num_buffers);

    gpu_memory_pool_ = std::make_shared<store::PinnedMemoryPool>(total_pool_size, chunk_size);
    gpu_tcp_stager_ = std::make_shared<GpuTcpStager>(chunk_size, num_buffers, gpu_memory_pool_);

    VLOG(2) << "Initialized GPU TCP stager with " << num_buffers << " buffers of " << GPU_TCP_STAGER_CHUNK_SIZE_MB
            << " MB each";
    VLOG(2) << "Initialized shared pinned memory pool with total size " << total_pool_size / (1024 * 1024) << " MB";
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

  // Mark tensors that need GPU->CPU staging in TCP mode
  if (!enable_rdma_ && dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
    tensor->set_needs_staging(true);
  }

  if (enable_rdma_) {
    net_dev->reg_async(tensor);
    if (!async) {
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
        payload->reason = STEPCAST_READ_FAILED_NO_TENSOR;
        COMM_CHECK(t->send(rsp));
      } else if (req->offset + req->bytes > tensor->get_bytes()) {
        auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
        auto* payload = rsp->get_payload<ProtoReadFailed>();
        memcpy(payload->tensor_key, req->tensor_key, 512);
        payload->offset = req->offset;
        payload->reason = STEPCAST_READ_FAILED_OVERFLOW;
        COMM_CHECK(t->send(rsp));
      } else {
        auto rsp = EngineMessage::make_message<ProtoReadResponse>(ENGINE_OP_READ_RESPONSE);
        auto* payload = rsp->get_payload<ProtoReadResponse>();
        memcpy(payload->tensor_key, req->tensor_key, kMaxTensorNameLen);
        payload->offset = req->offset;
        if (enable_rdma_ && req->transport_type == ENGINE_TRANSPORT_RDMA) {
          payload->transport_type = ENGINE_TRANSPORT_RDMA;
          auto dev = tensor->get_dev();
          STRNCPY(payload->nic_name, dev->get_name(), kMaxDevName);
          tensor->wait_read_ready();
          auto* mr = tensor->get_mr();
          payload->addr = tensor->get_uint64_addr() + req->offset;
          payload->rkey = mr->rkey;
          payload->bytes = req->bytes;
        } else {
          // using mtcp transport
          auto write_request = std::make_shared<WriteRequest>(tensor, payload->tensor_key, req->offset, req->bytes);
          auto transport = channel->get_mtcp();
          if (transport == nullptr) {
            transport = std::make_shared<MTcpTransport>(mtcp_conn_count_);
            channel->set_transport(transport);
          }
          if (gpu_tcp_stager_) {
            transport->set_gpu_tcp_stager(gpu_tcp_stager_);
          }
          if (gpu_memory_pool_) {
            transport->set_memory_pool(gpu_memory_pool_);
          }
          transport->send(write_request);
          payload->transport_type = ENGINE_TRANSPORT_MTCP;
          payload->bytes = req->bytes;
        }
        COMM_CHECK(t->send(rsp));
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

        LOG(INFO) << "[on_receive_response] Starting MTCP recv for " << tensor_key;
        CHECK_WARN(transport->recv(read_request), "failed to read");
      }

      pending_requests_.del(req_key);
      LOG(INFO) << "[on_receive_response] Removed pending request for " << req_key;
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
      if (DEFAULT_DEV.empty()) {
        LOG(WARNING) << "failed to get net dev for gpu=" << dev_id;
        return nullptr;
      }

      LOG(WARNING) << "failed to find a net dev for gpu" << dev_id << ", automatically, use the default net dev "
                   << DEFAULT_DEV;
      net_dev = rdma_context_->get_dev(DEFAULT_DEV);
      if (net_dev == nullptr) {
        LOG(WARNING) << "failed to get default net dev " << DEFAULT_DEV;
        return nullptr;
      }
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
  }
}

} // namespace stepcast::communicator
