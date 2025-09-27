
// Copyright (c) 2025, TensorCast Team.

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
// absl string utilities no longer needed in typed-config engine

#include "core/common/system_capabilities.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/dram_stager.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/gpu_net_stager.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/protocol.h"
#include "core/communicator/engine/staging_flow_controller.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/rdma_context.h"

namespace tensorcast::communicator::engine {

using base::CHANNEL_MTCP;
using base::CHANNEL_RDMA;
using base::COMMUNICATE_ENGINE_DEV_CPU;
using base::COMMUNICATE_ENGINE_DEV_GPU;
using misc::get_us;
using misc::INTERNAL_ERROR;
using misc::SUCCESS;
using transport::future_read_result_t;
using transport::net_dev_t;
using transport::PartitionTensor;
using transport::RdmaContext;
using transport::read_request_t;
using transport::tcp_transport_t;

struct RdmaReadSession {
  ProtoReadRequest request;
  std::string tensor_key;
  std::string request_key;
  std::shared_ptr<PartitionTensor> tensor;
  std::shared_ptr<MemoryStager> stager;
  net_dev_t dev;
  tcp_transport_t control_transport;
  std::unique_ptr<StagingWindow> window;
};

namespace {

struct RdmaDriveResult {
  absl::Status status = absl::OkStatus();
  bool made_progress = false;
  bool completed = false;
};

RdmaDriveResult DriveRdmaSession(Channel::FlowState& flow_state, RdmaReadSession& session) {
  RdmaDriveResult result;
  while (true) {
    auto window_or = session.window->stage_next();
    if (!window_or.ok()) {
      if (absl::IsOutOfRange(window_or.status())) {
        result.completed = true;
        result.status = absl::OkStatus();
        return result;
      }
      result.status = window_or.status();
      return result;
    }

    auto staged_window = std::move(window_or).value();
    if (staged_window.segments.empty()) {
      // No staged segments implies no credit; propagate as unavailable for the caller to retry later.
      result.status = absl::UnavailableError("staging produced no segments");
      return result;
    }

    result.made_progress = true;
    LOG(INFO) << "[staging_credit] request=" << session.request_key
              << " transport=rdma window=" << staged_window.window_seq << " granted=" << staged_window.granted_credit
              << " more=" << (staged_window.more_segments ? "yes" : "no")
              << " outstanding=" << flow_state.ledger.outstanding_credit();

    const uint32_t seg_count = static_cast<uint32_t>(staged_window.segments.size());
    auto rsp = std::make_shared<EngineMessage>(
        ENGINE_OP_READ_RESPONSE_EX,
        static_cast<uint32_t>(sizeof(ProtoReadResponseExHeader) + seg_count * sizeof(ProtoReadResponseExSeg)));

    auto* hdr = rsp->get_payload<ProtoReadResponseExHeader>();
    memcpy(hdr->tensor_key, session.request.tensor_key, kMaxTensorNameLen);
    hdr->transport_type = ENGINE_TRANSPORT_RDMA;
    hdr->staged = 1;
    misc::STRNCPY(hdr->nic_name, session.dev ? session.dev->get_name().c_str() : "", kMaxDevName);
    hdr->num_segments = seg_count;
    hdr->window_seq = staged_window.window_seq;
    hdr->credit_granted = static_cast<uint32_t>(staged_window.granted_credit);
    hdr->more_segments = staged_window.more_segments ? 1 : 0;

    std::vector<StageLeaseKey> inserted_keys;
    inserted_keys.reserve(seg_count);

    for (uint32_t i = 0; i < seg_count; ++i) {
      auto& segment = staged_window.segments[i];
      auto* seg_pl = reinterpret_cast<ProtoReadResponseExSeg*>(
          reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader) + i * sizeof(ProtoReadResponseExSeg));

      seg_pl->addr = reinterpret_cast<uint64_t>(segment.lease.host_ptr());
      seg_pl->offset = segment.offset;
      seg_pl->bytes = segment.bytes;
      seg_pl->rkey = segment.lease.mr() ? segment.lease.mr()->rkey : 0;

      StageLease::Metadata metadata = segment.lease.metadata();
      metadata.window_seq = staged_window.window_seq;
      metadata.segment_idx = segment.segment_idx;
      metadata.offset = segment.offset;
      metadata.bytes = segment.bytes;
      segment.lease.set_metadata(metadata);

      StageLeaseKey key{
          .request_key = metadata.request_key,
          .window_seq = metadata.window_seq,
          .segment_idx = metadata.segment_idx,
      };
      flow_state.registry.put(key, segment.lease);
      inserted_keys.push_back(key);
    }

    misc::result_t send_res = session.control_transport->send(rsp);
    if (send_res != misc::SUCCESS) {
      LOG(ERROR) << "Failed to send RDMA READ_RESPONSE_EX window: res=" << send_res;
      for (const auto& key : inserted_keys) {
        auto lease_or = flow_state.registry.take(key);
        if (lease_or.ok()) {
          lease_or->release();
        }
      }
      result.status = absl::InternalError("failed to send RDMA window");
      return result;
    }

    if (!staged_window.more_segments) {
      result.completed = true;
      result.status = absl::OkStatus();
      return result;
    }
  }
}

} // namespace

// Engine is fully typed-config driven; no environment-variable reads here.

// No legacy constructors; typed CommunicatorConfig is required.

Communicator::Communicator(const v1::CommunicatorConfig& config, uint32_t channel_expire_sec)
    : stop_(false),
      inited_(false),
      server_context_(new transport::TcpContext()),
      client_context_(new transport::TcpContext()),
      enable_rdma_(config.enable_rdma()),
      mtcp_conn_count_(config.transport().tcp_conn_count()),
      ack_ttl_ms_(config.rdma().ack_ttl_ms()),
      config_(config),
      channel_expire_(channel_expire_sec) {
  common::SystemCapabilities::instance().record_rdma_available(enable_rdma_);
  request_thread_ = std::thread([this]() { this->do_read_request_loop(); });
  gc_thread_ = std::thread([this]() { this->do_channel_gc_loop(); });
  // Apply typed config to TCP contexts
  server_context_->set_connect_timeout(config_.transport().connect_timeout_sec());
  client_context_->set_connect_timeout(config_.transport().connect_timeout_sec());

  // No default residency provider required; staging policy no longer consults UMA bridges.

  // Staging resources sized from config
  const size_t gpu_chunk_size =
      (config_.stager().stage_chunk_mb_gpu() > 0 ? config_.stager().stage_chunk_mb_gpu() : 16) * 1024ULL * 1024ULL;
  const size_t cpu_chunk_size =
      (config_.stager().stage_chunk_mb_cpu() > 0 ? config_.stager().stage_chunk_mb_cpu() : 4) * 1024ULL * 1024ULL;
  const size_t num_buffers = (config_.stager().buffers_per_flow() > 0 ? config_.stager().buffers_per_flow() : 4);
  buffers_per_flow_ = static_cast<int>(num_buffers);
  const uint32_t configured_max_window = config_.stager().max_window_segments();
  if (configured_max_window == 0) {
    max_window_segments_ = static_cast<uint32_t>(num_buffers);
  } else {
    if (configured_max_window > num_buffers) {
      LOG(WARNING) << "max_window_segments=" << configured_max_window << " exceeds buffers_per_flow=" << num_buffers
                   << "; clamping to buffers_per_flow";
    }
    max_window_segments_ = std::min(configured_max_window, static_cast<uint32_t>(num_buffers));
  }
  const size_t recv_num_buffers = num_buffers; // unify receiver buffering under stager policy
  const size_t total_pool_size = config_.pool().pool_size_bytes() > 0
      ? config_.pool().pool_size_bytes()
      : gpu_chunk_size * (num_buffers + recv_num_buffers);

  // GPU staging pool and stager
  gpu_memory_pool_ = std::make_shared<common::memory::PinnedBufferPool>(total_pool_size, gpu_chunk_size);
  gpu_memory_stager_ = std::make_shared<GpuNetStager>(gpu_chunk_size, num_buffers, gpu_memory_pool_);

  // CPU staging pool honors CPU chunk size; size conservatively for one flow
  if (cpu_chunk_size != gpu_chunk_size) {
    const size_t cpu_pool_size = cpu_chunk_size * num_buffers; // minimal to honor buffers_per_flow
    cpu_memory_pool_ = std::make_shared<common::memory::PinnedBufferPool>(cpu_pool_size, cpu_chunk_size);
  }
  auto dram_pool = cpu_memory_pool_ ? cpu_memory_pool_ : gpu_memory_pool_;
  memory_stager_ = std::make_shared<DRAMStager>(
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{dram_pool}, /*num_buffers_hint=*/num_buffers);
  if (auto ds = std::dynamic_pointer_cast<DRAMStager>(memory_stager_)) {
    ds->set_lease_provider(DRAMStager::make_noop_lease_provider());
  }

  if (enable_rdma_) {
    rdma_context_ = std::make_shared<RdmaContext>();
    mr_cache_ = std::make_unique<MrCache>();
    // Apply typed RDMA QP tuning
    rdma_context_->set_qp_params(
        config_.rdma().traffic_class(), config_.rdma().qp_timeout(), config_.rdma().qp_retry());

    if (config_.simple_numa().enable()) {
      for (const auto& node : config_.simple_numa().nodes()) {
        auto pool = std::make_shared<common::memory::PinnedBufferPool>(total_pool_size, gpu_chunk_size);
        numa_pools_.push_back(pool);
        auto cpu_stager = std::make_shared<DRAMStager>(
            gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{pool}, /*num_buffers_hint=*/num_buffers);
        if (auto ds = std::dynamic_pointer_cast<DRAMStager>(cpu_stager)) {
          ds->set_lease_provider(DRAMStager::make_noop_lease_provider());
        }
        auto gpu_mem_stager = std::make_shared<GpuNetStager>(gpu_chunk_size, num_buffers, pool);
        // Map GPU ids
        for (int gid : node.gpus()) {
          gpu_mem_stagers_[gid] = gpu_mem_stager;
        }
        // Map NIC names
        for (const auto& nic : node.nics()) {
          nic_cpu_stagers_[nic] = cpu_stager;
        }
      }
    }

    // Preregister MRs for all pools
    int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
    std::vector<std::shared_ptr<common::memory::PinnedBufferPool>> pools;
    pools.push_back(gpu_memory_pool_);
    if (cpu_memory_pool_ && cpu_memory_pool_.get() != gpu_memory_pool_.get()) {
      pools.push_back(cpu_memory_pool_);
    }
    for (auto& p : numa_pools_)
      pools.push_back(p);
    for (const auto& dev : rdma_context_->list_devs()) {
      for (auto& pool : pools) {
        auto buffers = pool->list_buffers();
        for (auto ptr : buffers) {
          auto* mr = mr_cache_->get_or_register(dev->get_pd(), ptr.get(), pool->slice_bytes(), access);
          if (mr == nullptr) {
            LOG(WARNING) << "Failed to preregister MR for buffer " << static_cast<void*>(ptr.get()) << " on PD";
          }
        }
      }
    }
  }
}

Communicator::~Communicator() {
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

void Communicator::set_dram_lease_provider(const std::shared_ptr<DRAMStager::LeaseProvider>& provider) {
  if (!memory_stager_)
    return;
  if (auto ds = std::dynamic_pointer_cast<DRAMStager>(memory_stager_)) {
    ds->set_lease_provider(provider);
  }
  // Also propagate to NUMA CPU stagers if present
  for (auto& kv : nic_cpu_stagers_) {
    if (auto ds2 = std::dynamic_pointer_cast<DRAMStager>(kv.second)) {
      ds2->set_lease_provider(provider);
    }
  }
}

absl::Status Communicator::init(const std::string& ip, uint16_t port, int conn_count) {
  inited_.store(true);
  if (server_context_->open(ip, port, [this](tcp_transport_t t) { return this->on_new_client(t); }) != SUCCESS) {
    return absl::InternalError("failed to open server " + ip + ":" + std::to_string(port));
  }
  mtcp_conn_count_ = conn_count;
  return absl::OkStatus();
}

uint16_t Communicator::listening_port() const {
  if (!server_context_) {
    return 0;
  }
  return server_context_->listening_port();
}

future_read_result_t Communicator::read_tensor(
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
    return transport::ReadRequest::get_read_result_future("failed to read tensor through un-initiated engine");
  }
  net_dev_t net_dev = nullptr;
  if (enable_rdma_) {
    net_dev = get_net_dev(dev_type, dev_id);
    if (net_dev == nullptr) {
      return transport::ReadRequest::get_read_result_future("failed to get net dev for the rdma connection");
    }
  } else if (COMMUNICATE_ENGINE_DEV_GPU == dev_type && !gpu_memory_stager_) {
    return transport::ReadRequest::get_read_result_future(
        "failed to read GPU tensor with tcp: GPU stager not initialized");
  }

  LOG(INFO) << "read tensor:"
            << " dst=" << dst_ip << ":" << dst_port << ", key=" << key << " ,offset=" << remote_offset
            << ", net_dev=" << (net_dev == nullptr ? "none" : net_dev->get_name());

  auto local_tensor = store_.get_tensor(key);
  const bool needs_new_tensor = local_tensor == nullptr || local_tensor->get_uint64_addr() != addr ||
      local_tensor->get_bytes() != bytes || local_tensor->get_mem_type() != dev_type ||
      (dev_type == COMMUNICATE_ENGINE_DEV_GPU && local_tensor->get_device_id() != dev_id);

  if (needs_new_tensor) {
    local_tensor = std::make_shared<PartitionTensor>(key, addr, bytes, dev_type, net_dev);
    if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
      local_tensor->set_device_id(dev_id);
    }
    if (enable_rdma_ && net_dev != nullptr) {
      net_dev->reg_async(local_tensor);
    }
    local_tensor->set_read_ready();
    store_.register_tensor(local_tensor);
  }

  auto req = std::make_shared<transport::ReadRequest>(key, dst_ip, dst_port, local_tensor, remote_offset);
  LOG(INFO) << "[read_tensor] Creating request: key=" << key << " dst=" << dst_ip << ":" << dst_port
            << " req_key=" << req->get_key();
  request_queue_.push(req);
  LOG(INFO) << "[read_tensor] Request pushed to queue successfully for key=" << key;
  return req->get_future();
}

absl::Status Communicator::register_tensor_ex(
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

absl::Status Communicator::handle_rdma_read_request(
    const channel_t& channel,
    const tcp_transport_t& control_transport,
    const ProtoReadRequest& request,
    const std::shared_ptr<PartitionTensor>& tensor) {
  if (!enable_rdma_) {
    return absl::FailedPreconditionError("RDMA transport disabled");
  }

  auto flow_state = channel->flow_state();
  if (!flow_state) {
    return absl::InternalError("channel missing flow state");
  }

  tensor->wait_read_ready();
  auto dev = tensor->get_dev();
  if (dev == nullptr) {
    return absl::InternalError("tensor missing RDMA device");
  }

  const bool tensor_on_cpu = tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU;
  std::shared_ptr<MemoryStager> stager =
      tensor_on_cpu ? get_cpu_stager_for_nic(dev->get_name()) : get_gpu_mem_stager_for_id(tensor->get_device_id());
  if (!stager) {
    stager = tensor_on_cpu ? memory_stager_ : gpu_memory_stager_;
  }
  if (!stager) {
    return absl::FailedPreconditionError("no staging backend available for tensor");
  }

  const uint64_t chunk_size = stager->get_chunk_size() > 0 ? stager->get_chunk_size() : request.bytes;
  const uint64_t total_bytes = request.bytes;
  const uint64_t start_offset = request.offset;
  const std::string tensor_key(reinterpret_cast<const char*>(request.tensor_key));
  const std::string request_key = transport::get_request_key(tensor_key, start_offset);
  FlowCreditLedger* ledger_ptr = &flow_state->ledger;
  MrCache* mr_cache_ptr = mr_cache_.get();

  auto stage_fn = [stager, tensor, dev, ledger_ptr, mr_cache_ptr, tensor_key](
                      uint64_t offset, uint32_t bytes, uint32_t /*segment_idx*/) -> absl::StatusOr<StageLease> {
    constexpr int kAccess = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
    auto staged_or = stager->stage(tensor, offset, bytes, MemoryStager::StageMode::kTry);
    if (!staged_or.ok()) {
      return staged_or.status();
    }
    void* host_ptr = *staged_or;
    ibv_mr* staged_mr = nullptr;
    bool deregister_mr = false;

    gsl::not_null<void*> host_ptr_nn{host_ptr};
    if (mr_cache_ptr) {
      staged_mr = mr_cache_ptr->get_or_register(dev->get_pd(), host_ptr_nn, bytes, kAccess);
      if (staged_mr == nullptr) {
        auto release_status = stager->release_staged_buffer(host_ptr_nn);
        if (!release_status.ok()) {
          LOG(WARNING) << "Failed to release staged buffer after MR cache failure: " << release_status;
        }
        return absl::InternalError("failed to register MR via cache");
      }
    } else {
      if (dev->reg_mr(&staged_mr, host_ptr, bytes, kAccess) != SUCCESS) {
        auto release_status = stager->release_staged_buffer(host_ptr_nn);
        if (!release_status.ok()) {
          LOG(WARNING) << "Failed to release staged buffer after MR registration failure: " << release_status;
        }
        return absl::InternalError("failed to register staged MR");
      }
      deregister_mr = true;
    }

    StageLease::Metadata metadata;
    metadata.transport = StageTransport::kRdma;
    metadata.request_key = transport::get_request_key(tensor_key, offset);
    metadata.offset = offset;
    metadata.bytes = bytes;

    return StageLease(stager, ledger_ptr, host_ptr, bytes, staged_mr, deregister_mr, metadata);
  };

  auto session = std::make_shared<RdmaReadSession>();
  session->request = request;
  session->tensor_key = tensor_key;
  session->request_key = request_key;
  session->tensor = tensor;
  session->stager = stager;
  session->dev = dev;
  session->control_transport = control_transport;
  session->window = std::make_unique<StagingWindow>(
      flow_state->ledger, stage_fn, total_bytes, chunk_size, start_offset, flow_state->max_window_segments);

  flow_state->rdma_pending_reads.push_back(session);

  auto status = resume_rdma_reads(channel);
  if (!status.ok()) {
    return status;
  }

  return absl::OkStatus();
}

absl::Status Communicator::resume_rdma_reads(const channel_t& channel) {
  auto flow_state = channel->flow_state();
  if (!flow_state) {
    return absl::OkStatus();
  }
  if (flow_state->rdma_refill_in_progress) {
    return absl::OkStatus();
  }

  flow_state->rdma_refill_in_progress = true;
  absl::Cleanup guard([flow_state]() { flow_state->rdma_refill_in_progress = false; });

  absl::Status first_error = absl::OkStatus();

  while (!flow_state->rdma_pending_reads.empty()) {
    auto session = flow_state->rdma_pending_reads.front();
    auto result = DriveRdmaSession(*flow_state, *session);

    if (!result.status.ok()) {
      if (absl::IsResourceExhausted(result.status) || absl::IsUnavailable(result.status)) {
        if (!result.made_progress && flow_state->rdma_pending_reads.size() > 1) {
          flow_state->rdma_pending_reads.pop_front();
          flow_state->rdma_pending_reads.push_back(std::move(session));
        }
        break;
      }

      LOG(ERROR) << "Failed to service RDMA read request=" << session->request_key << " status=" << result.status;

      auto rsp = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
      auto* payload = rsp->get_payload<ProtoReadFailed>();
      memcpy(payload->tensor_key, session->request.tensor_key, kMaxTensorNameLen);
      payload->offset = session->request.offset;
      payload->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
      misc::result_t send_res = session->control_transport->send(rsp);
      if (send_res != misc::SUCCESS) {
        LOG(WARNING) << "Failed to send READ_FAILED after staging failure: " << send_res;
      }

      flow_state->rdma_pending_reads.pop_front();
      if (first_error.ok()) {
        first_error = result.status;
      }
      continue;
    }

    if (result.completed) {
      flow_state->rdma_pending_reads.pop_front();
      continue;
    }

    if (result.made_progress) {
      // Wait for RDMA ACKs to return credit before continuing.
      break;
    }

    break; // Defensive: no progress and no status; avoid tight loop.
  }

  return first_error;
}

absl::Status Communicator::handle_mtcp_read_request(
    const channel_t& channel,
    const tcp_transport_t& control_transport,
    const ProtoReadRequest& request,
    const std::shared_ptr<PartitionTensor>& tensor) {
  auto flow_state = channel->flow_state();
  if (!flow_state) {
    return absl::InternalError("channel missing flow state");
  }

  auto transport = channel->get_mtcp();
  if (transport == nullptr) {
    transport = std::make_shared<transport::MTcpTransport>(mtcp_conn_count_);
    channel->set_transport(transport);
  }
  transport->set_tcp_tos(config_.transport().tcp_tos());
  if (gpu_memory_stager_) {
    transport->set_gpu_memory_stager(gpu_memory_stager_);
  }
  if (gpu_memory_pool_) {
    transport->set_memory_pool(gpu_memory_pool_);
  }
  if (memory_stager_) {
    transport->set_memory_stager(memory_stager_);
  }

  auto rsp = std::make_shared<EngineMessage>(
      ENGINE_OP_READ_RESPONSE_EX,
      static_cast<uint32_t>(sizeof(ProtoReadResponseExHeader) + sizeof(ProtoReadResponseExSeg)));
  auto* hdr = rsp->get_payload<ProtoReadResponseExHeader>();
  memcpy(hdr->tensor_key, request.tensor_key, kMaxTensorNameLen);
  hdr->transport_type = ENGINE_TRANSPORT_MTCP;
  hdr->staged = 0;
  misc::STRCPY(hdr->nic_name, "");
  hdr->num_segments = 1;
  hdr->window_seq = 0;
  hdr->credit_granted = 0;
  hdr->more_segments = 0;
  auto* s0 =
      reinterpret_cast<ProtoReadResponseExSeg*>(reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
  s0->addr = 0;
  s0->rkey = 0;
  s0->bytes = static_cast<uint32_t>(request.bytes);
  s0->offset = request.offset;

  misc::result_t send_res = control_transport->send(rsp);
  if (send_res != misc::SUCCESS) {
    return absl::InternalError("failed to send READ_RESPONSE_EX for MTCP");
  }

  const std::string tensor_key(reinterpret_cast<const char*>(request.tensor_key));
  const uint64_t total_bytes = request.bytes;
  const uint64_t start_offset = request.offset;
  const std::string request_key = transport::get_request_key(tensor_key, start_offset);

  const bool needs_gpu_staging = tensor->needs_staging() || tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_GPU;
  std::shared_ptr<MemoryStager> stager;
  if (needs_gpu_staging) {
    stager = get_gpu_mem_stager_for_id(tensor->get_device_id());
    if (!stager) {
      stager = gpu_memory_stager_;
    }
  } else {
    stager = memory_stager_;
  }
  if (!stager) {
    return absl::FailedPreconditionError("no staging backend available for MTCP tensor");
  }

  const uint64_t chunk_size = stager->get_chunk_size() > 0 ? stager->get_chunk_size() : total_bytes;

  auto stage_fn = [&](uint64_t offset, uint32_t bytes, uint32_t segment_idx) -> absl::StatusOr<StageLease> {
    auto staged_or = stager->stage(tensor, offset, bytes, MemoryStager::StageMode::kBlocking);
    if (!staged_or.ok()) {
      return staged_or.status();
    }
    void* host_ptr = *staged_or;

    StageLease::Metadata metadata;
    metadata.transport = StageTransport::kMtcp;
    metadata.request_key = transport::get_request_key(tensor_key, offset);
    metadata.offset = offset;
    metadata.bytes = bytes;
    metadata.segment_idx = segment_idx;

    return StageLease(stager, &flow_state->ledger, host_ptr, bytes, /*mr=*/nullptr, /*deregister_mr=*/false, metadata);
  };

  StagingWindow window(
      flow_state->ledger, stage_fn, total_bytes, chunk_size, start_offset, flow_state->max_window_segments);

  while (true) {
    auto window_or = window.stage_next();
    if (!window_or.ok()) {
      if (absl::IsOutOfRange(window_or.status())) {
        break;
      }
      LOG(ERROR) << "Failed to stage MTCP window: " << window_or.status();
      auto fail_msg = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
      auto* payload = fail_msg->get_payload<ProtoReadFailed>();
      memcpy(payload->tensor_key, request.tensor_key, kMaxTensorNameLen);
      payload->offset = request.offset;
      payload->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
      control_transport->send(fail_msg);
      return window_or.status();
    }

    auto staged_window = std::move(window_or).value();
    transport::MTcpTransport::StageSendWindow send_window;
    send_window.request_key = request_key;
    send_window.window_seq = staged_window.window_seq;
    send_window.final_window = !staged_window.more_segments;
    send_window.segments.reserve(staged_window.segments.size());

    LOG(INFO) << "[staging_credit] request=" << request_key << " transport=mtcp window=" << staged_window.window_seq
              << " granted=" << staged_window.granted_credit << " more=" << (staged_window.more_segments ? "yes" : "no")
              << " outstanding=" << flow_state->ledger.outstanding_credit();

    for (auto& segment : staged_window.segments) {
      StageLease lease = std::move(segment.lease);
      StageLease::Metadata metadata = lease.metadata();
      metadata.window_seq = staged_window.window_seq;
      metadata.segment_idx = segment.segment_idx;
      metadata.offset = segment.offset;
      metadata.bytes = segment.bytes;
      lease.set_metadata(metadata);

      StageLeaseKey key{
          .request_key = metadata.request_key,
          .window_seq = metadata.window_seq,
          .segment_idx = metadata.segment_idx,
      };

      flow_state->registry.put(key, lease);

      transport::MTcpTransport::StageSendSegment send_segment;
      send_segment.data = lease.host_ptr();
      send_segment.bytes = metadata.bytes;
      send_segment.metadata = metadata;

      auto flow_state_ref = flow_state;
      send_segment.on_complete =
          [flow_state_ref, key, metadata, lease = std::move(lease)](misc::result_t status) mutable {
            if (flow_state_ref) {
              auto lease_or = flow_state_ref->registry.take(key);
              if (lease_or.ok()) {
                lease_or->release();
              } else {
                VLOG(1) << "[MTCP] StageLease missing during release: key=" << key.request_key
                        << " window=" << key.window_seq << " segment=" << key.segment_idx;
              }
            }
            lease.release();
            if (status != misc::SUCCESS) {
              LOG(WARNING) << "[MTCP] StageLease send failure request=" << metadata.request_key
                           << " window=" << metadata.window_seq << " segment=" << metadata.segment_idx
                           << " status=" << status;
            }
          };

      send_window.segments.push_back(std::move(send_segment));
    }

    transport->enqueue_stage_window(std::move(send_window));
  }

  return absl::OkStatus();
}

absl::Status Communicator::unregister_tensor(const std::string& tensor_key) {
  // Make unregister idempotent: return OK if the key does not exist.
  if (store_.get_tensor(tensor_key) == nullptr) {
    VLOG(1) << "[unregister_tensor] key not found, treating as idempotent OK: " << tensor_key;
    return absl::OkStatus();
  }
  store_.unregister_tensor(tensor_key);
  return absl::OkStatus();
}

misc::result_t Communicator::on_new_client(const tcp_transport_t& t) {
  LOG(INFO) << "[on_new_client] New client connection from " << t->get_remote_url() << " fd=" << t->get_fd();
  auto channel =
      std::make_shared<Channel>(t, enable_rdma_ ? CHANNEL_RDMA : CHANNEL_MTCP, buffers_per_flow_, max_window_segments_);
  channels_.put(t->get_remote_url(), channel);
  t->set_recv_func([this](const tcp_transport_t& t) -> misc::result_t {
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
    return misc::SUCCESS;
  });
  return misc::SUCCESS;
}

absl::StatusOr<channel_t> Communicator::do_create_channel(const std::string& ip, uint16_t port) {
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
  auto channel = std::make_shared<Channel>(
      transport, enable_rdma_ ? CHANNEL_RDMA : CHANNEL_MTCP, buffers_per_flow_, max_window_segments_);

  VLOG(1) << "[Communicator] Control channel connected: local=" << server_context_->get_local_ip() << ":" << port
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

  VLOG(1) << "[Communicator] Channel stored: " << transport->get_remote_url();
  return channel;
}

void Communicator::do_read_request_loop() {
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
    misc::STRNCPY(request->tensor_key, req->tensor_key_, kMaxTensorNameLen);

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

misc::result_t Communicator::on_receive_request(
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

      if (transport->connect(&req->qp_info) == misc::SUCCESS) {
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
      if (transport->listen(ip, &port) == misc::SUCCESS) {
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
          auto status = handle_rdma_read_request(channel, t, *req, tensor);
          if (!status.ok()) {
            LOG(WARNING) << "RDMA read request failed: " << status;
            return misc::FAILED;
          }
        } else {
          auto status = handle_mtcp_read_request(channel, t, *req, tensor);
          if (!status.ok()) {
            LOG(WARNING) << "MTCP read request failed: " << status;
            return misc::FAILED;
          }
        }
      }
      break;
    }
    case ENGINE_OP_RDMA_READ_DONE_EX: {
      auto* hdr = msg->get_payload<ProtoRdmaReadDoneExHeader>();
      const std::string tensor_key = reinterpret_cast<char*>(hdr->tensor_key);
      auto flow_state = channel->flow_state();
      if (!flow_state) {
        LOG(WARNING) << "RDMA_READ_DONE_EX without channel flow state";
        break;
      }
      for (uint32_t i = 0; i < hdr->num_segments; ++i) {
        auto* s = reinterpret_cast<ProtoRdmaReadDoneExSeg*>(
            reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoRdmaReadDoneExHeader) + i * sizeof(ProtoRdmaReadDoneExSeg));
        StageLeaseKey key{
            .request_key = transport::get_request_key(tensor_key, s->offset),
            .window_seq = hdr->window_seq,
            .segment_idx = i,
        };
        auto lease_or = flow_state->registry.take(key);
        if (!lease_or.ok()) {
          LOG(WARNING) << "RDMA_READ_DONE_EX for unknown lease: key=" << key.request_key
                       << " window=" << hdr->window_seq << " segment=" << i;
          continue;
        }
        lease_or->release();
      }
      auto resume_status = resume_rdma_reads(channel);
      if (!resume_status.ok()) {
        LOG(WARNING) << "Failed to resume RDMA staging after ACK: " << resume_status;
      }
      break;
    }
    default:
      LOG(WARNING) << "failed to process request: " << msg->get_op();
      return misc::FAILED;
  }
  return misc::SUCCESS;
}

misc::result_t Communicator::on_receive_response(
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
    // case ENGINE_OP_READ_RESPONSE: (legacy) removed
    case ENGINE_OP_READ_RESPONSE_EX: {
      auto* hdr = msg->get_payload<ProtoReadResponseExHeader>();
      std::string tensor_key = reinterpret_cast<char*>(hdr->tensor_key);
      std::string peer_dev_name = reinterpret_cast<char*>(hdr->nic_name);

      LOG(INFO) << "[on_receive_response] READ_RESPONSE_EX: key=" << tensor_key << " segs=" << hdr->num_segments
                << " transport=" << (hdr->transport_type == ENGINE_TRANSPORT_MTCP ? "MTCP" : "RDMA");

      auto* seg0 = reinterpret_cast<ProtoReadResponseExSeg*>(
          reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader));
      auto req_key = transport::get_request_key(tensor_key, seg0->offset);
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
          misc::STRNCPY(payload->src_dev_name, tensor->get_dev()->get_name(), kMaxDevName);
          misc::STRNCPY(payload->dst_dev_name, peer_dev_name, kMaxDevName);
          COMM_CHECK(t->send(req));
          channel->set_transport(tensor->get_dev()->get_name(), peer_dev_name, transport);
        }
        if (!transport->ready()) {
          LOG(ERROR) << "[on_receive_response] RDMA transport not ready before read_multi: request="
                     << read_request->get_key() << " tensor=" << tensor_key
                     << " local_dev=" << tensor->get_dev()->get_name() << " peer_dev=" << peer_dev_name
                     << " remote=" << t->get_remote_url();
        }
        std::vector<transport::RdmaTransport::RdmaReadSeg> rdma_segs;
        rdma_segs.reserve(hdr->num_segments);
        std::vector<uint64_t> ack_offsets;
        ack_offsets.reserve(hdr->num_segments);
        uint64_t base_off = read_request->remote_offset_;
        for (uint32_t i = 0; i < hdr->num_segments; ++i) {
          auto* s = reinterpret_cast<ProtoReadResponseExSeg*>(
              reinterpret_cast<uint8_t*>(hdr) + sizeof(ProtoReadResponseExHeader) + i * sizeof(ProtoReadResponseExSeg));
          transport::RdmaTransport::RdmaReadSeg seg{};
          seg.remote_addr = s->addr;
          seg.rkey = s->rkey;
          seg.length = s->bytes;
          seg.local_addr = tensor->get_uint64_addr() + (s->offset - base_off);
          seg.window_seq = hdr->window_seq;
          seg.segment_idx = i;
          rdma_segs.emplace_back(seg);
          ack_offsets.emplace_back(s->offset);
        }
        read_request->add_expected_completions(static_cast<int>(rdma_segs.size()));
        if (hdr->staged) {
          auto ctrl = channel->get_control();
          const std::string staged_key = tensor_key;
          read_request->set_ack_sender(
              [ctrl, staged_key](uint32_t window_seq, const std::vector<uint64_t>& offsets, bool final_window) {
                auto ack = std::make_shared<EngineMessage>(
                    ENGINE_OP_RDMA_READ_DONE_EX,
                    static_cast<uint32_t>(
                        sizeof(ProtoRdmaReadDoneExHeader) + offsets.size() * sizeof(ProtoRdmaReadDoneExSeg)));
                auto* h = ack->get_payload<ProtoRdmaReadDoneExHeader>();
                misc::STRNCPY(h->tensor_key, staged_key, kMaxTensorNameLen);
                h->num_segments = static_cast<uint32_t>(offsets.size());
                h->window_seq = window_seq;
                h->final_window = final_window ? 1 : 0;
                for (size_t i = 0; i < offsets.size(); ++i) {
                  auto* seg_ack = reinterpret_cast<ProtoRdmaReadDoneExSeg*>(
                      reinterpret_cast<uint8_t*>(h) + sizeof(ProtoRdmaReadDoneExHeader) +
                      i * sizeof(ProtoRdmaReadDoneExSeg));
                  seg_ack->offset = offsets[i];
                }
                CHECK_WARN(ctrl->send(ack), "ack send failed");
              });

          read_request->enqueue_window_ack(hdr->window_seq, std::move(ack_offsets), hdr->more_segments == 0);
        }

        CHECK_WARN(transport->read_multi(read_request, rdma_segs), "failed to read (multi)");
      } else if (hdr->transport_type == ENGINE_TRANSPORT_MTCP) {
        // MTCP path using EX header (no segments)
        auto transport = channel->get_mtcp();
        if (transport == nullptr) {
          transport = std::make_shared<transport::MTcpTransport>(base::kMTcpConnCount);
          LOG(INFO) << "[on_receive_response] Sending MTCP_CONNECT_REQUEST for " << tensor_key;
          auto req = EngineMessage::make_message<ProtoMtcpConnectRequest>(ENGINE_OP_MTCP_CONNECT_REQUEST);
          auto* payload = req->get_payload<ProtoMtcpConnectRequest>();
          payload->conn_count = base::kMTcpConnCount;
          COMM_CHECK(t->send(req));
          channel->set_transport(transport);
        }
        transport->set_tcp_tos(config_.transport().tcp_tos());
        if (gpu_memory_pool_)
          transport->set_memory_pool(gpu_memory_pool_);
        if (memory_stager_)
          transport->set_memory_stager(memory_stager_);
        CHECK_WARN(transport->recv(read_request), "failed to recv via mtcp");
        // Remove pending entry now; completion is tracked in request future
        pending_requests_.del(req_key);
      } else {
        LOG(ERROR) << "[on_receive_response] READ_RESPONSE_EX unsupported transport type";
        read_request->set_result(absl::InternalError("READ_RESPONSE_EX unsupported transport type"));
        pending_requests_.del(req_key);
      }
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
      auto req_key = transport::get_request_key(tensor_key, rsp->offset);

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

  return misc::SUCCESS;
}

net_dev_t Communicator::get_net_dev(int dev_type, int dev_id) {
  net_dev_t net_dev(nullptr);
  if (enable_rdma_) {
    CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";
    if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
      net_dev = rdma_context_->get_best_dev(dev_id);
    } else {
      // CPU: choose the first available RDMA device when not specified via policy/mapping.
      const auto& devs = rdma_context_->list_devs();
      if (!devs.empty())
        net_dev = devs.front();
    }
    if (net_dev == nullptr) {
      LOG(WARNING) << "failed to select RDMA device (dev_type=" << dev_type
                   << ") — ensure CommunicatorConfig specifies device mapping";
      return nullptr;
    }
  }
  return net_dev;
}

absl::Status Communicator::close_connection(const std::string& dst_ip, uint16_t dst_port) {
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

void Communicator::do_channel_gc_loop() {
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

    const uint64_t ttl_ms = ack_ttl_ms_ ? ack_ttl_ms_ : 30000;
    if (ttl_ms > 0) {
      auto channels_copy = channels_.pairs();
      const absl::Duration ttl = absl::Milliseconds(static_cast<int64_t>(ttl_ms));
      for (auto& entry : channels_copy) {
        auto flow_state = entry.second->flow_state();
        if (!flow_state) {
          continue;
        }
        auto expired = flow_state->registry.snapshot_expired(ttl);
        bool resumed = false;
        for (const auto& stale : expired) {
          auto lease_or = flow_state->registry.take(stale.key);
          if (!lease_or.ok()) {
            continue;
          }
          auto metadata = lease_or->metadata();
          LOG(WARNING) << "[staging_credit] Reaping lease request=" << metadata.request_key
                       << " window=" << metadata.window_seq << " segment=" << metadata.segment_idx
                       << " bytes=" << metadata.bytes;
          lease_or->release();
          resumed = true;
        }
        if (resumed) {
          auto resume_status = resume_rdma_reads(entry.second);
          if (!resume_status.ok()) {
            LOG(WARNING) << "Failed to resume RDMA staging after lease reap: " << resume_status;
          }
        }
      }
    }
  }
}

std::shared_ptr<MemoryStager> Communicator::get_cpu_stager_for_nic(const std::string& nic_name) const {
  auto it = nic_cpu_stagers_.find(nic_name);
  if (it != nic_cpu_stagers_.end())
    return it->second;
  return nullptr;
}

std::shared_ptr<MemoryStager> Communicator::get_gpu_mem_stager_for_id(int gpu_id) const {
  auto it = gpu_mem_stagers_.find(gpu_id);
  if (it != gpu_mem_stagers_.end())
    return it->second;
  return nullptr;
}

} // namespace tensorcast::communicator::engine
