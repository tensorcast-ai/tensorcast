
// Copyright (c) 2025-2026, TensorCast Team.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

#include "core/common/system_capabilities.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/engine/gpu_vram_rdma_stager.h"
#include "core/communicator/engine/host_pinned_cpu_stager.h"
#include "core/communicator/engine/host_pinned_gpu_stager.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/protocol.h"
#include "core/communicator/engine/staging_flow_controller.h"
#include "core/communicator/misc/utils.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/cuda/cuda_api.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

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
  bool zero_copy = false;
};

struct Communicator::GpuChannelLease {
  explicit GpuChannelLease(Communicator* owner) : owner(owner) {}

  ~GpuChannelLease() {
    if (owner != nullptr) {
      owner->release_gpu_channel_slot();
    }
  }

  Communicator* owner;
};

namespace {

struct RdmaDriveResult {
  absl::Status status = absl::OkStatus();
  bool made_progress = false;
  bool completed = false;
};

// TODO: Is rdma really need this application layer windows control?
enum class DirectFallbackReason {
  kNone = 0,
  kNotGpu,
  kNeedsStaging,
  kMrUnavailable,
  kOutOfRange,
};

const char* DirectFallbackReasonToString(DirectFallbackReason reason) {
  switch (reason) {
    case DirectFallbackReason::kNone:
      return "none";
    case DirectFallbackReason::kNotGpu:
      return "non_gpu";
    case DirectFallbackReason::kNeedsStaging:
      return "requires_staging";
    case DirectFallbackReason::kMrUnavailable:
      return "mr_unavailable";
    case DirectFallbackReason::kOutOfRange:
      return "out_of_range";
  }
  return "unknown";
}

v1::RdmaConfig::StagedRdmaBackend NormalizeStagedBackend(const v1::CommunicatorConfig& config) {
  auto backend = config.rdma().staging_backend();
  if (backend == v1::RdmaConfig::STAGED_RDMA_BACKEND_UNSPECIFIED) {
    return v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED;
  }
  return backend;
}

const char* StagedBackendToString(v1::RdmaConfig::StagedRdmaBackend backend) {
  switch (backend) {
    case v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED:
      return "host_pinned";
    case v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM:
      return "gpu_vram";
    case v1::RdmaConfig::STAGED_RDMA_BACKEND_UNSPECIFIED:
    case v1::RdmaConfig_StagedRdmaBackend_RdmaConfig_StagedRdmaBackend_INT_MIN_SENTINEL_DO_NOT_USE_:
    case v1::RdmaConfig_StagedRdmaBackend_RdmaConfig_StagedRdmaBackend_INT_MAX_SENTINEL_DO_NOT_USE_:
      break;
  }
  return "host_pinned";
}

void record_staged_backend_metrics(v1::RdmaConfig::StagedRdmaBackend backend, bool tensor_on_cpu, uint64_t bytes);
void record_mr_register_metrics(const char* location, const char* reason, bool success);

} // namespace

StagingWindow::StageFn MakeStageFunction(
    const std::shared_ptr<PartitionTensor>& tensor,
    FlowCreditLedger* ledger,
    const std::shared_ptr<MemoryStager>& stager,
    const net_dev_t& dev,
    MrCache* mr_cache,
    std::string tensor_key,
    v1::RdmaConfig::StagedRdmaBackend staged_backend,
    bool use_direct,
    ibv_mr* direct_mr) {
  if (use_direct) {
    const uint64_t base_addr = tensor->get_uint64_addr();
    const uint64_t tensor_bytes = tensor->get_bytes();
    return [tensor, ledger, tensor_key = std::move(tensor_key), base_addr, tensor_bytes, direct_mr](
               uint64_t offset, uint32_t bytes, uint32_t /*segment_idx*/) -> absl::StatusOr<StageLease> {
      if (offset + bytes > tensor_bytes) {
        return absl::OutOfRangeError("direct RDMA stage exceeded tensor bounds");
      }
      void* device_ptr = reinterpret_cast<void*>(base_addr + offset);
      StageLease::Metadata metadata;
      metadata.transport = StageTransport::kRdma;
      metadata.request_key = transport::get_request_key(tensor_key, offset);
      metadata.offset = offset;
      metadata.bytes = bytes;
      metadata.zero_copy = true;
      return StageLease(
          /*stager=*/nullptr,
          ledger,
          device_ptr,
          bytes,
          direct_mr,
          /*deregister_mr=*/false,
          metadata);
    };
  }

  if (!stager) {
    return [tensor_key = std::move(tensor_key)](
               uint64_t /*offset*/, uint32_t /*bytes*/, uint32_t /*segment_idx*/) -> absl::StatusOr<StageLease> {
      return absl::FailedPreconditionError(absl::StrCat("no staging backend available for tensor=", tensor_key));
    };
  }

  auto fallback_stager = stager;
  return [fallback_stager, tensor, dev, ledger, mr_cache, tensor_key = std::move(tensor_key), staged_backend](
             uint64_t offset, uint32_t bytes, uint32_t /*segment_idx*/) -> absl::StatusOr<StageLease> {
    constexpr int kAccess = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
    auto staged_or = fallback_stager->stage(tensor, offset, bytes, MemoryStager::StageMode::kTry);
    if (!staged_or.ok()) {
      return staged_or.status();
    }
    void* exposed_ptr = *staged_or;
    ibv_mr* staged_mr = nullptr;
    bool deregister_mr = false;

    gsl::not_null<void*> exposed_ptr_nn{exposed_ptr};
    const char* mr_location = StagedBackendToString(staged_backend);
    if (mr_cache) {
      const auto slab = NormalizeMrRegion(*fallback_stager, exposed_ptr_nn, bytes);
      gsl::not_null<void*> mr_base = slab.base;
      size_t mr_bytes = slab.bytes;

      auto mr_result = mr_cache->get_or_register(dev->get_pd(), mr_base, mr_bytes, kAccess);
      staged_mr = mr_result.mr;
      if (staged_mr == nullptr) {
        record_mr_register_metrics(mr_location, "cache_register_failed", false);
        auto release_status = fallback_stager->release_staged_buffer(exposed_ptr_nn);
        if (!release_status.ok()) {
          LOG(WARNING) << "Failed to release staged buffer after MR cache failure: " << release_status;
        }
        return absl::InternalError("failed to register MR via cache");
      }
      if (mr_result.registered) {
        record_mr_register_metrics(mr_location, nullptr, true);
      }
    } else {
      if (dev->reg_mr(&staged_mr, exposed_ptr, bytes, kAccess) != SUCCESS) {
        record_mr_register_metrics(mr_location, "direct_register_failed", false);
        auto release_status = fallback_stager->release_staged_buffer(exposed_ptr_nn);
        if (!release_status.ok()) {
          LOG(WARNING) << "Failed to release staged buffer after MR registration failure: " << release_status;
        }
        return absl::InternalError("failed to register staged MR");
      }
      record_mr_register_metrics(mr_location, nullptr, true);
      deregister_mr = true;
    }

    record_staged_backend_metrics(staged_backend, tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU, bytes);

    StageLease::Metadata metadata;
    metadata.transport = StageTransport::kRdma;
    metadata.request_key = transport::get_request_key(tensor_key, offset);
    metadata.offset = offset;
    metadata.bytes = bytes;
    metadata.zero_copy = false;

    return StageLease(fallback_stager, ledger, exposed_ptr, bytes, staged_mr, deregister_mr, metadata);
  };
}

namespace {

opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> g_comm_meter;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_direct_segments_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_direct_bytes_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_direct_fallback_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> g_rdma_direct_window_bytes_hist;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_staged_backend_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_staged_bytes_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_mr_register_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_rdma_mr_register_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_pinned_rdma_prereg_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> g_pinned_rdma_prereg_latency_ms_hist;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> g_pinned_rdma_prereg_bytes_gauge;
std::atomic<double> g_pinned_rdma_prereg_bytes_last{0.0};
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_vram_rdma_prereg_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> g_vram_rdma_prereg_latency_ms_hist;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> g_vram_rdma_prereg_bytes_gauge;
std::atomic<double> g_vram_rdma_prereg_bytes_last{0.0};

absl::once_flag g_comm_meter_once;
absl::once_flag g_rdma_direct_window_metrics_once;
absl::once_flag g_rdma_direct_fallback_metric_once;
absl::once_flag g_rdma_staged_metrics_once;
absl::once_flag g_rdma_mr_register_metrics_once;
absl::once_flag g_pinned_rdma_prereg_metrics_once;
absl::once_flag g_vram_rdma_prereg_metrics_once;

inline void ensure_communicator_meter() {
  absl::call_once(g_comm_meter_once, []() {
    g_comm_meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.communicator", "1.0.0");
  });
}

inline void ensure_rdma_direct_window_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_direct_window_metrics_once, []() {
    g_rdma_direct_segments_total = g_comm_meter->CreateDoubleCounter("tc_rdma_direct_segments_total");
    g_rdma_direct_bytes_total = g_comm_meter->CreateDoubleCounter("tc_rdma_direct_bytes_total");
    g_rdma_direct_window_bytes_hist = g_comm_meter->CreateDoubleHistogram("tc_rdma_direct_window_bytes");
  });
}

inline void ensure_rdma_direct_fallback_metric() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_direct_fallback_metric_once, []() {
    g_rdma_direct_fallback_total = g_comm_meter->CreateDoubleCounter("tc_rdma_direct_fallback_total");
  });
}

inline void ensure_rdma_staged_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_staged_metrics_once, []() {
    g_rdma_staged_backend_total = g_comm_meter->CreateDoubleCounter("tc_rdma_staged_backend_total");
    g_rdma_staged_bytes_total = g_comm_meter->CreateDoubleCounter("tc_rdma_staged_bytes_total");
  });
}

inline void ensure_rdma_mr_register_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_rdma_mr_register_metrics_once, []() {
    g_rdma_mr_register_total = g_comm_meter->CreateDoubleCounter("tc_rdma_mr_register_total");
    g_rdma_mr_register_failures_total = g_comm_meter->CreateDoubleCounter("tc_rdma_mr_register_failures_total");
  });
}

void pinned_rdma_prereg_bytes_callback(opentelemetry::metrics::ObserverResult result, void* /*state*/) noexcept {
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  obs->Observe(g_pinned_rdma_prereg_bytes_last.load());
}

inline void ensure_pinned_rdma_prereg_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_pinned_rdma_prereg_metrics_once, []() {
    g_pinned_rdma_prereg_failures_total = g_comm_meter->CreateDoubleCounter("tc_pinned_rdma_prereg_failures_total");
    g_pinned_rdma_prereg_latency_ms_hist = g_comm_meter->CreateDoubleHistogram("tc_pinned_rdma_prereg_latency_ms");
    g_pinned_rdma_prereg_bytes_gauge = g_comm_meter->CreateDoubleObservableGauge("tc_pinned_rdma_prereg_bytes");
    g_pinned_rdma_prereg_bytes_gauge->AddCallback(&pinned_rdma_prereg_bytes_callback, nullptr);
  });
}

void vram_rdma_prereg_bytes_callback(opentelemetry::metrics::ObserverResult result, void* /*state*/) noexcept {
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  obs->Observe(g_vram_rdma_prereg_bytes_last.load());
}

inline void ensure_vram_rdma_prereg_metrics() {
  ensure_communicator_meter();
  absl::call_once(g_vram_rdma_prereg_metrics_once, []() {
    g_vram_rdma_prereg_failures_total = g_comm_meter->CreateDoubleCounter("tc_vram_rdma_prereg_failures_total");
    g_vram_rdma_prereg_latency_ms_hist = g_comm_meter->CreateDoubleHistogram("tc_vram_rdma_prereg_latency_ms");
    g_vram_rdma_prereg_bytes_gauge = g_comm_meter->CreateDoubleObservableGauge("tc_vram_rdma_prereg_bytes");
    g_vram_rdma_prereg_bytes_gauge->AddCallback(&vram_rdma_prereg_bytes_callback, nullptr);
  });
}

void record_direct_window_metrics(int device_id, uint64_t segments, uint64_t bytes) {
  if (segments == 0 && bytes == 0) {
    return;
  }
  ensure_rdma_direct_window_metrics();
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("device_id", opentelemetry::common::AttributeValue(device_id));
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  auto ctx = opentelemetry::context::Context{};
  if (segments > 0) {
    g_rdma_direct_segments_total->Add(static_cast<double>(segments), attr_view, ctx);
  }
  if (bytes > 0) {
    const double bytes_double = static_cast<double>(bytes);
    g_rdma_direct_bytes_total->Add(bytes_double, attr_view, ctx);
    g_rdma_direct_window_bytes_hist->Record(bytes_double, attr_view, ctx);
  }
}

void record_staged_backend_metrics(v1::RdmaConfig::StagedRdmaBackend backend, bool tensor_on_cpu, uint64_t bytes) {
  ensure_rdma_staged_metrics();
  if (!g_rdma_staged_backend_total || !g_rdma_staged_bytes_total) {
    return;
  }
  const char* backend_label = StagedBackendToString(backend);
  const char* mem_label = tensor_on_cpu ? "cpu" : "gpu";
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("backend", backend_label);
  attrs.emplace("mem", mem_label);
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  g_rdma_staged_backend_total->Add(1.0, attr_view, opentelemetry::context::Context{});
  g_rdma_staged_bytes_total->Add(static_cast<double>(bytes), attr_view, opentelemetry::context::Context{});
}

void record_mr_register_metrics(const char* location, const char* reason, bool success) {
  ensure_rdma_mr_register_metrics();
  if (!g_rdma_mr_register_total || !g_rdma_mr_register_failures_total) {
    return;
  }
  if (success) {
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("method", "ibv_reg_mr");
    attrs.emplace("location", location);
    auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
    g_rdma_mr_register_total->Add(1.0, attr_view, opentelemetry::context::Context{});
    return;
  }
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("method", "ibv_reg_mr");
  attrs.emplace("reason", reason);
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  g_rdma_mr_register_failures_total->Add(1.0, attr_view, opentelemetry::context::Context{});
}

void record_direct_fallback_metric(DirectFallbackReason reason) {
  if (reason == DirectFallbackReason::kNone) {
    return;
  }
  ensure_rdma_direct_fallback_metric();
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("reason", opentelemetry::common::AttributeValue(DirectFallbackReasonToString(reason)));
  auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
  g_rdma_direct_fallback_total->Add(1.0, attr_view, opentelemetry::context::Context{});
}

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

    uint64_t zero_copy_segments = 0;
    uint64_t zero_copy_bytes = 0;
    for (const auto& segment : staged_window.segments) {
      if (segment.lease.metadata().zero_copy) {
        ++zero_copy_segments;
        zero_copy_bytes += segment.bytes;
      }
    }
    if (zero_copy_segments > 0) {
      const int device_id = session.tensor ? session.tensor->get_device_id() : -1;
      record_direct_window_metrics(device_id, zero_copy_segments, zero_copy_bytes);
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
    hdr->rail_id = session.dev ? session.dev->get_rail_id() : 0;
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

      seg_pl->addr = reinterpret_cast<uint64_t>(segment.lease.exposed_ptr());
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

absl::Duration compute_handshake_backoff(int failure_count) {
  if (failure_count <= 0) {
    return absl::Milliseconds(50);
  }
  const int capped = std::min(failure_count, 5);
  return absl::Milliseconds(50 * (1 << capped));
}

std::vector<Channel::PendingRdmaRead> drain_pending_reads_for_generation(
    const std::shared_ptr<Channel::RdmaEndpoint>& endpoint,
    uint64_t generation) {
  std::vector<Channel::PendingRdmaRead> drained;
  absl::MutexLock lock(&endpoint->mu);
  for (auto it = endpoint->pending_reads.begin(); it != endpoint->pending_reads.end();) {
    if (generation == 0 || it->generation == generation) {
      drained.push_back(std::move(*it));
      it = endpoint->pending_reads.erase(it);
    } else {
      ++it;
    }
  }
  return drained;
}

void log_handshake_transition(
    const std::string& local_dev,
    const std::string& peer_dev,
    Channel::HandshakeState from,
    Channel::HandshakeState to,
    uint64_t generation,
    size_t queue_depth) {
  auto state_to_string = [](Channel::HandshakeState state) {
    switch (state) {
      case Channel::HandshakeState::kIdle:
        return "idle";
      case Channel::HandshakeState::kConnectRequested:
        return "connecting";
      case Channel::HandshakeState::kReady:
        return "ready";
      case Channel::HandshakeState::kFailed:
        return "failed";
    }
    return "unknown";
  };

  LOG(INFO) << "[rdma_handshake] dev=" << local_dev << " peer=" << peer_dev << " state=" << state_to_string(from)
            << " -> " << state_to_string(to) << " gen=" << generation << " pending=" << queue_depth;
}

} // namespace

Communicator::Communicator(const v1::CommunicatorConfig& config, uint32_t channel_expire_sec)
    : Communicator(
          config,
          [&config]() {
            constexpr size_t kDefaultGpuSliceBytes = 16ULL * 1024 * 1024;
            constexpr size_t kDefaultCpuSliceBytes = 4ULL * 1024 * 1024;
            PinnedStagingPools pools;
            const size_t num_buffers = static_cast<size_t>(std::max(1, config.stager().buffers_per_flow()));
            int conn = config.transport().tcp_conn_count();
            if (conn <= 0) {
              conn = base::kMTcpConnCount;
            }
            const size_t conn_count = static_cast<size_t>(std::max(2, conn));
            const size_t required_gpu_slices = num_buffers + (num_buffers * conn_count);
            pools.gpu_pool = std::make_shared<common::memory::PinnedBufferPool>(
                required_gpu_slices * kDefaultGpuSliceBytes, kDefaultGpuSliceBytes);
            pools.cpu_pool = std::make_shared<common::memory::PinnedBufferPool>(
                num_buffers * kDefaultCpuSliceBytes, kDefaultCpuSliceBytes);
            pools.preregister_gpu = config.enable_rdma();
            pools.preregister_cpu = config.enable_rdma();
            return pools;
          }(),
          channel_expire_sec) {}

Communicator::Communicator(const v1::CommunicatorConfig& config, PinnedStagingPools pools, uint32_t channel_expire_sec)
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
  bool so_reuseport_enabled = true;
  if (config_.transport().has_so_reuseport()) {
    so_reuseport_enabled = config_.transport().so_reuseport();
  }
  server_context_->set_so_reuseport(so_reuseport_enabled);
  client_context_->set_so_reuseport(so_reuseport_enabled);

  // No default residency provider required; staging policy no longer consults UMA bridges.

  // Staging resources are sized from the pinned-memory authority. The communicator config
  // controls fan-out and transport behavior only.
  if (!pools.gpu_pool) {
    LOG(FATAL) << "Communicator requires a non-null pinned gpu_pool";
  }
  gpu_memory_pool_ = std::move(pools.gpu_pool);
  cpu_memory_pool_ = pools.cpu_pool ? std::move(pools.cpu_pool) : gpu_memory_pool_;
  preregister_gpu_pool_ = pools.preregister_gpu;
  preregister_cpu_pool_ = pools.preregister_cpu;
  staging_wait_timeout_ = pools.staging_wait_timeout;
  if (staging_wait_timeout_ <= absl::ZeroDuration()) {
    LOG(WARNING) << "Communicator staging_wait_timeout is <= 0; defaulting to 30s";
    staging_wait_timeout_ = absl::Seconds(30);
  }
  const auto staging_backend = NormalizeStagedBackend(config_);
  if (staging_backend == v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM && !enable_rdma_) {
    LOG(FATAL) << "rdma.staging_backend=STAGED_RDMA_BACKEND_GPU_VRAM requires enable_rdma=true";
  }
  use_gpu_vram_staging_ = enable_rdma_ && staging_backend == v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM;
  if (use_gpu_vram_staging_) {
    const uint64_t pool_bytes = config_.rdma().vram_pool_bytes_per_gpu();
    const uint64_t slice_bytes = config_.rdma().vram_slice_bytes();
    if (pool_bytes == 0 || slice_bytes == 0) {
      LOG(FATAL) << "rdma.vram_pool_bytes_per_gpu and rdma.vram_slice_bytes must be > 0 when "
                    "STAGED_RDMA_BACKEND_GPU_VRAM staging is set";
    }
    int device_count = 0;
    auto count_status = cuda::get_device_count(&device_count);
    if (!count_status.ok()) {
      LOG(FATAL) << "Failed to query CUDA device count for GPU VRAM staging: " << count_status;
    }
    if (device_count <= 0) {
      LOG(FATAL) << "GPU VRAM staging requires at least one CUDA device";
    }
    for (int device_id = 0; device_id < device_count; ++device_id) {
      auto pool = std::make_shared<GpuVramStagingPool>(
          device_id, static_cast<size_t>(pool_bytes), static_cast<size_t>(slice_bytes));
      auto init_status = pool->initialize();
      if (!init_status.ok()) {
        LOG(FATAL) << "Failed to initialize GPU VRAM staging pool for device=" << device_id << ": " << init_status;
      }
      gpu_vram_pools_[device_id] = pool;
      gpu_vram_stagers_[device_id] = std::make_shared<GpuVramRdmaStager>(pool);
    }
  }

  if (config_.stager().buffers_per_flow() <= 0) {
    LOG(FATAL) << "stager.buffers_per_flow must be > 0";
  }
  const size_t gpu_chunk_size = gpu_memory_pool_->slice_bytes();
  const size_t cpu_chunk_size = cpu_memory_pool_->slice_bytes();
  if (gpu_chunk_size == 0 || cpu_chunk_size == 0) {
    LOG(FATAL) << "Pinned staging pool slice_bytes must be > 0";
  }

  const size_t num_buffers = static_cast<size_t>(config_.stager().buffers_per_flow());
  buffers_per_flow_ = static_cast<int>(num_buffers);
  direct_rdma_chunk_bytes_ = gpu_chunk_size;
  if (direct_rdma_chunk_bytes_ == 0) {
    LOG(FATAL) << "direct_rdma_chunk_bytes must be > 0";
  }
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
  int configured_conn = config_.transport().tcp_conn_count();
  if (configured_conn <= 0) {
    configured_conn = base::kMTcpConnCount;
  }
  configured_conn = std::max(2, configured_conn);
  mtcp_conn_count_ = configured_conn;
  const auto mtcp_conn_budget = static_cast<size_t>(configured_conn);
  const size_t recv_num_buffers = num_buffers * mtcp_conn_budget;
  const size_t computed_pool_buffers = num_buffers + recv_num_buffers;
  const size_t required_gpu_slices = computed_pool_buffers;
  const size_t capacity_gpu_slices = gpu_memory_pool_->capacity_slices();
  if (capacity_gpu_slices < required_gpu_slices) {
    LOG(FATAL) << "Pinned staging pool (comm_gpu) too small: capacity_slices=" << capacity_gpu_slices
               << " required_slices=" << required_gpu_slices << " slice_bytes=" << gpu_chunk_size
               << " (buffers_per_flow=" << num_buffers << " tcp_conn_count=" << config_.transport().tcp_conn_count()
               << ")";
  }
  gpu_memory_stager_ = std::make_shared<HostPinnedGpuStager>(gpu_chunk_size, num_buffers, gpu_memory_pool_);

  const size_t total_gpu_slices = capacity_gpu_slices;
  if (total_gpu_slices == 0) {
    LOG(FATAL) << "GPU pinned buffer pool initialized with zero slices";
  }
  const size_t stager_reserve = num_buffers;
  size_t available_gpu_slices = 0;
  if (total_gpu_slices > stager_reserve) {
    available_gpu_slices = total_gpu_slices - stager_reserve;
  }
  size_t computed_gpu_channel_limit = 0;
  if (buffers_per_flow_ > 0) {
    computed_gpu_channel_limit = available_gpu_slices / static_cast<size_t>(buffers_per_flow_);
  }

  const uint32_t configured_gpu_channels = config_.stager().expected_gpu_channels();
  if (configured_gpu_channels > 0) {
    if (configured_gpu_channels > computed_gpu_channel_limit) {
      LOG(FATAL) << "expected_gpu_channels=" << configured_gpu_channels
                 << " exceeds staging capacity. gpu_pool_slices=" << total_gpu_slices << " reserve=" << stager_reserve
                 << " buffers_per_flow=" << buffers_per_flow_ << " computed_limit=" << computed_gpu_channel_limit
                 << ". Increase pinned_memory.classes[name=comm_gpu].pool_bytes or reduce expected_gpu_channels.";
    }
    max_gpu_channels_ = static_cast<int>(configured_gpu_channels);
    enforce_gpu_channel_limit_ = max_gpu_channels_ > 0;
  } else {
    max_gpu_channels_ = static_cast<int>(computed_gpu_channel_limit);
    enforce_gpu_channel_limit_ = max_gpu_channels_ > 0;
  }

  if (enforce_gpu_channel_limit_) {
    LOG(INFO) << "[Communicator] GPU staging pool allows up to " << max_gpu_channels_
              << " concurrent MTCP transports before additional channels are rejected";
  } else {
    LOG(WARNING) << "[Communicator] GPU staging pool has insufficient headroom to compute MTCP channel limit;"
                 << " concurrent GPU reads may block waiting for staging buffers.";
  }

  const size_t required_cpu_slices = num_buffers;
  if (cpu_memory_pool_ && cpu_memory_pool_.get() != gpu_memory_pool_.get()) {
    const size_t available_cpu_slices = cpu_memory_pool_->capacity_slices();
    if (available_cpu_slices < required_cpu_slices) {
      LOG(FATAL) << "Pinned staging pool (comm_cpu) too small: capacity_slices=" << available_cpu_slices
                 << " required_slices=" << required_cpu_slices << " slice_bytes=" << cpu_chunk_size
                 << " (buffers_per_flow=" << num_buffers << ")";
    }
  }
  auto dram_pool = cpu_memory_pool_ ? cpu_memory_pool_ : gpu_memory_pool_;
  memory_stager_ = std::make_shared<HostPinnedCpuStager>(
      gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{dram_pool}, /*num_buffers_hint=*/num_buffers);
  if (auto ds = std::dynamic_pointer_cast<HostPinnedCpuStager>(memory_stager_)) {
    ds->set_lease_provider(HostPinnedCpuStager::make_noop_lease_provider());
  }

  if (enable_rdma_) {
    rdma_context_ = std::make_shared<RdmaContext>();
    meta_mr_cache_ = std::make_unique<MrCache>();
    // Apply typed RDMA QP tuning
    rdma_context_->set_qp_params(
        config_.rdma().traffic_class(), config_.rdma().qp_timeout(), config_.rdma().qp_retry());
    // Apply multi-QP configuration
    rdma_context_->set_multi_qp_config(config_.rdma().qp_count(), config_.rdma().bonding_balance());
    int access = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;

    if (config_.simple_numa().enable()) {
      for (const auto& node : config_.simple_numa().nodes()) {
        auto cpu_stager = std::make_shared<HostPinnedCpuStager>(
            gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{dram_pool},
            /*num_buffers_hint=*/num_buffers);
        if (auto ds = std::dynamic_pointer_cast<HostPinnedCpuStager>(cpu_stager)) {
          ds->set_lease_provider(HostPinnedCpuStager::make_noop_lease_provider());
        }
        auto gpu_mem_stager = std::make_shared<HostPinnedGpuStager>(gpu_chunk_size, num_buffers, gpu_memory_pool_);
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

    if (use_gpu_vram_staging_) {
      ensure_vram_rdma_prereg_metrics();
      const auto prereg_start = std::chrono::steady_clock::now();
      uint64_t prereg_bytes = 0;
      for (const auto& entry : gpu_vram_pools_) {
        prereg_bytes += entry.second->pool_bytes();
      }
      g_vram_rdma_prereg_bytes_last.store(static_cast<double>(prereg_bytes));

      uint64_t failures = 0;
      for (const auto& dev : rdma_context_->list_devs()) {
        for (const auto& entry : gpu_vram_pools_) {
          auto slab = entry.second->mr_slab();
          if (!slab.has_value()) {
            ++failures;
            LOG(WARNING) << "Failed to preregister VRAM MR for device=" << entry.first << ": missing slab";
            record_mr_register_metrics("gpu_vram", "preregister_failed", false);
            continue;
          }
          auto result = meta_mr_cache_->get_or_register(dev->get_pd(), slab->base, slab->bytes, access);
          if (result.mr == nullptr) {
            ++failures;
            LOG(WARNING) << "Failed to preregister VRAM MR for slab " << static_cast<void*>(slab->base.get())
                         << " bytes=" << slab->bytes << " on PD";
            record_mr_register_metrics("gpu_vram", "preregister_failed", false);
            continue;
          }
          if (result.registered) {
            record_mr_register_metrics("gpu_vram", nullptr, true);
          }
        }
      }
      const auto prereg_end = std::chrono::steady_clock::now();
      const double prereg_ms =
          std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(prereg_end - prereg_start).count();
      if (g_vram_rdma_prereg_latency_ms_hist) {
        std::map<std::string, opentelemetry::common::AttributeValue> attrs;
        auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
        g_vram_rdma_prereg_latency_ms_hist->Record(prereg_ms, attr_view, opentelemetry::context::Context{});
      }
      if (failures > 0 && g_vram_rdma_prereg_failures_total) {
        std::map<std::string, opentelemetry::common::AttributeValue> attrs;
        auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
        g_vram_rdma_prereg_failures_total->Add(
            static_cast<double>(failures), attr_view, opentelemetry::context::Context{});
      }
      if (failures > 0) {
        LOG(FATAL) << "GPU VRAM staging preregistration failed for " << failures << " slabs";
      }
    }

    // Preregister MRs for selected pools (slab-level).
    ensure_pinned_rdma_prereg_metrics();
    const auto prereg_start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<common::memory::PinnedBufferPool>> pools;
    if (preregister_gpu_pool_) {
      pools.push_back(gpu_memory_pool_);
    }
    if (preregister_cpu_pool_ && cpu_memory_pool_ && cpu_memory_pool_.get() != gpu_memory_pool_.get()) {
      pools.push_back(cpu_memory_pool_);
    }
    uint64_t prereg_bytes = 0;
    for (const auto& pool : pools) {
      for (const auto& slab : pool->list_slabs()) {
        prereg_bytes += slab.bytes;
      }
    }
    g_pinned_rdma_prereg_bytes_last.store(static_cast<double>(prereg_bytes));

    uint64_t failures = 0;
    for (const auto& dev : rdma_context_->list_devs()) {
      for (auto& pool : pools) {
        for (const auto& slab : pool->list_slabs()) {
          auto result = meta_mr_cache_->get_or_register(dev->get_pd(), slab.base.get(), slab.bytes, access);
          if (result.mr == nullptr) {
            ++failures;
            LOG(WARNING) << "Failed to preregister MR for slab " << static_cast<void*>(slab.base.get())
                         << " bytes=" << slab.bytes << " on PD";
            record_mr_register_metrics("host_pinned", "preregister_failed", false);
          } else if (result.registered) {
            record_mr_register_metrics("host_pinned", nullptr, true);
          }
        }
      }
    }
    const auto prereg_end = std::chrono::steady_clock::now();
    const double prereg_ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(prereg_end - prereg_start).count();
    if (g_pinned_rdma_prereg_latency_ms_hist) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
      g_pinned_rdma_prereg_latency_ms_hist->Record(prereg_ms, attr_view, opentelemetry::context::Context{});
    }
    if (failures > 0 && g_pinned_rdma_prereg_failures_total) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      auto attr_view = opentelemetry::common::KeyValueIterableView(attrs);
      g_pinned_rdma_prereg_failures_total->Add(
          static_cast<double>(failures), attr_view, opentelemetry::context::Context{});
    }

    handshake_retry_thread_ = std::thread([this]() { this->handshake_retry_loop(); });
    handshake_retry_thread_started_ = true;
  }

  mtcp_staging_thread_ = std::thread([this]() { this->mtcp_staging_loop(); });
}

Communicator::~Communicator() {
  store_.clear();
  stop_.store(true);
  request_queue_.stop();
  handshake_retry_stop_.store(true);
  handshake_retry_cv_.SignalAll();
  mtcp_staging_queue_.notify();
  if (handshake_retry_thread_started_ && handshake_retry_thread_.joinable()) {
    handshake_retry_thread_.join();
  }
  if (mtcp_staging_thread_.joinable()) {
    mtcp_staging_thread_.join();
  }
  mtcp_staging_queue_.stop();
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

void Communicator::mtcp_staging_loop() {
  while (!stop_.load()) {
    MtcpReadTask task = mtcp_staging_queue_.pop(true, 1000);
    if (!task.channel) {
      continue;
    }
    process_mtcp_read_task(std::move(task));
  }

  while (true) {
    MtcpReadTask task = mtcp_staging_queue_.pop(false);
    if (!task.channel) {
      break;
    }
    fail_mtcp_read_task(task, absl::CancelledError("communicator shutting down"));
  }
}

void Communicator::fail_mtcp_read_task(const MtcpReadTask& task, absl::Status status) {
  if (status.ok()) {
    return;
  }

  const std::string tensor_key(reinterpret_cast<const char*>(task.request.tensor_key));

  if (task.control_transport) {
    auto fail_msg = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
    auto* payload = fail_msg->get_payload<ProtoReadFailed>();
    memcpy(payload->tensor_key, task.request.tensor_key, kMaxTensorNameLen);
    payload->offset = task.request.offset;
    payload->reason = TENSORCAST_READ_FAILED_MEM_MISMATCH;
    misc::result_t send_res = task.control_transport->send(fail_msg);
    if (send_res != misc::SUCCESS) {
      LOG(WARNING) << "Failed to send READ_FAILED after staging failure: key=" << tensor_key << " res=" << send_res;
    }
  }

  LOG(WARNING) << "MTCP staging task failed for key=" << tensor_key << ": " << status;
}

absl::StatusOr<std::shared_ptr<void>> Communicator::acquire_gpu_channel_slot() {
  if (!enforce_gpu_channel_limit_ || max_gpu_channels_ <= 0) {
    return std::shared_ptr<void>{};
  }

  int current = active_gpu_channels_.load(std::memory_order_relaxed);
  while (true) {
    if (current >= max_gpu_channels_) {
      return absl::ResourceExhaustedError(
          absl::StrFormat(
              "GPU staging pool capacity exceeded: %d active MTCP channels, limit=%d (buffers_per_flow=%d)",
              current,
              max_gpu_channels_,
              buffers_per_flow_));
    }
    if (active_gpu_channels_.compare_exchange_weak(
            current, current + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
      auto lease = std::shared_ptr<GpuChannelLease>(new GpuChannelLease(this));
      return std::static_pointer_cast<void>(lease);
    }
  }
}

void Communicator::release_gpu_channel_slot() {
  int previous = active_gpu_channels_.fetch_sub(1, std::memory_order_acq_rel);
  if (previous <= 0) {
    active_gpu_channels_.store(0, std::memory_order_relaxed);
    LOG(WARNING) << "[Communicator] GPU channel slot release underflow";
  }
}

void Communicator::process_mtcp_read_task(MtcpReadTask task) {
  if (!task.channel || !task.tensor) {
    if (task.channel) {
      task.channel->mtcp_request_finished();
    }
    fail_mtcp_read_task(task, absl::FailedPreconditionError("invalid MTCP staging task"));
    return;
  }

  auto release_called = std::make_shared<std::atomic<bool>>(false);
  auto release_once = [channel = task.channel, release_called]() {
    if (!release_called->exchange(true, std::memory_order_acq_rel)) {
      channel->mtcp_request_finished();
    }
  };
  bool release_handed_off = false;

  auto flow_state = task.channel->flow_state();
  if (!flow_state) {
    release_once();
    release_handed_off = true;
    fail_mtcp_read_task(task, absl::InternalError("channel missing flow state"));
    return;
  }

  auto transport = task.channel->get_mtcp();
  if (transport == nullptr) {
    release_once();
    release_handed_off = true;
    fail_mtcp_read_task(task, absl::InternalError("missing MTCP transport"));
    return;
  }

  const ProtoReadRequest& request = task.request;
  const std::string tensor_key(reinterpret_cast<const char*>(request.tensor_key));
  const uint64_t total_bytes = request.bytes;
  const uint64_t start_offset = request.offset;
  const std::string request_key = transport::get_request_key(tensor_key, start_offset);

  const bool needs_gpu_staging =
      task.tensor->needs_staging() || task.tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_GPU;
  std::shared_ptr<MemoryStager> stager;
  if (needs_gpu_staging) {
    stager = get_gpu_mem_stager_for_id(task.tensor->get_device_id());
    if (!stager) {
      stager = gpu_memory_stager_;
    }
  } else {
    stager = memory_stager_;
  }
  if (!stager) {
    release_once();
    release_handed_off = true;
    fail_mtcp_read_task(task, absl::FailedPreconditionError("no staging backend available for MTCP tensor"));
    return;
  }

  const uint64_t chunk_size = stager->get_chunk_size() > 0 ? stager->get_chunk_size() : total_bytes;

  auto stage_fn = [&](uint64_t offset, uint32_t bytes, uint32_t segment_idx) -> absl::StatusOr<StageLease> {
    // Do not block inside stage(); bounded waiting and retry is handled by the
    // MTCP staging loop via backoff + retry_deadline.
    auto staged_or = stager->stage(task.tensor, offset, bytes, MemoryStager::StageMode::kTry);
    if (!staged_or.ok()) {
      return staged_or.status();
    }
    void* exposed_ptr = *staged_or;

    StageLease::Metadata metadata;
    metadata.transport = StageTransport::kMtcp;
    metadata.request_key = transport::get_request_key(tensor_key, offset);
    metadata.offset = offset;
    metadata.bytes = bytes;
    metadata.segment_idx = segment_idx;

    return StageLease(
        stager,
        &flow_state->ledger,
        exposed_ptr,
        bytes,
        /*mr=*/nullptr,
        /*deregister_mr=*/false,
        metadata);
  };

  StagingWindow window(
      flow_state->ledger, stage_fn, total_bytes, chunk_size, start_offset, flow_state->max_window_segments);

  absl::Time retry_deadline = absl::Now() + staging_wait_timeout_;
  absl::Duration backoff = absl::Milliseconds(1);
  constexpr absl::Duration kMaxBackoff = absl::Milliseconds(50);
  absl::Time last_warning = absl::InfinitePast();

  while (true) {
    auto window_or = window.stage_next();
    if (!window_or.ok()) {
      if (absl::IsOutOfRange(window_or.status())) {
        break;
      }

      if (absl::IsUnavailable(window_or.status()) || absl::IsResourceExhausted(window_or.status())) {
        const absl::Time now = absl::Now();
        if (last_warning == absl::InfinitePast() || now - last_warning >= absl::Seconds(1)) {
          LOG(WARNING) << "[staging_credit] request=" << request_key
                       << " transport=mtcp waiting for staging credit outstanding="
                       << flow_state->ledger.outstanding_credit() << "/" << flow_state->ledger.total_credit();
          last_warning = now;
        }

        if (now >= retry_deadline) {
          LOG(ERROR) << "MTCP staging credit wait exceeded deadline for request=" << request_key;
          release_once();
          release_handed_off = true;
          fail_mtcp_read_task(task, absl::ResourceExhaustedError("MTCP staging credit wait timed out"));
          return;
        }

        absl::SleepFor(backoff);
        backoff = std::min(backoff * 2, kMaxBackoff);
        continue;
      }

      LOG(ERROR) << "Failed to stage MTCP window: " << window_or.status();
      release_once();
      release_handed_off = true;
      fail_mtcp_read_task(task, window_or.status());
      return;
    }

    backoff = absl::Milliseconds(1);

    auto staged_window = std::move(window_or).value();
    transport::MTcpTransport::StageSendWindow send_window;
    send_window.request_key = request_key;
    send_window.window_seq = staged_window.window_seq;
    send_window.final_window = !staged_window.more_segments;
    send_window.total_bytes = total_bytes;
    send_window.segments.reserve(staged_window.segments.size());

    if (send_window.final_window) {
      send_window.pending_segments = std::make_shared<std::atomic<int>>(0);
      auto release_cb = release_once;
      send_window.on_window_complete = [release_cb]() { release_cb(); };
      release_handed_off = true;
    }

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
      send_segment.data = lease.exposed_ptr();
      send_segment.bytes = metadata.bytes;
      send_segment.metadata = metadata;

      send_segment.on_complete =
          [flow_state_ref = flow_state, key, metadata, lease = std::move(lease)](misc::result_t status) mutable {
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

    if (send_window.pending_segments) {
      send_window.pending_segments->store(static_cast<int>(send_window.segments.size()), std::memory_order_relaxed);
    }

    transport->enqueue_stage_window(std::move(send_window));
  }

  if (!release_handed_off) {
    release_once();
  }
}

void Communicator::set_dram_lease_provider(const std::shared_ptr<HostPinnedCpuStager::LeaseProvider>& provider) {
  if (!memory_stager_)
    return;
  if (auto ds = std::dynamic_pointer_cast<HostPinnedCpuStager>(memory_stager_)) {
    ds->set_lease_provider(provider);
  }
  // Also propagate to NUMA CPU stagers if present
  for (auto& kv : nic_cpu_stagers_) {
    if (auto ds2 = std::dynamic_pointer_cast<HostPinnedCpuStager>(kv.second)) {
      ds2->set_lease_provider(provider);
    }
  }
}

absl::Status Communicator::init(const std::string& ip, uint16_t port, int conn_count) {
  inited_.store(true);
  if (server_context_->open(ip, port, [this](tcp_transport_t t) { return this->on_new_client(t); }) != SUCCESS) {
    return absl::InternalError("failed to open server " + ip + ":" + std::to_string(port));
  }
  if (conn_count > 0) {
    mtcp_conn_count_ = conn_count;
  }
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
    int dev_id, // gpu_id
    const std::string& dst_ip,
    uint16_t dst_port,
    uint64_t remote_offset) {
  if (!inited_.load()) {
    LOG(ERROR) << "failed to read a tensor with a un-inited engine";
    return transport::ReadRequest::get_read_result_future("failed to read tensor through un-initiated engine");
  }
  net_dev_t net_dev = nullptr;
  if (enable_rdma_) {
    // with rail
    net_dev = get_net_dev(dev_type, dev_id, key);
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
    local_tensor->set_read_unready();
    if (dev_type == COMMUNICATE_ENGINE_DEV_GPU) {
      local_tensor->set_device_id(dev_id);
    }
    store_.register_tensor(local_tensor);
  }

  if (enable_rdma_ && net_dev != nullptr) {
    local_tensor->set_read_unready();
    if (local_tensor->get_dev_by_rail(net_dev->get_rail_id()) == nullptr) {
      local_tensor->add_dev(net_dev);
    }
    if (!local_tensor->is_registered(net_dev)) {
      net_dev->reg_async(local_tensor);
    }
  }
  local_tensor->set_read_ready();

  auto req = std::make_shared<transport::ReadRequest>(
      key, dst_ip, dst_port, local_tensor, remote_offset, net_dev != nullptr ? net_dev->get_rail_id() : -1);
  LOG(INFO) << "[read_tensor] Creating request: key=" << key << " dst=" << dst_ip << ":" << dst_port
            << " req_key=" << req->get_key() << " req_rail=" << req->get_rail_id();
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
    net_dev = get_net_dev(dev_type, dev_id, tensor_key);
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
  if (opts.direct_rdma_enabled) {
    tensor->set_direct_rdma_enabled(true);
  }

  if (enable_rdma_ && opts.register_mr) {
    net_dev->reg_async(tensor);
    if (!opts.async) {
      if (tensor->get_mr(net_dev) == nullptr) {
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
    ProtoReadRequest& request,
    const std::shared_ptr<PartitionTensor>& tensor) {
  if (!enable_rdma_) {
    return absl::FailedPreconditionError("RDMA transport disabled");
  }

  auto flow_state = channel->flow_state();
  if (!flow_state) {
    return absl::InternalError("channel missing flow state");
  }

  tensor->wait_read_ready();
  int16_t src_rail_id = request.rail_id;
  auto dev = tensor->get_dev();
  if (dev == nullptr) {
    return absl::InternalError("tensor missing RDMA device");
  }
  if (dev->get_rail_id() != src_rail_id) {
    auto dev_by_rail = rdma_context_->get_dev_by_rail(src_rail_id);
    if (dev_by_rail == nullptr) {
      LOG(WARNING) << "failed to get net dev by rail: rail_id=" << src_rail_id
                   << "use the default dev with rail_id=" << dev->get_rail_id();
      // use default rail dev
      request.rail_id = dev->get_rail_id();
    } else {
      tensor->add_dev(dev_by_rail);
    }

    dev = dev_by_rail;

    if (!tensor->is_registered(dev_by_rail)) {
      // sync or async?
      tensor->register_mr(dev_by_rail.get());
      // dev_by_rail->reg_async(tensor);
    }
    // reg tensor to the new rail
  }

  const bool tensor_on_cpu = tensor->get_mem_type() == COMMUNICATE_ENGINE_DEV_CPU;
  const int device_id = tensor->get_device_id();

  std::shared_ptr<MemoryStager> stager;
  if (tensor_on_cpu) {
    stager = get_cpu_stager_for_nic(dev->get_name());
  } else if (use_gpu_vram_staging_) {
    stager = get_gpu_vram_stager_for_id(device_id);
  } else {
    stager = get_gpu_mem_stager_for_id(device_id);
  }

  if (!stager) {
    if (tensor_on_cpu) {
      stager = memory_stager_;
    } else if (use_gpu_vram_staging_) {
      // VRAM staging is RDMA-only and depends on a per-GPU pool. If the tensor reports an
      // unexpected device id, fall back to host-pinned GPU staging so we never end up with
      // a nullptr stager later in the RDMA staging path.
      LOG_FIRST_N(WARNING, 1) << "GPU VRAM staging enabled but no VRAM stager for device_id=" << device_id
                              << "; falling back to host-pinned staging";
      stager = get_gpu_mem_stager_for_id(device_id);
      if (!stager) {
        stager = gpu_memory_stager_;
      }
    } else {
      stager = gpu_memory_stager_;
    }
  }

  const uint64_t total_bytes = request.bytes;
  const uint64_t start_offset = request.offset;
  const std::string tensor_key(reinterpret_cast<const char*>(request.tensor_key));
  const std::string request_key = transport::get_request_key(tensor_key, start_offset);
  FlowCreditLedger* ledger_ptr = &flow_state->ledger;
  MrCache* mr_cache_ptr = meta_mr_cache_.get();

  const bool direct_requested = tensor->direct_rdma_enabled();
  bool use_direct = false;
  ibv_mr* direct_mr = nullptr;
  DirectFallbackReason fallback_reason = DirectFallbackReason::kNone;

  if (direct_requested) {
    if (tensor_on_cpu || tensor->get_mem_type() != COMMUNICATE_ENGINE_DEV_GPU) {
      fallback_reason = DirectFallbackReason::kNotGpu;
    } else if (tensor->needs_staging()) {
      fallback_reason = DirectFallbackReason::kNeedsStaging;
    } else {
      const uint64_t tensor_bytes = tensor->get_bytes();
      if (start_offset > tensor_bytes || total_bytes > (tensor_bytes - start_offset)) {
        fallback_reason = DirectFallbackReason::kOutOfRange;
      } else {
        tensor->wait_mr_ready(dev);
        direct_mr = tensor->get_mr(dev);
        if (direct_mr == nullptr || !tensor->has_registered_mr(dev)) {
          fallback_reason = DirectFallbackReason::kMrUnavailable;
        } else {
          use_direct = true;
        }
      }
    }
  }

  if (!use_direct && !stager) {
    return absl::FailedPreconditionError("no staging backend available for tensor");
  }
  if (direct_requested && !use_direct && fallback_reason != DirectFallbackReason::kNone) {
    LOG_FIRST_N(WARNING, 1) << "RDMA zero-copy fallback for tensor=" << tensor_key
                            << " reason=" << DirectFallbackReasonToString(fallback_reason);
    record_direct_fallback_metric(fallback_reason);
  }

  const uint64_t chunk_size = use_direct
      ? (direct_rdma_chunk_bytes_ > 0 ? direct_rdma_chunk_bytes_ : request.bytes)
      : (stager && stager->get_chunk_size() > 0 ? stager->get_chunk_size() : request.bytes);

  const bool using_gpu_vram_stager =
      !tensor_on_cpu && use_gpu_vram_staging_ && stager && stager == get_gpu_vram_stager_for_id(device_id);
  const v1::RdmaConfig::StagedRdmaBackend staged_backend = using_gpu_vram_stager
      ? v1::RdmaConfig::STAGED_RDMA_BACKEND_GPU_VRAM
      : v1::RdmaConfig::STAGED_RDMA_BACKEND_HOST_PINNED;

  auto stage_fn = MakeStageFunction(
      tensor, ledger_ptr, stager, dev, mr_cache_ptr, tensor_key, staged_backend, use_direct, direct_mr);

  auto session = std::make_shared<RdmaReadSession>();
  session->request = request;
  session->tensor_key = tensor_key;
  session->request_key = request_key;
  session->tensor = tensor;
  session->stager = stager;
  session->dev = dev;
  session->control_transport = control_transport;
  session->zero_copy = use_direct;
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
  const std::string tensor_key(reinterpret_cast<const char*>(request.tensor_key));
  MtcpReadTask task;
  task.channel = channel;
  task.control_transport = control_transport;
  task.request = request;
  task.tensor = tensor;

  auto flow_state = channel->flow_state();
  if (!flow_state) {
    auto status = absl::InternalError("channel missing flow state");
    fail_mtcp_read_task(task, status);
    return status;
  }

  auto transport = channel->get_mtcp();
  if (transport == nullptr) {
    auto slot_or = acquire_gpu_channel_slot();
    if (!slot_or.ok()) {
      auto fail_msg = EngineMessage::make_message<ProtoReadFailed>(ENGINE_OP_READ_FAILED);
      auto* payload = fail_msg->get_payload<ProtoReadFailed>();
      memcpy(payload->tensor_key, request.tensor_key, kMaxTensorNameLen);
      payload->offset = request.offset;
      payload->reason = absl::IsResourceExhausted(slot_or.status()) ? TENSORCAST_READ_FAILED_RESOURCE_EXHAUSTED
                                                                    : TENSORCAST_READ_FAILED_MEM_MISMATCH;
      misc::result_t send_res = control_transport->send(fail_msg);
      if (send_res != misc::SUCCESS) {
        LOG(WARNING) << "Failed to send READ_FAILED after GPU channel limit hit: key=" << tensor_key
                     << " res=" << send_res;
      }
      return slot_or.status();
    }
    std::shared_ptr<void> gpu_slot_handle = std::move(slot_or).value();
    transport = std::make_shared<transport::MTcpTransport>(
        mtcp_conn_count_,
        gsl::not_null<std::shared_ptr<MemoryStager>>{memory_stager_},
        gsl::not_null<std::shared_ptr<MemoryStager>>{gpu_memory_stager_},
        gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{gpu_memory_pool_},
        buffers_per_flow_);
    channel->set_transport(transport);
    channel->set_gpu_slot_handle(std::move(gpu_slot_handle));
  }
  transport->set_tcp_tos(config_.transport().tcp_tos());

  channel->mtcp_request_started();
  bool request_handed_off = false;
  absl::Cleanup mtcp_request_guard = [&]() {
    if (!request_handed_off) {
      channel->mtcp_request_finished();
    }
  };

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
  absl::Status shutdown_status = absl::CancelledError("communicator shutting down");
  if (stop_.load(std::memory_order_relaxed)) {
    fail_mtcp_read_task(task, shutdown_status);
    return shutdown_status;
  }
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

  if (stop_.load(std::memory_order_relaxed)) {
    fail_mtcp_read_task(task, shutdown_status);
    return shutdown_status;
  }

  if (mtcp_staging_queue_.push(task) != misc::SUCCESS) {
    auto status = absl::InternalError("failed to enqueue MTCP staging task");
    fail_mtcp_read_task(task, status);
    return status;
  }

  request_handed_off = true;
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
    const std::string url_key = t->get_remote_url();
    LOG(INFO) << "[on_new_client] Client connection closed: " << url_key;
    auto channel = channels_.get(url_key);
    if (channel && channel->get_control().get() == t.get()) {
      channel->close();
      if (!channels_.erase_if(url_key, channel)) {
        VLOG(1) << "[on_new_client] Channel already removed or replaced for " << url_key;
      }
    } else {
      VLOG(1) << "[on_new_client] Channel mismatch or missing for " << url_key;
    }
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
      if (!channels_.erase_if(url_key, channel)) {
        VLOG(1) << "[do_create_channel] Channel already removed or replaced for " << url_key;
      }
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

    request->rail_id = req->get_rail_id();

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
      auto endpoint = channel->get_rdma_endpoint(local_dev_name, peer_dev_name);
      if (endpoint == nullptr) {
        LOG(WARNING) << "[rdma_handshake] received connect response for unknown endpoint: local_dev=" << local_dev_name
                     << " peer_dev=" << peer_dev_name;
        break;
      }

      transport::rdma_transport_t transport;
      uint64_t generation = 0;
      Channel::HandshakeState from_state = Channel::HandshakeState::kConnectRequested;
      bool already_ready = false;
      {
        absl::MutexLock lock(&endpoint->mu);
        if (endpoint->state == Channel::HandshakeState::kConnectRequested) {
          transport = endpoint->transport;
          generation = endpoint->generation;
          from_state = Channel::HandshakeState::kConnectRequested;
        } else if (endpoint->state == Channel::HandshakeState::kReady && !endpoint->pending_reads.empty()) {
          transport = endpoint->transport;
          generation = endpoint->generation;
          from_state = Channel::HandshakeState::kReady;
          already_ready = true;
        } else {
          LOG(INFO) << "[rdma_handshake] ignoring late connect response: local_dev=" << local_dev_name
                    << " peer_dev=" << peer_dev_name;
          break;
        }
      }

      if (transport == nullptr) {
        LOG(WARNING) << "[rdma_handshake] connect response missing transport: local_dev=" << local_dev_name
                     << " peer_dev=" << peer_dev_name;
        auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
        const absl::Status status = absl::UnavailableError("rdma transport missing while processing connect response");
        for (auto& pending : failed_reads) {
          pending_requests_.del(pending.request->get_key());
          pending.request->set_result(status);
        }
        {
          absl::MutexLock lock(&endpoint->mu);
          log_handshake_transition(
              local_dev_name,
              peer_dev_name,
              Channel::HandshakeState::kConnectRequested,
              Channel::HandshakeState::kFailed,
              endpoint->generation,
              endpoint->pending_reads.size());
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
        break;
      }

      if (!already_ready) {
        misc::result_t connect_res = transport->connect(&req->qp_info);
        if (connect_res != misc::SUCCESS) {
          LOG(WARNING) << "[rdma_handshake] transport connect failed: local_dev=" << local_dev_name
                       << " peer_dev=" << peer_dev_name << " res=" << connect_res;
          auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
          const absl::Status status = absl::UnavailableError("remote RDMA connect failed");
          for (auto& pending : failed_reads) {
            pending_requests_.del(pending.request->get_key());
            pending.request->set_result(status);
          }
          {
            absl::MutexLock lock(&endpoint->mu);
            log_handshake_transition(
                local_dev_name,
                peer_dev_name,
                Channel::HandshakeState::kConnectRequested,
                Channel::HandshakeState::kFailed,
                endpoint->generation,
                endpoint->pending_reads.size());
            endpoint->state = Channel::HandshakeState::kFailed;
            endpoint->transport.reset();
            endpoint->failure_count += 1;
            endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
            endpoint->retry_scheduled = false;
          }
          break;
        }
      }

      size_t queued = 0;
      {
        absl::MutexLock lock(&endpoint->mu);
        queued = endpoint->pending_reads.size();
        endpoint->state = Channel::HandshakeState::kReady;
        endpoint->failure_count = 0;
        endpoint->next_retry_at = absl::InfinitePast();
        endpoint->retry_scheduled = false;
      }
      log_handshake_transition(
          local_dev_name, peer_dev_name, from_state, Channel::HandshakeState::kReady, generation, queued);

      auto ready_reads = drain_pending_reads_for_generation(endpoint, generation);
      for (auto& pending : ready_reads) {
        pending.request->add_expected_completions(static_cast<int>(pending.segments.size()));
        auto res = transport->read_multi(pending.request, pending.segments);
        if (res != misc::SUCCESS) {
          pending_requests_.del(pending.request->get_key());
          pending.request->set_result(absl::UnavailableError("rdma read_multi failed after handshake"));
        }
      }
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
      const int negotiated_conn = std::max(2, static_cast<int>(rsp->conn_count));
      transport->set_conn_count(negotiated_conn);
      COMM_CHECK(transport->connect(ip, rsp->port, negotiated_conn));
      LOG(INFO) << "mtcp connect done " << ip << ":" << rsp->port << " " << negotiated_conn;
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
        auto dev = tensor->get_dev();
        CHECK(dev != nullptr) << "local tensor missing device metadata";
        const std::string local_dev_name = dev->get_name();

        auto endpoint = channel->ensure_rdma_endpoint(local_dev_name, peer_dev_name);

        std::vector<transport::RdmaTransport::RdmaReadSeg> rdma_segs;
        rdma_segs.reserve(hdr->num_segments);
        std::vector<uint64_t> ack_offsets;
        ack_offsets.reserve(hdr->num_segments);
        const uint64_t base_off = read_request->remote_offset_;
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

        auto now = absl::Now();
        transport::rdma_transport_t transport_to_use;
        transport::rdma_transport_t prepared_transport;
        std::shared_ptr<EngineMessage> connect_request_msg;
        bool issue_now = false;
        bool handshake_started = false;
        bool queued_current = false;
        bool deferred_for_backoff = false;
        bool schedule_retry = false;
        absl::Duration backoff_remaining = absl::ZeroDuration();
        absl::Status immediate_failure = absl::OkStatus();
        uint64_t generation = 0;

        while (true) {
          endpoint->mu.Lock();
          auto state = endpoint->state;
          if (state == Channel::HandshakeState::kReady) {
            transport_to_use = endpoint->transport;
            if (transport_to_use == nullptr || !transport_to_use->ready()) {
              log_handshake_transition(
                  local_dev_name,
                  peer_dev_name,
                  state,
                  Channel::HandshakeState::kIdle,
                  endpoint->generation,
                  endpoint->pending_reads.size());
              endpoint->state = Channel::HandshakeState::kIdle;
              endpoint->transport.reset();
              endpoint->mu.Unlock();
              continue;
            }
            generation = endpoint->generation;
            endpoint->mu.Unlock();
            issue_now = true;
            break;
          }

          if (state == Channel::HandshakeState::kConnectRequested) {
            generation = endpoint->generation;
            endpoint->pending_reads.push_back(
                Channel::PendingRdmaRead{
                    .request = read_request,
                    .segments = std::move(rdma_segs),
                    .enqueued_at = now,
                    .generation = generation,
                });
            queued_current = true;
            const size_t queue_depth = endpoint->pending_reads.size();
            endpoint->mu.Unlock();
            LOG(INFO) << "[rdma_handshake] queueing read while awaiting connect: request=" << read_request->get_key()
                      << " local_dev=" << local_dev_name << " peer_dev=" << peer_dev_name
                      << " queue_depth=" << queue_depth << " rail_id=" << read_request->get_rail_id();
            break;
          }

          if (state == Channel::HandshakeState::kIdle || state == Channel::HandshakeState::kFailed) {
            const bool can_retry = state == Channel::HandshakeState::kIdle || now >= endpoint->next_retry_at;
            if (!can_retry) {
              generation = endpoint->generation;
              endpoint->pending_reads.push_back(
                  Channel::PendingRdmaRead{
                      .request = read_request,
                      .segments = std::move(rdma_segs),
                      .enqueued_at = now,
                      .generation = generation,
                  });
              queued_current = true;
              deferred_for_backoff = true;
              backoff_remaining = endpoint->next_retry_at - now;
              if (!endpoint->retry_scheduled) {
                endpoint->retry_scheduled = true;
                schedule_retry = true;
              }
              const size_t queue_depth = endpoint->pending_reads.size();
              endpoint->mu.Unlock();
              LOG(WARNING) << "[rdma_handshake] deferring read during backoff: request=" << read_request->get_key()
                           << " local_dev=" << local_dev_name << " peer_dev=" << peer_dev_name
                           << " queue_depth=" << queue_depth
                           << " wait_ms=" << absl::ToInt64Milliseconds(backoff_remaining);
              break;
            }

            if (prepared_transport == nullptr) {
              endpoint->mu.Unlock();
              prepared_transport = rdma_context_->create_transport(local_dev_name);
              if (prepared_transport == nullptr) {
                immediate_failure = absl::InternalError("failed to allocate RDMA transport");
                break;
              }
              connect_request_msg =
                  EngineMessage::make_message<ProtoRdmaConnectRequest>(ENGINE_OP_RDMA_CONNECT_REQUEST);
              auto* payload = connect_request_msg->get_payload<ProtoRdmaConnectRequest>();
              misc::result_t info_res = prepared_transport->get_local_info(&payload->qp_info);
              if (info_res != misc::SUCCESS) {
                immediate_failure = absl::InternalError("failed to prepare RDMA connect info");
                prepared_transport.reset();
                connect_request_msg.reset();
                break;
              }
              misc::STRNCPY(payload->src_dev_name, local_dev_name, kMaxDevName);
              misc::STRNCPY(payload->dst_dev_name, peer_dev_name, kMaxDevName);
              continue;
            }

            Channel::HandshakeState from_state = state;
            endpoint->transport = prepared_transport;
            endpoint->generation += 1;
            generation = endpoint->generation;
            endpoint->state = Channel::HandshakeState::kConnectRequested;
            endpoint->failure_count = 0;
            endpoint->next_retry_at = absl::InfinitePast();
            endpoint->retry_scheduled = false;
            endpoint->pending_reads.push_back(
                Channel::PendingRdmaRead{
                    .request = read_request,
                    .segments = std::move(rdma_segs),
                    .enqueued_at = now,
                    .generation = generation,
                });
            queued_current = true;
            handshake_started = true;
            const size_t queue_depth = endpoint->pending_reads.size();
            endpoint->mu.Unlock();
            log_handshake_transition(
                local_dev_name,
                peer_dev_name,
                from_state,
                Channel::HandshakeState::kConnectRequested,
                generation,
                queue_depth);
            break;
          }

          endpoint->mu.Unlock();
          immediate_failure = absl::InternalError("unexpected RDMA handshake state");
          break;
        }

        if (!immediate_failure.ok()) {
          LOG(WARNING) << "[rdma_handshake] immediate failure handling read: request=" << read_request->get_key()
                       << " status=" << immediate_failure;
          read_request->set_result(immediate_failure);
          pending_requests_.del(req_key);
          break;
        }

        if (issue_now) {
          read_request->add_expected_completions(static_cast<int>(rdma_segs.size()));
          auto res = transport_to_use->read_multi(read_request, rdma_segs);
          if (res != misc::SUCCESS) {
            read_request->set_result(absl::UnavailableError("rdma read_multi failed before completion"));
            pending_requests_.del(req_key);
          }
          break;
        }

        if (handshake_started) {
          auto send_res = t->send(connect_request_msg);
          if (send_res != misc::SUCCESS) {
            LOG(WARNING) << "[rdma_handshake] failed to send connect request: request=" << read_request->get_key()
                         << " local_dev=" << local_dev_name << " peer_dev=" << peer_dev_name << " res=" << send_res;
            const absl::Status send_error = absl::UnavailableError("failed to send RDMA connect request to peer");
            auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
            for (auto& pending : failed_reads) {
              pending_requests_.del(pending.request->get_key());
              pending.request->set_result(send_error);
            }
            {
              absl::MutexLock lock(&endpoint->mu);
              log_handshake_transition(
                  local_dev_name,
                  peer_dev_name,
                  Channel::HandshakeState::kConnectRequested,
                  Channel::HandshakeState::kFailed,
                  endpoint->generation,
                  endpoint->pending_reads.size());
              endpoint->state = Channel::HandshakeState::kFailed;
              endpoint->transport.reset();
              endpoint->failure_count += 1;
              endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
              endpoint->retry_scheduled = false;
            }
          }
          break;
        }

        if (queued_current && deferred_for_backoff) {
          if (schedule_retry) {
            if (backoff_remaining <= absl::ZeroDuration()) {
              backoff_remaining = absl::Milliseconds(1);
            }
            schedule_handshake_retry(channel, local_dev_name, peer_dev_name, backoff_remaining);
          }
          break;
        }

        break;
      }
      if (hdr->transport_type == ENGINE_TRANSPORT_MTCP) {
        // MTCP path using EX header (no segments)
        auto transport = channel->get_mtcp();
        if (transport == nullptr) {
          const int requested_conn = std::max(2, mtcp_conn_count_);
          transport = std::make_shared<transport::MTcpTransport>(
              requested_conn,
              gsl::not_null<std::shared_ptr<MemoryStager>>{memory_stager_},
              gsl::not_null<std::shared_ptr<MemoryStager>>{gpu_memory_stager_},
              gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>{gpu_memory_pool_},
              buffers_per_flow_);
          LOG(INFO) << "[on_receive_response] Sending MTCP_CONNECT_REQUEST for " << tensor_key;
          auto req = EngineMessage::make_message<ProtoMtcpConnectRequest>(ENGINE_OP_MTCP_CONNECT_REQUEST);
          auto* payload = req->get_payload<ProtoMtcpConnectRequest>();
          payload->conn_count = requested_conn;
          COMM_CHECK(t->send(req));
          channel->set_transport(transport);
        }
        transport->set_tcp_tos(config_.transport().tcp_tos());
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

      auto endpoint = channel->get_rdma_endpoint(local_dev_name, peer_dev_name);
      if (endpoint == nullptr) {
        break;
      }

      uint64_t generation = 0;
      {
        absl::MutexLock lock(&endpoint->mu);
        generation = endpoint->generation;
      }

      auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
      const absl::Status status = absl::UnavailableError("remote RDMA connect failed");
      for (auto& pending : failed_reads) {
        pending_requests_.del(pending.request->get_key());
        pending.request->set_result(status);
      }

      {
        absl::MutexLock lock(&endpoint->mu);
        log_handshake_transition(
            local_dev_name,
            peer_dev_name,
            Channel::HandshakeState::kConnectRequested,
            Channel::HandshakeState::kFailed,
            endpoint->generation,
            endpoint->pending_reads.size());
        endpoint->state = Channel::HandshakeState::kFailed;
        endpoint->transport.reset();
        endpoint->failure_count += 1;
        endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
        endpoint->retry_scheduled = false;
      }
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
      absl::Status failure_status = absl::InternalError("failed to read from peer");
      switch (rsp->reason) {
        case TENSORCAST_READ_FAILED_NO_TENSOR:
          failure_status = absl::NotFoundError(absl::StrCat("tensor not found: ", tensor_key));
          break;
        case TENSORCAST_READ_FAILED_OVERFLOW:
          failure_status =
              absl::OutOfRangeError(absl::StrCat("read overflow for tensor ", tensor_key, " offset=", rsp->offset));
          break;
        case TENSORCAST_READ_FAILED_RESOURCE_EXHAUSTED:
          failure_status = absl::ResourceExhaustedError("GPU staging pool capacity exceeded");
          break;
        case TENSORCAST_READ_FAILED_MEM_MISMATCH:
        default:
          break;
      }
      read_request->set_result(failure_status);
      break;
    }
    default:
      LOG(WARNING) << "failed to process response: " << msg->get_op();
  }

  return misc::SUCCESS;
}

void Communicator::schedule_handshake_retry(
    const channel_t& channel,
    const std::string& local_dev_name,
    const std::string& peer_dev_name,
    absl::Duration delay) {
  if (!enable_rdma_ || !handshake_retry_thread_started_ || handshake_retry_stop_.load(std::memory_order_relaxed)) {
    return;
  }
  if (delay <= absl::ZeroDuration()) {
    delay = absl::Milliseconds(1);
  }

  HandshakeRetryTask task;
  task.resume_at = absl::Now() + delay;
  task.channel = channel;
  task.local_dev_name = local_dev_name;
  task.peer_dev_name = peer_dev_name;

  {
    absl::MutexLock lock(&handshake_retry_mu_);
    handshake_retry_queue_.push(std::move(task));
  }
  handshake_retry_cv_.Signal();
}

void Communicator::handshake_retry_loop() {
  while (!handshake_retry_stop_.load(std::memory_order_relaxed)) {
    HandshakeRetryTask task;
    bool has_task = false;
    {
      absl::MutexLock lock(&handshake_retry_mu_);
      while (!handshake_retry_stop_.load(std::memory_order_relaxed) && handshake_retry_queue_.empty()) {
        handshake_retry_cv_.Wait(&handshake_retry_mu_);
      }
      if (handshake_retry_stop_.load(std::memory_order_relaxed)) {
        break;
      }
      auto now = absl::Now();
      const auto& next = handshake_retry_queue_.top();
      if (next.resume_at > now) {
        handshake_retry_cv_.WaitWithDeadline(&handshake_retry_mu_, next.resume_at);
        continue;
      }
      task = handshake_retry_queue_.top();
      handshake_retry_queue_.pop();
      has_task = true;
    }

    if (!has_task) {
      continue;
    }

    process_handshake_retry_task(task.channel, task.local_dev_name, task.peer_dev_name);
  }
}

void Communicator::process_handshake_retry_task(
    const std::weak_ptr<Channel>& channel_weak,
    const std::string& local_dev_name,
    const std::string& peer_dev_name) {
  if (handshake_retry_stop_.load(std::memory_order_relaxed)) {
    return;
  }

  auto channel = channel_weak.lock();
  if (!channel) {
    return;
  }

  auto endpoint = channel->get_rdma_endpoint(local_dev_name, peer_dev_name);
  if (endpoint == nullptr) {
    return;
  }

  start_pending_rdma_handshake(channel, endpoint, local_dev_name, peer_dev_name);
}

void Communicator::start_pending_rdma_handshake(
    const channel_t& channel,
    const std::shared_ptr<Channel::RdmaEndpoint>& endpoint,
    const std::string& local_dev_name,
    const std::string& peer_dev_name) {
  if (!enable_rdma_ || handshake_retry_stop_.load(std::memory_order_relaxed)) {
    return;
  }

  auto control = channel->get_control();
  if (control == nullptr) {
    return;
  }

  transport::rdma_transport_t prepared_transport;
  std::shared_ptr<EngineMessage> connect_request_msg;

  while (!handshake_retry_stop_.load(std::memory_order_relaxed)) {
    endpoint->mu.Lock();
    endpoint->retry_scheduled = false;
    const auto state = endpoint->state;
    const bool has_pending = !endpoint->pending_reads.empty();
    const absl::Time now = absl::Now();

    if (!has_pending) {
      endpoint->mu.Unlock();
      return;
    }

    if (state == Channel::HandshakeState::kConnectRequested || state == Channel::HandshakeState::kReady) {
      endpoint->mu.Unlock();
      return;
    }

    if (state == Channel::HandshakeState::kFailed && now < endpoint->next_retry_at) {
      const absl::Duration delay = endpoint->next_retry_at - now;
      endpoint->retry_scheduled = true;
      endpoint->mu.Unlock();
      schedule_handshake_retry(channel, local_dev_name, peer_dev_name, delay);
      return;
    }

    if (prepared_transport == nullptr) {
      endpoint->mu.Unlock();
      prepared_transport = rdma_context_->create_transport(local_dev_name);
      if (prepared_transport == nullptr) {
        const absl::Status status = absl::InternalError("failed to allocate RDMA transport");
        auto failed_reads = drain_pending_reads_for_generation(endpoint, 0);
        for (auto& pending : failed_reads) {
          pending_requests_.del(pending.request->get_key());
          pending.request->set_result(status);
        }
        {
          absl::MutexLock lock(&endpoint->mu);
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->transport.reset();
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
        return;
      }

      connect_request_msg = EngineMessage::make_message<ProtoRdmaConnectRequest>(ENGINE_OP_RDMA_CONNECT_REQUEST);
      auto* payload = connect_request_msg->get_payload<ProtoRdmaConnectRequest>();
      misc::result_t info_res = prepared_transport->get_local_info(&payload->qp_info);
      if (info_res != misc::SUCCESS) {
        prepared_transport.reset();
        connect_request_msg.reset();
        const absl::Status status = absl::InternalError("failed to prepare RDMA connect info");
        auto failed_reads = drain_pending_reads_for_generation(endpoint, 0);
        for (auto& pending : failed_reads) {
          pending_requests_.del(pending.request->get_key());
          pending.request->set_result(status);
        }
        {
          absl::MutexLock lock(&endpoint->mu);
          endpoint->state = Channel::HandshakeState::kFailed;
          endpoint->transport.reset();
          endpoint->failure_count += 1;
          endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
          endpoint->retry_scheduled = false;
        }
        return;
      }

      misc::STRNCPY(payload->src_dev_name, local_dev_name, kMaxDevName);
      misc::STRNCPY(payload->dst_dev_name, peer_dev_name, kMaxDevName);
      continue;
    }

    Channel::HandshakeState from_state = state;
    endpoint->transport = prepared_transport;
    endpoint->generation += 1;
    uint64_t generation = endpoint->generation;
    endpoint->state = Channel::HandshakeState::kConnectRequested;
    endpoint->failure_count = 0;
    endpoint->next_retry_at = absl::InfinitePast();
    const size_t queue_depth = endpoint->pending_reads.size();
    for (auto& pending : endpoint->pending_reads) {
      pending.generation = generation;
    }
    endpoint->mu.Unlock();

    log_handshake_transition(
        local_dev_name, peer_dev_name, from_state, Channel::HandshakeState::kConnectRequested, generation, queue_depth);

    auto send_res = control->send(connect_request_msg);
    if (send_res != misc::SUCCESS) {
      LOG(WARNING) << "[rdma_handshake] failed to send connect request: local_dev=" << local_dev_name
                   << " peer_dev=" << peer_dev_name << " res=" << send_res;
      const absl::Status send_error = absl::UnavailableError("failed to send RDMA connect request to peer");
      auto failed_reads = drain_pending_reads_for_generation(endpoint, generation);
      for (auto& pending : failed_reads) {
        pending_requests_.del(pending.request->get_key());
        pending.request->set_result(send_error);
      }
      {
        absl::MutexLock lock(&endpoint->mu);
        log_handshake_transition(
            local_dev_name,
            peer_dev_name,
            Channel::HandshakeState::kConnectRequested,
            Channel::HandshakeState::kFailed,
            endpoint->generation,
            endpoint->pending_reads.size());
        endpoint->state = Channel::HandshakeState::kFailed;
        endpoint->transport.reset();
        endpoint->failure_count += 1;
        endpoint->next_retry_at = absl::Now() + compute_handshake_backoff(endpoint->failure_count);
        endpoint->retry_scheduled = false;
      }
    }
    return;
  }
}

net_dev_t Communicator::get_net_dev(int dev_type, int dev_id, const std::string& key, int rail_id) {
  net_dev_t net_dev(nullptr);
  if (enable_rdma_) {
    CHECK(rdma_context_ != nullptr) << "rdma context is not initialized";

    net_dev = rdma_context_->get_best_dev(dev_type, dev_id, -1, key);

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
  auto channel = channels_.get(url.str());
  if (channel == nullptr) {
    return absl::InternalError("could not find the connection");
  }
  if (!channels_.erase_if(url.str(), channel)) {
    VLOG(1) << "[close_connection] Channel already removed or replaced for " << url.str();
  }
  channel->close();
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
        if (!channels_.erase_if(p.first, p.second)) {
          VLOG(1) << "[channel gc] Channel already removed or replaced for " << p.first;
        }
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

std::shared_ptr<MemoryStager> Communicator::get_gpu_vram_stager_for_id(int gpu_id) const {
  auto it = gpu_vram_stagers_.find(gpu_id);
  if (it != gpu_vram_stagers_.end())
    return it->second;
  return nullptr;
}

} // namespace tensorcast::communicator::engine
