// Copyright (c) 2025, TensorCast Team.

#include "core/store/replica/transfer_service.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <utility>

#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "core/common/async_copy_manager.h"
#include "core/common/cuda_api.h"
#include "core/store/loader/cpu_va_sink.h"
#include "core/store/loader/file_partition_source.h"
#include "core/store/loader/gpu_memory_sink.h"
#include "core/store/loader/multi_safetensors_source.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/safetensors_source.h"
#include "core/store/loader/streaming_buffer_adapter.h"
#include "core/store/replica/transfer_helpers.h"
// OpenTelemetry metrics (inflight copy gauge)
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/observer_result.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::replica {

using common::memory::MemoryLocation;

namespace {
// No compatibility mode: enforce that transfer slice is a divisor of chunk size

// Minimal per-GPU scheduler scaffold (Phase B stub):
// Limits to 1 in-flight session per GPU for now, ready for extension to bytes/copies.
struct GpuSchedulerGate {
  bool active{false};
  absl::CondVar cv;
};

ABSL_CONST_INIT absl::Mutex g_sched_mu(absl::kConstInit);
std::unordered_map<int, GpuSchedulerGate> g_sched ABSL_GUARDED_BY(g_sched_mu);

class GpuSchedHandle {
 public:
  explicit GpuSchedHandle(int device_id) : device_id_(device_id) {
    absl::MutexLock lk(&g_sched_mu);
    GpuSchedulerGate& gate = g_sched[device_id_];
    while (gate.active)
      gate.cv.Wait(&g_sched_mu);
    gate.active = true;
    // Update inflight copies (+1)
    update_inflight_copies_(device_id_, +1);
  }

  ~GpuSchedHandle() {
    absl::MutexLock lk(&g_sched_mu);
    auto it = g_sched.find(device_id_);
    if (it != g_sched.end()) {
      it->second.active = false;
      it->second.cv.Signal();
    }
    // Update inflight copies (-1)
    update_inflight_copies_(device_id_, -1);
  }

 private:
  int device_id_;
  static void update_inflight_copies_(int device_id, int delta);
};

// --- Minimal inflight copies gauge (per GPU) ---
ABSL_CONST_INIT absl::Mutex g_if_mu(absl::kConstInit);
std::unordered_map<int, int> g_inflight_copies ABSL_GUARDED_BY(g_if_mu);
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> g_meter;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> g_if_copies_gauge;

static void inflight_copies_callback(opentelemetry::metrics::ObserverResult result, void* /*state*/) noexcept {
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs)
    return;
  absl::MutexLock lk(&g_if_mu);
  for (const auto& kv : g_inflight_copies) {
    const int gpu = kv.first;
    const int copies = kv.second;
    obs->Observe(static_cast<double>(copies), {{"gpu", opentelemetry::common::AttributeValue(std::to_string(gpu))}});
  }
}

void GpuSchedHandle::update_inflight_copies_(int device_id, int delta) {
  {
    absl::MutexLock lk(&g_if_mu);
    g_inflight_copies[device_id] += delta;
    if (g_inflight_copies[device_id] < 0)
      g_inflight_copies[device_id] = 0;
  }
  // Lazy-init meter and gauge
  if (!g_meter) {
    g_meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  }
  if (!g_if_copies_gauge) {
    g_if_copies_gauge = g_meter->CreateDoubleObservableGauge("tc_tx_inflight_copies_gauge");
    g_if_copies_gauge->AddCallback(&inflight_copies_callback, nullptr);
  }
}
} // namespace

namespace {

absl::Status synchronize_gpu_after_transfer(int device_id, absl::string_view context, uint64_t total_bytes) {
  if (device_id < 0) {
    return absl::InvalidArgumentError("GPU synchronize requested with invalid device id");
  }
  auto set_dev_status = tensorcast::cuda::set_device(device_id);
  if (!set_dev_status.ok()) {
    LOG(ERROR) << context << ": cuda set_device failed for device=" << device_id << ": " << set_dev_status;
    return set_dev_status;
  }
  const auto sync_start = std::chrono::steady_clock::now();
  auto sync_status = tensorcast::cuda::device_synchronize();
  const auto sync_end = std::chrono::steady_clock::now();
  const auto sync_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sync_end - sync_start);
  if (!sync_status.ok()) {
    LOG(ERROR) << context << ": device_synchronize failed for device=" << device_id
               << " duration_ms=" << sync_ms.count() << " status=" << sync_status;
    return sync_status;
  }
  VLOG(1) << context << " cuda synchronize device=" << device_id << " bytes=" << total_bytes
          << " duration_ms=" << sync_ms.count();
  return absl::OkStatus();
}

} // namespace

TransferService::TransferService(
    const gsl::not_null<std::shared_ptr<common::memory::PinnedBufferPool>>& pinned_pool,
    const gsl::not_null<std::shared_ptr<UnifiedMemoryAuthority>>& uma,
    loading::ReplicaKey replica_key,
    Config cfg)
    : pinned_pool_(pinned_pool),
      uma_(uma),
      replica_key_(std::move(replica_key)),
      cfg_(cfg),
      spb_(
          std::make_shared<common::memory::StreamingPinnedBuffer>(
              /*num_chunks=*/16,
              pinned_pool_->slice_bytes(),
              pinned_pool_)) {}

size_t TransferService::get_pool_chunk_size() const {
  return pinned_pool_->slice_bytes();
}

absl::Status TransferService::copy_cpu_to_gpu_streaming(
    uint32_t device_id,
    gsl::not_null<void*> gpu_ptr,
    size_t total_bytes) {
  // Validate parameters first
  if (total_bytes == 0) {
    return absl::InvalidArgumentError("Total bytes must be greater than 0");
  }

  void* vs_base = uma_->get_cpu_base_ptr(replica_key_);
  if (!vs_base) {
    return absl::FailedPreconditionError("CPU base not available via UMA");
  }
  // Gate per GPU (1 active session per GPU)
  std::unique_ptr<GpuSchedHandle> sched = std::make_unique<GpuSchedHandle>(static_cast<int>(device_id));
  // Create a per-session streaming buffer backed by the shared pinned pool
  const size_t slice_bytes = get_pool_chunk_size();
  auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
      /*num_chunks=*/16, slice_bytes, pinned_pool_);
  auto init_status = session_spb->initialize(cfg_.pinned_memory_timeout);
  if (!init_status.ok()) {
    return init_status;
  }
  return perform_copy_cpu_to_gpu_streaming(
      replica_key_.artifact_id,
      device_id,
      session_spb,
      gpu_ptr,
      total_bytes,
      gsl::not_null<void*>{vs_base},
      uma_,
      replica_key_);
}

absl::Status TransferService::copy_gpu_to_cpu_streaming(
    uint32_t device_id,
    gsl::not_null<void*> gpu_ptr,
    size_t total_bytes) {
  // Validate parameters first
  if (total_bytes == 0) {
    return absl::InvalidArgumentError("Total bytes must be greater than 0");
  }

  auto spb = get_streaming_buffer();
  if (!spb) {
    return absl::FailedPreconditionError("Streaming buffer not available");
  }
  void* vs_base = uma_->get_cpu_base_ptr(replica_key_);
  if (!vs_base) {
    return absl::FailedPreconditionError("CPU base not available via UMA");
  }
  return perform_copy_gpu_to_cpu_streaming(
      replica_key_.artifact_id,
      device_id,
      spb,
      gpu_ptr,
      total_bytes,
      gsl::not_null<void*>{vs_base},
      uma_,
      replica_key_);
}

std::unique_ptr<loader::PositionedSink> TransferService::build_sink_(
    MemoryLocation target_location,
    void* gpu_ptr,
    std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation,
    int device_id) {
  if (target_location == MemoryLocation::GPU) {
    auto sink = std::make_unique<loader::GpuMemorySink>(loader::GpuMemorySink::Options{
        .gpu_base_ptr = gsl::not_null<void*>{gpu_ptr},
        // Leave total_size=0 to indicate unknown/streamed; sink close() will not enforce completeness.
        .total_size = 0,
        .chunk_size = [&]() -> size_t {
          auto l = uma_->get_layout(replica_key_);
          if (l.ok() && l->artifact_chunk_bytes > 0)
            return l->artifact_chunk_bytes;
          return get_pool_chunk_size();
        }(),
        .device_id = device_id,
        .allocation = std::move(gpu_allocation)});
    return sink;
  }
  // CPU sink writes into VS region via PositionedSink
  loader::CpuVaSink::Options opts;
  opts.uma = uma_;
  opts.replica_key = replica_key_;
  opts.total_size = uma_->get_artifact_size(replica_key_).ok() ? *uma_->get_artifact_size(replica_key_) : 0;
  opts.plan_direct_write_fn =
      [uma = uma_, key = replica_key_](absl::Span<const VaRange> ranges) -> absl::StatusOr<DirectWriteGrant> {
    return uma->grant_direct_write(key, ranges);
  };
  auto sink = std::make_unique<loader::CpuVaSink>(std::move(opts));
  return sink;
}

std::vector<std::pair<uint64_t, size_t>> TransferService::build_ranges_(
    std::optional<absl::Span<const uint32_t>> chunk_indices,
    size_t chunk_size,
    uint64_t total_bytes) {
  std::vector<std::pair<uint64_t, size_t>> ranges;
  if (!chunk_indices.has_value() || chunk_indices->empty()) {
    // Always use chunk-aligned ranges in V3 final state.
    const uint64_t chunks = (chunk_size > 0) ? ((total_bytes + chunk_size - 1) / chunk_size) : 0;
    for (uint64_t i = 0; i < chunks; ++i) {
      const uint64_t off = i * chunk_size;
      const uint64_t len64 = std::min<uint64_t>(chunk_size, total_bytes - off);
      ranges.emplace_back(off, static_cast<size_t>(len64));
    }
    // Handle tail 0-case gracefully (when total_bytes==0)
    if (ranges.empty() && total_bytes > 0) {
      ranges.emplace_back(0ULL, total_bytes);
    }
    return ranges;
  }

  // Remove duplicates and sort in one pass
  std::vector<uint32_t> sorted(chunk_indices->begin(), chunk_indices->end());
  std::ranges::sort(sorted);
  sorted.erase(std::ranges::unique(sorted).begin(), sorted.end());
  uint32_t run_start = sorted.front();
  uint32_t prev = run_start;
  for (size_t i = 1; i < sorted.size(); ++i) {
    const uint32_t idx = sorted[i];
    if (idx == prev + 1) {
      prev = idx;
      continue;
    }
    const uint64_t off = static_cast<uint64_t>(run_start) * chunk_size;
    const uint64_t end = static_cast<uint64_t>(prev + 1) * chunk_size;
    const uint64_t len64 = (end > total_bytes) ? (total_bytes - off) : (end - off);
    const auto len = static_cast<size_t>(len64);
    ranges.emplace_back(off, len);
    run_start = prev = idx;
  }
  const uint64_t off = static_cast<uint64_t>(run_start) * chunk_size;
  const uint64_t end = static_cast<uint64_t>(prev + 1) * chunk_size;
  const uint64_t len64 = (end > total_bytes) ? (total_bytes - off) : (end - off);
  const auto len = static_cast<size_t>(len64);
  ranges.emplace_back(off, len);
  return ranges;
}

absl::Status TransferService::load_from_source(
    std::unique_ptr<loader::SeekableSource>& source,
    MemoryLocation target_location,
    int concurrency,
    std::optional<absl::Span<const uint32_t>> chunk_indices,
    void* gpu_ptr_or_null,
    std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation,
    int device_id) {
  LOG(INFO) << "TransferService::load_from_source begin: artifact_id=" << replica_key_.artifact_id
            << " target=" << static_cast<int>(target_location) << " concurrency=" << concurrency;
  // Validate parameters
  if (!source) {
    return absl::InvalidArgumentError("Source is null");
  }
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }
  if (target_location == MemoryLocation::GPU && !gpu_ptr_or_null) {
    return absl::InvalidArgumentError("GPU pointer required for GPU target location");
  }

  const size_t pool_chunk_size = get_pool_chunk_size();
  size_t layout_chunk_size = pool_chunk_size;
  {
    auto l = uma_->get_layout(replica_key_);
    if (l.ok() && l->artifact_chunk_bytes > 0) {
      layout_chunk_size = l->artifact_chunk_bytes;
    }
  }
  uint64_t total_size = 0;
  auto sz = uma_->get_artifact_size(replica_key_);
  if (sz.ok()) {
    total_size = *sz;
  }
  // Fallback: use source total size when UMA doesn't know
  if (total_size == 0) {
    if (auto* fps = dynamic_cast<loader::FilePartitionSource*>(source.get())) {
      total_size = fps->total_size();
    }
    if (total_size == 0) {
      if (auto* ss = dynamic_cast<loader::SafetensorsSource*>(source.get())) {
        total_size = ss->total_size();
      } else if (auto* ms = dynamic_cast<loader::MultiSafetensorsSource*>(source.get())) {
        total_size = ms->total_size();
      }
    }
  }

  // Gate per GPU (1 active session per GPU). Use scheduler gate unconditionally when targeting GPU.
  std::unique_ptr<GpuSchedHandle> sched;
  if (target_location == MemoryLocation::GPU) {
    sched = std::make_unique<GpuSchedHandle>(device_id);
  }

  // Create a per-session streaming buffer backed by the shared pinned pool
  size_t slice_bytes = get_pool_chunk_size();
  if (slice_bytes == 0 || layout_chunk_size == 0) {
    return absl::FailedPreconditionError("slice_bytes or layout_chunk_size is zero");
  }
  if ((layout_chunk_size % slice_bytes) != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "transfer_slice_bytes (%zu) must divide artifact_chunk_bytes (%zu) to avoid cross-chunk slices",
            slice_bytes,
            layout_chunk_size));
  }
  auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
      /*num_chunks=*/16, slice_bytes, pinned_pool_);
  auto init_status = session_spb->initialize(cfg_.pinned_memory_timeout);
  if (!init_status.ok()) {
    return init_status;
  }

  loader::StreamingBufferAdapter adapter(session_spb);
  auto sink = build_sink_(target_location, gpu_ptr_or_null, std::move(gpu_allocation), device_id);
  if (!sink) {
    return absl::FailedPreconditionError("Failed to construct sink for target location");
  }

  auto ranges = build_ranges_(chunk_indices, layout_chunk_size, total_size);
  const auto start_tp = std::chrono::steady_clock::now();
  absl::Status pump_status = loader::pump_ranges(*source, *sink, adapter, absl::MakeSpan(ranges), concurrency);
  if (!pump_status.ok()) {
    LOG(ERROR) << "TransferService::load_from_source pump_ranges failed: " << pump_status;
    return pump_status;
  }
  {
    absl::Status _st = sink->close();
    if (!_st.ok()) {
      LOG(WARNING) << "TransferService: sink->close failed: " << _st;
    }
  }
  const auto end_tp = std::chrono::steady_clock::now();
  const uint64_t dur_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(end_tp - start_tp).count());
  uint64_t total_bytes = 0;
  for (const auto& r : ranges)
    total_bytes += r.second;
  if (target_location == MemoryLocation::GPU && total_bytes > 0) {
    auto sync_status = synchronize_gpu_after_transfer(device_id, "TransferService::load_from_source", total_bytes);
    if (!sync_status.ok()) {
      return sync_status;
    }
  }
  // Record duration metric
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto tx_dur = meter->CreateDoubleHistogram("tc_tx_duration_ms");
    const char* target_str = (target_location == MemoryLocation::GPU) ? "GPU" : "CPU";
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("target", opentelemetry::common::AttributeValue(std::string(target_str)));
    tx_dur->Record(
        static_cast<double>(dur_ms),
        opentelemetry::common::KeyValueIterableView(attrs),
        opentelemetry::context::Context{});
  } catch (...) {
    VLOG(1) << "metrics histogram tc_tx_duration_ms unavailable";
  }
  LOG(INFO) << "TransferService::load_from_source end: artifact_id=" << replica_key_.artifact_id
            << " target=" << static_cast<int>(target_location) << " bytes=" << total_bytes << " duration_ms=" << dur_ms
            << " slices_per_chunk="
            << (layout_chunk_size == 0 ? 0 : (slice_bytes ? (layout_chunk_size / slice_bytes) : 0));
  return absl::OkStatus();
}

absl::Status TransferService::execute(
    const UnifiedMemoryAuthority::TransferPlan& plan,
    MemoryLocation target_location,
    loader::SeekableSource& source,
    int concurrency,
    void* gpu_ptr_or_null,
    std::shared_ptr<common::memory::GpuDeviceMemory> gpu_allocation,
    int device_id) {
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }
  if (target_location == MemoryLocation::GPU && !gpu_ptr_or_null) {
    return absl::InvalidArgumentError("GPU pointer required for GPU target location");
  }
  // Per-GPU limiter using scheduler gate when targeting GPU
  std::unique_ptr<GpuSchedHandle> sched;
  if (target_location == MemoryLocation::GPU) {
    sched = std::make_unique<GpuSchedHandle>(device_id);
  }
  // Per-session SPB
  size_t layout_chunk_size = get_pool_chunk_size();
  {
    auto l = uma_->get_layout(replica_key_);
    if (l.ok() && l->artifact_chunk_bytes > 0)
      layout_chunk_size = l->artifact_chunk_bytes;
  }
  size_t slice_bytes = get_pool_chunk_size();
  if (slice_bytes == 0 || layout_chunk_size == 0) {
    return absl::FailedPreconditionError("slice_bytes or layout_chunk_size is zero");
  }
  if ((layout_chunk_size % slice_bytes) != 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat(
            "transfer_slice_bytes (%zu) must divide artifact_chunk_bytes (%zu) to avoid cross-chunk slices",
            slice_bytes,
            layout_chunk_size));
  }
  auto session_spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
      /*num_chunks=*/16, slice_bytes, pinned_pool_);
  auto init_status = session_spb->initialize(cfg_.pinned_memory_timeout);
  if (!init_status.ok()) {
    return init_status;
  }
  loader::StreamingBufferAdapter adapter(session_spb);
  auto sink = build_sink_(target_location, gpu_ptr_or_null, std::move(gpu_allocation), device_id);
  if (!sink) {
    return absl::FailedPreconditionError("Failed to construct sink for target location");
  }
  const auto start_tp = std::chrono::steady_clock::now();
  absl::Status pump_status = loader::pump_ranges(source, *sink, adapter, absl::MakeSpan(plan.ranges), concurrency);
  if (!pump_status.ok()) {
    return pump_status;
  }
  {
    absl::Status _st = sink->close();
    if (!_st.ok()) {
      LOG(WARNING) << "TransferService: sink->close failed: " << _st;
    }
  }
  const auto end_tp = std::chrono::steady_clock::now();
  const uint64_t dur_ms =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(end_tp - start_tp).count());
  uint64_t total_bytes = 0;
  for (const auto& r : plan.ranges)
    total_bytes += r.second;
  if (target_location == MemoryLocation::GPU && total_bytes > 0) {
    auto sync_status = synchronize_gpu_after_transfer(device_id, "TransferService::execute", total_bytes);
    if (!sync_status.ok()) {
      return sync_status;
    }
  }
  if (target_location == MemoryLocation::GPU && gpu_ptr_or_null != nullptr && device_id >= 0 && total_bytes > 0 &&
      VLOG_IS_ON(1)) {
    constexpr size_t kTailSampleBytes = 16;
    const size_t tail_len = static_cast<size_t>(std::min<uint64_t>(kTailSampleBytes, total_bytes));
    const size_t tail_offset = static_cast<size_t>(total_bytes - tail_len);
    std::array<uint8_t, kTailSampleBytes> tail_bytes{};
    auto set_dev_status = tensorcast::cuda::set_device(device_id);
    if (!set_dev_status.ok()) {
      VLOG(1) << "TransferService::execute tail sample set_device failed: " << set_dev_status;
    } else {
      const auto* gpu_bytes = static_cast<const uint8_t*>(gpu_ptr_or_null);
      auto copy_status =
          tensorcast::cuda::memcpy(tail_bytes.data(), gpu_bytes + tail_offset, tail_len, cudaMemcpyDeviceToHost);
      if (!copy_status.ok()) {
        VLOG(1) << "TransferService::execute tail sample memcpy failed: " << copy_status;
      } else {
        std::string tail_hex;
        tail_hex.reserve(tail_len * 3);
        for (size_t i = 0; i < tail_len; ++i) {
          if (i != 0) {
            tail_hex.push_back(' ');
          }
          absl::StrAppendFormat(&tail_hex, "%02x", tail_bytes[i]);
        }
        VLOG(1) << "TransferService::execute tail bytes gpu_off=" << tail_offset << " len=" << tail_len << " ["
                << tail_hex << "]";
      }
    }
  }
  // Record duration metric
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
    static auto tx_dur = meter->CreateDoubleHistogram("tc_tx_duration_ms");
    const char* target_str = (target_location == MemoryLocation::GPU) ? "GPU" : "CPU";
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    attrs.emplace("target", opentelemetry::common::AttributeValue(std::string(target_str)));
    tx_dur->Record(
        static_cast<double>(dur_ms),
        opentelemetry::common::KeyValueIterableView(attrs),
        opentelemetry::context::Context{});
  } catch (...) {
    VLOG(1) << "metrics histogram tc_tx_duration_ms unavailable";
  }
  VLOG(1) << "transfer_execute session=" << plan.session_id << " artifact_id=" << replica_key_.artifact_id
          << " target=" << static_cast<int>(target_location) << " bytes=" << total_bytes << " duration_ms=" << dur_ms;
  return absl::OkStatus();
}

} // namespace tensorcast::store::replica
