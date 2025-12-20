// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/sinks/gpu_memory_sink.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/async_copy_manager.h"
#include "core/common/device_guard.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/observer_result.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::loader {

namespace {
// Simple per-GPU copy scheduler with inflight bytes and copy count limits.
struct GpuSchedDev {
  size_t inflight_bytes{0};
  size_t inflight_copies{0};
  size_t limit_bytes{static_cast<size_t>(DEFAULT_GPU_SCHED_LIMIT_BYTES)};
  size_t limit_copies{static_cast<size_t>(DEFAULT_GPU_SCHED_LIMIT_COPIES)};
  uint64_t waits{0};
  double wait_sec{0.0};
  bool enabled{true};
  absl::CondVar cv;
};

ABSL_CONST_INIT absl::Mutex g_sched_mu(absl::kConstInit);
absl::flat_hash_map<int, std::unique_ptr<GpuSchedDev>> g_sched ABSL_GUARDED_BY(g_sched_mu);

// OTel gauge for inflight bytes
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> g_meter;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> g_if_bytes_gauge;

void inflight_bytes_callback(opentelemetry::metrics::ObserverResult result, void* /*state*/) noexcept {
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs)
    return;
  absl::MutexLock lk(&g_sched_mu);
  for (const auto& kv : g_sched) {
    const int gpu = kv.first;
    const GpuSchedDev* dev = kv.second.get();
    if (!dev)
      continue;
    obs->Observe(
        static_cast<double>(dev->inflight_bytes),
        {{"gpu", opentelemetry::common::AttributeValue(std::to_string(gpu))}});
  }
}

inline void ensure_metrics_init_() {
  if (!g_meter) {
    g_meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  }
  if (!g_if_bytes_gauge) {
    g_if_bytes_gauge = g_meter->CreateDoubleObservableGauge("tc_tx_inflight_bytes_gauge");
    g_if_bytes_gauge->AddCallback(&inflight_bytes_callback, nullptr);
  }
}

inline size_t normalize_sched_limit_(size_t limit, size_t min) {
  if (limit == 0) {
    return std::numeric_limits<size_t>::max();
  }
  return std::max(limit, min);
}

inline bool debug_copy_checks_enabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("TENSORCAST_DEBUG_GPU_COPY");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return enabled;
}

std::string bytes_to_hex(absl::Span<const uint8_t> bytes) {
  std::string out;
  out.reserve(bytes.size() * 3);
  for (size_t i = 0; i < bytes.size(); ++i) {
    if (i != 0) {
      out.push_back(' ');
    }
    absl::StrAppendFormat(&out, "%02x", bytes[i]);
  }
  return out;
}

struct DebugCopyPayload {
  int device_id;
  uint64_t gpu_offset;
  size_t bytes;
  char* gpu_ptr;
  std::vector<uint8_t> host_head;
  std::vector<uint8_t> host_tail;

  void verify() const {
    if (bytes == 0) {
      return;
    }
    auto set_dev_status = tensorcast::cuda::set_device(device_id);
    if (!set_dev_status.ok()) {
      LOG(ERROR) << "GpuMemorySink debug: cudaSetDevice failed device=" << device_id << " status=" << set_dev_status;
      return;
    }
    const size_t head_len = host_head.size();
    if (head_len > 0) {
      std::vector<uint8_t> gpu_head(head_len);
      auto head_status = tensorcast::cuda::memcpy(gpu_head.data(), gpu_ptr, head_len, cudaMemcpyDeviceToHost);
      if (!head_status.ok()) {
        LOG(ERROR) << "GpuMemorySink debug: head sampling failed device=" << device_id << " offset=" << gpu_offset
                   << " len=" << head_len << " status=" << head_status;
      } else if (std::memcmp(gpu_head.data(), host_head.data(), head_len) != 0) {
        LOG(ERROR) << "GpuMemorySink debug: head mismatch device=" << device_id << " offset=" << gpu_offset
                   << " bytes=" << bytes << " host_head=[" << bytes_to_hex(absl::MakeConstSpan(host_head))
                   << "] gpu_head=[" << bytes_to_hex(absl::MakeConstSpan(gpu_head)) << "]";
      }
    }
    if (!host_tail.empty()) {
      const size_t tail_len = host_tail.size();
      std::vector<uint8_t> gpu_tail(tail_len);
      auto tail_status =
          tensorcast::cuda::memcpy(gpu_tail.data(), gpu_ptr + (bytes - tail_len), tail_len, cudaMemcpyDeviceToHost);
      if (!tail_status.ok()) {
        LOG(ERROR) << "GpuMemorySink debug: tail sampling failed device=" << device_id
                   << " offset=" << (gpu_offset + bytes - tail_len) << " len=" << tail_len << " status=" << tail_status;
      } else if (std::memcmp(gpu_tail.data(), host_tail.data(), tail_len) != 0) {
        LOG(ERROR) << "GpuMemorySink debug: tail mismatch device=" << device_id
                   << " offset=" << (gpu_offset + bytes - tail_len) << " bytes=" << bytes << " host_tail=["
                   << bytes_to_hex(absl::MakeConstSpan(host_tail)) << "] gpu_tail=["
                   << bytes_to_hex(absl::MakeConstSpan(gpu_tail)) << "]";
      }
    }
  }
};

inline void sched_acquire(int device_id, size_t bytes, bool enabled, size_t limit_bytes, size_t limit_copies) {
  absl::MutexLock lk(&g_sched_mu);
  ensure_metrics_init_();
  auto& ptr = g_sched[device_id];
  if (!ptr)
    ptr = std::make_unique<GpuSchedDev>();
  GpuSchedDev* dev = ptr.get();

  const size_t next_limit_bytes = normalize_sched_limit_(limit_bytes, /*min=*/1);
  const size_t next_limit_copies = normalize_sched_limit_(limit_copies, /*min=*/1);
  const bool config_changed =
      (dev->enabled != enabled) || (dev->limit_bytes != next_limit_bytes) || (dev->limit_copies != next_limit_copies);
  dev->enabled = enabled;
  dev->limit_bytes = next_limit_bytes;
  dev->limit_copies = next_limit_copies;
  if (config_changed) {
    dev->cv.SignalAll();
  }
  if (!enabled) {
    return;
  }

  while ((dev->inflight_bytes + bytes > std::max(dev->limit_bytes, bytes)) ||
         (dev->inflight_copies >= dev->limit_copies)) {
    dev->waits += 1;
    const absl::Time t0 = absl::Now();
    dev->cv.Wait(&g_sched_mu);
    dev->wait_sec += absl::ToDoubleSeconds(absl::Now() - t0);
  }
  dev->inflight_bytes += bytes;
  dev->inflight_copies += 1;
  // Metrics hooks can be added here (omitted to avoid extra deps in loader).
}

inline void sched_release(int device_id, size_t bytes) {
  absl::MutexLock lk(&g_sched_mu);
  auto it = g_sched.find(device_id);
  if (it != g_sched.end() && it->second) {
    GpuSchedDev* dev = it->second.get();
    if (dev->inflight_bytes >= bytes)
      dev->inflight_bytes -= bytes;
    else
      dev->inflight_bytes = 0;
    if (dev->inflight_copies > 0)
      dev->inflight_copies -= 1;
    else
      dev->inflight_copies = 0;
    dev->cv.Signal();
  }
  // Metrics hooks can be added here (omitted to avoid extra deps in loader).
}
} // namespace

GpuSchedulerStats get_gpu_scheduler_stats(int device_id) {
  absl::MutexLock lk(&g_sched_mu);
  auto it = g_sched.find(device_id);
  if (it == g_sched.end() || !it->second) {
    return GpuSchedulerStats{};
  }
  const GpuSchedDev& dev = *it->second;
  return GpuSchedulerStats{
      .waits = dev.waits,
      .wait_sec = dev.wait_sec,
      .inflight_bytes = static_cast<uint64_t>(dev.inflight_bytes),
      .inflight_copies = static_cast<uint64_t>(dev.inflight_copies),
      .limit_bytes = static_cast<uint64_t>(dev.limit_bytes),
      .limit_copies = static_cast<uint64_t>(dev.limit_copies),
      .enabled = dev.enabled,
  };
}

void reset_gpu_scheduler_stats_for_testing() {
  absl::MutexLock lk(&g_sched_mu);
  g_sched.clear();
}

GpuMemorySink::GpuMemorySink(Options options) : options_(std::move(options)) {}

GpuMemorySink::~GpuMemorySink() {}

absl::Status GpuMemorySink::write(const void* src, size_t bytes) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  auto st = write_at(current_offset_, src, bytes);
  if (st.ok()) {
    current_offset_ += bytes;
  }
  return st;
}

absl::Status GpuMemorySink::write_at(uint64_t offset, const void* src, size_t bytes) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }

  if (options_.total_size > 0 && offset + bytes > options_.total_size) {
    return absl::InvalidArgumentError("Write would exceed total GPU memory size");
  }

  // Set device context
  common::DeviceGuard guard(options_.device_id);
  if (!guard.status().ok()) {
    overall_status_ = guard.status();
    return overall_status_;
  }

  // Submit H2D via AsyncCopyManager and wait for completion to preserve
  // synchronous write_at semantics for non-pump callers. Pump will use
  // write_at_async to avoid per-chunk waits and achieve overlap.
  char* gpu_dest = static_cast<char*>(options_.gpu_base_ptr.get()) + offset;
  common::HostRegion h{.base = src, .length = bytes, .pinned = true};
  common::DeviceRegion d{.device_id = options_.device_id, .dev_ptr = gpu_dest, .length = bytes};
  auto hdl_or = common::AsyncCopyManager::instance().submit_h2d(h, d, {.tracing_stage = "H2D/Copy"});
  if (!hdl_or.ok()) {
    overall_status_ = hdl_or.status();
    LOG(ERROR) << "Failed to schedule H2D copy: " << overall_status_;
    return overall_status_;
  }
  auto st = hdl_or->wait();
  if (!st.ok()) {
    overall_status_ = st;
    LOG(ERROR) << "H2D copy failed: " << st;
    return st;
  }

  VLOG(3) << "Copied " << bytes << " bytes to GPU at offset " << offset;
  // Maintain compatibility with tests that check completeness on close().
  total_bytes_written_ += bytes;
  return absl::OkStatus();
}

absl::StatusOr<common::CopyHandle> GpuMemorySink::write_at_async(
    uint64_t offset,
    const void* src,
    size_t bytes,
    const AsyncWriteOptions& options) {
  if (!overall_status_.ok()) {
    return overall_status_;
  }
  if (options_.total_size > 0 && offset + bytes > options_.total_size) {
    return absl::InvalidArgumentError("Write would exceed total GPU memory size");
  }
  // Set device context
  common::DeviceGuard guard(options_.device_id);
  if (!guard.status().ok()) {
    overall_status_ = guard.status();
    return overall_status_;
  }

  // Submit H2D via ACM; caller is responsible for waiting before reusing src
  // memory (e.g., pump will defer returning SPB slot until handle completes).
  // Apply per-GPU scheduling limits when enabled.
  sched_acquire(
      options_.device_id,
      bytes,
      options_.gpu_sched_enabled,
      static_cast<size_t>(options_.gpu_sched_limit_bytes),
      static_cast<size_t>(options_.gpu_sched_limit_copies));
  char* gpu_dest = static_cast<char*>(options_.gpu_base_ptr.get()) + offset;
  common::HostRegion h{.base = src, .length = bytes, .pinned = true};
  common::DeviceRegion d{.device_id = options_.device_id, .dev_ptr = gpu_dest, .length = bytes};
  common::CopyOptions effective_opts = options.copy_options.has_value() ? *options.copy_options : common::CopyOptions{};
  if (effective_opts.tracing_stage == nullptr) {
    effective_opts.tracing_stage = "H2D/Copy";
  }
  const bool debug_enabled = debug_copy_checks_enabled();
  auto release_token = [dev_id = options_.device_id, bytes]() { sched_release(dev_id, bytes); };
  std::shared_ptr<DebugCopyPayload> debug_payload;
  if (debug_enabled && bytes > 0) {
    const auto* host_ptr = static_cast<const uint8_t*>(src);
    debug_payload = std::make_shared<DebugCopyPayload>();
    debug_payload->device_id = options_.device_id;
    debug_payload->gpu_offset = offset;
    debug_payload->bytes = bytes;
    debug_payload->gpu_ptr = gpu_dest;
    const size_t head_len = std::min<size_t>(64, bytes);
    debug_payload->host_head.assign(host_ptr, host_ptr + head_len);
    if (bytes > head_len) {
      const size_t tail_len = std::min<size_t>(64, bytes);
      debug_payload->host_tail.assign(host_ptr + (bytes - tail_len), host_ptr + bytes);
    }
  }
  auto user_cb = effective_opts.callbacks.on_copy_done;
  auto user_inline_cb = effective_opts.callbacks.on_copy_done_inline;
  effective_opts.callbacks.on_copy_done_inline = [release_cb = std::move(release_token),
                                                  user_inline_cb](absl::Status st) {
    release_cb();
    if (user_inline_cb) {
      user_inline_cb(st);
    }
  };
  effective_opts.callbacks.on_copy_done = [debug_payload, user_cb](absl::Status st) {
    if (st.ok() && debug_payload) {
      debug_payload->verify();
    } else if (!st.ok() && debug_payload) {
      LOG(ERROR) << "GpuMemorySink async copy failed before debug verify device=" << debug_payload->device_id
                 << " offset=" << debug_payload->gpu_offset << " bytes=" << debug_payload->bytes << " status=" << st;
    }
    if (user_cb) {
      user_cb(st);
    }
  };

  // Tail sampling guard – only emit when we are within the final 16 bytes of the allocation.
  constexpr size_t kTailProbeBytes = 16;
  const size_t allocation_size = options_.allocation ? options_.allocation->size() : options_.total_size;
  const bool is_last_chunk = allocation_size > 0 && (offset + bytes >= allocation_size);
  const bool tail_debug = is_last_chunk && VLOG_IS_ON(1);
  std::array<uint8_t, kTailProbeBytes> host_tail{};
  size_t host_tail_len = 0;
  if (tail_debug && bytes > 0) {
    host_tail_len = std::min(kTailProbeBytes, bytes);
    const auto* src_bytes = static_cast<const uint8_t*>(src);
    std::memcpy(host_tail.data(), src_bytes + bytes - host_tail_len, host_tail_len);
  }

  auto hdl_or = common::AsyncCopyManager::instance().submit_h2d(h, d, effective_opts);
  if (!hdl_or.ok()) {
    // Release budget on immediate failure path
    sched_release(options_.device_id, bytes);
    overall_status_ = hdl_or.status();
    LOG(ERROR) << "Failed to schedule H2D async copy: " << overall_status_;
    return overall_status_;
  }
  auto handle = std::move(*hdl_or);
  // Maintain compatibility with tests and callers that validate completeness on close().
  // Callers are responsible for awaiting returned handles; pump_ranges does so before calling close().
  total_bytes_written_ += bytes;

  if (is_last_chunk) {
    auto wait_status = handle.wait();
    if (!wait_status.ok()) {
      LOG(WARNING) << "GpuMemorySink tail wait failed offset=" << offset << " bytes=" << bytes << ": " << wait_status;
      return wait_status;
    }
  }

  if (tail_debug) {
    std::array<uint8_t, kTailProbeBytes> gpu_tail{};
    size_t tail_len = host_tail_len;
    if (tail_len == 0 && bytes > 0) {
      tail_len = std::min(kTailProbeBytes, bytes);
    }

    if (tail_len > 0) {
      const size_t tail_offset_in_chunk = bytes - tail_len;
      auto copy_status =
          tensorcast::cuda::memcpy(gpu_tail.data(), gpu_dest + tail_offset_in_chunk, tail_len, cudaMemcpyDeviceToHost);
      if (!copy_status.ok()) {
        LOG(WARNING) << "GpuMemorySink tail probe memcpy failed offset=" << offset << " bytes=" << bytes << ": "
                     << copy_status;
      } else {
        std::string host_hex;
        std::string gpu_hex;
        host_hex.reserve(tail_len * 3);
        gpu_hex.reserve(tail_len * 3);
        for (size_t i = 0; i < tail_len; ++i) {
          if (i != 0) {
            host_hex.push_back(' ');
            gpu_hex.push_back(' ');
          }
          absl::StrAppendFormat(&host_hex, "%02x", host_tail[i]);
          absl::StrAppendFormat(&gpu_hex, "%02x", gpu_tail[i]);
        }
        const size_t global_tail_offset = offset + bytes - tail_len;
        VLOG(1) << "GpuMemorySink tail probe offset=" << offset << " bytes=" << bytes
                << " global_tail_off=" << global_tail_offset << " host_tail=[" << host_hex << "] gpu_tail=[" << gpu_hex
                << "]";
      }
    }
  }

  return handle;
}

absl::Status GpuMemorySink::close() {
  // No-op: All synchronization responsibility lies with callers via CopyHandles.
  // For synchronous write_at callers, we already waited for completion.
  if (!overall_status_.ok())
    return overall_status_;
  if (options_.total_size > 0 && total_bytes_written_ != options_.total_size) {
    LOG(ERROR) << "GPU memory sink closed with incomplete transfer. "
               << "Expected " << options_.total_size << " bytes, "
               << "but only " << total_bytes_written_ << " bytes were written.";
    return absl::OutOfRangeError("incomplete GPU transfer: bytes written do not match expected total");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::store::loader
