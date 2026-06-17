// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/runtime/pump.h"

#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/async_copy_manager.h"

#include <cstdlib>

#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::loader {

namespace {

pid_t current_tid() {
  return static_cast<pid_t>(::syscall(SYS_gettid));
}

int current_cpu() {
  return ::sched_getcpu();
}

int addr_numa_node(const void* addr) {
  if (addr == nullptr) {
    return -1;
  }
  int node = -1;
  long rc = ::syscall(SYS_get_mempolicy, &node, nullptr, 0, const_cast<void*>(addr), MPOL_F_NODE | MPOL_F_ADDR);
  if (rc != 0) {
    return -1;
  }
  return node;
}

struct PumpState {
  std::atomic<bool> should_stop{false};
  std::atomic<bool> drain_requested{false};
  std::atomic<uint64_t> next_chunk_id{0};
  std::atomic<uint64_t> produced_chunks_total{0};
  std::atomic<uint64_t> produced_bytes_total{0};
  std::atomic<uint64_t> producer_read_at_us_total{0};
  std::atomic<uint64_t> consumer_gpu_write_wait_us_total{0};
  std::atomic<uint64_t> consumer_gpu_write_bytes_total{0};
  absl::Status producer_status;
  absl::Status consumer_status;
  absl::Mutex status_mutex;
  // Map global_chunk_id -> destination offset for range pumping
  std::unordered_map<uint64_t, uint64_t> chunk_offsets;
  absl::Mutex offsets_mutex;
};

// Lazily created OTel counter for copy failures
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> g_meter;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_copy_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_bytes_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_direct_win_failures_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_direct_win_retry_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> g_direct_win_fallback_total;
opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> g_direct_win_duration_ms;

inline void record_copy_failure(const char* device) {
  if (!g_meter) {
    g_meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  }
  if (!g_copy_failures_total) {
    g_copy_failures_total = g_meter->CreateDoubleCounter("tc_tx_copy_failures_total");
  }
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("device", opentelemetry::common::AttributeValue(device));
  g_copy_failures_total->Add(
      1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
}

void record_direct_bytes(std::string_view mode, size_t bytes) {
  if (bytes == 0) {
    return;
  }
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("mode", opentelemetry::common::AttributeValue(std::string(mode)));
  g_bytes_total->Add(
      static_cast<double>(bytes),
      opentelemetry::common::KeyValueIterableView(attrs),
      opentelemetry::context::Context{});
}

void record_direct_failure(std::string_view reason) {
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string(reason)));
  g_direct_win_failures_total->Add(
      1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
}

void record_direct_fallback(std::string_view reason) {
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("reason", opentelemetry::common::AttributeValue(std::string(reason)));
  g_direct_win_fallback_total->Add(
      1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
}

size_t saturating_mul(size_t lhs, size_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > std::numeric_limits<size_t>::max() / rhs) {
    return std::numeric_limits<size_t>::max();
  }
  return lhs * rhs;
}

size_t resolve_direct_write_batch_bytes(size_t window_bytes, const PumpDirectWriteOptions& options) {
  const size_t requested = options.direct_write_batch_bytes;
  if (requested > 0) {
    return std::max(window_bytes, requested);
  }
  return std::max(window_bytes, saturating_mul(window_bytes, 8));
}

size_t resolve_direct_write_batch_ops(const PumpDirectWriteOptions& options) {
  return std::max<size_t>(1, options.direct_write_batch_ops > 0 ? options.direct_write_batch_ops : 8);
}

absl::Status staged_copy_direct_write_batch(
    SeekableSource& src,
    PositionedSink& dst,
    absl::Span<const DirectWriteOp> ops,
    size_t staging_chunk_bytes,
    std::vector<uint8_t>& staging_buf) {
  if (ops.empty()) {
    return absl::OkStatus();
  }
  if (staging_chunk_bytes == 0) {
    return absl::InvalidArgumentError("staging chunk bytes must be positive");
  }
  if (staging_buf.size() < staging_chunk_bytes) {
    staging_buf.resize(staging_chunk_bytes);
  }
  for (const auto& op : ops) {
    uint64_t src_offset = op.src_offset;
    uint64_t dest_offset = op.dest_va_offset;
    uint64_t remaining = op.bytes;
    while (remaining > 0) {
      const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, staging_buf.size()));
      auto got_or = src.read_at(src_offset, staging_buf.data(), chunk);
      if (!got_or.ok()) {
        return got_or.status();
      }
      if (*got_or == 0) {
        return absl::OutOfRangeError("EOF during staged direct-write fallback");
      }
      auto write_status = dst.write_at(dest_offset, staging_buf.data(), *got_or);
      if (!write_status.ok()) {
        return write_status;
      }
      record_direct_bytes("staged", *got_or);
      src_offset += *got_or;
      dest_offset += *got_or;
      remaining -= *got_or;
    }
  }
  return absl::OkStatus();
}

absl::Status pump_ranges_direct_write_impl(
    SeekableSource& src,
    PositionedSink& dst,
    absl::Span<const Range> ranges,
    size_t window_bytes,
    int concurrency,
    PumpDebugStats* debug_stats,
    PumpDirectWriteOptions direct_write_options) {
  if (!src.supports_direct_write_at()) {
    return absl::FailedPreconditionError("pump_ranges_direct_write requires a direct-write-capable source");
  }
  auto* cap = dynamic_cast<DirectWriteCapable*>(&dst);
  if (cap == nullptr) {
    return absl::FailedPreconditionError("pump_ranges_direct_write requires a DirectWriteCapable destination");
  }
  if (window_bytes == 0) {
    return absl::InvalidArgumentError("pump_ranges_direct_write requires window_bytes > 0");
  }

  const size_t batch_bytes_limit = resolve_direct_write_batch_bytes(window_bytes, direct_write_options);
  const size_t batch_ops_limit = resolve_direct_write_batch_ops(direct_write_options);
  std::size_t direct_batch_count = 0;
  std::size_t fallback_batch_count = 0;
  absl::Duration plan_direct_write_elapsed = absl::ZeroDuration();
  absl::Duration source_readv_elapsed = absl::ZeroDuration();
  absl::Duration staged_fallback_elapsed = absl::ZeroDuration();
  LOG(INFO) << "pump_ranges using direct-write path"
            << " ranges=" << ranges.size() << " concurrency=" << concurrency << " window_bytes=" << window_bytes
            << " batch_bytes_limit=" << batch_bytes_limit << " batch_ops_limit=" << batch_ops_limit;
  if (!g_meter) {
    g_meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  }
  if (!g_bytes_total) {
    g_bytes_total = g_meter->CreateDoubleCounter("tc_tx_bytes_total");
  }
  if (!g_direct_win_failures_total) {
    g_direct_win_failures_total = g_meter->CreateDoubleCounter("tc_tx_direct_window_failures_total");
  }
  if (!g_direct_win_retry_total) {
    g_direct_win_retry_total = g_meter->CreateDoubleCounter("tc_tx_direct_window_retry_total");
  }
  if (!g_direct_win_fallback_total) {
    g_direct_win_fallback_total = g_meter->CreateDoubleCounter("tc_tx_direct_window_fallback_total");
  }
  if (!g_direct_win_duration_ms) {
    g_direct_win_duration_ms = g_meter->CreateDoubleHistogram("tc_tx_direct_window_duration_ms");
  }

  std::vector<VaRange> batch_ranges;
  std::vector<DirectWriteOp> batch_ops;
  std::vector<uint8_t> staging_buf;
  size_t batch_bytes = 0;
  uint64_t total_direct_path_bytes = 0;
  uint64_t total_direct_path_chunks = 0;

  auto clear_batch = [&]() {
    batch_ranges.clear();
    batch_ops.clear();
    batch_bytes = 0;
  };
  auto flush_batch = [&]() -> absl::Status {
    if (batch_ops.empty()) {
      return absl::OkStatus();
    }
    const size_t expected_bytes = batch_bytes;
    const absl::Time plan_direct_write_started_at = absl::Now();
    auto grant_or = cap->plan_direct_write(batch_ranges);
    plan_direct_write_elapsed += absl::Now() - plan_direct_write_started_at;
    if (!grant_or.ok()) {
      record_direct_failure("pre_issue_plan");
      record_direct_fallback("pre_issue_plan");
      ++fallback_batch_count;
      const absl::Time staged_fallback_started_at = absl::Now();
      auto staged_status =
          staged_copy_direct_write_batch(src, dst, absl::MakeConstSpan(batch_ops), window_bytes, staging_buf);
      staged_fallback_elapsed += absl::Now() - staged_fallback_started_at;
      if (staged_status.ok()) {
        total_direct_path_bytes += expected_bytes;
        total_direct_path_chunks += batch_ops.size();
      }
      clear_batch();
      return staged_status;
    }

    ++direct_batch_count;
    const absl::Time direct_started_at = absl::Now();
    auto got_or = src.readv_into_at(absl::MakeConstSpan(batch_ops), *grant_or);
    source_readv_elapsed += absl::Now() - direct_started_at;
    if (!got_or.ok()) {
      record_direct_failure("post_issue_readv");
      return got_or.status();
    }
    if (*got_or != expected_bytes) {
      record_direct_failure("post_issue_short_write");
      return absl::DataLossError("Short direct write batch");
    }
    record_direct_bytes("direct", *got_or);
    total_direct_path_bytes += *got_or;
    total_direct_path_chunks += batch_ops.size();
    const std::map<std::string, opentelemetry::common::AttributeValue> attrs{
        {"mode", opentelemetry::common::AttributeValue("direct")}};
    g_direct_win_duration_ms->Record(
        absl::ToDoubleMilliseconds(absl::Now() - direct_started_at),
        opentelemetry::common::KeyValueIterableView(attrs),
        opentelemetry::context::Context{});
    clear_batch();
    return absl::OkStatus();
  };

  for (const auto& [range_off, range_len] : ranges) {
    uint64_t off = range_off;
    const uint64_t end = range_off + range_len;
    while (off < end) {
      const size_t step = static_cast<size_t>(std::min<uint64_t>(window_bytes, end - off));
      const bool bytes_limit_hit = !batch_ops.empty() && batch_bytes + step > batch_bytes_limit;
      const bool ops_limit_hit = !batch_ops.empty() && batch_ops.size() >= batch_ops_limit;
      if (bytes_limit_hit || ops_limit_hit) {
        auto flush_status = flush_batch();
        if (!flush_status.ok()) {
          return flush_status;
        }
      }
      batch_ranges.push_back(VaRange{off, step});
      batch_ops.push_back(
          DirectWriteOp{
              .src_offset = off,
              .dest_va_offset = off,
              .bytes = step,
          });
      batch_bytes += step;
      off += step;
    }
  }
  auto flush_status = flush_batch();
  if (!flush_status.ok()) {
    return flush_status;
  }
  if (debug_stats != nullptr) {
    debug_stats->produced_chunks = total_direct_path_chunks;
    debug_stats->produced_bytes = total_direct_path_bytes;
  }
  VLOG(2) << "pump_ranges.direct_write_summary"
          << " ranges=" << ranges.size() << " concurrency=" << concurrency << " window_bytes=" << window_bytes
          << " batch_bytes_limit=" << batch_bytes_limit << " batch_ops_limit=" << batch_ops_limit
          << " direct_batches=" << direct_batch_count << " fallback_batches=" << fallback_batch_count
          << " total_chunks=" << total_direct_path_chunks << " total_bytes=" << total_direct_path_bytes
          << " plan_direct_write_ms=" << absl::ToDoubleMilliseconds(plan_direct_write_elapsed)
          << " source_readv_ms=" << absl::ToDoubleMilliseconds(source_readv_elapsed)
          << " staged_fallback_ms=" << absl::ToDoubleMilliseconds(staged_fallback_elapsed);
  return absl::OkStatus();
}

// RAII lease for a pool slot to guarantee return on all paths
class SlotLease {
 public:
  SlotLease(BufferPool& pool, int slot_id) : pool_(pool), slot_id_(slot_id), active_(true) {}

  SlotLease(const SlotLease&) = delete;
  SlotLease& operator=(const SlotLease&) = delete;

  ~SlotLease() {
    if (active_) {
      pool_.return_chunk(slot_id_);
    }
  }

  int id() const {
    return slot_id_;
  }

  void release() {
    active_ = false;
  }

 private:
  BufferPool& pool_;
  int slot_id_;
  bool active_;
};

static void run_consumer(PositionedSink& dst, BufferPool& pool, PumpState& state) {
  VLOG(1) << "Consumer thread started tid=" << current_tid() << " cpu=" << current_cpu();
  bool draining = false;

  struct AsyncSlot {
    AsyncSlot(
        BufferPool& pool_in,
        int slot_in,
        uint64_t chunk_in,
        uint64_t dest_in,
        size_t bytes_in,
        absl::Time submitted_at_in)
        : pool(pool_in),
          slot_id(slot_in),
          global_chunk_id(chunk_in),
          dest_offset(dest_in),
          bytes(bytes_in),
          submitted_at(submitted_at_in),
          lease(pool_in, slot_in) {}

    void return_if_needed() {
      bool expected = false;
      if (!returned.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
      }
      lease.release();
      pool.return_chunk(slot_id);
    }

    void record_status(const absl::Status& st) {
      {
        absl::MutexLock lock(&status_mutex);
        callback_status = st;
        completed_at = absl::Now();
        has_status = true;
      }
      status_cv.SignalAll();
    }

    absl::Status await_final_status(const absl::Status& fallback) {
      absl::MutexLock lock(&status_mutex);
      while (!has_status) {
        status_cv.Wait(&status_mutex);
      }
      if (callback_status.ok()) {
        return fallback;
      }
      return callback_status;
    }

    absl::Duration copy_elapsed() {
      absl::MutexLock lock(&status_mutex);
      if (!has_status) {
        return absl::ZeroDuration();
      }
      return completed_at - submitted_at;
    }

    void request_pool_shutdown() {
      pool.shutdown();
    }

    BufferPool& pool;
    int slot_id;
    uint64_t global_chunk_id;
    uint64_t dest_offset;
    size_t bytes;
    absl::Time submitted_at;
    SlotLease lease;
    std::atomic<bool> returned{false};
    absl::Mutex status_mutex;
    absl::CondVar status_cv;
    absl::Status callback_status ABSL_GUARDED_BY(status_mutex) = absl::OkStatus();
    absl::Time completed_at ABSL_GUARDED_BY(status_mutex);
    bool has_status ABSL_GUARDED_BY(status_mutex) = false;
  };

  // Track in-flight async copies when destination supports async writes
  struct InFlight {
    tensorcast::common::CopyHandle handle;
    int slot_id;
    uint64_t global_chunk_id;
    uint64_t dest_offset;
    size_t bytes;
    std::shared_ptr<AsyncSlot> slot_ctx;
  };

  std::vector<InFlight> inflight;
  auto* async = dynamic_cast<AsyncPositionedSink*>(&dst);
  auto finalize_inflight = [&](InFlight& entry, const absl::Status& handle_status) -> bool {
    absl::Status st = handle_status;
    if (entry.slot_ctx) {
      st = entry.slot_ctx->await_final_status(handle_status);
    }
    if (entry.slot_ctx) {
      state.consumer_gpu_write_wait_us_total.fetch_add(
          static_cast<uint64_t>(std::max<int64_t>(0, absl::ToInt64Microseconds(entry.slot_ctx->copy_elapsed()))),
          std::memory_order_relaxed);
      state.consumer_gpu_write_bytes_total.fetch_add(entry.bytes, std::memory_order_relaxed);
    }
    const bool failed = !st.ok();
    if (failed) {
      {
        absl::MutexLock lock(&state.status_mutex);
        if (state.consumer_status.ok()) {
          state.consumer_status = st;
        }
      }
      state.should_stop.store(true, std::memory_order_release);
      state.drain_requested.store(true, std::memory_order_release);
      pool.shutdown();
      draining = true;
    }
    if (entry.slot_ctx) {
      entry.slot_ctx->return_if_needed();
    }
    VLOG(2) << "pump_async_completion slot=" << entry.slot_id << " chunk_id=" << entry.global_chunk_id
            << " dest_offset=" << entry.dest_offset << " bytes=" << entry.bytes << " status=" << st;
    return true;
  };
  auto flush_inflight = [&]() {
    if (inflight.empty()) {
      return;
    }
    for (auto& entry : inflight) {
      absl::Status st = entry.handle.wait();
      finalize_inflight(entry, st);
    }
    inflight.clear();
  };
  absl::Cleanup flush_guard = [&]() { flush_inflight(); };
  while (true) {
    if (state.drain_requested.load(std::memory_order_acquire)) {
      draining = true;
    }
    if (state.should_stop.load(std::memory_order_acquire) && !draining) {
      break;
    }
    auto chunk_result = pool.get_ready_chunk();
    if (!chunk_result.ok()) {
      if (absl::IsUnavailable(chunk_result.status())) {
        // No more chunks available (EOF)
        LOG(INFO) << "Consumer: No more chunks (Unavailable status)";
        break;
      }
      if (absl::IsOutOfRange(chunk_result.status())) {
        LOG(INFO) << "Consumer: No more chunks (OutOfRange status)";
        break;
      }
      LOG(ERROR) << "Consumer failed to get ready chunk: " << chunk_result.status();
      absl::MutexLock lock(&state.status_mutex);
      if (state.consumer_status.ok()) {
        state.consumer_status = chunk_result.status();
      }
      state.should_stop.store(true, std::memory_order_release);
      // Wake producers/consumers and terminate to avoid spinning on the same error
      pool.shutdown();
      break;
    }

    const auto& chunk = *chunk_result;
    uint64_t dest_offset = 0;
    {
      absl::MutexLock lock(&state.offsets_mutex);
      auto it = state.chunk_offsets.find(chunk.global_chunk_id);
      if (it == state.chunk_offsets.end()) {
        absl::MutexLock s(&state.status_mutex);
        state.consumer_status = absl::InternalError("Missing destination offset for ready chunk");
        state.should_stop.store(true, std::memory_order_release);
        pool.return_chunk(chunk.slot_id);
        break;
      }
      dest_offset = it->second;
      state.chunk_offsets.erase(it);
    }

    if (async != nullptr) {
      // Submit async write and defer returning the slot until completion.
      AsyncPositionedSink::AsyncWriteOptions write_opts;
      tensorcast::common::CopyOptions copy_opts;
      copy_opts.tracing_stage = "Pump/H2D";
      const int slot_id = chunk.slot_id;
      const uint64_t chunk_id = chunk.global_chunk_id;
      const size_t chunk_bytes = chunk.bytes_in_chunk;
      write_opts.copy_options = copy_opts;
      auto slot_ctx = std::make_shared<AsyncSlot>(pool, slot_id, chunk_id, dest_offset, chunk_bytes, absl::Now());
      write_opts.copy_options->callbacks.on_copy_done = [slot_ctx](absl::Status st) {
        slot_ctx->record_status(st);
        slot_ctx->return_if_needed();
        if (!st.ok()) {
          record_copy_failure("gpu");
          slot_ctx->request_pool_shutdown();
        }
        VLOG(2) << "pump_async_callback slot=" << slot_ctx->slot_id << " chunk_id=" << slot_ctx->global_chunk_id
                << " dest_offset=" << slot_ctx->dest_offset << " bytes=" << slot_ctx->bytes << " status=" << st;
      };
      auto hdl_or = async->write_at_async(dest_offset, chunk.data_ptr, chunk.bytes_in_chunk, write_opts);
      if (!hdl_or.ok()) {
        record_copy_failure("gpu");
        absl::MutexLock lock(&state.status_mutex);
        if (state.consumer_status.ok()) {
          state.consumer_status = hdl_or.status();
        }
        // Return the slot and request shutdown
        slot_ctx->return_if_needed();
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        // Switch to drain mode to return remaining ready chunks
        draining = true;
      } else {
        inflight.push_back(
            InFlight{
                .handle = std::move(*hdl_or),
                .slot_id = chunk.slot_id,
                .global_chunk_id = chunk.global_chunk_id,
                .dest_offset = dest_offset,
                .bytes = chunk.bytes_in_chunk,
                .slot_ctx = std::move(slot_ctx)});
      }
    } else {
      const absl::Time sync_write_started_at = absl::Now();
      auto status = dst.write_at(dest_offset, chunk.data_ptr, chunk.bytes_in_chunk);
      state.consumer_gpu_write_wait_us_total.fetch_add(
          static_cast<uint64_t>(std::max<int64_t>(0, absl::ToInt64Microseconds(absl::Now() - sync_write_started_at))),
          std::memory_order_relaxed);
      state.consumer_gpu_write_bytes_total.fetch_add(chunk.bytes_in_chunk, std::memory_order_relaxed);
      pool.return_chunk(chunk.slot_id);
      if (!status.ok()) {
        absl::MutexLock lock(&state.status_mutex);
        if (state.consumer_status.ok()) {
          state.consumer_status = status;
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        draining = true;
      }
    }

    // Sweep in-flight operations: process any handle that is completed
    // (success or failure). Skip only pending operations by probing with a
    // zero-timeout wait.
    if (!inflight.empty()) {
      for (size_t i = 0; i < inflight.size();) {
        absl::Status st = inflight[i].handle.wait(absl::ZeroDuration());
        if (absl::IsDeadlineExceeded(st)) {
          ++i; // still pending
          continue;
        }
        finalize_inflight(inflight[i], st);
        inflight.erase(inflight.begin() + i);
      }
    }
  }

  // Drain any remaining in-flight operations at shutdown
  for (auto& item : inflight) {
    absl::Status st = item.handle.wait();
    finalize_inflight(item, st);
  }
  inflight.clear();
}

void run_range_producer(
    SeekableSource& src,
    BufferPool& pool,
    absl::Span<const std::pair<uint64_t, size_t>> ranges,
    std::atomic<size_t>& range_index,
    PumpState& state) {
  const absl::Time producer_start = absl::Now();
  VLOG(2) << "pump_producer_start tid=" << current_tid() << " cpu=" << current_cpu();
  uint64_t produced_chunks = 0;
  uint64_t produced_bytes = 0;
  uint64_t wait_free_chunk_us_total = 0;
  uint64_t read_at_us_total = 0;
  uint64_t mark_ready_us_total = 0;

  const auto fail_producer = [&](absl::Status status) {
    absl::MutexLock lock(&state.status_mutex);
    if (state.producer_status.ok()) {
      state.producer_status = std::move(status);
    }
    state.should_stop.store(true, std::memory_order_release);
    state.drain_requested.store(true, std::memory_order_release);
    pool.shutdown();
  };

  while (!state.should_stop.load(std::memory_order_acquire)) {
    size_t idx = range_index.fetch_add(1, std::memory_order_acq_rel);
    if (idx >= ranges.size()) {
      break;
    }

    const auto& [offset, size] = ranges[idx];
    size_t remaining = size;
    uint64_t current_offset = offset;

    while (remaining > 0 && !state.should_stop.load(std::memory_order_acquire)) {
      const absl::Time wait_free_start = absl::Now();
      auto slot_result = pool.get_free_chunk();
      const uint64_t wait_free_us =
          static_cast<uint64_t>(std::max<int64_t>(0, absl::ToInt64Microseconds(absl::Now() - wait_free_start)));
      wait_free_chunk_us_total += wait_free_us;
      if (!slot_result.ok()) {
        fail_producer(slot_result.status());
        break;
      }

      int slot_id = *slot_result;
      // Ensure slot is returned on any early exit
      SlotLease lease(pool, slot_id);

      if (state.should_stop.load(std::memory_order_acquire)) {
        // Respect cancellation promptly
        break;
      }
      size_t to_read = std::min(remaining, pool.chunk_size());

      // Get buffer pointer from pool using interface method
      void* buffer = pool.get_chunk_data_ptr(slot_id);
      if (!buffer) {
        fail_producer(absl::InternalError("Failed to get chunk buffer pointer"));
        break;
      }

      const absl::Time read_start = absl::Now();
      auto read_result = src.read_at(current_offset, buffer, to_read);
      const uint64_t read_us =
          static_cast<uint64_t>(std::max<int64_t>(0, absl::ToInt64Microseconds(absl::Now() - read_start)));
      read_at_us_total += read_us;
      if (!read_result.ok()) {
        fail_producer(read_result.status());
        break;
      }

      size_t bytes_read = *read_result;
      if (bytes_read == 0) {
        LOG(WARNING) << "Unexpected EOF at offset " << current_offset;
        // Treat as error to propagate failure instead of silently succeeding.
        fail_producer(absl::OutOfRangeError("Unexpected EOF while reading source"));
        break;
      }

      // Validate that Source respects requested read size limits
      if (bytes_read > to_read) {
        fail_producer(absl::InvalidArgumentError("Source returned more bytes than requested"));
        break;
      }

      auto chunk_id = state.next_chunk_id.fetch_add(1, std::memory_order_acq_rel);

      // Check for overflow - use max value as error indicator
      if (chunk_id == std::numeric_limits<uint64_t>::max()) {
        fail_producer(absl::ResourceExhaustedError("Chunk ID overflow"));
        break;
      }

      // Record destination offset for this produced chunk
      {
        absl::MutexLock lock(&state.offsets_mutex);
        state.chunk_offsets.emplace(chunk_id, current_offset);
      }

      const absl::Time mark_ready_start = absl::Now();
      auto status = pool.mark_chunk_ready(slot_id, chunk_id, bytes_read);
      const uint64_t mark_ready_us =
          static_cast<uint64_t>(std::max<int64_t>(0, absl::ToInt64Microseconds(absl::Now() - mark_ready_start)));
      mark_ready_us_total += mark_ready_us;
      if (!status.ok()) {
        record_copy_failure("cpu");
        fail_producer(status);
        break;
      }

      // Emit lightweight diagnostics for chunk payload (first/last 8 bytes).
      current_offset += bytes_read;
      remaining -= bytes_read;
      // Transfer ownership to consumer; avoid returning the slot here
      lease.release();
      produced_chunks += 1;
      produced_bytes += bytes_read;
      VLOG(2) << "pump_producer_chunk range_index=" << idx << " chunk_id=" << chunk_id << " slot=" << slot_id
              << " src_offset=" << (current_offset - bytes_read) << " bytes=" << bytes_read
              << " wait_free_chunk_us=" << wait_free_us << " read_at_us=" << read_us
              << " mark_chunk_ready_us=" << mark_ready_us << " remaining_in_range=" << remaining
              << " tid=" << current_tid() << " cpu=" << current_cpu() << " buffer_numa=" << addr_numa_node(buffer);
    }
  }
  VLOG(2) << "pump_producer_summary ranges=" << ranges.size() << " produced_chunks=" << produced_chunks
          << " produced_bytes=" << produced_bytes << " wait_free_chunk_us_total=" << wait_free_chunk_us_total
          << " read_at_us_total=" << read_at_us_total << " mark_chunk_ready_us_total=" << mark_ready_us_total
          << " duration_us="
          << static_cast<uint64_t>(std::max<int64_t>(0, absl::ToInt64Microseconds(absl::Now() - producer_start)))
          << " tid=" << current_tid() << " cpu=" << current_cpu();
  state.produced_chunks_total.fetch_add(produced_chunks, std::memory_order_relaxed);
  state.produced_bytes_total.fetch_add(produced_bytes, std::memory_order_relaxed);
  state.producer_read_at_us_total.fetch_add(read_at_us_total, std::memory_order_relaxed);
}

} // namespace

absl::Status pump_ranges_direct_write(
    SeekableSource& src,
    PositionedSink& dst,
    absl::Span<const Range> ranges,
    size_t window_bytes,
    int concurrency,
    PumpDebugStats* debug_stats,
    PumpDirectWriteOptions direct_write_options) {
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }
  if (ranges.empty()) {
    return absl::InvalidArgumentError("Ranges cannot be empty");
  }
  return pump_ranges_direct_write_impl(src, dst, ranges, window_bytes, concurrency, debug_stats, direct_write_options);
}

absl::Status pump_ranges(
    SeekableSource& src,
    PositionedSink& dst,
    BufferPool& pool,
    absl::Span<const Range> ranges,
    int concurrency,
    folly::Executor::KeepAlive<> executor,
    PumpDebugStats* debug_stats,
    PumpDirectWriteOptions direct_write_options) {
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }
  if (ranges.empty()) {
    return absl::InvalidArgumentError("Ranges cannot be empty");
  }
  if (!executor) {
    return absl::InvalidArgumentError("pump_ranges requires a non-null executor");
  }

  // Capability-driven direct-write path: if destination implements
  // DirectWriteCapable and source supports direct writes (e.g., RDMA),
  // plan windowed grants and stream directly into destination VA ranges.
  if (src.supports_direct_write_at()) {
    if (dynamic_cast<DirectWriteCapable*>(&dst) != nullptr) {
      auto direct_status = pump_ranges_direct_write_impl(
          src, dst, ranges, pool.chunk_size(), concurrency, debug_stats, direct_write_options);
      if (!direct_status.ok()) {
        return direct_status;
      }
      if (auto* base_sink = dynamic_cast<Sink*>(&dst)) {
        return base_sink->close();
      }
      return dst.close();
    }
    LOG(INFO) << "pump_ranges direct-write unavailable: destination is not DirectWriteCapable"
              << " ranges=" << ranges.size() << " concurrency=" << concurrency;
  }

  if (!src.supports_direct_write_at()) {
    LOG(INFO) << "pump_ranges staged path: source does not support direct-write"
              << " ranges=" << ranges.size() << " concurrency=" << concurrency << " chunk_bytes=" << pool.chunk_size();
  }

  PumpState state;
  std::atomic<size_t> range_index{0};
  auto producers_done = std::make_shared<absl::BlockingCounter>(concurrency);
  auto producers_remaining = std::make_shared<std::atomic<int>>(concurrency);

  for (int i = 0; i < concurrency; ++i) {
    executor->add([&src, &pool, ranges, &range_index, &state, producers_done, producers_remaining]() {
      run_range_producer(src, pool, ranges, range_index, state);
      if (producers_remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
        pool.signal_production_complete();
      }
      producers_done->DecrementCount();
    });
  }

  run_consumer(dst, pool, state);

  producers_done->Wait();

  // Producer-side failures can race with consumer shutdown: late ready chunks
  // may be enqueued after the consumer observed end-of-stream. Drain any
  // remaining ready slots so StreamingPinnedBuffer::release can reclaim all
  // pinned slices deterministically.
  if (state.should_stop.load(std::memory_order_acquire) || state.drain_requested.load(std::memory_order_acquire)) {
    while (true) {
      auto chunk_result = pool.get_ready_chunk();
      if (!chunk_result.ok()) {
        if (absl::IsOutOfRange(chunk_result.status()) || absl::IsUnavailable(chunk_result.status())) {
          break;
        }
        LOG(WARNING) << "pump_ranges cleanup drain failed to fetch ready chunk: " << chunk_result.status();
        break;
      }
      pool.return_chunk(chunk_result->slot_id);
    }
  }

  // Close the sink
  // Attempt to close if dst also implements Sink
  absl::Status close_status = absl::OkStatus();
  if (auto* base_sink = dynamic_cast<Sink*>(&dst)) {
    close_status = base_sink->close();
  }

  // Return first error encountered
  {
    absl::MutexLock lock(&state.status_mutex);
    if (!state.consumer_status.ok()) {
      LOG(ERROR) << "pump_ranges returning consumer error: " << state.consumer_status;
      return state.consumer_status;
    }
    if (!state.producer_status.ok()) {
      LOG(ERROR) << "pump_ranges returning producer error: " << state.producer_status;
      return state.producer_status;
    }
  }

  if (debug_stats != nullptr) {
    debug_stats->produced_chunks = state.produced_chunks_total.load(std::memory_order_relaxed);
    debug_stats->produced_bytes = state.produced_bytes_total.load(std::memory_order_relaxed);
    debug_stats->source_read_at_us_total = state.producer_read_at_us_total.load(std::memory_order_relaxed);
    debug_stats->gpu_write_wait_us_total = state.consumer_gpu_write_wait_us_total.load(std::memory_order_relaxed);
    debug_stats->gpu_write_bytes_total = state.consumer_gpu_write_bytes_total.load(std::memory_order_relaxed);
  }

  return close_status;
}

} // namespace tensorcast::store::loader
