// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/runtime/pump.h"

#include <atomic>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

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

struct PumpState {
  std::atomic<bool> should_stop{false};
  std::atomic<bool> drain_requested{false};
  std::atomic<uint64_t> next_chunk_id{0};
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
  VLOG(1) << "Consumer thread started";
  bool draining = false;

  struct AsyncSlot {
    AsyncSlot(BufferPool& pool_in, int slot_in, uint64_t chunk_in, uint64_t dest_in, size_t bytes_in)
        : pool(pool_in),
          slot_id(slot_in),
          global_chunk_id(chunk_in),
          dest_offset(dest_in),
          bytes(bytes_in),
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
        has_status = true;
      }
      status_cv.SignalAll();
    }

    absl::Status await_final_status(const absl::Status& fallback) {
      absl::MutexLock lock(&status_mutex);
      while (!has_status) {
        status_cv.Wait(&status_mutex);
      }
      has_status = false;
      if (callback_status.ok()) {
        return fallback;
      }
      return callback_status;
    }

    void request_pool_shutdown() {
      pool.shutdown();
    }

    BufferPool& pool;
    int slot_id;
    uint64_t global_chunk_id;
    uint64_t dest_offset;
    size_t bytes;
    SlotLease lease;
    std::atomic<bool> returned{false};
    absl::Mutex status_mutex;
    absl::CondVar status_cv;
    absl::Status callback_status ABSL_GUARDED_BY(status_mutex) = absl::OkStatus();
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
  while (!state.should_stop.load(std::memory_order_acquire) || draining) {
    if (state.drain_requested.load(std::memory_order_acquire)) {
      draining = true;
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
      auto slot_ctx = std::make_shared<AsyncSlot>(pool, slot_id, chunk_id, dest_offset, chunk_bytes);
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
      auto status = dst.write_at(dest_offset, chunk.data_ptr, chunk.bytes_in_chunk);
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
  while (!state.should_stop.load(std::memory_order_acquire)) {
    size_t idx = range_index.fetch_add(1, std::memory_order_acq_rel);
    if (idx >= ranges.size()) {
      break;
    }

    const auto& [offset, size] = ranges[idx];
    size_t remaining = size;
    uint64_t current_offset = offset;

    while (remaining > 0 && !state.should_stop.load(std::memory_order_acquire)) {
      auto slot_result = pool.get_free_chunk();
      if (!slot_result.ok()) {
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = slot_result.status();
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
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
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = absl::InternalError("Failed to get chunk buffer pointer");
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        break;
      }

      auto read_result = src.read_at(current_offset, buffer, to_read);
      if (!read_result.ok()) {
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = read_result.status();
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        break;
      }

      size_t bytes_read = *read_result;
      if (bytes_read == 0) {
        LOG(WARNING) << "Unexpected EOF at offset " << current_offset;
        // Treat as error to propagate failure instead of silently succeeding
        {
          absl::MutexLock lock(&state.status_mutex);
          if (state.producer_status.ok()) {
            state.producer_status = absl::OutOfRangeError("Unexpected EOF while reading source");
          }
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        break;
      }

      // Validate that Source respects requested read size limits
      if (bytes_read > to_read) {
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = absl::InvalidArgumentError("Source returned more bytes than requested");
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        break;
      }

      auto chunk_id = state.next_chunk_id.fetch_add(1, std::memory_order_acq_rel);

      // Check for overflow - use max value as error indicator
      if (chunk_id == std::numeric_limits<uint64_t>::max()) {
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = absl::ResourceExhaustedError("Chunk ID overflow");
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        break;
      }

      // Record destination offset for this produced chunk
      {
        absl::MutexLock lock(&state.offsets_mutex);
        state.chunk_offsets.emplace(chunk_id, current_offset);
      }

      auto status = pool.mark_chunk_ready(slot_id, chunk_id, bytes_read);
      if (!status.ok()) {
        record_copy_failure("cpu");
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = status;
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        break;
      }

      // Emit lightweight diagnostics for chunk payload (first/last 8 bytes).
      current_offset += bytes_read;
      remaining -= bytes_read;
      // Transfer ownership to consumer; avoid returning the slot here
      lease.release();
    }
  }
}

} // namespace

absl::Status pump_ranges(
    SeekableSource& src,
    PositionedSink& dst,
    BufferPool& pool,
    absl::Span<const Range> ranges,
    int concurrency,
    folly::Executor::KeepAlive<> executor) {
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
    if (auto* cap = dynamic_cast<DirectWriteCapable*>(&dst)) {
      const size_t window_bytes = pool.chunk_size();
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
      // Convert to VaRanges
      std::vector<VaRange> va_ranges;
      va_ranges.reserve(ranges.size());
      for (const auto& r : ranges)
        va_ranges.push_back({r.first, r.second});

      // Always use sliding window mode in V3 final state.

      // Sliding window mode: plan small per-window tokens to reduce lease lifetime.
      for (const auto& [range_off, range_len] : ranges) {
        uint64_t off = range_off;
        uint64_t end = range_off + range_len;
        bool range_staged_fallback = false;
        std::vector<uint8_t> staging_buf; // allocated on first fallback use
        while (off < end) {
          size_t step = static_cast<size_t>(std::min<uint64_t>(window_bytes, end - off));
          VaRange win{off, step};
          auto grant_or = cap->plan_direct_write(absl::Span<const VaRange>(&win, 1));
          if (!grant_or.ok()) {
            // classify as failure event and fallback for remaining range
            {
              std::map<std::string, opentelemetry::common::AttributeValue> attr{
                  {"reason", opentelemetry::common::AttributeValue("plan")}};
              g_direct_win_failures_total->Add(
                  1.0, opentelemetry::common::KeyValueIterableView(attr), opentelemetry::context::Context{});
            }
            range_staged_fallback = true;
          } else {
            auto t0 = absl::Now();
            auto got = src.read_into_at(off, off, step, *grant_or);
            if (!got.ok()) {
              // retry once on recoverable errors
              auto code = got.status().code();
              bool retriable =
                  (code == absl::StatusCode::kUnavailable || code == absl::StatusCode::kDeadlineExceeded ||
                   code == absl::StatusCode::kAborted || code == absl::StatusCode::kResourceExhausted ||
                   code == absl::StatusCode::kInternal);
              if (retriable) {
                static const std::map<std::string, opentelemetry::common::AttributeValue> kEmptyAttrs;
                g_direct_win_retry_total->Add(
                    1.0, opentelemetry::common::KeyValueIterableView(kEmptyAttrs), opentelemetry::context::Context{});
                got = src.read_into_at(off, off, step, *grant_or);
              }
              if (!got.ok()) {
                // mark failure and fallback sticky for remaining of this range
                static const std::map<std::string, opentelemetry::common::AttributeValue> kEmptyAttrs;
                g_direct_win_failures_total->Add(
                    1.0, opentelemetry::common::KeyValueIterableView(kEmptyAttrs), opentelemetry::context::Context{});
                range_staged_fallback = true;
              } else {
                if (*got != step)
                  return absl::DataLossError("Short direct write (window)");
                // metrics for successful window
                std::map<std::string, opentelemetry::common::AttributeValue> attrs;
                attrs.emplace("mode", opentelemetry::common::AttributeValue("direct"));
                g_bytes_total->Add(
                    static_cast<double>(*got),
                    opentelemetry::common::KeyValueIterableView(attrs),
                    opentelemetry::context::Context{});
                const double ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
                g_direct_win_duration_ms->Record(
                    ms, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
                off += step;
              }
            } else {
              if (*got != step)
                return absl::DataLossError("Short direct write (window)");
              std::map<std::string, opentelemetry::common::AttributeValue> attrs;
              attrs.emplace("mode", opentelemetry::common::AttributeValue("direct"));
              g_bytes_total->Add(
                  static_cast<double>(*got),
                  opentelemetry::common::KeyValueIterableView(attrs),
                  opentelemetry::context::Context{});
              const double ms = absl::ToDoubleMilliseconds(absl::Now() - t0);
              g_direct_win_duration_ms->Record(
                  ms, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
              off += step;
            }
          }

          if (range_staged_fallback) {
            static const std::map<std::string, opentelemetry::common::AttributeValue> kEmptyAttrs;
            g_direct_win_fallback_total->Add(
                1.0, opentelemetry::common::KeyValueIterableView(kEmptyAttrs), opentelemetry::context::Context{});
            // staged copy for remaining bytes in this range [off .. end)
            if (staging_buf.empty())
              staging_buf.resize(pool.chunk_size());
            while (off < end) {
              const size_t chunk = static_cast<size_t>(std::min<uint64_t>(staging_buf.size(), end - off));
              // read from source into host buffer, then write via PositionedSink
              auto got2 = src.read_at(off, staging_buf.data(), chunk);
              if (!got2.ok())
                return got2.status();
              if (*got2 == 0)
                return absl::OutOfRangeError("EOF during staged fallback");
              auto st2 = dst.write_at(off, staging_buf.data(), *got2);
              if (!st2.ok())
                return st2;
              // metrics for staged bytes
              std::map<std::string, opentelemetry::common::AttributeValue> attrs;
              attrs.emplace("mode", opentelemetry::common::AttributeValue("staged"));
              g_bytes_total->Add(
                  static_cast<double>(*got2),
                  opentelemetry::common::KeyValueIterableView(attrs),
                  opentelemetry::context::Context{});
              off += *got2;
            }
            break; // exit window loop for this range
          }
        }
      }
      if (auto* base_sink = dynamic_cast<Sink*>(&dst)) {
        return base_sink->close();
      }
      return dst.close();
    }
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

  return close_status;
}

} // namespace tensorcast::store::loader
