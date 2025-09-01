// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/pump.h"

#include <atomic>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "core/store/direct_write.h"

namespace tensorcast::store::loader {

namespace {

struct PumpState {
  std::atomic<bool> should_stop{false};
  std::atomic<uint64_t> next_chunk_id{0};
  absl::Status producer_status;
  absl::Status consumer_status;
  absl::Mutex status_mutex;
  // Map global_chunk_id -> destination offset for range pumping
  std::unordered_map<uint64_t, uint64_t> chunk_offsets;
  absl::Mutex offsets_mutex;
};

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

void RunConsumer(PositionedSink& dst, BufferPool& pool, PumpState& state) {
  LOG(INFO) << "Consumer thread started";
  bool draining = false;
  while (!state.should_stop.load(std::memory_order_acquire) || draining) {
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
      // Enter drain mode: keep attempting to consume any remaining ready chunks
      // to ensure slots are returned, then exit when queues are empty.
      draining = true;
      continue;
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

    auto status = dst.write_at(dest_offset, chunk.data_ptr, chunk.bytes_in_chunk);
    pool.return_chunk(chunk.slot_id);

    if (!status.ok()) {
      absl::MutexLock lock(&state.status_mutex);
      if (state.consumer_status.ok()) {
        state.consumer_status = status;
      }
      state.should_stop.store(true, std::memory_order_release);
      // Request pool to wake up any waiters so producers can exit
      pool.shutdown();
      // Switch to drain mode to return remaining ready chunks
      draining = true;
    }
  }
}

void RunRangeProducer(
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
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = status;
        }
        state.should_stop.store(true, std::memory_order_release);
        pool.shutdown();
        break;
      }

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
    int concurrency) {
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }
  if (ranges.empty()) {
    return absl::InvalidArgumentError("Ranges cannot be empty");
  }

  // Capability-driven direct-write fast path: if destination implements
  // DirectWritableSink and source supports direct writes (e.g., RDMA),
  // plan a token and stream directly into destination VA ranges.
  if (src.supports_direct_write()) {
    if (auto* cap = dynamic_cast<DirectWritableSink*>(&dst)) {
      // Convert to VaRanges
      std::vector<VaRange> va_ranges;
      va_ranges.reserve(ranges.size());
      for (const auto& r : ranges)
        va_ranges.push_back({r.first, r.second});

      auto token_or = cap->plan_direct_write(absl::MakeSpan(va_ranges));
      if (!token_or.ok()) {
        return token_or.status();
      }
      const auto& token = *token_or;
      for (const auto& [off, len] : ranges) {
        auto got = src.read_into(off, len, token);
        if (!got.ok()) {
          return got.status();
        }
        if (*got != len) {
          return absl::OutOfRangeError("Short direct write");
        }
      }
      // Close lifecycle
      if (auto* base_sink = dynamic_cast<Sink*>(&dst)) {
        return base_sink->close();
      }
      return dst.close();
    }
  }

  PumpState state;
  std::atomic<size_t> range_index{0};
  std::vector<std::thread> producers;
  producers.reserve(concurrency);

  // Start producer threads
  for (int i = 0; i < concurrency; ++i) {
    producers.emplace_back(
        RunRangeProducer, std::ref(src), std::ref(pool), ranges, std::ref(range_index), std::ref(state));
  }

  // Start consumer thread
  std::thread consumer(RunConsumer, std::ref(dst), std::ref(pool), std::ref(state));

  // Wait for all producers to finish
  for (auto& t : producers) {
    t.join();
  }

  // Signal that production is complete
  pool.signal_production_complete();

  // Wait for consumer to finish
  consumer.join();

  // Close the sink
  // Attempt to close if dst also implements Sink
  absl::Status close_status = absl::OkStatus();
  if (auto* base_sink = dynamic_cast<Sink*>(&dst)) {
    close_status = base_sink->close();
  }

  // Return first error encountered
  {
    absl::MutexLock lock(&state.status_mutex);
    if (!state.producer_status.ok()) {
      return state.producer_status;
    }
    if (!state.consumer_status.ok()) {
      return state.consumer_status;
    }
  }

  return close_status;
}

} // namespace tensorcast::store::loader
