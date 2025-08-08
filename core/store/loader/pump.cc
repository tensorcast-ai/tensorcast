// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/pump.h"

#include <atomic>
#include <limits>
#include <thread>
#include <unordered_map>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "core/store/direct_write.h"
#include "core/store/loader/dvmp_region_sink.h"

namespace stepcast::store::loader {

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

void RunProducer(Source& src, BufferPool& pool, PumpState& state) {
  while (!state.should_stop.load(std::memory_order_acquire)) {
    auto slot_result = pool.get_free_chunk();
    if (!slot_result.ok()) {
      absl::MutexLock lock(&state.status_mutex);
      if (state.producer_status.ok()) {
        state.producer_status = slot_result.status();
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }

    int slot_id = *slot_result;
    auto chunk_id = state.next_chunk_id.fetch_add(1, std::memory_order_acq_rel);

    // Check for overflow - use max value as error indicator
    if (chunk_id == std::numeric_limits<uint64_t>::max()) {
      pool.return_chunk(slot_id);
      absl::MutexLock lock(&state.status_mutex);
      if (state.producer_status.ok()) {
        state.producer_status = absl::ResourceExhaustedError("Chunk ID overflow");
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }

    size_t chunk_size = pool.chunk_size();

    // Get buffer pointer from pool using interface method
    void* buffer = pool.get_chunk_data_ptr(slot_id);
    if (!buffer) {
      pool.return_chunk(slot_id);
      absl::MutexLock lock(&state.status_mutex);
      if (state.producer_status.ok()) {
        state.producer_status = absl::InternalError("Failed to get chunk buffer pointer");
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }

    auto read_result = src.read(buffer, chunk_size);
    if (!read_result.ok()) {
      pool.return_chunk(slot_id);
      absl::MutexLock lock(&state.status_mutex);
      if (state.producer_status.ok()) {
        state.producer_status = read_result.status();
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }

    size_t bytes_read = *read_result;
    if (bytes_read == 0) {
      // EOF reached
      pool.return_chunk(slot_id);
      break;
    }

    // Validate that Source respects chunk size limits
    if (bytes_read > chunk_size) {
      pool.return_chunk(slot_id);
      absl::MutexLock lock(&state.status_mutex);
      if (state.producer_status.ok()) {
        state.producer_status = absl::InvalidArgumentError("Source returned more bytes than chunk size");
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }

    LOG(INFO) << "Producer marking chunk " << chunk_id << " as ready with " << bytes_read << " bytes";
    auto status = pool.mark_chunk_ready(slot_id, chunk_id, bytes_read);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to mark chunk ready: " << status;
      absl::MutexLock lock(&state.status_mutex);
      if (state.producer_status.ok()) {
        state.producer_status = status;
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }
  }
}

void RunConsumer(PositionedSink& dst, BufferPool& pool, PumpState& state) {
  LOG(INFO) << "Consumer thread started";
  while (!state.should_stop.load(std::memory_order_acquire)) {
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

    auto status = dst.write_at(dest_offset, chunk.data_ptr, chunk.bytes_in_chunk);
    pool.return_chunk(chunk.slot_id);

    if (!status.ok()) {
      absl::MutexLock lock(&state.status_mutex);
      if (state.consumer_status.ok()) {
        state.consumer_status = status;
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }
  }
}

void RunConsumerSequential(Sink& dst, BufferPool& pool, PumpState& state) {
  LOG(INFO) << "Consumer (sequential) thread started";
  while (!state.should_stop.load(std::memory_order_acquire)) {
    auto chunk_result = pool.get_ready_chunk();
    if (!chunk_result.ok()) {
      if (absl::IsUnavailable(chunk_result.status()) || absl::IsOutOfRange(chunk_result.status())) {
        LOG(INFO) << "Consumer (sequential): No more chunks";
        break;
      }
      LOG(ERROR) << "Consumer (sequential) failed to get ready chunk: " << chunk_result.status();
      absl::MutexLock lock(&state.status_mutex);
      if (state.consumer_status.ok()) {
        state.consumer_status = chunk_result.status();
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
    }

    const auto& chunk = *chunk_result;
    auto status = dst.write(chunk.data_ptr, chunk.bytes_in_chunk);
    pool.return_chunk(chunk.slot_id);

    if (!status.ok()) {
      absl::MutexLock lock(&state.status_mutex);
      if (state.consumer_status.ok()) {
        state.consumer_status = status;
      }
      state.should_stop.store(true, std::memory_order_release);
      break;
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
        break;
      }

      int slot_id = *slot_result;
      size_t to_read = std::min(remaining, pool.chunk_size());

      // Get buffer pointer from pool using interface method
      void* buffer = pool.get_chunk_data_ptr(slot_id);
      if (!buffer) {
        pool.return_chunk(slot_id);
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = absl::InternalError("Failed to get chunk buffer pointer");
        }
        state.should_stop.store(true, std::memory_order_release);
        break;
      }

      auto read_result = src.read_at(current_offset, buffer, to_read);
      if (!read_result.ok()) {
        pool.return_chunk(slot_id);
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = read_result.status();
        }
        state.should_stop.store(true, std::memory_order_release);
        break;
      }

      size_t bytes_read = *read_result;
      if (bytes_read == 0) {
        pool.return_chunk(slot_id);
        LOG(WARNING) << "Unexpected EOF at offset " << current_offset;
        break;
      }

      // Validate that Source respects requested read size limits
      if (bytes_read > to_read) {
        pool.return_chunk(slot_id);
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = absl::InvalidArgumentError("Source returned more bytes than requested");
        }
        state.should_stop.store(true, std::memory_order_release);
        break;
      }

      auto chunk_id = state.next_chunk_id.fetch_add(1, std::memory_order_acq_rel);

      // Check for overflow - use max value as error indicator
      if (chunk_id == std::numeric_limits<uint64_t>::max()) {
        pool.return_chunk(slot_id);
        absl::MutexLock lock(&state.status_mutex);
        if (state.producer_status.ok()) {
          state.producer_status = absl::ResourceExhaustedError("Chunk ID overflow");
        }
        state.should_stop.store(true, std::memory_order_release);
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
        break;
      }

      current_offset += bytes_read;
      remaining -= bytes_read;
    }
  }
}

} // namespace

absl::Status pump(Source& src, Sink& dst, BufferPool& pool, int concurrency) {
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }

  PumpState state;
  std::vector<std::thread> producers;
  producers.reserve(concurrency);

  // Start producer threads
  for (int i = 0; i < concurrency; ++i) {
    producers.emplace_back(RunProducer, std::ref(src), std::ref(pool), std::ref(state));
  }

  // Start consumer thread
  std::thread consumer(RunConsumerSequential, std::ref(dst), std::ref(pool), std::ref(state));

  // Wait for all producers to finish
  for (auto& t : producers) {
    t.join();
  }

  // Signal that production is complete
  pool.signal_production_complete();

  // Wait for consumer to finish
  consumer.join();

  // Close the sink
  auto close_status = dst.close();

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

  // Optional direct-write fast path: only when destination is DVMP region and
  // source supports direct writes (e.g., RDMA-enabled RemoteKeySource), and
  // explicitly enabled via env SCSTORE_ENABLE_DIRECT_DVMP=1.
  if (const char* env = ::getenv("SCSTORE_ENABLE_DIRECT_DVMP"); env && std::string(env) == "1") {
    if (auto* dvmp_sink = dynamic_cast<DVMPRegionSink*>(&dst)) {
      if (src.supports_direct_write()) {
        auto mm = dvmp_sink->get_memory_manager();
        if (mm) {
          // Build VaRanges and plan a direct write token
          std::vector<VaRange> va_ranges;
          va_ranges.reserve(ranges.size());
          for (const auto& r : ranges) {
            va_ranges.push_back({r.first, r.second});
          }
          auto token_or = mm->plan_direct_write(va_ranges);
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
          // Attempt to close if dst also implements Sink
          if (auto* base_sink = dynamic_cast<Sink*>(&dst)) {
            return base_sink->close();
          }
          return absl::OkStatus();
        }
      }
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

} // namespace stepcast::store::loader
