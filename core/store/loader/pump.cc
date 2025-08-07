// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/pump.h"

#include <atomic>
#include <limits>
#include <thread>
#include <vector>

#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"

namespace stepcast::store::loader {

namespace {

struct PumpState {
  std::atomic<bool> should_stop{false};
  std::atomic<uint64_t> next_chunk_id{0};
  absl::Status producer_status;
  absl::Status consumer_status;
  absl::Mutex status_mutex;
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

void RunConsumer(Sink& dst, BufferPool& pool, PumpState& state) {
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
    Sink& dst,
    BufferPool& pool,
    absl::Span<const std::pair<uint64_t, size_t>> ranges,
    int concurrency) {
  if (concurrency <= 0) {
    return absl::InvalidArgumentError("Concurrency must be positive");
  }
  if (ranges.empty()) {
    return absl::InvalidArgumentError("Ranges cannot be empty");
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

} // namespace stepcast::store::loader