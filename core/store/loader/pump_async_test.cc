// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/async_copy_manager.h"
#include "core/store/loader/buffer_pool.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/sink.h"
#include "core/store/loader/source.h"

using namespace tensorcast::store::loader;

// Minimal seekable source that produces deterministic bytes
class TestSeekableSource : public SeekableSource {
 public:
  explicit TestSeekableSource(size_t total) : total_(total), data_(total, '\0') {
    for (size_t i = 0; i < total_; ++i)
      data_[i] = static_cast<char>('a' + (i % 26));
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return read_at(offset_, dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= total_)
      return size_t{0};
    size_t n = std::min(bytes, total_ - static_cast<size_t>(offset));
    std::memcpy(dst, data_.data() + static_cast<size_t>(offset), n);
    offset_ = static_cast<size_t>(offset + n);
    return n;
  }

 private:
  size_t total_;
  size_t offset_ = 0;
  std::vector<char> data_;
};

// Minimal buffer pool that tracks returns
class TestBufferPool : public BufferPool {
 public:
  TestBufferPool(size_t chunk_size, int capacity) : chunk_size_(chunk_size), capacity_(capacity) {
    bufs_.resize(static_cast<size_t>(capacity_));
    for (int i = 0; i < capacity_; ++i) {
      bufs_[static_cast<size_t>(i)] = std::unique_ptr<char[]>(new char[chunk_size_]);
      free_.push_back(i);
    }
  }
  size_t chunk_size() const override {
    return chunk_size_;
  }
  int capacity() const override {
    return capacity_;
  }
  absl::StatusOr<int> get_free_chunk() override {
    std::unique_lock<std::mutex> lk(mu_);
    free_cv_.wait(lk, [&] { return !free_.empty() || stopped_; });
    if (free_.empty())
      return absl::OutOfRangeError("stopped");
    int id = free_.front();
    free_.pop_front();
    return id;
  }
  void return_chunk(int slot_id) override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      returned_.push_back(slot_id);
      free_.push_back(slot_id);
    }
    free_cv_.notify_one();
    ready_cv_.notify_one();
  }
  absl::Status mark_chunk_ready(int slot_id, uint64_t gid, size_t bytes) override {
    void* p = get_chunk_data_ptr(slot_id);
    if (!p)
      return absl::InternalError("bad slot");
    {
      std::lock_guard<std::mutex> lk(mu_);
      ready_.push(ReadyChunk{slot_id, gid, bytes, p});
    }
    ready_cv_.notify_one();
    return absl::OkStatus();
  }
  absl::StatusOr<ReadyChunk> get_ready_chunk() override {
    std::unique_lock<std::mutex> lk(mu_);
    ready_cv_.wait(lk, [&] { return !ready_.empty() || stopped_; });
    if (ready_.empty())
      return absl::UnavailableError("stopped");
    auto c = ready_.front();
    ready_.pop();
    return c;
  }
  void signal_production_complete() override {
    std::lock_guard<std::mutex> lk(mu_);
    stopped_ = true;
    ready_cv_.notify_all();
    free_cv_.notify_all();
  }
  void shutdown() override {
    signal_production_complete();
  }
  void* get_chunk_data_ptr(int slot_id) override {
    if (slot_id < 0 || slot_id >= capacity_)
      return nullptr;
    return bufs_[static_cast<size_t>(slot_id)].get();
  }

  // Test helpers
  std::vector<int> returned_slots() const {
    std::lock_guard<std::mutex> lk(mu_);
    return returned_;
  }

 private:
  size_t chunk_size_;
  int capacity_;
  std::vector<std::unique_ptr<char[]>> bufs_;
  std::deque<int> free_;
  std::queue<ReadyChunk> ready_;
  mutable std::mutex mu_;
  std::condition_variable free_cv_;
  std::condition_variable ready_cv_;
  bool stopped_ = false;
  std::vector<int> returned_;
};

// Async sink that exercises the AsyncPositionedSink path. Uses ACM::submit_h2h
// to avoid requiring CUDA; this returns a ready handle but still validates
// the pump branch and handle-driven return logic.
class TestAsyncSink : public PositionedSink, public AsyncPositionedSink {
 public:
  absl::Status write_at(uint64_t, const void*, size_t) override {
    // Should not be called in this test when AsyncPositionedSink is present.
    sync_called_++;
    return absl::OkStatus();
  }
  absl::Status close() override {
    return absl::OkStatus();
  }

  absl::StatusOr<tensorcast::common::CopyHandle> write_at_async(uint64_t, const void* src, size_t bytes) override {
    async_called_++;
    // Use H2H copy for a ready handle; this still goes through ACM and traces.
    tensorcast::common::HostRegion hsrc{.base = src, .length = bytes, .pinned = true};
    if (tmp_.size() < bytes)
      tmp_.resize(bytes);
    tensorcast::common::HostRegion hdst{.base = tmp_.data(), .length = bytes, .pinned = true};
    auto hdl_or =
        tensorcast::common::AsyncCopyManager::instance().submit_h2h(hsrc, hdst, {.tracing_stage = "H2H/Copy"});
    return hdl_or;
  }

  int async_called() const {
    return async_called_.load();
  }
  int sync_called() const {
    return sync_called_.load();
  }

 private:
  std::atomic<int> async_called_{0};
  std::atomic<int> sync_called_{0};
  std::vector<char> tmp_;
};

TEST_CASE("Pump uses AsyncPositionedSink path", "[pump][async]") {
  const size_t total = 32 * 1024;
  TestSeekableSource src(total);
  TestAsyncSink sink;
  TestBufferPool pool(/*chunk_size=*/4096, /*capacity=*/2);

  std::vector<Range> ranges{{0ULL, total}};
  auto st = pump_ranges(src, sink, pool, absl::MakeSpan(ranges), /*concurrency=*/2);
  REQUIRE(st.ok());
  REQUIRE(sink.async_called() > 0);
  REQUIRE(sink.sync_called() == 0);
  // Ensure slots have been returned at least once
  REQUIRE(pool.returned_slots().size() > 0);
}

TEST_CASE("Pump async path surfaces sink error", "[pump][async]") {
  const size_t total = 8 * 1024;
  TestSeekableSource src(total);

  // Sink that fails on first async call
  class FailingAsyncSink : public TestAsyncSink {
   public:
    absl::StatusOr<tensorcast::common::CopyHandle> write_at_async(uint64_t, const void*, size_t) override {
      return absl::InternalError("async sink failure");
    }
  } sink;

  TestBufferPool pool(/*chunk_size=*/4096, /*capacity=*/1);
  std::vector<Range> ranges{{0ULL, total}};
  auto st = pump_ranges(src, sink, pool, absl::MakeSpan(ranges), /*concurrency=*/1);
  REQUIRE(!st.ok());
  REQUIRE(st.message().find("async sink failure") != std::string::npos);
}
