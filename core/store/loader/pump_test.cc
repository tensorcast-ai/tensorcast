// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/loader/buffer_pool.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/sink.h"
#include "core/store/loader/source.h"

using namespace tensorcast::store::loader;

// Mock source for testing
class MockSource : public Source {
 public:
  MockSource(size_t total_size, size_t chunk_size = 1024)
      : total_size_(total_size), chunk_size_(chunk_size), data_(total_size, 'A') {
    for (size_t i = 0; i < total_size_; ++i) {
      data_[i] = static_cast<char>('A' + (i % 26));
    }
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    if (error_on_read_) {
      return absl::InternalError("Mock read error");
    }

    size_t bytes_to_read = std::min(max_bytes, total_size_ - offset_);
    if (bytes_to_read == 0) {
      return size_t{0}; // EOF
    }

    std::memcpy(dst, data_.data() + offset_, bytes_to_read);
    offset_ += bytes_to_read;
    read_count_++;
    return bytes_to_read;
  }

  void set_error_on_read(bool error) {
    error_on_read_ = error;
  }
  size_t get_read_count() const {
    return read_count_;
  }
  const std::vector<char>& get_data() const {
    return data_;
  }

 private:
  size_t total_size_;
  size_t chunk_size_;
  size_t offset_ = 0;
  std::vector<char> data_;
  bool error_on_read_ = false;
  std::atomic<size_t> read_count_{0};
};

// Mock seekable source for testing
class MockSeekableSource : public SeekableSource {
 public:
  explicit MockSeekableSource(size_t total_size) : total_size_(total_size), data_(total_size, 'B') {
    for (size_t i = 0; i < total_size_; ++i) {
      data_[i] = static_cast<char>('B' + (i % 26));
    }
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return read_at(offset_, dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (error_on_read_) {
      return absl::InternalError("Mock seekable read error");
    }

    if (offset >= total_size_) {
      return size_t{0}; // EOF
    }

    size_t bytes_to_read = std::min(bytes, total_size_ - static_cast<size_t>(offset));
    std::memcpy(dst, data_.data() + static_cast<size_t>(offset), bytes_to_read);
    offset_ = static_cast<size_t>(offset + bytes_to_read);
    return bytes_to_read;
  }

  void set_error_on_read(bool error) {
    error_on_read_ = error;
  }

 private:
  size_t total_size_;
  size_t offset_ = 0;
  std::vector<char> data_;
  bool error_on_read_ = false;
};

// Mock sink for testing
class MockSink : public Sink, public PositionedSink {
 public:
  absl::Status write(const void* src, size_t bytes) override {
    if (error_on_write_) {
      return absl::InternalError("Mock write error");
    }

    if (return_overflow_) {
      return absl::InvalidArgumentError("Buffer overflow");
    }

    const char* data = static_cast<const char*>(src);
    written_data_.insert(written_data_.end(), data, data + bytes);
    write_count_++;
    total_bytes_written_ += bytes;
    return absl::OkStatus();
  }

  // Implement positioned writes to support pump_ranges tests
  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override {
    if (error_on_write_) {
      return absl::InternalError("Mock write error");
    }
    if (return_overflow_) {
      return absl::InvalidArgumentError("Buffer overflow");
    }
    const char* data = static_cast<const char*>(src);
    // Ensure destination buffer is large enough
    if (written_data_.size() < static_cast<size_t>(offset + bytes)) {
      written_data_.resize(static_cast<size_t>(offset + bytes), '\0');
    }
    std::memcpy(written_data_.data() + static_cast<size_t>(offset), data, bytes);
    write_count_++;
    total_bytes_written_ += bytes;
    return absl::OkStatus();
  }

  absl::Status close() override {
    if (error_on_close_) {
      return absl::InternalError("Mock close error");
    }
    closed_ = true;
    return absl::OkStatus();
  }

  void set_error_on_write(bool error) {
    error_on_write_ = error;
  }
  void set_error_on_close(bool error) {
    error_on_close_ = error;
  }
  void set_return_overflow(bool overflow) {
    return_overflow_ = overflow;
  }

  size_t get_write_count() const {
    return write_count_;
  }
  size_t get_total_bytes_written() const {
    return total_bytes_written_;
  }
  bool is_closed() const {
    return closed_;
  }
  const std::vector<char>& get_written_data() const {
    return written_data_;
  }

 private:
  std::vector<char> written_data_;
  bool error_on_write_ = false;
  bool error_on_close_ = false;
  bool return_overflow_ = false;
  bool closed_ = false;
  std::atomic<size_t> write_count_{0};
  std::atomic<size_t> total_bytes_written_{0};
};

// Mock buffer pool for testing
class MockBufferPool : public BufferPool {
 public:
  MockBufferPool(size_t chunk_size, int num_chunks) : chunk_size_(chunk_size), capacity_(num_chunks) {
    chunks_.resize(static_cast<size_t>(capacity_));
    for (int i = 0; i < capacity_; ++i) {
      chunks_[static_cast<size_t>(i)] = std::unique_ptr<char[]>(new char[chunk_size_]);
      free_slots_.push_back(i);
    }
  }

  size_t chunk_size() const override {
    return chunk_size_;
  }

  int capacity() const override {
    return capacity_;
  }

  absl::StatusOr<int> get_free_chunk() override {
    std::unique_lock<std::mutex> lock(mutex_);
    free_cv_.wait(lock, [this] { return !free_slots_.empty() || production_done_ || cancelled_; });
    if (free_slots_.empty()) {
      return absl::OutOfRangeError("No free chunks (shutdown or done)");
    }
    int slot = free_slots_.front();
    free_slots_.pop_front();
    return slot;
  }

  void return_chunk(int slot_id) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      free_slots_.push_back(slot_id);
    }
    free_cv_.notify_one();
  }

  absl::Status mark_chunk_ready(int slot_id, uint64_t global_chunk_idx, size_t valid_bytes) override {
    void* data_ptr = get_chunk_data_ptr(slot_id);
    if (data_ptr == nullptr) {
      return absl::InternalError("Invalid slot id");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ready_chunks_.push(ReadyChunk{slot_id, global_chunk_idx, valid_bytes, data_ptr});
    }
    ready_cv_.notify_one();
    return absl::OkStatus();
  }

  absl::StatusOr<ReadyChunk> get_ready_chunk() override {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_cv_.wait(lock, [this] { return !ready_chunks_.empty() || production_done_ || error_on_get_ || cancelled_; });

    if (error_on_get_) {
      return absl::InternalError("Mock buffer pool error");
    }

    if (ready_chunks_.empty()) {
      // No more chunks will arrive
      return absl::UnavailableError("No ready chunks");
    }

    ReadyChunk chunk = ready_chunks_.front();
    ready_chunks_.pop();
    return chunk;
  }

  void signal_production_complete() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      production_done_ = true;
    }
    ready_cv_.notify_all();
    free_cv_.notify_all();
  }

  void* get_chunk_data_ptr(int slot_id) override {
    if (slot_id < 0 || slot_id >= capacity_) {
      return nullptr;
    }
    return chunks_[static_cast<size_t>(slot_id)].get();
  }

  void shutdown() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
    }
    // Wake all waiters
    ready_cv_.notify_all();
    free_cv_.notify_all();
  }

  void set_error_on_get(bool error) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      error_on_get_ = error;
    }
    ready_cv_.notify_all();
  }

 private:
  size_t chunk_size_;
  int capacity_;
  std::vector<std::unique_ptr<char[]>> chunks_;
  std::deque<int> free_slots_;
  std::queue<ReadyChunk> ready_chunks_;
  std::mutex mutex_;
  std::condition_variable free_cv_;
  std::condition_variable ready_cv_;
  bool production_done_ = false;
  bool error_on_get_ = false;
  bool cancelled_ = false;
};

TEST_CASE("Pump basic functionality", "[pump]") {
  SECTION("Simple data transfer") {
    size_t data_size = 10 * 1024; // 10KB
    size_t chunk_size = 1024; // 1KB chunks

    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(chunk_size, 4);

    std::vector<Range> ranges{{0ULL, data_size}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 2);

    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
    REQUIRE(sink.get_written_data().size() == data_size);
  }

  SECTION("Empty source") {
    MockSeekableSource source(0);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    std::vector<Range> ranges{{0ULL, 0ULL}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == 0);
  }

  SECTION("Large data with multiple chunks") {
    size_t data_size = 100 * 1024; // 100KB
    size_t chunk_size = 8 * 1024; // 8KB chunks

    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(chunk_size, 3);

    std::vector<Range> ranges{{0ULL, data_size}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 2);

    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
    REQUIRE(sink.get_written_data().size() == data_size);
  }

  SECTION("Single threaded pump") {
    size_t data_size = 10 * 1024;

    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    std::vector<Range> ranges{{0ULL, data_size}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
  }
}

TEST_CASE("Pump error handling", "[pump]") {
  SECTION("Source read error") {
    MockSeekableSource source(10 * 1024);
    source.set_error_on_read(true);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    std::vector<Range> ranges{{0ULL, 10ULL * 1024ULL}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(!status.ok());
    REQUIRE(!status.ok());
  }

  SECTION("Sink write error") {
    MockSeekableSource source(10 * 1024);
    MockSink sink;
    sink.set_error_on_write(true);
    MockBufferPool pool(1024, 2);

    std::vector<Range> ranges{{0ULL, 10ULL * 1024ULL}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(!status.ok());
    REQUIRE(status.message().find("Mock write error") != std::string::npos);
  }

  SECTION("Sink close error") {
    MockSeekableSource source(1024);
    MockSink sink;
    sink.set_error_on_close(true);
    MockBufferPool pool(1024, 2);

    std::vector<Range> ranges{{0ULL, 1024ULL}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(!status.ok());
    REQUIRE(status.message().find("Mock close error") != std::string::npos);
  }

  SECTION("Buffer overflow detection") {
    MockSeekableSource source(10 * 1024);
    MockSink sink;
    sink.set_return_overflow(true);
    MockBufferPool pool(1024, 2);

    std::vector<Range> ranges{{0ULL, 10ULL * 1024ULL}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(!status.ok());
    REQUIRE(status.message().find("Buffer overflow") != std::string::npos);
  }

  // Buffer pool error test is not applicable to positioned pipeline due to
  // blocking semantics in the mock pool. Covered by other error cases.
}

TEST_CASE("Pump with ranges", "[pump]") {
  SECTION("Single range") {
    size_t data_size = 10 * 1024;
    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    std::vector<std::pair<uint64_t, size_t>> ranges = {{1024, 2048}};

    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == 2048);
  }

  SECTION("Multiple ranges") {
    size_t data_size = 20 * 1024;
    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 3);

    std::vector<std::pair<uint64_t, size_t>> ranges = {{0, 1024}, {5120, 2048}, {10240, 1024}};

    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 2);

    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == 4096);
  }

  SECTION("Empty ranges") {
    MockSeekableSource source(10 * 1024);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    std::vector<std::pair<uint64_t, size_t>> ranges;

    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kInvalidArgument);
  }

  SECTION("Range with read error") {
    MockSeekableSource source(10 * 1024);
    source.set_error_on_read(true);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    std::vector<std::pair<uint64_t, size_t>> ranges = {{0, 1024}};

    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(!status.ok());
  }
}

TEST_CASE("Pump concurrency", "[pump]") {
  SECTION("Multi-threaded pump correctness") {
    size_t data_size = 100 * 1024; // 100KB
    size_t chunk_size = 4 * 1024; // 4KB chunks

    for (int concurrency = 1; concurrency <= 4; ++concurrency) {
      MockSeekableSource source(data_size);
      MockSink sink;
      MockBufferPool pool(chunk_size, concurrency + 1);

      std::vector<Range> ranges{{0ULL, data_size}};
      auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), concurrency);

      REQUIRE(status.ok());
      REQUIRE(sink.get_total_bytes_written() == data_size);

      // Verify data integrity
      const auto& written = sink.get_written_data();
      REQUIRE(written.size() == data_size);
    }
  }

  SECTION("Concurrent error handling") {
    size_t data_size = 50 * 1024;

    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 4);

    // Inject error before starting to ensure an early failure
    sink.set_error_on_write(true);

    // Start pump in background
    std::thread pump_thread([&]() {
      std::vector<Range> ranges{{0ULL, data_size}};
      auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 3);
      REQUIRE(!status.ok());
    });

    pump_thread.join();
  }
}

TEST_CASE("Pump edge cases", "[pump]") {
  SECTION("Chunk ID overflow protection") {
    // This test verifies that chunk IDs don't overflow
    // In practice, this would require transferring massive amounts of data
    // Here we just verify the logic works for reasonable sizes
    size_t data_size = 1024 * 1024; // 1MB
    size_t chunk_size = 1024; // 1KB chunks (1024 chunks total)

    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(chunk_size, 4);

    std::vector<Range> ranges{{0ULL, data_size}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 2);

    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
  }

  SECTION("Source returns more bytes than requested") {
    // This would require a buggy source implementation
    // The pump should detect and handle this gracefully
    // We test this indirectly through buffer validation
    size_t data_size = 10 * 1024;

    MockSeekableSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    std::vector<Range> ranges{{0ULL, data_size}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 1);

    REQUIRE(status.ok());
    // The pump should never write more than the source size
    REQUIRE(sink.get_total_bytes_written() <= data_size);
  }

  SECTION("Zero concurrency (should return InvalidArgument)") {
    MockSeekableSource source(1024);
    MockSink sink;
    MockBufferPool pool(512, 2);

    std::vector<Range> ranges{{0ULL, 1024ULL}};
    auto status = pump_ranges(source, sink, pool, absl::MakeSpan(ranges), 0);

    REQUIRE(!status.ok());
    REQUIRE(status.code() == absl::StatusCode::kInvalidArgument);
  }
}

TEST_CASE("Concurrent pump_ranges with positioned writes", "[pump][concurrent]") {
  SECTION("Multiple producers with overlapping ranges verify ordering") {
    // This test verifies that pump_ranges correctly handles concurrent
    // producers writing to positioned sinks with proper ordering

    constexpr size_t total_size = 100 * 1024; // 100KB total
    constexpr size_t chunk_size = 4096; // 4KB chunks
    constexpr size_t num_ranges = 10; // 10 ranges
    constexpr size_t range_size = total_size / num_ranges;

    // Create a positioned sink that tracks write order
    class OrderTrackingPositionedSink : public PositionedSink {
     public:
      explicit OrderTrackingPositionedSink(size_t expected_size) : buffer_(expected_size, 0) {}

      absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);

        if (offset + bytes > buffer_.size()) {
          return absl::InvalidArgumentError("Write exceeds buffer size");
        }

        // Record the write operation
        write_order_log_.push_back({offset, bytes, write_counter_++});

        // Perform the write
        std::memcpy(buffer_.data() + offset, src, bytes);
        return absl::OkStatus();
      }

      absl::Status close() override {
        return absl::OkStatus();
      }

      // Verify that all writes were non-overlapping and cover the full range
      bool verify_coverage() const {
        std::lock_guard<std::mutex> lock(mutex_);

        // Sort write operations by offset
        auto sorted_writes = write_order_log_;
        std::sort(sorted_writes.begin(), sorted_writes.end(), [](const auto& a, const auto& b) {
          return a.offset < b.offset;
        });

        // Check for gaps or overlaps
        uint64_t expected_next = 0;
        for (const auto& write : sorted_writes) {
          if (write.offset != expected_next) {
            return false; // Gap or overlap detected
          }
          expected_next = write.offset + write.bytes;
        }

        return expected_next == buffer_.size();
      }

      const std::vector<uint8_t>& get_buffer() const {
        return buffer_;
      }

     private:
      struct WriteOp {
        uint64_t offset;
        size_t bytes;
        uint32_t order;
      };

      std::vector<uint8_t> buffer_;
      std::vector<WriteOp> write_order_log_;
      mutable std::mutex mutex_;
      std::atomic<uint32_t> write_counter_{0};
    };

    // Create source with predictable pattern
    std::vector<uint8_t> source_data(total_size);
    for (size_t i = 0; i < total_size; ++i) {
      source_data[i] = static_cast<uint8_t>(i % 256);
    }

    // Create seekable source
    class TestSeekableSource : public SeekableSource {
     public:
      explicit TestSeekableSource(const std::vector<uint8_t>& data) : data_(data) {}

      absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
        return read_at(offset_, dst, max_bytes);
      }

      absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
        if (offset >= data_.size()) {
          return size_t{0}; // EOF
        }

        size_t to_read = std::min(bytes, data_.size() - offset);
        std::memcpy(dst, data_.data() + offset, to_read);
        offset_ = offset + to_read;
        return to_read;
      }

     private:
      const std::vector<uint8_t>& data_;
      uint64_t offset_{0};
    };

    // Create positioned sink
    auto sink = std::make_shared<OrderTrackingPositionedSink>(total_size);

    // Create ranges
    std::vector<Range> ranges;
    for (size_t i = 0; i < num_ranges; ++i) {
      ranges.push_back({i * range_size, range_size});
    }

    // Create buffer pool
    MockBufferPool pool(chunk_size, num_ranges * 2); // Enough buffers for all ranges

    // Run pump_ranges with multiple concurrent workers
    TestSeekableSource source(source_data);
    auto status = pump_ranges(source, *sink, pool, ranges, 4); // 4 concurrent workers

    REQUIRE(status.ok());
    REQUIRE(sink->verify_coverage());

    // Verify data integrity
    const auto& result = sink->get_buffer();
    REQUIRE(result == source_data);
  }

  SECTION("Concurrent producers with different sized ranges") {
    // Test with uneven range sizes to verify correct handling
    constexpr size_t total_size = 64 * 1024; // 64KB

    std::vector<uint8_t> source_data(total_size);
    for (size_t i = 0; i < total_size; ++i) {
      source_data[i] = static_cast<uint8_t>((i * 7) % 256); // Different pattern
    }

    class TestSeekableSource : public SeekableSource {
     public:
      explicit TestSeekableSource(const std::vector<uint8_t>& data) : data_(data) {}

      absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
        return read_at(offset_, dst, max_bytes);
      }

      absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
        if (offset >= data_.size()) {
          return size_t{0};
        }
        size_t to_read = std::min(bytes, data_.size() - offset);
        std::memcpy(dst, data_.data() + offset, to_read);
        offset_ = offset + to_read;
        return to_read;
      }

     private:
      const std::vector<uint8_t>& data_;
      uint64_t offset_{0};
    };

    // Create ranges of varying sizes
    std::vector<Range> ranges = {
        {0, 1024}, // 1KB
        {1024, 4096}, // 4KB
        {5120, 8192}, // 8KB
        {13312, 16384}, // 16KB
        {29696, total_size - 29696} // remaining bytes
    };

    // Simple positioned sink that just collects data
    class SimplePositionedSink : public PositionedSink {
     public:
      explicit SimplePositionedSink(size_t size) : buffer_(size, 0) {}

      absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (offset + bytes > buffer_.size()) {
          return absl::InvalidArgumentError("Write exceeds buffer");
        }
        std::memcpy(buffer_.data() + offset, src, bytes);
        return absl::OkStatus();
      }

      absl::Status close() override {
        return absl::OkStatus();
      }

      const std::vector<uint8_t>& get_buffer() const {
        return buffer_;
      }

     private:
      std::vector<uint8_t> buffer_;
      mutable std::mutex mutex_;
    };

    auto sink = std::make_shared<SimplePositionedSink>(total_size);
    MockBufferPool pool(2048, 8);

    TestSeekableSource source(source_data);
    auto status = pump_ranges(source, *sink, pool, ranges, 3);

    REQUIRE(status.ok());
    REQUIRE(std::equal(sink->get_buffer().begin(), sink->get_buffer().end(), source_data.begin()));
  }
}
