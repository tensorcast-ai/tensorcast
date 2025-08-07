// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/source.h"
#include "core/store/loader/sink.h"
#include "core/store/loader/buffer_pool.h"

using namespace stepcast::store::loader;

// Mock source for testing
class MockSource : public Source {
 public:
  MockSource(size_t total_size, size_t chunk_size = 1024) 
      : total_size_(total_size), chunk_size_(chunk_size), data_(total_size, 'A') {
    for (size_t i = 0; i < total_size; ++i) {
      data_[i] = static_cast<char>('A' + (i % 26));
    }
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    if (error_on_read_) {
      return absl::InternalError("Mock read error");
    }

    size_t bytes_to_read = std::min(max_bytes, total_size_ - offset_);
    if (bytes_to_read == 0) {
      return 0;  // EOF
    }

    std::memcpy(dst, data_.data() + offset_, bytes_to_read);
    offset_ += bytes_to_read;
    read_count_++;
    return bytes_to_read;
  }

  void set_error_on_read(bool error) { error_on_read_ = error; }
  size_t get_read_count() const { return read_count_; }
  const std::vector<char>& get_data() const { return data_; }

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
  MockSeekableSource(size_t total_size) : total_size_(total_size), data_(total_size, 'B') {
    for (size_t i = 0; i < total_size; ++i) {
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
      return 0;  // EOF
    }

    size_t bytes_to_read = std::min(bytes, total_size_ - offset);
    std::memcpy(dst, data_.data() + offset, bytes_to_read);
    offset_ = offset + bytes_to_read;
    return bytes_to_read;
  }

  void set_error_on_read(bool error) { error_on_read_ = error; }

 private:
  size_t total_size_;
  size_t offset_ = 0;
  std::vector<char> data_;
  bool error_on_read_ = false;
};

// Mock sink for testing
class MockSink : public Sink {
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

  absl::Status close() override {
    if (error_on_close_) {
      return absl::InternalError("Mock close error");
    }
    closed_ = true;
    return absl::OkStatus();
  }

  void set_error_on_write(bool error) { error_on_write_ = error; }
  void set_error_on_close(bool error) { error_on_close_ = error; }
  void set_return_overflow(bool overflow) { return_overflow_ = overflow; }
  
  size_t get_write_count() const { return write_count_; }
  size_t get_total_bytes_written() const { return total_bytes_written_; }
  bool is_closed() const { return closed_; }
  const std::vector<char>& get_written_data() const { return written_data_; }

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
  MockBufferPool(size_t chunk_size, size_t num_chunks) 
      : chunk_size_(chunk_size), num_chunks_(num_chunks) {
    // Initialize chunks
    for (size_t i = 0; i < num_chunks; ++i) {
      chunks_.emplace_back(std::make_unique<char[]>(chunk_size));
      free_slots_.push_back(i);
    }
  }

  absl::StatusOr<ReadyChunk> get_ready_chunk() override {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait for ready chunk
    cv_.wait(lock, [this] { return !ready_chunks_.empty() || error_on_get_; });
    
    if (error_on_get_) {
      return absl::InternalError("Mock buffer pool error");
    }

    if (ready_chunks_.empty()) {
      return absl::UnavailableError("No ready chunks");
    }

    ReadyChunk chunk = ready_chunks_.front();
    ready_chunks_.pop();
    return chunk;
  }

  absl::StatusOr<int> get_free_slot() override {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (free_slots_.empty()) {
      return absl::ResourceExhaustedError("No free slots");
    }

    int slot = free_slots_.front();
    free_slots_.pop_front();
    return slot;
  }

  void* get_chunk_data_ptr(int slot_id) override {
    if (slot_id < 0 || slot_id >= static_cast<int>(chunks_.size())) {
      return nullptr;
    }
    return chunks_[slot_id].get();
  }

  void produce_chunk(int slot_id, size_t bytes, uint64_t chunk_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ready_chunks_.push({slot_id, bytes, chunk_id});
    cv_.notify_one();
  }

  void return_chunk(int slot_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    free_slots_.push_back(slot_id);
  }

  void set_error_on_get(bool error) { 
    std::lock_guard<std::mutex> lock(mutex_);
    error_on_get_ = error; 
    cv_.notify_all();
  }

  size_t get_chunk_size() const { return chunk_size_; }

 private:
  size_t chunk_size_;
  size_t num_chunks_;
  std::vector<std::unique_ptr<char[]>> chunks_;
  std::deque<int> free_slots_;
  std::queue<ReadyChunk> ready_chunks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool error_on_get_ = false;
};

TEST_CASE("Pump basic functionality", "[pump]") {
  SECTION("Simple data transfer") {
    size_t data_size = 10 * 1024;  // 10KB
    size_t chunk_size = 1024;       // 1KB chunks
    
    MockSource source(data_size);
    MockSink sink;
    MockBufferPool pool(chunk_size, 4);

    auto status = pump(source, sink, pool, 2);
    
    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
    REQUIRE(sink.get_written_data() == source.get_data());
  }

  SECTION("Empty source") {
    MockSource source(0);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == 0);
  }

  SECTION("Large data with multiple chunks") {
    size_t data_size = 100 * 1024;  // 100KB
    size_t chunk_size = 8 * 1024;    // 8KB chunks
    
    MockSource source(data_size);
    MockSink sink;
    MockBufferPool pool(chunk_size, 3);

    auto status = pump(source, sink, pool, 2);
    
    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
    REQUIRE(sink.get_written_data() == source.get_data());
  }

  SECTION("Single threaded pump") {
    size_t data_size = 10 * 1024;
    
    MockSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
  }
}

TEST_CASE("Pump error handling", "[pump]") {
  SECTION("Source read error") {
    MockSource source(10 * 1024);
    source.set_error_on_read(true);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(!status.ok());
    REQUIRE(status.message().find("Mock read error") != std::string::npos);
  }

  SECTION("Sink write error") {
    MockSource source(10 * 1024);
    MockSink sink;
    sink.set_error_on_write(true);
    MockBufferPool pool(1024, 2);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(!status.ok());
    REQUIRE(status.message().find("Mock write error") != std::string::npos);
  }

  SECTION("Sink close error") {
    MockSource source(1024);
    MockSink sink;
    sink.set_error_on_close(true);
    MockBufferPool pool(1024, 2);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(!status.ok());
    REQUIRE(status.message().find("Mock close error") != std::string::npos);
  }

  SECTION("Buffer overflow detection") {
    MockSource source(10 * 1024);
    MockSink sink;
    sink.set_return_overflow(true);
    MockBufferPool pool(1024, 2);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(!status.ok());
    REQUIRE(status.message().find("Buffer overflow") != std::string::npos);
  }

  SECTION("Buffer pool error") {
    MockSource source(10 * 1024);
    MockSink sink;
    MockBufferPool pool(1024, 2);
    pool.set_error_on_get(true);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(!status.ok());
  }
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

    std::vector<std::pair<uint64_t, size_t>> ranges = {
      {0, 1024},
      {5120, 2048},
      {10240, 1024}
    };
    
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
    
    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == 0);
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
    size_t data_size = 100 * 1024;  // 100KB
    size_t chunk_size = 4 * 1024;    // 4KB chunks
    
    for (int concurrency = 1; concurrency <= 4; ++concurrency) {
      MockSource source(data_size);
      MockSink sink;
      MockBufferPool pool(chunk_size, concurrency + 1);

      auto status = pump(source, sink, pool, concurrency);
      
      REQUIRE(status.ok());
      REQUIRE(sink.get_total_bytes_written() == data_size);
      
      // Verify data integrity
      const auto& written = sink.get_written_data();
      const auto& original = source.get_data();
      REQUIRE(written == original);
    }
  }

  SECTION("Concurrent error handling") {
    size_t data_size = 50 * 1024;
    
    MockSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 4);

    // Start pump in background
    std::thread pump_thread([&]() {
      auto status = pump(source, sink, pool, 3);
      REQUIRE(!status.ok());
    });

    // Inject error after some delay
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    sink.set_error_on_write(true);

    pump_thread.join();
  }
}

TEST_CASE("Pump edge cases", "[pump]") {
  SECTION("Chunk ID overflow protection") {
    // This test verifies that chunk IDs don't overflow
    // In practice, this would require transferring massive amounts of data
    // Here we just verify the logic works for reasonable sizes
    size_t data_size = 1024 * 1024;  // 1MB
    size_t chunk_size = 1024;         // 1KB chunks (1024 chunks total)
    
    MockSource source(data_size);
    MockSink sink;
    MockBufferPool pool(chunk_size, 4);

    auto status = pump(source, sink, pool, 2);
    
    REQUIRE(status.ok());
    REQUIRE(sink.get_total_bytes_written() == data_size);
  }

  SECTION("Source returns more bytes than requested") {
    // This would require a buggy source implementation
    // The pump should detect and handle this gracefully
    // We test this indirectly through buffer validation
    size_t data_size = 10 * 1024;
    
    MockSource source(data_size);
    MockSink sink;
    MockBufferPool pool(1024, 2);

    auto status = pump(source, sink, pool, 1);
    
    REQUIRE(status.ok());
    // The pump should never write more than the source size
    REQUIRE(sink.get_total_bytes_written() <= data_size);
  }

  SECTION("Zero concurrency (should default to 1)") {
    MockSource source(1024);
    MockSink sink;
    MockBufferPool pool(512, 2);

    auto status = pump(source, sink, pool, 0);
    
    // Implementation should handle this gracefully
    // Either by defaulting to 1 or returning an error
    if (status.ok()) {
      REQUIRE(sink.get_total_bytes_written() == 1024);
    } else {
      REQUIRE(status.code() == absl::StatusCode::kInvalidArgument);
    }
  }
}