// Copyright (c) 2025, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/direct_write.h"
#include "core/store/loader/buffer_pool.h"
#include "core/store/loader/pump.h"
#include "core/store/loader/sink.h"
#include "core/store/loader/source.h"

using namespace tensorcast::store;
using namespace tensorcast::store::loader;

namespace {

// Minimal buffer pool (unused in direct path but required by API)
class DummyPool : public BufferPool {
 public:
  explicit DummyPool(size_t chunk_size, int capacity) : chunk_size_(chunk_size), capacity_(capacity) {
    (void)capacity_;
  }
  size_t chunk_size() const override {
    return chunk_size_;
  }
  int capacity() const override {
    return capacity_;
  }
  absl::StatusOr<int> get_free_chunk() override {
    return absl::InternalError("unused in direct path");
  }
  void return_chunk(int) override {}
  absl::Status mark_chunk_ready(int, uint64_t, size_t) override {
    return absl::OkStatus();
  }
  absl::StatusOr<ReadyChunk> get_ready_chunk() override {
    return absl::UnavailableError("no chunks");
  }
  void signal_production_complete() override {}
  void* get_chunk_data_ptr(int) override {
    return nullptr;
  }

 private:
  size_t chunk_size_;
  int capacity_;
};

// Direct-capable sink that exposes a buffer and plans token segments.
class DirectCapableSink : public PositionedSink, public DirectWritableSink {
 public:
  explicit DirectCapableSink(size_t size) : buffer_(size, 0) {}

  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override {
    ++write_calls_;
    if (offset + bytes > buffer_.size())
      return absl::InvalidArgumentError("overflow");
    std::memcpy(buffer_.data() + offset, src, bytes);
    return absl::OkStatus();
  }
  absl::Status close() override {
    return absl::OkStatus();
  }

  absl::StatusOr<DirectWriteToken> plan_direct_write(absl::Span<const VaRange> ranges) override {
    DirectWriteToken token;
    token.segments.reserve(ranges.size());
    for (const auto& r : ranges) {
      if (r.offset + r.length > buffer_.size())
        return absl::OutOfRangeError("range beyond buffer");
      token.segments.push_back(
          DirectWriteToken::Segment{
              .va_offset = r.offset,
              .local_addr = reinterpret_cast<uint64_t>(buffer_.data() + r.offset),
              .length = r.length});
    }
    token.keepalive = nullptr; // Not needed in test
    return token;
  }

  const std::vector<uint8_t>& buffer() const {
    return buffer_;
  }
  size_t write_calls() const {
    return write_calls_;
  }

 private:
  std::vector<uint8_t> buffer_;
  size_t write_calls_ = 0;
};

// Seekable source that supports direct write; read_into writes data into token segments.
class DirectSource : public SeekableSource {
 public:
  explicit DirectSource(std::vector<uint8_t> data) : data_(std::move(data)) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    // Not used in direct path; provide simple implementation
    size_t to_read = std::min(max_bytes, data_.size());
    std::memcpy(dst, data_.data(), to_read);
    return to_read;
  }
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size())
      return size_t{0};
    size_t to_read = std::min<size_t>(bytes, data_.size() - offset);
    std::memcpy(dst, data_.data() + offset, to_read);
    return to_read;
  }
  bool supports_direct_write() const override {
    return true;
  }
  absl::StatusOr<size_t> read_into(uint64_t dest_va_offset, size_t bytes, const DirectWriteToken& token) override {
    size_t total = 0;
    uint64_t pos = dest_va_offset;
    while (total < bytes) {
      const DirectWriteToken::Segment* seg = nullptr;
      for (const auto& s : token.segments) {
        if (pos >= s.va_offset && pos < s.va_offset + s.length) {
          seg = &s;
          break;
        }
      }
      if (!seg)
        return absl::InvalidArgumentError("no segment for offset");
      uint64_t seg_off = pos - seg->va_offset;
      uint64_t seg_left = seg->length - seg_off;
      size_t step = static_cast<size_t>(std::min<uint64_t>(seg_left, bytes - total));
      if (pos >= data_.size())
        return absl::OutOfRangeError("source eof");
      size_t can_copy = std::min(step, data_.size() - static_cast<size_t>(pos));
      std::memcpy(reinterpret_cast<void*>(seg->local_addr + seg_off), data_.data() + pos, can_copy);
      total += can_copy;
      pos += can_copy;
    }
    return total;
  }

 private:
  std::vector<uint8_t> data_;
};

} // namespace

TEST_CASE("pump_ranges uses capability-driven direct writes when available", "[pump][direct]") {
  const size_t total_size = 32 * 1024;
  // Prepare source data with a pattern
  std::vector<uint8_t> src(total_size);
  for (size_t i = 0; i < total_size; ++i)
    src[i] = static_cast<uint8_t>((i * 3) % 251);

  DirectSource source(src);
  auto sink = std::make_shared<DirectCapableSink>(total_size);
  DummyPool pool(4096, 2);

  // Define three ranges across the buffer
  std::vector<Range> ranges = {{0, 4096}, {8192, 4096}, {16384, 8192}};

  auto st = pump_ranges(source, *sink, pool, absl::MakeSpan(ranges), 2);
  REQUIRE(st.ok());

  // Ensure staged path was not used
  REQUIRE(sink->write_calls() == 0);

  // Verify that only the specified ranges were written and match source
  const auto& buf = sink->buffer();
  for (const auto& r : ranges) {
    for (size_t i = 0; i < r.second; ++i) {
      REQUIRE(buf[r.first + i] == src[r.first + i]);
    }
  }
}
