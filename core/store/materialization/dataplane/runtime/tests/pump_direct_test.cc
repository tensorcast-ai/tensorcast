// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/common/async_runtime.h"
#include "core/store/materialization/dataplane/contracts/buffer_pool.h"
#include "core/store/materialization/dataplane/contracts/sink.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/runtime/pump.h"
#include "core/store/replica/types/direct_write_grant.h"

using namespace tensorcast::store;
using namespace tensorcast::store::loader;

namespace {

static tensorcast::common::AsyncRuntime& pump_direct_test_runtime() {
  static tensorcast::common::AsyncRuntime runtime(
      tensorcast::common::AsyncRuntime::Options{
          .thread_name_prefix = "tensorcast-pump-direct-test",
      });
  return runtime;
}

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

  void shutdown() override {}

  void* get_chunk_data_ptr(int) override {
    return nullptr;
  }

 private:
  size_t chunk_size_;
  int capacity_;
};

// Direct-capable sink that exposes a buffer and plans token segments.
class DirectCapableSink : public PositionedSink, public DirectWriteCapable {
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

  absl::StatusOr<DirectWriteGrant> plan_direct_write(absl::Span<const VaRange> ranges) override {
    DirectWriteGrant grant;
    grant.windows.reserve(ranges.size());
    for (const auto& r : ranges) {
      if (r.offset + r.length > buffer_.size())
        return absl::OutOfRangeError("range beyond buffer");
      grant.windows.push_back(
          DirectWriteGrant::Window{
              .va_offset = r.offset,
              .local_addr = reinterpret_cast<uint64_t>(buffer_.data() + r.offset),
              .length = r.length});
    }
    grant.keepalive = nullptr; // Not needed in test
    return grant;
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

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(current_offset_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or;
    }
    current_offset_ += *read_or;
    return read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size())
      return size_t{0};
    size_t to_read = std::min<size_t>(bytes, data_.size() - offset);
    std::memcpy(dst, data_.data() + offset, to_read);
    return to_read;
  }

  bool supports_direct_write_at() const override {
    return true;
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override {
    size_t total = 0;
    uint64_t src_pos = src_offset;
    uint64_t dest_pos = dest_va_offset;
    while (total < bytes) {
      const DirectWriteGrant::Window* win = nullptr;
      for (const auto& w : grant.windows) {
        if (dest_pos >= w.va_offset && dest_pos < w.va_offset + w.length) {
          win = &w;
          break;
        }
      }
      if (!win)
        return absl::InvalidArgumentError("no segment for offset");
      uint64_t seg_off = dest_pos - win->va_offset;
      uint64_t seg_left = win->length - seg_off;
      size_t step = static_cast<size_t>(std::min<uint64_t>(seg_left, bytes - total));
      if (src_pos >= data_.size())
        return absl::OutOfRangeError("source eof");
      size_t can_copy = std::min(step, data_.size() - static_cast<size_t>(src_pos));
      std::memcpy(reinterpret_cast<void*>(win->local_addr + seg_off), data_.data() + src_pos, can_copy);
      total += can_copy;
      src_pos += can_copy;
      dest_pos += can_copy;
    }
    return total;
  }

 private:
  std::vector<uint8_t> data_;
  uint64_t current_offset_{0};
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

  auto st = pump_ranges(source, *sink, pool, absl::MakeSpan(ranges), 2, pump_direct_test_runtime().blocking_executor());
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
