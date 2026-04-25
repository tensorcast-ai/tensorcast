// Copyright (c) 2025-2026, TensorCast Team.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

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

absl::StatusOr<size_t> copy_into_grant(
    absl::Span<const uint8_t> data,
    uint64_t src_offset,
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
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
    if (win == nullptr) {
      return absl::InvalidArgumentError("no direct-write window for destination offset");
    }
    const uint64_t seg_off = dest_pos - win->va_offset;
    const uint64_t seg_left = win->length - seg_off;
    const size_t step = static_cast<size_t>(std::min<uint64_t>(seg_left, bytes - total));
    if (src_pos >= data.size()) {
      return absl::OutOfRangeError("source eof");
    }
    const size_t can_copy = std::min(step, data.size() - static_cast<size_t>(src_pos));
    std::memcpy(reinterpret_cast<void*>(win->local_addr + seg_off), data.data() + src_pos, can_copy);
    total += can_copy;
    src_pos += can_copy;
    dest_pos += can_copy;
  }
  return total;
}

class DummyPool : public BufferPool {
 public:
  explicit DummyPool(size_t chunk_size, int capacity) : chunk_size_(chunk_size), capacity_(capacity) {}

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

class DirectCapableSink : public PositionedSink, public DirectWriteCapable {
 public:
  explicit DirectCapableSink(size_t size) : buffer_(size, 0) {}

  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override {
    ++write_calls_;
    if (offset + bytes > buffer_.size()) {
      return absl::InvalidArgumentError("overflow");
    }
    std::memcpy(buffer_.data() + offset, src, bytes);
    return absl::OkStatus();
  }

  absl::Status close() override {
    closed_ = true;
    return absl::OkStatus();
  }

  absl::StatusOr<DirectWriteGrant> plan_direct_write(absl::Span<const VaRange> ranges) override {
    ++plan_calls_;
    DirectWriteGrant grant;
    grant.windows.reserve(ranges.size());
    for (const auto& range : ranges) {
      if (range.offset + range.length > buffer_.size()) {
        return absl::OutOfRangeError("range beyond buffer");
      }
      grant.windows.push_back(
          DirectWriteGrant::Window{
              .va_offset = range.offset,
              .local_addr = reinterpret_cast<uint64_t>(buffer_.data() + range.offset),
              .length = range.length,
          });
    }
    grant.keepalive = nullptr;
    return grant;
  }

  const std::vector<uint8_t>& buffer() const {
    return buffer_;
  }

  size_t write_calls() const {
    return write_calls_;
  }

  size_t plan_calls() const {
    return plan_calls_;
  }

  bool closed() const {
    return closed_;
  }

 protected:
  std::vector<uint8_t> buffer_;
  size_t write_calls_ = 0;
  size_t plan_calls_ = 0;
  bool closed_ = false;
};

class PlanFailingSink final : public DirectCapableSink {
 public:
  PlanFailingSink(size_t size, size_t fail_plan_calls) : DirectCapableSink(size), fail_plan_calls_(fail_plan_calls) {}

  absl::StatusOr<DirectWriteGrant> plan_direct_write(absl::Span<const VaRange> ranges) override {
    if (plan_calls_ < fail_plan_calls_) {
      ++plan_calls_;
      return absl::UnavailableError("synthetic plan failure");
    }
    return DirectCapableSink::plan_direct_write(ranges);
  }

 private:
  size_t fail_plan_calls_ = 0;
};

class DefaultLoopDirectSource final : public SeekableSource {
 public:
  explicit DefaultLoopDirectSource(std::vector<uint8_t> data) : data_(std::move(data)) {}

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
    if (offset >= data_.size()) {
      return size_t{0};
    }
    const size_t to_read = std::min<size_t>(bytes, data_.size() - static_cast<size_t>(offset));
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
    ++read_into_calls_;
    if (fail_on_call_.has_value() && read_into_calls_ == *fail_on_call_) {
      return absl::AbortedError("synthetic read_into failure");
    }
    return copy_into_grant(data_, src_offset, dest_va_offset, bytes, grant);
  }

  void set_fail_on_call(size_t call_index) {
    fail_on_call_ = call_index;
  }

  size_t read_into_calls() const {
    return read_into_calls_;
  }

 private:
  std::vector<uint8_t> data_;
  uint64_t current_offset_{0};
  size_t read_into_calls_ = 0;
  std::optional<size_t> fail_on_call_;
};

class RecordingBatchDirectSource final : public SeekableSource {
 public:
  explicit RecordingBatchDirectSource(std::vector<uint8_t> data) : data_(std::move(data)) {}

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
    if (offset >= data_.size()) {
      return size_t{0};
    }
    const size_t to_read = std::min<size_t>(bytes, data_.size() - static_cast<size_t>(offset));
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
    ++read_into_calls_;
    return copy_into_grant(data_, src_offset, dest_va_offset, bytes, grant);
  }

  absl::StatusOr<size_t> readv_into_at(absl::Span<const DirectWriteOp> ops, const DirectWriteGrant& grant) override {
    ++readv_calls_;
    batch_sizes_.push_back(ops.size());
    size_t batch_bytes = 0;
    for (const auto& op : ops) {
      batch_bytes += static_cast<size_t>(op.bytes);
    }
    batch_bytes_.push_back(batch_bytes);
    if (readv_failure_.has_value()) {
      if (partial_write_before_failure_ && !ops.empty()) {
        auto first_write_or = copy_into_grant(
            data_, ops.front().src_offset, ops.front().dest_va_offset, static_cast<size_t>(ops.front().bytes), grant);
        if (!first_write_or.ok()) {
          return first_write_or.status();
        }
      }
      return *readv_failure_;
    }
    size_t total = 0;
    for (const auto& op : ops) {
      auto write_or = copy_into_grant(data_, op.src_offset, op.dest_va_offset, static_cast<size_t>(op.bytes), grant);
      if (!write_or.ok()) {
        return write_or.status();
      }
      total += *write_or;
    }
    return total;
  }

  void set_readv_failure(absl::Status status, bool partial_write_before_failure) {
    readv_failure_ = std::move(status);
    partial_write_before_failure_ = partial_write_before_failure;
  }

  size_t readv_calls() const {
    return readv_calls_;
  }

  size_t read_into_calls() const {
    return read_into_calls_;
  }

  const std::vector<size_t>& batch_sizes() const {
    return batch_sizes_;
  }

  const std::vector<size_t>& batch_bytes() const {
    return batch_bytes_;
  }

 private:
  std::vector<uint8_t> data_;
  uint64_t current_offset_{0};
  size_t readv_calls_ = 0;
  size_t read_into_calls_ = 0;
  std::vector<size_t> batch_sizes_;
  std::vector<size_t> batch_bytes_;
  std::optional<absl::Status> readv_failure_;
  bool partial_write_before_failure_ = false;
};

std::vector<uint8_t> make_pattern(size_t total_size) {
  std::vector<uint8_t> data(total_size, 0);
  for (size_t i = 0; i < total_size; ++i) {
    data[i] = static_cast<uint8_t>((i * 3) % 251);
  }
  return data;
}

void expect_ranges_match(
    const std::vector<uint8_t>& actual,
    const std::vector<uint8_t>& expected,
    absl::Span<const Range> ranges) {
  for (const auto& [offset, length] : ranges) {
    for (size_t i = 0; i < length; ++i) {
      REQUIRE(actual[offset + i] == expected[offset + i]);
    }
  }
}

} // namespace

TEST_CASE("SeekableSource default readv_into_at loops through read_into_at operations", "[pump][direct]") {
  auto source_bytes = make_pattern(8192);
  DefaultLoopDirectSource source(source_bytes);
  DirectCapableSink sink(source_bytes.size());

  std::vector<VaRange> ranges = {{0, 1024}, {4096, 512}};
  auto grant_or = sink.plan_direct_write(ranges);
  REQUIRE(grant_or.ok());

  std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 1024},
      DirectWriteOp{.src_offset = 4096, .dest_va_offset = 4096, .bytes = 512},
  };
  auto wrote_or = source.readv_into_at(ops, *grant_or);
  REQUIRE(wrote_or.ok());
  REQUIRE(*wrote_or == 1536);
  REQUIRE(source.read_into_calls() == ops.size());
  const std::vector<Range> expected_ranges = {{0, 1024}, {4096, 512}};
  expect_ranges_match(sink.buffer(), source_bytes, absl::MakeConstSpan(expected_ranges));
}

TEST_CASE("SeekableSource default readv_into_at treats an empty batch as a no-op", "[pump][direct]") {
  auto source_bytes = make_pattern(4096);
  DefaultLoopDirectSource source(source_bytes);
  DirectWriteGrant empty_grant;

  const std::vector<DirectWriteOp> ops;
  auto wrote_or = source.readv_into_at(ops, empty_grant);

  REQUIRE(wrote_or.ok());
  REQUIRE(*wrote_or == 0);
  REQUIRE(source.read_into_calls() == 0);
}

TEST_CASE("SeekableSource default readv_into_at propagates the first read_into_at error", "[pump][direct]") {
  auto source_bytes = make_pattern(8192);
  DefaultLoopDirectSource source(source_bytes);
  source.set_fail_on_call(2);
  DirectCapableSink sink(source_bytes.size());

  std::vector<VaRange> ranges = {{0, 1024}, {2048, 1024}};
  auto grant_or = sink.plan_direct_write(ranges);
  REQUIRE(grant_or.ok());

  std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 1024},
      DirectWriteOp{.src_offset = 2048, .dest_va_offset = 2048, .bytes = 1024},
  };
  auto wrote_or = source.readv_into_at(ops, *grant_or);
  REQUIRE_FALSE(wrote_or.ok());
  REQUIRE(wrote_or.status().code() == absl::StatusCode::kAborted);
  REQUIRE(source.read_into_calls() == 2);
  const std::vector<Range> expected_ranges = {{0, 1024}};
  expect_ranges_match(sink.buffer(), source_bytes, absl::MakeConstSpan(expected_ranges));
  for (size_t i = 0; i < 1024; ++i) {
    REQUIRE(sink.buffer()[2048 + i] == 0);
  }
}

TEST_CASE("pump_ranges batches direct writes into readv_into_at calls when available", "[pump][direct]") {
  constexpr size_t kTotalSize = 32 * 1024;
  auto source_bytes = make_pattern(kTotalSize);
  RecordingBatchDirectSource source(source_bytes);
  auto sink = std::make_shared<DirectCapableSink>(kTotalSize);
  DummyPool pool(4096, 2);

  std::vector<Range> ranges = {{0, 4096}, {8192, 4096}, {16384, 8192}};
  PumpDirectWriteOptions options{
      .direct_write_batch_bytes = 64 * 1024,
      .direct_write_batch_ops = 8,
  };

  auto status = pump_ranges(
      source, *sink, pool, absl::MakeSpan(ranges), 2, pump_direct_test_runtime().blocking_executor(), nullptr, options);
  REQUIRE(status.ok());
  REQUIRE(source.readv_calls() == 1);
  REQUIRE(source.read_into_calls() == 0);
  REQUIRE(source.batch_sizes() == std::vector<size_t>{4});
  REQUIRE(sink->write_calls() == 0);
  REQUIRE(sink->closed());
  expect_ranges_match(sink->buffer(), source_bytes, absl::MakeConstSpan(ranges));
}

TEST_CASE("pump_ranges_direct_write batches direct writes without a buffer pool", "[pump][direct]") {
  constexpr size_t kTotalSize = 32 * 1024;
  auto source_bytes = make_pattern(kTotalSize);
  RecordingBatchDirectSource source(source_bytes);
  auto sink = std::make_shared<DirectCapableSink>(kTotalSize);

  std::vector<Range> ranges = {{0, 4096}, {8192, 4096}, {16384, 8192}};
  PumpDirectWriteOptions options{
      .direct_write_batch_bytes = 64 * 1024,
      .direct_write_batch_ops = 8,
  };

  auto status =
      pump_ranges_direct_write(source, *sink, absl::MakeSpan(ranges), /*window_bytes=*/4096, 2, nullptr, options);
  REQUIRE(status.ok());
  REQUIRE(source.readv_calls() == 1);
  REQUIRE(source.read_into_calls() == 0);
  REQUIRE(source.batch_sizes() == std::vector<size_t>{4});
  REQUIRE(sink->write_calls() == 0);
  REQUIRE_FALSE(sink->closed());
  expect_ranges_match(sink->buffer(), source_bytes, absl::MakeConstSpan(ranges));
  REQUIRE(sink->close().ok());
}

TEST_CASE("pump_ranges uses default direct-write batch limits when options are omitted", "[pump][direct]") {
  constexpr size_t kWindowBytes = 1024;
  constexpr size_t kWindowCount = 10;
  auto source_bytes = make_pattern(kWindowBytes * kWindowCount);
  RecordingBatchDirectSource source(source_bytes);
  auto sink = std::make_shared<DirectCapableSink>(source_bytes.size());
  DummyPool pool(kWindowBytes, 2);

  std::vector<Range> ranges = {{0, source_bytes.size()}};

  auto status =
      pump_ranges(source, *sink, pool, absl::MakeSpan(ranges), 1, pump_direct_test_runtime().blocking_executor());
  REQUIRE(status.ok());
  REQUIRE(source.batch_sizes() == std::vector<size_t>{8, 2});
  REQUIRE(source.batch_bytes() == std::vector<size_t>{8 * kWindowBytes, 2 * kWindowBytes});
  expect_ranges_match(sink->buffer(), source_bytes, absl::MakeConstSpan(ranges));
}

TEST_CASE("pump_ranges splits direct-write batches by op count limit", "[pump][direct]") {
  constexpr size_t kWindowBytes = 4096;
  constexpr size_t kWindowCount = 5;
  auto source_bytes = make_pattern(kWindowBytes * kWindowCount);
  RecordingBatchDirectSource source(source_bytes);
  auto sink = std::make_shared<DirectCapableSink>(source_bytes.size());
  DummyPool pool(kWindowBytes, 2);

  std::vector<Range> ranges = {{0, source_bytes.size()}};
  PumpDirectWriteOptions options{
      .direct_write_batch_bytes = source_bytes.size(),
      .direct_write_batch_ops = 2,
  };

  auto status = pump_ranges(
      source, *sink, pool, absl::MakeSpan(ranges), 1, pump_direct_test_runtime().blocking_executor(), nullptr, options);
  REQUIRE(status.ok());
  REQUIRE(source.batch_sizes() == std::vector<size_t>{2, 2, 1});
  expect_ranges_match(sink->buffer(), source_bytes, absl::MakeConstSpan(ranges));
}

TEST_CASE("pump_ranges splits direct-write batches by byte limit", "[pump][direct]") {
  constexpr size_t kWindowBytes = 1024;
  auto source_bytes = make_pattern(kWindowBytes * 3);
  RecordingBatchDirectSource source(source_bytes);
  auto sink = std::make_shared<DirectCapableSink>(source_bytes.size());
  DummyPool pool(kWindowBytes, 2);

  std::vector<Range> ranges = {{0, source_bytes.size()}};
  PumpDirectWriteOptions options{
      .direct_write_batch_bytes = 2 * kWindowBytes,
      .direct_write_batch_ops = 8,
  };

  auto status = pump_ranges(
      source, *sink, pool, absl::MakeSpan(ranges), 1, pump_direct_test_runtime().blocking_executor(), nullptr, options);
  REQUIRE(status.ok());
  REQUIRE(source.batch_sizes() == std::vector<size_t>{2, 1});
  REQUIRE(source.batch_bytes() == std::vector<size_t>{2 * kWindowBytes, kWindowBytes});
  expect_ranges_match(sink->buffer(), source_bytes, absl::MakeConstSpan(ranges));
}

TEST_CASE("pump_ranges falls back to staged writes when grant planning fails before issue", "[pump][direct]") {
  constexpr size_t kTotalSize = 8192;
  auto source_bytes = make_pattern(kTotalSize);
  RecordingBatchDirectSource source(source_bytes);
  PlanFailingSink sink(kTotalSize, /*fail_plan_calls=*/1);
  DummyPool pool(4096, 2);

  std::vector<Range> ranges = {{0, kTotalSize}};
  PumpDirectWriteOptions options{
      .direct_write_batch_bytes = kTotalSize,
      .direct_write_batch_ops = 8,
  };

  auto status = pump_ranges(
      source, sink, pool, absl::MakeSpan(ranges), 1, pump_direct_test_runtime().blocking_executor(), nullptr, options);
  REQUIRE(status.ok());
  REQUIRE(source.readv_calls() == 0);
  REQUIRE(sink.write_calls() > 0);
  expect_ranges_match(sink.buffer(), source_bytes, absl::MakeConstSpan(ranges));
}

TEST_CASE("pump_ranges surfaces post-issue direct-write failure without staged fallback", "[pump][direct]") {
  constexpr size_t kTotalSize = 8192;
  auto source_bytes = make_pattern(kTotalSize);
  RecordingBatchDirectSource source(source_bytes);
  source.set_readv_failure(absl::AbortedError("synthetic readv failure"), /*partial_write_before_failure=*/true);
  DirectCapableSink sink(kTotalSize);
  DummyPool pool(4096, 2);

  std::vector<Range> ranges = {{0, kTotalSize}};
  PumpDirectWriteOptions options{
      .direct_write_batch_bytes = kTotalSize,
      .direct_write_batch_ops = 8,
  };

  auto status = pump_ranges(
      source, sink, pool, absl::MakeSpan(ranges), 1, pump_direct_test_runtime().blocking_executor(), nullptr, options);
  REQUIRE_FALSE(status.ok());
  REQUIRE(status.code() == absl::StatusCode::kAborted);
  REQUIRE(source.readv_calls() == 1);
  REQUIRE(sink.write_calls() == 0);
  const std::vector<Range> expected_ranges = {{0, 4096}};
  expect_ranges_match(sink.buffer(), source_bytes, absl::MakeConstSpan(expected_ranges));
  for (size_t i = 4096; i < kTotalSize; ++i) {
    REQUIRE(sink.buffer()[i] == 0);
  }
}
