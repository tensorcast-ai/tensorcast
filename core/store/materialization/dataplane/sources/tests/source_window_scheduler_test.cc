// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/sources/source_window_scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"

namespace tensorcast::store::loader {
namespace {

class TestSource final : public SeekableSource {
 public:
  explicit TestSource(std::vector<uint8_t> data) : data_(std::move(data)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    if (cursor_ >= data_.size()) {
      return static_cast<size_t>(0);
    }
    const size_t remaining = data_.size() - cursor_;
    const size_t to_copy = std::min(max_bytes, remaining);
    std::memcpy(dst, data_.data() + cursor_, to_copy);
    cursor_ += to_copy;
    return to_copy;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    read_calls_ += 1;
    if (offset > data_.size()) {
      return absl::OutOfRangeError("read_at offset out of range");
    }
    if (bytes > data_.size() - offset) {
      return absl::OutOfRangeError("read_at length out of range");
    }
    if (bytes == 0) {
      return static_cast<size_t>(0);
    }
    std::memcpy(dst, data_.data() + offset, bytes);
    return bytes;
  }

  size_t read_calls() const {
    return read_calls_;
  }

 private:
  std::vector<uint8_t> data_;
  size_t cursor_{0};
  size_t read_calls_{0};
};

class BufferSink final : public PositionedSink {
 public:
  explicit BufferSink(size_t size) : buffer_(size, 0) {}

  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override {
    if (offset > buffer_.size()) {
      return absl::OutOfRangeError("sink write offset out of range");
    }
    if (bytes > buffer_.size() - offset) {
      return absl::OutOfRangeError("sink write length out of range");
    }
    if (bytes == 0) {
      return absl::OkStatus();
    }
    std::memcpy(buffer_.data() + offset, src, bytes);
    return absl::OkStatus();
  }

  const std::vector<uint8_t>& buffer() const {
    return buffer_;
  }

 private:
  std::vector<uint8_t> buffer_;
};

ByteRangeMap make_gap_map() {
  ByteRangeMap map;
  map.total_bytes = 12;
  map.num_sources = 1;
  map.segments = {
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kPad,
          .dst_offset = 4,
          .length = 4,
          .src_offset = 0,
          .source_index = 0,
      },
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 8,
          .length = 4,
          .src_offset = 8,
          .source_index = 0,
      },
  };
  return map;
}

} // namespace

TEST_CASE("SourceWindowScheduler merges adjacent source windows", "[source_window_scheduler]") {
  std::vector<uint8_t> data(12);
  std::iota(data.begin(), data.end(), 0);
  auto source = std::make_shared<TestSource>(data);

  BufferSink sink(/*size=*/12);
  SourceWindowScheduler::Options opts{
      .merge_max_gap_bytes = 4,
      .merge_max_amplification = 4,
      .prefetch_depth = 1,
      .window_cap_bytes = 0,
      .path = "test",
  };
  SourceWindowScheduler scheduler(opts);

  auto status = scheduler.Execute(
      make_gap_map(),
      absl::Span<const std::shared_ptr<SeekableSource>>{source},
      sink,
      /*pinned_pool=*/nullptr,
      /*pinned_timeout=*/std::chrono::milliseconds(0),
      /*use_pinned_buffers=*/false);
  REQUIRE(status.ok());
  CHECK(source->read_calls() == 1);

  const auto& out = sink.buffer();
  REQUIRE(out.size() == 12);
  CHECK(out[0] == 0);
  CHECK(out[1] == 1);
  CHECK(out[2] == 2);
  CHECK(out[3] == 3);
  CHECK(out[4] == 0);
  CHECK(out[5] == 0);
  CHECK(out[6] == 0);
  CHECK(out[7] == 0);
  CHECK(out[8] == 8);
  CHECK(out[9] == 9);
  CHECK(out[10] == 10);
  CHECK(out[11] == 11);
}

TEST_CASE("SourceWindowScheduler avoids merging when gaps exceed limits", "[source_window_scheduler]") {
  std::vector<uint8_t> data(12);
  std::iota(data.begin(), data.end(), 0);
  auto source = std::make_shared<TestSource>(data);

  BufferSink sink(/*size=*/12);
  SourceWindowScheduler::Options opts{
      .merge_max_gap_bytes = 0,
      .merge_max_amplification = 4,
      .prefetch_depth = 1,
      .window_cap_bytes = 0,
      .path = "test",
  };
  SourceWindowScheduler scheduler(opts);

  auto status = scheduler.Execute(
      make_gap_map(),
      absl::Span<const std::shared_ptr<SeekableSource>>{source},
      sink,
      /*pinned_pool=*/nullptr,
      /*pinned_timeout=*/std::chrono::milliseconds(0),
      /*use_pinned_buffers=*/false);
  REQUIRE(status.ok());
  CHECK(source->read_calls() == 2);
}

TEST_CASE("SourceWindowScheduler splits windows by cap", "[source_window_scheduler]") {
  ByteRangeMap map;
  map.total_bytes = 16;
  map.num_sources = 1;
  map.segments = {
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = 16,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  std::vector<uint8_t> data(16);
  std::iota(data.begin(), data.end(), 0);
  auto source = std::make_shared<TestSource>(data);

  BufferSink sink(/*size=*/16);
  SourceWindowScheduler::Options opts{
      .merge_max_gap_bytes = 0,
      .merge_max_amplification = 1,
      .prefetch_depth = 1,
      .window_cap_bytes = 8,
      .path = "test",
  };
  SourceWindowScheduler scheduler(opts);

  auto status = scheduler.Execute(
      map,
      absl::Span<const std::shared_ptr<SeekableSource>>{source},
      sink,
      /*pinned_pool=*/nullptr,
      /*pinned_timeout=*/std::chrono::milliseconds(0),
      /*use_pinned_buffers=*/false);
  REQUIRE(status.ok());
  CHECK(source->read_calls() == 2);
}

TEST_CASE("SourceWindowScheduler uses pinned slices for windows", "[source_window_scheduler]") {
  (void)setenv("TENSORCAST_CUDA_BACKEND", "fake", /*overwrite=*/0);

  constexpr size_t kSliceBytes = 1024 * 1024;
  constexpr size_t kTotalBytes = kSliceBytes;

  auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(
      /*total_size=*/kSliceBytes * 4,
      /*chunk_size=*/kSliceBytes,
      /*name=*/"test");

  ByteRangeMap map;
  map.total_bytes = kTotalBytes;
  map.num_sources = 1;
  map.segments = {
      ByteRangeSegment{
          .kind = ByteRangeSegment::Kind::kData,
          .dst_offset = 0,
          .length = kTotalBytes,
          .src_offset = 0,
          .source_index = 0,
      },
  };

  std::vector<uint8_t> data(kTotalBytes);
  std::iota(data.begin(), data.end(), 0);
  auto source = std::make_shared<TestSource>(data);

  BufferSink sink(/*size=*/kTotalBytes);
  SourceWindowScheduler::Options opts{
      .merge_max_gap_bytes = 0,
      .merge_max_amplification = 1,
      .prefetch_depth = 2,
      .window_cap_bytes = kSliceBytes / 4,
      .path = "test",
  };
  SourceWindowScheduler scheduler(opts);

  auto status = scheduler.Execute(
      map,
      absl::Span<const std::shared_ptr<SeekableSource>>{source},
      sink,
      pinned_pool,
      /*pinned_timeout=*/std::chrono::milliseconds(0),
      /*use_pinned_buffers=*/true);
  REQUIRE(status.ok());
  CHECK(source->read_calls() == 4);
  CHECK(sink.buffer() == data);
}

} // namespace tensorcast::store::loader
