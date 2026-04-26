// Copyright (c) 2026, TensorCast Team.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/materialization/contracts/byte_range/byte_range_map.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/sources/byte_range_mapped_source.h"
#include "core/store/materialization/dataplane/sources/byte_range_program.h"
#include "core/store/replica/types/direct_write_grant.h"

namespace tensorcast::store::loader {
namespace {

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
    const DirectWriteGrant::Window* target = nullptr;
    for (const auto& window : grant.windows) {
      if (dest_pos >= window.va_offset && dest_pos < window.va_offset + window.length) {
        target = &window;
        break;
      }
    }
    if (target == nullptr) {
      return absl::InvalidArgumentError("no direct-write window for destination offset");
    }
    const uint64_t window_offset = dest_pos - target->va_offset;
    const size_t available = static_cast<size_t>(target->length - window_offset);
    const size_t step = std::min(bytes - total, available);
    if (src_pos > data.size() || step > data.size() - src_pos) {
      return absl::OutOfRangeError("source eof");
    }
    std::memcpy(reinterpret_cast<void*>(target->local_addr + window_offset), data.data() + src_pos, step);
    total += step;
    src_pos += step;
    dest_pos += step;
  }
  return total;
}

class RecordingBatchDirectSource final : public SeekableSource {
 public:
  explicit RecordingBatchDirectSource(std::vector<uint8_t> data, bool batched_direct_write_supported = true)
      : data_(std::move(data)), batched_direct_write_supported_(batched_direct_write_supported) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(cursor_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or;
    }
    cursor_ += *read_or;
    return read_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size()) {
      return static_cast<size_t>(0);
    }
    const size_t to_read = std::min<size_t>(bytes, data_.size() - static_cast<size_t>(offset));
    std::memcpy(dst, data_.data() + offset, to_read);
    return to_read;
  }

  [[nodiscard]] bool supports_direct_write_at() const override {
    return true;
  }

  [[nodiscard]] bool supports_batched_direct_write_at() const override {
    return batched_direct_write_supported_;
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
      auto wrote_or = copy_into_grant(data_, op.src_offset, op.dest_va_offset, static_cast<size_t>(op.bytes), grant);
      if (!wrote_or.ok()) {
        return wrote_or.status();
      }
      total += *wrote_or;
    }
    return total;
  }

  void set_readv_failure(absl::Status status, bool partial_write_before_failure = false) {
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

 private:
  std::vector<uint8_t> data_;
  bool batched_direct_write_supported_{true};
  uint64_t cursor_{0};
  size_t readv_calls_{0};
  size_t read_into_calls_{0};
  std::vector<size_t> batch_sizes_;
  std::optional<absl::Status> readv_failure_;
  bool partial_write_before_failure_{false};
};

DirectWriteGrant make_grant(std::vector<uint8_t>& buffer) {
  DirectWriteGrant grant;
  grant.windows.push_back(
      DirectWriteGrant::Window{
          .va_offset = 0,
          .local_addr = reinterpret_cast<uint64_t>(buffer.data()),
          .length = buffer.size(),
      });
  return grant;
}

std::shared_ptr<const ByteRangeProgram> make_program(std::vector<ByteRangeRun> runs, uint64_t total_bytes) {
  auto program = std::make_shared<ByteRangeProgram>();
  program->total_bytes = total_bytes;
  program->runs = std::move(runs);
  program->run_starts.reserve(program->runs.size());
  for (const auto& run : program->runs) {
    program->run_starts.push_back(run.dst_begin);
  }
  program->has_strided_runs = false;
  return program;
}

std::vector<uint8_t> make_sequence(uint8_t start, size_t count) {
  std::vector<uint8_t> data(count, 0);
  for (size_t i = 0; i < count; ++i) {
    data[i] = static_cast<uint8_t>(start + i);
  }
  return data;
}

} // namespace

TEST_CASE(
    "ByteRangeMappedSource batches direct writes by source and zero-fills pads",
    "[byte_range_mapped_source][direct]") {
  auto source0 = std::make_shared<RecordingBatchDirectSource>(make_sequence(10, 16));
  auto source1 = std::make_shared<RecordingBatchDirectSource>(make_sequence(100, 8));

  ByteRangeMap map{
      .total_bytes = 16,
      .num_sources = 2,
      .segments =
          {
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 0,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 0,
              },
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 4,
                  .length = 4,
                  .src_offset = 8,
                  .source_index = 0,
              },
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kPad,
                  .dst_offset = 8,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 0,
              },
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 12,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 1,
              },
          },
  };
  auto program = make_program(
      {
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 0,
              .dst_end = 4,
              .source_index = 0,
              .src_begin = 0,
          },
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 4,
              .dst_end = 8,
              .source_index = 0,
              .src_begin = 8,
          },
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kPad,
              .dst_begin = 8,
              .dst_end = 12,
          },
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 12,
              .dst_end = 16,
              .source_index = 1,
              .src_begin = 0,
          },
      },
      /*total_bytes=*/16);

  auto source_or = ByteRangeMappedSource::Create(
      std::move(map),
      std::move(program),
      {source0, source1},
      ByteRangeMappedSource::Options{.path = "direct_batch_test"});
  REQUIRE(source_or.ok());

  std::vector<uint8_t> sink_buffer(16, 0xFF);
  auto grant = make_grant(sink_buffer);
  const std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 16},
  };

  auto wrote_or = (*source_or)->readv_into_at(ops, grant);
  REQUIRE(wrote_or.ok());
  REQUIRE(*wrote_or == 16);
  REQUIRE(source0->readv_calls() == 1);
  REQUIRE(source0->batch_sizes() == std::vector<size_t>{2});
  REQUIRE(source0->read_into_calls() == 0);
  REQUIRE(source1->readv_calls() == 1);
  REQUIRE(source1->batch_sizes() == std::vector<size_t>{1});
  REQUIRE(source1->read_into_calls() == 0);
  CHECK(sink_buffer[0] == 10);
  CHECK(sink_buffer[1] == 11);
  CHECK(sink_buffer[2] == 12);
  CHECK(sink_buffer[3] == 13);
  CHECK(sink_buffer[4] == 18);
  CHECK(sink_buffer[5] == 19);
  CHECK(sink_buffer[6] == 20);
  CHECK(sink_buffer[7] == 21);
  CHECK(sink_buffer[8] == 0);
  CHECK(sink_buffer[9] == 0);
  CHECK(sink_buffer[10] == 0);
  CHECK(sink_buffer[11] == 0);
  CHECK(sink_buffer[12] == 100);
  CHECK(sink_buffer[13] == 101);
  CHECK(sink_buffer[14] == 102);
  CHECK(sink_buffer[15] == 103);
}

TEST_CASE(
    "ByteRangeMappedSource skips vectored fallback probing for scalar-only children",
    "[byte_range_mapped_source][direct]") {
  auto source =
      std::make_shared<RecordingBatchDirectSource>(make_sequence(1, 16), /*batched_direct_write_supported=*/false);

  ByteRangeMap map{
      .total_bytes = 8,
      .num_sources = 1,
      .segments =
          {
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 0,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 0,
              },
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 4,
                  .length = 4,
                  .src_offset = 8,
                  .source_index = 0,
              },
          },
  };
  auto program = make_program(
      {
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 0,
              .dst_end = 4,
              .source_index = 0,
              .src_begin = 0,
          },
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 4,
              .dst_end = 8,
              .source_index = 0,
              .src_begin = 8,
          },
      },
      /*total_bytes=*/8);

  auto source_or = ByteRangeMappedSource::Create(
      std::move(map), std::move(program), {source}, ByteRangeMappedSource::Options{.path = "scalar_child_test"});
  REQUIRE(source_or.ok());

  std::vector<uint8_t> sink_buffer(8, 0);
  auto grant = make_grant(sink_buffer);
  const std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 8},
  };

  auto wrote_or = (*source_or)->readv_into_at(ops, grant);
  REQUIRE(wrote_or.ok());
  REQUIRE(*wrote_or == 8);
  REQUIRE(source->readv_calls() == 0);
  REQUIRE(source->read_into_calls() == 2);
  CHECK(sink_buffer[0] == 1);
  CHECK(sink_buffer[1] == 2);
  CHECK(sink_buffer[2] == 3);
  CHECK(sink_buffer[3] == 4);
  CHECK(sink_buffer[4] == 9);
  CHECK(sink_buffer[5] == 10);
  CHECK(sink_buffer[6] == 11);
  CHECK(sink_buffer[7] == 12);
}

TEST_CASE(
    "ByteRangeMappedSource rejects later invalid ops before issuing earlier writes",
    "[byte_range_mapped_source][direct]") {
  auto source = std::make_shared<RecordingBatchDirectSource>(make_sequence(1, 16));

  ByteRangeMap map{
      .total_bytes = 8,
      .num_sources = 1,
      .segments =
          {
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 0,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 0,
              },
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 4,
                  .length = 4,
                  .src_offset = 8,
                  .source_index = 0,
              },
          },
  };
  auto program = make_program(
      {
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 0,
              .dst_end = 4,
              .source_index = 0,
              .src_begin = 0,
          },
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 4,
              .dst_end = 8,
              .source_index = 0,
              .src_begin = 8,
          },
      },
      /*total_bytes=*/8);

  auto source_or = ByteRangeMappedSource::Create(
      std::move(map),
      std::move(program),
      {source},
      ByteRangeMappedSource::Options{.path = "prevalidate_invalid_later_op_test"});
  REQUIRE(source_or.ok());

  std::vector<uint8_t> sink_buffer(8, 0);
  auto grant = make_grant(sink_buffer);
  const std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 4},
      DirectWriteOp{.src_offset = 8, .dest_va_offset = 4, .bytes = 1},
  };

  auto wrote_or = (*source_or)->readv_into_at(ops, grant);
  REQUIRE_FALSE(wrote_or.ok());
  REQUIRE(wrote_or.status().code() == absl::StatusCode::kDataLoss);
  REQUIRE(source->readv_calls() == 0);
  REQUIRE(source->read_into_calls() == 0);
  REQUIRE(sink_buffer == std::vector<uint8_t>(sink_buffer.size(), 0));
}

TEST_CASE(
    "ByteRangeMappedSource falls back to per-op direct writes on pre-issue capability miss",
    "[byte_range_mapped_source][direct]") {
  auto source = std::make_shared<RecordingBatchDirectSource>(make_sequence(1, 16));
  source->set_readv_failure(absl::UnimplementedError("vectored direct write unavailable"));

  ByteRangeMap map{
      .total_bytes = 8,
      .num_sources = 1,
      .segments =
          {
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 0,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 0,
              },
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 4,
                  .length = 4,
                  .src_offset = 8,
                  .source_index = 0,
              },
          },
  };
  auto program = make_program(
      {
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 0,
              .dst_end = 4,
              .source_index = 0,
              .src_begin = 0,
          },
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 4,
              .dst_end = 8,
              .source_index = 0,
              .src_begin = 8,
          },
      },
      /*total_bytes=*/8);

  auto source_or = ByteRangeMappedSource::Create(
      std::move(map), std::move(program), {source}, ByteRangeMappedSource::Options{.path = "direct_fallback_test"});
  REQUIRE(source_or.ok());

  std::vector<uint8_t> sink_buffer(8, 0);
  auto grant = make_grant(sink_buffer);
  const std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 8},
  };

  auto wrote_or = (*source_or)->readv_into_at(ops, grant);
  REQUIRE(wrote_or.ok());
  REQUIRE(*wrote_or == 8);
  REQUIRE(source->readv_calls() == 1);
  REQUIRE(source->read_into_calls() == 2);
  CHECK(sink_buffer[0] == 1);
  CHECK(sink_buffer[1] == 2);
  CHECK(sink_buffer[2] == 3);
  CHECK(sink_buffer[3] == 4);
  CHECK(sink_buffer[4] == 9);
  CHECK(sink_buffer[5] == 10);
  CHECK(sink_buffer[6] == 11);
  CHECK(sink_buffer[7] == 12);
}

TEST_CASE(
    "ByteRangeMappedSource does not hide hard vectored direct-write failures",
    "[byte_range_mapped_source][direct]") {
  auto source = std::make_shared<RecordingBatchDirectSource>(make_sequence(1, 16));
  source->set_readv_failure(absl::AbortedError("synthetic transport failure"), /*partial_write_before_failure=*/true);

  ByteRangeMap map{
      .total_bytes = 8,
      .num_sources = 1,
      .segments =
          {
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 0,
                  .length = 4,
                  .src_offset = 0,
                  .source_index = 0,
              },
              ByteRangeSegment{
                  .kind = ByteRangeSegment::Kind::kData,
                  .dst_offset = 4,
                  .length = 4,
                  .src_offset = 8,
                  .source_index = 0,
              },
          },
  };
  auto program = make_program(
      {
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 0,
              .dst_end = 4,
              .source_index = 0,
              .src_begin = 0,
          },
          ByteRangeRun{
              .kind = ByteRangeRun::Kind::kContiguous,
              .dst_begin = 4,
              .dst_end = 8,
              .source_index = 0,
              .src_begin = 8,
          },
      },
      /*total_bytes=*/8);

  auto source_or = ByteRangeMappedSource::Create(
      std::move(map), std::move(program), {source}, ByteRangeMappedSource::Options{.path = "direct_hard_failure_test"});
  REQUIRE(source_or.ok());

  std::vector<uint8_t> sink_buffer(8, 0);
  auto grant = make_grant(sink_buffer);
  const std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = 8},
  };

  auto wrote_or = (*source_or)->readv_into_at(ops, grant);
  REQUIRE_FALSE(wrote_or.ok());
  REQUIRE(wrote_or.status().code() == absl::StatusCode::kAborted);
  REQUIRE(source->readv_calls() == 1);
  REQUIRE(source->read_into_calls() == 0);
  CHECK(sink_buffer[0] == 1);
  CHECK(sink_buffer[1] == 2);
  CHECK(sink_buffer[2] == 3);
  CHECK(sink_buffer[3] == 4);
  CHECK(sink_buffer[4] == 0);
  CHECK(sink_buffer[5] == 0);
  CHECK(sink_buffer[6] == 0);
  CHECK(sink_buffer[7] == 0);
}

} // namespace tensorcast::store::loader
