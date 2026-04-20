// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/sources/mux_seekable_source.h"
#include "core/store/replica/types/direct_write_grant.h"

using namespace tensorcast::store::loader;
using tensorcast::store::DirectWriteGrant;

namespace {

absl::StatusOr<size_t> copy_into_grant(
    const std::string& data,
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

class PrimaryErrorSource : public SeekableSource {
 public:
  explicit PrimaryErrorSource(size_t total) : total_(total) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_;
  }

  absl::StatusOr<size_t> read(void* /*dst*/, size_t /*max_bytes*/) override {
    return absl::InternalError("primary error");
  }

  absl::StatusOr<size_t> read_at(uint64_t /*offset*/, void* /*dst*/, size_t /*bytes*/) override {
    return absl::InternalError("primary error");
  }

 private:
  size_t total_;
};

class FallbackMemorySource : public SeekableSource {
 public:
  explicit FallbackMemorySource(std::string data) : data_(std::move(data)) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return read_at(offset_, dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size())
      return static_cast<size_t>(0);
    size_t to_read = std::min<size_t>(bytes, data_.size() - static_cast<size_t>(offset));
    std::memcpy(dst, data_.data() + offset, to_read);
    offset_ = offset + to_read;
    return to_read;
  }

 private:
  std::string data_;
  uint64_t offset_{0};
};

class RecordingDirectSource final : public SeekableSource {
 public:
  explicit RecordingDirectSource(std::string data, bool direct_write_supported = true)
      : data_(std::move(data)), direct_write_supported_(direct_write_supported) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return data_.size();
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto read_or = read_at(offset_, dst, max_bytes);
    if (!read_or.ok()) {
      return read_or;
    }
    offset_ += *read_or;
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
    return direct_write_supported_;
  }

  [[nodiscard]] bool supports_batched_direct_write_at() const override {
    return direct_write_supported_;
  }

  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override {
    ++read_into_calls_;
    if (read_into_failure_.has_value()) {
      return *read_into_failure_;
    }
    return copy_into_grant(data_, src_offset, dest_va_offset, bytes, grant);
  }

  absl::StatusOr<size_t> readv_into_at(absl::Span<const DirectWriteOp> ops, const DirectWriteGrant& grant) override {
    ++readv_calls_;
    if (readv_failure_.has_value()) {
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

  void set_read_into_failure(absl::Status status) {
    read_into_failure_ = std::move(status);
  }

  void set_readv_failure(absl::Status status) {
    readv_failure_ = std::move(status);
  }

  size_t read_into_calls() const {
    return read_into_calls_;
  }

  size_t readv_calls() const {
    return readv_calls_;
  }

 private:
  std::string data_;
  bool direct_write_supported_{true};
  uint64_t offset_{0};
  size_t read_into_calls_{0};
  size_t readv_calls_{0};
  std::optional<absl::Status> read_into_failure_;
  std::optional<absl::Status> readv_failure_;
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

} // namespace

TEST_CASE("MuxSeekableSource falls back on primary error") {
  std::string payload(4096, '\0');
  for (size_t i = 0; i < payload.size(); ++i)
    payload[i] = static_cast<char>('a' + (i % 26));

  auto primary = std::make_shared<PrimaryErrorSource>(payload.size());
  auto fallback = std::make_shared<FallbackMemorySource>(payload);
  MuxSeekableSource mux(primary, fallback);

  std::string out(payload.size(), '\0');
  auto st = mux.read_at(0, out.data(), out.size());
  REQUIRE(st.ok());
  REQUIRE(*st == out.size());
  REQUIRE(out == payload);
}

TEST_CASE("MuxSeekableSource direct write freezes to the primary branch", "[mux_seekable_source][direct]") {
  auto primary = std::make_shared<RecordingDirectSource>("primary!");
  auto fallback = std::make_shared<RecordingDirectSource>("fallback");
  MuxSeekableSource mux(primary, fallback);

  std::vector<uint8_t> sink_buffer(primary->total_bytes(), 0);
  auto grant = make_grant(sink_buffer);
  const std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = primary->total_bytes()},
  };

  auto wrote_or = mux.readv_into_at(ops, grant);
  REQUIRE(wrote_or.ok());
  REQUIRE(*wrote_or == primary->total_bytes());
  REQUIRE(primary->readv_calls() == 1);
  REQUIRE(fallback->readv_calls() == 0);
  REQUIRE(std::string(sink_buffer.begin(), sink_buffer.end()) == "primary!");
}

TEST_CASE(
    "MuxSeekableSource direct write returns capability miss when neither branch supports it",
    "[mux_seekable_source][direct]") {
  auto primary = std::make_shared<RecordingDirectSource>("primary", /*direct_write_supported=*/false);
  auto fallback = std::make_shared<RecordingDirectSource>("fallback", /*direct_write_supported=*/false);
  MuxSeekableSource mux(primary, fallback);

  std::vector<uint8_t> sink_buffer(8, 0);
  auto grant = make_grant(sink_buffer);

  auto wrote_or = mux.read_into_at(/*src_offset=*/0, /*dest_va_offset=*/0, /*bytes=*/4, grant);
  REQUIRE_FALSE(wrote_or.ok());
  REQUIRE(wrote_or.status().code() == absl::StatusCode::kUnimplemented);
  REQUIRE(primary->read_into_calls() == 0);
  REQUIRE(fallback->read_into_calls() == 0);
}

TEST_CASE(
    "MuxSeekableSource direct write does not switch to fallback after issue-time failure",
    "[mux_seekable_source][direct]") {
  auto primary = std::make_shared<RecordingDirectSource>("primary!");
  auto fallback = std::make_shared<RecordingDirectSource>("fallback");
  primary->set_readv_failure(absl::AbortedError("synthetic primary failure"));
  MuxSeekableSource mux(primary, fallback);

  std::vector<uint8_t> sink_buffer(primary->total_bytes(), 0);
  auto grant = make_grant(sink_buffer);
  const std::vector<DirectWriteOp> ops = {
      DirectWriteOp{.src_offset = 0, .dest_va_offset = 0, .bytes = primary->total_bytes()},
  };

  auto wrote_or = mux.readv_into_at(ops, grant);
  REQUIRE_FALSE(wrote_or.ok());
  REQUIRE(wrote_or.status().code() == absl::StatusCode::kAborted);
  REQUIRE(primary->readv_calls() == 1);
  REQUIRE(fallback->readv_calls() == 0);
  REQUIRE(sink_buffer == std::vector<uint8_t>(sink_buffer.size(), 0));
}
