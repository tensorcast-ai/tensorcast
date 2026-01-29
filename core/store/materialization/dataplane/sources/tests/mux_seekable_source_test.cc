// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/sources/mux_seekable_source.h"

using namespace tensorcast::store::loader;

namespace {

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
