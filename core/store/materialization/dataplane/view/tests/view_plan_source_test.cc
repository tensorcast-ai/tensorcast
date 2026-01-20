// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_plan_source.h"

#include <array>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/store/materialization/dataplane/view/view_planner.h"

using tensorcast::store::loader::SelectionPlan;
using tensorcast::store::loader::ViewPlanSource;

namespace {

class VectorSource final : public tensorcast::store::loader::SeekableSource {
 public:
  explicit VectorSource(std::vector<uint8_t> data) : data_(std::move(data)) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    auto bytes_or = read_at(cursor_, dst, max_bytes);
    if (!bytes_or.ok()) {
      return bytes_or;
    }
    cursor_ += *bytes_or;
    return bytes_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= data_.size() || bytes == 0) {
      return static_cast<size_t>(0);
    }
    const size_t available = static_cast<size_t>(std::min<uint64_t>(bytes, data_.size() - offset));
    std::memcpy(dst, data_.data() + offset, available);
    read_calls_.fetch_add(1, std::memory_order_relaxed);
    read_bytes_.fetch_add(available, std::memory_order_relaxed);
    return available;
  }

  uint64_t read_calls() const {
    return read_calls_.load(std::memory_order_relaxed);
  }

  uint64_t read_bytes() const {
    return read_bytes_.load(std::memory_order_relaxed);
  }

 private:
  std::vector<uint8_t> data_;
  uint64_t cursor_{0};
  std::atomic<uint64_t> read_calls_{0};
  std::atomic<uint64_t> read_bytes_{0};
};

SelectionPlan::Range data_range(uint64_t src, uint64_t dst, uint64_t len) {
  SelectionPlan::Range r;
  r.kind = SelectionPlan::Range::Kind::kData;
  r.src_offset = src;
  r.dst_offset = dst;
  r.length = len;
  return r;
}

SelectionPlan::Range pad_range(uint64_t dst, uint64_t len) {
  SelectionPlan::Range r;
  r.kind = SelectionPlan::Range::Kind::kPad;
  r.src_offset = 0;
  r.dst_offset = dst;
  r.length = len;
  return r;
}

SelectionPlan build_sample_plan() {
  SelectionPlan plan;
  plan.ranges.push_back(data_range(/*src=*/4, /*dst=*/0, /*len=*/4));
  plan.ranges.push_back(data_range(/*src=*/16, /*dst=*/4, /*len=*/4));
  plan.ranges.push_back(pad_range(/*dst=*/8, /*len=*/4));
  plan.ranges.push_back(data_range(/*src=*/24, /*dst=*/12, /*len=*/4));
  plan.total_bytes = 16;
  plan.num_ranges = static_cast<uint32_t>(plan.ranges.size());
  plan.is_contiguous = false;
  plan.is_segment_aligned = false;
  plan.requires_materialization = false;
  return plan;
}

SelectionPlan build_strided_plan(uint64_t src_base, uint64_t row_len, uint64_t stride, uint64_t rows) {
  SelectionPlan plan;
  plan.ranges.reserve(rows);
  for (uint64_t row = 0; row < rows; ++row) {
    plan.ranges.push_back(data_range(src_base + row * stride, row * row_len, row_len));
  }
  plan.total_bytes = rows * row_len;
  plan.num_ranges = static_cast<uint32_t>(plan.ranges.size());
  plan.is_contiguous = false;
  plan.is_segment_aligned = false;
  plan.requires_materialization = false;
  return plan;
}

std::vector<uint8_t> build_strided_expected(
    const std::vector<uint8_t>& backing,
    uint64_t src_base,
    uint64_t row_len,
    uint64_t stride,
    uint64_t rows) {
  std::vector<uint8_t> expected(rows * row_len);
  for (uint64_t row = 0; row < rows; ++row) {
    const uint64_t src_offset = src_base + row * stride;
    const uint64_t dst_offset = row * row_len;
    std::memcpy(expected.data() + dst_offset, backing.data() + src_offset, row_len);
  }
  return expected;
}

} // namespace

TEST_CASE("ViewPlanSource streams ranges with padding", "[view_plan_source]") {
  std::vector<uint8_t> backing(32);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>(i);
  }
  VectorSource base(std::move(backing));
  SelectionPlan plan = build_sample_plan();

  ViewPlanSource src(&base, plan);
  std::array<uint8_t, 16> out{};
  auto bytes_or = src.read(out.data(), out.size());
  REQUIRE(bytes_or.ok());
  CHECK(*bytes_or == 16);

  // Expect [4,5,6,7,16,17,18,19,0,0,0,0,24,25,26,27]
  std::array<uint8_t, 16> expected = {4, 5, 6, 7, 16, 17, 18, 19, 0, 0, 0, 0, 24, 25, 26, 27};
  CHECK(out == expected);
  CHECK_FALSE(src.alias_eligible());
  CHECK_FALSE(src.requires_materialization());
  CHECK(src.total_bytes() == 16);
}

TEST_CASE("ViewPlanSource read_at spans multiple ranges", "[view_plan_source]") {
  std::vector<uint8_t> backing(32);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>(i);
  }
  VectorSource base(std::move(backing));
  SelectionPlan plan = build_sample_plan();
  ViewPlanSource src(&base, plan);

  std::array<uint8_t, 8> out{};
  auto bytes_or = src.read_at(/*offset=*/2, out.data(), out.size());
  REQUIRE(bytes_or.ok());
  CHECK(*bytes_or == 8);

  // Should cover: plan[0] tail (6,7), plan[1] full (16-19), pad two zeros, start of last range (24,25)
  std::array<uint8_t, 8> expected = {6, 7, 16, 17, 18, 19, 0, 0};
  CHECK(out == expected);
}

TEST_CASE("ViewPlanSource reports alias eligibility from plan", "[view_plan_source]") {
  std::vector<uint8_t> data(32, 1);
  VectorSource base(data);

  SelectionPlan contiguous_plan;
  contiguous_plan.ranges.push_back(data_range(/*src=*/8, /*dst=*/0, /*len=*/16));
  contiguous_plan.total_bytes = 16;
  contiguous_plan.is_contiguous = true;
  contiguous_plan.is_segment_aligned = true;
  contiguous_plan.requires_materialization = false;

  ViewPlanSource src(&base, contiguous_plan);
  CHECK(src.alias_eligible());
  CHECK_FALSE(src.requires_materialization());
}

TEST_CASE("ViewPlanSource coalesces strided runs", "[view_plan_source]") {
  constexpr uint64_t kRowLen = 4096;
  constexpr uint64_t kStride = 8192;
  constexpr uint64_t kRows = 128;
  const uint64_t backing_bytes = (kRows - 1) * kStride + kRowLen;
  std::vector<uint8_t> backing(backing_bytes);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>(i % 251);
  }

  VectorSource base(backing);
  SelectionPlan plan = build_strided_plan(/*src_base=*/0, kRowLen, kStride, kRows);
  ViewPlanSource src(&base, plan);

  std::vector<uint8_t> out(plan.total_bytes);
  auto bytes_or = src.read_at(0, out.data(), out.size());
  REQUIRE(bytes_or.ok());
  CHECK(*bytes_or == out.size());

  auto expected = build_strided_expected(backing, /*src_base=*/0, kRowLen, kStride, kRows);
  CHECK(out == expected);
  CHECK(base.read_calls() < kRows);
}

TEST_CASE("ViewPlanSource handles partial strided reads across rows", "[view_plan_source]") {
  constexpr uint64_t kRowLen = 4096;
  constexpr uint64_t kStride = 8192;
  constexpr uint64_t kRows = 128;
  const uint64_t backing_bytes = (kRows - 1) * kStride + kRowLen;
  std::vector<uint8_t> backing(backing_bytes);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>((i * 7) % 255);
  }

  VectorSource base(backing);
  SelectionPlan plan = build_strided_plan(/*src_base=*/0, kRowLen, kStride, kRows);
  ViewPlanSource src(&base, plan);

  auto expected = build_strided_expected(backing, /*src_base=*/0, kRowLen, kStride, kRows);
  const uint64_t offset = 2000;
  const size_t bytes = 12000;
  std::vector<uint8_t> out(bytes);
  auto bytes_or = src.read_at(offset, out.data(), out.size());
  REQUIRE(bytes_or.ok());
  CHECK(*bytes_or == out.size());

  const auto slice_start = static_cast<std::vector<uint8_t>::difference_type>(offset);
  const auto slice_end = static_cast<std::vector<uint8_t>::difference_type>(offset + bytes);
  std::vector<uint8_t> sliced(expected.begin() + slice_start, expected.begin() + slice_end);
  CHECK(out == sliced);
}

TEST_CASE("ViewPlanSource supports concurrent strided read_at", "[view_plan_source]") {
  constexpr uint64_t kRowLen = 4096;
  constexpr uint64_t kStride = 8192;
  constexpr uint64_t kRows = 128;
  const uint64_t backing_bytes = (kRows - 1) * kStride + kRowLen;
  std::vector<uint8_t> backing(backing_bytes);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>(i % 199);
  }

  VectorSource base(backing);
  SelectionPlan plan = build_strided_plan(/*src_base=*/0, kRowLen, kStride, kRows);
  ViewPlanSource src(&base, plan);

  auto expected = build_strided_expected(backing, /*src_base=*/0, kRowLen, kStride, kRows);
  std::vector<uint8_t> out_a(kRowLen * 4);
  std::vector<uint8_t> out_b(kRowLen * 4);
  absl::StatusOr<size_t> st_a;
  absl::StatusOr<size_t> st_b;

  std::thread t1([&]() { st_a = src.read_at(0, out_a.data(), out_a.size()); });
  std::thread t2([&]() { st_b = src.read_at(kRowLen * 10, out_b.data(), out_b.size()); });
  t1.join();
  t2.join();

  REQUIRE(st_a.ok());
  REQUIRE(st_b.ok());
  CHECK(*st_a == out_a.size());
  CHECK(*st_b == out_b.size());
  const auto slice_a_end = static_cast<std::vector<uint8_t>::difference_type>(out_a.size());
  std::vector<uint8_t> sliced_a(expected.begin(), expected.begin() + slice_a_end);
  const auto slice_b_start = static_cast<std::vector<uint8_t>::difference_type>(kRowLen * 10);
  const auto slice_b_end = static_cast<std::vector<uint8_t>::difference_type>(kRowLen * 10 + out_b.size());
  std::vector<uint8_t> sliced_b(expected.begin() + slice_b_start, expected.begin() + slice_b_end);
  CHECK(out_a == sliced_a);
  CHECK(out_b == sliced_b);
}
