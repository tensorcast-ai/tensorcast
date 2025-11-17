// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_plan_source.h"

#include <array>
#include <cstring>
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
    return available;
  }

 private:
  std::vector<uint8_t> data_;
  uint64_t cursor_{0};
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
