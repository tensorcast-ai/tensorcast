// Copyright (c) 2025, TensorCast Team.

#include "core/store/store_engine.h"

#include <cstring>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/view/view_planner.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

using tensorcast::store::StoreEngine;
using tensorcast::store::loader::SelectionPlan;
using tensorcast::store::loader::TensorViewOps;
using tensorcast::store::loader::ViewOp;
using tensorcast::store::loader::ViewPlan;
using tensorcast::store::loader::ViewSpec;

namespace {

nlohmann::json tensor_entry(
    uint64_t offset,
    uint64_t size,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    const std::string& dtype,
    uint64_t storage_offset) {
  nlohmann::json arr = nlohmann::json::array();
  arr.push_back(offset);
  arr.push_back(size);
  nlohmann::json j_shape = nlohmann::json::array();
  for (auto v : shape) {
    j_shape.push_back(v);
  }
  nlohmann::json j_stride = nlohmann::json::array();
  for (auto v : stride) {
    j_stride.push_back(v);
  }
  arr.push_back(j_shape);
  arr.push_back(j_stride);
  arr.push_back(dtype);
  arr.push_back(storage_offset);
  return arr;
}

class VectorSource final : public tensorcast::store::loader::SeekableSource {
 public:
  explicit VectorSource(const std::vector<uint8_t>& data) : data_(data) {}

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
  const std::vector<uint8_t>& data_;
  uint64_t cursor_{0};
};

std::string canonical_index_for_tensor(
    const std::string& name,
    uint64_t offset,
    uint64_t size,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    const std::string& dtype) {
  nlohmann::json index = nlohmann::json::object();
  index[name] = tensor_entry(offset, size, shape, stride, dtype, /*storage_offset=*/0);
  return index.dump();
}

} // namespace

TEST_CASE("StoreEngine computes view hash for contiguous narrow", "[store_engine][view_plan]") {
  const std::string canonical_index =
      canonical_index_for_tensor("weights", /*offset=*/0, /*size=*/32, {8}, {1}, "torch.float32");

  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Narrow(tensorcast::store::loader::NarrowOp{.dim = 0, .start = 2, .length = 4}));
  spec.tensors.emplace("weights", ops);

  auto plan_or = StoreEngine::compute_view_plan(canonical_index, spec);
  REQUIRE(plan_or.ok());
  const ViewPlan& plan = *plan_or;
  CHECK_FALSE(plan.is_identity);
  CHECK(StoreEngine::view_plan_allows_alias(plan));

  std::vector<uint8_t> backing(32);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>(i);
  }
  VectorSource base(backing);
  auto hash_or = StoreEngine::compute_view_data_hash_from_source(base, plan);
  REQUIRE(hash_or.ok());

  std::vector<uint8_t> expected(backing.begin() + 8, backing.begin() + 24);
  auto expected_or = tensorcast::store::loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{expected.data()}, expected.size());
  REQUIRE(expected_or.ok());
  CHECK(*hash_or == *expected_or);
}

TEST_CASE("StoreEngine view hash for scatter narrow matches CPU materialization", "[store_engine][view_plan]") {
  const std::string canonical_index =
      canonical_index_for_tensor("tensor", /*offset=*/0, /*size=*/32, {2, 4}, {4, 1}, "torch.float32");

  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Narrow(tensorcast::store::loader::NarrowOp{.dim = 1, .start = 1, .length = 2}));
  spec.tensors.emplace("tensor", ops);

  auto plan_or = StoreEngine::compute_view_plan(canonical_index, spec);
  REQUIRE(plan_or.ok());
  const ViewPlan& plan = *plan_or;
  CHECK_FALSE(StoreEngine::view_plan_allows_alias(plan));

  std::vector<uint8_t> backing(32);
  for (size_t i = 0; i < backing.size(); ++i) {
    backing[i] = static_cast<uint8_t>(i);
  }
  VectorSource base(backing);
  auto hash_or = StoreEngine::compute_view_data_hash_from_source(base, plan);
  REQUIRE(hash_or.ok());

  const size_t element_size = 4;
  const size_t rows = 2;
  const size_t cols = 4;
  const size_t start = 1;
  const size_t length = 2;
  std::vector<uint8_t> expected;
  expected.reserve(rows * length * element_size);
  for (size_t row = 0; row < rows; ++row) {
    const size_t row_start = row * cols * element_size;
    const size_t slice_start = row_start + start * element_size;
    expected.insert(
        expected.end(),
        backing.begin() + static_cast<std::ptrdiff_t>(slice_start),
        backing.begin() + static_cast<std::ptrdiff_t>(slice_start + length * element_size));
  }

  auto expected_or = tensorcast::store::loader::compute_data_multihash_from_cpu_memory(
      gsl::not_null<const void*>{expected.data()}, expected.size());
  REQUIRE(expected_or.ok());
  CHECK(*hash_or == *expected_or);
}
