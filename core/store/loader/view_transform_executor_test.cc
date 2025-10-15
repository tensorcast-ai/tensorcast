// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/view_transform_executor.h"

#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/common/memory/memory_location.h"
#include "core/store/loader/view_planner.h"
#include "nlohmann/json.hpp"

using tensorcast::common::memory::MemoryLocation;
using tensorcast::store::loader::TensorViewOps;
using tensorcast::store::loader::ViewOp;
using tensorcast::store::loader::ViewPlan;
using tensorcast::store::loader::ViewPlanner;
using tensorcast::store::loader::ViewSpec;

namespace {

std::string CanonicalIndexForTensor(
    const std::string& name,
    uint64_t offset,
    uint64_t size,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& stride,
    const std::string& dtype) {
  nlohmann::json index = nlohmann::json::object();
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
  arr.push_back(0); // storage_offset
  index[name] = std::move(arr);
  return index.dump();
}

} // namespace

TEST_CASE("View transform executor performs in-place transpose on CPU", "[view_transform_executor][cpu]") {
  const std::string canonical_index =
      CanonicalIndexForTensor("tensor", /*offset=*/0, /*size=*/24, {2, 3}, {3, 1}, "torch.float32");

  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Transpose(tensorcast::store::loader::TransposeOp{.dim0 = 0, .dim1 = 1}));
  spec.tensors.emplace("tensor", ops);

  auto plan_or = ViewPlanner::compute_view_plan(canonical_index, spec);
  REQUIRE(plan_or.ok());
  const ViewPlan& plan = *plan_or;
  REQUIRE_FALSE(plan.is_identity);
  REQUIRE(plan.transform.requires_materialization);
  REQUIRE_FALSE(plan.transform.tensors.empty());

  std::vector<float> buffer(6);
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = static_cast<float>(i);
  }

  auto status = tensorcast::store::loader::execute_transform(
      plan.transform,
      MemoryLocation::CPU,
      static_cast<void*>(buffer.data()),
      /*device_id=*/-1);
  REQUIRE(status.ok());

  const std::vector<float> expected{0.0F, 3.0F, 1.0F, 4.0F, 2.0F, 5.0F};
  REQUIRE(buffer == expected);
}
