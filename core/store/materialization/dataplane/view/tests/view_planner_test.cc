// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_planner.h"

#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "nlohmann/json.hpp"

using tensorcast::store::loader::NarrowOp;
using tensorcast::store::loader::SelectionPlan;
using tensorcast::store::loader::TensorViewOps;
using tensorcast::store::loader::ViewOp;
using tensorcast::store::loader::ViewPlanner;
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

} // namespace

TEST_CASE("ViewPlanner returns identity plan when spec is empty", "[view_planner]") {
  nlohmann::json index = nlohmann::json::object();
  index["weights"] = tensor_entry(
      /*offset=*/0,
      /*size=*/32,
      /*shape=*/{8},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewPlanner planner;
  ViewSpec spec;

  auto plan_or = planner.compute_view_plan(canonical, spec);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  REQUIRE(plan.is_identity);
  CHECK(plan.selection.num_ranges >= 1);
  CHECK(plan.selection.total_bytes == 32);
  CHECK(plan.view_size_bytes == 32);
  CHECK(plan.selection.is_contiguous);
  CHECK_FALSE(plan.selection.requires_materialization);
  REQUIRE_FALSE(plan.selection.ranges.empty());
  const auto& range = plan.selection.ranges.front();
  CHECK(range.kind == SelectionPlan::Range::Kind::kData);
  CHECK(range.src_offset == 0);
  CHECK(range.dst_offset == 0);
  CHECK(range.length == 32);

  auto rebuilt_or = tensorcast::store::loader::rebuild_stable_canonical_index(canonical, /*default_device_id=*/0);
  REQUIRE(rebuilt_or.ok());
  CHECK(plan.view_index_json == *rebuilt_or);
}

TEST_CASE("ViewPlanner packs subset names into view index", "[view_planner]") {
  nlohmann::json index = nlohmann::json::object();
  index["alpha"] = tensor_entry(
      /*offset=*/0,
      /*size=*/16,
      /*shape=*/{4},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  index["beta"] = tensor_entry(
      /*offset=*/16,
      /*size=*/16,
      /*shape=*/{4},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/4);
  const std::string canonical = index.dump();

  ViewSpec spec;
  const std::vector<std::string> subset{"beta"};

  auto plan_or = ViewPlanner::compute_view_plan(canonical, spec, subset);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK_FALSE(plan.is_identity);
  REQUIRE(plan.selection.ranges.size() == 1);
  const auto& range = plan.selection.ranges.front();
  CHECK(range.kind == SelectionPlan::Range::Kind::kData);
  CHECK(range.src_offset == 16);
  CHECK(range.dst_offset == 0);
  CHECK(range.length == 16);
  CHECK(plan.view_size_bytes == 16);

  const auto view_json = nlohmann::json::parse(plan.view_index_json);
  REQUIRE(view_json.size() == 1);
  const auto& entry = view_json.at("beta");
  CHECK(entry[0].get<uint64_t>() == 0);
  CHECK(entry[1].get<uint64_t>() == 16);
}

TEST_CASE("ViewPlanner keeps identity for full-name subset", "[view_planner]") {
  nlohmann::json index = nlohmann::json::object();
  index["alpha"] = tensor_entry(
      /*offset=*/0,
      /*size=*/16,
      /*shape=*/{4},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  index["beta"] = tensor_entry(
      /*offset=*/16,
      /*size=*/16,
      /*shape=*/{4},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/4);
  const std::string canonical = index.dump();

  ViewSpec spec;
  const std::vector<std::string> subset{"alpha", "beta"};

  auto plan_or = ViewPlanner::compute_view_plan(canonical, spec, subset);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;
  CHECK_FALSE(plan.is_identity);
  CHECK(plan.view_index_json == tensorcast::store::loader::rebuild_stable_canonical_index(canonical, 0).value());
}

TEST_CASE("ViewPlanner preserves subset ordering", "[view_planner]") {
  nlohmann::json index = nlohmann::json::object();
  index["alpha"] = tensor_entry(
      /*offset=*/0,
      /*size=*/16,
      /*shape=*/{4},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  index["beta"] = tensor_entry(
      /*offset=*/16,
      /*size=*/16,
      /*shape=*/{4},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewSpec spec;
  const std::vector<std::string> subset{"beta", "alpha"};

  auto plan_or = ViewPlanner::compute_view_plan(canonical, spec, subset);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK_FALSE(plan.is_identity);
  const auto view_json = nlohmann::json::parse(plan.view_index_json);
  REQUIRE(view_json.size() == 2);
  CHECK(view_json.at("beta")[0].get<uint64_t>() == 0);
  CHECK(view_json.at("alpha")[0].get<uint64_t>() == 16);
}

TEST_CASE("ViewPlanner emits contiguous aligned range for 1D narrow", "[view_planner]") {
  nlohmann::json index = nlohmann::json::object();
  index["weights"] = tensor_entry(
      /*offset=*/0,
      /*size=*/32,
      /*shape=*/{8},
      /*stride=*/{1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewPlanner planner;
  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 0, .start = 2, .length = 4}));
  spec.tensors.emplace("weights", ops);

  auto plan_or = planner.compute_view_plan(canonical, spec);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK_FALSE(plan.is_identity);
  CHECK(plan.selection.num_ranges == plan.selection.ranges.size());
  REQUIRE(plan.selection.ranges.size() == 1);
  const auto& range = plan.selection.ranges.front();
  CHECK(range.kind == SelectionPlan::Range::Kind::kData);
  CHECK(range.src_offset == 8);
  CHECK(range.dst_offset == 0);
  CHECK(range.length == 16);
  CHECK(plan.selection.is_contiguous);
  CHECK(plan.selection.is_segment_aligned);
  CHECK_FALSE(plan.selection.requires_materialization);
  CHECK(plan.selection.total_bytes == 16);
  CHECK(plan.view_size_bytes == 16);

  const auto view_json = nlohmann::json::parse(plan.view_index_json);
  const auto& entry = view_json.at("weights");
  CHECK(entry[0].get<uint64_t>() == 0);
  CHECK(entry[1].get<uint64_t>() == 16);
  const auto shape = entry[2].get<std::vector<int64_t>>();
  REQUIRE(shape.size() == 1);
  CHECK(shape[0] == 4);
  const auto stride = entry[3].get<std::vector<int64_t>>();
  REQUIRE(stride.size() == 1);
  CHECK(stride[0] == 1);
  CHECK(entry[5].get<uint64_t>() == 0);
}

TEST_CASE("ViewPlanner emits scatter plan for inner-dimension narrow", "[view_planner]") {
  nlohmann::json index = nlohmann::json::object();
  index["activation"] = tensor_entry(
      /*offset=*/0,
      /*size=*/32,
      /*shape=*/{2, 4},
      /*stride=*/{4, 1},
      /*dtype=*/"torch.float32",
      /*storage_offset=*/0);
  const std::string canonical = index.dump();

  ViewPlanner planner;
  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(
      ViewOp::Narrow(
          NarrowOp{
              .dim = 1,
              .start = -3, // normalized to 1
              .length = 2}));
  spec.tensors.emplace("activation", ops);

  auto plan_or = planner.compute_view_plan(canonical, spec);
  REQUIRE(plan_or.ok());
  const auto& plan = *plan_or;

  CHECK_FALSE(plan.is_identity);
  CHECK(plan.selection.ranges.size() == 2);
  CHECK(plan.selection.num_ranges == 2);
  CHECK_FALSE(plan.selection.is_contiguous);
  CHECK_FALSE(plan.selection.is_segment_aligned);
  CHECK_FALSE(plan.selection.requires_materialization);
  CHECK(plan.selection.total_bytes == 16);
  CHECK(plan.view_size_bytes == 16);

  const auto& first = plan.selection.ranges[0];
  const auto& second = plan.selection.ranges[1];
  CHECK(first.kind == SelectionPlan::Range::Kind::kData);
  CHECK(second.kind == SelectionPlan::Range::Kind::kData);
  CHECK(first.dst_offset == 0);
  CHECK(second.dst_offset == 8);
  CHECK(first.length == 8);
  CHECK(second.length == 8);
  CHECK(first.src_offset == 4);
  CHECK(second.src_offset == 20);

  const auto view_json = nlohmann::json::parse(plan.view_index_json);
  const auto& entry = view_json.at("activation");
  CHECK(entry[0].get<uint64_t>() == 0);
  CHECK(entry[1].get<uint64_t>() == 16);
  const auto shape = entry[2].get<std::vector<int64_t>>();
  REQUIRE(shape.size() == 2);
  CHECK(shape[0] == 2);
  CHECK(shape[1] == 2);
  const auto stride = entry[3].get<std::vector<int64_t>>();
  REQUIRE(stride.size() == 2);
  CHECK(stride[0] == 2);
  CHECK(stride[1] == 1);
  CHECK(entry[5].get<uint64_t>() == 0);
}
