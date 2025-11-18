// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "core/store/materialization/contracts/view/view_plan.h"
#include "core/store/materialization/contracts/view/view_spec.h"

namespace tensorcast::store::loader {

using tensorcast::store::materialization::view::BidirectionalViewPlan;
using tensorcast::store::materialization::view::NarrowOp;
using tensorcast::store::materialization::view::SelectionPlan;
using tensorcast::store::materialization::view::TensorTransformPlan;
using tensorcast::store::materialization::view::TensorViewOps;
using tensorcast::store::materialization::view::TransformPlan;
using tensorcast::store::materialization::view::TransposeOp;
using tensorcast::store::materialization::view::ViewOp;
using tensorcast::store::materialization::view::ViewPlan;
using tensorcast::store::materialization::view::ViewSpec;
using tensorcast::store::materialization::view::ViewWritePlan;

class ViewPlanner {
 public:
  [[nodiscard]] static absl::StatusOr<ViewPlan> compute_view_plan(
      std::string_view canonical_index_json,
      const ViewSpec& spec);

  [[nodiscard]] static absl::StatusOr<BidirectionalViewPlan> compute_bidirectional_view_plan(
      std::string_view canonical_index_json,
      const ViewSpec& spec);
};

} // namespace tensorcast::store::loader
