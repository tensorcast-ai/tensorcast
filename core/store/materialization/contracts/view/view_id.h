// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <optional>
#include <string>

#include "core/store/materialization/contracts/view/view_plan.h"
#include "core/store/materialization/contracts/view/view_spec.h"

namespace tensorcast::store::materialization::view {

enum class TransformPlacement : uint8_t { kServer = 0, kClient = 1 };

struct VariantIdentity {
  std::string canonical_artifact_id;
  std::optional<std::string> view_id;
  std::optional<ViewSpec> view_spec;
  TransformPlacement placement{TransformPlacement::kServer};
  std::optional<std::string> canonical_index_json;
  std::optional<ViewPlan> cached_plan;
};

} // namespace tensorcast::store::materialization::view
