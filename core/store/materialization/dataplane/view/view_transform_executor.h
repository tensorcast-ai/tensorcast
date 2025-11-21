// Copyright (c) 2025, TensorCast Team.

#pragma once

#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/materialization/dataplane/view/view_planner.h"

namespace tensorcast::store::loader {

/**
 * @brief Execute any required materialization transforms (e.g., transpose) for a view.
 *
 * The transform operates in-place on the provided memory region, reusing the existing allocation.
 *
 * @param plan Transform plan produced by ViewPlanner.
 * @param location Memory location of the backing allocation (CPU or GPU).
 * @param base_ptr Base pointer to the replica byte space.
 * @param device_id CUDA device ordinal when location == GPU; ignored otherwise.
 * @return absl::Status indicating success or failure.
 */
absl::Status execute_transform(
    const TransformPlan& plan,
    common::memory::MemoryLocation location,
    void* base_ptr,
    int device_id);

} // namespace tensorcast::store::loader
