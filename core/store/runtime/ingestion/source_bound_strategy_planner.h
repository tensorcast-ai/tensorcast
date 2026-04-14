// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <optional>

#include "absl/status/statusor.h"
#include "core/store/runtime/ingestion/materialization_strategy_types.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime::ingestion::strategy {

absl::StatusOr<SourceBoundStrategyPlan> build_source_bound_execution_strategy_plan(
    const ResolvedMaterializationPlan& resolved_plan,
    const std::optional<SourceBoundLoweringArtifacts>& lowering_artifacts,
    SourceBoundPolicy policy,
    const StoreEngineOptions::MaterializationStrategyConfig& strategy_config,
    const loading::ExecutionTopologyContext& execution_topology,
    bool disk_source_available);

} // namespace tensorcast::store::runtime::ingestion::strategy
