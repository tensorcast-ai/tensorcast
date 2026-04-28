// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include "core/store/materialization/dataplane/runtime/pump.h"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::runtime::ingestion {

loader::PumpDirectWriteOptions make_pump_direct_write_options(
    const StoreEngineOptions::MaterializationStrategyConfig& strategy_config);

} // namespace tensorcast::store::runtime::ingestion
