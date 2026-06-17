// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/runtime/ingestion/materialization_pump_options.h"

#include <algorithm>
#include <limits>

namespace tensorcast::store::runtime::ingestion {

namespace {

size_t clamp_batch_bytes(uint64_t batch_bytes) {
  return static_cast<size_t>(
      std::min<uint64_t>(batch_bytes, static_cast<uint64_t>(std::numeric_limits<size_t>::max())));
}

} // namespace

loader::PumpDirectWriteOptions make_pump_direct_write_options(
    const StoreEngineOptions::MaterializationStrategyConfig& strategy_config) {
  return loader::PumpDirectWriteOptions{
      .direct_write_batch_bytes = clamp_batch_bytes(strategy_config.direct_write_batch_bytes),
      .direct_write_batch_ops = static_cast<size_t>(strategy_config.direct_write_batch_ops),
  };
}

} // namespace tensorcast::store::runtime::ingestion
