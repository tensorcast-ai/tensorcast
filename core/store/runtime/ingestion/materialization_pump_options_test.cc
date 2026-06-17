// Copyright (c) 2025-2026, TensorCast Team.

#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "core/store/runtime/ingestion/materialization_pump_options.h"

namespace {

using tensorcast::store::StoreEngineOptions;
using tensorcast::store::runtime::ingestion::make_pump_direct_write_options;

TEST_CASE("materialization pump options keep zero defaults", "[materialization_pump_options]") {
  StoreEngineOptions::MaterializationStrategyConfig config;

  const auto options = make_pump_direct_write_options(config);

  REQUIRE(options.direct_write_batch_bytes == 0);
  REQUIRE(options.direct_write_batch_ops == 0);
}

TEST_CASE("materialization pump options preserve explicit overrides", "[materialization_pump_options]") {
  StoreEngineOptions::MaterializationStrategyConfig config;
  config.direct_write_batch_bytes = 128ULL * 1024ULL * 1024ULL;
  config.direct_write_batch_ops = 32;

  const auto options = make_pump_direct_write_options(config);

  REQUIRE(options.direct_write_batch_bytes == 128ULL * 1024ULL * 1024ULL);
  REQUIRE(options.direct_write_batch_ops == 32);
}

TEST_CASE("materialization pump options saturate batch bytes to size_t", "[materialization_pump_options]") {
  StoreEngineOptions::MaterializationStrategyConfig config;
  config.direct_write_batch_bytes = std::numeric_limits<uint64_t>::max();

  const auto options = make_pump_direct_write_options(config);

  REQUIRE(options.direct_write_batch_bytes == std::numeric_limits<size_t>::max());
}

} // namespace
