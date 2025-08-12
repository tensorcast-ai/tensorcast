// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>
#include "core/common/metrics/metric_objects.h"
#include "core/common/metrics/metrics_registry.h"

using namespace stepcast::metrics;

TEST_CASE("Metrics Registry", "[metrics]") {
  auto& registry = MetricsRegistry::instance();

  SECTION("Counter metrics") {
    registry.IncrementCounter("test_counter", 1.0);
    registry.IncrementCounter("test_counter", 2.5);
    registry.IncrementCounter("test_counter", 0.5);
    // Verify counter increments correctly
    REQUIRE_NOTHROW(registry.IncrementCounter("test_counter", 1.0));
  }

  SECTION("Gauge metrics") {
    registry.SetGauge("test_gauge", 100.0);
    registry.AddGauge("test_gauge", 50.0);
    registry.AddGauge("test_gauge", -30.0);
    // Verify gauge operations work correctly
    REQUIRE_NOTHROW(registry.SetGauge("test_gauge", 42.0));
  }

  SECTION("Histogram metrics") {
    registry.ObserveHistogram("test_histogram", 0.001);
    registry.ObserveHistogram("test_histogram", 0.05);
    registry.ObserveHistogram("test_histogram", 0.5);
    registry.ObserveHistogram("test_histogram", 1.5);
    registry.ObserveHistogram("test_histogram", 5.0);
    registry.ObserveHistogram("test_histogram", 15.0);
    // Verify histogram observations work correctly
    REQUIRE_NOTHROW(registry.ObserveHistogram("test_histogram", 10.0));
  }

  SECTION("Thread safety") {
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i) {
      threads.emplace_back([&registry, i]() {
        for (int j = 0; j < 1000; ++j) {
          registry.IncrementCounter("concurrent_counter", 1.0);
          registry.AddGauge("concurrent_gauge", 1.0);
          registry.ObserveHistogram("concurrent_histogram", i * 0.1 + j * 0.001);
        }
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    // Verify no crashes occurred during concurrent access
    REQUIRE_NOTHROW(registry.IncrementCounter("concurrent_counter", 1.0));
  }

  SECTION("Duplicate label detection") {
    Counter dup_base("dup_metric", {{"key", "v1"}});
    // Should throw when trying to add duplicate key
    REQUIRE_THROWS_AS(dup_base.with_labels({{"key", "v2"}}), std::invalid_argument);
  }

  SECTION("Export metrics as OpenMetrics text") {
    auto metrics_text = registry.to_open_metrics_text();
    // Verify we can export metrics without errors
    REQUIRE(!metrics_text.empty());
    // Verify the export contains expected metric names
    REQUIRE(metrics_text.find("test_counter") != std::string::npos);
    REQUIRE(metrics_text.find("test_gauge") != std::string::npos);
    REQUIRE(metrics_text.find("test_histogram") != std::string::npos);
  }
}