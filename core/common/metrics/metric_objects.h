// Copyright (c) 2025, TensorCast Team.

#pragma once

// Lightweight value objects providing a **minimal** Prometheus-style API on
// top of the global `MetricsRegistry` singleton.  The wrappers trade a few
// extra CPU cycles (due to the registry lookup performed on every update)
// for maximum convenience and are therefore intended for *infrequent* metric
// updates outside tight inner loops.
//
// Example:
//   #include "core/common/metrics/metric_objects.h"
//
//   tensorcast::metrics::Counter requests_total("my_requests_total");
//   requests_total.Inc();
//
//   tensorcast::metrics::Gauge pending("my_queue_pending");
//   pending.Set(42);
//   pending.Dec();
//
// These helpers are **header-only** and introduce no additional link-time
// dependency – they simply forward to the underlying `MetricsRegistry`.

#include <string>
#include <unordered_set>
#include <utility>

#include "core/common/metrics/metrics_registry.h"

namespace tensorcast::metrics {

class Counter {
 public:
  explicit Counter(std::string name, Labels labels = {}) : name_(std::move(name)), labels_(std::move(labels)) {}

  // ------------------------------------------------------------------
  // Helper to create a *child* metric instance that carries an additional
  // label set.  This mirrors the Prometheus client `.labels()` pattern and
  // allows on-the-fly construction without keeping separate wrapper
  // objects around.
  //
  // Example:
  //   Counter req_total("http_requests_total");
  //   req_total.with_labels({{"method", "POST"}, {"code", "200"}}).inc();
  //
  // The returned Counter is a lightweight value object – copying is cheap
  // because it only stores the metric *name* plus the (canonicalised by the
  // registry) label vector.  There is **no** extra registry entry unless the
  // resulting label combination has not been observed before.
  [[nodiscard]] Counter with_labels(const Labels& extra_labels) const {
    Labels merged = labels_;
    std::unordered_set<std::string> seen_keys;
    for (const auto& kv : labels_) {
      seen_keys.insert(kv.first);
    }
    for (const auto& kv : extra_labels) {
      if (!seen_keys.insert(kv.first).second) {
        throw std::invalid_argument("duplicate label key '" + kv.first + "' passed to with_labels()");
      }
      merged.emplace_back(kv);
    }
    return Counter{name_, std::move(merged)};
  }

  // Increment the counter by `amount` (defaults to 1.0).
  void inc(double amount = 1.0) const {
    MetricsRegistry::instance().increment_counter(name_, labels_, amount);
  }

  // Convenience alias – mirrors Prometheus client API.
  void add(double amount) const {
    inc(amount);
  }

 private:
  std::string name_;
  Labels labels_;
};

class Gauge {
 public:
  explicit Gauge(std::string name, Labels labels = {}) : name_(std::move(name)), labels_(std::move(labels)) {}

  // Create a derived Gauge with extra labels (non-destructive).
  [[nodiscard]] Gauge with_labels(const Labels& extra_labels) const {
    Labels merged = labels_;
    std::unordered_set<std::string> seen_keys;
    for (const auto& kv : labels_) {
      seen_keys.insert(kv.first);
    }
    for (const auto& kv : extra_labels) {
      if (!seen_keys.insert(kv.first).second) {
        throw std::invalid_argument("duplicate label key '" + kv.first + "' passed to with_labels()");
      }
      merged.emplace_back(kv);
    }
    return Gauge{name_, std::move(merged)};
  }

  // Set the gauge to an explicit value.
  void set(double value) const {
    MetricsRegistry::instance().set_gauge(name_, labels_, value);
  }

  // Increment (or decrement when `amount` is negative) the gauge atomically.
  void add(double amount = 1.0) const {
    MetricsRegistry::instance().add_gauge(name_, labels_, amount);
  }

  // Increment by +1.
  void inc() const {
    add(1.0);
  }

  // Decrement by -1.
  void dec() const {
    add(-1.0);
  }

 private:
  std::string name_;
  Labels labels_;
};

class Histogram {
 public:
  explicit Histogram(std::string name, Labels labels = {}) : name_(std::move(name)), labels_(std::move(labels)) {}

  // Return a Histogram observing the same metric *name* with an additional
  // set of labels merged onto the base labels.
  [[nodiscard]] Histogram with_labels(const Labels& extra_labels) const {
    Labels merged = labels_;
    std::unordered_set<std::string> seen_keys;
    for (const auto& kv : labels_) {
      seen_keys.insert(kv.first);
    }
    for (const auto& kv : extra_labels) {
      if (!seen_keys.insert(kv.first).second) {
        throw std::invalid_argument("duplicate label key '" + kv.first + "' passed to with_labels()");
      }
      merged.emplace_back(kv);
    }
    return Histogram{name_, std::move(merged)};
  }

  // Record an observation.
  void observe(double value) const {
    MetricsRegistry::instance().observe_histogram(name_, labels_, value);
  }

 private:
  std::string name_;
  Labels labels_;
};

} // namespace tensorcast::metrics