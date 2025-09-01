// Copyright (c) 2025, TensorCast Team.

#pragma once

// Lightweight wrappers providing a minimal API over OpenTelemetry Metrics.
// They replace the legacy MetricsRegistry-based helpers and record directly to
// the global OpenTelemetry MeterProvider.

#include <atomic>
#include <map>
#include <string>
#include <unordered_set>
#include <utility>

#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/meter.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::metrics {

// Keep the public Labels alias used across call sites
using Labels = std::vector<std::pair<std::string, std::string>>;

namespace detail {
inline opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> get_meter() {
  return opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
}
} // namespace detail

class Counter {
 public:
  explicit Counter(std::string name, Labels labels = {}) : name_(std::move(name)), labels_(std::move(labels)) {
    auto meter = detail::get_meter();
    counter_ = meter->CreateDoubleCounter(name_.c_str());
  }

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
    if (!counter_)
      return;
    std::map<std::string, opentelemetry::common::AttributeValue> attrs_storage;
    for (const auto& kv : labels_) {
      attrs_storage.emplace(kv.first, opentelemetry::common::AttributeValue(kv.second));
    }
    counter_->Add(
        amount, opentelemetry::common::KeyValueIterableView(attrs_storage), opentelemetry::context::Context{});
  }

  // Convenience alias – mirrors Prometheus client API.
  void add(double amount) const {
    inc(amount);
  }

 private:
  std::string name_;
  Labels labels_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> counter_;
};

class Gauge {
 public:
  explicit Gauge(std::string name, Labels labels = {}) : name_(std::move(name)), labels_(std::move(labels)) {
    auto meter = detail::get_meter();
    updown_counter_ = meter->CreateDoubleUpDownCounter(name_.c_str());
  }

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
    double old = last_value_.load(std::memory_order_relaxed);
    double delta = value - old;
    last_value_.store(value, std::memory_order_relaxed);
    if (updown_counter_) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs_storage;
      for (const auto& kv : labels_) {
        attrs_storage.emplace(kv.first, opentelemetry::common::AttributeValue(kv.second));
      }
      updown_counter_->Add(
          delta, opentelemetry::common::KeyValueIterableView(attrs_storage), opentelemetry::context::Context{});
    }
  }

  // Increment (or decrement when `amount` is negative) the gauge atomically.
  void add(double amount = 1.0) const {
    const double old_val = last_value_.load(std::memory_order_relaxed);
    last_value_.store(old_val + amount, std::memory_order_relaxed);
    if (updown_counter_) {
      std::map<std::string, opentelemetry::common::AttributeValue> attrs_storage;
      for (const auto& kv : labels_) {
        attrs_storage.emplace(kv.first, opentelemetry::common::AttributeValue(kv.second));
      }
      updown_counter_->Add(
          amount, opentelemetry::common::KeyValueIterableView(attrs_storage), opentelemetry::context::Context{});
    }
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
  mutable std::atomic<double> last_value_{0.0};
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::UpDownCounter<double>> updown_counter_;
};

class Histogram {
 public:
  explicit Histogram(std::string name, Labels labels = {}) : name_(std::move(name)), labels_(std::move(labels)) {
    auto meter = detail::get_meter();
    histogram_ = meter->CreateDoubleHistogram(name_.c_str());
  }

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
    if (!histogram_)
      return;
    std::map<std::string, opentelemetry::common::AttributeValue> attrs_storage;
    for (const auto& kv : labels_) {
      attrs_storage.emplace(kv.first, opentelemetry::common::AttributeValue(kv.second));
    }
    histogram_->Record(
        value, opentelemetry::common::KeyValueIterableView(attrs_storage), opentelemetry::context::Context{});
  }

 private:
  std::string name_;
  Labels labels_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> histogram_;
};

} // namespace tensorcast::metrics
