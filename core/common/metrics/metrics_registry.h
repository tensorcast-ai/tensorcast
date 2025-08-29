// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorcast::metrics {

enum class MetricType {
  kCounter,
  kGauge,
  kHistogram,
};

// EDIT 1: introduce Labels alias for key/value label pairs
using Labels = std::vector<std::pair<std::string, std::string>>;

struct Metric {
  MetricType type;
  std::atomic<double> value{0.0};

  // Original metric name without label suffixes (e.g., "request_total")
  std::string name;

  // Optional set of key→value labels attached to this time series.  The
  // registry stores the *canonical* (sorted-by-key) order to guarantee a
  // unique mapping independent of the construction call order.
  Labels labels;

  // Histogram specific data (nullptr for non-histogram metrics)
  struct HistogramData {
    // Upper bounds for each bucket (following Prometheus convention, +Inf is implicit).
    std::vector<double> buckets;
    std::vector<std::unique_ptr<std::atomic<uint64_t>>> counts; // per-bucket event counters
    std::atomic<double> sum{0.0};
    std::atomic<uint64_t> total_count{0};

    explicit HistogramData(const std::vector<double>& bucket_bounds);

    // Custom copy because std::atomic is not copy-assignable.
    HistogramData(const HistogramData& other) noexcept;
    HistogramData& operator=(const HistogramData& other) noexcept;
  };

  std::shared_ptr<HistogramData> hist{nullptr};

  // Explicit constructor to set the metric type and (optionally) an initial value.
  explicit Metric(MetricType t, double initial = 0.0);

  // Custom copy constructor – atomic values cannot be copied directly, so we
  // load the current value from the source atomic and initialise our own
  // atomic with the same numerical value.
  Metric(const Metric& other) noexcept;

  // Custom copy assignment – we cannot copy-assign atomics, therefore we store
  // the loaded value into our existing atomic instead.
  Metric& operator=(const Metric& other) noexcept;

  // Move constructor – we treat it the same as copy because std::atomic is not
  // movable. We simply load the current value from the source.
  Metric(Metric&& other) noexcept;

  // Move assignment – also behaves like copy assignment.
  Metric& operator=(Metric&& other) noexcept;
};

class MetricsRegistry {
 public:
  static MetricsRegistry& instance();

  // Increment a counter by `amount` (defaults to 1). If the counter does not
  // exist it will be created automatically.
  void increment_counter(const std::string& name, const Labels& labels = {}, double amount = 1.0);

  // Backward-compatibility overloads (no labels)
  inline void IncrementCounter(const std::string& name, double amount) {
    increment_counter(name, {}, amount);
  }

  // Set a gauge to a specific value.  Gauges are created on first use.
  void set_gauge(const std::string& name, const Labels& labels, double value);

  inline void SetGauge(const std::string& name, double value) {
    set_gauge(name, {}, value);
  }

  // Increment (or decrement when ``amount`` is negative) the value of a gauge.
  // This is equivalent to ``SetGauge(name, GetGauge(name) + amount)`` but
  // implemented atomically without a read-modify-write race.  The metric is
  // created on first use if it does not yet exist.
  void add_gauge(const std::string& name, const Labels& labels = {}, double amount = 1.0);

  inline void AddGauge(const std::string& name, double amount) {
    add_gauge(name, {}, amount);
  }

  // Observe a value for a histogram metric (creates the histogram on first use).
  void observe_histogram(const std::string& name, const Labels& labels, double value);

  inline void ObserveHistogram(const std::string& name, double value) {
    observe_histogram(name, {}, value);
  }

  // Serialize current values using the Prometheus text exposition format.
  std::string to_open_metrics_text() const;

  MetricsRegistry(const MetricsRegistry&) = delete;
  MetricsRegistry& operator=(const MetricsRegistry&) = delete;

 private:
  MetricsRegistry() = default;

  Metric& get_or_create(const std::string& name, const Labels& labels, MetricType type);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, Metric> metrics_;
};

} // namespace tensorcast::metrics