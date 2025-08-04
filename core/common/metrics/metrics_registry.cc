// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/common/metrics/metrics_registry.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <tuple>

namespace {
// Default bucket boundaries following Prometheus client libraries (seconds/order magnitudes).
const std::vector<double> kDefaultBuckets = {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0};

// Convert a vector of label pairs into a canonical, sorted string representation
// of the form `key1="val1",key2="val2"`.  Sorting ensures that the
// representation is *unique* for a given set irrespective of input order so we
// can safely use it as part of the unordered_map key.
std::string LabelsToString(stepcast::metrics::Labels labels) {
  if (labels.empty()) {
    return "";
  }
  // Sort by key for determinism.
  std::sort(labels.begin(), labels.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::ostringstream oss;
  for (size_t i = 0; i < labels.size(); ++i) {
    oss << labels[i].first << "=\"" << labels[i].second << "\"";
    if (i + 1 < labels.size()) {
      oss << ",";
    }
  }
  return oss.str();
}

// Compose the map key: `<name>{sorted_labels}`.  For metrics without labels the
// braces are omitted so the key degenerates to the metric name – preserving
// backward-compatible behaviour.
std::string compose_key(const std::string& name, const stepcast::metrics::Labels& labels) {
  std::string lbl_str = LabelsToString(labels);
  if (lbl_str.empty()) {
    return name;
  }
  return name + "{" + lbl_str + "}"; // braces become part of key but not stored in Metric.name
}
} // namespace

namespace stepcast::metrics {

MetricsRegistry& MetricsRegistry::instance() {
  static MetricsRegistry instance;
  return instance;
}

Metric& MetricsRegistry::get_or_create(const std::string& name, const Labels& labels, MetricType type) {
  const std::string composite_key = compose_key(name, labels);
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = metrics_.find(composite_key);
  if (it == metrics_.end()) {
    Metric metric{type};
    metric.name = name;
    metric.labels = labels; // unsorted order preservation not required beyond serialisation
    it = metrics_.try_emplace(composite_key, std::move(metric)).first;
  }
  return it->second;
}

void MetricsRegistry::increment_counter(const std::string& name, const Labels& labels, double amount) {
  Metric& m = get_or_create(name, labels, MetricType::kCounter);
  double old_val = m.value.load(std::memory_order_relaxed);
  double new_val;
  do {
    new_val = old_val + amount;
  } while (!m.value.compare_exchange_weak(old_val, new_val, std::memory_order_relaxed));
}

void MetricsRegistry::set_gauge(const std::string& name, const Labels& labels, double value) {
  Metric& m = get_or_create(name, labels, MetricType::kGauge);
  m.value.store(value, std::memory_order_relaxed);
}

void MetricsRegistry::add_gauge(const std::string& name, const Labels& labels, double amount) {
  Metric& m = get_or_create(name, labels, MetricType::kGauge);
  double old_val = m.value.load(std::memory_order_relaxed);
  double new_val;
  do {
    new_val = old_val + amount;
  } while (!m.value.compare_exchange_weak(old_val, new_val, std::memory_order_relaxed));
}

void MetricsRegistry::observe_histogram(const std::string& name, const Labels& labels, double value) {
  Metric& m = get_or_create(name, labels, MetricType::kHistogram);
  auto& h = m.hist;
  if (!h) {
    // Should not happen, but safeguard.
    return;
  }

  // Update sum and count using compare_exchange loops for atomic<double>
  double old_sum = h->sum.load(std::memory_order_relaxed);
  double new_sum;
  do {
    new_sum = old_sum + value;
  } while (!h->sum.compare_exchange_weak(old_sum, new_sum, std::memory_order_relaxed));

  h->total_count.fetch_add(1, std::memory_order_relaxed);

  // Find bucket.
  for (size_t i = 0; i < h->buckets.size(); ++i) {
    if (value <= h->buckets[i]) {
      h->counts[i]->fetch_add(1, std::memory_order_relaxed);
      break;
    }
  }
}

std::string MetricsRegistry::to_open_metrics_text() const {
  std::ostringstream ss;
  std::lock_guard<std::mutex> guard(mutex_);
  for (const auto& [key, metric] : metrics_) {
    const std::string& name = metric.name; // base name without labels

    // Prepare common label string (may be empty)
    auto lbl_str = LabelsToString(metric.labels);
    std::string qualifier = lbl_str.empty() ? "" : ("{" + lbl_str + "}");
    std::string full_name_with_labels = name + qualifier;

    switch (metric.type) {
      case MetricType::kGauge:
        ss << "# TYPE " << name << " gauge\n";
        ss << std::setprecision(15) << std::fixed << full_name_with_labels << " "
           << metric.value.load(std::memory_order_relaxed) << "\n";
        break;
      case MetricType::kCounter:
        ss << "# TYPE " << name << " counter\n";
        ss << std::setprecision(15) << std::fixed << full_name_with_labels << " "
           << metric.value.load(std::memory_order_relaxed) << "\n";
        break;
      case MetricType::kHistogram: {
        ss << "# TYPE " << name << " histogram\n";
        const auto& h = metric.hist;
        if (!h) {
          break;
        }

        uint64_t cumulative = 0;
        for (size_t i = 0; i < h->buckets.size(); ++i) {
          cumulative += h->counts[i]->load(std::memory_order_relaxed);
          // Append le label last to keep deterministic ordering (all user
          // labels first, then `le`).
          std::string bucket_labels = lbl_str;
          if (!bucket_labels.empty()) {
            bucket_labels += ",";
          }
          bucket_labels += "le=\"" + std::to_string(h->buckets[i]) + "\"";
          ss << name << "_bucket{" << bucket_labels << "} " << cumulative << "\n";
        }
        // +Inf bucket
        {
          std::string bucket_labels = lbl_str;
          if (!bucket_labels.empty()) {
            bucket_labels += ",";
          }
          bucket_labels += "le=\"+Inf\"";
          ss << name << "_bucket{" << bucket_labels << "} " << h->total_count.load(std::memory_order_relaxed) << "\n";
        }

        // sum & count
        ss << name << "_sum" << qualifier << " " << h->sum.load(std::memory_order_relaxed) << "\n";
        ss << name << "_count" << qualifier << " " << h->total_count.load(std::memory_order_relaxed) << "\n";
        break;
      }
    }
  }
  return ss.str();
}

// ---------------------------------------------------------------------------
// HistogramData implementation
// ---------------------------------------------------------------------------

Metric::HistogramData::HistogramData(const std::vector<double>& bucket_bounds) {
  buckets = bucket_bounds;
  // Create atomic counters using unique_ptr
  counts.reserve(buckets.size());
  for (size_t i = 0; i < buckets.size(); ++i) {
    counts.emplace_back(std::make_unique<std::atomic<uint64_t>>(0));
  }
}

Metric::HistogramData::HistogramData(const Metric::HistogramData& other) noexcept {
  buckets = other.buckets;
  // Create new atomic counters with copied values
  counts.reserve(buckets.size());
  for (size_t i = 0; i < other.counts.size(); ++i) {
    counts.emplace_back(std::make_unique<std::atomic<uint64_t>>(other.counts[i]->load(std::memory_order_relaxed)));
  }
  sum.store(other.sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
  total_count.store(other.total_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

Metric::HistogramData& Metric::HistogramData::operator=(const Metric::HistogramData& other) noexcept {
  if (this == &other) {
    return *this;
  }
  buckets = other.buckets;
  counts.clear();
  // Create new atomic counters with copied values
  counts.reserve(buckets.size());
  for (size_t i = 0; i < other.counts.size(); ++i) {
    counts.emplace_back(std::make_unique<std::atomic<uint64_t>>(other.counts[i]->load(std::memory_order_relaxed)));
  }
  sum.store(other.sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
  total_count.store(other.total_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
  return *this;
}

// ---------------------------------------------------------------------------
// Metric constructors / assignments with histogram support
// ---------------------------------------------------------------------------

Metric::Metric(MetricType t, double initial) : type(t) {
  if (t == MetricType::kHistogram) {
    hist = std::make_shared<HistogramData>(kDefaultBuckets);
    if (initial != 0.0) {
      // Observe initial as value (rare).
      hist->sum.store(initial, std::memory_order_relaxed);
      hist->total_count.store(1, std::memory_order_relaxed);
      // Increment appropriate bucket.
      for (size_t i = 0; i < hist->buckets.size(); ++i) {
        if (initial <= hist->buckets[i]) {
          hist->counts[i]->store(1, std::memory_order_relaxed);
          break;
        }
      }
    }
  } else {
    value.store(initial, std::memory_order_relaxed);
  }
}

Metric::Metric(const Metric& other) noexcept : type(other.type) {
  name = other.name;
  labels = other.labels;
  if (type == MetricType::kHistogram) {
    hist = std::make_shared<HistogramData>(*other.hist);
  } else {
    value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
}

Metric& Metric::operator=(const Metric& other) noexcept {
  if (this == &other) {
    return *this;
  }
  type = other.type;
  name = other.name;
  labels = other.labels;
  if (type == MetricType::kHistogram) {
    hist = std::make_shared<HistogramData>(*other.hist);
  } else {
    value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
  return *this;
}

Metric::Metric(Metric&& other) noexcept : type(other.type) {
  name = other.name;
  labels = other.labels;
  if (type == MetricType::kHistogram) {
    hist = std::move(other.hist);
  } else {
    value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
}

Metric& Metric::operator=(Metric&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  type = other.type;
  name = other.name;
  labels = other.labels;
  if (type == MetricType::kHistogram) {
    hist = std::move(other.hist);
  } else {
    value.store(other.value.load(std::memory_order_relaxed), std::memory_order_relaxed);
  }
  return *this;
}

} // namespace stepcast::metrics