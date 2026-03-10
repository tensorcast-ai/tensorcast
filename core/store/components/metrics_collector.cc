// Copyright (c) 2025-2026, TensorCast Team.

#include "metrics_collector.h"

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/pinned_memory_authority.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/replica_registry.h"

#include <map>
#include <mutex>
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/observer_result.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::store::components {

void MetricsCollector::cpu_mem_available_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  obs->Observe(
      self->cpu_available_bytes_last_,
      {{"location", opentelemetry::common::AttributeValue("cpu")},
       {"memory_type", opentelemetry::common::AttributeValue("available")}});
}

void MetricsCollector::registration_pending_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  obs->Observe(
      self->registration_pending_last_,
      {{"state", opentelemetry::common::AttributeValue("inflight")},
       {"scope", opentelemetry::common::AttributeValue("registration")}});
}

void MetricsCollector::pinned_class_capacity_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  std::vector<PinnedClassSnapshot> snapshots;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    snapshots = self->pinned_classes_last_;
  }
  for (const auto& cls : snapshots) {
    obs->Observe(cls.capacity_slices, {{"class", opentelemetry::common::AttributeValue(cls.name)}});
  }
}

void MetricsCollector::pinned_class_in_use_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  std::vector<PinnedClassSnapshot> snapshots;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    snapshots = self->pinned_classes_last_;
  }
  for (const auto& cls : snapshots) {
    obs->Observe(cls.in_use_slices, {{"class", opentelemetry::common::AttributeValue(cls.name)}});
  }
}

void MetricsCollector::pinned_class_free_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  std::vector<PinnedClassSnapshot> snapshots;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    snapshots = self->pinned_classes_last_;
  }
  for (const auto& cls : snapshots) {
    obs->Observe(cls.free_slices, {{"class", opentelemetry::common::AttributeValue(cls.name)}});
  }
}

void MetricsCollector::pinned_class_waiters_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  std::vector<PinnedClassSnapshot> snapshots;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    snapshots = self->pinned_classes_last_;
  }
  for (const auto& cls : snapshots) {
    obs->Observe(cls.waiters, {{"class", opentelemetry::common::AttributeValue(cls.name)}});
  }
}

void MetricsCollector::pinned_class_timeouts_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  std::vector<PinnedClassSnapshot> snapshots;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    snapshots = self->pinned_classes_last_;
  }
  for (const auto& cls : snapshots) {
    obs->Observe(cls.acquire_timeouts_total, {{"class", opentelemetry::common::AttributeValue(cls.name)}});
  }
}

void MetricsCollector::pinned_total_bytes_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  double value = 0.0;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    value = self->pinned_total_bytes_last_;
  }
  obs->Observe(value);
}

void MetricsCollector::pinned_committed_bytes_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  double value = 0.0;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    value = self->pinned_committed_bytes_last_;
  }
  obs->Observe(value);
}

void MetricsCollector::pinned_budget_exhausted_callback(
    opentelemetry::metrics::ObserverResult result,
    void* state) noexcept {
  auto* self = static_cast<MetricsCollector*>(state);
  if (self == nullptr) {
    return;
  }
  auto obs =
      opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
          result);
  if (!obs) {
    return;
  }
  double value = 0.0;
  {
    const std::lock_guard<std::mutex> lock(self->pinned_snapshot_mutex_);
    value = self->pinned_budget_exhausted_total_last_;
  }
  obs->Observe(value);
}

MetricsCollector::MetricsCollector() {
  meter_ = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  p2p_bytes_total_ = meter_->CreateDoubleCounter("tc_p2p_bytes_total");
  artifact_load_seconds_ = meter_->CreateDoubleHistogram("tc_artifact_load_seconds");
  registration_commit_seconds_ = meter_->CreateDoubleHistogram("tc_register_commit_seconds");

  // Observable gauge for CPU available bytes
  cpu_memory_available_gauge_ = meter_->CreateDoubleObservableGauge("tc_memory_pool_bytes");
  cpu_memory_available_gauge_->AddCallback(&cpu_mem_available_callback, this);

  registration_pending_gauge_ = meter_->CreateDoubleObservableGauge("tc_register_pending_gauge");
  registration_pending_gauge_->AddCallback(&registration_pending_callback, this);

  pinned_class_capacity_slices_gauge_ = meter_->CreateDoubleObservableGauge("tc_pinned_class_capacity_slices");
  pinned_class_capacity_slices_gauge_->AddCallback(&pinned_class_capacity_callback, this);
  pinned_class_in_use_slices_gauge_ = meter_->CreateDoubleObservableGauge("tc_pinned_class_in_use_slices");
  pinned_class_in_use_slices_gauge_->AddCallback(&pinned_class_in_use_callback, this);
  pinned_class_free_slices_gauge_ = meter_->CreateDoubleObservableGauge("tc_pinned_class_free_slices");
  pinned_class_free_slices_gauge_->AddCallback(&pinned_class_free_callback, this);
  pinned_class_waiters_gauge_ = meter_->CreateDoubleObservableGauge("tc_pinned_class_waiters");
  pinned_class_waiters_gauge_->AddCallback(&pinned_class_waiters_callback, this);
  pinned_class_acquire_timeouts_total_gauge_ =
      meter_->CreateDoubleObservableGauge("tc_pinned_class_acquire_timeouts_total");
  pinned_class_acquire_timeouts_total_gauge_->AddCallback(&pinned_class_timeouts_callback, this);

  pinned_total_bytes_gauge_ = meter_->CreateDoubleObservableGauge("tc_pinned_total_bytes");
  pinned_total_bytes_gauge_->AddCallback(&pinned_total_bytes_callback, this);
  pinned_committed_bytes_gauge_ = meter_->CreateDoubleObservableGauge("tc_pinned_committed_bytes");
  pinned_committed_bytes_gauge_->AddCallback(&pinned_committed_bytes_callback, this);
  pinned_budget_exhausted_total_gauge_ = meter_->CreateDoubleObservableGauge("tc_pinned_budget_exhausted_total");
  pinned_budget_exhausted_total_gauge_->AddCallback(&pinned_budget_exhausted_callback, this);
}

void MetricsCollector::update_memory_pool_metrics(const common::memory::PinnedBufferPool& memory_pool) {
  // Track available size via ObservableGauge snapshot
  size_t available_size = memory_pool.get_available_size();
  cpu_available_bytes_last_ = static_cast<double>(available_size);
}

void MetricsCollector::update_pinned_authority_metrics(const common::memory::PinnedMemoryAuthority& authority) {
  const auto names = authority.class_names();
  std::vector<PinnedClassSnapshot> next;
  next.reserve(names.size());

  uint64_t budget_exhausted_total = 0;
  for (const auto& name : names) {
    auto pool_or = authority.get_class_pool(name);
    if (!pool_or.ok() || !*pool_or) {
      continue;
    }
    const auto& pool = *pool_or;
    PinnedClassSnapshot cls;
    cls.name = name;
    cls.capacity_slices = static_cast<double>(pool->capacity_slices());
    cls.free_slices = static_cast<double>(pool->free_slices());
    cls.in_use_slices = static_cast<double>(pool->in_use_slices());
    cls.waiters = static_cast<double>(pool->waiters());
    cls.acquire_timeouts_total = static_cast<double>(pool->acquire_timeouts_total());
    budget_exhausted_total += pool->budget_exhausted_total();
    next.push_back(std::move(cls));
  }

  {
    const std::lock_guard<std::mutex> lock(pinned_snapshot_mutex_);
    pinned_classes_last_ = std::move(next);
    pinned_total_bytes_last_ = static_cast<double>(authority.total_bytes());
    pinned_committed_bytes_last_ = static_cast<double>(authority.committed_bytes());
    pinned_budget_exhausted_total_last_ = static_cast<double>(budget_exhausted_total);
  }
}

void MetricsCollector::update_replica_metrics(const ReplicaRegistry& replica_registry) {
  // Phase 5: converge on tc_*; replica counters moved out of C++ daemon.
  (void)replica_registry; // no-op
}

void MetricsCollector::update_gpu_metrics(DeviceManager& device_manager) {
  // GPU metrics are exposed via DeviceManager's ObservableGauge; nothing to do here
  (void)device_manager;
}

void MetricsCollector::record_operation(const std::string& operation_type, double duration_seconds) {
  // Phase 5: legacy operation metrics removed; rely on record_artifact_load
  (void)operation_type;
  (void)duration_seconds;
}

void MetricsCollector::record_p2p_transfer(size_t bytes_transferred, bool success) {
  p2p_total_transfers_.fetch_add(1, std::memory_order_relaxed);
  if (success) {
    p2p_total_bytes_transferred_.fetch_add(static_cast<std::uint64_t>(bytes_transferred), std::memory_order_relaxed);
    // Unified counter for P2P throughput only
    p2p_bytes_total_->Add(static_cast<double>(bytes_transferred));
  } else {
    p2p_total_transfer_errors_.fetch_add(1, std::memory_order_relaxed);
  }
}

MetricsCollector::P2PTransferSnapshot MetricsCollector::get_p2p_transfer_snapshot() const {
  return P2PTransferSnapshot{
      .total_transfers = p2p_total_transfers_.load(std::memory_order_relaxed),
      .total_bytes_transferred = p2p_total_bytes_transferred_.load(std::memory_order_relaxed),
      .total_transfer_errors = p2p_total_transfer_errors_.load(std::memory_order_relaxed),
  };
}

void MetricsCollector::record_registration_pending(size_t pending_count) {
  registration_pending_last_ = static_cast<double>(pending_count);
}

void MetricsCollector::record_registration_commit(double duration_seconds, std::string_view result) {
  if (!registration_commit_seconds_) {
    return;
  }
  const double clamped_duration = duration_seconds < 0.0 ? 0.0 : duration_seconds;
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  std::string result_label = result.empty() ? std::string("unknown") : std::string(result);
  attrs.emplace("result", opentelemetry::common::AttributeValue(result_label));
  registration_commit_seconds_->Record(
      clamped_duration, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
}

void MetricsCollector::record_memory_eviction() {
  // Phase 5: evictions no longer reported from daemon; no-op
}

void MetricsCollector::update_all_metrics(
    const common::memory::PinnedBufferPool& memory_pool,
    const ReplicaRegistry& replica_registry,
    DeviceManager& device_manager) {
  update_memory_pool_metrics(memory_pool);
  update_replica_metrics(replica_registry);
  update_gpu_metrics(device_manager);
}

void MetricsCollector::record_artifact_load(
    const std::string& source,
    const std::string& device,
    const std::string& phase,
    double duration_seconds,
    std::optional<std::string_view> view_id) {
  // Record into unified histogram with low-cardinality labels
  std::map<std::string, opentelemetry::common::AttributeValue> attrs;
  attrs.emplace("source", opentelemetry::common::AttributeValue(source));
  attrs.emplace("device", opentelemetry::common::AttributeValue(device));
  attrs.emplace("phase", opentelemetry::common::AttributeValue(phase));
  attrs.emplace("view_scope", opentelemetry::common::AttributeValue(view_id.has_value() ? "view" : "canonical"));
  if (view_id.has_value() && !view_id->empty()) {
    std::string truncated{*view_id};
    constexpr size_t kMaxLen = 24;
    if (truncated.size() > kMaxLen) {
      truncated.resize(kMaxLen);
    }
    attrs.emplace("view_id_prefix", opentelemetry::common::AttributeValue(truncated));
  }
  artifact_load_seconds_->Record(
      duration_seconds, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
}

} // namespace tensorcast::store::components
