// Copyright (c) 2025, TensorCast Team.

#include "metrics_collector.h"

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/store/components/device_manager.h"
#include "core/store/components/replica_registry.h"

#include <map>
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

MetricsCollector::MetricsCollector() {
  meter_ = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
  p2p_bytes_total_ = meter_->CreateDoubleCounter("tc_p2p_bytes_total");
  artifact_load_seconds_ = meter_->CreateDoubleHistogram("tc_artifact_load_seconds");

  // Observable gauge for CPU available bytes
  cpu_memory_available_gauge_ = meter_->CreateDoubleObservableGauge("tc_memory_pool_bytes");
  cpu_memory_available_gauge_->AddCallback(&cpu_mem_available_callback, this);
}

void MetricsCollector::update_memory_pool_metrics(const common::memory::PinnedBufferPool& memory_pool) {
  // Track available size via ObservableGauge snapshot
  size_t available_size = memory_pool.get_available_size();
  cpu_available_bytes_last_ = static_cast<double>(available_size);
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
  if (success) {
    // Unified counter for P2P throughput only
    p2p_bytes_total_->Add(static_cast<double>(bytes_transferred));
  }
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
  attrs.emplace("view_scope", opentelemetry::common::AttributeValue(view_id.has_value() ? "variant" : "canonical"));
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
