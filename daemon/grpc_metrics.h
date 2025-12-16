// Copyright (c) 2025, TensorCast Team.

// RpcMethodMetricsTimer: RAII helper to record per-RPC metrics
// - Increments a request counter on construction
// - Records duration histogram on destruction
// - Defaults to error=true; call mark_success() on successful RPC paths
//
// Uses OpenTelemetry metrics API and standardizes attributes.

#pragma once

#include <chrono>
#include <map>
#include <string>

#include "core/common/trace/trace_manager.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::daemon::metrics {

class RpcMethodMetricsTimer {
 public:
  explicit RpcMethodMetricsTimer(const char* method, bool allow_high_card_attrs)
      : method_(method), allow_high_card_attrs_(allow_high_card_attrs), start_(std::chrono::steady_clock::now()) {
    // Best-effort: create instruments once per process; ignore failures
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      // Instruments
      static auto req_counter = meter->CreateDoubleCounter("tc_rpc_requests_total");
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      attrs.emplace("rpc.method", opentelemetry::common::AttributeValue(std::string(method_)));
      if (allow_high_card_attrs_) {
        const std::string& request_id = common::trace::TraceManager::current_request_id();
        const std::string& artifact_id = common::trace::TraceManager::current_artifact_id();
        if (!request_id.empty()) {
          attrs.emplace("tc.request.id", opentelemetry::common::AttributeValue(request_id));
        }
        if (!artifact_id.empty()) {
          attrs.emplace("tc.artifact.id", opentelemetry::common::AttributeValue(artifact_id));
        }
      }
      req_counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
    } catch (...) {
      // Silently degrade; metrics must not affect control flow
    }
  }

  ~RpcMethodMetricsTimer() {
    // Observe duration and (optionally) errors
    const auto end = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(end - start_).count();
    try {
      static auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      static auto hist = meter->CreateDoubleHistogram("tc_rpc_duration_seconds");
      std::map<std::string, opentelemetry::common::AttributeValue> attrs;
      attrs.emplace("rpc.method", opentelemetry::common::AttributeValue(std::string(method_)));
      if (allow_high_card_attrs_) {
        const std::string& request_id = common::trace::TraceManager::current_request_id();
        const std::string& artifact_id = common::trace::TraceManager::current_artifact_id();
        if (!request_id.empty()) {
          attrs.emplace("tc.request.id", opentelemetry::common::AttributeValue(request_id));
        }
        if (!artifact_id.empty()) {
          attrs.emplace("tc.artifact.id", opentelemetry::common::AttributeValue(artifact_id));
        }
      }
      hist->Record(secs, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
      if (error_) {
        static auto err_counter = meter->CreateDoubleCounter("tc_rpc_errors_total");
        err_counter->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
      }
    } catch (...) {
      // Silently degrade
    }
  }

  // Mark RPC as successful; by default an instance is considered error
  void mark_success() {
    error_ = false;
  }

 private:
  const char* method_;
  bool allow_high_card_attrs_{false};
  std::chrono::steady_clock::time_point start_;
  bool error_{true};
};

} // namespace tensorcast::daemon::metrics
