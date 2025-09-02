// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"

namespace tensorcast::common::otel {

// Bridge RAII that mirrors existing TraceScope into an OpenTelemetry Span.
// This is always compiled; if no provider/exporter is configured, spans are no-op.
class TraceScopeBridge {
 public:
  TraceScopeBridge(const std::string& artifact_id, const std::string& stage) : active_(true) {
    auto tracer = opentelemetry::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.core");
    span_ = tracer->StartSpan(stage);
    scope_ = std::make_unique<opentelemetry::trace::Scope>(span_);
    // Attach artifact_id as requested for cross-component correlation.
    if (!artifact_id.empty()) {
      span_->SetAttribute("tc.artifact.id", artifact_id);
    }
  }

  ~TraceScopeBridge() {
    if (active_ && span_) {
      span_->End();
    }
  }

  TraceScopeBridge(const TraceScopeBridge&) = delete;
  TraceScopeBridge& operator=(const TraceScopeBridge&) = delete;

  TraceScopeBridge(TraceScopeBridge&& other) noexcept
      : span_(std::move(other.span_)), scope_(std::move(other.scope_)), active_(other.active_) {
    other.active_ = false;
  }

  TraceScopeBridge& operator=(TraceScopeBridge&& other) noexcept {
    if (this != &other) {
      span_ = std::move(other.span_);
      scope_ = std::move(other.scope_);
      active_ = other.active_;
      other.active_ = false;
    }
    return *this;
  }

 private:
  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span_;
  std::unique_ptr<opentelemetry::trace::Scope> scope_;
  bool active_{true};
};

} // namespace tensorcast::common::otel
