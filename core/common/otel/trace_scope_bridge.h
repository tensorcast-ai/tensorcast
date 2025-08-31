// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "core/common/otel/config.h"

#if TC_ENABLE_OTEL_CXX
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"
#endif

namespace tensorcast::obs {

// Bridge RAII that mirrors existing TraceScope into an OpenTelemetry Span.
// This is a no-op unless TC_ENABLE_OTEL_CXX is enabled at build time.
class TraceScopeBridge {
 public:
  TraceScopeBridge(const std::string& artifact_id, const std::string& stage)
#if TC_ENABLE_OTEL_CXX
      : active_(true)
#endif
  {
#if TC_ENABLE_OTEL_CXX
    using namespace opentelemetry;
    auto tracer = trace::Provider::GetTracerProvider()->GetTracer("tensorcast.core");
    span_ = tracer->StartSpan(stage);
    scope_ = std::make_unique<trace::Scope>(span_);
    // Attach artifact_id as requested for cross-component correlation.
    if (!artifact_id.empty()) {
      span_->SetAttribute("tc.artifact.id", artifact_id);
    }
#endif
  }

  ~TraceScopeBridge() {
#if TC_ENABLE_OTEL_CXX
    if (active_ && span_) {
      span_->End();
    }
#endif
  }

  TraceScopeBridge(const TraceScopeBridge&) = delete;
  TraceScopeBridge& operator=(const TraceScopeBridge&) = delete;

  TraceScopeBridge(TraceScopeBridge&& other) noexcept
#if TC_ENABLE_OTEL_CXX
      : span_(std::move(other.span_)),
        scope_(std::move(other.scope_)),
        active_(other.active_)
#endif
  {
#if TC_ENABLE_OTEL_CXX
    other.active_ = false;
#endif
  }

  TraceScopeBridge& operator=(TraceScopeBridge&& other) noexcept {
    if (this != &other) {
#if TC_ENABLE_OTEL_CXX
      span_ = std::move(other.span_);
      scope_ = std::move(other.scope_);
      active_ = other.active_;
      other.active_ = false;
#endif
    }
    return *this;
  }

 private:
#if TC_ENABLE_OTEL_CXX
  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span_;
  std::unique_ptr<opentelemetry::trace::Scope> scope_;
  bool active_{true};
#endif
};

} // namespace tensorcast::obs
