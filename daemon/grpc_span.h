// Copyright (c) 2025, TensorCast Team.

// GrpcSpan: RAII helper to create OpenTelemetry server spans for gRPC handlers
// Reduces boilerplate and guarantees standard rpc.* attributes are set.

#pragma once

#include "core/common/otel/grpc_propagation.h"
#include "grpcpp/grpcpp.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"

namespace tensorcast::daemon {

class GrpcSpan {
 public:
  GrpcSpan(const char* method, grpc::ServerContext& ctx)
      : tracer_(opentelemetry::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon")) {
    auto parent_ctx = common::otel::extract_from_server_metadata(ctx);
    ctx_token_ = opentelemetry::context::RuntimeContext::Attach(parent_ctx);
    opentelemetry::trace::StartSpanOptions opts;
    opts.kind = opentelemetry::trace::SpanKind::kServer;
    span_ = tracer_->StartSpan(method, opts);
    scope_ = std::make_unique<opentelemetry::trace::Scope>(span_);
    // Standard attributes
    span_->SetAttribute("rpc.system", "grpc");
    span_->SetAttribute("rpc.service", "tensorcast.daemon.StoreDaemon");
    span_->SetAttribute("rpc.method", method);
  }

  ~GrpcSpan() {
    scope_.reset();
  }

  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span() {
    return span_;
  }

 private:
  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Tracer> tracer_;
  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span> span_;
  std::unique_ptr<opentelemetry::trace::Scope> scope_;
  opentelemetry::nostd::unique_ptr<opentelemetry::context::Token> ctx_token_;
};

} // namespace tensorcast::daemon
