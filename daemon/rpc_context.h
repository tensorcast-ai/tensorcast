// Copyright (c) 2025, TensorCast Team.

// RpcContext: thin RAII wrapper around GrpcSpan and RpcMethodMetricsTimer
// Provides a unified pattern for per-RPC observability and convenience helpers.

#pragma once

#include <string>

#include "absl/strings/string_view.h"
#include "daemon/grpc_metrics.h"
#include "daemon/grpc_span.h"
#include "grpcpp/grpcpp.h"

namespace tensorcast::daemon {

class RpcContext {
 public:
  RpcContext(const char* method, grpc::ServerContext& ctx, bool allow_high_card_attrs)
      : allow_high_card_attrs_(allow_high_card_attrs), mt_(method), span_(method, ctx), ctx_(ctx) {}

  [[nodiscard]] bool allow_high_card_attrs() const {
    return allow_high_card_attrs_;
  }

  // Access underlying span for attribute setting
  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span() {
    return span_.span();
  }

  // Mark successful completion for metrics
  void mark_success() {
    mt_.mark_success();
  }

  // Access ServerContext to compute deadlines, etc.
  grpc::ServerContext& server_context() {
    return ctx_;
  }

  // Convenience: return a grpc::Status failure with message
  static grpc::Status fail(grpc::StatusCode code, absl::string_view msg) {
    return {code, std::string(msg)};
  }

 private:
  bool allow_high_card_attrs_;
  metrics::RpcMethodMetricsTimer mt_;
  GrpcSpan span_;
  grpc::ServerContext& ctx_;
};

} // namespace tensorcast::daemon
