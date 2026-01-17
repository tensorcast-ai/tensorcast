// Copyright (c) 2025-2026, TensorCast Team.

// RpcContext: thin RAII wrapper around GrpcSpan and RpcMethodMetricsTimer
// Provides a unified pattern for per-RPC observability and convenience helpers.

#pragma once

#include <memory>
#include <optional>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/trace/trace_manager.h"
#include "core/common/trace/trace_request_data.h"
#include "daemon/service/grpc_metrics.h"
#include "daemon/service/grpc_span.h"
#include "folly/io/async/Request.h"
#include "grpcpp/grpcpp.h"

namespace tensorcast::daemon {

class RpcContext {
 public:
  RpcContext(const char* method, grpc::ServerContext& ctx, bool allow_high_card_attrs)
      : allow_high_card_attrs_(allow_high_card_attrs),
        trace_ids_(std::make_shared<common::trace::TraceIds>(make_request_id(method), /*artifact_id=*/"")),
        request_ctx_(make_request_context(method, trace_ids_)),
        request_ctx_guard_(std::in_place, request_ctx_),
        mt_(method, allow_high_card_attrs),
        span_(method, ctx),
        ctx_(ctx) {}

  [[nodiscard]] bool allow_high_card_attrs() const {
    return allow_high_card_attrs_;
  }

  // Access underlying span for attribute setting
  opentelemetry::nostd::shared_ptr<opentelemetry::trace::Span>& span() {
    return span_.span();
  }

  void set_artifact_id(absl::string_view artifact_id) {
    if (!trace_ids_) {
      return;
    }
    trace_ids_->set_artifact_id(std::string(artifact_id));
    common::trace::TraceManager::set_current_artifact_id(std::string(artifact_id));
  }

  void set_request_id(absl::string_view request_id) {
    if (!trace_ids_) {
      return;
    }
    trace_ids_->set_request_id(std::string(request_id));
    common::trace::TraceManager::set_current_request_id(std::string(request_id));
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
  static std::string make_request_id(const char* method) {
    return absl::StrCat("rpc_", method, "_", absl::ToUnixNanos(absl::Now()));
  }

  static std::shared_ptr<folly::RequestContext> make_request_context(
      const char* method,
      const std::shared_ptr<common::trace::TraceIds>& trace_ids) {
    auto request_ctx = std::make_shared<folly::RequestContext>();
    request_ctx->setContextData(
        common::trace::kRpcMethodToken,
        std::make_unique<folly::ImmutableRequestData<std::string>>(std::string(method)));
    request_ctx->setContextData(
        common::trace::kTraceIdsToken, std::make_unique<common::trace::TraceIdsRequestData>(trace_ids));
    request_ctx->setContextData(
        common::trace::kTraceRequestDataToken, std::make_unique<common::trace::TraceRequestData>(trace_ids));
    return request_ctx;
  }

  bool allow_high_card_attrs_;
  std::shared_ptr<common::trace::TraceIds> trace_ids_;
  std::shared_ptr<folly::RequestContext> request_ctx_;
  std::optional<folly::RequestContextScopeGuard> request_ctx_guard_;
  metrics::RpcMethodMetricsTimer mt_;
  GrpcSpan span_;
  grpc::ServerContext& ctx_;
};

} // namespace tensorcast::daemon
