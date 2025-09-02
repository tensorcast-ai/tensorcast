// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>

#include "grpcpp/grpcpp.h"

#include "opentelemetry/context/context.h"
#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/context/propagation/text_map_propagator.h"
#include "opentelemetry/context/runtime_context.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/trace/provider.h"
#include "opentelemetry/trace/scope.h"
#include "opentelemetry/trace/span.h"

namespace tensorcast::common::otel {

// TextMapCarrier for extracting trace context from gRPC ServerContext metadata.
class GrpcServerCarrier : public opentelemetry::context::propagation::TextMapCarrier {
 public:
  explicit GrpcServerCarrier(grpc::ServerContext& ctx) : ctx_(ctx) {}

  opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view key) const noexcept override {
    const auto& md = ctx_.client_metadata();
    auto it = md.find(grpc::string_ref{key.data(), key.size()});
    if (it != md.end()) {
      const grpc::string_ref& v = it->second;
      return opentelemetry::nostd::string_view(v.data(), v.length());
    }
    return opentelemetry::nostd::string_view{};
  }

  void Set(opentelemetry::nostd::string_view /*key*/, opentelemetry::nostd::string_view /*value*/) noexcept override {}

 private:
  grpc::ServerContext& ctx_;
};

// TextMapCarrier for injecting trace context into gRPC ClientContext metadata.
class GrpcClientCarrier : public opentelemetry::context::propagation::TextMapCarrier {
 public:
  explicit GrpcClientCarrier(grpc::ClientContext& ctx) : ctx_(ctx) {}

  opentelemetry::nostd::string_view Get(opentelemetry::nostd::string_view /*key*/) const noexcept override {
    return opentelemetry::nostd::string_view{}; // Not used in Inject path
  }

  void Set(opentelemetry::nostd::string_view key, opentelemetry::nostd::string_view value) noexcept override {
    ctx_.AddMetadata(std::string(key.data(), key.size()), std::string(value.data(), value.size()));
  }

 private:
  grpc::ClientContext& ctx_;
};

// Helper: extract parent context from ServerContext via global propagator.
inline opentelemetry::context::Context ExtractFromServerMetadata(grpc::ServerContext& ctx) {
  auto propagator = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
  GrpcServerCarrier carrier{ctx};
  opentelemetry::context::Context base;
  return propagator->Extract(carrier, base);
}

// Helper: inject current runtime context into ClientContext via global propagator.
inline void InjectIntoClientMetadata(grpc::ClientContext& ctx) {
  auto propagator = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
  auto current = opentelemetry::context::RuntimeContext::GetCurrent();
  GrpcClientCarrier carrier{ctx};
  propagator->Inject(carrier, current);
}

} // namespace tensorcast::common::otel
