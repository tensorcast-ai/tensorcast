// Copyright (c) 2025, TensorCast Team.

#include "core/common/otel/init.h"

#include <string>

#include "absl/log/log.h"
#include "tensorcast/config/v1/common.pb.h"

#include "opentelemetry/context/propagation/global_propagator.h"
#include "opentelemetry/trace/propagation/http_trace_context.h"
#include "opentelemetry/trace/provider.h"

#include "opentelemetry/exporters/otlp/otlp_grpc_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_grpc_exporter_options.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_factory.h"
#include "opentelemetry/exporters/otlp/otlp_http_exporter_options.h"

#include "opentelemetry/sdk/resource/resource.h"
#include "opentelemetry/sdk/trace/batch_span_processor_factory.h"
#include "opentelemetry/sdk/trace/tracer_provider_factory.h"

// Optional console exporter
// Console exporter (ostream) may not be available in the Bazel package; omit.

namespace sdktrace = opentelemetry::sdk::trace;
namespace resource = opentelemetry::sdk::resource;

namespace tensorcast::common::otel {

static void strip_scheme(std::string& s) {
  if (s.rfind("http://", 0) == 0) {
    s = s.substr(7);
  } else if (s.rfind("https://", 0) == 0) {
    s = s.substr(8);
  }
}

// Env-based initialization removed in final scheme.

bool init_from_config(const tensorcast::config::v1::Observability& obs, const std::string& role) {
  // Honor top-level enable switch
  if (!obs.otel().enabled()) {
    LOG(INFO) << "OpenTelemetry disabled via config";
    return false;
  }

  try {
    // Resource attributes
    std::string service_name =
        obs.otel().service_name().empty() ? std::string("tensorcast") : obs.otel().service_name();
    resource::ResourceAttributes attrs{
        {"service.name", service_name}, {"service.namespace", std::string("tensorcast")}, {"tc.node.role", role}};
    auto res = resource::Resource::Create(attrs);

    // Choose exporter based on protocol
    std::unique_ptr<sdktrace::SpanExporter> exporter;
    const auto proto = obs.otel().exporter_protocol();
    if (proto == tensorcast::config::v1::Observability_OTelProtocol_O_TEL_PROTOCOL_HTTP_PROTOBUF) {
      opentelemetry::exporter::otlp::OtlpHttpExporterOptions opts;
      std::string url = obs.otel().exporter_otlp_endpoint();
      if (url.empty()) {
        url = "http://127.0.0.1:4318/v1/traces";
      }
      opts.url = url;
      exporter = opentelemetry::exporter::otlp::OtlpHttpExporterFactory::Create(opts);
    } else {
      opentelemetry::exporter::otlp::OtlpGrpcExporterOptions opts;
      std::string ep = obs.otel().exporter_otlp_endpoint().empty() ? std::string("127.0.0.1:4317")
                                                                   : obs.otel().exporter_otlp_endpoint();
      strip_scheme(ep);
      opts.endpoint = ep;
      // C++ specific options from otel_cxx
      opts.use_ssl_credentials = !obs.otel_cxx().exporter_insecure();
      for (const auto& [k, v] : obs.otel_cxx().exporter_headers()) {
        opts.metadata.emplace(k, v);
      }
      exporter = opentelemetry::exporter::otlp::OtlpGrpcExporterFactory::Create(opts);
    }

    sdktrace::BatchSpanProcessorOptions bsp_opts;
    auto processor = sdktrace::BatchSpanProcessorFactory::Create(std::move(exporter), bsp_opts);
    auto provider_up = sdktrace::TracerProviderFactory::Create(std::move(processor), res);
    std::shared_ptr<opentelemetry::trace::TracerProvider> provider_sp(std::move(provider_up));
    opentelemetry::trace::Provider::SetTracerProvider(provider_sp);
    opentelemetry::context::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
        opentelemetry::nostd::shared_ptr<opentelemetry::context::propagation::TextMapPropagator>(
            new opentelemetry::trace::propagation::HttpTraceContext()));
    LOG(INFO) << "OpenTelemetry C++ SDK initialized from config for service: " << service_name;
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "OTel C++ SDK init (config) failed: " << e.what();
  } catch (...) {
    LOG(WARNING) << "OTel C++ SDK init (config) failed: unknown error";
  }
  return false;
}

} // namespace tensorcast::common::otel
