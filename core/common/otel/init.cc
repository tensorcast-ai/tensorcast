// Copyright (c) 2025, TensorCast Team.

#include "core/common/otel/init.h"

#include <cstdlib>
#include <string>

#include "absl/log/log.h"

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

namespace otel = opentelemetry;
namespace sdktrace = otel::sdk::trace;
namespace resource = otel::sdk::resource;

namespace tensorcast::obs {

static bool truthy(const char* v) {
  if (!v) {
    return false;
  }
  std::string s(v);
  for (auto& c : s) {
    c = static_cast<char>(::tolower(c));
  }
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

static std::string get_env(const char* k, const std::string& def = {}) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : def;
}

static void strip_scheme(std::string& s) {
  if (s.rfind("http://", 0) == 0) {
    s = s.substr(7);
  } else if (s.rfind("https://", 0) == 0) {
    s = s.substr(8);
  }
}

bool InitFromEnv(const std::string& service_default, const std::string& role) {
  if (truthy(std::getenv("OTEL_SDK_DISABLED"))) {
    LOG(INFO) << "OTel C++ SDK disabled via OTEL_SDK_DISABLED";
    return false;
  }

  try {
    // Resource attributes
    const std::string service_name = get_env("OTEL_SERVICE_NAME", service_default);
    resource::ResourceAttributes attrs{
        {"service.name", service_name},
        {"service.namespace", std::string("tensorcast")},
        {"tc.node.role", role},
    };
    auto res = resource::Resource::Create(attrs);

    // Choose exporter based on protocol
    std::unique_ptr<sdktrace::SpanExporter> exporter;
    std::string protocol = get_env("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc");
    // Trim spaces & lowercase
    for (auto& c : protocol) {
      c = static_cast<char>(::tolower(c));
    }

    if (protocol == "http/protobuf" || protocol == "http") {
      otel::exporter::otlp::OtlpHttpExporterOptions opts;
      std::string url = get_env(
          "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
          get_env("OTEL_EXPORTER_OTLP_ENDPOINT", std::string("http://127.0.0.1:4318/v1/traces")));
      opts.url = url;
      exporter = otel::exporter::otlp::OtlpHttpExporterFactory::Create(opts);
    } else { // grpc
      otel::exporter::otlp::OtlpGrpcExporterOptions opts;
      std::string ep = get_env(
          "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", get_env("OTEL_EXPORTER_OTLP_ENDPOINT", std::string("127.0.0.1:4317")));
      // Allow ep like http://host:4317
      strip_scheme(ep);
      opts.endpoint = ep;
      opts.use_ssl_credentials = !truthy(std::getenv("OTEL_EXPORTER_OTLP_INSECURE"));
      // Metadata headers
      std::string headers = get_env("OTEL_EXPORTER_OTLP_HEADERS");
      if (!headers.empty()) {
        // Parse k=v,k2=v2
        size_t start = 0;
        while (start < headers.size()) {
          size_t comma = headers.find(',', start);
          std::string_view kv = (comma == std::string::npos) ? std::string_view(headers).substr(start)
                                                             : std::string_view(headers).substr(start, comma - start);
          size_t eq = kv.find('=');
          if (eq != std::string::npos) {
            std::string key(kv.substr(0, eq));
            std::string val(kv.substr(eq + 1));
            opts.metadata.emplace(std::move(key), std::move(val));
          }
          if (comma == std::string::npos) {
            break;
          }
          start = comma + 1;
        }
      }
      exporter = otel::exporter::otlp::OtlpGrpcExporterFactory::Create(opts);
    }

    // Batch span processor with default options
    sdktrace::BatchSpanProcessorOptions bsp_opts;
    auto processor = sdktrace::BatchSpanProcessorFactory::Create(std::move(exporter), bsp_opts);
    auto provider_up = sdktrace::TracerProviderFactory::Create(std::move(processor), res);

    // Optional console exporter omitted (not available via Bazel package)

    // Convert unique_ptr -> shared_ptr for Provider API
    std::shared_ptr<otel::trace::TracerProvider> provider_sp(std::move(provider_up));
    otel::trace::Provider::SetTracerProvider(provider_sp);
    // Set W3C propagator
    otel::context::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
        opentelemetry::nostd::shared_ptr<opentelemetry::context::propagation::TextMapPropagator>(
            new opentelemetry::trace::propagation::HttpTraceContext()));

    LOG(INFO) << "OpenTelemetry C++ SDK initialized for service: " << service_name;
    return true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "OTel C++ SDK init failed: " << e.what();
  } catch (...) {
    LOG(WARNING) << "OTel C++ SDK init failed: unknown error";
  }
  return false;
}

} // namespace tensorcast::obs
