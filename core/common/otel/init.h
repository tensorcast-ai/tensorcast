// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>

namespace tensorcast::obs {

// Initialize OpenTelemetry C++ SDK + exporter from OTEL_* environment variables.
// - OTEL_SDK_DISABLED: if set truthy, initialization is skipped
// - OTEL_EXPORTER_OTLP_PROTOCOL: grpc | http/protobuf (default: grpc)
// - OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
// - OTEL_EXPORTER_OTLP_INSECURE: true|false (grpc only)
// - OTEL_EXPORTER_OTLP_HEADERS: key=val[,key=val]
// - OTEL_SERVICE_NAME: service name (overrides provided default)
// Extra:
// - TC_OTEL_CXX_CONSOLE: if truthy, attach console/ostream exporter for debugging
// Returns true if SDK initialized; false if disabled or on fatal error (logged).
bool InitFromEnv(const std::string& service_default, const std::string& role);

} // namespace tensorcast::obs
