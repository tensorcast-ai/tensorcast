// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "absl/log/log_sink.h"

namespace tensorcast::common::otel {

// Install an absl::LogSink that writes log entries enriched with
// OpenTelemetry trace_id/span_id. The sink is enabled by environment:
// - TC_LOG_OTEL_CONTEXT_ENABLED: truthy to enable (default: 1)
// - TC_LOG_SINK_FILE: path to append; if empty, no sink is installed
// This function is idempotent and safe to call multiple times.
void install_otel_log_sink_from_env();

// Remove the installed sink (if any). Intended for tests or controlled
// shutdown paths. Safe to call even if not installed.
void remove_otel_log_sink();
} // namespace tensorcast::common::otel
