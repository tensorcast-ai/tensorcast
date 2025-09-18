// Copyright (c) 2025, TensorCast Team.

#pragma once
// Prefer explicit proto include over forward declaration
#include "tensorcast/config/v1/common.pb.h"

namespace tensorcast::common::otel {

// Install an absl::LogSink that writes log entries enriched with
// OpenTelemetry trace_id/span_id (config-driven).

// Remove the installed sink (if any). Intended for tests or controlled
// shutdown paths. Safe to call even if not installed.
void remove_otel_log_sink();

// Install or update OTel-enriched log sink based on Observability.Logging.
// When otel_context_enabled is false or sink_file empty, the sink is removed.
void install_otel_log_sink_from_config(const tensorcast::config::v1::Observability_Logging& log_cfg);

// Optionally install a plain file sink (no OTel enrichment) if `logging.file`
// is set. Safe to call multiple times; later calls replace the sink if the
// target path changes or remove it when empty.
void install_plain_log_sink_from_config(const tensorcast::config::v1::Observability_Logging& log_cfg);
} // namespace tensorcast::common::otel
