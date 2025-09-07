// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>

namespace tensorcast::common::otel {

// Env-based initialization removed in final scheme.

// Initialize OpenTelemetry C++ SDK from Observability config (no env reliance).
// Returns true on success. On failure, logs a warning and returns false.
class ObservabilityCfgFwd; // forward-declare helper type

} // namespace tensorcast::common::otel

namespace tensorcast::config::v1 {
class Observability;
} // namespace tensorcast::config::v1

namespace tensorcast::common::otel {
bool init_from_config(const tensorcast::config::v1::Observability& obs, const std::string& role);
} // namespace tensorcast::common::otel
