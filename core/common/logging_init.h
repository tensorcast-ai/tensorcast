// Copyright (c) 2025, TensorCast Team.

#pragma once

namespace tensorcast::config::v1 {
class Observability_Logging;
} // namespace tensorcast::config::v1

namespace tensorcast::common {

/**
 * Ensure absl logging is initialized exactly once across all modules.
 *
 * Defaults to INFO level. Entry points that need configuration-driven setup
 * should call initialize_logging_from_config() after parsing runtime config.
 */
void ensure_logging_initialized();

/**
 * Initialize and configure logging based on the observability.logging proto.
 * Safe to call multiple times; updates severity thresholds, VLOG level, and
 * optional file/OTel sinks atomically. Subsequent calls with different config
 * replace previously installed sinks.
 */
void initialize_logging_from_config(const tensorcast::config::v1::Observability_Logging& log_cfg);

} // namespace tensorcast::common
