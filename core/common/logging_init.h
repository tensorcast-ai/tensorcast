// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMON_LOGGING_INIT_H_
#define CORE_COMMON_LOGGING_INIT_H_

namespace tensorcast::common {

/**
 * Ensure absl logging is initialized exactly once across all modules.
 *
 * Defaults to INFO level. For configuration-driven severity/VLOG and file
 * sinks, entry points should call:
 *  - tensorcast::common::otel::apply_absl_log_level_from_config
 *  - tensorcast::common::otel::install_plain_log_sink_from_config
 *  - tensorcast::common::otel::install_otel_log_sink_from_config
 */
void ensure_logging_initialized();

} // namespace tensorcast::common

#endif // CORE_COMMON_LOGGING_INIT_H_
