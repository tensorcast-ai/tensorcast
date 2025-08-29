// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMON_LOGGING_INIT_H_
#define CORE_COMMON_LOGGING_INIT_H_

namespace tensorcast::store {

/**
 * @brief Ensure absl logging is initialized exactly once across all modules
 *
 * This function configures logging based on environment variables:
 * - TENSORCAST_LOG_LEVEL: Set minimum log level (INFO, WARNING, ERROR, FATAL). Default: INFO
 * - TENSORCAST_VLOG_LEVEL: Set verbose log level (integer). Only used if TENSORCAST_VLOG_MODULE is also set
 * - TENSORCAST_VLOG_MODULE: Module pattern for verbose logging. Must be set with TENSORCAST_VLOG_LEVEL
 *
 * Example usage:
 *   export TENSORCAST_LOG_LEVEL=WARNING
 *   export TENSORCAST_VLOG_LEVEL=2
 */
void ensure_logging_initialized();

} // namespace tensorcast::store

#endif // CORE_COMMON_LOGGING_INIT_H_