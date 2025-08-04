// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMON_LOGGING_INIT_H_
#define CORE_COMMON_LOGGING_INIT_H_

namespace stepcast::store {

/**
 * @brief Ensure absl logging is initialized exactly once across all modules
 *
 * This function configures logging based on environment variables:
 * - SCSTORE_LOG_LEVEL: Set minimum log level (INFO, WARNING, ERROR, FATAL). Default: INFO
 * - SCSTORE_VLOG_LEVEL: Set verbose log level (integer). Only used if SCSTORE_VLOG_MODULE is also set
 * - SCSTORE_VLOG_MODULE: Module pattern for verbose logging. Must be set with SCSTORE_VLOG_LEVEL
 *
 * Example usage:
 *   export SCSTORE_LOG_LEVEL=WARNING
 *   export SCSTORE_VLOG_LEVEL=2
 */
void ensure_logging_initialized();

} // namespace stepcast::store

#endif // CORE_COMMON_LOGGING_INIT_H_