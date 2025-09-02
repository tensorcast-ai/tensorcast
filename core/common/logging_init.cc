
// Copyright (c) 2025, TensorCast Team.

#include "core/common/logging_init.h"

#include <cstdlib>
#include <mutex>
#include <string>
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

namespace tensorcast::common {

std::once_flag logging_init_flag;

void ensure_logging_initialized() {
  std::call_once(logging_init_flag, []() {
    absl::InitializeLog();
    // Set log level from environment variable, default to INFO
    const char* log_level_env = std::getenv("TENSORCAST_LOG_LEVEL");
    absl::LogSeverityAtLeast log_level = absl::LogSeverityAtLeast::kInfo;

    absl::SetStderrThreshold(log_level);

    if (log_level_env != nullptr) {
      std::string level_str(log_level_env);
      if (level_str == "INFO") {
        log_level = absl::LogSeverityAtLeast::kInfo;
      } else if (level_str == "WARNING") {
        log_level = absl::LogSeverityAtLeast::kWarning;
      } else if (level_str == "ERROR") {
        log_level = absl::LogSeverityAtLeast::kError;
      } else if (level_str == "FATAL") {
        log_level = absl::LogSeverityAtLeast::kFatal;
      }
      // If invalid value, keep default INFO
    }
    absl::SetMinLogLevel(log_level);

    // Set vlog level if specified
    const char* vlog_level_env = std::getenv("TENSORCAST_VLOG_LEVEL");
    if (vlog_level_env != nullptr) {
      int vlog_level = std::atoi(vlog_level_env);
      absl::SetGlobalVLogLevel(vlog_level);
    }
  });
}

} // namespace tensorcast::common
