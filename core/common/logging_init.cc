
// Copyright (c) 2025, TensorCast Team.

#include "core/common/logging_init.h"

#include <cstdlib>
#include <mutex>
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

namespace tensorcast::common {

std::once_flag logging_init_flag;

void ensure_logging_initialized() {
  std::call_once(logging_init_flag, []() {
    absl::InitializeLog();
    // Default log level INFO, no environment fallbacks
    absl::LogSeverityAtLeast log_level = absl::LogSeverityAtLeast::kInfo;
    absl::SetStderrThreshold(log_level);
    absl::SetMinLogLevel(log_level);
  });
}

} // namespace tensorcast::common
