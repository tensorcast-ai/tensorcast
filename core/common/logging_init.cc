// Copyright (c) 2025-2026, TensorCast Team.

#include "core/common/logging_init.h"

#include <algorithm>
#include <mutex>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "core/common/otel/logging_sink.h"
#include "tensorcast/config/v1/common.pb.h"

namespace tensorcast::common {

namespace {

std::once_flag logging_init_flag;
std::mutex logging_config_mu;

absl::LogSeverityAtLeast map_log_level(tensorcast::config::v1::Observability::LogLevel level) {
  using tensorcast::config::v1::Observability;
  switch (level) {
    case Observability::LOG_LEVEL_DEBUG:
    case Observability::LOG_LEVEL_INFO:
      return absl::LogSeverityAtLeast::kInfo; // DEBUG routed through VLOG
    case Observability::LOG_LEVEL_WARN:
      return absl::LogSeverityAtLeast::kWarning;
    case Observability::LOG_LEVEL_ERROR:
      return absl::LogSeverityAtLeast::kError;
    case Observability::LOG_LEVEL_UNSPECIFIED:
    default:
      return absl::LogSeverityAtLeast::kInfo;
  }
}

void set_vlog_level(int32_t vlog_level) {
  const int32_t clamped = std::max<int32_t>(0, vlog_level);
  if (clamped == 0) {
    return;
  }
  absl::SetVLogLevel("*", clamped);
}

} // namespace

void ensure_logging_initialized() {
  std::call_once(logging_init_flag, []() {
    absl::InitializeLog();
    const absl::LogSeverityAtLeast default_level = absl::LogSeverityAtLeast::kInfo;
    absl::SetStderrThreshold(default_level);
    absl::SetMinLogLevel(default_level);
  });
}

void initialize_logging_from_config(const tensorcast::config::v1::Observability_Logging& log_cfg) {
  ensure_logging_initialized();

  std::lock_guard<std::mutex> lk(logging_config_mu);

  const absl::LogSeverityAtLeast min_level = map_log_level(log_cfg.level());
  absl::SetStderrThreshold(min_level);
  absl::SetMinLogLevel(min_level);
  set_vlog_level(log_cfg.vlog_level());

  otel::install_plain_log_sink_from_config(log_cfg);
  otel::install_otel_log_sink_from_config(log_cfg);

  LOG(INFO) << "Logging initialized from config: \n" << log_cfg.DebugString();
}

} // namespace tensorcast::common
