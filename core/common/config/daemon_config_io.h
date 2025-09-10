// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "tensorcast/config/v1/daemon_config.pb.h"

namespace tensorcast::common::config {

// Load DaemonConfig from a YAML or JSON file and normalize defaults.
// - Strict unknown-field rejection
// - Supports human-friendly units for byte fields: KB/MB/GB/TB (case-insensitive)
// - Duration fields accept strings like "500ms", "30s", "2m"
// - Normalizes enums from convenient strings (e.g., logging.level: INFO)
absl::StatusOr<tensorcast::config::v1::DaemonConfig> load_daemon_config_from_file(const std::string& path);

// Load DaemonConfig from inline YAML/JSON text and normalize defaults.
// Content may be provided in either YAML or JSON; YAML parser is attempted first
// and falls back to JSON parsing. Unknown fields are rejected strictly.
absl::StatusOr<tensorcast::config::v1::DaemonConfig> load_daemon_config_from_text(const std::string& content);

// Apply runtime defaults to partially filled config (proto3 has no field defaults).
// Only numeric/time-like defaults are applied to avoid overriding explicit false booleans.
void normalize_defaults(tensorcast::config::v1::DaemonConfig* cfg);

} // namespace tensorcast::common::config
