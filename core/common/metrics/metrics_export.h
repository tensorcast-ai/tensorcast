// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <string>

namespace tensorcast::metrics {

// Global function to get the current metrics snapshot in OpenMetrics text format.
// This function is thread-safe and can be called from Python bindings.
std::string get_global_metrics_text();

} // namespace tensorcast::metrics