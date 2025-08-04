// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <string>

namespace stepcast::metrics {

// Global function to get the current metrics snapshot in OpenMetrics text format.
// This function is thread-safe and can be called from Python bindings.
std::string get_global_metrics_text();

} // namespace stepcast::metrics