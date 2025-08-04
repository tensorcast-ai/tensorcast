// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/common/metrics/metrics_export.h"
#include "core/common/metrics/metrics_registry.h"

namespace stepcast::metrics {

std::string get_global_metrics_text() {
  return MetricsRegistry::instance().to_open_metrics_text();
}

} // namespace stepcast::metrics