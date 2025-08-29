// Copyright (c) 2025, TensorCast Team.

#include "core/common/metrics/metrics_export.h"
#include "core/common/metrics/metrics_registry.h"

namespace tensorcast::metrics {

std::string get_global_metrics_text() {
  return MetricsRegistry::instance().to_open_metrics_text();
}

} // namespace tensorcast::metrics