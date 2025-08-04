// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "core/common/metrics/metrics_export.h"

namespace py = pybind11;

namespace stepcast {

void InitMetricsBindings(py::module_& m) {
  m.def(
      "get_global_metrics_text",
      []() {
        const std::string text = metrics::get_global_metrics_text();
        return py::bytes(text);
      },
      "Get the global metrics snapshot in OpenMetrics text format");
}

} // namespace stepcast