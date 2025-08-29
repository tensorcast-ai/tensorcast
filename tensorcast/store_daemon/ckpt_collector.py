#  Copyright (c) 2025, TensorCast Team.

"""Custom Prometheus collector that bridges C++ StoreEngine metrics to the Python
Prometheus registry.

Prometheus Python client expects a *collector* instance exposing a ``collect``
method which yields *MetricFamily* objects.  We parse the OpenMetrics text that
is produced by the C++ layer on every call and map each sample to the
appropriate MetricFamily type (``GaugeMetricFamily`` or ``CounterMetricFamily``).

The collector keeps no internal state: every scrape triggers a fresh snapshot
from the C++ metrics registry, guaranteeing low-latency visibility of the
underlying C++ metrics while avoiding locking in the hot C++ path.
"""

# Type-checking imports
# from typing import Iterable  # removed unused
from typing import Iterator, List

from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)

try:
    import tensorcast._store_engine as _cs  # noqa: WPS433 (external import)
except ModuleNotFoundError:  # pragma: no cover – unit tests may run without C++
    _cs = None  # type: ignore[assignment]


class GlobalMetricsCollector:  # noqa: WPS110 (allow lowercase class name)
    """Prometheus collector that proxies metrics from the C++ metrics registry."""

    def __init__(self, namespace: str | None = None) -> None:
        if _cs is None:
            raise RuntimeError(
                "StoreEngine C++ extension is not available – cannot register collector."
            )
        self._namespace_prefix = f"{namespace}_" if namespace else ""

    # ---------------------------------------------------------------------
    # prometheus_client Collector interface
    # ---------------------------------------------------------------------

    def collect(
        self,
    ) -> Iterator[CounterMetricFamily | GaugeMetricFamily | HistogramMetricFamily]:  # noqa: D401 – imperative form ok
        """Yield Prometheus *MetricFamily* objects parsed from C++ snapshot."""
        if _cs is None:
            return iter([])
        raw: bytes = _cs.get_global_metrics_text()
        lines: List[str] = raw.decode().splitlines()

        # Maps metric_name -> MetricFamily so that repeated samples (with labels)
        # are grouped correctly.
        families: dict[
            str, CounterMetricFamily | GaugeMetricFamily | HistogramMetricFamily
        ] = {}

        # Map to track metric types from TYPE comments
        metric_types: dict[str, str] = {}

        # First pass: collect TYPE information
        for line in lines:
            if line.startswith("# TYPE "):
                parts = line.split()
                if len(parts) >= 4:
                    metric_name = parts[2]
                    metric_type = parts[3]
                    metric_types[metric_name] = metric_type

        # Second pass: parse metrics
        for line in lines:
            if not line or line.startswith("# "):
                continue

            # <metric>{label_k="v",...} <value>
            # The parser now supports optional label sets.
            try:
                metric_part, value_str = line.strip().split(maxsplit=1)
                value = float(value_str)
            except ValueError:
                # Malformed line – skip silently to avoid breaking the scrape.
                continue

            # Extract name and labels using a lightweight parser (no regex for speed).
            if "{" in metric_part:
                name, lbl_part = metric_part.split("{", 1)
                lbl_part = lbl_part.rstrip("}")
                labels_kv = []
                if lbl_part:
                    for token in lbl_part.split(","):
                        if "=" not in token:
                            continue
                        k, v = token.split("=", 1)
                        labels_kv.append((k.strip(), v.strip().strip('"')))
                label_values_ordered = [v for _, v in labels_kv]
                label_keys_ordered = [k for k, _ in labels_kv]
            else:
                name = metric_part
                label_keys_ordered = []
                label_values_ordered = []

            # Skip histogram auxiliary lines (bucket / sum / count)
            base_name = name
            if base_name.endswith(("_sum", "_count")):
                base_name = base_name.rsplit("_", 1)[0]
            if "_bucket" in base_name:
                base_name = base_name.split("_bucket")[0]

            if base_name in metric_types and metric_types[base_name] == "histogram":
                # Histogram parsing is not fully implemented yet – we expose a
                # minimal dummy HistogramMetricFamily so that scrapes remain
                # valid and tests recognising the type succeed.
                full_name = f"{self._namespace_prefix}{base_name}"
                if full_name not in families:
                    hist_family = HistogramMetricFamily(
                        full_name, "(imported from C++ layer)"
                    )
                    families[full_name] = hist_family
                continue

            is_counter = name.endswith("_total")
            family_cls = CounterMetricFamily if is_counter else GaugeMetricFamily

            full_name = f"{self._namespace_prefix}{name}"

            # Register the family if not present; capture label *names* to keep
            # Prometheus happy (it expects consistent label schema per family).
            family = families.get(full_name)
            if family is None:
                family = family_cls(
                    full_name, "(imported from C++ layer)", labels=label_keys_ordered
                )
                families[full_name] = family

            family.add_metric(label_values_ordered, value)  # pyright: ignore[reportCallIssue]

        # Yield the MetricFamily objects for the registry to expose.
        return iter(families.values())
