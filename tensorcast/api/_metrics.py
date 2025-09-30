#  Copyright (c) 2025, TensorCast Team.

"""Store client metrics backed by OpenTelemetry."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.metrics import Counter as _Counter
    from opentelemetry.metrics import Histogram as _Histogram
    from opentelemetry.metrics import Meter as _Meter
else:  # pragma: no cover - runtime fallback for optional dependency
    _Counter = _Histogram = _Meter = object


_logger = logging.getLogger(__name__)


class _MeterState:
    """Lazily initialises OTel metric instruments for the Store client."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latency: _Histogram | None = None
        self._errors: _Counter | None = None
        self._retries: _Counter | None = None
        self._disabled = False

    @property
    def ready(self) -> bool:
        return not self._disabled and self._latency is not None

    def _build_meter(self) -> _Meter | None:
        try:
            from opentelemetry.metrics import get_meter
        except Exception:  # noqa: BLE001 - optional dependency missing
            return None
        try:
            return get_meter("tensorcast.api.store", "1.0.0")
        except Exception:  # noqa: BLE001 - fall back to disabled state
            return None

    def ensure(self) -> bool:
        if self.ready:
            return True
        with self._lock:
            if self.ready:
                return True
            if self._disabled:
                return False
            meter = self._build_meter()
            if meter is None:
                self._disabled = True
                _logger.debug("Store metrics disabled: OpenTelemetry meter unavailable")
                return False
            try:
                self._latency = meter.create_histogram(
                    name="tc_store_operation_latency_seconds",
                    unit="s",
                    description="Latency of TensorCast Store client operations.",
                )
                self._errors = meter.create_counter(
                    name="tc_store_operation_errors_total",
                    unit="1",
                    description="Count of TensorCast Store client operation errors.",
                )
                self._retries = meter.create_counter(
                    name="tc_store_operation_retries_total",
                    unit="1",
                    description="Retry attempts for TensorCast Store client operations.",
                )
            except Exception as exc:  # noqa: BLE001
                self._disabled = True
                self._latency = None
                self._errors = None
                self._retries = None
                _logger.debug("Failed to initialise store OTel metrics", exc_info=exc)
                return False
        return True

    def record_latency(
        self, verb: str, daemon: str, status: str, duration_s: float
    ) -> None:
        if not self.ensure():
            return
        assert self._latency is not None
        try:
            self._latency.record(
                max(duration_s, 0.0),
                attributes={
                    "verb": verb,
                    "daemon": daemon,
                    "status": status,
                },
            )
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to record store latency", exc_info=True)

    def increment_errors(self, verb: str, daemon: str, status: str) -> None:
        if not self.ensure():
            return
        assert self._errors is not None
        try:
            self._errors.add(
                1,
                attributes={
                    "verb": verb,
                    "daemon": daemon,
                    "status": status,
                },
            )
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment store error counter", exc_info=True)

    def increment_retries(self, verb: str, daemon: str, status: str) -> None:
        if not self.ensure():
            return
        assert self._retries is not None
        try:
            self._retries.add(
                1,
                attributes={
                    "verb": verb,
                    "daemon": daemon,
                    "status": status,
                },
            )
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment store retry counter", exc_info=True)


_METRICS = _MeterState()


def observe_latency(verb: str, daemon: str, status: str, duration_s: float) -> None:
    _METRICS.record_latency(verb, daemon, status, duration_s)


def increment_error(verb: str, daemon: str, status: str) -> None:
    _METRICS.increment_errors(verb, daemon, status)


def increment_retry(verb: str, daemon: str, status: str) -> None:
    _METRICS.increment_retries(verb, daemon, status)
