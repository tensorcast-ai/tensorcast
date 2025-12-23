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
        self._cache_hits: _Counter | None = None
        self._cache_misses: _Counter | None = None
        self._cache_evictions: _Counter | None = None
        self._cache_invalidations: _Counter | None = None
        self._batch_hits: _Counter | None = None
        self._batch_coalesced: _Counter | None = None
        self._batch_latency: _Histogram | None = None
        self._prefetch_events: _Counter | None = None
        self._region_backed_fallbacks: _Counter | None = None
        self._region_backed_verification_skipped: _Counter | None = None
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
                self._cache_hits = meter.create_counter(
                    name="tc_store_artifact_cache_hits_total",
                    unit="1",
                    description="Artifact cache hits in the TensorCast Store client.",
                )
                self._cache_misses = meter.create_counter(
                    name="tc_store_artifact_cache_misses_total",
                    unit="1",
                    description="Artifact cache misses in the TensorCast Store client.",
                )
                self._cache_evictions = meter.create_counter(
                    name="tc_store_artifact_cache_evictions_total",
                    unit="1",
                    description="Artifact cache evictions in the TensorCast Store client.",
                )
                self._cache_invalidations = meter.create_counter(
                    name="tc_store_artifact_cache_invalidations_total",
                    unit="1",
                    description="Artifact cache invalidations in the TensorCast Store client.",
                )
                self._batch_hits = meter.create_counter(
                    name="tc_store_batch_hits_total",
                    unit="1",
                    description="Count of batcher-served tensor fetches.",
                )
                self._batch_coalesced = meter.create_counter(
                    name="tc_store_batch_coalesced_total",
                    unit="1",
                    description="Count of coalesced batch RPCs.",
                )
                self._batch_latency = meter.create_histogram(
                    name="tc_store_batch_window_seconds",
                    unit="s",
                    description="Observed batch coalescing window durations.",
                )
                self._prefetch_events = meter.create_counter(
                    name="tc_store_prefetch_events_total",
                    unit="1",
                    description="Prefetch ticket lifecycle events.",
                )
                self._region_backed_fallbacks = meter.create_counter(
                    name="tc_store_region_backed_fallback_total",
                    unit="1",
                    description="Count of region-backed get_into fallbacks.",
                )
                self._region_backed_verification_skipped = meter.create_counter(
                    name="tc_store_region_backed_verification_skipped_total",
                    unit="1",
                    description="Count of region-backed get_into verification skips.",
                )
            except Exception as exc:  # noqa: BLE001
                self._disabled = True
                self._latency = None
                self._errors = None
                self._retries = None
                self._cache_hits = None
                self._cache_misses = None
                self._cache_evictions = None
                self._cache_invalidations = None
                self._batch_hits = None
                self._batch_coalesced = None
                self._batch_latency = None
                self._prefetch_events = None
                self._region_backed_fallbacks = None
                self._region_backed_verification_skipped = None
                _logger.debug("Failed to initialise store OTel metrics", exc_info=exc)
                return False
        return True

    def record_latency(
        self,
        verb: str,
        daemon: str,
        status: str,
        duration_s: float,
        *,
        source: str | None = None,
        selection: str | None = None,
    ) -> None:
        if not self.ensure():
            return
        assert self._latency is not None
        attributes = {
            "verb": verb,
            "daemon": daemon,
            "status": status,
        }
        if source:
            attributes["source"] = source
        if selection:
            attributes["selection"] = selection
        try:
            self._latency.record(max(duration_s, 0.0), attributes=attributes)
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to record store latency", exc_info=True)

    def increment_errors(
        self,
        verb: str,
        daemon: str,
        status: str,
        *,
        source: str | None = None,
        selection: str | None = None,
    ) -> None:
        if not self.ensure():
            return
        assert self._errors is not None
        attributes = {
            "verb": verb,
            "daemon": daemon,
            "status": status,
        }
        if source:
            attributes["source"] = source
        if selection:
            attributes["selection"] = selection
        try:
            self._errors.add(1, attributes=attributes)
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment store error counter", exc_info=True)

    def increment_retries(
        self,
        verb: str,
        daemon: str,
        status: str,
        *,
        source: str | None = None,
        selection: str | None = None,
    ) -> None:
        if not self.ensure():
            return
        assert self._retries is not None
        attributes = {
            "verb": verb,
            "daemon": daemon,
            "status": status,
        }
        if source:
            attributes["source"] = source
        if selection:
            attributes["selection"] = selection
        try:
            self._retries.add(1, attributes=attributes)
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment store retry counter", exc_info=True)

    def increment_cache_hit(self, daemon: str) -> None:
        if not self.ensure():
            return
        if self._cache_hits is None:
            return
        try:
            self._cache_hits.add(1, attributes={"daemon": daemon})
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment cache hit counter", exc_info=True)

    def increment_cache_miss(self, daemon: str) -> None:
        if not self.ensure():
            return
        if self._cache_misses is None:
            return
        try:
            self._cache_misses.add(1, attributes={"daemon": daemon})
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment cache miss counter", exc_info=True)

    def increment_cache_eviction(self, daemon: str, *, reason: str | None) -> None:
        if not self.ensure():
            return
        if self._cache_evictions is None:
            return
        attributes = {"daemon": daemon}
        if reason:
            attributes["reason"] = reason
        try:
            self._cache_evictions.add(1, attributes=attributes)
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment cache eviction counter", exc_info=True)

    def increment_cache_invalidation(self, daemon: str, *, reason: str | None) -> None:
        if not self.ensure():
            return
        if self._cache_invalidations is None:
            return
        attributes = {"daemon": daemon}
        if reason:
            attributes["reason"] = reason
        try:
            self._cache_invalidations.add(1, attributes=attributes)
        except Exception:  # noqa: BLE001
            _logger.debug(
                "Failed to increment cache invalidation counter", exc_info=True
            )

    def increment_batch_hit(self, daemon: str, *, coalesced: bool) -> None:
        if not self.ensure():
            return
        if self._batch_hits is None:
            return
        attrs = {"daemon": daemon, "coalesced": coalesced}
        try:
            self._batch_hits.add(1, attributes=attrs)
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment batch hit counter", exc_info=True)

    def increment_batch_coalesced(self, daemon: str) -> None:
        if not self.ensure():
            return
        if self._batch_coalesced is None:
            return
        try:
            self._batch_coalesced.add(1, attributes={"daemon": daemon})
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to increment batch coalesced counter", exc_info=True)

    def record_batch_latency(self, daemon: str, duration_s: float) -> None:
        if not self.ensure():
            return
        if self._batch_latency is None:
            return
        try:
            self._batch_latency.record(
                max(duration_s, 0.0), attributes={"daemon": daemon}
            )
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to record batch latency", exc_info=True)

    def record_prefetch_event(self, daemon: str, status: str) -> None:
        if not self.ensure():
            return
        if self._prefetch_events is None:
            return
        try:
            self._prefetch_events.add(
                1, attributes={"daemon": daemon, "status": status}
            )
        except Exception:  # noqa: BLE001
            _logger.debug("Failed to record prefetch event", exc_info=True)

    def record_region_backed_fallback(self, daemon: str, reason: str) -> None:
        if not self.ensure():
            return
        if self._region_backed_fallbacks is None:
            return
        try:
            self._region_backed_fallbacks.add(
                1, attributes={"daemon": daemon, "reason": reason}
            )
        except Exception:  # noqa: BLE001
            _logger.debug(
                "Failed to record region-backed fallback event", exc_info=True
            )

    def record_region_backed_verification_skipped(self, daemon: str) -> None:
        if not self.ensure():
            return
        if self._region_backed_verification_skipped is None:
            return
        try:
            self._region_backed_verification_skipped.add(
                1, attributes={"daemon": daemon}
            )
        except Exception:  # noqa: BLE001
            _logger.debug(
                "Failed to record region-backed verification skipped", exc_info=True
            )


_METRICS = _MeterState()


def observe_latency(
    verb: str,
    daemon: str,
    status: str,
    duration_s: float,
    *,
    source: str | None = None,
    selection: str | None = None,
) -> None:
    _METRICS.record_latency(
        verb, daemon, status, duration_s, source=source, selection=selection
    )


def increment_error(
    verb: str,
    daemon: str,
    status: str,
    *,
    source: str | None = None,
    selection: str | None = None,
) -> None:
    _METRICS.increment_errors(verb, daemon, status, source=source, selection=selection)


def increment_retry(
    verb: str,
    daemon: str,
    status: str,
    *,
    source: str | None = None,
    selection: str | None = None,
) -> None:
    _METRICS.increment_retries(verb, daemon, status, source=source, selection=selection)


def increment_artifact_cache_hit(daemon: str) -> None:
    _METRICS.increment_cache_hit(daemon)


def increment_artifact_cache_miss(daemon: str) -> None:
    _METRICS.increment_cache_miss(daemon)


def increment_artifact_cache_eviction(daemon: str, *, reason: str | None) -> None:
    _METRICS.increment_cache_eviction(daemon, reason=reason)


def increment_artifact_cache_invalidation(daemon: str, *, reason: str | None) -> None:
    _METRICS.increment_cache_invalidation(daemon, reason=reason)


def increment_batch_hit(daemon: str, *, coalesced: bool) -> None:
    _METRICS.increment_batch_hit(daemon, coalesced=coalesced)


def increment_batch_coalesced(daemon: str) -> None:
    _METRICS.increment_batch_coalesced(daemon)


def record_batch_latency(daemon: str, duration_s: float) -> None:
    _METRICS.record_batch_latency(daemon, duration_s)


def record_prefetch_event(daemon: str, status: str) -> None:
    _METRICS.record_prefetch_event(daemon, status)


def record_region_backed_fallback(daemon: str, reason: str) -> None:
    _METRICS.record_region_backed_fallback(daemon, reason)


def record_region_backed_verification_skipped(daemon: str) -> None:
    _METRICS.record_region_backed_verification_skipped(daemon)
