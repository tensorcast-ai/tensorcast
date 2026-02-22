#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tensorcast.global_store.maintenance_coordinator import (
    GlobalStoreMaintenanceCoordinator,
)


class _LoggerStub:
    def info(self, *args: Any, **kwargs: Any) -> None:
        return

    def exception(self, *args: Any, **kwargs: Any) -> None:
        return


class _TransportServiceStub:
    def __init__(self) -> None:
        self.expiration_calls: list[int | None] = []

    def cleanup_expired_transports(self, expiration_seconds: int | None = None) -> int:
        self.expiration_calls.append(expiration_seconds)
        return 0


def _make_config(
    *,
    heartbeat_timeout_ms: int,
    cleanup_interval_ms: int,
    optimize_interval_ms: int = 3_600_000,
) -> SimpleNamespace:
    return SimpleNamespace(
        heartbeat_timeout_ms=heartbeat_timeout_ms,
        cleanup_interval_ms=cleanup_interval_ms,
        optimize_interval_ms=optimize_interval_ms,
    )


def _make_coordinator(config: SimpleNamespace) -> GlobalStoreMaintenanceCoordinator:
    return GlobalStoreMaintenanceCoordinator(
        config=config,
        connection=None,
        get_worker_service=lambda: None,
        get_instance_service=lambda: None,
        get_transport_service=lambda: None,
        logger=_LoggerStub(),
    )


def test_initial_delay_and_transport_expiration_fast_profile() -> None:
    cfg = _make_config(heartbeat_timeout_ms=10_000, cleanup_interval_ms=10_000)
    coordinator = _make_coordinator(cfg)
    assert coordinator._initial_maintenance_delay_sec() == 10.0
    assert coordinator._transport_expiration_seconds() == 100


def test_initial_delay_and_transport_expiration_slow_profile() -> None:
    cfg = _make_config(heartbeat_timeout_ms=45_000, cleanup_interval_ms=180_000)
    coordinator = _make_coordinator(cfg)
    assert coordinator._initial_maintenance_delay_sec() == 90.0
    assert coordinator._transport_expiration_seconds() == 1800


def test_maintenance_loop_passes_profile_linked_transport_expiration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config(heartbeat_timeout_ms=10_000, cleanup_interval_ms=10_000)
    transport_service = _TransportServiceStub()
    coordinator = GlobalStoreMaintenanceCoordinator(
        config=cfg,
        connection=None,
        get_worker_service=lambda: None,
        get_instance_service=lambda: None,
        get_transport_service=lambda: transport_service,
        logger=_LoggerStub(),
    )
    monkeypatch.setattr(coordinator, "_run_retention_gc", lambda: None)

    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(float(seconds))
        if len(sleep_calls) >= 2:
            raise RuntimeError("stop_loop")

    monkeypatch.setattr(
        "tensorcast.global_store.maintenance_coordinator.time.sleep", _fake_sleep
    )

    with pytest.raises(RuntimeError, match="stop_loop"):
        coordinator._maintenance_loop()

    assert sleep_calls[0] == 10.0
    assert transport_service.expiration_calls
    assert transport_service.expiration_calls[0] == 100
