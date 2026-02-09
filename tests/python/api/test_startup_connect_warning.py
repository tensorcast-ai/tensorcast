#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from pathlib import Path

import pytest

from tensorcast import startup


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: object) -> None:
        self.warnings.append(message % args if args else message)

    def info(self, _message: str, *_args: object) -> None:
        return


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    monkeypatch.setattr(startup, "_current_ctx", None)


def test_connect_mode_warns_global_store_args_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    called: dict[str, object] = {}

    def _fake_connect_context(**kwargs: object) -> object:
        called.update(kwargs)
        return object()

    monkeypatch.setattr(startup, "init_logger", lambda _name: logger)
    monkeypatch.setattr(startup, "_connect_context", _fake_connect_context)

    startup.init(
        mode="connect",
        address="127.0.0.1:50052",
        global_store_mode="connect",
        global_store_address="127.0.0.1:50051",
    )

    assert called["target_address"] == "127.0.0.1:50052"
    assert len(logger.warnings) == 1
    assert "init(mode='connect')" in logger.warnings[0]
    assert "are ignored" in logger.warnings[0]
    assert "creating/starting daemon" in logger.warnings[0]


def test_connect_mode_without_global_store_args_has_no_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()

    monkeypatch.setattr(startup, "init_logger", lambda _name: logger)
    monkeypatch.setattr(startup, "_connect_context", lambda **_kwargs: object())

    startup.init(
        mode="connect",
        address="127.0.0.1:50052",
    )

    assert logger.warnings == []
