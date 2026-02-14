#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import tensorcast.daemon_ctl as daemon_ctl
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def test_import_artifact_from_path_uses_configurable_timeout_and_retries(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TENSORCAST_IMPORT_ARTIFACT_TIMEOUT_SECONDS", "123.5")
    monkeypatch.setenv("TENSORCAST_IMPORT_ARTIFACT_RETRIES", "2")
    daemon_ctl._import_artifact_from_path_timeout_seconds.cache_clear()
    daemon_ctl._import_artifact_from_path_retries.cache_clear()

    ctl = daemon_ctl.DaemonCtl.__new__(daemon_ctl.DaemonCtl)
    ctl.server_address = "127.0.0.1:50052"
    ctl.stub_v2 = SimpleNamespace(ImportArtifactFromPath=object())

    seen: dict[str, object] = {}

    def _fake_unary(method, request, *, timeout, retries, span):
        seen["method"] = method
        seen["request"] = request
        seen["timeout"] = timeout
        seen["retries"] = retries
        seen["span"] = span
        return SimpleNamespace(
            artifact_id="mi2:test:test",
            canonical_index_bytes=b"{}",
            generation=1,
        )

    @contextmanager
    def _fake_span(_name: str):
        yield SimpleNamespace(record_exception=lambda *_args, **_kwargs: None)

    ctl._unary_call = _fake_unary
    ctl._client_span = _fake_span

    response = daemon_ctl.DaemonCtl.import_artifact_from_path_v2(
        ctl,
        path="/tmp/model",
        verify_checksums=False,
    )

    assert response.artifact_id == "mi2:test:test"
    assert seen["timeout"] == 123.5
    assert seen["retries"] == 2

    monkeypatch.setenv("TENSORCAST_IMPORT_ARTIFACT_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("TENSORCAST_IMPORT_ARTIFACT_RETRIES", "0")
    daemon_ctl._import_artifact_from_path_timeout_seconds.cache_clear()
    daemon_ctl._import_artifact_from_path_retries.cache_clear()
    seen.clear()

    daemon_ctl.DaemonCtl.import_artifact_from_path_v2(
        ctl,
        path="/tmp/model",
        verify_checksums=False,
    )

    assert seen["timeout"] is None
    assert seen["retries"] == 0


def test_import_artifact_from_path_stream_uses_configurable_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TENSORCAST_IMPORT_ARTIFACT_TIMEOUT_SECONDS", "45")
    daemon_ctl._import_artifact_from_path_timeout_seconds.cache_clear()

    ctl = daemon_ctl.DaemonCtl.__new__(daemon_ctl.DaemonCtl)
    ctl.server_address = "127.0.0.1:50052"
    seen: dict[str, object] = {}

    final_resp = store_daemon_pb2.ImportArtifactFromPathResponse(
        artifact_id="mi2:test:test",
        canonical_index_bytes=b"{}",
        generation=1,
    )

    def _fake_stream(request, timeout=None):
        seen["request"] = request
        seen["timeout"] = timeout
        event = store_daemon_pb2.ImportArtifactFromPathStreamEvent(
            seq=1,
            phase=store_daemon_pb2.IMPORT_ARTIFACT_PHASE_DONE,
            done=True,
        )
        event.result.CopyFrom(final_resp)
        yield event

    ctl.stub_v2 = SimpleNamespace(ImportArtifactFromPathStream=_fake_stream)

    @contextmanager
    def _fake_span(_name: str):
        yield SimpleNamespace(record_exception=lambda *_args, **_kwargs: None)

    ctl._client_span = _fake_span

    events = list(
        daemon_ctl.DaemonCtl.import_artifact_from_path_stream_v2(
            ctl,
            path="/tmp/model",
            verify_checksums=False,
        )
    )

    assert seen["timeout"] == 45.0
    assert len(events) == 1
    assert events[0].done is True
    assert events[0].result.artifact_id == "mi2:test:test"
