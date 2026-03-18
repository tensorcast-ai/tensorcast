#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorcast import runtime, startup
from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.paths import session_paths
from tensorcast.cli_utils.process import read_json_default


@pytest.fixture(autouse=True)
def _isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))
    monkeypatch.setattr(startup, "discover_daemon_config", lambda: None)
    monkeypatch.setattr(startup, "wait_for_daemon", lambda *_args, **_kwargs: False)


def test_auto_mode_connects_existing_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    daemon_address = "127.0.0.1:61001"
    dummy_client = object()

    monkeypatch.setattr(
        startup.runtime,
        "status",
        lambda _session_id=None: SimpleNamespace(daemon_address=daemon_address),
    )
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: True)
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: dummy_client
    )

    ctx = startup.init(mode="auto")
    try:
        assert ctx.is_owner is False
        assert ctx.address == daemon_address
        assert ctx.client is dummy_client
    finally:
        startup.shutdown()


def test_auto_mode_multiprocess_singleflight_shares_one_daemon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mp_ctx = multiprocessing.get_context("fork")
    daemon_address = "127.0.0.1:61011"
    global_store_address = "10.0.0.1:50051"
    start_count = mp_ctx.Value("i", 0)
    daemon_ready = mp_ctx.Event()
    release_workers = mp_ctx.Event()
    results = mp_ctx.Queue()
    start_kwargs = mp_ctx.Queue()

    def _fake_status(_session_id=None):
        if not daemon_ready.is_set():
            return None
        return runtime.RuntimeSession(
            session_id="sess-multiprocess",
            daemon_pid=4242,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:61012",
            logs_dir=session_paths("sess-multiprocess").logs,
            started_at=time.time(),
            owner=True,
        )

    def _fake_start(**kwargs):
        with start_count.get_lock():
            start_count.value += 1
        start_kwargs.put(
            {
                "global_store_mode": kwargs.get("global_store_mode"),
                "global_store_address": kwargs.get("global_store_address"),
            }
        )
        time.sleep(0.2)
        daemon_ready.set()
        return runtime.RuntimeSession(
            session_id="sess-multiprocess",
            daemon_pid=4242,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:61012",
            logs_dir=session_paths("sess-multiprocess").logs,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "status", _fake_status)
    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)
    monkeypatch.setattr(
        startup,
        "ping_daemon",
        lambda addr: bool(daemon_ready.is_set() and addr == daemon_address),
    )
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )

    def _worker() -> None:
        try:
            ctx = startup.init(
                mode="auto",
                global_store_mode="connect",
                global_store_address=global_store_address,
                show_daemon_logs=False,
            )
            results.put(
                {
                    "address": ctx.address,
                    "is_owner": ctx.is_owner,
                    "session_id": ctx.session_id,
                }
            )
            release_workers.wait(timeout=5.0)
            startup.shutdown()
        except Exception as exc:  # noqa: BLE001
            results.put({"error": type(exc).__name__, "message": str(exc)})

    procs = [mp_ctx.Process(target=_worker) for _ in range(2)]
    for proc in procs:
        proc.start()

    collected = [results.get(timeout=10.0) for _ in range(2)]
    release_workers.set()

    for proc in procs:
        proc.join(timeout=10.0)
        assert proc.exitcode == 0

    assert all("error" not in item for item in collected)
    assert start_count.value == 1
    assert [item["address"] for item in collected] == [daemon_address, daemon_address]
    assert sorted(item["is_owner"] for item in collected) == [False, True]
    leader_start = start_kwargs.get(timeout=2.0)
    assert leader_start["global_store_mode"] == "connect"
    assert leader_start["global_store_address"] == global_store_address


def test_auto_mode_creates_daemon_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, object] = {}
    daemon_address = "127.0.0.1:61002"

    def _fake_start(**kwargs):
        started.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-auto",
            daemon_pid=4242,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:61003",
            logs_dir=session_paths("sess-auto").logs,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: True)
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        assert ctx.session_id == "sess-auto"
        assert started["session_id"] is None
        assert started["register_current"] is True
        assert started["ephemeral"] is False
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "LISTENING"
        assert auto_state["session_id"] == "sess-auto"
        assert auto_state["address"] == daemon_address
    finally:
        startup.shutdown()


def test_auto_mode_recovers_from_stale_starting_owner_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_address = "127.0.0.1:64002"
    started: dict[str, object] = {}

    def _fake_start(**kwargs):
        started.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-starting-recovered",
            daemon_pid=5454,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:64003",
            logs_dir=session_paths("sess-starting-recovered").logs,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(
        startup, "ping_daemon", lambda addr: bool(addr == daemon_address)
    )
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)

    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "STARTING",
            "epoch": 3,
            "owner_pid": 0,
            "owner_fingerprint": {},
            "config_hash": "stale-config-hash",
            "session_id": "sess-dead",
            "started_at": time.time(),
            "address": "",
            "logs_dir": "/tmp/daemon-logs",
            "error_code": "",
            "error_message": "",
        },
    )

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        assert started["session_id"] is None
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "LISTENING"
        assert auto_state["session_id"] == "sess-starting-recovered"
        assert auto_state["address"] == daemon_address
    finally:
        startup.shutdown()


def test_auto_mode_rejects_config_mismatch_when_owner_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: False)
    monkeypatch.setattr(
        startup.runtime,
        "start",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )

    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "STARTING",
            "epoch": 7,
            "owner_pid": os.getpid(),
            "owner_fingerprint": startup.instance_fingerprint(),
            "config_hash": "different-config-hash",
            "session_id": "sess-live",
            "started_at": time.time(),
            "address": "",
            "logs_dir": "/tmp/live-logs",
            "error_code": "",
            "error_message": "",
        },
    )

    with pytest.raises(RuntimeError, match="AUTO_CONFIG_MISMATCH"):
        startup.init(mode="auto")


def test_auto_mode_records_failed_state_on_start_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: False)
    monkeypatch.setattr(
        startup.runtime,
        "start",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("start boom")),
    )

    with pytest.raises(RuntimeError, match="start boom"):
        startup.init(mode="auto")

    auto_state = read_json_default(startup._auto_state_path(), {})
    assert auto_state["status"] == "FAILED"
    assert auto_state["error_code"] == "AUTO_START_FAILED"
    assert "start boom" in auto_state["error_message"]


def test_auto_mode_multiprocess_connect_requires_reachable_global_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mp_ctx = multiprocessing.get_context("fork")
    global_store_address = "10.0.0.1:50051"
    start_count = mp_ctx.Value("i", 0)
    results = mp_ctx.Queue()

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: False)

    def _fake_start(**_kwargs):
        with start_count.get_lock():
            start_count.value += 1
        raise ServiceError(
            "Global Store connect mode requires a reachable address, "
            f"got ['{global_store_address}']"
        )

    monkeypatch.setattr(startup.runtime, "start", _fake_start)

    def _worker() -> None:
        try:
            startup.init(
                mode="auto",
                global_store_mode="connect",
                global_store_address=global_store_address,
                show_daemon_logs=False,
            )
            results.put({"error": "unexpected-success"})
        except Exception as exc:  # noqa: BLE001
            results.put({"type": type(exc).__name__, "message": str(exc)})

    procs = [mp_ctx.Process(target=_worker) for _ in range(2)]
    for proc in procs:
        proc.start()

    collected = [results.get(timeout=10.0) for _ in range(2)]

    for proc in procs:
        proc.join(timeout=10.0)
        assert proc.exitcode == 0

    assert start_count.value == 1
    assert all(item.get("error") != "unexpected-success" for item in collected)
    assert any(
        "Global Store connect mode requires a reachable address" in item["message"]
        for item in collected
    )
    assert any("AUTO_START_FAILED" in item["message"] for item in collected)


def test_auto_mode_start_reports_existing_local_global_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup, "ping_daemon", lambda _address: False)
    monkeypatch.setattr(
        startup.runtime,
        "start",
        lambda **_kwargs: (_ for _ in ()).throw(
            ServiceError(
                "A local Global Store is already running for session gs-existing "
                "at 127.0.0.1:50051. Stop it before using global_store_mode='start'."
            )
        ),
    )

    with pytest.raises(ServiceError, match="A local Global Store is already running"):
        startup.init(mode="auto", global_store_mode="start")

    auto_state = read_json_default(startup._auto_state_path(), {})
    assert auto_state["status"] == "FAILED"
    assert auto_state["error_code"] == "AUTO_START_FAILED"
    assert "A local Global Store is already running" in auto_state["error_message"]


def test_auto_mode_recovers_from_stale_ready_owner_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_address = "127.0.0.1:62002"
    started: dict[str, object] = {}

    def _fake_start(**kwargs):
        started.update(kwargs)
        return runtime.RuntimeSession(
            session_id="sess-recovered",
            daemon_pid=5252,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:62003",
            logs_dir=session_paths("sess-recovered").logs,
            started_at=time.time(),
            owner=True,
        )

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(startup.runtime, "start", _fake_start)
    monkeypatch.setattr(
        startup,
        "ping_daemon",
        lambda addr: bool(addr == daemon_address),
    )
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)
    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "READY",
            "epoch": 9,
            "owner_pid": 99999999,
            "owner_fingerprint": {},
            "config_hash": startup._compute_auto_config_hash(
                daemon_config_path=None,
                global_store_mode="none",
                global_store_address=None,
                global_store_config_path=None,
                cluster_id=None,
                allow_gs_fallback=False,
                session_id=None,
                port_config=None,
            ),
            "session_id": "sess-stale",
            "address": "127.0.0.1:50052",
            "logs_dir": "/tmp/stale",
            "started_at": time.time() - 30,
        },
    )

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        assert started["session_id"] is None
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "LISTENING"
        assert auto_state["session_id"] == "sess-recovered"
        assert auto_state["address"] == daemon_address
    finally:
        startup.shutdown()


def test_auto_mode_recovers_from_stale_failed_owner_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon_address = "127.0.0.1:63002"

    monkeypatch.setattr(startup.runtime, "status", lambda _session_id=None: None)
    monkeypatch.setattr(
        startup,
        "ping_daemon",
        lambda addr: bool(addr == daemon_address),
    )
    monkeypatch.setattr(
        startup.runtime,
        "start",
        lambda **_kwargs: runtime.RuntimeSession(
            session_id="sess-failed-recover",
            daemon_pid=5353,
            daemon_address=daemon_address,
            daemon_p2p_address="127.0.0.1:63003",
            logs_dir=session_paths("sess-failed-recover").logs,
            started_at=time.time(),
            owner=True,
        ),
    )
    monkeypatch.setattr(
        "tensorcast.daemon_ctl.get_daemon_client", lambda _address: object()
    )
    monkeypatch.setattr(startup.runtime, "stop", lambda **_kwargs: None)
    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "FAILED",
            "epoch": 11,
            "owner_pid": 99999998,
            "owner_fingerprint": {},
            "config_hash": startup._compute_auto_config_hash(
                daemon_config_path=None,
                global_store_mode="none",
                global_store_address=None,
                global_store_config_path=None,
                cluster_id=None,
                allow_gs_fallback=False,
                session_id=None,
                port_config=None,
            ),
            "session_id": "sess-failed",
            "address": "127.0.0.1:50052",
            "logs_dir": "/tmp/stale-failed",
            "started_at": time.time() - 30,
            "error_code": "AUTO_START_FAILED",
            "error_message": "old error",
        },
    )

    ctx = startup.init(mode="auto", show_daemon_logs=False)
    try:
        assert ctx.is_owner is True
        assert ctx.address == daemon_address
        auto_state = read_json_default(startup._auto_state_path(), {})
        assert auto_state["status"] == "LISTENING"
        assert auto_state["session_id"] == "sess-failed-recover"
    finally:
        startup.shutdown()


def test_auto_state_promotes_listening_to_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup.atomic_write_json(
        startup._auto_state_path(),
        {
            "schema_version": 1,
            "status": "LISTENING",
            "epoch": 1,
            "owner_pid": os.getpid(),
            "owner_fingerprint": {},
            "session_id": "sess-ready",
            "address": "127.0.0.1:65001",
            "started_at": time.time(),
            "error_code": "",
            "error_message": "",
        },
    )
    monkeypatch.setattr(startup, "wait_for_daemon", lambda *_args, **_kwargs: True)

    startup._promote_auto_state_ready_when_rpc_ready(
        session_id="sess-ready",
        address="127.0.0.1:65001",
    )

    deadline = time.time() + 1.0
    while time.time() < deadline:
        auto_state = read_json_default(startup._auto_state_path(), {})
        if auto_state.get("status") == "READY":
            break
        time.sleep(0.01)

    auto_state = read_json_default(startup._auto_state_path(), {})
    assert auto_state["status"] == "READY"
