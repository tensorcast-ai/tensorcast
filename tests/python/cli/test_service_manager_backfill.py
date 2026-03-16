#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import io
import json
from pathlib import Path

from tensorcast.cli_utils import service_manager
from tensorcast.cli_utils.paths import runtime_state_path, session_paths
from tensorcast.daemon_runtime_config import dump_daemon_config, load_daemon_config
from tensorcast.proto.config.v1 import daemon_config_pb2


class _FakeProc:
    def __init__(self, pid: int):
        self.pid = pid
        self.args = ["tensorcast_daemon", "--config=cfg"]
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(b"")

    def poll(self):  # noqa: D401
        return None

    def wait(self, timeout=None):  # noqa: D401, ANN001
        return 0


def test_start_service_backfills_ports(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))

    cfg = daemon_config_pb2.DaemonConfig()
    cfg.server.listen.host = "0.0.0.0"
    cfg.server.listen.port = 0
    cfg.server.p2p_listen.host = ""
    cfg.server.p2p_listen.port = 0
    cfg_path = tmp_path / "daemon.yaml"
    dump_daemon_config(cfg, cfg_path)

    # Deterministic ports before backfill
    port_seq = [61000, 61001]
    monkeypatch.setattr(service_manager, "pick_free_tcp_port", lambda: port_seq.pop(0))

    proc_calls: list[_FakeProc] = []

    def _fake_popen(args, **kwargs):  # noqa: ANN001
        proc = _FakeProc(3000 + len(proc_calls))
        proc_calls.append(proc)
        return proc

    class _Listen:
        def __init__(self, host: str, port: int):
            self.host = host
            self.port = port

    class _Cfg:
        def __init__(self):
            self.server = type(
                "_Server",
                (),
                {
                    "listen": _Listen("0.0.0.0", 62000),
                    "p2p_listen": _Listen("127.0.0.1", 62001),
                },
            )()

    fake_cfg = _Cfg()

    monkeypatch.setattr(
        service_manager, "ensure_cpp_daemon_binary", lambda: Path("/bin/true")
    )
    monkeypatch.setattr(service_manager.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        service_manager, "ensure_process_started", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        service_manager, "wait_daemon_listening", lambda host, _port, **_kwargs: host
    )
    monkeypatch.setattr(
        service_manager, "start_log_threads", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        service_manager, "get_daemon_config", lambda *args, **kwargs: fake_cfg
    )
    monkeypatch.setattr(service_manager, "preexec_fate_sharing", lambda: None)

    inst = service_manager.start_service(
        config_path=cfg_path,
        register_current=False,
        publish_meta=True,
        to_console=False,
    )

    assert inst.address is not None
    assert inst.address.endswith(":62000")
    assert inst.p2p_address is not None
    assert inst.p2p_address.endswith(":62001")
    assert len(proc_calls) == 1

    state = json.loads(runtime_state_path().read_text(encoding="utf-8"))
    daemon_state = state.get("daemon", {})
    assert daemon_state.get("address", "").endswith(":62000")
    assert daemon_state.get("p2p_address", "").endswith(":62001")

    inst_paths = session_paths(inst.id)
    session_state = json.loads(
        inst_paths.session_state_json.read_text(encoding="utf-8")
    )
    assert session_state["daemon"]["address"].endswith(":62000")
    assert session_state["daemon"]["p2p_address"].endswith(":62001")


def test_start_service_applies_config_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))

    cfg = daemon_config_pb2.DaemonConfig()
    cfg.server.listen.host = "127.0.0.1"
    cfg.server.listen.port = 0
    cfg.server.p2p_listen.host = "127.0.0.1"
    cfg.server.p2p_listen.port = 0
    cfg_path = tmp_path / "daemon.yaml"
    dump_daemon_config(cfg, cfg_path)

    port_seq = [63000, 63001]
    monkeypatch.setattr(service_manager, "pick_free_tcp_port", lambda: port_seq.pop(0))

    proc_calls: list[_FakeProc] = []

    def _fake_popen(args, **kwargs):  # noqa: ANN001
        proc = _FakeProc(4000 + len(proc_calls))
        proc_calls.append(proc)
        return proc

    monkeypatch.setattr(
        service_manager, "ensure_cpp_daemon_binary", lambda: Path("/bin/true")
    )
    monkeypatch.setattr(service_manager.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        service_manager, "ensure_process_started", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        service_manager, "wait_daemon_listening", lambda host, _port, **_kwargs: host
    )
    monkeypatch.setattr(
        service_manager, "start_log_threads", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        service_manager, "get_daemon_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(service_manager, "preexec_fate_sharing", lambda: None)

    inst = service_manager.start_service(
        config_path=cfg_path,
        register_current=False,
        publish_meta=True,
        to_console=False,
        p2p_listen_port=64001,
        config_overrides=[
            "engine.memory_tiers.enable_preemptible=true",
            "engine.memory_tiers.stable_bytes=4GB",
            "engine.memory_tiers.preemptible_low_watermark_ratio=0.3",
        ],
    )

    inst_paths = session_paths(inst.id)
    effective_cfg = load_daemon_config(inst_paths.effective_config_path)
    assert effective_cfg.engine.HasField("memory_tiers")
    tiers = effective_cfg.engine.memory_tiers
    assert tiers.enable_preemptible is True
    assert tiers.stable_bytes == 4 * 1024**3
    assert tiers.preemptible_low_watermark_ratio == 0.3
    assert effective_cfg.server.p2p_listen.port == 64001


def test_start_service_passes_launcher_envs_to_daemon_process(monkeypatch, tmp_path):
    monkeypatch.setenv("TENSORCAST_HOME", str(tmp_path))

    cfg = daemon_config_pb2.DaemonConfig()
    cfg.server.listen.host = "127.0.0.1"
    cfg.server.listen.port = 0
    cfg.server.p2p_listen.host = "127.0.0.1"
    cfg.server.p2p_listen.port = 0
    cfg.envs["NCCL_DEBUG"] = "INFO"
    cfg.envs["LD_LIBRARY_PATH"] = "/data/cuda/compat"
    cfg.envs["TENSORCAST_INSTANCE"] = "user-value"
    cfg_path = tmp_path / "daemon.yaml"
    dump_daemon_config(cfg, cfg_path)

    port_seq = [65000, 65001]
    monkeypatch.setattr(service_manager, "pick_free_tcp_port", lambda: port_seq.pop(0))

    captured_build_env: dict[str, object] = {}
    proc_envs: list[dict[str, str]] = []

    def _fake_build_env(base_env, extra_env):  # noqa: ANN001
        captured_build_env["base_env"] = dict(base_env)
        captured_build_env["extra_env"] = dict(extra_env)
        env = dict(extra_env)
        env["LD_LIBRARY_PATH"] = "/merged/lib"
        env["TENSORCAST_INSTANCE"] = "ignored"
        return env

    def _fake_popen(args, **kwargs):  # noqa: ANN001
        proc_envs.append(dict(kwargs["env"]))
        return _FakeProc(5000)

    monkeypatch.setattr(service_manager, "build_daemon_process_env", _fake_build_env)
    monkeypatch.setattr(
        service_manager, "ensure_cpp_daemon_binary", lambda: Path("/bin/true")
    )
    monkeypatch.setattr(service_manager.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        service_manager, "ensure_process_started", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        service_manager, "wait_daemon_listening", lambda host, _port, **_kwargs: host
    )
    monkeypatch.setattr(
        service_manager, "start_log_threads", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(
        service_manager, "get_daemon_config", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(service_manager, "preexec_fate_sharing", lambda: None)

    inst = service_manager.start_service(
        config_path=cfg_path,
        register_current=False,
        publish_meta=True,
        to_console=False,
    )

    assert captured_build_env["extra_env"] == {
        "LD_LIBRARY_PATH": "/data/cuda/compat",
        "NCCL_DEBUG": "INFO",
        "TENSORCAST_INSTANCE": "user-value",
    }
    assert len(proc_envs) == 1
    assert proc_envs[0]["LD_LIBRARY_PATH"] == "/merged/lib"
    assert proc_envs[0]["NCCL_DEBUG"] == "INFO"
    assert proc_envs[0]["TENSORCAST_INSTANCE"] == inst.id
