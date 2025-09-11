#  Copyright (c) 2025, TensorCast Team.

"""
Service management utilities for StoreDaemon (Linux-only).

Implements the launch/connect best practices with per-session directories,
fate sharing, and unified config-only launching.

Terminology (user-facing):
- "daemon session": a locally managed StoreDaemon run (formerly "instance").
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import click
import grpc
import psutil

from tensorcast.daemon_runtime_config import dump_daemon_config, load_daemon_config
from tensorcast.logger import init_logger
from tensorcast.proto.daemon.v1 import store_daemon_pb2, store_daemon_pb2_grpc

logger = init_logger(__name__)


class ServiceError(Exception):
    pass


# Paths & Instances
HOME_DIRNAME = ".tensorcast"


def _home_dir() -> Path:
    return Path.home() / HOME_DIRNAME


def _instances_dir() -> Path:
    p = _home_dir() / "instances"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _current_instance_path() -> Path:
    return _home_dir() / "current_session"


def get_current_instance_id() -> str | None:
    p = _current_instance_path()
    if not p.exists():
        # Backward-compat: read legacy file name if present
        legacy = _home_dir() / "current_instance"
        if not legacy.exists():
            return None
        try:
            with open(legacy, "r", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = f.read()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data.strip() or None
        except Exception:
            return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = f.read()
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data.strip() or None
    except Exception:
        return None


def _set_current_instance_id(instance_id: str) -> None:
    p = _current_instance_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(instance_id)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp, p)
    with contextlib.suppress(Exception):
        os.chmod(p, 0o600)


def _clear_current_instance_if_matches(iid: str) -> None:
    p = _current_instance_path()
    if not p.exists():
        return
    try:
        current = p.read_text(encoding="utf-8").strip()
        if current == iid:
            with contextlib.suppress(Exception):
                p.unlink()
    except Exception:
        pass


def list_instances() -> list[str]:
    d = _instances_dir()
    return sorted([ch.name for ch in d.iterdir() if ch.is_dir()])


def get_instance_address(instance_id: str | None = None) -> str | None:
    """Return the host:port address recorded in meta.json for a session (default: current)."""
    iid = instance_id or get_current_instance_id()
    if not iid:
        return None
    try:
        meta = _read_json_locked((_instances_dir() / iid / "meta.json"))
        addr = meta.get("address")
        return addr if isinstance(addr, str) else None
    except Exception:
        return None


def _now_ts() -> float:
    return time.time()


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _rand4() -> str:
    import random

    return f"{random.randrange(0, 65536):04x}"


def _gen_instance_id() -> str:
    return f"{_now_id()}-{_rand4()}"


@dataclass(frozen=True)
class Instance:
    id: str
    root: Path
    session: Path
    logs: Path
    pids_json: Path
    meta_json: Path
    # Derived connection info (set by start_service)
    address: str | None = None
    p2p_address: str | None = None


def _instance_paths(instance_id: str | None = None) -> Instance:
    iid = instance_id or _gen_instance_id()
    root = _instances_dir() / iid
    session = root / "session"
    logs = root / "logs"
    session.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return Instance(
        id=iid,
        root=root,
        session=session,
        logs=logs,
        pids_json=root / "pids.json",
        meta_json=root / "meta.json",
        address=None,
        p2p_address=None,
    )


# Config discovery helpers
def discover_default_config_path() -> Path | None:
    """Discover a default StoreDaemon config path for convenience starts.

    Order:
    1) $TENSORCAST_DAEMON_CONFIG if set and exists
    2) ~/.tensorcast/store_daemon_config.yaml
       ~/.tensorcast/store_daemon.yaml
    3) examples/config/store_daemon_config.yaml (in repo checkout)
    """
    # 1) explicit env var
    env = os.environ.get("TENSORCAST_DAEMON_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    # 2) home defaults
    home = Path.home() / HOME_DIRNAME
    for name in ("store_daemon_config.yaml", "store_daemon.yaml"):
        candidate = home / name
        if candidate.exists():
            return candidate

    # 3) repo example
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "examples" / "config" / "store_daemon_config.yaml"
    if example.exists():
        return example
    return None


# File I/O helpers
def _write_json_locked(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_json_locked(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data


def _open_log_binary(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "ab", buffering=0)


def _resolve_connect_host(listen_host: str | None) -> str:
    """Choose a host suitable for client connections.

    If the daemon binds to 0.0.0.0/:: or empty, prefer 127.0.0.1 for local connects.
    """
    if not listen_host:
        return "127.0.0.1"
    s = str(listen_host).strip().lower()
    if s in {"0.0.0.0", "::", "[::]", "*"}:
        return "127.0.0.1"
    return listen_host


def _pick_free_tcp_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


def _tcp_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _wait_daemon_ready(host: str, port: int, timeout: float = 20.0) -> bool:
    """Wait until daemon responds to gRPC or at least the TCP port is open.

    Tries gRPC GetServerConfig; if it fails (e.g., TLS), falls back to TCP probe.
    """
    deadline = time.time() + timeout
    addr = f"{host}:{port}"
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            channel = grpc.insecure_channel(addr)
            stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
            # A small timeout per attempt to keep loop snappy
            stub.GetServerConfig(store_daemon_pb2.GetServerConfigRequest(), timeout=0.8)
            return True
        except Exception as e:
            last_err = e
            # fallback: plain TCP open indicates server is up (TLS/creds may block gRPC)
            if _tcp_port_open(host, port, timeout=0.4):
                return True
            time.sleep(0.2)
    if last_err:
        logger.debug("Daemon readiness wait exceeded: %s", last_err)
    return False


# Linux fate sharing
libc = ctypes.CDLL("libc.so.6", use_errno=True)
PR_SET_PDEATHSIG = 1


def _set_pdeathsig(sig: int = signal.SIGKILL) -> None:
    res = libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    if res != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_PDEATHSIG) failed: errno={err}")


def _preexec_fate_sharing(
    ignore_sigint: bool = True, pdeathsig: int = signal.SIGKILL
) -> None:
    os.setsid()
    _set_pdeathsig(pdeathsig)
    if ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    from contextlib import suppress

    with suppress(Exception):
        signal.pthread_sigmask(
            signal.SIG_BLOCK, [signal.SIGTTOU, signal.SIGTTIN, signal.SIGTSTP]
        )


def _pump(src, sinks: Iterable[Any]) -> None:
    import sys as _sys

    for line in iter(src.readline, b""):
        for s in sinks:
            try:
                if s in (_sys.stdout, _sys.stderr):
                    s.write(line.decode("utf-8", errors="replace"))
                    s.flush()
                else:
                    s.write(line)
                    s.flush()
            except Exception:
                pass


# Public API
def start_service(
    *,
    config_path: Path,
    instance_id: str | None = None,
    blocking: bool = False,
    to_console: bool = True,
    wait: bool = False,
    timeout: float = 15.0,
    register_current: bool = True,
    publish_meta: bool = True,
    restrict_to_localhost: bool = False,
) -> Instance:
    """Start the C++ StoreDaemon.

    - If config listen.port is 0 or missing, pick a free TCP port and pass derived config to daemon.
    - In non-blocking mode, when wait=True, wait up to `timeout` seconds for readiness before returning.
    """
    cfg = load_daemon_config(config_path)
    # Ensure host present; prefer explicit host from config
    # For private launches, force loopback visibility only
    if restrict_to_localhost:
        try:
            if cfg.server.listen.host != "127.0.0.1":
                cfg.server.listen.host = "127.0.0.1"
            if cfg.server.p2p_listen.host and cfg.server.p2p_listen.host != "127.0.0.1":
                cfg.server.p2p_listen.host = "127.0.0.1"
        except Exception:
            # Best-effort; continue
            pass

    host = cfg.server.listen.host or "127.0.0.1"
    port = int(cfg.server.listen.port or 0)

    # Auto-assign port when unspecified/zero, and materialize an effective config for the daemon
    cfg_modified = False
    if port <= 0:
        port = _pick_free_tcp_port()
        cfg.server.listen.port = port
        cfg_modified = True
    if not cfg.server.listen.host:
        cfg.server.listen.host = host
        cfg_modified = True

    # Handle P2P listen: assign port if unset/zero, and fill host if empty.
    # Default to listen.host when p2p host missing.
    p2p_host = cfg.server.p2p_listen.host or host
    p2p_port = int(cfg.server.p2p_listen.port or 0)
    if p2p_port <= 0:
        p2p_port = _pick_free_tcp_port()
        cfg.server.p2p_listen.port = p2p_port
        cfg_modified = True
    if not cfg.server.p2p_listen.host:
        cfg.server.p2p_listen.host = p2p_host
        cfg_modified = True

    inst = _instance_paths(instance_id)
    # Write effective config under the session directory to ensure C++ reads the final value
    effective_cfg_path = inst.session / "effective_daemon_config.yaml"
    try:
        if cfg_modified:
            dump_daemon_config(cfg, effective_cfg_path)
            daemon_cfg_arg = f"--config={str(effective_cfg_path)}"
            meta_cfg_path = str(effective_cfg_path)
        else:
            daemon_cfg_arg = f"--config={str(Path(config_path).expanduser().resolve())}"
            meta_cfg_path = str(Path(config_path).expanduser().resolve())
    except Exception as e:
        raise ServiceError(f"Failed to write effective config: {e}") from e

    bin_path = _ensure_cpp_daemon_binary()
    args = [str(bin_path), daemon_cfg_arg]

    so_path = inst.logs / "daemon.out"
    se_path = inst.logs / "daemon.err"
    so = _open_log_binary(so_path)
    se = _open_log_binary(se_path)

    env = {**os.environ, "TENSORCAST_INSTANCE": inst.id}

    if blocking:
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=inst.session,
            env=env,
            preexec_fn=lambda: _preexec_fate_sharing(),
            close_fds=True,
        )
        _register_process(inst, role="daemon", proc=proc, so=so_path, se=se_path)
        connect_host = _resolve_connect_host(host)
        # Publish metadata only if requested
        if publish_meta:
            _write_meta(
                inst,
                address=f"{connect_host}:{port}",
                config_path=meta_cfg_path,
                p2p_address=f"{_resolve_connect_host(p2p_host)}:{p2p_port}",
            )
        if register_current:
            _set_current_instance_id(inst.id)
        # Attach address info to instance
        object.__setattr__(inst, "address", f"{connect_host}:{port}")
        object.__setattr__(
            inst, "p2p_address", f"{_resolve_connect_host(p2p_host)}:{p2p_port}"
        )
        sinks_out: list[Any] = [so]
        sinks_err: list[Any] = [se]
        if to_console:
            sinks_out.append(sys.stdout)
            sinks_err.append(sys.stderr)
        t1 = threading.Thread(target=_pump, args=(proc.stdout, sinks_out), daemon=True)
        t2 = threading.Thread(target=_pump, args=(proc.stderr, sinks_err), daemon=True)
        t1.start()
        t2.start()
        ret = proc.wait()
    click.echo(f"daemon exited with code {ret}")
    return inst

    # Non-blocking
    proc = subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=so,
        stderr=se,
        cwd=inst.session,
        env=env,
        preexec_fn=lambda: _preexec_fate_sharing(),
        close_fds=True,
    )
    _register_process(inst, role="daemon", proc=proc, so=so_path, se=se_path)
    connect_host = _resolve_connect_host(host)
    if publish_meta:
        _write_meta(
            inst,
            address=f"{connect_host}:{port}",
            config_path=meta_cfg_path,
            p2p_address=f"{_resolve_connect_host(p2p_host)}:{p2p_port}",
        )
    if register_current:
        _set_current_instance_id(inst.id)
    # Attach address info to instance
    object.__setattr__(inst, "address", f"{connect_host}:{port}")
    object.__setattr__(
        inst, "p2p_address", f"{_resolve_connect_host(p2p_host)}:{p2p_port}"
    )

    # Optional readiness wait for non-blocking mode
    if wait:
        ready = _wait_daemon_ready(connect_host, port, timeout=timeout)
        if not ready:
            # best-effort: terminate the group and report failure
            try:
                pgid = os.getpgid(proc.pid)
                if not _kill_gracefully(pgid, grace=2.0):
                    _kill_force(pgid)
            except ProcessLookupError:
                pass
            raise ServiceError(
                f"Daemon failed to become ready within {timeout:.0f}s (address={connect_host}:{port}). "
                f"Check logs at {so_path}"
            )

    click.echo(
        f"StoreDaemon started (session={inst.id}) at {connect_host}:{port}. Logs: {so_path}"
    )
    return inst


def stop_service(
    *, instance_id: str | None = None, grace: float = 10.0, force: bool = False
) -> None:
    iid = instance_id or get_current_instance_id()
    if not iid:
        raise ServiceError("No session id provided and no current session found")
    inst = _instance_paths(iid)
    if not inst.pids_json.exists():
        click.echo(f"No pids.json found for daemon session {iid}")
        return
    data = _read_json_locked(inst.pids_json)
    procs = data.get("processes", [])
    for proc in reversed(procs):
        pid = int(proc.get("pid", 0))
        if pid <= 0:
            continue
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            continue
        if not _kill_gracefully(pgid, grace=grace) or force:
            _kill_force(pgid)
    click.echo(f"Stopped daemon session {iid}")
    _clear_current_instance_if_matches(iid)


def check_service_status(
    *, instance_id: str | None = None, host: str | None = None, port: int | None = None
) -> None:
    iid = instance_id or get_current_instance_id()
    if not iid:
        click.echo("No local daemon session found. Start one with 'tensorcast start'.")
        sys.exit(1)
    inst = _instance_paths(iid)
    try:
        data = _read_json_locked(inst.pids_json)
        pid = int(data.get("processes", [{}])[0].get("pid", 0))
    except Exception:
        pid = 0
    if pid <= 0:
        click.echo(f"Daemon session {iid} has no running process.")
        sys.exit(1)

    try:
        if host is None or port is None:
            derived_host = None
            derived_port: int | None = None
            try:
                meta = _read_json_locked(inst.meta_json)
                addr = meta.get("address")
                if addr and ":" in addr:
                    h, p = addr.split(":", 1)
                    derived_host = h
                    derived_port = int(p)
                else:
                    cfg_path = meta.get("config_path")
                    if cfg_path:
                        cfg = load_daemon_config(cfg_path)
                        if cfg.server.listen.host:
                            derived_host = cfg.server.listen.host
                        if cfg.server.listen.port:
                            derived_port = int(cfg.server.listen.port)
            except Exception:
                pass
            host = host or derived_host or "127.0.0.1"
            port = port or derived_port or 50052

        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = store_daemon_pb2_grpc.StoreDaemonServiceStub(channel)
        response = stub.GetDetailedStatus(
            store_daemon_pb2.GetDetailedStatusRequest(), timeout=5.0
        )
        _display_detailed_status(pid, response)
    except Exception as e:
        click.echo(f"StoreDaemon is running with PID {pid}")
        click.echo(f"  (Unable to connect to gRPC service: {e})")
        info = get_process_info(pid)
        if info and len(info) > 1:
            if "started" in info:
                click.echo(f"  Started: {info['started']}")
            if "cpu_percent" in info:
                click.echo(f"  CPU: {info['cpu_percent']}%")
            if "memory_mb" in info:
                click.echo(f"  Memory: {info['memory_mb']:.1f} MB")
    sys.exit(0)


def _ensure_cpp_daemon_binary() -> Path:
    env_path = os.environ.get("TENSORCAST_DAEMON_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists() and os.access(p, os.X_OK):
            return p
        raise ServiceError(f"TENSORCAST_DAEMON_BIN set but not executable: {p}")

    try:
        import importlib.resources as ir  # py3.9+

        pkg = ir.files("tensorcast").joinpath("bin").joinpath("tensorcast_daemon")
        p = Path(str(pkg))
        if p.exists() and os.access(p, os.X_OK):
            return p
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "bazel-bin" / "daemon" / "tensorcast_daemon"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return candidate

    raise ServiceError(
        "tensorcast_daemon binary not found. Set TENSORCAST_DAEMON_BIN, install a package containing tensorcast/bin/tensorcast_daemon, or build it via Bazel (bazel build //daemon:tensorcast_daemon)."
    )


def _format_bytes(bytes_value: int) -> str:
    if bytes_value < 0:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_value)
    while value >= 1024 and unit_index < len(units) - 1:
        value /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"


def _format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs}s"


def _display_detailed_status(
    pid: int, response: "store_daemon_pb2.GetDetailedStatusResponse"
) -> None:
    click.echo("\n" + "=" * 60)
    click.echo("StoreDaemon Status")
    click.echo("=" * 60)
    click.echo(f"PID: {pid}")
    click.echo(f"Uptime: {_format_duration(response.uptime_seconds)}")
    click.echo(f"Health: {'Healthy' if response.is_healthy else 'Unhealthy'}")
    if response.is_shutting_down:
        click.echo("Status: SHUTTING DOWN")
    click.echo(f"Registered: {'Yes' if response.is_registered else 'No'}")
    if response.worker_id:
        click.echo(f"Worker ID: {response.worker_id}")
    mp = response.memory_pool_info
    click.echo("\nMemory Pool Status")
    click.echo("=" * 60)
    click.echo(f"Total Size: {_format_bytes(mp.total_size_bytes)}")
    if mp.total_size_bytes > 0:
        percent_avail = (mp.available_bytes / mp.total_size_bytes) * 100
        click.echo(
            f"Available: {_format_bytes(mp.available_bytes)} ({percent_avail:.1f}%)"
        )
        click.echo(
            f"Allocated: {_format_bytes(mp.allocated_bytes)} ({100 - percent_avail:.1f}%)"
        )
    else:
        click.echo(f"Available: {_format_bytes(mp.available_bytes)}")
        click.echo(f"Allocated: {_format_bytes(mp.allocated_bytes)}")
    if mp.allocated_chunks_count > 0:
        click.echo(f"Chunks: {mp.allocated_chunks_count} allocated")
    if mp.chunk_size_bytes > 0:
        click.echo(f"Chunk Size: {_format_bytes(mp.chunk_size_bytes)}")
    if response.gpu_devices:
        click.echo(f"\nGPU Devices ({len(response.gpu_devices)} total)")
        click.echo("=" * 60)
        for gpu in response.gpu_devices:
            click.echo(f"\nGPU {gpu.device_id}")
            if gpu.device_uuid:
                click.echo(f"  UUID: {gpu.device_uuid}")
            click.echo(f"  Total Memory: {_format_bytes(gpu.total_memory_bytes)}")
            if gpu.total_memory_bytes > 0:
                used_percent = (gpu.used_memory_bytes / gpu.total_memory_bytes) * 100
                click.echo(
                    f"  Used: {_format_bytes(gpu.used_memory_bytes)} ({used_percent:.1f}%)"
                )
                click.echo(
                    f"  Free: {_format_bytes(gpu.free_memory_bytes)} ({100 - used_percent:.1f}%)"
                )
            else:
                click.echo(f"  Used: {_format_bytes(gpu.used_memory_bytes)}")
                click.echo(f"  Free: {_format_bytes(gpu.free_memory_bytes)}")
            if gpu.loaded_replicas:
                click.echo(f"  Replicas ({len(gpu.loaded_replicas)} loaded):")
                for artifact in gpu.loaded_replicas:
                    click.echo(
                        f"    - {artifact.artifact_id} ({_format_bytes(artifact.artifact_size_bytes)})"
                    )
    if response.cpu_replicas:
        click.echo(f"\nCPU Memory Replicas ({len(response.cpu_replicas)} total)")
        click.echo("=" * 60)
        for artifact in response.cpu_replicas:
            click.echo(
                f"- {artifact.artifact_id} ({_format_bytes(artifact.artifact_size_bytes)})"
            )
            if artifact.is_registered_for_comm:
                click.echo("  [Registered for RDMA]")
    comm = response.communication_info
    if comm.enabled:
        click.echo("\nCommunication Status")
        click.echo("=" * 60)
        click.echo("RDMA: Enabled")
        if comm.total_transfers > 0:
            click.echo(f"Total Transfers: {comm.total_transfers}")
            click.echo(
                f"Bytes Transferred: {_format_bytes(comm.total_bytes_transferred)}"
            )
            if comm.total_transfer_errors > 0:
                click.echo(f"Transfer Errors: {comm.total_transfer_errors}")
    click.echo("\nSummary")
    click.echo("=" * 60)
    click.echo(f"Total Replicas Loaded: {response.total_replicas_loaded}")
    click.echo(
        f"Total Artifact Size: {_format_bytes(response.total_artifact_size_bytes)}"
    )
    click.echo(f"Storage Path: {response.storage_path}")
    if response.num_worker_threads > 0:
        click.echo(f"Worker Threads: {response.num_worker_threads}")
    click.echo("")


def _kill_gracefully(pgid: int, grace: float = 10.0) -> bool:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False


def _kill_force(pgid: int) -> None:
    from contextlib import suppress

    with suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _register_process(
    inst: Instance, *, role: str, proc: subprocess.Popen, so: Path, se: Path
) -> None:
    entry = {
        "role": role,
        "pid": int(proc.pid),
        "cmd": [str(x) for x in proc.args]
        if isinstance(proc.args, (list, tuple))
        else [str(proc.args)],
        "stdout": str(so),
        "stderr": str(se),
        "start_time": _now_ts(),
    }
    data = {"instance_id": inst.id, "processes": [entry]}
    _write_json_locked(inst.pids_json, data)


def _write_meta(
    inst: Instance, *, address: str, config_path: str, p2p_address: str | None = None
) -> None:
    meta = {
        "instance_id": inst.id,
        "address": address,
        "config_path": config_path,
        "created_at": _now_ts(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME", ""),
    }
    if p2p_address:
        meta["p2p_address"] = p2p_address
    _write_json_locked(inst.meta_json, meta)


def logs_tail(
    *, instance_id: str | None = None, stderr: bool = False, follow: bool = False
) -> None:
    iid = instance_id or get_current_instance_id()
    if not iid:
        raise ServiceError("No session id provided and no current session found")
    inst = _instance_paths(iid)
    log_file = inst.logs / ("daemon.err" if stderr else "daemon.out")
    if not log_file.exists():
        click.echo(f"No log file found for daemon session: {log_file}")
        return
    click.echo(f"==> {log_file} <==")
    if not follow:
        try:
            with open(log_file, "rb") as f:
                data = f.read()
                lines = data.splitlines()[-200:]
                for ln in lines:
                    click.echo(ln.decode("utf-8", errors="replace"))
        except Exception as e:
            raise ServiceError(f"Failed to read logs: {e}") from e
        return
    try:
        with open(log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    click.echo(line.decode("utf-8", errors="replace"), nl=False)
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        return
    except Exception as e:
        raise ServiceError(f"Failed to tail logs: {e}") from e


def get_process_info(pid: int) -> dict[str, Any] | None:
    try:
        p = psutil.Process(pid)
        return {
            "pid": pid,
            "started": time.ctime(p.create_time()),
            "cpu_percent": p.cpu_percent(),
            "memory_mb": p.memory_info().rss / 1024 / 1024,
        }
    except Exception:
        return {"pid": pid}
