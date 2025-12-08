#  Copyright (c) 2025, TensorCast Team.

"""Shared configuration helpers for CLI/SDK orchestration.

- Discover default daemon / Global Store config paths
- Build embedded daemon / Global Store configs for local launches
- Allocate deterministic ports for daemon/Global Store
- Manage the implicit cluster token persisted under ~/.tensorcast/runtime
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
from pathlib import Path
from typing import Tuple

import yaml
from google.protobuf.json_format import MessageToDict

from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.filesys import write_text_atomic
from tensorcast.cli_utils.network import pick_free_tcp_port
from tensorcast.cli_utils.paths import (
    DaemonSession,
    GlobalSession,
    global_session_paths,
    home_dir,
    runtime_lock_path,
    runtime_root,
)
from tensorcast.cli_utils.process import file_lock
from tensorcast.daemon_runtime_config import dump_daemon_config
from tensorcast.proto.config.v1 import common_pb2 as common_pb
from tensorcast.proto.config.v1 import daemon_config_pb2 as cfg_pb
from tensorcast.proto.config.v1 import global_store_config_pb2 as gsc_pb

_CLUSTER_TOKEN_FILENAME = "cluster_token"
_CLUSTER_TOKEN_HMAC_KEY = b"tensorcast-cluster-token"


def discover_global_store_config() -> Path | None:
    """Discover a default Global Store config.

    Order:
    1) $TENSORCAST_GLOBAL_STORE_CONFIG
    2) ~/.tensorcast/config/global_store.yaml or .yml
    3) None (caller should fall back to embedded config)
    """

    env = os.environ.get("TENSORCAST_GLOBAL_STORE_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    cfg_dir = home_dir() / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name in ("global_store.yaml", "global_store.yml"):
        candidate = cfg_dir / name
        if candidate.exists():
            return candidate
    return None


def _can_bind(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


def select_free_port(
    preferred: int, *, host: str = "127.0.0.1", probe_span: int = 32
) -> int:
    """Select a free TCP port, preferring the provided base when possible."""

    if preferred > 0 and _can_bind(host, preferred):
        return preferred
    if preferred > 0:
        for offset in range(1, max(1, probe_span)):
            candidate = preferred + offset
            if _can_bind(host, candidate):
                return candidate
    return pick_free_tcp_port()


def choose_daemon_ports(
    preferred_rpc: int = 50052,
    preferred_p2p: int = 50053,
    probe_span: int = 32,
) -> Tuple[int, int]:
    """Deterministically pick RPC/P2P ports with minimal collisions."""

    rpc_port = select_free_port(preferred_rpc, probe_span=probe_span)
    p2p_base = preferred_p2p if preferred_p2p > 0 else rpc_port + 1
    p2p_port = select_free_port(p2p_base, probe_span=probe_span)
    if p2p_port == rpc_port:
        # Avoid self-conflict by probing forward then falling back to ephemeral.
        for offset in range(1, max(2, probe_span)):
            candidate = p2p_base + offset
            if candidate == rpc_port:
                continue
            if _can_bind("127.0.0.1", candidate):
                p2p_port = candidate
                break
        else:
            p2p_port = pick_free_tcp_port()
            if p2p_port == rpc_port:
                p2p_port = pick_free_tcp_port()
    return rpc_port, p2p_port


def discover_daemon_config() -> Path | None:
    """Discover a default daemon config path.

    Order:
    1) $TENSORCAST_DAEMON_CONFIG
    2) ~/.tensorcast/config/daemon.yaml or .yml
    3) None (callers should fall back to embedded defaults)
    """

    env = os.environ.get("TENSORCAST_DAEMON_CONFIG")
    if env:
        path = Path(env).expanduser()
        if path.exists():
            return path

    cfg_dir = home_dir() / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name in ("daemon.yaml", "daemon.yml"):
        candidate = cfg_dir / name
        if candidate.exists():
            return candidate
    return None


def build_embedded_daemon_config(
    session: DaemonSession, *, restrict_to_localhost: bool = False
) -> cfg_pb.DaemonConfig:
    """Create an embedded daemon config anchored to a session directory."""

    storage_dir = session.root / "data"
    fallback_dir = session.root / "p2p_cache"
    storage_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)

    cfg = cfg_pb.DaemonConfig()
    listen_host = (
        "127.0.0.1"
        if restrict_to_localhost
        else (cfg.server.listen.host or "127.0.0.1")
    )
    cfg.server.listen.host = listen_host
    cfg.server.listen.port = 0
    cfg.server.storage_path = str(storage_dir)
    cfg.server.num_threads = max(4, min(8, os.cpu_count() or 4))
    cfg.server.p2p_listen.host = listen_host
    cfg.server.p2p_listen.port = 0

    cfg.engine.mem_pool_size_bytes = 1 * 1024 * 1024 * 1024
    cfg.engine.tx_slice_bytes = 256 * 1024 * 1024
    cfg.engine.artifact_chunk_bytes = 256 * 1024 * 1024
    cfg.engine.streaming_buffer_max_concurrent_sessions = 1
    cfg.engine.p2p_fallback_disk_dir = str(fallback_dir)
    cfg.engine.disk_path_whitelist.extend([str(storage_dir), str(fallback_dir)])
    cfg.engine.pinned_allocation_timeout.FromSeconds(30)

    cfg.lifecycle.gpu_memory_limit_fraction = 0.75
    cfg.lifecycle.enable_periodic_eviction = False
    cfg.lifecycle.eviction_loop_interval.FromSeconds(1)
    cfg.lifecycle.proc_check_interval.FromSeconds(5)
    cfg.lifecycle.sessions_sweep_interval.FromSeconds(10)
    cfg.lifecycle.locks_sweep_interval.FromSeconds(10)
    cfg.lifecycle.verification_sweep_interval.FromMilliseconds(500)
    cfg.lifecycle.sessions_ttl.FromSeconds(60)
    cfg.lifecycle.locks_ttl.FromSeconds(120)

    cfg.high_availability.enabled = False
    cfg.communicator.enable_rdma = False
    cfg.checkpoint.streaming.num_buffers = 2
    cfg.checkpoint.streaming.io_chunk_bytes = 128 * 1024 * 1024
    cfg.checkpoint.streaming.pinned_pool_bytes = 512 * 1024 * 1024
    cfg.observability.logging.level = common_pb.Observability.LogLevel.LOG_LEVEL_INFO
    cfg.meta.description = "embedded-daemon-config"
    return cfg


def materialize_daemon_config(
    session: DaemonSession,
    config_path: Path | None,
    *,
    restrict_to_localhost: bool = False,
) -> Path:
    """Resolve or create a daemon config for the given session."""

    if config_path is None:
        config_path = discover_daemon_config()

    if config_path is not None:
        return Path(config_path).expanduser()

    cfg = build_embedded_daemon_config(
        session, restrict_to_localhost=restrict_to_localhost
    )
    effective_path = session.effective_config_path
    effective_path.parent.mkdir(parents=True, exist_ok=True)
    dump_daemon_config(cfg, effective_path)
    return effective_path


def _cluster_token_path() -> Path:
    return runtime_root() / _CLUSTER_TOKEN_FILENAME


def _token_hmac(token: str) -> str:
    return hmac.new(
        _CLUSTER_TOKEN_HMAC_KEY, token.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def load_or_create_cluster_token(existing: str | None = None) -> str:
    """Load the implicit cluster token, creating it if missing.

    - Token is 128-bit random hex with HMAC checksum to detect corruption.
    - Stored at ~/.tensorcast/runtime/cluster_token with mode 0600.
    - If *existing* is provided and no file exists, it is persisted verbatim.
    """

    path = _cluster_token_path()
    token: str | None = None
    with file_lock(runtime_lock_path()):
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                token = str(payload.get("token"))
                stored_hmac = str(payload.get("hmac") or "")
            except Exception as exc:  # noqa: BLE001
                raise ServiceError(
                    f"Failed to parse cluster token at {path}: {exc}"
                ) from exc
            if not token or not stored_hmac:
                raise ServiceError(
                    f"Cluster token at {path} is invalid; delete the file to regenerate."
                )
            if _token_hmac(token) != stored_hmac:
                raise ServiceError(
                    f"Cluster token checksum mismatch at {path}; delete the file to regenerate."
                )
            return token

        token = existing or secrets.token_hex(16)
        payload = {"token": token, "hmac": _token_hmac(token)}
        write_text_atomic(path, json.dumps(payload, sort_keys=True), mode=0o600)
        return token


def build_embedded_global_store_config(
    session: GlobalSession | None = None,
    *,
    cluster_token: str,
    listen_host: str = "127.0.0.1",
    listen_port: int = 0,
    metrics_port: int = 0,
) -> gsc_pb.GlobalStoreConfig:
    """Create an embedded Global Store config anchored to the given session."""

    inst = session or global_session_paths()
    cfg = gsc_pb.GlobalStoreConfig()
    cfg.server.listen.host = listen_host or "127.0.0.1"
    cfg.server.listen.port = max(0, int(listen_port or 0))
    cfg.server.max_workers = 10
    cfg.server.metrics_port = max(0, int(metrics_port or 0))
    cfg.database.db_file = str(inst.root / "global_store.duckdb")
    cfg.meta.schema_version = "v1"
    cfg.meta.description = "generated-by-cli"
    cfg.meta.cluster_token = cluster_token
    cfg.observability.logging.level = common_pb.Observability.LogLevel.LOG_LEVEL_INFO
    cfg.observability.logging.file = str(inst.logs / "global_store.out")
    return cfg


def dump_global_store_config(cfg: gsc_pb.GlobalStoreConfig, path: Path) -> Path:
    """Serialize GlobalStoreConfig proto to YAML."""

    data = MessageToDict(cfg, preserving_proto_field_name=True)
    yaml_text = yaml.safe_dump(data, sort_keys=False)
    write_text_atomic(path, yaml_text, mode=0o600)
    return path


__all__ = [
    "build_embedded_daemon_config",
    "build_embedded_global_store_config",
    "choose_daemon_ports",
    "discover_daemon_config",
    "discover_global_store_config",
    "dump_global_store_config",
    "load_or_create_cluster_token",
    "materialize_daemon_config",
    "select_free_port",
]
