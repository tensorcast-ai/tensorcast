#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tensorcast.cli_utils import global_store_manager, service_manager
from tensorcast.cli_utils.config import materialize_daemon_config
from tensorcast.cli_utils.errors import ServiceError
from tensorcast.cli_utils.health import (
    GlobalStoreHealth,
    ping_daemon,
    ping_global_store,
)
from tensorcast.cli_utils.paths import (
    clear_current_session_if_matches,
    get_current_session_id,
    runtime_lock_path,
    runtime_root,
    runtime_state_path,
    session_paths,
)
from tensorcast.cli_utils.proc import is_matching_daemon_process
from tensorcast.cli_utils.process import (
    clear_runtime_daemon,
    file_lock,
    instance_fingerprint,
    is_process_alive,
    prune_process_records,
    read_json_default,
    read_runtime_state,
    read_session_state,
    update_runtime_global_store,
)
from tensorcast.logger import init_logger

logger = init_logger(__name__)


@dataclass
class RuntimeSession:
    session_id: str
    daemon_pid: int | None
    daemon_address: str | None
    daemon_p2p_address: str | None
    logs_dir: Path | None
    started_at: float | None
    owner: bool | None = None
    global_store_mode: Literal["connect", "start", "none"] | None = None
    global_store_address: str | None = None
    global_store_session: str | None = None
    global_store_owner: bool | None = None
    cluster_token: str | None = None


@dataclass(frozen=True)
class _ResolvedGlobalStore:
    mode: Literal["connect", "start", "none"]
    address: str | None
    session_id: str | None
    owner: bool
    cluster_token: str | None


def _build_runtime_session(session_id: str) -> RuntimeSession | None:
    inst = session_paths(session_id)
    session_state: dict[str, Any] | None = None
    with contextlib.suppress(Exception):
        session_state = read_session_state(inst.session_state_json)
    runtime_state: dict[str, Any] | None = None
    with contextlib.suppress(Exception):
        runtime_state = read_runtime_state(runtime_state_path())
    daemon_state = None
    if isinstance(runtime_state, dict):
        daemon_state = runtime_state.get("daemon")
    daemon_info = (
        session_state.get("daemon") if isinstance(session_state, dict) else None
    )
    pid = None
    address = None
    p2p_address = None
    owner = None
    if isinstance(daemon_state, dict) and daemon_state.get("session_id") == session_id:
        pid = int(daemon_state.get("pid", 0) or 0) or None
        address = daemon_state.get("address")
        p2p_address = daemon_state.get("p2p_address")
        owner = daemon_state.get("owner")
    if isinstance(daemon_info, dict):
        pid = int(daemon_info.get("pid", 0) or 0) or pid
        address = daemon_info.get("address") or address
        p2p_address = daemon_info.get("p2p_address") or p2p_address
    logs_dir = inst.logs if inst.logs.exists() else None
    if isinstance(session_state, dict) and session_state.get("logs_dir"):
        logs_dir = Path(session_state["logs_dir"])
    started_at: float | None = None
    if isinstance(session_state, dict):
        started_at_raw = session_state.get("started_at")
        if started_at_raw is not None:
            started_at = float(started_at_raw)
    global_store_mode: Literal["connect", "start", "none"] | None = None
    global_store_address = None
    global_store_session = None
    global_store_owner = None
    cluster_token = None
    if isinstance(session_state, dict):
        gs = session_state.get("global_store")
        if isinstance(gs, dict):
            global_store_mode = gs.get("mode")
            global_store_address = gs.get("address")
            global_store_session = gs.get("session")
            global_store_owner = gs.get("owner")
            cluster_token = gs.get("cluster_token")
    if isinstance(runtime_state, dict):
        rs_gs = runtime_state.get("global_store")
        if isinstance(rs_gs, dict):
            cluster_token = cluster_token or rs_gs.get("cluster_token")
            if rs_gs.get("session_id") == global_store_session:
                global_store_address = global_store_address or rs_gs.get("address")
                global_store_owner = global_store_owner or rs_gs.get("owner")
    return RuntimeSession(
        session_id=session_id,
        daemon_pid=pid,
        daemon_address=address,
        daemon_p2p_address=p2p_address,
        logs_dir=logs_dir,
        started_at=started_at,
        owner=owner,
        global_store_mode=global_store_mode,
        global_store_address=global_store_address,
        global_store_session=global_store_session,
        global_store_owner=global_store_owner,
        cluster_token=cluster_token,
    )


def _prune_stale_session(
    session_id: str, pid: int | None, *, reason: str | None = None
) -> None:
    if reason:
        logger.warning(
            "Reconcile cleaning daemon session %s (pid=%s): %s",
            session_id,
            pid if pid is not None else "unknown",
            reason,
        )
    inst = session_paths(session_id)
    with file_lock(inst.pids_lock):
        prune_process_records(
            pids_path=inst.pids_json,
            predicate=(
                (lambda entry: int(entry.get("pid", 0)) == int(pid))
                if pid
                else (lambda _entry: True)
            ),
            lock_path=None,
        )
        with contextlib.suppress(FileNotFoundError):
            inst.meta_json.unlink()
        with contextlib.suppress(Exception):
            state = read_runtime_state(runtime_state_path())
            daemon_state = state.get("daemon")
            if not isinstance(daemon_state, dict):
                pass
            elif daemon_state.get("session_id") == session_id and (
                pid is None or int(daemon_state.get("pid", 0)) == int(pid)
            ):
                clear_runtime_daemon(runtime_state_path())
    clear_current_session_if_matches(session_id)


def reconcile(session_id: str | None = None) -> RuntimeSession | None:
    runtime_state: dict[str, Any] | None = None
    daemon_state: dict[str, Any] | None = None
    target_session: str | None = None
    pids_data: dict[str, Any] | None = None

    with contextlib.suppress(Exception), file_lock(runtime_lock_path()):
        runtime_state = read_runtime_state(runtime_state_path())
        daemon_state = (
            runtime_state.get("daemon") if isinstance(runtime_state, dict) else None
        )
        target_session = (
            session_id
            or (
                daemon_state.get("session_id")
                if isinstance(daemon_state, dict)
                else None
            )
            or get_current_session_id()
        )
        if target_session:
            inst = session_paths(target_session)
            with file_lock(inst.pids_lock):
                pids_data = read_json_default(inst.pids_json, {"processes": []})

    if not target_session:
        return None

    pid = None
    address = None
    runtime_fp: dict[str, Any] | None = None
    if (
        isinstance(daemon_state, dict)
        and daemon_state.get("session_id") == target_session
    ):
        pid = int(daemon_state.get("pid", 0) or 0) or None
        address = daemon_state.get("address")
        if isinstance(daemon_state.get("instance_fingerprint"), dict):
            runtime_fp = daemon_state["instance_fingerprint"]
    processes: list[dict[str, Any]] = []
    if isinstance(pids_data, dict):
        raw_procs = pids_data.get("processes", [])
        if isinstance(raw_procs, list):
            processes = [proc for proc in raw_procs if isinstance(proc, dict)]

    expected_cmd0: str | None = None
    if pid is None and processes:
        for proc in processes:
            try:
                candidate_pid = int(proc.get("pid", 0) or 0)
            except Exception:
                candidate_pid = 0
            if candidate_pid > 0:
                pid = candidate_pid
                cmd = proc.get("cmd")
                if isinstance(cmd, list) and cmd:
                    expected_cmd0 = str(cmd[0])
                break
    if expected_cmd0 is None and pid is not None and processes:
        for proc in processes:
            try:
                candidate_pid = int(proc.get("pid", 0) or 0)
            except Exception:
                candidate_pid = 0
            if candidate_pid == pid:
                cmd = proc.get("cmd")
                if isinstance(cmd, list) and cmd:
                    expected_cmd0 = str(cmd[0])
                break

    fingerprint_mismatch = False
    if runtime_fp is not None:
        if pid is None:
            fingerprint_mismatch = True
        else:
            current_fp = instance_fingerprint(pid)
            try:
                fingerprint_mismatch = runtime_fp.get("host_id") != current_fp.get(
                    "host_id"
                ) or runtime_fp.get("boot_id") != current_fp.get("boot_id")
            except Exception:
                fingerprint_mismatch = True

    alive = pid is not None and is_process_alive(pid)
    health_ok = False
    if address:
        health_ok = ping_daemon(address)

    if health_ok:
        session = _build_runtime_session(target_session)
        if session is not None and pid is not None and not alive:
            session.daemon_pid = None
        return session

    if not alive or fingerprint_mismatch or pid is None:
        reasons: list[str] = []
        if fingerprint_mismatch:
            reasons.append("instance fingerprint mismatch (boot/host changed)")
        if not pid:
            reasons.append("no recorded pid")
        elif not alive:
            reasons.append("pid not alive")
        _prune_stale_session(
            target_session, pid, reason="; ".join(reasons) or "unhealthy"
        )
        return None

    if expected_cmd0 and not is_matching_daemon_process(pid, expected_cmd0):
        _prune_stale_session(
            target_session,
            pid,
            reason="pid no longer matches tensorcast_daemon process",
        )
        return None

    return _build_runtime_session(target_session)


def _split_address(address: str | None) -> tuple[str | None, int | None]:
    if not address:
        return None, None
    try:
        host, port_s = address.rsplit(":", 1)
        return host, int(port_s)
    except Exception:
        return address, None


def _dial_address_from_health(candidate: str, health: GlobalStoreHealth) -> str:
    """Derive a routable dial address from a candidate host:port and server info payload.

    Global Store GetServerInfo responses expose listen_* (bind) and advertise_* (dial) addresses.
    listen_host may legitimately be "0.0.0.0" and is not a valid dial target.

    Policy:
    - Prefer the candidate host if provided and not unspecified (CLI / env / runtime record).
    - Prefer advertise_host/advertise_port from server info when candidate host is missing or unspecified,
      and the advertised host is routable (not unspecified).
    - Use listen_port to backfill the bound port (especially when config uses port=0).
    """

    host_hint, port_hint = _split_address(candidate)
    # Lazily import to avoid import cycles in CLI/runtime code paths.
    from tensorcast.cli_utils.network import is_unspecified_host, resolve_connect_host

    advertise_host = health.advertise_host
    advertise_port = health.advertise_port

    explicit_host = (
        host_hint if host_hint and not is_unspecified_host(host_hint) else None
    )
    if explicit_host:
        host = explicit_host
    elif advertise_host and not is_unspecified_host(advertise_host):
        host = advertise_host
    else:
        host = resolve_connect_host(health.listen_host)

    explicit_port = port_hint if port_hint and port_hint > 0 else None
    port = (
        explicit_port
        or int(advertise_port or 0)
        or int(health.listen_port or 0)
        or None
    )

    if host and port:
        return f"{host}:{port}"
    return candidate


def _resolve_global_store(
    *,
    mode: Literal["connect", "start", "none"],
    address: str | None,
    config_path: Path | None,
    allow_gs_fallback: bool,
    cluster_id: str | None,
    fate_share: bool = True,
    to_console: bool = False,
) -> _ResolvedGlobalStore:
    runtime_state = None
    with contextlib.suppress(Exception):
        runtime_state = read_runtime_state(runtime_state_path())
    runtime_gs = (
        runtime_state.get("global_store") if isinstance(runtime_state, dict) else None
    )
    cluster_token_hint = None
    if isinstance(runtime_gs, dict):
        cluster_token_hint = runtime_gs.get("cluster_token")

    if mode == "none":
        return _ResolvedGlobalStore(
            mode="none",
            address=None,
            session_id=None,
            owner=False,
            cluster_token=cluster_id or cluster_token_hint,
        )

    if address and mode != "connect":
        raise ServiceError("global_store_address requires global_store_mode=connect")

    runtime_gs_session = (
        runtime_gs.get("session_id") if isinstance(runtime_gs, dict) else None
    )
    runtime_gs_address = (
        runtime_gs.get("address") if isinstance(runtime_gs, dict) else None
    )

    if mode == "connect":
        candidates: list[str] = []
        seen: set[str] = set()
        for candidate in (
            address,
            os.environ.get("TENSORCAST_GLOBAL_STORE_ADDRESS"),
            os.environ.get("TENSORCAST_GLOBAL_STORE"),
            runtime_gs_address,
        ):
            if candidate and candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

        for candidate in candidates:
            health = ping_global_store(candidate, timeout=2.0)
            if not health:
                continue
            if (
                cluster_id
                and health.cluster_token
                and health.cluster_token != cluster_id
            ):
                raise ServiceError(
                    f"Global Store at {candidate} has mismatched cluster token; expected {cluster_id}"
                )
            if (
                cluster_token_hint
                and health.cluster_token
                and health.cluster_token != cluster_token_hint
            ):
                raise ServiceError(
                    f"Global Store at {candidate} has mismatched cluster token; expected "
                    "recorded runtime cluster token"
                )
            return _ResolvedGlobalStore(
                mode="connect",
                address=_dial_address_from_health(candidate, health),
                session_id=runtime_gs_session,
                owner=False,
                cluster_token=health.cluster_token or cluster_id or cluster_token_hint,
            )

        raise ServiceError("Global Store connect mode requires a reachable address")

    if runtime_gs_address:
        health = ping_global_store(runtime_gs_address, timeout=2.0)
        if health:
            if (
                cluster_id
                and health.cluster_token
                and health.cluster_token != cluster_id
            ):
                raise ServiceError(
                    f"Global Store cluster token mismatch; expected {cluster_id}"
                )
            if (
                cluster_token_hint
                and health.cluster_token
                and health.cluster_token != cluster_token_hint
            ):
                raise ServiceError(
                    "Global Store cluster token mismatch; expected recorded runtime cluster token"
                )
            return _ResolvedGlobalStore(
                mode="start",
                address=_dial_address_from_health(runtime_gs_address, health),
                session_id=runtime_gs_session,
                owner=False,
                cluster_token=health.cluster_token or cluster_id or cluster_token_hint,
            )

    if cluster_token_hint and not allow_gs_fallback:
        raise ServiceError(
            "Found existing Global Store cluster token but no healthy instance is reachable. "
            f"Clear {runtime_root()} if you intend to create a new cluster, or provide "
            "a reachable --global-store-address."
        )

    try:
        inst = global_store_manager.start_global_store(
            config_path=config_path,
            cluster_token=cluster_id or cluster_token_hint,
            fate_share=fate_share,
            to_console=to_console,
        )
    except ServiceError:
        if allow_gs_fallback:
            logger.warning(
                "Global Store failed to start; continuing with global_store_mode=none"
            )
            return _ResolvedGlobalStore(
                mode="none",
                address=None,
                session_id=None,
                owner=False,
                cluster_token=cluster_id or cluster_token_hint,
            )
        raise

    return _ResolvedGlobalStore(
        mode="start",
        address=inst.address,
        session_id=inst.id,
        owner=True,
        cluster_token=inst.cluster_token or cluster_id or cluster_token_hint,
    )


def start(
    *,
    daemon_config: Path | None = None,
    session_id: str | None = None,
    global_store_address: str | None = None,
    global_store_config: Path | None = None,
    global_store_mode: Literal["connect", "start", "none"] = "none",
    config_overrides: list[str] | tuple[str, ...] | None = None,
    ha_endpoints: list[str] | tuple[str, ...] | None = None,
    register_current: bool = True,
    ephemeral: bool = False,
    restrict_to_localhost: bool = False,
    listen_host: str | None = None,
    listen_port: int | None = None,
    blocking: bool = False,
    to_console: bool = True,
    allow_gs_fallback: bool = False,
    cluster_id: str | None = None,
    reuse_existing: bool = False,
    fate_share: bool = True,
) -> RuntimeSession:
    existing = reconcile()
    if existing is not None:
        if reuse_existing:
            return existing
        raise ServiceError(
            f"A StoreDaemon is already running for session {existing.session_id} "
            f"at {existing.daemon_address or 'unknown'}. Stop it before starting another."
        )

    resolved_gs = _resolve_global_store(
        mode=global_store_mode,
        address=global_store_address,
        config_path=global_store_config,
        allow_gs_fallback=allow_gs_fallback,
        cluster_id=cluster_id,
        fate_share=fate_share,
        to_console=blocking,
    )
    resolved_ha_endpoints: list[str] = []
    if resolved_gs.mode != "none":
        if ha_endpoints:
            resolved_ha_endpoints = list(ha_endpoints)
            if resolved_gs.address:
                if resolved_ha_endpoints:
                    resolved_ha_endpoints[0] = resolved_gs.address
                else:
                    resolved_ha_endpoints = [resolved_gs.address]
        elif resolved_gs.address:
            resolved_ha_endpoints = [resolved_gs.address]
    gs_state: dict[str, Any] = {
        "mode": resolved_gs.mode,
        "required": resolved_gs.mode != "none",
    }
    if resolved_gs.address:
        gs_state["address"] = resolved_gs.address
    if resolved_gs.session_id:
        gs_state["session"] = resolved_gs.session_id
    gs_state["owner"] = resolved_gs.owner
    if resolved_gs.cluster_token:
        gs_state["cluster_token"] = resolved_gs.cluster_token

    inst_paths = session_paths(session_id)
    config = materialize_daemon_config(
        inst_paths,
        daemon_config,
        restrict_to_localhost=restrict_to_localhost,
    )

    inst = service_manager.start_service(
        config_path=Path(config),
        session_id=inst_paths.id,
        blocking=blocking,
        to_console=to_console,
        register_current=register_current,
        publish_meta=True,
        ephemeral=ephemeral,
        persist_runtime_state=True,
        owner=True,
        restrict_to_localhost=restrict_to_localhost,
        config_overrides=config_overrides,
        global_store=gs_state,
        listen_host=listen_host,
        listen_port=listen_port,
        ha_endpoints=resolved_ha_endpoints or None,
        ha_enabled=resolved_gs.mode != "none",
        fate_share=fate_share,
    )

    if resolved_gs.address and not resolved_gs.owner and resolved_gs.mode != "none":
        host, port = _split_address(resolved_gs.address)
        with file_lock(runtime_lock_path()):
            update_runtime_global_store(
                path=runtime_state_path(),
                session_id=resolved_gs.session_id or "external",
                pid=None,
                address=resolved_gs.address,
                listen_host=host,
                listen_port=port,
                metrics_port=None,
                db_file=None,
                cluster_token=resolved_gs.cluster_token,
                owner=False,
                fingerprint=None,
            )

    return status(inst.id) or RuntimeSession(
        session_id=inst.id,
        daemon_pid=None,
        daemon_address=None,
        daemon_p2p_address=None,
        logs_dir=inst.logs,
        started_at=None,
    )


def stop(*, session_id: str | None = None, force: bool = False) -> None:
    sid = session_id
    if not sid:
        existing = reconcile()
        if existing is not None:
            sid = existing.session_id
    if not sid:
        sid = get_current_session_id()
    if not sid:
        raise ServiceError(
            "No local daemon session found. Start one with 'tensorcast daemon start'."
        )

    session_state = None
    with contextlib.suppress(Exception):
        session_state = read_session_state(session_paths(sid).session_state_json)
    runtime_state = None
    with contextlib.suppress(Exception):
        runtime_state = read_runtime_state(runtime_state_path())

    daemon_state = (
        runtime_state.get("daemon") if isinstance(runtime_state, dict) else None
    )
    daemon_owner = (
        bool(daemon_state.get("owner")) if isinstance(daemon_state, dict) else True
    )

    gs_info_raw = (
        session_state.get("global_store") if isinstance(session_state, dict) else None
    )
    gs_info = gs_info_raw if isinstance(gs_info_raw, dict) else {}
    gs_session = gs_info.get("session")
    gs_owner = bool(gs_info.get("owner"))
    if not gs_session and isinstance(runtime_state, dict):
        rs_gs = runtime_state.get("global_store")
        if isinstance(rs_gs, dict):
            gs_session = rs_gs.get("session_id")
            gs_owner = gs_owner or bool(rs_gs.get("owner"))

    if daemon_owner and gs_owner and gs_session:
        with contextlib.suppress(Exception):
            global_store_manager.stop_global_store(
                session_id=gs_session, force=force, quiet=True
            )

    if daemon_owner:
        service_manager.stop_service(session_id=sid, force=force)
    else:
        clear_current_session_if_matches(sid)


def status(session_id: str | None = None) -> RuntimeSession | None:
    session = reconcile(session_id)
    if session is not None:
        return session
    sid = session_id or get_current_session_id()
    if not sid:
        return None
    return _build_runtime_session(sid)


def is_running() -> bool:
    return status() is not None


__all__ = [
    "RuntimeSession",
    "start",
    "stop",
    "status",
    "is_running",
    "reconcile",
]
