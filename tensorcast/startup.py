#  Copyright (c) 2025-2026, TensorCast Team.

"""
Runtime startup and connection utilities for TensorCast.

- Provides a unified `init(mode=...)` entrypoint that either connects to an existing
  Store Daemon or launches one locally and connects to it.
- Returns a Context object whose lifecycle is process-scoped. The context is
  automatically closed at process exit via atexit when we launched the daemon
  (or connected). Use `shutdown()` to close early if needed. Context manager
  semantics are not required or supported for this object.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib
import json
import os
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, SupportsInt, cast

from pydantic import BaseModel, ConfigDict, field_validator

from tensorcast.api._config import clear_daemon_address, set_daemon_address
from tensorcast.cli_utils.config import discover_daemon_config
from tensorcast.cli_utils.health import ping_daemon, wait_for_daemon
from tensorcast.cli_utils.paths import runtime_lock_path, runtime_root, session_paths
from tensorcast.cli_utils.process import (
    atomic_write_json,
    file_lock,
    instance_fingerprint,
    is_process_alive,
    read_json_default,
)
from tensorcast.client_config_loader import discover_client_config, load_client_config
from tensorcast.client_runtime import daemon_target_default, set_client_config

if TYPE_CHECKING:
    from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.daemon_runtime_config import load_daemon_config
from tensorcast.logger import init_logger, setup_logging

runtime = importlib.import_module("tensorcast.runtime")

_current_ctx: Context | None = None
_atexit_registered = False
_ctx_lock: "threading.Lock" = threading.Lock()

_AUTO_STATE_SCHEMA_VERSION = 1
_AUTO_STATUS_EMPTY = "EMPTY"
_AUTO_STATUS_STARTING = "STARTING"
_AUTO_STATUS_LISTENING = "LISTENING"
_AUTO_STATUS_READY = "READY"
_AUTO_STATUS_FAILED = "FAILED"
_AUTO_STATUS_VALUES = {
    _AUTO_STATUS_EMPTY,
    _AUTO_STATUS_STARTING,
    _AUTO_STATUS_LISTENING,
    _AUTO_STATUS_READY,
    _AUTO_STATUS_FAILED,
}
_AUTO_WAIT_TIMEOUT_SECONDS = 180.0
_AUTO_POLL_INTERVAL_SECONDS = 0.2


class PortConfig(BaseModel):
    """Optional port overrides for SDK-managed create/auto startup."""

    model_config = ConfigDict(frozen=True)

    daemon_listen_port: int | None = None
    daemon_p2p_port: int | None = None
    global_store_listen_port: int | None = None
    global_store_metrics_port: int | None = None

    @field_validator(
        "daemon_listen_port",
        "daemon_p2p_port",
        "global_store_listen_port",
        "global_store_metrics_port",
        mode="before",
    )
    @classmethod
    def _validate_port(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("port values must be integers between 0 and 65535")
        try:
            port = int(
                cast(SupportsInt | SupportsIndex | str | bytes | bytearray, value)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "port values must be integers between 0 and 65535"
            ) from exc
        if port < 0 or port > 65535:
            raise ValueError("port values must be integers between 0 and 65535")
        return port


@dataclass(slots=True)
class Context:
    """Runtime context representing a client session against a Store Daemon.

    - If `is_owner` is True, the current process launched the daemon and is
      responsible for stopping it on close().
    - `address` is the gRPC address for the daemon (host:port).
    - `session_id` and `session_dir` are best-effort identifiers provided by
      the service manager when launching locally.
    - The object does not implement context manager methods; it is managed for
      the lifetime of the process and cleaned up on exit.
    """

    address: str
    is_owner: bool
    session_id: str | None
    session_dir: str | None
    client: DaemonCtl
    _closed: bool = False
    _handlers_installed: bool = False
    _client_released: bool = False

    def close(self) -> None:
        if self._closed:
            return

        with contextlib.suppress(Exception):
            from tensorcast.api.store import shutdown_process_store

            shutdown_process_store()
        if not self._client_released:
            with contextlib.suppress(Exception):
                from tensorcast.daemon_ctl import release_daemon_client

                release_daemon_client(self.address)
            self._client_released = True
        if self.is_owner and self.session_id:
            # Stop daemon session we launched.
            try:
                runtime.stop(session_id=self.session_id)
            finally:
                _clear_auto_state_if_matches(self.session_id, self.address)
        self._closed = True
        clear_daemon_address()

    def install_signal_handlers(
        self, mode: Literal["graceful", "hard-exit"] = "graceful"
    ) -> None:
        """Optionally install parent signal handlers (SDK opt-in).

        - graceful: on SIGINT/SIGTERM, attempt a cooperative shutdown via stop_service(),
          then exit the process with code 128+signal.
        - hard-exit: on SIGINT/SIGTERM, call os._exit(signal) to immediately terminate,
          relying on child PDEATHSIG for cleanup. Use sparingly.

        This method is idempotent.
        """
        if self._handlers_installed:
            return

        def _handle(signum, _frame):  # noqa: ANN001
            try:
                if self.is_owner and self.session_id:
                    from tensorcast.cli_utils.service_manager import stop_service

                    stop_service(session_id=self.session_id)
            except Exception:
                # Best-effort only; never block signal handling
                pass
            if mode == "hard-exit":
                os._exit(signum)
            else:
                # graceful: propagate exit so outer frameworks can unwind
                import sys as _sys

                try:
                    _sys.exit(128 + int(signum))
                except SystemExit:
                    raise

        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, _handle)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, _handle)
        self._handlers_installed = True


def _current_session_address() -> str | None:
    session = runtime.status()
    if session and session.daemon_address:
        return session.daemon_address
    from tensorcast.cli_utils.service_manager import get_session_address

    return get_session_address()


def _register_atexit_if_needed() -> None:
    global _atexit_registered
    if _atexit_registered:
        return
    atexit.register(lambda: _current_ctx and _current_ctx.close())  # type: ignore[func-returns-value]
    _atexit_registered = True


def _auto_state_path() -> Path:
    return runtime_root() / "daemon_auto_state.json"


def _default_auto_state() -> dict[str, object]:
    return {
        "schema_version": _AUTO_STATE_SCHEMA_VERSION,
        "status": _AUTO_STATUS_EMPTY,
        "epoch": 0,
        "updated_at": time.time(),
    }


def _read_auto_state_locked() -> dict[str, object]:
    state = read_json_default(_auto_state_path(), _default_auto_state())
    if not isinstance(state, dict):
        state = _default_auto_state()
    normalized = dict(_default_auto_state())
    normalized.update(state)
    status = str(normalized.get("status", _AUTO_STATUS_EMPTY)).upper()
    if status not in _AUTO_STATUS_VALUES:
        status = _AUTO_STATUS_EMPTY
    normalized["status"] = status
    normalized["epoch"] = max(0, _coerce_int(normalized.get("epoch"), default=0))
    return normalized


def _write_auto_state_locked(state: dict[str, object]) -> None:
    payload = dict(state)
    payload["schema_version"] = _AUTO_STATE_SCHEMA_VERSION
    payload["updated_at"] = time.time()
    atomic_write_json(_auto_state_path(), payload)


def _clear_auto_state_if_matches(
    session_id: str | None, address: str | None = None
) -> None:
    if not session_id and not address:
        return
    with file_lock(runtime_lock_path()):
        state = _read_auto_state_locked()
        tracked_session = state.get("session_id")
        tracked_address = state.get("address")
        if tracked_session != session_id and tracked_address != address:
            return
        with contextlib.suppress(FileNotFoundError):
            _auto_state_path().unlink()


def _promote_auto_state_ready_when_rpc_ready(
    *, session_id: str | None, address: str
) -> None:
    def _monitor() -> None:
        if not wait_for_daemon(address, timeout=None):
            return
        with file_lock(runtime_lock_path()):
            state = _read_auto_state_locked()
            tracked_session = str(state.get("session_id", "") or "")
            tracked_address = str(state.get("address", "") or "")
            if tracked_session != (session_id or "") or tracked_address != address:
                return
            if (
                str(state.get("status", _AUTO_STATUS_EMPTY)).upper()
                != _AUTO_STATUS_LISTENING
            ):
                return
            ready = dict(state)
            ready["status"] = _AUTO_STATUS_READY
            ready["error_code"] = ""
            ready["error_message"] = ""
            _write_auto_state_locked(ready)

    thread = threading.Thread(
        target=_monitor, name="tensorcast-auto-ready-promoter", daemon=True
    )
    thread.start()


def _wait_for_daemon_rpc_ready_or_raise(
    *,
    address: str,
    timeout_s: float,
    context: str,
) -> None:
    if wait_for_daemon(address, timeout=timeout_s):
        return
    raise RuntimeError(
        "TensorCast daemon did not become RPC-ready after startup: "
        f"context={context}, address={address}, timeout_s={timeout_s:.1f}"
    )


def _auto_wait_timeout_seconds() -> float:
    raw = os.environ.get("TENSORCAST_STARTUP_AUTO_WAIT_TIMEOUT_SECONDS")
    if raw is None or raw == "":
        return _AUTO_WAIT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _AUTO_WAIT_TIMEOUT_SECONDS
    if value <= 0:
        return _AUTO_WAIT_TIMEOUT_SECONDS
    return value


def _coerce_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _hash_file(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        data = path.expanduser().resolve().read_bytes()
    except Exception:
        return None
    return hashlib.sha256(data).hexdigest()


def _compute_auto_config_hash(
    *,
    daemon_config_path: Path | None,
    global_store_mode: Literal["connect", "start", "none"],
    global_store_address: str | None,
    global_store_config_path: str | None,
    cluster_id: str | None,
    allow_gs_fallback: bool,
    session_id: str | None,
    port_config: PortConfig | None,
) -> str:
    daemon_path = (
        str(daemon_config_path.expanduser().resolve()) if daemon_config_path else None
    )
    gs_cfg = (
        str(Path(global_store_config_path).expanduser().resolve())
        if global_store_config_path
        else None
    )
    payload = {
        "daemon_config_path": daemon_path,
        "daemon_config_hash": _hash_file(daemon_config_path),
        "global_store_mode": global_store_mode,
        "global_store_address": global_store_address,
        "global_store_config_path": gs_cfg,
        "global_store_config_hash": _hash_file(Path(gs_cfg)) if gs_cfg else None,
        "cluster_id": cluster_id,
        "allow_gs_fallback": bool(allow_gs_fallback),
        "session_id": session_id,
        "port_config": (
            port_config.model_dump(exclude_none=True) if port_config is not None else {}
        ),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _normalize_port_config(
    *,
    port_config: PortConfig | None,
    global_store_mode: Literal["connect", "start", "none"],
    logger,
) -> PortConfig | None:
    if port_config is None:
        return None
    if global_store_mode == "start":
        return port_config
    if (
        port_config.global_store_listen_port is None
        and port_config.global_store_metrics_port is None
    ):
        return port_config
    logger.warning(
        "port_config.global_store_listen_port/global_store_metrics_port are ignored "
        "unless global_store_mode='start'."
    )
    return port_config.model_copy(
        update={
            "global_store_listen_port": None,
            "global_store_metrics_port": None,
        }
    )


def _owner_alive(owner_pid: int, owner_fingerprint: object) -> bool:
    if owner_pid <= 0 or not is_process_alive(owner_pid):
        return False
    if not isinstance(owner_fingerprint, dict):
        return True
    current = instance_fingerprint(owner_pid)
    try:
        return owner_fingerprint.get("host_id") == current.get(
            "host_id"
        ) and owner_fingerprint.get("boot_id") == current.get("boot_id")
    except Exception:
        return False


def _auto_error(
    *,
    code: str,
    reason: str,
    state: dict[str, object],
    wait_s: float | None = None,
) -> RuntimeError:
    owner_pid = _coerce_int(state.get("owner_pid"), default=0)
    epoch = _coerce_int(state.get("epoch"), default=0)
    session = str(state.get("session_id", "") or "")
    address = str(state.get("address", "") or "")
    logs_dir = str(state.get("logs_dir", "") or "")
    config_hash = str(state.get("config_hash", "") or "")
    started_at = _coerce_float(state.get("started_at"), default=0.0)
    elapsed = max(0.0, time.time() - started_at) if started_at > 0 else 0.0
    wait_text = f", wait_s={wait_s:.2f}" if wait_s is not None else ""
    return RuntimeError(
        f"[{code}] {reason}; owner_pid={owner_pid}, epoch={epoch}, "
        f"elapsed_s={elapsed:.2f}{wait_text}, session_id={session or 'unknown'}, "
        f"address={address or 'unknown'}, logs_dir={logs_dir or 'unknown'}, "
        f"config_hash={config_hash or 'unknown'}"
    )


def _resolve_launch_config(
    daemon_config_path: str | None,
) -> tuple[Path | None, Any | None, bool]:
    cfg_path: Path | None = None
    # Defaults should remain routable. Explicit loopback configs still bind to
    # loopback because the host is carried by the config itself.
    restrict_localhost = False
    cfg: Any | None = None
    if daemon_config_path:
        cfg_path = Path(daemon_config_path)
    else:
        discovered = discover_daemon_config()
        if discovered:
            cfg_path = Path(discovered)
    if cfg_path is not None:
        cfg = load_daemon_config(cfg_path)
        listen_host = (cfg.server.listen.host or "").strip().lower()
        if listen_host and listen_host not in {"127.0.0.1", "localhost"}:
            restrict_localhost = False
        if cfg.server.HasField("p2p_listen"):
            p2p_host = cfg.server.p2p_listen.host.strip()
            p2p_port = int(cfg.server.p2p_listen.port)
            if p2p_host or p2p_port:
                restrict_localhost = False
    return cfg_path, cfg, restrict_localhost


def _connect_context(
    *,
    target_address: str,
    install_signal_handlers: bool,
    fate_share_sigterm: bool,
    logger,
) -> Context:
    global _current_ctx
    if not target_address:
        raise RuntimeError("Missing daemon address for connect mode.")
    _wait_for_daemon_rpc_ready_or_raise(
        address=target_address,
        timeout_s=_auto_wait_timeout_seconds(),
        context="connect",
    )
    set_daemon_address(target_address)
    from tensorcast.daemon_ctl import get_daemon_client

    client = get_daemon_client(target_address)
    logger.info("✅ Connected to daemon at %s", target_address)
    ctx = Context(
        address=target_address,
        is_owner=False,
        session_id=None,
        session_dir=None,
        client=client,
    )
    _current_ctx = ctx
    if install_signal_handlers:
        ctx.install_signal_handlers("hard-exit" if fate_share_sigterm else "graceful")
    _register_atexit_if_needed()
    return ctx


def _start_context(
    *,
    cfg_path: Path | None,
    cfg: Any | None,
    restrict_localhost: bool,
    show_daemon_logs: bool,
    install_signal_handlers: bool,
    fate_share_sigterm: bool,
    global_store_mode: Literal["connect", "start", "none"],
    global_store_address: str | None,
    global_store_config_path: str | None,
    cluster_id: str | None,
    allow_gs_fallback: bool,
    session_id: str | None,
    port_config: PortConfig | None,
    reuse_existing: bool,
    logger,
) -> Context:
    global _current_ctx
    session_obj = runtime.start(
        daemon_config=cfg_path,
        session_id=session_id,
        global_store_mode=global_store_mode,
        global_store_address=global_store_address,
        global_store_config=(
            Path(global_store_config_path).expanduser()
            if global_store_config_path
            else None
        ),
        allow_gs_fallback=allow_gs_fallback,
        cluster_id=cluster_id,
        register_current=session_id is None,
        ephemeral=session_id is not None,
        restrict_to_localhost=restrict_localhost,
        listen_port=(
            port_config.daemon_listen_port if port_config is not None else None
        ),
        p2p_listen_port=(
            port_config.daemon_p2p_port if port_config is not None else None
        ),
        global_store_listen_port=(
            port_config.global_store_listen_port if port_config is not None else None
        ),
        global_store_metrics_port=(
            port_config.global_store_metrics_port if port_config is not None else None
        ),
        to_console=show_daemon_logs,
        reuse_existing=reuse_existing,
    )

    daemon_address = session_obj.daemon_address or _current_session_address()
    if not daemon_address and cfg is not None:
        with contextlib.suppress(Exception):
            host = cfg.server.listen.host or "127.0.0.1"
            port = (
                port_config.daemon_listen_port
                if port_config is not None
                and port_config.daemon_listen_port is not None
                else int(cfg.server.listen.port or 0)
            )
            daemon_address = f"{host}:{port}"
    if not daemon_address:
        daemon_address = "127.0.0.1:0"
    _wait_for_daemon_rpc_ready_or_raise(
        address=daemon_address,
        timeout_s=_auto_wait_timeout_seconds(),
        context="start",
    )
    set_daemon_address(daemon_address)
    from tensorcast.daemon_ctl import get_daemon_client

    client = get_daemon_client(daemon_address)
    logger.info(
        "✅ tensorcast initialized; daemon at %s (session=%s)",
        daemon_address,
        session_obj.session_id,
    )
    ctx = Context(
        address=daemon_address,
        is_owner=True,
        session_id=session_obj.session_id,
        session_dir=str(session_paths(session_obj.session_id).session),
        client=client,
    )
    _current_ctx = ctx
    if install_signal_handlers:
        ctx.install_signal_handlers("hard-exit" if fate_share_sigterm else "graceful")
    _register_atexit_if_needed()
    return ctx


def _init_auto_mode(
    *,
    cfg_path: Path | None,
    cfg: Any | None,
    restrict_localhost: bool,
    address: str | None,
    show_daemon_logs: bool,
    install_signal_handlers: bool,
    fate_share_sigterm: bool,
    global_store_mode: Literal["connect", "start", "none"],
    global_store_address: str | None,
    global_store_config_path: str | None,
    cluster_id: str | None,
    allow_gs_fallback: bool,
    session_id: str | None,
    port_config: PortConfig | None,
    logger,
) -> Context:
    if address and address not in {"auto", "local"}:
        raise ValueError(
            "init(mode='auto') does not accept explicit 'address'. "
            "Use mode='connect' to attach to an existing daemon."
        )
    expected_hash = _compute_auto_config_hash(
        daemon_config_path=cfg_path,
        global_store_mode=global_store_mode,
        global_store_address=global_store_address,
        global_store_config_path=global_store_config_path,
        cluster_id=cluster_id,
        allow_gs_fallback=allow_gs_fallback,
        session_id=session_id,
        port_config=port_config,
    )
    timeout_s = _auto_wait_timeout_seconds()
    while True:
        existing = runtime.status()
        existing_address = (
            existing.daemon_address
            if existing
            and existing.daemon_address
            and ping_daemon(existing.daemon_address)
            else None
        )
        next_action = "wait"
        connect_address: str | None = None
        leader_state: dict[str, object] | None = None
        leader_session_id: str | None = None
        reset_state = False
        with file_lock(runtime_lock_path()):
            state = _read_auto_state_locked()
            status = str(state.get("status", _AUTO_STATUS_EMPTY)).upper()
            state_address_raw = state.get("address")
            state_address = (
                str(state_address_raw).strip()
                if isinstance(state_address_raw, str)
                else ""
            )
            state_hash_raw = state.get("config_hash")
            state_hash = (
                str(state_hash_raw).strip() if isinstance(state_hash_raw, str) else ""
            )
            enforce_hash = status in {
                _AUTO_STATUS_STARTING,
                _AUTO_STATUS_LISTENING,
            } or (
                status == _AUTO_STATUS_READY
                and bool(existing_address)
                and bool(state_address)
                and state_address == existing_address
            )
            if enforce_hash and state_hash and state_hash != expected_hash:
                hash_owner_pid = _coerce_int(state.get("owner_pid"), default=0)
                if not _owner_alive(hash_owner_pid, state.get("owner_fingerprint")):
                    reset_state = True
                    _write_auto_state_locked(_default_auto_state())
                    next_action = "reset"
                else:
                    raise _auto_error(
                        code="AUTO_CONFIG_MISMATCH",
                        reason="auto init configuration hash differs from active state",
                        state=state,
                    )
            if next_action == "reset":
                pass
            elif status == _AUTO_STATUS_FAILED:
                if existing_address:
                    connect_address = existing_address
                    next_action = "connect"
                else:
                    owner_pid = _coerce_int(state.get("owner_pid"), default=0)
                    if owner_pid > 0 and not _owner_alive(
                        owner_pid, state.get("owner_fingerprint")
                    ):
                        reset_state = True
                        _write_auto_state_locked(_default_auto_state())
                        next_action = "reset"
                    else:
                        raise _auto_error(
                            code="AUTO_START_FAILED",
                            reason=str(
                                state.get("error_message", "owner reported failure")
                            ),
                            state=state,
                        )
            elif status in {_AUTO_STATUS_LISTENING, _AUTO_STATUS_READY}:
                owner_pid = _coerce_int(state.get("owner_pid"), default=0)
                if owner_pid > 0 and not _owner_alive(
                    owner_pid, state.get("owner_fingerprint")
                ):
                    reset_state = True
                    _write_auto_state_locked(_default_auto_state())
                    next_action = "reset"
                else:
                    listening_address = state.get("address")
                    if isinstance(listening_address, str) and listening_address:
                        connect_address = listening_address
                        next_action = "connect"
                    elif existing_address:
                        connect_address = existing_address
                        next_action = "connect"
                    else:
                        raise _auto_error(
                            code="AUTO_LISTENING_INVALID",
                            reason="listening state is missing daemon address",
                            state=state,
                        )
            elif status == _AUTO_STATUS_STARTING:
                owner_pid = _coerce_int(state.get("owner_pid"), default=0)
                if not _owner_alive(owner_pid, state.get("owner_fingerprint")):
                    reset_state = True
                    _write_auto_state_locked(_default_auto_state())
                    next_action = "reset"
                else:
                    started_at = _coerce_float(state.get("started_at"), default=0.0)
                    elapsed = (
                        max(0.0, time.time() - started_at) if started_at > 0 else 0.0
                    )
                    if elapsed > timeout_s:
                        raise _auto_error(
                            code="AUTO_START_TIMEOUT",
                            reason="waiting for leader startup timed out",
                            state=state,
                            wait_s=elapsed,
                        )
                    if existing_address:
                        connect_address = existing_address
                        next_action = "connect"
                    else:
                        next_action = "wait"
            else:
                if existing_address:
                    connect_address = existing_address
                    next_action = "connect"
                else:
                    leader_session_id = session_id
                    epoch = _coerce_int(state.get("epoch"), default=0) + 1
                    leader_state = {
                        "schema_version": _AUTO_STATE_SCHEMA_VERSION,
                        "status": _AUTO_STATUS_STARTING,
                        "epoch": epoch,
                        "owner_pid": os.getpid(),
                        "owner_fingerprint": instance_fingerprint(),
                        "config_hash": expected_hash,
                        "session_id": leader_session_id or "",
                        "started_at": time.time(),
                        "address": "",
                        "logs_dir": "",
                        "error_code": "",
                        "error_message": "",
                    }
                    _write_auto_state_locked(leader_state)
                    next_action = "create"

        if next_action == "reset":
            if reset_state:
                logger.info(
                    "Auto init cleared stale daemon auto state; retrying election"
                )
            time.sleep(_AUTO_POLL_INTERVAL_SECONDS)
            continue

        if next_action == "connect":
            assert connect_address is not None
            if ping_daemon(connect_address):
                return _connect_context(
                    target_address=connect_address,
                    install_signal_handlers=install_signal_handlers,
                    fate_share_sigterm=fate_share_sigterm,
                    logger=logger,
                )
            with file_lock(runtime_lock_path()):
                latest_state = _read_auto_state_locked()
                latest_status = str(
                    latest_state.get("status", _AUTO_STATUS_EMPTY)
                ).upper()
                latest_owner_pid = _coerce_int(latest_state.get("owner_pid"), default=0)
                latest_owner_alive = _owner_alive(
                    latest_owner_pid, latest_state.get("owner_fingerprint")
                )
            if latest_status == _AUTO_STATUS_STARTING:
                time.sleep(_AUTO_POLL_INTERVAL_SECONDS)
                continue
            latest_address_raw = latest_state.get("address")
            latest_address = (
                str(latest_address_raw).strip()
                if isinstance(latest_address_raw, str)
                else ""
            )
            if (
                latest_status
                in {
                    _AUTO_STATUS_LISTENING,
                    _AUTO_STATUS_READY,
                    _AUTO_STATUS_FAILED,
                }
                and latest_address
                and latest_address == connect_address
                and not latest_owner_alive
            ):
                with file_lock(runtime_lock_path()):
                    refreshed = _read_auto_state_locked()
                    refreshed_status = str(
                        refreshed.get("status", _AUTO_STATUS_EMPTY)
                    ).upper()
                    refreshed_address_raw = refreshed.get("address")
                    refreshed_address = (
                        str(refreshed_address_raw).strip()
                        if isinstance(refreshed_address_raw, str)
                        else ""
                    )
                    refreshed_owner_pid = _coerce_int(
                        refreshed.get("owner_pid"), default=0
                    )
                    if (
                        refreshed_status
                        in {
                            _AUTO_STATUS_LISTENING,
                            _AUTO_STATUS_READY,
                            _AUTO_STATUS_FAILED,
                        }
                        and refreshed_address == connect_address
                        and not _owner_alive(
                            refreshed_owner_pid, refreshed.get("owner_fingerprint")
                        )
                    ):
                        _write_auto_state_locked(_default_auto_state())
                time.sleep(_AUTO_POLL_INTERVAL_SECONDS)
                continue
            raise RuntimeError(f"No daemon found at {connect_address}")

        if next_action == "create":
            assert leader_state is not None
            logger.info(
                "Auto init elected leader pid=%s for daemon session=%s",
                os.getpid(),
                leader_session_id or "<auto>",
            )
            try:
                ctx = _start_context(
                    cfg_path=cfg_path,
                    cfg=cfg,
                    restrict_localhost=restrict_localhost,
                    show_daemon_logs=show_daemon_logs,
                    install_signal_handlers=install_signal_handlers,
                    fate_share_sigterm=fate_share_sigterm,
                    global_store_mode=global_store_mode,
                    global_store_address=global_store_address,
                    global_store_config_path=global_store_config_path,
                    cluster_id=cluster_id,
                    allow_gs_fallback=allow_gs_fallback,
                    session_id=leader_session_id,
                    port_config=port_config,
                    reuse_existing=False,
                    logger=logger,
                )
            except Exception as exc:
                with file_lock(runtime_lock_path()):
                    failed = dict(leader_state)
                    failed["status"] = _AUTO_STATUS_FAILED
                    failed["error_code"] = "AUTO_START_FAILED"
                    failed["error_message"] = str(exc)
                    failed["logs_dir"] = (
                        str(session_paths(leader_session_id).logs)
                        if leader_session_id
                        else ""
                    )
                    _write_auto_state_locked(failed)
                raise
            with file_lock(runtime_lock_path()):
                ready = dict(leader_state)
                ready["status"] = _AUTO_STATUS_READY
                ready["address"] = ctx.address
                ready["session_id"] = ctx.session_id or ""
                ready["logs_dir"] = (
                    str(session_paths(ctx.session_id).logs) if ctx.session_id else ""
                )
                ready["error_code"] = ""
                ready["error_message"] = ""
                _write_auto_state_locked(ready)
            return ctx

        time.sleep(_AUTO_POLL_INTERVAL_SECONDS)


def init(
    *,
    mode: Literal["connect", "create", "auto"],
    address: str | None = None,
    daemon_config_path: str | None = None,
    install_signal_handlers: bool = False,
    fate_share_sigterm: bool = False,
    show_daemon_logs: bool = True,
    global_store_mode: Literal["connect", "start", "none"] = "none",
    global_store_address: str | None = None,
    global_store_config_path: str | None = None,
    cluster_id: str | None = None,
    allow_gs_fallback: bool = False,
    session_id: str | None = None,
    port_config: PortConfig | None = None,
) -> Context:
    """Launch or connect to a Store Daemon and set global address.

    Usage patterns:
    - Connect: `init(mode="connect", address="host:port")`
    - Connect (auto-discover local): `init(mode="connect")`
    - Launch:  `init(mode="create", global_store_mode="start", daemon_config_path="/path/to/config.yaml")`
    - Auto:    `init(mode="auto")` (connect if one exists, otherwise one process creates and others wait)

    When launching, the SDK will pick a config in the following order:
    1) user-provided `daemon_config_path`
    2) discovered default via `discover_default_config_path()` (env or
       examples/config/store_daemon_config.yaml when available)
    3) error if no config is found.

    Args:
        mode: Required init mode. Use "connect" to attach to an existing daemon,
            "create" to launch a new daemon owned by the current process,
            or "auto" for process-group singleflight connect-or-create.
        address: Optional target daemon address (connect-only).
        daemon_config_path: Config file when launching a local daemon.
        install_signal_handlers: Install SIGTERM/SIGINT handlers on success.
        fate_share_sigterm: Force hard exit on signals when handlers installed.
        show_daemon_logs: Mirror daemon stdout/stderr to the current console when
            launching locally.
        global_store_mode: connect|start|none orchestration mode shared with CLI.
            Applies only when launching daemon (mode="create"|"auto"). In
            start mode, startup creates a new local Global Store and fails only
            if a healthy local Global Store is already recorded for the current
            runtime root. Stale local state does not block startup; a fresh
            local Global Store is created with a new token unless `cluster_id`
            explicitly pins the identity.
        global_store_address: Optional Global Store host:port to connect.
            Applies only when launching daemon (mode="create"|"auto").
        global_store_config_path: Optional Global Store config path used when
            global_store_mode="start". Applies only when launching daemon
            (mode="create"|"auto"). If omitted, the Global Store launcher
            discovers one via $TENSORCAST_GLOBAL_STORE_CONFIG or
            examples/config/global_store_config.yaml (repo or packaged wheel).
        cluster_id: Optional cluster identity to enforce when connecting/starting GS.
        allow_gs_fallback: If True, start mode may fall back to none when GS
            startup itself fails.
        session_id: Optional explicit daemon session id (treated as private/ephemeral).
        port_config: Optional port overrides for SDK-managed daemon / Global
            Store launch. Supports daemon listen, daemon P2P, Global Store
            listen, and Global Store metrics ports. Applies only when launching
            locally (`mode="create"|"auto"`).
    """
    global _current_ctx, _atexit_registered, _ctx_lock

    logger = init_logger(__name__)

    # Do the entire init under the lock to avoid concurrent launches.
    with _ctx_lock:
        # If already initialized and context still open, return it
        if _current_ctx is not None and not _current_ctx._closed:  # noqa: SLF001
            return _current_ctx

        if mode == "connect":
            # Connect path: never start/stop daemon; just bind to an existing session.
            if (
                global_store_mode != "none"
                or global_store_address is not None
                or global_store_config_path is not None
                or port_config is not None
            ):
                logger.warning(
                    "init(mode='connect') does not reconfigure daemon Global Store settings; "
                    "global_store_mode/global_store_address/global_store_config_path/port_config are ignored. "
                    "Set Global Store when creating/starting daemon via "
                    "tc.init(mode='create'|'auto', ...) or "
                    "`uv run tensorcast-cli daemon start ...`."
                )
            target_address: str | None = None
            if address and address not in {"auto", "local"}:
                target_address = address
            else:
                target_address = _current_session_address()
            if not target_address:
                raise RuntimeError(
                    "No local daemon session found. Start one with "
                    "'tensorcast daemon start' or call "
                    "tensorcast.startup.init(mode='auto'|'create', ...)."
                )
            return _connect_context(
                target_address=target_address,
                install_signal_handlers=install_signal_handlers,
                fate_share_sigterm=fate_share_sigterm,
                logger=logger,
            )

        if mode not in {"create", "auto"}:
            raise ValueError(f"Unknown init mode: {mode}")

        if mode == "create" and address and address not in {"auto", "local"}:
            raise ValueError(
                "init(mode='create') does not accept 'address'. "
                "Use mode='connect' to attach to an existing daemon."
            )

        port_config = _normalize_port_config(
            port_config=port_config,
            global_store_mode=global_store_mode,
            logger=logger,
        )
        cfg_path, cfg, restrict_localhost = _resolve_launch_config(daemon_config_path)
        if mode == "auto":
            return _init_auto_mode(
                cfg_path=cfg_path,
                cfg=cfg,
                restrict_localhost=restrict_localhost,
                address=address,
                show_daemon_logs=show_daemon_logs,
                install_signal_handlers=install_signal_handlers,
                fate_share_sigterm=fate_share_sigterm,
                global_store_mode=global_store_mode,
                global_store_address=global_store_address,
                global_store_config_path=global_store_config_path,
                cluster_id=cluster_id,
                allow_gs_fallback=allow_gs_fallback,
                session_id=session_id,
                port_config=port_config,
                logger=logger,
            )
        return _start_context(
            cfg_path=cfg_path,
            cfg=cfg,
            restrict_localhost=restrict_localhost,
            show_daemon_logs=show_daemon_logs,
            install_signal_handlers=install_signal_handlers,
            fate_share_sigterm=fate_share_sigterm,
            global_store_mode=global_store_mode,
            global_store_address=global_store_address,
            global_store_config_path=global_store_config_path,
            cluster_id=cluster_id,
            allow_gs_fallback=allow_gs_fallback,
            session_id=session_id,
            port_config=port_config,
            reuse_existing=False,
            logger=logger,
        )


def is_initialized() -> bool:
    """Return True if a runtime Context has been established and not closed."""
    return _current_ctx is not None and not _current_ctx._closed  # noqa: SLF001


def require_initialized() -> Context:
    ctx = _current_ctx
    if ctx is None or ctx._closed:
        raise RuntimeError(
            "TensorCast runtime is not initialized. "
            "Call tensorcast.startup.init(mode='connect'|'create'|'auto') first."
        )
    return ctx


def current_client() -> DaemonCtl:
    return require_initialized().client


def shutdown() -> None:
    """Unified shutdown API to close the current daemon session context."""
    global _current_ctx
    ctx = _current_ctx
    if ctx is None:
        return
    ctx.close()
    _current_ctx = None
    with contextlib.suppress(Exception):
        from tensorcast.observability.otel import shutdown_otel

        shutdown_otel()


def init_from_client_config(config_path: str | Path | None) -> None:
    """Initialize client behavior from ClientConfig file (YAML/JSON).

    If *config_path* is None, the loader searches $TENSORCAST_CLIENT_CONFIG and
    ~/.tensorcast/config/client.{yaml,yml,json}. Sets default storage root,
    daemon target, and client load defaults. Also applies logging level from
    config if provided.
    """

    resolved_path: Path | None = (
        Path(config_path).expanduser()
        if config_path is not None
        else discover_client_config()
    )
    if resolved_path is None:
        raise RuntimeError(
            "No client config found. Provide a path or set $TENSORCAST_CLIENT_CONFIG."
        )

    cfg = load_client_config(str(resolved_path))
    set_client_config(cfg)

    # Apply daemon target default if present
    target = daemon_target_default()
    runtime_addr = _current_session_address()
    chosen_addr = target or runtime_addr
    if chosen_addr:
        set_daemon_address(chosen_addr)

    # Apply logging level if provided in config
    with contextlib.suppress(Exception):
        level_map = {1: "DEBUG", 2: "INFO", 3: "WARN", 4: "ERROR"}
        lvl_num = cfg.observability.logging.level
        if lvl_num in level_map:
            setup_logging(level_map[lvl_num])


__all__ = [
    "Context",
    "PortConfig",
    "init",
    "shutdown",
    "is_initialized",
    "require_initialized",
    "current_client",
    "init_from_client_config",
]
