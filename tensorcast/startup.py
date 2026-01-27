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
import os
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from tensorcast import runtime
from tensorcast.api._config import clear_daemon_address, set_daemon_address
from tensorcast.cli_utils.config import discover_daemon_config
from tensorcast.cli_utils.health import ping_daemon
from tensorcast.cli_utils.paths import session_paths
from tensorcast.client_config_loader import discover_client_config, load_client_config
from tensorcast.client_runtime import daemon_target_default, set_client_config

if TYPE_CHECKING:
    from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.daemon_runtime_config import load_daemon_config
from tensorcast.logger import init_logger, setup_logging

_current_ctx: Context | None = None
_atexit_registered = False
_ctx_lock: "threading.Lock" = threading.Lock()


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
            # Stop daemon session we launched
            runtime.stop(session_id=self.session_id)
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


def init(
    *,
    mode: Literal["connect", "create"],
    address: str | None = None,
    daemon_config_path: str | None = None,
    install_signal_handlers: bool = False,
    fate_share_sigterm: bool = False,
    show_daemon_logs: bool = True,
    global_store_mode: Literal["connect", "start", "none"] = "none",
    global_store_address: str | None = None,
    cluster_id: str | None = None,
    allow_gs_fallback: bool = False,
    session_id: str | None = None,
) -> Context:
    """Launch or connect to a Store Daemon and set global address.

    Usage patterns:
    - Connect: `init(mode="connect", address="host:port")`
    - Connect (auto-discover local): `init(mode="connect")`
    - Launch:  `init(mode="create", global_store_mode="start", daemon_config_path="/path/to/config.yaml")`

    When launching, the SDK will pick a config in the following order:
    1) user-provided `daemon_config_path`
    2) discovered default via `discover_default_config_path()` (env or
       examples/config/store_daemon_config.yaml when available)
    3) error if no config is found.

    Args:
        mode: Required init mode. Use "connect" to attach to an existing daemon,
            "create" to launch a new daemon owned by the current process.
        address: Optional target daemon address (connect-only).
        daemon_config_path: Config file when launching a local daemon.
        install_signal_handlers: Install SIGTERM/SIGINT handlers on success.
        fate_share_sigterm: Force hard exit on signals when handlers installed.
        show_daemon_logs: Mirror daemon stdout/stderr to the current console when
            launching locally.
        global_store_mode: connect|start|none orchestration mode shared with CLI.
        global_store_address: Optional Global Store host:port to connect.
        cluster_id: Optional cluster identity to enforce when connecting/starting GS.
        allow_gs_fallback: If True, start mode may fall back to none on GS failure.
        session_id: Optional explicit daemon session id (treated as private/ephemeral).
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
            target_address: str | None = None
            if address and address not in {"auto", "local"}:
                target_address = address
            else:
                target_address = _current_session_address()
            if not target_address:
                raise RuntimeError(
                    "No local daemon session found. Start one with "
                    "'tensorcast daemon start' or call "
                    "tensorcast.startup.init(mode='create', ...)."
                )
            if not ping_daemon(target_address):
                raise RuntimeError(f"No daemon found at {target_address}")
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
                ctx.install_signal_handlers(
                    "hard-exit" if fate_share_sigterm else "graceful"
                )
            if not _atexit_registered:
                atexit.register(lambda: _current_ctx and _current_ctx.close())  # type: ignore[func-returns-value]
                _atexit_registered = True
            return ctx

        if mode != "create":
            raise ValueError(f"Unknown init mode: {mode}")

        if address and address not in {"auto", "local"}:
            raise ValueError(
                "init(mode='create') does not accept 'address'. "
                "Use mode='connect' to attach to an existing daemon."
            )

        # Launch requires a config file (optional; defaults to example config when available).
        cfg_path: Path | None = None
        restrict_localhost = True
        cfg = None
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

        # Private launch for SDK: do not publish meta/current_session when session_id provided
        session_obj = runtime.start(
            daemon_config=cfg_path,
            session_id=session_id,
            global_store_mode=global_store_mode,
            global_store_address=global_store_address,
            allow_gs_fallback=allow_gs_fallback,
            cluster_id=cluster_id,
            register_current=session_id is None,
            ephemeral=session_id is not None,
            restrict_to_localhost=restrict_localhost,
            to_console=show_daemon_logs,
            reuse_existing=False,
        )

        daemon_address = session_obj.daemon_address or _current_session_address()
        if not daemon_address and cfg is not None:
            # Fallback: read original config (may be wrong when port was 0)
            host = cfg.server.listen.host or "127.0.0.1"
            port = int(cfg.server.listen.port or 0)
            daemon_address = f"{host}:{port}"
        if not daemon_address:
            daemon_address = "127.0.0.1:0"
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
            ctx.install_signal_handlers(
                "hard-exit" if fate_share_sigterm else "graceful"
            )
        if not _atexit_registered:
            atexit.register(lambda: _current_ctx and _current_ctx.close())  # type: ignore[func-returns-value]
            _atexit_registered = True
        return ctx


def is_initialized() -> bool:
    """Return True if a runtime Context has been established and not closed."""
    return _current_ctx is not None and not _current_ctx._closed  # noqa: SLF001


def require_initialized() -> Context:
    ctx = _current_ctx
    if ctx is None or ctx._closed:
        raise RuntimeError(
            "TensorCast runtime is not initialized. "
            "Call tensorcast.startup.init(mode='connect'|'create') first."
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
    "init",
    "shutdown",
    "is_initialized",
    "require_initialized",
    "current_client",
    "init_from_client_config",
]
