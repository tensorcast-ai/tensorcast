#  Copyright (c) 2025, TensorCast Team.

"""
Runtime startup and connection utilities for TensorCast.

- Provides a unified `init()` entrypoint that either connects to an existing
  Store Daemon or launches one locally and connects to it.
- Returns a Context object that can be used as a context manager to ensure
  cleanup when we launched the daemon.

This supersedes the older tensorcast.config module. Use this module directly
and rely on `tensorcast.config` only for backward compatibility.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
from dataclasses import dataclass
from pathlib import Path

from tensorcast.cli_utils.service_manager import discover_default_config_path

_current_ctx: Context | None = None
_atexit_registered = False


@dataclass(slots=True)
class Context:
    """Runtime context representing a client session against a Store Daemon.

    - If `is_owner` is True, the current process launched the daemon and is
      responsible for stopping it on close().
    - `address` is the gRPC address for the daemon (host:port).
    - `instance_id` and `session_dir` are best-effort identifiers provided by
      the service manager when launching locally.
    """

    address: str
    is_owner: bool
    instance_id: str | None
    session_dir: str | None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        if self.is_owner and self.instance_id:
            # Stop daemon session we launched
            from tensorcast.cli_utils.service_manager import stop_service

            stop_service(instance_id=self.instance_id)
        self._closed = True

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, D401
        self.close()


def _current_instance_address() -> str | None:
    from tensorcast.cli_utils.service_manager import get_instance_address

    return get_instance_address()


def init(
    *,
    address: str | None = None,
    daemon_config_path: str | None = None,
    wait: bool = True,
    timeout: float = 20.0,
) -> Context:
    """Launch or connect to a Store Daemon and set global address.

    Usage patterns:
    - Connect: `init(address="host:port" | "auto" | "local")`
    - Launch:  `init(address="local", daemon_config_path="/path/to/config.yaml")`

    When launching, a daemon config file is required. If `address` is None and
    there is a current session, connects to it; otherwise `address="local"`
    requires `daemon_config_path` (or a discoverable default).
    """
    global _current_ctx, _atexit_registered

    # Import here to avoid circular dependencies and heavy imports at module import time
    from tensorcast.api import set_daemon_address
    from tensorcast.cli_utils.resolve import ping_daemon, resolve_address_mode
    from tensorcast.cli_utils.service_manager import start_service
    from tensorcast.daemon_runtime_config import load_daemon_config
    from tensorcast.logger import init_logger

    logger = init_logger(__name__)

    # Parent SIGTERM: immediate exit to leverage child PDEATHSIG
    def _sigterm_handler(signum, _frame):  # noqa: ANN001
        os._exit(signum)

    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, _sigterm_handler)

    # If already initialized and context still open, return it
    if _current_ctx is not None and not _current_ctx._closed:  # noqa: SLF001
        return _current_ctx

    # Address resolution (shared helper)
    try:
        resolved = resolve_address_mode(address)
        resolved_mode, resolved_addr = resolved.mode, resolved.address
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(str(e)) from e

    if resolved_mode == "connect":
        assert resolved_addr is not None
        if not ping_daemon(resolved_addr):
            raise RuntimeError(f"No daemon found at {resolved_addr}")
        set_daemon_address(resolved_addr)
        logger.info("✅ Connected to daemon at %s", resolved_addr)
        ctx = Context(
            address=resolved_addr, is_owner=False, instance_id=None, session_dir=None
        )
        _current_ctx = ctx
        if not _atexit_registered:
            atexit.register(lambda: _current_ctx and _current_ctx.close())  # type: ignore[func-returns-value]
            _atexit_registered = True
        return ctx

    # Launch requires a config file (try discovery if not provided)
    if not daemon_config_path:
        cfg_path_opt = discover_default_config_path()
        if not cfg_path_opt:
            raise RuntimeError(
                "Launching locally requires daemon_config_path (--config) or a default config at "
                "$TENSORCAST_DAEMON_CONFIG, ~/.tensorcast/store_daemon_config.yaml, or examples/config/store_daemon_config.yaml."
            )
        cfg_path = Path(cfg_path_opt)
    else:
        cfg_path = Path(daemon_config_path)

    # Private launch for SDK: do not publish meta/current_instance; restrict to localhost
    inst = start_service(
        config_path=cfg_path,
        instance_id=None,
        blocking=False,
        to_console=False,
        wait=wait,
        timeout=timeout,
        register_current=False,
        publish_meta=False,
        restrict_to_localhost=True,
    )

    daemon_address = inst.address or _current_instance_address()
    if not daemon_address:
        # Fallback: read original config (may be wrong when port was 0)
        cfg = load_daemon_config(cfg_path)
        host = cfg.server.listen.host or "127.0.0.1"
        port = int(cfg.server.listen.port or 0)
        daemon_address = f"{host}:{port}"
    set_daemon_address(daemon_address)
    logger.info(
        "✅ tensorcast initialized; daemon at %s (session=%s)", daemon_address, inst.id
    )

    ctx = Context(
        address=daemon_address,
        is_owner=True,
        instance_id=inst.id,
        session_dir=str(inst.session),
    )
    _current_ctx = ctx
    if not _atexit_registered:
        atexit.register(lambda: _current_ctx and _current_ctx.close())  # type: ignore[func-returns-value]
        _atexit_registered = True
    return ctx


def is_initialized() -> bool:
    """Return True if a runtime Context has been established and not closed."""
    return _current_ctx is not None and not _current_ctx._closed  # noqa: SLF001


def init_from_client_config(config_path: str) -> None:
    """Initialize client behavior from ClientConfig file (YAML/JSON).

    Sets default storage root, daemon target, and client load defaults.
    Also applies logging level from config if provided.
    """
    from tensorcast.api import set_daemon_address
    from tensorcast.client_config_loader import load_client_config
    from tensorcast.client_runtime import daemon_target_default, set_client_config
    from tensorcast.logger import setup_logging

    cfg = load_client_config(config_path)
    set_client_config(cfg)

    # Apply daemon target default if present
    target = daemon_target_default()
    if target:
        set_daemon_address(target)

    # Apply logging level if provided in config
    with contextlib.suppress(Exception):
        level_map = {1: "DEBUG", 2: "INFO", 3: "WARN", 4: "ERROR"}
        lvl_num = cfg.observability.logging.level
        if lvl_num in level_map:
            setup_logging(level_map[lvl_num])


__all__ = [
    "Context",
    "init",
    "is_initialized",
    "init_from_client_config",
]
