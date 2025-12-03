#  Copyright (c) 2025, TensorCast Team.

"""
Runtime startup and connection utilities for TensorCast.

- Provides a unified `init()` entrypoint that either connects to an existing
  Store Daemon or launches one locally and connects to it.
- Returns a Context object whose lifecycle is process-scoped. The context is
  automatically closed at process exit via atexit when we launched the daemon
  (or connected). Use `shutdown()` to close early if needed. Context manager
  semantics are not required or supported for this object.

This supersedes the older tensorcast.config module. Use this module directly
and rely on `tensorcast.config` only for backward compatibility.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tensorcast.api._config import clear_daemon_address, set_daemon_address
from tensorcast.cli_utils.resolve import ping_daemon, resolve_address_mode
from tensorcast.cli_utils.service_manager import (
    discover_default_config_path,
    start_service,
)
from tensorcast.client_config_loader import load_client_config
from tensorcast.client_runtime import daemon_target_default, set_client_config
from tensorcast.daemon_ctl import (
    DaemonCtl,
    get_daemon_client,
    release_daemon_client,
)
from tensorcast.daemon_runtime_config import dump_daemon_config, load_daemon_config
from tensorcast.logger import init_logger, setup_logging
from tensorcast.proto.config.v1 import daemon_config_pb2 as cfg_pb

_current_ctx: Context | None = None
_atexit_registered = False
_ctx_lock: "threading.Lock" = threading.Lock()
_EMBEDDED_CONFIG_PATH: Path | None = None


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
                release_daemon_client(self.address)
            self._client_released = True
        if self.is_owner and self.session_id:
            # Stop daemon session we launched
            from tensorcast.cli_utils.service_manager import stop_service

            stop_service(session_id=self.session_id)
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
    from tensorcast.cli_utils.service_manager import get_session_address

    return get_session_address()


def _select_cache_root() -> Path:
    preferred = Path.home() / ".tensorcast"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except Exception:
        return Path(tempfile.mkdtemp(prefix="tensorcast-cache-"))


def _build_embedded_daemon_config(
    storage_dir: Path, fallback_dir: Path
) -> cfg_pb.DaemonConfig:
    cfg = cfg_pb.DaemonConfig()
    cfg.server.listen.host = "127.0.0.1"
    cfg.server.listen.port = 0
    cfg.server.storage_path = str(storage_dir)
    cfg.server.num_threads = max(4, min(8, os.cpu_count() or 4))

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
    cfg.observability.logging.level = cfg_pb.Observability.LOG_LEVEL_INFO
    cfg.meta.description = "Embedded default daemon config (tensorcast.startup)"
    return cfg


def _ensure_embedded_daemon_config_path() -> Path:
    global _EMBEDDED_CONFIG_PATH
    if _EMBEDDED_CONFIG_PATH and _EMBEDDED_CONFIG_PATH.exists():
        return _EMBEDDED_CONFIG_PATH

    cache_root = _select_cache_root()
    storage_dir = cache_root / "cache"
    fallback_dir = cache_root / "p2p_cache"
    storage_dir.mkdir(parents=True, exist_ok=True)
    fallback_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_embedded_daemon_config(storage_dir, fallback_dir)
    cfg_path = cache_root / "embedded_store_daemon.yaml"
    dump_daemon_config(cfg, cfg_path)
    _EMBEDDED_CONFIG_PATH = cfg_path
    return cfg_path


def init(
    *,
    address: str | None = None,
    daemon_config_path: str | None = None,
    wait: bool = True,
    timeout: float = 20.0,
    install_signal_handlers: bool = False,
    fate_share_sigterm: bool = False,
    show_daemon_logs: bool = True,
) -> Context:
    """Launch or connect to a Store Daemon and set global address.

    Usage patterns:
    - Connect: `init(address="host:port" | "auto" | "local")`
    - Launch:  `init(address="local", daemon_config_path="/path/to/config.yaml")`

    When launching, the SDK will pick a config in the following order:
    1) user-provided `daemon_config_path`
    2) discovered default via `discover_default_config_path()`
    3) embedded minimal config (loopback bind, ephemeral port, writable cache
       under ~/.tensorcast or a tempdir).
    If `address` is None and there is a current session, connects to it;
    otherwise `address="local"` launches a daemon with the resolved config.

    Args:
        address: Target daemon address or resolution mode.
        daemon_config_path: Config file when launching a local daemon.
        wait: Wait for startup readiness when launching.
        timeout: Startup readiness timeout in seconds.
        install_signal_handlers: Install SIGTERM/SIGINT handlers on success.
        fate_share_sigterm: Force hard exit on signals when handlers installed.
        show_daemon_logs: Mirror daemon stdout/stderr to the current console when
            launching locally. The SDK previously suppressed logs; defaulting to
            True now preserves full visibility during development.
    """
    global _current_ctx, _atexit_registered, _ctx_lock

    logger = init_logger(__name__)

    # Do the entire init under the lock to avoid concurrent launches.
    with _ctx_lock:
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
            client = get_daemon_client(resolved_addr)
            logger.info("✅ Connected to daemon at %s", resolved_addr)
            ctx = Context(
                address=resolved_addr,
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

        # Launch requires a config file (try discovery if not provided)
        used_embedded_config = False
        if not daemon_config_path:
            cfg_path_opt = discover_default_config_path()
            if not cfg_path_opt:
                cfg_path = _ensure_embedded_daemon_config_path()
                used_embedded_config = True
            else:
                cfg_path = Path(cfg_path_opt)
        else:
            cfg_path = Path(daemon_config_path)

        cfg = load_daemon_config(cfg_path)
        if used_embedded_config:
            logger.info("ℹ️ Using embedded default daemon config at %s", cfg_path)
        restrict_localhost = True
        if cfg.server.HasField("p2p_listen"):
            p2p_host = cfg.server.p2p_listen.host.strip()
            p2p_port = int(cfg.server.p2p_listen.port)
            if p2p_host or p2p_port:
                restrict_localhost = False

        # Private launch for SDK: do not publish meta/current_session; restrict to localhost unless P2P is configured
        inst = start_service(
            config_path=cfg_path,
            session_id=None,
            blocking=False,
            to_console=show_daemon_logs,
            wait=wait,
            timeout=timeout,
            register_current=False,
            publish_meta=False,
            restrict_to_localhost=restrict_localhost,
        )

        daemon_address = inst.address or _current_session_address()
        if not daemon_address:
            # Fallback: read original config (may be wrong when port was 0)
            host = cfg.server.listen.host or "127.0.0.1"
            port = int(cfg.server.listen.port or 0)
            daemon_address = f"{host}:{port}"
        set_daemon_address(daemon_address)
        client = get_daemon_client(daemon_address)
        logger.info(
            "✅ tensorcast initialized; daemon at %s (session=%s)",
            daemon_address,
            inst.id,
        )

        ctx = Context(
            address=daemon_address,
            is_owner=True,
            session_id=inst.id,
            session_dir=str(inst.session),
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
            "TensorCast runtime is not initialized. Call tensorcast.startup.init() first."
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


def init_from_client_config(config_path: str) -> None:
    """Initialize client behavior from ClientConfig file (YAML/JSON).

    Sets default storage root, daemon target, and client load defaults.
    Also applies logging level from config if provided.
    """

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
    "shutdown",
    "is_initialized",
    "require_initialized",
    "current_client",
    "init_from_client_config",
]
