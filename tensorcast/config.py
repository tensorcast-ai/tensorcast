#  Copyright (c) 2025, TensorCast Team.

"""
Global configuration management for tensorcast.

This module handles the centralized configuration and initialization
of the tensorcast system, including daemon management.
"""

import tempfile
from pathlib import Path

initialized = False


def init(
    storage_path: str = "",
    daemon_host: str = "127.0.0.1",
    daemon_port: int = 8073,
    daemon_mem_pool_size: str = "8GB",
    daemon_num_thread: int = 8,
    daemon_chunk_size: str = "32MB",
    daemon_enable_p2p_access: bool = False,
    daemon_comm_enabled: bool = False,
    auto_start: bool = True,
    connect_only: bool = False,
):
    """
    Initialize tensorcast client and optionally manage daemon lifecycle.

    This function serves as the central configuration point for tensorcast. It can:
    1. Connect to an existing daemon (if connect_only=True)
    2. Auto-manage daemon lifecycle (if auto_start=True, default)
    3. Just set configuration without starting daemon (if auto_start=False)

    Args:
        storage_path: Path to artifact storage directory
        daemon_host: Host address for daemon
        daemon_port: Port for daemon
        daemon_mem_pool_size: Memory pool size (e.g., "4GB", "64GB")
        daemon_num_thread: Number of worker threads
        daemon_chunk_size: Chunk size for memory operations (e.g., "32MB")
        daemon_enable_p2p_access: Whether artifact registration is required
        daemon_comm_enabled: Whether communication engine is enabled
        auto_start: Whether to automatically start daemon if not running
        connect_only: Only connect to existing daemon, don't start new one

    Examples:
        # Auto-manage daemon lifecycle (recommended)
        tensorcast.init(storage_path="./test-models", daemon_mem_pool_size="64GB")

        # Connect to existing daemon only
        tensorcast.init(connect_only=True)

        # Configure but don't start daemon yet
        tensorcast.init(auto_start=False, storage_path="./models")
    """

    # Import here to avoid circular imports
    from tensorcast.api import set_daemon_address
    from tensorcast.daemon_manager import get_daemon_manager
    from tensorcast.daemon_runtime_config import dump_daemon_config
    from tensorcast.logger import init_logger
    from tensorcast.proto.config.v1 import daemon_config_pb2 as cfg_pb
    from tensorcast.utils import to_num_bytes

    logger = init_logger(__name__)

    # No environment overrides in final scheme

    # Programmatically build a minimal DaemonConfig proto and write a temp config file
    cfg = cfg_pb.DaemonConfig()
    cfg.server.listen.host = daemon_host
    cfg.server.listen.port = daemon_port
    cfg.server.storage_path = (
        str(Path(storage_path)) if storage_path else str(Path("/tmp/models"))
    )
    cfg.server.num_threads = daemon_num_thread
    cfg.engine.mem_pool_size_bytes = int(to_num_bytes(daemon_mem_pool_size))
    cfg.engine.chunk_bytes = int(to_num_bytes(daemon_chunk_size))
    # communicator defaults
    cfg.communicator.enable_rdma = bool(daemon_comm_enabled)
    # compatibility block for old toggles mapping
    cfg.compatibility.auto_register_disk_loads = False
    cfg.compatibility.confirm_requires_disk_path = False
    cfg.compatibility.force_full_digest_on_load = False

    # Inline config support: we can now pass JSON/YAML text directly. Keep
    # ability to write a temp file only when connect_only path needs it.
    tmp_path: Path | None = None
    cfg_text = None
    if connect_only:
        with tempfile.NamedTemporaryFile(
            prefix="tc_daemon_", suffix=".yaml", delete=False
        ) as tmp_cfg:
            tmp_path = Path(tmp_cfg.name)
        dump_daemon_config(cfg, tmp_path)
    else:
        # Produce compact JSON for inline CLI flag
        from google.protobuf.json_format import MessageToJson

        cfg_text = MessageToJson(cfg, always_print_fields_with_no_presence=True)

    daemon_address = f"{daemon_host}:{daemon_port}"

    if connect_only:
        # Only try to connect to existing daemon
        logger.info(f"Connecting to existing daemon at {daemon_address}...")
        manager = get_daemon_manager(
            host=daemon_host,
            port=daemon_port,
            auto_start=False,
            config_path=str(tmp_path) if tmp_path else None,
            config_text=cfg_text if not tmp_path else None,
        )

        if manager.is_daemon_running():
            set_daemon_address(daemon_address)
            logger.info("✅ Successfully connected to existing daemon")
            return True
        else:
            logger.error(f"❌ No daemon found at {daemon_address}")
            return False

    elif auto_start:
        # Auto-manage daemon lifecycle
        logger.info("Initializing tensorcast with auto daemon management...")
        # Pass inline config JSON directly; daemon now supports --config_text
        from tensorcast.daemon_manager import ensure_daemon_running as _ensure

        success = _ensure(
            daemon_host,
            daemon_port,
            auto_start=True,
            config_path=None,
            config_text=cfg_text,
        )

        if success:
            set_daemon_address(daemon_address)
            logger.info("✅ tensorcast initialized successfully")
        else:
            logger.error("❌ Failed to initialize tensorcast")
            raise RuntimeError("Failed to initialize tensorcast")

    else:
        # Just set configuration, don't start daemon
        logger.info("tensorcast configured (daemon not started)")
        set_daemon_address(daemon_address)

    global initialized
    initialized = True


def is_initialized() -> bool:
    return initialized


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
    try:
        level_map = {
            1: "DEBUG",
            2: "INFO",
            3: "WARN",
            4: "ERROR",
        }
        lvl_num = cfg.observability.logging.level
        if lvl_num in level_map:
            setup_logging(level_map[lvl_num])
    except Exception:
        pass
