#  Copyright (c) 2025-2026, TensorCast Team.

"""Shared config loading/override helpers for Global Store launchers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from tensorcast.cli_utils.config import discover_global_store_config
from tensorcast.global_store.config import GlobalStoreConfig
from tensorcast.proto.config.v1 import global_store_config_pb2 as gsc_pb


def resolve_global_store_config_path(config_path: str | Path | None) -> Path:
    """Resolve explicit/default Global Store config path."""

    if config_path is not None:
        return Path(config_path).expanduser()
    discovered = discover_global_store_config()
    if discovered is not None:
        return Path(discovered).expanduser()
    raise FileNotFoundError(
        "No Global Store config found. Provide --config or set "
        "TENSORCAST_GLOBAL_STORE_CONFIG. Expected examples/config/global_store_config.yaml "
        "to be available in the repo or packaged wheel."
    )


def _copy_config_with_updates(
    config: GlobalStoreConfig, updates: dict[str, object]
) -> GlobalStoreConfig:
    if not updates:
        return config
    copy_fn = getattr(config, "model_copy", None)
    if callable(copy_fn):
        return cast(GlobalStoreConfig, copy_fn(update=updates))
    return cast(GlobalStoreConfig, config.copy(update=updates))


def apply_global_store_overrides(
    config: GlobalStoreConfig,
    pb_cfg: gsc_pb.GlobalStoreConfig,
    *,
    listen_host: str | None = None,
    listen_port: int | None = None,
    metrics_port: int | None = None,
) -> tuple[GlobalStoreConfig, gsc_pb.GlobalStoreConfig]:
    """Apply launcher-level listen/metrics overrides consistently."""

    updates: dict[str, object] = {}
    if listen_host is not None:
        updates["listen_host"] = listen_host
        pb_cfg.server.listen.host = listen_host
    if listen_port is not None:
        normalized_listen_port = max(0, int(listen_port))
        updates["listen_port"] = normalized_listen_port
        pb_cfg.server.listen.port = normalized_listen_port
    if metrics_port is not None:
        normalized_metrics_port = max(0, int(metrics_port))
        updates["metrics_port"] = normalized_metrics_port
        pb_cfg.server.metrics_port = normalized_metrics_port
    return _copy_config_with_updates(config, updates), pb_cfg


def load_global_store_config_with_overrides(
    config_path: str | Path | None,
    *,
    listen_host: str | None = None,
    listen_port: int | None = None,
    metrics_port: int | None = None,
) -> tuple[GlobalStoreConfig, gsc_pb.GlobalStoreConfig]:
    """Load strict config + proto and apply shared runtime overrides."""

    path_str = str(resolve_global_store_config_path(config_path))
    config = GlobalStoreConfig.from_file(path_str)
    pb_cfg = GlobalStoreConfig.load_proto_from_file(path_str)
    return apply_global_store_overrides(
        config,
        pb_cfg,
        listen_host=listen_host,
        listen_port=listen_port,
        metrics_port=metrics_port,
    )


def apply_global_store_proto_defaults(
    pb_cfg: gsc_pb.GlobalStoreConfig,
    *,
    default_listen_host: str | None = None,
    default_log_file: str | None = None,
    default_schema_version: str | None = None,
    default_description: str | None = None,
    cluster_token: str | None = None,
) -> None:
    """Fill launcher-managed proto defaults used for effective configs."""

    if default_listen_host and not pb_cfg.server.listen.host:
        pb_cfg.server.listen.host = default_listen_host
    if default_log_file and not pb_cfg.observability.logging.file:
        pb_cfg.observability.logging.file = default_log_file
    if default_schema_version and not pb_cfg.meta.schema_version:
        pb_cfg.meta.schema_version = default_schema_version
    if default_description and not pb_cfg.meta.description:
        pb_cfg.meta.description = default_description
    if cluster_token is not None:
        pb_cfg.meta.cluster_token = cluster_token


__all__ = [
    "apply_global_store_overrides",
    "apply_global_store_proto_defaults",
    "load_global_store_config_with_overrides",
    "resolve_global_store_config_path",
]
