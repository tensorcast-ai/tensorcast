#  Copyright (c) 2025, StepCast Team.

"""Configuration loading and validation utilities."""

from pathlib import Path
from typing import Any

import click
from pydantic import ByteSize

from scstore.store_daemon.config import NetworkConfig, ServerConfig, StoreDaemonConfig
from scstore.utils import to_num_bytes


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


def load_config(
    config_file: Path | None, cli_args: dict[str, Any]
) -> StoreDaemonConfig:
    """
    Load configuration from file or CLI arguments.

    Args:
        config_file: Optional path to YAML configuration file
        cli_args: Dictionary of CLI arguments

    Returns:
        Validated StoreDaemonConfig instance

    Raises:
        ConfigError: If configuration is invalid
    """
    if config_file:
        return _load_from_file(config_file)
    else:
        return _load_from_cli(cli_args)


def _load_from_file(config_file: Path) -> StoreDaemonConfig:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        StoreDaemonConfig instance

    Raises:
        ConfigError: If file cannot be read or parsed
    """
    if not config_file.exists():
        raise ConfigError(f"Configuration file not found: {config_file}")

    try:
        return StoreDaemonConfig.load(config_file)
    except Exception as e:
        raise ConfigError(
            f"Failed to load configuration from {config_file}: {e}"
        ) from e


def _load_from_cli(cli_args: dict[str, Any]) -> StoreDaemonConfig:
    """
    Create configuration from CLI arguments.

    Args:
        cli_args: Dictionary of CLI arguments

    Returns:
        StoreDaemonConfig instance
    """
    # Convert human-readable sizes to bytes
    chunk_size_str = cli_args.get("chunk_size")
    mem_pool_size_str = cli_args.get("mem_pool_size")
    chunk_size = to_num_bytes(chunk_size_str) if chunk_size_str else None
    mem_pool_size = to_num_bytes(mem_pool_size_str) if mem_pool_size_str else None

    # Create nested config structure
    server_config = ServerConfig()
    network_config = NetworkConfig()

    # Update server config with CLI args
    if cli_args.get("host") is not None:
        server_config.host = cli_args["host"]
    if cli_args.get("port") is not None:
        server_config.port = cli_args["port"]
    if cli_args.get("storage_path") is not None:
        server_config.storage_path = Path(cli_args["storage_path"])
    if mem_pool_size is not None:
        server_config.mem_pool_size = ByteSize(mem_pool_size)
    if cli_args.get("num_thread") is not None:
        server_config.num_threads = cli_args["num_thread"]
    if chunk_size is not None:
        server_config.chunk_size = ByteSize(chunk_size)
    if cli_args.get("enable_p2p_access") is not None:
        server_config.enable_p2p_access = cli_args["enable_p2p_access"]
    if cli_args.get("enable_p2p_engine") is not None:
        server_config.enable_p2p_engine = cli_args["enable_p2p_engine"]
    if cli_args.get("pinned_memory_timeout_ms") is not None:
        server_config.pinned_memory_timeout_ms = cli_args["pinned_memory_timeout_ms"]

    # Update network config with CLI args
    if cli_args.get("p2p_port") is not None:
        network_config.p2p_port = cli_args["p2p_port"]
    if cli_args.get("metrics_port") is not None:
        network_config.metrics_port = cli_args["metrics_port"]
    if cli_args.get("health_check_port") is not None:
        network_config.health_check_port = cli_args["health_check_port"]

    return StoreDaemonConfig(
        server=server_config,
        network=network_config,
        global_store_address=cli_args.get("global_store_address"),
    )


def validate_config(config: StoreDaemonConfig) -> None:
    """
    Validate configuration for logical consistency.

    Args:
        config: StoreDaemonConfig to validate

    Raises:
        ConfigError: If configuration is invalid
    """
    if config.server.port <= 0 or config.server.port > 65535:
        raise ConfigError(f"Invalid port number: {config.server.port}")

    if config.server.num_threads <= 0:
        raise ConfigError(f"Invalid number of threads: {config.server.num_threads}")

    # Allow 0 to disable pinned host pool (useful on systems without CUDA)
    if config.server.mem_pool_size < 0:
        raise ConfigError(f"Invalid memory pool size: {config.server.mem_pool_size}")

    if config.server.chunk_size <= 0:
        raise ConfigError(f"Invalid chunk size: {config.server.chunk_size}")


def print_config_summary(config: StoreDaemonConfig) -> None:
    """
    Print a summary of the configuration.

    Args:
        config: StoreDaemonConfig to summarize
    """
    click.echo("=" * 60)
    click.echo("StoreDaemon Configuration")
    click.echo("=" * 60)
    click.echo(f"gRPC Server: {config.server.host}:{config.server.port}")
    click.echo(f"Storage Path: {config.server.storage_path}")
    click.echo(f"Memory Pool: {config.server.mem_pool_size // (1024**3):.1f} GB")
    click.echo(f"Worker Threads: {config.server.num_threads}")
    click.echo(f"Chunk Size: {config.server.chunk_size // (1024**2):.1f} MB")
    click.echo(
        f"Communication: {'Enabled' if config.server.enable_p2p_engine else 'Disabled'}"
    )
    click.echo(f"Global Store: {config.global_store_address}")
    click.echo(f"Metrics: http://0.0.0.0:{config.network.metrics_port}/metrics")
    click.echo(
        f"Health Check: http://0.0.0.0:{config.network.health_check_port}/health"
    )
    click.echo("=" * 60)
