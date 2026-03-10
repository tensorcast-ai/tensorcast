#  Copyright (c) 2025-2026, TensorCast Team.

"""Main entry point for Global Store server."""

import argparse

from tensorcast.global_store.launcher import run_global_store


def main():
    """Start the Global Store server."""
    parser = argparse.ArgumentParser(
        description="Global Store Server - Centralized artifact registry"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to Global Store config (YAML/JSON)",
    )
    parser.add_argument(
        "--listen-host",
        type=str,
        default=None,
        help="Override listen host from config",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=None,
        help="Override listen port from config (0 = auto)",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=None,
        help="Override metrics port from config (0 = auto)",
    )
    args = parser.parse_args()
    run_global_store(
        config_path=args.config,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        metrics_port=args.metrics_port,
    )


if __name__ == "__main__":
    main()
