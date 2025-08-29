#  Copyright (c) 2025, TensorCast Team.

"""Simplified logging configuration for scstore."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Default format with detailed context
_FORMAT = "%(levelname)s %(asctime)s [%(threadName)s %(thread)d] %(filename)s:%(lineno)d: %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"

# Service manager format
SERVICE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    format: Optional[str] = None,
) -> None:
    """Simple logging setup with sensible defaults.

    Args:
        level: Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional path to log file
        format: Optional custom format string. If None, uses default format
    """
    # Convert string level to numeric level
    # Create a mapping of valid log levels to avoid runtime type checks
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.FATAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    # Get log level from map, defaulting to INFO if not found
    log_level: int = level_map.get(level.upper(), logging.INFO)
    if level.upper() not in level_map:
        logging.warning(f"Invalid log level '{level}', defaulting to INFO")

    # Use provided format or default
    log_format = format or _FORMAT

    # Build handlers list
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file_path = Path(log_file).expanduser().resolve()
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, mode="a"))

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
        force=True,  # Python 3.8+ - reconfigure even if already configured
    )

    # Ensure scstore logger uses same level
    logging.getLogger("scstore").setLevel(log_level)


def init_logger(name: str) -> logging.Logger:
    """Initialize a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Auto-configure logging if not already done
    if not logging.getLogger().handlers:
        # Use environment variable for log level if available
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        setup_logging(level=log_level)

    return logging.getLogger(name)
