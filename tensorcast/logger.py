#  Copyright (c) 2025, TensorCast Team.

"""Simplified logging configuration for tensorcast."""

import logging
import sys
from pathlib import Path
from typing import Optional

from opentelemetry import trace

# Default format with detailed context
_FORMAT = (
    "%(levelname)s %(asctime)s [%(threadName)s] "
    "trace_id=%(trace_id)s span_id=%(span_id)s "
    "%(filename)s:%(lineno)d: %(message)s"
)
_DATE_FORMAT = "%m-%d %H:%M:%S"

# Service manager format
SERVICE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    class _SafeFormatter(logging.Formatter):
        """Formatter that tolerates missing trace fields by injecting defaults.

        Some early logs may occur before OTel context is available. This
        formatter ensures %(trace_id)s / %(span_id)s exist to prevent KeyError.
        """

        def format(self, record: logging.LogRecord) -> str:
            if not hasattr(record, "trace_id"):
                record.trace_id = "-"
            if not hasattr(record, "span_id"):
                record.span_id = "-"
            return super().format(record)

    # Configure root logger with handlers and safe formatter
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    fmt = _SafeFormatter(log_format, _DATE_FORMAT)
    for h in logging.getLogger().handlers:
        h.setFormatter(fmt)

    # Attach a filter that injects OpenTelemetry trace_id/span_id into log records
    _prev_factory = logging.getLogRecordFactory()

    def _record_factory(*args, **kwargs):
        record = _prev_factory(*args, **kwargs)
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        if not hasattr(record, "span_id"):
            record.span_id = "-"

        span = trace.get_current_span()
        if span is not None:
            ctx = span.get_span_context()
            tid = getattr(ctx, "trace_id", 0)
            sid = getattr(ctx, "span_id", 0)
            if tid:
                record.trace_id = f"{tid:032x}"
            if sid:
                record.span_id = f"{sid:016x}"
        return record

    logging.setLogRecordFactory(_record_factory)

    class _OtelContextFilter(logging.Filter):
        """Inject OpenTelemetry trace_id/span_id into log records.

        Tolerant of missing OpenTelemetry dependencies and always ensures the
        fields exist to avoid formatting KeyError.
        """

        def __init__(self) -> None:
            super().__init__()
            try:
                from opentelemetry import trace as _trace

                self._trace = _trace
            except Exception:  # OTEL not installed
                self._trace = None

        def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
            # Default placeholders prevent formatter failures when tracing is off
            if not hasattr(record, "trace_id"):
                record.trace_id = "-"
            if not hasattr(record, "span_id"):
                record.span_id = "-"

            try:
                if self._trace is None:
                    return True
                span = self._trace.get_current_span()
                if span is None:
                    return True
                ctx = span.get_span_context()
                # ctx may be invalid if no active span
                trace_id = getattr(ctx, "trace_id", 0)
                span_id = getattr(ctx, "span_id", 0)
                if trace_id:
                    record.trace_id = f"{trace_id:032x}"
                if span_id:
                    record.span_id = f"{span_id:016x}"
            except Exception:
                # Best-effort only; never block logging
                pass
            return True

    root_logger = logging.getLogger()
    _filter_instance = _OtelContextFilter()
    # Attach to all handlers so records passing through root handlers get enriched
    for _h in root_logger.handlers:
        _h.addFilter(_filter_instance)
    # Also attach to root logger as a fallback
    root_logger.addFilter(_filter_instance)

    # Ensure tensorcast logger uses same level
    logging.getLogger("tensorcast").setLevel(log_level)


def init_logger(name: str) -> logging.Logger:
    """Initialize a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Auto-configure logging if not already done (default INFO, no env)
    if not logging.getLogger().handlers:
        setup_logging(level="INFO")

    return logging.getLogger(name)
