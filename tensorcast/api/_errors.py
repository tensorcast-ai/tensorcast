#  Copyright (c) 2025, TensorCast Team.


class TensorCastError(Exception):
    """Base class for TensorCast SDK errors."""


class DaemonUnavailable(TensorCastError):
    """Raised when the StoreDaemon is unavailable or not reachable."""


class VerificationFailed(TensorCastError):
    """Raised when asynchronous verification fails."""


class DeviceMismatch(TensorCastError):
    """Raised when provided devices or tensors do not match expectations."""


class DiskIndexMismatch(TensorCastError):
    """Raised when provided disk tensor_index.json does not match computed index."""


class InvalidPlan(TensorCastError):
    """Raised when an unknown or invalid registration plan is specified."""


class FeedFailed(TensorCastError):
    """Raised when feeding registration data to the daemon fails."""


class IndexParseError(TensorCastError):
    """Raised when parsing index artifacts fails."""
