#  Copyright (c) 2025, TensorCast Team.

"""Custom exceptions for Global Store."""

from .base import (
    ConflictError,
    DatabaseError,
    GlobalStoreError,
    NotFoundError,
    TimeoutError,
    TooManyRequestsError,
    ValidationError,
)

__all__ = [
    "GlobalStoreError",
    "NotFoundError",
    "TimeoutError",
    "TooManyRequestsError",
    "ValidationError",
    "ConflictError",
    "DatabaseError",
]
