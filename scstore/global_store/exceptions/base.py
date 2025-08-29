#  Copyright (c) 2025, TensorCast Team.

"""Base exception classes for Global Store."""


class GlobalStoreError(Exception):
    """Base exception for all Global Store errors."""

    pass


class NotFoundError(GlobalStoreError):
    """Raised when a requested resource is not found."""

    pass


class TimeoutError(GlobalStoreError):
    """Raised when an operation times out."""

    pass


class TooManyRequestsError(GlobalStoreError):
    """Raised when too many requests are made."""

    pass


class ValidationError(GlobalStoreError):
    """Raised when validation fails."""

    pass


class ConflictError(GlobalStoreError):
    """Raised when there's a conflict in the operation."""

    pass


class DatabaseError(GlobalStoreError):
    """Raised when database operations fail."""

    pass
