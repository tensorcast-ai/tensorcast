#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from typing import Literal

ArtifactStatusCode = Literal[
    "OK",
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
    "UNAUTHENTICATED",
]

CollectiveFailureClassName = Literal["not_eligible", "execution_failed"]


class ArtifactError(RuntimeError):
    """Structured exception surfaced by Store verbs."""

    def __init__(
        self,
        message: str,
        *,
        status_code: ArtifactStatusCode,
        retryable: bool,
        collective_failure_class: CollectiveFailureClassName | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code: ArtifactStatusCode = status_code
        self.retryable = bool(retryable)
        self.collective_failure_class: CollectiveFailureClassName | None = (
            collective_failure_class
        )


__all__ = [
    "ArtifactError",
    "ArtifactStatusCode",
    "CollectiveFailureClassName",
]
