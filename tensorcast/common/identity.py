#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import re
from enum import Enum

MI2_PREFIX: str = "mi2:"
CGID_PREFIX: str = "cgid:"
_CGID_ALLOWED: re.Pattern[str] = re.compile(r"^[-._~A-Za-z0-9]+$")
_CGID_MIN_LEN: int = 8
_CGID_MAX_LEN: int = 200


class ArtifactIdKind(str, Enum):
    """Classifier for artifact identity schemes."""

    MI2 = "MI2"
    CGID = "CGID"


def _ensure_ascii(value: str) -> None:
    if not value.isascii():
        raise ValueError("artifact_id must be ASCII")


def infer_artifact_id_kind(artifact_id: str) -> ArtifactIdKind | None:
    """Return the identity kind implied by the artifact_id prefix."""
    if artifact_id.startswith(MI2_PREFIX):
        return ArtifactIdKind.MI2
    if artifact_id.startswith(CGID_PREFIX):
        return ArtifactIdKind.CGID
    return None


def validate_client_generated_id(artifact_id: str) -> None:
    """Validate the grammar for a client-generated artifact identifier."""
    if not artifact_id.startswith(CGID_PREFIX):
        raise ValueError("client_artifact_id must start with 'cgid:'")
    _ensure_ascii(artifact_id)
    if not (_CGID_MIN_LEN <= len(artifact_id) <= _CGID_MAX_LEN):
        raise ValueError(
            "client_artifact_id length must be between 8 and 200 characters"
        )
    suffix = artifact_id[len(CGID_PREFIX) :]
    if not suffix:
        raise ValueError("client_artifact_id requires a suffix after 'cgid:'")
    if not _CGID_ALLOWED.fullmatch(suffix):
        raise ValueError("client_artifact_id contains invalid characters")


def validate_artifact_id(artifact_id: str) -> ArtifactIdKind:
    """Validate an artifact identifier and return its kind."""
    kind = infer_artifact_id_kind(artifact_id)
    if kind is ArtifactIdKind.MI2:
        return kind
    if kind is ArtifactIdKind.CGID:
        validate_client_generated_id(artifact_id)
        return kind
    raise ValueError("artifact_id must start with 'mi2:' or 'cgid:'")


__all__ = [
    "ArtifactIdKind",
    "CGID_PREFIX",
    "MI2_PREFIX",
    "infer_artifact_id_kind",
    "validate_artifact_id",
    "validate_client_generated_id",
]
