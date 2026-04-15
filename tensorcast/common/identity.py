#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from enum import Enum

MI2_PREFIX: str = "mi2:"
CGID_PREFIX: str = "cgid:"
MSA1_PREFIX: str = "msa1:"
_CGID_ALLOWED: re.Pattern[str] = re.compile(r"^[-._~A-Za-z0-9]+$")
_B64U_ALLOWED: re.Pattern[str] = re.compile(r"^[-_A-Za-z0-9]+$")
_CGID_MIN_LEN: int = 8
_CGID_MAX_LEN: int = 200
BYTE_ARTIFACT_CGID_NAMESPACE: str = "byte_artifact"
_CGID_SEGMENT_DELIM: str = "~"
_LEGACY_BYTE_ARTIFACT_SEGMENT_COUNT: int = 6
_BYTE_ARTIFACT_SEGMENT_COUNT: int = 7
_B64U_PREFIX: str = "b64u."


class ArtifactIdKind(str, Enum):
    """Classifier for artifact identity schemes."""

    MI2 = "MI2"
    CGID = "CGID"
    MSA1 = "MSA1"


@dataclass(frozen=True, slots=True)
class ByteArtifactCgidParts:
    namespace: str
    engine: str
    model_id_enc: str
    model_version_enc: str
    layout_id: str
    engine_key_enc: str


def _ensure_ascii(value: str) -> None:
    if not value.isascii():
        raise ValueError("artifact_id must be ASCII")


def infer_artifact_id_kind(artifact_id: str) -> ArtifactIdKind | None:
    """Return the identity kind implied by the artifact_id prefix."""
    if artifact_id.startswith(MI2_PREFIX):
        return ArtifactIdKind.MI2
    if artifact_id.startswith(CGID_PREFIX):
        return ArtifactIdKind.CGID
    if artifact_id.startswith(MSA1_PREFIX):
        return ArtifactIdKind.MSA1
    return None


def is_msa1_artifact_id(artifact_id: str) -> bool:
    return artifact_id.startswith(MSA1_PREFIX)


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


def encode_cgid_segment(raw: bytes | str) -> str:
    payload = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
    if not payload:
        raise ValueError("cgid segment payload must not be empty")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{_B64U_PREFIX}{encoded}"


def decode_cgid_segment(encoded: str) -> bytes:
    if not encoded.startswith(_B64U_PREFIX):
        raise ValueError("cgid segment must start with 'b64u.'")
    payload = encoded[len(_B64U_PREFIX) :]
    if not payload:
        raise ValueError("cgid segment payload is empty")
    if not _B64U_ALLOWED.fullmatch(payload):
        raise ValueError("cgid segment payload contains invalid base64url characters")
    padding = "=" * (-len(payload) % 4)
    try:
        return base64.urlsafe_b64decode(payload + padding)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("cgid segment payload is not valid base64url") from exc


def is_byte_artifact_id(artifact_id: str) -> bool:
    if not artifact_id.startswith(CGID_PREFIX):
        return False
    suffix = artifact_id[len(CGID_PREFIX) :]
    return suffix.startswith(f"{BYTE_ARTIFACT_CGID_NAMESPACE}{_CGID_SEGMENT_DELIM}")


def parse_byte_artifact_cgid(artifact_id: str) -> ByteArtifactCgidParts:
    validate_client_generated_id(artifact_id)
    suffix = artifact_id[len(CGID_PREFIX) :]
    segments = suffix.split(_CGID_SEGMENT_DELIM)
    if (
        len(segments)
        not in {
            _LEGACY_BYTE_ARTIFACT_SEGMENT_COUNT,
            _BYTE_ARTIFACT_SEGMENT_COUNT,
        }
        or segments[0] != BYTE_ARTIFACT_CGID_NAMESPACE
    ):
        raise ValueError(
            "byte_artifact cgid must match "
            "'cgid:byte_artifact~<namespace>~<engine>~<model_id_enc>~"
            "<layout_id>~<engine_key_enc>' or "
            "'cgid:byte_artifact~<namespace>~<engine>~<model_id_enc>~"
            "<model_version_enc>~<layout_id>~<engine_key_enc>'"
        )
    if any(not segment for segment in segments):
        raise ValueError("byte_artifact cgid segments must be non-empty")
    if any(
        ("|" in segment) or ("\n" in segment) or ("\r" in segment)
        for segment in segments
    ):
        raise ValueError("byte_artifact cgid segments contain forbidden delimiters")
    if len(segments) == _LEGACY_BYTE_ARTIFACT_SEGMENT_COUNT:
        return ByteArtifactCgidParts(
            namespace=segments[1],
            engine=segments[2],
            model_id_enc=segments[3],
            model_version_enc="",
            layout_id=segments[4],
            engine_key_enc=segments[5],
        )
    return ByteArtifactCgidParts(
        namespace=segments[1],
        engine=segments[2],
        model_id_enc=segments[3],
        model_version_enc=segments[4],
        layout_id=segments[5],
        engine_key_enc=segments[6],
    )


def validate_byte_artifact_cgid(artifact_id: str) -> None:
    parse_byte_artifact_cgid(artifact_id)


def build_byte_artifact_cgid(
    *,
    namespace: str,
    engine: str,
    model_id: bytes | str,
    model_version: bytes | str,
    layout_id: str,
    engine_key: bytes | str,
) -> str:
    for field_name, value in (
        ("namespace", namespace),
        ("engine", engine),
        ("layout_id", layout_id),
    ):
        if not value:
            raise ValueError(f"{field_name} must be non-empty")
        if (
            _CGID_SEGMENT_DELIM in value
            or "|" in value
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(f"{field_name} contains forbidden delimiters")
    artifact_id = (
        f"{CGID_PREFIX}"
        f"{BYTE_ARTIFACT_CGID_NAMESPACE}{_CGID_SEGMENT_DELIM}"
        f"{namespace}{_CGID_SEGMENT_DELIM}"
        f"{engine}{_CGID_SEGMENT_DELIM}"
        f"{encode_cgid_segment(model_id)}{_CGID_SEGMENT_DELIM}"
        f"{encode_cgid_segment(model_version)}{_CGID_SEGMENT_DELIM}"
        f"{layout_id}{_CGID_SEGMENT_DELIM}"
        f"{encode_cgid_segment(engine_key)}"
    )
    validate_client_generated_id(artifact_id)
    return artifact_id


def validate_artifact_id(artifact_id: str) -> ArtifactIdKind:
    """Validate an artifact identifier and return its kind."""
    kind = infer_artifact_id_kind(artifact_id)
    if kind is ArtifactIdKind.MI2:
        return kind
    if kind is ArtifactIdKind.CGID:
        validate_client_generated_id(artifact_id)
        return kind
    if kind is ArtifactIdKind.MSA1:
        return kind
    raise ValueError("artifact_id must start with 'mi2:', 'cgid:', or 'msa1:'")


__all__ = [
    "ArtifactIdKind",
    "BYTE_ARTIFACT_CGID_NAMESPACE",
    "CGID_PREFIX",
    "ByteArtifactCgidParts",
    "MI2_PREFIX",
    "MSA1_PREFIX",
    "build_byte_artifact_cgid",
    "decode_cgid_segment",
    "encode_cgid_segment",
    "infer_artifact_id_kind",
    "is_byte_artifact_id",
    "is_msa1_artifact_id",
    "parse_byte_artifact_cgid",
    "validate_artifact_id",
    "validate_byte_artifact_cgid",
    "validate_client_generated_id",
]
