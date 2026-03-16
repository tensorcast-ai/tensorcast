#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.common.identity import is_byte_artifact_id, validate_byte_artifact_cgid

_KEY_SET_DIGEST_PREFIX = "tensorcast.byte_artifact.keyset.v1\n"


@dataclass(frozen=True, slots=True)
class PutIfAbsentInvariant:
    layout_id: str
    byte_length: int
    payload_digest_alg: str
    payload_digest_hex: str


@dataclass(frozen=True, slots=True)
class OpenByteArtifact:
    artifact_id: str
    layout_id: str
    payload: bytes

    def seal(self) -> "SealedByteArtifact":
        return seal_byte_artifact(
            artifact_id=self.artifact_id,
            layout_id=self.layout_id,
            payload=self.payload,
        )


@dataclass(frozen=True, slots=True)
class SealedByteArtifact:
    artifact_id: str
    payload: bytes
    invariant: PutIfAbsentInvariant


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    artifact_id: str
    status_code: str
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestResult:
    engine_request_id: str
    layout_id: str
    artifact_ids: tuple[str, ...]
    key_set_digest_alg: str
    key_set_digest_hex: str

    @classmethod
    def from_artifact_ids(
        cls,
        *,
        engine_request_id: str,
        layout_id: str,
        artifact_ids: tuple[str, ...],
    ) -> "ManifestResult":
        digest_hex = compute_key_set_digest_hex(
            layout_id=layout_id,
            artifact_ids=artifact_ids,
        )
        return cls(
            engine_request_id=engine_request_id,
            layout_id=layout_id,
            artifact_ids=artifact_ids,
            key_set_digest_alg="sha256",
            key_set_digest_hex=digest_hex,
        )


@dataclass(frozen=True, slots=True)
class PublishResult:
    manifest: ManifestResult
    put_outcomes: tuple[BatchOutcome, ...]


@dataclass(frozen=True, slots=True)
class HydrateResult:
    manifest: ManifestResult | None
    get_outcomes: tuple[BatchOutcome, ...]
    missing_artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchResult:
    engine_request_id: str | None
    outcomes: tuple[BatchOutcome, ...]


class EngineArtifactAdapter(Protocol):
    def manifest(
        self,
        *,
        engine_request_id: str,
        ctx: CallContext | None = None,
    ) -> ManifestResult: ...

    def publish(
        self,
        *,
        engine_request_id: str,
        ttl_ms: int | None = None,
        sealed_artifacts: tuple[SealedByteArtifact, ...] = (),
        ctx: CallContext | None = None,
    ) -> PublishResult: ...

    def hydrate(
        self,
        *,
        engine_request_id: str,
        ctx: CallContext | None = None,
    ) -> HydrateResult: ...

    def evict_local(
        self,
        *,
        engine_request_id: str | None = None,
        ctx: CallContext | None = None,
    ) -> BatchResult: ...


def open_byte_artifact(
    *,
    artifact_id: str,
    layout_id: str,
    payload: bytes,
) -> OpenByteArtifact:
    if not is_byte_artifact_id(artifact_id):
        raise ArtifactError(
            "byte artifact id must use 'cgid:byte_artifact~...' namespace",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    try:
        validate_byte_artifact_cgid(artifact_id)
    except ValueError as exc:
        raise ArtifactError(
            f"byte artifact id must be a valid byte artifact cgid: {exc}",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        ) from exc
    if not layout_id:
        raise ArtifactError(
            "layout_id is required for byte artifact open state",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return OpenByteArtifact(
        artifact_id=artifact_id,
        layout_id=layout_id,
        payload=bytes(payload),
    )


def seal_byte_artifact(
    *,
    artifact_id: str,
    layout_id: str,
    payload: bytes,
) -> SealedByteArtifact:
    opened = open_byte_artifact(
        artifact_id=artifact_id,
        layout_id=layout_id,
        payload=payload,
    )
    payload_bytes = bytes(opened.payload)
    payload_digest_hex = hashlib.sha256(payload_bytes).hexdigest()
    invariant = PutIfAbsentInvariant(
        layout_id=opened.layout_id,
        byte_length=len(payload_bytes),
        payload_digest_alg="sha256",
        payload_digest_hex=payload_digest_hex,
    )
    return SealedByteArtifact(
        artifact_id=opened.artifact_id,
        payload=payload_bytes,
        invariant=invariant,
    )


def compute_key_set_digest_hex(*, layout_id: str, artifact_ids: tuple[str, ...]) -> str:
    if not layout_id:
        raise ArtifactError(
            "layout_id is required for key-set digest",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    normalized_ids = sorted({str(item) for item in artifact_ids if str(item)})
    payload = (
        _KEY_SET_DIGEST_PREFIX + layout_id + "\n" + "\n".join(normalized_ids)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "BatchOutcome",
    "BatchResult",
    "EngineArtifactAdapter",
    "HydrateResult",
    "ManifestResult",
    "OpenByteArtifact",
    "PublishResult",
    "PutIfAbsentInvariant",
    "SealedByteArtifact",
    "compute_key_set_digest_hex",
    "open_byte_artifact",
    "seal_byte_artifact",
]
