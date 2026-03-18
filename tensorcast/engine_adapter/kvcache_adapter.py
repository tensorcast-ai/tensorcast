#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Sequence

from tensorcast.api.context import CallContext
from tensorcast.api.errors import ArtifactError
from tensorcast.api.plan.artifact_set import (
    ARTIFACT_SET_CARRIER_MANIFEST_BACKED,
    ArtifactSetRef,
    canonicalize_artifact_selections,
    compute_artifact_set_digest_hex,
)
from tensorcast.common.identity import is_byte_artifact_id, validate_byte_artifact_cgid
from tensorcast.proto.common.v1 import common_pb2

_KEY_SET_DIGEST_PREFIX = "tensorcast.byte_artifact.keyset.v1\n"
MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA = "tensorcast.manifest_artifact_set_bridge"
MANIFEST_ARTIFACT_SET_BRIDGE_VERSION = 1


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
class ManifestArtifactSetBridge:
    bridge_schema: str
    bridge_version: int
    artifact_set_ref: ArtifactSetRef
    resolved_items: tuple[common_pb2.ArtifactSelection, ...]

    def __post_init__(self) -> None:
        bridge_schema = str(self.bridge_schema).strip()
        if not bridge_schema:
            raise ArtifactError(
                "ManifestArtifactSetBridge.bridge_schema is required",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        bridge_version = int(self.bridge_version)
        if bridge_version <= 0:
            raise ArtifactError(
                "ManifestArtifactSetBridge.bridge_version must be positive",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        if self.artifact_set_ref.carrier_form != ARTIFACT_SET_CARRIER_MANIFEST_BACKED:
            raise ArtifactError(
                "ManifestArtifactSetBridge requires manifest_backed ArtifactSetRef",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )
        canonical_items = canonicalize_artifact_selections(self.resolved_items)
        object.__setattr__(self, "bridge_schema", bridge_schema)
        object.__setattr__(self, "bridge_version", bridge_version)
        object.__setattr__(
            self,
            "resolved_items",
            tuple(_clone_selection(item.selection) for item in canonical_items),
        )

    @classmethod
    def from_resolved_items(
        cls,
        *,
        manifest_selection: common_pb2.ArtifactSelection,
        resolved_items: Sequence[common_pb2.ArtifactSelection],
        bridge_schema: str = MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA,
        bridge_version: int = MANIFEST_ARTIFACT_SET_BRIDGE_VERSION,
    ) -> "ManifestArtifactSetBridge":
        canonical_items = canonicalize_artifact_selections(resolved_items)
        artifact_set_ref = ArtifactSetRef.manifest_backed(
            set_digest_hex=compute_artifact_set_digest_hex(
                item.item_identity for item in canonical_items
            ),
            item_count=len(canonical_items),
            manifest_selection=manifest_selection,
        )
        return cls(
            bridge_schema=bridge_schema,
            bridge_version=bridge_version,
            artifact_set_ref=artifact_set_ref,
            resolved_items=tuple(
                _clone_selection(item.selection) for item in canonical_items
            ),
        )

    def to_proto(self):  # noqa: ANN201
        from tensorcast.proto.plan.v1 import plan_pb2

        message = plan_pb2.ManifestArtifactSetBridge(
            bridge_schema=str(self.bridge_schema),
            bridge_version=int(self.bridge_version),
        )
        message.artifact_set_ref.CopyFrom(self.artifact_set_ref.to_proto())
        message.resolved_items.extend(
            _clone_selection(selection) for selection in self.resolved_items
        )
        return message

    @classmethod
    def from_proto(cls, message) -> "ManifestArtifactSetBridge":  # noqa: ANN206, ANN001
        return cls(
            bridge_schema=str(message.bridge_schema),
            bridge_version=int(message.bridge_version),
            artifact_set_ref=ArtifactSetRef.from_proto(message.artifact_set_ref),
            resolved_items=tuple(
                _clone_selection(selection) for selection in message.resolved_items
            ),
        )

    def resolve_artifact_set(
        self, artifact_set_ref: ArtifactSetRef
    ) -> tuple[common_pb2.ArtifactSelection, ...]:
        if self.bridge_schema != MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA:
            raise ArtifactError(
                "ManifestArtifactSetBridge schema is unsupported",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if self.bridge_version != MANIFEST_ARTIFACT_SET_BRIDGE_VERSION:
            raise ArtifactError(
                "ManifestArtifactSetBridge version is unsupported",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        if not _artifact_set_ref_matches(self.artifact_set_ref, artifact_set_ref):
            raise ArtifactError(
                "ManifestArtifactSetBridge artifact_set_ref does not match requested ArtifactSetRef",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return tuple(_clone_selection(selection) for selection in self.resolved_items)


@dataclass(frozen=True, slots=True)
class ManifestResult:
    engine_request_id: str
    layout_id: str
    artifact_ids: tuple[str, ...]
    key_set_digest_alg: str
    key_set_digest_hex: str
    artifact_set_bridge: ManifestArtifactSetBridge | None = None

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

    @classmethod
    def from_artifact_selections(
        cls,
        *,
        engine_request_id: str,
        layout_id: str,
        manifest_selection: common_pb2.ArtifactSelection,
        artifact_selections: Sequence[common_pb2.ArtifactSelection],
        bridge_schema: str = MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA,
        bridge_version: int = MANIFEST_ARTIFACT_SET_BRIDGE_VERSION,
    ) -> "ManifestResult":
        bridge = ManifestArtifactSetBridge.from_resolved_items(
            manifest_selection=manifest_selection,
            resolved_items=artifact_selections,
            bridge_schema=bridge_schema,
            bridge_version=bridge_version,
        )
        return cls(
            engine_request_id=engine_request_id,
            layout_id=layout_id,
            artifact_ids=tuple(str(item.artifact_id) for item in artifact_selections),
            key_set_digest_alg="sha256",
            key_set_digest_hex=compute_key_set_digest_hex(
                layout_id=layout_id,
                artifact_ids=tuple(
                    str(item.artifact_id) for item in artifact_selections
                ),
            ),
            artifact_set_bridge=bridge,
        )

    def require_artifact_set_bridge(self) -> ManifestArtifactSetBridge:
        if self.artifact_set_bridge is None:
            raise ArtifactError(
                "ManifestResult does not carry an explicit ManifestArtifactSetBridge",
                status_code="FAILED_PRECONDITION",
                retryable=False,
            )
        return self.artifact_set_bridge

    def require_artifact_set_ref(self) -> ArtifactSetRef:
        return self.require_artifact_set_bridge().artifact_set_ref


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


def _clone_selection(
    selection: common_pb2.ArtifactSelection,
) -> common_pb2.ArtifactSelection:
    cloned = common_pb2.ArtifactSelection()
    cloned.CopyFrom(selection)
    return cloned


def _artifact_set_ref_matches(left: ArtifactSetRef, right: ArtifactSetRef) -> bool:
    return left.to_proto().SerializeToString(
        deterministic=True
    ) == right.to_proto().SerializeToString(deterministic=True)


__all__ = [
    "BatchOutcome",
    "BatchResult",
    "EngineArtifactAdapter",
    "HydrateResult",
    "MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA",
    "MANIFEST_ARTIFACT_SET_BRIDGE_VERSION",
    "ManifestResult",
    "ManifestArtifactSetBridge",
    "OpenByteArtifact",
    "PublishResult",
    "PutIfAbsentInvariant",
    "SealedByteArtifact",
    "compute_key_set_digest_hex",
    "open_byte_artifact",
    "seal_byte_artifact",
]
