#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from tensorcast.api.errors import ArtifactError
from tensorcast.api.operation import OperationError, OperationStatus
from tensorcast.common.selection_identity import (
    SelectionIdentity,
    build_selection_identity,
)
from tensorcast.proto.common.v1 import common_pb2

ARTIFACT_SET_CARRIER_INLINE = "inline"
ARTIFACT_SET_CARRIER_MANIFEST_BACKED = "manifest_backed"
MAX_INLINE_ARTIFACT_SET_ITEMS = 1024

_SET_DIGEST_PREFIX = b"tensorcast.artifact_set.v1\n"
_HEX_RE = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True, slots=True)
class ResolvedArtifactSetItem:
    selection: common_pb2.ArtifactSelection
    item_identity: SelectionIdentity


@dataclass(frozen=True, slots=True)
class ArtifactSetRef:
    set_digest_hex: str
    item_count: int
    carrier_form: str
    inline_items: tuple[common_pb2.ArtifactSelection, ...] | None = None
    manifest_selection: common_pb2.ArtifactSelection | None = None

    def __post_init__(self) -> None:
        digest_hex = str(self.set_digest_hex).strip().lower()
        if not digest_hex or _HEX_RE.fullmatch(digest_hex) is None:
            raise ArtifactError(
                "ArtifactSetRef.set_digest_hex must be non-empty lowercase hex",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        item_count = int(self.item_count)
        if item_count < 0:
            raise ArtifactError(
                "ArtifactSetRef.item_count must be non-negative",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        carrier_form = str(self.carrier_form).strip()
        if carrier_form not in {
            ARTIFACT_SET_CARRIER_INLINE,
            ARTIFACT_SET_CARRIER_MANIFEST_BACKED,
        }:
            raise ArtifactError(
                "ArtifactSetRef.carrier_form must be 'inline' or 'manifest_backed'",
                status_code="INVALID_ARGUMENT",
                retryable=False,
            )

        inline_items = (
            tuple(_clone_selection(selection) for selection in self.inline_items)
            if self.inline_items is not None
            else None
        )
        manifest_selection = (
            _clone_selection(self.manifest_selection)
            if self.manifest_selection is not None
            else None
        )

        if carrier_form == ARTIFACT_SET_CARRIER_INLINE:
            if manifest_selection is not None:
                raise ArtifactError(
                    "ArtifactSetRef inline carrier must not set manifest_selection",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if inline_items is None:
                inline_items = ()
            if len(inline_items) > MAX_INLINE_ARTIFACT_SET_ITEMS:
                raise ArtifactError(
                    "ArtifactSetRef inline carrier exceeds the explicit inline limit",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
        else:
            if manifest_selection is None:
                raise ArtifactError(
                    "ArtifactSetRef manifest_backed carrier requires manifest_selection",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            if inline_items not in (None, ()):
                raise ArtifactError(
                    "ArtifactSetRef manifest_backed carrier must not set inline_items",
                    status_code="INVALID_ARGUMENT",
                    retryable=False,
                )
            inline_items = None

        object.__setattr__(self, "set_digest_hex", digest_hex)
        object.__setattr__(self, "item_count", item_count)
        object.__setattr__(self, "carrier_form", carrier_form)
        object.__setattr__(self, "inline_items", inline_items)
        object.__setattr__(self, "manifest_selection", manifest_selection)

    @classmethod
    def inline(
        cls, selections: Sequence[common_pb2.ArtifactSelection]
    ) -> "ArtifactSetRef":
        canonical_items = canonicalize_inline_artifact_selections(selections)
        return cls(
            set_digest_hex=compute_artifact_set_digest_hex(
                item.item_identity for item in canonical_items
            ),
            item_count=len(canonical_items),
            carrier_form=ARTIFACT_SET_CARRIER_INLINE,
            inline_items=tuple(item.selection for item in canonical_items),
        )

    @classmethod
    def manifest_backed(
        cls,
        *,
        set_digest_hex: str,
        item_count: int,
        manifest_selection: common_pb2.ArtifactSelection,
    ) -> "ArtifactSetRef":
        return cls(
            set_digest_hex=set_digest_hex,
            item_count=item_count,
            carrier_form=ARTIFACT_SET_CARRIER_MANIFEST_BACKED,
            manifest_selection=manifest_selection,
        )

    def to_proto(self):  # noqa: ANN201
        from tensorcast.proto.plan.v1 import plan_pb2

        message = plan_pb2.ArtifactSetRef(
            set_digest_hex=str(self.set_digest_hex),
            item_count=int(self.item_count),
            carrier_form=str(self.carrier_form),
        )
        if self.inline_items:
            message.inline_items.extend(
                _clone_selection(selection) for selection in self.inline_items
            )
        if self.manifest_selection is not None:
            message.manifest_selection.CopyFrom(self.manifest_selection)
        return message

    @classmethod
    def from_proto(cls, message) -> "ArtifactSetRef":  # noqa: ANN206, ANN001
        manifest_selection = (
            _clone_selection(message.manifest_selection)
            if message.HasField("manifest_selection")
            else None
        )
        return cls(
            set_digest_hex=str(message.set_digest_hex),
            item_count=int(message.item_count),
            carrier_form=str(message.carrier_form),
            inline_items=tuple(_clone_selection(item) for item in message.inline_items),
            manifest_selection=manifest_selection,
        )


@dataclass(frozen=True, slots=True)
class ArtifactSetItemResult:
    item_identity: SelectionIdentity
    artifact_id: str | None = None
    status: OperationStatus | None = None


@dataclass(frozen=True, slots=True)
class ArtifactSetResult:
    set_digest_hex: str
    outcomes: tuple[ArtifactSetItemResult, ...]


ManifestArtifactSetResolver = Callable[
    [ArtifactSetRef],
    Sequence[common_pb2.ArtifactSelection] | Sequence[ResolvedArtifactSetItem],
]


def selection_identity_to_proto(
    identity: SelectionIdentity,
) -> common_pb2.SelectionIdentity:
    return common_pb2.SelectionIdentity(
        artifact_id=str(identity.artifact_id),
        logical_layout_hash=bytes(identity.logical_layout_hash),
        selection_hash=bytes(identity.selection_hash),
    )


def selection_identity_from_proto(
    identity: common_pb2.SelectionIdentity,
) -> SelectionIdentity:
    if not identity.artifact_id:
        raise ArtifactError(
            "SelectionIdentity.artifact_id is required",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if not identity.logical_layout_hash:
        raise ArtifactError(
            "SelectionIdentity.logical_layout_hash is required",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    if not identity.selection_hash:
        raise ArtifactError(
            "SelectionIdentity.selection_hash is required",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    return SelectionIdentity(
        artifact_id=str(identity.artifact_id),
        logical_layout_hash=bytes(identity.logical_layout_hash),
        selection_hash=bytes(identity.selection_hash),
    )


def compute_artifact_set_digest_hex(
    identities: Iterable[SelectionIdentity] | Iterable[ResolvedArtifactSetItem],
) -> str:
    digest = hashlib.sha256()
    digest.update(_SET_DIGEST_PREFIX)
    for artifact_id, logical_layout_hash, selection_hash in _unique_identity_keys(
        identities
    ):
        encoded_artifact_id = artifact_id.encode("utf-8")
        digest.update(len(encoded_artifact_id).to_bytes(8, "big"))
        digest.update(encoded_artifact_id)
        digest.update(len(logical_layout_hash).to_bytes(4, "big"))
        digest.update(logical_layout_hash)
        digest.update(len(selection_hash).to_bytes(4, "big"))
        digest.update(selection_hash)
    return digest.hexdigest()


def canonicalize_inline_artifact_selections(
    selections: Sequence[common_pb2.ArtifactSelection],
) -> tuple[ResolvedArtifactSetItem, ...]:
    return canonicalize_artifact_selections(
        selections,
        max_items=MAX_INLINE_ARTIFACT_SET_ITEMS,
    )


def canonicalize_artifact_selections(
    selections: Sequence[common_pb2.ArtifactSelection],
    *,
    max_items: int | None = None,
) -> tuple[ResolvedArtifactSetItem, ...]:
    if max_items is not None and len(selections) > max_items:
        raise ArtifactError(
            "inline ArtifactSetRef exceeds the explicit inline limit",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )

    canonical_items: dict[
        tuple[str, bytes, bytes], tuple[bytes, common_pb2.ArtifactSelection]
    ] = {}
    for selection in selections:
        cloned = _clone_selection(selection)
        identity = build_selection_identity(cloned)
        key = _selection_identity_key(identity)
        stable_bytes = cloned.SerializeToString(deterministic=True)
        current = canonical_items.get(key)
        if current is None or stable_bytes < current[0]:
            canonical_items[key] = (stable_bytes, cloned)

    resolved = [
        ResolvedArtifactSetItem(
            selection=selection,
            item_identity=SelectionIdentity(
                artifact_id=artifact_id,
                logical_layout_hash=logical_layout_hash,
                selection_hash=selection_hash,
            ),
        )
        for (artifact_id, logical_layout_hash, selection_hash), (
            _,
            selection,
        ) in canonical_items.items()
    ]
    resolved.sort(key=lambda item: _selection_identity_sort_key(item.item_identity))
    return tuple(resolved)


def resolve_artifact_set_ref(
    artifact_set: ArtifactSetRef,
    *,
    manifest_resolver: ManifestArtifactSetResolver | None = None,
) -> tuple[ResolvedArtifactSetItem, ...]:
    if artifact_set.carrier_form == ARTIFACT_SET_CARRIER_INLINE:
        resolved = canonicalize_inline_artifact_selections(
            artifact_set.inline_items or ()
        )
        _verify_resolved_artifact_set(
            artifact_set=artifact_set,
            resolved=resolved,
        )
        return resolved

    if manifest_resolver is None:
        raise ArtifactError(
            "manifest_backed ArtifactSetRef resolution requires an explicit owner-provided resolver",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )

    try:
        resolved_entries = tuple(manifest_resolver(artifact_set))
    except ArtifactError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(
            f"manifest_backed ArtifactSetRef resolution failed: {exc}",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        ) from exc
    resolved = canonicalize_artifact_selections(
        tuple(
            _clone_selection(
                entry.selection if isinstance(entry, ResolvedArtifactSetItem) else entry
            )
            for entry in resolved_entries
        )
    )
    _verify_resolved_artifact_set(
        artifact_set=artifact_set,
        resolved=resolved,
    )
    return resolved


def summarize_artifact_set_outcomes(
    *,
    action_name: str,
    outcomes: Sequence[ArtifactSetItemResult],
) -> OperationStatus:
    failed = [
        item
        for item in outcomes
        if item.status is None or item.status.state != "success"
    ]
    success_count = len(outcomes) - len(failed)
    message = (
        f"{action_name} guarantees local_replica_ready only: "
        f"{success_count}/{len(outcomes)} items reached that floor"
    )
    if not failed:
        return OperationStatus(
            state="success",
            message=message,
            as_of_ms=int(time.time() * 1000),
        )
    first_failed = failed[0].status
    if success_count == 0:
        return OperationStatus(
            state="failed",
            message=message,
            as_of_ms=int(time.time() * 1000),
            error=OperationError(
                status_code=(
                    first_failed.error.status_code
                    if first_failed is not None and first_failed.error is not None
                    else "FAILED_PRECONDITION"
                ),
                message=message,
                retryable=(
                    first_failed.error.retryable
                    if first_failed is not None and first_failed.error is not None
                    else False
                ),
            ),
        )
    return OperationStatus(
        state="degraded",
        message=message,
        as_of_ms=int(time.time() * 1000),
        error=OperationError(
            status_code=(
                first_failed.error.status_code
                if first_failed is not None and first_failed.error is not None
                else "FAILED_PRECONDITION"
            ),
            message=message,
            retryable=(
                first_failed.error.retryable
                if first_failed is not None and first_failed.error is not None
                else False
            ),
        ),
    )


def _verify_resolved_artifact_set(
    *,
    artifact_set: ArtifactSetRef,
    resolved: Sequence[ResolvedArtifactSetItem],
) -> None:
    resolved_digest = compute_artifact_set_digest_hex(resolved)
    resolved_count = len(resolved)
    if resolved_digest != artifact_set.set_digest_hex:
        raise ArtifactError(
            "ArtifactSetRef digest does not match resolved canonical item set",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )
    if resolved_count != artifact_set.item_count:
        raise ArtifactError(
            "ArtifactSetRef item_count does not match resolved canonical item set",
            status_code="FAILED_PRECONDITION",
            retryable=False,
        )


def _clone_selection(
    selection: common_pb2.ArtifactSelection | None,
) -> common_pb2.ArtifactSelection:
    if selection is None:
        raise ArtifactError(
            "ArtifactSelection is required",
            status_code="INVALID_ARGUMENT",
            retryable=False,
        )
    cloned = common_pb2.ArtifactSelection()
    cloned.CopyFrom(selection)
    return cloned


def _selection_identity_key(identity: SelectionIdentity) -> tuple[str, bytes, bytes]:
    return (
        str(identity.artifact_id),
        bytes(identity.logical_layout_hash),
        bytes(identity.selection_hash),
    )


def _selection_identity_sort_key(identity: SelectionIdentity) -> tuple[str, str, str]:
    return (
        str(identity.artifact_id),
        bytes(identity.logical_layout_hash).hex(),
        bytes(identity.selection_hash).hex(),
    )


def _unique_identity_keys(
    identities: Iterable[SelectionIdentity] | Iterable[ResolvedArtifactSetItem],
) -> tuple[tuple[str, bytes, bytes], ...]:
    unique_keys: set[tuple[str, bytes, bytes]] = set()
    for entry in identities:
        identity = (
            entry.item_identity if isinstance(entry, ResolvedArtifactSetItem) else entry
        )
        unique_keys.add(_selection_identity_key(identity))
    return tuple(
        sorted(unique_keys, key=lambda item: (item[0], item[1].hex(), item[2].hex()))
    )


__all__ = [
    "ARTIFACT_SET_CARRIER_INLINE",
    "ARTIFACT_SET_CARRIER_MANIFEST_BACKED",
    "MAX_INLINE_ARTIFACT_SET_ITEMS",
    "ArtifactSetItemResult",
    "ArtifactSetRef",
    "ArtifactSetResult",
    "ManifestArtifactSetResolver",
    "ResolvedArtifactSetItem",
    "canonicalize_artifact_selections",
    "canonicalize_inline_artifact_selections",
    "compute_artifact_set_digest_hex",
    "resolve_artifact_set_ref",
    "selection_identity_from_proto",
    "selection_identity_to_proto",
    "summarize_artifact_set_outcomes",
]
