#  Copyright (c) 2026, TensorCast Team.

"""Repository for immutable group version-set manifests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from google.protobuf.message import Message

from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.proto.common.v1 import common_pb2

_SAME_SELECTION = "same_selection"
_PER_PART_SELECTION = "per_part_selection"
_BYTE_SPACE_KIND_BY_VALUE: dict[int, common_pb2.ByteSpaceKind] = {
    int(common_pb2.BYTE_SPACE_KIND_UNSPECIFIED): common_pb2.BYTE_SPACE_KIND_UNSPECIFIED,
    int(common_pb2.BYTE_SPACE_KIND_CANONICAL): common_pb2.BYTE_SPACE_KIND_CANONICAL,
    int(common_pb2.BYTE_SPACE_KIND_VIEW): common_pb2.BYTE_SPACE_KIND_VIEW,
}


def _bytes_or_none(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    raise TypeError(f"expected bytes-like value, got {type(value).__name__}")


def _byte_space_kind_from_value(value: object) -> common_pb2.ByteSpaceKind:
    if isinstance(value, bool):
        raise TypeError("expected byte space kind int, got bool")
    if isinstance(value, int | str):
        kind_value = int(value)
    else:
        raise TypeError(f"expected byte space kind int, got {type(value).__name__}")
    try:
        return _BYTE_SPACE_KIND_BY_VALUE[kind_value]
    except KeyError as exc:
        raise ValueError(f"unsupported byte space kind: {kind_value}") from exc


def _byte_space_json(byte_space: common_pb2.ByteSpaceRef) -> str:
    return json.dumps(
        {
            "kind": int(byte_space.kind),
            "id": byte_space.id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _default_byte_space_for_selection(
    selection: common_pb2.ArtifactSelection,
) -> common_pb2.ByteSpaceRef:
    if selection.view_id:
        return common_pb2.ByteSpaceRef(
            kind=common_pb2.ByteSpaceKind.BYTE_SPACE_KIND_VIEW,
            id=selection.view_id,
        )
    return common_pb2.ByteSpaceRef(
        kind=common_pb2.ByteSpaceKind.BYTE_SPACE_KIND_CANONICAL,
    )


def parse_requested_byte_space_json(raw: str) -> common_pb2.ByteSpaceRef:
    data = json.loads(raw)
    if not isinstance(data, Mapping):
        raise ValueError("requested_byte_space must be a JSON object")
    return common_pb2.ByteSpaceRef(
        kind=_byte_space_kind_from_value(
            data.get("kind", int(common_pb2.BYTE_SPACE_KIND_UNSPECIFIED))
        ),
        id=str(data.get("id", "") or ""),
    )


def _deterministic_proto_bytes(message: Message) -> bytes:
    return message.SerializeToString(deterministic=True)


def _selection_hash(selection: common_pb2.ArtifactSelection) -> bytes:
    if selection.selection_hash:
        return bytes(selection.selection_hash)
    return hashlib.sha256(
        b"tensorcast.artifact_selection.v1\0" + _deterministic_proto_bytes(selection)
    ).digest()


def _manifest_hash(*, realization_kind: str, parts: list[dict[str, Any]]) -> bytes:
    payload = {
        "realization_kind": realization_kind,
        "parts": [
            {
                "part_id": str(part["part_id"]),
                "artifact_id": str(part["artifact_id"]),
                "view_id": str(part.get("view_id") or ""),
                "requested_byte_space": str(part["requested_byte_space"]),
                "selection_hash": bytes(part["selection_hash"]).hex(),
                "logical_layout_hash": bytes(
                    part.get("logical_layout_hash") or b""
                ).hex(),
            }
            for part in sorted(parts, key=lambda item: str(item["part_id"]))
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(b"tensorcast.group_version_set.v1\0" + canonical).digest()


def _version_set_id(manifest_hash: bytes) -> str:
    return f"gvs_{manifest_hash.hex()}"


class GroupVersionSetRepository(BaseRepository):
    """Data access for group version-set manifests and parts."""

    def _row_to_version_set(self, row: tuple[Any, ...]) -> dict[str, Any]:
        return {
            "version_set_id": str(row[0]),
            "realization_kind": str(row[1]),
            "namespace": row[2],
            "key": row[3],
            "key_generation": int(row[4]) if row[4] is not None else None,
            "total_parts": int(row[5]),
            "manifest_hash": bytes(row[6]),
            "manifest_generation": int(row[7]),
            "logical_layout_hash": _bytes_or_none(row[8]),
            "created_at": row[9],
        }

    def _row_to_part(self, row: tuple[Any, ...]) -> dict[str, Any]:
        selection = common_pb2.ArtifactSelection()
        if row[8] is not None:
            selection.ParseFromString(bytes(row[8]))
        else:
            selection.artifact_id = str(row[2])
            if row[3]:
                selection.view_id = str(row[3])
            selection.selection_hash = bytes(row[5])
            if row[6] is not None:
                selection.logical_layout_hash = bytes(row[6])
        return {
            "version_set_id": str(row[0]),
            "part_id": str(row[1]),
            "artifact_id": str(row[2]),
            "view_id": row[3],
            "requested_byte_space": str(row[4]),
            "selection_hash": bytes(row[5]),
            "logical_layout_hash": _bytes_or_none(row[6]),
            "part_metadata_json": row[7],
            "selection": selection,
        }

    def _get_by_manifest_hash(
        self, *, manifest_hash: bytes, cursor
    ) -> dict[str, Any] | None:
        row = cursor.execute(
            """
            SELECT version_set_id,
                   realization_kind,
                   namespace,
                   key,
                   key_generation,
                   total_parts,
                   manifest_hash,
                   manifest_generation,
                   logical_layout_hash,
                   created_at
            FROM group_version_sets
            WHERE manifest_hash = ?
            """,
            [manifest_hash],
        ).fetchone()
        if row is None:
            return None
        return self._row_to_version_set(row)

    def _select_parts(self, *, version_set_id: str, cursor) -> list[dict[str, Any]]:
        rows = cursor.execute(
            """
            SELECT version_set_id,
                   part_id,
                   artifact_id,
                   view_id,
                   requested_byte_space,
                   selection_hash,
                   logical_layout_hash,
                   part_metadata_json,
                   selection_proto
            FROM group_version_set_parts
            WHERE version_set_id = ?
            ORDER BY part_id
            """,
            [version_set_id],
        ).fetchall()
        return [self._row_to_part(row) for row in rows]

    def _insert_manifest(
        self,
        *,
        cursor,
        realization_kind: str,
        parts: list[dict[str, Any]],
        manifest_hash: bytes,
        namespace: str | None,
        key: str | None,
        key_generation: int | None,
    ) -> dict[str, Any]:
        version_set_id = _version_set_id(manifest_hash)
        logical_layout_hash = next(
            (
                bytes(part["logical_layout_hash"])
                for part in parts
                if part.get("logical_layout_hash")
            ),
            None,
        )
        cursor.execute(
            """
            INSERT INTO group_version_sets (
              version_set_id,
              realization_kind,
              namespace,
              key,
              key_generation,
              total_parts,
              manifest_hash,
              logical_layout_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                version_set_id,
                realization_kind,
                namespace,
                key,
                key_generation,
                len(parts),
                manifest_hash,
                logical_layout_hash,
            ],
        )
        for part in parts:
            selection = part["selection"]
            cursor.execute(
                """
                INSERT INTO group_version_set_parts (
                  version_set_id,
                  part_id,
                  artifact_id,
                  view_id,
                  requested_byte_space,
                  selection_hash,
                  logical_layout_hash,
                  part_metadata_json,
                  selection_proto
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    version_set_id,
                    part["part_id"],
                    part["artifact_id"],
                    part.get("view_id"),
                    part["requested_byte_space"],
                    part["selection_hash"],
                    part.get("logical_layout_hash"),
                    part.get("part_metadata_json"),
                    _deterministic_proto_bytes(selection),
                ],
            )
        inserted = self._get_by_manifest_hash(
            manifest_hash=manifest_hash,
            cursor=cursor,
        )
        if inserted is None:
            raise ValueError("group version set missing after insert")
        return inserted

    def create_or_get_same_selection(
        self,
        *,
        selection: common_pb2.ArtifactSelection,
        required_part_ids: list[str],
        requested_byte_space: common_pb2.ByteSpaceRef | None = None,
        namespace: str | None = None,
        key: str | None = None,
        key_generation: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Create or load a `same_selection` manifest for concrete parts."""
        part_ids = sorted({str(part_id).strip() for part_id in required_part_ids})
        if not part_ids:
            raise ValueError("required_part_ids must not be empty")
        if "" in part_ids:
            raise ValueError("part ids must not be empty")
        byte_space = requested_byte_space or _default_byte_space_for_selection(
            selection
        )
        byte_space_encoded = _byte_space_json(byte_space)
        selection_digest = _selection_hash(selection)
        logical_layout_hash = bytes(selection.logical_layout_hash or b"")
        parts = [
            {
                "part_id": part_id,
                "artifact_id": selection.artifact_id,
                "view_id": selection.view_id or None,
                "requested_byte_space": byte_space_encoded,
                "selection_hash": selection_digest,
                "logical_layout_hash": logical_layout_hash,
                "selection": selection,
            }
            for part_id in part_ids
        ]
        manifest_hash = _manifest_hash(
            realization_kind=_SAME_SELECTION,
            parts=parts,
        )
        with self.transaction() as cursor:
            existing = self._get_by_manifest_hash(
                manifest_hash=manifest_hash,
                cursor=cursor,
            )
            if existing is None:
                existing = self._insert_manifest(
                    cursor=cursor,
                    realization_kind=_SAME_SELECTION,
                    parts=parts,
                    manifest_hash=manifest_hash,
                    namespace=namespace,
                    key=key,
                    key_generation=key_generation,
                )
            return existing, self._select_parts(
                version_set_id=existing["version_set_id"],
                cursor=cursor,
            )

    def register_per_part_selection(
        self,
        *,
        parts: list[dict[str, Any]],
        namespace: str | None = None,
        key: str | None = None,
        key_generation: int | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Create or load a `per_part_selection` manifest."""
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for part in parts:
            part_id = str(part["part_id"]).strip()
            if not part_id:
                raise ValueError("part_id is required")
            if part_id in seen:
                raise ValueError(f"duplicate part_id: {part_id}")
            seen.add(part_id)
            selection = part["selection"]
            if not isinstance(selection, common_pb2.ArtifactSelection):
                raise TypeError("selection must be ArtifactSelection")
            byte_space = part.get("requested_byte_space")
            if byte_space is None:
                byte_space = _default_byte_space_for_selection(selection)
            if not isinstance(byte_space, common_pb2.ByteSpaceRef):
                raise TypeError("requested_byte_space must be ByteSpaceRef")
            normalized.append(
                {
                    "part_id": part_id,
                    "artifact_id": selection.artifact_id,
                    "view_id": selection.view_id or None,
                    "requested_byte_space": _byte_space_json(byte_space),
                    "selection_hash": _selection_hash(selection),
                    "logical_layout_hash": bytes(selection.logical_layout_hash or b""),
                    "part_metadata_json": part.get("part_metadata_json"),
                    "selection": selection,
                }
            )
        if not normalized:
            raise ValueError("parts must not be empty")
        manifest_hash = _manifest_hash(
            realization_kind=_PER_PART_SELECTION,
            parts=normalized,
        )
        with self.transaction() as cursor:
            existing = self._get_by_manifest_hash(
                manifest_hash=manifest_hash,
                cursor=cursor,
            )
            if existing is None:
                existing = self._insert_manifest(
                    cursor=cursor,
                    realization_kind=_PER_PART_SELECTION,
                    parts=normalized,
                    manifest_hash=manifest_hash,
                    namespace=namespace,
                    key=key,
                    key_generation=key_generation,
                )
            return existing, self._select_parts(
                version_set_id=existing["version_set_id"],
                cursor=cursor,
            )

    def get(
        self,
        *,
        version_set_id: str,
        cursor=None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        target = cursor if cursor is not None else self.get_cursor()
        close = cursor is None
        try:
            row = target.execute(
                """
                SELECT version_set_id,
                       realization_kind,
                       namespace,
                       key,
                       key_generation,
                       total_parts,
                       manifest_hash,
                       manifest_generation,
                       logical_layout_hash,
                       created_at
                FROM group_version_sets
                WHERE version_set_id = ?
                """,
                [version_set_id],
            ).fetchone()
            if row is None:
                return None
            version_set = self._row_to_version_set(row)
            parts = self._select_parts(version_set_id=version_set_id, cursor=target)
            return version_set, parts
        finally:
            if close:
                target.close()

    def parse_requested_byte_space(self, raw: str) -> common_pb2.ByteSpaceRef:
        return parse_requested_byte_space_json(raw)
