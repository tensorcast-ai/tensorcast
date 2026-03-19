#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for byte-space leaf digests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from tensorcast.global_store.repositories.base import BaseRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LeafRow:
    """Projected row from the `leaves` table."""

    artifact_id: str
    space_kind: str
    space_id: str
    leaf_idx: int
    digest: bytes


class LeafRepository(BaseRepository):
    """Data access layer for `leaves` table."""

    def delete_all_for_space(
        self,
        *,
        artifact_id: str,
        space_kind: str,
        space_id: str,
        cursor=None,
    ) -> None:
        normalized_kind = space_kind.upper()
        if normalized_kind not in {"C", "V"}:
            raise ValueError(f"Invalid space_kind: {space_kind}")
        target = cursor if cursor is not None else self.get_cursor()
        target.execute(
            """
            DELETE FROM leaves
            WHERE artifact_id = ?
              AND space_kind = ?
              AND space_id = ?
            """,
            [artifact_id, normalized_kind, space_id],
        )

    def delete_indices(
        self,
        *,
        artifact_id: str,
        space_kind: str,
        space_id: str,
        leaf_idxs: Sequence[int],
        cursor=None,
    ) -> None:
        normalized_kind = space_kind.upper()
        if normalized_kind not in {"C", "V"}:
            raise ValueError(f"Invalid space_kind: {space_kind}")
        normalized_idxs = sorted({int(idx) for idx in leaf_idxs})
        if not normalized_idxs:
            return
        target = cursor if cursor is not None else self.get_cursor()
        chunk_size = 1000
        for start in range(0, len(normalized_idxs), chunk_size):
            chunk = normalized_idxs[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            target.execute(
                f"""
                DELETE FROM leaves
                WHERE artifact_id = ?
                  AND space_kind = ?
                  AND space_id = ?
                  AND leaf_idx IN ({placeholders})
                """,
                [artifact_id, normalized_kind, space_id, *chunk],
            )

    def upsert_many(
        self,
        *,
        artifact_id: str,
        space_kind: str,
        space_id: str,
        entries: Iterable[Tuple[int, bytes]],
        cursor=None,
    ) -> int:
        """Insert many leaf digests with idempotent commit semantics.

        Returns the number of newly inserted digest entries. If an entry already
        exists, its digest must match or the write is rejected.
        """
        normalized_kind = space_kind.upper()
        if normalized_kind not in {"C", "V"}:
            raise ValueError(f"Invalid space_kind: {space_kind}")

        requested: dict[int, bytes] = {}
        for idx, digest in entries:
            leaf_idx = int(idx)
            if leaf_idx in requested and requested[leaf_idx] != digest:
                raise ValueError("leaves conflict")
            requested[leaf_idx] = digest
        if not requested:
            return 0

        try:
            target = cursor if cursor is not None else self.get_cursor()
            leaf_idxs = list(requested.keys())

            existing: dict[int, bytes] = {}
            chunk_size = 1000
            for start in range(0, len(leaf_idxs), chunk_size):
                chunk = leaf_idxs[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = target.execute(
                    f"""
                    SELECT leaf_idx, digest
                    FROM leaves
                    WHERE artifact_id = ?
                      AND space_kind = ?
                      AND space_id = ?
                      AND leaf_idx IN ({placeholders})
                    """,
                    [artifact_id, normalized_kind, space_id, *chunk],
                ).fetchall()
                for row in rows:
                    existing[int(row[0])] = bytes(row[1])

            to_insert = []
            for leaf_idx, digest in requested.items():
                if leaf_idx in existing:
                    if existing[leaf_idx] != digest:
                        raise ValueError("leaves conflict")
                    continue
                to_insert.append(
                    (artifact_id, normalized_kind, space_id, leaf_idx, digest)
                )

            if not to_insert:
                return 0

            target.executemany(
                """
                INSERT INTO leaves (
                    artifact_id,
                    space_kind,
                    space_id,
                    leaf_idx,
                    digest
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (artifact_id, space_kind, space_id, leaf_idx)
                DO NOTHING
                """,
                to_insert,
            )

            # Re-fetch inserted keys defensively (race-safe) and confirm equality.
            inserted_idxs = [int(row[3]) for row in to_insert]
            for start in range(0, len(inserted_idxs), chunk_size):
                chunk = inserted_idxs[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = target.execute(
                    f"""
                    SELECT leaf_idx, digest
                    FROM leaves
                    WHERE artifact_id = ?
                      AND space_kind = ?
                      AND space_id = ?
                      AND leaf_idx IN ({placeholders})
                    """,
                    [artifact_id, normalized_kind, space_id, *chunk],
                ).fetchall()
                for row in rows:
                    leaf_idx = int(row[0])
                    digest = bytes(row[1])
                    expected = requested.get(leaf_idx)
                    if expected is not None and expected != digest:
                        raise ValueError("leaves conflict")

            return len(to_insert)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to upsert leaves for artifact_id=%s space_id=%s",
                artifact_id,
                space_id,
            )
            raise

    def fetch(
        self,
        *,
        artifact_id: str,
        space_kind: str,
        space_id: str,
        leaf_idxs: Optional[Sequence[int]] = None,
        cursor=None,
    ) -> List[LeafRow]:
        """Fetch leaf digests for a given byte-space."""
        normalized_kind = space_kind.upper()
        if normalized_kind not in {"C", "V"}:
            raise ValueError(f"Invalid space_kind: {space_kind}")

        if leaf_idxs is not None and len(leaf_idxs) == 0:
            return []

        params: List[object] = [artifact_id, normalized_kind, space_id]
        where_clause = ""
        if leaf_idxs:
            placeholders = ",".join("?" for _ in leaf_idxs)
            where_clause = f" AND leaf_idx IN ({placeholders})"
            params.extend(int(idx) for idx in leaf_idxs)

        query = f"""
            SELECT artifact_id,
                   space_kind,
                   space_id,
                   leaf_idx,
                   digest
            FROM leaves
            WHERE artifact_id = ?
              AND space_kind = ?
              AND space_id = ?{where_clause}
            ORDER BY leaf_idx ASC
        """

        target = cursor if cursor is not None else self.get_cursor()
        rows = target.execute(query, params).fetchall()
        return [
            LeafRow(
                artifact_id=row[0],
                space_kind=row[1],
                space_id=row[2],
                leaf_idx=row[3],
                digest=row[4],
            )
            for row in rows
        ]
