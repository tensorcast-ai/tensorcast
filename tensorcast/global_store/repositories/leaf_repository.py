#  Copyright (c) 2025, TensorCast Team.

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

    def upsert_many(
        self,
        *,
        artifact_id: str,
        space_kind: str,
        space_id: str,
        entries: Iterable[Tuple[int, bytes]],
        cursor=None,
    ) -> None:
        """Insert or update many leaf digests."""
        normalized_kind = space_kind.upper()
        if normalized_kind not in {"C", "V"}:
            raise ValueError(f"Invalid space_kind: {space_kind}")

        batched = [
            (artifact_id, normalized_kind, space_id, idx, digest)
            for idx, digest in entries
        ]
        if not batched:
            return

        try:
            target = cursor if cursor is not None else self.get_cursor()
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
                DO UPDATE SET digest = EXCLUDED.digest
                """,
                batched,
            )
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
