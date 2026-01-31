#  Copyright (c) 2025-2026, TensorCast Team.

"""Repositories for overlap proof commitment tables (v2)."""

from __future__ import annotations

from tensorcast.global_store.repositories.base import BaseRepository


class ProofRepository(BaseRepository):
    """Data access helper for proof commitment tables."""

    def upsert_piece_proof_digest(
        self,
        *,
        assembly_id: str,
        view_id: str,
        tensor_name: str,
        proof_schema_version: str,
        proof_chunk_idx: int,
        digest: bytes,
        cursor=None,
    ) -> bool:
        target = cursor if cursor is not None else self.get_cursor()
        existing = target.execute(
            """
            SELECT digest
            FROM piece_proof_digests
            WHERE assembly_id = ?
              AND view_id = ?
              AND tensor_name = ?
              AND proof_schema_version = ?
              AND proof_chunk_idx = ?
            """,
            [
                assembly_id,
                view_id,
                tensor_name,
                proof_schema_version,
                int(proof_chunk_idx),
            ],
        ).fetchone()
        if existing is not None:
            if bytes(existing[0]) != digest:
                raise ValueError("piece_proof_digests conflict")
            return False
        target.execute(
            """
            INSERT INTO piece_proof_digests (
                assembly_id,
                view_id,
                tensor_name,
                proof_schema_version,
                proof_chunk_idx,
                digest
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                assembly_id,
                view_id,
                tensor_name,
                proof_schema_version,
                int(proof_chunk_idx),
                digest,
            ],
        )
        return True

    def upsert_assembly_proof_commitment(
        self,
        *,
        assembly_id: str,
        tensor_name: str,
        proof_schema_version: str,
        proof_chunk_idx: int,
        digest: bytes,
        cursor=None,
    ) -> bool:
        target = cursor if cursor is not None else self.get_cursor()
        existing = target.execute(
            """
            SELECT digest
            FROM assembly_proof_commitments
            WHERE assembly_id = ?
              AND tensor_name = ?
              AND proof_schema_version = ?
              AND proof_chunk_idx = ?
            """,
            [assembly_id, tensor_name, proof_schema_version, int(proof_chunk_idx)],
        ).fetchone()
        if existing is not None:
            if bytes(existing[0]) != digest:
                raise ValueError("assembly_proof_commitments conflict")
            return False
        target.execute(
            """
            INSERT INTO assembly_proof_commitments (
                assembly_id,
                tensor_name,
                proof_schema_version,
                proof_chunk_idx,
                digest
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                assembly_id,
                tensor_name,
                proof_schema_version,
                int(proof_chunk_idx),
                digest,
            ],
        )
        return True

    def upsert_tensor_proof_commitment(
        self,
        *,
        mi2_id: str,
        tensor_name: str,
        proof_schema_version: str,
        proof_chunk_idx: int,
        digest: bytes,
        cursor=None,
    ) -> bool:
        target = cursor if cursor is not None else self.get_cursor()
        existing = target.execute(
            """
            SELECT digest
            FROM tensor_proof_commitments
            WHERE mi2_id = ?
              AND tensor_name = ?
              AND proof_schema_version = ?
              AND proof_chunk_idx = ?
            """,
            [mi2_id, tensor_name, proof_schema_version, int(proof_chunk_idx)],
        ).fetchone()
        if existing is not None:
            if bytes(existing[0]) != digest:
                raise ValueError("tensor_proof_commitments conflict")
            return False
        target.execute(
            """
            INSERT INTO tensor_proof_commitments (
                mi2_id,
                tensor_name,
                proof_schema_version,
                proof_chunk_idx,
                digest
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [mi2_id, tensor_name, proof_schema_version, int(proof_chunk_idx), digest],
        )
        return True

    def commitments_match(
        self,
        *,
        assembly_id: str,
        mi2_id: str,
        proof_schema_version: str,
        tensor_names: list[str],
        cursor=None,
    ) -> bool:
        if not tensor_names:
            raise ValueError("tensor_names is required")
        for name in tensor_names:
            if not name:
                raise ValueError("tensor_names must be non-empty")

        names = sorted(set(tensor_names))
        if not names:
            raise ValueError("tensor_names is required")

        target = cursor if cursor is not None else self.get_cursor()
        placeholders = ",".join(["?"] * len(names))

        assembly_counts = target.execute(
            f"""
            SELECT tensor_name, COUNT(*) AS count
            FROM assembly_proof_commitments
            WHERE assembly_id = ?
              AND proof_schema_version = ?
              AND tensor_name IN ({placeholders})
            GROUP BY tensor_name
            """,
            [assembly_id, proof_schema_version, *names],
        ).fetchall()
        assembly_by_tensor = {row[0]: int(row[1]) for row in assembly_counts}
        for name in names:
            if assembly_by_tensor.get(name, 0) <= 0:
                return False

        tensor_counts = target.execute(
            f"""
            SELECT tensor_name, COUNT(*) AS count
            FROM tensor_proof_commitments
            WHERE mi2_id = ?
              AND proof_schema_version = ?
              AND tensor_name IN ({placeholders})
            GROUP BY tensor_name
            """,
            [mi2_id, proof_schema_version, *names],
        ).fetchall()
        tensor_by_name = {row[0]: int(row[1]) for row in tensor_counts}
        for name in names:
            if tensor_by_name.get(name, 0) <= 0:
                return False

        assembly_select = (
            "SELECT tensor_name, proof_chunk_idx, digest "
            "FROM assembly_proof_commitments "
            "WHERE assembly_id = ? "
            "AND proof_schema_version = ? "
            f"AND tensor_name IN ({placeholders})"
        )
        tensor_select = (
            "SELECT tensor_name, proof_chunk_idx, digest "
            "FROM tensor_proof_commitments "
            "WHERE mi2_id = ? "
            "AND proof_schema_version = ? "
            f"AND tensor_name IN ({placeholders})"
        )
        diff = target.execute(
            f"SELECT 1 FROM ({assembly_select} EXCEPT {tensor_select}) LIMIT 1",
            [
                assembly_id,
                proof_schema_version,
                *names,
                mi2_id,
                proof_schema_version,
                *names,
            ],
        ).fetchone()
        if diff is not None:
            return False

        diff = target.execute(
            f"SELECT 1 FROM ({tensor_select} EXCEPT {assembly_select}) LIMIT 1",
            [
                mi2_id,
                proof_schema_version,
                *names,
                assembly_id,
                proof_schema_version,
                *names,
            ],
        ).fetchone()
        return diff is None
