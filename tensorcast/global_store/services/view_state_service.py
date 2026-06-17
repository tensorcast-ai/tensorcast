#  Copyright (c) 2025-2026, TensorCast Team.

"""Service helpers for view metadata, leaf digests, and overlap proofs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tensorcast.global_store import metrics
from tensorcast.global_store.repositories.assembly_layout_binding_repository import (
    AssemblyLayoutBindingRepository,
)
from tensorcast.global_store.repositories.layout_spec_repository import (
    LayoutSpecRepository,
)
from tensorcast.global_store.repositories.leaf_repository import LeafRepository
from tensorcast.global_store.repositories.proof_repository import ProofRepository
from tensorcast.global_store.repositories.view_coverage_repository import (
    ViewCoverageRepository,
)
from tensorcast.global_store.repositories.view_repository import ViewRepository
from tensorcast.logger import init_logger
from tensorcast.proto.layout.v1 import layout_pb2

logger = init_logger(__name__)

PROOF_SCHEMA_V1 = "v1"
PROOF_CHUNK_BYTES_V1 = 4 * 1024 * 1024


@dataclass(frozen=True)
class ViewUpsertPayload:
    """Payload for view UPSERT requests."""

    artifact_id: str
    view_id: str
    view_spec_json: str
    view_size: int
    view_data_hash: Optional[str]
    verified_at: Optional[datetime]
    canonical_size_bytes: Optional[int] = None
    canonical_bytes_covered: Optional[int] = None
    canonical_ranges: Optional[Sequence[Tuple[int, int]]] = None


@dataclass(frozen=True)
class LeafWritePayload:
    """Payload for batched leaf writes."""

    artifact_id: str
    space_kind: str
    space_id: str
    leaf_idx: int
    digest: bytes


@dataclass(frozen=True)
class PieceProofDigestPayload:
    """Payload for per-piece overlap proof digests."""

    artifact_id: str
    view_id: str
    tensor_name: str
    proof_schema_version: str
    proof_chunk_idx: int
    digest: bytes


def _normalize_ranges(ranges: Sequence[Tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    for off, length in ranges:
        length_i = int(length)
        if length_i <= 0:
            continue
        off_i = int(off)
        if off_i < 0:
            raise ValueError("canonical range offset must be >= 0")
        cleaned.append((off_i, length_i))
    cleaned.sort(key=lambda x: x[0])
    out: list[tuple[int, int]] = []
    for off, length in cleaned:
        if not out:
            out.append((off, length))
            continue
        prev_off, prev_len = out[-1]
        prev_end = prev_off + prev_len
        if off < prev_end:
            raise ValueError("overlapping canonical ranges within a single view")
        if off == prev_end:
            out[-1] = (prev_off, prev_len + length)
            continue
        out.append((off, length))
    return out


def _ranges_cover_interval(
    ranges: Sequence[Tuple[int, int]], *, start: int, length: int
) -> bool:
    if length <= 0:
        return True
    cursor = int(start)
    end = int(start + length)
    for off, seg_len in ranges:
        seg_start = int(off)
        seg_end = int(off + seg_len)
        if seg_end <= cursor:
            continue
        if seg_start > cursor:
            return False
        cursor = min(end, seg_end)
        if cursor >= end:
            return True
    return cursor >= end


def _ranges_intersect_interval(
    ranges: Sequence[Tuple[int, int]], *, start: int, length: int
) -> bool:
    if length <= 0:
        return False
    end = int(start + length)
    for off, seg_len in ranges:
        seg_start = int(off)
        seg_end = int(off + seg_len)
        if seg_start < end and seg_end > start:
            return True
    return False


class ViewStateService:
    """Business logic bridging view metadata, leaf digests, and overlap proofs."""

    def __init__(
        self,
        view_repository: ViewRepository,
        leaf_repository: LeafRepository,
        coverage_repository: ViewCoverageRepository,
        layout_repository: LayoutSpecRepository,
        layout_binding_repository: AssemblyLayoutBindingRepository,
        proof_repository: ProofRepository,
    ) -> None:
        self.view_repository = view_repository
        self.leaf_repository = leaf_repository
        self.coverage_repository = coverage_repository
        self.layout_repository = layout_repository
        self.layout_binding_repository = layout_binding_repository
        self.proof_repository = proof_repository

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update_view_state(
        self,
        *,
        view: Optional[ViewUpsertPayload],
        leaf_writes: Sequence[LeafWritePayload],
        proof_digests: Sequence[PieceProofDigestPayload],
        tensor_intervals: dict[str, tuple[int, int]] | None,
    ) -> None:
        """Apply view metadata, leaf digests, and proof digests atomically."""
        if view is None and not leaf_writes and not proof_digests:
            return

        if view is None and proof_digests:
            raise ValueError("proof digests require a view payload")

        logger.debug(
            "Updating view state: view=%s leaf_count=%d proof_count=%d",
            view.view_id if view else "none",
            len(leaf_writes),
            len(proof_digests),
        )

        leaf_inserted = 0
        piece_proof_inserted = 0
        assembly_proof_inserted = 0

        artifact_id = view.artifact_id if view is not None else None
        if artifact_id is None and leaf_writes:
            artifact_id = leaf_writes[0].artifact_id
        if artifact_id is None and proof_digests:
            artifact_id = proof_digests[0].artifact_id

        if artifact_id is None:
            raise ValueError(
                "artifact_id must be provided for view, proof, or leaf writes"
            )

        for leaf_entry in leaf_writes:
            if leaf_entry.artifact_id != artifact_id:
                raise ValueError("artifact_id mismatch within leaf writes batch")
        for proof_entry in proof_digests:
            if proof_entry.artifact_id != artifact_id:
                raise ValueError("artifact_id mismatch within proof digest batch")

        artifact_is_assembly = artifact_id.startswith("cgid:")

        with self.view_repository.transaction() as cursor:
            active_layout: layout_pb2.LayoutSpec | None = None
            if artifact_is_assembly:
                binding = self.layout_binding_repository.get(
                    assembly_id=artifact_id, cursor=cursor
                )
                if binding is not None:
                    record = self.layout_repository.get(
                        layout_id=str(binding["layout_id"]), cursor=cursor
                    )
                    if record is None:
                        raise ValueError(
                            "layout_id not found for assembly layout binding"
                        )
                    active_layout = layout_pb2.LayoutSpec()
                    active_layout.ParseFromString(record["layout_proto"])

            replicated_tensors: set[str] = set()
            proof_schema_version: str | None = None
            replace_existing_assembly_view = False
            if active_layout is not None:
                for tensor_name, policy in active_layout.tensors.items():
                    if policy.overlap_mode == layout_pb2.OVERLAP_MODE_REPLICATE_EQUAL:
                        replicated_tensors.add(tensor_name)
                proof_schema_version = active_layout.proof_schema_version or None

            if view is not None:
                existing = self.view_repository.get(
                    artifact_id=view.artifact_id,
                    view_id=view.view_id,
                    cursor=cursor,
                )
                if (
                    existing
                    and existing.get("view_data_hash")
                    and view.view_data_hash
                    and existing.get("view_data_hash") != view.view_data_hash
                ):
                    if artifact_is_assembly:
                        replace_existing_assembly_view = True
                    else:
                        raise ValueError(
                            "view_data_hash conflict for existing view registration"
                        )

                # Metadata-only updates omit fields (empty/None); preserve the
                # previously stored values instead of clobbering them. The repo
                # upsert is a pure overwrite, so the merge happens here.
                view_spec_json = view.view_spec_json
                view_data_hash = view.view_data_hash
                verified_at = view.verified_at
                canonical_size_bytes = view.canonical_size_bytes
                canonical_bytes_covered = view.canonical_bytes_covered
                if existing is not None:
                    if not view_spec_json:
                        view_spec_json = existing.get("view_spec_json") or ""
                    if view_data_hash is None:
                        view_data_hash = existing.get("view_data_hash")
                    if verified_at is None:
                        verified_at = existing.get("verified_at")
                    if canonical_size_bytes is None:
                        canonical_size_bytes = existing.get("canonical_size_bytes")
                    if canonical_bytes_covered is None:
                        canonical_bytes_covered = existing.get(
                            "canonical_bytes_covered"
                        )

                self.view_repository.upsert(
                    artifact_id=view.artifact_id,
                    view_id=view.view_id,
                    view_spec_json=view_spec_json,
                    view_size=view.view_size,
                    view_data_hash=view_data_hash,
                    verified_at=verified_at,
                    canonical_size_bytes=canonical_size_bytes,
                    canonical_bytes_covered=canonical_bytes_covered,
                    cursor=cursor,
                )

                ranges = (
                    list(view.canonical_ranges or [])
                    if view.canonical_ranges is not None
                    else []
                )

                total = int(view.canonical_size_bytes or 0)
                covered = int(view.canonical_bytes_covered or 0)
                if total > 0 and covered > total:
                    raise ValueError(
                        "canonical_bytes_covered exceeds canonical_size_bytes"
                    )
                is_partial = total > 0 and covered < total
                if is_partial and not ranges:
                    raise ValueError("coverage metadata missing for partial view")

                if ranges:
                    normalized_ranges = _normalize_ranges(ranges)
                    existing_ranges = self.coverage_repository.get_ranges(
                        artifact_id=view.artifact_id,
                        view_id=view.view_id,
                        cursor=cursor,
                    )
                    if existing_ranges and existing_ranges != normalized_ranges:
                        raise ValueError("canonical range mismatch for view")

                    if artifact_is_assembly:
                        if tensor_intervals is None:
                            raise ValueError(
                                "tensor_intervals required for assembly view updates"
                            )
                        # Default is DISJOINT unless a layout explicitly opts in.
                        if not replicated_tensors:
                            overlaps = self.coverage_repository.find_overlaps(
                                artifact_id=view.artifact_id,
                                view_id=view.view_id,
                                ranges=normalized_ranges,
                                cursor=cursor,
                            )
                            if overlaps:
                                raise ValueError(
                                    "overlapping canonical coverage across views"
                                )
                        else:
                            if not proof_schema_version:
                                raise ValueError(
                                    "proof_schema_version required for replicated tensors"
                                )
                            if proof_schema_version != PROOF_SCHEMA_V1:
                                raise ValueError("unsupported proof_schema_version")
                            # Replicated tensors require full tensor coverage when present.
                            participating: set[str] = set()
                            for tensor_name in replicated_tensors:
                                if tensor_name not in tensor_intervals:
                                    raise ValueError("layout references unknown tensor")
                                tensor_off, tensor_bytes = tensor_intervals[tensor_name]
                                if _ranges_intersect_interval(
                                    normalized_ranges,
                                    start=tensor_off,
                                    length=tensor_bytes,
                                ):
                                    participating.add(tensor_name)
                                    if not _ranges_cover_interval(
                                        normalized_ranges,
                                        start=tensor_off,
                                        length=tensor_bytes,
                                    ):
                                        raise ValueError(
                                            "replicated tensor requires full coverage"
                                        )

                            # Disjoint tensors must not overlap.
                            disjoint_intervals: list[tuple[str, int, int]] = []
                            for tensor_name, (
                                tensor_off,
                                tensor_bytes,
                            ) in tensor_intervals.items():
                                if tensor_name in replicated_tensors:
                                    continue
                                disjoint_intervals.append(
                                    (
                                        tensor_name,
                                        int(tensor_off),
                                        int(tensor_off + tensor_bytes),
                                    )
                                )
                            disjoint_intervals.sort(key=lambda x: x[1])
                            t_idx = 0
                            for off, length in normalized_ranges:
                                r_start = int(off)
                                r_end = int(off + length)
                                while (
                                    t_idx < len(disjoint_intervals)
                                    and disjoint_intervals[t_idx][2] <= r_start
                                ):
                                    t_idx += 1
                                scan_idx = t_idx
                                while (
                                    scan_idx < len(disjoint_intervals)
                                    and disjoint_intervals[scan_idx][1] < r_end
                                ):
                                    tensor_name, t_start, t_end = disjoint_intervals[
                                        scan_idx
                                    ]
                                    inter_start = max(r_start, t_start)
                                    inter_end = min(r_end, t_end)
                                    overlap_hit = self.coverage_repository.find_any_overlap_in_interval(
                                        artifact_id=view.artifact_id,
                                        view_id=view.view_id,
                                        start=inter_start,
                                        end=inter_end,
                                        cursor=cursor,
                                    )
                                    if overlap_hit is not None:
                                        other_view_id, other_off, other_len = (
                                            overlap_hit
                                        )
                                        raise ValueError(
                                            f"overlap in DISJOINT tensor {tensor_name}: "
                                            f"other_view_id={other_view_id} off={other_off} len={other_len}"
                                        )
                                    scan_idx += 1

                            # Proof digests: required exactly for participating replicated tensors.
                            digest_map: dict[tuple[str, int], bytes] = {}
                            for proof_entry in proof_digests:
                                if proof_entry.view_id != view.view_id:
                                    raise ValueError("proof digest view_id mismatch")
                                if (
                                    proof_entry.proof_schema_version
                                    != proof_schema_version
                                ):
                                    raise ValueError("proof_schema_version mismatch")
                                if len(proof_entry.digest) != 32:
                                    raise ValueError(
                                        "proof digest must be 32 bytes (raw sha256)"
                                    )
                                if proof_entry.tensor_name not in tensor_intervals:
                                    raise ValueError(
                                        "proof digest references unknown tensor"
                                    )
                                key = (
                                    proof_entry.tensor_name,
                                    int(proof_entry.proof_chunk_idx),
                                )
                                if (
                                    key in digest_map
                                    and digest_map[key] != proof_entry.digest
                                ):
                                    raise ValueError("duplicate proof digest conflict")
                                digest_map[key] = proof_entry.digest

                            for tensor_name in participating:
                                tensor_off, tensor_bytes = tensor_intervals[tensor_name]
                                expected_chunks = int(
                                    (int(tensor_bytes) + PROOF_CHUNK_BYTES_V1 - 1)
                                    // PROOF_CHUNK_BYTES_V1
                                )
                                for idx in range(expected_chunks):
                                    if (tensor_name, idx) not in digest_map:
                                        raise ValueError(
                                            "missing proof digest for replicated tensor"
                                        )

                            for tensor_name, idx in digest_map:
                                if tensor_name not in participating:
                                    continue
                                _, tensor_bytes = tensor_intervals[tensor_name]
                                expected_chunks = int(
                                    (int(tensor_bytes) + PROOF_CHUNK_BYTES_V1 - 1)
                                    // PROOF_CHUNK_BYTES_V1
                                )
                                if idx < 0 or idx >= expected_chunks:
                                    raise ValueError("proof_chunk_idx out of range")

                            # Persist proof digests (idempotent; conflicts fail fast).
                            for (tensor_name, idx), digest in digest_map.items():
                                if tensor_name not in participating:
                                    continue
                                if self.proof_repository.upsert_piece_proof_digest(
                                    assembly_id=view.artifact_id,
                                    view_id=view.view_id,
                                    tensor_name=tensor_name,
                                    proof_schema_version=proof_schema_version,
                                    proof_chunk_idx=idx,
                                    digest=digest,
                                    cursor=cursor,
                                ):
                                    piece_proof_inserted += 1
                                if self.proof_repository.upsert_assembly_proof_commitment(
                                    assembly_id=view.artifact_id,
                                    tensor_name=tensor_name,
                                    proof_schema_version=proof_schema_version,
                                    proof_chunk_idx=idx,
                                    digest=digest,
                                    cursor=cursor,
                                ):
                                    assembly_proof_inserted += 1

                    self.coverage_repository.replace_ranges(
                        artifact_id=view.artifact_id,
                        view_id=view.view_id,
                        ranges=normalized_ranges,
                        cursor=cursor,
                    )
                elif proof_digests:
                    raise ValueError(
                        "proof digests provided without canonical coverage ranges"
                    )

            if leaf_writes:
                grouped: Dict[Tuple[str, str], List[Tuple[int, bytes]]] = defaultdict(
                    list
                )
                for leaf_entry in leaf_writes:
                    grouped[(leaf_entry.space_kind, leaf_entry.space_id)].append(
                        (leaf_entry.leaf_idx, leaf_entry.digest)
                    )

                for (space_kind, space_id), entries in grouped.items():
                    if replace_existing_assembly_view:
                        if space_kind == "V":
                            self.leaf_repository.delete_all_for_space(
                                artifact_id=artifact_id,
                                space_kind=space_kind,
                                space_id=space_id,
                                cursor=cursor,
                            )
                        else:
                            self.leaf_repository.delete_indices(
                                artifact_id=artifact_id,
                                space_kind=space_kind,
                                space_id=space_id,
                                leaf_idxs=[leaf_idx for leaf_idx, _ in entries],
                                cursor=cursor,
                            )
                    leaf_inserted += self.leaf_repository.upsert_many(
                        artifact_id=artifact_id,
                        space_kind=space_kind,
                        space_id=space_id,
                        entries=entries,
                        cursor=cursor,
                    )

        metrics.inc_digest_entries_written(grid="leaves", count=leaf_inserted)
        metrics.inc_digest_entries_written(
            grid="piece_proof_digests", count=piece_proof_inserted
        )
        metrics.inc_digest_entries_written(
            grid="assembly_proof_commitments", count=assembly_proof_inserted
        )

        if view is not None:
            canonical_total = view.canonical_size_bytes
            canonical_covered = view.canonical_bytes_covered
            backlog = None
            if (
                canonical_total is not None
                and canonical_covered is not None
                and canonical_total > 0
            ):
                capped_covered = min(canonical_total, canonical_covered)
                backlog = max(0, canonical_total - capped_covered)
                metrics.set_view_partial_backlog(
                    view.artifact_id, view.view_id, backlog
                )
            elif canonical_total is not None and canonical_total == 0:
                metrics.set_view_partial_backlog(view.artifact_id, view.view_id, 0)
            result = "partial" if backlog and backlog > 0 else "complete"
            metrics.inc_view_registration(result)
            if backlog is None:
                metrics.set_view_partial_backlog(view.artifact_id, view.view_id, 0)

    def record_view_registration(
        self,
        *,
        artifact_id: str,
        view_id: str,
        view_spec_json: str,
        view_size: int,
        view_data_hash: Optional[str],
        verified_at: Optional[datetime],
        canonical_size_bytes: Optional[int] = None,
        canonical_bytes_covered: Optional[int] = None,
        canonical_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        leaf_writes: Optional[Sequence[LeafWritePayload]] = None,
        proof_digests: Optional[Sequence[PieceProofDigestPayload]] = None,
        tensor_intervals: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Convenience wrapper to persist view metadata after registration."""
        view = ViewUpsertPayload(
            artifact_id=artifact_id,
            view_id=view_id,
            view_spec_json=view_spec_json,
            view_size=view_size,
            view_data_hash=view_data_hash,
            verified_at=verified_at,
            canonical_size_bytes=canonical_size_bytes,
            canonical_bytes_covered=canonical_bytes_covered,
            canonical_ranges=canonical_ranges,
        )
        self.update_view_state(
            view=view,
            leaf_writes=leaf_writes or (),
            proof_digests=proof_digests or (),
            tensor_intervals=tensor_intervals,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_view(
        self, *, artifact_id: str, view_id: str
    ) -> Optional[dict[str, object]]:
        """Fetch view metadata."""
        return self.view_repository.get(artifact_id=artifact_id, view_id=view_id)

    def get_leaves(
        self,
        *,
        artifact_id: str,
        space_kind: str,
        space_id: str,
        leaf_idxs: Optional[Sequence[int]] = None,
    ) -> Iterable[Tuple[int, bytes]]:
        """Fetch leaves for downstream proto mapping."""
        rows = self.leaf_repository.fetch(
            artifact_id=artifact_id,
            space_kind=space_kind,
            space_id=space_id,
            leaf_idxs=leaf_idxs,
        )
        return [(row.leaf_idx, row.digest) for row in rows]
