#  Copyright (c) 2025-2026, TensorCast Team.

"""Service helpers for variant metadata and leaf digests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tensorcast.global_store import metrics
from tensorcast.global_store.repositories.leaf_repository import LeafRepository
from tensorcast.global_store.repositories.variant_coverage_repository import (
    VariantCoverageRepository,
)
from tensorcast.global_store.repositories.variant_repository import VariantRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class VariantUpsertPayload:
    """Payload for variant UPSERT requests."""

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


class ViewStateService:
    """Business logic bridging variant metadata and leaf digests."""

    def __init__(
        self,
        variant_repository: VariantRepository,
        leaf_repository: LeafRepository,
        coverage_repository: VariantCoverageRepository,
    ) -> None:
        self.variant_repository = variant_repository
        self.leaf_repository = leaf_repository
        self.coverage_repository = coverage_repository

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update_view_state(
        self,
        *,
        variant: Optional[VariantUpsertPayload],
        leaf_writes: Sequence[LeafWritePayload],
    ) -> None:
        """Apply variant metadata and leaf digests atomically."""
        if variant is None and not leaf_writes:
            return

        logger.debug(
            "Updating view state: variant=%s leaf_count=%d",
            variant.view_id if variant else "none",
            len(leaf_writes),
        )

        artifact_id = variant.artifact_id if variant is not None else None
        if artifact_id is None and leaf_writes:
            artifact_id = leaf_writes[0].artifact_id

        if artifact_id is None:
            raise ValueError("artifact_id must be provided for variant or leaf writes")

        for entry in leaf_writes:
            if entry.artifact_id != artifact_id:
                raise ValueError("artifact_id mismatch within leaf writes batch")

        with self.variant_repository.transaction() as cursor:
            if variant is not None:
                existing = self.variant_repository.get(
                    artifact_id=variant.artifact_id,
                    view_id=variant.view_id,
                    cursor=cursor,
                )
                if (
                    existing
                    and existing.get("view_data_hash")
                    and variant.view_data_hash
                    and existing.get("view_data_hash") != variant.view_data_hash
                ):
                    raise ValueError(
                        "view_data_hash conflict for existing variant registration"
                    )

                self.variant_repository.upsert(
                    artifact_id=variant.artifact_id,
                    view_id=variant.view_id,
                    view_spec_json=variant.view_spec_json,
                    view_size=variant.view_size,
                    view_data_hash=variant.view_data_hash,
                    verified_at=variant.verified_at,
                    canonical_size_bytes=variant.canonical_size_bytes,
                    canonical_bytes_covered=variant.canonical_bytes_covered,
                    cursor=cursor,
                )

                ranges = (
                    list(variant.canonical_ranges or [])
                    if variant.canonical_ranges is not None
                    else []
                )
                if ranges:
                    existing_ranges = self.coverage_repository.get_ranges(
                        artifact_id=variant.artifact_id,
                        view_id=variant.view_id,
                        cursor=cursor,
                    )
                    if existing_ranges and existing_ranges != ranges:
                        raise ValueError("canonical range mismatch for variant")

                total = variant.canonical_size_bytes or 0
                covered = variant.canonical_bytes_covered or 0
                is_partial = total > 0 and covered < total
                if is_partial and not ranges:
                    raise ValueError("coverage metadata missing for partial variant")

                if is_partial and ranges:
                    overlaps = self.coverage_repository.find_overlaps(
                        artifact_id=variant.artifact_id,
                        view_id=variant.view_id,
                        ranges=ranges,
                        cursor=cursor,
                    )
                    if overlaps:
                        raise ValueError("overlapping canonical coverage across views")

                if ranges:
                    self.coverage_repository.replace_ranges(
                        artifact_id=variant.artifact_id,
                        view_id=variant.view_id,
                        ranges=ranges,
                        cursor=cursor,
                    )

            if leaf_writes:
                grouped: Dict[Tuple[str, str], List[Tuple[int, bytes]]] = defaultdict(
                    list
                )
                for entry in leaf_writes:
                    grouped[(entry.space_kind, entry.space_id)].append(
                        (entry.leaf_idx, entry.digest)
                    )

                for (space_kind, space_id), entries in grouped.items():
                    self.leaf_repository.upsert_many(
                        artifact_id=artifact_id,
                        space_kind=space_kind,
                        space_id=space_id,
                        entries=entries,
                        cursor=cursor,
                    )

        if variant is not None:
            canonical_total = variant.canonical_size_bytes
            canonical_covered = variant.canonical_bytes_covered
            backlog = None
            if (
                canonical_total is not None
                and canonical_covered is not None
                and canonical_total > 0
            ):
                capped_covered = min(canonical_total, canonical_covered)
                backlog = max(0, canonical_total - capped_covered)
                metrics.set_view_partial_backlog(
                    variant.artifact_id, variant.view_id, backlog
                )
            elif canonical_total is not None and canonical_total == 0:
                metrics.set_view_partial_backlog(
                    variant.artifact_id, variant.view_id, 0
                )
            result = "partial" if backlog and backlog > 0 else "complete"
            metrics.inc_view_registration(result)
            if backlog is None:
                metrics.set_view_partial_backlog(
                    variant.artifact_id, variant.view_id, 0
                )

    def record_variant_registration(
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
    ) -> None:
        """Convenience wrapper to persist variant metadata after registration."""
        variant = VariantUpsertPayload(
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
            variant=variant,
            leaf_writes=leaf_writes or (),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_variant(
        self, *, artifact_id: str, view_id: str
    ) -> Optional[dict[str, object]]:
        """Fetch variant metadata."""
        return self.variant_repository.get(artifact_id=artifact_id, view_id=view_id)

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
