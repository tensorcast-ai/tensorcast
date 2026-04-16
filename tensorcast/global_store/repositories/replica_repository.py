#  Copyright (c) 2025-2026, TensorCast Team.

"""Repository for artifact replica data access."""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
from uuid import UUID

from tensorcast.global_store.exceptions import NotFoundError
from tensorcast.global_store.metrics import inc_transport_filter
from tensorcast.global_store.models import (
    ByteSpaceRef,
    ExportState,
    MemoryType,
    Replica,
)
from tensorcast.global_store.repositories.base import (
    BaseRepository,
    is_transient_tx_conflict,
)
from tensorcast.logger import init_logger

logger = init_logger(__name__)


def _to_uuid(value: str | UUID) -> UUID:
    """Convert str or UUID to UUID.

    This helper is now branch-free and relies on UUID(str(value)) which
    is safe for both input variants.  See 250720-isinstance-removal.md §11.
    """
    return UUID(str(value))


@dataclass(frozen=True)
class TransportReplicaSnapshot:
    replica_id: str
    node_id: str
    node_address: str
    node_port: int
    memory_type: str
    device_id: int
    is_available: bool
    current_requests: int
    max_concurrency: int
    worker_id: str
    worker_present: bool
    worker_accepting: bool
    heartbeat_age_sec: float
    heartbeat_fresh: bool
    export_state: str
    transport_valid: bool

    def format_for_log(self) -> str:
        worker_label = self.worker_id if self.worker_id else "missing"
        heartbeat_age = f"{self.heartbeat_age_sec:.1f}s"
        if self.heartbeat_age_sec < 0:
            heartbeat_age = "n/a"
        return (
            "replica_id="
            f"{self.replica_id} "
            f"node={self.node_address}:{self.node_port} "
            f"mem={self.memory_type}/{self.device_id} "
            f"available={self.is_available} "
            f"load={self.current_requests}/{self.max_concurrency} "
            f"export_state={self.export_state} "
            f"transport_valid={self.transport_valid} "
            f"worker={worker_label} "
            f"accepting={self.worker_accepting} "
            f"hb_age={heartbeat_age} "
            f"hb_fresh={self.heartbeat_fresh}"
        )


@dataclass(frozen=True)
class TransportEligibilitySnapshot:
    artifact_id: str
    total_replicas: int
    exportable_replicas: int
    eligible_replicas: int
    available_replicas: int
    capacity_replicas: int
    over_capacity_replicas: int
    worker_present_replicas: int
    worker_missing_replicas: int
    accepting_workers: int
    fresh_heartbeat_replicas: int
    stale_heartbeat_replicas: int
    sample_replicas: list[TransportReplicaSnapshot]

    def format_for_log(self) -> str:
        samples = ", ".join(sample.format_for_log() for sample in self.sample_replicas)
        if not samples:
            samples = "none"
        return (
            f"replicas_total={self.total_replicas} "
            f"exportable={self.exportable_replicas} "
            f"eligible={self.eligible_replicas} "
            f"available={self.available_replicas} "
            f"capacity={self.capacity_replicas} "
            f"over_capacity={self.over_capacity_replicas} "
            f"workers_present={self.worker_present_replicas} "
            f"workers_missing={self.worker_missing_replicas} "
            f"accepting_workers={self.accepting_workers} "
            f"fresh_heartbeats={self.fresh_heartbeat_replicas} "
            f"stale_heartbeats={self.stale_heartbeat_replicas} "
            f"samples=[{samples}]"
        )


@dataclass(frozen=True)
class TransportCandidate:
    replica: Replica
    worker_present: bool
    worker_accepting: bool
    heartbeat_age_sec: float
    heartbeat_fresh: bool
    last_assigned_at: object | None


@dataclass(frozen=True)
class SourceBalanceWeights:
    replica_load: float = 1.0
    worker_load: float = 1.0
    recent_assignment_penalty: float = 1.0
    diffusion_bonus: float = 1.0


@dataclass(frozen=True)
class GroupSourceSpreadPolicy:
    spread_weight: float = 2.0
    soft_cap_ratio: float = 1.3
    min_candidates_for_enforce: int = 3


@dataclass(frozen=True)
class TransportSelectionResult:
    replica: Replica | None
    exportable_replicas: int


_TRANSPORT_FILTER_WORKER_MISSING = "worker_missing"
_TRANSPORT_FILTER_WORKER_NOT_ACCEPTING = "worker_not_accepting"
_TRANSPORT_FILTER_HEARTBEAT_STALE = "heartbeat_stale"
_TRANSPORT_FILTER_REPLICA_UNAVAILABLE = "replica_unavailable"
_TRANSPORT_FILTER_OVER_CAPACITY = "over_capacity"
_TRANSPORT_FILTER_EXPORT_STATE = "export_state"
_TRANSPORT_FILTER_TRANSPORT_MISSING = "transport_missing"
_TRANSPORT_FILTER_TRANSPORT_SIZE_MISMATCH = "transport_size_mismatch"
_TRANSPORT_FILTER_TRANSPORT_SIZE_ZERO = "transport_size_zero"
_TRANSPORT_FILTER_TRANSPORT_SIZE_SUM = "transport_size_sum"


class ReplicaRepository(BaseRepository):
    """Repository for managing artifact replicas in the database."""

    # Canonical column list for replica reads. Always keep in sync with schema.
    # We expose a single source of truth for SELECT projections to guarantee
    # column presence and ordering, while mapping by name in _row_to_model.
    _REPLICA_PROJECTION = (
        "mr.replica_id, mr.artifact_id, mr.view_id, mr.disk_path, mr.node_id, mr.node_address, mr.node_port, "
        "mr.memory_size, mr.memory_type, mr.device_id, mr.max_concurrency, "
        "COALESCE(rc.current_requests, 0) AS current_requests, "
        "rc.last_assigned_at AS last_assigned_at, "
        "mr.is_available, mr.remote_memory_keys, mr.buffer_sizes, mr.export_state, mr.export_generation, "
        "mr.verification_json, mr.worker_id, mr.created_at, mr.updated_at, mr.expires_at"
    )

    @staticmethod
    def _replica_select_sql(join_type: str = "LEFT JOIN") -> str:
        """Build the canonical SELECT for replicas with configurable join type.

        Args:
            join_type: SQL join type between artifact_replicas and replica_counters.
                       Common values: "LEFT JOIN" (default) or "JOIN".

        Returns:
            A SELECT ... FROM ... SQL string without trailing WHERE/ORDER clauses.
        """
        return (
            "SELECT "
            + ReplicaRepository._REPLICA_PROJECTION
            + " FROM artifact_replicas mr "
            + f"{join_type} replica_counters rc ON rc.replica_id = mr.replica_id"
        )

    @staticmethod
    def _evaluate_transport_metadata(
        replica: Replica,
    ) -> tuple[bool, str]:
        keys = list(replica.remote_memory_keys)
        sizes = list(replica.buffer_sizes)
        if replica.export_state is not ExportState.EXPORTABLE:
            return False, _TRANSPORT_FILTER_EXPORT_STATE
        if not keys:
            return False, _TRANSPORT_FILTER_TRANSPORT_MISSING
        if len(keys) != len(sizes):
            return False, _TRANSPORT_FILTER_TRANSPORT_SIZE_MISMATCH
        total = 0
        for size in sizes:
            if int(size) <= 0:
                return False, _TRANSPORT_FILTER_TRANSPORT_SIZE_ZERO
            total += int(size)
        if replica.memory_size > 0 and total != replica.memory_size:
            return False, _TRANSPORT_FILTER_TRANSPORT_SIZE_SUM
        return True, ""

    @staticmethod
    def _evaluate_transport_candidate(
        candidate: TransportCandidate,
    ) -> tuple[bool, str]:
        if not candidate.worker_present:
            return False, _TRANSPORT_FILTER_WORKER_MISSING
        if not candidate.worker_accepting:
            return False, _TRANSPORT_FILTER_WORKER_NOT_ACCEPTING
        if not candidate.heartbeat_fresh:
            return False, _TRANSPORT_FILTER_HEARTBEAT_STALE
        if not candidate.replica.is_available:
            return False, _TRANSPORT_FILTER_REPLICA_UNAVAILABLE
        if not candidate.replica.has_capacity:
            return False, _TRANSPORT_FILTER_OVER_CAPACITY
        ok, reason = ReplicaRepository._evaluate_transport_metadata(candidate.replica)
        if not ok:
            return False, reason
        return True, ""

    def _build_transport_candidate(
        self,
        row: tuple,
        columns: Sequence[str],
        *,
        now_ts: float,
        heartbeat_timeout_seconds: float,
    ) -> TransportCandidate:
        replica = self._row_to_model(row, columns)
        column_index = {column: idx for idx, column in enumerate(columns)}

        def get(column: str, *, default=None):
            idx = column_index.get(column)
            if idx is None:
                return default
            value = row[idx]
            if value is None:
                return default
            return value

        worker_id = get("gs_worker_id", default="")
        inactive_at = get("worker_inactive_at")
        worker_present = bool(worker_id) and inactive_at is None
        worker_accepting = (
            bool(get("worker_accepting", default=False)) if worker_present else False
        )
        last_heartbeat = get("worker_last_heartbeat")
        last_assigned_at = get("last_assigned_at")

        heartbeat_age_sec = -1.0
        heartbeat_fresh = False
        if worker_present and last_heartbeat is not None:
            last_heartbeat_ts = last_heartbeat.timestamp()
            heartbeat_age_sec = max(0.0, now_ts - last_heartbeat_ts)
            heartbeat_fresh = heartbeat_age_sec <= heartbeat_timeout_seconds

        return TransportCandidate(
            replica=replica,
            worker_present=worker_present,
            worker_accepting=worker_accepting,
            heartbeat_age_sec=heartbeat_age_sec,
            heartbeat_fresh=heartbeat_fresh,
            last_assigned_at=last_assigned_at,
        )

    def has_any_replica(self, artifact_id: str, view_id: str | None = None) -> bool:
        """Return True when at least one replica exists for *artifact_id*."""

        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                "SELECT 1 FROM artifact_replicas WHERE artifact_id = ? AND COALESCE(view_id, '') = COALESCE(?, '') LIMIT 1",
                [artifact_id, view_id or ""],
            ).fetchone()
            return row is not None
        finally:
            cursor.close()

    def find_by_id(
        self, replica_id: UUID, artifact_id: str, cursor=None
    ) -> Replica | None:
        """Find a replica by ID and content-addressed artifact_id."""
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            sql = (
                self._replica_select_sql("LEFT JOIN")
                + " WHERE mr.replica_id = ? AND mr.artifact_id = ?"
            )
            query = cursor.execute(sql, [str(replica_id), artifact_id])

            row = query.fetchone()
            if row:
                assert query.description is not None
                columns = [desc[0] for desc in query.description]
                return self._row_to_model(row, columns)
            return None
        finally:
            if owns_cursor:
                cursor.close()

    def find_by_replica_id(self, replica_id: UUID) -> Replica | None:
        """Find a replica by ID (artifact_id not required)."""
        cursor = self.get_cursor()
        try:
            sql = self._replica_select_sql("LEFT JOIN") + " WHERE mr.replica_id = ?"
            query = cursor.execute(sql, [str(replica_id)])
            row = query.fetchone()
            if row:
                assert query.description is not None
                columns = [desc[0] for desc in query.description]
                return self._row_to_model(row, columns)
            return None
        finally:
            cursor.close()

    def get_current_requests(self, replica_id: UUID) -> int | None:
        """Return current_requests for a replica, or None if not found."""
        cursor = self.get_cursor()
        try:
            row = cursor.execute(
                """
                SELECT COALESCE(rc.current_requests, 0) AS current_requests
                FROM artifact_replicas mr
                LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id
                WHERE mr.replica_id = ?
                """,
                [str(replica_id)],
            ).fetchone()
            if row is None:
                return None
            return int(row[0] or 0)
        finally:
            cursor.close()

    def find_existing(
        self,
        artifact_id: str,
        view_id: str | None,
        node_id: str,
        node_address: str,
        node_port: int,
        memory_type: MemoryType,
        device_id: int,
    ) -> Replica | None:
        """Find existing replica with same identifying information."""
        cursor = self.get_cursor()
        try:
            sql = (
                self._replica_select_sql("LEFT JOIN")
                + " WHERE mr.artifact_id = ?"
                + " AND COALESCE(mr.view_id, '') = COALESCE(?, '')"
                + " AND mr.node_id = ?"
                + " AND mr.node_address = ?"
                + " AND mr.node_port = ?"
                + " AND mr.memory_type = ?"
                + " AND mr.device_id = ?"
            )
            query = cursor.execute(
                sql,
                [
                    artifact_id,
                    view_id,
                    node_id,
                    node_address,
                    node_port,
                    memory_type.value,
                    device_id,
                ],
            )

            row = query.fetchone()
            if row:
                assert query.description is not None
                columns = [desc[0] for desc in query.description]
                return self._row_to_model(row, columns)
            return None
        finally:
            cursor.close()

    def find_available_for_transport(
        self,
        artifact_id: str,
        heartbeat_timeout_seconds: float,
        view_id: str | None = None,
        scheduler_mode: str = "LEGACY",
        source_balance_weights: SourceBalanceWeights | None = None,
        group_source_counts: dict[str, int] | None = None,
        group_source_policy: GroupSourceSpreadPolicy | None = None,
        cursor=None,
    ) -> TransportSelectionResult:
        """
        Find best available replica for transport with load balancing.

        Atomically increments current_requests counter.
        """
        owns_cursor = cursor is None
        if owns_cursor:
            cursor = self.get_cursor()
        try:
            query = (
                "SELECT "
                + self._REPLICA_PROJECTION
                + ", COALESCE(w.worker_id, '') AS gs_worker_id, "
                + "wl.accepting_new_requests AS worker_accepting, "
                + "wl.last_heartbeat AS worker_last_heartbeat, "
                + "w.inactive_at AS worker_inactive_at "
                + "FROM artifact_replicas mr "
                + "LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id "
                + "LEFT JOIN workers w ON mr.worker_id = w.worker_id "
                + "LEFT JOIN worker_liveness wl ON wl.worker_id = w.worker_id "
                + "WHERE mr.artifact_id = ? "
                + "AND COALESCE(mr.view_id, '') = COALESCE(?, '') "
                + "ORDER BY "
                + "CASE "
                + "WHEN mr.memory_type = 'GPU' THEN 0 "
                + "WHEN mr.memory_type = 'RAM' THEN 1 "
                + "WHEN mr.memory_type = 'DISK' THEN 2 "
                + "ELSE 3 "
                + "END, "
                + "(COALESCE(rc.current_requests, 0) * 1.0 / GREATEST(mr.max_concurrency, 1)), "
                + "COALESCE(rc.current_requests, 0) ASC, "
                + "mr.max_concurrency DESC, "
                + "COALESCE(rc.last_assigned_at, TIMESTAMP '1970-01-01') ASC, "
                + "mr.updated_at ASC"
            )
            result = cursor.execute(query, [artifact_id, view_id or ""])
            rows = result.fetchall()
            if not rows:
                return TransportSelectionResult(replica=None, exportable_replicas=0)

            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            now_ts = time.time()
            exportable_replicas = 0
            candidates: list[TransportCandidate] = []

            worker_request_totals: dict[str, int] = {}
            worker_capacity_totals: dict[str, int] = {}

            for row in rows:
                candidate = self._build_transport_candidate(
                    row,
                    columns,
                    now_ts=now_ts,
                    heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                )
                candidates.append(candidate)

                worker_id = candidate.replica.worker_id or ""
                if candidate.worker_present and worker_id:
                    worker_request_totals[worker_id] = worker_request_totals.get(
                        worker_id, 0
                    ) + int(candidate.replica.current_requests)
                    worker_capacity_totals[worker_id] = worker_capacity_totals.get(
                        worker_id, 0
                    ) + max(1, int(candidate.replica.max_concurrency))

            worker_loads: dict[str, float] = {}
            for worker_id, total_requests in worker_request_totals.items():
                total_capacity = max(1, worker_capacity_totals.get(worker_id, 0))
                worker_loads[worker_id] = float(total_requests) / float(total_capacity)

            eligible_candidates: list[TransportCandidate] = []
            for candidate in candidates:
                transport_ok, _ = self._evaluate_transport_metadata(candidate.replica)
                if transport_ok:
                    exportable_replicas += 1

                eligible, reason = self._evaluate_transport_candidate(candidate)
                if not eligible:
                    inc_transport_filter(artifact_id, reason)
                    continue
                eligible_candidates.append(candidate)

            normalized_group_source_counts = self._normalize_group_source_counts(
                group_source_counts
            )
            spread_policy = (
                group_source_policy
                if group_source_policy is not None
                else GroupSourceSpreadPolicy()
            )
            group_candidate_count = len(eligible_candidates)
            group_penalty_scale = self._group_source_penalty_scale(
                candidate_count=group_candidate_count,
                min_candidates_for_enforce=spread_policy.min_candidates_for_enforce,
            )

            ranked_candidates = eligible_candidates
            mode = str(scheduler_mode or "LEGACY").upper()
            if mode in ("SOURCE_BALANCE", "GROUP_DISPATCH"):
                weights = (
                    source_balance_weights
                    if source_balance_weights is not None
                    else SourceBalanceWeights()
                )
                ranked_candidates = sorted(
                    eligible_candidates,
                    key=lambda candidate: self._source_balance_sort_key(
                        candidate,
                        worker_loads,
                        now_ts,
                        weights,
                        group_source_counts=normalized_group_source_counts,
                        group_candidate_count=group_candidate_count,
                        group_source_spread_weight=spread_policy.spread_weight,
                        group_source_penalty_scale=group_penalty_scale,
                    ),
                )

            enforce_group_soft_cap = (
                mode == "GROUP_DISPATCH"
                and group_candidate_count
                >= max(1, int(spread_policy.min_candidates_for_enforce))
            )
            has_soft_cap_alternative = False
            if enforce_group_soft_cap:
                has_soft_cap_alternative = any(
                    not self._is_group_source_over_soft_cap(
                        replica_id=str(candidate.replica.replica_id),
                        group_source_counts=normalized_group_source_counts,
                        candidate_count=group_candidate_count,
                        soft_cap_ratio=spread_policy.soft_cap_ratio,
                    )
                    for candidate in ranked_candidates
                )

            for candidate in ranked_candidates:
                replica_id = str(candidate.replica.replica_id)
                if (
                    enforce_group_soft_cap
                    and has_soft_cap_alternative
                    and self._is_group_source_over_soft_cap(
                        replica_id=replica_id,
                        group_source_counts=normalized_group_source_counts,
                        candidate_count=group_candidate_count,
                        soft_cap_ratio=spread_policy.soft_cap_ratio,
                    )
                ):
                    continue
                claim = cursor.execute(
                    """
                    UPDATE replica_counters
                    SET current_requests = current_requests + 1,
                        last_assigned_at = now()
                    WHERE replica_id = ?
                      AND current_requests < (
                        SELECT max_concurrency FROM artifact_replicas WHERE replica_id = ?
                      )
                    RETURNING current_requests
                    """,
                    [replica_id, replica_id],
                ).fetchone()
                if not claim:
                    continue

                sql = self._replica_select_sql("JOIN") + " WHERE mr.replica_id = ?"
                full_result = cursor.execute(sql, [replica_id])
                full_row = full_result.fetchone()
                if full_row:
                    assert full_result.description is not None
                    full_columns = [desc[0] for desc in full_result.description]
                    replica = self._row_to_model(full_row, full_columns)
                    return TransportSelectionResult(
                        replica=replica,
                        exportable_replicas=exportable_replicas,
                    )
                break

            return TransportSelectionResult(
                replica=None, exportable_replicas=exportable_replicas
            )
        finally:
            if owns_cursor:
                cursor.close()

    @staticmethod
    def _memory_priority(memory_type: MemoryType) -> int:
        if memory_type is MemoryType.GPU:
            return 0
        if memory_type is MemoryType.RAM:
            return 1
        if memory_type is MemoryType.DISK:
            return 2
        return 3

    @staticmethod
    def _last_assigned_timestamp_seconds(last_assigned_at: object | None) -> float:
        if last_assigned_at is None:
            return 0.0
        timestamp_fn = getattr(last_assigned_at, "timestamp", None)
        if callable(timestamp_fn):
            try:
                return float(cast(float, timestamp_fn()))
            except Exception:  # noqa: BLE001
                return 0.0
        return 0.0

    @classmethod
    def _recent_assignment_penalty(
        cls, last_assigned_at: object | None, now_ts: float
    ) -> float:
        assigned_ts = cls._last_assigned_timestamp_seconds(last_assigned_at)
        if assigned_ts <= 0.0:
            return 0.0
        age_sec = max(0.0, now_ts - assigned_ts)
        return 1.0 / (1.0 + age_sec)

    @classmethod
    def _diffusion_bonus(cls, candidate: TransportCandidate, now_ts: float) -> float:
        replica = candidate.replica
        if replica.current_requests != 0:
            return 0.0
        if replica.export_state is not ExportState.EXPORTABLE:
            return 0.0
        assigned_ts = cls._last_assigned_timestamp_seconds(candidate.last_assigned_at)
        if assigned_ts <= 0.0:
            return 1.0
        age_sec = max(0.0, now_ts - assigned_ts)
        return min(1.0, age_sec / 30.0)

    @staticmethod
    def _normalize_group_source_counts(
        group_source_counts: dict[str, int] | None,
    ) -> dict[str, int]:
        if not group_source_counts:
            return {}
        normalized: dict[str, int] = {}
        for replica_id, count in group_source_counts.items():
            normalized[str(replica_id)] = max(0, int(count))
        return normalized

    @staticmethod
    def _group_source_penalty_scale(
        *, candidate_count: int, min_candidates_for_enforce: int
    ) -> float:
        bounded_min_candidates = max(1, int(min_candidates_for_enforce))
        if candidate_count <= 1:
            return 0.0
        if candidate_count >= bounded_min_candidates:
            return 1.0
        if bounded_min_candidates <= 1:
            return 1.0
        return float(candidate_count - 1) / float(bounded_min_candidates - 1)

    @staticmethod
    def _group_source_assignment_total(group_source_counts: dict[str, int]) -> int:
        return sum(max(0, int(count)) for count in group_source_counts.values())

    @classmethod
    def _group_hotspot_penalty(
        cls,
        *,
        replica_id: str,
        group_source_counts: dict[str, int],
        candidate_count: int,
    ) -> float:
        if candidate_count <= 1:
            return 0.0
        total_assignments = cls._group_source_assignment_total(group_source_counts)
        if total_assignments <= 0:
            return 0.0
        replica_assignments = max(0, int(group_source_counts.get(replica_id, 0)))
        actual_share = float(replica_assignments) / float(total_assignments)
        fair_share = 1.0 / float(candidate_count)
        if fair_share <= 0.0 or actual_share <= fair_share:
            return 0.0
        return (actual_share / fair_share) - 1.0

    @classmethod
    def _is_group_source_over_soft_cap(
        cls,
        *,
        replica_id: str,
        group_source_counts: dict[str, int],
        candidate_count: int,
        soft_cap_ratio: float,
    ) -> bool:
        if candidate_count <= 1:
            return False
        effective_soft_cap_ratio = max(0.0, float(soft_cap_ratio))
        if effective_soft_cap_ratio <= 0.0:
            return False
        total_assignments = cls._group_source_assignment_total(group_source_counts)
        if total_assignments <= 0:
            return False
        projected_assignments = float(
            max(0, int(group_source_counts.get(replica_id, 0))) + 1
        )
        projected_total = float(total_assignments + 1)
        projected_fair_assignments = projected_total / float(candidate_count)
        soft_cap = max(1.0, projected_fair_assignments * effective_soft_cap_ratio)
        return projected_assignments > soft_cap

    @classmethod
    def _source_balance_sort_key(
        cls,
        candidate: TransportCandidate,
        worker_loads: dict[str, float],
        now_ts: float,
        weights: SourceBalanceWeights,
        *,
        group_source_counts: dict[str, int],
        group_candidate_count: int,
        group_source_spread_weight: float,
        group_source_penalty_scale: float,
    ) -> tuple[int, float, int, int, float, str]:
        replica = candidate.replica
        replica_load = float(replica.current_requests) / float(
            max(1, replica.max_concurrency)
        )
        worker_id = replica.worker_id or ""
        worker_load = worker_loads.get(worker_id, 1.0)
        recent_penalty = cls._recent_assignment_penalty(
            candidate.last_assigned_at, now_ts
        )
        diffusion_bonus = cls._diffusion_bonus(candidate, now_ts)
        group_hotspot_penalty = cls._group_hotspot_penalty(
            replica_id=str(replica.replica_id),
            group_source_counts=group_source_counts,
            candidate_count=group_candidate_count,
        )
        source_score = (
            float(weights.replica_load) * replica_load
            + float(weights.worker_load) * worker_load
            + float(weights.recent_assignment_penalty) * recent_penalty
            - float(weights.diffusion_bonus) * diffusion_bonus
            + float(group_source_spread_weight)
            * float(group_source_penalty_scale)
            * group_hotspot_penalty
        )
        return (
            cls._memory_priority(replica.memory_type),
            source_score,
            int(replica.current_requests),
            -int(replica.max_concurrency),
            cls._last_assigned_timestamp_seconds(candidate.last_assigned_at),
            str(replica.replica_id),
        )

    def get_transport_eligibility_snapshot(
        self,
        artifact_id: str,
        view_id: str | None,
        heartbeat_timeout_seconds: float,
        sample_limit: int = 10,
    ) -> TransportEligibilitySnapshot:
        """Return a snapshot of replica eligibility for transport debugging."""
        cursor = self.get_cursor()
        try:
            query = (
                "SELECT "
                + self._REPLICA_PROJECTION
                + ", COALESCE(w.worker_id, '') AS gs_worker_id, "
                + "wl.accepting_new_requests AS worker_accepting, "
                + "wl.last_heartbeat AS worker_last_heartbeat, "
                + "w.inactive_at AS worker_inactive_at "
                + "FROM artifact_replicas mr "
                + "LEFT JOIN replica_counters rc ON rc.replica_id = mr.replica_id "
                + "LEFT JOIN workers w ON mr.worker_id = w.worker_id "
                + "LEFT JOIN worker_liveness wl ON wl.worker_id = w.worker_id "
                + "WHERE mr.artifact_id = ? "
                + "AND COALESCE(mr.view_id, '') = COALESCE(?, '') "
                + "ORDER BY mr.updated_at ASC"
            )
            result = cursor.execute(query, [artifact_id, view_id or ""])
            rows = result.fetchall()
            if not rows:
                return TransportEligibilitySnapshot(
                    artifact_id=artifact_id,
                    total_replicas=0,
                    exportable_replicas=0,
                    eligible_replicas=0,
                    available_replicas=0,
                    capacity_replicas=0,
                    over_capacity_replicas=0,
                    worker_present_replicas=0,
                    worker_missing_replicas=0,
                    accepting_workers=0,
                    fresh_heartbeat_replicas=0,
                    stale_heartbeat_replicas=0,
                    sample_replicas=[],
                )

            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            now_ts = time.time()

            total_replicas = 0
            exportable_replicas = 0
            eligible_replicas = 0
            available_replicas = 0
            capacity_replicas = 0
            over_capacity_replicas = 0
            worker_present_replicas = 0
            worker_missing_replicas = 0
            accepting_workers = 0
            fresh_heartbeat_replicas = 0
            stale_heartbeat_replicas = 0
            sample_replicas: list[TransportReplicaSnapshot] = []

            for row in rows:
                total_replicas += 1
                candidate = self._build_transport_candidate(
                    row,
                    columns,
                    now_ts=now_ts,
                    heartbeat_timeout_seconds=heartbeat_timeout_seconds,
                )
                if candidate.worker_present:
                    worker_present_replicas += 1
                    if candidate.worker_accepting:
                        accepting_workers += 1
                else:
                    worker_missing_replicas += 1

                if candidate.heartbeat_fresh:
                    fresh_heartbeat_replicas += 1
                elif candidate.worker_present:
                    stale_heartbeat_replicas += 1

                if candidate.replica.is_available:
                    available_replicas += 1

                if candidate.replica.has_capacity:
                    capacity_replicas += 1
                else:
                    over_capacity_replicas += 1

                transport_ok, _ = self._evaluate_transport_metadata(candidate.replica)
                if transport_ok:
                    exportable_replicas += 1

                eligible, _ = self._evaluate_transport_candidate(candidate)
                if eligible:
                    eligible_replicas += 1

                if sample_limit > 0 and len(sample_replicas) < sample_limit:
                    sample_replicas.append(
                        TransportReplicaSnapshot(
                            replica_id=str(candidate.replica.replica_id),
                            node_id=candidate.replica.node_id,
                            node_address=candidate.replica.node_address,
                            node_port=int(candidate.replica.node_port),
                            memory_type=candidate.replica.memory_type.value,
                            device_id=int(candidate.replica.device_id),
                            is_available=candidate.replica.is_available,
                            current_requests=int(candidate.replica.current_requests),
                            max_concurrency=int(candidate.replica.max_concurrency),
                            worker_id=candidate.replica.worker_id or "",
                            worker_present=candidate.worker_present,
                            worker_accepting=candidate.worker_accepting,
                            heartbeat_age_sec=candidate.heartbeat_age_sec,
                            heartbeat_fresh=candidate.heartbeat_fresh,
                            export_state=candidate.replica.export_state.value,
                            transport_valid=transport_ok,
                        )
                    )

            return TransportEligibilitySnapshot(
                artifact_id=artifact_id,
                total_replicas=total_replicas,
                exportable_replicas=exportable_replicas,
                eligible_replicas=eligible_replicas,
                available_replicas=available_replicas,
                capacity_replicas=capacity_replicas,
                over_capacity_replicas=over_capacity_replicas,
                worker_present_replicas=worker_present_replicas,
                worker_missing_replicas=worker_missing_replicas,
                accepting_workers=accepting_workers,
                fresh_heartbeat_replicas=fresh_heartbeat_replicas,
                stale_heartbeat_replicas=stale_heartbeat_replicas,
                sample_replicas=sample_replicas,
            )
        finally:
            cursor.close()

    def create(self, replica: Replica) -> Replica:
        """Create a new replica."""
        cursor = self.get_cursor()
        try:
            # Insert into artifact_replicas (without current_requests)
            cursor.execute(
                """
                INSERT INTO artifact_replicas (
                    replica_id, artifact_id, view_id, node_id, node_address, node_port,
                    memory_size, memory_type, device_id, max_concurrency,
                    is_available, remote_memory_keys, buffer_sizes, export_state, export_generation,
                    verification_json, worker_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(replica.replica_id),
                    replica.artifact_id,
                    replica.byte_space.id,
                    replica.node_id,
                    replica.node_address,
                    replica.node_port,
                    replica.memory_size,
                    replica.memory_type.value,
                    replica.device_id
                    if replica.device_id is not None
                    else -1,  # Use -1 for NULL device_id
                    replica.max_concurrency,
                    replica.is_available,
                    list(replica.remote_memory_keys),
                    list(replica.buffer_sizes),
                    replica.export_state.value,
                    replica.export_generation,
                    replica.verification_json,
                    replica.worker_id,
                ],
            )

            # Ensure corresponding counter record exists without resetting
            # in-flight transport counters on re-registration.
            cursor.execute(
                """
                INSERT INTO replica_counters (replica_id, current_requests, last_assigned_at)
                VALUES (?, ?, TIMESTAMP '1970-01-01 00:00:00')
                ON CONFLICT (replica_id) DO NOTHING
                """,
                [str(replica.replica_id), replica.current_requests],
            )

            return replica
        finally:
            cursor.close()

    def create_or_update(self, replica: Replica) -> Replica:
        """Create a new replica or update existing one."""
        existing = self.find_existing(
            replica.artifact_id,
            replica.byte_space.id,
            replica.node_id,
            replica.node_address,
            replica.node_port,
            replica.memory_type,
            replica.device_id,
        )

        if existing:
            # Update the existing replica with new information
            replica.replica_id = existing.replica_id
            return self.update(replica)
        else:
            return self.create(replica)

    def create_or_update_atomic(
        self, replica: Replica, cursor, *, preserve_transport: bool = False
    ) -> Replica:
        """
        Atomically create or update a replica within a transaction.

        This method prevents race conditions by using a transaction-safe
        approach to handle concurrent replica registrations.

        Args:
            replica: Replica to create or update
            cursor: Database cursor within an active transaction

        Returns:
            Created or updated Replica
        """
        # First, try to find existing replica within the transaction
        existing_result = cursor.execute(
            """
            SELECT replica_id, memory_size FROM artifact_replicas
            WHERE artifact_id = ? AND COALESCE(view_id, '') = COALESCE(?, '')
              AND node_id = ? AND node_address = ?
              AND node_port = ? AND memory_type = ?
              AND COALESCE(device_id, -1) = COALESCE(?, -1)
            """,
            [
                replica.artifact_id,
                replica.byte_space.id,
                replica.node_id,
                replica.node_address,
                replica.node_port,
                replica.memory_type.value,
                replica.device_id,
            ],
        ).fetchone()

        if existing_result:
            # Update existing replica
            existing_replica_id = existing_result[0]
            existing_memory_size = int(existing_result[1] or 0)
            if preserve_transport:
                cursor.execute(
                    """
                    UPDATE artifact_replicas
                    SET
                        updated_at = CURRENT_TIMESTAMP,
                        is_available = ?,
                        max_concurrency = ?,
                        worker_id = ?,
                        expires_at = ?
                    WHERE replica_id = ?
                    """,
                    [
                        replica.is_available,
                        replica.max_concurrency,
                        replica.worker_id,
                        replica.expires_at,
                        str(existing_replica_id),
                    ],
                )
                if (
                    existing_memory_size > 0
                    and replica.memory_size != existing_memory_size
                ):
                    logger.warning(
                        "RegisterReplica preserving transport metadata for replica_id=%s but memory_size differs (existing=%s new=%s); keeping existing",
                        existing_replica_id,
                        existing_memory_size,
                        replica.memory_size,
                    )
                if existing_memory_size > 0:
                    replica.memory_size = existing_memory_size
            else:
                cursor.execute(
                    """
                    UPDATE artifact_replicas
                    SET
                        updated_at = CURRENT_TIMESTAMP,
                        is_available = ?,
                        max_concurrency = ?,
                        remote_memory_keys = ?,
                        buffer_sizes = ?,
                        export_state = ?,
                        export_generation = ?,
                        verification_json = ?,
                        memory_size = ?,
                        worker_id = ?,
                        expires_at = ?
                    WHERE replica_id = ?
                    """,
                    [
                        replica.is_available,
                        replica.max_concurrency,
                        list(replica.remote_memory_keys),
                        list(replica.buffer_sizes),
                        replica.export_state.value,
                        replica.export_generation,
                        replica.verification_json,
                        replica.memory_size,
                        replica.worker_id,
                        replica.expires_at,
                        str(existing_replica_id),
                    ],
                )
            replica.replica_id = UUID(str(existing_replica_id))
        else:
            # Create new replica
            cursor.execute(
                """
                INSERT INTO artifact_replicas (
                    replica_id, artifact_id, view_id, node_id, node_address, node_port,
                    memory_size, memory_type, device_id, max_concurrency,
                    is_available, remote_memory_keys, buffer_sizes, export_state, export_generation,
                    verification_json, worker_id, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(replica.replica_id),
                    replica.artifact_id,
                    replica.byte_space.id,
                    replica.node_id,
                    replica.node_address,
                    replica.node_port,
                    replica.memory_size,
                    replica.memory_type.value,
                    replica.device_id
                    if replica.device_id is not None
                    else -1,  # Use -1 for NULL device_id
                    replica.max_concurrency,
                    replica.is_available,
                    list(replica.remote_memory_keys),
                    list(replica.buffer_sizes),
                    replica.export_state.value,
                    replica.export_generation,
                    replica.verification_json,
                    replica.worker_id,
                    replica.expires_at,
                ],
            )

        # Ensure corresponding counter record exists without clobbering
        # active request counters or assignment history.
        cursor.execute(
            """
            INSERT INTO replica_counters (replica_id, current_requests, last_assigned_at)
            VALUES (?, ?, TIMESTAMP '1970-01-01 00:00:00')
            ON CONFLICT (replica_id) DO NOTHING
            """,
            [str(replica.replica_id), replica.current_requests],
        )

        return replica

    def update_atomic(self, replica: Replica, cursor) -> Replica:
        """Update an existing replica within an active transaction."""
        result = cursor.execute(
            """
            UPDATE artifact_replicas
            SET
                updated_at = CURRENT_TIMESTAMP,
                node_id = ?,
                node_address = ?,
                node_port = ?,
                is_available = ?,
                max_concurrency = ?,
                remote_memory_keys = ?,
                buffer_sizes = ?,
                export_state = ?,
                export_generation = ?,
                verification_json = ?,
                memory_size = ?,
                worker_id = ?
            WHERE replica_id = ?
            RETURNING replica_id
            """,
            [
                replica.node_id,
                replica.node_address,
                replica.node_port,
                replica.is_available,
                replica.max_concurrency,
                list(replica.remote_memory_keys),
                list(replica.buffer_sizes),
                replica.export_state.value,
                replica.export_generation,
                replica.verification_json,
                replica.memory_size,
                replica.worker_id,
                str(replica.replica_id),
            ],
        )
        if result.fetchone() is None:
            raise NotFoundError(f"Replica {replica.replica_id} not found")
        return replica

    def delete_atomic(self, replica_id: UUID, cursor) -> bool:
        """Delete a replica within an existing transaction."""
        cursor.execute(
            """
            DELETE FROM replica_counters
            WHERE replica_id = ?
            """,
            [str(replica_id)],
        )
        result = cursor.execute(
            """
            DELETE FROM artifact_replicas
            WHERE replica_id = ?
            RETURNING replica_id
            """,
            [str(replica_id)],
        )
        return result.fetchone() is not None

    def update(self, replica: Replica) -> Replica:
        """Update an existing replica."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                """
                UPDATE artifact_replicas
                SET
                    updated_at = CURRENT_TIMESTAMP,
                    node_id = ?,
                    node_address = ?,
                    node_port = ?,
                    is_available = ?,
                    max_concurrency = ?,
                    remote_memory_keys = ?,
                    buffer_sizes = ?,
                    export_state = ?,
                    export_generation = ?,
                    verification_json = ?,
                    memory_size = ?,
                    worker_id = ?
                WHERE replica_id = ?
                RETURNING replica_id
                """,
                [
                    replica.node_id,
                    replica.node_address,
                    replica.node_port,
                    replica.is_available,
                    replica.max_concurrency,
                    list(replica.remote_memory_keys),
                    list(replica.buffer_sizes),
                    replica.export_state.value,
                    replica.export_generation,
                    replica.verification_json,
                    replica.memory_size,
                    replica.worker_id,
                    str(replica.replica_id),
                ],
            )

            if result.fetchone() is None:
                raise NotFoundError(f"Replica {replica.replica_id} not found")

            return replica
        finally:
            cursor.close()

    def update_heartbeat(self, replica_id: UUID, artifact_id: str) -> bool:
        """Update the heartbeat timestamp for a replica."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                """
                UPDATE artifact_replicas
                SET updated_at = CURRENT_TIMESTAMP
                WHERE replica_id = ? AND artifact_id = ?
                RETURNING replica_id
                """,
                [str(replica_id), artifact_id],
            )

            return result.fetchone() is not None
        finally:
            cursor.close()

    def decrement_requests(self, replica_id: UUID) -> tuple[int, int]:
        """
        Decrement the current_requests counter.

        Returns:
            Tuple of (current_requests, max_concurrency)
        """
        max_attempts = 8
        for attempt in range(max_attempts):
            cursor = self.get_cursor()
            try:
                cnt_row = self.decrement_requests_with_cursor(replica_id, cursor)
                if cnt_row is None:
                    return 0, 0

                current_requests = cnt_row
                max_row = cursor.execute(
                    """
                    SELECT max_concurrency FROM artifact_replicas WHERE replica_id = ?
                    """,
                    [str(replica_id)],
                ).fetchone()
                max_conc = int(max_row[0]) if max_row else 0
                return current_requests, max_conc
            except Exception as exc:  # noqa: BLE001
                if not is_transient_tx_conflict(exc) or attempt == max_attempts - 1:
                    raise
                backoff_sec = min(0.2, 0.005 * (2**attempt))
                logger.warning(
                    "Retrying decrement_requests after transient DB conflict replica_id=%s attempt=%s/%s backoff_s=%.3f error=%s",
                    replica_id,
                    attempt + 1,
                    max_attempts,
                    backoff_sec,
                    exc,
                )
                time.sleep(backoff_sec)
            finally:
                cursor.close()

        return 0, 0

    def decrement_requests_with_cursor(self, replica_id: UUID, cursor) -> int | None:
        """Decrement `current_requests` in an existing transaction."""
        cnt_row = cursor.execute(
            """
            UPDATE replica_counters
            SET current_requests = GREATEST(0, current_requests - 1)
            WHERE replica_id = ?
            RETURNING current_requests
            """,
            [str(replica_id)],
        ).fetchone()
        if not cnt_row:
            return None
        return int(cnt_row[0])

    def delete(self, replica_id: UUID, artifact_id: str | None = None) -> bool:
        """Delete a replica."""
        cursor = self.get_cursor()
        try:
            # First delete from replica_counters (due to foreign key constraint)
            cursor.execute(
                """
                DELETE FROM replica_counters
                WHERE replica_id = ?
                """,
                [str(replica_id)],
            )

            # Then delete from artifact_replicas
            if artifact_id:
                result = cursor.execute(
                    """
                    DELETE FROM artifact_replicas
                    WHERE replica_id = ? AND artifact_id = ?
                    RETURNING replica_id
                    """,
                    [str(replica_id), artifact_id],
                )
            else:
                result = cursor.execute(
                    """
                    DELETE FROM artifact_replicas
                    WHERE replica_id = ?
                    RETURNING replica_id
                    """,
                    [str(replica_id)],
                )

            return result.fetchone() is not None
        finally:
            cursor.close()

    def find_by_artifact(
        self, artifact_id: str, view_id: str | None = None
    ) -> list[Replica]:
        """Convenience helper to list replicas for a given content-addressed artifact_id."""
        return self.find_by_filters(artifact_id=artifact_id, view_id=view_id)

    def count_replicas_by_artifact_ids(
        self, artifact_ids: list[str]
    ) -> dict[str, tuple[int, int]]:
        """Return {artifact_id: (replica_count, available_count)} for the requested ids."""
        normalized_ids = [str(artifact_id).strip() for artifact_id in artifact_ids]
        requested_ids = [artifact_id for artifact_id in normalized_ids if artifact_id]
        if not requested_ids:
            return {}

        unique_ids = list(dict.fromkeys(requested_ids))
        placeholders = ", ".join("?" for _ in unique_ids)
        cursor = self.get_cursor()
        try:
            rows = cursor.execute(
                f"""
                SELECT
                    artifact_id,
                    COUNT(*) AS replica_count,
                    COALESCE(SUM(CASE WHEN is_available THEN 1 ELSE 0 END), 0) AS available_count
                FROM artifact_replicas
                WHERE artifact_id IN ({placeholders})
                GROUP BY artifact_id
                """,
                unique_ids,
            ).fetchall()
        finally:
            cursor.close()

        counts = cast(
            dict[str, tuple[int, int]],
            dict.fromkeys(unique_ids, (0, 0)),
        )
        for row in rows:
            artifact_id = str(row[0])
            replica_count = int(row[1])
            available_count = int(row[2])
            counts[artifact_id] = (replica_count, available_count)
        return counts

    def find_by_filters(
        self,
        artifact_id: str | None = None,
        view_id: str | None = None,
        node_id: str | None = None,
        node_address: str | None = None,
        node_port: int | None = None,
        memory_type: MemoryType | None = None,
        device_id: int | None = None,
    ) -> list[Replica]:
        """Find replicas matching the given filters."""
        cursor = self.get_cursor()
        try:
            # Build dynamic query joining with replica_counters
            query = self._replica_select_sql("LEFT JOIN") + " WHERE 1=1"
            params: list = []

            if artifact_id is not None:
                query += " AND mr.artifact_id = ?"
                params.append(artifact_id)

            if view_id is not None:
                query += " AND COALESCE(mr.view_id, '') = COALESCE(?, '')"
                params.append(view_id)

            if node_id is not None:
                query += " AND mr.node_id = ?"
                params.append(node_id)

            if node_address is not None:
                query += " AND mr.node_address = ?"
                params.append(node_address)

            if node_port is not None:
                query += " AND mr.node_port = ?"
                params.append(node_port)

            if memory_type is not None:
                query += " AND mr.memory_type = ?"
                params.append(memory_type.value)

            if device_id is not None:
                query += " AND mr.device_id = ?"
                params.append(device_id)

            result = cursor.execute(query, params)
            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            replicas = [self._row_to_model(row, columns) for row in rows]
            return replicas
        finally:
            cursor.close()

    def mark_unavailable_by_worker(self, worker_id: str) -> int:
        """Mark all replicas for a worker as unavailable."""
        cursor = self.get_cursor()
        try:
            # First count how many replicas will be affected
            count_result = cursor.execute(
                """
                SELECT COUNT(*) FROM artifact_replicas
                WHERE worker_id = ? AND is_available = TRUE
                """,
                [worker_id],
            ).fetchone()

            affected_count = count_result[0] if count_result else 0

            # Update the replicas
            cursor.execute(
                """
                UPDATE artifact_replicas
                SET is_available = FALSE
                WHERE worker_id = ?
                """,
                [worker_id],
            )

            return affected_count
        finally:
            cursor.close()

    # ========== High Availability Methods ==========

    def list_all_replicas(self) -> list[Replica]:
        """List all replicas in the database."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                self._replica_select_sql("LEFT JOIN") + " ORDER BY mr.created_at DESC"
            )

            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [self._row_to_model(row, columns) for row in rows]
        finally:
            cursor.close()

    def mark_as_stale(self, replica_id: UUID) -> bool:
        """Mark a replica as stale (for recovery purposes)."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                """
                UPDATE artifact_replicas
                SET updated_at = TIMESTAMP '1970-01-01 00:00:00',
                    is_available = FALSE
                WHERE replica_id = ?
                RETURNING replica_id
                """,
                [str(replica_id)],
            )

            updated = result.fetchone() is not None
            if updated:
                logger.debug(f"Marked replica {replica_id} as stale")
            return updated
        finally:
            cursor.close()

    def mark_unavailable(self, replica_id: UUID) -> bool:
        """Mark a replica as unavailable."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                """
                UPDATE artifact_replicas
                SET is_available = FALSE,
                    updated_at = CURRENT_TIMESTAMP
                WHERE replica_id = ?
                RETURNING replica_id
                """,
                [str(replica_id)],
            )

            updated = result.fetchone() is not None
            if updated:
                logger.debug(f"Marked replica {replica_id} as unavailable")
            return updated
        finally:
            cursor.close()

    def find_orphaned_replicas(self) -> list[Replica]:
        """Find replicas that reference non-existent workers."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                self._replica_select_sql("LEFT JOIN")
                + " LEFT JOIN workers w ON mr.worker_id = w.worker_id"
                + " WHERE mr.worker_id IS NOT NULL AND w.worker_id IS NULL"
            )

            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            orphaned = [self._row_to_model(row, columns) for row in rows]
            if orphaned:
                logger.warning(f"Found {len(orphaned)} orphaned replicas")
            return orphaned
        finally:
            cursor.close()

    def get_replicas_by_worker(self, worker_id: str) -> list[Replica]:
        """Get all replicas belonging to a specific worker."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                self._replica_select_sql("LEFT JOIN")
                + " WHERE mr.worker_id = ? ORDER BY mr.created_at DESC",
                [worker_id],
            )

            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [self._row_to_model(row, columns) for row in rows]
        finally:
            cursor.close()

    def get_replicas_by_worker_atomic(self, worker_id: str, cursor) -> list[Replica]:
        """Get replicas for a worker using an existing transaction cursor."""
        result = cursor.execute(
            self._replica_select_sql("LEFT JOIN")
            + " WHERE mr.worker_id = ? ORDER BY mr.created_at DESC",
            [worker_id],
        )
        assert result.description is not None
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [self._row_to_model(row, columns) for row in rows]

    def get_worker_replica_epoch_atomic(
        self, worker_id: str, cursor
    ) -> tuple[int, int]:
        """Return a compact epoch for detecting worker-replica set changes.

        The epoch combines:
        - row count for the worker
        - max(updated_at) in microseconds since epoch
        """
        row = cursor.execute(
            """
            SELECT
                COUNT(*) AS replica_count,
                CAST(COALESCE(MAX(EXTRACT(epoch FROM updated_at) * 1000000), 0) AS BIGINT) AS max_updated_epoch_us
            FROM artifact_replicas
            WHERE worker_id = ?
            """,
            [worker_id],
        ).fetchone()
        if row is None:
            return 0, 0
        return int(row[0] or 0), int(row[1] or 0)

    def update_worker_id(self, replica_id: UUID, new_worker_id: str) -> bool:
        """Update the worker_id for a replica."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                """
                UPDATE artifact_replicas
                SET worker_id = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE replica_id = ?
                RETURNING replica_id
                """,
                [new_worker_id, str(replica_id)],
            )

            updated = result.fetchone() is not None
            if updated:
                logger.debug(
                    f"Updated replica {replica_id} worker_id to {new_worker_id}"
                )
            return updated
        finally:
            cursor.close()

    def get_stale_replicas(self, recovery_time: int) -> list[Replica]:
        """Get replicas that haven't been updated since recovery started."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                self._replica_select_sql("LEFT JOIN")
                + " WHERE EXTRACT(epoch FROM mr.updated_at) < ? ORDER BY mr.updated_at DESC",
                [recovery_time],
            )

            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [self._row_to_model(row, columns) for row in rows]
        finally:
            cursor.close()

    def cleanup_stale_replicas(self, recovery_time: int) -> int:
        """
        Clean up replicas that remained stale after recovery period.

        Returns:
            Number of replicas cleaned up
        """
        cursor = self.get_cursor()
        try:
            # Mark stale replicas as unavailable instead of deleting
            result = cursor.execute(
                """
                UPDATE artifact_replicas
                SET is_available = FALSE
                WHERE EXTRACT(epoch FROM updated_at) < ?
                RETURNING replica_id
                """,
                [recovery_time],
            )

            cleaned_up_count = len(result.fetchall())
            if cleaned_up_count > 0:
                logger.info(f"Marked {cleaned_up_count} stale replicas as unavailable")
            return cleaned_up_count
        finally:
            cursor.close()

    def _row_to_model(self, row: tuple, columns: Sequence[str]) -> Replica:
        """Convert a database row to Replica object using column metadata."""

        if not columns:
            raise ValueError("Replica row conversion requires column metadata.")

        column_index = {column: idx for idx, column in enumerate(columns)}

        def get(column: str, *, default=None):
            idx = column_index.get(column)
            if idx is None:
                return default
            value = row[idx]
            if value is None and default is not None:
                return default
            return value

        def require(column: str):
            value = get(column)
            if value is None:
                raise ValueError(f"Replica row missing required column: {column}")
            return value

        replica_id = _to_uuid(require("replica_id"))
        artifact_id = require("artifact_id")
        view_id = get("view_id")
        disk_path = get("disk_path")
        node_id = require("node_id")
        node_address = require("node_address")
        node_port = int(require("node_port"))
        memory_size = int(require("memory_size"))
        memory_type = MemoryType(require("memory_type"))
        device_id_value = get("device_id")
        device_id = int(device_id_value) if device_id_value is not None else -1
        max_concurrency = int(require("max_concurrency"))
        current_requests_value = get("current_requests", default=0)
        current_requests = int(current_requests_value or 0)
        is_available = bool(require("is_available"))

        remote_memory_keys_value = get("remote_memory_keys", default=())
        remote_memory_keys = (
            list(remote_memory_keys_value) if remote_memory_keys_value else []
        )

        buffer_sizes_value = get("buffer_sizes", default=())
        buffer_sizes = list(buffer_sizes_value) if buffer_sizes_value else []

        export_state_value = get(
            "export_state", default=ExportState.PRESENCE_ONLY.value
        )
        try:
            export_state = ExportState(export_state_value)
        except Exception:
            export_state = ExportState.PRESENCE_ONLY
        export_generation_value = get("export_generation", default=0)
        export_generation = int(export_generation_value or 0)

        verification_json = get("verification_json")
        worker_id = get("worker_id")
        created_at = get("created_at")
        updated_at = get("updated_at")
        expires_at = get("expires_at")

        byte_space = (
            ByteSpaceRef.view(str(view_id))
            if view_id is not None
            else ByteSpaceRef.canonical()
        )

        return Replica(
            replica_id=replica_id,
            artifact_id=artifact_id,
            byte_space=byte_space,
            disk_path=disk_path,
            node_id=node_id,
            node_address=node_address,
            node_port=node_port,
            memory_size=memory_size,
            memory_type=memory_type,
            device_id=device_id,
            max_concurrency=max_concurrency,
            current_requests=current_requests,
            is_available=is_available,
            remote_memory_keys=remote_memory_keys,
            buffer_sizes=buffer_sizes,
            export_state=export_state,
            export_generation=export_generation,
            worker_id=worker_id,
            verification_json=verification_json,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
        )

    def find_by_disk_path(self, disk_path: str) -> list[Replica]:
        """Find all replicas registered under a given disk path (legacy flow)."""
        cursor = self.get_cursor()
        try:
            result = cursor.execute(
                self._replica_select_sql("LEFT JOIN") + " WHERE mr.disk_path = ?",
                [disk_path],
            )
            assert result.description is not None
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [self._row_to_model(row, columns) for row in rows]
        finally:
            cursor.close()
