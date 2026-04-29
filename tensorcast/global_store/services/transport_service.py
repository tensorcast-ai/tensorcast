#  Copyright (c) 2025-2026, TensorCast Team.

"""Service for transport operations."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import TypeVar
from uuid import UUID

from tensorcast.global_store.config import get_config
from tensorcast.global_store.exceptions import (
    DatabaseError,
    NotFoundError,
    TimeoutError,
    ValidationError,
)
from tensorcast.global_store.metrics import (
    dec_active_transports,
    inc_active_transports,
    inc_transport_dispatch_event,
    inc_transport_request,
    observe_transport_wait,
    record_transport_source_assignment,
)
from tensorcast.global_store.models import (
    BroadcastTransportHint,
    PendingTransportRequest,
    PendingTransportState,
    Replica,
    Transport,
    TransportCompletionOutcome,
    TransportSchedulingGroup,
)
from tensorcast.global_store.repositories import (
    PendingTransportRequestRepository,
    ReplicaRepository,
    TransportRepository,
)
from tensorcast.global_store.repositories.base import is_transient_tx_conflict
from tensorcast.global_store.repositories.replica_repository import (
    GroupSourceSpreadPolicy,
    SourceBalanceWeights,
)
from tensorcast.global_store.repositories.transport_repository import (
    TransportWindowRow,
)
from tensorcast.global_store.services.broadcast_service import BroadcastService
from tensorcast.logger import init_logger

logger = init_logger(__name__)
T = TypeVar("T")


class TransportService:
    """Business logic for transport operations."""

    def __init__(
        self,
        replica_repository: ReplicaRepository,
        transport_repository: TransportRepository,
        pending_transport_request_repository: PendingTransportRequestRepository,
        broadcast_service: BroadcastService | None = None,
    ):
        """Initialize service with repositories."""
        self.replica_repository = replica_repository
        self.transport_repository = transport_repository
        self.pending_transport_request_repository = pending_transport_request_repository
        self.broadcast_service = broadcast_service
        self.config = get_config()
        # Serialize queue-wide dispatch to avoid multi-thread transaction storms.
        self._dispatch_loop_lock = threading.Lock()

    def _retry_transient_db_call(self, *, op_name: str, fn: Callable[[], T]) -> T:
        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                if not is_transient_tx_conflict(exc) or attempt == max_attempts - 1:
                    raise
                backoff_sec = min(0.2, 0.005 * (2**attempt))
                logger.warning(
                    "Retrying transient DB conflict op=%s attempt=%s/%s backoff_s=%.3f error=%s",
                    op_name,
                    attempt + 1,
                    max_attempts,
                    backoff_sec,
                    exc,
                )
                time.sleep(backoff_sec)
        raise RuntimeError(f"{op_name} retry loop exhausted")

    def _source_balance_weights(self) -> SourceBalanceWeights:
        policy = self.config.transport_scheduler.source_balance_weights
        return SourceBalanceWeights(
            replica_load=float(policy.replica_load_weight),
            worker_load=float(policy.worker_load_weight),
            recent_assignment_penalty=float(policy.recent_assignment_penalty_weight),
            diffusion_bonus=float(policy.diffusion_bonus_weight),
        )

    def _group_source_spread_policy(self) -> GroupSourceSpreadPolicy:
        policy = self.config.transport_scheduler.group_dispatch
        return GroupSourceSpreadPolicy(
            spread_weight=float(policy.group_source_spread_weight),
            soft_cap_ratio=float(policy.group_source_soft_cap_ratio),
            min_candidates_for_enforce=max(
                1,
                int(policy.group_source_min_candidates_for_enforce),
            ),
        )

    @staticmethod
    def _normalize_request_id(request_id: str) -> str:
        normalized = request_id.strip()
        if not normalized:
            raise ValidationError("request_id is required")
        return normalized

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    @staticmethod
    def _build_request_fingerprint(
        *,
        artifact_id: str,
        view_id: str | None,
        source_node_id: str,
        source_address: str,
        source_port: int,
        requester_worker_id: str | None,
        scheduling_group: TransportSchedulingGroup | None,
        broadcast_hint: BroadcastTransportHint | None = None,
    ) -> str:
        group_kind = (
            str(scheduling_group.group_kind).strip().lower()
            if scheduling_group is not None
            else ""
        )
        normalized_view_id = view_id or ""
        if group_kind == "tp_version":
            normalized_view_id = ""
        payload = {
            "artifact_id": artifact_id,
            "view_id": normalized_view_id,
            "source_node_id": source_node_id,
            "source_address": source_address,
            "source_port": int(source_port),
            "requester_worker_id": (requester_worker_id or "").strip(),
            "broadcast": (
                {
                    "session_id": broadcast_hint.session_id,
                    "strict_parent": bool(broadcast_hint.strict_parent),
                }
                if broadcast_hint is not None
                else None
            ),
            "scheduling_group": (
                {
                    "group_id": scheduling_group.group_id,
                    "group_kind": scheduling_group.group_kind,
                    "total_parts": int(scheduling_group.total_parts),
                    "part_id": scheduling_group.part_id,
                    "priority": int(scheduling_group.priority),
                    "epoch": int(scheduling_group.epoch),
                }
                if scheduling_group is not None
                else None
            ),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_transport(
        self,
        *,
        replica: Replica,
        artifact_id: str,
        requested_view_id: str | None,
        source_node_id: str,
        source_address: str,
        source_port: int,
        request_id: str,
        request_fingerprint: str | None,
        requester_worker_id: str | None,
        scheduling_group: TransportSchedulingGroup | None,
        broadcast_session_id: str | None = None,
        broadcast_edge_id: str | None = None,
    ) -> Transport:
        transport = Transport(
            replica_id=replica.replica_id,
            artifact_id=artifact_id,
            requested_view_id=requested_view_id,
            source_node_id=source_node_id,
            source_address=source_address,
            source_port=source_port,
            replica_memory_size_bytes=max(0, int(replica.memory_size)),
            request_id=request_id,
            request_fingerprint=request_fingerprint,
            requester_worker_id=requester_worker_id,
            broadcast_session_id=broadcast_session_id,
            broadcast_edge_id=broadcast_edge_id,
        )
        transport.set_scheduling_group(scheduling_group)
        return transport

    @staticmethod
    def _is_terminal_failed_transport(transport: Transport) -> bool:
        status = str(transport.status or "").strip().lower()
        if status != "completed":
            return False
        return transport.completion_outcome in {
            TransportCompletionOutcome.FAILED,
            TransportCompletionOutcome.EXPIRED,
            TransportCompletionOutcome.CANCELLED,
        }

    def _reconcile_request_replay_state(self, *, request_id: str, tx) -> None:
        existing_transport = self.transport_repository.find_by_request_id(
            request_id, cursor=tx
        )
        if existing_transport is not None and self._is_terminal_failed_transport(
            existing_transport
        ):
            tx.execute(
                """
                UPDATE artifact_transports
                SET request_id = NULL,
                    request_fingerprint = NULL
                WHERE request_id = ?
                  AND status = 'completed'
                  AND completion_outcome IN ('failed', 'expired', 'cancelled')
                """,
                [request_id],
            )
            tx.execute(
                """
                DELETE FROM pending_transport_requests
                WHERE request_id = ?
                  AND state <> 'enqueued'
                """,
                [request_id],
            )
            inc_transport_dispatch_event("request_recycle_terminal_failure")
            return

        pending = self.pending_transport_request_repository.find_by_request_id(
            request_id, cursor=tx
        )
        if pending is None:
            return
        if pending.state == PendingTransportState.ENQUEUED:
            return
        if existing_transport is not None:
            return
        tx.execute(
            """
            DELETE FROM pending_transport_requests
            WHERE request_id = ?
              AND state <> 'enqueued'
            """,
            [request_id],
        )
        inc_transport_dispatch_event("request_recycle_stale_pending")

    def _resolve_existing_request(
        self,
        request_id: str,
        request_fingerprint: str | None,
    ) -> tuple[Replica, UUID] | None:
        existing = self.transport_repository.find_by_request_id(request_id)
        if existing is not None:
            if self._is_terminal_failed_transport(existing):
                return None
            if (
                existing.request_fingerprint is not None
                and request_fingerprint is not None
                and existing.request_fingerprint != request_fingerprint
            ):
                raise ValidationError(
                    f"request_id={request_id} already used with different payload"
                )
            replica = self.replica_repository.find_by_id(
                existing.replica_id, existing.artifact_id
            )
            if replica is None:
                logger.warning(
                    "Found transport for request_id=%s but source replica is missing",
                    request_id,
                )
                return None
            return replica, existing.transport_id

        pending = self.pending_transport_request_repository.find_by_request_id(
            request_id
        )
        if (
            pending is not None
            and pending.state == PendingTransportState.ENQUEUED
            and request_fingerprint is not None
            and pending.request_fingerprint
            and pending.request_fingerprint != request_fingerprint
        ):
            raise ValidationError(
                f"request_id={request_id} already used with different payload"
            )
        return None

    def request_transport(
        self,
        artifact_id: str,
        source_node_id: str,
        source_address: str,
        source_port: int,
        request_id: str,
        wait_timeout_ms: int = 0,
        view_id: str | None = None,
        scheduling_group: TransportSchedulingGroup | None = None,
        requester_worker_id: str | None = None,
        broadcast_hint: BroadcastTransportHint | None = None,
    ) -> tuple[Replica, UUID]:
        """
        Request an artifact transport via unified pending-queue dispatch.

        Args:
            artifact_id: Content-addressed artifact id (mi2:...)
            view_id: Optional view byte-space id (None for canonical)
            source_node_id: Source node ID
            source_address: Source node address
            source_port: Source node port
            wait_timeout_ms: Max time to wait for availability
            scheduling_group: Optional scheduling-group hint
            requester_worker_id: Optional requester worker identity
            request_id: Required idempotency key

        Returns:
            Tuple of (selected replica, transport ID)

        Raises:
            TimeoutError: If no replica available within timeout
        """
        normalized_request_id = self._normalize_request_id(request_id)
        normalized_requester_worker_id = self._normalize_optional_text(
            requester_worker_id
        )
        request_fingerprint = self._build_request_fingerprint(
            artifact_id=artifact_id,
            view_id=view_id,
            source_node_id=source_node_id,
            source_address=source_address,
            source_port=source_port,
            requester_worker_id=normalized_requester_worker_id,
            scheduling_group=scheduling_group,
            broadcast_hint=broadcast_hint,
        )
        existing = self._resolve_existing_request(
            normalized_request_id, request_fingerprint
        )
        if existing is not None:
            return existing

        if broadcast_hint is not None:
            return self._request_transport_broadcast(
                artifact_id=artifact_id,
                view_id=view_id,
                source_node_id=source_node_id,
                source_address=source_address,
                source_port=source_port,
                requester_worker_id=normalized_requester_worker_id,
                request_fingerprint=request_fingerprint,
                request_id=normalized_request_id,
                broadcast_hint=broadcast_hint,
            )

        return self._request_transport_group_dispatch(
            artifact_id=artifact_id,
            view_id=view_id,
            source_node_id=source_node_id,
            source_address=source_address,
            source_port=source_port,
            wait_timeout_ms=wait_timeout_ms,
            scheduling_group=scheduling_group,
            requester_worker_id=normalized_requester_worker_id,
            request_fingerprint=request_fingerprint,
            request_id=normalized_request_id,
        )

    def _request_transport_broadcast(
        self,
        *,
        artifact_id: str,
        view_id: str | None,
        source_node_id: str,
        source_address: str,
        source_port: int,
        requester_worker_id: str | None,
        request_fingerprint: str,
        request_id: str,
        broadcast_hint: BroadcastTransportHint,
    ) -> tuple[Replica, UUID]:
        start_time = time.time()
        if self.broadcast_service is None:
            raise ValidationError(
                "broadcast transport requested but service is unavailable"
            )
        if requester_worker_id is None:
            raise ValidationError("broadcast transport requires requester_worker_id")

        try:
            with self._dispatch_loop_lock, self.replica_repository.transaction() as tx:
                existing = self.transport_repository.find_by_request_id(
                    request_id,
                    cursor=tx,
                )
                if existing is not None:
                    if (
                        existing.request_fingerprint is not None
                        and existing.request_fingerprint != request_fingerprint
                    ):
                        raise ValidationError(
                            f"request_id={request_id} already used with different payload"
                        )
                    replica = self.replica_repository.find_by_id(
                        existing.replica_id,
                        existing.artifact_id,
                        cursor=tx,
                    )
                    if replica is not None:
                        return replica, existing.transport_id

                replica, edge = self.broadcast_service.claim_transport_edge(
                    session_id=broadcast_hint.session_id,
                    artifact_id=artifact_id,
                    requested_view_id=view_id,
                    requester_worker_id=requester_worker_id,
                    request_id=request_id,
                    heartbeat_timeout_seconds=self.config.heartbeat_timeout_ms / 1000,
                    cursor=tx,
                )
                transport = self._build_transport(
                    replica=replica,
                    artifact_id=artifact_id,
                    requested_view_id=view_id,
                    source_node_id=source_node_id,
                    source_address=source_address,
                    source_port=source_port,
                    request_id=request_id,
                    request_fingerprint=request_fingerprint,
                    requester_worker_id=requester_worker_id,
                    scheduling_group=None,
                    broadcast_session_id=broadcast_hint.session_id,
                    broadcast_edge_id=edge.edge_id,
                )
                resolved_transport, created = (
                    self.transport_repository.create_if_absent_with_cursor(
                        transport,
                        tx,
                    )
                )
                if not created:
                    self.replica_repository.decrement_requests_with_cursor(
                        replica.replica_id,
                        tx,
                    )
                else:
                    inc_active_transports()
                    record_transport_source_assignment(
                        artifact_id=artifact_id,
                        replica_id=str(replica.replica_id),
                        source_created_at=replica.created_at,
                    )
                inc_transport_request(artifact_id, "success")
                observe_transport_wait(artifact_id, time.time() - start_time)
                return replica, resolved_transport.transport_id
        except DatabaseError as exc:
            if isinstance(exc.__cause__, (NotFoundError, ValidationError)):
                raise exc.__cause__ from exc
            raise

    def _request_transport_group_dispatch(
        self,
        *,
        artifact_id: str,
        view_id: str | None,
        source_node_id: str,
        source_address: str,
        source_port: int,
        wait_timeout_ms: int,
        scheduling_group: TransportSchedulingGroup | None,
        requester_worker_id: str | None,
        request_fingerprint: str,
        request_id: str,
    ) -> tuple[Replica, UUID]:
        start_time = time.time()
        timeout_deadline = start_time + (max(0, wait_timeout_ms) / 1000.0)

        if not self.replica_repository.has_any_replica(artifact_id, view_id):
            inc_transport_request(artifact_id, "not_found")
            observe_transport_wait(artifact_id, time.time() - start_time)
            raise NotFoundError(f"No replicas registered for artifact {artifact_id}")

        deadline_at = None
        if wait_timeout_ms > 0:
            deadline_at = datetime.now(timezone.utc) + timedelta(
                milliseconds=wait_timeout_ms
            )

        pending_request = PendingTransportRequest(
            request_id=request_id,
            request_fingerprint=request_fingerprint,
            artifact_id=artifact_id,
            requested_view_id=view_id,
            source_node_id=source_node_id,
            source_address=source_address,
            source_port=source_port,
            requester_worker_id=self._normalize_optional_text(requester_worker_id),
            deadline_at=deadline_at,
        )
        pending_request.set_scheduling_group(scheduling_group)

        try:
            (
                resolved_replica,
                resolved_transport_id,
                terminal_pending_state,
            ) = self._retry_transient_db_call(
                op_name="request_transport_initial_dispatch",
                fn=lambda: self._enqueue_and_attempt_initial_dispatch(
                    pending_request=pending_request,
                    request_id=request_id,
                    request_fingerprint=request_fingerprint,
                    artifact_id=artifact_id,
                    view_id=view_id,
                    scheduling_group=scheduling_group,
                ),
            )
            if resolved_replica is not None and resolved_transport_id is not None:
                inc_transport_request(artifact_id, "success")
                observe_transport_wait(artifact_id, time.time() - start_time)
                return resolved_replica, resolved_transport_id
            if terminal_pending_state == PendingTransportState.EXPIRED:
                inc_transport_dispatch_event("request_terminal_expired")
                inc_transport_request(artifact_id, "timeout")
                observe_transport_wait(artifact_id, time.time() - start_time)
                raise TimeoutError("Queued transport request expired before dispatch")
            if terminal_pending_state == PendingTransportState.CANCELLED:
                inc_transport_dispatch_event("request_terminal_cancelled")
                inc_transport_request(artifact_id, "timeout")
                observe_transport_wait(artifact_id, time.time() - start_time)
                raise TimeoutError("Queued transport request cancelled before dispatch")
        except DatabaseError as exc:
            if isinstance(exc.__cause__, ValidationError):
                raise exc.__cause__ from exc
            raise

        while True:
            loop_terminal_pending_state: PendingTransportState | None = None
            try:
                loop_resolved_replica: Replica | None = None
                loop_resolved_transport_id: UUID | None = None
                dispatch_owner = self._dispatch_loop_lock.acquire(blocking=False)
                if dispatch_owner:
                    try:
                        with self.replica_repository.transaction() as tx:
                            self.pending_transport_request_repository.purge_malformed_rows(
                                cursor=tx
                            )
                            self.pending_transport_request_repository.expire_enqueued_deadlines(
                                now_utc=datetime.now(timezone.utc),
                                cursor=tx,
                            )
                            self._dispatch_pending_requests(tx=tx)
                            (
                                loop_resolved_replica,
                                loop_resolved_transport_id,
                                loop_terminal_pending_state,
                            ) = self._resolve_pending_request_state(
                                request_id=request_id,
                                request_fingerprint=request_fingerprint,
                                cursor=tx,
                            )
                    finally:
                        self._dispatch_loop_lock.release()
                else:
                    (
                        loop_resolved_replica,
                        loop_resolved_transport_id,
                        loop_terminal_pending_state,
                    ) = self._resolve_pending_request_state(
                        request_id=request_id,
                        request_fingerprint=request_fingerprint,
                    )

                if (
                    loop_resolved_replica is not None
                    and loop_resolved_transport_id is not None
                ):
                    inc_transport_request(artifact_id, "success")
                    observe_transport_wait(artifact_id, time.time() - start_time)
                    return loop_resolved_replica, loop_resolved_transport_id
            except (NotFoundError, TimeoutError):
                raise
            except Exception as exc:  # noqa: BLE001
                if is_transient_tx_conflict(exc):
                    inc_transport_dispatch_event("request_loop_tx_conflict_retry")
                    logger.debug(
                        "Group-dispatch request retry due to transient DB error: %s",
                        exc,
                    )
                else:
                    raise

            if loop_terminal_pending_state == PendingTransportState.EXPIRED:
                inc_transport_dispatch_event("request_terminal_expired")
                inc_transport_request(artifact_id, "timeout")
                observe_transport_wait(artifact_id, time.time() - start_time)
                raise TimeoutError("Queued transport request expired before dispatch")
            if loop_terminal_pending_state == PendingTransportState.CANCELLED:
                inc_transport_dispatch_event("request_terminal_cancelled")
                inc_transport_request(artifact_id, "timeout")
                observe_transport_wait(artifact_id, time.time() - start_time)
                raise TimeoutError("Queued transport request cancelled before dispatch")

            if time.time() >= timeout_deadline:
                inc_transport_dispatch_event("request_deadline_cancelled")
                self.pending_transport_request_repository.mark_cancelled(request_id)
                inc_transport_request(artifact_id, "timeout")
                observe_transport_wait(artifact_id, time.time() - start_time)
                raise TimeoutError(
                    f"No available replica for artifact {artifact_id} within timeout"
                )

            time.sleep(self.config.transport_wait_retry_interval_ms / 1000)

    def _enqueue_and_attempt_initial_dispatch(
        self,
        *,
        pending_request: PendingTransportRequest,
        request_id: str,
        request_fingerprint: str,
        artifact_id: str,
        view_id: str | None,
        scheduling_group: TransportSchedulingGroup | None,
    ) -> tuple[Replica | None, UUID | None, PendingTransportState | None]:
        # Keep the first request replay check, queue mutation, and initial dispatch
        # on a single serialized transaction timeline so competing callers either
        # observe the same pending row or retry on a fresh snapshot.
        with self._dispatch_loop_lock, self.replica_repository.transaction() as tx:
            self.pending_transport_request_repository.purge_malformed_rows(cursor=tx)
            self._reconcile_request_replay_state(request_id=request_id, tx=tx)
            existing_pending = (
                self.pending_transport_request_repository.find_by_request_id(
                    request_id, cursor=tx
                )
            )
            if (
                existing_pending is not None
                and existing_pending.request_fingerprint != request_fingerprint
            ):
                raise ValidationError(
                    f"request_id={request_id} already used with different payload"
                )
            self._validate_group_contract_with_cursor(
                tx=tx,
                request_id=request_id,
                artifact_id=artifact_id,
                view_id=view_id,
                scheduling_group=scheduling_group,
            )
            persisted_pending = (
                self.pending_transport_request_repository.create_if_absent_with_cursor(
                    pending_request,
                    tx,
                )
            )
            if persisted_pending.request_fingerprint != request_fingerprint:
                raise ValidationError(
                    f"request_id={request_id} already used with different payload"
                )
            existing_transport = self.transport_repository.find_by_request_id(
                request_id, cursor=tx
            )
            if (
                existing_transport is not None
                and existing_transport.request_fingerprint is not None
                and existing_transport.request_fingerprint != request_fingerprint
            ):
                raise ValidationError(
                    f"request_id={request_id} already used with different payload"
                )
            if existing_transport is not None:
                replica = self.replica_repository.find_by_id(
                    existing_transport.replica_id,
                    existing_transport.artifact_id,
                    cursor=tx,
                )
                if replica is not None:
                    return replica, existing_transport.transport_id, None

            self.pending_transport_request_repository.expire_enqueued_deadlines(
                now_utc=datetime.now(timezone.utc),
                cursor=tx,
            )
            self._dispatch_pending_requests(tx=tx)
            return self._resolve_pending_request_state(
                request_id=request_id,
                request_fingerprint=request_fingerprint,
                cursor=tx,
            )

    def _resolve_pending_request_state(
        self,
        *,
        request_id: str,
        request_fingerprint: str,
        cursor=None,
    ) -> tuple[Replica | None, UUID | None, PendingTransportState | None]:
        pending_row = self.pending_transport_request_repository.find_by_request_id(
            request_id, cursor=cursor
        )
        if pending_row is None:
            transport = self.transport_repository.find_by_request_id(
                request_id, cursor=cursor
            )
            if transport is None:
                inc_transport_dispatch_event("request_pending_missing")
                return None, None, None
            if self._is_terminal_failed_transport(transport):
                inc_transport_dispatch_event("request_terminal_replay_blocked")
                return None, None, PendingTransportState.EXPIRED
            if (
                transport.request_fingerprint is not None
                and transport.request_fingerprint != request_fingerprint
            ):
                raise ValidationError(
                    f"request_id={request_id} replay fingerprint mismatch"
                )
            replica = self.replica_repository.find_by_id(
                transport.replica_id,
                transport.artifact_id,
                cursor=cursor,
            )
            if replica is None:
                return None, None, None
            return replica, transport.transport_id, None

        if pending_row.state == PendingTransportState.DISPATCHED:
            transport = self.transport_repository.find_by_request_id(
                request_id, cursor=cursor
            )
            if transport is None:
                inc_transport_dispatch_event("request_dispatched_missing_transport")
                return None, None, None
            if self._is_terminal_failed_transport(transport):
                inc_transport_dispatch_event("request_terminal_replay_blocked")
                return None, None, PendingTransportState.EXPIRED

            if (
                transport.request_fingerprint is not None
                and transport.request_fingerprint != request_fingerprint
            ):
                raise ValidationError(
                    f"request_id={request_id} replay fingerprint mismatch"
                )

            replica = self.replica_repository.find_by_id(
                transport.replica_id,
                transport.artifact_id,
                cursor=cursor,
            )
            if replica is None:
                return None, None, None
            return replica, transport.transport_id, None

        if pending_row.state == PendingTransportState.EXPIRED:
            return None, None, PendingTransportState.EXPIRED
        if pending_row.state == PendingTransportState.CANCELLED:
            return None, None, PendingTransportState.CANCELLED
        return None, None, None

    def _validate_group_contract_with_cursor(
        self,
        *,
        tx,
        request_id: str,
        artifact_id: str,
        view_id: str | None,
        scheduling_group: TransportSchedulingGroup | None,
    ) -> None:
        if scheduling_group is None:
            return
        group = scheduling_group
        normalized_view_id = view_id or ""
        normalized_group_kind = str(group.group_kind).strip().lower()
        group_epoch = int(group.epoch)
        group_total_parts = int(group.total_parts)
        # tp_version groups are keyed by logical version, and key remap retries
        # may transiently produce different artifact ids within the same epoch.
        # Keep total_parts/part_id invariants strict, but do not fail the whole
        # request path on artifact/view variance for tp_version.
        enforce_view_consistency = normalized_group_kind != "tp_version"
        enforce_artifact_consistency = normalized_group_kind != "tp_version"
        if group_total_parts <= 0:
            raise ValidationError(
                "group contract violation: total_parts must be > 0 for "
                f"group={group.group_kind}:{group.group_id}:{group_epoch}"
            )
        if not group.part_id:
            raise ValidationError(
                "group contract violation: part_id must be non-empty for "
                f"group={group.group_kind}:{group.group_id}:{group_epoch}"
            )

        mismatch_pending = tx.execute(
            """
            SELECT request_id
            FROM pending_transport_requests
            WHERE group_kind = ?
              AND group_id = ?
              AND COALESCE(group_epoch, 0) = ?
              AND state = 'enqueued'
              AND request_id <> ?
              AND (
                (? = 1 AND artifact_id <> ?)
                OR (? = 1 AND COALESCE(requested_view_id, '') <> ?)
                OR COALESCE(group_total_parts, 0) <> ?
              )
            LIMIT 1
            """,
            [
                group.group_kind,
                group.group_id,
                group_epoch,
                request_id,
                int(enforce_artifact_consistency),
                artifact_id,
                int(enforce_view_consistency),
                normalized_view_id,
                group_total_parts,
            ],
        ).fetchone()
        if mismatch_pending is not None:
            raise ValidationError(
                "group contract violation: artifact/view/total_parts mismatch in "
                f"pending queue group={group.group_kind}:{group.group_id}:{group_epoch}"
            )

        mismatch_transport = tx.execute(
            """
            SELECT request_id
            FROM artifact_transports
            WHERE group_kind = ?
              AND group_id = ?
              AND COALESCE(group_epoch, 0) = ?
              AND COALESCE(request_id, '') <> ?
              AND (
                (? = 1 AND artifact_id <> ?)
                OR (? = 1 AND COALESCE(requested_view_id, '') <> ?)
                OR COALESCE(group_total_parts, 0) <> ?
              )
            LIMIT 1
            """,
            [
                group.group_kind,
                group.group_id,
                group_epoch,
                request_id,
                int(enforce_artifact_consistency),
                artifact_id,
                int(enforce_view_consistency),
                normalized_view_id,
                group_total_parts,
            ],
        ).fetchone()
        if mismatch_transport is not None:
            raise ValidationError(
                "group contract violation: artifact/view/total_parts mismatch in "
                f"transport history group={group.group_kind}:{group.group_id}:{group_epoch}"
            )

        duplicate_pending_part = tx.execute(
            """
            SELECT request_id
            FROM pending_transport_requests
            WHERE group_kind = ?
              AND group_id = ?
              AND COALESCE(group_epoch, 0) = ?
              AND state = 'enqueued'
              AND group_part_id = ?
              AND request_id <> ?
            LIMIT 1
            """,
            [
                group.group_kind,
                group.group_id,
                group_epoch,
                group.part_id,
                request_id,
            ],
        ).fetchone()
        if duplicate_pending_part is not None:
            raise ValidationError(
                "group contract violation: duplicate part_id in pending queue "
                f"group={group.group_kind}:{group.group_id}:{group_epoch} part_id={group.part_id}"
            )

        duplicate_transport_part = tx.execute(
            """
            SELECT request_id
            FROM artifact_transports
            WHERE group_kind = ?
              AND group_id = ?
              AND COALESCE(group_epoch, 0) = ?
              AND group_part_id = ?
              AND COALESCE(request_id, '') <> ?
            LIMIT 1
            """,
            [
                group.group_kind,
                group.group_id,
                group_epoch,
                group.part_id,
                request_id,
            ],
        ).fetchone()
        if duplicate_transport_part is not None:
            raise ValidationError(
                "group contract violation: duplicate part_id in transport history "
                f"group={group.group_kind}:{group.group_id}:{group_epoch} part_id={group.part_id}"
            )

    def _group_dispatch_sort_key(
        self,
        *,
        pending_request: PendingTransportRequest,
        min_completion_ratio: float,
        now_utc: datetime,
        tx,
    ) -> tuple[int, int, float, int, datetime, str]:
        group = pending_request.scheduling_group
        if group is None:
            return (
                1,
                1,
                0.0,
                -int(pending_request.group_priority or 0),
                pending_request.created_at or now_utc,
                pending_request.request_id,
            )

        progress = self.transport_repository.get_group_progress(
            group_kind=group.group_kind,
            group_id=group.group_id,
            group_epoch=group.epoch,
            total_parts_hint=group.total_parts,
            cursor=tx,
        )
        ratio = progress.completion_ratio

        aging_threshold_ms = max(
            1,
            int(
                self.config.transport_scheduler.group_dispatch.starvation_aging_threshold_ms
            ),
        )
        stale = True
        if progress.last_success_at is not None:
            stale = (now_utc - progress.last_success_at) >= timedelta(
                milliseconds=aging_threshold_ms
            )
        starvation_rank = 0 if stale else 1

        fairness_floor_ratio = float(
            self.config.transport_scheduler.group_dispatch.fairness_floor_ratio
        )
        fairness_rank = (
            0 if ratio <= (min_completion_ratio + fairness_floor_ratio) else 1
        )

        completion_bias_weight = float(
            self.config.transport_scheduler.group_dispatch.completion_bias_weight
        )
        completion_rank = -ratio * completion_bias_weight

        created_at = pending_request.created_at or now_utc
        priority_rank = -int(pending_request.group_priority or 0)
        return (
            starvation_rank,
            fairness_rank,
            completion_rank,
            priority_rank,
            created_at,
            pending_request.request_id,
        )

    def _dispatch_pending_requests(self, *, tx) -> int:
        dispatch_cfg = self.config.transport_scheduler.group_dispatch
        scan_limit = max(1, int(dispatch_cfg.queue_scan_limit))
        dispatch_limit = max(1, int(dispatch_cfg.dispatch_batch_limit))

        pending = self.pending_transport_request_repository.list_enqueued(
            limit=scan_limit, cursor=tx
        )
        if not pending:
            return 0

        now_utc = datetime.now(timezone.utc)
        group_ratios: list[float] = []
        for pending_request in pending:
            group = pending_request.scheduling_group
            if group is None:
                continue
            progress = self.transport_repository.get_group_progress(
                group_kind=group.group_kind,
                group_id=group.group_id,
                group_epoch=group.epoch,
                total_parts_hint=group.total_parts,
                cursor=tx,
            )
            group_ratios.append(progress.completion_ratio)
        min_completion_ratio = min(group_ratios) if group_ratios else 0.0

        pending_sorted = sorted(
            pending,
            key=lambda req: self._group_dispatch_sort_key(
                pending_request=req,
                min_completion_ratio=min_completion_ratio,
                now_utc=now_utc,
                tx=tx,
            ),
        )

        source_weights = self._source_balance_weights()
        group_source_policy = self._group_source_spread_policy()
        dispatched = 0
        for pending_request in pending_sorted:
            if dispatched >= dispatch_limit:
                break
            group_source_counts: dict[str, int] = {}
            group = pending_request.scheduling_group
            if group is not None:
                group_source_counts = self.transport_repository.get_group_source_counts(
                    group_kind=group.group_kind,
                    group_id=group.group_id,
                    group_epoch=group.epoch,
                    cursor=tx,
                )

            selection = self.replica_repository.find_available_for_transport(
                artifact_id=pending_request.artifact_id,
                view_id=pending_request.requested_view_id,
                heartbeat_timeout_seconds=self.config.heartbeat_timeout_ms / 1000,
                scheduler_mode="GROUP_DISPATCH",
                source_balance_weights=source_weights,
                group_source_counts=group_source_counts,
                group_source_policy=group_source_policy,
                cursor=tx,
            )
            if selection.replica is None:
                continue

            replica = selection.replica
            transport = self._build_transport(
                replica=replica,
                artifact_id=pending_request.artifact_id,
                requested_view_id=pending_request.requested_view_id,
                source_node_id=pending_request.source_node_id,
                source_address=pending_request.source_address,
                source_port=pending_request.source_port,
                request_id=pending_request.request_id,
                request_fingerprint=pending_request.request_fingerprint,
                requester_worker_id=pending_request.requester_worker_id,
                scheduling_group=pending_request.scheduling_group,
            )

            resolved_transport, created_transport = (
                self.transport_repository.create_if_absent_with_cursor(transport, tx)
            )
            if (
                resolved_transport.request_fingerprint is not None
                and resolved_transport.request_fingerprint
                != pending_request.request_fingerprint
            ):
                raise ValidationError(
                    "request replay fingerprint mismatch while dispatching "
                    f"request_id={pending_request.request_id}"
                )
            if not created_transport:
                inc_transport_dispatch_event("claim_rollback_existing_transport")
                self.replica_repository.decrement_requests_with_cursor(
                    replica.replica_id, tx
                )
            marked_dispatched = (
                self.pending_transport_request_repository.mark_dispatched(
                    pending_request.request_id, tx
                )
            )
            if not marked_dispatched and created_transport:
                inc_transport_dispatch_event("claim_rollback_dispatch_state_race")
                self.transport_repository.delete_with_cursor(
                    transport.transport_id,
                    tx,
                )
                self.replica_repository.decrement_requests_with_cursor(
                    replica.replica_id,
                    tx,
                )
                continue
            if marked_dispatched:
                if created_transport:
                    inc_transport_dispatch_event("dispatched_new_transport")
                    inc_active_transports()
                    record_transport_source_assignment(
                        artifact_id=pending_request.artifact_id,
                        replica_id=str(replica.replica_id),
                        source_created_at=replica.created_at,
                    )
                else:
                    inc_transport_dispatch_event("dispatched_reused_transport")
                dispatched += 1

        return dispatched

    def complete_transport(
        self,
        transport_id: UUID,
        outcome: TransportCompletionOutcome,
        outcome_detail: str | None = None,
    ) -> tuple[int, int]:
        """
        Complete a transport and release resources.

        Args:
            transport_id: ID of the transport to complete
            outcome: requester-reported completion outcome
            outcome_detail: optional detail for failure diagnostics

        Returns:
            Tuple of (current_requests, max_concurrency) after completion

        Raises:
            NotFoundError: If transport not found
        """
        if outcome == TransportCompletionOutcome.UNSPECIFIED:
            raise ValidationError(
                "complete_transport requires explicit outcome "
                "(SUCCESS/FAILED/EXPIRED/CANCELLED)"
            )
        # Share the same mutation gate with dispatch to avoid hot-row conflicts
        # on replica counters under high fanout completion bursts.
        try:
            with self._dispatch_loop_lock:
                current, max_conc, transport, released = self._retry_transient_db_call(
                    op_name="complete_transport_tx",
                    fn=lambda: self._complete_transport_in_single_tx(
                        transport_id=transport_id,
                        outcome=outcome,
                        outcome_detail=outcome_detail,
                    ),
                )
        except DatabaseError as exc:
            # Keep public contract stable for callers/tests.
            if isinstance(exc.__cause__, NotFoundError):
                raise exc.__cause__ from exc
            raise

        if released:
            dec_active_transports()
            logger.info(
                "Completed transport %s for %s, replica=%s, outcome=%s, new_load=%s/%s",
                transport_id,
                transport.artifact_id,
                transport.replica_id,
                outcome.value,
                current,
                max_conc,
            )
        return current, max_conc

    def _complete_transport_in_single_tx(
        self,
        *,
        transport_id: UUID,
        outcome: TransportCompletionOutcome,
        outcome_detail: str | None,
    ) -> tuple[int, int, Transport, bool]:
        with self.replica_repository.transaction() as tx:
            transport = self.transport_repository.find_by_id(transport_id, cursor=tx)
            if transport is None:
                raise NotFoundError(f"Transport {transport_id} not found")

            status_updated = self.transport_repository.complete_if_in_progress(
                transport_id,
                completed_at=datetime.now(timezone.utc),
                outcome=outcome,
                outcome_detail=outcome_detail,
                cursor=tx,
            )
            if not status_updated:
                replica = self.replica_repository.find_by_id(
                    transport.replica_id, transport.artifact_id, cursor=tx
                )
                if replica is None:
                    return 0, 0, transport, False
                return (
                    int(replica.current_requests),
                    int(replica.max_concurrency),
                    transport,
                    False,
                )

            # Release source quota in the same transaction as status completion.
            self.replica_repository.decrement_requests_with_cursor(
                transport.replica_id, tx
            )
            if (
                self.broadcast_service is not None
                and transport.broadcast_session_id
                and transport.broadcast_edge_id
            ):
                self.broadcast_service.complete_transport_edge(
                    session_id=transport.broadcast_session_id,
                    edge_id=transport.broadcast_edge_id,
                    transport_outcome=outcome,
                    outcome_detail=outcome_detail,
                    cursor=tx,
                )
            replica_after = self.replica_repository.find_by_id(
                transport.replica_id, transport.artifact_id, cursor=tx
            )
            if replica_after is None:
                return 0, 0, transport, True
            return (
                int(replica_after.current_requests),
                int(replica_after.max_concurrency),
                transport,
                True,
            )

    def query_transport_window(
        self,
        *,
        started_at: datetime,
        finished_at: datetime,
        limit: int,
    ) -> list[TransportWindowRow]:
        if finished_at < started_at:
            raise ValidationError("created_at_end must be >= created_at_start")
        bounded_limit = min(max(1, int(limit)), 1_000_000)
        return self.transport_repository.list_rows_in_created_window(
            started_at=started_at,
            finished_at=finished_at,
            limit=bounded_limit,
        )

    @staticmethod
    def _is_malformed_inflight_transport(transport: Transport) -> bool:
        status = str(transport.status or "").strip().lower()
        if "in_progress" not in status:
            return False
        return status != "in_progress"

    def cleanup_expired_transports(self, expiration_seconds: int | None = None) -> int:
        """Release transports that have been in *in_progress* state for too long."""
        effective_expiration = (
            max(60, int(self.config.cleanup_interval_ms / 1000) * 10)
            if expiration_seconds is None
            else int(expiration_seconds)
        )
        expired: list[Transport] = []
        malformed: list[Transport] = []
        cleaned = 0

        try:
            pending = self.transport_repository.list_inflight(limit=10_000)
            malformed = [
                transport
                for transport in pending
                if self._is_malformed_inflight_transport(transport)
            ]
            malformed_ids = {transport.transport_id for transport in malformed}
            expired = [
                transport
                for transport in pending
                if transport.transport_id not in malformed_ids
                and transport.age_seconds > effective_expiration
            ]
        except Exception:  # noqa: BLE001
            logger.exception("Failed to fetch pending transports for cleanup")
            return 0

        malformed_cleaned = 0
        stale_cleaned = 0
        for transport in malformed:
            try:
                self.complete_transport(
                    transport.transport_id,
                    outcome=TransportCompletionOutcome.EXPIRED,
                    outcome_detail=(
                        f"cleanup_malformed_inflight status={transport.status}"
                    ),
                )
                malformed_cleaned += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to force-complete malformed transport %s: %s",
                    transport.transport_id,
                    exc,
                )

        for transport in expired:
            try:
                self.complete_transport(
                    transport.transport_id,
                    outcome=TransportCompletionOutcome.EXPIRED,
                    outcome_detail=(
                        f"cleanup_expired_transports age_seconds={transport.age_seconds:.3f}"
                    ),
                )
                stale_cleaned += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to force-complete stale transport %s: %s",
                    transport.transport_id,
                    exc,
                )

        if malformed_cleaned or stale_cleaned:
            logger.info(
                "Cleaned up transports malformed=%s stale=%s (>%ss)",
                malformed_cleaned,
                stale_cleaned,
                effective_expiration,
            )

        cleaned += malformed_cleaned + stale_cleaned
        return cleaned
