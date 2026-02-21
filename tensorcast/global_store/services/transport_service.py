#  Copyright (c) 2025-2026, TensorCast Team.

"""Service for transport operations."""

import time
from typing import Tuple
from uuid import UUID

from tensorcast.global_store.config import get_config
from tensorcast.global_store.exceptions import NotFoundError, TimeoutError
from tensorcast.global_store.metrics import (
    dec_active_transports,
    inc_active_transports,
    inc_transport_no_exportable,
    inc_transport_request,
    observe_transport_wait,
)
from tensorcast.global_store.models import Replica, Transport
from tensorcast.global_store.repositories import (
    ReplicaRepository,
    TransportRepository,
)
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class TransportService:
    """Business logic for transport operations."""

    def __init__(
        self,
        replica_repository: ReplicaRepository,
        transport_repository: TransportRepository,
    ):
        """Initialize service with repositories."""
        self.replica_repository = replica_repository
        self.transport_repository = transport_repository
        self.config = get_config()

    def _log_eligibility_snapshot(
        self,
        *,
        artifact_id: str,
        view_id: str | None,
        wait_timeout_ms: int,
        outcome: str,
        waited_sec: float,
    ) -> None:
        try:
            snapshot = self.replica_repository.get_transport_eligibility_snapshot(
                artifact_id=artifact_id,
                view_id=view_id,
                heartbeat_timeout_seconds=self.config.heartbeat_timeout_ms / 1000,
            )
            logger.warning(
                "Transport request %s for %s (view_id=%s) after %.3fs "
                "(wait_timeout_ms=%s). %s",
                outcome,
                artifact_id,
                view_id or "",
                waited_sec,
                wait_timeout_ms,
                snapshot.format_for_log(),
            )
        except Exception:
            logger.exception(
                "Failed to build transport eligibility snapshot for %s", artifact_id
            )

    def request_transport(
        self,
        artifact_id: str,
        source_node_id: str,
        source_address: str,
        source_port: int,
        wait_timeout_ms: int = 0,
        view_id: str | None = None,
    ) -> Tuple[Replica, UUID]:
        """
        Request a artifact transport with load balancing.

        Args:
            artifact_id: Content-addressed artifact id (mi2:...)
            view_id: Optional view byte-space id (None for canonical)
            source_node_id: Source node ID
            source_address: Source node address
            source_port: Source node port
            wait_timeout_ms: Max time to wait for availability

        Returns:
            Tuple of (selected replica, transport ID)

        Raises:
            TimeoutError: If no replica available within timeout
        """
        start_time = time.time()
        end_time = start_time + (wait_timeout_ms / 1000 if wait_timeout_ms > 0 else 0)

        if not self.replica_repository.has_any_replica(artifact_id, view_id):
            inc_transport_request(artifact_id, "not_found")
            wait_sec = time.time() - start_time
            self._log_eligibility_snapshot(
                artifact_id=artifact_id,
                view_id=view_id,
                wait_timeout_ms=wait_timeout_ms,
                outcome="not_found_no_replica",
                waited_sec=wait_sec,
            )
            observe_transport_wait(artifact_id, wait_sec)
            raise NotFoundError(f"No replicas registered for artifact {artifact_id}")

        while True:
            try:
                # Try to find and claim an available replica
                selection = self.replica_repository.find_available_for_transport(
                    artifact_id=artifact_id,
                    view_id=view_id,
                    heartbeat_timeout_seconds=self.config.heartbeat_timeout_ms / 1000,
                )

                if selection.replica:
                    replica = selection.replica
                    # Create transport record
                    transport = Transport(
                        replica_id=replica.replica_id,
                        artifact_id=artifact_id,
                        source_node_id=source_node_id,
                        source_address=source_address,
                        source_port=source_port,
                    )

                    self.transport_repository.create(transport)

                    # Metrics
                    inc_transport_request(artifact_id, "success")
                    wait_sec = time.time() - start_time
                    observe_transport_wait(artifact_id, wait_sec)
                    inc_active_transports()

                    logger.info(
                        f"Transport requested for {artifact_id}, "
                        f"replica: {replica.replica_id}, "
                        f"transport: {transport.transport_id}, "
                        f"load: {replica.current_requests}/{replica.max_concurrency}"
                    )

                    return replica, transport.transport_id
                if selection.exportable_replicas == 0:
                    inc_transport_request(artifact_id, "not_found")
                    inc_transport_no_exportable(artifact_id)
                    wait_sec = time.time() - start_time
                    self._log_eligibility_snapshot(
                        artifact_id=artifact_id,
                        view_id=view_id,
                        wait_timeout_ms=wait_timeout_ms,
                        outcome="not_found_no_exportable",
                        waited_sec=wait_sec,
                    )
                    observe_transport_wait(artifact_id, wait_sec)
                    raise NotFoundError(
                        f"No exportable replicas available for artifact {artifact_id}"
                    )

            except (NotFoundError, TimeoutError):
                raise
            except Exception as e:  # noqa: BLE001
                # Handle database conflicts/retries - retry on conflicts
                message = str(e).lower()
                if any(
                    token in message
                    for token in ("transaction", "conflict", "serialization")
                ):
                    logger.debug(
                        "Transport request retry due to transient DB error: %s", e
                    )
                else:
                    raise
                # Fall through to timeout/sleep handling for transient errors.

            # Check timeout
            if time.time() >= end_time:
                if not self.replica_repository.has_any_replica(artifact_id, view_id):
                    inc_transport_request(artifact_id, "not_found")
                    wait_sec = time.time() - start_time
                    self._log_eligibility_snapshot(
                        artifact_id=artifact_id,
                        view_id=view_id,
                        wait_timeout_ms=wait_timeout_ms,
                        outcome="timeout_not_found",
                        waited_sec=wait_sec,
                    )
                    observe_transport_wait(artifact_id, wait_sec)
                    raise NotFoundError(
                        f"No replicas registered for artifact {artifact_id}"
                    )

                self._log_eligibility_snapshot(
                    artifact_id=artifact_id,
                    view_id=view_id,
                    wait_timeout_ms=wait_timeout_ms,
                    outcome="timeout",
                    waited_sec=time.time() - start_time,
                )

                inc_transport_request(artifact_id, "timeout")
                wait_sec = time.time() - start_time
                observe_transport_wait(artifact_id, wait_sec)
                raise TimeoutError(
                    f"No available replica for artifact {artifact_id} within timeout"
                )

            # Wait before retry
            time.sleep(self.config.transport_wait_retry_interval_ms / 1000)

    def complete_transport(self, transport_id: UUID) -> Tuple[int, int]:
        """
        Complete a transport and release resources.

        Args:
            transport_id: ID of the transport to complete

        Returns:
            Tuple of (current_requests, max_concurrency) after completion

        Raises:
            NotFoundError: If transport not found
        """
        # Find transport record
        transport = self.transport_repository.find_by_id(transport_id)
        if not transport:
            raise NotFoundError(f"Transport {transport_id} not found")

        # Decrement replica requests counter
        current, max_conc = self.replica_repository.decrement_requests(
            transport.replica_id
        )

        # Update transport status to completed
        from datetime import datetime

        self.transport_repository.update_status(
            transport_id, "completed", completed_at=datetime.now()
        )

        dec_active_transports()

        logger.info(
            f"Completed transport {transport_id} for {transport.artifact_id}, "
            f"replica: {transport.replica_id}, "
            f"new load: {current}/{max_conc}"
        )

        return current, max_conc

    def cleanup_expired_transports(self, expiration_seconds: int = 600) -> int:
        """Release transports that have been in *in_progress* state for too long.

        This serves as a safety-net for StoreDaemon crashes or network
        partitions that prevent the normal *CompleteReplicaTransport*
        call from reaching the Global-Store.  By periodically invoking this
        method, leaked *current_requests* counters on the source replicas are
        automatically decremented, avoiding long-term load-balancing issues.

        Args:
            expiration_seconds: Age threshold (in seconds).  Any transport
                whose *created_at* is older than this value and still in
                ``in_progress`` status will be force-completed.

        Returns:
            Number of transports that were force-completed.
        """

        expired: list[Transport] = []

        try:
            pending = self.transport_repository.list_with_filters(
                status="in_progress", limit=10_000
            )
            expired = [t for t in pending if t.age_seconds > expiration_seconds]
        except Exception:
            logger.exception("Failed to fetch pending transports for cleanup")
            return 0

        cleaned = 0
        for tr in expired:
            try:
                self.complete_transport(tr.transport_id)
                cleaned += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to force-complete stale transport %s: %s",
                    tr.transport_id,
                    exc,
                )

        if cleaned:
            logger.info(
                "Cleaned up %s stale transports (>%ss)", cleaned, expiration_seconds
            )

        return cleaned
