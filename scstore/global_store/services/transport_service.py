#  Copyright (c) 2025, TensorCast Team.

"""Service for transport operations."""

import time
from typing import Tuple
from uuid import UUID

from scstore.global_store.config import get_config
from scstore.global_store.exceptions import NotFoundError, TimeoutError
from scstore.global_store.metrics import (
    dec_active_transports,
    inc_active_transports,
    inc_transport_request,
    observe_transport_wait,
)
from scstore.global_store.models import Replica, Transport
from scstore.global_store.repositories import (
    ReplicaRepository,
    TransportRepository,
)
from scstore.logger import init_logger

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

    def request_transport(
        self,
        artifact_id: str,
        source_node_id: str,
        source_address: str,
        source_port: int,
        wait_timeout_ms: int = 0,
    ) -> Tuple[Replica, UUID]:
        """
        Request a artifact transport with load balancing.

        Args:
            artifact_id: Content-addressed artifact id (mi2:...)
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

        while True:
            try:
                # Try to find and claim an available replica
                replica = self.replica_repository.find_available_for_transport(
                    artifact_id=artifact_id,
                    heartbeat_timeout_seconds=self.config.heartbeat_timeout_ms / 1000,
                )

                if replica:
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

            except Exception as e:
                # Handle database conflicts/retries - retry on conflicts
                logger.debug(f"Transport request retry due to: {e}")

            # Check timeout
            if time.time() >= end_time:
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
