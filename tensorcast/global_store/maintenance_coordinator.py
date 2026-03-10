#  Copyright (c) 2025-2026, TensorCast Team.

"""Background maintenance coordinator for Global Store."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from tensorcast.global_store import metrics as gs_metrics
from tensorcast.global_store.db_utils import optimize_db
from tensorcast.global_store.repositories.base import db_execution_lock


class GlobalStoreMaintenanceCoordinator:
    """Runs periodic cleanup, retention GC, and database optimization."""

    def __init__(
        self,
        *,
        config: Any,
        connection,
        get_worker_service: Callable[[], Any],
        get_instance_service: Callable[[], Any],
        get_transport_service: Callable[[], Any],
        logger,
    ) -> None:
        self._config = config
        self._connection = connection
        self._get_worker_service = get_worker_service
        self._get_instance_service = get_instance_service
        self._get_transport_service = get_transport_service
        self._logger = logger
        self._thread: threading.Thread | None = None

    def _initial_maintenance_delay_sec(self) -> float:
        """Compute startup delay before first maintenance pass.

        Delay is profile-linked to both heartbeat timeout and cleanup interval:
        - wait enough time for initial worker registrations/heartbeats
        - never wait longer than one cleanup interval
        """
        heartbeat_timeout_sec = max(0.0, self._config.heartbeat_timeout_ms / 1000.0)
        cleanup_interval_sec = max(1.0, self._config.cleanup_interval_ms / 1000.0)
        if heartbeat_timeout_sec <= 0:
            return cleanup_interval_sec
        return min(cleanup_interval_sec, max(1.0, heartbeat_timeout_sec * 2.0))

    def _transport_expiration_seconds(self) -> int:
        """Compute stale transport expiration threshold from profile settings."""
        heartbeat_timeout_sec = max(1, int(self._config.heartbeat_timeout_ms / 1000))
        cleanup_interval_sec = max(1, int(self._config.cleanup_interval_ms / 1000))
        profile_window_sec = max(heartbeat_timeout_sec * 2, cleanup_interval_sec * 10)
        return max(60, int(profile_window_sec))

    def start(self) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self._thread.start()
        return self._thread

    def _maintenance_loop(self) -> None:
        initial_delay_sec = self._initial_maintenance_delay_sec()
        cleanup_interval_sec = max(1.0, self._config.cleanup_interval_ms / 1000.0)
        optimize_interval_sec = max(1.0, self._config.optimize_interval_ms / 1000.0)
        transport_expiration_seconds = self._transport_expiration_seconds()
        self._logger.info(
            "GlobalStore maintenance policy: initial_delay_sec=%.3f cleanup_interval_sec=%.3f "
            "optimize_interval_sec=%.3f transport_expiration_seconds=%s",
            initial_delay_sec,
            cleanup_interval_sec,
            optimize_interval_sec,
            transport_expiration_seconds,
        )
        time.sleep(initial_delay_sec)

        last_optimize_ts = time.time()

        while True:
            try:
                worker_service = self._get_worker_service()
                if worker_service is not None:
                    worker_service.cleanup_inactive_workers()

                instance_service = self._get_instance_service()
                if instance_service is not None:
                    try:
                        instance_service.cleanup_inactive_instances()
                    except Exception:
                        self._logger.exception("Error cleaning up inactive instances")

                transport_service = self._get_transport_service()
                if transport_service is not None:
                    try:
                        transport_service.cleanup_expired_transports(
                            expiration_seconds=transport_expiration_seconds
                        )
                    except Exception:
                        self._logger.exception("Error cleaning up expired transports")

                try:
                    self._run_retention_gc()
                except Exception:
                    self._logger.exception("Error applying retention / GC policies")

                if time.time() - last_optimize_ts >= optimize_interval_sec:
                    try:
                        optimize_db(self._connection)
                    except Exception:
                        # optimize_db already logs; safeguard thread
                        self._logger.exception("Error optimizing database")
                    last_optimize_ts = time.time()

            except Exception:
                self._logger.exception("Error in maintenance thread")

            time.sleep(cleanup_interval_sec)

    def _run_retention_gc(self) -> None:
        retention = self._config.limits.retention
        now = datetime.now(timezone.utc)

        with db_execution_lock():
            cursor = self._connection.cursor()
            try:
                if retention.operations_ttl_ms > 0:
                    cutoff = now - timedelta(
                        milliseconds=int(retention.operations_ttl_ms)
                    )
                    row = cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM operations
                        WHERE state IN ('success','failed','cancelled','degraded')
                          AND updated_at < ?
                        """,
                        [cutoff],
                    ).fetchone()
                    count = int(row[0]) if row else 0
                    if count > 0:
                        cursor.execute(
                            """
                            DELETE FROM operations
                            WHERE state IN ('success','failed','cancelled','degraded')
                              AND updated_at < ?
                            """,
                            [cutoff],
                        )
                        gs_metrics.inc_gc_rows_deleted(table="operations", count=count)

                if retention.assembly_proof_commitments_ttl_ms > 0:
                    cutoff = now - timedelta(
                        milliseconds=int(retention.assembly_proof_commitments_ttl_ms)
                    )
                    row = cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM assembly_proof_commitments
                        WHERE created_at < ?
                        """,
                        [cutoff],
                    ).fetchone()
                    count = int(row[0]) if row else 0
                    if count > 0:
                        cursor.execute(
                            """
                            DELETE FROM assembly_proof_commitments
                            WHERE created_at < ?
                            """,
                            [cutoff],
                        )
                        gs_metrics.inc_gc_rows_deleted(
                            table="assembly_proof_commitments", count=count
                        )

                if retention.piece_proof_digests_ttl_ms > 0:
                    cutoff = now - timedelta(
                        milliseconds=int(retention.piece_proof_digests_ttl_ms)
                    )
                    row = cursor.execute(
                        """
                        SELECT COUNT(*)
                        FROM piece_proof_digests
                        WHERE created_at < ?
                        """,
                        [cutoff],
                    ).fetchone()
                    count = int(row[0]) if row else 0
                    if count > 0:
                        cursor.execute(
                            """
                            DELETE FROM piece_proof_digests
                            WHERE created_at < ?
                            """,
                            [cutoff],
                        )
                        gs_metrics.inc_gc_rows_deleted(
                            table="piece_proof_digests", count=count
                        )
            finally:
                cursor.close()
