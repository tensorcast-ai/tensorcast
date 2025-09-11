#  Copyright (c) 2025, TensorCast Team.

"""Repository for chunk directory operations."""

from typing import List, Optional, Tuple

from tensorcast.logger import init_logger
from tensorcast.proto.global_store.v1 import global_store_pb2

from .base import BaseRepository

logger = init_logger(__name__)


class ChunkDirectoryRepository(BaseRepository):
    """Repository for managing chunk directory entries."""

    def batch_update_chunk_states(
        self,
        worker_id: str,
        node_id: str,
        updates: List[global_store_pb2.ChunkStateUpdate],
    ) -> int:
        """
        Batch update chunk states from a worker.

        Args:
            worker_id: Worker ID performing the update
            node_id: Node ID where chunks are located
            updates: List of chunk state updates

        Returns:
            Number of updates applied
        """
        if not updates:
            return 0

        cursor = self.get_cursor()
        updates_applied = 0

        try:
            # Use ON CONFLICT DO UPDATE to perform atomic upsert on composite PK
            for update in updates:
                cursor.execute(
                    """
                    INSERT INTO chunk_directory (
                        artifact_id, chunk_idx, node_id, device_uuid, replica,
                        chunk_state, last_update_time, node_load_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, now(), 0.0)
                    ON CONFLICT (artifact_id, device_uuid, replica, chunk_idx, node_id)
                    DO UPDATE SET
                        chunk_state = EXCLUDED.chunk_state,
                        last_update_time = now(),
                        node_load_ratio = EXCLUDED.node_load_ratio
                    """,
                    (
                        update.artifact_id,
                        update.chunk_idx,
                        node_id,
                        update.device_uuid,
                        update.replica,
                        update.state,
                    ),
                )
                updates_applied += 1

            logger.info(
                f"Applied {updates_applied} chunk state updates from worker {worker_id}"
            )

        except Exception as e:
            logger.exception(f"Error updating chunk states: {e}")
            raise

        return updates_applied

    def query_chunk_locations(
        self, artifact_id: str, chunk_indices: Optional[List[int]] = None
    ) -> List[Tuple]:
        """
        Query chunk locations for a artifact.

        Args:
            artifact_id: Artifact identifier
            chunk_indices: Optional list of specific chunks to query

        Returns:
            List of tuples containing chunk location information
        """
        # Explicitly treat an empty list as an empty result set (do not fall back to full scan)
        if chunk_indices is not None and len(chunk_indices) == 0:
            return []

        cursor = self.get_cursor()

        try:
            if chunk_indices:
                # Query specific chunks
                placeholders = ",".join("?" for _ in chunk_indices)
                query = f"""
                    SELECT DISTINCT
                        cd.chunk_idx, w.node_id, w.node_address, w.p2p_port,
                        cd.chunk_state, cd.node_load_ratio, cd.device_uuid, cd.replica
                    FROM chunk_directory cd
                    JOIN workers w ON cd.node_id = w.node_id
                    WHERE cd.artifact_id = ?
                      AND cd.chunk_idx IN ({placeholders})
                      AND cd.chunk_state != 4  -- Exclude EVICTED chunks
                    ORDER BY cd.chunk_idx, cd.chunk_state, cd.node_load_ratio
                """
                params = [artifact_id] + chunk_indices
            else:
                # Query all chunks for the artifact
                query = """
                    SELECT DISTINCT
                        cd.chunk_idx, w.node_id, w.node_address, w.p2p_port,
                        cd.chunk_state, cd.node_load_ratio, cd.device_uuid, cd.replica
                    FROM chunk_directory cd
                    JOIN workers w ON cd.node_id = w.node_id
                    WHERE cd.artifact_id = ?
                      AND cd.chunk_state != 4  -- Exclude EVICTED chunks
                    ORDER BY cd.chunk_idx, cd.chunk_state, cd.node_load_ratio
                """
                params = [artifact_id]

            return cursor.execute(query, params).fetchall()

        except Exception as e:
            logger.exception(f"Error querying chunk locations: {e}")
            raise

    def cleanup_stale_chunks(self, stale_threshold_seconds: int = 3600) -> int:
        """
        Remove chunk entries that haven't been updated recently.

        Args:
            stale_threshold_seconds: Threshold in seconds for considering chunks stale

        Returns:
            Number of entries removed
        """
        try:
            # Execute count + delete atomically to avoid interleaving changes
            with self.transaction() as cursor:
                count_row = cursor.execute(
                    """
                    SELECT COUNT(*) FROM chunk_directory
                    WHERE EXTRACT(epoch FROM last_update_time) < EXTRACT(epoch FROM current_timestamp) - ?
                    """,
                    (stale_threshold_seconds,),
                ).fetchone()
                deleted_count = int(count_row[0]) if count_row is not None else 0

                if deleted_count > 0:
                    cursor.execute(
                        """
                        DELETE FROM chunk_directory
                        WHERE EXTRACT(epoch FROM last_update_time) < EXTRACT(epoch FROM current_timestamp) - ?
                        """,
                        (stale_threshold_seconds,),
                    )
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} stale chunk entries")

            return deleted_count

        except Exception as e:
            logger.exception(f"Error cleaning up stale chunks: {e}")
            raise

    def get_chunk_distribution(self, artifact_id: str) -> dict:
        """
        Get chunk distribution statistics for a artifact.

        Args:
            artifact_id: Artifact identifier

        Returns:
            Dictionary with distribution statistics
        """
        cursor = self.get_cursor()

        try:
            # Get chunk count by state
            state_counts = cursor.execute(
                """
                SELECT chunk_state, COUNT(DISTINCT chunk_idx) as count
                FROM chunk_directory
                WHERE artifact_id = ?
                GROUP BY chunk_state
                """,
                (artifact_id,),
            ).fetchall()

            # Get chunk count by node
            node_counts = cursor.execute(
                """
                SELECT node_id, COUNT(DISTINCT chunk_idx) as count
                FROM chunk_directory
                WHERE artifact_id = ?
                GROUP BY node_id
                """,
                (artifact_id,),
            ).fetchall()

            return {
                "state_distribution": dict(state_counts),
                "node_distribution": dict(node_counts),
            }

        except Exception as e:
            logger.exception(f"Error getting chunk distribution: {e}")
            raise
