#  Copyright (c) 2025, StepCast Team.

"""Service for chunk directory operations."""

from typing import List, Optional

from scstore.logger import init_logger
from scstore.proto import global_store_pb2

from ..repositories.chunk_directory_repository import ChunkDirectoryRepository

logger = init_logger(__name__)


class ChunkService:
    """Service for managing chunk directory operations."""

    def __init__(self, chunk_repository: ChunkDirectoryRepository):
        """
        Initialize the chunk service.

        Args:
            chunk_repository: Repository for chunk directory operations
        """
        self.chunk_repository = chunk_repository

    def query_chunk_locations(
        self, artifact_id: str, chunk_indices: Optional[List[int]] = None
    ) -> List[global_store_pb2.ChunkLocation]:
        """
        Query chunk locations for a artifact.

        Args:
            artifact_id: Artifact identifier
            chunk_indices: Optional list of specific chunks to query

        Returns:
            List of ChunkLocation protobuf messages
        """
        try:
            # Query from repository
            locations = self.chunk_repository.query_chunk_locations(
                artifact_id, chunk_indices
            )

            # Convert to protobuf messages
            chunk_locations = []
            for row in locations:
                (
                    chunk_idx,
                    node_id,
                    node_address,
                    p2p_port,
                    chunk_state,
                    node_load_ratio,
                    device_uuid,
                    replica,
                ) = row

                location = global_store_pb2.ChunkLocation(
                    chunk_idx=chunk_idx,
                    node_id=node_id,
                    node_address=node_address,
                    p2p_port=p2p_port,
                    state=chunk_state,
                    node_load_ratio=node_load_ratio,
                    device_uuid=device_uuid,
                    replica=replica,
                )
                chunk_locations.append(location)

            logger.debug(
                f"Found {len(chunk_locations)} chunk locations for artifact {artifact_id}"
            )

            return chunk_locations

        except Exception as e:
            logger.exception(f"Error querying chunk locations: {e}")
            raise

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
        try:
            updates_applied = self.chunk_repository.batch_update_chunk_states(
                worker_id, node_id, updates
            )

            logger.info(
                f"Applied {updates_applied}/{len(updates)} chunk state updates "
                f"from worker {worker_id} on node {node_id}"
            )

            return updates_applied

        except Exception as e:
            logger.exception(f"Error updating chunk states: {e}")
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
            return self.chunk_repository.cleanup_stale_chunks(stale_threshold_seconds)
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
        try:
            return self.chunk_repository.get_chunk_distribution(artifact_id)
        except Exception as e:
            logger.exception(f"Error getting chunk distribution: {e}")
            raise
