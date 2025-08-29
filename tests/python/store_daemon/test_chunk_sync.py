#  Copyright (c) 2025, TensorCast Team.

# mypy: disable-error-code=method-assign
"""Tests for chunk state synchronization worker."""

import time
import threading
from unittest.mock import Mock, MagicMock, patch, call
import grpc
import pytest

from scstore.proto import global_store_pb2
from scstore.store_daemon.chunk_sync import ChunkSyncWorker


class TestChunkSyncWorker:
    """Test chunk synchronization functionality."""

    @pytest.fixture
    def mock_servicer(self):
        """Create mock servicer with necessary attributes."""
        servicer = Mock()
        servicer.worker_id = "test-worker-123"
        servicer.store_engine = Mock()
        servicer.replica_manager = Mock()
        servicer.replica_manager.get_loaded_replicas.return_value = ["model1", "model2"]
        return servicer

    @pytest.fixture
    def mock_stub(self):
        """Create mock gRPC stub."""
        stub = Mock()
        # Mock successful response
        response = Mock()
        response.status.code = 0
        response.updates_applied = 10
        stub.BatchUpdateChunkStates.return_value = response
        return stub

    def test_initialization(self, mock_servicer):
        """Test worker initialization."""
        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
            sync_interval_seconds=0.1,
            batch_size=500,
        )

        assert worker.servicer == mock_servicer
        assert worker.global_store_address == "localhost:50051"
        assert worker.sync_interval == 0.1
        assert worker.batch_size == 500
        assert worker._sync_thread is None
        assert worker._channel is None

    @patch('grpc.insecure_channel')
    @patch('scstore.store_daemon.chunk_sync.global_store_pb2_grpc.GlobalStoreStub')
    def test_start_stop(self, mock_stub_class, mock_channel, mock_servicer):
        """Test starting and stopping the worker."""
        # Setup mocks
        mock_channel_instance = Mock()
        mock_channel.return_value = mock_channel_instance
        mock_stub_instance = Mock()
        mock_stub_class.return_value = mock_stub_instance

        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
            sync_interval_seconds=0.1,
        )

        # Start the worker
        worker.start()

        # Verify channel and stub created
        mock_channel.assert_called_once_with("localhost:50051")
        mock_stub_class.assert_called_once_with(mock_channel_instance)

        # Verify thread started
        assert worker._sync_thread is not None
        assert worker._sync_thread.is_alive()

        # Stop the worker
        worker.stop()

        # Verify thread stopped
        assert worker._sync_thread is None
        mock_channel_instance.close.assert_called_once()

    def test_chunk_state_mapping(self, mock_servicer):
        """Test chunk state enum mapping."""
        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
        )

        # Test all state mappings
        assert worker._map_chunk_state(0) == global_store_pb2.ChunkState.CHUNK_HOT
        assert worker._map_chunk_state(1) == global_store_pb2.ChunkState.CHUNK_LOCKED_TX
        assert worker._map_chunk_state(2) == global_store_pb2.ChunkState.CHUNK_COPIED_GPU
        assert worker._map_chunk_state(3) == global_store_pb2.ChunkState.CHUNK_COLD
        assert worker._map_chunk_state(4) == global_store_pb2.ChunkState.CHUNK_EVICTED

        # Unknown state should map to HOT (default)
        assert worker._map_chunk_state(99) == global_store_pb2.ChunkState.CHUNK_HOT

    @patch('grpc.insecure_channel')
    @patch('scstore.store_daemon.chunk_sync.global_store_pb2_grpc.GlobalStoreStub')
    def test_sync_with_updates(self, mock_stub_class, mock_channel, mock_servicer, mock_stub):
        """Test synchronization with chunk updates."""
        # Setup mocks
        mock_channel.return_value = Mock()
        mock_stub_class.return_value = mock_stub

        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
            sync_interval_seconds=0.1,
        )

        # Mock chunk states
        worker._get_all_chunk_states = Mock(return_value={
            "model1": {0: 0, 1: 2, 2: 4},  # HOT, COPIED_GPU, EVICTED
            "model2": {0: 1, 1: 3},  # LOCKED_TX, COLD
        })

        # Perform sync
        worker._channel = Mock()
        worker._stub = mock_stub
        worker._perform_sync()

        # Verify BatchUpdateChunkStates was called
        assert mock_stub.BatchUpdateChunkStates.called

        # Get the request that was sent
        request = mock_stub.BatchUpdateChunkStates.call_args[0][0]
        assert request.worker_id == "test-worker-123"
        assert request.node_id == "test-worker-123"
        assert len(request.updates) == 5  # All chunks should be sent on first sync

        # Verify updates contain correct data
        update_map = {(u.artifact_id, u.chunk_idx): u.state for u in request.updates}
        assert update_map[("model1", 0)] == global_store_pb2.ChunkState.CHUNK_HOT
        assert update_map[("model1", 1)] == global_store_pb2.ChunkState.CHUNK_COPIED_GPU
        assert update_map[("model1", 2)] == global_store_pb2.ChunkState.CHUNK_EVICTED
        assert update_map[("model2", 0)] == global_store_pb2.ChunkState.CHUNK_LOCKED_TX
        assert update_map[("model2", 1)] == global_store_pb2.ChunkState.CHUNK_COLD

    @patch('grpc.insecure_channel')
    @patch('scstore.store_daemon.chunk_sync.global_store_pb2_grpc.GlobalStoreStub')
    def test_sync_delta_updates(self, mock_stub_class, mock_channel, mock_servicer, mock_stub):
        """Test that only changed chunks are synchronized."""
        # Setup mocks
        mock_channel.return_value = Mock()
        mock_stub_class.return_value = mock_stub

        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
            sync_interval_seconds=0.1,
        )

        # Initial state
        initial_states = {
            "model1": {0: 0, 1: 2},
            "model2": {0: 1},
        }
        worker._get_all_chunk_states = Mock(return_value=initial_states)

        # First sync
        worker._channel = Mock()
        worker._stub = mock_stub
        worker._perform_sync()

        # Reset mock
        mock_stub.BatchUpdateChunkStates.reset_mock()

        # Update states - only model1 chunk 0 changes
        updated_states = {
            "model1": {0: 3, 1: 2},  # Chunk 0 changed from HOT to COLD
            "model2": {0: 1},
        }
        worker._get_all_chunk_states = Mock(return_value=updated_states)

        # Second sync
        worker._perform_sync()

        # Verify only the changed chunk was sent
        request = mock_stub.BatchUpdateChunkStates.call_args[0][0]
        assert len(request.updates) == 1
        assert request.updates[0].artifact_id == "model1"
        assert request.updates[0].chunk_idx == 0
        assert request.updates[0].state == global_store_pb2.ChunkState.CHUNK_COLD

    @patch('grpc.insecure_channel')
    @patch('scstore.store_daemon.chunk_sync.global_store_pb2_grpc.GlobalStoreStub')
    def test_sync_batch_size(self, mock_stub_class, mock_channel, mock_servicer, mock_stub):
        """Test that updates are sent in batches."""
        # Setup mocks
        mock_channel.return_value = Mock()
        mock_stub_class.return_value = mock_stub

        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
            sync_interval_seconds=0.1,
            batch_size=3,  # Small batch size for testing
        )

        # Create many chunk updates
        states = {}
        for i in range(10):
            states[f"model_{i}"] = {0: i % 5}  # Various states

        worker._get_all_chunk_states = Mock(return_value=states)

        # Perform sync
        worker._channel = Mock()
        worker._stub = mock_stub
        worker._perform_sync()

        # Verify multiple batches were sent
        assert mock_stub.BatchUpdateChunkStates.call_count == 4  # 10 updates / 3 batch size = 4 calls

        # Verify batch sizes
        call_args_list = mock_stub.BatchUpdateChunkStates.call_args_list
        assert len(call_args_list[0][0][0].updates) == 3
        assert len(call_args_list[1][0][0].updates) == 3
        assert len(call_args_list[2][0][0].updates) == 3
        assert len(call_args_list[3][0][0].updates) == 1  # Last batch has remainder

    @patch('grpc.insecure_channel')
    @patch('scstore.store_daemon.chunk_sync.global_store_pb2_grpc.GlobalStoreStub')
    def test_rpc_error_handling(self, mock_stub_class, mock_channel, mock_servicer):
        """Test handling of RPC errors."""
        # Setup mocks
        mock_channel.return_value = Mock()
        mock_stub_instance = Mock()
        mock_stub_instance.BatchUpdateChunkStates.side_effect = grpc.RpcError("Connection failed")
        mock_stub_class.return_value = mock_stub_instance

        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
            sync_interval_seconds=0.1,
        )

        # Mock chunk states
        worker._get_all_chunk_states = Mock(return_value={
            "model1": {0: 0},
        })

        # Perform sync - should not raise exception
        worker._channel = Mock()
        worker._stub = mock_stub_instance
        worker._perform_sync()  # Should log error but not crash

    def test_artifact_id_extraction(self, mock_servicer):
        """Test artifact ID extraction from paths."""
        worker = ChunkSyncWorker(
            servicer=mock_servicer,
            global_store_address="localhost:50051",
        )

        # Test various path formats
        assert worker._extract_artifact_id("/path/to/model1") == "model1"
        assert worker._extract_artifact_id("model2") == "model2"
        assert worker._extract_artifact_id("/a/b/c/d/model3") == "model3"
        assert worker._extract_artifact_id("") == ""