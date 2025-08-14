#  Copyright (c) 2025, StepCast Team.

"""Test Web UI with gRPC backend integration."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from scstore.global_store.webui_backend.grpc_client import (
    GlobalStoreClient,
    GlobalStoreClientConfig,
)
from scstore.proto import global_store_pb2
from scstore.global_store.webui_backend.grpc_client import WorkerInfoWrapper


@pytest.fixture
def mock_grpc_client():
    """Create a mock gRPC client."""
    client = AsyncMock(spec=GlobalStoreClient)

    # Mock list_active_workers
    worker1 = WorkerInfoWrapper(global_store_pb2.ListActiveWorkersResponse.WorkerInfo(
        worker_id="worker-1",
        node_id="node-1",
        node_address="192.168.1.1",
        grpc_port=50052,
        p2p_port=50053,
        mem_pool_total_size=10737418240,  # 10GB
        mem_pool_available_size=5368709120,  # 5GB
        accepting_new_requests=True,
        last_heartbeat_timestamp=1234567890,
    ))
    client.list_active_workers.return_value = [worker1]

    # Mock list_model_replicas
    replica1 = global_store_pb2.MemoryInfo(
        node_id="node-1",
        node_address="192.168.1.1",
        node_port=50052,
        memory_size=1073741824,  # 1GB
        memory_type=global_store_pb2.MemoryType.GPU,
        device_id=0,
    )
    client.list_model_replicas.return_value = {
        "model1": [replica1],
    }

    # Mock get_summary_stats
    client.get_summary_stats.return_value = {
        "total_workers": 1,
        "active_workers": 1,
        "total_models": 1,
        "total_replicas": 1,
        "gpu_replicas": 1,
        "ram_replicas": 0,
        "disk_replicas": 0,
        "active_transports": 0,
    }

    return client


@pytest.mark.asyncio
async def test_grpc_client_connection():
    """Test gRPC client connection."""
    config = GlobalStoreClientConfig(host="127.0.0.1", port=50051)

    # Patch the synchronous gRPC channel and stub used inside GlobalStoreClient.
    # The implementation relies on the *blocking* (non-aio) API, so patch that
    # instead of ``grpc.aio`` to avoid real network traffic during the unit test.
    with patch("grpc.insecure_channel") as mock_channel, patch(
        "scstore.global_store.webui_backend.grpc_client.global_store_pb2_grpc.GlobalModelStoreStub"
    ) as mock_stub:
        # Mock channel instance returned by grpc.insecure_channel
        mock_channel_instance = MagicMock()
        mock_channel.return_value = mock_channel_instance

        # Mock stub instance and have the health-check RPC return an empty list
        mock_stub_instance = MagicMock()
        mock_stub_instance.ListActiveWorkers.return_value = (
            global_store_pb2.ListActiveWorkersResponse(workers=[])
        )
        mock_stub.return_value = mock_stub_instance

        client = GlobalStoreClient(config)
        await client.connect()

        assert client._channel is mock_channel_instance
        assert client._stub is mock_stub_instance
        mock_channel.assert_called_once_with("127.0.0.1:50051")


@pytest.mark.asyncio
async def test_api_endpoints_with_grpc(mock_grpc_client):
    """Test API endpoints using gRPC client."""
    from scstore.global_store.webui_backend.api import get_summary, list_workers, list_models

    # Test summary endpoint
    summary_response = await get_summary(mock_grpc_client)

    # ``data`` is a ``GlobalMetrics`` Pydantic model – assert via attributes.
    assert summary_response.data.total_workers == 1
    assert summary_response.data.active_workers == 1
    assert summary_response.data.total_models == 1

    # Test workers endpoint
    workers_response = await list_workers(mock_grpc_client, False, 1, 50)
    assert len(workers_response.data) == 1
    assert workers_response.data[0]["worker_id"] == "worker-1"
    assert workers_response.meta is not None
    assert workers_response.meta["total_count"] == 1

    # Test models endpoint
    models_response = await list_models(mock_grpc_client)
    assert len(models_response.data) == 1
    assert models_response.data[0]["model_name"] == "model1"
    assert models_response.data[0]["gpu_replicas"] == 1


@pytest.mark.asyncio
async def test_websocket_polling(mock_grpc_client):
    """Test WebSocket polling mechanism."""
    from scstore.global_store.webui_backend.websocket import WebSocketManager

    ws_manager = WebSocketManager()
    ws_manager.set_grpc_client(mock_grpc_client)

    # Let polling run for a short time
    await asyncio.sleep(0.1)

    # Check that the manager has initialized its state
    assert ws_manager.grpc_client is not None
    assert isinstance(ws_manager._last_state, dict)



if __name__ == "__main__":
    print("\nTesting gRPC client connection...")
    asyncio.run(test_grpc_client_connection())
    print("✓ gRPC client connection test passed")

    print("\nAll tests passed!")